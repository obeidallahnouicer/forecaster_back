from typing import Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import HumanMessage
from . import config, retriever, data_loader
import pandas as pd
import threading
import json
from pathlib import Path

# Keep an in-memory map of thread_id -> memory
_memories: Dict[str, ConversationBufferMemory] = {}
_memories_lock = threading.Lock()


class SimpleMemorySaver(BaseCallbackHandler):
    """A simple memory saver that appends assistant/user messages to an external store.
    For now it uses the ConversationBufferMemory in-memory object.
    """

    def __init__(self, thread_id: str):
        self.thread_id = thread_id

    def on_chat_model_start(self, serialized, messages, **kwargs):
        return None


def get_memory(thread_id: str) -> ConversationBufferMemory:
    with _memories_lock:
        if thread_id not in _memories:
            mem = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            # Load persisted memory if exists
            mem_file = Path(config.MEMORY_DIR) / f"{thread_id}.json"
            if mem_file.exists():
                try:
                    data = json.loads(mem_file.read_text(encoding="utf-8"))
                    # data is expected to be list of messages: [{"role":"user|ai","text":...}, ...]
                    for m in data:
                        if m.get("role") == "user":
                            mem.chat_memory.add_user_message(m.get("text"))
                        else:
                            mem.chat_memory.add_ai_message(m.get("text"))
                except Exception:
                    pass
            _memories[thread_id] = mem
        return _memories[thread_id]


def reset_memory(thread_id: str):
    with _memories_lock:
        if thread_id in _memories:
            del _memories[thread_id]
    # remove persisted memory file
    try:
        mem_file = Path(config.MEMORY_DIR) / f"{thread_id}.json"
        if mem_file.exists():
            mem_file.unlink()
    except Exception:
        pass


def _build_chain(thread_id: str):
    """Build a ConversationalRetrievalChain for the given thread_id."""
    # Use Groq ChatGroq LLM (requires GROQ_API_KEY in .env)
    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY must be set in .env to use Groq LLM")
    llm = ChatGroq(api_key=config.GROQ_API_KEY, model=config.GROQ_MODEL, temperature=0.0, max_tokens=512)
    retr = retriever.get_retriever()
    memory = get_memory(thread_id)
    # Build a prompt that tells the model about the dataset columns and desired behavior.
    columns = (
        "Client_Principal; Code_Client; Intitule_client; Categorie_Client; Pays; Zone; Gouvernorat; Adresse; "
        "Representant; NDocument; Type_Document; Date; Mois; Annee; Marque; Famille; Sous_Famille; Ref_Article; "
        "Designation; Qte_Vendu; CA_HT_BRUT; Tx_Remise; CA_HT_NET; Année"
    )

    prompt_template = (
        "You are a senior data/business analyst assistant. "
        "You have access to retrieved documents (context) coming from a CSV with the following columns:\n"
        "{columns}\n\n"
        "Rules:\n"
        "- Use ONLY the information from the retrieved context and the data when producing answers.\n"
        "- Ground your reasoning in the retrieved documents; cite sources using metadata when possible (e.g., source_row).\n"
        "- Provide clear analysis, actionable insights, and recommendations.\n"
        "- If the information is insufficient, say so and suggest what additional data is needed.\n"
        "- Be concise and show results as bullet points, tables or short summaries.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:" 
    )

    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template.replace("{columns}", columns))

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retr,
        memory=memory,
        return_source_documents=True,
        verbose=False,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
    )
    return chain


def _answer_with_dataframe(query: str, df: pd.DataFrame) -> Optional[str]:
    """Lightweight placeholder.

    We intentionally avoid hardcoded analytics here. Prefer RAG + LLM to interpret
    analytical requests. This function is kept as a stub for explicit structured
    commands in the future. Currently it always returns None so the query falls
    back to the retrieval-augmented LLM pipeline.
    """
    return None


def chat(message: str, thread_id: str) -> Dict[str, Any]:
    """Main chat entrypoint. Tries dataframe answers first, then retrieval chain."""
    # Try dataframe-based quick answers
    df = data_loader.load_and_preprocess()
    df_answer = _answer_with_dataframe(message, df)
    chain = _build_chain(thread_id)

    if df_answer:
        # Log to memory as assistant reply
        memory = get_memory(thread_id)
        memory.chat_memory.add_user_message(message)
        memory.chat_memory.add_ai_message(df_answer)
        # persist memory
        _persist_memory(thread_id)
        return {"answer": df_answer, "source": "dataframe", "thread_id": thread_id}

    res = chain.run(input=message)

    # ConversationalRetrievalChain.run usually returns a string answer; retrieve source docs via call
    # To get source documents, call the chain as dict
    try:
        res_dict = chain({"question": message})
        answer = res_dict.get("answer") or res_dict.get("result") or res
        sourcedocs = res_dict.get("source_documents")
    except Exception:
        answer = res
        sourcedocs = None

    # Log and persist memory
    memory = get_memory(thread_id)
    memory.chat_memory.add_user_message(message)
    memory.chat_memory.add_ai_message(answer)
    _persist_memory(thread_id)

    return {"answer": answer, "source": "rag", "thread_id": thread_id, "source_documents": [
        {"page_content": d.page_content, "metadata": d.metadata} for d in (sourcedocs or [])
    ]}


def _persist_memory(thread_id: str):
    """Persist conversation history for a thread to disk as JSON list of messages."""
    try:
        mem = _memories.get(thread_id)
        if not mem:
            return
        messages = []
        for m in mem.chat_memory.messages:
            role = "user" if m.type == "human" else "ai"
            messages.append({"role": role, "text": m.content})
        mem_file = Path(config.MEMORY_DIR) / f"{thread_id}.json"
        mem_file.write_text(json.dumps(messages, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass