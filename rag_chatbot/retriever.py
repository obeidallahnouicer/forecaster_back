from typing import Optional
from pathlib import Path
import pickle
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from . import config, data_loader
import os
import hashlib
from tqdm import tqdm
import json


def _load_cached_vectorstore() -> Optional[FAISS]:
    if config.FAISS_INDEX_PATH.exists() and config.METADATA_PATH.exists():
        try:
            # Load with embeddings signature (actual embedding object not required for loading)
            emb = HuggingFaceEmbeddings(model_name=config.HUGGINGFACE_EMBEDDING_MODEL)
            store = FAISS.load_local(str(config.VECTORSTORE_DIR), emb)
            return store
        except Exception:
            return None
    return None


def build_vectorstore(recreate: bool = False):
    """Build or load a FAISS vectorstore with OpenAI embeddings and cache it locally."""
    store = None
    if not recreate:
        store = _load_cached_vectorstore()
    if store:
        return store

    # Load data and create documents
    df = data_loader.load_and_preprocess()
    docs = data_loader.df_to_documents(df)
    lc_docs = [Document(page_content=d["content"], metadata=d["metadata"]) for d in docs]

    # Embeddings (sentence-transformers via local HF library)
    embeddings = HuggingFaceEmbeddings(model_name=config.HUGGINGFACE_EMBEDDING_MODEL)

    # Prepare checkpoint
    checkpoint_path = Path(config.VECTORSTORE_DIR) / "embedding_checkpoint.json"
    processed_hashes = set()
    if checkpoint_path.exists() and not recreate:
        try:
            processed_hashes = set(json.loads(checkpoint_path.read_text(encoding="utf-8")))
        except Exception:
            processed_hashes = set()

    # Build or append to FAISS index in batches
    store = None
    batch_size = max(1, int(config.EMBEDDING_BATCH_SIZE))
    pending = []
    ids = []
    # If there is an existing index load it to append
    if (Path(config.VECTORSTORE_DIR) / "index.faiss").exists() and not recreate:
        try:
            store = FAISS.load_local(str(config.VECTORSTORE_DIR), embeddings)
        except Exception:
            store = None

    for doc in tqdm(lc_docs, desc="Preparing documents"):
        content = doc.page_content
        h = hashlib.sha1(content.encode("utf-8")).hexdigest()
        if h in processed_hashes:
            continue
        pending.append(content)
        ids.append({"hash": h, "metadata": doc.metadata})

        if len(pending) >= batch_size:
            # build Document objects for this batch
            batch_docs = [Document(page_content=c, metadata=m["metadata"]) for c, m in zip(pending, ids)]
            if store is None:
                store = FAISS.from_documents(batch_docs, embeddings)
            else:
                # add_texts expects texts and metadatas
                store.add_texts([d.page_content for d in batch_docs], metadatas=[d.metadata for d in batch_docs])

            # mark processed
            for p in ids:
                processed_hashes.add(p["hash"])

            # persist checkpoint and store
            checkpoint_path.write_text(json.dumps(list(processed_hashes)), encoding="utf-8")
            store.save_local(str(config.VECTORSTORE_DIR))

            # reset buffers
            pending = []
            ids = []

    # process remaining
    if pending:
        batch_docs = [Document(page_content=c, metadata=m["metadata"]) for c, m in zip(pending, ids)]
        if store is None:
            store = FAISS.from_documents(batch_docs, embeddings)
        else:
            store.add_texts([d.page_content for d in batch_docs], metadatas=[d.metadata for d in batch_docs])
        for p in ids:
            processed_hashes.add(p["hash"])
        checkpoint_path.write_text(json.dumps(list(processed_hashes)), encoding="utf-8")
        store.save_local(str(config.VECTORSTORE_DIR))

    return store


def get_retriever(k: int = 5, recreate: bool = False):
    store = build_vectorstore(recreate=recreate)
    return store.as_retriever(search_kwargs={"k": k})
