from typing import Optional
from dataclasses import dataclass
import os
from .config import settings
from .logger import logger


@dataclass
class MockModel:
    """A tiny mock model used in tests and CI to avoid loading a real LLM.

    The mock returns predictable SQL when asked; it's deterministic and safe.
    """

    def generate_sql(self, prompt: str) -> str:
        # Heuristics to map common French/English questions from the TABLE Chatbot.md
        q = prompt.lower()

        # How many / Count questions
        if any(x in q for x in ("combien", "how many", "count(")) and "client" in q or "clients" in q:
            return "<SQL>SELECT COUNT(*) AS count FROM clients;</SQL>"

        # Inactive clients (clients inactifs depuis X mois)
        if "inactif" in q or "inactifs" in q or "inactives" in q:
            # Return clients with Mois_Depuis_Derniere_Vente >= 3
            return (
                "<SQL>SELECT Code_Client, Intitule_Client, Mois_Depuis_Derniere_Vente "
                "FROM clients WHERE Mois_Depuis_Derniere_Vente >= 3;</SQL>"
            )

        # Average revenue for a client (CA moyen)
        if "chiffre d'affaires moyen" in q or "ca moyen" in q or "chiffre d’affaires moyen" in q:
            # Attempt to extract client name from prompt
            import re

            m = re.search(r"client\s+([A-Z\-\_\w\s]+)\b", prompt, flags=re.I)
            if m:
                client = m.group(1).strip()
                # Use Intitule_Client lookup
                return f"<SQL>SELECT CA_Moyen_Annuel FROM clients WHERE Intitule_Client = '{client}';</SQL>"
            return "<SQL>SELECT Code_Client, CA_Moyen_Annuel FROM clients LIMIT 10;</SQL>"

        # Clients at risk of churn / perte (chute > 20%)
        if "risque" in q and "perte" in q or "chut" in q and "%" in q or "-20" in q or "20 %" in q:
            # Use a safe column name Variation_CA_Percent
            return (
                "<SQL>SELECT Code_Client, Intitule_Client, Variation_CA_Percent "
                "FROM clients WHERE Variation_CA_Percent <= -20;</SQL>"
            )

        # Default: simple select to keep tests predictable
        return "<SQL>SELECT Code_Client FROM clients LIMIT 5;</SQL>"


class ModelLoader:
    """Responsible for loading the Chat2DB-SQL model (or a mock) lazily.

    The real loader uses transformers + torch but that is only invoked when
    load() is called, so importing this module is safe during unit tests.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or settings.MODEL_PATH
        self._model = None

    def load(self):
        """Load the model. If settings.MODEL_USE_MOCK is true, return a MockModel.

        The real model is loaded with transformers and configured for GPU if available.
        """
        if settings.MODEL_USE_MOCK or os.environ.get("MODEL_USE_MOCK") == "1":
            logger.info("Using MockModel (MODEL_USE_MOCK=True)")
            self._model = MockModel()
            return self._model

        # Lazy import heavy packages
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        except Exception as exc:
            logger.exception("Failed to import transformers/torch: %s", exc)
            raise

        device = 0 if torch.cuda.is_available() else -1
        logger.info("Loading model '%s' on device %s", self.model_path, "GPU" if device == 0 else "CPU")

        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto" if device == 0 else {"": "cpu"},
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            use_cache=True,
        )

        # Create a simple text-generation wrapper
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            return_full_text=False,
        )

        # Wrap the pipeline to expose generate_sql(prompt)
        class RealModelWrapper:
            def __init__(self, pipe, tokenizer):
                self.pipe = pipe
                self.tokenizer = tokenizer

            def generate_sql(self, prompt: str) -> str:
                out = self.pipe(prompt, max_new_tokens=settings.MODEL_MAX_NEW_TOKENS, do_sample=False)
                return out[0]["generated_text"]

        self._model = RealModelWrapper(pipe, tokenizer)
        return self._model

    @property
    def model(self):
        if self._model is None:
            return self.load()
        return self._model
