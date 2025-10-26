"""Sample runner demonstrating how to load the real Chat2DB model and run a prompt.

This script is optional and will only run if `transformers` and `torch` are installed
and the environment has sufficient resources. It is safe: it prints the generated SQL
and does not execute it against any database.
"""
from text2sql.model_loader import ModelLoader
from text2sql.config import settings
from text2sql.logger import logger


def main():
    # This will attempt to load the real model unless MODEL_USE_MOCK is set.
    loader = ModelLoader()
    try:
        model = loader.load()
    except Exception as exc:
        logger.exception("Unable to load real model: %s", exc)
        print("Real model not available in this environment. Set MODEL_USE_MOCK=1 to use the mock model.")
        return

    prompt = (
        "### Database Schema\nTABLE clients: Code_Client, Intitule_Client, CA_Moyen_Annuel\n\n"
        "### Instruction:\nWrite ONLY SQL inside <SQL>...</SQL>.\nQuestion: How many clients are inactive for more than 3 months?"
    )

    out = model.generate_sql(prompt)
    print("Model output:")
    print(out)


if __name__ == '__main__':
    main()
