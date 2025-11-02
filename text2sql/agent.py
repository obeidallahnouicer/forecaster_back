import time
from typing import Dict, Any
from .model_loader import ModelLoader
from .db import DBConnection
from .schema_inspector import load_schema
from .validator import validate_and_autocorrect
from .logger import logger
from .config import settings


class Chat2DBQueryAgent:
    """Orchestrates text->SQL generation, validation, and execution using Chat2DB model.

    This agent uses the Chat2DB-SQL-7B model (or mock) for text-to-SQL conversion.
    It handles schema loading, prompt building, SQL generation, validation, and execution.

    Usage:
        agent = Chat2DBQueryAgent(db_url)
        results = agent.generate_and_run(question)
    """

    def __init__(self, db_url: str | None = None, model_loader: ModelLoader | None = None):
        self.db = DBConnection(db_url or settings.DATABASE_URL)
        self.model_loader = model_loader or ModelLoader()
        self.model = None

    def _ensure_model(self):
        if self.model is None:
            self.model = self.model_loader.model

    def _build_prompt(self, question: str, schema: Dict[str, list]) -> str:
        # Build a concise prompt including schema and strict instructions
        schema_lines = []
        for t, cols in schema.items():
            schema_lines.append(f"TABLE {t}: {', '.join(cols)}")
        schema_text = "\n".join(schema_lines)

        prompt = (
            "### Database Schema\n\n"
            f"{schema_text}\n\n"
            "### Instruction:\n"
            "Answer ONLY with valid SQL inside <SQL>...</SQL> markers. Use standard SQL. No commentary.\n\n"
            f"Question: {question}\n"
        )
        return prompt

    def generate_and_run(self, question: str, retries: int = 3, backoff: float = 1.0) -> Dict[str, Any]:
        """Full flow: generate SQL, validate/autocorrect, execute, return structured response."""
        schema = load_schema(self.db)
        prompt = self._build_prompt(question, schema)
        self._ensure_model()

        last_issues = []
        generated_sql = None

        for attempt in range(1, retries + 1):
            logger.info("Generation attempt %s for question: %s", attempt, question)
            raw = self.model.generate_sql(prompt)
            logger.debug("Raw model output: %s", raw)

            # extract sql between markers
            import re

            m = re.search(r"<SQL>(.*?)</SQL>", raw, flags=re.S | re.I)
            if m:
                sql = m.group(1).strip()
            else:
                # fallback: first line
                sql = raw.strip().splitlines()[0]

            corrected_sql, issues = validate_and_autocorrect(sql, schema)
            logger.info("Validation issues: %s", issues)
            last_issues = issues

            if not issues:
                generated_sql = corrected_sql
                break

            # If only autocorrections are present, accept corrected SQL
            if all(i.startswith("Autocorrected") or i.startswith("Autocorrected") for i in issues):
                generated_sql = corrected_sql
                break

            # otherwise retry after a backoff
            logger.warning("SQL validation failed, retrying after %.1fs", backoff)
            time.sleep(backoff)
            backoff *= 2

        if generated_sql is None:
            raise RuntimeError(f"Unable to produce valid SQL after {retries} attempts. Last issues: {last_issues}")

        logger.info("Executing SQL: %s", generated_sql)
        rows = self.db.execute_select(generated_sql)

        return {
            "question": question,
            "sql": generated_sql,
            "issues": last_issues,
            "rows": rows,
        }
