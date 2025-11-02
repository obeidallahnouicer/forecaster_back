"""CLI for running batch queries against a database using the QueryAgent.

Example:
    python -m text2sql.cli --query "How many singers do we have?"
    python -m text2sql.cli --file queries.txt --db sqlite:///./data.db
"""
import argparse
from .agent import Chat2DBQueryAgent
from .logger import logger


def main():
    p = argparse.ArgumentParser(prog="text2sql-cli")
    p.add_argument("--db", help="SQLAlchemy DB URL", default=None)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--query", help="Single natural language query")
    g.add_argument("--file", help="File with one query per line")
    args = p.parse_args()

    agent = Chat2DBQueryAgent(db_url=args.db)

    queries = []
    if args.query:
        queries = [args.query]
    else:
        with open(args.file, "r", encoding="utf-8") as fh:
            queries = [l.strip() for l in fh if l.strip()]

    for q in queries:
        try:
            res = agent.generate_and_run(q)
            logger.info("Result for query: %s -> rows=%s", q, len(res.get("rows", [])))
            print(res)
        except Exception as exc:
            logger.exception("Failed to process query: %s", q)
            print({"error": str(exc)})


if __name__ == "__main__":
    main()
