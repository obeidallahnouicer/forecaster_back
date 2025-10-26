from typing import Dict, List
from .db import DBConnection
from .logger import logger


def load_schema(db: DBConnection) -> Dict[str, List[str]]:
    """Return table->columns mapping using the DBConnection inspector."""
    logger.info("Loading DB schema")
    return db.get_schema()
