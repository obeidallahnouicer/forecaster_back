"""Simple token budget monitor for requests.

Tracks estimated token usage per request and enforces a budget by
indicating when context should be summarized or trimmed.
"""
import logging
from typing import List

logger = logging.getLogger("llm.token_manager")


class TokenBudget:
    def __init__(self, budget: int = 2000):
        self.budget = budget
        self.used = 0

    def estimate_and_add(self, texts: List[str]) -> int:
        """Estimate tokens for a list of text chunks and add to used.

        Estimation heuristic: 1 token ~= 4 characters.
        """
        est = sum(max(1, len(t) // 4) for t in texts)
        self.used += est
        logger.debug(f"TokenBudget: added {est} tokens, now used={self.used}/{self.budget}")
        return est

    def remaining(self) -> int:
        return max(0, self.budget - self.used)

    def exceeded(self) -> bool:
        return self.used >= self.budget

    def reset(self):
        self.used = 0

