import re
from typing import List
from pathlib import Path


def load_example_questions(md_path: str) -> List[str]:
    """Load the markdown file and extract example questions contained in French guillemets « ... »

    Returns a list of stripped question strings.
    """
    p = Path(md_path)
    if not p.exists():
        raise FileNotFoundError(md_path)

    txt = p.read_text(encoding="utf-8")
    # Extract content inside French quotes « ... » and standard quotes
    q1 = re.findall(r'«([^»]+)»', txt)
    q2 = re.findall(r'"([^"]+)"', txt)
    q3 = re.findall(r"'([^']+)'", txt)

    # Merge and filter likely questions (contain ? or words like 'Quel', 'Quels', 'Combien')
    candidates = list(dict.fromkeys([q.strip() for q in (q1 + q2 + q3)]))
    questions = [q for q in candidates if any(w in q for w in ("?", "Quel", "Quels", "Combien", "Combien"))]
    return questions
