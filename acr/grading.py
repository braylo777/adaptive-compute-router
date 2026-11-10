"""Answer grading: numeric equality for numbers, normalized containment for names."""
from __future__ import annotations

import re

from .gsm8k import extract_answer


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def grade(text: str | None, gold: str, numeric: bool) -> bool:
    if not text:
        return False
    if numeric:
        pred = extract_answer(text)
        try:
            return pred is not None and abs(float(pred) - float(gold)) < 1e-6
        except ValueError:
            return False
    return norm(gold) in norm(text)
