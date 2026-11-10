"""The five compute strategies. Each returns (answer_text, tokens_processed) per query.
Cost = 2 x N_non_embedding x tokens (forward FLOPs), N per model."""
from __future__ import annotations

from . import calc

DIRECT_SYS = "Answer the question. Reply with only the final answer, no explanation."
COT_SYS = ("Solve the problem step by step. Finish with a final line of the form "
           "'The answer is X.'")
RAG_SYS = "Use the context to answer the question. Reply with only the final answer, no explanation."
POT_SYS = ("Solve the problem by writing a short Python program using only numbers, arithmetic and "
           "variable assignments, one per line. Store the final result in a variable named answer. Output only code.")

STRATEGIES = ["direct", "cot", "retrieval", "verifier", "bigger"]


def msgs(system: str, user: str) -> list[dict]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def rag_prompt(question: str, passages: list[str]) -> str:
    ctx = "\n".join(f"- {p}" for p in passages)
    return f"Context:\n{ctx}\n\nQuestion: {question}"


def pot_value(code: str) -> str | None:
    try:
        return calc.format_number(calc.run_program(code))
    except Exception:  # noqa: BLE001 - any failure of model-written code means "no tool answer"
        return None


def verifier_combine(cot_text: str, pot_code: str) -> tuple[str, str]:
    """CoT answer checked by a program-of-thought run through the safe calculator.
    Agree -> keep; disagree and tool ran -> trust the tool; tool failed -> keep CoT."""
    from .gsm8k import extract_answer
    c, p = extract_answer(cot_text), pot_value(pot_code)
    if p is None:
        return cot_text, "tool_failed"
    if c is not None and abs(float(c) - float(p)) < 1e-6:
        return cot_text, "agree"
    return f"The answer is {p}.", "tool_override"
