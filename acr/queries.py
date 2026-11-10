"""Query pool: three families that need different amounts of compute.
- gsm8k: real GSM8K test problems (multi-step math)
- fact: questions about a SYNTHETIC fictional knowledge base (unanswerable without retrieval)
- arith: one-step arithmetic generated from a seed
The fictional KB and arithmetic items are synthetic; model outcomes on them are real runs."""
from __future__ import annotations

import random

SYL = ["ta", "ve", "lo", "ri", "mar", "keth", "an", "sol", "dru", "nix", "por", "ela", "quin", "zor", "bel", "ith"]
FIELDS = ["glassblowing", "tidal mapping", "beekeeping", "clockmaking", "salt farming", "kite design"]


def _name(rng, n=3):
    return "".join(rng.choice(SYL) for _ in range(n)).capitalize()


def knowledge_base(n: int = 60, seed: int = 11) -> list[dict]:
    rng = random.Random(seed)
    used, kb = set(), []
    while len(kb) < n:
        country, capital, founder = _name(rng), _name(rng, 2), f"{_name(rng, 2)} {_name(rng, 2)}"
        if country in used:
            continue
        used.add(country)
        year, field = rng.randint(1100, 1950), rng.choice(FIELDS)
        doc = (f"{country} is a small fictional nation known for {field}. Its capital city is {capital}. "
               f"{country} was founded in {year} by {founder}.")
        kind = rng.choice(["capital", "year", "founder"])
        q, a = {"capital": (f"What is the capital city of {country}?", capital),
                "year": (f"In what year was {country} founded?", str(year)),
                "founder": (f"Who founded {country}?", founder)}[kind]
        kb.append({"doc": doc, "question": q, "answer": a, "kind": kind})
    return kb


def arithmetic(n: int = 60, seed: int = 13) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        a, b = rng.randint(2, 99), rng.randint(2, 99)
        op = rng.choice(["+", "-", "*"])
        val = {"+": a + b, "-": a - b, "*": a * b}[op]
        out.append({"question": f"What is {a} {op} {b}?", "answer": str(val)})
    return out


def pool(n_each: int = 60, seed: int = 20261110) -> list[dict]:
    from datasets import load_dataset
    from .gsm8k import gold_answer
    ds = load_dataset("openai/gsm8k", "main", split="test")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    q = [{"id": f"gsm8k-{i}", "family": "gsm8k", "question": ds[i]["question"], "answer": gold_answer(ds[i]["answer"]),
          "numeric": True} for i in idx[-n_each:]]  # tail of the shuffle: no overlap with rlvr-vs-sft-vs-harness subset
    q += [{"id": f"fact-{j}", "family": "fact", "question": x["question"], "answer": x["answer"], "numeric": x["kind"] == "year"}
          for j, x in enumerate(knowledge_base(n_each))]
    q += [{"id": f"arith-{j}", "family": "arith", "question": x["question"], "answer": x["answer"], "numeric": True}
          for j, x in enumerate(arithmetic(n_each))]
    return q
