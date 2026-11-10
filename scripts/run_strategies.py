"""Run all 5 strategies on every query with real models. Writes results/outcomes.jsonl."""
import argparse, json
from pathlib import Path

from acr import grading, queries, strategies as S
from acr.model import BIG, SMALL, generate, load, non_embedding_params
from acr.retrieval import Retriever

ap = argparse.ArgumentParser()
ap.add_argument("--n-each", type=int, default=60)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--out", default="results/outcomes.jsonl")
a = ap.parse_args()

Q = queries.pool(a.n_each)
kb = queries.knowledge_base(a.n_each)
ret = Retriever([x["doc"] for x in kb])
res = {q["id"]: {"id": q["id"], "family": q["family"], "question": q["question"], "answer": q["answer"]} for q in Q}


def per_item(model, tok, convs, max_new):
    """Generate one by one in batches but keep per-item token counts."""
    outs, toks = [], []
    for i in range(0, len(convs), a.batch):
        o, _, _ = generate(model, tok, convs[i:i + a.batch], max_new, a.batch)
        for c, text in zip(convs[i:i + a.batch], o):
            p = len(tok.apply_chat_template(c, tokenize=True, add_generation_prompt=True))
            if not isinstance(p, int):
                p = len(p["input_ids"]) if isinstance(p, dict) else len(p)
            toks.append(p + len(tok(text)["input_ids"]) + 1)
            outs.append(text)
    return outs, toks


model, tok = load(SMALL)
n_small = non_embedding_params(model)
qs = [q["question"] for q in Q]
runs = {
    "direct": per_item(model, tok, [S.msgs(S.DIRECT_SYS, q) for q in qs], 32),
    "cot": per_item(model, tok, [S.msgs(S.COT_SYS, q) for q in qs], 320),
    "retrieval": per_item(model, tok, [S.msgs(S.RAG_SYS, S.rag_prompt(q, ret.top(q))) for q in qs], 32),
    "pot": per_item(model, tok, [S.msgs(S.POT_SYS, q) for q in qs], 200),
}
del model
import gc, torch
gc.collect()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
big, btok = load(BIG)
n_big = non_embedding_params(big)
runs["bigger"] = per_item(big, btok, [S.msgs(S.COT_SYS, q) for q in qs], 320)

for i, q in enumerate(Q):
    r = res[q["id"]]
    for name in ["direct", "cot", "retrieval"]:
        text, t = runs[name][0][i], runs[name][1][i]
        r[name] = {"correct": grading.grade(text, q["answer"], q["numeric"]), "tokens": t, "flops": 2 * n_small * t,
                   "tail": text[-160:]}
    vtext, route = S.verifier_combine(runs["cot"][0][i], runs["pot"][0][i])
    vt = runs["cot"][1][i] + runs["pot"][1][i]
    r["verifier"] = {"correct": grading.grade(vtext, q["answer"], q["numeric"]), "tokens": vt, "flops": 2 * n_small * vt,
                     "route": route, "tail": vtext[-160:]}
    text, t = runs["bigger"][0][i], runs["bigger"][1][i]
    r["bigger"] = {"correct": grading.grade(text, q["answer"], q["numeric"]), "tokens": t, "flops": 2 * n_big * t,
                   "tail": text[-160:]}
Path(a.out).write_text("\n".join(json.dumps(res[q["id"]]) for q in Q) + "\n")
meta = {"small": SMALL, "big": BIG, "n_small_nonemb": n_small, "n_big_nonemb": n_big, "n_each": a.n_each,
        "retriever": "TF-IDF (1-2 grams), top-2 passages"}
Path("results/outcomes_meta.json").write_text(json.dumps(meta, indent=2))
for s in S.STRATEGIES:
    acc = sum(res[q["id"]][s]["correct"] for q in Q) / len(Q)
    print(s, round(acc, 3), sum(res[q["id"]][s]["flops"] for q in Q) / len(Q) / 1e9, "GFLOPs/q")
