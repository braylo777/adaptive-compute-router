"""Router vs fixed strategies vs oracle, from results/outcomes.jsonl. Writes results/frontier.json + REPORT.md."""
import json
from pathlib import Path

import numpy as np

from acr import router
from acr.strategies import STRATEGIES

rows = [json.loads(l) for l in Path("results/outcomes.jsonl").read_text().splitlines() if l.strip()]
Q = [r["question"] for r in rows]
fam = [r["family"] for r in rows]
C = np.array([[r[s]["correct"] for s in STRATEGIES] for r in rows], dtype=bool)
F = np.array([[r[s]["flops"] for s in STRATEGIES] for r in rows], dtype=float)
P = router.oof_success_probs(Q, C, fam)
mean_cost = F.mean(0) / F.mean(0).max()  # router sees expected cost per strategy, not the realized cost
fixed = {s: (float(F[:, i].mean()), float(C[:, i].mean())) for i, s in enumerate(STRATEGIES)}
sweep = []
for lam in np.concatenate([[0.0], np.geomspace(1e-3, 10, 40)]):
    acc, cost = router.evaluate(router.route(P, mean_cost, lam), C, F)
    sweep.append({"lam": float(lam), "flops_per_q": cost, "accuracy": acc})
oracle_choice = np.where(C.any(1), np.argmin(np.where(C, F, np.inf), axis=1), np.argmin(F, axis=1))
oracle = router.evaluate(oracle_choice, C, F)
front = router.pareto_front([(s["flops_per_q"], s["accuracy"]) for s in sweep])
out = {"n": len(rows), "fixed": fixed, "router_sweep": sweep, "router_front": front,
       "oracle": {"accuracy": oracle[0], "flops_per_q": oracle[1]}}
Path("results/frontier.json").write_text(json.dumps(out, indent=2))
md = ["| policy | accuracy | GFLOPs / query |", "|---|---|---|"]
md += [f"| fixed: {s} | {100*a:.1f}% | {c/1e9:.1f} |" for s, (c, a) in fixed.items()]
md += [f"| router (front point) | {100*a:.1f}% | {c/1e9:.1f} |" for c, a in front]
md.append(f"| oracle (cheapest correct) | {100*oracle[0]:.1f}% | {oracle[1]/1e9:.1f} |")
Path("results/REPORT.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
