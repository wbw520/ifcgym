"""Score a featurised run with the calibrated BIM-Edit scorer config. Usage: score_run.py <feat_dir> <calib.json> [out.jsonl]"""
import json, sys, os, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ifcedit.bimedit_calib import load_feats, score_from_features
feats = load_feats(sys.argv[1]); cfg = json.load(open(sys.argv[2]))["best"]; cfg["rels"] = tuple(cfg["rels"])
rows = [score_from_features(F, cfg) for F in feats.values()]
tasks = {json.loads(l)["task_id"]: json.loads(l) for l in open("${IFCGYM_ROOT}/data/bimedit-runs/tasks.jsonl")}
for r in rows:
    r["operation"] = tasks[r["task_id"]]["operation"]; r["category"] = tasks[r["task_id"]]["category"]
if len(sys.argv) > 3:
    with open(sys.argv[3], "w") as f:
        for r in sorted(rows, key=lambda r: r["task_id"]): f.write(json.dumps(r) + "\n")
F = np.array([r["final_score"] for r in rows]); strict = np.mean([all(r[k] >= 0.98 for k in ("geometry", "semantics", "topology")) for r in rows])
boot = [np.mean(np.random.default_rng(s).choice(F, len(F))) for s in range(1000)]
print(f"n={len(rows)} Final={100*F.mean():.2f} (95% CI {100*np.percentile(boot,2.5):.2f}-{100*np.percentile(boot,97.5):.2f}) strict={100*strict:.1f}% geo={100*np.mean([r['geometry'] for r in rows]):.1f} sem={100*np.mean([r['semantics'] for r in rows]):.1f} topo={100*np.mean([r['topology'] for r in rows]):.1f}")
for key in ("category", "operation"):
    g = collections.defaultdict(list)
    for r in rows: g[r[key]].append(r["final_score"])
    print(" ", key, {k: round(100*float(np.mean(v)), 1) for k, v in sorted(g.items())})
print(f"metric=bimedit_final {100*F.mean():.2f}")
