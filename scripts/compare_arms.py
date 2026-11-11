"""Pre-registered comparison of two arms on the frozen dev-val list (H001/E006 analysis plan, prereg #3).

    python scripts/compare_arms.py --tasks outputs/E006/devval_tasks.jsonl --a outputs/E005/q8b_nothink_none --b outputs/E006/student_s1_none [outputs/E006/student_s2_none] [--subset ids.txt] [--seed 0]

Arm B may be several run dirs (trainer seeds): each task's B outcome is the mean over seeds (0, 0.5, 1). Primary statistic: paired
difference in strict success (B - A, percentage points) with a CLUSTER bootstrap over target-element clusters (the task's first target
GlobalId), stratified by project, 10,000 resamples, percentile CI, fixed RNG seed; exact McNemar on discordant pairs as cross-check;
per-project differences; task-level bootstrap and design effect reported for reference. Decision rule (prereg): supported if the point
estimate >= 10.0 and the cluster-CI lower bound > 0 and both projects' differences have the same sign and McNemar p < 0.05;
falsified if the point estimate < 5.0 and the cluster-CI upper bound < 10.0; otherwise inconclusive.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os

import numpy as np


def load_outcomes(run_dirs: list[str], ids: set[str]) -> dict[str, float]:
    per = collections.defaultdict(list)
    for rd in run_dirs:
        for p in glob.glob(os.path.join(rd, "trajectories", "*.json")):
            try:
                s = json.load(open(p))["summary"]
            except Exception:  # noqa: BLE001
                continue
            if s["task_id"] in ids:
                per[s["task_id"]].append(1.0 if str(s.get("strict")) == "True" else 0.0)   # older summaries hold the string "False" (numpy bool via default=str)
    return {t: float(np.mean(v)) for t, v in per.items()}


def cluster_of(t: dict) -> str:
    """Target-element cluster: the first target GlobalId; a compound task (intent 'all', E009 prereg) uses its FIRST part's element."""
    if t["intent"].get("type") == "all":
        p0 = t["intent"]["parts"][0]
        for k in ("guid", "host", "storey"):
            if p0.get(k):
                return p0[k]
        if p0.get("guids"):
            return p0["guids"][0]
    return t["target"][0]


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c); p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True); ap.add_argument("--a", nargs="+", required=True); ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--sign-by", choices=["project", "file"], default="project", help="E009: 'file' = the same-sign rule over the --tasks files "
                    "(real / held-out synthetic compound dev-val) instead of per project; the bootstrap stays stratified by project")
    ap.add_argument("--subset", default=None); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--resamples", type=int, default=10000)
    a = ap.parse_args()
    tasks = {}; src = {}
    for f_ in a.tasks:
        for l in open(f_):
            if l.strip():
                t_ = json.loads(l); tasks[t_["task_id"]] = t_; src[t_["task_id"]] = os.path.basename(f_)
    if a.subset:
        keep = {l.strip() for l in open(a.subset) if l.strip()}; tasks = {k: v for k, v in tasks.items() if k in keep}
    A = load_outcomes(a.a, set(tasks)); B = load_outcomes(a.b, set(tasks))
    ids = sorted(t for t in tasks if t in A and t in B)
    missing = len(tasks) - len(ids)
    x = np.array([A[t] for t in ids]); y = np.array([B[t] for t in ids]); d = y - x
    clusters = np.array([cluster_of(tasks[t]) for t in ids]); projects = np.array([tasks[t]["project"] for t in ids]); tiers = np.array([tasks[t]["tier"] for t in ids])
    est = 100 * d.mean()
    rng = np.random.default_rng(a.seed)
    # cluster bootstrap stratified by project
    cl_by_proj = {p: sorted(set(clusters[projects == p])) for p in sorted(set(projects))}
    idx_by_cl = {c: np.where(clusters == c)[0] for c in set(clusters)}
    boots = []
    for _ in range(a.resamples):
        sel = []
        for p, cls in cl_by_proj.items():
            pick = rng.choice(cls, size=len(cls), replace=True)
            for c in pick:
                sel.append(idx_by_cl[c])
        s = np.concatenate(sel); boots.append(100 * d[s].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    tb = [100 * d[rng.choice(len(d), len(d), replace=True)].mean() for _ in range(a.resamples)]
    tlo, thi = np.percentile(tb, [2.5, 97.5])
    deff = (np.var(boots) / np.var(tb)) if np.var(tb) > 0 else float("nan")
    bc = int(((y > x)).sum()); cc = int(((y < x)).sum()); p_mc = mcnemar_exact(bc, cc)
    per_proj = {p: 100 * float(d[projects == p].mean()) for p in cl_by_proj}
    per_tier = {int(t): {"n": int((tiers == t).sum()), "A": 100 * float(x[tiers == t].mean()), "B": 100 * float(y[tiers == t].mean()), "diff": 100 * float(d[tiers == t].mean())} for t in sorted(set(tiers))}
    groups = np.array([src[t] for t in ids]) if a.sign_by == "file" else projects
    per_group = {g: 100 * float(d[groups == g].mean()) for g in sorted(set(groups))}
    same_sign = all((v > 0) == (est > 0) for v in per_group.values()) if est != 0 else False
    if est >= 10.0 and lo > 0 and same_sign and p_mc < 0.05:
        verdict = "supported"
    elif est < 5.0 and hi < 10.0:
        verdict = "falsified"
    else:
        verdict = "inconclusive"
    out = {"n": len(ids), "missing": missing, "A_strict": 100 * float(x.mean()), "B_strict": 100 * float(y.mean()), "diff": est,
           "cluster_ci95": [float(lo), float(hi)], "task_ci95": [float(tlo), float(thi)], "design_effect": float(deff), "n_clusters": len(idx_by_cl),
           "discordant_B_only": bc, "discordant_A_only": cc, "mcnemar_p": p_mc, "per_project_diff": per_proj, "sign_by": a.sign_by, "per_sign_group_diff": per_group,
           "per_file": {g: {"n": int((np.array([src[t] for t in ids]) == g).sum()), "A": 100 * float(x[np.array([src[t] for t in ids]) == g].mean()),
                            "B": 100 * float(y[np.array([src[t] for t in ids]) == g].mean())} for g in sorted({src[t] for t in ids})}, "per_tier": per_tier, "verdict": verdict,
           "rule": "supported: diff>=10 and cluster-CI low>0 and same sign per project (or per --tasks file with --sign-by file) and McNemar p<0.05; falsified: diff<5 and cluster-CI high<10; else inconclusive"}
    print(json.dumps(out, indent=1))
    print(f"metric=strict_diff {est:.2f}")
    print(f"metric=verdict {verdict}")


if __name__ == "__main__":
    main()
