"""E012 RL pool from the k = 4 pre-filter (user 2026-10-24: drop tasks the start policy solves 4/4, keep 1-3/4, cap 0/4 at ~20 %;
no online filtering). Inputs: the pilot and the pre-filter runs of the start policy (run_tasks --repeats 4, ids '<task>@<i>').
    python train/rl/build_pool.py   -> outputs/E012/rl_pool_final.jsonl + rl_pool_card.json"""
import collections
import glob
import json
import os
import random

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
O = os.path.join(C, "outputs/E012")
ZERO_CAP = 0.20


def grp(tp: str) -> str:
    return tp.rsplit("-", 1)[0] if tp[-2:] in ("-D", "-T", "-S", "-P", "-V", "-C") else tp


def main():
    tasks = {json.loads(l)["task_id"]: json.loads(l) for l in open(os.path.join(O, "rl_candidates.jsonl"))}
    res = collections.defaultdict(dict)
    for f in glob.glob(os.path.join(O, "pilot/results*.jsonl")) + glob.glob(os.path.join(O, "prefilter/results*.jsonl")):
        for l in open(f):
            r = json.loads(l); base, i = r["task_id"].rsplit("@", 1)
            res[base][i] = bool(r["strict"])           # duplicates (resumed runs) collapse on the sample index
    full = {k: sum(v.values()) for k, v in res.items() if len(v) == 4}
    hist = collections.Counter(full.values())
    mixed = [k for k, s in full.items() if 0 < s < 4]; zero = sorted(k for k, s in full.items() if s == 0)
    random.Random(0).shuffle(zero)
    n_zero = min(len(zero), int(ZERO_CAP / (1 - ZERO_CAP) * len(mixed)))
    keep = sorted(mixed + zero[:n_zero])
    pool = [dict(tasks[k], rl_k4=full[k]) for k in keep]
    open(os.path.join(O, "rl_pool_final.jsonl"), "w").writelines(json.dumps(t) + "\n" for t in pool)
    card = {"candidates": len(tasks), "scored_k4": len(full), "incomplete": len(res) - len(full), "k4_hist": {str(k): hist[k] for k in range(5)},
            "kept": len(pool), "kept_mixed": len(mixed), "kept_zero": n_zero, "zero_available": len(zero), "zero_cap": ZERO_CAP,
            "templates": dict(collections.Counter(grp(t["template"]) for t in pool).most_common()),
            "sources": dict(collections.Counter(t.get("rl_source") for t in pool))}
    json.dump(card, open(os.path.join(O, "rl_pool_card.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in card.items() if k != "templates"}))


if __name__ == "__main__":
    main()
