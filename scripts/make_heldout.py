"""Held-out test list (the 1,500-task core of IfcEditBench): covering as much as possible, aligned with the training data types.
Allocation follows the composition of the student's SFT training set (element/operation kinds of train/build_e010_mix.py):
  1. kind quota = n x the kind's training share, at least --floor tasks, never more than the kind has available (water-filling);
  2. within a kind, variant quota in proportion to the variant's training count, at least 1 per available variant (coverage);
  3. within a variant, tasks one at a time from the building with the fewest tasks so far (buildings balanced).
    python scripts/make_heldout.py --tasks 'outputs/heldout_gate/*/survivors.jsonl' --out heldout.jsonl --n 1500 --floor 30 --seed 0"""
import argparse, collections, glob, json, math, os, random, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))
from build_e010_mix import element, operation      # noqa: E402

TRAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs/sft/data/train.parquet")


def kind(tpl):
    return f"{element(tpl)}/{operation(tpl)}"


def waterfill(n, weight, avail, floor):
    """Integer quotas summing to min(n, sum(avail)): q_k = min(avail_k, max(floor_k, s * weight_k)), largest remainder rounding."""
    n = min(n, sum(avail.values())); lo_k = {k: min(floor, avail[k]) for k in avail}
    f = lambda s: {k: min(avail[k], max(lo_k[k], s * weight.get(k, 0))) for k in avail}
    lo, hi = 0.0, 1.0
    while sum(f(hi).values()) < n and hi < 1e12:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2; lo, hi = (mid, hi) if sum(f(mid).values()) < n else (lo, mid)
    x = f(hi); q = {k: int(math.floor(v)) for k, v in x.items()}
    for k in sorted(x, key=lambda k: (q[k] - x[k], k))[: n - sum(q.values())]:
        q[k] += 1
    return q


def quotas(n, avail_tpl, train_tpl, floor):
    """Per-variant quotas: kinds by training share (floor per kind), variants within a kind by training count (at least 1)."""
    kinds = collections.defaultdict(dict)
    for t, a in avail_tpl.items():
        kinds[kind(t)][t] = a
    tk = collections.Counter()
    for t, c in train_tpl.items():
        tk[kind(t)] += c
    qk = waterfill(n, tk, {k: sum(v.values()) for k, v in kinds.items()}, floor)
    q = {}
    for k, vs in kinds.items():
        q.update(waterfill(qk[k], {t: train_tpl.get(t, 0) for t in vs}, vs, 1))
    return q


def draw(rows, q, rng, count=None):
    """Take q[template] tasks per template; each from the building with the fewest tasks so far (ties at random)."""
    pool = collections.defaultdict(lambda: collections.defaultdict(list))
    for t in sorted(rows, key=lambda t: t["task_id"]):
        pool[t["template"]][t["project"]].append(t)
    for d in pool.values():
        for ts in d.values():
            rng.shuffle(ts)
    count = collections.Counter() if count is None else count; out = []
    for tpl in sorted(q, key=lambda t: (q[t], t)):        # scarce templates first, so balance is spent where it is free
        for _ in range(q[tpl]):
            cand = [p for p in pool[tpl] if pool[tpl][p]]; lo = min(count[p] for p in cand); p = rng.choice([p for p in cand if count[p] == lo])
            out.append(pool[tpl][p].pop()); count[p] += 1
    return out


def train_templates():
    import pandas as pd
    return collections.Counter(pd.read_parquet(TRAIN, columns=["template"]).template)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tasks", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--floor", type=int, default=15)
    a = ap.parse_args(); rng = random.Random(a.seed)
    rows = [json.loads(l) for p in sorted(glob.glob(a.tasks)) for l in open(p) if l.strip()]
    assert len({t["task_id"] for t in rows}) == len(rows), "duplicate task ids"
    tr = train_templates(); rows = [t for t in rows if t["template"] in tr]       # only types the student was trained on
    q = quotas(a.n, collections.Counter(t["template"] for t in rows), tr, a.floor)
    out = sorted(draw(rows, q, rng), key=lambda t: t["task_id"])
    with open(a.out, "w") as f:
        for t in out:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    n_tr = sum(tr.values()); tk = collections.Counter()
    for t, c in tr.items():
        tk[kind(t)] += c
    ok = collections.Counter(kind(t["template"]) for t in out)
    print(f"{len(rows)} candidates -> {len(out)} tasks; variants {len({t['template'] for t in out})} of {len(tr)} trained; projects "
          f"{dict(sorted(collections.Counter(t['project'] for t in out).items()))}")
    for k in sorted(set(tk) | set(ok), key=lambda k: -tk[k]):
        print(f"   {k:20s} train {100 * tk[k] / n_tr:5.1f} %   test {ok[k]:4d} ({100 * ok[k] / max(1, len(out)):5.1f} %)")


if __name__ == "__main__":
    main()
