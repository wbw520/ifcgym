"""RL prompt pool for GRPO: a stratified sample over
buildings x template x reference style, with tier-2/3 and topological/spatial/path references up-weighted, excluding the teacher
list and the dev-val buildings. The rest of the generated tasks stays on disk as a reserve for pass-rate filtering.
    python scripts/sample_rl_pool.py --tasks 'outputs/tasks_synth2k_v7/*.jsonl' --exclude outputs/<run>/synth_teacher_tasks.jsonl --out outputs/<run>/rl_pool.jsonl --n 30000 --seed 0"""
import argparse, collections, glob, json, random

def _body(pid):
    base = pid[:-7] if pid.endswith("_sparse") else pid
    for k in ("brep", "clipped", "faceset"):
        if base.endswith("_" + k):
            return k
    return "extrusion"

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", required=True); ap.add_argument("--exclude", nargs="*", default=[]); ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=30000); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); rng = random.Random(a.seed)
skip = set()
for p in a.exclude:
    skip |= {json.loads(l)["task_id"] for l in open(p) if l.strip()}
rows = []
for p in sorted(glob.glob(a.tasks)):
    for l in open(p):
        if l.strip():
            t = json.loads(l)
            if t["task_id"] not in skip:
                rows.append(t)
def weight(t):
    w = 1.0
    if t["tier"] >= 2: w *= 3.0
    st = t.get("meta", {}).get("style")
    if st in ("T", "S", "P"): w *= 2.0
    if t["template"].split("-")[0] in ("A",): w *= 2.0
    return w
# stratify: cap per building so every building contributes, then weighted sampling inside the cap
by_b = collections.defaultdict(list)
for t in rows:
    by_b[t["project"]].append(t)
cap = max(1, a.n // max(1, len(by_b)) + 2)
out = []
for b, ts in sorted(by_b.items()):
    ws = [weight(t) for t in ts]; pick = []
    idx = list(range(len(ts)))
    for _ in range(min(cap, len(ts))):
        j = rng.choices(idx, weights=[ws[i] for i in idx])[0]; pick.append(ts[j]); idx.remove(j)
    out += pick
rng.shuffle(out); out = out[: a.n]
with open(a.out, "w") as f:
    for t in out:
        f.write(json.dumps(t) + "\n")
c = collections.Counter(t["tier"] for t in out); s = collections.Counter(t.get("meta", {}).get("style", "D") for t in out); fam = collections.Counter(t["template"].split("-")[0] for t in out)
body = collections.Counter(_body(t["project"]) for t in out)
print(f"{len(rows)} candidates in {len(by_b)} buildings -> pool {len(out)}: tiers {dict(sorted(c.items()))}, styles {dict(s)}, families {dict(fam)}, bodies {dict(body)}")
missing = [(tpl, b) for tpl in {t["template"] for t in rows} for b in ("extrusion", "brep", "clipped") if not any(t["template"] == tpl and (_body(t["project"]) == b) for t in out)]
print("template x body combinations absent from the pool:", len(missing), missing[:10])
