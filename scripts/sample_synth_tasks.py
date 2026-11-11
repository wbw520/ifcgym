"""Sample the teacher task list for the synthetic buildings: per building k tier-2/3 tasks and m tier-0/1 tasks, so the
teacher budget (~6000 episodes) is spread over all buildings; the rest of the tasks stay as the RL prompt pool.
    python scripts/sample_synth_tasks.py --tasks 'outputs/tasks_synth2k/*.jsonl' --out outputs/<run>/synth_teacher_tasks.jsonl --k23 2 --k01 1 --seed 0
"""
import argparse, collections, glob, json, random

def _body(pid):
    base = pid[:-7] if pid.endswith("_sparse") else pid
    for k in ("brep", "clipped", "faceset"):
        if base.endswith("_" + k):
            return k
    return "extrusion"

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--k23", type=int, default=2); ap.add_argument("--k01", type=int, default=1); ap.add_argument("--kx", type=int, default=1, help="create/retrieve/spatial/update/delete/concept tasks per building (families in rotation)"); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = random.Random(a.seed)
FAMS = ["C", "R", "S", "U", "D", "A"]
by = collections.defaultdict(lambda: {"t23": [], "t01": [], **{k: [] for k in FAMS}})
n = 0
for p in sorted(glob.glob(a.tasks)):
    for l in open(p):
        if not l.strip():
            continue
        t = json.loads(l); n += 1
        fam = t["template"][0]
        if fam in FAMS:
            by[t["project"]][fam].append(t)
        else:
            by[t["project"]]["t23" if t["tier"] >= 2 else "t01"].append(t)
out = []
for proj in sorted(by):
    g = by[proj]
    rng.shuffle(g["t23"]); rng.shuffle(g["t01"])
    out += g["t23"][: a.k23] + g["t01"][: a.k01]
    fams = list(FAMS); rng.shuffle(fams)
    for i in range(a.kx):
        fam = fams[i % len(FAMS)]; rng.shuffle(g[fam]); out += g[fam][:1]
# every body style (extrusion / brep / clipped) must be represented in proportion to its buildings: the per-building quotas above
# already guarantee that, since every building gets k23 + k01 + kx tasks; report the split so it is visible
import re as _re
body = collections.Counter(_body(t["project"]) for t in out)
print("body styles:", dict(body))
rng.shuffle(out)
with open(a.out, "w") as f:
    for t in out:
        f.write(json.dumps(t) + "\n")
print(f"{n} tasks in {len(by)} buildings -> {len(out)} sampled:", dict(collections.Counter(t['tier'] for t in out)), "families", dict(collections.Counter(t['template'].split('-')[0] for t in out)),
      "| tiled:", sum(1 for t in out if 'tiled' in t['project']), "| units mm:", sum(1 for t in out if '_mm_' in t['project']))
