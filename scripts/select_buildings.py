"""Select the training buildings (about 3000 in total) from everything synthesised, with target shares per
schema (IFC4 40 / IFC4X3 40 / IFC2X3 20), body style (extrusion 45 / brep 20 / clipped 15 / faceset 20), sparse relations (20 %)
and tiled blocks (25 %). Greedy: repeatedly pick the building that most reduces the deviation from the targets.
    python scripts/select_buildings.py --root ${IFCGYM_ROOT}/data/synth/buildings2k --n 3000 --out <root>/selected.txt"""
import argparse, collections, glob, json, os, random

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--n", type=int, default=3000); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); rng = random.Random(a.seed)
T = {"schema": {"ifc4": 0.40, "ifc4x3": 0.40, "ifc2x3": 0.20}, "body": {"extrusion": 0.45, "brep": 0.20, "clipped": 0.15, "faceset": 0.20}, "sparse": {"1": 0.20, "0": 0.80}, "tiled": {"1": 0.25, "0": 0.75}}

def attrs(bid):
    parts = bid.split("_"); sparse = "1" if parts[-1] == "sparse" else "0"
    core = parts[:-1] if sparse == "1" else parts
    body = core[-1] if core[-1] in ("brep", "clipped", "faceset") else "extrusion"
    core = core[:-1] if body != "extrusion" else core
    return {"schema": core[-1], "body": body, "sparse": sparse, "tiled": "1" if parts[1] == "tiled" else "0"}

ids = [os.path.basename(d) for d in glob.glob(os.path.join(a.root, "synth_*")) if os.path.exists(os.path.join(d, "arc.ifc"))]
quality = {}
for qp in glob.glob(os.path.join(a.root, "quality_*.jsonl")):     # scripts/building_quality.py: hard flags exclude, score weights
    for l in open(qp):
        if l.strip():
            r = json.loads(l); quality[r["id"]] = r
if quality:
    before = len(ids); ids = [b for b in ids if b in quality and not quality[b]["flags"]]
    print(f"quality filter: {before} -> {len(ids)} buildings (flags: {collections.Counter(fl for r in quality.values() for fl in r['flags'])})")
    ids.sort(key=lambda b: -quality[b]["score"])     # best first: the greedy window below then prefers higher scores
else:
    rng.shuffle(ids)
pool = {b: attrs(b) for b in ids}
chosen = []; counts = {k: collections.Counter() for k in T}
def score_gain(at):
    n = len(chosen) + 1; g = 0.0
    for k, v in at.items():
        want = T[k].get(v, 0.0) * n; have = counts[k][v]
        g += (want - have)        # positive when this category is under-represented
    return g
remaining = list(ids)
while len(chosen) < a.n and remaining:
    # evaluate a random window of candidates (greedy over all 6k each step would be slow but fine; keep it simple with a window)
    window = remaining[:400]
    best = max(window, key=lambda b: score_gain(pool[b]))
    chosen.append(best); remaining.remove(best)
    for k, v in pool[best].items():
        counts[k][v] += 1
with open(a.out, "w") as f:
    for b in sorted(chosen):
        f.write(b + "\n")
n = len(chosen)
print(f"selected {n} of {len(ids)}:", {k: {v: f"{c / n:.2f}" for v, c in sorted(counts[k].items())} for k in T})
