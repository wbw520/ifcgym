"""Compound-instruction and quantity take-off lists (E009 / E010, decisions 2026-10-26): M-compound-2/3 tasks (several numbered edits in one prompt) and the rel-D
variants (direct target + a relationship clause). Each kind is capped; within a kind the draw is round-robin over part-family
combinations (compound) or templates (rel-D), then body styles (extrusion / brep / clipped / faceset), one task per building and
kind first, so no combination or body dominates. Excluded buildings (dev-val / held-out lists, prior teacher lists) are skipped.
    python scripts/sample_compound.py "outputs/tasks_synth2k_v8c/*.jsonl" --out L.jsonl [--cap M-compound-2=700 M-compound-3=500 rel-D=400]
        [--exclude-ids ids.txt] [--exclude-buildings names.txt] [--seed 0]"""
import argparse, collections, glob, json, random


def kind(t):
    tp = t["template"]
    if tp.startswith("R-qto-"):
        return "qto"          # E010: IFC base quantities
    if any(k in tp for k in ("-box-", "slab-storey", "stack-column", "thickness-bbox", "slab-extend")):
        return "cover"        # E010: column / slab coverage
    return tp if tp.startswith("M-compound") else "rel-D" if tp.endswith("-rel-D") else None


def stratum(t):
    if t["template"].startswith("M-"):
        return "+".join(sorted("-".join(p.split("-")[:2]) for p in t["meta"]["parts"]))
    if t["template"].startswith("R-qto-") or kind(t) == "cover":
        return t["template"].rsplit("-", 1)[0]     # quantity (element x property) / cover operation, any reference style
    return t["template"]


def body(tid):
    return "brep" if "_brep" in tid else "faceset" if "faceset" in tid else "clipped" if "clipped" in tid else "extrusion"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("glob"); ap.add_argument("--out", required=True)
    ap.add_argument("--cap", nargs="+", default=["M-compound-2=700", "M-compound-3=500", "rel-D=400"])
    ap.add_argument("--exclude-ids"); ap.add_argument("--exclude-buildings"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); rng = random.Random(a.seed); caps = {k: int(v) for k, v in (c.split("=") for c in a.cap)}
    ex_ids = set(open(a.exclude_ids).read().split()) if a.exclude_ids else set()
    ex_b = set(open(a.exclude_buildings).read().split()) if a.exclude_buildings else set()
    pools = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))  # kind -> stratum -> body -> tasks
    for p in sorted(glob.glob(a.glob)):
        for l in open(p):
            if not l.strip():
                continue
            t = json.loads(l); k = kind(t)
            if k in caps and t["task_id"] not in ex_ids and t["project"] not in ex_b:
                pools[k][stratum(t)][body(t["task_id"])].append(t)
    out = []
    for k, cap in caps.items():
        for s in pools[k].values():
            for v in s.values():
                rng.shuffle(v)
        used_b = set(); picked = []; strata = sorted(pools[k]); rng.shuffle(strata); progress = True
        for first_pass in (True, False):       # one task per building first, then repeats
            progress = True
            while len(picked) < cap and progress:
                progress = False
                for s in strata:
                    for b in sorted(pools[k][s]):
                        if len(picked) >= cap:
                            break
                        lst = pools[k][s][b]
                        for i, t in enumerate(lst):
                            if not first_pass or t["project"] not in used_b:
                                picked.append(lst.pop(i)); used_b.add(t["project"]); progress = True; break
        n_pool = sum(len(v) for s in pools[k].values() for v in s.values()) + len(picked)
        print(f"{k:14s} pool {n_pool:6d} picked {len(picked):4d} buildings {len({t['project'] for t in picked}):4d} strata {len(strata)} "
              f"by body {dict(collections.Counter(body(t['task_id']) for t in picked))}")
        out += picked
    with open(a.out, "w") as fo:
        for t in out:
            fo.write(json.dumps(t) + "\n")
    print(f"{len(out)} tasks -> {a.out}")


if __name__ == "__main__":
    main()
