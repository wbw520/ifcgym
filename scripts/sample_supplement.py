"""Supplementary teacher list (decision 2026-10-24): v8 tasks of the kinds the v7 teacher list lacks, so the SFT student sees every task
type before RL. Kinds = templates absent from v7 + reference phrasings absent from v7 (matched on prompt / prompt_template). Each kind
is capped (--per-kind), drawn round-robin over body styles (extrusion / brep / clipped / faceset) and buildings, one task per building
and kind first. Buildings of the v7 dev-val / held-out lists are never used (the v8 set is synthetic only).
    python scripts/sample_supplement.py "outputs/tasks_synth2k_v8/*.jsonl" --out outputs/E008/supplement_v8.jsonl [--per-kind 120]"""
import argparse, collections, glob, json, random, re

KINDS = [   # (name, predicate on the task)
    ("slab-thickness", lambda t: t["template"].startswith("U-set-slab-thickness")),
    ("slab-resize", lambda t: t["template"].startswith("U-resize-slab")),
    ("slab-delete", lambda t: t["template"].startswith("D-delete-slab")),
    ("storey-gross-area", lambda t: t["template"] == "R-storey-gross-area-T"),
    ("space-perimeter", lambda t: t["template"].startswith("R-space-perimeter")),
    ("wall-net-area", lambda t: t["template"].startswith("R-wall-net-area")),
    ("non-adjacent-space", lambda t: t["template"] == "R-non-adjacent-space-T"),
    ("rotate-not-plus90-end", lambda t: t["template"].startswith("U-rotate-wall") and not (t["meta"].get("angle", 90) == 90 and "mid-point" not in t["prompt_template_or"] and "through the point" not in t["prompt_template_or"])),
    ("ref-view", lambda t: "look through the window with GlobalId" in t["prompt_template_or"]),
    ("ref-leave", lambda t: "after leaving the space with GlobalId" in t["prompt_template_or"]),
    ("ref-corner", lambda t: "closest to the corner where the wall with GlobalId" in t["prompt_template_or"]),
    ("ref-path-wall-side", lambda t: re.search(r"the wall on your (left|right)-hand side", t["prompt_template_or"]) is not None),
    ("ref-path-wall-facing", lambda t: "the wall facing you after entering" in t["prompt_template_or"]),
]


def body(tid):
    return "brep" if "_brep" in tid else "faceset" if "faceset" in tid else "clipped" if "clipped" in tid else "extrusion"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("glob"); ap.add_argument("--out", required=True); ap.add_argument("--per-kind", type=int, default=120); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); rng = random.Random(a.seed); pools = collections.defaultdict(list)
    for p in sorted(glob.glob(a.glob)):
        for l in open(p):
            if not l.strip():
                continue
            t = json.loads(l); t["prompt_template_or"] = t.get("prompt_template", t["prompt"])
            for name, pred in KINDS:
                if pred(t):
                    pools[name].append(t); break
    out, seen = [], set()
    for name, _ in KINDS:
        by = collections.defaultdict(lambda: collections.defaultdict(list))
        for t in pools[name]:
            by[body(t["task_id"])][t["project"]].append(t)
        for b in by.values():
            for v in b.values():
                rng.shuffle(v)
        picked = []; bodies = sorted(by); projs = {b: rng.sample(sorted(by[b]), len(by[b])) for b in bodies}; rnd = 0
        while len(picked) < a.per_kind and any(projs[b] for b in bodies):
            for b in bodies:
                if len(picked) >= a.per_kind:
                    break
                while projs[b]:
                    pr = projs[b].pop(0); lst = by[b][pr]
                    if lst:
                        t = lst.pop(0)
                        if t["task_id"] not in seen:
                            picked.append(t); seen.add(t["task_id"]); break
            rnd += 1
        print(f"{name:24s} pool {len(pools[name]):6d}  picked {len(picked):4d}  by body {dict(collections.Counter(body(t['task_id']) for t in picked))}")
        out += picked
    with open(a.out, "w") as fo:
        for t in out:
            t.pop("prompt_template_or", None); t.setdefault("meta", {})["supplement_kind"] = next(n for n, p in KINDS if p({**t, "prompt_template_or": t.get("prompt_template", t["prompt"])}))
            fo.write(json.dumps(t) + "\n")
    print(f"{len(out)} tasks -> {a.out}")


if __name__ == "__main__":
    main()
