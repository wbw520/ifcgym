"""Helper ceiling test (E007): a scripted agent performs the cascade recipe with ONLY the h.* primitives on generated tasks and the
frozen checkers judge it. Measures whether the helpers suffice when the decisions are right (per project / profile type).
    python scripts/oracle_cascade.py <tasks.jsonl> [--limit N]"""
import json, os, sys, tempfile, shutil, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import ifcopenshell
from ifcgym import ifctools as h
from ifcgym.checkers import check

def cascade(f, t):
    WG = t["target"][0]; sc = h.unit_scale(f) or 1.0
    vec = [c / sc for c in t["intent"]["vector"]]; v = np.array(vec, dtype=float)    # task vectors are metres; helpers take model units
    info = h.inspect(f, WG)
    ax = int(np.argmax(np.abs(v))); oax = 1 - ax if ax < 2 else 0
    axis = info["axis_world"]
    bb = info.get("bbox_world")
    lo_face, hi_face = (bb["min"][ax], bb["max"][ax]) if bb else (None, None)
    extent = {"xyz"[oax]: (min(axis[0][oax], axis[1][oax]) - 0.05 / sc, max(axis[0][oax], axis[1][oax]) + 0.05 / sc)} if axis else None
    log = {"joined": 0, "spaces": 0, "boundaries": 0}
    h.translate(f, WG, vec)
    seen = set()
    for c in info["connected_walls"]:
        if c["other_guid"] in seen:
            continue
        seen.add(c["other_guid"])
        e = c["other_end_touching_this_wall"]
        if e:
            h.move_wall_end(f, c["other_guid"], e, vec); log["joined"] += 1
    for sp in info["bounded_spaces"]:
        moved = 0
        for face in (hi_face, lo_face):
            if face is None: break
            moved = h.shift_vertices_on_plane(f, sp["guid"], "xyz"[ax], face, vec, within=extent)["vertices_moved"]
            if moved: break
        log["spaces"] += bool(moved)
        for b in h.inspect(f, sp["guid"], psets=False, geometry=False, related_to=WG)["boundaries"]:
            log["boundaries"] += h.shift_boundary_curve(f, b["rel_guid"], None, None, vec)["points_moved"] > 0
    return log

tasks = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
tasks = [t for t in tasks if t["intent"]["type"] == "translate"][:limit]
res = []
for t in tasks:
    f = ifcopenshell.open(t["ifc"])
    try:
        log = cascade(f, t)
        out = os.path.join(tempfile.mkdtemp(prefix="oracle_"), "edited.ifc"); f.write(out)
        r = check(t["ifc"], out, t)
        shutil.rmtree(os.path.dirname(out), ignore_errors=True)   # temp dir cleanup 2026-10-23
        fails = [x.strip().split(":")[1] for x in r.details["dependency"].split(";") if ":FAIL" in x]
        res.append((t["project"], t["tier"], r.strict, r.intent, r.dependency, r.preservation, fails, log))
        print(f"{t['task_id']:40s} strict={r.strict} intent={r.intent} dep={r.dependency} pres={r.preservation} side={r.side_effects} fails={sorted(set(fails))} {log}", flush=True)
    except Exception as e:  # noqa: BLE001
        res.append((t["project"], t["tier"], False, False, False, False, ["EXC"], {}))
        print(f"{t['task_id']:40s} EXCEPTION {type(e).__name__}: {str(e)[:120]}", flush=True)
import collections
by = collections.defaultdict(list)
for r in res: by[(r[0], r[1])].append(r)
for k, v in sorted(by.items()):
    print(f"{k}: n={len(v)} strict={sum(r[2] for r in v)} intent={sum(r[3] for r in v)} dep={sum(r[4] for r in v)} pres={sum(r[5] for r in v)} fails={dict(collections.Counter(x for r in v for x in set(r[6])))}")
