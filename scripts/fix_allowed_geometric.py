"""Widen `allowed` of wall-move tasks with GEOMETRIC neighbours (2026-10-23): walls whose axis end touches the moved wall and spaces
whose footprint edge lies on one of its faces. In buildings with full relations these coincide with the relation-based neighbours
already listed; in sparse-relation buildings (no IfcRelConnectsPathElements / IfcRelSpaceBoundary) nothing was listed, so a teacher
that tidied a touching wall was charged with a side effect. Prompts, intents and dependents are unchanged.
    python scripts/fix_allowed_geometric.py <tasks.jsonl ...>   (in place; shardable by giving different files)"""
import json, sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, ifcopenshell
from ifcgym.ifctools import wall_axis_world, _wall_thickness, unit_scale
from ifcgym.checkers import _footprint
from shapely.geometry import LineString, Point

def neighbours(f, guid):
    sc = unit_scale(f) or 1.0; w = f.by_guid(guid); ax = wall_axis_world(w)
    if ax is None:
        return []
    a, b = np.array(ax[0][:2]) * sc, np.array(ax[1][:2]) * sc; seg = LineString([a, b]); t = (_wall_thickness(w) or 0.2) * sc
    out = []
    for o in f.by_type("IfcWall"):
        if o == w:
            continue
        axo = wall_axis_world(o)
        if axo is None:
            continue
        oa, ob = np.array(axo[0][:2]) * sc, np.array(axo[1][:2]) * sc
        if abs(float(axo[0][2]) * sc - float(ax[0][2]) * sc) > 0.5:
            continue
        if min(seg.distance(Point(oa)), seg.distance(Point(ob))) <= t / 2 + 0.3 or LineString([oa, ob]).distance(Point(a)) <= 0.3 or LineString([oa, ob]).distance(Point(b)) <= 0.3:
            out.append(o.GlobalId)
    face = seg.buffer(t / 2 + 0.05, cap_style=2)
    for s in f.by_type("IfcSpace"):
        fp = _footprint(s)
        if fp is None:
            continue
        if fp.distance(seg) <= t / 2 + 0.05 and fp.boundary.intersection(face).length > 0.2:
            out.append(s.GlobalId)
    return out

for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    cache = {}; changed = 0
    for t in rows:
        it = t["intent"]
        if it.get("type") != "translate" or not t["template"].startswith(("T1-translate", "T2-", "T3-")):
            continue
        key = (t["ifc"], it["guid"])
        if key not in cache:
            if t["ifc"] not in cache:
                cache[t["ifc"]] = ifcopenshell.open(t["ifc"])
            cache[key] = neighbours(cache[t["ifc"]], it["guid"])
        add = [g for g in cache[key] if g not in t["allowed"]]
        if add:
            t["allowed"] = t["allowed"] + add; t.setdefault("meta", {})["allowed_geometric_added"] = len(add); changed += 1
    with open(path, "w") as fh:
        for t in rows:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"{os.path.basename(path)}: {len(rows)} tasks, {changed} wall-move tasks widened", flush=True)
