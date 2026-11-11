"""Cascade-tiered task generator on real IFC projects (DESIGN.md §1, decision 2026-10-19).

Tier 0  attribute / property edits of one element                (allowed = target)
Tier 1  host-dependent edits: translate a wall WITH its openings and fills; delete a door WITH its opening   (dependents listed)
Tier 2  space-affecting edits: translate a wall that bounds spaces (space boundaries / quantities must follow)  [spec'd, checker TODO]
Tier 3  connection-affecting edits: translate a wall joined to other walls                                  [spec'd, checker TODO]

    python -m ifcgym.tasks.generate --ifc data/ifc-bench/wbdg_office/arc.ifc --project wbdg_office --out outputs/tasks/wbdg_office.jsonl --per-template 20 --seed 0

Each record: {task_id, project, template, tier, prompt, target, allowed, intent, dependents, meta}. Prompts are templated (direct
references by GlobalId, BIM-Edit 'direct' style); spatial / topological phrasings are added later as separate templates.
"""
from __future__ import annotations

import argparse
import json
import os
import random

import ifcopenshell
import ifcopenshell.util.element as eu

DIRS = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0)}


def _openings(wall):
    return [r.RelatedOpeningElement for r in getattr(wall, "HasOpenings", []) or []]


def _fills(opening):
    return [r.RelatedBuildingElement for r in getattr(opening, "HasFillings", []) or []]


def _spaces_bounded(wall):
    out, seen = [], set()
    for r in getattr(wall, "ProvidesBoundaries", []) or []:
        s = r.RelatingSpace
        if s is not None and s.GlobalId not in seen:
            seen.add(s.GlobalId); out.append(s)
    return out


def _connected_walls(wall):
    out, seen = [], set()
    for r in list(getattr(wall, "ConnectedTo", []) or []) + list(getattr(wall, "ConnectedFrom", []) or []):
        other = r.RelatedElement if r.RelatingElement == wall else r.RelatingElement
        if other is not None and other.is_a("IfcWall") and other.GlobalId not in seen:
            seen.add(other.GlobalId); out.append(other)
    return out


def _space_side(f, space, wall, vec):
    """+1 if the wall lies on the max side of the space along the move axis, -1 on the min side, None if unclear."""
    try:
        from ifcgym.checkers import mesh
        import numpy as np
        sm, wm = mesh(space), mesh(wall)
        if sm is None or wm is None:
            return None
        ax = int(np.argmax(np.abs(np.array(vec, dtype=float))))
        c = wm[:, ax].mean(); lo, hi = sm[:, ax].min(), sm[:, ax].max()
        if abs(c - hi) < abs(c - lo):
            return 1
        return -1
    except Exception:  # noqa: BLE001
        return None


def _flat(f, w) -> bool:
    """Body = plain vertical extrusion(s) whose depth equals the mesh height (false for clipped gable walls, Breps, face sets)."""
    try:
        from ifcgym.tasks.generate_v4 import _body_items, _extrusion_depth
        from ifcgym.checkers import mesh as _m
        from ifcgym.ifctools import unit_scale as _us
        v = _m(w); sc_ = _us(f) or 1.0
        if v is None:
            return False
        h_ = float(v[:, 2].max() - v[:, 2].min()); items = _body_items(w)
        return bool(items) and all((_extrusion_depth(it) is not None and abs(_extrusion_depth(it) * sc_ - h_) <= 0.02) or it.is_a("IfcFacetedBrep") or it.is_a("IfcPolygonalFaceSet") for it in items)
    except Exception:  # noqa: BLE001
        return False


def _perp_dirs(w) -> list[str]:
    """World axis directions perpendicular to the wall's axis (empty for oblique walls or walls without a usable axis)."""
    try:
        from ifcgym.ifctools import wall_axis_world
        import numpy as np
        ax = wall_axis_world(w)
        if ax is None:
            return []
        u = np.array(ax[1][:2]) - np.array(ax[0][:2]); u = u / (np.linalg.norm(u) or 1.0)
        if abs(u[0]) > 0.98:
            return ["+y", "-y"]
        if abs(u[1]) > 0.98:
            return ["+x", "-x"]
        return []
    except Exception:  # noqa: BLE001
        return []


def _collinear(w, o) -> bool:
    try:
        from ifcgym.ifctools import wall_axis_world
        import numpy as np
        a, b = wall_axis_world(w), wall_axis_world(o)
        if a is None or b is None:
            return False
        u = np.array(a[1][:2]) - np.array(a[0][:2]); u /= np.linalg.norm(u) or 1.0
        v = np.array(b[1][:2]) - np.array(b[0][:2]); v /= np.linalg.norm(v) or 1.0
        return abs(float(u @ v)) > 0.95
    except Exception:  # noqa: BLE001
        return False


def gen(f: ifcopenshell.file, project: str, per_template: int, rng: random.Random, emit_t2: bool = True, perpendicular: bool = False, families: tuple = (), emit_t1: bool = True, per_template_move: int | None = None) -> list[dict]:
    per_template_move = per_template_move or per_template
    walls = [w for w in f.by_type("IfcWall") if getattr(w, "Representation", None)]
    doors = [d for d in f.by_type("IfcDoor") if getattr(d, "Representation", None)]
    tasks: list[dict] = []
    n = 0
    def tid(tpl):
        nonlocal n; n += 1; return f"{project}-{tpl}-{n:04d}"

    # tier 0: set a property on a wall (Pset_WallCommon.FireRating) / set the Name attribute
    for w in rng.sample(walls, min(per_template, len(walls))):
        val = rng.choice(["REI 30", "REI 60", "REI 90", "REI 120"])
        tasks.append({"task_id": tid("T0-fire"), "project": project, "template": "T0-set-fire-rating", "tier": 0,
                      "prompt": f"Set the fire rating (Pset_WallCommon.FireRating) of the wall with GlobalId {w.GlobalId} to '{val}'. Do not change anything else.",
                      "target": [w.GlobalId], "allowed": [w.GlobalId],
                      "intent": {"type": "set_property", "guid": w.GlobalId, "pset": "Pset_WallCommon", "prop": "FireRating", "value": val}, "dependents": []})
    for w in rng.sample(walls, min(per_template, len(walls))):
        name = f"Wall-{rng.randint(100, 999)}"
        tasks.append({"task_id": tid("T0-name"), "project": project, "template": "T0-set-name", "tier": 0,
                      "prompt": f"Rename the wall with GlobalId {w.GlobalId}: set its Name attribute to '{name}'. Do not change anything else.",
                      "target": [w.GlobalId], "allowed": [w.GlobalId],
                      "intent": {"type": "set_attribute", "guid": w.GlobalId, "name": "Name", "value": name}, "dependents": []})

    # tier 1a: translate a wall that has openings; openings and their fills must move with it, host relations kept
    hosts = [w for w in walls if _openings(w)]
    if perpendicular:   # generator v2 (2026-10-20): a wall is moved across its axis only - sliding a wall along its axis has no
        hosts = [w for w in hosts if _perp_dirs(w)]   # well-defined joint propagation (corner walls would have to follow, then theirs...)
    for w in rng.sample(hosts, min(per_template_move, len(hosts))):
        d = rng.choice(_perp_dirs(w) if perpendicular else list(DIRS)); dist = rng.choice([0.25, 0.5, 0.75, 1.0]); vec = [c * dist for c in DIRS[d]]
        deps = []; allowed = [w.GlobalId]
        for op in _openings(w):
            deps.append({"guid": op.GlobalId, "rule": "translate_with", "vector": vec}); deps.append({"guid": op.GlobalId, "rule": "stays_hosted"}); allowed.append(op.GlobalId)
            for fl in _fills(op):
                deps.append({"guid": fl.GlobalId, "rule": "translate_with", "vector": vec}); allowed.append(fl.GlobalId)
        sp = _spaces_bounded(w); cw = _connected_walls(w); tier = 1
        for s in sp:   # tier 2: bounded spaces must follow: the space face on the wall side and the space boundary move with the wall
            allowed.append(s.GlobalId); tier = max(tier, 2)
            side = _space_side(f, s, w, vec)
            if side is not None:
                deps.append({"guid": s.GlobalId, "rule": "space_meets_wall", "wall": w.GlobalId, "vector": vec, "side": side})
            deps.append({"guid": s.GlobalId, "rule": "boundary_on_wall", "wall": w.GlobalId, "vector": vec})
        deps_t2 = list(deps); allowed_t2 = list(allowed) + [o.GlobalId for o in cw]   # tier 2: joints may change but are not required to re-join
        for o in cw:   # tier 3: connected walls must stay joined to the moved wall
            allowed.append(o.GlobalId); tier = max(tier, 3)
            if perpendicular and _collinear(w, o):
                continue   # generator v2: a collinear continuation cannot keep its joint when the wall moves sideways (released, may change)
            deps.append({"guid": o.GlobalId, "rule": "joined_to", "other": w.GlobalId})
        if tier > 1 and emit_t1:   # generator v3 (2026-10-22, user): a separate tier-1 task on the same wall - openings must follow, spaces and
            # joints are allowed to change but not required (synthetic buildings model every space and joint, so a natural tier-1 wall never occurs)
            deps_t1 = [d for d in deps if d["rule"] in ("translate_with", "stays_hosted")]
            allowed_t1 = [w.GlobalId] + [op.GlobalId for op in _openings(w)] + [fl.GlobalId for op in _openings(w) for fl in _fills(op)] + [s.GlobalId for s in sp] + [o.GlobalId for o in cw]
            tasks.append({"task_id": tid("T1-move"), "project": project, "template": "T1-translate-wall-with-openings", "tier": 1,
                          "prompt": f"Move the wall with GlobalId {w.GlobalId} by {dist} m in the {d} direction of the project coordinate system. Its openings and the doors/windows in them must move with it and stay hosted. The spaces it bounds and the walls connected to it may be adjusted as needed. Do not change anything else.",
                          "target": [w.GlobalId], "allowed": allowed_t1, "intent": {"type": "translate", "guid": w.GlobalId, "vector": vec, "tol": 0.02},
                          "dependents": deps_t1, "meta": {"n_openings": len(_openings(w)), "n_spaces": len(sp), "n_connected": len(cw), "relaxed": True}})
        if sp and cw and emit_t2:   # a separate tier-2 task on the same wall: space propagation required, joints relaxed
            tasks.append({"task_id": tid("T2-move"), "project": project, "template": "T2-translate-wall-bounding-spaces", "tier": 2,
                          "prompt": f"Move the wall with GlobalId {w.GlobalId} by {dist} m in the {d} direction of the project coordinate system. Its openings and the doors/windows in them must move with it and stay hosted, and the spaces it bounds and their space boundaries must follow the new wall face. Walls connected to it may be adjusted as needed. Do not change anything else.",
                          "target": [w.GlobalId], "allowed": allowed_t2, "intent": {"type": "translate", "guid": w.GlobalId, "vector": vec, "tol": 0.02},
                          "dependents": deps_t2, "meta": {"n_openings": len(_openings(w)), "n_spaces": len(sp), "n_connected": len(cw)}})
        tpl = {1: "T1-translate-wall-with-openings", 2: "T2-translate-wall-bounding-spaces", 3: "T3-translate-joined-wall"}[tier]
        tasks.append({"task_id": tid("T%d-move" % tier), "project": project, "template": tpl, "tier": tier,
                      "prompt": f"Move the wall with GlobalId {w.GlobalId} by {dist} m in the {d} direction of the project coordinate system. Everything that depends on this wall must stay consistent: its openings and the doors/windows in them move with it and stay hosted, the spaces it bounds and their space boundaries follow the new wall face, and walls connected to it stay joined. Do not change anything else.",
                      "target": [w.GlobalId], "allowed": allowed, "intent": {"type": "translate", "guid": w.GlobalId, "vector": vec, "tol": 0.02},
                      "dependents": deps, "meta": {"n_openings": len(_openings(w)), "n_spaces": len(sp), "n_connected": len(_connected_walls(w))}})

    # tier 1b: delete a door; its opening must be removed too (BIBIMBAP integrity rule), the host wall stays
    for dr in rng.sample(doors, min(per_template, len(doors))):
        ops = [r.RelatingOpeningElement for r in getattr(dr, "FillsVoids", []) or []]
        deps = [{"guid": op.GlobalId, "rule": "removed"} for op in ops]
        hostw = [r.RelatingBuildingElement for op in ops for r in (op.VoidsElements or [])]
        for h in hostw:
            deps.append({"guid": h.GlobalId, "rule": "present"})
        tasks.append({"task_id": tid("T1-deldoor"), "project": project, "template": "T1-delete-door-with-opening", "tier": 1,
                      "prompt": f"Delete the door with GlobalId {dr.GlobalId} and its opening, so that the host wall becomes solid again. Do not change anything else.",
                      "target": [dr.GlobalId], "allowed": [dr.GlobalId] + [op.GlobalId for op in ops] + [h.GlobalId for h in hostw],
                      "intent": {"type": "delete", "guid": dr.GlobalId}, "dependents": deps, "meta": {"n_openings": len(ops)}})

    # tier 1c: change wall height (geometry edit without dependents) - only walls whose body is a plain vertical extrusion whose depth is
    # the mesh height (clipped gable walls would not change height when the depth changes)
    plain = [w for w in walls if not _openings(w) and _flat(f, w)]
    for w in rng.sample(plain, min(per_template, len(plain))):
        h = rng.choice([2.7, 3.0, 3.3, 3.6])
        tasks.append({"task_id": tid("T1-height"), "project": project, "template": "T1-set-wall-height", "tier": 1,
                      "prompt": f"Change the height of the wall with GlobalId {w.GlobalId} to {h} m, keeping its footprint, position and thickness. Do not change anything else.",
                      "target": [w.GlobalId], "allowed": [w.GlobalId], "intent": {"type": "resize_height", "guid": w.GlobalId, "height": h, "tol": 0.02}, "dependents": []})
    if families:
        tasks += gen_extra(f, project, per_template, rng, tid, walls, families)
    return tasks


# ------------------------------------------------------------------------------------------ 2026-10-22: create / retrieve / spatial-phrasing templates
def _axis(w):
    from ifcgym.ifctools import wall_axis_world
    try:
        return wall_axis_world(w)
    except Exception:  # noqa: BLE001
        return None


def _north_of(f, walls, w, scale):
    """The unique parallel wall directly north (+Y) of w (x-extent overlap > 50 percent, nearest), or None."""
    a = _axis(w)
    if a is None or abs(a[0][1] - a[1][1]) > 1e-6 * (1 / scale):   # must run along x
        return None
    y = a[0][1]; x0, x1 = sorted([a[0][0], a[1][0]])
    if (x1 - x0) * scale < 2.0:
        return None
    best = None; partial = None
    try:
        st = eu.get_container(w)
    except Exception:  # noqa: BLE001
        st = None
    for o in walls:
        if o == w:
            continue
        b = _axis(o)
        if b is None or abs(b[0][1] - b[1][1]) > 1e-6 * (1 / scale):
            continue
        if st is not None:      # same storey container (the prompt says "on the same storey"), whatever the elevation
            try:
                if eu.get_container(o) != st:
                    continue
            except Exception:  # noqa: BLE001
                continue
        elif abs(b[0][2] - a[0][2]) > 0.5 / scale:
            continue
        ox0, ox1 = sorted([b[0][0], b[1][0]]); ov = min(x1, ox1) - max(x0, ox0)
        if ov <= -0.3 / scale or b[0][1] <= y + 0.05 / scale:
            continue
        d = b[0][1] - y
        if ov <= 0.6 * (x1 - x0) or ov <= 0.6 * (ox1 - ox0):   # touching / partial overlap closer than the answer -> no task      # closer wall with partial overlap -> the reference would be arguable (review 2026-10-22)
            partial = min(partial, d) if partial is not None else d
            continue
        if best is None or d < best[0] - 1e-9:
            best = (d, o, 1)
        elif abs(d - best[0]) <= 1e-9:
            best = (d, o, best[2] + 1)
    if best and partial is not None and partial < best[0] + 1e-9:
        return None
    return best[1] if best and best[2] == 1 else None


def gen_extra(f, project, per_template, rng, tid, walls, families):
    from ifcgym.ifctools import wall_axis_world, _wall_thickness, space_area, unit_scale, world_matrix
    import numpy as np
    sc = unit_scale(f) or 1.0           # metres per model unit; prompts and intents are in METRES
    tasks = []
    storeys = [s for s in f.by_type("IfcBuildingStorey") if s.Name]
    def storey_of(e):
        try:
            import ifcopenshell.util.element as ue
            c = ue.get_container(e); return c if c is not None and c.is_a("IfcBuildingStorey") else None
        except Exception:  # noqa: BLE001
            return None
    def storey_ref(s):
        return f"the storey named '{s.Name}' (GlobalId {s.GlobalId})"
    def height_of(w):
        from ifcgym.checkers import mesh
        m = mesh(w); return None if m is None else float(m[:, 2].max() - m[:, 2].min())
    long_walls = [w for w in walls if (a := _axis(w)) and np.linalg.norm(np.array(a[1][:2]) - np.array(a[0][:2])) * sc >= 2.5 and (_wall_thickness(w) > 0 or any(it.is_a("IfcFacetedBrep") or it.is_a("IfcPolygonalFaceSet") for it in (lambda e: [i for r in e.Representation.Representations if r.RepresentationIdentifier == "Body" for i in r.Items])(w)))]
    if "create" in families:
        for kind in ("door", "window"):
            for w in rng.sample(long_walls, min(per_template, len(long_walls))):
                width, height, sill = ((rng.choice([0.8, 0.9, 1.0]), 2.1, 0.0) if kind == "door" else (rng.choice([0.9, 1.2, 1.5]), 1.2, 0.9))
                hw = height_of(w)
                if hw is None or hw < height + sill + 0.1:
                    continue
                from ifcgym.tasks.generate_v4 import Ctx as _Ctx, free_slot as _free_slot
                if not hasattr(gen_extra, "_ctx") or gen_extra._ctx[0] is not f:
                    gen_extra._ctx = (f, _Ctx(f, random.Random(0)))
                if w.GlobalId not in gen_extra._ctx[1].axis or not _free_slot(gen_extra._ctx[1], w, width):
                    continue
                tasks.append({"task_id": tid(f"C-{kind}"), "project": project, "template": f"C-create-{kind}", "tier": 1,
                              "prompt": f"Add a {kind} {width} m wide and {height} m high to the wall with GlobalId {w.GlobalId}, anywhere along the wall" + (" with its sill at 0.9 m" if kind == "window" else "") + f". The {kind} must be hosted by the wall through an opening (IfcRelVoidsElement / IfcRelFillsElement). Do not change anything else.",
                              "target": [w.GlobalId], "allowed": [w.GlobalId], "allowed_new_classes": ["IfcOpeningElement", "IfcDoor" if kind == "door" else "IfcWindow"],
                              "intent": {"type": "create_fill", "host": w.GlobalId, "kind": kind, "width": width, "height": height, "tol": 0.03}, "dependents": []})
        for s in rng.sample(storeys, min(per_template, len(storeys))) * max(1, per_template // max(1, len(storeys))):
            ws = [w for w in walls if storey_of(w) == s and _axis(w)]
            if len(ws) < 3:
                continue
            pts = np.array([p[:2] for w in ws for p in _axis(w)]) * sc; lo, hi = pts.min(0), pts.max(0)
            if (hi - lo).min() < 4:
                continue
            L = rng.choice([2.0, 3.0, 4.0]); t = rng.choice([0.1, 0.15, 0.2, 0.25]); h = rng.choice([2.7, 3.0])
            x = round(rng.uniform(lo[0] + 1, hi[0] - L - 1), 2); y = round(rng.uniform(lo[1] + 1, hi[1] - 1), 2)
            if rng.random() < 0.5 or y + L >= hi[1] - 1:
                p0, p1 = (x, y), (round(x + L, 2), y)
            else:
                p0, p1 = (x, y), (x, round(y + L, 2))
            tasks.append({"task_id": tid("C-wall"), "project": project, "template": "C-create-wall", "tier": 1,
                          "prompt": f"Create a new wall on {storey_ref(s)} from ({p0[0]}, {p0[1]}) to ({p1[0]}, {p1[1]}) in metres of the project coordinate system (XY on that storey), {t} m thick and {h} m high, contained in that storey. Do not change anything else.",
                          "target": [s.GlobalId], "allowed": [], "allowed_new_classes": ["IfcWall", "IfcWallStandardCase"],
                          "intent": {"type": "create_wall", "p0": list(p0), "p1": list(p1), "thickness": t, "height": h, "storey": s.GlobalId, "tol": 0.03}, "dependents": []})
            if rng.random() < 0.5:
                size = rng.choice([0.3, 0.4, 0.5]); cx, cy = round(rng.uniform(lo[0] + 1, hi[0] - 1), 2), round(rng.uniform(lo[1] + 1, hi[1] - 1), 2)
                tasks.append({"task_id": tid("C-column"), "project": project, "template": "C-create-column", "tier": 1,
                              "prompt": f"Create a square column {size} m by {size} m and {h} m high centred at ({cx}, {cy}) metres on {storey_ref(s)}, contained in that storey. Do not change anything else.",
                              "target": [s.GlobalId], "allowed": [], "allowed_new_classes": ["IfcColumn"],
                              "intent": {"type": "create_column", "xy": [cx, cy], "size": size, "height": h, "storey": s.GlobalId, "tol": 0.03}, "dependents": []})
            else:
                a_, b_ = rng.choice([3.0, 4.0, 5.0]), rng.choice([3.0, 4.0]); th = rng.choice([0.15, 0.2, 0.25])
                px, py = round(rng.uniform(lo[0] + 0.5, hi[0] - a_ - 0.5), 2), round(rng.uniform(lo[1] + 0.5, hi[1] - b_ - 0.5), 2)
                poly = [[px, py], [round(px + a_, 2), py], [round(px + a_, 2), round(py + b_, 2)], [px, round(py + b_, 2)]]
                tasks.append({"task_id": tid("C-slab"), "project": project, "template": "C-create-slab", "tier": 1,
                              "prompt": f"Create a floor slab {th} m thick on {storey_ref(s)} with the rectangular footprint {poly} (metres, project coordinate system, corners in order), its top at the storey level, contained in that storey. Do not change anything else.",
                              "target": [s.GlobalId], "allowed": [], "allowed_new_classes": ["IfcSlab"],
                              "intent": {"type": "create_slab", "polygon": poly, "thickness": th, "storey": s.GlobalId, "tol": 0.03}, "dependents": []})
    if "retrieve" in families:
        spaces = [s for s in f.by_type("IfcSpace") if getattr(s, "Representation", None)]
        for sp in rng.sample(spaces, min(per_template, len(spaces))):
            try:
                area = space_area(f, sp)
                from ifcgym.checkers import _footprint
                fp = _footprint(sp); marea = fp.area if fp is not None else None   # mesh footprint: the truth must not depend on the representation
            except Exception:  # noqa: BLE001
                continue
            if not (area == area) or area < 1 or marea is None or abs(marea - area) > 0.01 * area:
                continue
            area = marea
            tasks.append({"task_id": tid("R-area"), "project": project, "template": "R-space-area", "tier": 0,
                          "prompt": f"What is the net floor area, in square metres, of the space with GlobalId {sp.GlobalId}? Compute it from the model. Do not modify the model. End your final message with a JSON object {{\"value\": <number>}}.",
                          "target": [sp.GlobalId], "allowed": [], "intent": {"type": "answer", "kind": "number", "value": round(area, 2), "rel_tol": 0.01, "tol": 0.1}, "dependents": []})
        for s in rng.sample(storeys, min(per_template, len(storeys))):
            doors = [d for d in f.by_type("IfcDoor") if storey_of(d) == s]
            tasks.append({"task_id": tid("R-count"), "project": project, "template": "R-count-doors", "tier": 0,
                          "prompt": f"How many doors (IfcDoor) are contained in {storey_ref(s)}? Do not modify the model. End your final message with a JSON object {{\"value\": <integer>}}.",
                          "target": [s.GlobalId], "allowed": [], "intent": {"type": "answer", "kind": "number", "value": len(doors), "tol": 0.0}, "dependents": []})
        cands = [(w, _north_of(f, walls, w, sc)) for w in walls]; cands = [(w, o) for w, o in cands if o is not None]
        for w, o in rng.sample(cands, min(per_template, len(cands))):
            tasks.append({"task_id": tid("R-north"), "project": project, "template": "R-find-wall-north", "tier": 0,
                          "prompt": f"Which wall lies directly north (+Y direction) of the wall with GlobalId {w.GlobalId} on the same storey (the nearest parallel wall in +Y on the same storey; the two walls' axes overlap along at least half of each wall's length)? Do not modify the model. End your final message with a JSON object {{\"global_id\": \"<GlobalId>\"}}.",
                          "target": [w.GlobalId], "allowed": [], "intent": {"type": "answer", "kind": "guid", "value": o.GlobalId}, "dependents": []})
    if "spatial" in families:
        cands = [(w, _north_of(f, walls, w, sc)) for w in walls]; cands = [(w, o) for w, o in cands if o is not None]
        for w, o in rng.sample(cands, min(per_template, len(cands))):
            val = rng.choice(["REI 30", "REI 60", "REI 90"])
            tasks.append({"task_id": tid("S-fire"), "project": project, "template": "S-set-fire-rating-north-of", "tier": 0,
                          "prompt": f"Set the fire rating (Pset_WallCommon.FireRating) of the wall that lies directly north (+Y) of the wall with GlobalId {w.GlobalId} (the nearest parallel wall in +Y on the same storey; the two walls' axes overlap along at least half of each wall's length) to '{val}'. Do not change anything else.",
                          "target": [o.GlobalId], "allowed": [o.GlobalId], "intent": {"type": "set_property", "guid": o.GlobalId, "pset": "Pset_WallCommon", "prop": "FireRating", "value": val}, "dependents": [], "meta": {"anchor": w.GlobalId}})
        for w, o in rng.sample(cands, min(per_template, len(cands))):
            h = rng.choice([2.7, 3.0, 3.3])
            if any(True for _ in (getattr(o, "HasOpenings", None) or [])) or not _flat(f, o):
                continue
            tasks.append({"task_id": tid("S-height"), "project": project, "template": "S-set-height-north-of", "tier": 1,
                          "prompt": f"Change the height of the wall that lies directly north (+Y) of the wall with GlobalId {w.GlobalId} (the nearest parallel wall in +Y on the same storey; the two walls' axes overlap along at least half of each wall's length) to {h} m, keeping its footprint, position and thickness. Do not change anything else.",
                          "target": [o.GlobalId], "allowed": [o.GlobalId], "intent": {"type": "resize_height", "guid": o.GlobalId, "height": h, "tol": 0.02}, "dependents": [], "meta": {"anchor": w.GlobalId}})
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ifc", required=True); ap.add_argument("--project", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--per-template", type=int, default=20); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-template-move", type=int, default=None, help="density for the wall-move tiers 1-3 (default = --per-template)")
    ap.add_argument("--perpendicular", action="store_true", help="generator v2: move walls across their axis only (tiers 1-3 translate templates)")
    ap.add_argument("--families", nargs="*", default=[], choices=["create", "retrieve", "spatial", "update", "delete", "create2", "retrieve2", "concept", "compound", "qto", "cover", "cover2", "cover3", "cover4"],
                    help="extra template families (2026-10-22): create door/window/wall/column/slab, retrieve area/count/north-wall, spatial phrasing; "
                         "generator v4 (BIM-Edit / BIBIMBAP grid, derived input arc_v4.ifc): update, delete, create2, retrieve2, concept")
    a = ap.parse_args()
    f = ifcopenshell.open(a.ifc)
    v1 = tuple(x for x in a.families if x in ("create", "retrieve", "spatial")); v4 = tuple(x for x in a.families if x not in v1)
    tasks = gen(f, a.project, a.per_template, random.Random(a.seed), perpendicular=a.perpendicular, families=v1, per_template_move=a.per_template_move)
    if v4:
        from ifcgym.tasks.generate_v4 import gen_v4
        n = [len(tasks)]
        def tid4(tpl):
            n[0] += 1; return f"{a.project}-{tpl}-{n[0]:04d}"
        tasks += gen_v4(a.ifc, a.project, a.per_template, random.Random(a.seed + 1000), tid4, families=v4)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        for t in tasks:
            t.setdefault("ifc", a.ifc); fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"{len(tasks)} tasks -> {a.out}", dict(Counter(t['template'] for t in tasks)))


if __name__ == "__main__":
    main()
