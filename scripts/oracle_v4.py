"""Checker + helper validation for generator v4 templates: a scripted agent solves each task from its intent with the h.* helpers
(or answers from the intent), the frozen checkers judge it. Prints pass counts per template.
    python scripts/oracle_v4.py <tasks.jsonl> [--limit N]"""
import json, os, sys, tempfile, shutil, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import ifcopenshell
from ifcgym import ifctools as h
from ifcgym.checkers import check

args = [a for a in sys.argv[1:] if not a.startswith("--")]; lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
NEG = "--negative" in sys.argv   # negative control: a wrong solution (half the edit / no edit / wrong answer) must FAIL the checker
tasks = [json.loads(l) for l in open(args[0]) if l.strip()]
tasks = [t for t in tasks if t["template"][0] in "UDCRAM" and ("-" in t["template"]) and t["template"].split("-")[0] in ("U", "D", "C", "R", "A", "M")][:lim]
res = collections.defaultdict(list)
def rel_deps(f, t):
    """Scripted relationships for the E010 cover templates: space boundary / connection for the new element, re-containment, connection."""
    import ifcopenshell.api.spatial, ifcopenshell.guid
    oh = (f.by_type("IfcOwnerHistory") or [None])[0]
    new = None
    for d in t.get("dependents", []):
        cls = d.get("cls")
        if cls and new is None:
            cands = [e for e in f.by_type(cls) if e.id() > MAXID[0]]
            new = cands[-1] if cands else None
        if d["rule"] == "space_boundary_new" and new is not None:
            f.create_entity("IfcRelSpaceBoundary", GlobalId=ifcopenshell.guid.new(), OwnerHistory=oh, RelatingSpace=f.by_guid(d["guid"]), RelatedBuildingElement=new,
                            PhysicalOrVirtualBoundary="PHYSICAL", InternalOrExternalBoundary="INTERNAL")
        elif d["rule"] == "connected_new" and new is not None and cls == "IfcColumn":
            f.create_entity("IfcRelConnectsElements", GlobalId=ifcopenshell.guid.new(), OwnerHistory=oh, RelatingElement=f.by_guid(d["guid"]), RelatedElement=new)
        elif d["rule"] == "rel_connects":
            f.create_entity("IfcRelConnectsElements", GlobalId=ifcopenshell.guid.new(), OwnerHistory=oh, RelatingElement=f.by_guid(d["guid"]), RelatedElement=f.by_guid(d["el"]))
        elif d["rule"] == "in_storey":
            ifcopenshell.api.spatial.assign_container(f, relating_structure=f.by_guid(d["guid"]), products=[f.by_guid(d["el"])])
        elif d["rule"] == "space_boundary_new" and new is not None and cls == "IfcWall":
            f.create_entity("IfcRelSpaceBoundary", GlobalId=ifcopenshell.guid.new(), OwnerHistory=oh, RelatingSpace=f.by_guid(d["guid"]), RelatedBuildingElement=new,
                            PhysicalOrVirtualBoundary="PHYSICAL", InternalOrExternalBoundary="INTERNAL")
        elif d["rule"] == "new_space_bounds":
            sp = [e for e in f.by_type("IfcSpace") if e.id() > MAXID[0]][-1]
            f.create_entity("IfcRelSpaceBoundary", GlobalId=ifcopenshell.guid.new(), OwnerHistory=oh, RelatingSpace=sp, RelatedBuildingElement=f.by_guid(d["guid"]),
                            PhysicalOrVirtualBoundary="PHYSICAL", InternalOrExternalBoundary="INTERNAL")
        elif d["rule"] == "space_boundary_with":
            f.create_entity("IfcRelSpaceBoundary", GlobalId=ifcopenshell.guid.new(), OwnerHistory=oh, RelatingSpace=f.by_guid(d["guid"]), RelatedBuildingElement=f.by_guid(d["el"]),
                            PhysicalOrVirtualBoundary="PHYSICAL", InternalOrExternalBoundary="INTERNAL")
        elif d["rule"] == "not_connected":
            e = f.by_guid(d["el"])
            for rel in list(getattr(e, "ConnectedTo", []) or []) + list(getattr(e, "ConnectedFrom", []) or []):
                o = rel.RelatedElement if rel.RelatingElement == e else rel.RelatingElement
                if o is not None and o.GlobalId == d["guid"]:
                    f.remove(rel)
        elif d["rule"] == "opening_covers":
            e = f.by_guid(d["guid"]); op = e.FillsVoids[0].RelatingOpeningElement; be, bo = h._mesh_bbox(e), h._mesh_bbox(op)
            if bo["max"][2] < be["max"][2]:
                h.set_extrusion_depth(f, op, (be["max"][2] - bo["min"][2]) + 0.01 / max(1e-9, 1.0))


MAXID = [0]


def _a2p(p):
    import ifcopenshell.util.placement as pl
    return np.eye(4) if p is None else np.array(pl.get_axis2placement(p))


def solve(f, sc, it, tpl, t):
    """Scripted solution of one (sub-)task; returns the final message (answers)."""
    final = ""
    typ = it["type"]
    if typ == "translate":
        e = f.by_guid(it["guid"])
        if e.is_a("IfcDoor") or e.is_a("IfcWindow"):
            op = e.FillsVoids[0].RelatingOpeningElement; h.translate(f, op, (np.array(it["vector"]) / sc).tolist())
        else:
            h.translate(f, e, (np.array(it["vector"]) / sc).tolist())
    elif typ == "set_thickness":
        h.set_wall_thickness(f, it["guid"], it["thickness"] / sc)
    elif typ == "wall_axis" and tpl.startswith("U-shorten"):
        w = f.by_guid(it["guid"]); ax = h.wall_axis_world(w); a, b = np.array(ax[0][:2]) * sc, np.array(ax[1][:2]) * sc
        p0, p1 = np.array(it["p0"]), np.array(it["p1"])      # p0 = fixed end, p1 = new end
        end = "end" if np.linalg.norm(a - p0) < np.linalg.norm(b - p0) else "start"
        delta = float(np.linalg.norm(a - b) - np.linalg.norm(p0 - p1))
        h.stretch_wall_end(f, w, end, -delta / sc)
    elif typ == "wall_axis" and tpl.startswith("U-rotate"):
        h.rotate(f, it["guid"], float(t["meta"].get("angle", 90)), (np.array(t["meta"]["pivot"]) / sc).tolist())
    elif typ == "resize_height":
        e = f.by_guid(it["guid"]); old_h = None
        if it.get("top") is not None:
            bb = h._mesh_bbox(e); old_h = bb["max"][2] - bb["min"][2]
        h.set_extrusion_depth(f, e, it["height"] / sc)
        if old_h is not None:   # keep the top: the extrusion grows upward, so move the slab down by the thickness change
            h.translate(f, e, [0, 0, -(it["height"] / sc - old_h)])
    elif typ == "bbox":            # E010: every side that changes (x / y planes; z = extrusion depth with the bottom kept)
        e = f.by_guid(it["guid"]); bb = h._mesh_bbox(e)
        for ax, name in ((0, "x"), (1, "y")):
            for side, want in (("min", it["lo"][ax]), ("max", it["hi"][ax])):
                cur = bb[side][ax]
                if abs(want / sc - cur) > 1e-6:
                    vec = [0.0, 0.0, 0.0]; vec[ax] = want / sc - cur; h.shift_vertices_on_plane(f, e, name, cur, vec); bb = h._mesh_bbox(e)
        if len(it["lo"]) == 3:
            h.set_extrusion_depth(f, e, (it["hi"][2] - it["lo"][2]) / sc); bb = h._mesh_bbox(e)
            if abs(it["lo"][2] / sc - bb["min"][2]) > 1e-6:
                h.translate(f, e, [0, 0, it["lo"][2] / sc - bb["min"][2]])
    elif typ == "delete_all":
        for g_ in it["guids"]:
            e = f.by_guid(g_)
            if e.is_a("IfcDoor") or e.is_a("IfcWindow"):
                op = e.FillsVoids[0].RelatingOpeningElement; h.delete(f, e); h.delete(f, op)
            else:
                h.delete(f, e)
    elif typ == "resize_height_all":
        for g_ in it["guids"]:
            h.set_extrusion_depth(f, g_, it["height"] / sc)
    elif typ == "delete":
        e = f.by_guid(it["guid"]); cls = e.is_a()
        if e.is_a("IfcWall") or e.is_a("IfcSlab"):
            for r in list(e.HasOpenings or []):
                op = r.RelatedOpeningElement
                for fr in list(op.HasFillings or []):
                    h.delete(f, fr.RelatedBuildingElement)
                h.delete(f, op)
        elif e.is_a("IfcDoor") or e.is_a("IfcWindow"):
            op = e.FillsVoids[0].RelatingOpeningElement; h.delete(f, e); h.delete(f, op); op = None
        elif e.is_a("IfcOpeningElement"):
            for fr in list(e.HasFillings or []):
                h.delete(f, fr.RelatedBuildingElement)
        if cls not in ("IfcDoor", "IfcWindow"):
            h.delete(f, e)
    elif typ == "create_space":
        r_ = h.add_space(f, it["storey"], (np.array(it["polygon"]) / sc).tolist(), it["height"] / sc)
        if it.get("z0") is not None:
            e_ = f.by_guid(r_["guid"]); dz = it["z0"] / sc - h._mesh_bbox(e_)["min"][2]
            if abs(dz) > 1e-6:
                h.translate(f, e_, [0, 0, dz])
    elif typ == "create_wall_bbox":        # E010: centre line of the bbox along its long side, connected to the walls at its ends
        lo, hi = np.array(it["lo"]) / sc, np.array(it["hi"]) / sc; dx, dy = hi[0] - lo[0], hi[1] - lo[1]
        if dx >= dy:
            p0, p1, th = np.array([lo[0], (lo[1] + hi[1]) / 2]), np.array([hi[0], (lo[1] + hi[1]) / 2]), dy
        else:
            p0, p1, th = np.array([(lo[0] + hi[0]) / 2, lo[1]]), np.array([(lo[0] + hi[0]) / 2, hi[1]]), dx
        r_ = h.add_wall(f, it["storey"], p0, p1, th, hi[2] - lo[2]); new = f.by_guid(r_["guid"])
        dz = lo[2] - h._mesh_bbox(new)["min"][2]
        if abs(dz) > 1e-6:
            h.translate(f, new, [0, 0, dz])
        for d_ in t.get("dependents", []):
            if d_["rule"] == "connected_new":
                o = f.by_guid(d_["guid"]); ax = h.wall_axis_world(o); oa, ob = np.array(ax[0][:2]), np.array(ax[1][:2])
                o_end = "ATSTART" if min(np.linalg.norm(oa - p0), np.linalg.norm(oa - p1)) < min(np.linalg.norm(ob - p0), np.linalg.norm(ob - p1)) else "ATEND"
                n_end = "ATSTART" if min(np.linalg.norm(oa - p0), np.linalg.norm(ob - p0)) < min(np.linalg.norm(oa - p1), np.linalg.norm(ob - p1)) else "ATEND"
                h.connect_walls(f, o, new, o_end, n_end)
    elif typ == "set_thickness_side":     # E010: centred change, then shift by half the change towards n
        w = f.by_guid(it["guid"]); t0 = h.wall_thickness_any(w) if hasattr(h, "wall_thickness_any") else h._wall_thickness(w)
        h.set_wall_thickness(f, it["guid"], it["thickness"] / sc)
        n = np.array(it["n"]); h.translate(f, w, [float(n[0]) * (it["thickness"] / sc - t0) / 2, float(n[1]) * (it["thickness"] / sc - t0) / 2, 0.0])
    elif typ == "rehost_fill":            # E010: opening down to the new host and re-voided
        e = f.by_guid(it["guid"]); op = e.FillsVoids[0].RelatingOpeningElement; be = h._mesh_bbox(e); bo = h._mesh_bbox(op)
        cen = (np.array(be["min"][:2]) + np.array(be["max"][:2])) / 2; dxy = np.array(it["xy"]) / sc - cen; dz = it["zmin"] / sc - bo["min"][2]
        h.translate(f, op, [float(dxy[0]), float(dxy[1]), dz]); be2 = h._mesh_bbox(e)
        if np.linalg.norm((np.array(be2["min"][:2]) + np.array(be2["max"][:2])) / 2 - np.array(it["xy"]) / sc) > 0.01 / sc or abs(be2["min"][2] - be["min"][2] - dz) > 0.01 / sc:
            h.translate(f, e, [float(dxy[0]), float(dxy[1]), dz])     # fill not placed relative to the opening: move it too
        op.VoidsElements[0].RelatingBuildingElement = f.by_guid(it["host"])
    elif typ == "opening_width":          # E010: both planes along the wall axis move by half the change
        e = f.by_guid(it["guid"]); op = e.FillsVoids[0].RelatingOpeningElement; bb = h._mesh_bbox(op); u = np.array(it["u"]); ax = 0 if abs(u[0]) > 0.5 else 1
        w0 = bb["max"][ax] - bb["min"][ax]; d = (it["width"] / sc - w0) / 2; nm = "xy"[ax]
        vec = [0.0, 0.0, 0.0]; vec[ax] = -d; h.shift_vertices_on_plane(f, op, nm, bb["min"][ax], vec); bb = h._mesh_bbox(op)
        vec = [0.0, 0.0, 0.0]; vec[ax] = d; h.shift_vertices_on_plane(f, op, nm, bb["max"][ax], vec)
    elif typ == "create_slab_opening":    # E010: box opening through the slab
        import ifcopenshell.api.root, ifcopenshell.api.feature
        sl = f.by_guid(it["host"]); bb = h._mesh_bbox(sl); lo, hi = np.array(it["lo"]) / sc, np.array(it["hi"]) / sc; m = 0.05 / sc
        op = ifcopenshell.api.root.create_entity(f, ifc_class="IfcOpeningElement", name="Opening")
        op.ObjectPlacement = f.createIfcLocalPlacement(None, f.createIfcAxis2Placement3D(f.createIfcCartesianPoint([float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2), float(bb["min"][2] - m)]), f.createIfcDirection([0.0, 0.0, 1.0]), f.createIfcDirection([1.0, 0.0, 0.0])))
        prof = f.createIfcRectangleProfileDef("AREA", None, f.createIfcAxis2Placement2D(f.createIfcCartesianPoint([0.0, 0.0]), None), float(hi[0] - lo[0]), float(hi[1] - lo[1]))
        op.Representation = f.createIfcProductDefinitionShape(None, None, [f.createIfcShapeRepresentation(h._ctx(f, "Body"), "Body", "SweptSolid", [h._extrusion(f, prof, float(bb["max"][2] - bb["min"][2] + 2 * m))])])
        ifcopenshell.api.feature.add_feature(f, feature=op, element=sl)
    elif typ == "space_footprint":        # E010: merge - delete the wall (and its openings) and the other room, then re-profile the kept room
        for d_ in t.get("dependents", []):
            if d_["rule"] == "removed":
                try:
                    e_ = f.by_guid(d_["guid"])
                except Exception:  # noqa: BLE001
                    continue          # already removed with its wall
                for r__ in list(getattr(e_, "HasOpenings", None) or []):
                    for fr in list(r__.RelatedOpeningElement.HasFillings or []):
                        h.delete(f, fr.RelatedBuildingElement)
                    h.delete(f, r__.RelatedOpeningElement)
                h.delete(f, e_)
        sp = f.by_guid(it["guid"]); ex = sp.Representation.Representations[0].Items[0]
        while ex.is_a("IfcMappedItem"):
            ex = ex.MappingSource.MappedRepresentation.Items[0]
        M = np.linalg.inv(np.array(h.world_matrix(sp)) @ _a2p(ex.Position)); P = (np.array(it["polygon"]) / sc)
        loc = [tuple((M @ np.array([x, y, 0.0, 1.0]))[:2].tolist()) for x, y in P]
        ex.SweptArea = f.createIfcArbitraryClosedProfileDef("AREA", None, f.createIfcPolyline([f.createIfcCartesianPoint(list(map(float, p_))) for p_ in loc + [loc[0]]]))
    elif typ == "create_fill":
        w = f.by_guid(it["host"]); L = h.wall_length(w); W = it["width"] / sc
        ax = h.wall_axis_world(w); a = np.array(ax[0][:2]); u = np.array(ax[1][:2]) - a; u = u / (np.linalg.norm(u) or 1.0); busy = []
        for r_ in w.HasOpenings or []:   # occupied intervals along the axis (model units)
            bb = h._mesh_bbox(r_.RelatedOpeningElement)
            if bb:
                cs = [np.array([bb["min"][0], bb["min"][1]]), np.array([bb["max"][0], bb["max"][1]]), np.array([bb["min"][0], bb["max"][1]]), np.array([bb["max"][0], bb["min"][1]])]
                ps = [float((c_ - a) @ u) for c_ in cs]; busy.append((min(ps), max(ps)))
        along = None; step = 0.05 / sc
        x = 0.1 / sc
        if it.get("centre") is not None:       # E010: position given by the opening's bounding box
            along = float((np.array(it["centre"]) / sc - a) @ u) - W / 2; x = L
        while x + W <= L - 0.1 / sc:
            if all(x + W <= lo - 0.02 / sc or x >= hi + 0.02 / sc for lo, hi in busy):
                along = x; break
            x += step
        if along is None:
            raise ValueError("no free slot on the wall")
        h.add_opening_fill(f, w, along=along, width=W, height=it["height"] / sc, kind=it["kind"], sill=(it.get("sill", 0.9 if it["kind"] == "window" else 0.0) / sc))
    elif typ == "create_wall":
        r_ = h.add_wall(f, it["storey"], np.array(it["p0"]) / sc, np.array(it["p1"]) / sc, it["thickness"] / sc, it["height"] / sc)
        new = f.by_guid(r_["guid"]) if isinstance(r_, dict) and "guid" in r_ else None
        for d_ in t.get("dependents", []):
            if d_["rule"] == "connected_new" and new is not None:
                o = f.by_guid(d_["guid"]); ax = h.wall_axis_world(o); pa, pb = np.array(it["p0"]) / sc, np.array(it["p1"]) / sc
                oa, ob = np.array(ax[0][:2]), np.array(ax[1][:2])
                o_end = "ATSTART" if min(np.linalg.norm(oa - pa), np.linalg.norm(oa - pb)) < min(np.linalg.norm(ob - pa), np.linalg.norm(ob - pb)) else "ATEND"
                n_end = "ATSTART" if min(np.linalg.norm(oa - pa), np.linalg.norm(ob - pa)) < min(np.linalg.norm(oa - pb), np.linalg.norm(ob - pb)) else "ATEND"
                h.connect_walls(f, o, new, o_end, n_end)
    elif typ == "create_slab":
        r_ = h.add_slab(f, it["storey"], (np.array(it["polygon"]) / sc).tolist(), it["thickness"] / sc, elevation=-it["thickness"] / sc)
        if it.get("top") is not None:          # E010: slab on columns - top not at the storey level
            e_ = f.by_guid(r_["guid"]); dz = it["top"] / sc - h._mesh_bbox(e_)["max"][2]
            if abs(dz) > 1e-6:
                h.translate(f, e_, [0, 0, dz])
    elif typ == "create_column":
        if "sx" in it:              # E010: rectangular section, base elevation
            r_ = h.add_column(f, it["storey"], np.array(it["xy"]) / sc, 1.0, it["height"] / sc); col = f.by_guid(r_["guid"])
            ex = col.Representation.Representations[0].Items[0]; ex.SweptArea.XDim = it["sx"] / sc; ex.SweptArea.YDim = it["sy"] / sc
            if it.get("z0") is not None:
                z_now = h._mesh_bbox(col)["min"][2]; h.translate(f, col, [0, 0, it["z0"] / sc - z_now])
        else:
            h.add_column(f, it["storey"], np.array(it["xy"]) / sc, it["size"] / sc, it["height"] / sc)
    if typ in ("create_column", "create_slab", "translate", "create_wall_bbox", "create_space", "wall_axis", "space_footprint", "resize_height"):    # E010 relationships named in the task
        rel_deps(f, t)
    elif typ == "noop":
        pass
    elif typ == "answer":
        final = json.dumps({{"guid": "global_id", "guids": "global_ids"}.get(it["kind"], "value"): it["value"]})
    return final


for t in tasks:
    print("..", t["task_id"], file=sys.stderr, flush=True)
    f = ifcopenshell.open(t["ifc"]); sc = h.unit_scale(f) or 1.0; it = dict(t["intent"]); tpl = t["template"]; final = ""; MAXID[0] = max(e.id() for e in f)
    if NEG:   # perturb: scale vectors/values, or do nothing for structural edits
        if it["type"] == "all": pass       # compound: handled below (one part left undone)
        elif it["type"] == "translate": it["vector"] = [v * 0.5 for v in it["vector"]]
        elif it["type"] in ("set_thickness",): it["thickness"] = it["thickness"] + 0.1
        elif it["type"] == "resize_height": it["height"] = it["height"] + 0.3
        elif it["type"] == "answer": it["value"] = (it["value"] * 1.5 + 1) if it["kind"] == "number" else ("0000000000000000000000" if it["kind"] == "guid" else [])
        else: it["type"] = "noop"
    try:
        if it["type"] == "all":   # compound instruction (2026-10-26): solve every part in order; negative control leaves the last part undone
            parts = it["parts"]; tpls = t["meta"]["parts"]; metas = t["meta"].get("part_meta") or [{} for _ in parts]
            todo = list(range(len(parts)))[:-1] if NEG else list(range(len(parts)))
            for i_ in todo:
                final = solve(f, sc, dict(parts[i_]), tpls[i_], {**t, "meta": metas[i_], "template": tpls[i_], "intent": parts[i_]}) or final
        else:
            final = solve(f, sc, it, tpl, t)
        out = os.path.join(tempfile.mkdtemp(prefix="oracle_v4_"), "e.ifc"); f.write(out)
        r = check(t["ifc"], out, t, final)
        shutil.rmtree(os.path.dirname(out), ignore_errors=True)   # 29,768 leftover temp dirs on lustre, 2026-10-23
        res[tpl].append((not r.strict) if NEG else r.strict)
        if NEG and r.strict:
            print(f"NEG-PASSED {t['task_id']}: a wrong solution was accepted | {r.details['intent'][:120]}", flush=True)
        if not r.strict and not NEG:
            print(flush=True); print(f"FAIL {t['task_id']}: intent={r.intent} dep={r.dependency} pres={r.preservation} valid={r.validity} | {r.details['intent'][:140]} | {r.details['dependency'][:100]} | {r.details['preservation'][:120]} | {r.details['validity'][:60]} {r.details.get('new_errors', [])[:2]}")
    except Exception as e:  # noqa: BLE001
        res[tpl].append(False); print(f"EXC {t['task_id']}: {type(e).__name__}: {str(e)[:160]}", flush=True)
tot = 0; ok = 0
for k, v in sorted(res.items()):
    print(f"{k:32s} {sum(v)}/{len(v)}"); tot += len(v); ok += sum(v)
print(f"TOTAL {ok}/{tot}")
