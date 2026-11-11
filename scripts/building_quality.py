"""Quality metrics for every synthetic building (select high-quality ones). Writes <root>/quality.jsonl with one
record per building: plan metrics (rooms, areas, fragments, overlaps, door connectivity), IFC metrics (schema errors, boundaries
per space, connections per wall, stairs) and hard flags. Run inside the python:3.12 container; shardable.
    python scripts/building_quality.py --root <root> --shard 0 --nshards 4"""
import argparse, collections, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, ifcopenshell, ifcopenshell.validate, logging
from shapely.geometry import Polygon
from ifcgym.ifctools import wall_axis_world, profile_world_polygon, unit_scale

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
a = ap.parse_args()
dirs = sorted(d for d in glob.glob(os.path.join(a.root, "synth_*")) if os.path.exists(os.path.join(d, "arc.ifc")))
dirs = [d for i, d in enumerate(dirs) if i % a.nshards == a.shard]
out = open(os.path.join(a.root, f"quality_{a.shard}.jsonl"), "w")
for d in dirs:
    bid = os.path.basename(d); rec = {"id": bid, "flags": []}
    try:
        f = ifcopenshell.open(os.path.join(d, "arc.ifc")); sc = unit_scale(f) or 1.0
        errs = []
        class L(logging.Handler):
            def emit(self, r): errs.append(r.getMessage())
        log = logging.getLogger("q" + bid); log.setLevel(logging.ERROR); log.addHandler(L()); ifcopenshell.validate.validate(f, log)
        rec["schema_errors"] = len(errs)
        sparse = bid.endswith("_sparse")
        storeys = f.by_type("IfcBuildingStorey"); walls = f.by_type("IfcWall"); spaces = f.by_type("IfcSpace"); doors = f.by_type("IfcDoor")
        rec["storeys"] = len(storeys); rec["walls"] = len(walls); rec["spaces"] = len(spaces); rec["doors"] = len(doors)
        # per-storey plan metrics on the first storey (plans repeat on every storey)
        import ifcopenshell.util.element as ue
        st0 = min(storeys, key=lambda s: float(s.Elevation or 0))
        sp0 = [s for s in spaces if ue.get_aggregate(s) == st0]; w0 = [w for w in walls if ue.get_container(w) == st0]
        from ifcgym.checkers import _footprint
        polys = []; poly_of = {}
        for s in sp0:
            fp = _footprint(s)          # mesh footprint: works for extruded, Brep and face-set spaces alike
            if fp is not None and not fp.is_empty:
                polys.append(fp); poly_of[s.GlobalId] = fp
        areas = [p.area for p in polys]
        rec["rooms"] = len(sp0); rec["area_min"] = round(min(areas), 2) if areas else None; rec["area_max"] = round(max(areas), 2) if areas else None
        ov = 0
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                if polys[i].intersection(polys[j]).area > 0.05 * min(polys[i].area, polys[j].area):
                    ov += 1
        rec["room_overlaps"] = ov
        lens = []
        for w in w0:
            ax = wall_axis_world(w)
            if ax:
                lens.append(float(np.linalg.norm(np.array(ax[1][:2]) - np.array(ax[0][:2])) * sc))
        rec["wall_fragments"] = sum(1 for L_ in lens if L_ < 0.3); rec["wall_len_median"] = round(float(np.median(lens)), 2) if lens else None
        # geometric checks for every building (sparse ones have no relations to read): room edges on wall faces, door connectivity
        wall_fp = []
        for w in w0:
            ax = wall_axis_world(w)
            if not ax:
                continue
            from ifcgym.ifctools import _wall_thickness
            t = _wall_thickness(w) * sc
            if t <= 0:
                bb = None
                from ifcgym.checkers import mesh as _m
                v = _m(w)
                if v is not None:
                    a_, b_ = np.array(ax[0][:2]) * sc, np.array(ax[1][:2]) * sc; u_ = (b_ - a_) / (np.linalg.norm(b_ - a_) or 1.0); n_ = np.array([-u_[1], u_[0]])
                    pn = (v[:, :2] - a_) @ n_; t = float(pn.max() - pn.min())
            a_, b_ = np.array(ax[0][:2]) * sc, np.array(ax[1][:2]) * sc
            from shapely.geometry import LineString
            wall_fp.append((w, LineString([a_, b_]).buffer(max(t, 0.1) / 2 + 0.03, cap_style=2), a_, b_))
        from shapely.ops import unary_union
        wall_union = unary_union([g for _, g, _, _ in wall_fp]) if wall_fp else None
        edges_total = 0; edges_on = 0
        for p_ in polys:
            cs = list(p_.exterior.coords)
            for i in range(len(cs) - 1):
                a_, b_ = np.array(cs[i]), np.array(cs[i + 1])
                if np.linalg.norm(b_ - a_) < 0.2:
                    continue
                edges_total += 1
                from shapely.geometry import Point as _P
                if wall_union is not None and wall_union.distance(_P((a_ + b_) / 2)) < 0.05 and wall_union.distance(_P(a_)) < 0.1 and wall_union.distance(_P(b_)) < 0.1:
                    edges_on += 1
        rec["edge_on_wall_frac"] = round(edges_on / edges_total, 3) if edges_total else None
        # doors: the two rooms on either side of the door (nearest polygon within 0.3 m along the wall normal)
        from ifcgym.checkers import mesh as _m
        adj = collections.defaultdict(set); rooms_with_door = set()
        sp_ids = [s.GlobalId for s in sp0]
        for dr in doors:
            try:
                w = dr.FillsVoids[0].RelatingOpeningElement.VoidsElements[0].RelatingBuildingElement
                if ue.get_container(dr) != st0:
                    continue
                v = _m(dr); ax = wall_axis_world(w)
                if v is None or not ax:
                    continue
                c = v[:, :2].mean(0); a_, b_ = np.array(ax[0][:2]) * sc, np.array(ax[1][:2]) * sc; u_ = (b_ - a_) / (np.linalg.norm(b_ - a_) or 1.0); n_ = np.array([-u_[1], u_[0]])
                from shapely.geometry import Point as _P
                sides = []
                for sgn in (1, -1):
                    q = _P(c + n_ * sgn * 0.35)
                    hits = sorted((poly_of[g].distance(q), g) for g in poly_of if poly_of[g].distance(q) < 0.15)
                    sides.append(hits[0][1] if hits else None)
                for sd in sides:
                    if sd: rooms_with_door.add(sd)
                if sides[0] and sides[1] and sides[0] != sides[1]:
                    adj[sides[0]].add(sides[1]); adj[sides[1]].add(sides[0])
            except Exception:  # noqa: BLE001
                continue
        seen = set(); comp = 0; largest = 0
        for s_ in sp_ids:
            if s_ in seen:
                continue
            comp += 1; stack = [s_]; size = 0
            while stack:
                x = stack.pop()
                if x in seen:
                    continue
                seen.add(x); size += 1; stack.extend(adj.get(x, ()))
            largest = max(largest, size)
        rec["door_components"] = comp; rec["largest_component_frac"] = round(largest / max(1, len(sp_ids)), 2); rec["rooms_without_door"] = sum(1 for s_ in sp_ids if s_ not in rooms_with_door)
        if not sparse:
            bnd = collections.Counter(r.RelatingSpace.GlobalId for r in f.by_type("IfcRelSpaceBoundary") if r.RelatingSpace)
            rec["spaces_lt3_boundaries"] = sum(1 for s_ in spaces if bnd.get(s_.GlobalId, 0) < 3)
            conn = collections.Counter()
            for r in f.by_type("IfcRelConnectsPathElements"):
                conn[r.RelatingElement.GlobalId] += 1; conn[r.RelatedElement.GlobalId] += 1
            rec["walls_unconnected"] = sum(1 for w in walls if conn.get(w.GlobalId, 0) == 0)
        rec["stairs"] = len(f.by_type("IfcStair"))
        # coverage: how much of the storey's slab footprint is covered by rooms (hallways missing from the plan annotation lower it)
        slabs0 = [x for x in f.by_type("IfcSlab") if ue.get_container(x) == st0]
        cov = None
        if slabs0 and polys:
            sfp = _footprint(slabs0[0])
            if sfp is not None and sfp.area > 0:
                from shapely.ops import unary_union as _uu
                cov = round(_uu(polys).intersection(sfp).area / sfp.area, 3)
        rec["room_coverage"] = cov
        # flags (hard exclusions)
        if rec["schema_errors"]: rec["flags"].append("schema")
        if rec["rooms"] < 3: rec["flags"].append("few_rooms")
        if areas and (min(areas) < 2.0 or max(areas) > 120.0): rec["flags"].append("odd_area")
        if ov: rec["flags"].append("overlap")
        if lens and rec["wall_fragments"] > 0.15 * len(lens): rec["flags"].append("fragments")
        if not sparse and rec.get("spaces_lt3_boundaries", 0) > 0.3 * max(1, len(spaces)): rec["flags"].append("boundaries")
        if not sparse and rec.get("walls_unconnected", 0) > 0.2 * max(1, len(walls)): rec["flags"].append("connections")
        if rec.get("edge_on_wall_frac") is None or rec["edge_on_wall_frac"] < 0.6: rec["flags"].append("edges_off_wall")
        if rec.get("room_coverage") is not None and rec["room_coverage"] < 0.5: rec["flags"].append("low_coverage")   # door connectivity stays a soft score
        if rec.get("rooms_without_door", 0) > 0.3 * max(1, rec["rooms"]): rec["flags"].append("doorless")
        if len(storeys) >= 2 and rec["stairs"] == 0: rec["flags"].append("no_stair")
        # soft score 0..1
        pen = 0.0
        pen += 0.3 * (rec["wall_fragments"] / max(1, len(lens))) if lens else 0.5
        pen += 0.2 * (rec.get("rooms_without_door", 0) / max(1, rec["rooms"]))
        pen += 0.2 * (1.0 - rec.get("largest_component_frac", 1.0))
        pen += 0.2 * (rec.get("spaces_lt3_boundaries", 0) / max(1, len(spaces)))
        rec["score"] = round(max(0.0, 1.0 - pen), 3)
    except Exception as e:  # noqa: BLE001
        rec["flags"].append("exception"); rec["error"] = f"{type(e).__name__}: {str(e)[:100]}"; rec["score"] = 0.0
    out.write(json.dumps(rec) + "\n"); out.flush()
print("done", len(dirs))
