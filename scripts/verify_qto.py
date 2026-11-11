"""Independent check of the R-qto-* answers. The generator (ifcgym.quantities) takes every quantity from the
triangulated geometry of ifcopenshell.geom (boolean openings included); this script re-derives it from the IFC representation alone,
without the geometry engine:
  * bodies: vertical IfcExtrudedAreaSolid (profile area x depth) or IfcFacetedBrep / IfcPolygonalFaceSet (volume of the stored
    polyhedron by the divergence theorem); anything else (clipping, sloped extrusions, several items) is 'unhandled', never guessed;
  * openings: the 8 corners of the opening's extruded rectangle placed in world coordinates, projected on the host's frame (along the
    wall axis / across the wall / vertical, or plan for slabs) and clipped to the host; net = gross - the clipped opening boxes;
  * side areas: axis length x height minus the clipped elevation rectangles of the openings.
A task is a mismatch when the two values differ by more than the task's own tolerance.
    python scripts/verify_qto.py <tasks.jsonl> [--out mismatch_ids.txt] [--ok checked_ok_ids.txt]"""
import collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import ifcopenshell
import ifcopenshell.util.placement as pl
from shapely.geometry import Polygon
from shapely.ops import unary_union
from ifcgym.ifctools import unit_scale, world_matrix

args = [a for a in sys.argv[1:] if not a.startswith("--")]; out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
ok_out = sys.argv[sys.argv.index("--ok") + 1] if "--ok" in sys.argv else None     # ids that were checked and agree (lists keep only these)
tasks = [json.loads(l) for l in open(args[0]) if l.strip() and '"R-qto-' in l]


class Unhandled(Exception):
    pass


def a2p(p):
    """4x4 of an IfcAxis2Placement2D/3D (or identity)."""
    if p is None:
        return np.eye(4)
    if p.is_a("IfcAxis2Placement2D"):
        M = np.eye(4); M[:2, 3] = p.Location.Coordinates[:2]
        if p.RefDirection is not None:
            x = np.array(p.RefDirection.DirectionRatios[:2], float); x /= np.linalg.norm(x); M[:2, 0] = x; M[:2, 1] = [-x[1], x[0]]
        return M
    return np.array(pl.get_axis2placement(p))


def ring(cu):
    if cu.is_a("IfcPolyline"):
        return [tuple(p.Coordinates[:2]) for p in cu.Points]
    if cu.is_a("IfcIndexedPolyCurve") and not cu.Segments:
        return [tuple(p[:2]) for p in cu.Points.CoordList]
    raise Unhandled(f"curve {cu.is_a()}")


def profile(pr):
    """Shapely polygon in the extrusion's local XY (model units)."""
    if pr.is_a("IfcRectangleProfileDef"):
        x, y = pr.XDim / 2, pr.YDim / 2; M = a2p(pr.Position)
        return Polygon([tuple((M @ np.array([a, b, 0, 1]))[:2]) for a, b in ((-x, -y), (x, -y), (x, y), (-x, y))])
    if pr.is_a("IfcArbitraryClosedProfileDef"):
        return Polygon(ring(pr.OuterCurve), [ring(c) for c in (pr.InnerCurves if pr.is_a("IfcArbitraryProfileDefWithVoids") else [])])
    raise Unhandled(f"profile {pr.is_a()}")


def body_item(e):
    items = []
    for r in (e.Representation.Representations if e.Representation else []):
        if r.RepresentationIdentifier == "Body":
            for it in r.Items:
                M = np.eye(4)
                while it.is_a("IfcMappedItem"):
                    M = M @ a2p(it.MappingSource.MappingOrigin); it = it.MappingSource.MappedRepresentation.Items[0]
                items.append((it, M))
    if len(items) != 1:
        raise Unhandled(f"body items {len(items)}")
    return items[0]


def poly_faces(it):
    """Polygons (n x 3 arrays, item coordinates) of a faceted Brep / polygonal face set."""
    if it.is_a("IfcFacetedBrep"):
        out = []
        for fc in it.Outer.CfsFaces:
            if len(fc.Bounds) != 1:
                raise Unhandled("brep face with holes")
            out.append(np.array([p.Coordinates for p in fc.Bounds[0].Bound.Polygon], float))
        return out
    if it.is_a("IfcPolygonalFaceSet"):
        C = np.array(it.Coordinates.CoordList, float); out = []
        for fc in it.Faces:
            if fc.is_a("IfcIndexedPolygonalFaceWithVoids"):
                raise Unhandled("face set face with voids")
            out.append(C[np.array(fc.CoordIndex) - 1])
        return out
    raise Unhandled(f"body {it.is_a()}")


def body(e, sc):
    """dict: volume (m3), z0, z1 (world, m), plan (world shapely, m; extrusions only), pts (world points, m)."""
    it, Mi = body_item(e); W = world_matrix(e) @ Mi
    if it.is_a("IfcExtrudedAreaSolid"):
        P = profile(it.SweptArea); Mp = W @ a2p(it.Position)
        d = np.array(it.ExtrudedDirection.DirectionRatios, float); d /= np.linalg.norm(d); dw = Mp[:3, :3] @ d
        base = [(Mp @ np.array([x, y, 0, 1]))[:3] for x, y in P.exterior.coords[:-1]]
        pts = np.array(base + [b + dw * it.Depth for b in base]) * sc
        plan = None     # the plan outline is the profile only for a vertical extrusion of a horizontal profile
        if abs(dw[2]) > 0.999 and abs(Mp[2, 2]) > 0.999:
            holes = [[tuple((Mp @ np.array([x, y, 0, 1]))[:2] * sc) for x, y in h.coords] for h in P.interiors]
            plan = Polygon([tuple(p[:2]) for p in pts[:len(base)]], holes)
        return {"volume": P.area * it.Depth * sc ** 3, "z0": float(pts[:, 2].min()), "z1": float(pts[:, 2].max()), "plan": plan, "pts": pts}
    F = poly_faces(it); vol = 0.0
    for p in F:                                  # divergence theorem on a fan triangulation of each planar face
        for i in range(1, len(p) - 1):
            vol += np.dot(p[0], np.cross(p[i], p[i + 1])) / 6.0
    Fw = [np.array([(W @ np.append(q, 1))[:3] for q in p]) * sc for p in F]; pts = np.vstack(Fw); z0 = float(pts[:, 2].min())
    bottom = [Polygon(p[:, :2]) for p in Fw if np.all(np.abs(p[:, 2] - z0) < 1e-4) and Polygon(p[:, :2]).area > 1e-8]
    plan = unary_union(bottom).buffer(1e-5).buffer(-1e-5) if bottom else None   # plan = the bottom faces of the polyhedron
    if plan is not None and plan.geom_type != "Polygon":
        plan = None
    return {"volume": abs(vol) * sc ** 3, "z0": z0, "z1": float(pts[:, 2].max()), "plan": plan, "pts": pts}


def wall_frame(w, sc):
    for r in w.Representation.Representations:
        if r.RepresentationIdentifier == "Axis" and r.Items[0].is_a("IfcPolyline"):
            P = [np.array(list(p.Coordinates[:2]) + [0.0], float) for p in r.Items[0].Points]
            M = world_matrix(w); a = (M @ np.append(P[0], 1))[:2] * sc; b = (M @ np.append(P[-1], 1))[:2] * sc
            L = float(np.linalg.norm(b - a)); u = (b - a) / L; return a, u, np.array([-u[1], u[0]]), L
    raise Unhandled("no polyline axis")


def openings(e):
    return [r.RelatedOpeningElement for r in (e.HasOpenings or [])]


def clip_box(q, a, u, n, L, Bh, pn):
    """(elevation area, subtraction volume) of an opening's corner points q against a wall frame, clipped to the wall."""
    s = (q[:, :2] - a) @ u; nn = (q[:, :2] - a) @ n
    ds = max(0.0, min(L, s.max()) - max(0.0, s.min())); dz = max(0.0, min(Bh["z1"], q[:, 2].max()) - max(Bh["z0"], q[:, 2].min()))
    dn = max(0.0, min(pn.max(), nn.max()) - max(pn.min(), nn.min()))
    return ds * dz, ds * dz * dn


def wall_vals(w, sc):
    B = body(w, sc); a, u, n, L = wall_frame(w, sc); H = B["z1"] - B["z0"]; pn = (B["pts"][:, :2] - a) @ n
    elev = sub = 0.0
    for o in openings(w):
        e_, v_ = clip_box(body(o, sc)["pts"], a, u, n, L, B, pn); elev += e_; sub += v_
    return {"GrossSideArea": L * H, "NetSideArea": L * H - elev, "GrossVolume": B["volume"], "NetVolume": B["volume"] - sub}


def opening_vals(o, sc):
    host = o.VoidsElements[0].RelatingBuildingElement; q = body(o, sc)["pts"]; Bh = body(host, sc)
    if host.is_a("IfcWall"):
        a, u, n, L = wall_frame(host, sc); pn = (Bh["pts"][:, :2] - a) @ n
        A, V = clip_box(q, a, u, n, L, Bh, pn); return {"Area": A, "Volume": V}
    if Bh["plan"] is None:
        raise Unhandled("slab plan needs an extrusion")
    A = float(Polygon(q[:, :2]).convex_hull.intersection(Polygon(Bh["plan"].exterior)).area)
    dz = max(0.0, min(Bh["z1"], q[:, 2].max()) - max(Bh["z0"], q[:, 2].min()))
    return {"Area": A, "Volume": A * dz}


def slab_vals(sl, sc):
    B = body(sl, sc)
    if B["plan"] is None:
        raise Unhandled("slab plan needs an extrusion")
    T = B["z1"] - B["z0"]; outer = Polygon(B["plan"].exterior); ops = sum(opening_vals(o, sc)["Area"] for o in openings(sl))
    G = outer.area; N = B["plan"].area - ops
    return {"GrossArea": G, "NetArea": N, "Perimeter": outer.length, "GrossVolume": G * T, "NetVolume": N * T}


def space_vals(f, s, sc):
    B = body(s, sc)
    if B["plan"] is None:
        raise Unhandled("space plan needs an extrusion")
    A = B["plan"].area; H = B["z1"] - B["z0"]; inside = []
    for col in f.by_type("IfcColumn"):    # elements inside the space: columns whose height range overlaps the space's
        Bc = body(col, sc)
        if Bc["z1"] > B["z0"] + 0.01 and Bc["z0"] < B["z1"] - 0.01:
            pc = Bc["plan"] if Bc["plan"] is not None else Polygon(Bc["pts"][:, :2]).convex_hull
            if pc.intersection(B["plan"]).area > 1e-6:
                inside.append(pc.intersection(B["plan"]))
    Ai = unary_union(inside).area if inside else 0.0
    return {"GrossFloorArea": A, "NetFloorArea": A - Ai, "GrossPerimeter": Polygon(B["plan"].exterior).length, "Height": H,
            "GrossVolume": B["volume"], "NetVolume": B["volume"] - Ai * H}


def contained(st, cls):
    els = [e for r in (st.ContainsElements or []) for e in r.RelatedElements if e.is_a(cls)]
    if cls == "IfcSpace":
        els += [o for a in (st.IsDecomposedBy or []) for o in a.RelatedObjects if o.is_a("IfcSpace")]
    return els


def value(f, sc, t):
    tpl = t["template"]; prop = t["meta"]["qto"].split(".")[1]
    if tpl.startswith("R-qto-walls-"):
        return sum(wall_vals(f.by_guid(g), sc)[prop] for g in t["target"])
    if tpl.startswith("R-qto-storey-walls-"):
        return sum(wall_vals(w, sc)[prop] for w in contained(f.by_guid(t["target"][0]), "IfcWall"))
    if tpl.startswith("R-qto-wall-"):
        return wall_vals(f.by_guid(t["target"][0]), sc)[prop]
    if tpl.startswith("R-qto-slab-opening-"):
        return opening_vals(f.by_guid(t["target"][0]), sc)[prop]
    if tpl.startswith("R-qto-slab-"):
        return slab_vals(f.by_guid(t["target"][0]), sc)[prop]
    if tpl.startswith("R-qto-space-"):
        return space_vals(f, f.by_guid(t["target"][0]), sc)[prop]
    if tpl.startswith("R-qto-opening-"):
        return opening_vals(f.by_guid(t["target"][1]), sc)[prop]
    if tpl.startswith("R-qto-storey-"):
        st = f.by_guid(t["target"][0]); sps = contained(st, "IfcSpace")
        if prop == "NetFloorArea":
            return sum(space_vals(f, s, sc)["NetFloorArea"] for s in sps)
        parts = [body(e, sc)["plan"] for e in sps + contained(st, "IfcWall") + contained(st, "IfcColumn")]
        if any(p is None for p in parts):
            raise Unhandled("storey outline needs extrusions")
        U = unary_union(parts).buffer(0.05).buffer(-0.05)
        if U.geom_type != "Polygon":
            raise Unhandled("several outlines")
        return Polygon(U.exterior).area
    raise Unhandled(tpl)


by = collections.defaultdict(lambda: [0, 0, 0]); bad = []; good = []; cache = {}; why = collections.Counter()
for t in tasks:
    if t["ifc"] not in cache:
        cache.clear(); f_ = ifcopenshell.open(t["ifc"]); cache[t["ifc"]] = (f_, unit_scale(f_) or 1.0)
    f, sc = cache[t["ifc"]]; key = t["template"].rsplit("-", 1)[0]; want = t["intent"]["value"]
    try:
        got = value(f, sc, t)
    except Unhandled as e:
        by[key][2] += 1; why[str(e)[:60]] += 1; continue
    except Exception as e:  # noqa: BLE001
        by[key][2] += 1; why[f"{type(e).__name__}: {str(e)[:50]}"] += 1; continue
    tol = max(t["intent"].get("tol", 0.0), t["intent"].get("rel_tol", 0.0) * abs(want))
    if abs(got - want) > tol:
        by[key][1] += 1; bad.append(t["task_id"]); print(f"MISMATCH {t['task_id']}: generator {want} vs representation {got:.3f} | {t['prompt'][:110]}")
    else:
        by[key][0] += 1; good.append(t["task_id"])
for k, (ok, ko, un) in sorted(by.items()):
    print(f"{k:36s} ok {ok:4d} mismatch {ko:3d} unhandled {un:4d}")
print(f"TOTAL checked {sum(v[0] + v[1] for v in by.values())} mismatches {len(bad)} unhandled {sum(v[2] for v in by.values())}")
print("unhandled because:", dict(why.most_common(12)))
if out:
    open(out, "w").write("\n".join(bad) + ("\n" if bad else ""))
if ok_out:
    open(ok_out, "w").write("\n".join(good) + ("\n" if good else ""))
