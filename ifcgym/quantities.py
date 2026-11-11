"""IFC base quantities computed from geometry (E010, user 2026-10-26: BIBIMBAP numeric tasks fail because the student does not know
quantity take-off definitions; the old R-* numeric templates spelled the formula out in the prompt). Definitions follow the
buildingSMART IFC4X3 quantity-set templates shipped with ifcopenshell (ifcopenshell/util/schema/Pset_IFC4X3.ifc):

  Qto_WallBaseQuantities   Length (along centre line), Width (perpendicular to the path, constant), Height (nominal),
                           GrossSideArea (elevation of the middle plane, openings NOT deducted), NetSideArea (openings deducted),
                           GrossVolume (openings, recesses not taken into account), NetVolume (after subtracting the openings).
  Qto_SlabBaseQuantities   Width (thickness), Perimeter (outer boundary), GrossArea (openings not taken into account),
                           NetArea (openings subtracted), GrossVolume, NetVolume (openings subtracted).
  Qto_SpaceBaseQuantities  Height, GrossPerimeter (outer contour), GrossFloorArea (includes the area covered by elements inside the
                           space, e.g. columns), NetFloorArea (excludes it), GrossVolume (body volume), NetVolume (excludes construction elements inside); Height only for prismatic spaces.
  Qto_OpeningElementBaseQuantities  Area (elevation view for wall openings, plan view for slab openings), Volume (the subtraction
                           volume from the voided element), Depth.
  Qto_BuildingStoreyBaseQuantities  GrossFloorArea (includes the area of construction elements within the storey), NetFloorArea
                           (sum of the spaces' net floor areas).
  Qto_ColumnBaseQuantities CrossSectionArea, OuterSurfaceArea (perimeter x length, no end caps), GrossVolume.

Everything here is in metres and comes from the triangulated geometry (ifcopenshell.geom, world coordinates); the gate re-computes
the same quantities from the IFC representation (profiles x extrusion depths) in scripts/verify_qto.py."""
from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from ifcgym.checkers import _tri_mesh


def volume(inst, no_openings: bool) -> float | None:
    """Enclosed volume of the product's mesh (divergence theorem), m3."""
    t = _tri_mesh(inst, no_openings)
    if t is None or len(t[1]) == 0:
        return None
    v, fc = t; A, B, C = v[fc[:, 0]], v[fc[:, 1]], v[fc[:, 2]]
    return float(abs(np.einsum("ij,ij->i", A, np.cross(B, C)).sum()) / 6.0)


def plan(inst, no_openings: bool):
    """Plan footprint: union of the mesh triangles projected on XY (shapely, m)."""
    t = _tri_mesh(inst, no_openings)
    if t is None or len(t[1]) == 0:
        return None
    v, fc = t; polys = []
    for f_ in fc:
        p = v[f_][:, :2]
        if abs((p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[2, 0] - p[0, 0]) * (p[1, 1] - p[0, 1])) > 1e-9:
            polys.append(Polygon(p))
    if not polys:
        return None
    return unary_union(polys).buffer(1e-4).buffer(-1e-4)


def z_extent(inst) -> tuple[float, float] | None:
    t = _tri_mesh(inst, True)
    if t is None or len(t[0]) == 0:
        return None
    return float(t[0][:, 2].min()), float(t[0][:, 2].max())


def opening_rect_on_wall(c, w, op):
    """Elevation rectangle (s0, s1, z0, z1) of an opening on the wall's axis frame, clipped to the wall's extent (m)."""
    a, b = c.axis[w.GlobalId]; L = float(np.linalg.norm(b - a)); u = (b - a) / (L or 1.0)
    t = _tri_mesh(op, True); wz = z_extent(w)
    if t is None or wz is None:
        return None
    s = (t[0][:, :2] - a) @ u; z = t[0][:, 2]
    s0, s1 = max(0.0, float(s.min())), min(L, float(s.max())); z0, z1 = max(wz[0], float(z.min())), min(wz[1], float(z.max()))
    return (s0, s1, z0, z1) if (s1 > s0 and z1 > z0) else None


def wall(c, w) -> dict | None:
    """Qto_WallBaseQuantities of one wall."""
    wz = z_extent(w); L = c.length(w); W = c.thick(w)
    if wz is None or not L or not W:
        return None
    # side areas = volume / width: the elevation area of the middle plane for a wall of constant thickness, exact also for clipped
    # (gable) bodies and for the real hole an opening cuts; scripts/verify_qto.py checks it against length x height minus opening rectangles
    from ifcgym.tasks.generate import _openings
    vg, vn = volume(w, True), volume(w, False)
    if vg is None or vn is None:
        return None
    return {"Length": L, "Width": W, "Height": wz[1] - wz[0], "GrossSideArea": vg / W, "NetSideArea": vn / W, "GrossVolume": vg, "NetVolume": vn,
            "n_openings": len(_openings(w))}


def slab(sl) -> dict | None:
    """Qto_SlabBaseQuantities of one slab."""
    pg, pn = plan(sl, True), plan(sl, False); z = z_extent(sl)
    if pg is None or pn is None or z is None:
        return None
    outer = unary_union([Polygon(g.exterior) for g in getattr(pg, "geoms", [pg])])
    return {"Width": z[1] - z[0], "Perimeter": float(outer.length), "GrossArea": float(outer.area), "NetArea": float(pn.area),
            "GrossVolume": volume(sl, True), "NetVolume": volume(sl, False)}


def column(col) -> dict | None:
    p = plan(col, True); z = z_extent(col)
    if p is None or z is None:
        return None
    Lc = z[1] - z[0]
    return {"Length": Lc, "CrossSectionArea": float(p.area), "OuterSurfaceArea": float(p.exterior.length) * Lc if hasattr(p, "exterior") else None,
            "GrossVolume": volume(col, True)}


def space(c, s) -> dict | None:
    """Qto_SpaceBaseQuantities; columns standing in the space are the 'elements inside the space' (net = gross minus their plan area)."""
    fp = c.fp(s.GlobalId); z = z_extent(s)
    if fp is None or z is None:
        return None
    H = z[1] - z[0]; outer = Polygon(fp.exterior)
    cols = []
    for col in c.columns:
        pc = plan(col, True)
        if pc is not None and pc.intersection(fp).area > 1e-6:
            cols.append(pc.intersection(fp))
    inside = unary_union(cols).area if cols else 0.0
    vg = volume(s, True)      # the body's volume: floor area x height only for prismatic spaces (attic under a sloped roof: review 2026-10-26)
    if vg is None:
        return None
    prismatic = abs(vg - float(fp.area) * H) <= 0.01 * vg
    return {"Height": H if prismatic else None, "GrossPerimeter": float(outer.length), "GrossFloorArea": float(fp.area), "NetFloorArea": float(fp.area) - inside,
            "GrossVolume": vg, "NetVolume": vg - inside * H, "n_columns": len(cols)}


def opening(c, op) -> dict | None:
    """Qto_OpeningElementBaseQuantities: Area as seen in elevation (wall) or plan (slab); Volume = subtraction volume from the host."""
    host = op.VoidsElements[0].RelatingBuildingElement if getattr(op, "VoidsElements", None) else None
    if host is None:
        return None
    if host.is_a("IfcWall") and host.GlobalId in c.axis:
        r = opening_rect_on_wall(c, host, op)
        if r is None:
            return None
        A = (r[1] - r[0]) * (r[3] - r[2]); D = c.thick(host)
    elif host.is_a("IfcSlab"):
        po, ph = plan(op, True), plan(host, True); z = z_extent(host)
        if po is None or ph is None or z is None:
            return None
        A = float(po.intersection(ph).area); D = z[1] - z[0]
    else:
        return None
    return {"Area": A, "Depth": D, "Volume": A * D, "host": host.GlobalId}


def storey(c, st) -> dict | None:
    """Qto_BuildingStoreyBaseQuantities: GrossFloorArea = plan area of the spaces together with the walls and columns of the storey
    (construction elements included; gaps < 10 cm closed; one outline with holes <= 1 % of its area, else None); NetFloorArea = sum of the spaces' NetFloorArea."""
    sps = [s for s in c.spaces if c.storey_of(s) == st]; ws = [w for w in c.walls if c.storey_of(w) == st]
    if not sps or not ws:
        return None
    parts = [c.fp(s.GlobalId) for s in sps] + [plan(w, True) for w in ws] + [plan(col, True) for col in c.columns if c.storey_of(col) == st]
    if any(p is None for p in parts):
        return None
    U = unary_union(parts).buffer(0.05).buffer(-0.05)    # closes the < 10 cm gaps between room polygons and wall faces
    if U.geom_type != "Polygon":
        return None                                       # several separate outlines: which belong to the storey's floor is unclear
    holes = sum(Polygon(h).area for h in U.interiors)     # floor not covered by any space (CubiCasa leftovers): counted or not is
    if holes > 0.01 * U.area:                             # ambiguous, so only storeys where both readings agree within 1 %
        return None
    U = Polygon(U.exterior)
    nets = [space(c, s) for s in sps]
    if any(n is None for n in nets):
        return None
    return {"GrossFloorArea": float(U.area), "NetFloorArea": sum(n["NetFloorArea"] for n in nets), "n_spaces": len(sps)}
