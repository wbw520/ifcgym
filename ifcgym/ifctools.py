"""Tools-v2 helpers (H001/E007, user 2026-10-20: the BIM-Edit single-tool protocol need not be kept for comparability).

Two things live here, both imported into the sandbox worker as `h`:
  1. `inspect(ifc, guid)` - the structured, read-only `inspect_element` tool: one call returns what an agent otherwise spends
     10-15 exploratory calls (and attribute-name errors) on: placement chain, representation summary, openings/fills, connected
     walls with connection types, space boundaries, bounded spaces, property sets, shared-reference flags.
  2. edit primitives - low-level, element-local operations (translate a placement, stretch a wall at one end, move profile
     vertices on a world plane, move a space-boundary curve, set a property on a PRIVATE pset, delete a product). They do not
     decide WHAT to propagate; the cascade (which dependents, which end, which sign) stays with the agent - that is the knowledge
     being trained. Every primitive first gives the edited entity a private copy of any sub-entity shared with other products
     (the wall #76 output-E failure: a shared IfcCartesianPoint moved two walls).
Geometry conventions: world coordinates in MODEL units (see `unit_scale`); extruded profiles live in the item's XY plane.
"""
from __future__ import annotations

from typing import Any

import numpy as np

import ifcopenshell
import ifcopenshell.api
import ifcopenshell.api.attribute
import ifcopenshell.api.geometry
import ifcopenshell.api.pset
import ifcopenshell.api.root
import ifcopenshell.guid
import ifcopenshell.util.element as ue
import ifcopenshell.util.placement as up

TOL = 0.02   # METRES (converted to model units inside the helpers); the generated tasks use 0.25-1.0 m moves


def _tol(f, tol=None) -> float:
    """tolerance in model units: TOL metres / unit scale (0.02 m -> 20 in a millimetre model)."""
    return (TOL if tol is None else tol) / (unit_scale(f) or 1.0)


_BODY_TYPES = {"SweptSolid", "AdvancedSweptSolid", "Brep", "AdvancedBrep", "Tessellation", "Clipping", "CSG", "SurfaceModel", "MappedRepresentation"}


def _is_rep(pds, r, identifier: str) -> bool:
    """r is the `identifier` representation; a body without RepresentationIdentifier (some Revit exports: ($,'Tessellation',...))
    counts as 'Body' when the product has no explicit 'Body' representation (E011 BIM-Edit SLB-UPD-DIR-R-001: helpers moved 0 vertices)."""
    if r.RepresentationIdentifier == identifier:
        return True
    return (identifier == "Body" and not r.RepresentationIdentifier and r.RepresentationType in _BODY_TYPES
            and not any(q.RepresentationIdentifier == "Body" for q in pds.Representations))


# ------------------------------------------------------------------------------------------ small utilities
def _pt(f, coords):
    return f.createIfcCartesianPoint(tuple(float(c) for c in coords))


def _r(x, n=4):
    try:
        return [round(float(v), n) for v in x]
    except TypeError:
        return round(float(x), n)


def private(f, parent, attr: str, idx: int | None = None):
    """Return parent.attr (or parent.attr[idx]) after making it a private copy when other entities also reference it."""
    child = getattr(parent, attr) if idx is None else getattr(parent, attr)[idx]
    if child is not None and len(f.get_inverse(child)) > 1:
        child = ue.copy(f, child)
        if idx is None:
            setattr(parent, attr, child)
        else:
            items = list(getattr(parent, attr)); items[idx] = child; setattr(parent, attr, items)
    return child


def unit_scale(f) -> float:
    """metres per model length unit (1.0 for a model in metres, 0.001 for millimetres)."""
    try:
        import ifcopenshell.util.unit as uu
        return float(uu.calculate_unit_scale(f))
    except Exception:  # noqa: BLE001
        return 1.0


def world_matrix(product) -> np.ndarray:
    """4x4 world transform of a product's ObjectPlacement (identity if none)."""
    if getattr(product, "ObjectPlacement", None) is None:
        return np.eye(4)
    return np.array(up.get_local_placement(product.ObjectPlacement), dtype=float)


def _rep_items(product) -> dict[str, list]:
    out: dict[str, list] = {}
    rep = getattr(product, "Representation", None)
    for r in (rep.Representations if rep else []):
        out.setdefault(r.RepresentationIdentifier or r.RepresentationType or "?", []).extend(list(r.Items))
    return out


def _extrusion_of(item):
    """The IfcExtrudedAreaSolid at the root of a Body item: the item itself, or the FirstOperand chain of an IfcBooleanResult
    (Revit exports clip wall extrusions with half-spaces under sloped roofs). None for Breps and other solids."""
    while item is not None and item.is_a("IfcBooleanResult"):
        item = item.FirstOperand
    return item if item is not None and item.is_a("IfcExtrudedAreaSolid") else None


def _private_extrusion(f, rep_, j):
    """Same as _extrusion_of for editing: every entity on the way is made private first; returns the extrusion or None."""
    it = private(f, rep_, "Items", j)
    while it is not None and it.is_a("IfcBooleanResult"):
        it = private(f, it, "FirstOperand")
    return it if it is not None and it.is_a("IfcExtrudedAreaSolid") else None


def _item_frame(product, item) -> np.ndarray:
    m = world_matrix(product)
    pos = getattr(item, "Position", None)
    if pos is not None:
        m = m @ np.array(up.get_axis2placement(pos), dtype=float)
    return m


def _profile_frame(product, item) -> tuple[Any, np.ndarray]:
    """(profile, 4x4 world frame of the profile's 2D coordinates) for an IfcExtrudedAreaSolid item."""
    m = _item_frame(product, item)
    pr = item.SweptArea
    pos = getattr(pr, "Position", None)
    if pos is not None:
        q = np.eye(4)
        d = pos.RefDirection.DirectionRatios if pos.RefDirection else (1.0, 0.0)
        q[:2, 0], q[:2, 1], q[:2, 3] = d[:2], (-d[1], d[0]), pos.Location.Coordinates[:2]
        m = m @ q
    return pr, m


def _profile_local_points(pr) -> list[tuple[float, float]] | None:
    if pr.is_a("IfcRectangleProfileDef"):
        x, y = pr.XDim / 2, pr.YDim / 2
        return [(-x, -y), (x, -y), (x, y), (-x, y)]
    if pr.is_a("IfcArbitraryClosedProfileDef") and pr.OuterCurve.is_a("IfcPolyline"):
        pts = [tuple(p.Coordinates[:2]) for p in pr.OuterCurve.Points]
        return pts[:-1] if len(pts) > 1 and pts[0] == pts[-1] else pts
    return None


def profile_world_polygon(product, item=None) -> list[list[float]] | None:
    """World XY polygon of a product's extruded Body profile (None if the profile is not rectangle / polyline)."""
    if item is None:
        items = [e for e in (_extrusion_of(i) for i in _rep_items(product).get("Body", [])) if e is not None]
        if not items:
            return None
        item = items[0]
    pr, m = _profile_frame(product, item)
    pts = _profile_local_points(pr)
    if pts is None:
        return None
    return [_r((m @ np.array([a, b, 0, 1]))[:3]) for a, b in pts]


def wall_axis_world(wall) -> list[list[float]] | None:
    """World end points of a wall's axis (Axis polyline, else the Body rectangle's long axis)."""
    m = world_matrix(wall)
    for it in _rep_items(wall).get("Axis", []):
        if it.is_a("IfcPolyline") and len(it.Points) >= 2:
            p0, p1 = it.Points[0].Coordinates, it.Points[-1].Coordinates
            return [_r((m @ np.array([p0[0], p0[1], 0, 1]))[:3]), _r((m @ np.array([p1[0], p1[1], 0, 1]))[:3])]
    for it in _rep_items(wall).get("Body", []):
        it = _extrusion_of(it)
        if it is not None and it.SweptArea.is_a("IfcRectangleProfileDef"):
            pr, mp = _profile_frame(wall, it)
            x = pr.XDim / 2
            return [_r((mp @ np.array([-x, 0, 0, 1]))[:3]), _r((mp @ np.array([x, 0, 0, 1]))[:3])]
    return None


def _wall_thickness(wall) -> float:
    for it in _rep_items(wall).get("Body", []):
        it = _extrusion_of(it)
        if it is not None:
            pr = it.SweptArea
            if pr.is_a("IfcRectangleProfileDef"):
                return float(pr.YDim)
            pts = _profile_local_points(pr)
            if pts:
                ys = [b for a, b in pts]; return float(max(ys) - min(ys))
    return 0.0


def wall_thickness_any(wall) -> float:
    """Wall thickness in MODEL units: the extrusion profile, else (Brep / faceset bodies, review 2026-10-24) the extent of the body mesh
    across the wall axis. _wall_thickness alone returns 0 for those bodies, which made _touching_end miss joined walls (helper-only
    cascade on T3 tasks: 60 % on Brep / faceset walls vs 98 % on extrusions) and add_opening_fill cut 0.2 m deep openings."""
    t = _wall_thickness(wall)
    if t > 0:
        return t
    try:
        from ifcgym.checkers import mesh
        v = mesh(wall); ax = wall_axis_world(wall); sc = unit_scale(wall.file) or 1.0
        if v is None or ax is None:
            return 0.0
        a, b = np.array(ax[0][:2]) * sc, np.array(ax[1][:2]) * sc; u = (b - a) / (np.linalg.norm(b - a) or 1.0); n = np.array([-u[1], u[0]])
        pn = np.asarray(v)[:, :2] @ n
        return float(pn.max() - pn.min()) / sc
    except Exception:  # noqa: BLE001
        return 0.0


def _touching_end(wall, other_axis, tol: float | None = None, other=None) -> str | None:
    """'start' / 'end' if that end point of the other wall's axis lies on this wall (within half its thickness + tol of the axis
    segment), else None - a geometric 'is joined' test independent of the IfcRelConnectsPathElements flags."""
    ax = wall_axis_world(wall)
    if ax is None or other_axis is None:
        return None
    tol = _tol(wall.file, tol); t = wall_thickness_any(wall); t2 = wall_thickness_any(other) if other is not None else t
    a, b = np.array(ax[0][:2]), np.array(ax[1][:2]); ab = b - a; L2 = float(ab @ ab) or 1.0
    best = None
    for name, p in (("start", np.array(other_axis[0][:2])), ("end", np.array(other_axis[1][:2]))):
        u = max(0.0, min(1.0, float((p - a) @ ab) / L2)); d = float(np.linalg.norm(p - (a + u * ab)))
        if d <= (t + t2) / 2 + tol and (best is None or d < best[1]):   # mitred/butt corners: the axis end sits up to a thickness away
            best = (name, d)
    return best[0] if best else None


def _mesh_bbox(product):
    """World bounding box in MODEL units (ifcopenshell.geom meshes are in metres: divided by the unit scale)."""
    try:
        from ifcgym.checkers import mesh
        v = mesh(product)
        if v is None:
            return None
        sc = unit_scale(product.file) or 1.0
        return {"min": _r(v.min(0) / sc), "max": _r(v.max(0) / sc)}
    except Exception:  # noqa: BLE001
        return None


def _item_summary(product, item) -> dict:
    d: dict[str, Any] = {"type": item.is_a(), "id": item.id()}
    if item.is_a("IfcExtrudedAreaSolid"):
        pr = item.SweptArea
        d["profile"] = {"type": pr.is_a(), "id": pr.id()}
        if pr.is_a("IfcRectangleProfileDef"):
            d["profile"].update({"XDim": _r(pr.XDim), "YDim": _r(pr.YDim)})
        poly = profile_world_polygon(product, item)
        if poly is not None:
            d["profile"]["world_xy_polygon"] = [p[:2] for p in poly]
        d["depth"] = _r(item.Depth); d["extrude_direction_world"] = _r((_item_frame(product, item)[:3, :3] @ np.array(item.ExtrudedDirection.DirectionRatios, dtype=float)))
    elif item.is_a("IfcPolyline"):
        m = world_matrix(product)
        d["world_points"] = [_r((m @ np.array([*list(p.Coordinates[:2]), 0, 1]))[:3]) for p in item.Points][:12]
    elif item.is_a("IfcMappedItem"):
        d["note"] = "mapped item (shared type geometry): edit the placement, not the mapped representation"
    elif item.is_a("IfcBooleanResult") or item.is_a("IfcBooleanClippingResult"):
        d["operands"] = [item.FirstOperand.is_a(), item.SecondOperand.is_a()]
    return d


def _shared(f, inst) -> bool:
    return inst is not None and len(f.get_inverse(inst)) > 1


def _rel_boundary(f, b, m0=None) -> dict:
    d = {"rel_guid": b.GlobalId, "rel_id": b.id(), "space_guid": b.RelatingSpace.GlobalId if b.RelatingSpace else None,
         "space_name": getattr(b.RelatingSpace, "Name", None) if b.RelatingSpace else None,
         "element_guid": b.RelatedBuildingElement.GlobalId if b.RelatedBuildingElement else None,
         "element_class": b.RelatedBuildingElement.is_a() if b.RelatedBuildingElement else None,
         "physical_or_virtual": b.PhysicalOrVirtualBoundary, "internal_or_external": b.InternalOrExternalBoundary}
    g = b.ConnectionGeometry.SurfaceOnRelatingElement if b.ConnectionGeometry else None
    if g is not None:
        d["geometry"] = {"type": g.is_a(), "id": g.id()}
        try:
            m = (np.array(up.get_local_placement(b.RelatingSpace.ObjectPlacement), dtype=float) if m0 is None else m0)
            if g.is_a("IfcSurfaceOfLinearExtrusion") and g.SweptCurve.Curve.is_a("IfcPolyline"):
                mm = m @ np.array(up.get_axis2placement(g.Position), dtype=float)
                pts = []
                for p in g.SweptCurve.Curve.Points:
                    c = list(p.Coordinates) + [0.0] * (3 - len(p.Coordinates))
                    pts.append(_r((mm @ np.array([c[0], c[1], c[2], 1]))[:3]))
                d["geometry"]["world_points"] = pts[:12]; d["geometry"]["depth"] = _r(g.Depth)
            elif g.is_a("IfcCurveBoundedPlane") and g.OuterBoundary.is_a("IfcPolyline"):
                mm = m @ np.array(up.get_axis2placement(g.BasisSurface.Position), dtype=float)
                d["geometry"]["world_points"] = [_r((mm @ np.array([*(list(p.Coordinates) + [0.0] * (3 - len(p.Coordinates))), 1]))[:3]) for p in g.OuterBoundary.Points][:12]
        except Exception as e:  # noqa: BLE001
            d["geometry"]["error"] = str(e)[:80]
    return d


# ------------------------------------------------------------------------------------------ the inspection tool
def inspect(f, guid: str, psets: bool = False, geometry: bool = True, related_to: str | None = None) -> dict:
    """Structured description of one product: identity, placement, representation, dependents, property sets, shared references.
    related_to=<guid>: for a space, list only its boundaries with that element (large spaces have hundreds)."""
    try:
        e = f.by_guid(guid)
    except Exception:  # noqa: BLE001
        return {"error": f"no entity with GlobalId {guid!r}"}
    out: dict[str, Any] = {"guid": guid, "id": e.id(), "class": e.is_a(), "name": getattr(e, "Name", None),
                           "predefined_type": getattr(e, "PredefinedType", None), "object_type": getattr(e, "ObjectType", None),
                           "unit_scale_m_per_unit": unit_scale(f)}
    try:
        c = ue.get_container(e)
        out["container"] = {"class": c.is_a(), "guid": c.GlobalId, "name": c.Name} if c is not None else None
    except Exception:  # noqa: BLE001
        pass
    pl = getattr(e, "ObjectPlacement", None)
    if pl is not None:
        m = world_matrix(e)
        rp = getattr(pl, "RelativePlacement", None)
        parent = getattr(pl, "PlacementRelTo", None)
        users = [x for x in f.get_inverse(pl) if x.is_a("IfcProduct")] if pl else []
        children = [x for x in f.get_inverse(pl) if x.is_a("IfcLocalPlacement")]
        out["placement"] = {"placement_id": pl.id(), "local_origin": _r(rp.Location.Coordinates) if rp is not None and hasattr(rp, "Location") else None,
                            "world_origin": _r(m[:3, 3]), "world_x_axis": _r(m[:3, 0]), "world_z_axis": _r(m[:3, 2]),
                            "placement_rel_to_id": parent.id() if parent is not None else None,
                            "products_sharing_this_placement": [x.GlobalId for x in users if x != e][:8],
                            "child_placements": len(children), "relative_placement_shared": _shared(f, rp)}
    if geometry:
        reps = _rep_items(e)
        out["representations"] = {k: [_item_summary(e, it) for it in v[:4]] for k, v in reps.items()}
        for k, v in reps.items():
            for it in v[:4]:
                if it.is_a("IfcExtrudedAreaSolid") and _shared(f, it.SweptArea):
                    out.setdefault("shared_geometry", []).append(f"{k}: profile #{it.SweptArea.id()} is shared with other products")
        if e.is_a("IfcWall") or e.is_a("IfcSpace") or e.is_a("IfcSlab"):
            out["bbox_world"] = _mesh_bbox(e)
    if e.is_a("IfcWall"):
        out["axis_world"] = wall_axis_world(e)
        ops = []
        for r in getattr(e, "HasOpenings", []) or []:
            o = r.RelatedOpeningElement
            ops.append({"guid": o.GlobalId, "name": o.Name, "world_origin": _r(world_matrix(o)[:3, 3]),
                        "placement_relative_to_wall": (o.ObjectPlacement.PlacementRelTo == e.ObjectPlacement) if o.ObjectPlacement else False,
                        "fills": [{"guid": x.RelatedBuildingElement.GlobalId, "class": x.RelatedBuildingElement.is_a(), "name": x.RelatedBuildingElement.Name,
                                   "placement_relative_to_opening": (x.RelatedBuildingElement.ObjectPlacement.PlacementRelTo == o.ObjectPlacement) if x.RelatedBuildingElement.ObjectPlacement else False}
                                  for x in (o.HasFillings or [])]})
        out["openings"] = ops
        cw = []
        for r in list(getattr(e, "ConnectedTo", []) or []) + list(getattr(e, "ConnectedFrom", []) or []):
            if not r.is_a("IfcRelConnectsPathElements"):
                continue
            relating = r.RelatingElement == e
            o = r.RelatedElement if relating else r.RelatingElement
            oax = wall_axis_world(o)
            cw.append({"rel_guid": r.GlobalId, "rel_id": r.id(), "other_guid": o.GlobalId, "other_name": o.Name, "other_class": o.is_a(),
                       "this_wall_end": r.RelatingConnectionType if relating else r.RelatedConnectionType,
                       "other_wall_end": r.RelatedConnectionType if relating else r.RelatingConnectionType,
                       "other_axis_world": oax, "other_end_touching_this_wall": _touching_end(e, oax, other=o), "other_has_openings": len(o.HasOpenings or []) > 0})
        merged = {}
        for c in cw:   # one entry per other wall; all relation guids kept
            k = c["other_guid"]
            if k in merged:
                merged[k]["rel_guids"] = merged[k].get("rel_guids", [merged[k]["rel_guid"]]) + [c["rel_guid"]]
            else:
                merged[k] = c
        out["connected_walls"] = list(merged.values())
        out["space_boundaries"] = [_rel_boundary(f, b) for b in (getattr(e, "ProvidesBoundaries", []) or [])][:16]; out["space_boundary_count"] = len(getattr(e, "ProvidesBoundaries", []) or [])
        spaces = {}
        for b in getattr(e, "ProvidesBoundaries", []) or []:
            if b.RelatingSpace is not None and b.RelatingSpace.GlobalId not in spaces:
                s = b.RelatingSpace
                spaces[s.GlobalId] = {"guid": s.GlobalId, "name": s.Name, "long_name": getattr(s, "LongName", None), "footprint_world_xy": [p[:2] for p in (profile_world_polygon(s) or [])][:80]}
        out["bounded_spaces"] = list(spaces.values())
    if e.is_a("IfcOpeningElement"):
        out["voids"] = [{"guid": r.RelatingBuildingElement.GlobalId, "class": r.RelatingBuildingElement.is_a()} for r in (e.VoidsElements or [])]
        out["fills"] = [{"guid": x.RelatedBuildingElement.GlobalId, "class": x.RelatedBuildingElement.is_a()} for x in (e.HasFillings or [])]
    if e.is_a("IfcDoor") or e.is_a("IfcWindow"):
        fv = getattr(e, "FillsVoids", []) or []
        out["fills_opening"] = [{"opening_guid": r.RelatingOpeningElement.GlobalId,
                                 "host_wall_guid": [v.RelatingBuildingElement.GlobalId for v in (r.RelatingOpeningElement.VoidsElements or [])]} for r in fv]
    if e.is_a("IfcSpace"):
        out["footprint_world_xy"] = [p[:2] for p in (profile_world_polygon(e) or [])]
        m0 = world_matrix(e)
        bs = [b for b in (getattr(e, "BoundedBy", []) or []) if related_to is None or (b.RelatedBuildingElement is not None and b.RelatedBuildingElement.GlobalId == related_to)]
        out["boundaries"] = [_rel_boundary(f, b, m0) for b in bs][:24]; out["boundary_count"] = len(bs)
    if psets:
        try:
            ps = ue.get_psets(e)
            trimmed = {}
            for name, props in list(ps.items())[:20]:
                pid = props.get("id")
                shared = False
                if pid:
                    try:
                        shared = len(ue.get_elements_by_pset(f.by_id(pid))) > 1
                    except Exception:  # noqa: BLE001
                        pass
                trimmed[name] = {"id": pid, "shared_with_other_elements": shared, **{k: v for k, v in list(props.items())[:40] if k != "id"}}
            out["psets"] = trimmed
        except Exception as e2:  # noqa: BLE001
            out["psets"] = {"error": str(e2)[:80]}
    return out


# ------------------------------------------------------------------------------------------ edit primitives
def translate(f, product, vec_world) -> dict:
    """Move a product by a WORLD vector (model units) by editing its own IfcLocalPlacement. Products whose placement is relative
    to this one (e.g. a wall's openings and doors) move with it automatically - do not move them again."""
    if isinstance(product, str):
        product = f.by_guid(product)
    pl = product.ObjectPlacement
    if pl is None or not pl.is_a("IfcLocalPlacement"):
        raise ValueError("product has no IfcLocalPlacement")
    others = [x for x in f.get_inverse(pl) if x.is_a("IfcProduct") and x != product]
    if others:   # the placement itself is shared by other products: give this product a private one
        pl = ue.copy(f, pl); product.ObjectPlacement = pl
    parent = np.array(up.get_local_placement(pl.PlacementRelTo), dtype=float) if pl.PlacementRelTo else np.eye(4)
    d = np.linalg.inv(parent[:3, :3]) @ np.array(vec_world, dtype=float)
    rp = private(f, pl, "RelativePlacement")
    loc = private(f, rp, "Location")
    c = list(loc.Coordinates) + [0.0] * (3 - len(loc.Coordinates))
    loc.Coordinates = tuple(float(c[i] + d[i]) for i in range(len(loc.Coordinates)))
    return {"guid": product.GlobalId, "world_origin": _r(world_matrix(product)[:3, 3]), "children_moved_with_it": len([x for x in f.get_inverse(pl) if x.is_a("IfcLocalPlacement")])}


def set_extrusion_depth(f, product, depth: float, identifier: str = "Body") -> dict:
    """Set the Depth (height for a wall) of the product's extruded solid(s) in the given representation."""
    if isinstance(product, str):
        product = f.by_guid(product)
    n = 0
    pds = private(f, product, "Representation")
    for i, r in enumerate(pds.Representations):
        if not _is_rep(pds, r, identifier):
            continue
        r = private(f, pds, "Representations", i)
        for j, it0 in enumerate(list(r.Items)):
            if _extrusion_of(it0) is not None:        # plain extrusion, or one inside an IfcBooleanClippingResult (Revit walls under roofs)
                it = _private_extrusion(f, r, j); it.Depth = float(depth); n += 1
    if n == 0:   # Brep / face-set bodies: move the top face vertices (world z = current top) by the height change
        bb = _mesh_bbox(product)
        if bb is None:
            raise ValueError("no extrusion and no geometry to edit")
        h_now = bb["max"][2] - bb["min"][2]
        r_ = shift_vertices_on_plane(f, product, "z", float(bb["max"][2]), [0.0, 0.0, float(depth - h_now)], identifier=identifier)
        n = 1 if r_.get("vertices_moved") else 0
    return {"guid": product.GlobalId, "items_changed": n}


def stretch_wall_end(f, wall, end: str, delta: float, tol: float | None = None) -> dict:
    """Lengthen (delta > 0) or shorten (delta < 0) a wall at one end ('start' = axis start, 'end' = far end) along its own axis,
    keeping the other end fixed. The placement is NOT moved (so openings/doors keep their entities untouched): the Axis polyline
    end point and the Body profile vertices at that end move. Rectangle profiles are converted to polylines when the start end
    moves (the rectangle is centred on its Position)."""
    if isinstance(wall, str):
        wall = f.by_guid(wall)
    if end not in ("start", "end"):
        raise ValueError("end must be 'start' or 'end'")
    tol = _tol(f, tol)
    reps = _rep_items(wall)
    bodies = [e for e in (_extrusion_of(it) for it in reps.get("Body", [])) if e is not None and _profile_local_points(e.SweptArea) is not None]
    if not bodies:   # Brep / other solids: move the vertices on the end plane (axis-aligned walls), then update the Axis polyline below
        ax = wall_axis_world(wall)
        if ax is None:
            raise ValueError("wall has neither an extruded profile nor an Axis polyline")
        p0, p1 = np.array(ax[0][:2]), np.array(ax[1][:2]); u = (p1 - p0) / (np.linalg.norm(p1 - p0) or 1.0)
        if abs(u[0]) < 0.98 and abs(u[1]) < 0.98:
            raise ValueError("Brep wall is not axis-aligned; edit it with shift_vertices_on_plane by hand")
        axis_name = "x" if abs(u[0]) >= 0.98 else "y"; other = "y" if axis_name == "x" else "x"; k = 0 if axis_name == "x" else 1
        end_pt = p1 if end == "end" else p0; sgn = 1.0 if end == "end" else -1.0
        bb = _mesh_bbox(wall); t_half = (abs(bb["max"][1 - k] - bb["min"][1 - k]) / 2 + tol) if bb else 1.0
        vec = (u * sgn * delta).tolist() + [0.0]
        shift_vertices_on_plane(f, wall, axis_name, float(end_pt[k]), vec, within={other: (float(end_pt[1 - k]) - t_half, float(end_pt[1 - k]) + t_half)})   # tol: let it derive (passing model units would be rescaled again - mm bug 2026-10-22)
    pds = private(f, wall, "Representation")
    for i, r in enumerate(pds.Representations):
        if _is_rep(pds, r, "Body") and bodies:
            r = private(f, pds, "Representations", i)
            for j, it0 in enumerate(list(r.Items)):
                e0 = _extrusion_of(it0)
                if e0 is None or _profile_local_points(e0.SweptArea) is None:
                    continue
                it = _private_extrusion(f, r, j); pr = private(f, it, "SweptArea")
                if pr.is_a("IfcRectangleProfileDef"):
                    pr.XDim = float(pr.XDim + delta)
                    pos = private(f, pr, "Position"); loc = private(f, pos, "Location")
                    cx, cy = loc.Coordinates[:2]
                    loc.Coordinates = (float(cx + (delta / 2 if end == "end" else -delta / 2)), float(cy))
                else:
                    curve = private(f, pr, "OuterCurve")
                    pts = [tuple(p.Coordinates[:2]) for p in curve.Points]
                    xs = [a for a, b in pts]; xmin, xmax = min(xs), max(xs)
                    if end == "end":
                        new = [(a + delta, b) if abs(a - xmax) <= tol else (a, b) for a, b in pts]
                    else:
                        new = [(a - delta, b) if abs(a - xmin) <= tol else (a, b) for a, b in pts]
                    curve.Points = [_pt(f, q) for q in new]
        elif r.RepresentationIdentifier == "Axis":
            r = private(f, pds, "Representations", i)
            for j, it in enumerate(r.Items):
                if it.is_a("IfcPolyline") and len(it.Points) >= 2:
                    it = private(f, r, "Items", j)
                    pts = list(it.Points); k = len(pts) - 1 if end == "end" else 0
                    pk = private(f, it, "Points", k)
                    x, y = pk.Coordinates[:2]; pk.Coordinates = (float(x + (delta if end == "end" else -delta)), float(y))
    return {"guid": wall.GlobalId, "axis_world": wall_axis_world(wall)}


def move_wall_end(f, wall, end: str, vec_world) -> dict:
    """Move ONE end of a wall ('start' or 'end') by a world vector, projected on the wall's axis, keeping the other end fixed:
    the way a joined wall follows a wall that was moved (its touching end moves by the same vector)."""
    if isinstance(wall, str):
        wall = f.by_guid(wall)
    ax = wall_axis_world(wall)
    if ax is None:
        raise ValueError("wall has no usable axis / extruded profile")
    p0, p1 = np.array(ax[0]), np.array(ax[1]); u = (p1 - p0) / (np.linalg.norm(p1 - p0) or 1.0)
    sgn = float(np.array(vec_world, dtype=float) @ u)
    return stretch_wall_end(f, wall, end, sgn if end == "end" else -sgn)


def _match(world_pt, axis: str | None, value: float | None, vec_world, within, tol) -> bool:
    if axis is not None:
        i = "xyz".index(axis)
        if abs(world_pt[i] - value) > tol:
            return False
    for ax, (lo, hi) in (within or {}).items():
        k = "xyz".index(ax)
        if not (lo - tol <= world_pt[k] <= hi + tol):
            return False
    return True


def shift_profile_vertices(f, product, plane_axis: str, plane_value: float, vec_world, within: dict | None = None, tol: float | None = None, identifier: str = "Body") -> dict:
    """Move the vertices of the product's extruded profile that lie on the world plane {plane_axis = plane_value} (optionally only
    within world ranges, e.g. within={'x': (3.0, 9.5)}) by vec_world. Rectangle profiles are converted to a polyline profile first.
    Typical use: a space's footprint edge that sat on a wall face follows the moved wall."""
    if isinstance(product, str):
        product = f.by_guid(product)
    tol = _tol(f, tol); moved = 0
    pds = private(f, product, "Representation")
    for i, r in enumerate(pds.Representations):
        if not _is_rep(pds, r, identifier):
            continue
        r = private(f, pds, "Representations", i)
        for j, it in enumerate(r.Items):
            if not it.is_a("IfcExtrudedAreaSolid"):
                continue
            it = private(f, r, "Items", j)
            pr, m = _profile_frame(product, it)
            pts = _profile_local_points(pr)
            if pts is None:
                continue
            d = (np.linalg.inv(m[:3, :3]) @ np.array(vec_world, dtype=float))[:2]
            new = []
            for a, b in pts:
                w = (m @ np.array([a, b, 0, 1]))[:3]
                if _match(w, plane_axis, plane_value, vec_world, within, tol):
                    new.append((a + d[0], b + d[1])); moved += 1
                else:
                    new.append((a, b))
            if not any(n != o for n, o in zip(new, pts)):
                continue                                   # nothing on that plane: leave the profile untouched
            if pr.is_a("IfcRectangleProfileDef"):
                # replace by an arbitrary closed profile: its points are in the ITEM's XY frame, so apply the rectangle's Position
                q = np.eye(3); pos = getattr(pr, "Position", None)
                if pos is not None:
                    dd = pos.RefDirection.DirectionRatios if pos.RefDirection else (1.0, 0.0)
                    q[:2, 0], q[:2, 1], q[:2, 2] = dd[:2], (-dd[1], dd[0]), pos.Location.Coordinates[:2]
                new = [tuple((q @ np.array([a, b, 1.0]))[:2]) for a, b in new]
                poly = f.createIfcPolyline([_pt(f, p) for p in new] + [_pt(f, new[0])])
                it.SweptArea = f.createIfcArbitraryClosedProfileDef("AREA", pr.ProfileName, poly)
            else:
                pr = private(f, it, "SweptArea"); curve = private(f, pr, "OuterCurve")
                curve.Points = [_pt(f, p) for p in new] + [_pt(f, new[0])]
    return {"guid": product.GlobalId, "vertices_moved": moved, "footprint_world_xy": [p[:2] for p in (profile_world_polygon(product) or [])]}


def _axis2(pl):
    return np.array(up.get_axis2placement(pl), dtype=float) if pl is not None else np.eye(4)


def shift_vertices_on_plane(f, product, plane_axis: str, plane_value: float, vec_world, within: dict | None = None, tol: float | None = None, identifier: str = "Body") -> dict:
    """Move every vertex of the product's geometry that lies on the world plane {plane_axis = plane_value} (optionally within
    world ranges) by vec_world, whatever the representation: extruded profiles (rectangle -> polyline), faceted Breps, polylines,
    indexed poly-curves, boolean results (operands recursed). Mapped items (shared type geometry) are reported, not edited.
    The general form of shift_profile_vertices for models whose spaces are Breps or clipped solids (ArchiCAD/Revit exports)."""
    if isinstance(product, str):
        product = f.by_guid(product)
    tol = _tol(f, tol); v = np.array(vec_world, dtype=float); moved = 0; skipped = []
    pds = private(f, product, "Representation")
    base = world_matrix(product)

    def shift_pt(point, m, dims):
        nonlocal moved
        c = list(point.Coordinates) + [0.0] * (3 - len(point.Coordinates))
        w = (m @ np.array([c[0], c[1], c[2], 1.0]))[:3]
        if _match(w, plane_axis, plane_value, v, within, tol):
            d = np.linalg.inv(m[:3, :3]) @ v
            moved += 1
            return tuple(float(c[i] + d[i]) for i in range(dims))
        return tuple(float(c[i]) for i in range(dims))

    def do_curve(owner, attr, m):
        curve = private(f, owner, attr)
        if curve.is_a("IfcPolyline"):
            pts = list(curve.Points); new = []
            for q in pts:
                dims = len(q.Coordinates); new.append(_pt(f, shift_pt(q, m, dims)))
            curve.Points = new
        elif curve.is_a("IfcIndexedPolyCurve"):
            pl = private(f, curve, "Points"); dims = 2 if pl.is_a("IfcCartesianPointList2D") else 3
            new = []
            for c in pl.CoordList:
                c3 = list(c) + [0.0] * (3 - len(c)); w = (m @ np.array([c3[0], c3[1], c3[2], 1.0]))[:3]
                if _match(w, plane_axis, plane_value, v, within, tol):
                    d = np.linalg.inv(m[:3, :3]) @ v; moved += 1; new.append(tuple(float(c3[i] + d[i]) for i in range(dims)))
                else:
                    new.append(tuple(float(c3[i]) for i in range(dims)))
            pl.CoordList = new
        else:
            skipped.append(curve.is_a())

    def do_item(owner, attr, idx, m):
        nonlocal moved
        item = private(f, owner, attr, idx)
        if item.is_a("IfcExtrudedAreaSolid"):
            mi = m @ _axis2(item.Position)
            pr = item.SweptArea
            if pr.is_a("IfcRectangleProfileDef"):
                q = np.eye(4); pos = pr.Position
                if pos is not None:
                    dd = pos.RefDirection.DirectionRatios if pos.RefDirection else (1.0, 0.0)
                    q[:2, 0], q[:2, 1], q[:2, 3] = dd[:2], (-dd[1], dd[0]), pos.Location.Coordinates[:2]
                x, y = pr.XDim / 2, pr.YDim / 2
                pts = [(-x, -y), (x, -y), (x, y), (-x, y)]
                mp = mi @ q; new = []; hit = False
                for a, b in pts:
                    w = (mp @ np.array([a, b, 0, 1.0]))[:3]
                    if _match(w, plane_axis, plane_value, v, within, tol):
                        d = np.linalg.inv(mp[:3, :3]) @ v; moved += 1; hit = True; new.append((a + d[0], b + d[1]))
                    else:
                        new.append((a, b))
                if hit:
                    new = [tuple((q @ np.array([a, b, 0, 1.0]))[:2]) for a, b in new]
                    poly = f.createIfcPolyline([_pt(f, pt) for pt in new] + [_pt(f, new[0])])
                    item.SweptArea = f.createIfcArbitraryClosedProfileDef("AREA", pr.ProfileName, poly)
            elif pr.is_a("IfcArbitraryClosedProfileDef"):
                pr = private(f, item, "SweptArea"); mp = mi @ _axis2(getattr(pr, "Position", None)) if hasattr(pr, "Position") else mi
                do_curve(pr, "OuterCurve", mp)
                if pr.is_a("IfcArbitraryProfileDefWithVoids"):
                    for k in range(len(pr.InnerCurves)):
                        inner = private(f, pr, "InnerCurves", k); do_curve(pr, "InnerCurves", mp) if False else None
            else:
                skipped.append(pr.is_a())
        elif item.is_a("IfcPolygonalFaceSet"):      # Revit 2022+ tessellated bodies: one shared point list
            pl = private(f, item, "Coordinates"); new = []
            for c in pl.CoordList:
                c3 = list(c) + [0.0] * (3 - len(c)); w = (m @ np.array([c3[0], c3[1], c3[2], 1.0]))[:3]
                if _match(w, plane_axis, plane_value, v, within, tol):
                    d = np.linalg.inv(m[:3, :3]) @ v; moved += 1; new.append(tuple(float(c3[i] + d[i]) for i in range(3)))
                else:
                    new.append(tuple(float(c3[i]) for i in range(3)))
            pl.CoordList = new
        elif item.is_a("IfcFacetedBrep") or item.is_a("IfcFacetedBrepWithVoids"):
            shell = private(f, item, "Outer"); seen = set()
            loops = [b.Bound for face in shell.CfsFaces for b in face.Bounds if b.Bound.is_a("IfcPolyLoop")]; loop_ids = {lp.id() for lp in loops}
            for fi, face in enumerate(shell.CfsFaces):
                for bi, bound in enumerate(face.Bounds):
                    loop = bound.Bound
                    if not loop.is_a("IfcPolyLoop"):
                        continue
                    for pi, q in enumerate(loop.Polygon):
                        if q.id() in seen:
                            continue
                        seen.add(q.id())
                        c = list(q.Coordinates) + [0.0] * (3 - len(q.Coordinates)); w = (m @ np.array([c[0], c[1], c[2], 1.0]))[:3]
                        if _match(w, plane_axis, plane_value, v, within, tol):
                            d = np.linalg.inv(m[:3, :3]) @ v; moved += 1
                            newc = tuple(float(c[i] + d[i]) for i in range(len(q.Coordinates)))
                            inv = f.get_inverse(q)
                            if all(x.id() in loop_ids for x in inv):    # shared only between the faces of THIS solid: edit in place
                                q.Coordinates = newc
                            else:                                         # shared with other solids (ArchiCAD/Revit): copy-on-write in every loop of this solid
                                nq = _pt(f, newc)
                                for lp in loops:
                                    if q in lp.Polygon:
                                        lp.Polygon = [nq if qq == q else qq for qq in lp.Polygon]
        elif item.is_a("IfcBooleanResult") or item.is_a("IfcBooleanClippingResult"):
            do_item(item, "FirstOperand", None, m)
            if item.SecondOperand.is_a("IfcHalfSpaceSolid") or item.SecondOperand.is_a("IfcPolygonalBoundedHalfSpace"):
                skipped.append("half-space operand (clipping plane not moved)")
            else:
                do_item(item, "SecondOperand", None, m)
        elif item.is_a("IfcMappedItem"):
            skipped.append("IfcMappedItem (shared type geometry)")
        elif item.is_a("IfcPolyline") or item.is_a("IfcIndexedPolyCurve"):
            do_curve(owner, attr, m) if idx is None else None
        else:
            skipped.append(item.is_a())

    for i, r in enumerate(pds.Representations):
        if not _is_rep(pds, r, identifier):
            continue
        r = private(f, pds, "Representations", i)
        for j in range(len(r.Items)):
            do_item(r, "Items", j, base)
    out = {"guid": product.GlobalId, "vertices_moved": moved, "skipped": sorted(set(skipped))}
    if not moved:      # say why, instead of a bare 0 the model retries blindly (E011 BIM-Edit no-commit loops)
        bb = _mesh_bbox(product); k = "xyz".index(plane_axis) if plane_axis else None
        out["hint"] = ("no vertex within %g of %s = %g%s; the body spans %s" % (tol, plane_axis, plane_value, " inside `within`" if within else "",
                       ("%s in [%g, %g] (model units)" % (plane_axis, bb["min"][k], bb["max"][k])) if bb and k is not None else ("bbox %s" % bb)))
    return out


def shift_boundary_curve(f, rel_boundary, plane_axis: str | None = None, plane_value: float | None = None, vec_world=(0, 0, 0), within: dict | None = None, tol: float | None = None) -> dict:
    """Move the points of an IfcRelSpaceBoundary's connection geometry by vec_world - all points (plane_axis=None: the boundary
    surface of a space on a moved wall follows the wall as a whole), or only those on the world plane {plane_axis = plane_value}."""
    if isinstance(rel_boundary, str):
        rel_boundary = f.by_guid(rel_boundary)
    tol = _tol(f, tol); b = rel_boundary
    cg = b.ConnectionGeometry
    if cg is None or cg.SurfaceOnRelatingElement is None:
        return {"rel_guid": b.GlobalId, "points_moved": 0, "note": "no connection geometry"}
    m0 = world_matrix(b.RelatingSpace)
    cg = private(f, b, "ConnectionGeometry"); g = private(f, cg, "SurfaceOnRelatingElement")
    if g.is_a("IfcSurfaceOfLinearExtrusion") and g.SweptCurve.Curve.is_a("IfcPolyline"):
        m = m0 @ np.array(up.get_axis2placement(g.Position), dtype=float)
        sc = private(f, g, "SweptCurve"); curve = private(f, sc, "Curve")
    elif g.is_a("IfcCurveBoundedPlane") and g.OuterBoundary.is_a("IfcPolyline"):
        m = m0 @ np.array(up.get_axis2placement(g.BasisSurface.Position), dtype=float)
        curve = private(f, g, "OuterBoundary")
    else:
        return {"rel_guid": b.GlobalId, "points_moved": 0, "note": f"unsupported geometry {g.is_a()}"}
    d = np.linalg.inv(m[:3, :3]) @ np.array(vec_world, dtype=float)
    moved = 0; pts = []
    for p in curve.Points:
        c = list(p.Coordinates) + [0.0] * (3 - len(p.Coordinates))
        w = (m @ np.array([c[0], c[1], c[2], 1]))[:3]
        if _match(w, plane_axis, plane_value, vec_world, within, tol):
            pts.append(_pt(f, tuple(c[i] + d[i] for i in range(len(p.Coordinates))))); moved += 1
        else:
            pts.append(p)
    curve.Points = pts
    return {"rel_guid": b.GlobalId, "points_moved": moved}


def set_property(f, product, pset_name: str, prop_name: str, value) -> dict:
    """Set one property, creating the pset if needed. A pset shared with other elements is first replaced by a private copy on
    this product so the others keep their values."""
    if isinstance(product, str):
        product = f.by_guid(product)
    existing = ue.get_psets(product).get(pset_name)
    old = {}
    if existing and existing.get("id"):
        pset = f.by_id(existing["id"])
        old = {k: v for k, v in existing.items() if k != "id"}
        if len(ue.get_elements_by_pset(pset)) > 1:
            for rel in list(getattr(product, "IsDefinedBy", []) or []):
                if rel.is_a("IfcRelDefinesByProperties") and rel.RelatingPropertyDefinition == pset:
                    rel.RelatedObjects = tuple(o for o in rel.RelatedObjects if o != product)
            pset = ifcopenshell.api.pset.add_pset(f, product=product, name=pset_name)
            if old:
                ifcopenshell.api.pset.edit_pset(f, pset=pset, properties=old)
    else:
        pset = ifcopenshell.api.pset.add_pset(f, product=product, name=pset_name)
    ifcopenshell.api.pset.edit_pset(f, pset=pset, properties={prop_name: value})
    return {"guid": product.GlobalId, "pset": pset_name, "pset_id": pset.id(), prop_name: ue.get_psets(product).get(pset_name, {}).get(prop_name)}


def set_attribute(f, product, name: str, value) -> dict:
    if isinstance(product, str):
        product = f.by_guid(product)
    ifcopenshell.api.attribute.edit_attributes(f, product=product, attributes={name: value})
    return {"guid": product.GlobalId, name: getattr(product, name)}


def delete(f, product) -> dict:
    """Remove a product with its openings/fillings relationships and orphaned geometry (ifcopenshell.api.root.remove_product).
    Deleting a door leaves its opening in the wall unless the opening is deleted too."""
    if isinstance(product, str):
        product = f.by_guid(product)
    g = product.GlobalId
    # ifcopenshell 0.8.5 remove_product can delete a point / direction that the product's geometry shares with an entity it keeps
    # (BIM-Edit DOR-DEL-DIR-R-001: an opening's mapped-item operator shared the origin of a representation context, which was left
    # as IfcAxis2Placement3D($,$,$) - a new schema error the student then chased until the budget ran out). Record every outside
    # reference into the product's subtree and re-create what remove_product deleted under a surviving referrer.
    sub = {e.id() for root in (product.Representation, product.ObjectPlacement) if root is not None for e in f.traverse(root)} - {0}   # 0: typed values (IfcLabel ...)
    watch = []; infos = {}
    for sid in sub:
        ent = f.by_id(sid)
        if ent.is_a("IfcRepresentationContext") or ent.is_a("IfcObjectPlacement") and ent.id() != getattr(product.ObjectPlacement, "id", lambda: None)():
            continue    # contexts and parent placements are never removed by remove_product (and are referenced by thousands)
        if f.get_total_inverses(ent) < 2:
            continue    # referenced only by its parent in the subtree: deleting it cannot break anything outside
        for ref in f.get_inverse(ent):
            if ref.id() == product.id():
                continue
            for i in range(len(ref)):
                v = ref[i]
                if isinstance(v, ifcopenshell.entity_instance) and v.id() == sid:
                    if sid not in infos:
                        infos[sid] = ent.get_info(recursive=True, include_identifier=False)
                    watch.append((ref.id(), i, infos[sid]))
    ifcopenshell.api.root.remove_product(f, product=product)
    repaired = 0
    for rid, i, info in watch:
        try:
            ref = f.by_id(rid)
        except RuntimeError:
            continue
        if ref[i] is None:
            setattr(ref, ref.attribute_name(i), _recreate(f, info)); repaired += 1
    return {"deleted": g, **({"shared_geometry_restored": repaired} if repaired else {})}


def _recreate(f, info):
    """Entity (tree) from get_info(recursive=True, include_identifier=False)."""
    if isinstance(info, dict) and "type" in info:
        return f.create_entity(info["type"], **{k: _recreate(f, v) for k, v in info.items() if k != "type"})
    if isinstance(info, (list, tuple)):
        return tuple(_recreate(f, v) for v in info)
    return info


def connect_walls(f, wall_a, wall_b, a_end: str = "ATEND", b_end: str = "ATSTART") -> dict:
    """Create (or replace) the IfcRelConnectsPathElements between two walls; ends in {ATSTART, ATEND, ATPATH, NOTDEFINED}."""
    if isinstance(wall_a, str):
        wall_a = f.by_guid(wall_a)
    if isinstance(wall_b, str):
        wall_b = f.by_guid(wall_b)
    rel = ifcopenshell.api.geometry.connect_path(f, relating_element=wall_a, related_element=wall_b, relating_connection=a_end, related_connection=b_end)
    return {"rel_guid": rel.GlobalId}


# ------------------------------------------------------------------------------------------ creation primitives (tools v2.1, 2026-10-22)
def _ctx(f, identifier: str):
    """Representation (sub)context with the given identifier ('Body', 'Axis'); falls back to any Model/Plan context, creating one."""
    for c in f.by_type("IfcGeometricRepresentationSubContext"):
        if c.ContextIdentifier == identifier:
            return c
    for c in f.by_type("IfcGeometricRepresentationContext"):
        if c.ContextType == ("Plan" if identifier == "Axis" else "Model") and not c.is_a("IfcGeometricRepresentationSubContext"):
            parent = c; break
    else:
        import ifcopenshell.api.context
        parent = ifcopenshell.api.context.add_context(f, context_type="Plan" if identifier == "Axis" else "Model")
    import ifcopenshell.api.context
    return ifcopenshell.api.context.add_context(f, context_type=parent.ContextType, context_identifier=identifier, target_view="GRAPH_VIEW" if identifier == "Axis" else "MODEL_VIEW", parent=parent)


def _storey_of(f, storey):
    if isinstance(storey, str):
        try:
            return f.by_guid(storey)
        except Exception:  # noqa: BLE001
            m = [s for s in f.by_type("IfcBuildingStorey") if s.Name == storey]
            if not m:
                raise ValueError(f"no storey {storey!r}")
            return m[0]
    return storey


def _extrusion(f, profile, depth, origin=(0, 0, 0), xdir=(1, 0, 0)):
    import ifcopenshell.util.placement as up  # noqa: F401
    pl = f.createIfcAxis2Placement3D(f.createIfcCartesianPoint([float(c) for c in origin]), f.createIfcDirection([0.0, 0.0, 1.0]), f.createIfcDirection([float(c) for c in xdir]))
    return f.createIfcExtrudedAreaSolid(profile, pl, f.createIfcDirection([0.0, 0.0, 1.0]), float(depth))


def add_wall(f, storey, p0, p1, thickness: float, height: float, name: str | None = None) -> dict:
    """New standard wall (model units) from p0 to p1 on the storey (guid, name or entity): placement at p0 with x along the wall,
    Axis polyline, rectangular Body profile, contained in the storey. Returns its GlobalId. Joints and openings are separate steps."""
    import ifcopenshell.api.root, ifcopenshell.api.spatial
    storey = _storey_of(f, storey)
    p0, p1 = np.array(p0, dtype=float)[:2], np.array(p1, dtype=float)[:2]; u = (p1 - p0) / (np.linalg.norm(p1 - p0) or 1.0); L = float(np.linalg.norm(p1 - p0))
    cls = "IfcWallStandardCase" if f.schema == "IFC2X3" else "IfcWall"
    wall = ifcopenshell.api.root.create_entity(f, ifc_class=cls, name=name or "Wall", predefined_type=None if f.schema == "IFC2X3" else "STANDARD")
    sp = storey.ObjectPlacement
    wall.ObjectPlacement = f.createIfcLocalPlacement(sp, f.createIfcAxis2Placement3D(f.createIfcCartesianPoint([float(p0[0]), float(p0[1]), 0.0]), f.createIfcDirection([0.0, 0.0, 1.0]), f.createIfcDirection([float(u[0]), float(u[1]), 0.0])))
    axis = f.createIfcPolyline([f.createIfcCartesianPoint([0.0, 0.0]), f.createIfcCartesianPoint([L, 0.0])])
    prof = f.createIfcRectangleProfileDef("AREA", None, f.createIfcAxis2Placement2D(f.createIfcCartesianPoint([L / 2, 0.0]), None), L, float(thickness))
    wall.Representation = f.createIfcProductDefinitionShape(None, None, [f.createIfcShapeRepresentation(_ctx(f, "Axis"), "Axis", "Curve2D", [axis]),
                                                                        f.createIfcShapeRepresentation(_ctx(f, "Body"), "Body", "SweptSolid", [_extrusion(f, prof, height)])])
    ifcopenshell.api.spatial.assign_container(f, relating_structure=storey, products=[wall])
    return {"guid": wall.GlobalId, "axis_world": wall_axis_world(wall)}


def add_opening_fill(f, wall, along: float, width: float, height: float, kind: str = "door", sill: float = 0.0, name: str | None = None) -> dict:
    """Opening in the wall (model units: `along` = distance of the opening's left edge from the wall's start along its axis) voided
    by IfcRelVoidsElement, filled by a new IfcDoor / IfcWindow (IfcRelFillsElement), both placed relative to the wall."""
    import ifcopenshell.api.root, ifcopenshell.api.spatial
    if isinstance(wall, str):
        wall = f.by_guid(wall)
    t = wall_thickness_any(wall) or 0.2 / (unit_scale(f) or 1.0)
    storey = None
    try:
        storey = ue.get_container(wall)
    except Exception:  # noqa: BLE001
        pass
    op = ifcopenshell.api.root.create_entity(f, ifc_class="IfcOpeningElement", name="Opening")
    op.ObjectPlacement = f.createIfcLocalPlacement(wall.ObjectPlacement, f.createIfcAxis2Placement3D(f.createIfcCartesianPoint([float(along), float(-t / 2 - 0.05 * (1.0 / (unit_scale(f) or 1.0))), float(sill)]), f.createIfcDirection([0.0, 0.0, 1.0]), f.createIfcDirection([1.0, 0.0, 0.0])))
    extra = 0.1 / (unit_scale(f) or 1.0)
    prof = f.createIfcRectangleProfileDef("AREA", None, f.createIfcAxis2Placement2D(f.createIfcCartesianPoint([float(width / 2), float((t + extra) / 2)]), None), float(width), float(t + extra))
    op.Representation = f.createIfcProductDefinitionShape(None, None, [f.createIfcShapeRepresentation(_ctx(f, "Body"), "Body", "SweptSolid", [_extrusion(f, prof, height)])])
    import ifcopenshell.api.owner
    f.create_entity("IfcRelVoidsElement", GlobalId=ifcopenshell.guid.new(), OwnerHistory=ifcopenshell.api.owner.create_owner_history(f), RelatingBuildingElement=wall, RelatedOpeningElement=op)
    fill = ifcopenshell.api.root.create_entity(f, ifc_class="IfcDoor" if kind == "door" else "IfcWindow", name=name or kind.capitalize())
    fill.OverallHeight = float(height); fill.OverallWidth = float(width)
    fill.ObjectPlacement = f.createIfcLocalPlacement(op.ObjectPlacement, f.createIfcAxis2Placement3D(f.createIfcCartesianPoint([0.0, float(0.05 / (unit_scale(f) or 1.0)), 0.0]), f.createIfcDirection([0.0, 0.0, 1.0]), f.createIfcDirection([1.0, 0.0, 0.0])))
    leaf = 0.05 / (unit_scale(f) or 1.0)
    prof2 = f.createIfcRectangleProfileDef("AREA", None, f.createIfcAxis2Placement2D(f.createIfcCartesianPoint([float(width / 2), float(t / 2)]), None), float(width), float(leaf))
    fill.Representation = f.createIfcProductDefinitionShape(None, None, [f.createIfcShapeRepresentation(_ctx(f, "Body"), "Body", "SweptSolid", [_extrusion(f, prof2, height)])])
    f.create_entity("IfcRelFillsElement", GlobalId=ifcopenshell.guid.new(), OwnerHistory=ifcopenshell.api.owner.create_owner_history(f), RelatingOpeningElement=op, RelatedBuildingElement=fill)
    if storey is not None:
        ifcopenshell.api.spatial.assign_container(f, relating_structure=storey, products=[fill])
    return {"opening": op.GlobalId, "fill": fill.GlobalId, "class": fill.is_a()}


def add_slab(f, storey, polygon, thickness: float, elevation: float = 0.0, name: str | None = None) -> dict:
    """New floor slab (model units): polygon in storey XY, extruded by thickness upwards from `elevation` (storey-relative)."""
    import ifcopenshell.api.root, ifcopenshell.api.spatial
    storey = _storey_of(f, storey)
    slab = ifcopenshell.api.root.create_entity(f, ifc_class="IfcSlab", name=name or "Slab", predefined_type="FLOOR")
    slab.ObjectPlacement = f.createIfcLocalPlacement(storey.ObjectPlacement, f.createIfcAxis2Placement3D(f.createIfcCartesianPoint([0.0, 0.0, float(elevation)]), f.createIfcDirection([0.0, 0.0, 1.0]), f.createIfcDirection([1.0, 0.0, 0.0])))
    pts = [tuple(float(c) for c in p[:2]) for p in polygon]
    prof = f.createIfcArbitraryClosedProfileDef("AREA", None, f.createIfcPolyline([f.createIfcCartesianPoint(list(p)) for p in pts] + [f.createIfcCartesianPoint(list(pts[0]))]))
    slab.Representation = f.createIfcProductDefinitionShape(None, None, [f.createIfcShapeRepresentation(_ctx(f, "Body"), "Body", "SweptSolid", [_extrusion(f, prof, thickness)])])
    ifcopenshell.api.spatial.assign_container(f, relating_structure=storey, products=[slab])
    return {"guid": slab.GlobalId}


def add_column(f, storey, xy, size: float, height: float, name: str | None = None) -> dict:
    """New square column (model units) centred at xy on the storey, extruded by height."""
    import ifcopenshell.api.root, ifcopenshell.api.spatial
    storey = _storey_of(f, storey)
    col = ifcopenshell.api.root.create_entity(f, ifc_class="IfcColumn", name=name or "Column", predefined_type="COLUMN")
    col.ObjectPlacement = f.createIfcLocalPlacement(storey.ObjectPlacement, f.createIfcAxis2Placement3D(f.createIfcCartesianPoint([float(xy[0]), float(xy[1]), 0.0]), f.createIfcDirection([0.0, 0.0, 1.0]), f.createIfcDirection([1.0, 0.0, 0.0])))
    prof = f.createIfcRectangleProfileDef("AREA", None, f.createIfcAxis2Placement2D(f.createIfcCartesianPoint([0.0, 0.0]), None), float(size), float(size))
    col.Representation = f.createIfcProductDefinitionShape(None, None, [f.createIfcShapeRepresentation(_ctx(f, "Body"), "Body", "SweptSolid", [_extrusion(f, prof, height)])])
    ifcopenshell.api.spatial.assign_container(f, relating_structure=storey, products=[col])
    return {"guid": col.GlobalId}


def set_wall_thickness(f, wall, thickness: float) -> dict:
    """Set the thickness (model units) of a wall's extruded Body profile, keeping its axis line: rectangle profiles get YDim =
    thickness; polyline profiles have their points moved to +-thickness/2 about the profile's centre line."""
    if isinstance(wall, str):
        wall = f.by_guid(wall)
    n = 0
    if not any(_extrusion_of(it) is not None for it in _rep_items(wall).get("Body", [])):   # Brep: move both side faces (axis-aligned walls)
        ax = wall_axis_world(wall); bb = _mesh_bbox(wall)
        if ax is None or bb is None:
            raise ValueError("Brep wall without Axis/geometry: edit it with shift_vertices_on_plane by hand")
        p0, p1 = np.array(ax[0][:2]), np.array(ax[1][:2]); u = (p1 - p0) / (np.linalg.norm(p1 - p0) or 1.0)
        if abs(u[0]) < 0.98 and abs(u[1]) < 0.98:
            raise ValueError("Brep wall is not axis-aligned; edit it with shift_vertices_on_plane by hand")
        k = 0 if abs(u[0]) >= 0.98 else 1; side = "y" if k == 0 else "x"; along = "x" if k == 0 else "y"
        lo, hi = bb["min"][1 - k], bb["max"][1 - k]; t_old = hi - lo; dt = (thickness - t_old) / 2
        rng_along = (min(bb["min"][k], bb["max"][k]) - 1.0, max(bb["min"][k], bb["max"][k]) + 1.0)
        v_hi = [0.0, 0.0, 0.0]; v_hi[1 - k] = dt; v_lo = [0.0, 0.0, 0.0]; v_lo[1 - k] = -dt
        shift_vertices_on_plane(f, wall, side, float(hi), v_hi, within={along: rng_along}); shift_vertices_on_plane(f, wall, side, float(lo), v_lo, within={along: rng_along})
        return {"guid": wall.GlobalId, "items_changed": 1, "thickness": float(thickness), "mode": "brep-vertex-shift"}
    pds = private(f, wall, "Representation")
    for i, r in enumerate(pds.Representations):
        if not _is_rep(pds, r, "Body"):
            continue
        r = private(f, pds, "Representations", i)
        for j, it0 in enumerate(list(r.Items)):
            if _extrusion_of(it0) is None:
                continue
            it = _private_extrusion(f, r, j); pr = private(f, it, "SweptArea")
            if pr.is_a("IfcRectangleProfileDef"):
                pr.YDim = float(thickness); n += 1
            elif pr.is_a("IfcArbitraryClosedProfileDef") and pr.OuterCurve.is_a("IfcPolyline"):
                curve = private(f, pr, "OuterCurve"); ys = [pt.Coordinates[1] for pt in curve.Points]; cy = (max(ys) + min(ys)) / 2
                for k, pt in enumerate(list(curve.Points)):
                    pt = private(f, curve, "Points", k); c = list(pt.Coordinates)
                    c[1] = cy + (thickness / 2 if c[1] >= cy else -thickness / 2); pt.Coordinates = tuple(float(v) for v in c)
                n += 1
    return {"guid": wall.GlobalId, "items_changed": n, "thickness": float(thickness)}


def rotate(f, product, angle_deg: float, center_world=(0.0, 0.0)) -> dict:
    """Rotate a product about the vertical axis through a WORLD point (model units) by editing its own IfcLocalPlacement;
    children placed relative to it (openings, doors) rotate with it."""
    if isinstance(product, str):
        product = f.by_guid(product)
    pl = product.ObjectPlacement
    if pl is None or not pl.is_a("IfcLocalPlacement"):
        raise ValueError("product has no IfcLocalPlacement")
    others = [x for x in f.get_inverse(pl) if x.is_a("IfcProduct") and x != product]
    if others:
        pl = ue.copy(f, pl); product.ObjectPlacement = pl
    P = np.array(up.get_local_placement(pl.PlacementRelTo), dtype=float) if pl.PlacementRelTo else np.eye(4)
    M = np.array(up.get_local_placement(pl), dtype=float)
    a = np.radians(angle_deg); c, s_ = np.cos(a), np.sin(a)
    R = np.eye(4); R[0, 0], R[0, 1], R[1, 0], R[1, 1] = c, -s_, s_, c
    T1 = np.eye(4); T1[:2, 3] = -np.array(center_world[:2], dtype=float); T2 = np.eye(4); T2[:2, 3] = np.array(center_world[:2], dtype=float)
    L = np.linalg.inv(P) @ (T2 @ R @ T1 @ M)
    rp = private(f, pl, "RelativePlacement")
    rp.Location = f.createIfcCartesianPoint([float(v) for v in L[:3, 3]])
    rp.Axis = f.createIfcDirection([float(v) for v in L[:3, 2]]); rp.RefDirection = f.createIfcDirection([float(v) for v in L[:3, 0]])
    return {"guid": product.GlobalId, "world_origin": _r(world_matrix(product)[:3, 3]), "x_dir": _r(world_matrix(product)[:3, 0])}


def add_space(f, storey, polygon, height: float, name: str | None = None, long_name: str | None = None) -> dict:
    """New IfcSpace (model units) with a polygon footprint in storey XY extruded by height, aggregated into the storey."""
    import ifcopenshell.api.root, ifcopenshell.api.aggregate
    storey = _storey_of(f, storey)
    sp = ifcopenshell.api.root.create_entity(f, ifc_class="IfcSpace", name=name or "Space")
    if long_name:
        sp.LongName = long_name
    if f.schema == "IFC2X3":
        sp.InteriorOrExteriorSpace = "INTERNAL"
    sp.ObjectPlacement = f.createIfcLocalPlacement(storey.ObjectPlacement, f.createIfcAxis2Placement3D(f.createIfcCartesianPoint([0.0, 0.0, 0.0]), f.createIfcDirection([0.0, 0.0, 1.0]), f.createIfcDirection([1.0, 0.0, 0.0])))
    pts = [tuple(float(c) for c in p[:2]) for p in polygon]
    prof = f.createIfcArbitraryClosedProfileDef("AREA", None, f.createIfcPolyline([f.createIfcCartesianPoint(list(p)) for p in pts] + [f.createIfcCartesianPoint(list(pts[0]))]))
    sp.Representation = f.createIfcProductDefinitionShape(None, None, [f.createIfcShapeRepresentation(_ctx(f, "Body"), "Body", "SweptSolid", [_extrusion(f, prof, height)])])
    ifcopenshell.api.aggregate.assign_object(f, relating_object=storey, products=[sp])
    return {"guid": sp.GlobalId}


def find(f, ifc_class: str | None = None, storey: str | None = None, name_contains: str | None = None, near=None, radius: float | None = None,
         predefined_type: str | None = None, limit: int = 50) -> dict:
    """Find elements by class (e.g. 'IfcWall', 'IfcSpace'), storey (GlobalId or Name), name substring, predefined type and/or
    distance in plan from a world point `near` = (x, y) in MODEL units within `radius`. Returns up to `limit` matches with their
    centre (model units, from the geometry) and bounding box, plus the total count."""
    import ifcopenshell.util.element as _ue
    classes = [ifc_class] if ifc_class else ["IfcProduct"]
    cands = []
    for cls in classes:
        try:
            cands += list(f.by_type(cls))
        except Exception:  # noqa: BLE001
            raise ValueError(f"unknown IFC class {cls!r}")
    st = None
    if storey:
        try:
            st = _storey_of(f, storey)
        except Exception:  # noqa: BLE001
            raise ValueError(f"no storey {storey!r}")
    out = []; total = 0
    for e in cands:
        if not getattr(e, "GlobalId", None) or e.is_a("IfcSpatialStructureElement") and not ifc_class:
            continue
        if st is not None:
            c = _ue.get_container(e) if not e.is_a("IfcSpace") else _ue.get_aggregate(e)
            if c != st:
                continue
        if name_contains and name_contains.lower() not in ((e.Name or "") + " " + (getattr(e, "LongName", "") or "")).lower():
            continue
        if predefined_type and (getattr(e, "PredefinedType", None) or "") != predefined_type:
            continue
        bb = _mesh_bbox(e) if (near is not None or len(out) < limit) else None
        centre = None if bb is None else [round((bb["min"][0] + bb["max"][0]) / 2, 4), round((bb["min"][1] + bb["max"][1]) / 2, 4)]
        if near is not None:
            if centre is None:
                continue
            d = float(np.hypot(centre[0] - float(near[0]), centre[1] - float(near[1])))
            if radius is not None and d > float(radius):
                continue
        total += 1
        if len(out) < limit:
            try:
                cont = _ue.get_container(e) if not e.is_a("IfcSpace") else _ue.get_aggregate(e); cont = cont.Name if cont is not None else None
            except Exception:  # noqa: BLE001
                cont = None
            row = {"guid": e.GlobalId, "class": e.is_a(), "name": e.Name, "storey": cont, "centre_xy": centre, "bbox": bb}
            if getattr(e, "LongName", None):
                row["long_name"] = e.LongName
            if near is not None:
                row["distance"] = round(d, 4)
            out.append(row)
    if near is not None:
        out.sort(key=lambda r: r.get("distance", 1e9))
    return {"count": total, "matches": out, "units": "model units (h.unit_scale(ifc) metres per unit)"}


def check_model(f, original_path: str) -> dict:
    """Self-check of the CURRENT state against the original file: new schema errors, entities added / removed / modified (class, name),
    relationships added / removed, and whether the geometry of changed elements still builds. Facts only - it does not know the task."""
    from ifcgym.checkers import schema_errors, before_model, geometry_ok
    from ifcgym.ifcgraph import edit_graph_from
    graph, err_before = before_model(original_path)
    err_now = set(schema_errors(f)); new_err = sorted(err_now - set(err_before))
    eg = edit_graph_from(graph, f)
    def rows(d, kind):
        out = []
        for g in sorted(d)[:30]:
            n = d[g] if kind != "modified" else d[g][1]
            out.append({"guid": g, "class": n.cls, "name": n.name})
        return out
    touched = [g for g in sorted(eg.node_ids) if g not in eg.removed]
    gok, gbad = geometry_ok(f, touched[:60]) if touched else (0, 0)
    return {"new_schema_errors": new_err[:10], "n_new_schema_errors": len(new_err),
            "added": rows(eg.added, "added"), "removed": rows(eg.removed, "removed"), "modified": rows(eg.modified, "modified"),
            "n_added": len(eg.added), "n_removed": len(eg.removed), "n_modified": len(eg.modified),
            "relationships_added": len(eg.rel_added), "relationships_removed": len(eg.rel_removed),
            "geometry_rebuild_failures": gbad, "geometry_checked": gok + gbad,
            "note": "modified = any attribute, property, placement or geometry change; compare with what the task allows to change"}


def wall_length(wall) -> float:
    """Axis length in model units."""
    ax = wall_axis_world(wall)
    return 0.0 if ax is None else float(np.linalg.norm(np.array(ax[1][:2]) - np.array(ax[0][:2])))


def space_area(f, space) -> float:
    """Net floor area (m2) of a space from its footprint polygon (world), falling back to the mesh footprint."""
    if isinstance(space, str):
        space = f.by_guid(space)
    poly = profile_world_polygon(space)
    sc = unit_scale(f) or 1.0
    if poly:
        P = np.array(poly)[:, :2] * sc; x, y = P[:, 0], P[:, 1]
        return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    from ifcgym.checkers import mesh
    m = mesh(space)
    if m is None:
        return float("nan")
    from scipy.spatial import ConvexHull
    return float(ConvexHull(m[:, :2]).volume)


HELP = """h.inspect(ifc, guid, psets=False, related_to=None)    -> dict: placement, representation, openings, connected walls, space boundaries (related_to=<wall guid> filters a space's boundaries); psets=True adds property sets
h.unit_scale(ifc)                                     -> metres per model unit (coordinates below are in MODEL units)
h.world_matrix(product) / h.wall_axis_world(wall) / h.profile_world_polygon(product)
h.translate(ifc, product_or_guid, [dx,dy,dz])         -> moves ONE product's placement; children placed relative to it (openings, doors) follow
h.move_wall_end(ifc, wall, 'start'|'end', [dx,dy,dz]) -> move one end of a wall by a world vector along its axis (joined wall follows a moved wall)
h.stretch_wall_end(ifc, wall, 'start'|'end', delta)   -> lengthen (delta>0) / shorten a rectangular wall at one end along its own axis
h.shift_vertices_on_plane(ifc, product, 'x'|'y'|'z', value, [dx,dy,dz], within={'x': (lo,hi)}) -> move ALL vertices of the product's geometry on a world plane (extrusions, breps, polylines)
h.shift_profile_vertices(ifc, product, 'x'|'y'|'z', value, [dx,dy,dz], within={'x': (lo,hi)})  -> same, extruded footprint profiles only
h.shift_boundary_curve(ifc, rel_boundary_or_guid, None, None, [dx,dy,dz])  -> move a space boundary surface as a whole (or only points on a plane)
h.set_extrusion_depth(ifc, product, depth)            -> wall/space height
h.set_property(ifc, product, pset, name, value)       -> private pset (shared psets are copied first), h.set_attribute(ifc, product, name, value)
h.delete(ifc, product)                                -> smart delete (relationships cleaned), h.connect_walls(ifc, a, b, a_end, b_end)
h.private(ifc, parent, attr[, idx])                   -> make a sub-entity private before editing it by hand
h.add_wall(ifc, storey, p0, p1, thickness, height)    -> new standard wall on a storey (guid/name/entity); then h.connect_walls / h.add_opening_fill
h.add_opening_fill(ifc, wall, along, width, height, 'door'|'window', sill=0) -> opening + door/window hosted by the wall
h.add_slab(ifc, storey, polygon, thickness, elevation=0), h.add_column(ifc, storey, (x, y), size, height)
h.space_area(ifc, space)                              -> net floor area in m2 (footprint polygon); h.wall_length(wall) -> axis length (model units)
h.set_wall_thickness(ifc, wall, t)                    -> wall thickness about its axis; h.rotate(ifc, product, angle_deg, (cx, cy)) -> rotate about a world point
(stretch_wall_end / set_wall_thickness / wall_axis_world also handle extrusions clipped by half-spaces (IfcBooleanClippingResult, Revit gable walls); true Breps need shift_vertices_on_plane)
h.add_space(ifc, storey, polygon, height)             -> new IfcSpace aggregated into the storey (footprint polygon in storey XY)
h.find(ifc, ifc_class, storey, name_contains, near=(x, y), radius, predefined_type, limit) -> elements matching (also the find_elements tool)
h.check_model(ifc, original_path)                     -> diff against the original: new schema errors, added/removed/modified entities (also the check_model tool)"""
