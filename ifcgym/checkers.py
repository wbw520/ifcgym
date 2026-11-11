"""Gated checkers of IfcEdit (DESIGN.md §1): validity, intent, dependency, preservation, under-edit. All checks compare the input
model with the agent's final model; the task spec (tasks/generate.py) says what was asked, which entities may change (allowed set)
and which dependents must follow. Nothing here needs a target file: several legal solutions pass.

Task spec (JSON):
  {"task_id", "project", "template", "tier", "prompt", "target": [guid...], "allowed": [guid...],
   "intent": {"type": "translate"|"set_attribute"|"set_property"|"delete"|"resize_height", ...params},
   "dependents": [{"guid", "rule": "translate_with"|"removed"|"stays_hosted", ...params}]}
"""
from __future__ import annotations

import io
import logging
import math
from dataclasses import dataclass, field

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element as eu
import ifcopenshell.util.placement as pl
import numpy as np

from ifcgym.ifcgraph import EditGraph, edit_graph


@dataclass
class CheckReport:
    validity: bool
    intent: bool
    dependency: bool
    preservation: bool
    under_edit: bool
    side_effects: int
    new_errors: int
    details: dict = field(default_factory=dict)

    @property
    def strict(self) -> bool:
        return self.validity and self.intent and self.dependency and self.preservation

    def text(self) -> str:
        """Feedback text shown to agents in `all` mode (no target values are revealed beyond the task statement)."""
        lines = [f"validity: {'ok' if self.validity else 'FAIL'} ({self.details.get('validity', '')})",
                 f"intent: {'ok' if self.intent else 'FAIL'} ({self.details.get('intent', '')})",
                 f"dependency: {'ok' if self.dependency else 'FAIL'} ({self.details.get('dependency', '')})",
                 f"preservation: {'ok' if self.preservation else 'FAIL'} (side-effect entities: {self.side_effects}; {self.details.get('preservation', '')})"]
        return "\n".join(lines)


# ------------------------------------------------------------------------------------------ validity
def schema_errors(f: ifcopenshell.file) -> list[str]:
    import ifcopenshell.validate as v
    log = logging.getLogger("ifcedit.validate." + str(id(f))); log.handlers.clear(); log.propagate = False
    buf = io.StringIO(); h = logging.StreamHandler(buf); log.addHandler(h); log.setLevel(logging.ERROR)
    try:
        v.validate(f, log, express_rules=False)
    except Exception as e:  # noqa: BLE001
        return [f"validator crashed: {e}"]
    return [l for l in buf.getvalue().splitlines() if l.strip()]


def decomposition_cycle(f: ifcopenshell.file) -> str | None:
    """GlobalId of an object that is its own IfcRelAggregates / IfcRelNests ancestor, else None. A Gemma 4 31B episode (E021, task
    fantasy_office_building_3-M-compound-3-0584-s2, 2026-11-04) aggregated a storey into itself; ifcopenshell.geom.create_shape then
    recursed forever in C++ and the check never returned, so a cyclic decomposition is reported as a validity failure before any geometry."""
    parents: dict[int, list] = {}
    for cls in ("IfcRelAggregates", "IfcRelNests"):
        for rel in f.by_type(cls):
            p = getattr(rel, "RelatingObject", None)
            if p is None:
                continue
            for o in rel.RelatedObjects or ():
                parents.setdefault(o.id(), []).append(p)
    for start in list(parents):
        seen = {start}; stack = list(parents[start])
        while stack:
            e = stack.pop()
            if e.id() == start:
                return getattr(e, "GlobalId", None) or str(e.id())
            if e.id() in seen:
                continue
            seen.add(e.id()); stack.extend(parents.get(e.id(), []))
    return None


def geometry_ok(f: ifcopenshell.file, guids: list[str]) -> tuple[int, int]:
    s = ifcopenshell.geom.settings(); ok = bad = 0
    for g in guids:
        try:
            inst = f.by_guid(g)
        except Exception:  # noqa: BLE001
            continue
        if not getattr(inst, "Representation", None):
            continue
        try:
            shp = ifcopenshell.geom.create_shape(s, inst)
            ok += len(shp.geometry.verts) > 0
        except Exception:  # noqa: BLE001
            bad += 1
    return ok, bad


# ------------------------------------------------------------------------------------------ geometry helpers
def abs_origin(inst) -> np.ndarray | None:
    try:
        return np.array(pl.get_local_placement(inst.ObjectPlacement))[:3, 3]
    except Exception:  # noqa: BLE001
        return None


def _settings(no_openings: bool = True):
    s = ifcopenshell.geom.settings()
    if no_openings:
        try:
            s.set("disable-opening-subtractions", True)
        except Exception:  # noqa: BLE001
            try:
                s.set(s.DISABLE_OPENING_SUBTRACTIONS, True)
            except Exception:  # noqa: BLE001
                pass
    return s


def mesh(inst, no_openings: bool = True) -> np.ndarray | None:
    from ifcgym.ifcgraph import world_verts
    try:
        return world_verts(ifcopenshell.geom.create_shape(_settings(no_openings), inst))
    except Exception:  # noqa: BLE001
        return None


def centroid(f: ifcopenshell.file, inst) -> np.ndarray | None:
    v = mesh(inst)
    return None if v is None else v.mean(0)


def _min_dist(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.spatial import cKDTree
    d, _ = cKDTree(b).query(a[:: max(1, len(a) // 4000)], k=1)
    return float(d.min())


def _tri_mesh(inst, no_openings: bool = True):
    """(vertices, faces) of a product's mesh in metres, world coordinates."""
    from ifcgym.ifcgraph import shape_matrix
    try:
        shp = ifcopenshell.geom.create_shape(_settings(no_openings), inst)
    except Exception:  # noqa: BLE001
        return None
    v = np.array(shp.geometry.verts, dtype=float).reshape(-1, 3); fc = np.array(shp.geometry.faces, dtype=int).reshape(-1, 3)
    M = shape_matrix(shp); v = (M[:3, :3] @ v.T).T + M[:3, 3]
    return v, fc


def surface_samples(v: np.ndarray, fc: np.ndarray, spacing: float = 0.05, cap: int = 20000) -> np.ndarray:
    """Points on the triangle surfaces, roughly one per spacing^2 of area (vertices and edge midpoints always included)."""
    if len(fc) == 0:
        return v
    A, B, C = v[fc[:, 0]], v[fc[:, 1]], v[fc[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
    n = np.minimum(np.ceil(area / (spacing * spacing)).astype(int), 2000)
    if n.sum() > cap:
        n = np.maximum(1, (n * cap / n.sum()).astype(int))
    rng = np.random.default_rng(0); out = [v, (A + B) / 2, (B + C) / 2, (A + C) / 2]
    idx = np.repeat(np.arange(len(fc)), n)
    r1, r2 = rng.random((len(idx), 1)), rng.random((len(idx), 1)); s1 = np.sqrt(r1)
    out.append((1 - s1) * A[idx] + s1 * (1 - r2) * B[idx] + s1 * r2 * C[idx])
    return np.vstack(out)


def _point_tri_dist(P: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Exact distance from each point P[i] to the nearest of the triangles (A, B, C) (Ericson, vectorised over triangles)."""
    best = np.full(len(P), np.inf)
    for k in range(0, len(A), 256):
        a, b, c = A[k:k + 256], B[k:k + 256], C[k:k + 256]
        ab, ac = b - a, c - a
        p = P[:, None, :]; ap = p - a[None]
        d1 = np.einsum("ijk,jk->ij", ap, ab); d2 = np.einsum("ijk,jk->ij", ap, ac)
        bp = p - b[None]; d3 = np.einsum("ijk,jk->ij", bp, ab); d4 = np.einsum("ijk,jk->ij", bp, ac)
        cp = p - c[None]; d5 = np.einsum("ijk,jk->ij", cp, ab); d6 = np.einsum("ijk,jk->ij", cp, ac)
        va = d3 * d6 - d5 * d4; vb = d5 * d2 - d1 * d6; vc = d1 * d4 - d3 * d2
        denom = 1.0 / np.maximum(va + vb + vc, 1e-30); v_ = vb * denom; w_ = vc * denom
        q = a[None] + v_[..., None] * ab[None] + w_[..., None] * ac[None]          # interior projection
        # region tests
        r_a = (d1 <= 0) & (d2 <= 0); r_b = (d3 >= 0) & (d4 <= d3); r_c = (d6 >= 0) & (d5 <= d6)
        r_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0); r_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0); r_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
        t_ab = np.clip(d1 / np.where(d1 - d3 == 0, 1e-30, d1 - d3), 0, 1); t_ac = np.clip(d2 / np.where(d2 - d6 == 0, 1e-30, d2 - d6), 0, 1)
        t_bc = np.clip((d4 - d3) / np.where((d4 - d3) + (d5 - d6) == 0, 1e-30, (d4 - d3) + (d5 - d6)), 0, 1)
        q = np.where(r_bc[..., None], b[None] + t_bc[..., None] * (c - b)[None], q)
        q = np.where(r_ac[..., None], a[None] + t_ac[..., None] * ac[None], q)
        q = np.where(r_ab[..., None], a[None] + t_ab[..., None] * ab[None], q)
        q = np.where(r_c[..., None], c[None], q); q = np.where(r_b[..., None], b[None], q); q = np.where(r_a[..., None], a[None], q)
        d = np.linalg.norm(p - q, axis=2).min(axis=1)
        best = np.minimum(best, d)
    return best


def _inside(P: np.ndarray, v: np.ndarray, fc: np.ndarray) -> np.ndarray:
    """Point-in-solid by ray parity along +x (Moller-Trumbore, vectorised); robust enough for closed box-like solids."""
    A, B, C = v[fc[:, 0]], v[fc[:, 1]], v[fc[:, 2]]
    e1, e2 = B - A, C - A; d = np.array([1.0, 0.0, 0.0])
    pvec = np.cross(d, e2); det = np.einsum("jk,jk->j", e1, pvec); ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    cnt = np.zeros(len(P), dtype=int)
    for k in range(0, len(A), 512):
        sl = slice(k, k + 512)
        tvec = P[:, None, :] - A[None, sl]; u = np.einsum("ijk,jk->ij", tvec, pvec[sl]) * inv[sl]
        qvec = np.cross(tvec, e1[None, sl]); vv = np.einsum("ijk,k->ij", qvec, d) * inv[sl]
        t = np.einsum("ijk,jk->ij", qvec, e2[sl]) * inv[sl]
        hit = ok[sl][None] & (u >= 0) & (vv >= 0) & (u + vv <= 1) & (t > 1e-9)
        cnt += hit.sum(axis=1)
    return cnt % 2 == 1


def _gap(a_inst, b_inst) -> float | None:
    """Surface-to-surface gap between two products (metres): 0 if their solids overlap or touch, else the exact minimum
    distance from sampled surface points of each to the other's triangles (symmetric)."""
    ma, mb = _tri_mesh(a_inst), _tri_mesh(b_inst)
    if ma is None or mb is None:
        return None
    (va, fa), (vb, fb) = ma, mb
    pa, pb = surface_samples(va, fa), surface_samples(vb, fb)
    if len(fa) and len(fb):
        if _inside(pa[:: max(1, len(pa) // 3000)], vb, fb).any() or _inside(pb[:: max(1, len(pb) // 3000)], va, fa).any():
            return 0.0
        return float(min(_point_tri_dist(pa[:: max(1, len(pa) // 3000)], vb[fb[:, 0]], vb[fb[:, 1]], vb[fb[:, 2]]).min(),
                         _point_tri_dist(pb[:: max(1, len(pb) // 3000)], va[fa[:, 0]], va[fa[:, 1]], va[fa[:, 2]]).min()))
    return float(min(_min_dist(pa, pb), _min_dist(pb, pa)))


def unit_scale(f: ifcopenshell.file) -> float:
    try:
        import ifcopenshell.util.unit as uu
        return float(uu.calculate_unit_scale(f))
    except Exception:  # noqa: BLE001
        return 1.0


# ------------------------------------------------------------------------------------------ predicates
def _translated(before: ifcopenshell.file, after: ifcopenshell.file, guid: str, vec, tol: float, scale: float = 1.0) -> tuple[bool, str]:
    """vec and tol in METRES; mesh centroids are metres (ifcopenshell.geom converts), placements are model units (x scale)."""
    try:
        a, b = before.by_guid(guid), after.by_guid(guid)
    except Exception:  # noqa: BLE001
        return False, f"{guid} missing after edit"
    ca, cb = centroid(before, a), centroid(after, b)
    if ca is None or cb is None:
        oa, ob = abs_origin(a), abs_origin(b)
        if oa is None or ob is None:
            return False, "no geometry/placement"
        d = (ob - oa) * scale
    else:
        d = cb - ca
    err = float(np.linalg.norm(d - np.array(vec, dtype=float)))
    return err <= tol, f"moved {np.round(d, 4).tolist()} vs expected {list(vec)} (err {err:.4f} m)"


def _attribute_equals(after: ifcopenshell.file, guid: str, name: str, value) -> tuple[bool, str]:
    try:
        inst = after.by_guid(guid)
    except Exception:  # noqa: BLE001
        return False, f"{guid} missing"
    got = getattr(inst, name, None)
    return got == value, f"{name}={got!r}"


def _property_equals(after: ifcopenshell.file, guid: str, pset: str, prop: str, value, tol=0.0) -> tuple[bool, str]:
    try:
        inst = after.by_guid(guid)
    except Exception:  # noqa: BLE001
        return False, f"{guid} missing"
    ps = eu.get_psets(inst)
    got = (ps.get(pset) or {}).get(prop)
    if isinstance(value, (int, float)) and isinstance(got, (int, float)):
        return abs(got - value) <= tol * max(abs(value), 1e-9) + 1e-9, f"{pset}.{prop}={got}"
    return got == value, f"{pset}.{prop}={got!r}"


def _deleted(after: ifcopenshell.file, guid: str) -> tuple[bool, str]:
    try:
        after.by_guid(guid); return False, f"{guid} still present"
    except Exception:  # noqa: BLE001
        return True, "removed"


def _height_equals(after: ifcopenshell.file, guid: str, height: float, tol: float, top: float | None = None, bottom: float | None = None) -> tuple[bool, str]:
    try:
        inst = after.by_guid(guid)
    except Exception:  # noqa: BLE001
        return False, f"{guid} missing"
    s = ifcopenshell.geom.settings()
    try:
        v = mesh(inst)
        if v is None:
            raise RuntimeError("no mesh")
        h = float(v[:, 2].max() - v[:, 2].min())
    except Exception as e:  # noqa: BLE001
        return False, f"no geometry: {e}"
    if top is not None:   # v4 slab thickness: the top face must stay where it was
        return abs(h - height) <= tol and abs(float(v[:, 2].max()) - top) <= tol, f"height {h:.3f} vs {height}, top {float(v[:, 2].max()):.3f} vs {top}"
    if bottom is not None:   # E010: the bottom must stay where it was (relative height change of a window / space)
        return abs(h - height) <= tol and abs(float(v[:, 2].min()) - bottom) <= tol, f"height {h:.3f} vs {height}, bottom {float(v[:, 2].min()):.3f} vs {bottom}"
    return abs(h - height) <= tol, f"height {h:.3f} vs {height}"


def _footprint(inst):
    """Shapely polygon of the element's bottom face (metres) from its mesh; None without geometry."""
    try:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        r = _tri_mesh(inst)
        if r is None:
            return None
        v, fc = r; z0 = v[:, 2].min(); tris = [Polygon(v[t][:, :2]) for t in fc if np.all(np.abs(v[t][:, 2] - z0) < 1e-3)]
        return unary_union(tris).buffer(1e-4) if tris else None
    except Exception:  # noqa: BLE001
        return None


def _poly_match(inst, polygon, tol_frac=0.05, tol_abs=0.05) -> tuple[bool, str]:
    from shapely.geometry import Polygon
    P = Polygon(polygon); fp = _footprint(inst)
    if fp is None or P.area <= 0:
        return False, "no footprint"
    d = P.symmetric_difference(fp).area
    return d <= tol_frac * P.area + tol_abs, f"footprint symmetric-difference {d:.3f} m2 of {P.area:.2f}"


def _added_of(before, after, classes: tuple) -> list:
    old = {e.GlobalId for c in classes for e in before.by_type(c)}
    return [e for c in classes for e in after.by_type(c) if e.GlobalId not in old]


def _bbox_m(inst, scale):
    v = mesh(inst)
    return None if v is None else (v.min(0), v.max(0))


def _create_fill(before, after, intent, scale):
    cls = "IfcDoor" if intent["kind"] == "door" else "IfcWindow"
    new = _added_of(before, after, (cls,)); tol = intent.get("tol", 0.05)
    hosted = []
    for e in new:
        hosts = [r.RelatingOpeningElement for r in (getattr(e, "FillsVoids", []) or [])]
        voided = [v.RelatingBuildingElement.GlobalId for o in hosts for v in (o.VoidsElements or [])]
        if intent["host"] in voided:
            hosted.append((e, hosts[0]))
    if len(hosted) != 1:
        return False, f"expected exactly one new {cls} hosted by {intent['host']}, found {len(hosted)} (added {len(new)})"
    e, op = hosted[0]
    bb = _bbox_m(e, scale)
    if bb is None:
        return False, f"new {cls} has no geometry"
    dims = bb[1] - bb[0]; w = float(max(dims[0], dims[1])); h = float(dims[2]); msgs = []
    ok = abs(w - intent["width"]) <= tol and abs(h - intent["height"]) <= tol
    msgs.append(f"width {w:.3f} vs {intent['width']}, height {h:.3f} vs {intent['height']}")
    try:   # inside the host wall's extent and clear of its other openings (along the wall axis), sill height for windows
        from ifcgym.ifctools import wall_axis_world
        wall = after.by_guid(intent["host"]); ax = wall_axis_world(wall)
        if ax is not None:
            a, b = np.array(ax[0][:2]) * scale, np.array(ax[1][:2]) * scale; u = (b - a) / (np.linalg.norm(b - a) or 1.0); L = float(np.linalg.norm(b - a))
            lo, hi = sorted([float((bb[0][:2] - a) @ u), float((bb[1][:2] - a) @ u)])
            lo2, hi2 = sorted([float((np.array([bb[0][0], bb[1][1]]) - a) @ u), float((np.array([bb[1][0], bb[0][1]]) - a) @ u)])
            lo, hi = min(lo, lo2), max(hi, hi2)
            if lo < -tol or hi > L + tol:
                ok = False; msgs.append(f"outside the wall extent ({lo:.2f}-{hi:.2f} of {L:.2f})")
            for r in wall.HasOpenings or []:
                o = r.RelatedOpeningElement
                if o == op:
                    continue
                ob = _bbox_m(o, scale)
                if ob is None:
                    continue
                olo, ohi = sorted([float((ob[0][:2] - a) @ u), float((ob[1][:2] - a) @ u)])
                olo2, ohi2 = sorted([float((np.array([ob[0][0], ob[1][1]]) - a) @ u), float((np.array([ob[1][0], ob[0][1]]) - a) @ u)])
                olo, ohi = min(olo, olo2), max(ohi, ohi2)
                if min(hi, ohi) - max(lo, olo) > 0.02 and (ob[1][2] > bb[0][2] + 0.02 and bb[1][2] > ob[0][2] + 0.02):
                    ok = False; msgs.append(f"overlaps existing opening {o.GlobalId}"); break
            wb = _bbox_m(wall, scale)
            if intent.get("sill") is not None and wb is not None:
                sill = float(bb[0][2] - wb[0][2])
                if abs(sill - intent["sill"]) > tol:
                    ok = False
                msgs.append(f"sill {sill:.3f} vs {intent['sill']}")
    except Exception as ex:  # noqa: BLE001
        msgs.append(f"(position check skipped: {type(ex).__name__})")
    if intent.get("centre") is not None:   # E010: position given by the opening's bounding box -> plan centre of the new opening
        ob = _bbox_m(op, scale) or bb
        cxy = (ob[0][:2] + ob[1][:2]) / 2; d = float(np.linalg.norm(cxy - np.array(intent["centre"])))
        ok = ok and d <= max(tol, 0.05); msgs.append(f"centre {np.round(cxy, 3).tolist()} vs {intent['centre']} ({d:.3f} m)")
    return ok, f"new {cls} {e.GlobalId} hosted by the wall, " + ", ".join(msgs)


def _in_storey(e, storey_guid) -> bool:
    """Created element contained in (walls/columns/slabs) or aggregated into (spaces) the requested storey; stacked storeys share XY."""
    try:
        import ifcopenshell.util.element as _ue
        c = _ue.get_container(e)
        if c is None:
            c = _ue.get_aggregate(e)
        return c is not None and c.GlobalId == storey_guid
    except Exception:  # noqa: BLE001
        return False


def _create_wall(before, after, intent, scale):
    new = _added_of(before, after, ("IfcWall",))
    tol = intent.get("tol", 0.05); want = [np.array(intent["p0"], dtype=float), np.array(intent["p1"], dtype=float)]
    for e in new:
        from ifcgym.ifctools import wall_axis_world, _wall_thickness
        ax = wall_axis_world(e)
        if ax is None:
            continue
        a, b = np.array(ax[0][:2]) * scale, np.array(ax[1][:2]) * scale
        d = min(np.linalg.norm(a - want[0]) + np.linalg.norm(b - want[1]), np.linalg.norm(a - want[1]) + np.linalg.norm(b - want[0]))
        t = _wall_thickness(e) * scale; bb = _bbox_m(e, scale); h = float(bb[1][2] - bb[0][2]) if bb is not None else -1
        st = None
        try:
            st = pl_container(e)
        except Exception:  # noqa: BLE001
            pass
        ok = d <= 2 * tol and abs(t - intent["thickness"]) <= tol and abs(h - intent["height"]) <= tol and (not intent.get("storey") or _in_storey(e, intent["storey"]))
        return ok, f"new wall {e.GlobalId}: end-point error {d:.3f} m, thickness {t:.3f} vs {intent['thickness']}, height {h:.3f} vs {intent['height']}, in storey {_in_storey(e, intent.get('storey'))}"
    return False, f"no new wall (added {len(new)})"


def _create_box(before, after, intent, scale):
    cls = {"create_slab": "IfcSlab", "create_column": "IfcColumn"}[intent["type"]]
    new = _added_of(before, after, (cls,)); tol = intent.get("tol", 0.05)
    for e in new:
        bb = _bbox_m(e, scale)
        if bb is None:
            continue
        lo, hi = bb
        st_ok = (not intent.get("storey")) or _in_storey(e, intent["storey"])
        if intent["type"] == "create_slab":
            P = np.array(intent["polygon"], dtype=float); pm, pmsg = _poly_match(e, intent["polygon"])
            top_ok = intent.get("top") is None or abs(float(hi[2]) - intent["top"]) <= tol
            ok = pm and abs((hi[2] - lo[2]) - intent["thickness"]) <= tol and st_ok and top_ok
            return ok, f"new slab {e.GlobalId}: {pmsg}, thickness {hi[2]-lo[2]:.3f} vs {intent['thickness']}, top {hi[2]:.3f} vs {intent.get('top')}, in storey {st_ok}"
        c = (lo[:2] + hi[:2]) / 2; size = float(max(hi[0] - lo[0], hi[1] - lo[1]))
        if "sx" in intent:      # E010 coverage: rectangular cross-section given as bounding-box extents in X and Y
            size_ok = abs(float(hi[0] - lo[0]) - intent["sx"]) <= tol and abs(float(hi[1] - lo[1]) - intent["sy"]) <= tol
            smsg = f"extent {hi[0]-lo[0]:.3f} x {hi[1]-lo[1]:.3f} vs {intent['sx']} x {intent['sy']}"
        else:
            size_ok = abs(size - intent["size"]) <= tol; smsg = f"size {size:.3f}"
        z_ok = intent.get("z0") is None or abs(float(lo[2]) - intent["z0"]) <= tol    # E010: base elevation of the column
        ok = np.linalg.norm(c - np.array(intent["xy"])) <= tol and size_ok and abs((hi[2] - lo[2]) - intent["height"]) <= tol and st_ok and z_ok
        return ok, f"new column {e.GlobalId}: centre {np.round(c,3).tolist()} vs {intent['xy']}, {smsg}, height {hi[2]-lo[2]:.3f}, base {lo[2]:.3f} vs {intent.get('z0')}, in storey {st_ok}"
    return False, f"no new {cls}"


def _wall_axis_equals(after, intent, scale, before=None):   # before: the original model (body movement is judged relative to it)
    """Axis end points as requested AND the body following the axis: the mesh's extent along the new axis direction must match the
    new axis length (joint extensions allowed up to 0.5 m) and its centre must lie on the axis line (representation-independent)."""
    try:
        e = after.by_guid(intent["guid"])
    except Exception:  # noqa: BLE001
        return False, f"{intent['guid']} missing"
    from ifcgym.ifctools import wall_axis_world
    ax = wall_axis_world(e)
    if ax is None:
        return False, "no wall axis"
    a, b = np.array(ax[0][:2]) * scale, np.array(ax[1][:2]) * scale; want = [np.array(intent["p0"], dtype=float), np.array(intent["p1"], dtype=float)]
    d = min(np.linalg.norm(a - want[0]) + np.linalg.norm(b - want[1]), np.linalg.norm(a - want[1]) + np.linalg.norm(b - want[0]))
    ok = d <= 2 * intent.get("tol", 0.05); msg = f"axis {np.round(a, 3).tolist()}-{np.round(b, 3).tolist()} vs {intent['p0']}-{intent['p1']} (end-point error {d:.3f} m)"
    v = mesh(e)
    if v is None:
        return False, msg + "; no body geometry"
    u = (want[1] - want[0]) / (np.linalg.norm(want[1] - want[0]) or 1.0); n = np.array([-u[1], u[0]])
    pu = v[:, :2] @ u; lo, hi = sorted([float(want[0] @ u), float(want[1] @ u)])
    body_ok = True; bmsg = ""
    if before is not None:   # the body must follow the axis: its extent along the new axis must match the axis change, relative to the original body
        try:
            v0 = mesh(before.by_guid(intent["guid"])); ax0 = wall_axis_world(before.by_guid(intent["guid"]))
            if v0 is not None and ax0 is not None:
                a0, b0 = np.array(ax0[0][:2]) * scale, np.array(ax0[1][:2]) * scale
                u0 = (b0 - a0) / (np.linalg.norm(b0 - a0) or 1.0)
                if abs(float(u0 @ u)) > 0.98:      # same direction (shorten): each body end moves as much as the axis end nearest to it
                    pu0 = v0[:, :2] @ u; lo0, hi0 = sorted([float(a0 @ u), float(b0 @ u)])
                    d_lo, d_hi = lo - lo0, hi - hi0
                    body_ok = abs((pu.min() - pu0.min()) - d_lo) <= 0.1 and abs((pu.max() - pu0.max()) - d_hi) <= 0.1
                    bmsg = f"; body ends moved {pu.min() - pu0.min():+.2f}/{pu.max() - pu0.max():+.2f} vs axis {d_lo:+.2f}/{d_hi:+.2f}"
                else:                              # rotated: the body's principal direction follows the new axis and its length is kept
                    vv = v[:, :2] - v[:, :2].mean(0); sv = np.linalg.svd(vv, full_matrices=False)[2][0]
                    L0 = float((v0[:, :2] @ u0).max() - (v0[:, :2] @ u0).min()); L1 = float(pu.max() - pu.min())
                    body_ok = abs(float(sv @ u)) > 0.98 and abs(L1 - L0) <= 0.1 and abs(float((v[:, :2].mean(0) - want[0]) @ n)) <= 0.5
                    bmsg = f"; body direction follows axis {abs(float(sv @ u)) > 0.98}, length {L1:.2f} vs {L0:.2f}"
        except Exception as ex:  # noqa: BLE001
            bmsg = f"; body check skipped ({type(ex).__name__})"
    ok = ok and body_ok
    return ok, msg + bmsg


def _thickness_equals(after, intent, scale, before=None):
    try:
        e = after.by_guid(intent["guid"])
    except Exception:  # noqa: BLE001
        return False, f"{intent['guid']} missing"
    from ifcgym.ifctools import _wall_thickness, wall_axis_world
    t = _wall_thickness(e) * scale
    ok = (abs(t - intent["thickness"]) <= intent.get("tol", 0.02)) if t > 0 else True   # Brep walls have no profile: judged by the body below
    msg = f"profile thickness {t:.3f} vs {intent['thickness']}" if t > 0 else "no profile (Brep body)"
    v = mesh(e); ax = wall_axis_world(e)
    if v is not None and ax is not None:   # body: extent across the axis = thickness, centred on the axis line
        a, b = np.array(ax[0][:2]) * scale, np.array(ax[1][:2]) * scale; u = (b - a) / (np.linalg.norm(b - a) or 1.0); n = np.array([-u[1], u[0]])
        pn = (v[:, :2] - a) @ n; tm = float(pn.max() - pn.min()); cm = float((pn.max() + pn.min()) / 2)
        cm0 = 0.0
        if before is not None:   # the centre line must stay where it was (exporters may put the Axis on a face)
            try:
                v0 = mesh(before.by_guid(intent["guid"])); pn0 = (v0[:, :2] - a) @ n; cm0 = float((pn0.max() + pn0.min()) / 2)
            except Exception:  # noqa: BLE001
                cm0 = cm
        ok = ok and abs(tm - intent["thickness"]) <= intent.get("tol", 0.02) + 0.01 and abs(cm - cm0) <= 0.03
        msg += f"; body thickness {tm:.3f}, centre line moved {cm - cm0:+.3f}"
    return ok, msg


def _bbox_equals(after, intent, scale):
    try:
        e = after.by_guid(intent["guid"])
    except Exception:  # noqa: BLE001
        return False, f"{intent['guid']} missing"
    bb = _bbox_m(e, scale)
    if bb is None:
        return False, "no geometry"
    lo, hi = np.array(intent["lo"], dtype=float), np.array(intent["hi"], dtype=float); n = len(lo); tol = intent.get("tol", 0.05)
    ok = np.allclose(bb[0][:n], lo, atol=tol) and np.allclose(bb[1][:n], hi, atol=tol)
    return ok, f"bbox {np.round(bb[0][:n], 3).tolist()}-{np.round(bb[1][:n], 3).tolist()} vs {lo.tolist()}-{hi.tolist()}"


def _create_space(before, after, intent, scale):
    new = _added_of(before, after, ("IfcSpace",)); tol = intent.get("tol", 0.1); P = np.array(intent["polygon"], dtype=float)
    for e in new:
        bb = _bbox_m(e, scale)
        if bb is None:
            continue
        lo, hi = bb
        pm, pmsg = _poly_match(e, intent["polygon"], tol_frac=0.05, tol_abs=0.1)
        z_ok = intent.get("z0") is None or abs(float(lo[2]) - intent["z0"]) <= tol    # E010: base elevation of a space given by its bbox
        ok = pm and abs((hi[2] - lo[2]) - intent["height"]) <= tol and ((not intent.get("storey")) or _in_storey(e, intent["storey"])) and z_ok
        return ok, f"new space {e.GlobalId}: {pmsg}, height {hi[2]-lo[2]:.3f} vs {intent['height']}, base {lo[2]:.3f} vs {intent.get('z0')}, in storey {_in_storey(e, intent.get('storey'))}"
    return False, "no new IfcSpace"


def _answer(intent, final_text: str | None):
    import json as _json, re as _re
    key = {"guid": "global_id", "guids": "global_ids"}.get(intent["kind"], "value")
    got = None
    for m in reversed(list(_re.finditer(r"\{[^{}]*\}", final_text or ""))):
        try:
            d = _json.loads(m.group(0))
        except _json.JSONDecodeError:
            continue
        if key in d:
            got = d[key]; break
    if got is None:
        return False, f"no JSON with key {key!r} in the final message"
    if intent["kind"] == "guid":
        return str(got).strip() == intent["value"], f"answered {got!r} vs {intent['value']!r}"
    if intent["kind"] == "guids":
        want = set(intent["value"]); have = {str(x).strip() for x in got} if isinstance(got, (list, tuple)) else {str(got).strip()}
        return have == want, f"answered {sorted(have)} vs {sorted(want)}"
    try:
        g = float(got)
    except (TypeError, ValueError):
        return False, f"answer {got!r} is not a number"
    tol = intent.get("tol", 0.0); rel = intent.get("rel_tol", 0.0)
    ok = abs(g - intent["value"]) <= max(tol, rel * abs(intent["value"]))
    return ok, f"answered {g} vs {intent['value']} (tol {max(tol, rel * abs(intent['value'])):.3g})"


def check_intent(before, after, intent: dict, scale: float, final_text: str | None = None) -> tuple[bool, str]:
    t = intent["type"]
    if t == "all":   # compound instruction (E009, 2026-10-26): every part must hold; the message lists each part
        res = [check_intent(before, after, p_, scale, final_text) for p_ in intent["parts"]]
        return all(r_[0] for r_ in res), f"parts {sum(1 for r_ in res if r_[0])}/{len(res)} | " + " | ".join(f"{i_ + 1}:{'ok' if r_[0] else 'FAIL'} {r_[1][:100]}" for i_, r_ in enumerate(res))
    if t == "create_fill":
        return _create_fill(before, after, intent, scale)
    if t == "create_wall":
        return _create_wall(before, after, intent, scale)
    if t in ("create_slab", "create_column"):
        return _create_box(before, after, intent, scale)
    if t == "answer":
        return _answer(intent, final_text)
    if t == "translate":
        return _translated(before, after, intent["guid"], np.array(intent["vector"]), intent.get("tol", 0.02) + 1e-6, scale)
    if t == "set_attribute":
        return _attribute_equals(after, intent["guid"], intent["name"], intent["value"])
    if t == "set_property":
        return _property_equals(after, intent["guid"], intent["pset"], intent["prop"], intent["value"], intent.get("tol", 0.0))
    if t == "delete":
        return _deleted(after, intent["guid"])
    if t == "resize_height":
        return _height_equals(after, intent["guid"], intent["height"], intent.get("tol", 0.02), intent.get("top"), intent.get("bottom"))
    if t == "delete_all":
        res = [_deleted(after, g_) for g_ in intent["guids"]]
        return all(r_[0] for r_ in res), f"removed {sum(1 for r_ in res if r_[0])}/{len(res)}"
    if t == "resize_height_all":
        res = [_height_equals(after, g_, intent["height"], intent.get("tol", 0.02)) for g_ in intent["guids"]]
        bad = [m for ok_, m in res if not ok_]
        return not bad, f"{len(res) - len(bad)}/{len(res)} at height {intent['height']}" + (f"; e.g. {bad[0]}" if bad else "")
    if t == "wall_axis":          # v4: shorten / rotate a wall - expected end points in metres
        return _wall_axis_equals(after, intent, scale, before)
    if t == "set_thickness":
        return _thickness_equals(after, intent, scale, before)
    if t == "bbox":               # v4: resize a slab - expected world bbox (metres, 2D or 3D)
        return _bbox_equals(after, intent, scale)
    if t == "create_space":
        return _create_space(before, after, intent, scale)
    if t in ("create_wall_bbox", "set_thickness_side", "rehost_fill", "opening_width", "create_slab_opening", "space_footprint"):
        return _e010_intent(before, after, intent, scale)
    return False, f"unknown intent {t}"


def _fill_opening(f, guid):
    e = f.by_guid(guid); r = list(getattr(e, "FillsVoids", None) or [])
    return e, (r[0].RelatingOpeningElement if r else None)


def _e010_intent(before, after, intent, scale):
    """E010 coverage intents (2026-10-26), all judged on meshes in metres."""
    t = intent["type"]; tol = intent.get("tol", 0.05)
    if t == "create_wall_bbox":       # a new wall whose body fills the given bounding box (representation-independent)
        new = _added_of(before, after, ("IfcWall", "IfcWallStandardCase"))
        for e in new:
            bb = _bbox_m(e, scale)
            if bb is None:
                continue
            ok = np.allclose(bb[0], intent["lo"], atol=tol) and np.allclose(bb[1], intent["hi"], atol=tol) and ((not intent.get("storey")) or _in_storey(e, intent["storey"]))
            return ok, f"new wall {e.GlobalId}: bbox {np.round(bb[0], 3).tolist()}-{np.round(bb[1], 3).tolist()} vs {intent['lo']}-{intent['hi']}, in storey {_in_storey(e, intent.get('storey'))}"
        return False, f"no new wall (added {len(new)})"
    if t == "set_thickness_side":     # thickness grows / shrinks towards n only: the face on the other side keeps its position
        try:
            v = mesh(after.by_guid(intent["guid"]))
        except Exception:  # noqa: BLE001
            return False, f"{intent['guid']} missing"
        if v is None:
            return False, "no body"
        n = np.array(intent["n"], dtype=float); pn = v[:, :2] @ n; th = float(pn.max() - pn.min())
        ok = abs(th - intent["thickness"]) <= intent.get("tol", 0.02) and abs(float(pn.min()) - intent["fixed"]) <= intent.get("tol", 0.02)
        return ok, f"thickness {th:.3f} vs {intent['thickness']}, fixed face {float(pn.min()):.3f} vs {intent['fixed']}"
    if t == "rehost_fill":            # the fill's opening now voids the new host; plan position kept; opening bottom at zmin
        try:
            e, op = _fill_opening(after, intent["guid"])
        except Exception:  # noqa: BLE001
            return False, f"{intent['guid']} missing"
        if op is None:
            return False, "fill has no opening"
        hosts = [r.RelatingBuildingElement.GlobalId for r in (op.VoidsElements or [])]
        bb, ob = _bbox_m(e, scale), _bbox_m(op, scale)
        if bb is None or ob is None:
            return False, "no geometry"
        cxy = (bb[0][:2] + bb[1][:2]) / 2; d = float(np.linalg.norm(cxy - np.array(intent["xy"])))
        ok = hosts == [intent["host"]] and d <= tol and abs(float(ob[0][2]) - intent["zmin"]) <= tol
        return ok, f"opening voids {hosts} (want {intent['host']}), plan centre off by {d:.3f}, opening min z {ob[0][2]:.3f} vs {intent['zmin']}"
    if t == "opening_width":          # the fill's opening has the new width along its wall, centred where it was
        try:
            e, op = _fill_opening(after, intent["guid"])
        except Exception:  # noqa: BLE001
            return False, f"{intent['guid']} missing"
        ob = _bbox_m(op, scale) if op is not None else None
        if ob is None:
            return False, "no opening geometry"
        u = np.array(intent["u"], dtype=float); v = mesh(op); pu = v[:, :2] @ u; w = float(pu.max() - pu.min()); c = float((pu.max() + pu.min()) / 2)
        ok = abs(w - intent["width"]) <= tol and abs(c - intent["centre_u"]) <= tol
        return ok, f"opening width {w:.3f} vs {intent['width']}, centre {c:.3f} vs {intent['centre_u']}"
    if t == "create_slab_opening":    # a new opening voids the slab, plan bbox as requested, through the slab's thickness
        new = _added_of(before, after, ("IfcOpeningElement",))
        try:
            hb = _bbox_m(after.by_guid(intent["host"]), scale)
        except Exception:  # noqa: BLE001
            return False, "host slab missing"
        for o in new:
            if intent["host"] not in [r.RelatingBuildingElement.GlobalId for r in (o.VoidsElements or [])]:
                continue
            ob = _bbox_m(o, scale)
            if ob is None or hb is None:
                continue
            ok = np.allclose(ob[0][:2], intent["lo"], atol=tol) and np.allclose(ob[1][:2], intent["hi"], atol=tol) and ob[0][2] <= hb[0][2] + 0.01 and ob[1][2] >= hb[1][2] - 0.01
            return ok, f"new opening {o.GlobalId}: plan {np.round(ob[0][:2], 3).tolist()}-{np.round(ob[1][:2], 3).tolist()} vs {intent['lo']}-{intent['hi']}, z {ob[0][2]:.3f}-{ob[1][2]:.3f} vs slab {hb[0][2]:.3f}-{hb[1][2]:.3f}"
        return False, f"no new opening voiding {intent['host']} (added {len(new)})"
    if t == "space_footprint":        # an existing space now has this footprint (rooms merged) and keeps its height
        try:
            e = after.by_guid(intent["guid"])
        except Exception:  # noqa: BLE001
            return False, f"{intent['guid']} missing"
        pm, pmsg = _poly_match(e, intent["polygon"], tol_frac=0.03, tol_abs=0.1); bb = _bbox_m(e, scale)
        ok = pm and bb is not None and abs(float(bb[1][2] - bb[0][2]) - intent["height"]) <= intent.get("tol", 0.05)
        return ok, f"{pmsg}, height {float(bb[1][2] - bb[0][2]) if bb is not None else None} vs {intent['height']}"
    return False, f"unknown intent {t}"


# names used by the generator before commit 740fa37 (outputs/tasks/wbdg_office.jsonl, 10 frozen dev-val tier-3 tasks, was not regenerated):
# same rules, same keys - resolved here so frozen task files need not change (incident 2026-10-20, research log)
_RULE_ALIASES = {"space_face_follows": "space_meets_wall", "boundary_follows": "boundary_on_wall"}


def _e010_rule(before, after, d, scale):
    """E010 coverage rules (2026-10-26)."""
    r = d["rule"]
    def conns(e):
        out = set()
        for rel in list(getattr(e, "ConnectedTo", []) or []) + list(getattr(e, "ConnectedFrom", []) or []):
            o = rel.RelatedElement if rel.RelatingElement == e else rel.RelatingElement
            if o is not None:
                out.add(o.GlobalId)
        return out
    try:
        if r == "not_connected":     # the connection between d['el'] and d['guid'] was deleted (both elements remain)
            e = after.by_guid(d["el"]); return d["guid"] not in conns(e), "connected" if d["guid"] in conns(e) else "no connection"
        if r == "space_boundary_with":   # an IfcRelSpaceBoundary relates the space d['guid'] to the element d['el']
            sp = after.by_guid(d["guid"]); els = {b.RelatedBuildingElement.GlobalId for b in (sp.BoundedBy or []) if b.RelatedBuildingElement is not None}
            return d["el"] in els, f"bounded by {len(els)} elements"
        if r == "new_space_bounds":      # the NEW space has an IfcRelSpaceBoundary with the element d['guid']
            for sp in _added_of(before, after, ("IfcSpace",)):
                if d["guid"] in {b.RelatedBuildingElement.GlobalId for b in (sp.BoundedBy or []) if b.RelatedBuildingElement is not None}:
                    return True, "bounded"
            return False, "no new space bounded by it"
        if r == "opening_covers":        # the fill's opening still covers the fill vertically (enlarged with it)
            e, op = _fill_opening(after, d["guid"]); fb, ob = _bbox_m(e, scale), _bbox_m(op, scale) if op is not None else None
            if fb is None or ob is None:
                return False, "no geometry"
            return ob[0][2] <= fb[0][2] + 0.02 and ob[1][2] >= fb[1][2] - 0.02, f"opening z {ob[0][2]:.3f}-{ob[1][2]:.3f} vs fill {fb[0][2]:.3f}-{fb[1][2]:.3f}"
    except Exception as e:  # noqa: BLE001
        return False, f"error {type(e).__name__}"
    return False, "unknown rule"


def check_dependents(before, after, deps: list[dict], scale: float, eg=None) -> tuple[bool, str]:
    msgs = []; ok = True
    for d in deps:
        r = d["rule"]
        if r == "new_in_storey":   # E009: every NEW element of class d['cls'] is contained in the storey d['guid'] (IfcRelContainedInSpatialStructure)
            new = _added_of(before, after, (d.get("cls", "IfcDoor"),))
            good = bool(new)
            for e in new:
                cs = [rel.RelatingStructure for rel in (getattr(e, "ContainedInStructure", None) or [])]
                good &= any(x is not None and x.GlobalId == d["guid"] for x in cs)
            ok &= good; msgs.append(f"{d['guid']}:{r}:{'ok' if good else 'FAIL'}"); continue
        if r == "connected_new":   # v4: the NEW element of class d['cls'] must be connected to d['guid'] by IfcRelConnectsPathElements
            new = _added_of(before, after, (d.get("cls", "IfcWall"),))
            good = False
            for e in new:
                for rel in list(getattr(e, "ConnectedTo", []) or []) + list(getattr(e, "ConnectedFrom", []) or []):
                    other = rel.RelatedElement if rel.RelatingElement == e else rel.RelatingElement
                    if other is not None and other.GlobalId == d["guid"]:
                        good = True
            ok &= good; msgs.append(f"{d['guid']}:{r}:{'ok' if good else 'FAIL'}"); continue
        if r == "space_boundary_new":   # E010: an IfcRelSpaceBoundary relates the space d['guid'] to the NEW element of class d['cls']
            new = _added_of(before, after, (d.get("cls", "IfcColumn"),)); good = False
            for e in new:
                for rel in getattr(e, "ProvidesBoundaries", None) or []:
                    if rel.RelatingSpace is not None and rel.RelatingSpace.GlobalId == d["guid"]:
                        good = True
            ok &= good; msgs.append(f"{d['guid']}:{r}:{'ok' if good else 'FAIL'}"); continue
        if r == "in_storey":            # E010: the existing element d['el'] is contained in the storey d['guid'] and in no other structure
            try:
                cs = [rel.RelatingStructure.GlobalId for rel in (after.by_guid(d["el"]).ContainedInStructure or []) if rel.RelatingStructure is not None]
                good = cs == [d["guid"]]; m = f"contained in {cs}"
            except Exception:  # noqa: BLE001
                good, m = False, "element missing"
            ok &= good; msgs.append(f"{d['guid']}:{r}:{'ok' if good else 'FAIL'}({m})"); continue
        if r == "rel_connects":         # E010: existing elements d['el'] and d['guid'] are connected by an IfcRelConnectsElements (any subtype)
            good = False
            try:
                e = after.by_guid(d["el"])
                for rel in list(getattr(e, "ConnectedTo", []) or []) + list(getattr(e, "ConnectedFrom", []) or []):
                    other = rel.RelatedElement if rel.RelatingElement == e else rel.RelatingElement
                    good |= other is not None and other.GlobalId == d["guid"]
            except Exception:  # noqa: BLE001
                good = False
            ok &= good; msgs.append(f"{d['guid']}:{r}:{'ok' if good else 'FAIL'}"); continue
        if r in ("not_connected", "space_boundary_with", "new_space_bounds", "opening_covers"):
            good, m = _e010_rule(before, after, d, scale); ok &= good; msgs.append(f"{d['guid']}:{r}:{'ok' if good else 'FAIL'}({m})"); continue
        if r == "unchanged":       # v4: an element that "must remain": present and not modified at all (attributes, geometry, placement)
            if eg is None:
                good, m = _deleted(after, d["guid"])[0] is False, "present"
            else:
                good = d["guid"] not in eg.removed and d["guid"] not in eg.modified; m = "unchanged" if good else ("removed" if d["guid"] in eg.removed else "modified")
            ok &= good; msgs.append(f"{d['guid']}:{r}:{'ok' if good else 'FAIL'}({m})"); continue
        if r == "translate_with":
            good, m = _translated(before, after, d["guid"], np.array(d["vector"]), d.get("tol", 0.02) + 1e-6, scale)
        elif r == "removed":
            good, m = _deleted(after, d["guid"])
        elif r == "stays_hosted":   # opening still voids its wall and, if it was filled, is still filled
            try:
                op = after.by_guid(d["guid"])
                try:
                    was_filled = bool(before.by_guid(d["guid"]).HasFillings)
                except Exception:  # noqa: BLE001
                    was_filled = True
                good = bool(op.VoidsElements) and (bool(op.HasFillings) or not was_filled); m = "hosted" if good else "host/fill relation lost"
            except Exception:  # noqa: BLE001
                good, m = False, "opening missing"
        elif r == "joined_to":        # tier 3: a connected wall must not lose its joint: its gap to the moved wall may not grow beyond the gap it had before (+tol)
            try:
                g0 = _gap(before.by_guid(d["guid"]), before.by_guid(d["other"]))
                g1 = _gap(after.by_guid(d["guid"]), after.by_guid(d["other"]))
                if g0 is None or g1 is None:
                    good, m = False, "missing geometry"
                else:   # surface gap (2026-10-20: was corner-vertex distance, which grew when a stem wall slid along a wall face)
                    good = g1 <= g0 + d.get("tol", 0.02) + 1e-6
                    m = f"gap before {g0:.4f} after {g1:.4f}"
            except Exception as e:  # noqa: BLE001
                good, m = False, f"missing: {e}"
        elif r == "space_meets_wall":     # tier 2: the bounded space's geometry still meets a CURRENT wall face (only enforced if it met the wall in the original model)
            try:
                v = np.array(d["vector"], dtype=float); ax = int(np.argmax(np.abs(v))); tol = d.get("tol", 0.02) + 1e-6   # metres (meshes)
                def gap_of(f):
                    sm, wm = mesh(f.by_guid(d["guid"])), mesh(f.by_guid(d["wall"]))
                    if sm is None or wm is None:
                        return None
                    oax = 1 - ax if ax < 2 else 0
                    lo, hi = wm[:, oax].min() - 0.05, wm[:, oax].max() + 0.05
                    faces = np.array([wm[:, ax].min(), wm[:, ax].max()]); window = abs(float(v[ax])) + 0.5
                    near = sm[(sm[:, oax] > lo) & (sm[:, oax] < hi) & (np.min(np.abs(sm[:, [ax]] - faces[None, :]), axis=1) <= window)]
                    return None if len(near) == 0 else float(np.min(np.abs(near[:, [ax]] - faces[None, :])))
                g0 = gap_of(before)
                if g0 is None or g0 > tol:
                    good, m = True, f"n/a: space did not meet the wall in the original model (gap {g0})"
                else:
                    g1 = gap_of(after)
                    good, m = (g1 is not None and g1 <= tol), f"gap to nearest wall face {g1}"
            except Exception as e:  # noqa: BLE001
                good, m = False, f"error: {e}"
        elif r == "boundary_on_wall":     # tier 2: space-boundary surfaces of this space with this wall lie on the wall's current centreline/faces (enforced only if they did in the original)
            try:
                v = np.array(d["vector"], dtype=float); ax = int(np.argmax(np.abs(v))); tol = d.get("tol", 0.02) + 1e-6   # metres
                def deviation(f):
                    wm = mesh(f.by_guid(d["wall"]))
                    if wm is None:
                        return None
                    wmin, wmax = wm[:, ax].min(), wm[:, ax].max(); centre = (wmin + wmax) / 2
                    sp = f.by_guid(d["guid"]); pts = []
                    for b in getattr(sp, "BoundedBy", []) or []:
                        if not (b.RelatedBuildingElement and b.RelatedBuildingElement.GlobalId == d["wall"] and b.ConnectionGeometry):
                            continue
                        g = b.ConnectionGeometry.SurfaceOnRelatingElement
                        if g is None or not hasattr(g, "Position"):
                            continue
                        M = np.array(pl.get_local_placement(sp.ObjectPlacement)) @ np.array(pl.get_axis2placement(g.Position))
                        curve = getattr(getattr(g, "SweptCurve", None), "Curve", None) or getattr(g, "OuterBoundary", None)
                        for p in list(getattr(curve, "Points", []) or []):
                            c = list(p.Coordinates) + [0.0] * (3 - len(p.Coordinates)); pts.append((M @ np.array(c + [1.0]))[:3] * scale)   # model units -> metres
                    if not pts:
                        return None
                    P = np.array(pts); return float(np.max(np.min(np.abs(P[:, [ax]] - np.array([[centre, wmin, wmax]])), axis=1))), len(P)
                d0 = deviation(before)
                if d0 is None or d0[0] > tol:
                    good, m = True, f"n/a: no evaluable boundary on the wall in the original model ({d0})"
                else:
                    d1 = deviation(after)
                    good = d1 is not None and d1[0] <= tol; m = f"boundary deviation {d1}"
            except Exception as e:  # noqa: BLE001
                good, m = False, f"error: {e}"
        elif r == "present":
            try:
                after.by_guid(d["guid"]); good, m = True, "present"
            except Exception:  # noqa: BLE001
                good, m = False, "missing"
        else:
            good, m = False, f"unknown rule {r}"
        ok &= good; msgs.append(f"{d['guid']}:{r}:{'ok' if good else 'FAIL'} {m}")
    return ok, "; ".join(msgs)


# ------------------------------------------------------------------------------------------ cache of the original model
import functools, os, pickle, hashlib, threading


def _cache_valid(f, graph, n_sample: int = 24) -> bool:
    """Recompute the hashes of a deterministic sample of products and compare with the cached graph."""
    from ifcgym.ifcgraph import build_graph
    nodes = graph[0]
    guids = sorted(nodes)
    if not guids:
        return True
    step = max(1, len(guids) // n_sample)
    sample = guids[::step][:n_sample]
    for g in sample:
        try:
            inst = f.by_guid(g)
        except Exception:  # noqa: BLE001
            return False
        from ifcgym.ifcgraph import _hash_subtree
        rep = getattr(inst, "Representation", None); pl = getattr(inst, "ObjectPlacement", None)
        gh = _hash_subtree(rep, depth=24) if rep else None; ph = _hash_subtree(pl, depth=24) if pl else None
        if nodes[g].geom_hash != gh or nodes[g].placement_hash != ph:
            return False
    return True


@functools.lru_cache(maxsize=8)
def before_model(before_path: str):
    """(graph, schema_errors) of an ORIGINAL model - pure data, cached in memory and on disk. The ifcopenshell.file object is NOT
    cached: run_tasks checks from several threads at once and ifcopenshell file objects are not thread-safe (2026-10-19: a shared
    object produced hundreds of phantom side effects and false intent failures in every concurrent run)."""
    from ifcgym.ifcgraph import build_graph
    st = os.stat(before_path); key = hashlib.sha1(f"{before_path}|{st.st_size}|{int(st.st_mtime)}|v2".encode()).hexdigest()
    cdir = os.environ.get("IFCGYM_CACHE", os.path.join(os.environ.get("IFCGYM_ROOT", os.getcwd()), "outputs", "cache")); os.makedirs(cdir, exist_ok=True)
    cp = os.path.join(cdir, key + ".pkl")
    f = ifcopenshell.open(before_path)
    if os.path.exists(cp):
        try:
            with open(cp, "rb") as fh:
                graph, errs = pickle.load(fh)
            if _cache_valid(f, graph):
                return graph, errs
            os.remove(cp)      # built under a thread-contaminated condition (seen 2026-10-19): rebuild
        except Exception:  # noqa: BLE001
            pass
    graph = build_graph(f); errs = set(schema_errors(f))
    tmp = cp + f".{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp, "wb") as fh:
        pickle.dump((graph, errs), fh)
    os.replace(tmp, cp)
    return graph, errs


# ------------------------------------------------------------------------------------------ the gate
def pl_container(e):
    import ifcopenshell.util.element as _ue
    return _ue.get_container(e)


def check(before_path: str, after_path: str, spec: dict, final_text: str | None = None) -> CheckReport:
    before_graph, err_before = before_model(before_path)
    before = ifcopenshell.open(before_path)          # a fresh, thread-private file object every time
    try:
        if not os.path.exists(after_path) and os.path.exists(after_path + ".gz"):   # archived outputs (edited IFCs gzipped 2026-10-23)
            import gzip, tempfile
            with gzip.open(after_path + ".gz", "rb") as gz, tempfile.NamedTemporaryFile("wb", suffix=".ifc", delete=False) as tf:
                tf.write(gz.read()); after_path = tf.name
        after = ifcopenshell.open(after_path)
    except Exception as e:  # noqa: BLE001
        return CheckReport(False, False, False, False, False, 0, 1, {"validity": f"unreadable: {e}"})
    cyc = decomposition_cycle(after)
    if cyc:   # non-terminating geometry otherwise; treated like an unreadable output (2026-11-04)
        return CheckReport(False, False, False, False, False, 0, 1, {"validity": f"decomposition cycle at {cyc}", "new_errors": [f"decomposition cycle at {cyc}"]})
    scale = unit_scale(before)
    from ifcgym.ifcgraph import edit_graph_from
    eg: EditGraph = edit_graph_from(before_graph, after)
    touched = sorted(eg.node_ids)
    allowed = set(spec.get("allowed") or []) | set(spec.get("target") or []) | {d["guid"] for d in spec.get("dependents") or []}
    # validity: new schema errors vs the input (pre-existing errors are not the agent's fault), geometry of touched elements rebuilds
    err_after = set(schema_errors(after))
    new_err = sorted(err_after - err_before)
    gok, gbad = geometry_ok(after, [g for g in touched if g not in eg.removed])
    validity = not new_err and gbad == 0
    intent_ok, intent_msg = check_intent(before, after, spec["intent"], scale, final_text)
    deps = [dict(d, rule=_RULE_ALIASES[d["rule"]], wall=d.get("wall") or (spec.get("target") or [None])[0]) if d.get("rule") in _RULE_ALIASES else d
            for d in (spec.get("dependents") or [])]       # pre-740fa37 records carry no 'wall': it is the task's target wall
    dep_ok, dep_msg = check_dependents(before, after, deps, scale, eg)
    new_ok = set(spec.get("allowed_new_classes") or [])      # create tasks: new entities of these classes are expected, not side effects
    if new_ok:
        added_ok = {e.GlobalId for c in new_ok for e in after.by_type(c)} - {e.GlobalId for c in new_ok for e in before.by_type(c)}
        allowed |= added_ok
    side = [g for g in touched if g not in allowed]
    # relationship-level side effects: a relation added/removed between elements that are all outside the allowed set (e.g. a space
    # boundary or a wall connection of unrelated elements deleted) - invisible to the node diff (review 2026-10-22)
    rel_side = [k for k, e in list(eg.rel_added.items()) + list(eg.rel_removed.items()) if e.src not in allowed and e.dst not in allowed]
    side += [f"rel:{k[0]}:{k[1]}->{k[2]}" for k in rel_side]
    preservation = not side
    under_edit = (not intent_ok) and len(touched) == 0 and spec["intent"]["type"] != "answer"
    return CheckReport(validity, intent_ok, dep_ok, preservation, under_edit, len(side), len(new_err),
                       {"validity": f"new schema errors {len(new_err)}, geometry failures {gbad}/{gok + gbad}", "intent": intent_msg,
                        "dependency": dep_msg, "preservation": f"changed outside the allowed set: {side[:8]}", "edit": eg.summary(),
                        "new_errors": new_err[:5]})
