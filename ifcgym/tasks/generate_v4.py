"""Generator v4: the operation x entity x reference
grid of BIM-Edit (create / update / delete on walls, doors, windows, columns, slabs, spaces; each addressed DIRECTLY by GlobalId,
TOPOLOGICALLY through relationships, SPATIALLY through geometry, or along a PATH: "after entering B from A through door D, the wall on your right") and the basic / geometry / topological / numeric retrieval
types of BIBIMBAP, plus the half of BIBIMBAP's "conceptual" questions that admit an operational definition (corridor, hub,
entrance, shortest door path, vertical connector). Truth is programmatic. Template ids: <family>-<name>-<D|T|S|P>.

All v4 tasks run on a derived input <dir>/arc_v4.ifc written next to the building's arc.ifc: 3-6 columns added inside rooms (the
builder has no columns) and 1-2 spaces removed (so "create a room" tasks have an empty area with its walls still there).
Prompts and intents are in METRES; helpers take model units.
"""
from __future__ import annotations

import json
import math
import os
import random

import numpy as np
import uuid

import ifcopenshell
import ifcopenshell.guid
import ifcopenshell.util.element as ue

from ifcgym.tasks.generate import _openings, _fills, _spaces_bounded, _connected_walls
from ifcgym.ifctools import wall_axis_world, _wall_thickness, profile_world_polygon, world_matrix, unit_scale, space_area, add_column, delete, _mesh_bbox
from ifcgym.checkers import _footprint, mesh as _mesh
from shapely.geometry import Point

DERIVE_VERSION = "v4.3"   # bump when derive_input changes: arc_v4.ifc is re-derived when the recorded version differs
DIRS = {"+x": np.array([1.0, 0.0]), "-x": np.array([-1.0, 0.0]), "+y": np.array([0.0, 1.0]), "-y": np.array([0.0, -1.0])}
DIR_WORD = {"+x": "east (+X)", "-x": "west (-X)", "+y": "north (+Y)", "-y": "south (-Y)"}


def _seg_d(p, q0, q1) -> float:
    v = q1 - q0; tt = np.clip(np.dot(p - q0, v) / (np.dot(v, v) or 1e-12), 0, 1); return float(np.linalg.norm(p - (q0 + tt * v)))


def _body_items(e):
    try:
        return [it for r in e.Representation.Representations if r.RepresentationIdentifier == "Body" for it in r.Items]
    except Exception:  # noqa: BLE001
        return []


def _extrusion_depth(it):
    from ifcgym.ifctools import _extrusion_of
    ex = _extrusion_of(it)
    return None if ex is None or abs(abs(ex.ExtrudedDirection.DirectionRatios[2]) - 1) > 1e-3 else float(ex.Depth)


def free_slot(c, w, width: float) -> bool:
    """Is there room along wall w (metres) for a new opening of `width` that overlaps no existing opening? (mirrors the checker)"""
    a, b = c.axis[w.GlobalId]; u = c.udir(w); L = c.length(w); busy = []
    for o in _openings(w):
        bb = c.bbox(o)
        if bb is None:
            continue
        cs = [np.array([bb[0][0], bb[0][1]]), np.array([bb[1][0], bb[1][1]]), np.array([bb[0][0], bb[1][1]]), np.array([bb[1][0], bb[0][1]])]
        ps = [float((p_ - a) @ u) for p_ in cs]; busy.append((min(ps), max(ps)))
    x = 0.1
    while x + width <= L - 0.1:
        if all(x + width <= lo - 0.02 or x >= hi + 0.02 for lo, hi in busy):
            return True
        x += 0.05
    return False


def axis_centred(c, w, tol: float = 0.03) -> bool:
    """The Axis line runs along the body's centre line (ArchiCAD can put it on a face: then 'symmetric about its axis' is wrong)."""
    bb = c.bbox(w)
    if bb is None:
        return False
    a, b = c.axis[w.GlobalId]; u = c.udir(w); n = np.array([-u[1], u[0]])
    v = _mesh(w)
    if v is None:
        return False
    pn = (v[:, :2] - a) @ n
    return abs(float((pn.max() + pn.min()) / 2)) <= tol


def _poly_area(P):
    x, y = P[:, 0], P[:, 1]; return float(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _inside(pt, P) -> bool:
    x, y = pt; n = len(P); ins = False; j = n - 1
    for i in range(n):
        xi, yi = P[i]; xj, yj = P[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi:
            ins = not ins
        j = i
    return ins


def _dist_margin(pt, P) -> float:
    """Distance from a point to the polygon boundary (metres)."""
    best = 1e9
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]; ab = b - a; t = np.clip(np.dot(pt - a, ab) / (np.dot(ab, ab) or 1e-12), 0, 1)
        best = min(best, float(np.linalg.norm(pt - (a + t * ab))))
    return best


class Ctx:
    """Indexed view of one model (metres)."""
    def __init__(self, f, rng):
        self.f = f; self.rng = rng; self.sc = unit_scale(f) or 1.0
        self.storeys = sorted([s for s in f.by_type("IfcBuildingStorey") if s.Name], key=lambda s: float(s.Elevation or 0))
        self.walls = [w for w in f.by_type("IfcWall") if getattr(w, "Representation", None) and wall_axis_world(w)]
        self.axis = {w.GlobalId: tuple(np.array(p[:2]) * self.sc for p in wall_axis_world(w)) for w in self.walls}
        self.z = {w.GlobalId: float(wall_axis_world(w)[0][2]) * self.sc for w in self.walls}
        self.spaces = [s for s in f.by_type("IfcSpace") if getattr(s, "Representation", None)]
        self.poly = {}      # outer profile polygon (metres) of void-free spaces: used where a polygon is written into a prompt
        self._fpc = {}      # shapely footprint with holes from the mesh (lazy): used for containment / area / representative points
        for s in self.spaces:
            if self._has_voids(s):      # the outer ring of a profile with voids is not the room (centroid may be in the hole)
                continue
            P = profile_world_polygon(s)
            if P:
                self.poly[s.GlobalId] = np.array(P)[:, :2] * self.sc
        self._bbox = {}; self._fc = {}; self._sides = {}     # before self.slabs: _flat_floor -> height -> bbox needs the cache (
        # with the cache created after, every slab raised inside _flat_floor's try and was dropped, so v7 had no slab update/delete task)
        self.columns = list(f.by_type("IfcColumn")); self.slabs = [sl for sl in f.by_type("IfcSlab") if self._flat_floor(sl)]
        self.conn = {w.GlobalId: {o.GlobalId for o in _connected_walls(w)} for w in self.walls}
        self.bounded = {w.GlobalId: {s.GlobalId for s in _spaces_bounded(w)} for w in self.walls}
        self.space_walls = {}
        for w, ss in self.bounded.items():
            for s in ss:
                self.space_walls.setdefault(s, set()).add(w)
        wall_ids = {w.GlobalId for w in self.walls}
        self.aligned = {w.GlobalId: (abs(self.udir(w)[0]) > 0.98 or abs(self.udir(w)[1]) > 0.98) for w in self.walls}
        def usable(d):   # centre / width come from the mesh projected on the wall axis, so oblique walls are fine too
            h = self.host(d)
            return h is not None and h[1].GlobalId in wall_ids and (self.bbox(d) is not None or self.bbox(h[0]) is not None)
        self.doors = [d for d in f.by_type("IfcDoor") if usable(d)]; self.windows = [d for d in f.by_type("IfcWindow") if usable(d)]

    @staticmethod
    def _has_voids(e) -> bool:
        try:
            for r in e.Representation.Representations:
                for it in r.Items:
                    if it.is_a("IfcExtrudedAreaSolid") and it.SweptArea.is_a("IfcArbitraryProfileDefWithVoids"):
                        return True
        except Exception:  # noqa: BLE001
            return True
        return False

    def _flat_floor(self, sl) -> bool:
        """A flat floor slab: FLOOR/BASESLAB/undefined type (no ROOF, no LANDING), one vertical extrusion whose depth equals the
        mesh z-extent (sloped roofs and clipped slabs are excluded)."""
        if (sl.PredefinedType or "NOTDEFINED") not in ("FLOOR", "BASESLAB", "NOTDEFINED", "USERDEFINED"):
            return False
        try:
            items = [it for r in sl.Representation.Representations if r.RepresentationIdentifier == "Body" for it in r.Items]
            if len(items) != 1 or not items[0].is_a("IfcExtrudedAreaSolid") or abs(abs(items[0].ExtrudedDirection.DirectionRatios[2]) - 1) > 1e-3:
                return False
            h = self.height(sl)
            return h is not None and abs(h - float(items[0].Depth) * self.sc) <= 0.02
        except Exception:  # noqa: BLE001
            return False

    # ---- footprints (holes included)
    def fp(self, sg):
        if sg not in self._fpc:
            try:
                self._fpc[sg] = _footprint(self.f.by_guid(sg))
            except Exception:  # noqa: BLE001
                self._fpc[sg] = None
        return self._fpc[sg]

    def ring(self, sg):
        """Outer ring of the footprint as an (n, 2) array: the profile polygon when there is one, else the mesh footprint's exterior."""
        P = self.poly.get(sg)
        if P is not None:
            return P
        fp = self.fp(sg)
        return None if fp is None else np.array(fp.exterior.coords)[:-1, :2]

    def contains(self, sg, pt, margin: float = 0.0) -> bool:
        """Point inside the space footprint, at least `margin` away from its boundary (negative margin = within that distance outside)."""
        fp = self.fp(sg)
        if fp is not None:
            return fp.buffer(-margin).contains(Point(float(pt[0]), float(pt[1])))
        P = self.poly.get(sg)
        if P is None:
            return False
        return (_inside(pt, P) and _dist_margin(pt, P) >= margin) if margin >= 0 else (_inside(pt, P) or _dist_margin(pt, P) <= -margin)

    def area(self, sg):
        fp = self.fp(sg)
        if fp is not None:
            return float(fp.area)
        P = self.poly.get(sg); return None if P is None else abs(_poly_area(P))

    def rep_point(self, sg, margin: float = 0.3):
        """A point well inside the space: the centroid when it is inside with the margin, else shapely's representative point."""
        fp = self.fp(sg)
        if fp is None:
            c = self.centroid(self.f.by_guid(sg)); P = self.poly.get(sg)
            return c if (c is not None and P is not None and _inside(c, P) and _dist_margin(c, P) >= margin) else None
        c = np.array(fp.centroid.coords[0])
        if fp.buffer(-margin).contains(Point(c)):
            return c
        inner = fp.buffer(-margin)
        return np.array(inner.representative_point().coords[0]) if not inner.is_empty else None

    # ---- geometry
    def bbox(self, e):
        if e.GlobalId not in self._bbox:
            bb = _mesh_bbox(e); self._bbox[e.GlobalId] = None if bb is None else (np.array(bb["min"]) * self.sc, np.array(bb["max"]) * self.sc)
        return self._bbox[e.GlobalId]

    def storey_of(self, e):
        try:
            c = ue.get_container(e)
            if c is not None and c.is_a("IfcBuildingStorey"):
                return c
            a = ue.get_aggregate(e)
            return a if a is not None and a.is_a("IfcBuildingStorey") else None
        except Exception:  # noqa: BLE001
            return None

    def storey_ref(self, s):
        return f"the storey named '{s.Name}' (GlobalId {s.GlobalId})"

    def length(self, w):
        a, b = self.axis[w.GlobalId]; return float(np.linalg.norm(b - a))

    def udir(self, w):
        a, b = self.axis[w.GlobalId]; u = b - a; return u / (np.linalg.norm(u) or 1.0)

    def host(self, fill):
        for r in getattr(fill, "FillsVoids", []) or []:
            op = r.RelatingOpeningElement
            for v in op.VoidsElements or []:
                return op, v.RelatingBuildingElement
        return None

    def fill_centre(self, d):
        """(centre xy, width) in metres from the door/window MESH (placement conventions differ between exporters: Revit puts the
        opening origin at the centre, our builder at the left edge); width = mesh extent along the host wall's axis."""
        if d.GlobalId in self._fc:
            return self._fc[d.GlobalId]
        op, w = self.host(d); v = _mesh(d)
        if v is None:
            v = _mesh(op)
        if v is None:
            raise ValueError(f"no geometry for {d.GlobalId}")
        u = self.udir(w); pu = v[:, :2] @ u
        self._fc[d.GlobalId] = (v[:, :2].mean(0), float(pu.max() - pu.min()))
        return self._fc[d.GlobalId]

    def col_centre(self, c):
        bb = self.bbox(c)
        return (bb[0][:2] + bb[1][:2]) / 2 if bb is not None else world_matrix(c)[:3, 3][:2] * self.sc

    def thick(self, w) -> float:
        """Wall thickness in metres: the extrusion profile, else (Brep / faceset bodies) the mesh extent across the wall axis."""
        if not hasattr(self, "_thick"):
            self._thick = {}
        if w.GlobalId not in self._thick:
            t = _wall_thickness(w) * self.sc
            if t <= 0:
                v = _mesh(w)
                if v is not None:
                    u = self.udir(w); n = np.array([-u[1], u[0]]); pn = np.asarray(v)[:, :2] @ n; t = float(pn.max() - pn.min())   # checkers.mesh is in metres
            self._thick[w.GlobalId] = t
        return self._thick[w.GlobalId]

    def is_external(self, w) -> bool:
        """Pset_WallCommon.IsExternal when present (prompts must not contradict the file), else geometric: one side of the wall
        inside the storey outline (union of space footprints), the other outside (ifcgym.geomext; replaced
        'bounds exactly one space', which misses facade walls bounding several rooms)."""
        try:
            v = (ue.get_psets(w).get("Pset_WallCommon") or {}).get("IsExternal")
        except Exception:  # noqa: BLE001
            v = None
        if v is not None:
            return bool(v)
        from ifcgym.geomext import storey_outline, wall_is_exterior
        st = self.storey_of(w); key = st.GlobalId if st is not None else None
        if not hasattr(self, "_outline"):
            self._outline = {}
        if key not in self._outline:
            ws = [o for o in self.walls if self.storey_of(o) == st]
            self._outline[key] = storey_outline([self.fp(s.GlobalId) for s in self.spaces if self.storey_of(s) == st],
                                                max([self.thick(o) for o in ws] or [0.2]),
                                                [(*self.axis[o.GlobalId], self.thick(o)) for o in ws])
        a, b = self.axis[w.GlobalId]
        return wall_is_exterior(a, b, self.thick(w), self._outline[key])

    def centroid(self, s):
        P = self.poly.get(s.GlobalId)
        if P is None:
            return None
        A = _poly_area(P)
        if abs(A) < 1e-6:
            return P.mean(0)
        cx = np.sum((P[:, 0] + np.roll(P[:, 0], -1)) * (P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])) / (6 * A)
        cy = np.sum((P[:, 1] + np.roll(P[:, 1], -1)) * (P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])) / (6 * A)
        return np.array([cx, cy])

    def height(self, e):
        bb = self.bbox(e); return None if bb is None else float(bb[1][2] - bb[0][2])

    def neighbour(self, w, d: str):
        """Nearest parallel wall in direction d whose extent overlaps w by > 50 percent, same level; None if absent or tied."""
        a, b = self.axis[w.GlobalId]; u = self.udir(w); n = DIRS[d]
        if abs(float(u @ n)) > 0.05 or np.linalg.norm(b - a) < 2.0:   # short anchors make "half of its length" fragile
            return None
        lo, hi = sorted([float(a @ u), float(b @ u)]); base = float(a @ n); best = None; partial = None
        st = self.storey_of(w)
        for o in self.walls:
            if o.GlobalId == w.GlobalId:
                continue
            if st is not None:          # "on the same storey" = same IfcBuildingStorey container, whatever the wall's own elevation
                if self.storey_of(o) != st:
                    continue
            elif abs(self.z[o.GlobalId] - self.z[w.GlobalId]) > 0.5:
                continue
            oa, ob = self.axis[o.GlobalId]; ou = self.udir(o)
            if abs(float(ou @ u)) < 0.95:
                continue
            olo, ohi = sorted([float(oa @ u), float(ob @ u)]); ov = min(hi, ohi) - max(lo, olo)
            dd = float(oa @ n) - base
            if ov <= -0.3 or dd <= 0.05:
                continue
            if ov <= 0.6 * (hi - lo) or ov <= 0.6 * (ohi - olo):   # touching / partially overlapping walls (bodies overlap by half a thickness
                partial = min(partial, dd) if partial is not None else dd   # even at axis overlap 0) make a closer candidate arguable: block
                continue
            if best is None or dd < best[0] - 1e-6:
                best = (dd, o, 1)
            elif abs(dd - best[0]) <= 1e-6:
                best = (dd, o, best[2] + 1)
        if best and partial is not None and partial < best[0] + 1e-6:
            return None
        return best[1] if best and best[2] == 1 else None

    def above(self, w):
        """The wall on the next storey with the same axis (stacked plans), None if absent or ambiguous."""
        a, b = self.axis[w.GlobalId]; z = self.z[w.GlobalId]; cands = []
        for o in self.walls:
            if o.GlobalId == w.GlobalId or not (0.5 < self.z[o.GlobalId] - z < 6.0):
                continue
            oa, ob = self.axis[o.GlobalId]
            if min(np.linalg.norm(oa - a) + np.linalg.norm(ob - b), np.linalg.norm(oa - b) + np.linalg.norm(ob - a)) < 0.05:
                cands.append(o)
        if not cands:
            return None
        zmin = min(self.z[o.GlobalId] for o in cands); cands = [o for o in cands if abs(self.z[o.GlobalId] - zmin) < 0.05]
        return cands[0] if len(cands) == 1 else None

    # ---- references: (phrase, style, meta) or None
    def ref_wall(self, w, styles=("D", "T", "S", "P", "V")):
        opts = []
        g = w.GlobalId
        if "D" in styles:
            opts.append((f"the wall with GlobalId {g}", "D", {}))
        if "T" in styles:
            cw = sorted(self.conn[g])
            if len(cw) >= 2:
                for _ in range(4):
                    A, B = self.rng.sample(cw, 2)
                    if len(self.conn.get(A, set()) & self.conn.get(B, set())) == 1:
                        opts.append((f"the wall that is connected to both the wall with GlobalId {A} and the wall with GlobalId {B}", "T", {"anchors": [A, B]})); break
            sp = sorted(self.bounded[g])
            if len(sp) == 2 and len(self.space_walls[sp[0]] & self.space_walls[sp[1]]) == 1:
                opts.append((f"the wall that separates the space with GlobalId {sp[0]} from the space with GlobalId {sp[1]}", "T", {"anchors": sp}))
        if "S" in styles:
            for d in self.rng.sample(list(DIRS), 4):
                o = self.neighbour(w, d); opp = {"+x": "-x", "-x": "+x", "+y": "-y", "-y": "+y"}[d]
                if o is not None and getattr(self.neighbour(o, opp), 'GlobalId', None) == w.GlobalId:   # o lies in direction d of w, so w is the wall in direction opp of o
                    opts.append((f"the wall directly {DIR_WORD[opp]} of the wall with GlobalId {o.GlobalId} (the nearest parallel wall in that direction on the same storey; the two walls' axes overlap along at least half of each wall's length)", "S", {"anchors": [o.GlobalId]})); break
            below = [o for o in self.walls if self.above(o) is not None and self.above(o).GlobalId == w.GlobalId]
            if len(below) == 1:
                opts.append((f"the wall directly above the wall with GlobalId {below[0].GlobalId}, on the storey above it", "S", {"anchors": [below[0].GlobalId]}))
            ab = self.above(w)
            if ab is not None and [o.GlobalId for o in self.walls if self.above(o) is not None and self.above(o).GlobalId == ab.GlobalId] == [w.GlobalId]:
                opts.append((f"the wall directly below the wall with GlobalId {ab.GlobalId}, on the storey below it (same axis end points in XY)", "S", {"anchors": [ab.GlobalId]}))
        if "P" in styles:
            opts += self.path_wall(w)
        if "V" in styles:
            opts += self.view_wall(w)
        return self._pick(opts)

    def ref_fill(self, d, kind, styles=("D", "T", "S", "P")):
        opts = []; g = d.GlobalId; op, w = self.host(d); cls = "IfcDoor" if kind == "door" else "IfcWindow"
        same = [x for o in _openings(w) for x in _fills(o) if x.is_a(cls)]
        if "D" in styles:
            opts.append((f"the {kind} with GlobalId {g}", "D", {}))
        if "T" in styles and len(same) == 1:
            opts.append((f"the {kind} hosted by the wall with GlobalId {w.GlobalId}", "T", {"anchors": [w.GlobalId]}))
            sp = sorted(self.bounded[w.GlobalId])
            sides = self.door_sides(d) if kind == "door" else None
            if sides is not None and len(self.space_walls[sides[0]] & self.space_walls[sides[1]]) == 1 and sum(1 for x in same if self.door_sides(x) and set(self.door_sides(x)) == set(sides)) == 1:
                a_, b_ = sorted(sides)
                opts.append((f"the door connecting the space with GlobalId {a_} and the space with GlobalId {b_} (the door in their common wall that opens between the two)", "T", {"anchors": [a_, b_]}))
        if "S" in styles:
            c, _ = self.fill_centre(d); a, b = self.axis[w.GlobalId]
            for end_pt in (a, b):
                others = [o for o in self.walls if o.GlobalId in self.conn[w.GlobalId] and min(np.linalg.norm(self.axis[o.GlobalId][0] - end_pt), np.linalg.norm(self.axis[o.GlobalId][1] - end_pt)) < 0.4]
                if not others:
                    continue
                o = others[0]
                # candidates = every fill of that class on the storey; corner = the
                # intersection of the two walls' axis lines; the prompt states both
                ua, ub = self.udir(w), self.udir(o); A_ = np.array([ua, -ub]).T
                if abs(float(np.linalg.det(A_))) < 1e-6:
                    continue
                s_ = np.linalg.solve(A_, self.axis[o.GlobalId][0] - a); corner = a + ua * float(s_[0]); st_ = self.storey_of(d)
                cands = [x for x in (self.doors if cls == "IfcDoor" else self.windows) if self.storey_of(x) == st_]
                ds = sorted((float(np.linalg.norm(self.fill_centre(x)[0] - corner)), x.GlobalId) for x in cands)
                if ds and ds[0][1] == g and (len(ds) == 1 or ds[1][0] - ds[0][0] > 0.3):
                    opts.append((f"the {kind} closest to the corner where the wall with GlobalId {w.GlobalId} meets the wall with GlobalId {o.GlobalId} (among all {kind}s on that storey; distance in plan from the {kind}'s centre to the intersection point of the two walls' axis lines)", "S", {"anchors": [w.GlobalId, o.GlobalId]})); break
            st = self.storey_of(d)
            if st is not None:
                pool = [x for x in (self.doors if kind == "door" else self.windows) if self.storey_of(x) == st]
                for d_, name in (("-x", "westernmost"), ("+x", "easternmost"), ("+y", "northernmost"), ("-y", "southernmost")):
                    ds = sorted((-float(self.fill_centre(x)[0] @ DIRS[d_]), x.GlobalId) for x in pool)
                    if ds and ds[0][1] == g and (len(ds) == 1 or ds[1][0] - ds[0][0] > 0.3):
                        opts.append((f"the {name} {kind} on {self.storey_ref(st)} (+X is east, +Y is north)", "S", {"anchors": [st.GlobalId]})); break
            # stacked storeys: the door/window straight above / below (same XY centre, next storey up / down), unique both ways
            c0, _ = self.fill_centre(d); bz0 = self.bbox(d)
            if bz0 is not None:
                z0 = float(bz0[0][2]); pool_all = self.doors if kind == "door" else self.windows
                col = [x for x in pool_all if self.bbox(x) is not None and np.linalg.norm(self.fill_centre(x)[0] - c0) <= 0.05]   # the vertical stack through d
                zs = sorted((float(self.bbox(x)[0][2]), x.GlobalId) for x in col); i = [k_ for k_, (z_, g_) in enumerate(zs) if g_ == g][0]
                if i >= 1 and 0.5 < z0 - zs[i - 1][0] < 6.0:
                    opts.append((f"the {kind} directly above the {kind} with GlobalId {zs[i - 1][1]}, on the storey above it (same XY centre)", "S", {"anchors": [zs[i - 1][1]]}))
                if i + 1 < len(zs) and 0.5 < zs[i + 1][0] - z0 < 6.0:
                    opts.append((f"the {kind} directly below the {kind} with GlobalId {zs[i + 1][1]}, on the storey below it (same XY centre)", "S", {"anchors": [zs[i + 1][1]]}))
            if len(same) >= 2:   # BIM-Edit: "the most central window of wall W"
                a, b = self.axis[w.GlobalId]; mid = (a + b) / 2
                ds = sorted((float(np.linalg.norm(self.fill_centre(x)[0] - mid)), x.GlobalId) for x in same)
                if ds[0][1] == g and ds[1][0] - ds[0][0] > 0.3:
                    opts.append((f"the most central {kind} of the wall with GlobalId {w.GlobalId} (the one closest to the wall's mid-point)", "S", {"anchors": [w.GlobalId]}))
        if "P" in styles:
            opts += (self.path_item(d, kind, None) or []) + (self.leave_item(d, kind) or [])   # both P phrasings compete in _pick
        return self._pick(opts)

    def ref_space(self, s, styles=("D", "T", "S")):
        opts = []; g = s.GlobalId
        if "D" in styles:
            opts.append((f"the space with GlobalId {g}", "D", {}))
        ws = sorted(self.space_walls.get(g, ()))
        if "T" in styles and 3 <= len(ws) <= 8 and not any(set(ws) <= self.space_walls.get(o, set()) for o in self.space_walls if o != g):
            opts.append((f"the space bounded by exactly the walls with GlobalIds {', '.join(ws[:-1])} and {ws[-1]}", "T", {"anchors": ws}))
        if "S" in styles:
            c = self.rep_point(g); st = self.storey_of(s)
            if c is not None and st is not None:
                pt = (round(float(c[0]), 2), round(float(c[1]), 2))
                hits = [o for o in self.spaces if self.storey_of(o) == st and self.contains(o.GlobalId, np.array(pt), -0.0)]
                if hits == [s]:
                    opts.append((f"the space on {self.storey_ref(st)} that contains the point ({pt[0]}, {pt[1]}) (metres, project XY)", "S", {"anchors": [st.GlobalId]}))
        return self._pick(opts)

    def ref_column(self, c, styles=("D", "T", "S", "P")):
        opts = []; g = c.GlobalId; ctr = self.col_centre(c); st = self.storey_of(c)
        if "D" in styles:
            opts.append((f"the column with GlobalId {g}", "D", {}))
        if "T" in styles and st is not None:
            room = [s for s in self.spaces if self.storey_of(s) == st and self.contains(s.GlobalId, ctr)]
            if len(room) == 1 and sum(1 for o in self.columns if self.storey_of(o) == st and self.contains(room[0].GlobalId, self.col_centre(o))) == 1:
                opts.append((f"the column standing inside the space with GlobalId {room[0].GlobalId}", "T", {"anchors": [room[0].GlobalId]}))
        if "S" in styles and st is not None:
            def seg_d(p, w):
                a, b = self.axis[w.GlobalId]; ab = b - a; t = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) or 1e-12), 0, 1); return float(np.linalg.norm(p - (a + t * ab)))
            ws = [w for w in self.walls if abs(self.z[w.GlobalId] - float(st.Elevation or 0) * self.sc) < 0.5]
            if ws:
                w = min(ws, key=lambda w: seg_d(ctr, w))
                ds = sorted((seg_d(self.col_centre(o), w), o.GlobalId) for o in self.columns if self.storey_of(o) == st)
                if ds[0][1] == g and (len(ds) == 1 or ds[1][0] - ds[0][0] > 0.3):
                    opts.append((f"the column on {self.storey_ref(st)} closest to the wall with GlobalId {w.GlobalId} (distance in plan from the column centre to the wall axis)", "S", {"anchors": [w.GlobalId, st.GlobalId]}))
        if "P" in styles:
            opts += self.path_item(c, "column", None)
        return self._pick(opts)

    def ref_slab(self, sl, styles=("D", "T", "S")):
        opts = []; g = sl.GlobalId; st = self.storey_of(sl)
        if "D" in styles:
            opts.append((f"the slab with GlobalId {g}", "D", {}))
        if "T" in styles and st is not None and sum(1 for o in self.slabs if self.storey_of(o) == st) == 1:
            opts.append((f"the floor slab (IfcSlab, not a roof or landing) contained in {self.storey_ref(st)}", "T", {"anchors": [st.GlobalId]}))
        if "S" in styles:
            zs = sorted((float(self.bbox(o)[0][2]), o.GlobalId) for o in self.slabs if self.bbox(o) is not None)
            if zs and zs[0][1] == g and (len(zs) == 1 or zs[1][0] - zs[0][0] > 0.3):
                opts.append(("the lowest floor slab of the building (IfcSlab, flat floor slabs only: not roofs or landings)", "S", {}))
            elif zs and zs[-1][1] == g and (len(zs) == 1 or zs[-1][0] - zs[-2][0] > 0.3):
                opts.append(("the highest floor slab of the building (IfcSlab, flat floor slabs only: not roofs or landings)", "S", {}))
        return self._pick(opts)

    # ---- path references (BIM-Edit style: "after entering space B from space A through door D, the X on your right")
    def exit_door(self, d) -> bool:
        """A space just beyond one face of the door's wall and no space of the storey at all beyond the other (0.15 / 0.3 / 0.5 m)."""
        op, w = self.host(d); u = self.udir(w); n = np.array([-u[1], u[0]]); c, _ = self.fill_centre(d); a0, _b0 = self.axis[w.GlobalId]
        base = a0 + u * float((c - a0) @ u); half = self.thick(w) / 2; st = self.storey_of(d)
        geoms = [g_ for s in self.spaces if self.storey_of(s) == st and (g_ := self._side_geom(s.GlobalId)) is not None]
        hit = {sd: any(g_.contains(Point(base + sd * n * (half + gap))) for gap in (0.15, 0.3, 0.5) for g_ in geoms) for sd in (1, -1)}
        return hit[1] != hit[-1]

    def _side_geom(self, sg):
        """Footprint polygon of a space for side tests: the mesh footprint, else the outer ring of its profile."""
        fp = self.fp(sg)
        if fp is not None:
            return fp
        P = self.ring(sg)
        if P is None or len(P) < 3:
            return None
        from shapely.geometry import Polygon as _Poly
        return _Poly(P).buffer(0)

    def door_sides(self, d):
        """The two spaces a door really connects: among the spaces bounded by its wall, the one whose footprint contains a point just
        beyond each face of the wall at the door (merged walls bound several rooms; entrance doors have one side only).
        Returns (space_a, space_b) GlobalIds (a on the -n side, b on the +n side) or None."""
        if d.GlobalId in self._sides:
            return self._sides[d.GlobalId]
        # 'nearest reference point on each side' picked a large L-shaped room whose inside point lay near the wall
        # (wrong room) and dropped doors whose two rooms' inside points fell on the same side. Now: the space whose footprint contains
        # a probe point just beyond each face of the wall at the door centre (0.15 / 0.3 / 0.5 m, first distance that hits).
        op, w = self.host(d); u = self.udir(w); n = np.array([-u[1], u[0]]); c, width = self.fill_centre(d)
        a0, _b0 = self.axis[w.GlobalId]; base = a0 + u * float((c - a0) @ u); half = self.thick(w) / 2; best = {}
        for side in (1, -1):
            hit = []
            for gap in (0.15, 0.3, 0.5):
                pt = Point(base + side * n * (half + gap))
                hit = [sg for sg in self.bounded[w.GlobalId] if (g_ := self._side_geom(sg)) is not None and g_.contains(pt)]
                if hit:
                    break
            best[side] = hit[0] if len(hit) == 1 else None
        res = (best[-1], best[1]) if best[1] and best[-1] else None
        self._sides[d.GlobalId] = res
        return res

    def entries(self):
        """(door, from_space, to_space, e, r, door_centre) for every door that really lies between two spaces (one on each side
        of its wall at the door's position): e = walking direction (unit, into to_space), r = right-hand direction (e rotated
        clockwise)."""
        if hasattr(self, "_entries"):
            return self._entries
        out = []
        for d in self.doors:
            sides = self.door_sides(d)
            if sides is None:
                continue
            op, w = self.host(d); u = self.udir(w); n = np.array([-u[1], u[0]]); c, _ = self.fill_centre(d)
            for a, b in ((sides[0], sides[1]), (sides[1], sides[0])):
                cb = self.rep_point(b, 0.0)
                if cb is None:
                    continue
                e = n if float((cb - c) @ n) > 0 else -n; r = np.array([e[1], -e[0]])
                out.append((d, a, b, e, r, c))
        self._entries = out
        return out

    def in_space(self, pt, sg, tol=0.3) -> bool:
        return self.contains(sg, pt, -tol)

    def edge_on_wall(self, a, b, walls) -> bool:
        """A footprint edge a-b (metres) lies on a face of one of the walls (parallel, at half thickness, covered along the axis)."""
        for wg in walls:
            w = self.f.by_guid(wg); u = self.udir(w); n = np.array([-u[1], u[0]]); q0, q1 = self.axis[wg]; t2 = _wall_thickness(w) * self.sc / 2
            if abs(float((b - a) @ n)) > 0.02 * max(1.0, np.linalg.norm(b - a)):
                continue
            if abs(abs(float((a - q0) @ n)) - t2) > 0.05:
                continue
            lo, hi = sorted([0.0, float((q1 - q0) @ u)]); ta, tb = sorted([float((a - q0) @ u), float((b - q0) @ u)])
            if ta >= lo - 0.1 and tb <= hi + 0.1:
                return True
        return False

    @staticmethod
    def side_def(side):
        """Define only the side the reference uses ('farthest to your left ... your right = clockwise' made the
        paraphraser write 'right' in 60-70 % of left prompts)."""
        return {"left": "your left = that direction turned counter-clockwise, seen from above", "right": "your right = that direction turned clockwise, seen from above"}.get(side, "you face along that direction")

    def path_phrase(self, d, a, b, side=None):
        return f"after entering the space with GlobalId {b} from the space with GlobalId {a} through the door with GlobalId {d.GlobalId} (walking direction = perpendicular to that door's wall, into the entered space; {self.side_def(side)})"

    def view_wall(self, w):
        """'the wall facing you when looking through window W into the building': from the window centre, along the inward normal of
        its wall, the first wall footprint hit on the same storey (excluding the host wall); unique when the next hit is > 0.5 m farther."""
        opts = []
        for win in self.windows:
            op, host = self.host(win)
            if self.storey_of(win) != self.storey_of(w) or host.GlobalId == w.GlobalId:
                continue
            c, _ = self.fill_centre(win); u = self.udir(host); n = np.array([-u[1], u[0]])
            inside = [sg for sg in self.bounded[host.GlobalId] if self.fp(sg) is not None]
            if not self.is_external(host):      # 'into the building' needs an outside: interior walls are ambiguous
                continue
            # inward = towards the bounded spaces (all on one side, else skip), or towards the storey centre for walls without boundaries
            if inside:
                qs = [self.rep_point(sg, 0.0) for sg in inside]
                if any(q is None for q in qs):
                    continue
                sides = {float((q - c) @ n) > 0 for q in qs}
                if len(sides) != 1:
                    continue
                e = n if sides.pop() else -n
            elif True:      # no bounded space (sparse export): 'into the building' is not checkable from relations - skip (gate 20)
                continue
            else:
                st = self.storey_of(host); cen = np.mean([self.col_centre(x) for x in self.columns if self.storey_of(x) == st] or [np.array(self.axis[x.GlobalId][0]) for x in self.walls if self.storey_of(x) == st], axis=0)
                e = n if float((cen - c) @ n) > 0 else -n
            hits = []
            for o in self.walls:
                if o.GlobalId == host.GlobalId or self.storey_of(o) != self.storey_of(host):
                    continue
                a, b = self.axis[o.GlobalId]; t = max(self.thick(o), 0.1)
                # ray c + s*e, s > 0, against the wall's face strip: solve for intersection with the axis line, then check extent and thickness
                d = b - a; den = float(e[0] * (-d[1]) + e[1] * d[0])
                if abs(den) < 1e-9:
                    continue
                s_ = float(((a - c)[0] * (-d[1]) + (a - c)[1] * d[0]) / den); tt = float(((a - c)[0] * (-e[1]) + (a - c)[1] * e[0]) / den) if False else None
                p_ = c + e * s_
                if s_ <= 0.3:
                    continue
                along = float((p_ - a) @ (d / (np.linalg.norm(d) or 1.0)))
                L_ = float(np.linalg.norm(d))
                if -t / 2 - 0.3 <= along <= L_ + t / 2 + 0.3:      # anything that may block the sight counts (corner extensions included)
                    hits.append((s_ - t / 2, o.GlobalId, 0.3 <= along <= L_ - 0.3))
            hits.sort()
            # the first wall must be hit clearly inside its length (supplement gate: rays grazing wall ends disagreed with the verifier)
            if hits and hits[0][1] == w.GlobalId and hits[0][2] and (len(hits) == 1 or hits[1][0] - hits[0][0] > 0.5):
                opts.append((f"the wall facing you when you look through the window with GlobalId {win.GlobalId} into the building (the first wall met by the horizontal line of sight perpendicular to that window's wall, pointing inwards)", "V", {"anchors": [win.GlobalId]}))
                break
        return opts

    def path_wall(self, w):
        """Phrase for wall w as 'the wall on your right/left' or 'the wall facing you' on some entry into a space it bounds."""
        opts = []
        for d, a, b, e, r, c in self.entries():
            if b not in self.bounded[w.GlobalId]:
                continue
            ws = [self.f.by_guid(g) for g in self.space_walls[b]]
            par = [x for x in ws if abs(float(self.udir(x) @ e)) > 0.9]; perp = [x for x in ws if abs(float(self.udir(x) @ e)) < 0.1]
            def off(x, v):
                aa, bb = self.axis[x.GlobalId]; return float(((aa + bb) / 2 - c) @ v)
            for side, v in (("right", r), ("left", -r)):
                cand = [x for x in par if off(x, v) > 0.3]
                if len(cand) == 1 and cand[0].GlobalId == w.GlobalId:
                    opts.append((f"the wall on your {side}-hand side {self.path_phrase(d, a, b, side)} (the wall of that space running parallel to your walking direction on your {side})", "P", {"anchors": [a, b, d.GlobalId]}))
            ahead = [x for x in perp if off(x, e) > 0.3]
            if len(ahead) == 1 and ahead[0].GlobalId == w.GlobalId:
                opts.append((f"the wall facing you {self.path_phrase(d, a, b)} (the wall of that space in front of you, perpendicular to your walking direction)", "P", {"anchors": [a, b, d.GlobalId]}))
            if opts:
                break
        return opts

    def leave_item(self, item, cls_word):
        """'after leaving space A through door D to the outside, the window/door farthest to your right': D is hosted by an exterior wall
        with one bounded space; walking direction = outward normal; candidates = fills in exterior walls of that storey within 8 m."""
        opts = []
        if cls_word == "column":
            return opts
        cls = "IfcDoor" if cls_word == "door" else "IfcWindow"
        for d in self.doors:
            op, w = self.host(d)
            if d.GlobalId == item.GlobalId or not self.is_external(w) or self.storey_of(d) != self.storey_of(item):
                continue
            # the room left = the one space found just beyond one face of the wall at the door; nothing beyond the other face
            c, _ = self.fill_centre(d); u = self.udir(w); n = np.array([-u[1], u[0]]); a0, _b0 = self.axis[w.GlobalId]
            base = a0 + u * float((c - a0) @ u); off = self.thick(w) / 2 + 0.3
            near = {sd: [sg for sg in self.bounded[w.GlobalId] if self.fp(sg) is not None and self.fp(sg).contains(Point(base + sd * n * off))] for sd in (1, -1)}
            sides = [sd for sd in (1, -1) if near[sd]]
            if len(sides) != 1 or len(near[sides[0]]) != 1:
                continue
            inside = near[sides[0]]; e = -sides[0] * n          # outward
            r = np.array([e[1], -e[0]]); st = self.storey_of(d)
            pool = [x for x in (self.doors if cls == "IfcDoor" else self.windows) if x.GlobalId != d.GlobalId and self.storey_of(x) == st and self.is_external(self.host(x)[1]) and np.linalg.norm(self.fill_centre(x)[0] - c) <= 8.0]
            if item.GlobalId not in {x.GlobalId for x in pool}:
                continue
            for side, v in (("right", r), ("left", -r)):
                ds = sorted((float((self.fill_centre(x)[0] - c) @ v), x.GlobalId) for x in pool)
                if ds[-1][1] == item.GlobalId and ds[-1][0] > 0.3 and (len(ds) == 1 or ds[-1][0] - ds[-2][0] > 0.3):
                    opts.append((f"the {cls_word} farthest to your {side} after leaving the space with GlobalId {inside[0]} through the door with GlobalId {d.GlobalId} to the outside (walking direction = perpendicular to that door's wall, outwards; {self.side_def(side)}; among the {cls_word}s in the exterior walls of that storey within 8 m of the door)", "P", {"anchors": [inside[0], d.GlobalId]}))
                    return opts
        return opts

    def path_item(self, item, cls_word, centre):
        """Phrase for a door/window/column inside/bounding space b as 'the <cls> on your right/left' (largest lateral offset, unique)."""
        opts = []
        for d, a, b, e, r, c in self.entries():
            if cls_word == "column":
                stb = self.storey_of(self.f.by_guid(b))
                if stb is None or self.storey_of(item) != stb:   # stacked plans repeat the footprint on every storey: stay on B's storey
                    continue
                pool = [x for x in self.columns if self.storey_of(x) == stb and self.contains(b, self.col_centre(x))]
                cen = {x.GlobalId: self.col_centre(x) for x in pool}
            else:
                cls = "IfcDoor" if cls_word == "door" else "IfcWindow"
                ok_ids = {y.GlobalId for y in (self.doors + self.windows)}
                pool = [x for g in self.space_walls[b] for o in _openings(self.f.by_guid(g)) for x in _fills(o) if x.is_a(cls) and x.GlobalId != d.GlobalId and x.GlobalId in ok_ids and self.in_space(self.fill_centre(x)[0], b)]
                cen = {x.GlobalId: self.fill_centre(x)[0] for x in pool}
            if item.GlobalId not in cen or len(pool) < 1:
                continue
            for side, v in (("right", r), ("left", -r)):
                ds = sorted(((float((cen[x.GlobalId] - c) @ v)), x.GlobalId) for x in pool)
                if ds[-1][1] == item.GlobalId and ds[-1][0] > 0.3 and (len(ds) == 1 or ds[-1][0] - ds[-2][0] > 0.3):
                    opts.append((f"the {cls_word} farthest to your {side} {self.path_phrase(d, a, b, side)} (among the {cls_word}s " + ("standing inside that space" if cls_word == "column" else "in the walls of that space and lying within its footprint") + f", the one with the largest offset to your {side})", "P", {"anchors": [a, b, d.GlobalId]}))
            if opts:
                break
        return opts

    def _pick(self, opts):
        if not opts:
            return None
        by = {}
        for o in opts:
            by.setdefault(o[1], []).append(o)
        style = self.rng.choices(list(by), weights=[{"D": 0.26, "T": 0.24, "S": 0.24, "P": 0.16, "V": 0.10}[k] for k in by])[0]
        return self.rng.choice(by[style])


def derive_input(src: str, dst: str, rng: random.Random) -> dict:
    """arc_v4.ifc: columns added inside rooms, 1-2 spaces removed (recorded for create-space tasks)."""
    f = ifcopenshell.open(src); c = Ctx(f, rng); sc = c.sc
    rec = {"columns": [], "removed_spaces": [], "derive_version": DERIVE_VERSION}
    rooms = [s for s in c.spaces if (c.area(s.GlobalId) or 0) >= 8.0 and c.storey_of(s) is not None and c.rep_point(s.GlobalId, 0.8) is not None]
    rng.shuffle(rooms)
    for s in rooms[: rng.randint(3, 6)]:
        ctr = c.rep_point(s.GlobalId, 0.8)
        for _ in range(10):
            p = ctr + np.array([rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)])
            if c.contains(s.GlobalId, p, 0.6):
                break
        else:
            continue
        h = c.height(s) or 2.8; st = c.storey_of(s)
        r = add_column(f, st, (np.array([round(float(p[0]), 2), round(float(p[1]), 2)]) / sc).tolist(), 0.3 / sc, h / sc)
        col = f.by_guid(r["guid"]); col.GlobalId = ifcopenshell.guid.compress(uuid.UUID(int=rng.getrandbits(128)).hex); r["guid"] = col.GlobalId   # reproducible ids
        rec["columns"].append({"guid": r["guid"], "xy": [round(float(p[0]), 2), round(float(p[1]), 2)], "space": s.GlobalId, "storey": st.GlobalId, "height": round(h, 3)})
    cands = [s for s in c.spaces if c.ring(s.GlobalId) is not None and 3 <= len(c.space_walls.get(s.GlobalId, ())) <= 8 and (c.area(s.GlobalId) or 0) >= 4.0
             and c.storey_of(s) is not None and s.GlobalId not in {col["space"] for col in rec["columns"]}]
    rng.shuffle(cands)
    for s in cands[: rng.randint(1, 2)]:
        st = c.storey_of(s); P = c.ring(s.GlobalId); h = c.height(s) or 2.8; ctr = c.rep_point(s.GlobalId, 0.3)
        rec["removed_spaces"].append({"guid": s.GlobalId, "polygon": [[round(float(x), 3), round(float(y), 3)] for x, y in P], "height": round(h, 2), "storey": st.GlobalId,
                                      "walls": sorted(c.space_walls[s.GlobalId]), "centre": [round(float(ctr[0]), 2), round(float(ctr[1]), 2)] if ctr is not None else None, "name": s.Name})
        delete(f, s)
    tmp = f"{dst}.{os.getpid()}.tmp"; f.write(tmp); os.replace(tmp, dst)
    return rec


def gen_v4(ifc_path: str, project: str, per_template: int, rng: random.Random, tid, families=("update", "delete", "create2", "retrieve2", "concept")) -> list[dict]:
    d = os.path.dirname(ifc_path); dst = os.path.join(d, "arc_v4.ifc"); recp = os.path.join(d, "arc_v4.json")
    stale = True
    if os.path.exists(dst) and os.path.exists(recp):
        try:
            stale = json.load(open(recp)).get("derive_version") != DERIVE_VERSION
        except Exception:  # noqa: BLE001
            stale = True
    if stale:
        rec = derive_input(ifc_path, dst, random.Random(f"{project}:derive"))
        tmp = f"{recp}.{os.getpid()}.tmp"; json.dump(rec, open(tmp, "w")); os.replace(tmp, recp)
    rec = json.load(open(recp))
    f = ifcopenshell.open(dst); c = Ctx(f, rng); sc = c.sc; k = max(1, per_template // 2)
    T: list[dict] = []
    import hashlib
    sha = hashlib.sha1(open(dst, "rb").read()).hexdigest()[:16]
    def add(tpl, prompt, target, allowed, intent, deps, meta=None, new_classes=None, tier=1):
        meta = dict(meta or {}); meta.setdefault("style", tpl[-1] if tpl[-2:] in ("-D", "-T", "-S", "-P", "-V") else "D"); meta["ifc_sha1"] = sha
        t = {"task_id": tid(tpl), "project": project, "template": tpl, "tier": tier, "prompt": prompt, "target": target, "allowed": allowed, "intent": intent, "dependents": deps, "meta": meta, "ifc": dst}
        if new_classes:
            t["allowed_new_classes"] = new_classes
        T.append(t)
    def sample(xs, n=k):
        xs = list(xs); rng.shuffle(xs); return xs[:n]
    def fmt(v):
        return [round(float(x), 3) + 0.0 for x in v]   # + 0.0 turns -0.0 into 0.0
    NC = "Do not change anything else."

    # ------------------------------------------------------------------ UPDATE
    if "update" in families:
        for kind, pool in (("door", c.doors), ("window", c.windows)):
            for dd in sample(pool, k * 2):
                r = c.ref_fill(dd, kind)
                if r is None:
                    continue
                op, w = c.host(dd); u = c.udir(w); L = c.length(w); ctr, width = c.fill_centre(dd); a, _ = c.axis[w.GlobalId]
                t0 = float((ctr - a) @ u); dist = rng.choice([0.3, 0.5]); sgn = rng.choice([1, -1])
                if not (0.2 < t0 + sgn * dist - width / 2 and t0 + sgn * dist + width / 2 < L - 0.2):
                    sgn = -sgn
                    if not (0.2 < t0 + sgn * dist - width / 2 and t0 + sgn * dist + width / 2 < L - 0.2):
                        continue
                vec = [round(float(x), 3) + 0.0 for x in (u * sgn * dist)] + [0.0]
                dws = [k_ for k_, v in DIRS.items() if float(v @ (u * sgn)) > 0.9]
                end_xy = c.axis[w.GlobalId][1] if sgn > 0 else c.axis[w.GlobalId][0]
                towards = f"towards {DIR_WORD[dws[0]]}" if dws else f"towards the wall's end at ({round(float(end_xy[0]), 2)}, {round(float(end_xy[1]), 2)})"
                add(f"U-move-{kind}-{r[1]}", f"Move {r[0]} by {dist} m along its wall {towards}, i.e. by the vector {vec} m in the project coordinate system. Its opening must move with it and the {kind} must stay hosted by the same wall. {NC}",
                    [dd.GlobalId], [dd.GlobalId, op.GlobalId], {"type": "translate", "guid": dd.GlobalId, "vector": vec, "tol": 0.02},
                    [{"guid": op.GlobalId, "rule": "translate_with", "vector": vec}, {"guid": op.GlobalId, "rule": "stays_hosted"}], {"style": r[1], **r[2]})
        for col in sample(c.columns, k):
            r = c.ref_column(col)
            if r is None:
                continue
            ctr = c.col_centre(col); st = c.storey_of(col)
            room = [s for s in c.spaces if c.storey_of(s) == st and c.contains(s.GlobalId, ctr)]
            for _ in range(6):
                dw = rng.choice(list(DIRS)); dist = rng.choice([0.5, 1.0]); p = ctr + DIRS[dw] * dist
                if room and c.contains(room[0].GlobalId, p, 0.4):
                    vec = fmt(DIRS[dw] * dist) + [0.0]
                    add(f"U-move-column-{r[1]}", f"Move {r[0]} by {dist} m towards {DIR_WORD[dw]} ({vec} m in the project coordinate system), keeping its size, height and storey. {NC}",
                        [col.GlobalId], [col.GlobalId], {"type": "translate", "guid": col.GlobalId, "vector": vec, "tol": 0.02}, [], {"style": r[1], **r[2]}); break
        for w in sample(c.walls, k):
            r = c.ref_wall(w)
            if r is None:
                continue
            t_now = _wall_thickness(w) * sc
            if t_now <= 0 and c.aligned[w.GlobalId] and c.bbox(w) is not None:   # Brep wall: thickness from the body (axis-aligned walls only)
                u = c.udir(w); bb = c.bbox(w); t_now = float((bb[1] - bb[0])[1] if abs(u[0]) > 0.98 else (bb[1] - bb[0])[0])
            if t_now <= 0.08 or c.length(w) < 1.0 or not axis_centred(c, w):    # no usable body; thin panels/stubs; axis on a face (ArchiCAD)
                continue
            t_new = rng.choice([t for t in (0.15, 0.2, 0.25, 0.3) if abs(t - t_now) > 0.03])
            ops = _openings(w); fl = [x for o in ops for x in _fills(o)]
            add(f"U-set-wall-thickness-{r[1]}", f"Change the thickness of {r[0]} to {t_new} m, keeping its axis line, length and height (the wall grows or shrinks symmetrically about its axis). Its openings and the doors/windows in them must stay hosted. {NC}",
                [w.GlobalId], [w.GlobalId] + [o.GlobalId for o in ops] + [x.GlobalId for x in fl], {"type": "set_thickness", "guid": w.GlobalId, "thickness": t_new, "tol": 0.02},
                [{"guid": o.GlobalId, "rule": "stays_hosted"} for o in ops], {"style": r[1], "thickness_before": round(t_now, 3), **r[2]})
        def body_reaches_axis(w):
            """Both body ends reach the axis ends, and the visible body is not cut shorter than its profile by a clipping half-space
            (ArchiCAD clips wall ends at joints: shortening the profile then moves the visible end by less than asked)."""
            v = _mesh(w)
            if v is None:
                return False
            a, b = c.axis[w.GlobalId]; u = c.udir(w); pu = v[:, :2] @ u; lo, hi = sorted([float(a @ u), float(b @ u)])
            if not (pu.min() <= lo + 0.03 and pu.max() >= hi - 0.03):
                return False
            from ifcgym.ifctools import _extrusion_of, _profile_local_points
            for it in _body_items(w):
                ex = _extrusion_of(it)
                if ex is not None:
                    pts = _profile_local_points(ex.SweptArea)
                    if pts is None:
                        return False
                    xs = sorted({round(p_[0], 3) for p_ in pts}); ys = sorted({round(p_[1], 3) for p_ in pts})
                    if len(pts) != 4 or len(xs) != 2 or len(ys) != 2:   # mitred / non-rectangular profiles (ArchiCAD joints): the end is not one plane
                        return False
                    prof_len = (xs[1] - xs[0]) * sc
                    if it.is_a("IfcBooleanResult") and (pu.max() - pu.min()) < prof_len - 0.02:   # the clip cuts the ends
                        return False
            return True
        plain = [w for w in c.walls if not _openings(w) and c.length(w) >= 2.0 and (_wall_thickness(w) > 0 or c.aligned[w.GlobalId]) and body_reaches_axis(w)]
        def bigger_end(a, b):
            """(end, fixed) with 'end' the point with the larger X (larger Y if X equal): unambiguous for every orientation."""
            return (a, b) if (a[0] > b[0] + 1e-3 or (abs(a[0] - b[0]) <= 1e-3 and a[1] > b[1])) else (b, a)
        END_WORD = "its end with the larger X coordinate (if both ends have the same X, the end with the larger Y)"
        START_WORD = "its end with the smaller X coordinate (if both ends have the same X, the end with the smaller Y)"
        for w in sample(plain, k):
            r = c.ref_wall(w)
            if r is None:
                continue
            a, b = c.axis[w.GlobalId]; u = c.udir(w); L = c.length(w); delta = rng.choice([0.5, 1.0])
            if L - delta < 1.0:
                continue
            end_pt, fixed = bigger_end(a, b)   # shorten at the end with the larger coordinate
            new_end = end_pt - (end_pt - fixed) / (np.linalg.norm(end_pt - fixed) or 1.0) * delta
            def seg_dist(p, q0, q1):
                v = q1 - q0; tt = np.clip(np.dot(p - q0, v) / (np.dot(v, v) or 1e-12), 0, 1); return float(np.linalg.norm(p - (q0 + tt * v)))
            touching = [o for o in c.walls if o.GlobalId in c.conn[w.GlobalId] and min(seg_dist(c.axis[o.GlobalId][0], new_end, end_pt), seg_dist(c.axis[o.GlobalId][1], new_end, end_pt)) < 0.5]
            add(f"U-shorten-wall-{r[1]}", f"Shorten {r[0]} by {delta} m at {END_WORD}, keeping the other end fixed and keeping its thickness and height. Connections at the shortened end may be removed and the spaces it bounds may be adjusted. {NC}",
                [w.GlobalId], [w.GlobalId] + [o.GlobalId for o in touching] + sorted(c.bounded[w.GlobalId]), {"type": "wall_axis", "guid": w.GlobalId, "p0": fmt(fixed), "p1": fmt(new_end), "tol": 0.03}, [], {"style": r[1], **r[2]})
        for w in sample(plain, k):
            r = c.ref_wall(w)
            if r is None:
                continue
            a, b = c.axis[w.GlobalId]; far, piv = bigger_end(a, b)
            ang = rng.choice([90, -90, 45, -45, 30, 180]); th = np.radians(ang); R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            mode = rng.choice(["end", "end", "centre", "point"] if ang != 180 else ["end", "end", "point"])   # 180 deg about the mid-point = the same wall (gate 20: no-op passed)
            if mode == "end":
                pv = piv; where = f"the vertical axis through {START_WORD}"
            elif mode == "centre":
                pv = (a + b) / 2; where = "the vertical axis through the mid-point of its axis"
            else:
                pv = np.round(piv + np.array([rng.choice([-1.0, 1.0]), rng.choice([-1.0, 1.0])]), 2); where = f"the vertical axis through the point ({pv[0]}, {pv[1]}) (metres, project XY)"
            p0n = pv + R @ (piv - pv); p1n = pv + R @ (far - pv); sense = "counter-clockwise" if ang > 0 else "clockwise"
            add(f"U-rotate-wall-{r[1]}", f"Rotate {r[0]} by {ang:+d} degrees ({sense} seen from above; the sign follows the right-hand rule about +Z) about {where}, keeping its length, thickness and height. Connected walls and the spaces it bounds may be adjusted. {NC}",
                [w.GlobalId], [w.GlobalId] + sorted(c.conn[w.GlobalId]) + sorted(c.bounded[w.GlobalId]), {"type": "wall_axis", "guid": w.GlobalId, "p0": fmt(p0n), "p1": fmt(p1n), "tol": 0.03}, [], {"style": r[1], "pivot": fmt(pv), "angle": ang, **r[2]})
        for sl in sample(c.slabs, k):
            r = c.ref_slab(sl); h = c.height(sl)
            if r is None or h is None:
                continue
            t_new = rng.choice([t for t in (0.2, 0.25, 0.3, 0.35) if abs(t - h) > 0.03]); ops = _openings(sl); top = round(float(c.bbox(sl)[1][2]), 3)
            add(f"U-set-slab-thickness-{r[1]}", f"Change the thickness of {r[0]} to {t_new} m, keeping its footprint and its top level (top stays at z = {top} m). {NC}",
                [sl.GlobalId], [sl.GlobalId] + [o.GlobalId for o in ops], {"type": "resize_height", "guid": sl.GlobalId, "height": t_new, "top": top, "tol": 0.02}, [], {"style": r[1], **r[2]})
        for s in sample(c.spaces, k):
            r = c.ref_space(s); h = c.height(s)
            if r is None or h is None:
                continue
            h_new = rng.choice([x for x in (2.5, 2.6, 3.0, 3.2) if abs(x - h) > 0.05])
            add(f"U-set-space-height-{r[1]}", f"Set the height of {r[0]} (an IfcSpace) to {h_new} m, keeping its footprint. {NC}",
                [s.GlobalId], [s.GlobalId], {"type": "resize_height", "guid": s.GlobalId, "height": h_new, "tol": 0.02}, [], {"style": r[1], **r[2]})
        for col in sample(c.columns, k):
            r = c.ref_column(col); h = c.height(col)
            if r is None or h is None:
                continue
            h_new = rng.choice([x for x in (2.4, 2.6, 3.0, 3.3) if abs(x - h) > 0.05])
            add(f"U-set-column-height-{r[1]}", f"Change the height of {r[0]} to {h_new} m, keeping its position and cross-section. {NC}",
                [col.GlobalId], [col.GlobalId], {"type": "resize_height", "guid": col.GlobalId, "height": h_new, "tol": 0.02}, [], {"style": r[1], **r[2]})
        for sl in sample([s for s in c.slabs if not _openings(s)], k):
            r = c.ref_slab(sl); bb = c.bbox(sl)
            if r is None or bb is None or bb[1][0] - bb[0][0] < 4.0:
                continue
            delta = rng.choice([1.0, 2.0])
            add(f"U-resize-slab-{r[1]}", f"Shorten {r[0]} by {delta} m in the X direction: keep its edge with the smallest X fixed and move its edge with the largest X by -{delta} m. Keep its thickness, elevation and Y extent. {NC}",
                [sl.GlobalId], [sl.GlobalId], {"type": "bbox", "guid": sl.GlobalId, "lo": fmt(bb[0][:2]), "hi": fmt([bb[1][0] - delta, bb[1][1]]), "tol": 0.05}, [], {"style": r[1], **r[2]})

    # ------------------------------------------------------------------ DELETE
    if "delete" in families:
        for kind, pool, styles in (("window", c.windows, ("D", "T", "S")), ("door", c.doors, ("T", "S", "P"))):
            for dd in sample(pool, k):
                r = c.ref_fill(dd, kind, styles)
                if r is None:
                    continue
                op, w = c.host(dd)
                add(f"D-delete-{kind}-{r[1]}", f"Delete {r[0]} together with its opening, so that the host wall becomes solid again. The wall itself must remain. {NC}",
                    [dd.GlobalId], [dd.GlobalId, op.GlobalId, w.GlobalId], {"type": "delete", "guid": dd.GlobalId}, [{"guid": op.GlobalId, "rule": "removed"}, {"guid": w.GlobalId, "rule": "unchanged"}], {"style": r[1], **r[2]})
                st_ = c.storey_of(dd)
                if "compound" in families and st_ is not None:   # BIM-Edit-style direct prompt spelling out every relationship
                    add(f"D-delete-{kind}-rel-D", f"Delete the {kind} with GlobalId {dd.GlobalId}. Also delete its opening (the IfcOpeningElement with GlobalId {op.GlobalId}) and remove the relationships of that opening to the wall with GlobalId {w.GlobalId} (IfcRelVoidsElement) and to the {kind} (IfcRelFillsElement), as well as the {kind}'s containment in {c.storey_ref(st_)}. The wall itself must remain unchanged. {NC}",
                        [dd.GlobalId], [dd.GlobalId, op.GlobalId, w.GlobalId], {"type": "delete", "guid": dd.GlobalId}, [{"guid": op.GlobalId, "rule": "removed"}, {"guid": w.GlobalId, "rule": "unchanged"}], {"style": "D"})
        for w in sample(c.walls, k):
            r = c.ref_wall(w)
            if r is None:
                continue
            ops = _openings(w); fl = [x for o in ops for x in _fills(o)]
            deps = [{"guid": o.GlobalId, "rule": "removed"} for o in ops] + [{"guid": x.GlobalId, "rule": "removed"} for x in fl] + [{"guid": g, "rule": "unchanged"} for g in sorted(c.conn[w.GlobalId] | c.bounded[w.GlobalId])]
            add(f"D-delete-wall-{r[1]}", f"Delete {r[0]} together with its openings and the doors/windows in them. The walls connected to it and the spaces it bounds must remain (only their relationships to the deleted wall are removed). {NC}",
                [w.GlobalId], [w.GlobalId] + [o.GlobalId for o in ops] + [x.GlobalId for x in fl] + sorted(c.conn[w.GlobalId] | c.bounded[w.GlobalId]), {"type": "delete", "guid": w.GlobalId}, deps, {"style": r[1], **r[2]})
        for col in sample(c.columns, k):
            r = c.ref_column(col)
            if r is None:
                continue
            add(f"D-delete-column-{r[1]}", f"Delete {r[0]}. {NC}", [col.GlobalId], [col.GlobalId], {"type": "delete", "guid": col.GlobalId}, [], {"style": r[1], **r[2]})
        for s in sample(c.spaces, k):
            r = c.ref_space(s)
            if r is None:
                continue
            ws = sorted(c.space_walls.get(s.GlobalId, ()))
            add(f"D-delete-space-{r[1]}", f"Delete {r[0]} (the IfcSpace) together with its space-boundary relationships. Its bounding walls must remain. {NC}",
                [s.GlobalId], [s.GlobalId] + ws, {"type": "delete", "guid": s.GlobalId}, [{"guid": g, "rule": "unchanged"} for g in ws], {"style": r[1], **r[2]})
        for sl in sample(c.slabs, k):
            r = c.ref_slab(sl)
            if r is None:
                continue
            ops = _openings(sl)
            add(f"D-delete-slab-{r[1]}", f"Delete {r[0]}" + (" together with its openings" if ops else "") + f". {NC}",
                [sl.GlobalId], [sl.GlobalId] + [o.GlobalId for o in ops], {"type": "delete", "guid": sl.GlobalId}, [{"guid": o.GlobalId, "rule": "removed"} for o in ops], {"style": r[1], **r[2]})
        for dd in sample(c.doors + c.windows, k):
            op, w = c.host(dd); kind = "door" if dd.is_a("IfcDoor") else "window"
            add("D-delete-opening-D", f"Delete the opening with GlobalId {op.GlobalId} and the {kind} that fills it, so that the wall becomes solid there. The wall must remain. {NC}",
                [op.GlobalId], [op.GlobalId, dd.GlobalId, w.GlobalId], {"type": "delete", "guid": op.GlobalId}, [{"guid": dd.GlobalId, "rule": "removed"}, {"guid": w.GlobalId, "rule": "unchanged"}], {"style": "D"})

        # multi-target (BIBIMBAP style): delete all windows of a wall; set the height of all walls of a storey
        for w in sample([w for w in c.walls if sum(1 for o in _openings(w) for x in _fills(o) if x.is_a("IfcWindow")) >= 2], k):
            wins = [x for o in _openings(w) for x in _fills(o) if x.is_a("IfcWindow")]; ops = [o for o in _openings(w) if any(x.is_a("IfcWindow") for x in _fills(o))]
            add("D-delete-windows-of-wall-D", f"Delete all {len(wins)} windows hosted by the wall with GlobalId {w.GlobalId} together with their openings, so the wall becomes solid there. The wall itself must remain. {NC}",
                [x.GlobalId for x in wins], [x.GlobalId for x in wins] + [o.GlobalId for o in ops] + [w.GlobalId], {"type": "delete_all", "guids": [x.GlobalId for x in wins]},
                [{"guid": o.GlobalId, "rule": "removed"} for o in ops] + [{"guid": w.GlobalId, "rule": "unchanged"}], {"style": "D", "n": len(wins)})
        for st in sample(c.storeys, k):
            ws_all = [w for w in f.by_type("IfcWall") if c.storey_of(w) == st]
            flat = [w for w in ws_all if w in c.walls and c.height(w) is not None and _body_items(w) and
                    all((_extrusion_depth(it) is not None and abs(_extrusion_depth(it) * sc - c.height(w)) <= 0.02) or it.is_a("IfcFacetedBrep") or it.is_a("IfcPolygonalFaceSet") for it in _body_items(w))]
            if not ws_all or len(flat) != len(ws_all) or len(ws_all) > 60:
                continue
            hs = {round(c.height(w), 2) for w in flat}; h_new = rng.choice([x for x in (2.6, 2.8, 3.0, 3.2) if all(abs(x - h) > 0.05 for h in hs)])
            ops = [o for w in flat for o in _openings(w)]; fl = [x for o in ops for x in _fills(o)]
            add("U-set-walls-height-storey-D", f"Change the height of all {len(flat)} walls (IfcWall) contained in {c.storey_ref(st)} to {h_new} m, keeping their footprints, positions and thicknesses. Their openings and the doors/windows in them must stay hosted. {NC}",
                [w.GlobalId for w in flat], [w.GlobalId for w in flat] + [o.GlobalId for o in ops] + [x.GlobalId for x in fl], {"type": "resize_height_all", "guids": [w.GlobalId for w in flat], "height": h_new, "tol": 0.02},
                [{"guid": o.GlobalId, "rule": "stays_hosted"} for o in ops], {"style": "D", "n": len(flat)})

    # ------------------------------------------------------------------ CREATE (reference variants + spaces)
    if "create2" in families:
        for rs in rec["removed_spaces"]:
            st = f.by_guid(rs["storey"]); P = np.array(rs["polygon"]); h = rs["height"]; ws = rs["walls"]
            opts = [(f"on {c.storey_ref(st)} with the footprint {rs['polygon']} (metres, project XY, corners in order)", "D")]
            closed = all(c.edge_on_wall(P[i], P[(i + 1) % len(P)], ws) for i in range(len(P))) if ws else False
            if closed and 3 <= len(ws) <= 8:
                opts.append((f"filling the area enclosed by the walls with GlobalIds {', '.join(ws[:-1])} and {ws[-1]} on {c.storey_ref(st)}, with its footprint along the inner faces of those walls", "T"))
            if closed and rs.get("centre"):
                opts.append((f"filling the room around the point ({rs['centre'][0]}, {rs['centre'][1]}) on {c.storey_ref(st)} (metres, project XY), with its footprint along the inner faces of the surrounding walls", "S"))
            for phrase, style in opts:
                add(f"C-create-space-{style}", f"Create a space (IfcSpace) {phrase}, {h} m high, aggregated into that storey. {NC}",
                    [st.GlobalId], [], {"type": "create_space", "polygon": rs["polygon"], "height": h, "storey": st.GlobalId, "tol": 0.1}, [], {"style": style, **({"anchors": ws} if style == "T" else {})}, ["IfcSpace"])
        long_walls = [w for w in c.walls if c.length(w) >= 2.5]
        for kind in ("door", "window"):
            for w in sample(long_walls, k):
                r = c.ref_wall(w, ("T", "S", "P")); hw = c.height(w)
                width, height, sill = ((rng.choice([0.8, 0.9, 1.0]), 2.1, 0.0) if kind == "door" else (rng.choice([0.9, 1.2, 1.5]), 1.2, 0.9))
                if r is None or hw is None or hw < height + sill + 0.1 or not free_slot(c, w, width):
                    continue
                add(f"C-create-{kind}-{r[1]}", f"Add a {kind} {width} m wide and {height} m high to {r[0]}, anywhere along the wall where it does not overlap an existing opening" + (" with its sill 0.9 m above the wall's base" if kind == "window" else ", starting at the wall's base") + f". The {kind} must be hosted by that wall through an opening (IfcRelVoidsElement / IfcRelFillsElement). {NC}",
                    [w.GlobalId], [w.GlobalId], {"type": "create_fill", "host": w.GlobalId, "kind": kind, "width": width, "height": height, "sill": sill, "tol": 0.03}, [], {"style": r[1], **r[2]}, ["IfcOpeningElement", "IfcDoor" if kind == "door" else "IfcWindow"])
                st_ = c.storey_of(w)
                if "compound" in families and st_ is not None:   # direct, every relationship spelled out (BIM-Edit direct prompts)
                    cls_ = "IfcDoor" if kind == "door" else "IfcWindow"
                    add(f"C-create-{kind}-rel-D", f"Add a {kind} {width} m wide and {height} m high to the wall with GlobalId {w.GlobalId}, anywhere along the wall where it does not overlap an existing opening" + (" with its sill 0.9 m above the wall's base" if kind == "window" else ", starting at the wall's base") + f". Also create an opening (IfcOpeningElement) for it and establish the relationships between the wall and the opening (IfcRelVoidsElement) and between the opening and the {kind} (IfcRelFillsElement), and assign the new {kind} to {c.storey_ref(st_)} (IfcRelContainedInSpatialStructure). {NC}",
                        [w.GlobalId], [w.GlobalId], {"type": "create_fill", "host": w.GlobalId, "kind": kind, "width": width, "height": height, "sill": sill, "tol": 0.03},
                        [{"guid": st_.GlobalId, "rule": "new_in_storey", "cls": cls_}], {"style": "D"}, ["IfcOpeningElement", cls_])
        for w in sample([w for w in long_walls if c.aligned[w.GlobalId]], k):
            st = c.storey_of(w); u = c.udir(w); a, b = c.axis[w.GlobalId]; n = np.array([-u[1], u[0]]); hw = c.height(w)
            if st is None or hw is None:
                continue
            dw = [k_ for k_, v in DIRS.items() if float(v @ n) > 0.9 or float(v @ -n) > 0.9]; dw = rng.choice(dw); off = rng.choice([1.0, 1.5]); p0, p1 = a + DIRS[dw] * off, b + DIRS[dw] * off; t = 0.15; h = round(hw, 2)
            add("C-create-wall-S", f"Create a new wall parallel to the wall with GlobalId {w.GlobalId}, {off} m away from it towards {DIR_WORD[dw]}, with the same length and the same extent along the wall (its axis is that wall's axis shifted by {off} m), {t} m thick and {h} m high, contained in the same storey. {NC}",
                [st.GlobalId], [], {"type": "create_wall", "p0": fmt(p0), "p1": fmt(p1), "thickness": t, "height": h, "storey": st.GlobalId, "tol": 0.03}, [], {"style": "S", "anchors": [w.GlobalId]}, ["IfcWall", "IfcWallStandardCase"])
        # a new wall between two free wall ends, connected to both (BIM-Edit: "connected to the walls with Ids A and B")
        free = []
        for w in c.walls:
            a, b = c.axis[w.GlobalId]
            for end_name, pt in (("start", a), ("end", b)):
                if not any(min(np.linalg.norm(c.axis[o][0] - pt), np.linalg.norm(c.axis[o][1] - pt)) < 0.3 or _seg_d(pt, *c.axis[o]) < 0.3 for o in c.conn[w.GlobalId] if o in c.axis):
                    free.append((w, end_name, pt))
        cands = [(x, y) for x in free for y in free if x[0].GlobalId != y[0].GlobalId and c.storey_of(x[0]) == c.storey_of(y[0]) and 1.0 <= np.linalg.norm(x[2] - y[2]) <= 6.0]
        for (wa, ea, pa), (wb, eb, pb) in sample(cands, k):
            st = c.storey_of(wa); hw = c.height(wa)
            if st is None or hw is None:
                continue
            add("C-create-wall-C", f"Create a new wall from the {ea} point of the axis of the wall with GlobalId {wa.GlobalId}, at ({round(float(pa[0]), 3)}, {round(float(pa[1]), 3)}), to the {eb} point of the axis of the wall with GlobalId {wb.GlobalId}, at ({round(float(pb[0]), 3)}, {round(float(pb[1]), 3)}) (metres, XY on {c.storey_ref(st)}), 0.15 m thick and {round(hw, 2)} m high, contained in that storey, and connect it to both walls with IfcRelConnectsPathElements. {NC}",
                [st.GlobalId], [wa.GlobalId, wb.GlobalId], {"type": "create_wall", "p0": fmt(pa), "p1": fmt(pb), "thickness": 0.15, "height": round(hw, 2), "storey": st.GlobalId, "tol": 0.03},
                [{"guid": wa.GlobalId, "rule": "connected_new", "cls": "IfcWall"}, {"guid": wb.GlobalId, "rule": "connected_new", "cls": "IfcWall"}], {"style": "D", "anchors": [wa.GlobalId, wb.GlobalId]}, ["IfcWall", "IfcWallStandardCase"])
        for s in sample([s for s in c.spaces if (c.area(s.GlobalId) or 0) >= 6], k):
            r = c.ref_space(s, ("D", "T")); st = c.storey_of(s); h = c.height(s)
            fp = c.fp(s.GlobalId); ctr = np.array(fp.centroid.coords[0]) if fp is not None else None
            if r is None or st is None or ctr is None or h is None or not c.contains(s.GlobalId, ctr, 0.5):
                continue
            size = rng.choice([0.3, 0.4]); cxy = [round(float(ctr[0]), 2), round(float(ctr[1]), 2)]
            add(f"C-create-column-{r[1]}", f"Create a square column {size} m by {size} m and {round(h, 2)} m high at the centroid of the footprint of {r[0]} (compute the centroid from the space's footprint polygon), contained in that storey. {NC}",
                [s.GlobalId], [], {"type": "create_column", "xy": cxy, "size": size, "height": round(h, 2), "storey": st.GlobalId, "tol": 0.1}, [], {"style": r[1], **r[2]}, ["IfcColumn"])
        for s in sample([s for s in c.spaces if c.ring(s.GlobalId) is not None], k):
            r = c.ref_space(s, ("D", "T")); st = c.storey_of(s)
            if r is None or st is None:
                continue
            th = rng.choice([0.15, 0.2]); poly = [[round(float(x), 3), round(float(y), 3)] for x, y in c.ring(s.GlobalId)]; top = round(float(world_matrix(st)[2, 3]) * sc, 3)
            add(f"C-create-slab-{r[1]}", f"Create a floor slab {th} m thick whose footprint is exactly the footprint of {r[0]}, with its top at the storey level (z = {top} m), contained in that storey. {NC}",
                [s.GlobalId], [], {"type": "create_slab", "polygon": poly, "thickness": th, "storey": st.GlobalId, "top": top, "tol": 0.03}, [], {"style": r[1], **r[2]}, ["IfcSlab"])

    # ------------------------------------------------------------------ RETRIEVE (numeric / geometry / topological)
    JSN = 'Do not modify the model. End your final message with a JSON object {"value": <number>}.'
    JSG = 'Do not modify the model. End your final message with a JSON object {"global_id": "<GlobalId>"}.'
    JSL = 'Do not modify the model. End your final message with a JSON object {"global_ids": [<GlobalIds>]}.'
    if "retrieve2" in families:
        def q(tpl, prompt, target, intent, meta=None):
            add(tpl, prompt, target, [], intent, [], meta, tier=0)
        for w in sample(c.walls, k):
            r = c.ref_wall(w)
            if r:
                q(f"R-wall-length-{r[1]}", f"What is the length in metres of {r[0]}, measured along its axis? {JSN}", [w.GlobalId], {"type": "answer", "kind": "number", "value": round(c.length(w), 2), "rel_tol": 0.01, "tol": 0.1}, {"style": r[1], **r[2]})
        for st in sample(c.storeys, k):
            ws = [w for w in c.walls if c.storey_of(w) == st]
            if len(ws) != sum(1 for w in f.by_type("IfcWall") if c.storey_of(w) == st):
                ws = []    # walls without an Axis on this storey: the "all walls" questions would count a subset
            if ws:
                q("R-storey-wall-length-D", f"What is the total length in metres of all walls (IfcWall) contained in {c.storey_ref(st)}, measured along their axes? {JSN}", [st.GlobalId], {"type": "answer", "kind": "number", "value": round(sum(c.length(w) for w in ws), 2), "rel_tol": 0.01, "tol": 0.1})
            wins = [x for x in f.by_type("IfcWindow") if c.storey_of(x) == st]
            q("R-count-windows-D", f"How many windows (IfcWindow) are contained in {c.storey_ref(st)}? {JSN}", [st.GlobalId], {"type": "answer", "kind": "number", "value": len(wins), "tol": 0.0})
            fl = [x for x in f.by_type("IfcDoor") + f.by_type("IfcWindow") if c.storey_of(x) == st]
            areas = [(float(x.OverallWidth or 0) * sc) * (float(x.OverallHeight or 0) * sc) if (x.OverallWidth and x.OverallHeight) else None for x in fl]
            if fl and all(a_ is not None for a_ in areas) and all(abs(a_ - 2.0) > 0.15 for a_ in areas):
                q("R-count-large-fills-D", f"How many doors and windows (IfcDoor, IfcWindow) contained in {c.storey_ref(st)} have OverallWidth times OverallHeight larger than 2 square metres? {JSN}", [st.GlobalId], {"type": "answer", "kind": "number", "value": sum(1 for a_ in areas if a_ > 2.0), "tol": 0.0})
            sls = [s for s in c.slabs if c.storey_of(s) == st]
            if len(sls) == 1 and (P := profile_world_polygon(sls[0])):
                A = abs(_poly_area(np.array(P)[:, :2] * sc))
                q("R-storey-gross-area-T", f"What is the gross floor area in square metres of {c.storey_ref(st)}, measured as the area enclosed by the outer boundary of its floor slab's footprint (openings in the slab are included)? {JSN}", [st.GlobalId], {"type": "answer", "kind": "number", "value": round(A, 2), "rel_tol": 0.01, "tol": 0.1})
            if len(ws) >= 3:
                for name, fn in (("shortest", min), ("longest", max)):
                    ls = sorted((c.length(w), w.GlobalId) for w in ws); pick = ls[0] if name == "shortest" else ls[-1]; nxt = ls[1] if name == "shortest" else ls[-2]
                    if abs(pick[0] - nxt[0]) > 0.05:
                        q(f"R-{name}-wall-S", f"Which wall (IfcWall) contained in {c.storey_ref(st)} is the {name}, measured along its axis? {JSG}", [pick[1]], {"type": "answer", "kind": "guid", "value": pick[1]})
        for s in sample(c.spaces, k):
            r = c.ref_space(s); h = c.height(s); A = c.area(s.GlobalId)
            if r is None or h is None or A is None:
                continue
            q(f"R-space-volume-{r[1]}", f"What is the net volume in cubic metres of {r[0]} (footprint area times height)? {JSN}", [s.GlobalId], {"type": "answer", "kind": "number", "value": round(A * h, 2), "rel_tol": 0.01, "tol": 0.1}, {"style": r[1], **r[2]})
            adj = sorted({o for w in c.space_walls.get(s.GlobalId, ()) for o in c.bounded[w]} - {s.GlobalId})
            if adj:
                q(f"R-adjacent-spaces-{r[1]}", f"Which spaces (IfcSpace) are adjacent to {r[0]}, i.e. share at least one bounding wall (IfcRelSpaceBoundary) with it? {JSL}", [s.GlobalId], {"type": "answer", "kind": "guids", "value": adj}, {"style": r[1], **r[2]})
        pairs = [(a, b) for a in sorted(c.space_walls) for b in sorted(c.space_walls) if a < b and len(c.space_walls[a] & c.space_walls[b]) == 1]
        for a, b in sample(pairs, k):
            w = next(iter(c.space_walls[a] & c.space_walls[b]))
            q("R-wall-separating-T", f"Which wall separates the space with GlobalId {a} from the space with GlobalId {b} (the only wall bounding both)? {JSG}", [w], {"type": "answer", "kind": "guid", "value": w}, {"anchors": [a, b]})
        for w in sample([w for w in c.walls if c.conn[w.GlobalId]], k):
            q("R-connected-walls-T", f"Which walls are connected to the wall with GlobalId {w.GlobalId} through IfcRelConnectsPathElements? {JSL}", [w.GlobalId], {"type": "answer", "kind": "guids", "value": sorted(c.conn[w.GlobalId])}, {"anchors": [w.GlobalId]})
        cols_by_st = {}
        for col in c.columns:
            st = c.storey_of(col)
            if st is not None:
                cols_by_st.setdefault(st.GlobalId, []).append(col)
        for st_g, cols in sample(list(cols_by_st.items()), k):
            if len(cols) >= 2:
                a, b = rng.sample(cols, 2); dd = float(np.linalg.norm(c.col_centre(a) - c.col_centre(b)))
                q("R-column-distance-D", f"What is the distance in metres in plan (XY, centre to centre) between the column with GlobalId {a.GlobalId} and the column with GlobalId {b.GlobalId}? {JSN}", [a.GlobalId, b.GlobalId], {"type": "answer", "kind": "number", "value": round(dd, 2), "rel_tol": 0.01, "tol": 0.1})
        for s in sample(c.spaces, k):
            st = c.storey_of(s); ctr = c.rep_point(s.GlobalId)
            if st is None or ctr is None:
                continue
            pt = (round(float(ctr[0]), 2), round(float(ctr[1]), 2))
            hits = [o for o in c.spaces if c.storey_of(o) == st and c.contains(o.GlobalId, np.array(pt))]
            if hits == [s]:
                q("R-space-at-point-S", f"Which space (IfcSpace) on {c.storey_ref(st)} contains the point ({pt[0]}, {pt[1]}) (metres, project XY)? {JSG}", [s.GlobalId], {"type": "answer", "kind": "guid", "value": s.GlobalId})
        for s in sample(c.spaces, k):      # BIBIMBAP numeric: room perimeter; topological: a room not adjacent to R
            r = c.ref_space(s); fp = c.fp(s.GlobalId); st = c.storey_of(s)
            if r is None or fp is None or st is None:
                continue
            q(f"R-space-perimeter-{r[1]}", f"What is the perimeter in metres of the footprint of {r[0]} (outer boundary of its floor polygon, holes excluded)? {JSN}", [s.GlobalId], {"type": "answer", "kind": "number", "value": round(float(fp.exterior.length), 2), "rel_tol": 0.01, "tol": 0.1}, {"style": r[1], **r[2]})
            adj = {o for w in c.space_walls.get(s.GlobalId, ()) for o in c.bounded[w]} - {s.GlobalId}
            others = [o for o in c.spaces if c.storey_of(o) == st and o.GlobalId != s.GlobalId and o.GlobalId not in adj]
            if 1 <= len(others) <= 4 and c.space_walls.get(s.GlobalId):   # list answer (a unique non-adjacent room was rare: gate 20)
                q("R-non-adjacent-space-T", f"Which spaces (IfcSpace) on {c.storey_ref(st)} share no bounding wall (IfcRelSpaceBoundary) with the space with GlobalId {s.GlobalId}? List all of them. {JSL}", sorted(o.GlobalId for o in others), {"type": "answer", "kind": "guids", "value": sorted(o.GlobalId for o in others)}, {"anchors": [s.GlobalId, st.GlobalId]})
        for w in sample([w for w in c.walls if c.height(w) is not None], k):   # BIBIMBAP numeric: net face area of a wall (length x height minus openings)
            r = c.ref_wall(w); h = c.height(w)
            if r is None:
                continue
            op_area = 0.0
            for o in _openings(w):
                bb = c.bbox(o)
                if bb is not None:
                    u = c.udir(w); ext = bb[1] - bb[0]; op_area += float(abs(ext[0] * u[0]) + abs(ext[1] * u[1])) * float(ext[2])
            q(f"R-wall-net-area-{r[1]}", f"What is the net area in square metres of one face of {r[0]}: its axis length times its height, minus the face area (width times height) of its openings? {JSN}", [w.GlobalId], {"type": "answer", "kind": "number", "value": round(c.length(w) * h - op_area, 2), "rel_tol": 0.02, "tol": 0.1}, {"style": r[1], **r[2]})
        for dd in sample(c.doors, k):
            r = c.ref_fill(dd, "door"); _, width = c.fill_centre(dd)
            if r:
                q(f"R-door-width-{r[1]}", f"What is the width in metres of {r[0]}, measured as the extent of the door's geometry along the axis of its wall? {JSN}", [dd.GlobalId], {"type": "answer", "kind": "number", "value": round(width, 2), "rel_tol": 0.01, "tol": 0.02}, {"style": r[1], **r[2]})

    # ------------------------------------------------------------------ CONCEPT (BIBIMBAP 'conceptual' questions with an operational definition)
    if "concept" in families:
        def q(tpl, prompt, target, intent, meta=None):
            add(tpl, prompt, target, [], intent, [], meta, tier=0)
        for st in c.storeys:
            sp = [s for s in c.spaces if c.storey_of(s) == st]
            if len(sp) < 3:
                continue
            ndoors = {s.GlobalId: sum(1 for w in c.space_walls.get(s.GlobalId, ()) for o in _openings(f.by_guid(w)) for x in _fills(o) if x.is_a("IfcDoor")) for s in sp}
            top = sorted(ndoors.items(), key=lambda kv: -kv[1])
            if top[0][1] - top[1][1] >= 1:
                q("A-corridor-T", f"Which space on {c.storey_ref(st)} acts as the corridor, defined as the space whose bounding walls host the most doors? {JSG}", [top[0][0]], {"type": "answer", "kind": "guid", "value": top[0][0]})
            nadj = {s.GlobalId: len({o for w in c.space_walls.get(s.GlobalId, ()) for o in c.bounded[w]} - {s.GlobalId}) for s in sp}
            top = sorted(nadj.items(), key=lambda kv: -kv[1])
            if top[0][1] - top[1][1] >= 1:
                q("A-hub-T", f"Which space on {c.storey_ref(st)} is the hub, defined as the space adjacent to (sharing a bounding wall with) the most other spaces? {JSG}", [top[0][0]], {"type": "answer", "kind": "guid", "value": top[0][0]})
            stairs = [x for x in f.by_type("IfcStair") if c.storey_of(x) == st]
            if len(stairs) == 1:
                q("A-vertical-connector-T", f"Which element on {c.storey_ref(st)} provides the vertical circulation connecting it to the next storey (an IfcStair)? {JSG}", [stairs[0].GlobalId], {"type": "answer", "kind": "guid", "value": stairs[0].GlobalId})
            # shortest door path between two spaces (unique)
            edges = {}
            for x in c.doors:
                sides = c.door_sides(x)
                if sides is not None and c.storey_of(x) == st:
                    edges.setdefault(sides[0], []).append((sides[1], x.GlobalId)); edges.setdefault(sides[1], []).append((sides[0], x.GlobalId))
            ids = [s.GlobalId for s in sp if s.GlobalId in edges]
            for _ in range(min(k, 3)):
                if len(ids) < 2:
                    break
                src, dst_ = rng.sample(ids, 2)
                from collections import deque
                dist = {src: 0}; paths = {src: [[]]}; dq = deque([src])
                while dq:
                    u = dq.popleft()
                    for v, door in edges.get(u, []):
                        if v not in dist:
                            dist[v] = dist[u] + 1; paths[v] = [p + [door] for p in paths[u]]; dq.append(v)
                        elif dist[v] == dist[u] + 1:
                            paths[v] += [p + [door] for p in paths[u]]
                if dst_ in paths and len(paths[dst_]) == 1 and 2 <= dist[dst_] <= 4:
                    q("A-door-path-T", f"Which doors lie on the shortest path (counted in doors) from the space with GlobalId {src} to the space with GlobalId {dst_}? A door connects the two spaces that lie on either side of its wall at the door's position (both bounded by that wall, nearest on each side); the shortest path is unique here. {JSL}", [src, dst_], {"type": "answer", "kind": "guids", "value": sorted(paths[dst_][0])}, {"anchors": [src, dst_]})
        ground = c.storeys[0] if c.storeys else None
        if ground is not None:
            # with correct exterior labels many ground-floor doors sit in exterior walls (balconies, terraces); the
            # entrance is the exterior door with a space beyond one face of its wall and no space at all beyond the other
            ext = [d for d in c.doors if c.storey_of(d) == ground and c.is_external(c.host(d)[1]) and c.exit_door(d)]
            if len(ext) == 1:
                q("A-entrance-T", f"Which door is the building entrance, defined as the only door on the lowest storey ({c.storey_ref(ground)}) that is hosted by an exterior wall (Pset_WallCommon.IsExternal = true) and leads outside: just beyond one face of its wall at the door there is a space (IfcSpace), just beyond the other face there is none (doors to balconies or terraces modelled as spaces do not count)? {JSG}", [ext[0].GlobalId], {"type": "answer", "kind": "guid", "value": ext[0].GlobalId})
    # ------------------------------------------------------------------ COVER: generic IFC edits the older templates never asked for: bounding-box
    # sizing with a base elevation, relationships for a NEW column / slab (containment, IfcRelSpaceBoundary, IfcRelConnectsElements),
    # moving a slab to another storey's level with re-containment, stacking a column onto the one below, slab thickness by bounding box
    # with the bottom fixed, one-sided slab extension / shortening.
    if "cover" in families:
        def elev(st):
            return round(float(world_matrix(st)[2, 3]) * sc, 3)
        storey_idx = {s.GlobalId: i for i, s in enumerate(c.storeys)}
        cols_on = {}
        for col in c.columns:
            st_ = c.storey_of(col)
            if st_ is not None:
                cols_on.setdefault(st_.GlobalId, []).append(col)
        # 1 new column by bounding box + relationships
        for s in sample([s for s in c.spaces if (c.area(s.GlobalId) or 0) >= 6], k):
            st = c.storey_of(s); h = c.height(s); p = c.rep_point(s.GlobalId, 0.8)
            if st is None or h is None or p is None:
                continue
            sx, sy = rng.choice([(0.24, 0.24), (0.3, 0.3), (0.4, 0.4), (0.3, 0.5), (0.25, 0.4)])
            x, y = round(float(p[0]), 3), round(float(p[1]), 3); z0 = elev(st); hh = round(h, 2)
            ph = rng.choice(["centre", "bbox", "corner"])
            if ph == "centre":
                geo = f"with the centre of its lower face at ({x}, {y}, {z0}), {sx} m long in X, {sy} m wide in Y and {hh} m high"
            elif ph == "bbox":
                geo = f"on {c.storey_ref(st)} at point ({x}, {y}) with the length, width and height of its bounding box ({sx}, {sy}, {hh}), its base at z = {z0}"
            else:
                geo = f"whose bounding box spans from ({round(x - sx / 2, 3)}, {round(y - sy / 2, 3)}, {z0}) to ({round(x + sx / 2, 3)}, {round(y + sy / 2, 3)}, {round(z0 + hh, 3)}) ({sx} x {sy} m, {hh} m high)"
            deps = [{"guid": st.GlobalId, "rule": "new_in_storey", "cls": "IfcColumn"}]; rels = [f"contained in the storey with GlobalId {st.GlobalId} (IfcRelContainedInSpatialStructure)"]; allowed = [st.GlobalId]
            if rng.random() < 0.6:
                deps.append({"guid": s.GlobalId, "rule": "space_boundary_new", "cls": "IfcColumn"}); allowed.append(s.GlobalId)
                rels.append(f"an IfcRelSpaceBoundary between the space with GlobalId {s.GlobalId} and the new column")
            others = cols_on.get(st.GlobalId, [])
            if others and rng.random() < 0.6:
                o = rng.choice(others); deps.append({"guid": o.GlobalId, "rule": "connected_new", "cls": "IfcColumn"}); allowed.append(o.GlobalId)
                rels.append(f"a connection to the column with GlobalId {o.GlobalId} using IfcRelConnectsElements")
            rel_txt = rels[0] if len(rels) == 1 else ", ".join(rels[:-1]) + " and " + rels[-1]
            add(f"C-create-column-box-D", f"Create a column (IfcColumn) {geo} (metres, project coordinates). Give it {rel_txt}. {NC}",
                [st.GlobalId], allowed, {"type": "create_column", "xy": [x, y], "sx": sx, "sy": sy, "height": hh, "z0": z0, "storey": st.GlobalId, "tol": 0.05},
                deps, {"style": "D", "phrase": ph, "anchors": sorted(set(allowed))}, ["IfcColumn"])
        # 2 new slab by bounding box (rectangular rooms) + relationships
        rect = []
        for s in c.spaces:
            R_ = c.ring(s.GlobalId); A_ = c.area(s.GlobalId)
            if R_ is not None and A_ and len(R_) == 4:
                bx = R_[:, 0].max() - R_[:, 0].min(); by_ = R_[:, 1].max() - R_[:, 1].min()
                if abs(bx * by_ - A_) <= 0.01 * A_ and bx >= 1.5 and by_ >= 1.5:
                    rect.append((s, R_))
        for s, R_ in sample(rect, k):
            st = c.storey_of(s)
            if st is None:
                continue
            t = rng.choice([0.2, 0.25, 0.3]); x0, y0 = round(float(R_[:, 0].min()), 3), round(float(R_[:, 1].min()), 3)
            L, W = round(float(R_[:, 0].max() - R_[:, 0].min()), 3), round(float(R_[:, 1].max() - R_[:, 1].min()), 3); top = elev(st); z0 = round(top - t, 3)
            ph = rng.choice(["start", "bbox"])
            geo = (f"starting at ({x0}, {y0}, {z0}) with a length of {L} m in X, a width of {W} m in Y and a height of {t} m in Z" if ph == "start" else
                   f"that spans the bounding box from ({x0}, {y0}, {z0}) to ({round(x0 + L, 3)}, {round(y0 + W, 3)}, {top}) ({L} m x {W} m, {t} m thick)")
            deps = [{"guid": st.GlobalId, "rule": "new_in_storey", "cls": "IfcSlab"}]; allowed = [st.GlobalId]; extra = ""
            if rng.random() < 0.6:
                deps.append({"guid": s.GlobalId, "rule": "space_boundary_new", "cls": "IfcSlab"}); allowed.append(s.GlobalId)
                extra = f" and add an IfcRelSpaceBoundary between the space with GlobalId {s.GlobalId} and the new slab"
            poly = [[x0, y0], [round(x0 + L, 3), y0], [round(x0 + L, 3), round(y0 + W, 3)], [x0, round(y0 + W, 3)]]
            add("C-create-slab-box-D", f"Create a slab (IfcSlab) {geo} (metres, project coordinates). Assign it to the storey with GlobalId {st.GlobalId} (IfcRelContainedInSpatialStructure){extra}. {NC}",
                [st.GlobalId], allowed, {"type": "create_slab", "polygon": poly, "thickness": t, "storey": st.GlobalId, "top": top, "tol": 0.03},
                deps, {"style": "D", "phrase": ph, "anchors": sorted(set(allowed))}, ["IfcSlab"])
        # 3 move a slab to the next storey's level and re-contain it there
        for sl in sample(c.slabs, k):
            st = c.storey_of(sl); r = c.ref_slab(sl, styles=("D", "S"))
            if st is None or r is None or storey_idx.get(st.GlobalId) is None:
                continue
            i = storey_idx[st.GlobalId]; nxt = c.storeys[i + 1] if i + 1 < len(c.storeys) else None
            if nxt is None:
                continue
            dz = round(elev(nxt) - elev(st), 3)
            if dz <= 0.5:
                continue
            ops = _openings(sl)
            add(f"U-move-slab-storey-{r[1]}", f"Move {r[0]} by {dz} m in +Z ([0.0, 0.0, {dz}] m) and change its storey to the storey with GlobalId {nxt.GlobalId}: it must be contained in that storey (IfcRelContainedInSpatialStructure) and no longer in the storey with GlobalId {st.GlobalId}. Keep its footprint and thickness. {NC}",
                [sl.GlobalId], [sl.GlobalId, st.GlobalId, nxt.GlobalId] + [o.GlobalId for o in ops], {"type": "translate", "guid": sl.GlobalId, "vector": [0.0, 0.0, dz], "tol": 0.02},
                [{"guid": nxt.GlobalId, "rule": "in_storey", "el": sl.GlobalId}] + [{"guid": o.GlobalId, "rule": "translate_with", "vector": [0.0, 0.0, dz]} for o in ops],
                {"style": r[1], **r[2], "anchors": sorted({nxt.GlobalId, st.GlobalId} | set(r[2].get("anchors", [])))})
        # 4 stack a column onto the column below it (XY aligned) and connect them
        for st in c.storeys[1:]:
            below = c.storeys[storey_idx[st.GlobalId] - 1]; up, dn = cols_on.get(st.GlobalId, []), cols_on.get(below.GlobalId, [])
            if not up or not dn:
                continue
            a = rng.choice(up); b = rng.choice(dn); ca, cb = c.col_centre(a), c.col_centre(b)
            vec = fmt(list(cb - ca)) + [0.0]
            if float(np.linalg.norm(cb - ca)) < 0.3:
                continue
            ra = c.ref_column(a, styles=("D", "S"))
            if ra is None:
                continue
            add(f"U-stack-column-{ra[1]}", f"Move {ra[0]} horizontally so that it stands on top of the column with GlobalId {b.GlobalId} on the storey below: the same X and Y centre ({round(float(cb[0]), 3)}, {round(float(cb[1]), 3)}), i.e. by the vector {vec} m. Keep its height and storey, and establish a connection relationship between the two columns (IfcRelConnectsElements). {NC}",
                [a.GlobalId], [a.GlobalId, b.GlobalId], {"type": "translate", "guid": a.GlobalId, "vector": vec, "tol": 0.03},
                [{"guid": b.GlobalId, "rule": "rel_connects", "el": a.GlobalId}], {"style": ra[1], **ra[2], "anchors": sorted({b.GlobalId} | set(ra[2].get("anchors", [])))})
        # 5 slab thickness by bounding-box height, bottom fixed
        for sl in sample(c.slabs, k):
            r = c.ref_slab(sl); bb = c.bbox(sl)
            if r is None or bb is None or _openings(sl):
                continue
            h = float(bb[1][2] - bb[0][2]); t_new = rng.choice([t for t in (0.2, 0.25, 0.3, 0.35) if abs(t - h) > 0.03])
            lo = fmt(bb[0]); hi = fmt([bb[1][0], bb[1][1], bb[0][2] + t_new])
            add(f"U-slab-thickness-bbox-{r[1]}", f"{'Increase' if t_new > h else 'Decrease'} the thickness of {r[0]} so that its bounding box has a height of {t_new} m. Keep the lowest z value of its bounding box fixed (z = {lo[2]}) and keep its footprint. {NC}",
                [sl.GlobalId], [sl.GlobalId], {"type": "bbox", "guid": sl.GlobalId, "lo": lo, "hi": hi, "tol": 0.02}, [], {"style": r[1], **r[2]})
        # 6 one-sided slab extension / shortening with the other side fixed at a named coordinate
        for sl in sample([s for s in c.slabs if not _openings(s)], k):
            r = c.ref_slab(sl); bb = c.bbox(sl)
            if r is None or bb is None or bb[1][0] - bb[0][0] < 4.0:
                continue
            d = rng.choice([1.0, 1.5, 2.0]); side = rng.choice(["+x", "-x"]); grow = rng.random() < 0.5
            lo, hi = [float(bb[0][0]), float(bb[0][1])], [float(bb[1][0]), float(bb[1][1])]
            if side == "+x":
                fixed = round(lo[0], 3); hi[0] += d if grow else -d
            else:
                fixed = round(hi[0], 3); lo[0] -= d if grow else -d
            verb = (f"Extend {r[0]} by {d} m in {'positive' if side == '+x' else 'negative'} X direction" if grow else f"Decrease the length of {r[0]} in X by {d} m")
            add(f"U-slab-extend-{r[1]}", f"{verb}, keeping its edge at x = {fixed} fixed. Keep its Y extent, thickness and elevation, and preserve its relationships. {NC}",
                [sl.GlobalId], [sl.GlobalId], {"type": "bbox", "guid": sl.GlobalId, "lo": fmt(lo), "hi": fmt(hi), "tol": 0.05}, [], {"style": r[1], **r[2]})

    # ------------------------------------------------------------------ COVER2: the operation types the
    # two benchmarks ask for that no template covered: relationship-spelling deletes of walls / columns / slabs / rooms; wall by bounding
    # box with connections and space boundaries; room by bounding box with boundaries, room merge, room bbox shrink, relative room height;
    # one-sided wall thickness; shorten + delete a named connection; window moved to the wall below; opening widened symmetrically; window
    # made taller; door / window by the opening's bounding box; opening cut in a slab; slab on the tops of columns; slab laid on a slab.
    if "cover2" in families or "cover3" in families:
        R3 = "cover3" in families     # the same operations with the target referenced by description
        def elev2(st):
            return round(float(world_matrix(st)[2, 3]) * sc, 3)
        def lst(ids):
            ids = sorted(ids); return ", ".join(ids[:-1]) + (" and " if len(ids) > 1 else "") + ids[-1] if ids else ""
        def contained_storey(e):
            for rel in (getattr(e, "ContainedInStructure", None) or []):
                if rel.RelatingStructure is not None and rel.RelatingStructure.is_a("IfcBuildingStorey"):
                    return rel.RelatingStructure
            for rel in (getattr(e, "Decomposes", None) or []):
                if rel.RelatingObject is not None and rel.RelatingObject.is_a("IfcBuildingStorey"):
                    return rel.RelatingObject
            return None
        def sb_elems(e):      # elements bounded by / bounding through IfcRelSpaceBoundary
            if e.is_a("IfcSpace"):
                return sorted({b.RelatedBuildingElement.GlobalId for b in (e.BoundedBy or []) if b.RelatedBuildingElement is not None})
            return sorted({b.RelatingSpace.GlobalId for b in (getattr(e, "ProvidesBoundaries", None) or []) if b.RelatingSpace is not None})
        def conns_of(e):
            out = set()
            for rel in list(getattr(e, "ConnectedTo", []) or []) + list(getattr(e, "ConnectedFrom", []) or []):
                o = rel.RelatedElement if rel.RelatingElement == e else rel.RelatingElement
                if o is not None:
                    out.add(o.GlobalId)
            return sorted(out)
        def rel_clause(conn, conn_cls, sbs, st, bim):
            parts = []
            if bim:
                if conn: parts.append(f"Delete {conn_cls} from the elements with the IDs {lst(conn)}.")
                if sbs: parts.append(f"Delete IfcRelSpaceBoundary from the elements with IDs {lst(sbs)}.")
                if st is not None: parts.append(f"Delete IfcRelContainedInSpatialStructure from the level with ID {st.GlobalId}.")
                return " ".join(parts)
            if conn: parts.append(f"its connections to {lst(conn)} ({conn_cls})")
            if sbs: parts.append(f"its space boundaries to {lst(sbs)} (IfcRelSpaceBoundary)")
            if st is not None: parts.append(f"its assignment to the storey {st.GlobalId}")
            return ("Also delete " + "; ".join(parts) + ".") if parts else ""
        # 1 relationship-spelling deletes: wall / column / slab / space
        for w in sample(c.walls, k):
            st = contained_storey(w); ops = _openings(w); fl = [x for o in ops for x in _fills(o)]; cn = conns_of(w); sbs = sb_elems(w)
            if not (cn or sbs):
                continue
            bim = rng.random() < 0.5
            extra = (f" Also delete its openings ({lst([o.GlobalId for o in ops])}) and the doors / windows in them." if ops else "")
            rr = c.ref_wall(w, styles=("T", "S", "P", "V")) if R3 else ("the wall with ID " + w.GlobalId, "D", {})
            if rr is None:
                continue
            add(f"D-delete-wall-rel-{rr[1]}", f"Delete {rr[0]}.{extra} {rel_clause(cn, 'IfcRelConnectsPathElements', sbs, st, bim)} The other elements must remain unchanged. {NC}",
                [w.GlobalId], [w.GlobalId] + [o.GlobalId for o in ops] + [x.GlobalId for x in fl] + cn + sbs, {"type": "delete", "guid": w.GlobalId},
                [{"guid": o.GlobalId, "rule": "removed"} for o in ops] + [{"guid": x.GlobalId, "rule": "removed"} for x in fl] + [{"guid": g, "rule": "unchanged"} for g in cn + sbs], {"style": rr[1], **rr[2], "bim_phrase": bim})
        for col in sample(c.columns, k):
            st = contained_storey(col); cn = conns_of(col); sbs = sb_elems(col)
            bim = rng.random() < 0.5
            rr = c.ref_column(col, styles=("T", "S", "P")) if R3 else ("the column with ID " + col.GlobalId, "D", {})
            if rr is None:
                continue
            add(f"D-delete-column-rel-{rr[1]}", f"Delete {rr[0]}. {rel_clause(cn, 'IfcRelConnectsElements', sbs, st, bim)} The other elements must remain unchanged. {NC}",
                [col.GlobalId], [col.GlobalId] + cn + sbs, {"type": "delete", "guid": col.GlobalId}, [{"guid": g, "rule": "unchanged"} for g in cn + sbs], {"style": rr[1], **rr[2], "bim_phrase": bim})
        for sl in sample(c.slabs, k):
            st = contained_storey(sl); sbs = sb_elems(sl); ops = _openings(sl)
            bim = rng.random() < 0.5; extra = (f" Also delete its openings ({lst([o.GlobalId for o in ops])})." if ops else "")
            rr = c.ref_slab(sl, styles=("T", "S")) if R3 else ("the slab with ID " + sl.GlobalId, "D", {})
            if rr is None:
                continue
            add(f"D-delete-slab-rel-{rr[1]}", f"Delete {rr[0]}.{extra} {rel_clause([], '', sbs, st, bim)} The other elements must remain unchanged. {NC}",
                [sl.GlobalId], [sl.GlobalId] + [o.GlobalId for o in ops] + sbs, {"type": "delete", "guid": sl.GlobalId},
                [{"guid": o.GlobalId, "rule": "removed"} for o in ops] + [{"guid": g, "rule": "unchanged"} for g in sbs], {"style": rr[1], **rr[2], "bim_phrase": bim})
        for s in sample(c.spaces, k):
            st = contained_storey(s); sbs = sb_elems(s)
            if not sbs:
                continue
            bim = rng.random() < 0.5
            agg = (f" Delete IfcRelAggregates / IfcRelContainedInSpatialStructure to the level with ID {st.GlobalId}." if (bim and st is not None) else (f" and its assignment to the storey {st.GlobalId}" if st is not None else ""))
            txt = (f"Delete IfcRelSpaceBoundary from the elements with IDs {lst(sbs)}.{agg}" if bim else f"Also delete its space boundary relationships to the elements {lst(sbs)}{agg}.")
            rr = c.ref_space(s, styles=("T", "S")) if R3 else ("the room (IfcSpace) with ID " + s.GlobalId, "D", {})
            if rr is None or (R3 and rr[1] == "T"):      # a T reference lists the bounding walls, i.e. the relationships to delete: too trivial
                continue
            add(f"D-delete-space-rel-{rr[1]}", f"Delete {rr[0]}. {txt} The bounding elements must remain unchanged. {NC}",
                [s.GlobalId], [s.GlobalId] + sbs, {"type": "delete", "guid": s.GlobalId}, [{"guid": g, "rule": "unchanged"} for g in sbs], {"style": rr[1], **rr[2], "bim_phrase": bim})
        # 2 wall by bounding box across a rectangular room (partition), connected to the two walls its ends meet, optional space boundary
        rects = []
        for s in c.spaces:
            R_ = c.ring(s.GlobalId); A_ = c.area(s.GlobalId)
            if R_ is not None and A_ and len(R_) == 4:
                x0, x1, y0, y1 = R_[:, 0].min(), R_[:, 0].max(), R_[:, 1].min(), R_[:, 1].max()
                if abs((x1 - x0) * (y1 - y0) - A_) <= 0.01 * A_ and min(x1 - x0, y1 - y0) >= 2.0:
                    rects.append((s, float(x0), float(x1), float(y0), float(y1)))
        def walls_at(p, st):
            hits = []
            for o in c.walls:
                if c.storey_of(o) != st:
                    continue
                bb = c.bbox(o)
                if bb is not None and bb[0][0] - 0.03 <= p[0] <= bb[1][0] + 0.03 and bb[0][1] - 0.03 <= p[1] <= bb[1][1] + 0.03:
                    hits.append(o)
            return hits
        for s, x0, x1, y0, y1 in sample(rects, k):
            st = c.storey_of(s); hs = c.height(s); bbs = c.bbox(s)
            if st is None or hs is None or bbs is None:
                continue
            t = rng.choice([0.1, 0.12, 0.15]); along_x = (x1 - x0) >= (y1 - y0); z0 = round(float(bbs[0][2]), 3); hh = round(hs, 3)
            if along_x:
                ym = round((y0 + y1) / 2, 3); lo = [round(x0, 3), round(ym - t / 2, 3), z0]; hi = [round(x1, 3), round(ym + t / 2, 3), round(z0 + hh, 3)]
                ends = (np.array([x0 - 0.02, ym]), np.array([x1 + 0.02, ym]))
            else:
                xm = round((x0 + x1) / 2, 3); lo = [round(xm - t / 2, 3), round(y0, 3), z0]; hi = [round(xm + t / 2, 3), round(y1, 3), round(z0 + hh, 3)]
                ends = (np.array([xm, y0 - 0.02]), np.array([xm, y1 + 0.02]))
            wa, wb = walls_at(ends[0], st), walls_at(ends[1], st)
            if len(wa) != 1 or len(wb) != 1 or wa[0] == wb[0]:
                continue
            deps = [{"guid": wa[0].GlobalId, "rule": "connected_new", "cls": "IfcWall"}, {"guid": wb[0].GlobalId, "rule": "connected_new", "cls": "IfcWall"}, {"guid": st.GlobalId, "rule": "new_in_storey", "cls": "IfcWall"}]
            allowed = [wa[0].GlobalId, wb[0].GlobalId, st.GlobalId]; sbx = ""
            if rng.random() < 0.5:
                deps.append({"guid": s.GlobalId, "rule": "space_boundary_new", "cls": "IfcWall"}); allowed.append(s.GlobalId); sbx = f" and add an IfcRelSpaceBoundary between the space with GlobalId {s.GlobalId} and the new wall"
            L_, W_ = round(hi[0] - lo[0], 3), round(hi[1] - lo[1], 3)
            geo = rng.choice([f"at ({lo[0]}, {lo[1]}, {lo[2]}) with a bounding box given by the distances x={L_}, y={W_}, z={hh}",
                              f"on the level with the ID {st.GlobalId} at ({lo[0]}, {lo[1]}) with length, width and height of the bounding box ({L_}, {W_}, {hh}), its base at z = {z0}"])
            add("C-create-wall-box-D", f"Create a wall {geo} (metres, project coordinates). Connect it to the walls with IDs {wa[0].GlobalId} and {wb[0].GlobalId} (IfcRelConnectsPathElements), assign it to the storey {st.GlobalId} (IfcRelContainedInSpatialStructure){sbx}. {NC}",
                [st.GlobalId], allowed, {"type": "create_wall_bbox", "lo": lo, "hi": hi, "storey": st.GlobalId, "tol": 0.05}, deps, {"style": "D", "anchors": sorted(set(allowed))}, ["IfcWall", "IfcWallStandardCase"])
        # 3 rooms: by bounding box (the removed spaces of arc_v4), merge, bbox shrink, relative height
        for rs in sample(rec.get("removed_spaces", []), k):
            P = np.array(rs["polygon"]); st = f.by_guid(rs["storey"])
            x0_, x1_, y0_, y1_ = P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max()
            if len(P) != 4 or abs((x1_ - x0_) * (y1_ - y0_) - abs(_poly_area(P))) > 0.01 * abs(_poly_area(P)):
                continue
            z0 = elev2(st); L_, W_ = round(float(x1_ - x0_), 3), round(float(y1_ - y0_), 3); hh = rs["height"]
            ws = rs["walls"][: rng.choice([2, 3, 4])]
            add("C-create-space-box-D", f"Create a room (IfcSpace) at ({round(float(x0_), 3)}, {round(float(y0_), 3)}, {z0}) that spans the bounding box given by distances x={L_} m, y={W_} m, z={hh} m, assigned to the storey {st.GlobalId}. Add IfcRelSpaceBoundary relationships between the new room and the walls with IDs {lst(ws)}. {NC}",
                [st.GlobalId], [st.GlobalId] + ws, {"type": "create_space", "polygon": [[round(float(x), 3), round(float(y), 3)] for x, y in P], "height": hh, "storey": st.GlobalId, "z0": z0, "tol": 0.1},
                [{"guid": g, "rule": "new_space_bounds"} for g in ws], {"style": "D", "anchors": [st.GlobalId] + ws}, ["IfcSpace"])
        from shapely.geometry import Polygon as _SP
        from shapely.ops import unary_union as _uu
        pairs = [(a, b, next(iter(c.space_walls[a] & c.space_walls[b]))) for a in sorted(c.space_walls) for b in sorted(c.space_walls) if a < b and len(c.space_walls[a] & c.space_walls[b]) == 1]
        for a, b, wsep in sample(pairs, k):
            fa, fb = c.fp(a), c.fp(b); W = f.by_guid(wsep); pw = c.bbox(W); ha = c.height(f.by_guid(a))
            if fa is None or fb is None or pw is None or ha is None or a not in c.poly:   # the kept room is re-profiled: extrusion body needed
                continue
            wops = _openings(W); wfl = [x for o_ in wops for x in _fills(o_)]
            wpoly = _SP([(pw[0][0], pw[0][1]), (pw[1][0], pw[0][1]), (pw[1][0], pw[1][1]), (pw[0][0], pw[1][1])])
            U = _uu([fa, fb, wpoly.intersection(fa.buffer(0.5)).intersection(fb.buffer(0.5))]).buffer(0.03).buffer(-0.03)
            if U.geom_type != "Polygon" or U.interiors or U.area > fa.area + fb.area + wpoly.area + 0.5:
                continue
            others = sorted(c.space_walls[b] - {wsep} - c.space_walls[a])
            if not others:
                continue
            poly = [[round(float(x), 3), round(float(y), 3)] for x, y in list(U.simplify(0.02).exterior.coords)[:-1]]
            dtxt = f"the separating wall {wsep}" + (f" (with its openings and the doors / windows in them: {lst([x.GlobalId for x in wfl])})" if wfl else "")
            ra = c.ref_space(f.by_guid(a), styles=("S",)) if R3 else (f"the room (IfcSpace) with ID {a}", "D", {})   # T lists bounding walls next to the new-boundary walls: ambiguous
            if ra is None:
                continue
            add(f"U-merge-space-{ra[1]}", f"Extend {ra[0]} so that it also covers the room with ID {b} on the other side of the wall {wsep}: delete {dtxt} and the room {b}, extend the geometry of the extended room over both areas (keeping its height), and add IfcRelSpaceBoundary relationships between it and the walls {lst(others)}. {NC}",
                [a], [a, b, wsep] + [o_.GlobalId for o_ in wops] + [x.GlobalId for x in wfl] + others, {"type": "space_footprint", "guid": a, "polygon": poly, "height": round(ha, 3), "tol": 0.05},
                [{"guid": b, "rule": "removed"}, {"guid": wsep, "rule": "removed"}] + [{"guid": x, "rule": "removed"} for x in [o_.GlobalId for o_ in wops] + [y.GlobalId for y in wfl]]
                + [{"guid": a, "rule": "space_boundary_with", "el": g} for g in others], {"style": ra[1], **ra[2]} if R3 else {"style": "D", "anchors": [b, wsep] + others})
        for s, x0, x1, y0, y1 in sample(rects, k):
            r = c.ref_space(s, ("D", "T", "S")); d = rng.choice([0.5, 1.0, 1.5])
            if r is None or (x1 - x0) - d < 1.5:
                continue
            add(f"U-shrink-space-{r[1]}", f"Reduce the x-length of the bounding box of {r[0]} by {d} m, keeping its edge with the smallest X (x = {round(x0, 3)}) fixed and keeping its Y extent and height. {NC}",
                [s.GlobalId], [s.GlobalId], {"type": "bbox", "guid": s.GlobalId, "lo": [round(x0, 3), round(y0, 3)], "hi": [round(x1 - d, 3), round(y1, 3)], "tol": 0.05}, [], {"style": r[1], **r[2]})
        for s in sample(c.spaces, k):
            r = c.ref_space(s); bb = c.bbox(s)
            if r is None or bb is None:
                continue
            h0 = float(bb[1][2] - bb[0][2]); d = rng.choice([0.3, 0.5]); up = rng.random() < 0.5; hn = round(h0 + (d if up else -d), 3)
            add(f"U-space-height-rel-{r[1]}", f"{'Increase' if up else 'Reduce'} the height of {r[0]} by {d} m, keeping its floor level (bottom at z = {round(float(bb[0][2]), 3)}) and footprint. {NC}",
                [s.GlobalId], [s.GlobalId], {"type": "resize_height", "guid": s.GlobalId, "height": hn, "bottom": round(float(bb[0][2]), 3), "tol": 0.02}, [], {"style": r[1], **r[2]})
        # 4 walls: one-sided thickness; shorten + delete the connection at that end
        for w in sample([w for w in c.walls if c.aligned[w.GlobalId] and _wall_thickness(w) > 0], k):
            r = c.ref_wall(w, styles=("D", "T", "S")); v = _mesh(w)
            if r is None or v is None:
                continue
            u = c.udir(w); nvec = np.array([-u[1], u[0]]); nvec = nvec if abs(nvec[0]) > 0.9 or abs(nvec[1]) > 0.9 else None
            if nvec is None:
                continue
            nvec = np.round(nvec) * rng.choice([1, -1]); t_old = c.thick(w); t_new = rng.choice([x for x in (0.15, 0.2, 0.24, 0.3) if abs(x - t_old) > 0.03])
            fixed = round(float((v[:, :2] @ nvec).min()), 3); axis = "x" if abs(nvec[0]) > 0.5 else "y"; sign = "+" if nvec[0] + nvec[1] > 0 else "-"
            add(f"U-set-wall-thickness-side-{r[1]}", f"Change the thickness of {r[0]} to {t_new} m in {sign}{axis}-direction: its face on the {'-' if sign == '+' else '+'}{axis} side stays where it is and only the opposite face moves. Preserve all relationships; its openings must stay hosted. {NC}",
                [w.GlobalId], [w.GlobalId] + [o.GlobalId for o in _openings(w)] + [x.GlobalId for o in _openings(w) for x in _fills(o)] + sorted(c.conn[w.GlobalId]) + sorted(c.bounded[w.GlobalId]),
                {"type": "set_thickness_side", "guid": w.GlobalId, "thickness": t_new, "n": [float(nvec[0]), float(nvec[1])], "fixed": fixed, "tol": 0.02},
                [{"guid": o.GlobalId, "rule": "stays_hosted"} for o in _openings(w)], {"style": r[1], **r[2]})
        for w in sample([w for w in c.walls if c.aligned[w.GlobalId] and not _openings(w) and c.length(w) >= 3.0 and len(c.conn[w.GlobalId]) >= 2], k):
            a, b = c.axis[w.GlobalId]; d = rng.choice([0.5, 1.0])
            def seg_d(p, q0, q1):
                vv = q1 - q0; tt = np.clip(np.dot(p - q0, vv) / (np.dot(vv, vv) or 1e-12), 0, 1); return float(np.linalg.norm(p - (q0 + tt * vv)))
            for end, fx in ((a, b), (b, a)):
                at_end = [o for o in c.conn[w.GlobalId] if seg_d(end, *c.axis[o]) < 0.3]; at_fix = [o for o in c.conn[w.GlobalId] if seg_d(fx, *c.axis[o]) < 0.3]
                if len(at_end) == 1 and at_fix and not (set(at_end) & set(at_fix)):
                    break
            else:
                continue
            new_end = end - (end - fx) / (np.linalg.norm(end - fx) or 1.0) * d; axk = 0 if abs(c.udir(w)[0]) > 0.98 else 1
            rw = c.ref_wall(w, styles=("T", "S")) if R3 else (f"the wall with ID {w.GlobalId}", "D", {})
            if rw is None:
                continue
            add(f"U-shorten-wall-delconn-{rw[1]}", f"Decrease the length of {rw[0]} by {d} m fixing one side at {'xy'[axk]}={round(float(fx[axk]), 3)}. Delete its connecting relationship to the wall {at_end[0]} (at the shortened end). Make sure that the other relationships stay consistent. {NC}",
                [w.GlobalId], [w.GlobalId, at_end[0]] + at_fix + sorted(c.bounded[w.GlobalId]), {"type": "wall_axis", "guid": w.GlobalId, "p0": fmt(fx), "p1": fmt(new_end), "tol": 0.03},
                [{"guid": at_end[0], "rule": "not_connected", "el": w.GlobalId}] + [{"guid": g, "rule": "rel_connects", "el": w.GlobalId} for g in at_fix], {"style": rw[1], **rw[2]} if R3 else {"style": "D", "anchors": [at_end[0]]})
        # 5 windows / doors: move to the wall below, widen the opening, make taller, create by the opening's bbox
        below_of = {}
        for wl in c.walls:
            up_ = c.above(wl)
            if up_ is not None:
                below_of[up_.GlobalId] = wl
        for win in sample([x for x in c.windows if c.host(x) is not None and c.host(x)[1].GlobalId in below_of], k):
            op, wu = c.host(win); wl = below_of[wu.GlobalId]; bo = c.bbox(op); bu, bl = c.bbox(wu), c.bbox(wl); bw = c.bbox(win)
            if bo is None or bu is None or bl is None or bw is None:
                continue
            if any(c.bbox(o_) is not None and c.bbox(o_)[1][2] > bl[0][2] and abs(float(((c.bbox(o_)[0] + c.bbox(o_)[1]) / 2 - (bo[0] + bo[1]) / 2)[:2] @ c.udir(wl))) < 1.5 for o_ in _openings(wl)):
                continue      # the wall below already has an opening near that position
            zmin = round(float(bo[0][2] - bu[0][2] + bl[0][2]), 3); xy = fmt((bw[0][:2] + bw[1][:2]) / 2)
            add("U-rehost-window-D", f"Move the window with ID {win.GlobalId} down to the wall with ID {wl.GlobalId}. Keep its plan position and dimensions and only adjust the z-values (min z of the new opening = {zmin} m). Also make sure the opening voids the new wall (IfcRelVoidsElement) instead of {wu.GlobalId} and the window stays in its opening. {NC}",
                [win.GlobalId], [win.GlobalId, op.GlobalId, wu.GlobalId, wl.GlobalId], {"type": "rehost_fill", "guid": win.GlobalId, "host": wl.GlobalId, "xy": xy, "zmin": zmin, "tol": 0.05}, [],
                {"style": "D", "anchors": [wl.GlobalId, wu.GlobalId]})
        for win in sample([x for x in c.windows if c.host(x) is not None and c.aligned[c.host(x)[1].GlobalId]], k):    # move to the facing parallel wall
            op, wu = c.host(win); bw = c.bbox(win); bo = c.bbox(op)
            if bw is None or bo is None:
                continue
            u = c.udir(wu); nrm = np.array([-u[1], u[0]]); cen = (bw[0][:2] + bw[1][:2]) / 2; target = None; best = None
            for sg in c.bounded[wu.GlobalId]:          # the facing wall of the window's room: parallel, opposite side, covering the window position
                for og in c.space_walls.get(sg, ()):
                    if og == wu.GlobalId:
                        continue
                    o_ = f.by_guid(og); oa, ob_ = c.axis[og]; ou = (ob_ - oa) / (np.linalg.norm(ob_ - oa) or 1.0)
                    if abs(float(ou @ u)) < 0.98 or not c.aligned[og]:
                        continue
                    dist = abs(float((oa - cen) @ nrm)); s0 = float((cen - oa) @ ou)
                    if dist < 1.0 or not (0.5 < s0 < float(np.linalg.norm(ob_ - oa)) - 0.5):
                        continue
                    if best is None or dist < best:
                        best, target = dist, o_
            if target is None:
                continue
            ta, tb = c.axis[target.GlobalId]; tu = (tb - ta) / (np.linalg.norm(tb - ta) or 1.0); s_ = float((cen - ta) @ tu); half = float(abs((bo[1][:2] - bo[0][:2]) @ tu)) / 2
            if s_ - half < 0.2 or s_ + half > float(np.linalg.norm(tb - ta)) - 0.2:
                continue
            busy = [(float((_mesh(o2)[:, :2] @ tu).min()), float((_mesh(o2)[:, :2] @ tu).max())) for o2 in _openings(target) if _mesh(o2) is not None]
            p_ = float(ta @ tu) + s_
            if any(min(p_ + half, hi_) - max(p_ - half, lo_) > -0.1 for lo_, hi_ in busy):
                continue
            new_c = ta + tu * s_; xy = fmt(new_c); zmin = round(float(bo[0][2]), 3)
            rf = c.ref_fill(win, "window", styles=("T", "S", "P")) if R3 else (f"the window with ID {win.GlobalId}", "D", {})
            if rf is None:
                continue
            add(f"U-rehost-window-{rf[1]}", f"Move {rf[0]} to the parallel wall with ID {target.GlobalId} on the other side of the room: place it straight across, with its centre at ({xy[0]}, {xy[1]}) on that wall, keeping its dimensions and its height (min z of the opening = {zmin} m). The opening must void the new wall (IfcRelVoidsElement) instead of {wu.GlobalId} and the window must stay in its opening. {NC}",
                [win.GlobalId], [win.GlobalId, op.GlobalId, wu.GlobalId, target.GlobalId], {"type": "rehost_fill", "guid": win.GlobalId, "host": target.GlobalId, "xy": xy, "zmin": zmin, "tol": 0.05}, [],
                ({"style": rf[1], **rf[2], "variant": "across"} if R3 else {"style": "D", "anchors": [target.GlobalId, wu.GlobalId], "variant": "across"}))
        for win in sample([x for x in c.windows + c.doors if c.host(x) is not None and c.aligned[c.host(x)[1].GlobalId]], k):
            op, wall = c.host(win); vo = _mesh(op); u = c.udir(wall); u = np.round(u) if abs(u[0]) > 0.98 or abs(u[1]) > 0.98 else None
            if vo is None or u is None:
                continue
            pu = vo[:, :2] @ u; w0 = float(pu.max() - pu.min()); cen = float((pu.max() + pu.min()) / 2); wn = round(w0 + rng.choice([0.2, 0.4]), 3)
            others = [(_mesh(o_)[:, :2] @ u) for o_ in _openings(wall) if o_ != op and _mesh(o_) is not None]
            vw = _mesh(wall)
            if vw is None:
                continue
            wu_ = vw[:, :2] @ u
            if any(min(cen + wn / 2, p_.max()) - max(cen - wn / 2, p_.min()) > -0.05 for p_ in others) or cen - wn / 2 < wu_.min() + 0.1 or cen + wn / 2 > wu_.max() - 0.1:
                continue
            kind = "window" if win.is_a("IfcWindow") else "door"
            rf = c.ref_fill(win, kind, styles=("T", "S", "P")) if R3 else (f"the {kind} with ID {win.GlobalId}", "D", {})
            if rf is None:
                continue
            add(f"U-widen-opening-{rf[1]}", f"Change the width of the opening for {rf[0]} to {wn} m equally to both sides (its centre along the wall stays where it is). The {kind} must stay in the opening and the opening must stay in its wall. {NC}",
                [win.GlobalId], [win.GlobalId, op.GlobalId, wall.GlobalId], {"type": "opening_width", "guid": win.GlobalId, "u": [float(u[0]), float(u[1])], "width": wn, "centre_u": round(cen, 3), "tol": 0.03},
                [{"guid": op.GlobalId, "rule": "stays_hosted"}], {"style": rf[1], **rf[2], "kind": kind})
        for win in sample([x for x in c.windows if c.host(x) is not None], k):
            r = c.ref_fill(win, "window", styles=("D", "S")); bw = c.bbox(win); op, wall = c.host(win); bwall = c.bbox(wall); bo = c.bbox(op)
            if r is None or bw is None or bwall is None or bo is None:
                continue
            d = rng.choice([0.2, 0.3, 0.5]); hn = round(float(bw[1][2] - bw[0][2]) + d, 3)
            if bw[0][2] + hn > bwall[1][2] - 0.1:
                continue
            add(f"U-window-taller-{r[1]}", f"Increase the height of {r[0]} by {d} m, keeping its sill (bottom at z = {round(float(bw[0][2]), 3)}) fixed, and enlarge its opening so that the window still fits in it. {NC}",
                [win.GlobalId], [win.GlobalId, op.GlobalId, wall.GlobalId], {"type": "resize_height", "guid": win.GlobalId, "height": hn, "bottom": round(float(bw[0][2]), 3), "tol": 0.02},
                [{"guid": win.GlobalId, "rule": "opening_covers"}, {"guid": op.GlobalId, "rule": "stays_hosted"}], {"style": r[1], **r[2]})
        for w in sample([w for w in c.walls if c.aligned[w.GlobalId] and c.length(w) >= 2.5], k):
            st = c.storey_of(w); bwl = c.bbox(w); v = _mesh(w)
            if st is None or bwl is None or v is None:
                continue
            u = np.round(c.udir(w)); pu = v[:, :2] @ u; busy = [(float((_mesh(o_)[:, :2] @ u).min()), float((_mesh(o_)[:, :2] @ u).max())) for o_ in _openings(w) if _mesh(o_) is not None]
            kind = rng.choice(["door", "window"]); W = rng.choice([0.9, 1.0]) if kind == "door" else rng.choice([0.8, 1.2]); H = 2.1 if kind == "door" else 1.2; sill = 0.0 if kind == "door" else 0.9
            slot = None; x = float(pu.min()) + 0.3
            while x + W <= float(pu.max()) - 0.3:
                if all(x + W <= lo_ - 0.1 or x >= hi_ + 0.1 for lo_, hi_ in busy):
                    slot = x; break
                x += 0.1
            if slot is None:
                continue
            n_ = np.array([-u[1], u[0]]); pn = v[:, :2] @ n_; Tw = float(pn.max() - pn.min()); ccen = u * (slot + W / 2) + n_ * float((pn.max() + pn.min()) / 2)
            z0 = round(float(bwl[0][2]) + sill, 3); lx = round(W if abs(u[0]) > 0.5 else Tw, 3); ly = round(Tw if abs(u[0]) > 0.5 else W, 3)
            mn = (round(float(ccen[0] - lx / 2), 3), round(float(ccen[1] - ly / 2), 3))
            rw = c.ref_wall(w, styles=("T", "S")) if R3 else (f"the wall with the ID {w.GlobalId}", "D", {})
            if rw is None:
                continue
            add(f"C-create-{kind}-box-{rw[1]}", f"Create a {kind} in {rw[0]}. The bounding box of its opening starts at ({mn[0]}, {mn[1]}, {z0}) with length, width and height ({lx}, {ly}, {H}). Create the opening (IfcOpeningElement), add IfcRelVoidsElement between the wall and the opening and IfcRelFillsElement between the opening and the {kind}, and assign the {kind} to the storey {st.GlobalId}. {NC}",
                [w.GlobalId], [w.GlobalId, st.GlobalId], {"type": "create_fill", "host": w.GlobalId, "kind": kind, "width": W, "height": H, "sill": sill, "centre": [round(float(ccen[0]), 3), round(float(ccen[1]), 3)], "tol": 0.05},
                [{"guid": st.GlobalId, "rule": "new_in_storey", "cls": "IfcDoor" if kind == "door" else "IfcWindow"}], {"style": rw[1], **rw[2], "kind": kind}, ["IfcDoor" if kind == "door" else "IfcWindow", "IfcOpeningElement"])
        # 6 slabs: opening cut, slab on the tops of columns, slab laid on another slab
        for sl in sample([s for s in c.slabs if not _openings(s)], k):
            st = c.storey_of(sl); cand = [s_ for s_ in c.spaces if c.storey_of(s_) == st] or [s_ for s_ in c.spaces]
            fp = None
            for s_ in rng.sample(cand, min(4, len(cand))):
                p = c.rep_point(s_.GlobalId, 1.2)
                if p is not None:
                    fp = p; break
            if fp is None:
                continue
            a_, b_ = rng.choice([(1.0, 1.0), (1.2, 0.8), (0.8, 2.0), (1.5, 1.5)]); lo = [round(float(fp[0] - a_ / 2), 3), round(float(fp[1] - b_ / 2), 3)]; hi = [round(float(fp[0] + a_ / 2), 3), round(float(fp[1] + b_ / 2), 3)]
            rs_ = c.ref_slab(sl, styles=("T", "S")) if R3 else (f"the slab with ID {sl.GlobalId}", "D", {})
            if rs_ is None:
                continue
            add(f"C-create-slab-opening-{rs_[1]}", f"Add an opening (IfcOpeningElement) of size {a_} m (X) by {b_} m (Y) to {rs_[0]}, centred at ({round(float(fp[0]), 3)}, {round(float(fp[1]), 3)}), cutting through the full slab thickness, and relate it to the slab with IfcRelVoidsElement. {NC}",
                [sl.GlobalId], [sl.GlobalId], {"type": "create_slab_opening", "host": sl.GlobalId, "lo": lo, "hi": hi, "tol": 0.05}, [], {"style": rs_[1], **rs_[2]}, ["IfcOpeningElement"])
        cols_by = {}
        for col in c.columns:
            st_ = c.storey_of(col)
            if st_ is not None:
                cols_by.setdefault(st_.GlobalId, []).append(col)
        for st_g, cols in sample(list(cols_by.items()), k):
            if len(cols) < 2:
                continue
            pick = rng.sample(cols, min(len(cols), rng.choice([2, 3, 4]))); bbs_ = [c.bbox(x) for x in pick]
            if any(b_ is None for b_ in bbs_) or max(b_[1][2] for b_ in bbs_) - min(b_[1][2] for b_ in bbs_) > 0.01:
                continue
            ztop = round(float(bbs_[0][1][2]), 3); t = rng.choice([0.2, 0.25]); x0_, y0_ = min(b_[0][0] for b_ in bbs_), min(b_[0][1] for b_ in bbs_); x1_, y1_ = max(b_[1][0] for b_ in bbs_), max(b_[1][1] for b_ in bbs_)
            if (x1_ - x0_) < 1.0 or (y1_ - y0_) < 1.0:
                continue
            poly = [[round(float(x0_), 3), round(float(y0_), 3)], [round(float(x1_), 3), round(float(y0_), 3)], [round(float(x1_), 3), round(float(y1_), 3)], [round(float(x0_), 3), round(float(y1_), 3)]]
            ids = [x.GlobalId for x in pick]
            add("C-create-slab-on-columns-D", f"Create a slab (IfcSlab) {t} m thick that is placed on the upper outer edges of the columns with IDs {lst(ids)}: its bottom lies on their tops (z = {ztop}) and its footprint is the bounding rectangle of the columns' outlines. Assign it to the storey {st_g}. {NC}",
                [st_g], [st_g], {"type": "create_slab", "polygon": poly, "thickness": t, "storey": st_g, "top": round(ztop + t, 3), "tol": 0.03},
                [{"guid": st_g, "rule": "new_in_storey", "cls": "IfcSlab"}], {"style": "D", "anchors": ids + [st_g]}, ["IfcSlab"])
        for i_ in range(1, len(c.storeys)):
            lo_st, up_st = c.storeys[i_ - 1], c.storeys[i_]
            A_ = [s_ for s_ in c.slabs if c.storey_of(s_) == up_st]; B_ = [s_ for s_ in c.slabs if c.storey_of(s_) == lo_st]
            if len(A_) != 1 or len(B_) != 1 or rng.random() > 0.7:
                continue
            ba, bb_ = c.bbox(A_[0]), c.bbox(B_[0])
            if ba is None or bb_ is None:
                continue
            dz = round(float(bb_[1][2] - ba[0][2]), 3)
            if abs(dz) < 0.3:
                continue
            ops = _openings(A_[0])
            if R3:
                ta_, tb_ = f"the floor slab (IfcSlab) contained in {c.storey_ref(up_st)}", f"the floor slab contained in {c.storey_ref(lo_st)}"; sty = "T"
            else:
                ta_, tb_ = f"the slab with ID {A_[0].GlobalId}", f"the slab with ID {B_[0].GlobalId}"; sty = "D"
            add(f"U-slab-on-slab-{sty}", f"Change the position of {ta_} so that it lies on {tb_}: move it vertically by {dz} m so that its bottom is at the top of that slab (z = {round(float(bb_[1][2]), 3)}), keeping its footprint, thickness and storey. {NC}",
                [A_[0].GlobalId], [A_[0].GlobalId] + [o.GlobalId for o in ops], {"type": "translate", "guid": A_[0].GlobalId, "vector": [0.0, 0.0, dz], "tol": 0.02},
                [{"guid": o.GlobalId, "rule": "translate_with", "vector": [0.0, 0.0, dz]} for o in ops], {"style": sty, "anchors": [B_[0].GlobalId] if sty == "D" else [up_st.GlobalId, lo_st.GlobalId]})
        if R3:   # geometric-reasoning edits (BIBIMBAP style): no target numbers given, the agent derives them
            for i_ in range(len(c.storeys) - 1):
                st, nxt = c.storeys[i_], c.storeys[i_ + 1]; cols = cols_by.get(st.GlobalId, []); sl_up = [s_ for s_ in c.slabs if c.storey_of(s_) == nxt]
                if len(cols) < 2 or len(sl_up) != 1:
                    continue
                bbs_ = [c.bbox(x) for x in cols]; bu = c.bbox(sl_up[0])
                if any(b_ is None for b_ in bbs_) or bu is None or max(b_[0][2] for b_ in bbs_) - min(b_[0][2] for b_ in bbs_) > 0.01:
                    continue
                hn = round(float(bu[0][2] - bbs_[0][0][2]), 3)
                if hn <= 0.5 or all(abs(float(b_[1][2] - b_[0][2]) - hn) < 0.02 for b_ in bbs_):
                    continue
                add("U-columns-to-slab-D", f"Change the height of all columns (IfcColumn) contained in {c.storey_ref(st)} so that they touch the slab with ID {sl_up[0].GlobalId}: each column's top must reach the bottom of that slab; keep their bases and cross-sections. {NC}",
                    [st.GlobalId] + [x.GlobalId for x in cols], [x.GlobalId for x in cols], {"type": "resize_height_all", "guids": [x.GlobalId for x in cols], "height": hn, "tol": 0.02}, [], {"style": "D", "anchors": [sl_up[0].GlobalId, st.GlobalId]})
            for sl in sample([s_ for s_ in c.slabs if not _openings(s_)], k):
                bb = c.bbox(sl); r = c.ref_slab(sl, styles=("D", "T", "S")); R_ = profile_world_polygon(sl)
                if bb is None or r is None or R_ is None or len(R_) != 4:
                    continue
                dx, dy = float(bb[1][0] - bb[0][0]), float(bb[1][1] - bb[0][1])
                if abs(dx - dy) < 0.5 or abs(dx * dy - abs(_poly_area(np.array(R_)[:, :2] * sc))) > 0.01 * dx * dy:
                    continue
                lo, hi = [float(bb[0][0]), float(bb[0][1])], [float(bb[1][0]), float(bb[1][1])]
                if dx < dy:
                    hi[0] = lo[0] + dy
                else:
                    hi[1] = lo[1] + dx
                add(f"U-slab-square-{r[1]}", f"Increase the length of one side of {r[0]} so that it forms a square: extend its shorter side, keeping the edge with the smaller coordinate on that axis fixed, and keep its thickness and elevation. {NC}",
                    [sl.GlobalId], [sl.GlobalId], {"type": "bbox", "guid": sl.GlobalId, "lo": fmt(lo), "hi": fmt(hi), "tol": 0.05}, [], {"style": r[1], **r[2]})
            for st_g, cols in cols_by.items():
                st = f.by_guid(st_g); sps = [s_ for s_ in c.spaces if c.storey_of(s_) == st]
                inside = {s_.GlobalId: [x for x in cols if c.contains(s_.GlobalId, c.col_centre(x), 0.05)] for s_ in sps}
                full = [g for g, v in inside.items() if v]
                if not full:
                    continue
                g = rng.choice(full); rsp = c.ref_space(f.by_guid(g), styles=("D", "S"))
                if rsp is None:
                    continue
                ids = sorted(x.GlobalId for x in inside[g])
                add(f"D-delete-columns-in-space-{rsp[1]}", f"Delete the columns (IfcColumn) that stand inside {rsp[0]}. {NC}",
                    [g] + ids, ids, {"type": "delete_all", "guids": ids}, [], {"style": rsp[1], **rsp[2]})
                keep = rng.sample(full, min(2, len(full))); out_ = sorted(x.GlobalId for x in cols if not any(x in inside[k_] for k_ in keep))
                if out_ and len(out_) < len(cols):
                    add("D-delete-columns-outside-D", f"Delete the columns (IfcColumn) contained in {c.storey_ref(st)} that are outside of the rooms with GlobalIds {lst(keep)}. {NC}",
                        [st_g] + out_, out_, {"type": "delete_all", "guids": out_}, [], {"style": "D", "anchors": keep + [st_g]})

    # ------------------------------------------------------------------ COVER4:
    # the operations behind an earlier student's BIM-Edit / BIBIMBAP failures: rooms referenced by their bounding walls (merge, create by
    # enclosure), merge deleting "the elements that separate", room extended to a coordinate; storey length (its floor slab); slab lying
    # directly above the walls of a storey; slab under the window walls / the walls of a room; column moved towards the nearest exterior
    # wall; column on top of a column; wall centred between two facing walls; windows of the walls facing a compass direction (a window
    # aligned with the one above was dropped: stacked windows of the synthetic buildings are always aligned). Generic wording; existing
    # intents and dependency rules only.
    if "cover4" in families:
        from shapely.geometry import Polygon as _SP4
        from shapely.ops import unary_union as _uu4
        def lst4(ids):
            ids = sorted(ids); return ", ".join(ids[:-1]) + (" and " if len(ids) > 1 else "") + ids[-1] if ids else ""
        def elev4(st):
            return round(float(world_matrix(st)[2, 3]) * sc, 3)
        def walls_ref(sg):      # bounding-wall reference: unique if no other space is bounded by all of these walls
            ws = sorted(c.space_walls.get(sg, ()))
            if not (3 <= len(ws) <= 8) or any(set(ws) <= c.space_walls.get(o, set()) for o in c.space_walls if o != sg):
                return None
            return ws
        def rect_of(sg):
            R_ = c.ring(sg); A_ = c.area(sg)
            if R_ is None or not A_ or len(R_) != 4:
                return None
            x0, x1, y0, y1 = float(R_[:, 0].min()), float(R_[:, 0].max()), float(R_[:, 1].min()), float(R_[:, 1].max())
            return (x0, x1, y0, y1) if abs((x1 - x0) * (y1 - y0) - A_) <= 0.01 * A_ else None
        # 1 room merge: both rooms by their bounding walls (T) / "the elements that separate them" (E); room extended to a coordinate (D)
        pairs = [(a, b, next(iter(c.space_walls[a] & c.space_walls[b]))) for a in sorted(c.space_walls) for b in sorted(c.space_walls) if a != b and len(c.space_walls[a] & c.space_walls[b]) == 1]
        merges = []
        for a, b, wsep in pairs:
            fa, fb = c.fp(a), c.fp(b); W = f.by_guid(wsep); pw = c.bbox(W); ha = c.height(f.by_guid(a))
            if fa is None or fb is None or pw is None or ha is None or a not in c.poly:
                continue
            wpoly = _SP4([(pw[0][0], pw[0][1]), (pw[1][0], pw[0][1]), (pw[1][0], pw[1][1]), (pw[0][0], pw[1][1])])
            U = _uu4([fa, fb, wpoly.intersection(fa.buffer(0.5)).intersection(fb.buffer(0.5))]).buffer(0.03).buffer(-0.03)
            if U.geom_type != "Polygon" or U.interiors or U.area > fa.area + fb.area + wpoly.area + 0.5:
                continue
            others = sorted(c.space_walls[b] - {wsep} - c.space_walls[a])
            if others:
                merges.append((a, b, wsep, U, others, ha))
        def merge_task(a, b, wsep, U, others, ha, tpl, prompt, meta):
            W = f.by_guid(wsep); wops = _openings(W); wfl = [x for o_ in wops for x in _fills(o_)]
            poly = [[round(float(x), 3), round(float(y), 3)] for x, y in list(U.simplify(0.02).exterior.coords)[:-1]]
            add(tpl, prompt, [a], [a, b, wsep] + [o_.GlobalId for o_ in wops] + [x.GlobalId for x in wfl] + others, {"type": "space_footprint", "guid": a, "polygon": poly, "height": round(ha, 3), "tol": 0.05},
                [{"guid": b, "rule": "removed"}, {"guid": wsep, "rule": "removed"}] + [{"guid": x, "rule": "removed"} for x in [o_.GlobalId for o_ in wops] + [y.GlobalId for y in wfl]]
                + [{"guid": a, "rule": "space_boundary_with", "el": g} for g in others], meta)
        for a, b, wsep, U, others, ha in sample(merges, k):
            wa_, wb_ = walls_ref(a), walls_ref(b)
            if wa_ is None or wb_ is None:
                continue
            dele = " together with the doors / windows in it and their openings" if _openings(f.by_guid(wsep)) else ""
            merge_task(a, b, wsep, U, others, ha, "U-merge-space-T",
                       f"Extend the room that is bounded by the walls {lst4(wa_)} so that it also covers the room that is bounded by the walls {lst4(wb_)}. Remove the wall between the two rooms{dele}, delete the covered room, extend the geometry of the extended room over both areas keeping its height, and add IfcRelSpaceBoundary relationships between the extended room and the walls that bounded the covered room. {NC}",
                       {"style": "T", "anchors": sorted(set(wa_) | set(wb_))})
        for a, b, wsep, U, others, ha in sample(merges, k):
            merge_task(a, b, wsep, U, others, ha, "U-merge-space-E",
                       f"Extend the room with ID {a} so that it also covers the room with ID {b} by deleting the elements that physically separate those two rooms (the wall between them with any doors / windows and openings in it) and the room {b} itself. Extend the geometry of {a} over both areas keeping its height, and add IfcRelSpaceBoundary relationships between it and the walls that bounded {b}. {NC}",
                       {"style": "D", "anchors": [b]})
        exts = []
        for m_ in merges:
            ra_ = rect_of(m_[0]); minx, miny, maxx, maxy = m_[3].bounds
            if ra_ is None or rect_of(m_[1]) is None or abs((maxx - minx) * (maxy - miny) - m_[3].area) > 0.04 * m_[3].area:
                continue      # the union is not (nearly) a rectangle - small notches at the wall ends - so no single coordinate describes it
            ext = [(ax, sg, v) for ax, sg, v, old in (("x", "-", minx, ra_[0]), ("x", "+", maxx, ra_[1]), ("y", "-", miny, ra_[2]), ("y", "+", maxy, ra_[3])) if abs(v - old) > 0.3]
            if len(ext) == 1:
                exts.append((m_, ext[0]))
        for (a, b, wsep, U, others, ha), (ax, sg, v) in sample(exts, k):
            dele = " (with the doors / windows in it and their openings)" if _openings(f.by_guid(wsep)) else ""
            merge_task(a, b, wsep, U, others, ha, "U-extend-space-D",
                       f"Extend the room with ID {a} in {'negative' if sg == '-' else 'positive'} {ax} direction to {ax}={round(float(v), 3)}, by removing the wall {wsep}{dele} and the room {b} that occupies that area. Keep its height. Update the IfcRelSpaceBoundary relationships so that the room is additionally bounded by the walls {lst4(others)}. {NC}",
                       {"style": "D", "anchors": [wsep, b] + others})
        # 2 room created in the area enclosed by given walls (the spaces removed in arc_v4)
        for rs in sample(rec.get("removed_spaces", []), k):
            P = np.array(rs["polygon"]); st = f.by_guid(rs["storey"]); ws = rs["walls"]
            if len(P) > 8 or not (3 <= len(ws) <= 8):
                continue
            hh = rs["height"]; z0 = elev4(st)
            add("C-create-space-enclosed-D", f"Create a room (IfcSpace) in the area that is enclosed by the walls {lst4(ws)}, {hh} m high, on the storey with GlobalId {st.GlobalId} (its floor at the storey level). Add IfcRelSpaceBoundary relationships between the new room and these walls. {NC}",
                [st.GlobalId], [st.GlobalId] + ws, {"type": "create_space", "polygon": [[round(float(x), 3), round(float(y), 3)] for x, y in P], "height": hh, "storey": st.GlobalId, "z0": z0, "tol": 0.1},
                [{"guid": g, "rule": "new_space_bounds"} for g in ws], {"style": "D", "anchors": [st.GlobalId] + ws}, ["IfcSpace"])
        # 3 storey length = its only floor slab, one end fixed
        for st in sample(c.storeys, k):
            sls = [s_ for s_ in c.slabs if c.storey_of(s_) == st]
            if len(sls) != 1 or _openings(sls[0]):
                continue
            sl = sls[0]; bb = c.bbox(sl); R_ = profile_world_polygon(sl)
            if bb is None or R_ is None or len(R_) != 4:
                continue
            ax = rng.choice([0, 1]); L = float(bb[1][ax] - bb[0][ax]); d = rng.choice([1.0, 2.0, 3.0])
            if L - d < 2.5:
                continue
            keep_min = rng.random() < 0.5; lo, hi = [float(bb[0][0]), float(bb[0][1])], [float(bb[1][0]), float(bb[1][1])]
            if keep_min:
                hi[ax] -= d
            else:
                lo[ax] += d
            add("U-storey-length-T", f"Decrease the length of the building storey with ID {st.GlobalId} in {'xy'[ax]} direction by {d} m, keeping it fixed at the {'smallest' if keep_min else 'highest'} {'xy'[ax]} value. The storey's extent is given by its floor slab (the IfcSlab contained in it): shorten that slab and keep its thickness and elevation; walls and rooms stay as they are. {NC}",
                [sl.GlobalId], [sl.GlobalId], {"type": "bbox", "guid": sl.GlobalId, "lo": fmt(lo), "hi": fmt(hi), "tol": 0.05}, [], {"style": "T", "anchors": [st.GlobalId]})
        # 4 the slab lying on the walls of a storey, extended / shortened on one side
        for st in c.storeys:
            ws = [w for w in c.walls if c.storey_of(w) == st and c.bbox(w) is not None]
            if not ws:
                continue
            ztop = max(float(c.bbox(w)[1][2]) for w in ws); wx0 = min(float(c.bbox(w)[0][0]) for w in ws); wx1 = max(float(c.bbox(w)[1][0]) for w in ws)
            wy0 = min(float(c.bbox(w)[0][1]) for w in ws); wy1 = max(float(c.bbox(w)[1][1]) for w in ws)
            on = sorted((float(c.bbox(s_)[0][2]), s_.GlobalId) for s_ in c.slabs if c.bbox(s_) is not None and ztop - 0.02 <= float(c.bbox(s_)[0][2]) <= ztop + 0.3
                        and min(float(c.bbox(s_)[1][0]), wx1) - max(float(c.bbox(s_)[0][0]), wx0) > 1.0 and min(float(c.bbox(s_)[1][1]), wy1) - max(float(c.bbox(s_)[0][1]), wy0) > 1.0)
            if not on or (len(on) > 1 and on[1][0] - on[0][0] < 0.1) or rng.random() > 0.8:
                continue      # the lowest slab whose underside is within 0.3 m above the walls' top, unambiguous
            on = [f.by_guid(on[0][1])]
            sl = on[0]; bb = c.bbox(sl); ax = rng.choice([0, 1]); neg = rng.random() < 0.5; d = rng.choice([1.0, 1.5, 2.0])
            lo, hi = [float(bb[0][0]), float(bb[0][1])], [float(bb[1][0]), float(bb[1][1])]
            if neg:
                lo[ax] -= d
            else:
                hi[ax] += d
            add("U-slab-above-walls-S", f"Extend the slab that is directly above the walls of {c.storey_ref(st)} by {d} m in {'negative' if neg else 'positive'} {'xy'[ax]}-direction, keeping its opposite edge, its thickness and its elevation. Preserve all its relationships. {NC}",
                [sl.GlobalId], [sl.GlobalId], {"type": "bbox", "guid": sl.GlobalId, "lo": fmt(lo), "hi": fmt(hi), "tol": 0.05}, [], {"style": "S", "anchors": [st.GlobalId]})
        # 5 a slab under selected walls of a storey (the walls hosting a window / the walls bounding a room)
        for st in sample(c.storeys, k):
            ws_st = [w for w in c.walls if c.storey_of(w) == st and c.bbox(w) is not None]
            groups = [("window", None, [w for w in ws_st if any(x.is_a("IfcWindow") for o_ in _openings(w) for x in _fills(o_))])]
            sps = [s_ for s_ in c.spaces if c.storey_of(s_) == st and len(c.space_walls.get(s_.GlobalId, ())) >= 3]
            if sps:
                s_ = rng.choice(sps); groups.append(("room", s_.GlobalId, [w for w in ws_st if w.GlobalId in c.space_walls[s_.GlobalId]]))
            kind, sg, ws = rng.choice(groups)
            if len(ws) < 2:
                continue
            bbs_ = [c.bbox(w) for w in ws]; x0_, y0_ = min(float(b_[0][0]) for b_ in bbs_), min(float(b_[0][1]) for b_ in bbs_); x1_, y1_ = max(float(b_[1][0]) for b_ in bbs_), max(float(b_[1][1]) for b_ in bbs_)
            zb = [float(b_[0][2]) for b_ in bbs_]
            if x1_ - x0_ < 1.5 or y1_ - y0_ < 1.5 or max(zb) - min(zb) > 0.02:
                continue
            t = rng.choice([0.2, 0.25]); top = round(min(zb), 3)
            poly = [[round(x0_, 3), round(y0_, 3)], [round(x1_, 3), round(y0_, 3)], [round(x1_, 3), round(y1_, 3)], [round(x0_, 3), round(y1_, 3)]]
            which = "all walls of that storey that host a window" if kind == "window" else f"the walls of that storey that bound the room with GlobalId {sg}"
            add(f"C-create-slab-under-{kind}-walls-D", f"Create a slab (IfcSlab) {t} m thick for {c.storey_ref(st)} so that it lies exactly below {which} and aligns with them: its footprint is the bounding rectangle of those walls' outlines and its top is at the base of the walls. Assign it to that storey. {NC}",
                [st.GlobalId], [st.GlobalId], {"type": "create_slab", "polygon": poly, "thickness": t, "storey": st.GlobalId, "top": top, "tol": 0.03},
                [{"guid": st.GlobalId, "rule": "new_in_storey", "cls": "IfcSlab"}], {"style": "D", "anchors": [st.GlobalId] + ([sg] if sg else [])}, ["IfcSlab"])
        # 6 columns: moved towards the nearest exterior wall; a new column on top of a column, connected to it
        def segd4(p, w):
            a_, b_ = c.axis[w.GlobalId]; ab = b_ - a_; tt = np.clip(np.dot(p - a_, ab) / (np.dot(ab, ab) or 1e-12), 0, 1); return float(np.linalg.norm(p - (a_ + tt * ab))), a_ + tt * ab
        for col in sample(c.columns, k):
            st = c.storey_of(col); ctr = c.col_centre(col)
            if st is None or ctr is None:
                continue
            ext = sorted((segd4(ctr, w)[0], w.GlobalId) for w in c.walls if c.storey_of(w) == st and c.aligned[w.GlobalId] and c.is_external(w))
            if not ext or (len(ext) > 1 and ext[1][0] - ext[0][0] < 0.3):
                continue
            w = f.by_guid(ext[0][1]); dist, foot = segd4(ctr, w); u = c.udir(w); nrm = np.array([-u[1], u[0]]); nrm = np.round(nrm) if abs(nrm[0]) > 0.98 or abs(nrm[1]) > 0.98 else None
            if nrm is None:
                continue
            nrm = nrm if float((foot - ctr) @ nrm) > 0 else -nrm; d = rng.choice([0.5, 0.75, 1.0, 1.125])
            room = [s_ for s_ in c.spaces if c.storey_of(s_) == st and c.contains(s_.GlobalId, ctr)]
            if dist - d < c.thick(w) / 2 + 0.4 or (room and not c.contains(room[0].GlobalId, ctr + nrm * d, 0.3)):
                continue
            r = c.ref_column(col, styles=("D", "T"))
            if r is None:
                continue
            vec = [round(float(nrm[0] * d), 3), round(float(nrm[1] * d), 3), 0.0]
            add(f"U-move-column-extwall-{r[1]}", f"Move {r[0]} by {d} m in the direction of the nearest exterior wall (perpendicular to that wall, towards it), keeping its size, height and storey. {NC}",
                [col.GlobalId], [col.GlobalId], {"type": "translate", "guid": col.GlobalId, "vector": vec, "tol": 0.03}, [], {"style": r[1], **r[2], "ext_wall": w.GlobalId})
        for col in sample(c.columns, k):
            st = c.storey_of(col); bb = c.bbox(col); ctr = c.col_centre(col)
            if st is None or bb is None or ctr is None:
                continue
            i_ = next((i for i, s_ in enumerate(c.storeys) if s_ == st), None); ztop = round(float(bb[1][2]), 3)
            nxt = c.storeys[i_ + 1] if i_ is not None and i_ + 1 < len(c.storeys) else None
            stn = nxt if nxt is not None and abs(elev4(nxt) - ztop) < 0.35 else st
            sx, sy = round(float(bb[1][0] - bb[0][0]), 3), round(float(bb[1][1] - bb[0][1]), 3); h = rng.choice([1.75, 2.0, 2.5, 3.0])
            ztxt = f", its base at the top of that column (z = {ztop})" if rng.random() < 0.5 else ""
            add("C-create-column-on-top-D", f"Create a column (IfcColumn) that is connected to the column with ID {col.GlobalId} (IfcRelConnectsElements), sits directly on top of it with the same X / Y centre and cross-section{ztxt}, and has a height of {h} m. Assign it to the storey with GlobalId {stn.GlobalId}. {NC}",
                [stn.GlobalId], [col.GlobalId, stn.GlobalId], {"type": "create_column", "xy": [round(float(ctr[0]), 3), round(float(ctr[1]), 3)], "sx": sx, "sy": sy, "height": h, "z0": ztop, "storey": stn.GlobalId, "tol": 0.05},
                [{"guid": col.GlobalId, "rule": "connected_new", "cls": "IfcColumn"}, {"guid": stn.GlobalId, "rule": "new_in_storey", "cls": "IfcColumn"}], {"style": "D", "anchors": [col.GlobalId, stn.GlobalId]}, ["IfcColumn"])
        # 7 wall centred between two facing parallel walls, orthogonal to both, face to face
        cand = []
        for A in c.walls:
            for B in c.walls:
                if A.GlobalId >= B.GlobalId or c.storey_of(A) != c.storey_of(B) or c.storey_of(A) is None or not (c.aligned[A.GlobalId] and c.aligned[B.GlobalId]):
                    continue
                ua, ub = c.udir(A), c.udir(B)
                if abs(float(ua @ ub)) < 0.99:
                    continue
                u = np.round(np.abs(ua)); nrm = np.array([u[1], u[0]]); (a0, a1), (b0, b1) = c.axis[A.GlobalId], c.axis[B.GlobalId]
                sa = sorted((float(a0 @ u), float(a1 @ u))); sb = sorted((float(b0 @ u), float(b1 @ u))); ov = (max(sa[0], sb[0]), min(sa[1], sb[1]))
                gap = float((b0 - a0) @ nrm)
                if ov[1] - ov[0] < 1.5 or not (1.5 <= abs(gap) <= 8.0):
                    continue
                cand.append((A, B, u, nrm, ov, gap))
        for A, B, u, nrm, ov, gap in sample(cand, k):
            st = c.storey_of(A); ba, bb_ = c.bbox(A), c.bbox(B)
            if ba is None or bb_ is None or abs(float(ba[0][2]) - float(bb_[0][2])) > 0.02:
                continue
            t = rng.choice([0.1, 0.12, 0.15]); s_mid = (ov[0] + ov[1]) / 2; na = float(c.axis[A.GlobalId][0] @ nrm); sgn = 1.0 if gap > 0 else -1.0
            fa_ = na + sgn * c.thick(A) / 2; fb_ = na + gap - sgn * c.thick(B) / 2
            if (fb_ - fa_) * sgn < 1.0:
                continue
            p_lo = u * (s_mid - t / 2) + nrm * min(fa_, fb_); p_hi = u * (s_mid + t / 2) + nrm * max(fa_, fb_)
            lo2, hi2 = np.minimum(p_lo, p_hi), np.maximum(p_lo, p_hi); z0 = round(float(ba[0][2]), 3); hh = round(min(float(ba[1][2] - ba[0][2]), float(bb_[1][2] - bb_[0][2])), 3)
            box_ = _SP4([(lo2[0], lo2[1]), (hi2[0], lo2[1]), (hi2[0], hi2[1]), (lo2[0], hi2[1])]).buffer(-0.03)
            def hits(e):
                b2 = c.bbox(e)
                return b2 is not None and b2[1][2] > z0 + 0.1 and b2[0][2] < z0 + hh - 0.1 and box_.intersects(_SP4([(b2[0][0], b2[0][1]), (b2[1][0], b2[0][1]), (b2[1][0], b2[1][1]), (b2[0][0], b2[1][1])]))
            if any(hits(o) for o in c.walls if o not in (A, B)) or any(hits(o) for o in c.columns) or any(hits(o) for o in c.doors + c.windows):
                continue
            lo = [round(float(lo2[0]), 3), round(float(lo2[1]), 3), z0]; hi = [round(float(hi2[0]), 3), round(float(hi2[1]), 3), round(z0 + hh, 3)]
            add("C-create-wall-centred-D", f"Create a wall connected to the walls with IDs {A.GlobalId} and {B.GlobalId} (IfcRelConnectsPathElements), orthogonal to both of them and placed exactly centred relative to both (in the middle of the stretch where they face each other). It runs from the face of one wall to the face of the other, is {t} m thick and {hh} m high, its base at z = {z0}. Assign it to the storey {st.GlobalId}. {NC}",
                [st.GlobalId], [A.GlobalId, B.GlobalId, st.GlobalId], {"type": "create_wall_bbox", "lo": lo, "hi": hi, "storey": st.GlobalId, "tol": 0.05},
                [{"guid": A.GlobalId, "rule": "connected_new", "cls": "IfcWall"}, {"guid": B.GlobalId, "rule": "connected_new", "cls": "IfcWall"}, {"guid": st.GlobalId, "rule": "new_in_storey", "cls": "IfcWall"}],
                {"style": "D", "anchors": [A.GlobalId, B.GlobalId, st.GlobalId]}, ["IfcWall", "IfcWallStandardCase"])
        # 8 windows of the exterior walls facing a compass direction (+Y north, +X east), deleted with their openings
        facing = {}
        for w in c.walls:
            if not (c.aligned[w.GlobalId] and c.is_external(w)) or not c.bounded[w.GlobalId]:
                continue
            a_, b_ = c.axis[w.GlobalId]; mid = (a_ + b_) / 2; u = c.udir(w); nrm = np.array([-u[1], u[0]])
            sides = {float((c.rep_point(sg) - mid) @ nrm) > 0 for sg in c.bounded[w.GlobalId] if c.rep_point(sg) is not None}
            if len(sides) != 1:
                continue
            out = -nrm if sides.pop() else nrm; dname = ("east" if out[0] > 0 else "west") if abs(out[0]) > 0.9 else ("north" if out[1] > 0 else "south")
            facing.setdefault((c.storey_of(w), dname), []).append(w)
        def wins_of(ws):
            return [x for w in ws for o_ in _openings(w) for x in _fills(o_) if x.is_a("IfcWindow")]
        n_all = len(c.windows)
        for _ in range(k):
            dname = rng.choice(["north", "south", "east", "west"]); per_storey = rng.random() < 0.5
            keys = [kk for kk in facing if kk[1] == dname]
            if not keys:
                continue
            st = rng.choice(keys)[0] if per_storey else None
            ws = [w for kk in keys if (st is None or kk[0] == st) for w in facing[kk]]; wins = wins_of(ws)
            if not wins or len(wins) >= n_all:
                continue
            ops = sorted({c.host(x)[0].GlobalId for x in wins}); hosts = sorted({w.GlobalId for w in ws if wins_of([w])})
            where = f" of {c.storey_ref(st)}" if st is not None else ""
            add("D-delete-windows-facing-S", f"Delete the windows that are in the {dname} walls{where} (the exterior walls whose outside faces {dname}; +Y is north and +X is east), together with their openings, so the walls become solid there. The walls themselves remain. {NC}",
                sorted(x.GlobalId for x in wins), sorted(x.GlobalId for x in wins) + ops + hosts, {"type": "delete_all", "guids": sorted(x.GlobalId for x in wins)},
                [{"guid": o_, "rule": "removed"} for o_ in ops] + [{"guid": g, "rule": "unchanged"} for g in hosts], {"style": "S", "direction": dname, "anchors": [st.GlobalId] if st is not None else [], "n": len(wins)})

    # ------------------------------------------------------------------ QTO: IFC base quantities by NAME only
    # (BIBIMBAP numeric tasks name the quantity and expect the take-off definition; every older R-* template spelled the formula out).
    # Values from ifcgym.quantities (IFC4X3 Qto_* definitions), re-derived from the representation by scripts/verify_qto.py.
    if "qto" in families:
        from ifcgym import quantities as Q
        UNIT = {"A": "square metres", "V": "cubic metres", "L": "metres"}
        PLAIN = {"GrossSideArea": "gross side area", "NetSideArea": "net side area", "GrossVolume": "gross volume", "NetVolume": "net volume",
                 "GrossArea": "gross area", "NetArea": "net area", "Perimeter": "perimeter", "GrossFloorArea": "gross floor area",
                 "NetFloorArea": "net floor area", "GrossPerimeter": "gross perimeter", "Height": "height", "Area": "area", "Volume": "volume"}
        def qq(tpl, ref, qset, prop, kind, value, target, meta, tol=0.05):
            ph = rng.choice(["named", "plain"])
            what = (f"the {prop} (IFC base quantity set {qset})" if ph == "named" else f"the {PLAIN[prop]}")
            add(tpl, f"What is {what} of {ref} in {UNIT[kind]}? Compute it from the model geometry. {JSN}", target, [],
                {"type": "answer", "kind": "number", "value": round(float(value), 3), "rel_tol": 0.02, "tol": tol}, [], {**meta, "qto": f"{qset}.{prop}", "qto_phrase": ph}, tier=0)
        wq = {}
        for w in c.walls:
            try:
                wq[w.GlobalId] = Q.wall(c, w)
            except Exception:  # noqa: BLE001
                wq[w.GlobalId] = None
        walls_q = [w for w in c.walls if wq[w.GlobalId]]
        with_op = [w for w in walls_q if wq[w.GlobalId]["n_openings"]]
        for w in sample(with_op, k) + sample(walls_q, max(1, k // 2)):    # mostly walls with openings: there net and gross differ
            r = c.ref_wall(w, styles=("D", "T", "S"))
            if r:
                prop = rng.choice(["GrossSideArea", "NetSideArea", "GrossVolume", "NetVolume"])
                qq(f"R-qto-wall-{prop}-{r[1]}", r[0], "Qto_WallBaseQuantities", prop, "V" if "Volume" in prop else "A", wq[w.GlobalId][prop], [w.GlobalId], {"style": r[1], **r[2]})
        for _ in range(k):
            if len(with_op) >= 2:
                ws_ = rng.sample(with_op, min(len(with_op), rng.choice([2, 3, 4]))); prop = rng.choice(["NetSideArea", "GrossSideArea", "NetVolume"])
                ids = sorted(w.GlobalId for w in ws_)
                qq(f"R-qto-walls-{prop}-D", "the walls with GlobalIds " + ", ".join(ids) + " (their total)", "Qto_WallBaseQuantities", prop,
                   "V" if "Volume" in prop else "A", sum(wq[g][prop] for g in ids), ids, {"style": "D"})
        for st in sample(c.storeys, k):
            ws_ = [w for w in walls_q if c.storey_of(w) == st]
            if ws_ and len(ws_) == sum(1 for w in f.by_type("IfcWall") if c.storey_of(w) == st):
                prop = rng.choice(["NetSideArea", "NetVolume", "GrossVolume"])
                qq(f"R-qto-storey-walls-{prop}-D", f"all walls (IfcWall) contained in {c.storey_ref(st)} (their total)", "Qto_WallBaseQuantities", prop,
                   "V" if "Volume" in prop else "A", sum(wq[w.GlobalId][prop] for w in ws_), [st.GlobalId], {"style": "D"})
            try:
                sq = Q.storey(c, st)
            except Exception:  # noqa: BLE001
                sq = None
            if sq:
                prop = rng.choice(["GrossFloorArea", "NetFloorArea"])
                qq(f"R-qto-storey-{prop}-D", c.storey_ref(st), "Qto_BuildingStoreyBaseQuantities", prop, "A", sq[prop], [st.GlobalId], {"style": "D"}, tol=0.1)
        slabs_q = []
        for sl in c.slabs:
            try:
                qv = Q.slab(sl)
            except Exception:  # noqa: BLE001
                qv = None
            if qv:
                slabs_q.append((sl, qv))
        for sl, qv in sample([x for x in slabs_q if _openings(x[0])], k) + sample(slabs_q, max(1, k // 2)):
            r = c.ref_slab(sl)
            if r:
                prop = rng.choice(["GrossArea", "NetArea", "Perimeter", "GrossVolume", "NetVolume"])
                qq(f"R-qto-slab-{prop}-{r[1]}", r[0], "Qto_SlabBaseQuantities", prop, {"Perimeter": "L"}.get(prop, "V" if "Volume" in prop else "A"), qv[prop], [sl.GlobalId], {"style": r[1], **r[2]})
        for s in sample(c.spaces, k):
            r = c.ref_space(s)
            try:
                qv = Q.space(c, s)
            except Exception:  # noqa: BLE001
                qv = None
            if r and qv:
                prop = rng.choice(["GrossFloorArea", "NetFloorArea", "GrossPerimeter", "GrossVolume", "NetVolume"] + (["Height"] if qv["Height"] is not None else []))
                qq(f"R-qto-space-{prop}-{r[1]}", r[0], "Qto_SpaceBaseQuantities", prop, {"GrossPerimeter": "L", "Height": "L"}.get(prop, "V" if "Volume" in prop else "A"),
                   qv[prop], [s.GlobalId], {"style": r[1], **r[2]})
        for dd in sample(c.doors + c.windows, k):
            h_ = c.host(dd)
            if h_ is None:
                continue
            try:
                qv = Q.opening(c, h_[0])
            except Exception:  # noqa: BLE001
                qv = None
            kind = "door" if dd.is_a("IfcDoor") else "window"; r = c.ref_fill(dd, kind, styles=("D", "S"))
            if qv and r:
                prop = rng.choice(["Area", "Volume"])
                qq(f"R-qto-opening-{prop}-{r[1]}", f"the opening (IfcOpeningElement) that is filled by {r[0]}", "Qto_OpeningElementBaseQuantities", prop,
                   "V" if prop == "Volume" else "A", qv[prop], [dd.GlobalId, h_[0].GlobalId], {"style": r[1], **r[2]}, tol=0.01)   # target[0] = the described fill (reference verifier)
        for sl, _ in slabs_q:
            ops_ = _openings(sl)
            if len(ops_) == 1:
                try:
                    qv = Q.opening(c, ops_[0])
                except Exception:  # noqa: BLE001
                    qv = None
                if qv:
                    prop = rng.choice(["Area", "Volume"])
                    qq(f"R-qto-slab-opening-{prop}-D", f"the opening (IfcOpeningElement) in the slab with GlobalId {sl.GlobalId}", "Qto_OpeningElementBaseQuantities", prop,
                       "V" if prop == "Volume" else "A", qv[prop], [ops_[0].GlobalId], {"style": "D"}, tol=0.01)
    if "compound" in families:      # several independent edits in one instruction (BIM-Edit direct prompts)
        T += compound_tasks(T, rng, tid, sha, {s_.GlobalId for s_ in c.storeys}, n=max(1, per_template))
    return T



COMPOUND_TYPES = {"translate", "set_thickness", "resize_height", "set_property", "set_attribute", "delete", "delete_all", "create_column",
                  "create_fill", "create_slab", "create_wall", "wall_axis", "bbox", "create_space", "resize_height_all"}
CREATE_TYPES = {"create_column", "create_fill", "create_slab", "create_wall", "create_space"}


def compound_tasks(T: list, rng: random.Random, tid, sha: str, storeys: set, n: int = 2) -> list:
    """Compound instructions. Parts = single v4 edit tasks of the same building whose element sets are disjoint (storeys excepted),
    at most one part per creation type (the creation checkers expect exactly one new element of their class). Intent {"type": "all"}:
    strict success needs every part; target / allowed / dependents / new classes are the unions; the parts' templates and metas are kept
    for the oracle and for part-level rewards."""
    NC = "Do not change anything else."
    pool = [t for t in T if t["intent"]["type"] in COMPOUND_TYPES and t["template"][:2] in ("U-", "D-", "C-")]
    if len(pool) < 2:
        return []
    def footprint(t):
        return ({*(t.get("target") or []), *(t.get("allowed") or []), *(d["guid"] for d in t.get("dependents") or [])} - storeys)
    out = []
    for k in (2, 3):
        made = 0
        for _ in range(60 * n):
            if made >= n:
                break
            parts = rng.sample(pool, k) if len(pool) >= k else None
            if not parts:
                break
            fps = [footprint(t) for t in parts]
            if any(fps[i] & fps[j] for i in range(k) for j in range(i + 1, k)):
                continue
            ctypes = [t["intent"]["type"] for t in parts if t["intent"]["type"] in CREATE_TYPES]
            if len(ctypes) != len(set(ctypes)) or len({t["template"] for t in parts}) < k:
                continue
            body = "\n".join(f"{i + 1}. {t['prompt'].replace(NC, '').strip()}" for i, t in enumerate(parts))
            prompt = f"Make the following {k} changes to the model in one go:\n{body}\n{NC}"
            tpl = f"M-compound-{k}"
            new_classes = sorted({c_ for t in parts for c_ in (t.get("allowed_new_classes") or [])})
            task = {"task_id": tid(tpl), "project": parts[0]["project"], "template": tpl, "tier": max(t["tier"] for t in parts),
                    "prompt": prompt, "target": sorted({g for t in parts for g in t.get("target") or []}),
                    "allowed": sorted({g for t in parts for g in t.get("allowed") or []}),
                    "intent": {"type": "all", "parts": [t["intent"] for t in parts]},
                    "dependents": [d for t in parts for d in t.get("dependents") or []],
                    "meta": {"style": "M", "ifc_sha1": sha, "parts": [t["template"] for t in parts], "part_ids": [t["task_id"] for t in parts],
                             "part_meta": [t.get("meta") or {} for t in parts], "anchors": sorted({a for t in parts for a in (t.get("meta") or {}).get("anchors", [])})},
                    "ifc": parts[0]["ifc"]}
            if new_classes:
                task["allowed_new_classes"] = new_classes
            out.append(task); made += 1
    return out
