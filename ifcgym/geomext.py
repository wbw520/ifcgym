"""Geometric exterior-wall test: the CubiCasa importer flagged a wall as exterior only when it lay on the plan's
bounding box, so facades of non-rectangular plans were written with Pset_WallCommon.IsExternal = False. A wall is exterior here when,
at most of three stations along it (25/50/75 %), exactly one of the two probe points just beyond its faces lies inside the storey
outline (union of the room polygons, closed by the thickest wall so the wall strips between rooms are filled, enclosed holes filled)."""
from __future__ import annotations

import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union


def wall_strip(p0, p1, thickness: float):
    from shapely.geometry import LineString
    return LineString([tuple(p0), tuple(p1)]).buffer(thickness / 2, cap_style=2, join_style=2)


def storey_outline(rooms, t_max: float, walls=()):
    """rooms: shapely polygons (metres); walls: [(p0, p1, thickness)] whose strips join the outline, so an area without a space that
    walls enclose (closet, shaft) becomes a hole and is filled. Returns the closed outline or None."""
    rooms = [r for r in rooms if r is not None and not r.is_empty]
    if not rooms:
        return None
    c = max(float(t_max), 0.05) * 0.6 + 0.05
    U = unary_union(rooms + [wall_strip(p0, p1, t) for p0, p1, t in walls]).buffer(c, join_style=2).buffer(-c, join_style=2)
    return fill_holes(U)


def fill_holes(U):
    """Enclosed holes (shafts, closets without a space) are inside the building, not outside."""
    from shapely.geometry import Polygon, MultiPolygon
    parts = [Polygon(P.exterior) for P in getattr(U, "geoms", [U]) if P.geom_type == "Polygon" and not P.is_empty]
    return unary_union(parts) if parts else U


def wall_is_exterior(p0, p1, thickness: float, outline, gap: float = 0.25) -> bool:
    if outline is None:
        return False
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float); d = p1 - p0; L = float(np.linalg.norm(d))
    if L < 1e-6:
        return False
    u = d / L; n = np.array([-u[1], u[0]]); off = thickness / 2 + gap; votes = 0
    for s in (0.25, 0.5, 0.75):
        m = p0 + d * s
        a = outline.contains(Point(m + n * off)); b = outline.contains(Point(m - n * off))
        votes += a != b
    return votes >= 2


def external_flags(walls, rooms) -> list[bool]:
    """walls: [(p0, p1, thickness)], rooms: shapely polygons, all on one storey, metres."""
    if not walls:
        return []
    outline = storey_outline(rooms, max(t for _, _, t in walls), walls)
    return [wall_is_exterior(p0, p1, t, outline) for p0, p1, t in walls]
