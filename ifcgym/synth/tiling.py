"""Stage 2 of the synthetic buildings (user 2026-10-21): LARGE storeys by tiling several apartment plans along a corridor, and a
vertical core (elevator shaft, stair, slab openings) so multi-storey buildings have the entities real ones have.

tile_plans(plans, corridor=2.0) -> Plan: apartments are normalised to their bounding box, placed left to right on both sides of
a corridor (even ones above, odd ones below, mirrored so their entrance doors face the corridor), the corridor becomes a Space,
party walls between neighbours are shared (one wall). add_core(plan, ...) appends an elevator shaft and a stair well at the
corridor end (4 shaft walls each, a shaft Space, and a marker for the slab opening); Builder.add_storey reads plan.core to void the
slab and to place IfcStair / IfcTransportElement on every storey.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ifcgym.synth.ifcbuild import Opening, Plan, Room, WallSeg


@dataclass
class Core:
    elevator: tuple[float, float, float, float] | None = None   # (x0, y0, x1, y1) in plan coordinates
    stair: tuple[float, float, float, float] | None = None
    open_stair: bool = False          # house pattern (fzk_house): the stair stands inside a room, no stair-well walls
    slab_openings: bool = True        # some exports (wbdg_office) carry no slab openings at all


def _bbox(plan: Plan):
    xs = [p[0] for w in plan.walls for p in (w.p0, w.p1)]; ys = [p[1] for w in plan.walls for p in (w.p0, w.p1)]
    return min(xs), min(ys), max(xs), max(ys)


def _transform(plan: Plan, dx: float, dy: float, flip_y: bool, y0: float, y1: float) -> Plan:
    """Translate (and mirror in y about the bbox when flip_y) a plan; openings keep their wall index and t."""
    def tp(p):
        x, y = p
        if flip_y:
            y = y0 + y1 - y
        return (x + dx, y + dy)
    walls = [WallSeg(tp(w.p0), tp(w.p1), w.thickness, w.external) for w in plan.walls]
    rooms = [Room(r.name, [tp(p) for p in r.polygon], r.long_name) for r in plan.rooms]
    return Plan(walls, [Opening(o.wall, o.t, o.width, o.kind, o.height, o.sill) for o in plan.doors], rooms, plan.height)


def tile_plans(plans: list[Plan], corridor: float = 2.0, party: float = 0.24, height: float = 2.8) -> Plan:
    """Two rows of apartments facing a corridor. Apartments must be axis-aligned plans with their entrance on a long side; we
    simply put even-indexed plans above the corridor (mirrored so their bottom edge touches it) and odd ones below."""
    above = plans[0::2]; below = plans[1::2]
    walls: list[WallSeg] = []; doors: list[Opening] = []; rooms: list[Room] = []; rects: list[tuple] = []
    OUTDOOR = ("outdoor", "balcony", "terrace", "garden", "yard")   # quality review 2026-10-22: outdoor rooms stick out of the apartment envelope and overlap neighbours / the corridor
    def place(row: list[Plan], y_edge: float, up: bool):
        x = 0.0; widths = []
        for p in row:
            x0, y0, x1, y1 = _bbox(p); w = x1 - x0; h = y1 - y0
            dy = (y_edge - y0) if up else (y_edge - y1)
            q = _transform(p, x - x0, dy, flip_y=not up, y0=y0, y1=y1) if False else _transform(p, x - x0, dy, flip_y=False, y0=y0, y1=y1)
            off = len(walls)
            walls.extend(q.walls); rooms.extend(r for r in q.rooms if not (r.long_name and any(k in r.long_name.lower() for k in OUTDOOR)))
            rects.append((x, y0 + dy, x + w, y1 + dy))
            doors.extend(Opening(o.wall + off, o.t, o.width, o.kind, o.height, o.sill) for o in q.doors)
            widths.append(w); x += w + party
        return x
    y_top = corridor / 2; y_bot = -corridor / 2
    xa = place(above, y_top, True); xb = place(below, y_bot, False)
    X = max(xa, xb) - party
    # corridor: a space between the rows plus closing walls at both ends; doors from each apartment into the corridor
    rooms.append(Room("000", [(0.0, y_bot), (X, y_bot), (X, y_top), (0.0, y_top)], "Corridor")); rects.append((0.0, y_bot, X, y_top))
    walls.append(WallSeg((0.0, y_bot), (0.0, y_top), party, True)); walls.append(WallSeg((X, y_bot), (X, y_top), party, True))
    for i, w in enumerate(walls):
        # apartment walls lying on the corridor edges get a door to the corridor (one per apartment: the first such wall)
        pass
    added = set()
    for i, w in enumerate(walls[:-2]):
        on_top = abs(w.p0[1] - y_top) < 1e-6 and abs(w.p1[1] - y_top) < 1e-6
        on_bot = abs(w.p0[1] - y_bot) < 1e-6 and abs(w.p1[1] - y_bot) < 1e-6
        if (on_top or on_bot) and abs(w.p1[0] - w.p0[0]) > 1.6:
            key = (round(min(w.p0[0], w.p1[0]) / 3), on_top)
            if key in added:
                continue
            added.add(key); doors.append(Opening(i, 0.5, 0.9, "door"))
    plan = Plan(walls, doors, rooms, height); plan.outline = _union_outline(rects)
    return plan


def _union_outline(rects, pad: float = 0.15):
    """Slab outline = union of the apartment / corridor / core rectangles (plus the wall half thickness), not the global bbox:
    apartments of different depth leave notches that must not count as floor."""
    from shapely.geometry import box
    from shapely.ops import unary_union
    u = unary_union([box(x0 - pad, y0 - pad, x1 + pad, y1 + pad) for x0, y0, x1, y1 in rects]).simplify(0.02)
    if u.geom_type == "MultiPolygon":
        u = max(u.geoms, key=lambda g: g.area)
    return [(float(x), float(y)) for x, y in list(u.exterior.coords)[:-1]]


def add_house_stair(plan: Plan, rng, width: float = 1.0, length: float = 3.0) -> Plan:
    """House / small-block pattern seen in fzk_house and ac20: one open stair per storey standing inside the entrance hall,
    corridor or living room (preferred by room name, else the largest room), same footprint on every storey, the slab above
    voided over the stair. The room polygon is left unchanged (the stair is furniture-like inside it)."""
    pref = ("Entry", "Hall", "Corridor", "Lobby", "LivingRoom", "Living")
    def area(r):
        P = r.polygon; return 0.5 * abs(sum(P[i][0] * P[(i + 1) % len(P)][1] - P[(i + 1) % len(P)][0] * P[i][1] for i in range(len(P))))
    rooms = [r for r in plan.rooms if area(r) >= 6.0]
    if not rooms:
        return plan
    named = [r for r in rooms if r.long_name and any(k.lower() in r.long_name.lower() for k in pref)]
    room = rng.choice(named) if named else max(rooms, key=area)
    xs = [p[0] for p in room.polygon]; ys = [p[1] for p in room.polygon]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    along_x = (x1 - x0) >= (y1 - y0)
    L = min(length, (x1 - x0 if along_x else y1 - y0) - 1.0); W = width
    if L < 2.2:
        return plan
    # against one long side of the room, 0.3 m off the wall, near a corner
    if along_x:
        sx = x0 + 0.5; sy = rng.choice([y0 + 0.3, y1 - 0.3 - W]); rect = (sx, sy, sx + L, sy + W)
    else:
        sy = y0 + 0.5; sx = rng.choice([x0 + 0.3, x1 - 0.3 - W]); rect = (sx, sy, sx + W, sy + L)
    plan.core = Core(elevator=None, stair=rect, open_stair=True, slab_openings=(rng.random() > 0.2))
    return plan


def add_core(plan: Plan, at_x: float | None = None, shaft_w: float = 2.4, stair_w: float = 3.0, stair_l: float = 5.0, t: float = 0.25) -> Plan:
    """Elevator shaft + stair well attached to the corridor end (or at x = at_x): 4 walls each, shaft spaces, core geometry kept
    on plan.core for the builder (slab openings, IfcStair, IfcTransportElement)."""
    x0, y0, x1, y1 = _bbox(plan)
    cx = x1 + t if at_x is None else at_x
    ey0, ey1 = -shaft_w / 2, shaft_w / 2
    sx0, sx1 = cx + shaft_w + t, cx + shaft_w + t + stair_w
    sy0, sy1 = -stair_l / 2, stair_l / 2
    core = Core(elevator=(cx, ey0, cx + shaft_w, ey1), stair=(sx0, sy0, sx1, sy1))
    n = len(plan.walls)
    plan.walls += [WallSeg((cx, ey0), (cx + shaft_w, ey0), t, False), WallSeg((cx + shaft_w, ey0), (cx + shaft_w, ey1), t, False),
                   WallSeg((cx + shaft_w, ey1), (cx, ey1), t, False), WallSeg((cx, ey1), (cx, ey0), t, False),
                   WallSeg((sx0, sy0), (sx1, sy0), t, True), WallSeg((sx1, sy0), (sx1, sy1), t, True), WallSeg((sx1, sy1), (sx0, sy1), t, True), WallSeg((sx0, sy1), (sx0, sy0), t, True)]
    plan.doors.append(Opening(n + 3, 0.5, 1.0, "door"))          # elevator door on the corridor side
    plan.doors.append(Opening(n + 7, 0.5, 1.0, "door"))          # stair well door
    plan.rooms.append(Room("E00", [(cx + t / 2, ey0 + t / 2), (cx + shaft_w - t / 2, ey0 + t / 2), (cx + shaft_w - t / 2, ey1 - t / 2), (cx + t / 2, ey1 - t / 2)], "ElevatorShaft"))
    plan.rooms.append(Room("S00", [(sx0 + t / 2, sy0 + t / 2), (sx1 - t / 2, sy0 + t / 2), (sx1 - t / 2, sy1 - t / 2), (sx0 + t / 2, sy1 - t / 2)], "Stairwell"))
    plan.core = core
    if plan.outline:   # extend the slab outline over the core
        from shapely.geometry import Polygon, box
        from shapely.ops import unary_union
        u = unary_union([Polygon(plan.outline), box(cx - 0.15, ey0 - 0.15, sx1 + 0.15, max(ey1, sy1) + 0.15), box(cx - 0.15, min(ey0, sy0) - 0.15, sx1 + 0.15, ey1 + 0.15)]).simplify(0.02)
        if u.geom_type == "MultiPolygon":
            u = max(u.geoms, key=lambda g: g.area)
        plan.outline = [(float(x), float(y)) for x, y in list(u.exterior.coords)[:-1]]
    return plan
