"""CubiCasa5K floor plan (model.svg) -> Plan for ifcbuild.

CubiCasa5K (Zenodo 2613548, CC BY-NC-SA 4.0; data/cubicasa5k/, training-data use only): each sample folder has model.svg with
polygons: class "Wall" (wall outlines as quadrilaterals), "Door"/"Window" (opening outlines), "Space <RoomType>" (room interior
polygons), plus furniture and symbols we ignore. Coordinates are pixels of F1_scaled.png; the scale is calibrated so that the
median door width is DOOR_WIDTH_M (the dataset has no metric scale).

What we keep: axis-aligned wall quads only (the task generator v2 moves walls across their axis and needs axis-aligned walls); a
plan is rejected when more than 15 percent of its wall length is oblique, when it has fewer than 3 rooms, or when room polygons
are not simple. Wall quads -> centreline segments + thickness; segments on the same line are merged; T/L joints are left to the
builder's corner extension. Doors/windows -> the wall whose centreline is nearest their centre, position along it.
"""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from ifcgym.synth.ifcbuild import Opening, Plan, Room, WallSeg

DOOR_WIDTH_M = 0.9
ROOM_MIN_AREA_M2 = 3.0


def _polys(svg_path: str, prefix: str):
    """[(class, [(x, y), ...])] for polygons whose class starts with prefix (search all elements, polygons may be nested in <g>)."""
    root = ET.parse(svg_path).getroot()
    out = []
    for el in root.iter():
        cls = el.attrib.get("class", "")
        if not cls.startswith(prefix):
            continue
        pts = None
        tag = el.tag.split("}")[-1]
        if tag == "polygon":
            pts = el.attrib.get("points", "")
        else:
            for sub in el.iter():
                if sub.tag.split("}")[-1] == "polygon":
                    pts = sub.attrib.get("points", ""); break
        if not pts:
            continue
        nums = [float(x) for x in re.split(r"[ ,]+", pts.strip()) if x]
        poly = [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]
        if len(poly) >= 3:
            out.append((cls, poly))
    return out


def _obb_axis_aligned(poly):
    """For an axis-aligned quad: (p0, p1, thickness) of its centreline; None if oblique."""
    P = np.array(poly, dtype=float)
    xs, ys = P[:, 0], P[:, 1]
    w, h = xs.max() - xs.min(), ys.max() - ys.min()
    # oblique test: the polygon area must be close to its bounding box area
    area = 0.5 * abs(sum(xs[i] * ys[(i + 1) % len(P)] - xs[(i + 1) % len(P)] * ys[i] for i in range(len(P))))
    if area < 0.8 * w * h:
        return None
    if w >= h:
        y = (ys.max() + ys.min()) / 2
        return (xs.min(), y), (xs.max(), y), h
    x = (xs.max() + xs.min()) / 2
    return (x, ys.min()), (x, ys.max()), w


def _merge_collinear(segs, tol):
    """Merge wall segments that are collinear, overlapping/abutting and of similar thickness."""
    segs = [list(s) for s in segs]
    changed = True
    while changed:
        changed = False
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                (a0, a1, ta), (b0, b1, tb) = segs[i], segs[j]
                if abs(ta - tb) > tol:
                    continue
                horiz_a, horiz_b = abs(a0[1] - a1[1]) < 1e-9, abs(b0[1] - b1[1]) < 1e-9
                if horiz_a != horiz_b:
                    continue
                if horiz_a and abs(a0[1] - b0[1]) <= tol:
                    lo, hi = min(a0[0], a1[0], b0[0], b1[0]), max(a0[0], a1[0], b0[0], b1[0])
                    if hi - lo <= (abs(a1[0] - a0[0]) + abs(b1[0] - b0[0])) + tol:
                        segs[i] = [(lo, a0[1]), (hi, a0[1]), (ta + tb) / 2]; segs.pop(j); changed = True; break
                if (not horiz_a) and abs(a0[0] - b0[0]) <= tol:
                    lo, hi = min(a0[1], a1[1], b0[1], b1[1]), max(a0[1], a1[1], b0[1], b1[1])
                    if hi - lo <= (abs(a1[1] - a0[1]) + abs(b1[1] - b0[1])) + tol:
                        segs[i] = [(a0[0], lo), (a0[0], hi), (ta + tb) / 2]; segs.pop(j); changed = True; break
            if changed:
                break
    return [tuple(s) for s in segs]


def _simple(poly) -> bool:
    try:
        from shapely.geometry import Polygon
        return Polygon(poly).is_valid and Polygon(poly).area > 0
    except Exception:  # noqa: BLE001
        return True


def load_plan(sample_dir: str, height: float = 2.8) -> Plan | None:
    svg = str(Path(sample_dir) / "model.svg")
    walls = _polys(svg, "Wall"); doors = _polys(svg, "Door"); windows = _polys(svg, "Window"); spaces = _polys(svg, "Space")
    if not walls or not spaces:
        return None
    # scale: median door width (short side of the door quads) = DOOR_WIDTH_M
    dw = []
    for _, p in doors:
        P = np.array(p); dw.append(max(P[:, 0].max() - P[:, 0].min(), P[:, 1].max() - P[:, 1].min()))
    if not dw:
        return None
    px_per_m = float(np.median(dw)) / DOOR_WIDTH_M
    S = 1.0 / px_per_m
    segs = []; oblique = 0.0; total = 0.0
    for _, p in walls:
        r = _obb_axis_aligned(p)
        P = np.array(p); L = max(P[:, 0].max() - P[:, 0].min(), P[:, 1].max() - P[:, 1].min()) * S
        total += L
        if r is None:
            oblique += L; continue
        (x0, y0), (x1, y1), t = r
        if max(abs(x1 - x0), abs(y1 - y0)) * S < 0.3 or t * S < 0.05:
            continue
        segs.append(((x0 * S, -y0 * S), (x1 * S, -y1 * S), float(np.clip(t * S, 0.08, 0.5))))   # SVG y points down -> flip
    if total == 0 or oblique / total > 0.15 or len(segs) < 4:
        return None
    segs = _merge_collinear(segs, tol=0.06)
    # snap near-equal coordinates so joints meet exactly
    coords = sorted({c for s in segs for p in s[:2] for c in p})
    snap = {}
    for c in coords:
        for k in snap:
            if abs(k - c) < 0.04:
                snap[c] = snap[k]; break
        else:
            snap[c] = c
    segs = [((snap[a[0]], snap[a[1]]), (snap[b[0]], snap[b[1]]), t) for a, b, t in segs]
    wall_segs = []
    xs_all = [p[0] for s in segs for p in s[:2]]; ys_all = [p[1] for s in segs for p in s[:2]]
    for a, b, t in segs:
        ext = (abs(a[0] - min(xs_all)) < 0.05 and abs(b[0] - min(xs_all)) < 0.05) or (abs(a[0] - max(xs_all)) < 0.05 and abs(b[0] - max(xs_all)) < 0.05) \
            or (abs(a[1] - min(ys_all)) < 0.05 and abs(b[1] - min(ys_all)) < 0.05) or (abs(a[1] - max(ys_all)) < 0.05 and abs(b[1] - max(ys_all)) < 0.05)
        wall_segs.append(WallSeg(a, b, t, ext))
    # openings: nearest wall centreline to the opening centre
    ops = []
    for kind, lst in (("door", doors), ("window", windows)):
        for _, p in lst:
            P = np.array(p) * S; P[:, 1] *= -1; c = P.mean(0); width = float(max(P[:, 0].max() - P[:, 0].min(), P[:, 1].max() - P[:, 1].min()))
            best = None
            for i, w in enumerate(wall_segs):
                a, b = np.array(w.p0), np.array(w.p1); ab = b - a; L2 = float(ab @ ab)
                tt = float(np.clip((c - a) @ ab / L2, 0.0, 1.0)); d = float(np.linalg.norm(c - (a + tt * ab)))
                if best is None or d < best[0]:
                    best = (d, i, tt, float(np.linalg.norm(ab)))
            if best is None or best[0] > 0.4:
                continue
            d, i, tt, L = best
            width = float(np.clip(width, 0.6, min(2.4, L - 0.2)))
            tt = float(np.clip(tt, (width / 2 + 0.05) / L, 1 - (width / 2 + 0.05) / L))
            if kind == "door":
                ops.append(Opening(i, tt, width, "door", height=2.1, sill=0.0))
            else:
                ops.append(Opening(i, tt, width, "window", height=1.2, sill=0.9))
    # rooms
    rooms = []
    for k, (cls, p) in enumerate(spaces):
        P = [(x * S, -y * S) for x, y in p]
        # drop consecutive duplicates / collinear points
        Q = []
        for q in P:
            if not Q or math.dist(Q[-1], q) > 0.02:
                Q.append(q)
        if len(Q) >= 3 and math.dist(Q[0], Q[-1]) < 0.02:
            Q.pop()
        if len(Q) < 3 or not _simple(Q):
            continue
        area = 0.5 * abs(sum(Q[i][0] * Q[(i + 1) % len(Q)][1] - Q[(i + 1) % len(Q)][0] * Q[i][1] for i in range(len(Q))))
        if area < ROOM_MIN_AREA_M2:
            continue
        rtype = cls.replace("Space", "").strip().split()[0] if cls.replace("Space", "").strip() else "Room"
        rooms.append(Room(f"{101 + k}", Q, rtype))
    if len(rooms) < 3:
        return None
    return Plan(wall_segs, ops, rooms, height)


def main() -> None:
    import argparse, json, glob
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="cubicasa5k root (contains high_quality/, high_quality_architectural/, colorful/)")
    ap.add_argument("--limit", type=int, default=20); ap.add_argument("--out", default=None, help="write accepted plans as JSON lines")
    a = ap.parse_args()
    dirs = sorted(glob.glob(str(Path(a.root) / "*" / "*" / "model.svg")))[: a.limit * 5]
    kept = 0; rows = []
    for svg in dirs:
        plan = load_plan(str(Path(svg).parent))
        if plan is None:
            continue
        kept += 1
        rows.append({"sample": str(Path(svg).parent), "walls": len(plan.walls), "doors": sum(1 for o in plan.doors if o.kind == "door"), "windows": sum(1 for o in plan.doors if o.kind == "window"), "rooms": len(plan.rooms)})
        print(rows[-1])
        if kept >= a.limit:
            break
    print(f"accepted {kept} of {len(dirs)} scanned")
    if a.out:
        with open(a.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
