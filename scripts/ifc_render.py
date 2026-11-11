"""Render an IFC model before / after an edit to PNG (isometric + plan) with the changed elements highlighted, and summarise the
edit (elements and relationships added / removed / modified). Used by scripts/humaneval_build.py (expert evaluation pages) and
scripts/examples_fig.py (paper figure with edit examples). Needs the ifc-audit venv (ifcopenshell 0.8.5, matplotlib): run through
scripts/py312.sh or the python:3.12 container.

    python scripts/ifc_render.py BEFORE.ifc AFTER.ifc [AFTER2.ifc ...] --out PREFIX [--focus GUID ...] [--prompt "text with GUIDs"]

Conventions: unchanged elements grey (doors tan, windows blue), added green, removed red (ghost of the original geometry),
modified orange (plus a faint ghost of the original), elements referenced by the instruction outlined in blue. The view is a box
around the changed + referenced elements (margin 4 m, at most 36 m wide); ceilings / upper storeys above the band are hidden.
Element and relationship changes outside the view are listed in the summary so that they are not lost.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

import numpy as np
import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element as eu
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ifcgym import ifcgraph  # noqa: E402

KEEP = {"IfcWall", "IfcWallStandardCase", "IfcWallElementedCase", "IfcSlab", "IfcColumn", "IfcBeam", "IfcDoor", "IfcWindow", "IfcSpace",
        "IfcOpeningElement", "IfcStair", "IfcStairFlight", "IfcRoof", "IfcRailing", "IfcPlate", "IfcMember", "IfcCurtainWall",
        "IfcBuildingElementProxy", "IfcFooting", "IfcRamp", "IfcRampFlight", "IfcShadingDevice", "IfcChimney"}
GHOST_ONLY = {"IfcSpace", "IfcOpeningElement"}          # drawn only when changed or referenced (spaces also in the plan, translucent)
CONTEXT_OFF = {"IfcBuildingElementProxy", "IfcRailing", "IfcMember", "IfcPlate", "IfcShadingDevice"}   # clutter in real projects: only when changed / referenced
CLASS_COLOR = {"IfcDoor": "#c9a66b", "IfcWindow": "#8fbcd4", "IfcSlab": "#e6e6e6", "IfcSpace": "#9ecae1", "IfcOpeningElement": "#f0e442",
               "IfcColumn": "#b0b0b0", "IfcBeam": "#b8b8b8", "IfcRailing": "#c0c0c0", "IfcStair": "#c8c8c8", "IfcStairFlight": "#c8c8c8",
               "IfcRoof": "#dcdcdc", "IfcBuildingElementProxy": "#c4c4c4"}
DEFAULT_COLOR = "#cfcfcf"
KIND_COLOR = {"added": "#2ca02c", "removed": "#d62728", "modified": "#ff7f0e", "focus": "#1f77b4"}
GUID_RE = re.compile(r"(?<![0-9A-Za-z_$])[0-9A-Za-z_$]{22}(?![0-9A-Za-z_$])")
LIGHT = np.array([-0.45, 0.55, 1.0]); LIGHT /= np.linalg.norm(LIGHT)
MAX_EXTENT = 36.0
MARGIN = 4.0


# ----------------------------------------------------------------------------------------------------------------- model
class Model:
    """An IFC file with the world-coordinate triangle meshes of its building elements (metres) and a lazily built edit graph."""

    def __init__(self, path: str, threads: int = 8):
        self.path = path
        self.f = ifcopenshell.open(path)
        self.shapes: dict[str, dict] = {}
        s = ifcopenshell.geom.settings()
        s.set("use-world-coords", True)
        it = ifcopenshell.geom.iterator(s, self.f, threads)
        if it.initialize():
            while True:
                sh = it.get()
                if sh.type in KEEP:
                    v = np.array(sh.geometry.verts, dtype=float).reshape(-1, 3)
                    fc = np.array(sh.geometry.faces, dtype=int).reshape(-1, 3)
                    if len(v) and len(fc):
                        e = self.f.by_id(sh.id)
                        self.shapes[sh.guid] = {"type": sh.type, "name": getattr(e, "Name", None), "verts": v, "faces": fc,
                                                "lo": v.min(0), "hi": v.max(0)}
                if not it.next():
                    break
        self._graph = None

    def graph(self):
        if self._graph is None:
            self._graph = ifcgraph.build_graph(self.f)
        return self._graph

    def label(self, guid: str) -> str:
        try:
            e = self.f.by_guid(guid)
        except Exception:  # noqa: BLE001
            return guid
        return f"{e.is_a()} '{e.Name}'" if getattr(e, "Name", None) else e.is_a()

    def storey(self, guid: str) -> str | None:
        try:
            c = eu.get_container(self.f.by_guid(guid))
            return getattr(c, "Name", None) if c is not None else None
        except Exception:  # noqa: BLE001
            return None

    def bbox(self):
        if not self.shapes:
            return np.zeros(3), np.ones(3)
        lo = np.min([s["lo"] for s in self.shapes.values()], axis=0); hi = np.max([s["hi"] for s in self.shapes.values()], axis=0)
        return lo, hi


_CACHE: dict[str, Model] = {}


def load(path: str, cache: bool = True) -> Model:
    if cache and path in _CACHE:
        return _CACHE[path]
    m = Model(path)
    if cache:
        _CACHE.clear(); _CACHE[path] = m          # one cached "before" model at a time (real projects are large)
    return m


def guids_in(text: str, model: Model) -> list[str]:
    out = []
    for g in GUID_RE.findall(text or ""):
        try:
            model.f.by_guid(g); out.append(g)
        except Exception:  # noqa: BLE001
            pass
    return list(dict.fromkeys(out))


# ------------------------------------------------------------------------------------------------------------------ diff
def diff(before: Model, after: Model) -> dict:
    """Element / relationship changes between two models (edit graph of ifcgym.ifcgraph) in a readable form."""
    eg = ifcgraph.diff_graphs(before.graph(), after.graph())
    changes = []
    for g, n in sorted(eg.added.items()):
        changes.append({"guid": g, "kind": "added", "cls": n.cls, "name": n.name, "storey": after.storey(g), "detail": ""})
    for g, n in sorted(eg.removed.items()):
        changes.append({"guid": g, "kind": "removed", "cls": n.cls, "name": n.name, "storey": before.storey(g), "detail": ""})
    for g, (x, y) in sorted(eg.modified.items()):
        parts = []
        if x.geom_hash != y.geom_hash or x.placement_hash != y.placement_hash:
            sb, sa = before.shapes.get(g), after.shapes.get(g)
            if sb is not None and sa is not None and len(sb["verts"]) == len(sa["verts"]):
                d = sa["verts"].mean(0) - sb["verts"].mean(0)
                if np.allclose(sb["verts"] + d, sa["verts"], atol=1e-3):
                    parts.append("moved by (%.2f, %.2f, %.2f) m" % tuple(d) if np.linalg.norm(d) > 1e-3 else "representation rewritten in place")
                else:
                    parts.append("geometry changed (extent %.2f x %.2f x %.2f -> %.2f x %.2f x %.2f m)" % (tuple(sb["hi"] - sb["lo"]) + tuple(sa["hi"] - sa["lo"])))
            else:
                parts.append("geometry or placement changed")
        for k in sorted(set(x.attrs) | set(y.attrs)):
            if x.attrs.get(k) != y.attrs.get(k):
                parts.append(f"{k}: {x.attrs.get(k)!r} -> {y.attrs.get(k)!r}")
        changes.append({"guid": g, "kind": "modified", "cls": y.cls, "name": y.name, "storey": after.storey(g), "detail": "; ".join(parts)})

    def rel_text(e, m):
        a = m.label(e.src) if e.src else "-"; b = m.label(e.dst) if e.dst else (" ".join(str(k) for k in e.key) if e.key else "-")
        return f"{e.cls}: {a} -> {b}"
    rel_added = [rel_text(e, after) for e in eg.rel_added.values()]
    rel_removed = [rel_text(e, before) for e in eg.rel_removed.values()]
    return {"changes": changes, "rel_added": sorted(rel_added), "rel_removed": sorted(rel_removed), "counts": eg.summary(),
            "rel_added_by_type": dict(Counter(e.cls for e in eg.rel_added.values())),
            "rel_removed_by_type": dict(Counter(e.cls for e in eg.rel_removed.values()))}


# ---------------------------------------------------------------------------------------------------------------- region
def region_of(before: Model, afters: list[Model], diffs: list[dict], focus: list[str], margin: float = MARGIN) -> dict:
    """View box around the changed and referenced elements: xy with a margin (capped at MAX_EXTENT around their centre),
    z band from the lowest changed element to at least 3.2 m above it."""
    boxes = []
    for a, d in zip(afters, diffs):
        for c in d["changes"]:
            m = before if c["kind"] == "removed" else a
            if c["guid"] in m.shapes:
                s = m.shapes[c["guid"]]; boxes.append((s["lo"], s["hi"]))
    for g in focus:
        if g in before.shapes:
            s = before.shapes[g]; boxes.append((s["lo"], s["hi"]))
    fin = lambda b: np.all(np.isfinite(b[0])) and np.all(np.isfinite(b[1])) and np.abs(b[0]).max() < 1e5 and np.abs(b[1]).max() < 1e5  # noqa: E731
    boxes = [b for b in boxes if fin(b)]
    fboxes = [(before.shapes[g]["lo"], before.shapes[g]["hi"]) for g in focus if g in before.shapes and fin((before.shapes[g]["lo"], before.shapes[g]["hi"]))]
    if fboxes:                            # frame the referenced elements; changes far from them (a broken edit touching the whole building) are listed, not framed
        c = (np.min([b[0] for b in fboxes], axis=0) + np.max([b[1] for b in fboxes], axis=0)) / 2
        boxes = fboxes + [b for b in boxes if np.linalg.norm(((b[0] + b[1]) / 2 - c)[:2]) < MAX_EXTENT / 2 and abs(((b[0] + b[1]) / 2 - c)[2]) < 6]
    whole = not boxes                     # (elements placed at absurd coordinates by a broken edit are listed, not framed)
    if whole:
        boxes = [before.bbox()]
    lo = np.min([b[0] for b in boxes], axis=0); hi = np.max([b[1] for b in boxes], axis=0)
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    if whole:
        x0, x1, y0, y1 = lo[0] - 1, hi[0] + 1, lo[1] - 1, hi[1] + 1; z0, z1 = lo[2] - 0.2, hi[2] + 0.2
    else:
        hx = min((hi[0] - lo[0]) / 2 + margin, MAX_EXTENT / 2); hy = min((hi[1] - lo[1]) / 2 + margin, MAX_EXTENT / 2)
        x0, x1, y0, y1 = cx - hx, cx + hx, cy - hy, cy + hy
        z0 = lo[2] - 0.3; z1 = max(hi[2] + 0.3, z0 + 3.2)
    return {"x0": float(x0), "x1": float(x1), "y0": float(y0), "y1": float(y1), "z0": float(z0), "z1": float(z1), "whole": whole}


def _in_region(s: dict, reg: dict) -> bool:
    if not (np.all(np.isfinite(s["lo"])) and np.all(np.isfinite(s["hi"])) and np.abs(s["lo"]).max() < 1e5 and np.abs(s["hi"]).max() < 1e5):
        return False                                       # degenerate geometry (NaN / absurd coordinates) is listed in the summary, never drawn
    return not (s["hi"][0] < reg["x0"] or s["lo"][0] > reg["x1"] or s["hi"][1] < reg["y0"] or s["lo"][1] > reg["y1"]
                or s["hi"][2] < reg["z0"] or s["lo"][2] > reg["z1"])


def _visible(s: dict, reg: dict, special: bool) -> bool:
    if not _in_region(s, reg):
        return False
    if special:
        return True
    if s["type"] in GHOST_ONLY or (s["type"] in CONTEXT_OFF and not reg["whole"]):
        return False
    if not reg["whole"] and s["lo"][2] > reg["z0"] + 2.6:   # ceiling slab / upper storey would hide the band
        return False
    return True


# -------------------------------------------------------------------------------------------------------------- geometry
def _clip_poly(poly: np.ndarray, axis: int, value: float, keep_greater: bool) -> np.ndarray:
    out = []
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        pin = (p[axis] >= value) if keep_greater else (p[axis] <= value)
        qin = (q[axis] >= value) if keep_greater else (q[axis] <= value)
        if pin:
            out.append(p)
        if pin != qin:
            t = (value - p[axis]) / (q[axis] - p[axis]); out.append(p + t * (q - p))
    return np.array(out) if len(out) >= 3 else np.zeros((0, 3))


def clip_tris(verts: np.ndarray, faces: np.ndarray, reg: dict) -> list[np.ndarray]:
    """Triangles clipped to the region box (section box; polygons with 3-9 vertices)."""
    P = verts[faces]                                         # (n, 3, 3)
    inside = ((P[:, :, 0] >= reg["x0"]) & (P[:, :, 0] <= reg["x1"]) & (P[:, :, 1] >= reg["y0"]) & (P[:, :, 1] <= reg["y1"])
              & (P[:, :, 2] >= reg["z0"]) & (P[:, :, 2] <= reg["z1"]))
    out = [p for p in P[inside.all(1)]]
    for p in P[~inside.all(1)]:
        poly = p
        for axis, value, kg in ((0, reg["x0"], True), (0, reg["x1"], False), (1, reg["y0"], True), (1, reg["y1"], False),
                                (2, reg["z0"], True), (2, reg["z1"], False)):
            poly = _clip_poly(poly, axis, value, kg)
            if len(poly) < 3:
                break
        if len(poly) >= 3:
            out.append(poly)
    return out


def feature_edges(verts: np.ndarray, faces: np.ndarray, angle_deg: float = 25.0) -> np.ndarray:
    """Outline segments of a mesh: boundary edges and edges between faces whose normals differ by more than angle_deg."""
    if len(faces) == 0:
        return np.zeros((0, 2, 3))
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.sort(e, axis=1)
    fid = np.tile(np.arange(len(faces)), 3)
    u, inv, cnt = np.unique(e, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    P = verts[faces]
    nrm = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    ln = np.linalg.norm(nrm, axis=1); ln[ln == 0] = 1; nrm /= ln[:, None]
    keep = np.zeros(len(u), dtype=bool)
    keep[cnt != 2] = True
    order = np.argsort(inv, kind="stable"); inv_s = inv[order]; fid_s = fid[order]
    starts = np.searchsorted(inv_s, np.arange(len(u)))
    two = np.where(cnt == 2)[0]
    f1 = fid_s[starts[two]]; f2 = fid_s[starts[two] + 1]
    cosang = np.abs((nrm[f1] * nrm[f2]).sum(1))
    keep[two[cosang < np.cos(np.radians(angle_deg))]] = True
    seg = verts[u[keep]]                                     # (k, 2, 3)
    # drop degenerate / duplicate-position edges
    seg = seg[np.linalg.norm(seg[:, 0] - seg[:, 1], axis=1) > 1e-4]
    return seg


def _clip_segments_xy(seg: np.ndarray, reg: dict) -> np.ndarray:
    if len(seg) == 0:
        return seg
    out = []
    for a, b in seg:
        t0, t1 = 0.0, 1.0
        d = b - a
        ok = True
        for axis, lo, hi in ((0, reg["x0"], reg["x1"]), (1, reg["y0"], reg["y1"]), (2, reg["z0"], reg["z1"])):
            for pval, qval in ((-d[axis], a[axis] - lo), (d[axis], hi - a[axis])):
                if abs(pval) < 1e-12:
                    if qval < 0:
                        ok = False
                else:
                    t = qval / pval
                    if pval < 0:
                        t0 = max(t0, t)
                    else:
                        t1 = min(t1, t)
        if ok and t0 <= t1:
            out.append((a + t0 * d, a + t1 * d))
    return np.array(out) if out else np.zeros((0, 2, 3))


def _shade(polys: list[np.ndarray], rgb) -> list:
    cols = []
    for p in polys:
        n = np.cross(p[1] - p[0], p[2] - p[0]); ln = np.linalg.norm(n)
        f = 0.62 + 0.38 * abs(float(n @ LIGHT) / ln) if ln > 0 else 0.8
        cols.append(tuple(min(1.0, c * f) for c in rgb[:3]))
    return cols


def _rgb(hexcol: str):
    return matplotlib.colors.to_rgb(hexcol)


# ------------------------------------------------------------------------------------------------------------- draw list
def draw_items(before: Model, after: Model | None, d: dict | None, reg: dict, focus: list[str]) -> list[dict]:
    """What to draw for one state: the 'before' model alone (after=None) or the edited model with the diff highlighted."""
    items = []
    changed = {c["guid"]: c for c in (d["changes"] if d else [])}
    if after is None:                                        # the original model: neutral colours, referenced elements outlined
        for g, s in before.shapes.items():
            if _visible(s, reg, g in focus):
                items.append({"guid": g, "shape": s, "color": CLASS_COLOR.get(s["type"], DEFAULT_COLOR), "alpha": 0.25 if s["type"] in GHOST_ONLY else 1.0,
                              "kind": "focus" if g in focus else None, "z": s["lo"][2]})
        return items
    for g, s in after.shapes.items():
        c = changed.get(g)
        special = c is not None or g in focus
        if not _visible(s, reg, special):
            continue
        if c is None:
            items.append({"guid": g, "shape": s, "color": CLASS_COLOR.get(s["type"], DEFAULT_COLOR), "alpha": 0.25 if s["type"] in GHOST_ONLY else 1.0,
                          "kind": "focus" if g in focus else None, "z": s["lo"][2]})
        else:
            items.append({"guid": g, "shape": s, "color": KIND_COLOR[c["kind"]], "alpha": 0.45 if s["type"] in GHOST_ONLY else 1.0,
                          "kind": c["kind"], "z": s["lo"][2]})
            if c["kind"] == "modified" and g in before.shapes:           # ghost of the original geometry
                sb = before.shapes[g]
                if _in_region(sb, reg):
                    items.append({"guid": g, "shape": sb, "color": KIND_COLOR["modified"], "alpha": 0.18, "kind": "ghost", "z": sb["lo"][2]})
    for g, c in changed.items():                             # removed elements: ghost of the original
        if c["kind"] == "removed" and g in before.shapes and _in_region(before.shapes[g], reg):
            items.append({"guid": g, "shape": before.shapes[g], "color": KIND_COLOR["removed"], "alpha": 0.4, "kind": "removed", "z": before.shapes[g]["lo"][2]})
    return items


# ------------------------------------------------------------------------------------------------------------------ views
def draw_iso(ax, items: list[dict], reg: dict, elev: float = 36, azim: float = -58, zoom: float = 1.0):
    polys, cols, edges, ecols = [], [], [], []
    ghosts = []
    for it in sorted(items, key=lambda i: i["alpha"] < 1.0):          # opaque first
        s = it["shape"]
        tris = [p for p in clip_tris(s["verts"], s["faces"], reg) if np.all(np.isfinite(p)) and np.abs(p).max() < 1e6]
        if not tris:
            continue
        rgb = _rgb(it["color"])
        if it["alpha"] < 1.0:
            ghosts.append((tris, rgb, it["alpha"]))
        else:
            polys += tris; cols += _shade(tris, rgb)
        seg = _clip_segments_xy(feature_edges(s["verts"], s["faces"]), reg)
        if len(seg):
            ec = (0.15, 0.15, 0.15, 0.55 if it["alpha"] >= 1.0 else 0.3) if it["kind"] in (None, "focus") else (*_rgb(it["color"]), 0.9 if it["alpha"] >= 1.0 else 0.5)
            edges.append(seg); ecols += [ec] * len(seg)
    if polys:
        ax.add_collection3d(Poly3DCollection(polys, facecolors=cols, edgecolors="none", linewidths=0), autolim=False)
    for tris, rgb, a in ghosts:
        ax.add_collection3d(Poly3DCollection(tris, facecolors=[(*rgb, a)] * len(tris), edgecolors="none"), autolim=False)
    if edges:
        ax.add_collection3d(Line3DCollection(np.concatenate(edges), colors=ecols, linewidths=0.5), autolim=False)
    for it in items:                                        # referenced elements: dashed blue box
        if it["kind"] == "focus":
            lo, hi = it["shape"]["lo"], it["shape"]["hi"]
            lo = np.maximum(lo, [reg["x0"], reg["y0"], -1e9]); hi = np.minimum(hi, [reg["x1"], reg["y1"], 1e9])
            ax.add_collection3d(Line3DCollection(_box_edges(lo, hi), colors=KIND_COLOR["focus"], linewidths=1.2, linestyles="--"), autolim=False)
    ax.set_xlim(reg["x0"], reg["x1"]); ax.set_ylim(reg["y0"], reg["y1"]); ax.set_zlim(reg["z0"], reg["z1"])
    ax.set_box_aspect((reg["x1"] - reg["x0"], reg["y1"] - reg["y0"], reg["z1"] - reg["z0"]), zoom=zoom)
    ax.set_proj_type("ortho"); ax.view_init(elev=elev, azim=azim); ax.set_axis_off()


def _box_edges(lo, hi):
    x0, y0, z0 = lo; x1, y1, z1 = hi
    c = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    idx = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    return [(c[i], c[j]) for i, j in idx]


def draw_plan(ax, items: list[dict], reg: dict, label_changes: bool = True):
    order = sorted(items, key=lambda i: (i["shape"]["type"] != "IfcSpace", i["shape"]["type"] != "IfcSlab", i["z"]))
    for it in order:
        s = it["shape"]
        rgb = _rgb(it["color"])
        tris = [p[:, :2] for p in clip_tris(s["verts"], s["faces"], reg)]
        if not tris:
            continue
        if it["kind"] == "ghost" or it["kind"] == "removed":
            ax.add_collection(PolyCollection(tris, facecolors=[(*rgb, it["alpha"])], edgecolors="none"))
            seg = _clip_segments_xy(feature_edges(s["verts"], s["faces"]), reg)
            seg = seg[np.linalg.norm(seg[:, 0, :2] - seg[:, 1, :2], axis=1) > 1e-4] if len(seg) else seg
            if len(seg):
                ax.add_collection(LineCollection(seg[:, :, :2], colors=[(*rgb, 0.9)], linewidths=0.8, linestyles="--"))
            continue
        if s["type"] == "IfcSpace":
            ax.add_collection(PolyCollection(tris, facecolors=[(*rgb, 0.18 if it["kind"] is None else 0.35)], edgecolors="none"))
            cx, cy = (s["lo"][:2] + s["hi"][:2]) / 2
            if reg["x0"] < cx < reg["x1"] and reg["y0"] < cy < reg["y1"]:
                ax.text(cx, cy, s["name"] or "space", ha="center", va="center", fontsize=6, color="#1f4e79", alpha=0.8)
            continue
        a = it["alpha"] if s["type"] in GHOST_ONLY else 1.0
        ax.add_collection(PolyCollection(tris, facecolors=[(*rgb, a)], edgecolors="none"))
        seg = _clip_segments_xy(feature_edges(s["verts"], s["faces"]), reg)
        seg = seg[np.linalg.norm(seg[:, 0, :2] - seg[:, 1, :2], axis=1) > 1e-4] if len(seg) else seg
        if len(seg):
            ec = (0.1, 0.1, 0.1, 0.5) if it["kind"] in (None, "focus") else (*rgb, 0.95)
            ax.add_collection(LineCollection(seg[:, :, :2], colors=[ec], linewidths=0.45))
    for it in items:
        s = it["shape"]
        if it["kind"] == "focus":
            lo, hi = s["lo"], s["hi"]
            ax.add_patch(Rectangle((lo[0] - 0.15, lo[1] - 0.15), hi[0] - lo[0] + 0.3, hi[1] - lo[1] + 0.3, fill=False, ls="--", lw=1.2, ec=KIND_COLOR["focus"]))
        elif label_changes and it["kind"] in ("added", "removed", "modified") and s["type"] != "IfcOpeningElement":
            cx, cy = (s["lo"][:2] + s["hi"][:2]) / 2
            if reg["x0"] < cx < reg["x1"] and reg["y0"] < cy < reg["y1"]:
                ax.annotate(s["type"].replace("Ifc", "").replace("StandardCase", ""), (cx, cy), xytext=(0, 6), textcoords="offset points",
                            ha="center", fontsize=6.5, color=it["color"], fontweight="bold")
    ax.set_xlim(reg["x0"], reg["x1"]); ax.set_ylim(reg["y0"], reg["y1"]); ax.set_aspect("equal")
    ax.tick_params(labelsize=6, length=2); ax.grid(True, lw=0.3, alpha=0.4)
    ax.set_xlabel("x (m)", fontsize=6, labelpad=1); ax.set_ylabel("y (m)", fontsize=6, labelpad=1)


def legend_handles(kinds=("added", "removed", "modified", "focus")):
    names = {"added": "added", "removed": "removed (original shown)", "modified": "modified (original faint)", "focus": "referenced by the instruction"}
    hs = []
    for k in kinds:
        if k == "focus":
            hs.append(Line2D([0], [0], color=KIND_COLOR[k], ls="--", lw=1.2, label=names[k]))
        else:
            hs.append(Patch(facecolor=KIND_COLOR[k], alpha=0.5 if k == "removed" else 1.0, label=names[k]))
    return hs


def save_views(items: list[dict], reg: dict, prefix: str, title: str | None = None, legend: bool = True, dpi: int = 150) -> dict:
    out = {}
    fig = plt.figure(figsize=(6.4, 4.8)); ax = fig.add_subplot(111, projection="3d")
    draw_iso(ax, items, reg)
    if title:
        fig.suptitle(title, fontsize=9)
    if legend:
        ax.legend(handles=legend_handles(), loc="lower left", fontsize=6.5, frameon=True)
    fig.subplots_adjust(0, 0, 1, 1 if not title else 0.95)
    fig.savefig(prefix + "_iso.png", dpi=dpi); plt.close(fig); out["iso"] = prefix + "_iso.png"
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    draw_plan(ax, items, reg)
    if title:
        ax.set_title(title, fontsize=9)
    if legend:
        ax.legend(handles=legend_handles(), loc="upper left", fontsize=6.5, frameon=True, bbox_to_anchor=(0, -0.08), ncol=2)
    fig.tight_layout()
    fig.savefig(prefix + "_plan.png", dpi=dpi); plt.close(fig); out["plan"] = prefix + "_plan.png"
    return out


# ---------------------------------------------------------------------------------------------------------- high level
def render_set(before_path: str, after_paths: list[str], out_prefix: str, focus: list[str] | None = None, prompt: str | None = None,
               names: list[str] | None = None, legend: bool = True, region: dict | None = None) -> dict:
    """Render the original model and each edited model in one common view box; returns the diffs and image paths."""
    before = load(before_path)
    focus = list(focus or []) + ([g for g in guids_in(prompt, before)] if prompt else [])
    focus = list(dict.fromkeys(focus))
    afters = [Model(p) for p in after_paths]
    diffs = [diff(before, a) for a in afters]
    reg = region or region_of(before, afters, diffs, focus)
    names = names or [f"after{i + 1}" for i in range(len(afters))]
    res = {"before": before_path, "focus": focus, "region": reg, "images": {}, "diffs": {}}
    res["images"]["before"] = save_views(draw_items(before, None, None, reg, focus), reg, out_prefix + "_before", legend=legend)
    for a, d, n in zip(afters, diffs, names):
        items = draw_items(before, a, d, reg, focus)
        res["images"][n] = save_views(items, reg, f"{out_prefix}_{n}", legend=legend)
        shown = {it["guid"] for it in items if it["kind"] in ("added", "removed", "modified")}
        for c in d["changes"]:
            c["in_view"] = c["guid"] in shown
        res["diffs"][n] = d
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("before"); ap.add_argument("after", nargs="+"); ap.add_argument("--out", required=True)
    ap.add_argument("--focus", nargs="*", default=None); ap.add_argument("--prompt", default=None); ap.add_argument("--names", nargs="*", default=None)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    res = render_set(a.before, a.after, a.out, a.focus, a.prompt, a.names)
    json.dump(res, open(a.out + "_render.json", "w"), indent=1, default=str)
    for n, d in res["diffs"].items():
        print(f"[{n}] {d['counts']}  region={ {k: round(v, 1) if isinstance(v, float) else v for k, v in res['region'].items()} }")
        for c in d["changes"][:20]:
            print("   ", c["kind"], c["cls"], repr(c["name"]), c.get("storey"), "| in view" if c.get("in_view") else "| NOT in view", "|", c["detail"][:160])
        for r in d["rel_added"][:8]:
            print("    + rel", r[:140])
        for r in d["rel_removed"][:8]:
            print("    - rel", r[:140])


if __name__ == "__main__":
    main()
