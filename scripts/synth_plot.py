"""Draw a synthetic plan (walls, openings, rooms) to PNG for a quick visual check: python scripts/synth_plot.py <arc.ifc> <out.png>"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, ifcopenshell
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from ifcgym import ifctools as h
f = ifcopenshell.open(sys.argv[1]); sc = h.unit_scale(f)
storey = f.by_type("IfcBuildingStorey")[0]
fig, ax = plt.subplots(figsize=(9, 7))
for s in f.by_type("IfcSpace"):
    if h.world_matrix(s)[2, 3] * sc > 0.5: continue
    poly = h.profile_world_polygon(s)
    if poly:
        P = np.array(poly)[:, :2] * sc; ax.fill(P[:, 0], P[:, 1], alpha=0.15); ax.text(P[:, 0].mean(), P[:, 1].mean(), f"{s.Name}\n{s.LongName or ''}", ha="center", va="center", fontsize=7)
for w in f.by_type("IfcWall"):
    if h.world_matrix(w)[2, 3] * sc > 0.5: continue
    poly = h.profile_world_polygon(w)
    if poly:
        P = np.array(poly + [poly[0]])[:, :2] * sc; ax.plot(P[:, 0], P[:, 1], "k-", lw=1)
for cls, col in (("IfcDoor", "tab:red"), ("IfcWindow", "tab:blue")):
    for d in f.by_type(cls):
        m = h.world_matrix(d)
        if m[2, 3] * sc > 2.5: continue
        ax.plot(m[0, 3] * sc, m[1, 3] * sc, "s", color=col, ms=5)
ax.set_aspect("equal"); ax.set_title(os.path.basename(os.path.dirname(sys.argv[1])) + f"  (storey 1, {len(f.by_type('IfcWall'))} walls, {len(f.by_type('IfcSpace'))} spaces, units={'mm' if sc < 1 else 'm'})", fontsize=9)
ax.set_xlabel("m"); fig.tight_layout(); fig.savefig(sys.argv[2], dpi=110); print("wrote", sys.argv[2])
