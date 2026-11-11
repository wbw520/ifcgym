"""Re-implementation of the BIM-Edit scorer (arXiv:2606.20146 Sec. 3.2, Eqs. 1-4) with the released evaluator defaults
(data/bimedit-runs/*/eval/resolved_defaults.json). Validated against the official per-task scores of the released edited outputs.

    python -m ifcedit.bimedit_eval --run outputs/<run> [--tasks ...] [--workers 8] [--report]
    python -m ifcedit.bimedit_eval --official data/bimedit-runs/claude-sonnet --report      # validation against per_task.csv

Score per task: S = mean(S_geo, S_sem, S_topo) on the edit graph G_delta (input -> ground truth) vs (input -> prediction).
  S_geo  = exp(-CD_med(P*, P') / D * alpha), alpha=5, point clouds pooled over the edited nodes (4096 samples/object, max 16384).
  S_sem  = mean over reference edited nodes of 0.5*(class match) + 0.5*(property agreement), Hungarian matching by OBB-IoU (min 0.05).
  S_topo = 0.3*F1(edited nodes) + 0.7*F1(edited relations); 1 if neither side has topological modifications, 0 if only one has.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from ifcgym.config import resolve_bimedit_ifc, BIMEDIT_TASKS as TASKS
DEFAULTS = {"alpha": 5.0, "samples_per_object": 4096, "max_total_samples": 16384, "match_min_iou": 0.05, "lambda_node": 0.3,
            "properties_tolerance": 0.05, "properties_ignore": ["Tag", "Description", "LongName"]}


# ----------------------------------------------------------------------------- geometry helpers
def _mesh(f, inst):
    import ifcopenshell.geom
    s = ifcopenshell.geom.settings()
    try:
        shp = ifcopenshell.geom.create_shape(s, inst)
    except Exception:  # noqa: BLE001
        return None
    v = np.array(shp.geometry.verts, dtype=float).reshape(-1, 3)
    fcs = np.array(shp.geometry.faces, dtype=int).reshape(-1, 3)
    if len(v) == 0 or len(fcs) == 0:
        return None
    from ifcgym.ifcgraph import world_verts
    v = world_verts(shp)
    return v, fcs


def _sample_surface(v, fcs, n, rng):
    a, b, c = v[fcs[:, 0]], v[fcs[:, 1]], v[fcs[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    if areas.sum() <= 0:
        return v[rng.integers(0, len(v), n)]
    idx = rng.choice(len(fcs), size=n, p=areas / areas.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s1 = np.sqrt(r1)
    return (1 - s1)[:, None] * a[idx] + (s1 * (1 - r2))[:, None] * b[idx] + (s1 * r2)[:, None] * c[idx]


def _obb(v):
    """Oriented bounding box via PCA: (center, axes 3x3, half extents)."""
    c = v.mean(0); X = v - c
    if len(v) < 3:
        return c, np.eye(3), np.maximum(np.abs(X).max(0) if len(v) else np.zeros(3), 1e-6)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    axes = vt
    proj = X @ axes.T
    lo, hi = proj.min(0), proj.max(0)
    center = c + ((lo + hi) / 2) @ axes
    return center, axes, np.maximum((hi - lo) / 2, 1e-6)


def _obb_iou(p, q, n=20000, rng=None):
    """Monte-Carlo IoU of two OBBs (point clouds p, q)."""
    rng = rng or np.random.default_rng(0)
    (c1, a1, h1), (c2, a2, h2) = _obb(p), _obb(q)
    def inside(x, c, a, h):
        y = (x - c) @ a.T
        return np.all(np.abs(y) <= h, axis=1)
    def sample(c, a, h, k):
        u = rng.uniform(-1, 1, (k, 3)) * h
        return c + u @ a
    s1, s2 = sample(c1, a1, h1, n), sample(c2, a2, h2, n)
    v1, v2 = 8 * np.prod(h1), 8 * np.prod(h2)
    inter = 0.5 * (inside(s1, c2, a2, h2).mean() * v1 + inside(s2, c1, a1, h1).mean() * v2)
    union = v1 + v2 - inter
    return float(inter / union) if union > 0 else 0.0


def _chamfer_median(P, Q):
    from scipy.spatial import cKDTree
    d1, _ = cKDTree(Q).query(P, k=1); d2, _ = cKDTree(P).query(Q, k=1)
    return 0.5 * (float(np.median(d1)) + float(np.median(d2)))


# ----------------------------------------------------------------------------- scoring
def _node_points(f, guid, n, rng):
    try:
        inst = f.by_guid(guid)
    except Exception:  # noqa: BLE001
        return None
    if not getattr(inst, "Representation", None):
        return None
    m = _mesh(f, inst)
    return None if m is None else _sample_surface(m[0], m[1], n, rng)


def _props_agreement(a: dict, b: dict, tol: float) -> float:
    keys = [k for k in a.keys() if k not in DEFAULTS["properties_ignore"]]
    if not keys:
        return 1.0
    ok = 0
    for k in keys:
        x, y = a.get(k), b.get(k)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
            ok += abs(x - y) <= tol * max(abs(x), 1e-9) + 1e-9
        else:
            ok += x == y
    return ok / len(keys)


def score_task(task: dict, pred_path: str, cfg: dict | None = None) -> dict:
    import ifcopenshell
    from ifcgym.ifcgraph import edit_graph
    cfg = {**DEFAULTS, **(cfg or {})}
    rng = np.random.default_rng(0)
    fin = ifcopenshell.open(resolve_bimedit_ifc(task["input_ifc"]))
    fgt = ifcopenshell.open(resolve_bimedit_ifc(task["ground_truth_ifc"]))
    try:
        fpr = ifcopenshell.open(pred_path)
    except Exception as e:  # noqa: BLE001
        return {"task_id": task["task_id"], "geometry": 0.0, "semantics": 0.0, "topology": 0.0, "final_score": 0.0, "error": f"unreadable prediction: {e}"}
    g_ref, g_pred = edit_graph(fin, fgt), edit_graph(fin, fpr)
    ref_nodes, pred_nodes = g_ref.node_ids, g_pred.node_ids
    out = {"task_id": task["task_id"], "ref_edit": g_ref.summary(), "pred_edit": g_pred.summary()}

    # --- geometry: pooled surface point clouds of the edited nodes (reference geometry from GT, predicted from the prediction)
    def pooled(f, ids, removed_ids):
        pts = []
        live = [g for g in ids if g not in removed_ids]
        if not live:
            return None
        per = max(64, min(cfg["samples_per_object"], cfg["max_total_samples"] // len(live)))
        for g in live:
            p = _node_points(f, g, per, rng)
            if p is not None:
                pts.append(p)
        return np.vstack(pts) if pts else None
    P_ref = pooled(fgt, ref_nodes, set(g_ref.removed))
    P_pred = pooled(fpr, pred_nodes, set(g_pred.removed))
    if task["operation"] == "delete" and P_ref is None:
        # reference edit removes geometry; prediction is right when the removed targets are gone and nothing else was changed
        out["geometry"] = 1.0 if (set(g_ref.removed) <= set(g_pred.removed) and P_pred is None) else (0.5 if set(g_ref.removed) <= set(g_pred.removed) else 0.0)
    elif P_ref is None or P_pred is None:
        out["geometry"] = 0.0
    else:
        both = np.vstack([P_ref, P_pred]); D = float(np.linalg.norm(both.max(0) - both.min(0))) or 1.0
        cd = _chamfer_median(P_ref, P_pred)
        out["geometry"] = float(math.exp(-cd / D * cfg["alpha"]))
        out["chamfer_median"] = cd; out["bbox_diag"] = D

    # --- semantics: Hungarian matching of reference edited nodes to predicted edited nodes by OBB-IoU; class + properties
    ref_live = [g for g in ref_nodes if g not in g_ref.removed]
    pred_live = [g for g in pred_nodes if g not in g_pred.removed]
    if task["operation"] == "delete" and not ref_live:
        out["semantics"] = 1.0 if set(g_ref.removed) <= set(g_pred.removed) else 0.0
    elif not ref_live:
        out["semantics"] = 1.0 if not pred_live else 0.0
    else:
        from scipy.optimize import linear_sum_assignment
        ref_pts = {g: _node_points(fgt, g, 2048, rng) for g in ref_live}
        pred_pts = {g: _node_points(fpr, g, 2048, rng) for g in pred_live}
        pairs = {}
        if pred_live:
            M = np.zeros((len(ref_live), len(pred_live)))
            for i, a in enumerate(ref_live):
                for j, b in enumerate(pred_live):
                    if a == b and a in g_pred.modified and a in g_ref.modified:
                        M[i, j] = 1.0        # same identity edited on both sides
                    elif ref_pts[a] is not None and pred_pts[b] is not None:
                        M[i, j] = _obb_iou(ref_pts[a], pred_pts[b], rng=rng)
            r, c = linear_sum_assignment(-M)
            for i, j in zip(r, c):
                if M[i, j] >= cfg["match_min_iou"]:
                    pairs[ref_live[i]] = pred_live[j]
        tot = 0.0
        for a in ref_live:
            b = pairs.get(a)
            if b is None:
                continue
            na = fgt.by_guid(a); nb = fpr.by_guid(b)
            cls = 1.0 if na.is_a() == nb.is_a() else 0.0
            from ifcgym.ifcgraph import build_graph  # noqa: F401  (attrs come from the graphs below)
            ra = (g_ref.added.get(a) or (g_ref.modified.get(a) or (None, None))[1])
            pb = (g_pred.added.get(b) or (g_pred.modified.get(b) or (None, None))[1])
            props = _props_agreement(ra.attrs if ra else {}, pb.attrs if pb else {}, cfg["properties_tolerance"])
            tot += 0.5 * (cls + props)
        out["semantics"] = tot / len(ref_live)
        out["matched"] = len(pairs)

    # --- topology: F1 over edited nodes and edited relations (greedy identity alignment: same guid, else the semantic match)
    align = {a: a for a in ref_nodes if a in pred_nodes}
    if "pairs" in dir() and isinstance(pairs, dict):
        for a, b in pairs.items():
            align.setdefault(a, b)
    def f1(ref: set, pred: set):
        if not ref and not pred:
            return None
        tp = len(ref & pred)
        p = tp / len(pred) if pred else 0.0; r = tp / len(ref) if ref else 0.0
        return 0.0 if p + r == 0 else 2 * p * r / (p + r)
    ref_rel = {tuple(align.get(x, x) if isinstance(x, str) else x for x in k) for k in list(g_ref.rel_added) + list(g_ref.rel_removed)}
    pred_rel = set(list(g_pred.rel_added) + list(g_pred.rel_removed))
    n_ref = {align.get(a, a) for a in ref_nodes}; n_pred = set(pred_nodes)
    fn, fr = f1(n_ref, n_pred), f1(ref_rel, pred_rel)
    if fr is None:
        out["topology"] = 1.0 if not pred_rel else 0.0
        if fn is not None and not pred_rel:
            out["topology"] = 1.0
    else:
        out["topology"] = cfg["lambda_node"] * (fn if fn is not None else 1.0) + (1 - cfg["lambda_node"]) * fr
    out["final_score"] = float(np.mean([out["geometry"], out["semantics"], out["topology"]]))
    return out


def _score_wrapper(args):
    task, pred = args
    try:
        return score_task(task, pred)
    except Exception as e:  # noqa: BLE001
        return {"task_id": task["task_id"], "geometry": 0.0, "semantics": 0.0, "topology": 0.0, "final_score": 0.0, "error": f"{type(e).__name__}: {e}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="our run dir with edited/<task_id>.ifc")
    ap.add_argument("--official", help="official BIM-Edit-runs/<model> dir with edited/<task_id>/<file>_0.ifc and eval/per_task.csv")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    tasks = [json.loads(l) for l in open(TASKS) if l.strip()]
    if a.tasks:
        tasks = [t for t in tasks if t["task_id"] in set(a.tasks)]
    jobs = []
    for t in tasks:
        if a.run:
            p = os.path.join(a.run, "edited", t["task_id"] + ".ifc")
        else:
            d = os.path.join(a.official, "edited", t["task_id"])
            fs = sorted(os.listdir(d)) if os.path.isdir(d) else []
            p = os.path.join(d, fs[0]) if fs else None
        if p and os.path.exists(p):
            jobs.append((t, p))
    jobs = jobs[: a.limit]
    out_dir = a.run or a.official
    res = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(_score_wrapper, j) for j in jobs]):
            r = f.result(); res.append(r)
            print(f"{r['task_id']}: geo={r.get('geometry',0):.3f} sem={r.get('semantics',0):.3f} topo={r.get('topology',0):.3f} final={r.get('final_score',0):.3f} {r.get('error','')}", flush=True)
    res.sort(key=lambda r: r["task_id"])
    with open(os.path.join(out_dir, "ifcedit_eval.jsonl"), "w") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")
    final = np.array([r["final_score"] for r in res]); strict = np.mean([all(r[k] >= 0.98 for k in ("geometry", "semantics", "topology")) for r in res]) if res else 0
    print(f"metric=bimedit_final {100*final.mean():.2f}")
    print(f"metric=bimedit_strict {100*strict:.2f}")
    print(json.dumps({"n": len(res), "final_mean": 100 * float(final.mean()) if res else None, "strict": 100 * float(strict),
                      "geometry_mean": 100 * float(np.mean([r['geometry'] for r in res])) if res else None,
                      "semantics_mean": 100 * float(np.mean([r['semantics'] for r in res])) if res else None,
                      "topology_mean": 100 * float(np.mean([r['topology'] for r in res])) if res else None}))
    if a.report and a.official and os.path.exists(os.path.join(a.official, "eval/per_task.csv")):
        off = {r["task_id"]: r for r in csv.DictReader(open(os.path.join(a.official, "eval/per_task.csv")))}
        diffs = []
        for r in res:
            o = off.get(r["task_id"])
            if not o:
                continue
            d = {k: float(r[k]) - float(o[k]) for k in ("geometry", "semantics", "topology", "final_score")}
            diffs.append(d)
            print(f"  {r['task_id']}: ours final={r['final_score']:.3f} official={float(o['final_score']):.3f} "
                  f"(geo {float(o['geometry']):.2f}/{r['geometry']:.2f} sem {float(o['semantics']):.2f}/{r['semantics']:.2f} topo {float(o['topology']):.2f}/{r['topology']:.2f})")
        if diffs:
            A = np.array([[d[k] for k in ("geometry", "semantics", "topology", "final_score")] for d in diffs])
            print("agreement vs official (n=%d): mean abs diff geo=%.3f sem=%.3f topo=%.3f final=%.3f; |final diff|<=0.05 on %.1f%% of tasks"
                  % (len(diffs), *np.abs(A).mean(0), 100 * np.mean(np.abs(A[:, 3]) <= 0.05)))
            print(f"metric=evaluator_agreement {100*np.mean(np.abs(A[:, 3]) <= 0.05):.1f}")


if __name__ == "__main__":
    main()
