"""Per-task feature cache for calibrating the BIM-Edit evaluator re-implementation.

Everything expensive (edit graphs, point clouds, OBB-IoU match matrix, relation sets by type) is computed ONCE per (prediction file)
and pickled; `score_from_features(feat, cfg)` then evaluates any scoring configuration in milliseconds, so the rule choices that the
paper leaves open (which relation types count, how deletes are scored, when a task collapses to zero) can be fitted on the released
per-task scores of two models and validated on two others.

    python -m ifcedit.bimedit_features --official data/bimedit-runs/claude-sonnet --out outputs/<run>/feat/claude-sonnet [--workers 30]
    python -m ifcedit.bimedit_features --run outputs/<run> --out outputs/<run>/feat
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from ifcgym.config import resolve_bimedit_ifc, BIMEDIT_TASKS as TASKS


def _points(f, guid, n, rng, no_openings=False):
    import ifcopenshell.geom
    from ifcgym.ifcgraph import world_verts
    try:
        inst = f.by_guid(guid)
    except Exception:  # noqa: BLE001
        return None
    if not getattr(inst, "Representation", None):
        return None
    s = ifcopenshell.geom.settings()
    try:
        shp = ifcopenshell.geom.create_shape(s, inst)
    except Exception:  # noqa: BLE001
        return None
    v = world_verts(shp); fcs = np.array(shp.geometry.faces, dtype=int).reshape(-1, 3)
    if len(v) == 0 or len(fcs) == 0:
        return None
    a, b, c = v[fcs[:, 0]], v[fcs[:, 1]], v[fcs[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    if areas.sum() <= 0:
        return v[rng.integers(0, len(v), n)]
    idx = rng.choice(len(fcs), size=n, p=areas / areas.sum()); r1, r2 = rng.random(n), rng.random(n); s1 = np.sqrt(r1)
    return (1 - s1)[:, None] * a[idx] + (s1 * (1 - r2))[:, None] * b[idx] + (s1 * r2)[:, None] * c[idx]


def _obb(v):
    c = v.mean(0); X = v - c
    if len(v) < 3:
        return c, np.eye(3), np.maximum(np.abs(X).max(0) if len(v) else np.zeros(3), 1e-6)
    _, _, vt = np.linalg.svd(X, full_matrices=False); proj = X @ vt.T; lo, hi = proj.min(0), proj.max(0)
    return c + ((lo + hi) / 2) @ vt, vt, np.maximum((hi - lo) / 2, 1e-6)


def _obb_iou(p, q, rng, n=20000):
    (c1, a1, h1), (c2, a2, h2) = _obb(p), _obb(q)
    def inside(x, c, a, h):
        y = (x - c) @ a.T; return np.all(np.abs(y) <= h, axis=1)
    s1 = c1 + (rng.uniform(-1, 1, (n, 3)) * h1) @ a1; s2 = c2 + (rng.uniform(-1, 1, (n, 3)) * h2) @ a2
    v1, v2 = 8 * np.prod(h1), 8 * np.prod(h2)
    inter = 0.5 * (inside(s1, c2, a2, h2).mean() * v1 + inside(s2, c1, a1, h1).mean() * v2); union = v1 + v2 - inter
    return float(inter / union) if union > 0 else 0.0


def _aabb_iou(p, q):
    lo1, hi1, lo2, hi2 = p.min(0), p.max(0), q.min(0), q.max(0)
    inter = np.prod(np.clip(np.minimum(hi1, hi2) - np.maximum(lo1, lo2), 0, None)); v1, v2 = np.prod(hi1 - lo1), np.prod(hi2 - lo2)
    u = v1 + v2 - inter; return float(inter / u) if u > 0 else 0.0


GRAPH_CACHE_DIR = os.environ.get("BIMEDIT_GRAPH_CACHE", os.path.join(os.environ.get("IFCGYM_ROOT", os.getcwd()), "outputs", "graph_cache"))
_GMEMO: dict = {}


def graph_cached(path: str):
    """build_graph of an IMMUTABLE benchmark file (input / ground truth), cached on disk and in memory (the 45 tasks on
    the 154-349 MB models rebuilt the same graph up to five times per task; results are identical to building it every time)."""
    import hashlib, pickle
    import ifcopenshell
    from ifcgym.ifcgraph import build_graph
    st = os.stat(path); key = hashlib.sha1(f"{os.path.realpath(path)}|{st.st_size}|{int(st.st_mtime)}|graph-v1".encode()).hexdigest()[:20]
    if key in _GMEMO:
        return _GMEMO[key]
    cp = os.path.join(GRAPH_CACHE_DIR, key + ".pkl")
    if os.path.exists(cp):
        g = pickle.load(open(cp, "rb"))
    else:
        g = build_graph(ifcopenshell.open(path)); os.makedirs(GRAPH_CACHE_DIR, exist_ok=True)
        tmp = f"{cp}.{os.getpid()}.tmp"; pickle.dump(g, open(tmp, "wb"), protocol=5); os.replace(tmp, cp)
    _GMEMO[key] = g
    return g


def features(task: dict, pred_path: str) -> dict:
    import ifcopenshell
    from ifcgym.ifcgraph import build_graph, diff_graphs
    rng = np.random.default_rng(0)
    in_path, gt_path = resolve_bimedit_ifc(task["input_ifc"]), resolve_bimedit_ifc(task["ground_truth_ifc"])
    G_in, G_gt = graph_cached(in_path), graph_cached(gt_path)
    _fin = []
    def fin_file():
        if not _fin:
            _fin.append(ifcopenshell.open(in_path))
        return _fin[0]
    fgt = ifcopenshell.open(gt_path)
    out = {"task_id": task["task_id"], "operation": task["operation"], "category": task["category"], "tags": task.get("tags", []),
           "target": task.get("target", {}), "pred_readable": True}
    try:
        fpr = ifcopenshell.open(pred_path)
    except Exception as e:  # noqa: BLE001
        out["pred_readable"] = False; out["error"] = str(e); return out
    G_pr = build_graph(fpr)
    g_ref, g_pred = diff_graphs(G_in, G_gt), diff_graphs(G_in, G_pr)
    def pack(g):
        return {"added": {k: (n.cls, n.attrs) for k, n in g.added.items()}, "removed": {k: (n.cls, n.attrs) for k, n in g.removed.items()},
                "modified": {k: (b.cls, b.attrs, a.cls, a.attrs, b.geom_hash != a.geom_hash, b.placement_hash != a.placement_hash, b.attrs != a.attrs) for k, (b, a) in g.modified.items()},
                "rel_added": sorted(g.rel_added.keys(), key=str), "rel_removed": sorted(g.rel_removed.keys(), key=str)}
    out["ref"], out["pred"] = pack(g_ref), pack(g_pred)
    # pinned target guids (anchored tasks): present in input / gt / pred?
    tg = [g for g in (task.get("target") or {}).get("guids", [])]
    def has(f, g):
        try: f.by_guid(g); return True
        except Exception: return False  # noqa: BLE001
    out["target_presence"] = {g: (has(fin_file(), g), has(fgt, g), has(fpr, g)) for g in tg}
    # point clouds of the live edited nodes on both sides (4096 each, capped at 16384 total) and per-node clouds for matching
    def live(g):
        return [k for k in (set(g.added) | set(g.modified)) ]
    ref_live, pred_live = sorted(live(g_ref)), sorted(live(g_pred))
    per = lambda n: max(64, min(4096, 16384 // max(1, n)))
    out["ref_pts"] = {k: _points(fgt, k, per(len(ref_live)), rng) for k in ref_live}
    out["pred_pts"] = {k: _points(fpr, k, per(len(pred_live)), rng) for k in pred_live}
    out["ref_cls"] = {k: fgt.by_guid(k).is_a() for k in ref_live}; out["pred_cls"] = {k: fpr.by_guid(k).is_a() for k in pred_live}
    # target class in the GT (for class_match semantics) and whether the target still exists in pred
    M_obb = np.zeros((len(ref_live), len(pred_live))); M_aabb = np.zeros_like(M_obb)
    for i, a in enumerate(ref_live):
        for j, b in enumerate(pred_live):
            pa, pb = out["ref_pts"][a], out["pred_pts"][b]
            if pa is not None and pb is not None:
                M_obb[i, j] = _obb_iou(pa, pb, rng); M_aabb[i, j] = _aabb_iou(pa, pb)
    out["ref_live"], out["pred_live"], out["M_obb"], out["M_aabb"] = ref_live, pred_live, M_obb, M_aabb
    # removed targets still present?
    out["ref_removed_present_in_pred"] = {k: has(fpr, k) for k in g_ref.removed}
    out["pred_removed_cls"] = {k: n.cls for k, n in g_pred.removed.items()}
    # GUID-aligned view (official-evaluator hypothesis: GT-edited nodes are compared with the prediction's SAME-GUID nodes
    # whatever the prediction changed): for every GT live node, the prediction's node of the same GUID - its cloud, class and attrs
    pred_nodes = G_pr[0]
    out["pred_same"] = {}
    for k in ref_live:
        if k in pred_nodes:
            out["pred_same"][k] = {"cls": pred_nodes[k].cls, "attrs": pred_nodes[k].attrs, "pts": _points(fpr, k, per(len(ref_live)), rng)}
    out["ref_attrs"] = {k: (g_ref.added[k].attrs if k in g_ref.added else g_ref.modified[k][1].attrs) for k in ref_live}
    out["feat_version"] = 2
    return out


def _wrap(args):
    task, pred, outp = args
    try:
        f = features(task, pred)
    except (OSError, MemoryError) as e:   # environment (missing input / torn-down mount / memory), not the prediction: write nothing so
        return task["task_id"], f"NOT WRITTEN {type(e).__name__}: {e}"   # a rerun retries it (190 such records scored 0)
    except Exception as e:  # noqa: BLE001
        f = {"task_id": task["task_id"], "operation": task["operation"], "category": task["category"], "pred_readable": False, "error": f"{type(e).__name__}: {e}"}
    with open(outp, "wb") as fh:
        pickle.dump(f, fh)
    return f["task_id"], f.get("error")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run"); ap.add_argument("--official"); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--limit", type=int, default=None); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tasks = [json.loads(l) for l in open(TASKS) if l.strip()][: a.limit]
    jobs = []
    for t in tasks:
        if a.run:
            p = os.path.join(a.run, "edited", t["task_id"] + ".ifc")
        else:
            d = os.path.join(a.official, "edited", t["task_id"]); fs = sorted(os.listdir(d)) if os.path.isdir(d) else []
            p = os.path.join(d, fs[0]) if fs else None
        outp = os.path.join(a.out, t["task_id"] + ".pkl")
        if p and os.path.exists(p) and (a.force or not os.path.exists(outp)):
            jobs.append((t, p, outp))
    print(f"{len(jobs)} tasks to featurise -> {a.out}", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(_wrap, j) for j in jobs]):
            tid, err = f.result(); done += 1
            print(f"[{done}/{len(jobs)}] {tid} {err or ''}", flush=True)


if __name__ == "__main__":
    main()
