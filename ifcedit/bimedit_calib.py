"""Score BIM-Edit tasks from cached features under a configurable rule set, and fit the open rule choices to the official per-task
scores (calibration models) while reporting agreement on held-out models.

    python -m ifcedit.bimedit_calib --feat outputs/E001/feat --calib claude-sonnet gemini-flash-3.0 --valid gpt-5.4 gpt-5.4-mini
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import pickle

import numpy as np

ROOT = os.environ.get("IFCGYM_ROOT", os.getcwd())
REL_ALL = ["IfcRelVoidsElement", "IfcRelFillsElement", "IfcRelContainedInSpatialStructure", "IfcRelAggregates", "IfcRelConnectsPathElements",
           "IfcRelConnectsElements", "IfcRelSpaceBoundary", "IfcRelDefinesByType", "IfcRelAssociatesMaterial", "IfcRelNests", "IfcRelConnectsPorts"]
DEFAULT = {"align": "gt_guid", "props_mode": "task_relevant", "alpha": 5.0, "lambda_node": 0.3, "min_iou": 0.05, "match": "obb", "require_class_match": True, "zero_if_no_match": True,
           "delete_semantics": "present_is_one", "delete_geometry": "removed_is_one", "rels": tuple(REL_ALL), "props_tol": 0.05,
           "topo_none_rule": True, "anchored_pin": True, "node_f1_align": True}


def _chamfer_median(P, Q):
    from scipy.spatial import cKDTree
    d1, _ = cKDTree(Q).query(P, k=1); d2, _ = cKDTree(P).query(Q, k=1)
    return 0.5 * (float(np.median(d1)) + float(np.median(d2)))


def _props(a: dict, b: dict, tol: float) -> float:
    keys = [k for k in a.keys() if k not in ("Tag", "Description", "LongName")]
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


def score_from_features(F: dict, cfg: dict) -> dict:
    cfg = {**DEFAULT, **cfg}
    z = {"task_id": F["task_id"], "geometry": 0.0, "semantics": 0.0, "topology": 0.0, "final_score": 0.0}
    if not F.get("pred_readable", True):
        return z
    ref, pred = F["ref"], F["pred"]; op = F["operation"]
    ref_live, pred_live = F["ref_live"], F["pred_live"]
    M = F["M_obb"] if cfg["match"] == "obb" else F["M_aabb"]
    # ---- matching (Hungarian) between reference and predicted edited nodes; identity first
    pairs = {}
    if len(ref_live) and len(pred_live):
        from scipy.optimize import linear_sum_assignment
        MM = M.copy()
        for i, a in enumerate(ref_live):
            for j, b in enumerate(pred_live):
                if a == b:
                    MM[i, j] = max(MM[i, j], 1.0)
                if cfg["require_class_match"] and F["ref_cls"].get(a) != F["pred_cls"].get(b):
                    MM[i, j] = 0.0
        r, c = linear_sum_assignment(-MM)
        for i, j in zip(r, c):
            if MM[i, j] >= cfg["min_iou"]:
                pairs[ref_live[i]] = pred_live[j]
    # ---- anchored tasks: pinned target missing in the prediction -> zero
    if cfg["anchored_pin"] and "anchored" in F.get("tags", []) and op != "delete":
        for g, (in_in, in_gt, in_pr) in F.get("target_presence", {}).items():
            if in_gt and not in_pr:
                return z
    # ---- GUID-aligned scoring (official-evaluator hypothesis): compare GT-edited nodes with the prediction's same-GUID nodes
    if cfg["align"] == "gt_guid" and F.get("feat_version", 1) >= 2 and op != "delete":
        same = F["pred_same"]; ref_attrs = F["ref_attrs"]
        # anchored tasks: pinned target missing -> zero
        if cfg["anchored_pin"] and "anchored" in F.get("tags", []):
            for g, (in_in, in_gt, in_pr) in F.get("target_presence", {}).items():
                if in_gt and not in_pr:
                    return z
        # counterparts: same GUID, else OBB-match among prediction's added nodes (new GUIDs)
        pairs = {}
        for a in ref_live:
            if a in same:
                pairs[a] = ("same", a)
        unmatched = [a for a in ref_live if a not in pairs]; pred_added = [b for b in pred_live if b in pred["added"]]
        if unmatched and pred_added:
            from scipy.optimize import linear_sum_assignment
            MM = np.zeros((len(unmatched), len(pred_added)))
            for i, a in enumerate(unmatched):
                for j, b in enumerate(pred_added):
                    ii, jj = ref_live.index(a), pred_live.index(b)
                    MM[i, j] = M[ii, jj] if not (cfg["require_class_match"] and F["ref_cls"].get(a) != F["pred_cls"].get(b)) else 0.0
            r, c = linear_sum_assignment(-MM)
            for i, j in zip(r, c):
                if MM[i, j] >= cfg["min_iou"]:
                    pairs[unmatched[i]] = ("added", pred_added[j])
        if cfg["zero_if_no_match"] and ref_live and not pairs:
            return z
        # geometry: pooled GT clouds vs pooled counterpart clouds
        key = "_geo_gt_guid"
        if key not in F:
            A = [F["ref_pts"][a] for a in ref_live if F["ref_pts"].get(a) is not None]
            B = []
            for a in ref_live:
                if a not in pairs: continue
                kind, b = pairs[a]
                pts = same[b]["pts"] if kind == "same" else F["pred_pts"].get(b)
                if pts is not None: B.append(pts)
            if not A:
                F[key] = ("empty_ref", bool(B))
            elif not B:
                F[key] = ("empty_pred", None)
            else:
                A = np.vstack(A); B = np.vstack(B); both = np.vstack([A, B]); D = float(np.linalg.norm(both.max(0) - both.min(0))) or 1.0
                F[key] = ("cd", _chamfer_median(A, B) / D)
        kind, val = F[key]
        geo = (0.0 if val else 1.0) if kind == "empty_ref" else (0.0 if kind == "empty_pred" else float(math.exp(-val * cfg["alpha"])))
        # semantics: class + properties of each GT node vs its counterpart
        if not ref_live:
            sem = 1.0 if not pred_live else 0.0
        else:
            tot = 0.0
            for a in ref_live:
                if a not in pairs: continue
                kind, b = pairs[a]
                cls_b = same[b]["cls"] if kind == "same" else F["pred_cls"].get(b)
                attrs_b = same[b]["attrs"] if kind == "same" else (pred["added"].get(b) or (None, {}))[1]
                ra_attrs = ref_attrs.get(a, {})
                if cfg["props_mode"] == "task_relevant" and a in ref["modified"]:
                    before = ref["modified"][a][1]; ra_attrs = {k: v for k, v in ra_attrs.items() if before.get(k) != v}
                tot += 0.5 * ((1.0 if F["ref_cls"].get(a) == cls_b else 0.0) + _props(ra_attrs, attrs_b, cfg["props_tol"]))
            sem = tot / len(ref_live)
        align_pairs = {a: b for a, (k, b) in pairs.items()}
        return _finish(F, cfg, geo, sem, align_pairs, ref_live, pred_live, ref, pred)
    # ---- delete
    if op == "delete":
        removed_ok = all(not present for present in F["ref_removed_present_in_pred"].values()) and len(F["ref_removed_present_in_pred"]) > 0
        if cfg["delete_geometry"] == "removed_is_one":
            geo = 1.0 if removed_ok else 0.0
        else:
            geo = 1.0 if removed_ok and not pred_live else 0.0
        if cfg["delete_semantics"] == "present_is_one":
            sem = 0.0 if removed_ok else 1.0
        elif cfg["delete_semantics"] == "removed_is_one":
            sem = 1.0 if removed_ok else 0.0
        else:
            sem = 0.0
    else:
        if cfg["zero_if_no_match"] and ref_live and not pairs:
            return z
        # ---- geometry: pooled clouds of matched (or all live) nodes
        if "_geo_cd" not in F:
            P_ref = [F["ref_pts"][k] for k in ref_live if F["ref_pts"].get(k) is not None]
            P_pred = [F["pred_pts"][k] for k in pred_live if F["pred_pts"].get(k) is not None]
            if not P_ref:
                F["_geo_cd"] = ("empty_ref", bool(P_pred))
            elif not P_pred:
                F["_geo_cd"] = ("empty_pred", None)
            else:
                A, B = np.vstack(P_ref), np.vstack(P_pred); both = np.vstack([A, B]); D = float(np.linalg.norm(both.max(0) - both.min(0))) or 1.0
                F["_geo_cd"] = ("cd", _chamfer_median(A, B) / D)
        kind, val = F["_geo_cd"]
        if kind == "empty_ref":
            geo = 0.0 if val else 1.0
        elif kind == "empty_pred":
            geo = 0.0
        else:
            geo = float(math.exp(-val * cfg["alpha"]))
        # ---- semantics
        if not ref_live:
            sem = 1.0 if not pred_live else 0.0
        else:
            tot = 0.0
            for a in ref_live:
                b = pairs.get(a)
                if b is None:
                    continue
                ra = ref["added"].get(a) or (ref["modified"].get(a) and (ref["modified"][a][2], ref["modified"][a][3]))
                pb = pred["added"].get(b) or (pred["modified"].get(b) and (pred["modified"][b][2], pred["modified"][b][3]))
                cls = 1.0 if F["ref_cls"].get(a) == F["pred_cls"].get(b) else 0.0
                ra_attrs, pb_attrs = (ra[1] if ra else {}), (pb[1] if pb else {})
                if cfg["props_mode"] == "task_relevant" and a in ref["modified"]:
                    before = ref["modified"][a][1]; ra_attrs = {k: v for k, v in ra_attrs.items() if before.get(k) != v}
                pr = _props(ra_attrs, pb_attrs, cfg["props_tol"])
                tot += 0.5 * (cls + pr)
            sem = tot / len(ref_live)
    return _finish(F, cfg, geo, sem, pairs, ref_live, pred_live, ref, pred)


def _finish(F, cfg, geo, sem, pairs, ref_live, pred_live, ref, pred):
    # ---- topology on edited nodes and relations (restricted to the chosen relation types)
    rels = set(cfg["rels"])
    align = {a: a for a in ref_live}; align.update(pairs)
    def relset(g):
        return {tuple(k) for k in g["rel_added"] + g["rel_removed"] if k[0] in rels}
    ref_rel = {tuple(align.get(x, x) if isinstance(x, str) else x for x in k) for k in relset(ref)}
    pred_rel = relset(pred)
    n_ref = {align.get(a, a) for a in set(ref["added"]) | set(ref["removed"]) | set(ref["modified"])}
    n_pred = set(pred["added"]) | set(pred["removed"]) | set(pred["modified"])
    def f1(x, y):
        if not x and not y:
            return None
        tp = len(x & y); p = tp / len(y) if y else 0.0; r = tp / len(x) if x else 0.0
        return 0.0 if p + r == 0 else 2 * p * r / (p + r)
    fn, fr = f1(n_ref, n_pred), f1(ref_rel, pred_rel)
    if fr is None:
        topo = 1.0 if (cfg["topo_none_rule"] and not pred_rel) else 0.0
    else:
        topo = cfg["lambda_node"] * (fn if fn is not None else 1.0) + (1 - cfg["lambda_node"]) * fr
    return {"task_id": F["task_id"], "geometry": geo, "semantics": sem, "topology": topo, "final_score": float(np.mean([geo, sem, topo]))}


def load_feats(d: str) -> dict:
    out = {}
    for fn in os.listdir(d):
        if fn.endswith(".pkl"):
            with open(os.path.join(d, fn), "rb") as fh:
                F = pickle.load(fh); out[F["task_id"]] = F
    return out


def official(model: str) -> dict:
    p = os.path.join(ROOT, "data/bimedit-runs", model, "eval/per_task.csv")
    return {r["task_id"]: r for r in csv.DictReader(open(p))}


def agreement(feats: dict, off: dict, cfg: dict) -> dict:
    rows = []
    for t, F in feats.items():
        if t not in off:
            continue
        s = score_from_features(F, cfg); o = off[t]
        rows.append([s[k] - float(o[k]) for k in ("geometry", "semantics", "topology", "final_score")] + [s["final_score"], float(o["final_score"])])
    A = np.array(rows)
    return {"n": len(rows), "mae": np.abs(A[:, :4]).mean(0).round(4).tolist(), "within05": round(float(np.mean(np.abs(A[:, 3]) <= 0.05)), 3),
            "ours_mean": round(float(A[:, 4].mean()), 4), "off_mean": round(float(A[:, 5].mean()), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True); ap.add_argument("--calib", nargs="+", required=True); ap.add_argument("--valid", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    feats = {m: load_feats(os.path.join(a.feat, m)) for m in a.calib + a.valid if os.path.isdir(os.path.join(a.feat, m))}
    offs = {m: official(m) for m in feats}
    grid = {"align": ["gt_guid", "edit"], "props_mode": ["task_relevant", "all"], "match": ["obb", "aabb"], "require_class_match": [True, False], "zero_if_no_match": [True, False],
            "delete_semantics": ["present_is_one", "removed_is_one"], "min_iou": [0.05, 0.2],
            "rels": [tuple(REL_ALL), ("IfcRelVoidsElement", "IfcRelFillsElement", "IfcRelContainedInSpatialStructure", "IfcRelAggregates", "IfcRelConnectsPathElements", "IfcRelSpaceBoundary"),
                     ("IfcRelVoidsElement", "IfcRelFillsElement", "IfcRelContainedInSpatialStructure", "IfcRelConnectsPathElements"),
                     ("IfcRelVoidsElement", "IfcRelFillsElement", "IfcRelContainedInSpatialStructure", "IfcRelAggregates", "IfcRelConnectsPathElements", "IfcRelSpaceBoundary", "IfcRelDefinesByType", "IfcRelAssociatesMaterial")],
            "anchored_pin": [True, False]}
    keys = list(grid); best = None; results = []
    for vals in itertools.product(*[grid[k] for k in keys]):
        cfg = dict(zip(keys, vals))
        score = np.mean([agreement(feats[m], offs[m], cfg)["mae"][3] for m in a.calib if m in feats])
        results.append((score, cfg))
        if best is None or score < best[0]:
            best = (score, cfg)
    results.sort(key=lambda x: x[0])
    print("top 5 configs on calibration models (mean abs final diff):")
    for s, cfg in results[:5]:
        print(f"  {s:.4f}  {json.dumps({k: (v if not isinstance(v, tuple) else len(v)) for k, v in cfg.items()})}")
    print("\nbest config per model:")
    for m in feats:
        print(f"  {m:18s} {'calib' if m in a.calib else 'VALID'} {agreement(feats[m], offs[m], best[1])}")
    print("\ndefault config per model:")
    for m in feats:
        print(f"  {m:18s} {agreement(feats[m], offs[m], {})}")
    if a.out:
        json.dump({"best": {k: (list(v) if isinstance(v, tuple) else v) for k, v in best[1].items()}, "score": best[0],
                   "per_model": {m: agreement(feats[m], offs[m], best[1]) for m in feats}}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
