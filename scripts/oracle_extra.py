"""Checker validation for the extra template families: a scripted agent performs create tasks with h.add_* and answers
retrieval tasks from the model (spatial tasks reuse the plain edits); the frozen checkers must accept them.
    python scripts/oracle_extra.py <tasks.jsonl>"""
import json, os, sys, tempfile, shutil, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import ifcopenshell
from ifcgym import ifctools as h
from ifcgym.checkers import check

tasks = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
OLD_PLAIN = ("T0-set-fire-rating", "T0-set-name", "T1-delete-door-with-opening", "T1-set-wall-height")   # no solver covered these
tasks = [t for t in tasks if (t["template"].startswith(("C-", "R-", "S-")) and t["template"][-2:] not in ("-D", "-T", "-S", "-P", "-C", "-V")) or t["template"] in OLD_PLAIN]   # v4 templates: scripts/oracle_v4.py; T1-T3 moves: oracle_cascade.py
NEG = "--negative" in sys.argv   # negative control: a wrong solution must FAIL the checker
res = collections.defaultdict(list)
for t in tasks:
    f = ifcopenshell.open(t["ifc"]); sc = h.unit_scale(f) or 1.0; it = dict(t["intent"]); final = ""
    if NEG:
        if it["type"] in ("set_property", "set_attribute"): it["value"] = str(it["value"]) + "_x"
        elif it["type"] == "resize_height": it["height"] = it["height"] + 0.3
        elif it["type"] == "delete": it["type"] = "delete_fill_only"
        elif it["type"] == "answer": it["value"] = (it["value"] * 1.5 + 1) if isinstance(it["value"], (int, float)) else "0000000000000000000000"
        else: it["type"] = "noop"
    try:
        if it["type"] == "create_fill":   # first free slot along the wall (the checker rejects overlaps with existing openings)
            w = f.by_guid(it["host"]); ax = h.wall_axis_world(w); L = float(np.linalg.norm(np.array(ax[1]) - np.array(ax[0]))); W = it["width"] / sc
            a = np.array(ax[0][:2]); u = np.array(ax[1][:2]) - a; u = u / (np.linalg.norm(u) or 1.0); busy = []
            for r_ in w.HasOpenings or []:
                bb = h._mesh_bbox(r_.RelatedOpeningElement)
                if bb:
                    cs = [np.array([bb["min"][0], bb["min"][1]]), np.array([bb["max"][0], bb["max"][1]]), np.array([bb["min"][0], bb["max"][1]]), np.array([bb["max"][0], bb["min"][1]])]
                    ps = [float((c_ - a) @ u) for c_ in cs]; busy.append((min(ps), max(ps)))
            along = None; x = 0.1 / sc
            while x + W <= L - 0.1 / sc:
                if all(x + W <= lo - 0.02 / sc or x >= hi + 0.02 / sc for lo, hi in busy):
                    along = x; break
                x += 0.05 / sc
            if along is None:
                raise ValueError("no free slot on the wall")
            h.add_opening_fill(f, w, along=along, width=W, height=it["height"] / sc, kind=it["kind"], sill=(it.get("sill", 0.9 if it["kind"] == "window" else 0.0) / sc))
        elif it["type"] == "create_wall":
            h.add_wall(f, it["storey"], np.array(it["p0"]) / sc, np.array(it["p1"]) / sc, it["thickness"] / sc, it["height"] / sc)
        elif it["type"] == "create_slab":
            h.add_slab(f, it["storey"], (np.array(it["polygon"]) / sc).tolist(), it["thickness"] / sc, elevation=-it["thickness"] / sc)
        elif it["type"] == "create_column":
            h.add_column(f, it["storey"], np.array(it["xy"]) / sc, it["size"] / sc, it["height"] / sc)
        elif it["type"] == "answer":
            if NEG:
                final = json.dumps({"value": it["value"]} if isinstance(it["value"], (int, float)) else {"global_id": it["value"]})
            elif t["template"] == "R-space-area":
                final = json.dumps({"value": round(h.space_area(f, t["target"][0]), 2)})
            elif t["template"] == "R-count-doors":
                import ifcopenshell.util.element as ue
                st = f.by_guid(t["target"][0]); final = json.dumps({"value": sum(1 for d in f.by_type("IfcDoor") if ue.get_container(d) == st)})
            else:
                final = json.dumps({"global_id": it["value"]})
        elif it["type"] == "set_property":
            h.set_property(f, it["guid"], it["pset"], it["prop"], it["value"])
        elif it["type"] == "resize_height":
            h.set_extrusion_depth(f, it["guid"], it["height"] / sc)
        elif it["type"] == "set_attribute":
            h.set_attribute(f, it["guid"], it["name"], it["value"])
        elif it["type"] == "delete":     # door with its opening (dependents: the opening 'removed', the host wall 'present')
            e = f.by_guid(it["guid"]); ops = [r.RelatingOpeningElement for r in (getattr(e, "FillsVoids", None) or [])]
            h.delete(f, e)
            for op in ops:
                h.delete(f, op)
        elif it["type"] == "delete_fill_only":
            h.delete(f, f.by_guid(it["guid"]))
        out = os.path.join(tempfile.mkdtemp(prefix="oracle_x_"), "e.ifc"); f.write(out)
        r = check(t["ifc"], out, t, final)
        shutil.rmtree(os.path.dirname(out), ignore_errors=True)   # no leftover temp dirs
        res[t["template"]].append(r.strict != NEG)
        if r.strict == NEG:
            print(f"FAIL {t['task_id']}: intent={r.intent} dep={r.dependency} pres={r.preservation} valid={r.validity} | {r.details['intent'][:120]} | {r.details['preservation'][:100]} | {r.details['validity'][:60]}")
    except Exception as e:  # noqa: BLE001
        res[t["template"]].append(False); print(f"EXC {t['task_id']}: {type(e).__name__}: {str(e)[:140]}")
for k, v in sorted(res.items()):
    print(f"{k:28s} {sum(v)}/{len(v)}")
