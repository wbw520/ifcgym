"""Static invariants of a task set (layer 1 of the task-set gate, 2026-10-22): no model geometry, only the task records and the
entity ids of the input files. Prints violations per rule and exits 1 if any.
    python scripts/lint_tasks.py 'outputs/tasks_synth2k_v7/*.jsonl' [--limit-files N]
Rules: every GlobalId in the prompt exists in the input file; target / dependents / allowed ids exist; a DIRECT (-D or classic)
prompt names its target, a -T/-S/-P prompt must NOT contain the target id (leak); allowed ⊇ target ∪ dependents for edits; edit
intents point at the target; numbers in the intent (distance, vector, thickness, height, polygon) appear in the prompt; create tasks
declare allowed_new_classes; answer tasks carry a value; the input file exists."""
import glob, json, re, sys, collections, os
GUID = re.compile(r"(?<![0-9A-Za-z_$])[0-9A-Za-z_$]{22}(?![0-9A-Za-z_$])")   # \b fails next to $ and _
files = sorted(glob.glob(sys.argv[1])); lim = int(sys.argv[sys.argv.index("--limit-files") + 1]) if "--limit-files" in sys.argv else None
files = files[:lim]
viol = collections.Counter(); examples = {}; n = 0; ids_cache = {}; seen_ids = set(); sha_cache = {}

def ids_of(path):
    if path not in ids_cache:
        s = set()
        with open(path, "rb") as fh:
            for line in fh:
                m = re.search(rb"^#\d+\s*=\s*IFC[A-Z0-9]+\(\s*'([0-9A-Za-z_$]{22})'", line)
                if m:
                    s.add(m.group(1).decode())
        ids_cache[path] = s
    return ids_cache[path]

def flag(rule, t, extra=""):
    viol[rule] += 1; examples.setdefault(rule, f"{t['task_id']}: {extra} | {t['prompt'][:140]}")

def nums_in(x):
    out = []
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        out.append(float(x))
    elif isinstance(x, (list, tuple)):
        for y in x: out += nums_in(y)
    elif isinstance(x, dict):
        for k, y in x.items():
            if k in ("tol", "rel_tol", "value", "top"): continue
            out += nums_in(y)
    return out

for p in files:
    for line in open(p):
        if not line.strip(): continue
        t = json.loads(line); n += 1; pr = t["prompt"]; it = t["intent"]; tpl = t["template"]; style = t.get("meta", {}).get("style")
        if not os.path.exists(t["ifc"]):
            flag("ifc-missing", t, t["ifc"]); continue
        ids = ids_of(t["ifc"]); pg = {g for g in GUID.findall(pr) if not g.startswith(("Qto_", "Pset_", "Ifc"))}   # 22-char quantity-set / IFC class names (E010)
        for g in pg - ids:
            flag("prompt-id-not-in-model", t, g)
        for g in t.get("target", []) + t.get("allowed", []) + [d["guid"] for d in t.get("dependents", [])]:
            if g not in ids: flag("task-id-not-in-model", t, g)
        tgt = t.get("target", [])
        direct = style == "D" or (style is None and not tpl.startswith(("S-", "R-find")))
        if it["type"] == "all":    # compound (E009): every non-creation part's element(s) must be allowed; parts are linted as their own tasks
            allowed = set(t.get("allowed", []))
            for p_ in it.get("parts", []):
                if p_["type"] in ("create_wall", "create_slab", "create_column", "create_space", "create_fill", "answer"):
                    continue
                gs = set(p_.get("guids") or []) | ({p_["guid"]} if p_.get("guid") else set())
                if not gs <= allowed: flag("allowed-missing-target-or-dependent", t, f"part {p_['type']}")
            if len(it.get("parts", [])) != len(t.get("meta", {}).get("parts", [])): flag("compound-parts-meta-mismatch", t)
        elif it["type"] not in ("answer", "create_wall", "create_slab", "create_column", "create_space", "create_fill") or tpl.startswith("R-") and it.get("kind") == "number" and style:
            pass
        if tgt and it["type"] not in ("answer", "delete_all", "resize_height_all", "all") and not it["type"].startswith("create"):
            if direct and tgt[0] not in pg: flag("direct-prompt-lacks-target", t, tgt[0])
            if not direct and style in ("T", "S", "P", "V") and tgt[0] in pg: flag("LEAK-indirect-prompt-names-target", t, tgt[0])
            if it.get("guid") != tgt[0]: flag("intent-guid-not-target", t, f"{it.get('guid')} vs {tgt[0]}")
            allowed = set(t.get("allowed", []))
            if not (set(tgt) | {d["guid"] for d in t.get("dependents", [])}) <= allowed: flag("allowed-missing-target-or-dependent", t)
        if it["type"] == "answer":
            if it.get("value") in (None, "", []): flag("answer-without-value", t)
            if it.get("kind") == "guid" and it["value"] in pg: flag("LEAK-answer-guid-in-prompt", t, it["value"])
            if it.get("kind") == "guids" and set(it["value"]) & pg: flag("LEAK-answer-guid-in-prompt", t)
            if it.get("kind") in ("guid", "guids") and not all(g in ids for g in ([it["value"]] if it["kind"] == "guid" else it["value"])): flag("answer-id-not-in-model", t)
        if it["type"].startswith("create") and not t.get("allowed_new_classes"): flag("create-without-new-classes", t)
        if it["type"].startswith("create_fill") and it["host"] not in ids: flag("task-id-not-in-model", t, it["host"])
        if style in ("T", "S", "P", "V") and it["type"] == "create_fill" and it["host"] in pg: flag("LEAK-indirect-prompt-names-target", t, it["host"])
        # numbers of the intent must be visible in the prompt (distance/vector/thickness/height/polygon/xy), except derived end points
        if it["type"] in ("translate", "set_thickness", "resize_height", "create_wall", "create_column", "create_slab", "create_space", "create_fill"):
            keys = {"translate": ["vector"], "set_thickness": ["thickness"], "resize_height": ["height"], "create_wall": ["p0", "p1", "thickness", "height"], "create_column": ["size", "height"],
                    "create_slab": ["thickness"], "create_space": ["height"], "create_fill": ["width", "height"]}[it["type"]]
            if style in ("S",) and it["type"] in ("create_wall", "create_column"): keys = [k for k in keys if k not in ("p0", "p1", "xy")]
            if any(x in tpl for x in ("-height-rel-", "U-window-taller")): keys = [k for k in keys if k != "height"]   # E010: relative change - the prompt gives the delta
            pn = {x.lstrip("-") for x in re.findall(r"-?\d+(?:\.\d+)?", GUID.sub("", pr))}
            for k in keys:
                for v in nums_in(it.get(k)):
                    if v == 0.0 and k == "vector": continue
                    v = abs(v); s1 = f"{v:g}"; s2 = f"{v:.1f}"; s3 = f"{v:.2f}"; s4 = f"{v:.3f}"
                    if not any(x in pn for x in (s1, s2, s3, s4, str(v))): flag("intent-number-absent-from-prompt", t, f"{k}={v}")
        if it["type"] == "resize_height" and it.get("top") is not None and f"{it['top']}" not in pr and f"{it['top']:.3f}" not in pr: flag("intent-number-absent-from-prompt", t, "top")
        meta = t.get("meta", {})
        if t["task_id"] in seen_ids: flag("duplicate-task-id", t)
        seen_ids.add(t["task_id"])
        if tpl[-2:] in ("-D", "-T", "-S", "-P", "-V") and meta.get("style") != tpl[-1]: flag("style-suffix-mismatch", t, f"{meta.get('style')} vs {tpl[-1]}")
        for g in meta.get("anchors", []):
            if style in ("T", "S", "P", "V") and g not in pg: flag("anchor-not-in-prompt", t, g)
            if tgt and g == tgt[0] and style in ("T", "S", "P", "V") and it["type"] != "answer": flag("LEAK-anchor-is-target", t, g)
        for k in ("storey", "host"):
            if it.get(k) and it[k] not in ids: flag("task-id-not-in-model", t, f"{k}={it[k]}")
        if it["type"] in ("delete_all", "resize_height_all"):
            if not it.get("guids") or not set(it["guids"]) <= set(tgt) or (set(it["guids"]) != set(tgt) and not tpl.startswith(("D-delete-columns-", "U-columns-to-slab", "D-delete-windows-facing"))): flag("multi-intent-guids-not-target", t)   # E010: derived sets keep their reference element in target
            if not set(it["guids"]) <= set(t.get("allowed", [])): flag("allowed-missing-target-or-dependent", t)
            if str(len(it["guids"])) not in pr and not tpl.startswith(("D-delete-columns-", "U-columns-to-slab", "D-delete-windows-facing")): flag("intent-number-absent-from-prompt", t, "count")   # E010 / E011: the set is derived, not given
        if "ifc_sha1" in meta:
            if t["ifc"] not in sha_cache:
                import hashlib; sha_cache[t["ifc"]] = hashlib.sha1(open(t["ifc"], "rb").read()).hexdigest()[:16]
            if sha_cache[t["ifc"]] != meta["ifc_sha1"]: flag("input-file-changed-since-generation", t, t["ifc"])
        if it["type"] == "translate":
            m = re.search(r"by ([\d.]+) m", pr); norm = (sum(v * v for v in it["vector"])) ** 0.5
            if m and abs(float(m.group(1)) - norm) > 0.005: flag("translate-distance-mismatch", t, f"{m.group(1)} vs {norm:.3f}")
print(f"{n} tasks in {len(files)} files; violations: {sum(viol.values())}")
for k, v in sorted(viol.items(), key=lambda kv: -kv[1]):
    print(f"  {k:40s} {v:6d}   e.g. {examples[k][:220]}")
sys.exit(1 if viol else 0)
