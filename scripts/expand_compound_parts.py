"""Expand compound tasks (M-compound-k) into their single-edit source tasks so the reference verifier (scripts/verify_refs_v4.py, which
resolves one description per task) checks every part's T/S/P/V reference. The source tasks are looked up by meta.part_ids in the
generated files of the same building; each numbered line of the compound prompt (template wording) must equal the source prompt minus
the closing "Do not change anything else." and the part intents must equal the source intents (else reported and exit 1). Non-compound
tasks are passed through unchanged.
    python scripts/expand_compound_parts.py <tasks.jsonl> "<generated glob>" --out parts.jsonl"""
import argparse, collections, glob, json, re, sys

NC = "Do not change anything else."


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("tasks"); ap.add_argument("src_glob"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tasks = [json.loads(l) for l in open(a.tasks) if l.strip()]
    need = {pid for t in tasks if t["template"].startswith("M-") for pid in t["meta"]["part_ids"]}
    src = {}
    for p in sorted(glob.glob(a.src_glob)):
        for l in open(p):
            m = re.match(r'\{"task_id": "([^"]+)"', l)
            if m and m.group(1) in need:
                src[m.group(1)] = json.loads(l)
    bad = collections.Counter(); out = []
    for t in tasks:
        if not t["template"].startswith("M-"):
            out.append(t); continue
        body = t.get("prompt_template", t["prompt"])
        lines = re.findall(r"^\d+\. (.*)$", body, flags=re.M)
        for i, pid in enumerate(t["meta"]["part_ids"]):
            s = src.get(pid)
            if s is None:
                bad["source missing"] += 1; print(f"MISSING {t['task_id']} part {pid}"); continue
            if i >= len(lines) or lines[i].strip() != s.get("prompt_template", s["prompt"]).replace(NC, "").strip():
                bad["prompt line differs"] += 1; print(f"PROMPT {t['task_id']} part {i + 1}")
            if t["intent"]["parts"][i] != s["intent"]:
                bad["intent differs"] += 1; print(f"INTENT {t['task_id']} part {i + 1}")
            out.append({**s, "task_id": f"{t['task_id']}#p{i + 1}"})
    with open(a.out, "w") as fo:
        for t in out:
            fo.write(json.dumps(t) + "\n")
    print(f"{len(tasks)} tasks -> {len(out)} rows ({len(src)} source parts); problems: {dict(bad) or 0}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
