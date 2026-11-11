"""Balance how GlobalIds are introduced in paraphrased prompts (2026-10-23, user): the LLM rewrite keeps the template's word
"GlobalId" almost always, while BIM-Edit writes "with ID X" / "(X)" and BIBIMBAP "with id X" / 'GlobalId = "X"'. Deterministic
per-task relabelling (seeded by task_id), ids themselves untouched; the original rewrite is kept in prompt_pre_label.
Target mix: ID 35 %, id 25 %, GlobalId 25 %, GUID / parenthetical 15 %.
    python scripts/balance_id_labels.py <tasks.jsonl ...>   (in place)"""
import json, random, re, sys

STYLES = [("ID", 35), ("id", 25), ("GlobalId", 25), ("GUID", 8), ("paren", 7)]
PAT = re.compile(r"\b(GlobalIds?|GUIDs?|IDs?|ids?)(\s*[:=]?\s*)(?=['\"]?[0-9A-Za-z_$]{22}\b)")

def relabel(prompt: str, style: str) -> str:
    def sub(m):
        word, sep = m.group(1), m.group(2)
        plural = word.endswith("s")
        if style == "paren":
            return ""          # handled below: "wall with GlobalId X" -> "wall (X)"
        base = {"ID": "ID", "id": "id", "GlobalId": "GlobalId", "GUID": "GUID"}[style]
        return (base + ("s" if plural else "")) + (sep if sep.strip() else " ")
    if style == "paren":
        # "the wall with (the) GlobalId X" -> "the wall (X)" (BIM-Edit style); every other label becomes "ID"
        out = re.sub(r"\bwith (?:the )?(?:GlobalId|GUID|ID|id) ([0-9A-Za-z_$]{22})\b", r"(\1)", prompt)
        return relabel(out, "ID")
    return PAT.sub(sub, prompt)

for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    for t in rows:
        rng = random.Random("label:" + t["task_id"]); pick = rng.choices([s for s, _ in STYLES], weights=[w for _, w in STYLES])[0]
        if "prompt_pre_label" not in t:
            t["prompt_pre_label"] = t["prompt"]
        t["prompt"] = relabel(t["prompt_pre_label"], pick); t.setdefault("meta", {})["id_label"] = pick
        ids0 = set(re.findall(r"(?<![0-9A-Za-z_$])[0-9A-Za-z_$]{22}(?![0-9A-Za-z_$])", t["prompt_pre_label"])); ids1 = set(re.findall(r"(?<![0-9A-Za-z_$])[0-9A-Za-z_$]{22}(?![0-9A-Za-z_$])", t["prompt"]))
        assert ids0 == ids1, (t["task_id"], ids0 ^ ids1)
    with open(path, "w") as fh:
        for t in rows:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    import collections
    print(path.split("/")[-1], len(rows), dict(collections.Counter(t["meta"]["id_label"] for t in rows)))
