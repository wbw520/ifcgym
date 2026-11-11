"""Paraphrase fidelity audit: applies ifcgym.tasks.fidelity.problems (identifier set, numbers, operative left/right,
polarity words) to every paraphrased task; prints counts and writes flagged task ids. Semantic checks: scripts/judge_paraphrase.py.
    python scripts/audit_paraphrase.py <tasks.jsonl> [--out ids.txt]"""
import json, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ifcgym.tasks.fidelity import problems

path = sys.argv[1]; out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
cnt = collections.Counter(); flagged = []; n = 0
for l in open(path):
    if not l.strip():
        continue
    t = json.loads(l); tp = t.get("prompt_template")
    if not tp or tp == t["prompt"]:
        cnt["not paraphrased"] += 1; continue
    n += 1; pr = problems(tp, t["prompt"])
    for x in pr:
        cnt[x.split(" [")[0].split(" '")[0]] += 1
    if pr:
        flagged.append(t["task_id"])
print(f"{path}: {n} paraphrased tasks; flagged {len(flagged)}")
for k, v in cnt.most_common():
    print(f"  {k:45s} {v}")
if out:
    open(out, "w").write("".join(x + "\n" for x in flagged))
