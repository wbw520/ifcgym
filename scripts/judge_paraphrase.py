"""LLM equivalence judge for paraphrased prompts (review 2026-10-24: rule checks found left/right flips in 60-70 % of paraphrased
'farthest to your left' prompts and GlobalId role shifts that no rule detects). For each task with prompt_template != prompt, asks the
model whether an agent following the rewrite would do exactly the original task. Output: one JSON line per task
{task_id, equivalent, severity none|minor|major, problems}. Resumable (skips task ids already in --out).
    python scripts/judge_paraphrase.py <tasks.jsonl> --out judged.jsonl --base-url http://host:port/v1 [--workers 8] [--effort high]"""
import argparse, json, os, re, sys, concurrent.futures as cf

SYSTEM = """You audit rewrites of machine-generated BIM (IFC) task prompts. ORIGINAL is authoritative. Decide whether a careful agent that reads only REWRITE would perform exactly the task of ORIGINAL.
Check every point:
1. Each identifier (22-character GlobalId) refers to the same element and the same role (e.g. an id that names a storey or an anchor wall in ORIGINAL must not become the id of the element to edit).
2. The same target element(s) are designated by the same description (directions, left/right, above/below, nearest/farthest, north/south/east/west, highest/lowest, inwards/outwards, connected/adjacent).
3. The same operation, with the same numbers, units, signs, vectors and angles.
4. The same constraints (what must stay unchanged, what may change, what must follow) and the same answer format.
Dropping a purely explanatory definition is acceptable only if the rewrite stays unambiguous and still means the same.
severity: "none" = equivalent; "minor" = wording looser or a definition dropped but the task is still the same and unambiguous; "major" = the rewrite designates a different element, a different operation or value, contradicts itself, or is ambiguous enough that a careful agent could do something else.
Reply with one JSON object only: {"equivalent": true|false, "severity": "none"|"minor"|"major", "problems": ["short description", ...]}"""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("tasks"); ap.add_argument("--out", required=True); ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default=None); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--effort", default="high")
    ap.add_argument("--max-tokens", type=int, default=8192); ap.add_argument("--ids", default=None, help="only these task ids (file)")
    a = ap.parse_args()
    from openai import OpenAI
    cl = OpenAI(base_url=a.base_url, api_key="EMPTY", timeout=1800); model = a.model or cl.models.list().data[0].id
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["task_id"] for l in open(a.out) if l.strip()}
    only = set(open(a.ids).read().split()) if a.ids else None
    todo = [t for t in map(json.loads, open(a.tasks)) if t.get("prompt_template") and t["prompt_template"] != t["prompt"] and t["task_id"] not in done and (only is None or t["task_id"] in only)]
    print(f"{len(todo)} to judge ({len(done)} done) with {model}", flush=True)

    def one(t):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"ORIGINAL:\n{t['prompt_template']}\n\nREWRITE:\n{t['prompt']}"}]
        for attempt in range(3):
            try:
                r = cl.chat.completions.create(model=model, messages=msgs, temperature=0.0, max_tokens=a.max_tokens, extra_body={"reasoning_effort": a.effort})
                txt = r.choices[0].message.content or ""
                m = re.search(r"\{.*\}", txt, re.S)
                v = json.loads(m.group(0))
                return {"task_id": t["task_id"], "template": t["template"], "equivalent": bool(v.get("equivalent")), "severity": v.get("severity"), "problems": v.get("problems", [])}
            except Exception as e:  # noqa: BLE001
                err = repr(e)[:200]
        return {"task_id": t["task_id"], "template": t["template"], "equivalent": None, "severity": "error", "problems": [err]}

    with open(a.out, "a") as fo, cf.ThreadPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(one, todo)):
            fo.write(json.dumps(r) + "\n"); fo.flush()
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(todo)}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
