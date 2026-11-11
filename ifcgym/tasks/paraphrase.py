"""Rewrite templated task prompts into natural language with an LLM (ALL prompts, training and dev-val, use the
rewritten phrasing). The program keeps the task definition and the truth; the model only changes the wording.

    python -m ifcgym.tasks.paraphrase --model deepseek-v4-flash --base-url http://host:30004/v1 --tasks 'outputs/tasks_synth2k_v3/*.jsonl' --out outputs/tasks_synth2k_v3p --workers 16
Guards (programmatic): every GlobalId, number and unit string of the original prompt must appear verbatim in the rewrite, the rewrite
must not add a different GlobalId, must keep the JSON-answer instruction for retrieval tasks, and must stay under 4x the original
length; otherwise retry once with the violation named, then fall back to the template prompt (flagged `paraphrased: false`).
The original prompt is kept as `prompt_template`.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor

GUID = re.compile(r"(?<![0-9A-Za-z_$])[0-9A-Za-z_$]{22}(?![0-9A-Za-z_$])")   # \b fails next to $ and _
NUM = re.compile(r"-?\d+(?:\.\d+)?")
STYLES = [
    "a facility manager writing a short work request to a colleague",
    "an architect giving a quick instruction in a design review",
    "a BIM coordinator's issue ticket, one or two sentences",
    "a terse site engineer, no pleasantries",
    "a slightly verbose client explaining what they want and why",
]
SYSTEM = ("You rewrite BIM editing requests into natural language. Keep every GlobalId EXACTLY as given (they are identifiers like "
          "'2UsXAbj6n0fwEWqdkVuHvf'), keep every number with its unit, keep the meaning and every requirement (what must follow, what must "
          "not change, the requested answer format). Change only the wording, order and tone. Do not add requirements, do not drop any. "
          "Reply with the rewritten request only.")


def check(orig: str, new: str) -> str | None:
    g0, g1 = set(GUID.findall(orig)), set(GUID.findall(new))
    if g0 - g1:
        return f"missing identifier(s) {sorted(g0 - g1)}"
    if g1 - g0:
        return f"added identifier(s) {sorted(g1 - g0)}"
    n0 = [x for x in NUM.findall(orig) if x not in g0]
    for x in n0:
        if x not in new:
            return f"missing number {x}"
    if '{"value"' in orig and '"value"' not in new or '"global_id"' in orig and '"global_id"' not in new:
        return "missing the JSON answer instruction"
    if len(new) > 4 * len(orig) + 80 or len(new) < 0.3 * len(orig):
        return "length out of range"
    if "GlobalId" in orig and "GlobalId" not in new and "global id" not in new.lower() and "id " not in new.lower():
        return "identifiers must still be introduced as ids"
    from ifcgym.tasks.fidelity import problems      # left/right and other polarity words
    pr = problems(orig, new)
    if pr:
        return "; ".join(pr) + " (keep every direction / side / superlative word of the original)"
    return None


def rewrite(client, model: str, prompt: str, rng: random.Random, temperature: float, max_tokens: int = 600, extra_body: dict | None = None,
            attempts: int = 2) -> tuple[str, bool]:
    style = rng.choice(STYLES)
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Style: {style}.\n\nRequest:\n{prompt}"}]
    for attempt in range(attempts):
        r = client.chat.completions.create(model=model, messages=msgs, temperature=temperature, max_tokens=max_tokens, extra_body=extra_body or {})
        new = (r.choices[0].message.content or "").strip().strip('"')
        err = check(prompt, new)
        if err is None and new:
            return new, True
        msgs += [{"role": "assistant", "content": new}, {"role": "user", "content": f"Your rewrite is invalid: {err}. Rewrite again, keeping everything from the original request."}]
    return prompt, False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--base-url", required=True, help="one URL or a comma-separated list of replica URLs"); ap.add_argument("--tasks", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--temperature", type=float, default=0.9); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--effort", default=None, help="reasoning_effort for reasoning models (Kimi K3: low/high/max)")
    ap.add_argument("--max-tokens", type=int, default=600, help="raise for reasoning models (the reasoning counts against it)")
    ap.add_argument("--attempts", type=int, default=2); ap.add_argument("--ids", default=None, help="file of task ids: rewrite only these, copy the others unchanged")
    a = ap.parse_args(); extra = {"reasoning_effort": a.effort} if a.effort else {}; only = set(open(a.ids).read().split()) if a.ids else None
    from openai import OpenAI
    clients = [OpenAI(base_url=u.strip(), api_key="EMPTY", timeout=600) for u in a.base_url.split(",") if u.strip()]   # several replicas: round-robin
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(a.tasks))
    stats = {"tasks": 0, "paraphrased": 0}
    for p in files:
        outp = os.path.join(a.out, os.path.basename(p))
        if os.path.exists(outp):
            continue
        tasks = [json.loads(l) for l in open(p) if l.strip()][: a.limit]
        def one(t):
            if only is not None and t["task_id"] not in only:
                return t
            src = t.get("prompt_template", t["prompt"])     # always rewrite the GENERATED wording, never an earlier paraphrase
            rng = random.Random(f"{a.seed}:{t['task_id']}")
            new, ok = rewrite(clients[hash(t["task_id"]) % len(clients)], a.model, src, rng, a.temperature, a.max_tokens, extra, a.attempts)
            row = {k: v for k, v in t.items() if k != "prompt_pre_label"}   # a stale pre-label copy of an EARLIER paraphrase would make
            # scripts/balance_id_labels.py relabel that old text instead of this rewrite (on the E-class re-paraphrase)
            return {**row, "prompt": new, "prompt_template": src, "paraphrased": ok, "paraphrase_model": os.path.basename(a.model.rstrip("/")) if ok else None}
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            rows = list(ex.map(one, tasks))
        with open(outp, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats["tasks"] += len(rows); stats["paraphrased"] += sum(bool(r.get("paraphrased")) for r in rows)
        print(os.path.basename(p), len(rows), "paraphrased", sum(bool(r.get("paraphrased")) for r in rows), flush=True)
    print(stats)


if __name__ == "__main__":
    main()
