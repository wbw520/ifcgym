"""BIBIMBAP (EC3 2026) external test with the authors' evaluation code: our agent (protocol v1 or v2) answers the 100 prompts on the
benchmark's IFC files; the edited file and the structured answer are scored by the benchmark's own per-task tests
(data/tests/<test>.py: execute_test(ifc_file, edited_ifc_file, model_output) -> {metric: bool}); task score = satisfied /
evaluated; overall = mean over tasks (the paper's protocol). Code + data: data/bibimbap/code (github.com/nbharathik/bibimbap,
HF clemenskujat/bibimbap-data, CC BY 4.0).

    python -m ifcedit.run_bibimbap --model <student> --protocol v3 --out outputs/bibimbap/<student> [--limit N] [--score-only]
Structured-output tasks (numeric / conceptual retrieval): the prompt is extended with the output schema and the agent's final
message must contain a JSON object with those fields; it is parsed into model_output (plus raw_model_response).
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

BIB = os.environ.get("IFCGYM_BIBIMBAP", os.path.join(os.environ.get("IFCGYM_DATA", "data"), "bibimbap", "code"))
sys.path.insert(0, BIB)


def load_tasks():
    rows = list(csv.DictReader(open(os.path.join(BIB, "data/prompts/all_questions.csv"))))
    for i, r in enumerate(rows):
        r["id"] = i; r["category"] = r["test"].split("_test_")[0] if "_test_" in r["test"] else "basic"
    return rows


def schema_text(name: str) -> tuple[str, list[str]]:
    mod = importlib.import_module(f"data.structured_outputs.{name}")
    fields = [(k, v.description or "") for k, v in mod.ModelOutput.model_fields.items() if k != "raw_model_response"]
    return ("\n\nWhen you are done, end your final message with a JSON object with exactly these keys: "
            + ", ".join(f'"{k}" ({d})' for k, d in fields) + "."), [k for k, _ in fields]


def parse_json(text: str, keys: list[str]) -> dict:
    out = {k: None for k in keys}
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", text or ""))):
        try:
            d = json.loads(m.group(0))
            if any(k in d for k in keys):
                out.update({k: d.get(k) for k in keys}); break
        except json.JSONDecodeError:
            continue
    return out


def run_test(test_name: str, ifc_path: str, edited_path: str, model_output) -> dict:
    mod = importlib.import_module(f"data.tests.{test_name}")
    try:
        return {k: bool(v) for k, v in mod.execute_test(ifc_path, edited_path, model_output).items()}
    except Exception as e:  # noqa: BLE001
        return {"error": False, "_exception": f"{type(e).__name__}: {str(e)[:120]}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None); ap.add_argument("--base-url", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--protocol", default="v2", choices=["v1", "v2", "v2r", "v3"]); ap.add_argument("--budget", type=int, default=25)
    ap.add_argument("--no-think", action="store_true"); ap.add_argument("--think", action="store_true"); ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--limit", type=int, default=None); ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--score-only", action="store_true", help="score existing edited files / answers in --out (no model calls)")
    ap.add_argument("--effort", default=None, choices=["low", "medium", "high", "max"], help="reasoning_effort request field (Kimi-K3), as run_tasks --effort")
    a = ap.parse_args()
    if a.effort:   # shared EXTRA_BODY of run_tasks, which run_episode receives below
        from ifcedit.run_tasks import EXTRA_BODY as _EB
        _EB["reasoning_effort"] = a.effort
    a.out = a.out if os.path.isabs(a.out) else os.path.join(os.getcwd(), a.out)   # relative --out broke the sandbox worker path
    tasks = load_tasks()[: a.limit]
    os.makedirs(os.path.join(a.out, "edited"), exist_ok=True); os.makedirs(os.path.join(a.out, "trajectories"), exist_ok=True)

    def one(t):
        tid = t["id"]; ifc_path = os.path.join(BIB, "data/ifc", t["ifc-file"]); edited = os.path.join(a.out, "edited", f"{tid:03d}.ifc")
        traj_path = os.path.join(a.out, "trajectories", f"{tid:03d}.json")
        if not a.score_only:
            from ifcgym.sandbox import Sandbox
            from ifcedit.agent import run_episode
            from ifcedit.run_tasks import EXTRA_BODY
            from openai import OpenAI
            client = OpenAI(base_url=a.base_url or os.environ.get("RESEARCH_SERVICE_URL"), api_key="EMPTY")
            prompt = t["question"]; keys = []
            if t.get("structured-output"):
                extra, keys = schema_text(t["structured-output"]); prompt += extra
            wd = os.path.join(a.out, "tmp", f"task_{tid:03d}")
            sb = Sandbox(input_ifc=ifc_path, workdir=wd, python=a.python, timeout_s=120.0, helpers=(a.protocol != "v1"), isolate=(a.protocol == "v3"))
            t0 = time.time()
            ep = run_episode(client, a.model, str(tid), prompt, sb.execute, budget=a.budget, max_tokens=a.max_tokens, extra_body=EXTRA_BODY, protocol=a.protocol)
            sb.close()
            shutil.copyfile(sb.final_path(), edited)
            model_output = parse_json(ep.final_text, keys) if keys else {}
            model_output["raw_model_response"] = ep.final_text
            json.dump({"task": t, "messages": ep.messages, "final_text": ep.final_text, "model_output": model_output, "tool_calls": ep.tool_calls, "commits": ep.commits,
                       "outcome": ep.outcome, "seconds": time.time() - t0, "tokens_in": ep.tokens_in, "tokens_out": ep.tokens_out}, open(traj_path, "w"), ensure_ascii=False, indent=1, default=str)
            shutil.rmtree(wd, ignore_errors=True)
        else:
            d = json.load(open(traj_path)) if os.path.exists(traj_path) else {"model_output": {"raw_model_response": ""}}
            model_output = d.get("model_output") or {"raw_model_response": ""}
            if not os.path.exists(edited):
                shutil.copyfile(ifc_path, edited)
        metrics = run_test(t["test"], ifc_path, edited, model_output)
        ev = [v for k, v in metrics.items() if not k.startswith("_")]
        score = sum(ev) / len(ev) if ev else 0.0
        return {"id": tid, "category": t["category"], "crud": t["CRUD"], "test": t["test"], "metrics": metrics, "score": score}

    def safe(t):   # one failing episode must not abort the run; it is scored on the unedited input and reported
        try:
            return one(t)
        except Exception as e:  # noqa: BLE001
            tid = t["id"]; edited = os.path.join(a.out, "edited", f"{tid:03d}.ifc")
            shutil.copyfile(os.path.join(BIB, "data/ifc", t["ifc-file"]), edited)
            metrics = run_test(t["test"], os.path.join(BIB, "data/ifc", t["ifc-file"]), edited, {"raw_model_response": ""})
            ev = [v for k, v in metrics.items() if not k.startswith("_")]
            print(f"EPISODE-ERROR {tid}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            return {"id": tid, "category": t["category"], "crud": t["CRUD"], "test": t["test"], "metrics": metrics, "score": sum(ev) / len(ev) if ev else 0.0, "episode_error": repr(e)[:300]}

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        results = list(ex.map(safe, tasks))
    with open(os.path.join(a.out, "scores.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    import collections
    by = collections.defaultdict(list)
    for r in results:
        by[r["category"]].append(r["score"])
    overall = sum(r["score"] for r in results) / len(results)
    print(f"BIBIMBAP n={len(results)} overall={100*overall:.2f} | " + " ".join(f"{k}={100*sum(v)/len(v):.1f}({len(v)})" for k, v in sorted(by.items())))
    crud = collections.defaultdict(list)
    for r in results:
        crud[r["crud"]].append(r["score"])
    print("  by CRUD:", {k: round(100 * sum(v) / len(v), 1) for k, v in sorted(crud.items())})
    print(f"metric=bibimbap_overall {100*overall:.2f}")


if __name__ == "__main__":
    main()
