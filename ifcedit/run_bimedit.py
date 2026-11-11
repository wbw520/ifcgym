"""Run a served model on BIM-Edit tasks with our CodeAct harness (H001/E002 baseline and later comparisons).

    python -m ifcedit.run_bimedit --model gemma4-31b --feedback none --budget 20 --out outputs/E002 [--workers 4] [--tasks ids...]

Endpoint discovery: --base-url, else RESEARCH_SERVICE_<NAME> (set by `research.py job submit --needs <name>`), else
the RESEARCH_SERVICE_<NAME> environment variable. Writes outputs/<E>/edited/<task_id>.ifc, trajectories/<task_id>.json and runs.jsonl.
Tasks: data/bimedit-runs/tasks.jsonl (BIM-Edit-Tasks revision 2dc6462); IFC files under data/bimedit/ (BIM-Edit revision 0ae510bd).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from ifcedit.agent import run_episode
from ifcgym.sandbox import Sandbox
from ifcgym.config import resolve_bimedit_ifc, BIMEDIT_TASKS as TASKS

ROOT = os.environ.get("IFCGYM_ROOT", os.getcwd())
EXTRA_BODY: dict = {}


def endpoint_for(name: str, base_url: str | None) -> str:
    if base_url:
        return base_url
    env = os.environ.get("RESEARCH_SERVICE_" + name.upper().replace("-", "_"))
    if env:
        return env
    raise SystemExit(f"no endpoint for model {name}: pass --base-url or set RESEARCH_SERVICE_{name.upper().replace('-', '_')}")


def load_tasks(ids: list[str] | None) -> list[dict]:
    tasks = [json.loads(l) for l in open(TASKS) if l.strip()]
    if ids:
        want = set(ids); tasks = [t for t in tasks if t["task_id"] in want]
    return tasks


def run_one(task: dict, client: OpenAI, model: str, out: str, budget: int, feedback: str, temperature: float, top_p: float,
            max_tokens: int, python: str, protocol: str = "v1") -> dict:
    tid = task["task_id"]
    inp = resolve_bimedit_ifc(task["input_ifc"])
    edited = os.path.join(out, "edited", tid + ".ifc"); traj = os.path.join(out, "trajectories", tid + ".json")
    if os.path.exists(traj) and os.path.exists(edited):
        return json.load(open(traj))["summary"]
    with tempfile.TemporaryDirectory(prefix=f"bimedit_{tid}_", dir=os.path.join(out, "tmp"), ignore_cleanup_errors=True) as wd:   # lustre rmtree race (ENOTEMPTY)
        sb = Sandbox(input_ifc=inp, workdir=wd, python=python, timeout_s=120.0, helpers=(protocol != "v1"), isolate=(protocol == "v3"))
        fb = None  # feedback modes other than `none` are wired in E003 (checkers on BIM-Edit tasks have no intent spec)
        def execute(code):    # protocol v2 (tools inspect / find / check_model + h.* helpers), same wiring as ifcedit.run_tasks
            r = sb.execute(code); r.work_path = sb.final_path(); return r
        ep = run_episode(client, model, tid, task["prompt"], execute, budget=budget, temperature=temperature, top_p=top_p,
                         max_tokens=max_tokens, feedback=fb, extra_body=EXTRA_BODY, protocol=protocol)
        sb.close()
        shutil.copyfile(sb.final_path(), edited)
        summary = {"task_id": tid, "model": model, "protocol": protocol, "operation": task["operation"], "category": task["category"],
                   "tool_calls": ep.tool_calls, "commits": ep.commits, "tokens_in": ep.tokens_in, "tokens_out": ep.tokens_out,
                   "seconds": round(ep.seconds, 2), "outcome": ep.outcome, "error": ep.error, "snapshots": len(sb.snapshots),
                   "input_ifc": task["input_ifc"], "ground_truth_ifc": task["ground_truth_ifc"], "edited": os.path.relpath(edited, out)}
        json.dump({"summary": summary, "messages": ep.messages, "final_text": ep.final_text}, open(traj, "w"), ensure_ascii=False, indent=1)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="served model name (serve/models.json key)")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--feedback", default="none", choices=["none", "validity", "all"])
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--protocol", default="v1", choices=["v1", "v2", "v3"], help="v1 = BIM-Edit single tool (published setting); v2 = our tools + helpers; v3 = v2 + sandbox jail + budget note")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--python", default=sys.executable, help="interpreter with ifcopenshell for the sandbox worker")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-think", action="store_true", help="disable Qwen3-style thinking via chat_template_kwargs (declared per experiment)"); ap.add_argument("--shard", default=None, help="i/n: run only tasks with index %% n == i (disjoint shards against different replicas)")
    ap.add_argument("--effort", default=None, choices=["low", "medium", "high", "max"], help="reasoning_effort request field (Kimi-K3)")
    a = ap.parse_args()
    if a.no_think:
        EXTRA_BODY["chat_template_kwargs"] = {"enable_thinking": False}
    if a.effort:   # 2026-11-08: Kimi-K3 reasoning_effort, as run_tasks --effort (the K3 benchmark rows use high, like its IfcEditBench row)
        EXTRA_BODY["reasoning_effort"] = a.effort
    out = a.out if os.path.isabs(a.out) else os.path.join(os.getcwd(), a.out)
    for d in ("edited", "trajectories", "tmp"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    url = endpoint_for(a.model, a.base_url)
    client = OpenAI(base_url=url, api_key=os.environ.get("OPENAI_API_KEY", "none"), timeout=600, max_retries=0)
    tasks = load_tasks(a.tasks)[: a.limit]
    if a.shard:
        i, n = map(int, a.shard.split("/")); tasks = [t for k, t in enumerate(tasks) if k % n == i]
    json.dump({"model": a.model, "base_url": url, "protocol": a.protocol, "feedback": a.feedback, "budget": a.budget, "temperature": a.temperature,
               "top_p": a.top_p, "max_tokens": a.max_tokens, "no_think": a.no_think, "n_tasks": len(tasks), "started": time.strftime("%Y-%m-%dT%H:%M:%S")},
              open(os.path.join(out, "config.json"), "w"), indent=1)
    runs = open(os.path.join(out, "runs.jsonl"), "a")
    done = 0; t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_one, t, client, a.model, out, a.budget, a.feedback, a.temperature, a.top_p, a.max_tokens, a.python, a.protocol): t for t in tasks}
        for f in as_completed(futs):
            t = futs[f]
            try:
                s = f.result()
            except Exception as e:  # noqa: BLE001
                s = {"task_id": t["task_id"], "model": a.model, "outcome": "other", "error": f"{type(e).__name__}: {e}"[:500]}
            runs.write(json.dumps(s, ensure_ascii=False) + "\n"); runs.flush()
            done += 1
            print(f"[{done}/{len(tasks)}] {s['task_id']} outcome={s.get('outcome')} calls={s.get('tool_calls')} commits={s.get('commits')} "
                  f"{s.get('seconds')}s  elapsed {time.time()-t0:.0f}s", flush=True)
    runs.close()


if __name__ == "__main__":
    main()
