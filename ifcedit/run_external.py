"""Run a prior IFC-agent framework (ifcedit.frameworks.<name>) on IfcEdit generated tasks and score the IFC file it writes with the gated checkers.

    python -m ifcedit.run_external --framework mcp4ifc --model deepseek-v4-flash --base-url http://host:30004/v1 --tasks <jsonl> --out outputs/<run>/eval/heldout_MCP4IFC \
        [--workers 16] [--budget 20] [--max-tokens 16384] [--temperature 0] [--limit N] [--shard i/n]

Same output layout and summary fields as ifcedit.run_tasks (trajectories/<task>.json with "summary", edited/<task>.ifc, results.jsonl, config.json), so
ifcedit.result_tables and scripts/compare_arms.py read these runs unchanged. feedback is always "none": the frameworks have no checker feedback;
protocol is recorded as "ext:<framework>". The framework sees only the input IFC copy and the task prompt, never the task specification.
"""
from __future__ import annotations

import argparse
import glob
import importlib
import json
import multiprocessing
import os
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from openai import OpenAI

from ifcgym.checkers import check
from ifcedit.run_bimedit import endpoint_for

CHECK_POOL = None


def run_check(before, after, spec, final_text=None):
    if CHECK_POOL is None:
        return check(before, after, spec, final_text)
    return CHECK_POOL.submit(check, before, after, spec, final_text).result()


def run_one(task: dict, fw, name: str, client: OpenAI, model: str, out: str, budget: int, max_tokens: int, temperature: float) -> dict:
    tid = task["task_id"]; traj = os.path.join(out, "trajectories", tid + ".json"); edited = os.path.join(out, "edited", tid + ".ifc")
    if os.path.exists(traj):
        return json.load(open(traj))["summary"]
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix=f"task_{tid}_", dir=os.path.join(out, "tmp"), ignore_cleanup_errors=True) as wd:
        src = os.path.join(wd, "model.ifc"); shutil.copyfile(task["ifc"], src)
        try:
            r = fw.run(prompt=task["prompt"], ifc_path=src, workdir=wd, client=client, model=model, budget=budget, max_tokens=max_tokens, temperature=temperature)
        except Exception as e:  # noqa: BLE001
            r = {"output_ifc": src, "final_text": None, "tool_calls": 0, "tokens_in": 0, "tokens_out": 0, "outcome": "other", "error": f"{type(e).__name__}: {e}"[:500], "messages": []}
        final = r.get("output_ifc") or src
        if not os.path.exists(final):
            final = src
        shutil.copyfile(final, edited)
        rep = run_check(task["ifc"], edited, task, r.get("final_text"))
        summary = {"task_id": tid, "project": task["project"], "template": task["template"], "tier": task["tier"], "model": model, "feedback": "none", "protocol": f"ext:{name}",
                   "strict": bool(rep.strict), "validity": bool(rep.validity), "intent": bool(rep.intent), "dependency": bool(rep.dependency), "preservation": bool(rep.preservation),
                   "under_edit": bool(rep.under_edit), "side_effects": rep.side_effects, "new_errors": rep.new_errors,
                   "tool_calls": int(r.get("tool_calls", 0)), "commits": int(r.get("commits", 0)), "tokens_in": int(r.get("tokens_in", 0)), "tokens_out": int(r.get("tokens_out", 0)),
                   "seconds": round(time.time() - t0, 1), "outcome": r.get("outcome", "executed"), "error": r.get("error"), "details": rep.details}
        json.dump({"summary": summary, "messages": r.get("messages", []), "final_text": r.get("final_text")}, open(traj, "w"), ensure_ascii=False, indent=1, default=str)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--framework", required=True); ap.add_argument("--model", required=True); ap.add_argument("--base-url", nargs="*", default=None)
    ap.add_argument("--tasks", nargs="+", required=True); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--budget", type=int, default=20); ap.add_argument("--max-tokens", type=int, default=16384); ap.add_argument("--temperature", type=float, default=0.0); ap.add_argument("--timeout", type=float, default=900.0, help="per-request client timeout in seconds (long tool outputs prefill slowly)")
    ap.add_argument("--resume", action="store_true", help="skip task_ids already present in the out dir's results*.jsonl (a run continued after a stop; nothing is re-scored)")
    ap.add_argument("--limit", type=int, default=None); ap.add_argument("--shard", default=None)
    a = ap.parse_args()
    fw = importlib.import_module(f"ifcedit.frameworks.{a.framework}")
    out = a.out if os.path.isabs(a.out) else os.path.join(os.getcwd(), a.out)
    for d in ("edited", "trajectories", "tmp"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    paths = []
    for p in a.tasks:
        for q in p.split():
            paths += sorted(glob.glob(q)) or [q]
    tasks = [json.loads(l) for p in paths for l in open(p) if l.strip()][: a.limit]
    if a.shard:
        i, n = map(int, a.shard.split("/")); tasks = [t for k, t in enumerate(tasks) if k % n == i]
    if a.resume:
        done_ids = {json.loads(l)["task_id"] for f in glob.glob(os.path.join(out, "results*.jsonl")) for l in open(f) if l.strip()}
        tasks = [t for t in tasks if t["task_id"] not in done_ids]
        print(f"[resume] {len(done_ids)} tasks already scored, {len(tasks)} to run", flush=True)
    urls = [endpoint_for(a.model, u) for u in (a.base_url or [None])]
    clients = [OpenAI(base_url=u, api_key=os.environ.get("OPENAI_API_KEY", "none"), timeout=a.timeout, max_retries=0) for u in urls]
    json.dump({"framework": a.framework, "model": a.model, "base_url": urls if len(urls) > 1 else urls[0], "max_tokens": a.max_tokens, "timeout": a.timeout, "resume": a.resume, "feedback": "none", "budget": a.budget,
               "temperature": a.temperature, "protocol": f"ext:{a.framework}", "tasks": a.tasks, "started": time.strftime("%Y-%m-%dT%H:%M:%S")},
              open(os.path.join(out, "config.json"), "w"), indent=1)
    global CHECK_POOL
    CHECK_POOL = ProcessPoolExecutor(max_workers=max(2, a.workers), mp_context=multiprocessing.get_context("forkserver"))
    res = open(os.path.join(out, f"results{('_' + a.shard.replace('/', 'of')) if a.shard else ''}.jsonl"), "a"); done = 0; t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_one, t, fw, a.framework, clients[k % len(clients)], a.model, out, a.budget, a.max_tokens, a.temperature): t for k, t in enumerate(tasks)}
        for f in as_completed(futs):
            t = futs[f]
            try:
                s = f.result()
            except Exception as e:  # noqa: BLE001
                s = {"task_id": t["task_id"], "project": t["project"], "template": t["template"], "tier": t["tier"], "model": a.model, "feedback": "none", "protocol": f"ext:{a.framework}",
                     "strict": False, "outcome": "other", "error": f"{type(e).__name__}: {e}"[:500], "side_effects": 0, "tool_calls": 0, "validity": False, "intent": False,
                     "dependency": False, "preservation": False, "under_edit": False}
            res.write(json.dumps(s, ensure_ascii=False, default=str) + "\n"); res.flush(); done += 1
            print(f"[{done}/{len(tasks)}] {s['task_id']} strict={s.get('strict')} I={s.get('intent')} D={s.get('dependency')} P={s.get('preservation')} V={s.get('validity')} "
                  f"side={s.get('side_effects')} calls={s.get('tool_calls')} {s.get('outcome')} {s.get('seconds')}s elapsed {time.time()-t0:.0f}s", flush=True)
    res.close()


if __name__ == "__main__":
    main()
