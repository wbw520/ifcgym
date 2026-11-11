"""Run a served model on IfcEdit generated tasks (tasks/generate.py records) with the gated checkers; feedback modes none / validity / all.

    python -m ifcedit.run_tasks --model deepseek-v4-flash --tasks outputs/tasks/wbdg_office.jsonl --feedback all --out outputs/E003/ds_all [--workers 4] [--limit N]
    python -m ifcedit.run_tasks --report outputs/E003            # aggregate every run dir below it

Per task: sandbox on the project IFC, agent episode, final check -> results.jsonl rows with strict / per-check flags / side effects / cost.
The task prompt is the generated prompt; the system prompt is BIM-Edit's (agent.py). `all` feedback appends the checker report after
every commit (only pass/fail per check and the side-effect count; never the expected values beyond what the prompt already states).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import multiprocessing
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from openai import OpenAI

from ifcedit.agent import run_episode
from ifcgym.checkers import check, schema_errors
from ifcgym.sandbox import Sandbox
from ifcedit.run_bimedit import endpoint_for

ROOT = os.environ.get("IFCGYM_ROOT", os.getcwd())
EXTRA_BODY: dict = {}
CHECK_POOL = None   # process pool: IfcOpenShell is not thread-safe, every check runs in its own worker process (2026-10-19)


def run_check(before: str, after: str, spec: dict, final_text: str | None = None):
    if CHECK_POOL is None:
        return check(before, after, spec, final_text)
    return CHECK_POOL.submit(check, before, after, spec, final_text).result()


def make_feedback(mode: str, spec: dict, input_ifc: str):
    if mode == "none":
        return None
    def fb(r):
        try:
            rep = run_check(input_ifc, r.work_path, spec) if hasattr(r, "work_path") else None
        except Exception as e:  # noqa: BLE001
            return f"checker error: {e}"
        if rep is None:
            return ""
        if mode == "validity":
            return f"validity: {'ok' if rep.validity else 'FAIL'} ({rep.details.get('validity', '')})"
        return rep.text()
    return fb


def run_one(task: dict, client: OpenAI, model: str, out: str, budget: int, feedback: str, temperature: float, top_p: float, max_tokens: int, python: str, protocol: str = "v1") -> dict:
    tid = task["task_id"]; traj = os.path.join(out, "trajectories", tid + ".json"); edited = os.path.join(out, "edited", tid + ".ifc")
    if os.path.exists(traj):
        return json.load(open(traj))["summary"]
    with tempfile.TemporaryDirectory(prefix=f"task_{tid}_", dir=os.path.join(out, "tmp"), ignore_cleanup_errors=True) as wd:   # lustre rmtree race (ENOTEMPTY) after the trajectory is written
        sb = Sandbox(input_ifc=task["ifc"], workdir=wd, python=python, timeout_s=120.0, helpers=(protocol != "v1"), isolate=(protocol == "v3"))
        def execute(code):
            r = sb.execute(code); r.work_path = sb.final_path(); return r
        ep = run_episode(client, model, tid, task["prompt"], execute, budget=budget, temperature=temperature, top_p=top_p, max_tokens=max_tokens, protocol=protocol,
                         feedback=make_feedback(feedback, task, task["ifc"]), extra_body=EXTRA_BODY)
        sb.close()
        shutil.copyfile(sb.final_path(), edited)
        rep = run_check(task["ifc"], edited, task, ep.final_text)
        summary = {"task_id": tid, "project": task["project"], "template": task["template"], "tier": task["tier"], "model": model, "feedback": feedback, "protocol": protocol,
                   # bool(): some checks return numpy.bool_, which json default=str wrote as the STRING "False" - truthy for every reader (bug found 2026-10-26)
                   "strict": bool(rep.strict), "validity": bool(rep.validity), "intent": bool(rep.intent), "dependency": bool(rep.dependency), "preservation": bool(rep.preservation),
                   "under_edit": bool(rep.under_edit), "side_effects": rep.side_effects, "new_errors": rep.new_errors,
                   "tool_calls": ep.tool_calls, "commits": ep.commits, "tokens_in": ep.tokens_in, "tokens_out": ep.tokens_out, "seconds": round(ep.seconds, 1),
                   "outcome": ep.outcome, "error": ep.error, "details": rep.details}
        json.dump({"summary": summary, "messages": ep.messages, "final_text": ep.final_text}, open(traj, "w"), ensure_ascii=False, indent=1, default=str)
    return summary


def report(root: str) -> None:
    import collections
    for rd in sorted(glob.glob(os.path.join(root, "*"))):
        tj = sorted(glob.glob(os.path.join(rd, "trajectories", "*.json")))
        if not tj:
            continue
        rows = []
        for q in tj:   # trajectories are authoritative (one file per task); results*.jsonl may interleave when shards share a dir
            try:
                rows.append(json.load(open(q))["summary"])
            except Exception:  # noqa: BLE001
                pass
        if not rows:
            continue
        by = collections.defaultdict(list)
        for r in rows:
            by[r["tier"]].append(r)
        print(f"== {os.path.basename(rd)}: model={rows[0]['model']} feedback={rows[0]['feedback']} n={len(rows)}")
        for tier in sorted(by):
            rs = by[tier]; n = len(rs)
            def rate(k):
                return sum(1 for r in rs if str(r.get(k)) == 'True') / n
            print(f"   tier {tier} n={n:3d} strict={rate('strict'):.2f} intent={rate('intent'):.2f} dep={rate('dependency'):.2f} pres={rate('preservation'):.2f} valid={rate('validity'):.2f} "
                  f"under_edit={rate('under_edit'):.2f} side_eff/task={sum(r['side_effects'] for r in rs)/n:.1f} calls={sum(r['tool_calls'] for r in rs)/n:.1f} "
                  f"budget={sum(1 for r in rs if r['outcome']=='budget')/n:.2f} crash/api={sum(1 for r in rs if r['outcome'] in ('crash','api','other'))/n:.2f}")
        n = len(rows)
        print(f"   ALL    n={n:3d} strict={sum(1 for r in rows if str(r['strict']) == 'True')/n:.3f}")
        print(f"metric=strict_success {100*sum(1 for r in rows if str(r['strict']) == 'True')/n:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--base-url", nargs="*", default=None, help="one or more replicas of the same model: tasks are spread round-robin (2026-10-28)"); ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--feedback", default="none", choices=["none", "validity", "all"]); ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--out"); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=1, help="k samples per task (RL pool pre-filter): task ids get the suffix @<i>")
    ap.add_argument("--temperature", type=float, default=0.0); ap.add_argument("--top-p", type=float, default=1.0); ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--python", default=sys.executable); ap.add_argument("--report", default=None); ap.add_argument("--no-think", action="store_true", help="disable Qwen3-style thinking via chat_template_kwargs (declared per experiment)"); ap.add_argument("--shard", default=None, help="i/n: disjoint task shard"); ap.add_argument("--tiers", nargs="*", type=int, default=None); ap.add_argument("--protocol", default="v1", choices=["v1", "v2", "v2r", "v3"], help="v1 = BIM-Edit single tool; v2 = inspect_element + code tool with helpers (E007); v2r = v2 + cascade recipe (teacher data collection only); v3 = v2 + sandbox jail + budget note (2026-10-28)"); ap.add_argument("--think", action="store_true", help="DeepSeek-V4 thinking mode (chat_template_kwargs thinking=true); reasoning stored as reasoning_content"); ap.add_argument("--effort", default=None, choices=["low", "medium", "high", "max"], help="reasoning_effort request field (Kimi-K3: low/high/max)")
    a = ap.parse_args()
    if a.no_think:
        EXTRA_BODY["chat_template_kwargs"] = {"enable_thinking": False}
    if a.think:
        EXTRA_BODY["chat_template_kwargs"] = {"thinking": True}
    if a.effort:
        EXTRA_BODY["reasoning_effort"] = a.effort
    if a.report:
        report(a.report); return
    out = a.out if os.path.isabs(a.out) else os.path.join(os.getcwd(), a.out)
    for d in ("edited", "trajectories", "tmp"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    tasks = []
    paths = []
    for p in a.tasks:                       # each entry may be a glob or a whitespace-separated list (zsh does not word-split variables)
        for q in p.split():
            paths += sorted(glob.glob(q)) or [q]
    for p in paths:
        tasks += [json.loads(l) for l in open(p) if l.strip()]
    if a.tiers is not None:
        tasks = [t for t in tasks if t["tier"] in set(a.tiers)]
    tasks = tasks[: a.limit]
    if a.shard:
        i, n = map(int, a.shard.split("/")); tasks = [t for k, t in enumerate(tasks) if k % n == i]
    if a.repeats > 1:
        tasks = [dict(t, task_id=f"{t['task_id']}@{i}", base_task_id=t["task_id"]) for i in range(a.repeats) for t in tasks]
    urls = [endpoint_for(a.model, u) for u in (a.base_url or [None])]
    clients = [OpenAI(base_url=u, api_key=os.environ.get("OPENAI_API_KEY", "none"), timeout=900, max_retries=0) for u in urls]
    json.dump({"model": a.model, "base_url": urls if len(urls) > 1 else urls[0], "repeats": a.repeats, "max_tokens": a.max_tokens, "top_p": a.top_p, "feedback": a.feedback, "budget": a.budget, "temperature": a.temperature, "no_think": a.no_think, "think": a.think, "effort": a.effort, "protocol": a.protocol, "n_tasks": len(tasks),
               "tasks": a.tasks, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, open(os.path.join(out, "config.json"), "w"), indent=1)
    global CHECK_POOL
    # forkserver, not fork: the pool forks its workers lazily while the episode threads (and the HTTP client) are running, and a forked
    # child can inherit a lock held by another thread - the first checks of a run then hang forever (E010 / E011 evals, 2026-10-27)
    CHECK_POOL = ProcessPoolExecutor(max_workers=max(2, a.workers), mp_context=multiprocessing.get_context("forkserver"))
    res = open(os.path.join(out, f"results{('_' + a.shard.replace('/', 'of')) if a.shard else ''}.jsonl"), "a"); done = 0; t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_one, t, clients[k % len(clients)], a.model, out, a.budget, a.feedback, a.temperature, a.top_p, a.max_tokens, a.python, a.protocol): t for k, t in enumerate(tasks)}
        for f in as_completed(futs):
            t = futs[f]
            try:
                s = f.result()
            except Exception as e:  # noqa: BLE001
                s = {"task_id": t["task_id"], "project": t["project"], "template": t["template"], "tier": t["tier"], "model": a.model, "feedback": a.feedback,
                     "strict": False, "outcome": "other", "error": f"{type(e).__name__}: {e}"[:500], "side_effects": 0, "tool_calls": 0, "validity": False, "intent": False,
                     "dependency": False, "preservation": False, "under_edit": False}
            res.write(json.dumps(s, ensure_ascii=False, default=str) + "\n"); res.flush(); done += 1
            print(f"[{done}/{len(tasks)}] {s['task_id']} strict={s.get('strict')} I={s.get('intent')} D={s.get('dependency')} P={s.get('preservation')} V={s.get('validity')} "
                  f"side={s.get('side_effects')} calls={s.get('tool_calls')} {s.get('outcome')} {s.get('seconds')}s elapsed {time.time()-t0:.0f}s", flush=True)
    res.close()


if __name__ == "__main__":
    main()
