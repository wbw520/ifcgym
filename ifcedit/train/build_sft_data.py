"""Build the SFT dataset from teacher episodes (run_tasks trajectories): keep strict-passing episodes, drop tool calls that
errored (step filter), and write verl-style parquet with a `messages` column (system / user / assistant(tool_calls) / tool ...).

    python train/build_sft_data.py --runs outputs/E004/teacher outputs/E004/teacher_b --out outputs/E005/data --val-projects fzk_house wbdg_office
Leakage checks: every episode's project must be a DEVELOPMENT project (configs/splits/dev_ids.txt); val = held-out dev projects.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random

import pandas as pd
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ifcedit.agent import prune_messages, _DSML_BLOCK, _TEXT_CALL  # noqa: E402  (tools v2: the student is trained on the same pruned view it is served with)


def _clean_text(content: str) -> str:
    """Older trajectories (before 2026-10-20) kept the raw DSML / Gemma text tool-call markup in the assistant content next to the
    parsed tool_calls; the student must not learn to emit it."""
    return _DSML_BLOCK.sub("", _TEXT_CALL.sub("", content or "")).rstrip()

ROOT = os.environ.get("IFCGYM_ROOT", os.getcwd())
AMBIGUOUS_IDS = {"hitos-T2-move-0092", "hitos-T2-move-0112"}    # same id in tasks_train and tasks_train_t2 with different targets
DEV = {l.strip().split(":")[1] for l in open(f"{ROOT}/configs/splits/dev_ids.txt") if l.strip()}


def _with_params(tc: dict) -> dict:
    """Qwen3.5's template iterates `tool_call.arguments|items` (a mapping). Serving passes a parsed dict; parquet cannot hold dicts
    with varying keys without null fields, so the training rows carry `params` = [{name, value}] (strings, in order) and the
    patched template (outputs/E006/Qwen3.5-9B-ifcedit/chat_template.jinja) renders those - same text as the mapping path."""
    fn = tc["function"]
    try:
        args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
    except json.JSONDecodeError:
        args = {"code": fn["arguments"]}
    if not isinstance(args, dict):
        args = {"code": str(args)}
    params = [{"name": k, "value": (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v))} for k, v in args.items()]
    return {"id": tc.get("id", ""), "type": "function", "function": {"name": fn["name"], "arguments": json.dumps(args, ensure_ascii=False), "params": params}}


_SPECS: dict = {}


def _spec_target(task_id: str, specs_glob: str) -> str | None:
    """Target GlobalId of a task (its cluster key) from the task spec files, loaded once."""
    if not _SPECS:
        for p in glob.glob(specs_glob):
            for l in open(p):
                if l.strip():
                    t = json.loads(l); _SPECS[t["task_id"]] = t["target"][0]
    return _SPECS.get(task_id)


def clean_episode(d: dict, keep_failed_steps: bool, keep_reasoning: bool = True) -> list[dict] | None:
    msgs = d["messages"]
    if not msgs or msgs[0]["role"] != "system":
        return None
    if d["summary"].get("protocol", "v1") != "v1":
        msgs = prune_messages(msgs)
        if d["summary"].get("protocol") == "v2r":      # context distillation: the student never sees the teacher's recipe
            from ifcedit.agent import SYSTEM_PROMPT_V2
            msgs = [{"role": "system", "content": SYSTEM_PROMPT_V2}] + msgs[1:]
    out = [msgs[0], msgs[1]]
    i = 2
    while i < len(msgs):
        m = msgs[i]
        if m["role"] == "assistant" and m.get("tool_calls"):
            tools = []
            j = i + 1
            while j < len(msgs) and msgs[j]["role"] == "tool":
                tools.append(msgs[j]); j += 1
            failed = any("Traceback" in t["content"] or t["content"].startswith("Error:") for t in tools)
            if failed and not keep_failed_steps:
                i = j; continue        # step filter: drop the failed call and its tool output
            tool_msgs = [{"role": "tool", "content": t["content"].split("\n\n[checker report]")[0]} for t in tools]   # feedback text is not shown to the student
            rc = {"reasoning_content": m["reasoning_content"]} if keep_reasoning and m.get("reasoning_content") else {}
            out.append({"role": "assistant", "content": _clean_text(m.get("content")), **rc, "tool_calls": [_with_params(tc) for tc in m["tool_calls"]]}); out += tool_msgs
            i = j
        elif m["role"] == "assistant":
            rc = {"reasoning_content": m["reasoning_content"]} if keep_reasoning and m.get("reasoning_content") else {}
            out.append({"role": "assistant", "content": _clean_text(m.get("content")), **rc}); i += 1
        else:
            i += 1
    n_calls = sum(1 for m in out if m["role"] == "assistant" and m.get("tool_calls"))
    return out if n_calls >= 1 else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--val-projects", nargs="*", default=["fzk_house"]); ap.add_argument("--keep-failed-steps", action="store_true")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--protocol", default=None, help="keep only episodes of this protocol (v1 | v2 | v2r)")
    ap.add_argument("--specs", default=f"{ROOT}/outputs/tasks_train_v2/*.jsonl", help="task spec files (cluster = target GlobalId)")
    ap.add_argument("--val-fraction", type=float, default=None, help="instead of --val-projects: hold out this fraction of episodes by target-element cluster (RNG --seed)")
    ap.add_argument("--drop-reasoning", action="store_true", help="strip teacher reasoning_content (no-think student)")
    ap.add_argument("--synthetic-only", action="store_true", help="E008: accept only synthetic buildings (synth_*); any real building -> refuse")
    ap.add_argument("--exclude-ids", nargs="*", default=[], help="files of task ids to drop (verifier mismatches, rejected paraphrases, ...)")
    ap.add_argument("--val-building-fraction", type=float, default=None, help="hold out this fraction of BUILDINGS (projects) as val (RNG --seed)")
    ap.add_argument("--prefer-later", action="store_true", help="a task id present in several --runs uses the episode of the LAST run listed "
                    "(recollections replace bug-affected episodes; a failed recollection drops the task instead of falling back)")
    a = ap.parse_args()
    excluded = set()
    for p_ in a.exclude_ids:
        excluded |= set(open(p_).read().split())
    rows = []; stats = {"episodes": 0, "strict": 0, "kept": 0, "non_dev": 0}
    chosen = {}
    for rd in a.runs:
        for p in glob.glob(os.path.join(rd, "trajectories", "*.json")):
            k = os.path.basename(p)[:-5] if a.prefer_later else p
            if k in chosen:
                stats["replaced"] = stats.get("replaced", 0) + 1
            chosen[k] = p
    for p in sorted(chosen.values()):
        if True:
            d = json.load(open(p)); s = d["summary"]; stats["episodes"] += 1
            if s["task_id"] in AMBIGUOUS_IDS or (a.protocol and s.get("protocol", "v1") != a.protocol):
                continue
            if s["task_id"] in excluded:
                stats["excluded"] = stats.get("excluded", 0) + 1; continue
            if a.synthetic_only:
                if not s["project"].startswith("synth_"):
                    stats["non_dev"] += 1; continue
            elif s["project"] not in DEV:
                stats["non_dev"] += 1; continue
            if str(s.get("strict")) != "True":   # older summaries hold the string "False" (numpy bool via default=str): it is truthy
                continue
            stats["strict"] += 1
            msgs = clean_episode(d, a.keep_failed_steps, keep_reasoning=not a.drop_reasoning)
            if msgs is None:
                continue
            stats["kept"] += 1
            rows.append({"task_id": s["task_id"], "project": s["project"], "tier": s["tier"], "template": s["template"], "model": s["model"], "messages": msgs,
                         "cluster": (_spec_target(s["task_id"], a.specs) or s["task_id"])})
    assert stats["non_dev"] == 0, f"{stats['non_dev']} episodes from non-development projects - refusing to build training data"
    random.Random(a.seed).shuffle(rows)
    if a.val_building_fraction:   # E008: whole buildings held out
        projs = sorted({r["project"] for r in rows}); random.Random(a.seed).shuffle(projs)
        held = set(projs[: max(1, int(round(a.val_building_fraction * len(projs))))])
        val = [r for r in rows if r["project"] in held]; train = [r for r in rows if r["project"] not in held]; a.val_projects = [f"building-holdout {a.val_building_fraction} ({len(held)} buildings)"]
    elif a.val_fraction:   # E007: hold out whole target-element clusters (a wall and all its tasks) so val never sees a train target
        clusters = sorted({r["cluster"] for r in rows}); random.Random(a.seed).shuffle(clusters)
        held = set(clusters[: max(1, int(round(a.val_fraction * len(clusters))))])
        val = [r for r in rows if r["cluster"] in held]; train = [r for r in rows if r["cluster"] not in held]; a.val_projects = [f"cluster-holdout {a.val_fraction}"]
    else:
        val = [r for r in rows if r["project"] in set(a.val_projects)]; train = [r for r in rows if r["project"] not in set(a.val_projects)]
    os.makedirs(a.out, exist_ok=True)
    cols = ["task_id", "project", "tier", "template", "model", "messages"]
    pd.DataFrame(train, columns=cols).to_parquet(os.path.join(a.out, "train.parquet")); pd.DataFrame(val, columns=cols).to_parquet(os.path.join(a.out, "val.parquet"))
    assert len(val) > 0, "empty validation set: choose --val-projects that have episodes"
    card = {**stats, "train": len(train), "val": len(val), "val_projects": a.val_projects, "by_tier": {t: sum(1 for r in train if r["tier"] == t) for t in sorted({r["tier"] for r in train})},
            "by_project": {p: sum(1 for r in train if r["project"] == p) for p in sorted({r["project"] for r in train})}, "runs": a.runs, "step_filter": not a.keep_failed_steps,
            "exclude_id_files": a.exclude_ids, "n_excluded_ids": len(excluded), "synthetic_only": a.synthetic_only, "prefer_later": a.prefer_later}
    json.dump(card, open(os.path.join(a.out, "data_card.json"), "w"), indent=1)
    print(json.dumps(card, indent=1))


if __name__ == "__main__":
    main()
