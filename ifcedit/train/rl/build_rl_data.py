"""RL dataset for the E012 GRPO run (verl RLHFDataset parquet): one row per task, the evaluation prompt of protocol v3
(system = SYSTEM_PROMPT_V2, user = task prompt), agent_name = ifcedit_agent, the task record in extra_info.task.

    python train/rl/build_rl_data.py --tasks <tasks.jsonl> --out <dir> [--val <tasks.jsonl>]   (verl container: pandas)"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from ifcedit.agent import SYSTEM_PROMPT_V2  # noqa: E402


def rows(path: str) -> list[dict]:
    out = []
    for i, l in enumerate(open(path)):
        if not l.strip():
            continue
        t = json.loads(l)
        out.append({"data_source": "ifcedit", "agent_name": "ifcedit_agent",
                    "prompt": [{"role": "system", "content": SYSTEM_PROMPT_V2}, {"role": "user", "content": t["prompt"]}],
                    "reward_model": {"style": "rule", "ground_truth": ""},
                    "extra_info": {"index": i, "task_id": t["task_id"], "template": t["template"], "task": json.dumps(t)}})
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tasks", required=True); ap.add_argument("--val", default=None); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    tr = rows(a.tasks); pd.DataFrame(tr).to_parquet(os.path.join(a.out, "train.parquet"))
    va = rows(a.val) if a.val else tr[:16]; pd.DataFrame(va).to_parquet(os.path.join(a.out, "val.parquet"))
    print(json.dumps({"train": len(tr), "val": len(va), "out": a.out}))


if __name__ == "__main__":
    main()
