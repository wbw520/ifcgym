"""Drop SFT samples longer than a token budget. Length = Qwen3.5 tokens of all message contents, reasoning and tool calls, plus a margin
for the chat-template markup. Rewrites train.parquet / val.parquet in place and records the counts in data_card.json.
    python train/filter_sft_by_tokens.py <sft_data_dir> [--max 32768] [--margin 1024]   (verl container: transformers)"""
import argparse, json, os
import numpy as np, pandas as pd
from transformers import AutoTokenizer

ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--max", type=int, default=32768); ap.add_argument("--margin", type=int, default=1024)
ap.add_argument("--tokenizer", default="${IFCGYM_MODELS:-models}/Qwen3.5-9B"); a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.tokenizer)


def text(m):
    out = []
    for x in m:
        out += [str(x.get("content") or ""), str(x.get("reasoning_content") or "")]
        if x.get("tool_calls") is not None:
            out.append(str(x.get("tool_calls")))
    return "\n".join(out)


card_p = os.path.join(a.dir, "data_card.json"); card = json.load(open(card_p)); res = {}
for split in ("train", "val"):
    p = os.path.join(a.dir, f"{split}.parquet"); df = pd.read_parquet(p)
    L = np.array([len(tok(text(m), add_special_tokens=False).input_ids) for m in df.messages])
    keep = L <= a.max - a.margin
    res[split] = {"before": int(len(df)), "dropped": int((~keep).sum()), "after": int(keep.sum()), "max_tokens_kept": int(L[keep].max()), "p99": int(np.percentile(L[keep], 99))}
    df[keep].reset_index(drop=True).to_parquet(p)
card["token_filter"] = {"max_length": a.max, "margin": a.margin, **res}; card["train"] = res["train"]["after"]; card["val"] = res["val"]["after"]
json.dump(card, open(card_p, "w"), indent=1); print(json.dumps(card["token_filter"]))
