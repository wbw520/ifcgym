"""LoRA adapter of a GRPO checkpoint (verl FSDP2, save_lora_only) as a PEFT adapter dir, for train/export_like_base.py.

The actor checkpoint holds one file per FSDP rank with the lora_A / lora_B tensors as DTensors sharded on dim 0; the shards are
concatenated in rank order. r / alpha come from lora_train_meta.json. Then build the servable model on top of the start policy:

    python3 train/rl/grpo_adapter.py --actor <ckpt>/global_step_N/actor --out <dir>/lora_adapter        (verl container: torch)
    python3 train/export_like_base.py --base <merged SFT student> --adapter <dir>/lora_adapter --out <dir>/merged
"""
import argparse
import glob
import json
import os

import torch
from safetensors.torch import save_file
from torch.distributed.tensor import DTensor, Shard


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--actor", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.actor, "lora_train_meta.json")))
    ws = json.load(open(os.path.join(a.actor, "fsdp_config.json")))["world_size"]
    files = [os.path.join(a.actor, f"model_world_size_{ws}_rank_{r}.pt") for r in range(ws)]
    assert len(glob.glob(os.path.join(a.actor, "model_world_size_*_rank_*.pt"))) == ws
    parts = {}
    for f in files:
        for k, v in torch.load(f, map_location="cpu", weights_only=False).items():
            if isinstance(v, DTensor):
                assert v.placements == (Shard(0),), (k, v.placements)
                parts.setdefault(k, []).append((v.to_local(), tuple(v.shape)))
            else:
                parts.setdefault(k, []).append((v, None))
    sd, targets = {}, set()
    for k, ps in parts.items():
        assert "lora_" in k, f"non-LoRA tensor in a save_lora_only checkpoint: {k}"
        full = ps[0][1]
        t = torch.cat([p for p, _ in ps], 0) if full else ps[0][0]
        if full:
            assert tuple(t.shape) == full, (k, t.shape, full)
        name = k.replace(".default.weight", ".weight")
        sd[name] = t.contiguous(); targets.add(name.split(".")[-3])
    os.makedirs(a.out, exist_ok=True)
    save_file(sd, os.path.join(a.out, "adapter_model.safetensors"))
    cfg = {"peft_type": "LORA", "task_type": meta.get("task_type", "CAUSAL_LM"), "r": int(meta["r"]), "lora_alpha": int(meta["lora_alpha"]),
           "target_modules": sorted(targets), "lora_dropout": 0.0, "bias": "none", "use_rslora": False}
    json.dump(cfg, open(os.path.join(a.out, "adapter_config.json"), "w"), indent=1)
    print(json.dumps({"tensors": len(sd), "modules": len(sd) // 2, "targets": sorted(targets), "r": cfg["r"], "alpha": cfg["lora_alpha"]}))


if __name__ == "__main__":
    main()
