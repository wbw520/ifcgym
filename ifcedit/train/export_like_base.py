"""Write a tuned Qwen3.5 checkpoint in exactly the layout of the base checkpoint (same shards, same tensor names).

transformers 5.5.3 `save_pretrained` renames the vision tower (model.visual.* -> model.language_model.visual.*) and drops
the mtp.* tensors, and vLLM then refuses the checkpoint ("There is no module or parameter named 'visual' in Qwen3_5Model").
So the servable model is built from the BASE shards, replacing only what training changed:

    --adapter <lora_adapter dir>   LoRA run: W = W_base + (alpha / r) * B @ A for every adapted module (float32, cast back)
    --tuned   <hf model dir>       full-parameter run: every base tensor that exists in the tuned checkpoint (under its own
                                   name or under the renamed vision prefix) is taken from it; the rest (mtp.*) stays base

    python3 train/export_like_base.py --base <base dir> (--adapter DIR | --tuned DIR) --out <out dir>

Tokenizer, chat template, config and processor files are copied from --base. A report (EXPORT_INFO.json) lists how many
tensors were replaced and the largest change.
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

RENAMED = [("model.visual.", "model.language_model.visual.")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--adapter"); ap.add_argument("--tuned")
    a = ap.parse_args()
    assert bool(a.adapter) != bool(a.tuned), "give exactly one of --adapter / --tuned"
    base = Path(a.base); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    shards = sorted(glob.glob(str(base / "model*.safetensors")))
    delta = {}; tuned = None; tuned_keys = set()
    if a.adapter:
        ad = load_file(str(Path(a.adapter) / "adapter_model.safetensors"))
        cfg = json.load(open(Path(a.adapter) / "adapter_config.json")); scale = cfg["lora_alpha"] / cfg["r"]
        for ka in [k for k in ad if k.endswith(".lora_A.weight")]:
            kb = ka.replace(".lora_A.", ".lora_B.")
            name = ka[len("base_model.model."):-len(".lora_A.weight")] + ".weight"
            delta[name] = scale * (ad[kb].float() @ ad[ka].float())
        print(f"[export] LoRA: {len(delta)} adapted modules, r={cfg['r']} alpha={cfg['lora_alpha']}")
    else:
        tfiles = sorted(glob.glob(str(Path(a.tuned) / "model*.safetensors")))
        tuned = {}
        for f in tfiles:
            with safe_open(f, "pt") as h:
                for k in h.keys():
                    tuned[k] = f
        tuned_keys = set(tuned)
        print(f"[export] full-parameter: {len(tuned)} tuned tensors in {len(tfiles)} file(s)")
    handles = {}
    def tuned_tensor(k):
        f = tuned[k]
        if f not in handles:
            handles[f] = safe_open(f, "pt")
        return handles[f].get_tensor(k)
    replaced = 0; kept = 0; max_move = 0.0; used = set()
    for sf in shards:
        with safe_open(sf, "pt") as h:
            meta = h.metadata(); sd = {k: h.get_tensor(k) for k in h.keys()}
        for k in list(sd):
            w = sd[k]
            if a.adapter:
                if k in delta:
                    new = (w.float() + delta[k]).to(w.dtype); used.add(k)
                else:
                    kept += 1; continue
            else:
                src = k if k in tuned_keys else next((k.replace(o, n, 1) for o, n in RENAMED if k.startswith(o) and k.replace(o, n, 1) in tuned_keys), None)
                if src is None:
                    kept += 1; continue
                used.add(src); new = tuned_tensor(src).to(w.dtype)
                assert new.shape == w.shape, (k, new.shape, w.shape)
            max_move = max(max_move, (new.float() - w.float()).abs().max().item()); sd[k] = new; replaced += 1
        save_file(sd, str(out / Path(sf).name), metadata=meta or {"format": "pt"})
        print(f"[export] wrote {Path(sf).name} ({len(sd)} tensors)")
    missing = (set(delta) if a.adapter else tuned_keys) - used
    assert not missing, f"{len(missing)} tuned tensors have no place in the base layout, e.g. {sorted(missing)[:3]}"
    assert replaced > 0 and max_move > 0
    for f in base.iterdir():
        if f.name.startswith("model") and f.name.endswith(".safetensors"):
            continue
        if f.is_file() or f.is_symlink():
            shutil.copy(f.resolve(), out / f.name)
    info = {"base": str(base), "adapter": a.adapter, "tuned": a.tuned, "replaced": replaced, "kept_from_base": kept, "max_abs_change": max_move}
    json.dump(info, open(out / "EXPORT_INFO.json", "w"), indent=1)
    print(f"[export] {info}")


if __name__ == "__main__":
    main()
