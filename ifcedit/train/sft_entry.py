"""verl SFT trainer entry point with one patch, for multi-GPU runs of Qwen3.5 in the verl 0.9.0 container.

verl initialises the model on non-zero ranks with accelerate.init_empty_weights (meta tensors) whenever
config.tie_word_embeddings is false; with transformers 5.5.3 that path fails for Qwen3.5
(TypeError: Parameter.__new__() got an unexpected keyword argument '_is_hf_initialized'). Here every rank loads the real
weights on CPU instead (19 GB of host memory per rank; FSDP then shards them as usual).

    torchrun --nproc_per_node=N train/sft_entry.py <hydra overrides as for verl.trainer.sft_trainer>
"""
import torch

import verl.utils.fsdp_utils as _fu


def _cpu_init(use_meta_tensor=True, mesh=None):  # noqa: ARG001
    return lambda: torch.device("cpu")


_fu.get_init_weight_context_manager = _cpu_init
import verl.workers.engine.fsdp.transformer_impl as _ti  # noqa: E402

if hasattr(_ti, "get_init_weight_context_manager"):
    _ti.get_init_weight_context_manager = _cpu_init

# Patch 2 (2026-10-25): verl 0.9.0's SFT trainer never uses trainer.seed - the DistributedSampler shuffles with its default seed 0 and
# no RNG is seeded, so "seed 1" and "seed 2" trained on the same order from the same LoRA init (E008 first launch: identical losses to
# 3 decimals; E007's two seeds were affected the same way). Seed every RNG and the sampler from trainer.seed (same on all ranks).
import random  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

SEED = next((int(x.split("=", 1)[1]) for x in sys.argv[1:] if x.startswith("trainer.seed=")), 1)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

import verl.trainer.sft_trainer as _st  # noqa: E402

_BaseSampler = _st.DistributedSampler


class _SeededSampler(_BaseSampler):
    def __init__(self, *args, seed=None, **kwargs):
        super().__init__(*args, seed=SEED if seed is None else seed, **kwargs)


_st.DistributedSampler = _SeededSampler
from verl.trainer.sft_trainer import main  # noqa: E402

if __name__ == "__main__":
    print(f"[sft_entry] seeding RNGs and the DistributedSampler with {SEED}", flush=True)
    main()
