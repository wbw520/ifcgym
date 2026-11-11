"""Site hook for the GRPO run.
Loaded by every Python process whose PYTHONPATH contains this directory (the verl driver and all its Ray workers).

verl initialises the model on non-zero FSDP ranks with accelerate.init_empty_weights whenever config.tie_word_embeddings is
false; with transformers 5.5.3 that fails for Qwen3.5 (Parameter.__new__() got an unexpected keyword argument
'_is_hf_initialized'). The hook below replaces verl.utils.fsdp_utils.get_init_weight_context_manager at import time, so that
every rank loads the real weights on CPU (FSDP shards them afterwards). Nothing else is changed.
"""
import importlib.abc
import importlib.util
import sys


class _PatchFsdpUtils(importlib.abc.MetaPathFinder):
    target = "verl.utils.fsdp_utils"

    def find_spec(self, name, path, target=None):
        if name != self.target:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return spec
        orig = spec.loader.exec_module

        def exec_module(module):
            orig(module)
            import torch

            def cpu_init(use_meta_tensor=True, mesh=None):  # noqa: ARG001
                return lambda: torch.device("cpu")

            module.get_init_weight_context_manager = cpu_init

        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _PatchFsdpUtils())
