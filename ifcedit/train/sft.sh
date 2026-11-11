#!/bin/bash
# SFT of the IfcEdit student (H001/E005) on teacher episodes. Runs INSIDE the verl container on a held node (see run_in_container.sh).
# Base: Qwen3-8B (or BASE=.../Qwen3-14B). LoRA r=64 on all linear layers; multi-turn messages with tool calls rendered by the
# model's own chat template; loss on assistant turns only.
set -euo pipefail
R=${IFCGYM_ROOT:?set IFCGYM_ROOT to the repository root}
BASE=${BASE:-${IFCGYM_MODELS:-models}/Qwen3.5-9B}   # student since 2026-10-20 (user); Qwen3-8B = the E004 smoke base
# Qwen3.5: LoRA on the language model only; its linear-attention layers need flash-linear-attention + causal-conv1d (installed for this
# (flash-linear-attention + causal-conv1d, FLA_DEPS; FASTPATH=0 switches them off)
# need train/sft_entry.py (CPU weight init on every rank).
DEPS=${FLA_DEPS:-}   # optional: directory with flash-linear-attention + causal-conv1d for the Qwen3.5 fast path
if [ "${FASTPATH:-1}" = 1 ] && [ -d "$DEPS/fla" ]; then export PYTHONPATH="$DEPS:${PYTHONPATH:-}"; echo "[sft] fast path libraries on PYTHONPATH"; fi
O=${OUT:-$R/outputs/E006}
RUN=${RUN:-r1}
DATA=${DATA:-$O/data}
NGPU=$(echo "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c .)
if [ "$NGPU" -gt 1 ]; then ENTRY="$R/train/sft_entry.py"; else ENTRY="-m verl.trainer.sft_trainer"; fi
unset ROCR_VISIBLE_DEVICES ZE_AFFINITY_MASK HIP_VISIBLE_DEVICES
export HF_HUB_OFFLINE=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false HOME=/tmp/ac-home-$USER
mkdir -p "$HOME" "$O/runs/$RUN"; cd "$O/runs/$RUN"
exec torchrun --nnodes=1 --nproc_per_node="$NGPU" --master_port="${MASTER_PORT:-$((29600 + RANDOM % 2000))}" $ENTRY \
  data.train_files="$DATA/train.parquet" data.val_files="$DATA/val.parquet" data.messages_key=messages data.enable_thinking_default=false data.ignore_input_ids_mismatch=True \
  data.train_batch_size=16 data.micro_batch_size_per_gpu=1 data.max_token_len_per_gpu=${MAXTOK:-${MAXLEN:-16384}} data.use_dynamic_bsz=True \
  data.pad_mode=no_padding data.max_length=${MAXLEN:-16384} data.truncation=right data.num_workers=4 \
  ${SP:+engine.ulysses_sequence_parallel_size=$SP} \
  model.path="$BASE" model.trust_remote_code=False model.use_remove_padding=True \
  model.lora_rank=64 model.lora_alpha=128 model.target_modules=all-linear "model.exclude_modules='.*visual.*'" \
  engine=fsdp engine.strategy=fsdp2 engine.model_dtype=bf16 \
  optim.lr=1e-4 optim.lr_scheduler_type=cosine optim.lr_warmup_steps_ratio=0.05 optim.weight_decay=0.0 optim.clip_grad=1.0 \
  trainer.total_epochs=${EPOCHS:-3} trainer.seed=${SEED:-1} "trainer.logger=[console]" \
  trainer.save_freq=${SAVE_FREQ:-after_each_epoch} trainer.test_freq=after_each_epoch trainer.resume_mode=${RESUME:-disable} "checkpoint.save_contents=${CKPT_CONTENTS:-[model]}" \
  trainer.max_ckpt_to_keep=${KEEP_CKPT:-1} \
  trainer.default_local_dir="$O/runs/$RUN/ckpt" trainer.project_name=ifcedit trainer.experiment_name="sft_$RUN" \
  trainer.n_gpus_per_node="$NGPU" "$@"
