#!/bin/bash
# Multi-turn GRPO of the IfcEdit student. Runs inside the verl container on one GPU node (8 cards).
# LoRA on the language model, hybrid FSDP2 actor + one vLLM engine per card, custom agent loop, rollout log-probs as old log-probs).
# Start policy: the merged SFT student (MODEL). Episode, tools, reward: ifcedit/train/rl/agent_loop.py.
#   MODEL=<merged SFT student> RUN=<name> DATA=<dir with train/val.parquet> bash ifcedit/train/rl/grpo.sh [hydra overrides...]
# 27B student: TP=2 PPO_TOK=16384 SP=4 GPU_MEM=0.6 (defaults = the 9B values).
set -euo pipefail
R=${IFCGYM_ROOT:?set IFCGYM_ROOT to the repository root}
C=$R
O=${OUT:-$C/outputs/grpo}
D=${DATA:-$O/data}
RUN=${RUN:-grpo}
MODEL=${MODEL:?set MODEL to the merged SFT student}
NGPU=$(echo "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c .)
unset ROCR_VISIBLE_DEVICES ZE_AFFINITY_MASK HIP_VISIBLE_DEVICES
export HF_HUB_OFFLINE=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false HOME=/tmp/ac-home-$USER
export RAY_local_fs_capacity_threshold=0.9995
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-600}
# pyhook: CPU weight init on every FSDP rank; code: train.rl.agent_loop + ifcedit; pydeps: Qwen3.5 linear-attention fast path (read-only reuse)
export PYTHONPATH="$C/train/pyhook:$C:${FLA_DEPS:-}:${PYTHONPATH:-}"
export IFCGYM_RL_WORK=${IFCGYM_RL_WORK:-$O/runs/$RUN/work}
export IFCGYM_REWARD=${IFCGYM_REWARD:-strict}
mkdir -p "$HOME" "$O/runs/$RUN" "$IFCGYM_RL_WORK"
cd "$O/runs/$RUN"
echo "[grpo] run $RUN on $(hostname -s) cards $CUDA_VISIBLE_DEVICES model $MODEL data $D reward $IFCGYM_REWARD $(date -Is)"
exec python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
  algorithm.rollout_correction.bypass_mode=True algorithm.rollout_correction.loss_type=ppo_clip \
  data.train_files="$D/train.parquet" data.val_files="$D/val.parquet" \
  data.train_batch_size=${BATCH:-32} data.max_prompt_length=6144 data.max_response_length=${RESP:-32768} \
  data.filter_overlong_prompts=True data.truncation=error data.shuffle=True \
  actor_rollout_ref.model.path="$MODEL" actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.lora_rank=64 actor_rollout_ref.model.lora_alpha=128 \
  actor_rollout_ref.model.target_modules=all-linear "actor_rollout_ref.model.exclude_modules='.*visual.*'" \
  actor_rollout_ref.model.lora.merge=True \
  actor_rollout_ref.actor.optim.lr=${LR:-1e-5} actor_rollout_ref.actor.optim.weight_decay=0.0 \
  actor_rollout_ref.actor.ppo_mini_batch_size=${MINI:-8} actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_TOK:-40960} \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=${SP:-2} \
  actor_rollout_ref.actor.use_kl_loss=${USE_KL:-False} actor_rollout_ref.actor.kl_loss_coef=${KL_COEF:-0.001} actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.strategy=fsdp2 actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking=True actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True \
  "actor_rollout_ref.actor.checkpoint.save_contents=[model]" +actor_rollout_ref.actor.checkpoint.save_lora_only=True \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n=${GROUP:-8} actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=${TP:-1} actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM:-0.45} \
  actor_rollout_ref.rollout.max_model_len=$((6144 + ${RESP:-32768})) actor_rollout_ref.rollout.max_num_seqs=64 \
  actor_rollout_ref.rollout.enable_prefix_caching=True actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_TOK:-40960} \
  actor_rollout_ref.rollout.multi_turn.enable=True actor_rollout_ref.rollout.multi_turn.format=qwen3_coder \
  actor_rollout_ref.rollout.agent.default_agent_loop=ifcedit_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$C/train/rl/agent_loops.yaml" \
  actor_rollout_ref.rollout.agent.num_workers=${AGENT_WORKERS:-8} \
  trainer.n_gpus_per_node="$NGPU" trainer.nnodes=${NNODES:-1} trainer.total_epochs=${EPOCHS:-1} trainer.critic_warmup=0 \
  trainer.save_freq=${SAVE_FREQ:-10} trainer.test_freq=-1 trainer.val_before_train=False "trainer.logger=[console]" \
  trainer.project_name=ifcedit trainer.experiment_name="grpo_e012_$RUN" trainer.resume_mode=${RESUME:-auto} \
  trainer.default_local_dir="$O/runs/$RUN/ckpt" trainer.rollout_data_dir="$O/runs/$RUN/rollout_dumps" "$@"
