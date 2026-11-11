#!/bin/bash
# verl FSDP checkpoint -> servable HF model . Runs INSIDE the verl container; no GPU needed
# (CPU is fine).
#   SFT run:  RUN=r1 [STEP=N] [OUT=...] bash ifcedit/train/merge_ckpt.sh      -> outputs/sft/runs/$RUN/merged
#   any run:  CKPT=<dir with model_world_size_*.pt> BASE=<model the run started from> OUT=<dir> bash train/merge_ckpt.sh
set -euo pipefail
C=${IFCGYM_ROOT}
O=${SFT_ROOT:-$C/outputs/sft}
unset ROCR_VISIBLE_DEVICES ZE_AFFINITY_MASK HIP_VISIBLE_DEVICES
export HF_HUB_OFFLINE=1 HOME=/tmp/ac-home-$USER; mkdir -p "$HOME"
if [ -z "${CKPT:-}" ]; then
  RUN=${RUN:?RUN or CKPT}
  STEP=${STEP:-$(cat "$O/runs/$RUN/ckpt/latest_checkpointed_iteration.txt")}
  CKPT=$O/runs/$RUN/ckpt/global_step_$STEP; OUT=${OUT:-$O/runs/$RUN/merged}
fi
BASE=${BASE:-${IFCGYM_MODELS:-models}/Qwen3.5-9B}; OUT=${OUT:?OUT}; EXP=$OUT.export
echo "[merge] $CKPT (base $BASE) -> $OUT"
rm -rf "$EXP" "$OUT"
python3 -m verl.model_merger merge --backend fsdp --local_dir "$CKPT" --target_dir "$EXP" || [ -d "$EXP/lora_adapter" ]
if [ -d "$EXP/lora_adapter" ]; then
  rm -f "$EXP"/model*.safetensors
  python3 "$C/train/export_like_base.py" --base "$BASE" --adapter "$EXP/lora_adapter" --out "$OUT"
else
  python3 "$C/train/export_like_base.py" --base "$BASE" --tuned "$EXP" --out "$OUT"
  rm -f "$EXP"/model*.safetensors
fi
echo "$CKPT" > "$OUT/MERGED_FROM"
ls -la "$OUT" | awk '{print $5, $9}'
