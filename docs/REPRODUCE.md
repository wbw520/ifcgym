# Reproducing the paper

Paths below assume `IFCGYM_ROOT` is the repository root and the datasets are in place (`docs/DATA.md`). Every model is served
as an OpenAI-compatible endpoint (vLLM or SGLang; the flags are in `configs/models.example.json`).

## 1. Synthetic buildings and tasks (or download `wbw520/ifcgym-synthetic`)

```bash
python -m ifcgym.synth.make_buildings --root data/cubicasa5k/cubicasa5k --out data/synth/buildings --n 7000 --seed 0
python scripts/building_quality.py data/synth/buildings            # quality metrics per building
python scripts/select_buildings.py --root data/synth/buildings --n 3000 --out data/synth/buildings/selected.txt
bash scripts/regen_synth_tasks.sh outputs/tasks_synth               # ifcgym.tasks.generate on every selected building (all families)
bash scripts/gate_tasks.sh outputs/tasks_synth                      # lint + scripted oracles + independent verification
python scripts/sample_synth_tasks.py --tasks 'outputs/tasks_synth/*.jsonl' --out outputs/teacher_tasks.jsonl --k23 2 --k01 1 --seed 0
python -m ifcgym.tasks.paraphrase --model kimi-k3 --base-url $K3 --tasks outputs/teacher_tasks.jsonl --out outputs/teacher_paraphrased --workers 64 --effort high --max-tokens 8192
python scripts/audit_paraphrase.py outputs/teacher_paraphrased/teacher_tasks.jsonl --out outputs/teacher_paraphrased/audit_ids.txt
python scripts/judge_paraphrase.py outputs/teacher_paraphrased/teacher_tasks.jsonl --out outputs/teacher_paraphrased/judge.jsonl --base-url $K3 --model kimi-k3
python scripts/balance_id_labels.py outputs/teacher_final.jsonl
```

## 2. Teacher episodes

Kimi K3 in thinking mode, protocol `v2r` (helpers + cascade recipe), checker feedback after every commit, 40 rounds:

```bash
python -m ifcedit.run_tasks --model kimi-k3 --base-url $K3 --tasks outputs/teacher_final.jsonl --protocol v2r --feedback all \
  --budget 40 --max-tokens 16384 --out outputs/teacher --workers 48
```

## 3. SFT (9B and 27B students)

```bash
python ifcedit/train/build_sft_data.py --runs outputs/teacher --out outputs/sft_data --protocol v2r --synthetic-only \
  --specs outputs/teacher_final.jsonl --val-building-fraction 0.1 --seed 0
python ifcedit/train/filter_sft_by_tokens.py outputs/sft_data --max 32768 --margin 1024 --tokenizer $IFCGYM_MODELS/Qwen3.5-9B
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 BASE=$IFCGYM_MODELS/Qwen3.5-9B DATA=outputs/sft_data OUT=outputs/sft RUN=s2 EPOCHS=3 MAXLEN=32768 SEED=2 \
  bash ifcedit/train/sft.sh                                         # LoRA r=64 alpha=128 all-linear, lr 1e-4 cosine, batch 16
RUN=s2 SFT_ROOT=outputs/sft BASE=$IFCGYM_MODELS/Qwen3.5-9B bash ifcedit/train/merge_ckpt.sh   # -> outputs/sft/runs/s2/merged
```

The student base is the released Qwen3.5 checkpoint with `configs/chat_template_qwen35.jinja` as its chat template (it renders the
tool-calling turns and the preserved reasoning exactly as the SFT data does). The 27B student uses the same data and recipe with
`BASE=$IFCGYM_MODELS/Qwen3.5-27B`.

## 4. GRPO

```bash
# k = 4 pre-filter of the candidate tasks with the SFT student, then the pool rule (drop 4/4 solved, keep 1-3/4, cap 0/4 at 20 %)
python -m ifcedit.run_tasks --model sft-s2 --base-url $SFT --tasks outputs/rl_candidates.jsonl --protocol v3 --feedback none --repeats 4 --out outputs/prefilter
python ifcedit/train/rl/build_pool.py                               # -> rl_pool_final.jsonl + rl_pool_card.json (edit the input paths at the top)
python ifcedit/train/rl/build_rl_data.py --tasks outputs/rl_pool_final.jsonl --out outputs/grpo_data
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MODEL=outputs/sft/runs/s2/merged DATA=outputs/grpo_data OUT=outputs/grpo RUN=grpo_s2 \
  BATCH=64 GROUP=8 MINI=8 LR=1e-5 KL_COEF=1e-2 USE_KL=True EPOCHS=1 bash ifcedit/train/rl/grpo.sh
```

The reward is strict success computed by `ifcedit/train/rl/score_one.py` inside the agent loop (`ifcedit/train/rl/agent_loop.py`,
registered through `agent_loops.yaml`); online filtering drops groups with uniform rewards. The reward-term ablation sets
`IFCGYM_REWARD=intent_validity`. Export the final step with `ifcedit/train/rl/grpo_adapter.py` and `ifcedit/train/export_like_base.py`.

## 5. Evaluation

```bash
python -m ifcedit.run_tasks   --model <m> --base-url ... --tasks data/ifceditbench/tasks.jsonl --protocol v3 --feedback none --budget 20 --max-tokens 16384 --out outputs/eval/<m>
python -m ifcedit.run_bimedit  --model <m> --base-url ... --feedback none --budget 20 --max-tokens 16384 --out outputs/bimedit/<m>
python -m ifcedit.run_bibimbap --model <m> --base-url ... --protocol v3 --budget 25 --max-tokens 16384 --out outputs/bibimbap/<m>
python scripts/compare_arms.py outputs/eval/<a>/results.jsonl outputs/eval/<b>/results.jsonl   # paired difference, 95 % cluster-bootstrap CI
```

Prompted comparisons use the same commands with the untrained models; the two frameworks run through `ifcedit.run_external`
(`--framework mcp4ifc` with `MCP4IFC_TOOL_OUTPUT_CAP=8000` for the reported row, `--framework ifcagent`).
