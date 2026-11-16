# IfcGym and IfcEdit

Code for **Learning to Edit IFC Building Models: Verifiable Checkers as the Training Signal for Dependency-Aware Editing Agents**.

* **IfcGym** (`ifcgym/`) is an editing environment for IFC building models: a replayable CodeAct sandbox with one tool,
  `execute_ifc_code`, a helper package `h` of 28 element-local functions, four gated checkers (validity, intent, dependency,
  preservation) that turn a task specification into a strict pass/fail verdict, a synthetic-building generator and a task generator.
* **IfcEdit** (`ifcedit/`) is the agent trained in it: the tool-calling agent loop, the evaluation runners for IfcEditBench,
  BIM-Edit and BIBIMBAP, re-implementations of two prior frameworks (MCP4IFC, IFC-Agent) for comparison, and the two-stage
  training recipe (distillation of verified teacher episodes by SFT, then GRPO with strict success as the reward).

The datasets (IfcEditBench, the synthetic buildings, the SFT corpus and the RL pool) are distributed on Hugging Face, see
[`docs/DATA.md`](docs/DATA.md). The paper's numbers are reproduced by the steps in [`docs/REPRODUCE.md`](docs/REPRODUCE.md).

## Installation

Python 3.12 is required (IfcOpenShell 0.8.5 wheels are built for it).

```bash
git clone https://github.com/wbw520/ifcgym && cd ifcgym
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e .                      # ifcopenshell, numpy, shapely, scipy, openai, pyyaml, pyarrow, pydantic
pip install -e '.[figures]'           # optional: matplotlib + pillow for renders
export IFCGYM_ROOT=$PWD               # repository root; data/ outputs/ models/ default to subdirectories of it
```

Training additionally needs the verl 0.9.0 stack (`pip install -e '.[train]'`, or the verl container) and a GPU node.

## Data layout

Download the datasets as described in `docs/DATA.md` and place them under `$IFCGYM_DATA` (default `data/`):

```
data/ifc-bench/<project>/arc.ifc         real projects (13 of them form IfcEditBench)
data/ifceditbench/tasks.jsonl            the 1,990 IfcEditBench tasks (ifc paths relative to data/)
data/synth/buildings/<building>/arc.ifc  synthetic training buildings and their generated tasks
data/bimedit/{complex,simple}/*.ifc      BIM-Edit models;  data/bimedit-runs/tasks.jsonl   its 324 tasks
data/bibimbap/code                       the BIBIMBAP repository with its per-task tests and IFC files
```

## Quick start: evaluate a served model on IfcEditBench

Every runner talks to an OpenAI-compatible chat endpoint. Serve a model (vLLM example, flags as used in the paper):

```bash
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3.5-9B --served-model-name qwen35-9b \
  --max-model-len 65536 --language-model-only --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --port 30001
```

Run the tasks with the four checkers and no checker feedback (the evaluation protocol: 20 tool-call rounds, temperature 0,
thinking on, 16,384 output tokens per turn):

```bash
python -m ifcedit.run_tasks --model qwen35-9b --base-url http://localhost:30001/v1 \
  --tasks data/ifceditbench/tasks.jsonl --protocol v3 --feedback none --budget 20 --max-tokens 16384 \
  --out outputs/eval/qwen35-9b --workers 8
```

`outputs/eval/qwen35-9b/results.jsonl` holds one line per task with the verdict of each checker and `strict` (all four pass);
`trajectories/<task_id>.json` the full tool-calling episode and `edited/<task_id>.ifc` the committed model. Strict success is the
share of tasks with `strict == true`. `scripts/compare_arms.py` computes paired differences between two runs with cluster-bootstrap
intervals, `scripts/rescore_tasks.py` re-runs the checkers on saved outputs, `scripts/ifc_render.py` renders before/after views.

Checker feedback after every executed step (`--feedback all`) and a doubled budget (`--budget 40`) are the evaluation-time
variants of the paper.

## External benchmarks and prior frameworks

```bash
python -m ifcedit.run_bimedit  --model qwen35-9b --base-url ... --feedback none --budget 20 --out outputs/bimedit/qwen35-9b
python -m ifcedit.run_bibimbap --model qwen35-9b --base-url ... --protocol v3 --budget 25 --out outputs/bibimbap/qwen35-9b
python -m ifcedit.run_external --framework mcp4ifc  --model deepseek-v4-flash --base-url ... --tasks data/ifceditbench/tasks.jsonl --out outputs/frameworks/mcp4ifc
python -m ifcedit.run_external --framework ifcagent --model deepseek-v4-flash --base-url ... --tasks data/ifceditbench/tasks.jsonl --out outputs/frameworks/ifcagent
```

BIM-Edit outputs are scored by our calibrated re-implementation of its evaluator (`ifcedit/bimedit_eval.py`, features in
`ifcedit/bimedit_features.py`, rule set `ifcedit/bimedit_scorer_config.json`); BIBIMBAP outputs by the benchmark's own tests.

## The environment in code

```python
import json
from ifcgym.sandbox import Sandbox
from ifcgym.checkers import check

task = json.loads(open("data/ifceditbench/tasks.jsonl").readline())
sb = Sandbox(input_ifc=task["ifc"], workdir="outputs/demo", helpers=True, isolate=True)   # protocol v3: helpers as `h`, namespace jail
r = sb.execute("result = h.inspect(ifc.by_guid('%s'))" % task["target"][0])              # ToolResult: output, committed, error, seconds
print(r.output)
sb.execute("h.translate(ifc.by_guid('%s'), (0.3, 0.0, 0.0)); commit()" % task["target"][0])
report = check(task["ifc"], sb.final_path(), task)          # CheckReport: validity, intent, dependency, preservation, under_edit, strict
print(report.strict, report.details)
sb.close()
```

`ifcgym/ifctools.py` documents every helper (the table in the paper); `ifcgym/checkers.py` the four gates and their tolerances
(`docs/CHECKER_TOLERANCES.md` gives their provenance).

## Generating buildings and tasks

```bash
python -m ifcgym.synth.make_buildings --root data/cubicasa5k/cubicasa5k --out data/synth/buildings --n 50 --seed 0
python -m ifcgym.tasks.generate --ifc data/synth/buildings/<b>/arc.ifc --project <b> --out outputs/tasks/<b>.jsonl \
  --per-template 2 --per-template-move 6 --seed 1 --perpendicular --families create retrieve spatial update delete create2 retrieve2 concept compound qto cover cover2 cover3 cover4
python -m ifcgym.tasks.paraphrase --model <llm> --base-url ... --tasks 'outputs/tasks/*.jsonl' --out outputs/tasks_paraphrased
bash scripts/gate_tasks.sh outputs/tasks                       # static lint, scripted oracles, independent reference verification
```

## Training

`docs/REPRODUCE.md` lists the exact sequence: teacher episodes with checker feedback (`ifcedit.run_tasks --protocol v2r --feedback all`),
`ifcedit/train/build_sft_data.py` (strict episodes only, failed tool calls dropped, recipe and feedback stripped), `ifcedit/train/sft.sh`
(LoRA rank 64 on all linear layers, lr 1e-4 cosine, 3 epochs, batch 16, sequences up to 32,768 tokens), `ifcedit/train/rl/build_pool.py`
(offline k = 4 pre-filter of the task pool), `ifcedit/train/rl/build_rl_data.py` and `ifcedit/train/rl/grpo.sh` (64 prompts x 8 rollouts
per step, lr 1e-5, KL 1e-2, 100 steps), then `ifcedit/train/merge_ckpt.sh` to serve the result. The shell scripts were written for a
Slurm cluster with the verl container; the Python entry points and the hyperparameters are cluster-independent.


## Licence

Apache License 2.0 for the code (`LICENSE`). Third-party components and the licences of the datasets are listed in `NOTICE.md`;
in particular the synthetic buildings and everything derived from them are CC BY-NC 4.0 because their floor plans come from CubiCasa5K.
