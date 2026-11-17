# Datasets

All datasets are distributed on Hugging Face; the code expects them under `$IFCGYM_DATA` (default `data/`).

| Hugging Face dataset | Contents | Licence | Size |
|---|---|---|---|
| [wbw520wbw520/ifceditbench](https://huggingface.co/datasets/wbw520wbw520/ifceditbench) | the 1,990 IfcEditBench tasks (`tasks.jsonl`) and the 13 real projects they edit (`ifc-bench/<project>/arc.ifc` with each project's own licence file) | CC BY 4.0 (collection); per-project licences inside | 0.3 GB |
| [wbw520wbw520/ifcgym-synthetic](https://huggingface.co/datasets/wbw520wbw520/ifcgym-synthetic) | 3,400 synthetic training buildings (3,000 of the main set + 400 with the second door/window/opening generator) with randomised export styles, their generated tasks and the nine paraphrased teacher task lists, the SFT corpus (19,141 train / 2,270 val episodes, parquet) and the GRPO task pool (5,895 tasks + parquet) | CC BY-NC 4.0 (derived from CubiCasa5K) | 2.4 GB |

The two datasets are opened together with the preprint of the paper.

External benchmarks are downloaded from their authors:

| Benchmark | Where | Place under |
|---|---|---|
| BIM-Edit (324 tasks) | Hugging Face `BIM-Edit` organisation, revision used in the paper: `0ae510bd` | `data/bimedit/` (IFC files, `complex/` and `simple/`), `data/bimedit-runs/tasks.jsonl` (tasks), `data/bimedit-runs/<model>/` (released per-task scores, for the scorer calibration) |
| BIBIMBAP (100 tasks) | github.com/nbharathik/bibimbap and Hugging Face `clemenskujat/bibimbap-data` | `data/bibimbap/code` (repository with `benchmark.py`, `data/`, `llm_agents/`) |
| CubiCasa5K (floor plans, only needed to generate new buildings) | zenodo record of Kalervo et al. (2019) | `data/cubicasa5k/cubicasa5k/{colorful,high_quality,high_quality_architectural}` |
| ifc-bench (all 22 projects, for the development split) | Hugging Face `sylvainHellin/ifc-bench` | `data/ifc-bench/<project>/` |

## Task records

One JSON object per line:

```
task_id            unique id, <project>-<template>-<n>-s<seed>
project            building name (data/ifc-bench/<project> or a synthetic building)
template           generator template, e.g. C-create-column, U-move-door-T, T2-translate-wall-space
tier               dependency tier 0-3 (Table 2 of the paper)
prompt             natural-language instruction (paraphrased by an LLM for training tasks)
ifc                input model, relative to data/
target             GlobalIds the instruction refers to
allowed            GlobalIds that may change (dependents included); everything else is protected by the preservation gate
allowed_new_classes IFC classes that may be created
intent             the checkable specification (type and parameters) tested by the intent gate
dependents         [{guid, rule}] what must follow the target, tested by the dependency gate
meta               optional generator metadata
```

`ifcgym.checkers.check(task, input_ifc, output_ifc)` returns the four verdicts and `strict`.

## Episodes (SFT corpus)

verl-style parquet with a `messages` column: system prompt, user prompt, assistant turns with `tool_calls` and `reasoning_content`,
tool outputs. Only episodes that passed all four checkers are kept; tool calls that raised an exception were dropped, and the teacher's
cascade recipe and checker feedback were stripped (`ifcedit/train/build_sft_data.py`). The teacher was Kimi K3 in thinking mode.

## Development split

`configs/splits/dev_ids.txt` lists the seven ifc-bench projects used for instrument development (fzk_house, wbdg_office, ac20, hitos,
west_riverside_hospital, fantasy_office_building_2, fantasy_residential_building_1). They never appear in IfcEditBench.
