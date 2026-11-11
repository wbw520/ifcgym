> Historical design notes of the study (October 2026), kept for provenance. References to `research/...` point to the internal research ledger, which is not part of this repository.

# IfcEdit — design of record (H001, study ifcedit)

Status: created 2026-10-19 after the comparability review passed (baseline target: BIM-Edit Gemma 4 31B Final 37.54, Table 1 row 7).
Everything below is developed on DEVELOPMENT projects (research/splits dev ids) and frozen in pilot H001/E001 before any comparison runs.
Cluster conventions: project CLAUDE.md (containers only; `ac-*` Slurm holds; python 3.12 container for CPU work).

## 1. Objects

| object | what it is | where |
|---|---|---|
| sandbox `S` | one IFC file loaded with IfcOpenShell in a subprocess; the agent's only tool is `execute_ifc_code(code)` exactly as in BIM-Edit F.4 (`ifc`, `api`, `util`, `element_util`, `guid` preloaded; `result` returned; `commit()` saves); 120 s per call; per-step snapshot of the file for replay and rollback | `ifcedit/sandbox.py` |
| agent `π` | OpenAI-compatible tool-calling loop (served local models via SGLang/vLLM; BIM-Edit system prompt F.3 verbatim; budget N tool calls; temperature 0 for baselines) | `ifcedit/agent.py` |
| feedback modes | `none` = BIM-Edit protocol (tool output only); `all` = after every commit the full checker report (validity, intent, dependency, preservation) is appended to the tool output; `validity` = schema/geometry report only | `ifcedit/feedback.py` |
| checkers `C` | validity: IfcOpenShell schema validation + geometry rebuild of the touched elements; intent: task-specific predicate; dependency: every element in the task's dependent set updated consistently; preservation: entity-level diff outside the allowed set is empty; under-edit: intent failed with zero modified entities | `ifcedit/checkers/` |
| semantic diff | before/after correspondence of entities by GlobalId (rooted) and by content hash (unrooted), reporting added / removed / modified entities and modified attributes; the decoupling of exporter-shared placements (research/case-wall76) is handled by the diff, not by the agent | `ifcedit/diff.py` |
| task generator `G` | cascade-tiered edit templates applied to real projects: tier 0 attribute/property edits; tier 1 host-dependent edits (move/resize a wall with its openings and fills); tier 2 space-affecting edits (space boundaries, quantities); tier 3 connection-affecting edits (joined walls). Each task = prompt + checker spec (intent predicate, dependent set, allowed-change set); no target file | `ifcedit/tasks/` |
| BIM-Edit evaluator `E_B` | re-implementation of the BIM-Edit scorer from the paper (Sec. 3, Eq. 4) and the released evaluator defaults (`data/bimedit-runs/*/eval/resolved_defaults.json`); validated against the official per-task scores of the four API models whose edited IFC outputs are released | `ifcedit/bimedit_eval.py` |
| metrics | strict success (intent ∧ dependency ∧ preservation ∧ validity), side-effect entity count, new-error delta, under-edit rate, turns/tokens/wall time; BIM-Edit Final/strict with `E_B` | `ifcedit/metrics.py` |

Model roles (2026-10-19): baseline anchor = google/gemma-4-31b-it (the BIM-Edit open model, Table 1 row 7); strongest local prompted
baseline and teacher = DeepSeek-V4-Flash (fp8, 4 cards, vLLM); Kimi-K3 (1.6 TB) is the reserve teacher if DeepSeek trajectories are too
weak; student = Qwen3-8B (Qwen3-14B as the scale check). All served from /lustre/share/downloaded/models through the shared container images.

## 2. Data

- Corpus: ifc-bench (22 projects, CC BY 4.0; the GIS city model excluded). Project-level split (research/splits): development ids for instrument work
  and task-template design, evaluation ids never touched until the comparisons run. `west_riverside_hospital` IFC2X3/IFC4 are one project.
- External tests: BIM-Edit (324 tasks, official evaluator defaults; CC BY 4.0, pinned revision 0ae510bd) and BIBIMBAP (100 tasks; re-implemented
  if the scripts are not released). Never used for training or template design.
- Teacher trajectories and generated tasks live under `outputs/` (ignored by git); provenance (project, template, seed) in each task record.

## 3. Experiments (research/experiments/H001)

| id | kind | what | frozen artefact / target |
|---|---|---|---|
| E001 | pilot | sandbox, checkers, diff, evaluator re-implementation, task generator on dev projects | freeze: evaluator agreement with official per-task Final (claude-sonnet / gemini / gpt-5.4 / gpt-5.4-mini edited outputs), wall #76 case (only C passes), generated-task validity on dev projects |
| E002 | baseline | Gemma 4 31B on BIM-Edit, our harness, check-none, budget 20, T=0 | Final 37.54 ± 5% rel (Table 1 row 7); strict 0.9%; failure profile vs Table 3 row 7 |
| E003 | pilot | headroom: DeepSeek-V4-Flash and Gemma 4 31B, check-none vs check-all, on dev-project tasks by tier + wall #76 | failure-class profile (dependency / preservation / intent / validity) per tier; decides task difficulty and the visual-input question |
| later | treatment / control / ablation / heldout | SFT student; +GRPO gated reward; reward-term ablations; matched-SFT control; held-out projects, BIM-Edit, BIBIMBAP; 3 seeds | preregistered before any result |

## 4. Decisions log

- 2026-10-19 sandbox tool = BIM-Edit's single `execute_ifc_code` (not a curated toolset): keeps E002 comparable and gives the policy the broadest action space; curated helper tools, if any, are an ablation later.
- 2026-10-19 visual input: not in the reward; whether screenshots enter the agent's observation is decided by E003 (with vs without on spatial tasks).
- 2026-10-20 student backbone = Qwen3.5-9B (user: "why not Qwen3.5"): native thinking with reasoning_content kept on every turn
  after the single user message, verl 0.9 recipe from depression-paper (fla fast path, sft_entry for multi-GPU, export_like_base);
  Qwen3-8B stays only as the E004 smoke base. E006 (prereg #6) = Qwen3.5-9B, tools v1, DeepSeek no-think data.
- 2026-10-20 tools v2 (user: identical tool definitions are NOT required for comparability; BIM-Edit E002 is closed): `inspect_element`
  (structured read-only description: placement, representation polygon, openings/fills, connected walls with the touching end, space
  boundaries, bounded-space footprints, psets on request) + `execute_ifc_code` with the helper module `h` (`ifcedit/ifctools.py`:
  translate, move_wall_end/stretch_wall_end, shift_profile_vertices, shift_boundary_curve, set_extrusion_depth, set_property on a
  private pset, delete, connect_walls, private) + an IFC reference card. The helpers are element-local primitives; which dependents
  to update, at which end and in which direction stays with the agent. Request-side pruning keeps the last 6 tool outputs in full
  (applied identically in training). Self-check: the wall #76 cascade through the helpers is strict. Used by every system on the
  IfcEdit benchmark from E007 on; BIM-Edit external numbers of v2 systems carry a "protocol differs" note.
- 2026-10-20 think distillation (user: "I want to distil the teacher's think"): the E004 DeepSeek episodes were collected in chat mode
  (no reasoning); E007 re-collects with a thinking teacher (Kimi-K3 on a 4-node job - the user's cap is 4 nodes, so ac-gym must be
  released first - or DeepSeek-V4-Pro thinking on 2 nodes), storing reasoning_content and passing it back (preserved thinking).
- 2026-10-20 E007 plan (prereg #1): teacher = DeepSeek-V4-Flash thinking + cascade recipe (v2r) on generator-v2 training tasks
  (trial on 40 solvable tier-2/3 tasks: v2 10/40, v2r 17/40, v2r+think 17/40 against a helper ceiling of 30/40); student Qwen3.5-9B
  thinking, protocol v2 prompt (recipe stripped), reasoning traces kept; dev-val v2 = 300 stratified perpendicular-move tasks on
  fzk_house + wbdg_office; same decision rule as E006. Kimi-K3 not started (4-node cap; undocumented on H100).
- 2026-10-21 BIM-Edit comparison (user): v2 systems are compared DIRECTLY with the published BIM-Edit numbers; the published
  baselines are not re-run under v2. The tool protocol is part of the contribution (system = model + protocol); the paper says so
  and additionally reports the student under protocol v1 as the like-for-like row.
- 2026-10-21 external benchmarks: BIM-Edit (324 tasks; compared directly with the published table, protocol is part of the
  system) and BIBIMBAP (EC3 2026, 100 tasks, authors' own tests and data public: github.com/nbharathik/bibimbap + HF
  clemenskujat/bibimbap-data, CC BY 4.0; data/bibimbap/; adapter ifcedit/run_bibimbap.py scores with the authors' execute_test).
- 2026-10-20 incidents (research log): rescore_tasks used dev specs for training ids (fixed, --tasks); outputs/tasks/wbdg_office.jsonl
  carried pre-rename rule names (checker aliases); IfcOpenShell thread-safety (process pool); the millimetre-model unit inversion
  (hitos: metre vectors were multiplied by the unit scale while meshes are metres - fixed, 118 hitos verdicts flipped, E006 unaffected);
  joined_to measured corner-vertex distance (now surface-sampled gap); six E003 hospital trajectories concatenated by concurrent
  writes (repaired, re-scored).

## 5. Instrument freeze (E001, 2026-10-19)

- BIM-Edit scorer = our re-implementation (`ifcedit/bimedit_calib.py`, config `ifcedit/bimedit_scorer_config.json`, features from
  `ifcedit/bimedit_features.py`). The official evaluator is not released (authors declined). Calibrated on the released per-task scores
  of claude-sonnet-4.6 + gemini-3-flash, validated on gpt-5.4 + gpt-5.4-mini. Agreement: aggregate Final ours minus official =
  claude -3.2, gemini +1.0, gpt-5.4 +1.3, gpt-5.4-mini -2.2 points; per-task MAE 0.18-0.23, |diff| <= 0.05 on 20-46% of tasks; by
  operation ours underscores updates by ~8 and deletes by ~4 points, creates agree within 2-5. Every BIM-Edit number in this study is
  reported together with this bias; the strict 5% reproduction tolerance of E002 is evaluated on the aggregate and the per-operation
  pattern, and the evaluator deviation is recorded in the comparability review.
- Gated checkers: wall #76 case reproduced exactly (only C_complete passes; A space/boundary, B boundary, D double translation,
  E 194 side-effect products); unchanged model fails intent on 10/10 sampled tasks with preservation intact and under-edit flagged.
  Tier-2 rules (space meets wall face, boundary on wall) are enforced only where they held in the original model.
- Task generator v0: 5 templates, 338 tasks on the 7 development projects (tier 0: 140, tier 1: 150, tier 3: 48; tier 2 is folded
  into tier 3 in these models because every hosted wall is also joined).
- Known limits to carry into the design review: tier-3 dependency rules are strict (joint gap may not grow; space faces and
  boundaries must follow); agents can satisfy intent while leaving hundreds of side effects (shared property sets, shared placements);
  no spatial/topological phrasings yet (all prompts name GlobalIds).

## 6. Checker tolerances and their provenance (2026-10-22, user: "where do these definitions come from?")

The numeric tolerances were set by the author when the checkers were written (E001, 2026-10-19) as engineering judgement; no
external source was recorded then. Anchors found afterwards and applied:

| check | tolerance | origin |
|---|---|---|
| translate (wall, door/window, column), heights, wall thickness, slab thickness | 0.02 m | ours; same order as BIBIMBAP geometry tests (`TOLERANCE = 0.02` in several of its tests, 0.05-0.1 in others) |
| shorten / rotate: axis end points | 0.03 m per end (0.05 until 2026-10-22) | tightened after the standards search: DIN 18202 +-30 mm max, ACI 117 +-25 mm, USIBD LOA20 ceiling 50 mm (research/literature/tolerance-anchors-2026-10-22.md) |
| create door/window/wall/column: dimensions, end points, centre | 0.03 m (0.05 until 2026-10-22) | same anchors; BIBIMBAP uses 0.05-0.1 m |
| create slab / space footprint | symmetric difference <= 5% of the area + 0.05 m^2 (space: + 0.1) | ours |
| joined_to (wall joint) | surface gap <= 0.02 m or solids overlap | ours |
| numeric answers (area, volume, length, distance) | relative 1% + absolute 0.1 | aligned 2026-10-22 to BIBIMBAP's numeric tests (`rel_tol = 0.001, abs_tol = 0.1`), slightly looser relatively because our truths come from meshes |
| door width answer | relative 1% + absolute 0.02 m | ours (0.1 m would not separate 0.8 / 0.9 m doors) |
| counts, GlobalId answers | exact | both benchmarks |
| BIM-Edit external scoring | bbox IoU threshold 0.05 | BIM-Edit paper / our calibrated re-implementation (section 5) |

Independent review (2026-10-22) confirmed that a perfectly executed prompt never fails on rounding (prompts carry 2-3 decimals).
Standards and literature anchors for every class, with verdicts, are in research/literature/tolerance-anchors-2026-10-22.md
(DIN 18202, ACI 117, BS 5606, USIBD LOA, GSA/NIST, Bonduel 2017, Bassier 2020, BIM-Edit, BIBIMBAP).
What is NOT established: whether practitioners accept these thresholds as "correct edit" and whether the dependency rules match
professional expectations; this is the planned 150-task expert audit (research/tiers-evidence-2026-10-19.md section 5), still open.
