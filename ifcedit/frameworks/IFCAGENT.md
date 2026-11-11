# IFC-Agent reproduction (DeepSeek-V4-Flash backbone)

Reproduction of Gao, Hu, Chai, Weng, Li (2026), "Multi-agent framework for schema-guided reasoning and tool-augmented interaction with
IFC models", Automation in Construction 186, 106888 (DOI 10.1016/j.autcon.2026.106888; no code released) as a prior-method row for the
IfcEdit held-out tasks. Written from the publisher PDF.

## Files

| File | Content |
|---|---|
| `ifcedit/frameworks/ifcagent_tools.py` | the paper's tool set over IfcOpenShell 0.8.5 as one class `IfcAgentTools` (file, `globalid_dict`, `stepid_dict`, procedure memory, edit log per instance; tool execution serialised per instance, file IO per process); CHOICES C1-C14 in the module docstring |
| `ifcedit/frameworks/ifcagent.py` | `run(prompt, ifc_path, workdir, client, model, budget, max_tokens, temperature)`: Main Agent + TaskDecompositionAgent + 7 Sequential executors + per-entity text-ReAct sub-agents (Mode I) |
| `ifcedit/frameworks/IFCAGENT.md` | this note |

Run: `python -m ifcedit.run_external --framework ifcagent --model deepseek-v4-flash --base-url http://<node>:30004/v1 --tasks <jsonl> --out <dir> [--workers 4] [--budget 20]`
inside the py312 container (ifcopenshell is not on the login node). `run_external.py` was not changed.

## What is reproduced

Mode I of the framework end to end:

1. **Scripted prefix** (shown to the main agent as its first four ReAct steps, Fig. 4 style): `LoadIFCFile` -> `RewriteIfNeeded`
   (regex) -> `QueryMemory` (fresh per task, so "Memory is empty") -> `TaskDecompositionAgent` (LLM call, Fig. 9 principles verbatim,
   output `{"subtasks": [...]}`).
2. **Main Agent**: text ReAct, max 25 steps, tools `GetGlobalIdsByAttributeValue`, the seven `Sequential*Executor`s, `SaveFinalAnswer`,
   `ExportModifiedIFC` (plus the prefix tools). It resolves the target entities of each subtask (GlobalIds in the query,
   `GetGlobalIdsByAttributeValue`, or the previous subtask's results = "working scope"), picks ONE executor with the Fig. 8 table
   (verbatim, in its system prompt; the executors are the tools it chooses among), then `SaveFinalAnswer` and, for modifications,
   `ExportModifiedIFC` (export first; `SaveFinalAnswer` is the terminal action, like the `FinalAnswer` tool that ends the Fig. 4 trace).
3. **Executors**: one text-ReAct sub-agent per GlobalId (8 concurrent, results aggregated in entity order), each with the executor's
   fixed toolset of Table 2, max 25 steps; the executor returns `- <gid>: <sub-agent final answer>` lines to the main agent.
4. **ReAct format**: LangChain zero-shot template (Thought / Action / Action Input JSON / Observation / Final Answer), own parser
   (LangChain's MRKL regex; JSON, Python-literal or first balanced `{...}` as Action Input), stop sequences `\nObservation:` as
   LangChain's `ZeroShotAgent` sets them; temperature 0, top_p 1, presence/frequency penalty 0; no native function calling.
5. **Edits**: only `SetFieldByStepId` (field overwrite on an existing STEP entity, wrapper types preserved) + `ExportModifiedIFC`.
   If the model was changed and never exported, the in-memory model is written to `ifc_path` and `fallback_write` is recorded in the
   trajectory (generous reading, as for the MCP4IFC port).

Not reproduced: **Mode II** (ObserverAgent, BulkExecutor, AggregationAgent, PythonExecutor). An executor call with more than 20
GlobalIds processes the first 20 in Mode I and writes `mode2_not_reproduced` into the observation and the trajectory notes.

Budget semantics: `budget` (runner default 20) = max executor calls per task -> outcome `budget` when exceeded, or when the main agent
hits its 25-step cap without a final answer. `tool_calls` = all ReAct actions that reached a tool (sub-agents + main-agent tools; executor
calls themselves are not counted). `final_text` = the `SaveFinalAnswer` text (else the main agent's Final Answer).

## Tool list (deviations from the paper)

All names / argument names as in the brief's table. Aliases for the paper's inconsistent spellings are accepted
(`QuerySingleGlobalIdDict`, `ExtractConnectivityRelationsByGlobalId`, `ExtractAssignmentRelationsByGlobalId`, `FinalAnswer`).

| Tool | Args | Deviation / note |
|---|---|---|
| LoadIFCFile | file_path | observation adds schema, IfcProduct count and the length unit (`1 model unit = 0.3048 m`) - the paper's models are in metres, ours are partly in feet |
| GetInfoByGlobalId | global_id | Fig. 2 verbatim; stores the Fig. 2 field list; missing fields (Tag on IfcSpace, PredefinedType in IFC2X3) stored as None |
| QueryWithGlobalIdDict | global_id, field | dotted path, lists mapped over, `Items[0]` indexing, one-level fall-through into pset dicts (`PropertySet.FireRating`); entity values shown as raw STEP expressions |
| GetInfoByStepId | step_ids | Fig. 4 success line + `str(entity)` + parsed attribute list (Mode I regime of Sec. 6.1.1) |
| QueryWithStepIdDict | step_ids, field | as above |
| ExtractPropertySetByGlobalId | global_id | nested by pset; the observation lists the STEP id and wrapper type of every IfcPropertySingleValue (needed for SetFieldByStepId; the paper's Q51/Q52 imply the agent had them) |
| ExtractQuantitySetByGlobalId | global_id | same, IfcElementQuantity |
| ExtractRelationsByGlobalId | global_id | all inverse relations, Appendix C line format `HasAssociations: [#17045] -> STEP entity: IfcRelAssociatesMaterial; ...` |
| ExtractOpeningsByGlobalId | global_id | HasOpenings with fills, FillsVoids with host, VoidsElements, HasFillings |
| ExtractConnectivityByGlobalId | global_id | ConnectedTo / ConnectedFrom with connection types |
| ExtractAssignmentByGlobalId | global_id | HasAssignments, IsGroupedBy, ReferencedBy |
| ExtractBoundariesByGlobalId | global_id | BoundedBy (space) / ProvidesBoundaries (element) |
| ExtractContainsElementsByGlobalId | global_id | ContainsElements grouped by type with GlobalIds, ContainedInStructure, (De)composes |
| FindAdjacentSpacesByElement | global_id | space -> spaces sharing a bounding element; element -> spaces it bounds |
| DescribeRelationStep | step_id | deterministic (C7) |
| DecideRelationNextHop | step_id, goal? | deterministic: candidate fields / inverse relations with target types, keyword-ranked by `goal` (C7) |
| ComputePlacementMatrixFromStepId | step_id | LOCAL matrix of one IfcLocalPlacement / IfcAxis2Placement3D; the chain is multiplied by the agent (Appendix D) |
| GetGlobalIdsByAttributeValue | ifc_type, attribute?, value? | by_type with subtypes + optional equality filter; count + complete list (200 shown) |
| SetFieldByStepId | step_id, field, value | value coerced to the current type; wrappers (IfcLabel, IfcReal, IfcBoolean, ...) re-created; `#id` strings resolve entity references; no creation / deletion / propagation; caches refreshed after a write (C5, C6) |
| ExportModifiedIFC | output_path | relative to the task workdir; never overwrites the source |
| QueryMemory | - | ISARC-style `### Procedure Context (Tools) ###` block or "Memory is empty" |
| RewriteIfNeeded | query | regex: `>`, `<`, `>=`, `<=`, `!=`, `=` -> words; no unit insertion |
| SaveFinalAnswer | answer | stores the final text |
| TaskDecompositionAgent | question | LLM call (main-agent tool) |
| Sequential*Executor (7) | subtask, global_ids, context? | main-agent tools; `context` is OURS (facts the sub-agents need, e.g. the new value) |

Observations are cut at 8000 characters, entity lists at 50 entries; `ExtractContainsElementsByGlobalId` prints the per-type counts first and names only for groups of <= 10 so that the cap never hides a type. All errors are returned as `Error: ...` observations (no raise).

## OURS prompts (verbatim)

Everything the paper prints is used verbatim (Fig. 2 GetInfoByGlobalId description, Fig. 5 memory rules, Fig. 8 executor table,
Fig. 9 decomposition principles, Appendix D placement procedure, ISARC-2025 instruction lines a-e, LangChain's zero-shot ReAct template).
The following were written by us.

### Sub-agent system prompt (`SUBAGENT_SYSTEM`; `{isarc}`, `{memory}` = verbatim blocks; `{placement}` = Appendix D for the Geometry executor, empty otherwise)

```
You are a sub-agent of IFC-Agent. You work on ONE IFC entity with a fixed set of tools over the IFC model (IfcOpenShell). Follow the ReAct format exactly: write "Thought:", then "Action:" with exactly one tool name, then "Action Input:" with one JSON object, then stop and wait for the Observation. Give "Final Answer:" only when the subtask is done for this entity; the final answer must contain the entity's GlobalId and the values found or the change made.
Instructions:
{isarc}
Memory access rules (only the tools listed below exist for you):
{memory}
Schema path preferences:
- Depth of an element: prefer the geometric definition (IfcShapeRepresentation -> IfcExtrudedAreaSolid -> Depth) over the approximate bounding dimension (IfcShapeRepresentation -> IfcBoundingBox -> ZDim).
- Wall thickness: geometry path IfcExtrudedAreaSolid -> SweptArea -> IfcRectangleProfileDef -> YDim, or material path IfcMaterialLayerSetUsage -> ForLayerSet -> MaterialLayers[n] -> LayerThickness.
Observations show STEP entities as '#id=IFCTYPE(...)': read the referenced STEP ids from them and follow them with GetInfoByStepId. Values are in the model's length unit (see the context line). Modifying the model is only possible with SetFieldByStepId on the STEP entity that holds the field (an attribute of the element, the NominalValue of an IfcPropertySingleValue, a LayerThickness, a Depth, ...); do not export, the main agent exports. If a tool is missing for the subtask, say so in the Final Answer.
{placement}
```

### Sub-agent question (per entity)

```
Subtask: {subtask}
Target entity: GlobalId {gid} = STEP id #{id} ({IfcType}, Name={name!r})[; ObjectPlacement=#...; Representation=#...  (Geometry executor only)]
Length unit of the model: 1 model unit = {unit_scale} m.
[Context from the main agent: {context}]
Work on this one entity only and give the result for it in the Final Answer.
```

### Main-agent system prompt (`MAIN_SYSTEM`; `{table}` = Fig. 8 verbatim, `{limit}` = 20)

```
You are the Main Agent of IFC-Agent, the central controller for one task on an IFC model. The IFC file is loaded, the query was rewritten, the memory was checked and the task was decomposed into sequential subtasks (see the trace). Follow the ReAct format exactly: "Thought:", "Action:" with one tool name, "Action Input:" with one JSON object, then stop and wait for the Observation.
For each subtask in order: (1) identify the target entities: GlobalIds given in the task, found with GetGlobalIdsByAttributeValue, or taken from the previous subtask's results (the output of a subtask is the working scope of the next); (2) choose ONE sequential executor with the table below and call it with the subtask text and the list of target GlobalIds; it runs one sub-agent per entity with that executor's tools and returns their results. Pass the facts the sub-agents need (values from the task, results of earlier subtasks) in the "context" field. After the last subtask: if the model was modified, call ExportModifiedIFC with "export/updated_model.ifc"; then call SaveFinalAnswer with the complete answer (for a question: the answer in exactly the form the task asks for; for a modification: what was changed) - SaveFinalAnswer ends the task. Do not answer from your own knowledge; everything must come from the executors' results.
{table}
Each executor call handles at most {limit} entities (Mode I).
```

The main agent's user message is the task prompt unchanged (no export sentence appended; the export instruction lives in the system
prompt because our retrieval tasks say "Do not modify the model").

### TaskDecompositionAgent prompt (`DECOMPOSITION_SYSTEM`; `{principles}` = Fig. 9 verbatim; user message = `{"question": "<rewritten query>"}`)

```
You are the TaskDecompositionAgent of IFC-Agent. Decompose the user's query about an IFC model into subtasks that GlobalId-level executors run one after another.
{principles}
Examples:
"Find all walls and calculate their total area" -> {"subtasks": ["Find all walls", "Find the area of each wall", "Calculate the total area of walls"]}
"Rename the wall with GlobalId X: set its Name to 'W1'" -> {"subtasks": ["Set the Name attribute of the wall with GlobalId X to 'W1'"]}
Output exactly one JSON object {"subtasks": ["<subtask 1>", "<subtask 2>", ...]} and nothing else.
```

### Tool descriptions

All tool descriptions except `GetInfoByGlobalId` (Fig. 2 verbatim) are OURS, written in the Fig. 2 style (purpose, listed fields,
"Input must be a JSON object with keys", "Output"); see `IfcAgentTools.DESCRIPTIONS` and `MAIN_DESCRIPTIONS` in the code. The
executor description: "Run the subtask on each listed entity with one sub-agent per GlobalId; the sub-agents use the tools [...].
Input must be a JSON object with keys: 'subtask' (string), 'global_ids' (list of GlobalId strings), 'context' (string, optional). Output:
the result of every sub-agent, keyed by GlobalId."

## Choices taken (beyond the tool-level C1-C14 in `ifcagent_tools.py`)

| # | Choice | Why |
|---|---|---|
| A1 | The four prefix steps (LoadIFCFile, RewriteIfNeeded, QueryMemory, TaskDecompositionAgent) are scripted and injected as the main agent's first ReAct steps instead of being chosen by the LLM | they are fixed in the paper's loop (Sec. 4); saves 4 of the 25 main-agent steps for real work |
| A2 | Executor selection = the main agent picks the executor as its Action with the Fig. 8 table in its prompt (LLM, "semantic reasoning"), no separate routing call | Fig. 8 is titled "Executor Tool Name"; brief #9 |
| A3 | 25 steps per ReAct agent instance (main agent and every sub-agent separately) | brief #11 |
| A4 | Sub-agents get the entity's STEP id / type / Name (and placement + representation ids for Geometry) in their question | Geometry / Quantity executors have no GetInfoByGlobalId in Table 2 and could not reach their entity otherwise; the paper's global caches made it visible |
| A5 | `context` field on executor calls | without it the sub-agents never see the value to set ("set FireRating to 'REI 90'") when the main agent shortens the subtask |
| A6 | Stop sequences `\nObservation:` | LangChain's ZeroShotAgent always passes them, whatever the authors set at the API level |
| A7 | Request-side pruning of the ReAct scratchpad above 60k characters (older observations cut to 400 chars, last 6 kept in full), `max_tokens` reduced to fit `CONTEXT_TOKENS` (65,536, env `IFCAGENT_CONTEXT_TOKENS`), one hard-prune retry (keep 2, 200 chars) on a context-length 400 | the served DeepSeek-V4-Flash has a 65,536-token context; the trajectory keeps everything |
| A8 | Invalid ReAct output -> LangChain's "Invalid Format: Missing 'Action:' after 'Thought:'" observation, counts as a step; unknown tool -> "not a valid tool, try one of [...]" | brief #12 |
| A9 | API error inside a sub-agent -> that entity reports "no final answer (api error ...)" and the task continues; API error in the main agent -> outcome `api` | keeps one flaky request from voiding a 20-entity executor call |
| A10 | Memory reset per task; 8 concurrent sub-agents; results in entity order | brief #14, #10 |
| A11 | No Mode II: > 20 entities -> first 20 + `mode2_not_reproduced` | task statement |
| A12 | `SaveFinalAnswer` ends the main agent's loop (no separate `Final Answer:` turn needed) | Fig. 4 ends with the FinalAnswer tool; without this DeepSeek printed the saved answer without the `Final Answer:` prefix until the 25-step cap (smoke run 2) |
| A13 | Each sub-agent sees only the Fig. 5 memory example whose tools it has (Example 1 needs QueryWithGlobalIdDict, Example 2 QueryWithStepIdDict) | Geometry / Quantity / Relation sub-agents otherwise call tools they are not bound to (55 wasted steps in smoke run 2) |

## Tests

* `test_tools.py` (container, no LLM): every tool once on a copy of `data/ifc-bench/4351/arc.ifc` (IFC2X3, feet), including miss paths,
  unknown tool, unknown attribute; `SetFieldByStepId` on a wall Name, an IfcLabel NominalValue, an IfcBoolean NominalValue and an
  extrusion Depth; `ExportModifiedIFC`; reopen and check all four edits and that the source file is untouched. PASSED.
* `test_mock.py` (container, scripted fake LLM): parser cases (plain JSON, fenced multi-line JSON, bare string input, Final Answer,
  invalid output, hallucinated Observation after the action), a full rename task through decomposition -> main agent -> BasicInfo
  executor -> sub-agent (memory miss -> GetInfoByGlobalId -> SetFieldByStepId) -> SaveFinalAnswer -> ExportModifiedIFC, the fallback
  write without export, and budget exhaustion (`outcome == "budget"`). PASSED.

## Smoke runs (DeepSeek-V4-Flash, http://gpu-node:30004/v1, vLLM, max_model_len 65536)

Columns: strict / intent / dependency / preservation; `exec/sub/calls` = executor calls / sub-agents / ReAct tool actions; `in` = prompt tokens.
Outputs: `outputs/<run>/dev/ifcagent_smoke` (prescribed command, first 8 held-out tasks, `--limit 8 --workers 4`), `ifcagent_smoke_sel`
(10 selected retrieval / tier-0 tasks), `ifcagent_smoke_fix` (4 tasks re-run after the fixes
below). No crashes, no `api`/`other` outcomes; validity was true and preservation true for every task (the framework never damaged a file).

### Run 1: first 8 held-out tasks (all tier-1 creation tasks) - 0/8 strict, as expected by construction

| task | template | S/I/D/P | outcome | exec/sub/calls | in | s | cause |
|---|---|---|---|---|---|---|---|
| 4351-C-column-0202-s0 | C-create-column | 0/0/1/1 | executed | 1/1/4 | 19k | 47 | no creation tool; sub-agent and main agent report "cannot create IfcColumn" |
| 4351-C-create-column-T-0473-s2 | C-create-column-T | 0/0/1/1 | executed | 5/15/41 | 161k | 134 | computed the space centroid (5 executor calls), then "cannot create" |
| 4351-C-create-door-P-0384-s2 | C-create-door-P | 0/0/1/1 | executed | 15/30/349 | 1715k | 442 | resolved the path reference (left-hand wall after entering) with 15 executor calls, then "cannot create" |
| 4351-C-create-door-box-D-0693-s0 | C-create-door-box-D | 0/0/0/1 | executed | 1/1/8 | 29k | 52 | "cannot create IfcOpeningElement / IfcDoor / IfcRelVoidsElement" |
| 4351-C-create-door-box-D-0697-s2 | C-create-door-box-D | 0/0/0/1 | executed | 1/1/17 | 61k | 47 | same |
| 4351-C-create-door-rel-D-0387-s2 | C-create-door-rel-D | 0/0/0/1 | budget | 6/6/46 | 477k | 97 | main agent spent its 25 steps re-trying executors that cannot create entities |
| 4351-C-create-door-rel-D-0404-s0 | C-create-door-rel-D | 0/0/0/1 | executed | 5/5/24 | 107k | 84 | "cannot create" after inspecting the wall |
| 4351-C-create-slab-T-0475-s0 | C-create-slab-T | 0/0/1/1 | budget | 5/15/85 | 657k | 98 | centroid / footprint computed, then 25-step cap while looking for a creation path |

### Run 2: 10 selected retrieval and tier-0 tasks (before the fixes) - 6/10 strict

| task | template | S/I/D/P | outcome | exec/sub/calls | in | s | cause of failure / note |
|---|---|---|---|---|---|---|---|
| fantasy_office_building_1-T0-name-0049-s2 | T0-set-name | 1/1/1/1 | executed | 1/1/4 | 16k | 22 | GetInfoByGlobalId -> SetFieldByStepId(Name) -> ExportModifiedIFC: the paper's edit path works |
| digital_hub-T0-fire-0007-s2 | T0-set-fire-rating | 0/0/1/1 | executed | 1/1/4 | 18k | 26 | Pset_WallCommon of that wall has no FireRating property; the framework cannot add a property (by construction) |
| fantasy_hotel_1-R-count-0162-s0 | R-count-doors | 0/0/1/1 | executed | 1/1/4 | 26k | 24 | OUR BUG: the 230-element containment observation was cut before the IfcDoor line; the sub-agent counted 4 doors from the truncated STEP expression (11 correct). Fixed (run 3) |
| digital_hub-R-area-0177-s0 | R-space-area | 1/1/1/1 | budget | 1/1/9 | 99k | 35 | answer correct (from Qto NetFloorArea), but after SaveFinalAnswer the main agent printed the answer without `Final Answer:` 18 times -> 25-step cap. Fixed (run 3) |
| 4351-R-adjacent-spaces-D-0517-s0 | R-adjacent-spaces-D | 1/1/1/1 | executed | 6/20/111 | 664k | 124 | correct via ExtractBoundariesByGlobalId / FindAdjacentSpacesByElement |
| samuel_macalister_sample_house-R-door-width-D-0687-s2 | R-door-width-D | 0/0/1/1 | executed | 4/4/45 | 190k | 127 | Geometry sub-agent hit 25 steps walking the door's B-rep; the main agent fell back to Pset_DoorCommon Width 0.8 m (geometric extent 0.96 m). Method limitation (hop-by-hop STEP traversal, arithmetic in the LLM) |
| digital_hub-R-qto-space-Height-D-0605-s2 | R-qto-space-Height-D | 1/1/1/1 | executed | 1/1/3 | 13k | 19 | from the quantity set |
| samuel_macalister_sample_house-R-wall-length-D-0569-s2 | R-wall-length-D | 1/1/1/1 | executed | 2/2/9 | 29k | 55 | axis polyline via GetInfoByStepId hops |
| 4351-R-north-0214-s0 | R-find-wall-north | 1/1/1/1 | executed | 10/79/724 | 2977k | 1010 | correct, but 79 sub-agents (Geometry executor over all storey walls twice, 21 and 42 entities -> `mode2_not_reproduced`), one sub-agent 400 context overflow |
| 4351-S-fire-0215-s0 | S-set-fire-rating-north-of | 0/0/1/1 | budget | 20/70/708 | 2648k | 1107 | found the north wall correctly (same route as R-north), but its Pset_WallCommon has no FireRating; the main agent then repeated the same executor call until the 20-call budget. By construction (no property creation) |

### Run 3: the 4 affected tasks after the fixes (terminal SaveFinalAnswer, compact containment output, per-toolset memory rules, context handling)

| task | template | S/I/D/P | outcome | exec/sub/calls | in | s |
|---|---|---|---|---|---|---|
| digital_hub-R-area-0177-s0 | R-space-area | 1/1/1/1 | executed | 1/1/4 | 13k | 19 |
| fantasy_hotel_1-R-count-0162-s0 | R-count-doors | 1/1/1/1 | executed | 1/1/2 | 13k | 20 |
| fantasy_office_building_1-T0-name-0049-s2 | T0-set-name | 1/1/1/1 | executed | 1/1/5 | 15k | 23 |
| samuel_macalister_sample_house-R-door-width-D-0687-s2 | R-door-width-D | 0/0/1/1 | executed | 3/3/35 | 129k | 93 |

Net on the selected set after the fixes: 8/10 strict (T0-set-name, R-count-doors, R-space-area, R-adjacent-spaces, R-qto-space-Height,
R-wall-length, R-find-wall-north pass; T0-set-fire-rating and S-set-fire-rating-north-of fail by construction (property absent);
R-door-width fails on geometry). Creation / deletion / move / geometry tasks fail by construction (no tool creates, deletes or moves entities).

Observed behaviour of DeepSeek-V4-Flash in the text-ReAct format: Action / Action Input JSON parsed cleanly in all 18 tasks (one
"Invalid Format" per task at most, except the SaveFinalAnswer loop above); the model sometimes writes "Thought:" twice or omits it, which
the parser tolerates. Sub-agents of executors without GetInfoByGlobalId / QueryWithGlobalIdDict (Geometry, Quantity, Relation) tried those
tools 55 times in run 2 (the Fig. 5 rules name them); run 3 shows only the applicable Fig. 5 example to each executor.

## Open issues

1. **Cost on reference-resolution tasks.** Geometric references ("the wall north of X", "the left-hand wall after entering") make the
   main agent fan out Geometry sub-agents over whole storeys: 0.7-3.0 M prompt tokens and 7-18 minutes per task (R-north, S-fire,
   C-create-door-P). A full held-out run (1500 tasks, 4 workers) would take days and ~1-2 G tokens; the cost is intrinsic to Mode I
   (one ReAct agent per entity, hop-by-hop STEP traversal) and the paper's Mode II would not change the LLM-per-entity exploration for
   geometry. Consider `--limit`/sharding across several nodes, or sampling the held-out set for this row.
2. **Mode II not reproduced** (ObserverAgent / BulkExecutor / AggregationAgent / PythonExecutor): executor calls with > 20 entities process
   the first 20 and log `mode2_not_reproduced` (happened 3 times in the smoke runs, always Geometry over a storey's walls). This
   under-represents the method on "all walls of storey S" style tasks.
3. **Repeated identical actions** are not detected (LangChain and the paper have no loop guard): S-fire spent 10 of its 20 executor calls
   on the same request. Faithful, but it inflates `budget` outcomes and cost.
4. **Context length.** The served DeepSeek-V4-Flash has a 65,536-token context; sub-agents that follow deep STEP chains reach it (2 HTTP 400s
   in run 2). Now handled by earlier request-side pruning, a dynamic `max_tokens`, and one hard-prune retry; `IFCAGENT_CONTEXT_TOKENS`
   overrides the limit for other servers. Thinking-mode output (`<think>`) is stripped if the server does not separate it.
5. **Property creation** is impossible by construction, so every `T0-set-fire-rating` / `S-set-fire-rating-*` task whose wall has no
   FireRating in Pset_WallCommon fails (both such tasks in the smoke set). Check how many held-out property tasks have the property
   present before interpreting the row.
6. **Units.** Values are reported in model units with the metre factor in the context line; the sub-agents converted correctly in the
   smoke runs (feet in 4351, millimetres in samuel_macalister), but the conversion is done by the LLM, not by a tool (as in the paper).
7. **Unverified against the paper's own queries**: the Appendix C trace (wall layer thickness -> 0.3 on AC20-FZK-Haus) was not replayed
   because that model is not on disk; the tool chain it uses (ExtractRelationsByGlobalId -> GetInfoByStepId -> SetFieldByStepId
   LayerThickness -> ExportModifiedIFC) is covered by the tool-level test on arc.ifc (Depth / NominalValue / Name edits).
