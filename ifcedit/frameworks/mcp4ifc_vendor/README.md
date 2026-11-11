# MCP4IFC vendored port

Source: **Show2Instruct/ifc-bonsai-mcp** (package `ifc-bonsai-mcp` 0.1.0), `main` @ `6215493` (2025), MIT licence
(`LICENSE`, "Copyright (c) 2025 Bharathi Kannan N."). Paper: Nithyanantham et al., "MCP4IFC: IFC-Based Building Design using
Large Language Models", arXiv 2511.05533. Ported for the IfcEdit comparison rows.

The authors' system is `AI client (Claude Desktop) -> FastMCP server (src/blender_mcp) -> TCP socket -> Blender add-on
(blender_addon, inside Blender 4.4 + Bonsai 0.8.2)`; every tool body forwards a command to the add-on, where the IFC work is done
with `ifcopenshell.api`, and Bonsai is only the wrapper that holds the open file (`tool.Ifc.get()`), the active storey and the
save/reload. This directory is the "Route 2" port described in the re-implementation brief: the Bonsai wrapper is replaced by
plain IfcOpenShell (0.8.5, the project's `.venvs/ifc-audit`), the socket by a subprocess pipe, FastMCP by a registry + pydantic
schemas. Tool names, docstrings, parameters and all `ifcopenshell.api` logic are the authors' verbatim.

## Files

| Path | Origin | State |
|---|---|---|
| `LICENSE` | `SRC/LICENSE` | identical |
| `blender_addon/api/{wall,slab,door,window,roof,stairs,style,root,feature,mesh_ifc,mesh_trimesh,system}.py` | `SRC/blender_addon/api/` | **identical** (byte-for-byte) |
| `blender_addon/api/__init__.py` | same | 1 line: the import of the two dropped scene functions removed |
| `blender_addon/api/code.py` (sandbox) | same | 1 hunk: the unused `try: from ...src.blender_mcp.mcp_instance import mcp` removed; blacklist, builtins whitelist, `safe_import`, 60 s SIGALRM `execution_timeout`, result format unchanged |
| `blender_addon/api/ifc_utils.py` | same | the four Bonsai-bound helpers (original L15-66) replaced: `IfcSession` (ifcopenshell.open of the task copy) + `bind_session`/`current_session`; `get_ifc_file()` = the session file; `get_default_container()` = first `IfcBuildingStorey`, else `IfcBuilding`, else `IfcSite` (Bonsai's was the storey active in the GUI); `save_and_load_ifc()` = `ifc.write(path)` in place (no scene reload); `get_selected_guids()` = `[]`. Everything after the marker `END OF ROUTE-2 PORT` (contexts, unit scale, matrices, polylines) is unchanged |
| `blender_addon/api/scene.py` | same | header/imports rewritten (no `bpy`, `mathutils`, `bonsai.tool`); `get_scene_info` reimplemented over `ifc.by_type("IfcProduct")` with the same fields (`name` = `"<IfcClass>/<Name>"` as Bonsai names objects, `type` MESH/EMPTY by representation, `location` = world placement in metres, `visible` True, `selected` False, `guid`, `ifc_class`; `include_transform` / `include_bbox` from the placement matrix and `ifcopenshell.geom` world AABB; `detailed` = `get_object_info(detailed=True)`); `get_blender_object_info`, `get_selected_objects` removed; `get_object_info` and `get_ifc_scene_overview` unchanged except `tool.Ifc.get()` -> `get_ifc_file()`, the Blender-object `geometry` block replaced by placement/bbox from the IFC, `use_selection` / `include_selection_summary` answer "No objects selected" |
| `blender_addon/commands.py` | same | `import bpy`, `from bonsai import tool`, `from .scene_analysis.scene_analysis import *` removed (3 lines); `execute_command` unchanged |
| `blender_addon/__init__.py` | new | plain package marker (the original registered the Blender add-on: bl_info, UI, socket server `core.py`) |
| `blender_mcp/api_tools.py` | `SRC/src/blender_mcp/mcp_functions/api_tools.py` | import block replaced by `from .runtime import logger, get_blender_connection, mcp`; `ctx: Context` removed from every signature (FastMCP already hid it from the schema; docstrings untouched); 3 tools deleted: `execute_blender_code`, `get_blender_object_info`, `get_selected_objects`. 142 diff lines in total, all of these kinds |
| `blender_mcp/mcp_instance.py` | `SRC/src/blender_mcp/mcp_instance.py` | the `instructions` string kept as `INSTRUCTIONS`; no FastMCP |
| `blender_mcp/prompts.py` | `SRC/src/blender_mcp/mcp_functions/prompts.py` | text verbatim, decorator removed |
| `blender_mcp/runtime.py` | new | `mcp.tool()` registry (stand-in for FastMCP), thread-local `get_blender_connection()`, `WorkerConnection` (one worker subprocess per task, JSON lines; `send_command` returns `{"error": msg}` on command errors like `server.py`; 30 s per command = `core.py` L153, 60 s + 10 s margin for `execute_ifc_code` so the sandbox's own SIGALRM message comes back; a timed-out command returns `"Command execution timed out"` (core.py's text) while the worker finishes it and later commands queue behind it, late answers are dropped by id) |
| `blender_mcp/schema.py` | new | OpenAI tool schemas built as FastMCP did (mcp 1.14): name, description = raw `fn.__doc__`, parameters = `pydantic.create_model("<name>Arguments", ...).model_json_schema()`; argument validation = JSON-string pre-parse for non-str fields + `model_validate` |
| `worker.py` | new | the Blender-process stand-in: imports `blender_addon` under its original name (so the sandbox whitelist `blender_addon.api.*` and the ICL prelude's `from blender_addon.api.ifc_utils import ...` keep working), dispatches `commands.execute_command` on its main thread (SIGALRM works), `__open__` / `__write__` / `__quit__` |

Not vendored: `analysis_tools.py` (2 Blender-viewport screenshot tools), `rag_tools.py` (8 ChromaDB/sentence-transformers tools; no
embedding model offline), `server.py`/`core.py`/`ui.py`/`scene_analysis/` (transport and GUI), `scripts/`, `docs/`, `experiments/`.

## Final tool list (39)

`list_blender_commands execute_ifc_code_tool get_scene_info get_object_info get_ifc_scene_overview create_wall create_two_point_wall
create_polyline_walls update_wall get_wall_properties get_roof_types create_roof update_roof delete_roof create_slab update_slab
get_slab_properties get_door_operation_types create_door update_door get_door_properties get_window_partition_types create_window
update_window create_trimesh_ifc get_window_properties get_stairs_types create_stairs update_stairs delete_stairs
create_surface_style create_pbr_style apply_style_to_object list_styles update_style remove_style create_mesh_ifc list_ifc_entities
get_trimesh_examples`

= the 42 live tools of `api_tools.py` on `main` minus `execute_blender_code`, `get_blender_object_info`, `get_selected_objects`
(the 2 screenshot and 8 RAG tools were in other modules). The serialised OpenAI `tools` list is 113,684 characters; the server
counted **28,958 prompt tokens** for the first request of a held-out task (265 without `tools`, so the 39 schemas are ~28.7k tokens
under DeepSeek-V4's tokenizer; the paper quotes "around 40k tokens" for its 52 tools on Claude).

Process and limits: one `worker.py` subprocess per task (IfcOpenShell is not thread-safe and a segfault in geometry code must not
take the runner down); 30 s per predefined tool, 60 s sandbox limit for `execute_ifc_code_tool`; `trimesh` is **not** installed in
`.venvs/ifc-audit`, so `create_trimesh_ifc` answers "trimesh not available" (the module's own guard); `mesh_ifc.apply_solidify` is a
no-op without `bpy` (the module's own guard); `get_default_container()` is the file's first storey.

## The loop (`ifcedit/frameworks/mcp4ifc.py`)

The repository has no client, system prompt, step cap or retry logic (the paper used Claude Desktop's generic tool-use loop), so the
loop is ours and minimal. System prompt = three parts joined by blank lines:

1. FastMCP `instructions` verbatim: `Blender integration through the Model Context Protocol`
2. `prompts.ifc_building_element_creation_strategy()` verbatim (incl. indentation and the `#ToDo:` item):
   ```
       When creating IFC building elements in Bonsai (formerly BlenderBIM), follow these guidelines:

       1. IFC Structure and Hierarchy:
          - Respect the IFC hierarchy: Project → Site → Building → Building Story → Building Elements
          - Every element must belong to a proper container (typically a Building Story)
          - Use proper IFC entity types (IfcWall, IfcSlab, IfcDoor, etc.) that match the element's function

       2. #ToDo:

       Following these guidelines will ensure a well-structured, standards-compliant IFC model.
   ```
3. our framing sentence: `The IFC model is loaded. Apply the instruction to it using the tools; predefined tools save the file; if
   you edit with execute_ifc_code_tool, the model in memory is written at the end. When finished, answer in text.`

User message = the task prompt. Then: `chat.completions.create(model, messages, tools=<39 schemas>, tool_choice="auto",
temperature, max_tokens)`; every tool call of the reply is executed (`call_tool`: unknown tool / bad JSON / pydantic validation
error -> `Error executing tool <name>: ...` as FastMCP would answer) and appended as a `tool` message; one assistant turn with tool
calls = one round; the episode ends when the assistant answers without a tool call (`executed`), after `budget` rounds (`budget`),
when the worker process dies (`crash`), or on an API error after 3 retries 30 s apart (`api`; a context-length rejection is
reported immediately as `context_overflow: ...` without pruning tools). DeepSeek conventions of `ifcedit.agent` are reused
(`reasoning_content` / `</think>` split, DSML text-tool-call fallback when the server parser misses a call). After the loop the
in-memory file is always written to `ifc_path` (`__write__`) - the generous reading of the original, where only predefined tools
and an explicit `save_and_load_ifc()` persisted edits - and `output_ifc = ifc_path`. Returned dict: the interface fields plus
`saves` (number of writes incl. the final one), `timeouts`, `first_prompt_tokens`, `seconds`.

## Tests

(a) Tool level, no LLM, in a Python 3.12 environment, on a copy of `data/ifc-bench/4351/arc.ifc`
(IFC2X3, 441 products): `list_blender_commands`, `get_ifc_scene_overview`, `get_scene_info` (plain / MESH + bbox + transform),
`get_object_info` (detailed, multi-GUID, bad GUID), `get_wall_properties`, `create_wall`, `create_slab`, `execute_ifc_code_tool`
(prelude import + `save_and_load_ifc()`, blacklisted `import os`, NameError traceback, infinite loop -> "Code execution timed out
after 60 seconds" from the sandbox), `get_slab_properties` bad GUID, pydantic validation error, final write; the written file
reopens in a fresh process with the new wall and slab (44 walls).

(b) Smoke run, `python -m ifcedit.run_external --framework mcp4ifc --model deepseek-v4-flash --base-url http://gpu-node:30004/v1
--tasks data/ifceditbench/tasks.jsonl --out outputs/<run>/dev/mcp4ifc_smoke2 --limit 8 --workers 4` (vLLM,
`max_model_len` 65,536; budget 20, max_tokens 16,384, temperature 0). `mcp4ifc_smoke` is the first attempt before two loop fixes
(stored arguments not normalised -> DeepSeek's `{"arguments": ...}` wrap nested deeper every turn; `max_tokens` not reduced on a
context rejection) and the `detailed` block change; keep it only as a record.

| task | strict | intent | dep | pres | valid | calls | tokens in/out | outcome | note |
|---|---|---|---|---|---|---|---|---|---|
| 4351-C-column-0202-s0 | **True** | True | True | True | True | 7 | 257k / 2.5k | executed | column via `create_mesh_ifc` after `create_trimesh_ifc` failed (no trimesh) |
| 4351-C-create-column-T-0473-s2 | False | False | True | True | True | 14 | 657k / 16.4k | executed | column created at the wrong centre ([-18.64, 43.32] vs [-5.81, 13.02]) and in feet-as-metres (size 1.312) - model error; 1 malformed call |
| 4351-C-create-slab-T-0475-s0 | False | False | True | True | True | 18 | 620k / 18.0k | api | `create_slab` done with a wrong footprint (sym. diff. 391 m2), then `get_scene_info` x2 -> context_overflow |
| 4351-C-create-door-P-0384-s2 | False | False | True | True | True | 3 | 92k / 0.5k | api | `get_scene_info(obj_type=MESH, bbox, transform, detailed)` = 1.29 MB -> context_overflow |
| 4351-C-create-door-box-D-0693-s0 | False | False | False | True | True | 2 | 29k / 0.1k | api | `get_scene_info(detailed=true)` = 0.99 MB (443 products) -> context_overflow |
| 4351-C-create-door-box-D-0697-s2 | False | False | False | True | True | 2 | 29k / 0.1k | api | same |
| 4351-C-create-door-rel-D-0387-s2 | False | False | False | True | True | 2 | 29k / 0.1k | api | same |
| 4351-C-create-door-rel-D-0404-s0 | False | False | False | True | True | 2 | 29k / 0.1k | api | same |

1/8 strict; 2 executed, 6 `api` = `context_overflow` (after halving `max_tokens` down to 1,024 the prompt alone still exceeds
65,536 tokens). No crash, no worker timeout, no preservation/validity failure; every error of the `api` rows is the vLLM 400
"maximum context length is 65536 tokens ... your prompt contains at least 64513 input tokens". Worker-side errors seen in tool
outputs were all raised by the model's own code (IndexError, wrong attribute names), none by the port.

## Open issues

1. **Context**: 29k tokens of tool schemas + DeepSeek-V4-Flash's 64k window leave ~20k tokens for the whole episode at
   `max_tokens` 16,384 (the loop halves `max_tokens` on a rejection, down to 1,024, before giving up with `context_overflow`). The
   model's first move is often `get_scene_info(detailed=true)` without `limit`, whose answer (0.99-1.29 MB for the 443 products of
   4351/arc.ifc) exceeds any context - the original add-on would have returned a listing of the same order (Blender object info
   per object). The loop does not truncate tool outputs or prune tools (the brief's instruction); whether to cap tool-output size
   (as some MCP clients do) or serve a longer context is a decision for the comparison protocol, not taken here.
2. `trimesh` is not installed in `.venvs/ifc-audit`: `create_trimesh_ifc` returns "trimesh not available" (the module's own guard);
   the model then falls back to `create_mesh_ifc` or code. Install only if the comparison should include that tool.
3. Malformed tool calls: DeepSeek sometimes wraps the arguments (`{"arguments": "<json>"}`); the loop unwraps up to 6 levels and
   stores the normalised form. When the inner string is itself invalid JSON the call is answered with "Error executing tool ...:
   arguments must be a JSON object" (1 occurrence in 8 tasks). Fenced code in plain text is never executed.
4. Units: the predefined tools take metres (as in the original, where Bonsai is in SI) but `execute_ifc_code_tool` works in file
   units (feet for 4351); the model mixed them in 2 of 3 geometry edits. Framework behaviour, not a port issue.
5. `get_default_container()` = first `IfcBuildingStorey` (Bonsai: the storey active in the GUI); predefined create tools use it
   when the model gives no container, so elements may land on the wrong storey - framework behaviour under Route 2.
6. The summary written by `run_external` does not persist the extra fields of the returned dict (`saves`, `timeouts`,
   `context_clamps`, `max_tokens_final`, `first_prompt_tokens`); they are available only from a direct `run()` call.
7. `get_scene_info` with `detailed` / `include_bbox` / `include_transform` runs `ifcopenshell.geom.create_shape` per product
   (~28 s for all 443 products of arc.ifc, close to the 30 s command timeout; bigger models will time out, which returns
   "Command execution timed out" like the original add-on).
