"""IFC-Agent (Gao, Hu, Chai, Weng, Li, 2026, Automation in Construction 186:106888) - Mode I reproduction.

run(prompt, ifc_path, workdir, client, model, budget, max_tokens, temperature) -> dict per ifcedit.frameworks.__init__.

What is reproduced (paper Sec. 3-5, Table 2, Fig. 4/5/8/9, Appendix C/D; details in IFCAGENT.md):
  Main Agent  : LoadIFCFile -> RewriteIfNeeded -> QueryMemory -> TaskDecompositionAgent (LLM, Fig. 9 principles) -> a text-ReAct loop
                (max 25 steps) that, per subtask, resolves the target entities (GlobalIds in the query, GetGlobalIdsByAttributeValue, or
                the previous subtask's results = "working scope"), picks ONE of the 7 Sequential*Executors with the Fig. 8 table, and finally
                calls ExportModifiedIFC (for modifications) and SaveFinalAnswer, which ends the task (the FinalAnswer tool of Fig. 4).
  Executors   : SequentialBasicinfo / Geometry / Quantity / Property / Relation / SpaceBoundary / SpatialContainment - each launches one
                text-ReAct sub-agent per entity (8 concurrent, results in entity order) with the executor's FIXED toolset (Table 2);
                every sub-agent has at most 25 reasoning steps.
  ReAct       : LangChain zero-shot format (Thought / Action / Action Input JSON / Observation / Final Answer), own parser, stop sequences
                "\\nObservation:" as LangChain's ZeroShotAgent uses; no native function calling, temperature 0, top_p 1, penalties 0.
  Tools       : ifcedit.frameworks.ifcagent_tools (raw STEP expressions in observations; SetFieldByStepId + ExportModifiedIFC are the only
                ways to change the model).
NOT reproduced: Mode II (ObserverAgent / BulkExecutor / AggregationAgent + PythonExecutor for > 20 entities). When an executor call lists
more than 20 GlobalIds, the first 20 are processed in Mode I and the observation / trajectory carry the note "mode2_not_reproduced".
CustomMemory is reset per task (fresh instance); no cross-task reuse.

Prompts: the paper's printed prompts are used verbatim (Fig. 5 memory rules, Fig. 8 executor table, Fig. 9 decomposition principles,
Appendix D placement procedure, Fig. 2 GetInfoByGlobalId description, the ISARC-2025 instruction lines a-e). Everything the paper does
not print (system prompts, decomposition wrapper + exemplars, executor / main-agent tool descriptions, the entity line of a sub-agent's
question) is written here and marked OURS.

Budget: `budget` (runner default 20) caps the number of executor calls of the main agent; outcome "budget" when exceeded or when the
main agent hits its 25-step cap without a final answer. If ExportModifiedIFC was never called but SetFieldByStepId changed something,
the in-memory model is written to ifc_path (generous reading, as for the MCP4IFC port) and recorded as "fallback_write" in messages.
"""
from __future__ import annotations

import ast
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from ifcedit.frameworks.ifcagent_tools import IfcAgentTools

MAX_STEPS = 25            # paper: "maximum of 25 reasoning steps" per ReAct agent
CONTEXT_TOKENS = int(os.environ.get("IFCAGENT_CONTEXT_TOKENS", "65536"))   # OURS: the served DeepSeek-V4-Flash (vLLM) has max_model_len 65536
PRUNE_CHARS = 60_000      # OURS: beyond this many scratchpad characters the older observations are shortened in the REQUEST (the trajectory keeps them)
PRUNE_KEEP = 6
PRUNE_HEAD = 400
CHARS_PER_TOKEN = 2.5     # conservative estimate for STEP-heavy text
MODE_I_LIMIT = 20         # paper: threshold of 20 entities for adaptive mode switching
SUBAGENT_WORKERS = 8      # brief #10
STOP = ["\nObservation:", "\n\tObservation:"]     # LangChain ZeroShotAgent._stop
API_RETRIES = 2
RETRY_DELAY_S = 20.0

EXECUTOR_TOOLS = {   # Table 2 of the paper (static agent-tool binding)
    "SequentialBasicinfoExecutor": ["GetInfoByGlobalId", "QueryWithGlobalIdDict", "GetInfoByStepId", "QueryWithStepIdDict", "SetFieldByStepId"],
    "SequentialGeometryExecutor": ["GetInfoByStepId", "QueryWithStepIdDict", "ComputePlacementMatrixFromStepId", "SetFieldByStepId"],
    "SequentialQuantityExecutor": ["ExtractQuantitySetByGlobalId", "QueryWithGlobalIdDict", "SetFieldByStepId"],
    "SequentialPropertyExecutor": ["ExtractPropertySetByGlobalId", "ExtractRelationsByGlobalId", "GetInfoByStepId", "QueryWithStepIdDict", "SetFieldByStepId"],
    "SequentialRelationExecutor": ["ExtractRelationsByGlobalId", "ExtractOpeningsByGlobalId", "ExtractConnectivityByGlobalId", "ExtractAssignmentByGlobalId",
                                   "DescribeRelationStep", "DecideRelationNextHop"],
    "SequentialSpaceBoundaryExecutor": ["ExtractBoundariesByGlobalId", "FindAdjacentSpacesByElement", "DescribeRelationStep", "DecideRelationNextHop",
                                        "GetInfoByStepId", "QueryWithStepIdDict"],
    "SequentialSpatialContainmentExecutor": ["ExtractContainsElementsByGlobalId", "GetInfoByStepId", "QueryWithStepIdDict", "SetFieldByStepId"],
}
MAIN_TOOLS = ["LoadIFCFile", "RewriteIfNeeded", "QueryMemory", "TaskDecompositionAgent", "GetGlobalIdsByAttributeValue"] + list(EXECUTOR_TOOLS) + \
             ["SaveFinalAnswer", "ExportModifiedIFC"]

# ---------------------------------------------------------------------------------------------------------------------------- prompts
# LangChain's classic zero-shot ReAct template (hwchase17/react), verbatim; the paper's loop is LangChain ReAct with text parsing.
REACT_TEMPLATE = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

# Fig. 5 of the paper, verbatim (Example 1 is shown to sub-agents whose toolset has QueryWithGlobalIdDict, Example 2 to those with QueryWithStepIdDict).
MEMORY_RULES_1 = """Example 1 - GlobalIdDict
1. Always begin by checking if 'BasicInfo' exists in globalid_dict:
   Action: QueryWithGlobalIdDict
   Action Input: {"global_id": "<id>", "field": "BasicInfo"}
2. If not found, run:
   Action: GetInfoByGlobalId
   Action Input: {"global_id": "<id>"}
3. Then repeat step 1 to extract the required field from BasicInfo."""
MEMORY_RULES_2 = """Example 2 - StepIdDict
1. Always begin by checking if the field exists in stepid_dict:
   Action: QueryWithStepIdDict
   Action Input: {"step_ids": ["#xxxx"], "field": "Depth"}
2. If not found, run:
   Action: GetInfoByStepId
   Action Input: {"step_ids": ["#xxxx"]}
3. Then repeat step 1 to extract the field."""

# Fig. 8 of the paper, verbatim.
EXECUTOR_TABLE = """### Choose the Correct Sequential Executor
| Executor Tool Name                    | When Question Mentions ...
| SequentialBasicinfoExecutor           | basic info, type, name, when others not match
| SequentialGeometryExecutor            | XDim, YDim, depth, thickness, placement, position, location, coordinates
| SequentialQuantityExecutor            | volume, area, width, length, height
| SequentialPropertyExecutor            | room name or number, fire rating, thermal transmittance
| SequentialRelationExecutor            | openings, coverings, assignments, associations
| SequentialSpaceBoundaryExecutor       | room, walls, doors, bounding elements, spatial connections
| SequentialSpatialContainmentExecutor  | storeys, spatial structures, contained elements
Use semantic reasoning to pick one executor only."""

# Fig. 9 of the paper, verbatim.
DECOMPOSITION_PRINCIPLES = """### Task
1. Analyze the query complexity:
   - Simple query: Single entity type + Single field extraction -> Return 1 subtask (unchanged).
   - Compound query: Multiple steps (filter + extract + compute) -> Decompose into sequential subtasks.
2. Each subtask must be:
   - Independently executable by GlobalId-level executors.
   - Clear and specific (no ambiguity).
   - Ordered logically (filtering before computation, extraction before aggregation).
3. Preserve dependency order:
   - If subtask B depends on results from subtask A, list A before B.
4. Do NOT invent subtasks that are not implied by the query."""

# Appendix D of the paper, verbatim (given to the Geometry sub-agents).
PLACEMENT_PROMPT = """If the question asks about world or absolute placement:
Step A: Build the placement hierarchy
- Start from ObjectPlacement in BasicInfo -> get the IfcLocalPlacement STEP ID
- Use GetInfoByStepId to get its details (PlacementRelTo, RelativePlacement)
- If PlacementRelTo exists, recursively follow the chain until PlacementRelTo = None
- Record the hierarchy: [root_placement_id, parent_placement_id, ..., current_placement_id]
Step B: Collect all transformation data
For each IfcLocalPlacement in the hierarchy (from root to current):
- Use GetInfoByStepId to get the RelativePlacement (IfcAxis2Placement3D) STEP ID
- Use GetInfoByStepId to get Location, Axis, RefDirection references
- Call ComputePlacementMatrixFromStepId to get the 4x4 transformation matrix
Step C: Matrix multiplication
After collecting all matrices [M0, M1, M2, ..., Mn] from root to current:
1. Multiply matrices step by step: Result = M0; Result = Result x M1; ...; Final_Matrix = M0 x M1 x ... x Mn
2. Apply to origin point: Point = [0, 0, 0, 1]; Absolute_Position = Final_Matrix x Point; Extract [x, y, z]
3. Show your work: Display each matrix clearly; Show the step-by-step multiplication process; Verify the final coordinates
CRITICAL REQUIREMENTS:
ALWAYS call ComputePlacementMatrixFromStepId for each IfcLocalPlacement in the hierarchy
ALWAYS show the actual matrix multiplication calculations step by step
ALWAYS multiply ALL matrices in the chain, don't skip any
ALWAYS extract coordinates from the final matrix-vector multiplication
Do NOT assume any matrix is identity without verification
Do NOT skip intermediate matrices
Do NOT just use the last matrix's origin as the absolute position"""

# ISARC 2025 (Du, Gao, Tong, Weng) Fig. 2 instruction lines, verbatim; the paper cites it as the parent design.
ISARC_INSTRUCTIONS = """a) Check for required data before each task.
b) Reuse available data from memory.
c) Use the provided tools and update results.
d) Transit between tasks with clear statements.
e) Provide reasoning and results for each task before proceeding."""

# OURS: sub-agent system prompt (the paper prints no system prompt).
SUBAGENT_SYSTEM = """You are a sub-agent of IFC-Agent. You work on ONE IFC entity with a fixed set of tools over the IFC model (IfcOpenShell). Follow the ReAct format exactly: write "Thought:", then "Action:" with exactly one tool name, then "Action Input:" with one JSON object, then stop and wait for the Observation. Give "Final Answer:" only when the subtask is done for this entity; the final answer must contain the entity's GlobalId and the values found or the change made.
Instructions:
{isarc}
Memory access rules (only the tools listed below exist for you):
{memory}
Schema path preferences:
- Depth of an element: prefer the geometric definition (IfcShapeRepresentation -> IfcExtrudedAreaSolid -> Depth) over the approximate bounding dimension (IfcShapeRepresentation -> IfcBoundingBox -> ZDim).
- Wall thickness: geometry path IfcExtrudedAreaSolid -> SweptArea -> IfcRectangleProfileDef -> YDim, or material path IfcMaterialLayerSetUsage -> ForLayerSet -> MaterialLayers[n] -> LayerThickness.
Observations show STEP entities as '#id=IFCTYPE(...)': read the referenced STEP ids from them and follow them with GetInfoByStepId. Values are in the model's length unit (see the context line). Modifying the model is only possible with SetFieldByStepId on the STEP entity that holds the field (an attribute of the element, the NominalValue of an IfcPropertySingleValue, a LayerThickness, a Depth, ...); do not export, the main agent exports. If a tool is missing for the subtask, say so in the Final Answer.
{placement}"""

# OURS: main-agent system prompt (the paper prints no system prompt; Fig. 8 is verbatim inside).
MAIN_SYSTEM = """You are the Main Agent of IFC-Agent, the central controller for one task on an IFC model. The IFC file is loaded, the query was rewritten, the memory was checked and the task was decomposed into sequential subtasks (see the trace). Follow the ReAct format exactly: "Thought:", "Action:" with one tool name, "Action Input:" with one JSON object, then stop and wait for the Observation.
For each subtask in order: (1) identify the target entities: GlobalIds given in the task, found with GetGlobalIdsByAttributeValue, or taken from the previous subtask's results (the output of a subtask is the working scope of the next); (2) choose ONE sequential executor with the table below and call it with the subtask text and the list of target GlobalIds; it runs one sub-agent per entity with that executor's tools and returns their results. Pass the facts the sub-agents need (values from the task, results of earlier subtasks) in the "context" field. After the last subtask: if the model was modified, call ExportModifiedIFC with "export/updated_model.ifc"; then call SaveFinalAnswer with the complete answer (for a question: the answer in exactly the form the task asks for; for a modification: what was changed) - SaveFinalAnswer ends the task. Do not answer from your own knowledge; everything must come from the executors' results.
{table}
Each executor call handles at most {limit} entities (Mode I)."""

# OURS: TaskDecompositionAgent prompt around the verbatim Fig. 9 principles; the exemplar is the one the paper names (Sec. 3.3).
DECOMPOSITION_SYSTEM = """You are the TaskDecompositionAgent of IFC-Agent. Decompose the user's query about an IFC model into subtasks that GlobalId-level executors run one after another.
{principles}
Examples:
"Find all walls and calculate their total area" -> {{"subtasks": ["Find all walls", "Find the area of each wall", "Calculate the total area of walls"]}}
"Rename the wall with GlobalId X: set its Name to 'W1'" -> {{"subtasks": ["Set the Name attribute of the wall with GlobalId X to 'W1'"]}}
Output exactly one JSON object {{"subtasks": ["<subtask 1>", "<subtask 2>", ...]}} and nothing else."""

# OURS: tool descriptions of the main agent's control tools (Fig. 2 style).
MAIN_DESCRIPTIONS = {
    "TaskDecompositionAgent": "Decompose the query into a JSON list of sequential subtasks.\nInput must be a JSON object with keys:\n- 'question': string\nOutput: {\"subtasks\": [...]}.",
}
for _name, _tools in EXECUTOR_TOOLS.items():
    MAIN_DESCRIPTIONS[_name] = (f"Run the subtask on each listed entity with one sub-agent per GlobalId; the sub-agents use the tools [{', '.join(_tools)}].\n"
                                "Input must be a JSON object with keys:\n- 'subtask': string (what to do or find for each entity)\n- 'global_ids': list of GlobalId strings (the target entities)\n"
                                "- 'context': string (optional; facts the sub-agents need, e.g. values given in the task or results of previous subtasks)\n"
                                "Output: the result of every sub-agent, keyed by GlobalId.")


# ---------------------------------------------------------------------------------------------------------------------------- LLM access
class ContextOverflow(Exception):
    """The server rejected the request for exceeding its context length."""


class _LLM:
    """One chat endpoint with usage accounting and a flat exchange log shared by all agents of one task."""

    def __init__(self, client, model: str, max_tokens: int, temperature: float):
        self.client, self.model, self.max_tokens, self.temperature = client, model, max_tokens, temperature
        self.tokens_in = 0; self.tokens_out = 0; self.calls = 0
        self.lock = threading.Lock()
        self.messages: list[dict] = []

    def log(self, agent: str, role: str, content: str, **extra):
        with self.lock:
            self.messages.append({"agent": agent, "role": role, "content": content, **extra})

    def chat(self, agent: str, system: str, user: str) -> str:
        est = int((len(system) + len(user)) / CHARS_PER_TOKEN) + 64
        max_tokens = max(1024, min(self.max_tokens, CONTEXT_TOKENS - est))
        for attempt in range(API_RETRIES + 1):
            try:
                resp = self.client.chat.completions.create(model=self.model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                                                           temperature=self.temperature, top_p=1.0, max_tokens=max_tokens, stop=STOP,
                                                           presence_penalty=0.0, frequency_penalty=0.0)
                break
            except Exception as e:  # noqa: BLE001
                if "maximum context length" in str(e) or "context length" in str(e):
                    raise ContextOverflow(str(e)[:300])
                if attempt >= API_RETRIES:
                    raise
                time.sleep(RETRY_DELAY_S)
        u = getattr(resp, "usage", None)
        with self.lock:
            self.calls += 1
            if u:
                self.tokens_in += int(getattr(u, "prompt_tokens", 0) or 0); self.tokens_out += int(getattr(u, "completion_tokens", 0) or 0)
        msg = resp.choices[0].message
        content = msg.content or ""
        rc = getattr(msg, "reasoning_content", None) or (getattr(msg, "model_extra", None) or {}).get("reasoning_content")
        if "</think>" in content:       # thinking served without a reasoning parser
            rc2, content = content.rsplit("</think>", 1)
            rc = (rc or "") + rc2.replace("<think>", "", 1)
        self.log(agent, "assistant", content, **({"reasoning_content": rc[:4000]} if rc else {}))
        return content.strip()


# ---------------------------------------------------------------------------------------------------------------------------- ReAct parsing
_ACTION_RE = re.compile(r"Action\s*\d*\s*:[ \t]*(.*?)\s*Action\s*\d*\s*Input\s*\d*\s*:[ \t]*(.*)", re.S)     # LangChain MRKL regex
_FINAL = "Final Answer:"


def _clean_name(s: str) -> str:
    return s.strip().strip("`*\"' ").split("\n")[0].strip().strip("`*\"' ")


def _parse_input(raw: str):
    """Action Input -> dict (JSON, Python literal, or the first balanced {...}); a bare string is returned as str."""
    t = raw.strip()
    t = re.sub(r"^```(?:json|python)?\s*|\s*```$", "", t, flags=re.S).strip()
    if t.startswith("Observation"):
        t = ""
    for cand in (t, t.split("\nObservation")[0].strip(), t.split("\nThought")[0].strip()):
        if not cand:
            continue
        try:
            v = json.loads(cand)
            if isinstance(v, dict):
                return v
        except json.JSONDecodeError:
            pass
        try:
            v = ast.literal_eval(cand)
            if isinstance(v, dict):
                return v
        except (ValueError, SyntaxError):
            pass
    depth, start = 0, None
    for i, ch in enumerate(t):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                for parser in (json.loads, ast.literal_eval):
                    try:
                        v = parser(t[start:i + 1])
                        if isinstance(v, dict):
                            return v
                    except Exception:  # noqa: BLE001
                        pass
                break
    return t.strip("\"' ")


def parse_react(text: str) -> tuple[str, str | None, object]:
    """Returns ("final", answer, None) | ("action", tool, args) | ("invalid", message, None)."""
    m = _ACTION_RE.search(text)
    fi = text.find(_FINAL)
    if m and (fi < 0 or m.start() < fi):
        tool = _clean_name(m.group(1))
        raw = m.group(2)
        if fi > m.end():
            raw = raw[: fi - m.end()]
        return "action", tool, _parse_input(raw)
    if fi >= 0:
        return "final", text[fi + len(_FINAL):].strip(), None
    if re.search(r"Action\s*:", text):
        return "invalid", "Invalid Format: Missing 'Action Input:' after 'Action:'", None
    return "invalid", "Invalid Format: Missing 'Action:' after 'Thought:'", None


# ---------------------------------------------------------------------------------------------------------------------------- ReAct agent
class ReActAgent:
    """Text ReAct loop in the LangChain zero-shot format with a JSON Action Input; `execute(tool, args)` returns the observation."""

    def __init__(self, name: str, llm: _LLM, system: str, tool_names: list[str], descriptions: str, execute, max_steps: int = MAX_STEPS):
        self.name, self.llm, self.system, self.tool_names, self.descriptions, self.execute, self.max_steps = name, llm, system, tool_names, descriptions, execute, max_steps
        self.steps = 0; self.actions = 0; self.final: str | None = None; self.stop_reason = "running"; self.scratchpad = ""
        self.keep, self.head = PRUNE_KEEP, PRUNE_HEAD

    def prompt(self, question: str) -> str:
        return REACT_TEMPLATE.format(tools=self.descriptions, tool_names=", ".join(self.tool_names), input=question, agent_scratchpad=self._pruned())

    def _pruned(self) -> str:
        """Request view of the scratchpad: when it exceeds PRUNE_CHARS, all but the last PRUNE_KEEP observations are cut to PRUNE_HEAD characters
        (the stored trajectory keeps them in full). Mirrors ifcedit.agent.prune_messages; the paper used larger-context GPT models."""
        sp = self.scratchpad
        if len(sp) <= PRUNE_CHARS and (self.keep, self.head) == (PRUNE_KEEP, PRUNE_HEAD):
            return sp
        parts = sp.split("\nObservation: ")
        n_obs = len(parts) - 1
        out = [parts[0]]
        for k, part in enumerate(parts[1:], 1):
            if k <= n_obs - self.keep:
                obs, sep, rest = part.partition("\nThought:")
                if len(obs) > self.head:
                    obs = obs[:self.head] + " ... [earlier observation shortened; call the tool again if you need it]"
                part = obs + sep + rest
            out.append(part)
        return "\nObservation: ".join(out)

    def run(self, question: str, scratchpad: str = "") -> str | None:
        self.scratchpad = scratchpad
        self.llm.log(self.name, "system", self.system)
        self.llm.log(self.name, "user", question + (("\n[scratchpad]\n" + scratchpad) if scratchpad else ""))
        while self.steps < self.max_steps:
            self.steps += 1
            try:
                text = self.llm.chat(self.name, self.system, self.prompt(question))
            except ContextOverflow:
                if (self.keep, self.head) == (PRUNE_KEEP, PRUNE_HEAD):      # OURS: one retry with hard pruning (keep 2 observations, 200 chars each)
                    self.keep, self.head = 2, 200
                    self.llm.log(self.name, "observation", "[context overflow: request pruned harder and resent]")
                    text = self.llm.chat(self.name, self.system, self.prompt(question))
                else:
                    raise
            kind, a, b = parse_react(text)
            if kind == "final":
                self.final = a; self.stop_reason = "final"
                return a
            if kind == "action":
                self.actions += 1
                if a not in self.tool_names:
                    obs = f"Error: '{a}' is not a valid tool, try one of [{', '.join(self.tool_names)}]."
                else:
                    obs = self.execute(a, b)
                    if obs == "__STOP__":
                        self.stop_reason = "stopped"
                        return None
                    if obs.startswith("__FINAL__"):     # terminal action (SaveFinalAnswer of the main agent, as the FinalAnswer tool ending the Fig. 4 trace)
                        obs = obs[len("__FINAL__"):]
                        self.llm.log(self.name, "observation", obs, tool=a)
                        self.scratchpad += (" " if not self.scratchpad else "") + text + "\nObservation: " + obs
                        self.final = b.get("answer") if isinstance(b, dict) else str(b); self.stop_reason = "final"
                        return self.final
            else:
                obs = a
            self.llm.log(self.name, "observation", obs, tool=(a if kind == "action" else None))
            self.scratchpad += (" " if not self.scratchpad else "") + text + "\nObservation: " + obs + "\nThought:"
        self.stop_reason = "max_steps"
        return None


# ---------------------------------------------------------------------------------------------------------------------------- framework
class IfcAgentRun:
    def __init__(self, prompt: str, ifc_path: str, workdir: str, client, model: str, budget: int, max_tokens: int, temperature: float):
        self.prompt, self.ifc_path, self.workdir, self.budget = prompt, ifc_path, workdir, budget
        self.llm = _LLM(client, model, max_tokens, temperature)
        self.tools = IfcAgentTools(workdir=workdir)
        self.executor_calls = 0; self.subagents = 0; self.react_actions = 0
        self.notes: list[str] = []
        self.outcome = "executed"; self.error: str | None = None
        self.main: ReActAgent | None = None
        self.lock = threading.Lock(); self.api_errors: list[str] = []

    # ---- scripted prefix of the main agent (LoadIFCFile, RewriteIfNeeded, QueryMemory, TaskDecompositionAgent) shown as ReAct steps
    def _prefix(self) -> tuple[str, str]:
        obs_load = self.tools.call("LoadIFCFile", {"file_path": self.ifc_path})
        rewritten = self.tools.call("RewriteIfNeeded", {"query": self.prompt})
        obs_mem = self.tools.call("QueryMemory", {})
        subtasks = self.decompose(rewritten)
        obs_dec = json.dumps({"subtasks": subtasks}, ensure_ascii=False)
        self.tools.procedure_context.append({"tool": "TaskDecompositionAgent", "time": time.strftime("%H:%M:%S"), "summary": obs_dec[:160]})
        pad = (" I load the IFC file first.\nAction: LoadIFCFile\nAction Input: " + json.dumps({"file_path": os.path.basename(self.ifc_path)}) + "\nObservation: " + obs_load +
               "\nThought: I rewrite the query if needed.\nAction: RewriteIfNeeded\nAction Input: " + json.dumps({"query": self.prompt}, ensure_ascii=False) + "\nObservation: " + rewritten +
               "\nThought: I check the memory for reusable results.\nAction: QueryMemory\nAction Input: {}\nObservation: " + obs_mem +
               "\nThought: I decompose the task into sequential subtasks.\nAction: TaskDecompositionAgent\nAction Input: " + json.dumps({"question": rewritten}, ensure_ascii=False) +
               "\nObservation: " + obs_dec + "\nThought:")
        return rewritten, pad

    def decompose(self, query: str) -> list[str]:
        system = DECOMPOSITION_SYSTEM.format(principles=DECOMPOSITION_PRINCIPLES)
        self.llm.log("TaskDecompositionAgent", "system", system)
        user = json.dumps({"question": query}, ensure_ascii=False)
        self.llm.log("TaskDecompositionAgent", "user", user)
        try:
            text = self.llm.chat("TaskDecompositionAgent", system, user)
        except Exception as e:  # noqa: BLE001
            self.notes.append(f"decomposition api error: {e}"[:300])
            return [query]
        v = _parse_input(text)
        subs: list = []
        if isinstance(v, dict):
            subs = v.get("subtasks") or [x for k, x in v.items() if k.startswith("subtask")]
        if not subs:
            m = re.search(r"\[.*\]", text, re.S)
            if m:
                try:
                    subs = json.loads(m.group(0))
                except json.JSONDecodeError:
                    subs = []
        out = []
        for s in subs if isinstance(subs, list) else []:
            if isinstance(s, dict):
                s = s.get("subtask") or s.get("task") or next(iter(s.values()), "")
            if isinstance(s, str) and s.strip():
                out.append(s.strip())
        return out or [query]

    # ---- executors (Mode I)
    def run_executor(self, name: str, args) -> str:
        if not isinstance(args, dict):
            return "Error: executor input must be a JSON object {\"subtask\": ..., \"global_ids\": [...]}."
        subtask = str(args.get("subtask") or args.get("task") or args.get("question") or "").strip()
        gids = args.get("global_ids") or args.get("global_id") or args.get("globalids") or []
        if isinstance(gids, str):
            gids = [g.strip() for g in re.split(r"[,\s]+", gids) if g.strip()]
        gids = [str(g).strip() for g in gids if str(g).strip()]
        if not subtask or not gids:
            return "Error: give both 'subtask' (string) and 'global_ids' (non-empty list of GlobalIds). Use GetGlobalIdsByAttributeValue to find GlobalIds."
        if self.executor_calls >= self.budget:
            self.outcome = "budget"; self.error = f"executor budget of {self.budget} calls exhausted"
            return "__STOP__"
        self.executor_calls += 1
        context = str(args.get("context") or "")
        note = ""
        if len(gids) > MODE_I_LIMIT:
            note = f" [mode2_not_reproduced: {len(gids)} entities > {MODE_I_LIMIT}; only the first {MODE_I_LIMIT} were processed in Mode I]"
            self.notes.append(f"mode2_not_reproduced: {name} on {len(gids)} entities")
            gids = gids[:MODE_I_LIMIT]
        toolset = EXECUTOR_TOOLS[name]
        memory = "\n\n".join(r for r, t in ((MEMORY_RULES_1, "QueryWithGlobalIdDict"), (MEMORY_RULES_2, "QueryWithStepIdDict")) if t in toolset) or "(none)"
        system = SUBAGENT_SYSTEM.format(isarc=ISARC_INSTRUCTIONS, memory=memory, placement=(PLACEMENT_PROMPT if name == "SequentialGeometryExecutor" else ""))
        descriptions = IfcAgentTools.describe(toolset)

        def one(k: int, gid: str) -> str:
            try:
                e = self.tools.file.by_guid(gid)
                line = f"GlobalId {gid} = STEP id #{e.id()} ({e.is_a()}, Name={e.Name!r})"
                if name == "SequentialGeometryExecutor":
                    line += f"; ObjectPlacement={('#%d' % e.ObjectPlacement.id()) if e.ObjectPlacement else None}; Representation={('#%d' % e.Representation.id()) if e.Representation else None}"
            except Exception:  # noqa: BLE001
                return f"Error: GlobalId {gid} not found in the model."
            # OURS: the sub-agent's question (subtask + entity line + context); the paper does not print it.
            q = (f"Subtask: {subtask}\nTarget entity: {line}\nLength unit of the model: 1 model unit = {self.tools.unit_scale:g} m."
                 + (f"\nContext from the main agent: {context[:3000]}" if context else "") + "\nWork on this one entity only and give the result for it in the Final Answer.")
            agent = ReActAgent(f"{name}[{k}:{gid}]", self.llm, system, toolset, descriptions, self.tools.call)
            with self.lock:
                self.subagents += 1
            try:
                final = agent.run(q)
            except Exception as ex:  # noqa: BLE001
                final = None; agent.stop_reason = f"api error: {type(ex).__name__}: {ex}"[:300]
                with self.lock:
                    self.api_errors.append(agent.stop_reason)
            with self.lock:
                self.react_actions += agent.actions
            if final is None:
                return f"no final answer ({agent.stop_reason}, {agent.steps} steps)"
            return final

        with ThreadPoolExecutor(max_workers=min(SUBAGENT_WORKERS, len(gids))) as ex:
            results = list(ex.map(lambda kg: one(*kg), enumerate(gids)))
        lines = [f"{name} ran {len(gids)} sub-agent(s) for subtask '{subtask}'.{note} Results:"]
        for gid, r in zip(gids, results):
            lines.append(f"- {gid}: {r[:1500]}")
        return "\n".join(lines)

    def main_execute(self, tool: str, args) -> str:
        if tool in EXECUTOR_TOOLS:
            return self.run_executor(tool, args)
        if tool == "TaskDecompositionAgent":
            q = args.get("question") if isinstance(args, dict) else str(args)
            return json.dumps({"subtasks": self.decompose(str(q or self.prompt))}, ensure_ascii=False)
        if tool == "LoadIFCFile":
            return "Error: the IFC file is already loaded; LoadIFCFile is not called again within a task."
        self.react_actions += 1
        out = self.tools.call(tool, args)
        if tool == "SaveFinalAnswer" and self.tools.final_answer is not None:
            return "__FINAL__" + out
        return out

    def run(self) -> dict:
        t0 = time.time()
        try:
            rewritten, pad = self._prefix()
            system = MAIN_SYSTEM.format(table=EXECUTOR_TABLE, limit=MODE_I_LIMIT)
            descriptions = "\n\n".join(f"{n}: {MAIN_DESCRIPTIONS.get(n) or IfcAgentTools.DESCRIPTIONS[n]}" for n in MAIN_TOOLS)
            self.main = ReActAgent("MainAgent", self.llm, system, MAIN_TOOLS, descriptions, self.main_execute)
            final = self.main.run(rewritten, scratchpad=pad)
            if self.outcome == "executed":
                if self.main.stop_reason == "max_steps":
                    self.outcome = "budget"; self.error = f"main agent hit the {MAX_STEPS}-step cap without a final answer"
                elif self.api_errors and self.tools.final_answer is None and final is None:
                    self.outcome = "api"; self.error = self.api_errors[0]
        except Exception as e:  # noqa: BLE001
            name = type(e).__name__
            self.outcome = "api" if ("openai" in type(e).__module__ or "Timeout" in name or "Connection" in name) else "other"
            self.error = f"{name}: {e}"[:500]
            final = None
        # ---- output file
        output_ifc = self.ifc_path
        if self.tools.exported:
            output_ifc = self.tools.exported[-1]
        elif self.tools.edits and self.tools.file is not None:
            try:
                self.tools.file.write(self.ifc_path)
                self.notes.append(f"fallback_write: {len(self.tools.edits)} SetFieldByStepId edit(s) but no ExportModifiedIFC; in-memory model written to ifc_path")
            except Exception as e:  # noqa: BLE001
                self.notes.append(f"fallback_write failed: {e}"[:300])
        final_text = self.tools.final_answer if self.tools.final_answer is not None else final
        summary = {"executor_calls": self.executor_calls, "subagents": self.subagents, "edits": self.tools.edits, "exported": self.tools.exported,
                   "notes": self.notes, "llm_calls": self.llm.calls, "main_steps": self.main.steps if self.main else 0, "seconds": round(time.time() - t0, 1)}
        self.llm.log("runner", "system", json.dumps(summary, ensure_ascii=False, default=str))
        return {"output_ifc": output_ifc, "final_text": final_text, "tool_calls": self.react_actions, "tokens_in": self.llm.tokens_in, "tokens_out": self.llm.tokens_out,
                "outcome": self.outcome, "error": self.error, "messages": self.llm.messages, "commits": len(self.tools.exported)}


def run(prompt: str, ifc_path: str, workdir: str, client, model: str, budget: int = 20, max_tokens: int = 16384, temperature: float = 0.0) -> dict:
    return IfcAgentRun(prompt, ifc_path, workdir, client, model, budget, max_tokens, temperature).run()
