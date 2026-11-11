"""Tool layer of the IFC-Agent reproduction (baseline B008): the paper's atomic IfcOpenShell tools as plain Python.

Paper: Gao, Hu, Chai, Weng, Li (2026), "Multi-agent framework for schema-guided reasoning and tool-augmented interaction with IFC
models", Automation in Construction 186, 106888 (no code released). Reproduced from the paper text, Table 1/2, Fig. 2/4/5, Appendix C/D
and the reproduction brief (scratchpad/ifcagent_brief.md). Every tool is a method of `IfcAgentTools`; all state (the in-memory
ifcopenshell file, `globalid_dict`, `stepid_dict`, the procedure memory, the edit log) lives on the instance, one instance per task,
because ifcedit.run_external calls run() from several threads. Tool execution is serialised per instance (ifcopenshell is not
thread-safe; sub-agents of one task run concurrently), file open/write is serialised per process.

Tool names and argument names follow the brief's table (Sec. 3):
    LoadIFCFile(file_path)                      GetInfoByGlobalId(global_id)          QueryWithGlobalIdDict(global_id, field)
    GetInfoByStepId(step_ids)                   QueryWithStepIdDict(step_ids, field)  ExtractPropertySetByGlobalId(global_id)
    ExtractQuantitySetByGlobalId(global_id)     ExtractRelationsByGlobalId(global_id) ExtractOpeningsByGlobalId(global_id)
    ExtractConnectivityByGlobalId(global_id)    ExtractAssignmentByGlobalId(global_id) ExtractBoundariesByGlobalId(global_id)
    ExtractContainsElementsByGlobalId(global_id) FindAdjacentSpacesByElement(global_id) DescribeRelationStep(step_id)
    DecideRelationNextHop(step_id[, goal])      ComputePlacementMatrixFromStepId(step_id) GetGlobalIdsByAttributeValue(ifc_type[, attribute, value])
    SetFieldByStepId(step_id, field, value)     ExportModifiedIFC(output_path)        QueryMemory()   RewriteIfNeeded(query)
    SaveFinalAnswer(answer)
Aliases accepted for the paper's inconsistent spellings: QuerySingleGlobalIdDict = QueryWithGlobalIdDict,
ExtractConnectivityRelationsByGlobalId = ExtractConnectivityByGlobalId, ExtractAssignmentRelationsByGlobalId = ExtractAssignmentByGlobalId,
FinalAnswer = SaveFinalAnswer.

CHOICES (every under-specified point of the paper and the default taken; numbers refer to the brief's Sec. 9.2 where one exists):
 C1  Observations contain raw STEP expressions (brief #3, Mode I regime of Sec. 6.1.1): GetInfoByStepId returns the success line of
     Fig. 4 followed by `str(entity)` (`#id=IFCTYPE(...)`) and a parsed attribute list (entity-valued attributes as `#id (IfcType)`);
     QueryWith*Dict formats entity values as `str(entity)`. `bare_values=True` switches to bare `#id` strings (the Mode II regime; not used).
 C2  Dotted-path semantics (brief #4): the path is resolved segment by segment; a dict segment is a key (with a one-level fall-through into
     child dicts so `PropertySet.FireRating` finds `PropertySet.Pset_WallCommon.FireRating`); an entity segment is an attribute (or inverse
     attribute); when the current value is a list the segment is mapped over it; integer segments / `Items[0]` index into lists; a miss
     returns "Field '<x>' not found ..." (never raises).
 C3  GetInfoByGlobalId stores exactly the Fig. 2 field list (type, GlobalId, Name, Description, Tag, PredefinedType, ObjectPlacement,
     Representation, HasAssociations, ContainedInStructure) in globalid_dict[gid]['BasicInfo']; fields the entity type does not have
     (e.g. Tag on IfcSpace in IFC2X3, PredefinedType in IFC2X3) are stored as None. Its observation is the success message of Fig. 4
     (no field values) - the agent must call QueryWithGlobalIdDict, as the Fig. 5 rules say.
 C4  Property / quantity sets are stored nested by set name: globalid_dict[gid]['PropertySet'][<pset>][<prop>] = value (bare value,
     like ifcopenshell.util.element.get_psets); the observation additionally lists the STEP id and value type of every property /
     quantity entity (`#456 IfcPropertySingleValue NominalValue=IFCLABEL('REI 30')`) because SetFieldByStepId needs a STEP id to edit
     a property value; the paper's Q51/Q52 show property edits succeeded, so the agent must have seen these ids somehow.
 C5  SetFieldByStepId (brief #5): the value is coerced to the type of the current value (float/int/bool/str); when the current value
     is a typed wrapper (IfcLabel, IfcReal, ...) a new wrapper of the same type is created; a None NominalValue is wrapped by Python
     type (str->IfcLabel, bool->IfcBoolean, int->IfcInteger, float->IfcReal); a "#123" string for an entity-typed attribute is resolved
     to that entity. No attribute / entity creation, no deletion, no propagation. Unknown attribute -> error string.
 C6  Cache update on write (brief #6): after a successful SetFieldByStepId the stepid_dict entry of that entity is refreshed and every
     BasicInfo / PropertySet / QuantitySet bucket in globalid_dict is re-extracted (so re-reads see the new value).
 C7  DescribeRelationStep / DecideRelationNextHop are deterministic (brief #7): describe = type, GlobalId/Name and every Relating* /
     Related* field with the referenced entities; next-hop = the entity-valued fields of the given entity with their target types (and
     inverse relations), ranked by a keyword match against the optional `goal` text; no LLM call inside the tool.
 C8  GetGlobalIdsByAttributeValue(ifc_type, attribute=None, value=None) (brief #8): by_type with subtypes, optional equality filter
     on one direct attribute (strings compared exactly, then case-insensitively; numbers within 1e-9); returns the count and the
     complete GlobalId list (the observation shows at most 200 ids; the full list is kept in `last_global_ids`).
 C9  Relation tools' output formats (unknown in the paper): one line per relationship `<inverse>: [#rel] -> STEP entity: <IfcRelType>`
     (Appendix C style) followed by the related entities as `#id IfcType GlobalId 'Name'`. Lists are cut after 50 entries,
     observations after MAX_OBS characters.
 C10 ComputePlacementMatrixFromStepId returns the LOCAL 4x4 matrix of one IfcLocalPlacement (its RelativePlacement) or of an
     IfcAxis2Placement3D/2D; the chain is multiplied by the agent as Appendix D instructs (the tool does not resolve PlacementRelTo).
 C11 ExportModifiedIFC writes to output_path (relative paths resolved under the task workdir; "export/..." directories created);
     the source file is never overwritten by the tool itself.
 C12 RewriteIfNeeded (regex, no LLM; the paper shows one example): `>=` -> "is greater than or equal to", `<=` -> "is smaller than or
     equal to", `>` -> "is greater than", `<` -> "is smaller than", `!=` -> "is not equal to", `=`/`==` between a word and a value ->
     "is equal to"; units are not added (the paper's example adds "meters", which needs knowledge of the field; we do not guess).
 C13 Memory (brief #14): fresh per instance; every tool call appends `Tool: <name> - Result: <first 160 chars>` with a timestamp; QueryMemory
     prints the "### Procedure Context (Tools) ###" block of the ISARC trace or "Memory is empty".
 C14 Errors are returned as observation strings starting with "Error:" (brief #12); nothing is raised to the agent loop.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time

import ifcopenshell
import ifcopenshell.util.element as eu
import ifcopenshell.util.placement as up
import ifcopenshell.util.unit as uu
import numpy as np

MAX_OBS = 8000          # characters per observation
MAX_LIST = 50           # entities shown per list
MAX_IDS = 200           # GlobalIds shown by GetGlobalIdsByAttributeValue
_IO_LOCK = threading.Lock()   # ifcopenshell.open / file.write across threads of one process

BASIC_INFO_FIELDS = ("type", "GlobalId", "Name", "Description", "Tag", "PredefinedType", "ObjectPlacement", "Representation",
                     "HasAssociations", "ContainedInStructure")
INVERSE_RELATIONS = ("HasAssociations", "IsDefinedBy", "ContainedInStructure", "ContainsElements", "HasOpenings", "FillsVoids", "HasFillings",
                     "VoidsElements", "ConnectedTo", "ConnectedFrom", "HasAssignments", "ProvidesBoundaries", "BoundedBy", "Decomposes",
                     "IsDecomposedBy", "ReferencedBy", "HasCoverings", "HasProjections", "HasStructuralMember", "ReferencedInStructures",
                     "IsTypedBy", "IsNestedBy", "Nests", "HasContext", "ServicesBuildings", "IsGroupedBy", "IsInterferedByElements", "InterferesElements")
WRAPPER_BY_PYTYPE = {bool: "IfcBoolean", int: "IfcInteger", float: "IfcReal", str: "IfcLabel"}
ALIASES = {"QuerySingleGlobalIdDict": "QueryWithGlobalIdDict", "ExtractConnectivityRelationsByGlobalId": "ExtractConnectivityByGlobalId",
           "ExtractAssignmentRelationsByGlobalId": "ExtractAssignmentByGlobalId", "FinalAnswer": "SaveFinalAnswer"}


def _is_entity(v) -> bool:
    return isinstance(v, ifcopenshell.entity_instance)


def _is_wrapper(v) -> bool:
    """IfcLabel('x'), IfcReal(0.3) ... : entity instances of a simple type carrying wrappedValue."""
    return _is_entity(v) and v.id() == 0 and hasattr(v, "wrappedValue")


def _step(e) -> str:
    return f"#{e.id()}"


def _cut(s: str, n: int = MAX_OBS) -> str:
    return s if len(s) <= n else s[:n] + f"\n... [observation truncated at {n} characters]"


def _coerce(value, like):
    """Coerce `value` to the Python type of `like` (the current attribute value)."""
    if isinstance(like, bool):
        return value if isinstance(value, bool) else str(value).strip().lower() in ("true", "1", "yes", ".t.")
    if isinstance(like, int) and not isinstance(like, bool):
        return int(float(value))
    if isinstance(like, float):
        return float(value)
    if isinstance(like, str):
        return str(value)
    return value


class IfcAgentTools:
    """One instance per task: the in-memory model, both caches, the procedure memory and the edit log."""

    def __init__(self, workdir: str | None = None, bare_values: bool = False):
        self.workdir = workdir or os.getcwd()
        self.bare_values = bare_values
        self.lock = threading.RLock()
        self.file: ifcopenshell.file | None = None
        self.file_path: str | None = None
        self.globalid_dict: dict[str, dict] = {}
        self.stepid_dict: dict[str, dict] = {}
        self.procedure_context: list[dict] = []
        self.edits: list[dict] = []
        self.exported: list[str] = []
        self.final_answer: str | None = None
        self.last_global_ids: list[str] = []
        self.unit_scale = 1.0
        self.calls = 0

    # ------------------------------------------------------------------ dispatch
    def names(self) -> list[str]:
        return [n for n in dir(self) if n[:1].isupper() and callable(getattr(self, n))]

    def call(self, name: str, args) -> str:
        """Dispatch one tool call; args is a dict (JSON Action Input) or a scalar. Returns the observation string, never raises."""
        name = ALIASES.get(name, name)
        fn = getattr(self, name, None) if name[:1].isupper() else None
        if fn is None or not callable(fn):
            return f"Error: unknown tool '{name}'. Valid tools: {', '.join(self.names())}."
        if not isinstance(args, dict):
            args = {"query": args} if name == "RewriteIfNeeded" else {"answer": args} if name == "SaveFinalAnswer" else {"global_id": args} if isinstance(args, str) else {}
        with self.lock:
            self.calls += 1
            try:
                out = fn(**args)
            except TypeError as e:
                out = f"Error: wrong arguments for {name}: {e}. Input must be a JSON object with the documented keys."
            except Exception as e:  # noqa: BLE001
                out = f"Error: {name} failed: {type(e).__name__}: {e}"
            out = _cut(str(out))
            self.procedure_context.append({"tool": name, "time": time.strftime("%H:%M:%S"), "summary": out[:160].replace("\n", " ")})
            return out

    # ------------------------------------------------------------------ helpers
    def _need_file(self):
        if self.file is None:
            raise RuntimeError("no IFC file loaded; call LoadIFCFile first")

    def _by_guid(self, global_id: str):
        self._need_file()
        try:
            return self.file.by_guid(str(global_id).strip())
        except Exception:  # noqa: BLE001
            raise RuntimeError(f"GlobalId '{global_id}' not found in the model")

    def _by_step(self, step_id):
        self._need_file()
        s = str(step_id).strip()
        try:
            n = int(s.lstrip("#"))
        except ValueError:
            raise RuntimeError(f"invalid STEP id '{step_id}' (expected '#123')")
        try:
            return self.file.by_id(n)
        except Exception:  # noqa: BLE001
            raise RuntimeError(f"STEP id #{n} not found in the model")

    def _ref(self, e) -> str:
        """Short reference `#id (IfcType)` for an entity-valued attribute (wrappers show their value)."""
        if _is_wrapper(e):
            return f"{e.is_a()}({e.wrappedValue!r})"
        return _step(e) if self.bare_values else f"{_step(e)} ({e.is_a()})"

    def _fmt(self, v, depth: int = 0) -> str:
        """Observation formatting of a resolved value (CHOICE C1)."""
        if _is_wrapper(v):
            return f"{v.is_a()}({v.wrappedValue!r})"
        if _is_entity(v):
            return _step(v) if self.bare_values else str(v)
        if isinstance(v, (list, tuple)):
            items = [self._fmt(x, depth + 1) for x in v[:MAX_LIST]]
            more = f", ... ({len(v) - MAX_LIST} more)" if len(v) > MAX_LIST else ""
            return "[" + ", ".join(items) + more + "]"
        if isinstance(v, dict):
            return json.dumps({k: self._fmt(x, depth + 1) if (_is_entity(x) or isinstance(x, (list, tuple, dict))) else x for k, x in v.items()},
                              ensure_ascii=False, default=str)
        return repr(v) if isinstance(v, str) else str(v)

    def _label(self, e) -> str:
        """`#id IfcType GlobalId 'Name'` for related entities."""
        gid = getattr(e, "GlobalId", None) if hasattr(e, "GlobalId") else None
        name = getattr(e, "Name", None) if hasattr(e, "Name") else None
        return f"{_step(e)} {e.is_a()}" + (f" {gid}" if gid else "") + (f" {name!r}" if name else "")

    def _resolve(self, obj, segments: list[str]):
        """Dotted-path resolution (CHOICE C2). Returns (found: bool, value, missing_segment)."""
        cur = obj
        for i, seg in enumerate(segments):
            m = re.fullmatch(r"(\w+)\[(\d+)\]", seg)
            idx = None
            if m:
                seg, idx = m.group(1), int(m.group(2))
            if isinstance(cur, (list, tuple)):
                if seg.isdigit():
                    k = int(seg)
                    if k >= len(cur):
                        return False, None, seg
                    cur = cur[k]
                    continue
                vals, missing = [], []
                for x in cur:
                    ok, v, miss = self._resolve(x, [seg] + segments[i + 1:])
                    vals.append(v if ok else None)
                    if not ok:
                        missing.append(miss)
                found = any(v is not None for v in vals)
                return found, vals, (None if found else (max(set(missing), key=missing.count) if missing else seg))
            if isinstance(cur, dict):
                if seg in cur:
                    cur = cur[seg]
                else:
                    hit = [c[seg] for c in cur.values() if isinstance(c, dict) and seg in c]   # one-level fall-through
                    if not hit:
                        return False, None, seg
                    cur = hit[0]
            elif _is_entity(cur):
                if _is_wrapper(cur) and seg in ("wrappedValue", "value"):
                    cur = cur.wrappedValue
                elif seg == "type":
                    cur = cur.is_a()
                elif seg in ("id", "step_id"):
                    cur = _step(cur)
                else:
                    try:
                        cur = getattr(cur, seg)
                    except AttributeError:
                        return False, None, seg
            else:
                return False, None, seg
            if idx is not None:
                if not isinstance(cur, (list, tuple)) or idx >= len(cur):
                    return False, None, f"{seg}[{idx}]"
                cur = cur[idx]
        return True, cur, None

    def _attr_names(self, e) -> list[str]:
        return [e.attribute_name(i) for i in range(len(e))]

    def _entity_lines(self, e) -> str:
        """`str(entity)` plus the parsed attribute list (CHOICE C1)."""
        parts = []
        for i, n in enumerate(self._attr_names(e)):
            try:
                v = e[i]
            except Exception:  # noqa: BLE001
                v = "<unreadable>"
            if _is_entity(v):
                parts.append(f"{n}={self._ref(v)}")
            elif isinstance(v, (list, tuple)):
                if v and all(_is_entity(x) for x in v):
                    parts.append(f"{n}=[{', '.join(self._ref(x) for x in v[:MAX_LIST])}{', ...' if len(v) > MAX_LIST else ''}] ({len(v)} items)")
                else:
                    parts.append(f"{n}={self._fmt(v)}")
            else:
                parts.append(f"{n}={v!r}")
        head = _step(e) if self.bare_values else str(e)
        return head + "\n  Attributes: " + ", ".join(parts)

    def _rel_lines(self, inv: str, rels, related_fn=None) -> list[str]:
        lines = []
        for r in rels:
            lines.append(f"{inv}: [{_step(r)}] -> STEP entity: {r.is_a()}")
            targets = related_fn(r) if related_fn else []
            for t in (targets or [])[:MAX_LIST]:
                lines.append(f"    {self._label(t)}")
            if targets and len(targets) > MAX_LIST:
                lines.append(f"    ... ({len(targets) - MAX_LIST} more)")
        return lines

    def _refresh_caches(self):
        """CHOICE C6: re-extract cached buckets after an edit."""
        for sid in list(self.stepid_dict):
            try:
                self.stepid_dict[sid] = self._by_step(sid).get_info(recursive=False, include_identifier=True)
            except Exception:  # noqa: BLE001
                pass
        for gid, buckets in list(self.globalid_dict.items()):
            try:
                e = self._by_guid(gid)
            except Exception:  # noqa: BLE001
                continue
            if "BasicInfo" in buckets:
                buckets["BasicInfo"] = self._basic_info(e)
            if "PropertySet" in buckets:
                buckets["PropertySet"] = eu.get_psets(e, psets_only=True)
            if "QuantitySet" in buckets:
                buckets["QuantitySet"] = eu.get_psets(e, qtos_only=True)

    def _basic_info(self, e) -> dict:
        info = {"type": e.is_a()}
        for f in BASIC_INFO_FIELDS[1:]:
            try:
                v = getattr(e, f)
            except AttributeError:
                v = None
            if f in ("HasAssociations", "ContainedInStructure"):
                v = list(v) if v else []
            info[f] = v
        return info

    # ------------------------------------------------------------------ setup / control tools
    def LoadIFCFile(self, file_path: str) -> str:
        """Load the IFC file into memory and initialise globalid_dict / stepid_dict."""
        p = file_path if os.path.isabs(file_path) else os.path.join(self.workdir, file_path)
        if not os.path.exists(p):
            return f"Error: file not found: {file_path}"
        with _IO_LOCK:
            self.file = ifcopenshell.open(p)
        self.file_path = p
        self.globalid_dict.clear(); self.stepid_dict.clear()
        try:
            self.unit_scale = float(uu.calculate_unit_scale(self.file))   # metres per model length unit
        except Exception:  # noqa: BLE001
            self.unit_scale = 1.0
        return (f"{os.path.basename(p)} loaded successfully. Global variables initialized: globalid_dict (by GlobalId), step_id_dict (by STEP ID). "
                f"Schema: {self.file.schema}; {len(self.file.by_type('IfcProduct'))} IfcProduct entities; length unit: 1 model unit = {self.unit_scale:g} m.")

    def QueryMemory(self) -> str:
        if not self.procedure_context:
            return "Memory is empty"
        lines = ["### Procedure Context (Tools) ###"] + [f"[{c['time']}] Tool: {c['tool']} - Result: {c['summary']}" for c in self.procedure_context[-40:]]
        return "\n".join(lines)

    def RewriteIfNeeded(self, query: str) -> str:
        """Rewrite comparison operators into words (CHOICE C12)."""
        q = str(query)
        q = re.sub(r"\s*>=\s*", " is greater than or equal to ", q)
        q = re.sub(r"\s*<=\s*", " is smaller than or equal to ", q)
        q = re.sub(r"\s*!=\s*", " is not equal to ", q)
        q = re.sub(r"(?<![-=<>!])\s*>\s*(?![=>-])", " is greater than ", q)
        q = re.sub(r"(?<![-=<>!])\s*<\s*(?![=<-])", " is smaller than ", q)
        q = re.sub(r"(?<=[A-Za-z_)\]])\s*==?\s*(?=[\w'\"(-])", " is equal to ", q)
        return re.sub(r"[ \t]{2,}", " ", q).strip()

    def SaveFinalAnswer(self, answer=None, **kw) -> str:
        if answer is None and kw:
            answer = json.dumps(kw, ensure_ascii=False) if len(kw) > 1 else next(iter(kw.values()))
        self.final_answer = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False, default=str)
        return "Final answer saved."

    # ------------------------------------------------------------------ cache / query tools
    def GetInfoByGlobalId(self, global_id: str) -> str:
        e = self._by_guid(global_id)
        self.globalid_dict.setdefault(e.GlobalId, {})["BasicInfo"] = self._basic_info(e)
        return (f"Success: basic info of {e.GlobalId} ({e.is_a()}, STEP id {_step(e)}) stored in globalid_dict['{e.GlobalId}']['BasicInfo'] "
                f"(fields: {', '.join(BASIC_INFO_FIELDS)}). Use QueryWithGlobalIdDict with \"field\": \"BasicInfo\" (or \"BasicInfo.<Field>\") to read it.")

    def QueryWithGlobalIdDict(self, global_id: str, field: str = "BasicInfo") -> str:
        gid = str(global_id).strip()
        if gid not in self.globalid_dict:
            return (f"Not found: GlobalId {gid} is not in globalid_dict. Run GetInfoByGlobalId (BasicInfo), ExtractPropertySetByGlobalId (PropertySet), "
                    f"ExtractQuantitySetByGlobalId (QuantitySet) or ExtractRelationsByGlobalId (Relations) first.")
        segs = [s for s in str(field).replace("/", ".").split(".") if s]
        buckets = self.globalid_dict[gid]
        if not segs:
            return f"globalid_dict['{gid}'] has: {', '.join(buckets)}"
        if segs[0] not in buckets:
            return f"Not found: field '{segs[0]}' is not in globalid_dict['{gid}'] (available: {', '.join(buckets) or 'nothing'}). Run the matching Extract*/GetInfo* tool first."
        ok, v, miss = self._resolve(buckets[segs[0]], segs[1:])
        if not ok:
            return f"Not found: '{miss}' in globalid_dict['{gid}']['{field}']. Available at that level: {self._keys_at(buckets[segs[0]], segs[1:])}"
        return f"globalid_dict['{gid}']['{field}'] = {self._fmt(v)}"

    def _keys_at(self, obj, segs: list[str]) -> str:
        cur = obj
        for s in segs:
            ok, nxt, _ = self._resolve(cur, [s])
            if not ok:
                break
            cur = nxt
        if isinstance(cur, dict):
            return ", ".join(map(str, cur))[:600]
        if _is_entity(cur):
            return ", ".join(self._attr_names(cur))[:600]
        if isinstance(cur, (list, tuple)):
            return f"list of {len(cur)}"
        return type(cur).__name__

    def GetInfoByStepId(self, step_ids) -> str:
        if isinstance(step_ids, (str, int)):
            step_ids = [step_ids]
        done, lines = [], []
        for s in step_ids:
            try:
                e = self._by_step(s)
            except RuntimeError as ex:
                lines.append(f"Error: {ex}")
                continue
            self.stepid_dict[_step(e)] = e.get_info(recursive=False, include_identifier=True)
            done.append(_step(e))
            lines.append(self._entity_lines(e))
        head = f"STEP IDs parsed and stored in step_id_dict: {done}. Use QueryWithStepIdDict to inspect fields."
        return head + ("\n" + "\n".join(lines) if lines else "")

    def QueryWithStepIdDict(self, step_ids, field: str) -> str:
        if isinstance(step_ids, (str, int)):
            step_ids = [step_ids]
        segs = [s for s in str(field).replace("/", ".").split(".") if s]
        out = []
        for s in step_ids:
            sid = "#" + str(s).strip().lstrip("#")
            if sid not in self.stepid_dict:
                out.append(f"{sid}: Not found in stepid_dict. Run GetInfoByStepId with {{\"step_ids\": [\"{sid}\"]}} first.")
                continue
            try:
                e = self._by_step(sid)
            except RuntimeError as ex:
                out.append(f"{sid}: Error: {ex}")
                continue
            ok, v, miss = self._resolve(e, segs)
            if not ok:
                out.append(f"{sid}: Not found: field '{miss}' (attributes of {sid} {e.is_a()}: {', '.join(self._attr_names(e))})")
            else:
                out.append(f"{sid}.{field} = {self._fmt(v)}")
        return "\n".join(out)

    # ------------------------------------------------------------------ hard-coded extractors
    def _pset_lines(self, e, qto: bool) -> tuple[dict, list[str]]:
        data = eu.get_psets(e, qtos_only=True) if qto else eu.get_psets(e, psets_only=True)
        lines = []
        for rel in getattr(e, "IsDefinedBy", []) or []:
            if not rel.is_a("IfcRelDefinesByProperties"):
                continue
            pd = rel.RelatingPropertyDefinition
            is_q = pd.is_a("IfcElementQuantity")
            if is_q != qto:
                continue
            lines.append(f"- {pd.Name} ({_step(pd)} {pd.is_a()}, via {_step(rel)} IfcRelDefinesByProperties):")
            items = pd.Quantities if is_q else pd.HasProperties
            for p in items or []:
                if p.is_a("IfcPropertySingleValue"):
                    nv = p.NominalValue
                    lines.append(f"    {p.Name} = {nv.wrappedValue if nv is not None else None!r}  [{_step(p)} IfcPropertySingleValue, NominalValue={self._ref(nv) if nv is not None else None}]")
                elif p.is_a("IfcPhysicalSimpleQuantity"):
                    vn = next((n for n in self._attr_names(p) if n.endswith("Value")), None)
                    lines.append(f"    {p.Name} = {getattr(p, vn, None)!r}  [{_step(p)} {p.is_a()}, field {vn}]")
                else:
                    lines.append(f"    {p.Name}  [{_step(p)} {p.is_a()}]")
        if hasattr(e, "IsTypedBy") or hasattr(e, "ObjectType"):
            t = eu.get_type(e)
            if t is not None:
                lines.append(f"- (inherited from type {_step(t)} {t.is_a()} {t.GlobalId}: sets {', '.join(k for k in data if k not in [l[2:].split(' (')[0] for l in lines])})")
        return data, lines

    def ExtractPropertySetByGlobalId(self, global_id: str) -> str:
        e = self._by_guid(global_id)
        data, lines = self._pset_lines(e, qto=False)
        self.globalid_dict.setdefault(e.GlobalId, {})["PropertySet"] = data
        if not data:
            return f"No property sets on {e.GlobalId} ({e.is_a()}). globalid_dict['{e.GlobalId}']['PropertySet'] = {{}}"
        return (f"Property sets of {e.GlobalId} ({e.is_a()}, {_step(e)}) stored in globalid_dict['{e.GlobalId}']['PropertySet'] "
                f"(query with field \"PropertySet.<Pset>.<Property>\" or \"PropertySet.<Property>\"). To change a property value use SetFieldByStepId on the "
                f"IfcPropertySingleValue STEP id with field \"NominalValue\".\n" + "\n".join(lines))

    def ExtractQuantitySetByGlobalId(self, global_id: str) -> str:
        e = self._by_guid(global_id)
        data, lines = self._pset_lines(e, qto=True)
        self.globalid_dict.setdefault(e.GlobalId, {})["QuantitySet"] = data
        if not data:
            return f"No quantity sets (IfcElementQuantity) on {e.GlobalId} ({e.is_a()}). globalid_dict['{e.GlobalId}']['QuantitySet'] = {{}}"
        return (f"Quantity sets of {e.GlobalId} ({e.is_a()}, {_step(e)}) stored in globalid_dict['{e.GlobalId}']['QuantitySet'] "
                f"(query with field \"QuantitySet.<Qto>.<Quantity>\" or \"QuantitySet.<Quantity>\").\n" + "\n".join(lines))

    def ExtractRelationsByGlobalId(self, global_id: str) -> str:
        e = self._by_guid(global_id)
        rels, lines = {}, []
        for inv in INVERSE_RELATIONS:
            try:
                v = getattr(e, inv)
            except AttributeError:
                continue
            if not v:
                continue
            rels[inv] = list(v)
            for r in v[:MAX_LIST]:
                tgt = []
                for n in self._attr_names(r):
                    if n.startswith(("Relating", "Related")):
                        x = getattr(r, n)
                        if _is_entity(x):
                            tgt.append(f"{n}={self._label(x)}")
                        elif isinstance(x, (list, tuple)):
                            tgt.append(f"{n}=[{', '.join(self._label(y) for y in x[:8])}{', ...' if len(x) > 8 else ''}] ({len(x)} items)")
                lines.append(f"{inv}: [{_step(r)}] -> STEP entity: {r.is_a()}; " + "; ".join(tgt))
        self.globalid_dict.setdefault(e.GlobalId, {})["Relations"] = rels
        if not lines:
            return f"No relationships found on {e.GlobalId} ({e.is_a()})."
        return f"Relationships of {e.GlobalId} ({e.is_a()}, {_step(e)}) stored in globalid_dict['{e.GlobalId}']['Relations']:\n" + "\n".join(lines)

    def ExtractOpeningsByGlobalId(self, global_id: str) -> str:
        e = self._by_guid(global_id)
        lines = []
        if hasattr(e, "HasOpenings") and e.HasOpenings:
            for r in e.HasOpenings:
                o = r.RelatedOpeningElement
                fills = [f.RelatedBuildingElement for f in (getattr(o, "HasFillings", None) or [])]
                lines.append(f"HasOpenings: [{_step(r)}] -> STEP entity: {r.is_a()}; opening {self._label(o)}; filled by: "
                             + (", ".join(self._label(f) for f in fills) if fills else "nothing"))
        if hasattr(e, "FillsVoids") and e.FillsVoids:
            for r in e.FillsVoids:
                o = r.RelatingOpeningElement
                host = [v.RelatingBuildingElement for v in (getattr(o, "VoidsElements", None) or [])]
                lines.append(f"FillsVoids: [{_step(r)}] -> STEP entity: {r.is_a()}; opening {self._label(o)}; opening voids: "
                             + (", ".join(self._label(h) for h in host) if host else "nothing"))
        if hasattr(e, "VoidsElements") and e.VoidsElements:
            for r in e.VoidsElements:
                lines.append(f"VoidsElements: [{_step(r)}] -> STEP entity: {r.is_a()}; voids {self._label(r.RelatingBuildingElement)}")
        if hasattr(e, "HasFillings") and e.HasFillings:
            for r in e.HasFillings:
                lines.append(f"HasFillings: [{_step(r)}] -> STEP entity: {r.is_a()}; filled by {self._label(r.RelatedBuildingElement)}")
        self.globalid_dict.setdefault(e.GlobalId, {})["Openings"] = lines
        return (f"Openings of {e.GlobalId} ({e.is_a()}):\n" + "\n".join(lines)) if lines else f"No opening relationships (IfcRelVoidsElement / IfcRelFillsElement) on {e.GlobalId} ({e.is_a()})."

    def ExtractConnectivityByGlobalId(self, global_id: str) -> str:
        e = self._by_guid(global_id)
        lines = []
        for inv, other in (("ConnectedTo", "RelatedElement"), ("ConnectedFrom", "RelatingElement")):
            for r in (getattr(e, inv, None) or []):
                extra = ""
                if r.is_a("IfcRelConnectsPathElements"):
                    extra = f"; RelatingConnectionType={r.RelatingConnectionType}, RelatedConnectionType={r.RelatedConnectionType}"
                lines.append(f"{inv}: [{_step(r)}] -> STEP entity: {r.is_a()}; {other}={self._label(getattr(r, other))}{extra}")
        self.globalid_dict.setdefault(e.GlobalId, {})["Connectivity"] = lines
        return (f"Connectivity of {e.GlobalId} ({e.is_a()}):\n" + "\n".join(lines)) if lines else f"No connectivity relationships (IfcRelConnectsElements / IfcRelConnectsPathElements) on {e.GlobalId} ({e.is_a()})."

    def ExtractAssignmentByGlobalId(self, global_id: str) -> str:
        e = self._by_guid(global_id)
        lines = []
        for r in (getattr(e, "HasAssignments", None) or []):
            tgt = [f"{n}={self._label(getattr(r, n))}" for n in self._attr_names(r) if n.startswith("Relating") and _is_entity(getattr(r, n))]
            lines.append(f"HasAssignments: [{_step(r)}] -> STEP entity: {r.is_a()}; " + "; ".join(tgt))
        for inv in ("IsGroupedBy", "ReferencedBy"):
            for r in (getattr(e, inv, None) or []):
                lines.append(f"{inv}: [{_step(r)}] -> STEP entity: {r.is_a()}")
        self.globalid_dict.setdefault(e.GlobalId, {})["Assignments"] = lines
        return (f"Assignments of {e.GlobalId} ({e.is_a()}):\n" + "\n".join(lines)) if lines else f"No assignment relationships (IfcRelAssigns*) on {e.GlobalId} ({e.is_a()})."

    def ExtractBoundariesByGlobalId(self, global_id: str) -> str:
        e = self._by_guid(global_id)
        lines = []
        for r in (getattr(e, "BoundedBy", None) or []):
            b = r.RelatedBuildingElement
            lines.append(f"BoundedBy: [{_step(r)}] -> STEP entity: {r.is_a()}; RelatedBuildingElement={self._label(b) if b else None}; "
                         f"{r.PhysicalOrVirtualBoundary}/{r.InternalOrExternalBoundary}; ConnectionGeometry={self._ref(r.ConnectionGeometry) if r.ConnectionGeometry else None}")
        for r in (getattr(e, "ProvidesBoundaries", None) or []):
            lines.append(f"ProvidesBoundaries: [{_step(r)}] -> STEP entity: {r.is_a()}; RelatingSpace={self._label(r.RelatingSpace)}; "
                         f"{r.PhysicalOrVirtualBoundary}/{r.InternalOrExternalBoundary}")
        self.globalid_dict.setdefault(e.GlobalId, {})["Boundaries"] = lines
        return (f"Space boundaries of {e.GlobalId} ({e.is_a()}):\n" + "\n".join(lines)) if lines else f"No IfcRelSpaceBoundary on {e.GlobalId} ({e.is_a()})."

    def ExtractContainsElementsByGlobalId(self, global_id: str) -> str:
        e = self._by_guid(global_id)
        lines = []
        for r in (getattr(e, "ContainsElements", None) or []):
            els = list(r.RelatedElements or [])
            by_type: dict[str, list] = {}
            for x in els:
                by_type.setdefault(x.is_a(), []).append(x)
            lines.append(f"ContainsElements: [{_step(r)}] -> STEP entity: {r.is_a()}; {len(els)} elements: " + ", ".join(f"{t} ({len(xs)})" for t, xs in sorted(by_type.items())))
            for t, xs in sorted(by_type.items()):   # GlobalIds per type; names only for small groups (the observation cap must not hide a type)
                lines.append(f"    {t} ({len(xs)}): " + ", ".join(f"{x.GlobalId}{(' ' + repr(x.Name)) if (x.Name and len(xs) <= 10) else ''}" for x in xs[:MAX_LIST]) + (f", ... ({len(xs) - MAX_LIST} more)" if len(xs) > MAX_LIST else ""))
        for r in (getattr(e, "ContainedInStructure", None) or []):
            lines.append(f"ContainedInStructure: [{_step(r)}] -> STEP entity: {r.is_a()}; RelatingStructure={self._label(r.RelatingStructure)}")
        for r in (getattr(e, "IsDecomposedBy", None) or []):
            lines.append(f"IsDecomposedBy: [{_step(r)}] -> STEP entity: {r.is_a()}; RelatedObjects=[{', '.join(self._label(x) for x in r.RelatedObjects[:MAX_LIST])}]")
        for r in (getattr(e, "Decomposes", None) or []):
            lines.append(f"Decomposes: [{_step(r)}] -> STEP entity: {r.is_a()}; RelatingObject={self._label(r.RelatingObject)}")
        self.globalid_dict.setdefault(e.GlobalId, {})["SpatialContainment"] = lines
        return (f"Spatial containment of {e.GlobalId} ({e.is_a()}):\n" + "\n".join(lines)) if lines else f"No spatial containment relationships on {e.GlobalId} ({e.is_a()})."

    def FindAdjacentSpacesByElement(self, global_id: str) -> str:
        e = self._by_guid(global_id)
        lines = []
        if e.is_a("IfcSpace"):
            mine = {r.RelatedBuildingElement.id(): r.RelatedBuildingElement for r in (e.BoundedBy or []) if r.RelatedBuildingElement}
            adj: dict[str, list] = {}
            for b in mine.values():
                for r in (getattr(b, "ProvidesBoundaries", None) or []):
                    s = r.RelatingSpace
                    if s is not None and s.id() != e.id():
                        adj.setdefault(s.GlobalId, [s, []])[1].append(b)
            for gid, (s, shared) in adj.items():
                lines.append(f"adjacent space {self._label(s)} via shared elements: " + ", ".join(self._label(b) for b in shared[:10]))
            head = f"Spaces adjacent to space {e.GlobalId} (sharing at least one bounding element, IfcRelSpaceBoundary): {len(adj)}"
        else:
            spaces = [r.RelatingSpace for r in (getattr(e, "ProvidesBoundaries", None) or []) if r.RelatingSpace]
            for s in spaces:
                lines.append(f"bounded space {self._label(s)}")
            head = f"Spaces bounded by element {e.GlobalId} ({e.is_a()}): {len(spaces)}"
        self.globalid_dict.setdefault(e.GlobalId, {})["AdjacentSpaces"] = lines
        return head + ("\n" + "\n".join(lines) if lines else "")

    # ------------------------------------------------------------------ relation navigation helpers (deterministic, CHOICE C7)
    def DescribeRelationStep(self, step_id) -> str:
        r = self._by_step(step_id)
        parts = [f"{_step(r)} is {r.is_a()}" + (f" GlobalId {r.GlobalId}" if hasattr(r, "GlobalId") else "") + (f" Name {r.Name!r}" if getattr(r, "Name", None) else "")]
        for n in self._attr_names(r):
            v = getattr(r, n)
            if n.startswith(("Relating", "Related")) or _is_entity(v) or (isinstance(v, (list, tuple)) and v and _is_entity(v[0])):
                if _is_entity(v):
                    parts.append(f"  {n} = {self._label(v) if hasattr(v, 'GlobalId') else self._ref(v)}")
                elif isinstance(v, (list, tuple)):
                    parts.append(f"  {n} = [{', '.join((self._label(x) if hasattr(x, 'GlobalId') else self._ref(x)) for x in v[:MAX_LIST])}{', ...' if len(v) > MAX_LIST else ''}] ({len(v)} items)")
            elif v is not None:
                parts.append(f"  {n} = {v!r}")
        return "\n".join(parts)

    def DecideRelationNextHop(self, step_id, goal: str = "") -> str:
        e = self._by_step(step_id)
        words = set(re.findall(r"[a-z]+", str(goal).lower()))
        cands = []
        for n in self._attr_names(e):
            v = getattr(e, n)
            if _is_entity(v) and not _is_wrapper(v):
                cands.append((n, [v]))
            elif isinstance(v, (list, tuple)) and v and all(_is_entity(x) and not _is_wrapper(x) for x in v):
                cands.append((n, list(v)))
        for inv in INVERSE_RELATIONS:
            try:
                v = getattr(e, inv)
            except AttributeError:
                continue
            if v:
                cands.append((inv + " (inverse)", list(v)))
        def score(c):
            n, xs = c
            tokens = set(re.findall(r"[a-z]+", (n + " " + " ".join(x.is_a() for x in xs[:5])).lower()))
            return -len(tokens & words)
        cands.sort(key=score)
        if not cands:
            return f"{_step(e)} {e.is_a()} has no entity-valued fields to follow."
        lines = [f"Candidate next hops from {_step(e)} {e.is_a()}" + (f" for goal {goal!r}" if goal else "") + " (best first):"]
        for n, xs in cands:
            lines.append(f"  {n} -> {', '.join(self._ref(x) for x in xs[:8])}{', ...' if len(xs) > 8 else ''} ({len(xs)} target{'s' if len(xs) != 1 else ''})")
        lines.append("Use GetInfoByStepId on the chosen target STEP id(s) to continue.")
        return "\n".join(lines)

    # ------------------------------------------------------------------ geometry
    def ComputePlacementMatrixFromStepId(self, step_id) -> str:
        e = self._by_step(step_id)
        if e.is_a("IfcLocalPlacement"):
            rel = e.RelativePlacement
            m = up.get_axis2placement(rel)
            parent = e.PlacementRelTo
            note = f"local matrix of {_step(e)} IfcLocalPlacement (RelativePlacement {_step(rel)} {rel.is_a()}); PlacementRelTo = {self._ref(parent) if parent else None}"
        elif e.is_a("IfcAxis2Placement3D") or e.is_a("IfcAxis2Placement2D"):
            m = up.get_axis2placement(e)
            note = f"matrix of {_step(e)} {e.is_a()}"
        elif e.is_a("IfcProduct"):
            return self.ComputePlacementMatrixFromStepId(e.ObjectPlacement) if e.ObjectPlacement else f"Error: {_step(e)} has no ObjectPlacement"
        else:
            return f"Error: {_step(e)} is {e.is_a()}, expected IfcLocalPlacement or IfcAxis2Placement3D"
        m = np.asarray(m, dtype=float)
        rows = "\n".join("  [" + ", ".join(f"{x:.6f}" for x in row) + "]" for row in m)
        return f"4x4 {note}:\n{rows}\nOrigin (translation) = [{m[0, 3]:.6f}, {m[1, 3]:.6f}, {m[2, 3]:.6f}]"

    # ------------------------------------------------------------------ entity lookup
    def GetGlobalIdsByAttributeValue(self, ifc_type: str, attribute: str | None = None, value=None, **kw) -> str:
        self._need_file()
        t = str(ifc_type).strip()
        try:
            ents = self.file.by_type(t)
        except Exception:  # noqa: BLE001
            return f"Error: unknown IFC entity type '{t}' in schema {self.file.schema}."
        if attribute:
            hit = []
            for e in ents:
                try:
                    v = getattr(e, attribute)
                except AttributeError:
                    return f"Error: {t} has no attribute '{attribute}'."
                if isinstance(v, str) and isinstance(value, str):
                    if v == value or v.strip().lower() == value.strip().lower():
                        hit.append(e)
                elif isinstance(v, (int, float)) and not isinstance(v, bool):
                    try:
                        if abs(float(v) - float(value)) <= 1e-9:
                            hit.append(e)
                    except (TypeError, ValueError):
                        pass
                elif v == value or str(v) == str(value):
                    hit.append(e)
            ents = hit
        gids = [e.GlobalId for e in ents if hasattr(e, "GlobalId")]
        self.last_global_ids = gids
        head = f"{len(gids)} {t} entities" + (f" with {attribute} = {value!r}" if attribute else "") + "."
        shown = ", ".join(gids[:MAX_IDS]) + (f", ... ({len(gids) - MAX_IDS} more)" if len(gids) > MAX_IDS else "")
        return head + (f" GlobalIds: [{shown}]" if gids else "")

    # ------------------------------------------------------------------ edit tools
    def SetFieldByStepId(self, step_id, field: str, value) -> str:
        e = self._by_step(step_id)
        names = self._attr_names(e)
        if field not in names:
            return f"Error: {_step(e)} {e.is_a()} has no attribute '{field}'. Attributes: {', '.join(names)}"
        idx = names.index(field)
        old = e[idx]
        try:
            if _is_wrapper(old):
                new = self.file.create_entity(old.is_a(), _coerce(value, old.wrappedValue))
            elif _is_entity(old):
                if isinstance(value, str) and re.fullmatch(r"#?\d+", value.strip()):
                    new = self._by_step(value)
                else:
                    return f"Error: {_step(e)}.{field} refers to an entity ({self._ref(old)}); give a STEP id like \"#123\" or edit that entity's fields instead."
            elif old is None and field == "NominalValue":
                new = self.file.create_entity(WRAPPER_BY_PYTYPE.get(type(value), "IfcLabel"), value)
            elif old is None and isinstance(value, str) and re.fullmatch(r"#\d+", value.strip()) and e.attribute_type(idx) == "ENTITY INSTANCE":
                new = self._by_step(value)
            elif isinstance(old, (list, tuple)):
                if not isinstance(value, (list, tuple)):
                    return f"Error: {_step(e)}.{field} is a list of {len(old)}; give a list value."
                new = type(old)(_coerce(v, old[0]) if old else v for v in value)
            else:
                new = _coerce(value, old) if old is not None else value
            setattr(e, field, new)
        except Exception as ex:  # noqa: BLE001
            return f"Error: could not set {_step(e)}.{field} = {value!r}: {type(ex).__name__}: {ex}"
        shown = new.wrappedValue if _is_wrapper(new) else (self._ref(new) if _is_entity(new) else new)
        self.edits.append({"step_id": _step(e), "type": e.is_a(), "field": field, "old": self._fmt(old), "new": self._fmt(new)})
        self._refresh_caches()
        return f"Updated {_step(e)}.{field} = {shown}"

    def ExportModifiedIFC(self, output_path: str = "export/updated_model.ifc") -> str:
        self._need_file()
        p = output_path if os.path.isabs(output_path) else os.path.join(self.workdir, output_path)
        if os.path.abspath(p) == os.path.abspath(self.file_path or ""):
            p = os.path.join(self.workdir, "export", "updated_model.ifc")   # the tool never overwrites the source (CHOICE C11)
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with _IO_LOCK:
            self.file.write(p)
        self.exported.append(p)
        return f"IFC model exported to: {output_path if not os.path.isabs(output_path) else p}"

    # ------------------------------------------------------------------ tool descriptions (Fig. 2 style; OURS except GetInfoByGlobalId)
    DESCRIPTIONS = {
        "LoadIFCFile": "Load an IFC file into memory and initialise globalid_dict / step_id_dict.\nInput must be a JSON object with keys:\n- 'file_path': string\nOutput: Success message.",
        "QueryMemory": "Return the procedure context of previous tool calls (tool name and result summary), or 'Memory is empty'.\nInput: {} (no keys).",
        "RewriteIfNeeded": "Rewrite comparison operators (>, <, >=, <=, =) in a query into words.\nInput must be a JSON object with keys:\n- 'query': string\nOutput: the rewritten query.",
        "SaveFinalAnswer": "Save the final answer of the task.\nInput must be a JSON object with keys:\n- 'answer': string (the complete final answer text)\nOutput: confirmation.",
        # Fig. 2 of the paper, verbatim:
        "GetInfoByGlobalId": ("Perform a one-time get_info() on the IFC element with the given GlobalId.\n\n"
                              "Extracts and stores basic IFC object fields into globalid_dict[<GlobalId>]['BasicInfo'], including:\n"
                              "- type, GlobalId, Name, Description, Tag, PredefinedType\n"
                              "- ObjectPlacement, Representation, HasAssociations, ContainedInStructure\n\n"
                              "Input must be a Python dict or JSON object with the following keys:\n"
                              "- 'global_id': string (single GlobalId to query)\n\n"
                              "Output: Success message. Fields are written to globalid_dict under 'BasicInfo'."),
        "QueryWithGlobalIdDict": ("Read a field from globalid_dict[<GlobalId>] (filled by GetInfoByGlobalId / ExtractPropertySetByGlobalId / ExtractQuantitySetByGlobalId / ExtractRelationsByGlobalId).\n"
                                  "Input must be a JSON object with keys:\n- 'global_id': string\n- 'field': dotted path, e.g. 'BasicInfo', 'BasicInfo.Name', 'BasicInfo.ObjectPlacement', "
                                  "'PropertySet.FireRating', 'QuantitySet.GrossVolume', 'BasicInfo.Representation.Representations.Items.SweptArea.YDim' (lists are mapped over)\n"
                                  "Output: the value; referenced entities are shown as STEP expressions '#id=IFCTYPE(...)'. 'Not found' if the field is not cached yet."),
        "GetInfoByStepId": ("Parse the STEP entities with the given STEP ids and store their fields in step_id_dict.\nInput must be a JSON object with keys:\n"
                            "- 'step_ids': list of strings, e.g. [\"#15042\"]\nOutput: Success message plus each entity's STEP expression and its attributes (referenced entities as '#id (IfcType)')."),
        "QueryWithStepIdDict": ("Read a field of STEP entities stored in step_id_dict (run GetInfoByStepId first).\nInput must be a JSON object with keys:\n"
                                "- 'step_ids': list of strings, e.g. [\"#15042\"]\n- 'field': attribute name or dotted path, e.g. 'Depth', 'RelativePlacement.Location.Coordinates', 'Representations.Items'\n"
                                "Output: '#id.field = value' per entity; referenced entities as STEP expressions."),
        "ExtractPropertySetByGlobalId": ("Extract all property sets (IfcPropertySet via IfcRelDefinesByProperties) of the element and store them in globalid_dict[<GlobalId>]['PropertySet'].\n"
                                         "Input must be a JSON object with keys:\n- 'global_id': string\nOutput: every property with its value, STEP id and value type (e.g. FireRating, LoadBearing, IsExternal, ThermalTransmittance)."),
        "ExtractQuantitySetByGlobalId": ("Extract all quantity sets (IfcElementQuantity) of the element and store them in globalid_dict[<GlobalId>]['QuantitySet'].\n"
                                         "Input must be a JSON object with keys:\n- 'global_id': string\nOutput: every quantity (Length, Width, Height, GrossArea, NetVolume, ...) with its value and STEP id."),
        "ExtractRelationsByGlobalId": ("List all relationships of the element (HasAssociations, IsDefinedBy, ContainedInStructure, HasOpenings, FillsVoids, ConnectedTo/From, HasAssignments, "
                                       "ProvidesBoundaries, BoundedBy, Decomposes, IsDecomposedBy, ...) with the relationship STEP ids and related entities; stored in globalid_dict[<GlobalId>]['Relations'].\n"
                                       "Input must be a JSON object with keys:\n- 'global_id': string\nOutput: lines '<relation>: [#rel] -> STEP entity: IfcRel...; Relating...=..., Related...=...'."),
        "ExtractOpeningsByGlobalId": ("List the openings of an element (IfcRelVoidsElement: HasOpenings) with the doors/windows filling them, or the opening an element fills (IfcRelFillsElement: FillsVoids) and its host.\n"
                                      "Input must be a JSON object with keys:\n- 'global_id': string"),
        "ExtractConnectivityByGlobalId": ("List the connectivity relationships of an element (IfcRelConnectsElements / IfcRelConnectsPathElements: ConnectedTo, ConnectedFrom) with the connected elements and connection types.\n"
                                          "Input must be a JSON object with keys:\n- 'global_id': string"),
        "ExtractAssignmentByGlobalId": ("List the assignment relationships of an element (IfcRelAssigns*: HasAssignments, groups, systems).\nInput must be a JSON object with keys:\n- 'global_id': string"),
        "ExtractBoundariesByGlobalId": ("List the space boundaries (IfcRelSpaceBoundary): for a space its bounding elements (BoundedBy), for an element the spaces it bounds (ProvidesBoundaries).\n"
                                        "Input must be a JSON object with keys:\n- 'global_id': string"),
        "ExtractContainsElementsByGlobalId": ("For a spatial structure element (site, building, storey, space): the elements it contains (IfcRelContainedInSpatialStructure) grouped by type with GlobalIds, "
                                              "and its decomposition; for an element: the storey/space that contains it.\nInput must be a JSON object with keys:\n- 'global_id': string"),
        "FindAdjacentSpacesByElement": ("For a space: the spaces adjacent to it (sharing at least one bounding element via IfcRelSpaceBoundary) with the shared elements; for an element: the spaces it bounds.\n"
                                        "Input must be a JSON object with keys:\n- 'global_id': string"),
        "DescribeRelationStep": ("Describe a relationship entity (IfcRel...) or any entity by STEP id: its type and its Relating*/Related* and other entity-valued fields with the referenced entities.\n"
                                 "Input must be a JSON object with keys:\n- 'step_id': string, e.g. \"#17045\""),
        "DecideRelationNextHop": ("List the fields of an entity that can be followed next (entity references and inverse relationships) with their target entity types, best match to the goal first.\n"
                                  "Input must be a JSON object with keys:\n- 'step_id': string\n- 'goal': string (optional, what you are looking for)"),
        "ComputePlacementMatrixFromStepId": ("Compute the 4x4 transformation matrix of one IfcLocalPlacement (its RelativePlacement) or IfcAxis2Placement3D. The matrix is LOCAL: multiply the matrices of the "
                                             "whole PlacementRelTo chain (root first) for an absolute placement.\nInput must be a JSON object with keys:\n- 'step_id': string, e.g. \"#15030\"\nOutput: the matrix rows and the origin."),
        "GetGlobalIdsByAttributeValue": ("Return the entity count and the complete list of GlobalIds of an IFC entity type, optionally filtered by one attribute value.\n"
                                         "Input must be a JSON object with keys:\n- 'ifc_type': string, e.g. 'IfcWall', 'IfcDoor', 'IfcSpace', 'IfcBuildingStorey'\n- 'attribute': string (optional), e.g. 'Name'\n"
                                         "- 'value': the value to match (optional)\nOutput: '<n> <type> entities. GlobalIds: [...]'."),
        "SetFieldByStepId": ("Set one attribute of one STEP entity in the in-memory model (the value type is kept: strings, numbers, booleans; a property value is set on the IfcPropertySingleValue with "
                             "field 'NominalValue'; a reference with a STEP id string \"#123\").\nInput must be a JSON object with keys:\n- 'step_id': string, e.g. \"#15067\"\n- 'field': string, e.g. \"Name\", "
                             "\"LayerThickness\", \"NominalValue\"\n- 'value': the new value\nOutput: 'Updated #id.field = value' or an error."),
        "ExportModifiedIFC": ("Write the in-memory IFC model (with all SetFieldByStepId changes) to a new file.\nInput must be a JSON object with keys:\n- 'output_path': string, e.g. \"export/updated_model.ifc\"\nOutput: 'IFC model exported to: <path>'."),
    }

    @classmethod
    def describe(cls, names: list[str]) -> str:
        return "\n\n".join(f"{n}: {cls.DESCRIPTIONS[n]}" for n in names)
