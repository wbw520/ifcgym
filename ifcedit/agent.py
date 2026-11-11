"""Tool-calling CodeAct agent over an OpenAI-compatible endpoint (SGLang / vLLM services on the held nodes).

Protocol = BIM-Edit Sec. 4.1 / Appendix F: system prompt F.3 verbatim, one tool `execute_ifc_code`, budget of N tool-call rounds,
the run ends when the model answers without a tool call or the budget is exhausted. Feedback modes (DESIGN.md): `none` (BIM-Edit),
`validity`, `all` — extra checker text appended to the tool output after a commit.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable

from openai import OpenAI

SYSTEM_PROMPT = (
    "You are a BIM assistant, with a deep knowledge in Building Information Modeling. You are working with IFC files and need to "
    "create IFCOpenshell calls to fulfill the task. Make sure to understand the current model before modification and modify the "
    "model always correctly on geometry, topology and semantics. Always understand the unit scale of the model. Use sensible defaults "
    "for unspecified values. Execute commands directly and never ask for confirmation."
)
TOOL_DESCRIPTION = (
    "Execute Python code against the current IFC file. Input: {code: str}. The IFC model is pre-loaded as ifc (ifcopenshell.file). "
    "Also available: ifcopenshell, api (ifcopenshell.api), util (ifcopenshell.util), element_util (ifcopenshell.util.element), and "
    "guid (ifcopenshell.guid). Assign to result to return data. Call commit() to save modifications."
)
_TEXT_CALL = re.compile(r"<\|tool_call>\s*call:execute_ifc_code\{code:<\|\"\|>(.*?)<\|\"\|>\}", re.S)
_FENCE = re.compile(r"```(?:python|py)?\n(.*?)```", re.S)


_DSML_CALL = re.compile(r'<｜DSML｜invoke name="([A-Za-z_]+)">(.*?)</｜DSML｜invoke>', re.S)
_DSML_BLOCK = re.compile(r'<｜DSML｜tool_calls>.*?(</｜DSML｜tool_calls>|$)', re.S)


def _unwrap_args(args):
    """Models sometimes wrap the arguments once more ({"arguments": "{\"code\": ...}"}, nesting deeper each turn when the
    wrapped form is echoed back): unwrap until a dict with real parameters appears."""
    for _ in range(6):
        if isinstance(args, dict) and set(args) == {"arguments"}:
            inner = args["arguments"]
            try:
                args = json.loads(inner) if isinstance(inner, str) else inner
            except json.JSONDecodeError:
                return inner
        else:
            break
    return args
_DSML_PARAM = re.compile(r'<｜DSML｜parameter name="([A-Za-z_]+)"[^>]*>(.*?)</｜DSML｜parameter>', re.S)


def _text_tool_calls(content: str) -> list[tuple[str, str]]:
    """Fallback when the server did not parse the model's tool call: returns (tool name, JSON arguments) pairs.
    Gemma 4 text format `<|tool_call>call:execute_ifc_code{code:<|"|>...<|"|>}` and DeepSeek V4 DSML invokes (any tool name; the
    vLLM parser deepseek_v31 does not catch them). Only the explicit tool-call syntax is accepted; fenced code is NOT executed."""
    if not content:
        return []
    out = [("execute_ifc_code", json.dumps({"code": m.group(1)})) for m in _TEXT_CALL.finditer(content)]
    for m in _DSML_CALL.finditer(content):
        args = {k: v.strip() for k, v in _DSML_PARAM.findall(m.group(2))}
        out.append((m.group(1), json.dumps(args)))
    return out


TOOLS = [{"type": "function", "function": {"name": "execute_ifc_code", "description": TOOL_DESCRIPTION,
          "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}]

# ---- tools v2 (H001/E007; user 2026-10-20: the BIM-Edit tool definition need not be kept): a structured read-only inspection
# tool plus the code tool with the helper module `h` (ifcedit/ifctools.py) preloaded, and a short IFC reference card.
from ifcgym.ifctools import HELP as _HELP  # noqa: E402

TOOL_DESCRIPTION_V2 = (
    "Execute Python code against the current IFC file. Input: {code: str}. Pre-loaded: ifc (ifcopenshell.file), ifcopenshell, api "
    "(ifcopenshell.api), util (ifcopenshell.util), element_util (ifcopenshell.util.element), guid (ifcopenshell.guid) and the helper "
    "module h (see the system prompt). Assign to result to return data (JSON-serialisable). Every call continues from the state left by "
    "the previous call (a call that raises an exception is rolled back); call commit() to save the current state as the result - "
    "without a commit the result is the last committed state."
)
INSPECT_DESCRIPTION = (
    "Return a structured JSON description of one element by GlobalId: class, name, container, placement (world origin/axes, shared "
    "flags), representation items (extrusion profile as a world XY polygon, depth, direction), bounding box; for walls: axis end points, "
    "openings with their fills, connected walls (connection types, the other wall's axis and WHICH of its ends touches this wall), "
    "space boundaries and the bounded spaces' footprints; for spaces: footprint and boundaries (related_to=<wall guid> filters them); "
    "for doors/windows: the opening and host wall; with psets=true also the property sets (flagged when shared with other elements)."
)
TOOLS_V2 = [
    {"type": "function", "function": {"name": "inspect_element", "description": INSPECT_DESCRIPTION,
     "parameters": {"type": "object", "properties": {"guid": {"type": "string", "description": "GlobalId of the element"},
                                                      "related_to": {"type": "string", "description": "optional: for a space, only boundaries with this element's GlobalId"},
                                                      "psets": {"type": "boolean", "description": "include property sets and their values (default false; needed for property edits)"}},
                    "required": ["guid"]}}},
    {"type": "function", "function": {"name": "execute_ifc_code", "description": TOOL_DESCRIPTION_V2,
     "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
    # protocol v2.2 (2026-10-23, user): a query tool for references without a GlobalId and a self-check tool before committing
    {"type": "function", "function": {"name": "find_elements", "description": (
        "Find elements WITHOUT knowing their GlobalId: filter by IFC class (e.g. IfcWall, IfcDoor, IfcSpace, IfcColumn, IfcSlab), storey "
        "(GlobalId or name), name substring, predefined type, and/or distance in plan from a point (near=[x, y] in MODEL units, radius). "
        "Returns matches with GlobalId, class, name, storey, centre and bounding box (model units) and the total count. Use it to resolve "
        "descriptions like 'the walls of storey X', 'the space containing point (x, y)', 'columns near wall W' before inspecting them."),
     "parameters": {"type": "object", "properties": {"ifc_class": {"type": "string"}, "storey": {"type": "string"}, "name_contains": {"type": "string"},
                                                      "predefined_type": {"type": "string"}, "near": {"type": "array", "items": {"type": "number"}, "description": "[x, y] in model units"},
                                                      "radius": {"type": "number"}, "limit": {"type": "integer"}}, "required": []}}},
    {"type": "function", "function": {"name": "check_model", "description": (
        "Self-check of the current model state against the original file: new schema errors, which entities were added / removed / "
        "modified (class, name), relationships added / removed, and whether the geometry of the changed elements still builds. It states "
        "facts only (it does not judge the task): it sees the current working state (edits of earlier calls included); use it before commit() "
        "to confirm that only the intended elements changed and nothing broke."),
     "parameters": {"type": "object", "properties": {}, "required": []}}},
]
PROTOCOL_VERSION = "v2.2"
SYSTEM_PROMPT_V2 = SYSTEM_PROMPT + (
    "\n\nTools: find_elements(...) locates elements from a description (class, storey, name, position) when no GlobalId is given; "
    "inspect_element(guid) describes an element and everything that depends on it; execute_ifc_code(code) edits the model (each call "
    "continues from the previous one; commit() saves the result); check_model() lists what changed against the original and any new "
    "schema errors. Locate and inspect the target first, "
    "decide what must change with it, edit, verify with inspect_element and check_model, then commit.\n"
    "Helper module h (inside execute_ifc_code):\n" + _HELP + "\n"
    "IFC facts: coordinates are in MODEL units (h.unit_scale(ifc) = metres per unit). A wall's openings and the doors/windows filling "
    "them are placed relative to the wall's placement and move with it (do not move them again). IfcRelConnectsPathElements: "
    "RelatingElement/RelatedElement, RelatingConnectionType/RelatedConnectionType in {ATSTART, ATEND, ATPATH}. IfcRelSpaceBoundary: "
    "RelatingSpace, RelatedBuildingElement, ConnectionGeometry.SurfaceOnRelatingElement. IfcRelVoidsElement: RelatingBuildingElement/"
    "RelatedOpeningElement. IfcRelFillsElement: RelatingOpeningElement/RelatedBuildingElement. Property sets: element.IsDefinedBy -> "
    "IfcRelDefinesByProperties.RelatingPropertyDefinition. Psets, placements, points and profiles may be SHARED between elements: "
    "editing them in place changes other elements too - the helpers make private copies first (h.private does it by hand).\n"
    "Consistency: when an element is moved, resized or deleted, its hosted elements, joined walls, bounded spaces and their space "
    "boundaries must follow; change nothing else."
)
# Teacher-only recipe (context distillation, E007): the data-collection teacher gets an explicit cascade checklist; the student is
# trained on the resulting traces WITHOUT it (its prompt is SYSTEM_PROMPT_V2), so the procedure must be internalised.
CASCADE_RECIPE = (
    "\n\nProcedure for a wall move (vector v): (1) inspect_element(wall) - note the axis end points, thickness, the world y/x of the "
    "two faces, the connected walls and for each 'other_end_touching_this_wall', the bounded spaces' footprints. (2) h.translate(ifc, "
    "wall, v) - openings and doors follow automatically. (3) For every connected wall whose other_end_touching_this_wall is 'start' "
    "or 'end': h.move_wall_end(ifc, other, that_end, v) - the touching end follows the moved wall, the far end stays; connected walls "
    "with no touching end are left alone. (4) For every bounded space: find which face plane of the OLD wall position its footprint "
    "touches (a vertex coordinate equal to the old face value along the move axis, within the wall's extent on the other axis) and "
    "call h.shift_profile_vertices(ifc, space, axis, old_face_value, v, within={other_axis: (x_min, x_max)}); check vertices_moved > 0, "
    "otherwise try the other face. (5) For every space boundary of that space with this wall (inspect_element(space, related_to=wall)): "
    "h.shift_boundary_curve(ifc, rel_guid, None, None, v). (6) inspect_element(wall) once more to verify, then commit(). Do not move the "
    "connected walls or the spaces as a whole; do not edit points by hand; stop after committing. For a wall height change: "
    "h.set_extrusion_depth(ifc, wall, new_height), commit. For a property: inspect with psets=true, h.set_property, commit. For a door "
    "deletion with its opening: h.delete(ifc, door) then h.delete(ifc, opening), commit. For creating a door/window: inspect the wall, "
    "h.add_opening_fill(ifc, wall, along, width, height, kind, sill) with along/width/height in MODEL units (convert metres with "
    "1/h.unit_scale), commit. For a new wall / column / slab: h.add_wall / h.add_column / h.add_slab on the named storey with model-unit "
    "coordinates, commit. For a question (area, count, which wall): compute it from the model with inspect_element / h.space_area / "
    "ifc.by_type, do NOT commit any change, and end the final message with the requested JSON object. For an element described by its "
    "position ('the wall directly north of X'): inspect X, compare axes of candidate walls in +Y with overlapping extent, pick the nearest. "
    "Other edits (generator v4): move a door/window along its wall = h.translate(ifc, its OPENING, v) (the door follows; inspect the door "
    "to find its opening); move a column = h.translate; wall thickness = h.set_wall_thickness(ifc, wall, t); shorten a wall at one end = "
    "h.stretch_wall_end(ifc, wall, 'start'|'end', -delta) after reading the axis end points to see which end has the larger coordinate; "
    "rotate a wall = h.rotate(ifc, wall, degrees, (cx, cy)) about the named end point (these helpers also edit extrusions clipped by half-spaces, as in Revit gable walls); slab thickness / space height / column height = "
    "h.set_extrusion_depth; shorten a slab in X = h.shift_vertices_on_plane(ifc, slab, 'x', x_max, [-delta, 0, 0]); delete a wall = delete "
    "its doors/windows, then its openings, then the wall (h.delete each); delete a space or column = h.delete; delete an opening = delete its "
    "fill first; create a space = h.add_space(ifc, storey, polygon, height) with the polygon along the inner faces of the surrounding walls "
    "(inspect the walls: face coordinates = axis +- thickness/2). Resolving references: 'connected to both A and B' = the wall listed as "
    "connected in inspect_element(A) and in inspect_element(B); 'separates spaces A and B' = the wall that appears in the boundaries of both "
    "spaces; 'the door hosted by wall W' / 'connecting spaces A and B' = the door in W's openings; 'directly above wall W' = the wall with the "
    "same XY axis on the next storey; 'closest to the corner of W and V' = the door/window whose centre is nearest to the shared end point; "
    "'contains the point (x, y)' = the space whose footprint polygon contains it; 'the floor slab of storey S' = the slab contained in S; "
    "'the door connecting spaces A and B' = the door in their common wall whose position lies between the two footprints (one on each side); "
    "'directly north of W' requires the candidate to overlap W along at least half of W's length and to be the nearest such wall. "
    "Path references ('after entering space B from space A through door D, the wall on your right-hand side / facing you', 'the column farthest "
    "to your left'): walking direction e = the normal of D's host wall pointing into B (towards B's centroid), right = e rotated clockwise "
    "(e=(ex,ey) -> right=(ey,-ex)); 'on your right' walls run parallel to e with centre offset > 0 along right; 'facing you' is the wall of B "
    "perpendicular to e ahead of the door; 'farthest to your right' is the element of B with the largest offset along right, measured from the "
    "door centre. 'The most central window of wall W' = the window closest to W's axis mid-point. "
    "Tasks naming several targets ('all windows of wall W', 'all walls of storey S') are done element by element with the same helpers. A new wall "
    "that must be connected to existing walls: h.add_wall, then h.connect_walls(ifc, existing, new, 'ATSTART'|'ATEND', 'ATSTART'|'ATEND') "
    "for each (the end of each wall at the shared point). 'Directly above / below X' = same XY on the next storey up / down. "
    "Answers that ask for several ids end with {\"global_ids\": [...]}. All prompt coordinates are METRES: convert with 1/h.unit_scale(ifc). "
    "Instructions with several numbered changes (E009): do them one after another and call commit() after EACH completed change - later "
    "calls continue from the committed state and the last commit is the result - so finished changes are kept even if the tool budget runs "
    "out; inspect only what the next change needs and do not re-verify finished changes. Relationship wording ('create an opening and "
    "establish the relationships wall-opening (IfcRelVoidsElement) and opening-door (IfcRelFillsElement)', 'assign it to storey S'): "
    "h.add_opening_fill creates the opening, both relationships and the storey containment of the host wall's storey; 'remove the "
    "relationships of the opening to the wall / the storey': h.delete of the door and of the opening removes them. "
    "Quantity take-off (E010): quantity names (GrossSideArea, net volume, gross floor area, ...) are the IFC base quantities "
    "(Qto_*BaseQuantities, IFC4X3); compute them from the geometry (ifcopenshell.geom shapes, ifcopenshell.util.shape.get_volume / "
    "get_area / get_footprint_area), never from quantity or property values stored in the file. Wall: Length along the centre line; Width "
    "= thickness; GrossSideArea = elevation area of the wall's middle plane WITHOUT deducting openings, NetSideArea = the same WITH the "
    "openings deducted (for a wall of constant thickness = volume / Width); GrossVolume = body volume without the openings (geometry "
    "setting disable-opening-subtractions), NetVolume = body volume after subtracting the openings. Slab: Width = thickness; Perimeter = "
    "outer boundary; GrossArea = plan area inside the outer boundary (openings not deducted), NetArea = openings deducted; Gross/NetVolume "
    "likewise. Space: GrossFloorArea includes the area of elements standing inside the space (columns), NetFloorArea deducts it; "
    "GrossPerimeter = outer contour; Height = bottom to top of the space; GrossVolume / NetVolume = the corresponding floor area x height. "
    "Opening (IfcOpeningElement): Area = as seen in elevation for a wall opening (along the wall x vertical) or in plan for a slab "
    "opening, clipped to the host; Volume = the volume it removes from its host (Area x the host's thickness). Storey: GrossFloorArea = "
    "plan area covered by its spaces together with its walls and columns (construction elements included), NetFloorArea = sum of its "
    "spaces' NetFloorArea. A total over several elements = sum of the per-element quantities. Answer in the unit asked. "
    "Columns and slabs (E010): a column / slab given by its bounding box (corner, centre of the lower face, or length / width / height) "
    "is created with h.add_column / h.add_slab and then adjusted (rectangular section: the profile's XDim / YDim; base elevation: "
    "h.translate in Z so that the lowest z of the body equals the requested value). Relationships without a helper: oh = "
    "(ifc.by_type('IfcOwnerHistory') or [None])[0]; a connection = ifc.create_entity('IfcRelConnectsElements', "
    "GlobalId=ifcopenshell.guid.new(), OwnerHistory=oh, RelatingElement=a, RelatedElement=b); a space boundary = "
    "ifc.create_entity('IfcRelSpaceBoundary', GlobalId=ifcopenshell.guid.new(), OwnerHistory=oh, RelatingSpace=space, "
    "RelatedBuildingElement=element, PhysicalOrVirtualBoundary='PHYSICAL', InternalOrExternalBoundary='INTERNAL'); a change of storey = "
    "ifcopenshell.api.spatial.assign_container(ifc, relating_structure=new_storey, products=[element]) (it replaces the old containment). "
    "'Keep the lowest z of the bounding box fixed' while changing a thickness: h.set_extrusion_depth keeps the base for bodies extruded "
    "upwards - check the bounding box afterwards. One-sided extension 'keeping the edge at x = X fixed': move only the opposite edge "
    "(h.shift_vertices_on_plane on that plane). "
    "More edits (E010): deleting an element with listed relationships: h.delete removes the element and every relationship that "
    "references it (voids, fills, connections, space boundaries, containment) - delete its openings and their doors / windows "
    "first when asked; the related elements stay unchanged. A wall given by its bounding box: the axis is the centre line along "
    "the long side, thickness = the short side; h.add_wall then h.translate in Z to the requested base; connect with h.connect_walls. "
    "A room given by its bounding box: h.add_space with the rectangle, then h.translate in Z to the requested base; space "
    "boundaries to walls as IfcRelSpaceBoundary entities (see above). Merging rooms: delete the separating wall (and its "
    "openings / fills) and the other room, then replace the kept room's profile by the union outline (transform the world points "
    "into the extrusion's local coordinates: inverse of world_matrix(room) @ the extrusion Position). Wall thickness 'in -x "
    "direction' (one face fixed): h.set_wall_thickness (centred) and then h.translate the wall by half the change towards the "
    "named direction. Moving a window to another wall: h.translate its opening (and the window, if it is not placed relative to "
    "the opening) and set the opening's IfcRelVoidsElement.RelatingBuildingElement to the new wall. Widening an opening equally "
    "to both sides: h.shift_vertices_on_plane on both planes along the wall axis by half the change each. A taller window: "
    "h.set_extrusion_depth keeps its base; enlarge the opening the same way if the window would stick out. An opening in a slab: "
    "create an IfcOpeningElement with a box body through the slab (h._extrusion of an IfcRectangleProfileDef) and "
    "ifcopenshell.api.feature.add_feature(ifc, feature=opening, element=slab). Relative changes ('reduce the height by 0.5 m') "
    "start from the current value read from the model. "
    "More edits (E011): 'the room bounded by the walls A, B, C' = the IfcSpace whose IfcRelSpaceBoundary relationships include all "
    "of them; 'the elements that separate two rooms' = the wall bounding both rooms with the openings and doors / windows in it. "
    "A storey's length is that of its floor slab (the IfcSlab contained in it): change the slab with one edge fixed (vertices on "
    "the moving edge shifted, h.shift_vertices_on_plane). 'The slab that lies on the walls of a storey' = the slab whose underside is "
    "at the walls' top. A slab 'under the walls' that bound a room / host windows covers the bounding rectangle of those walls' "
    "complete outlines, also where a wall runs on past the room - do not trim it to the room. An exterior wall 'faces' the side without a room (+Y north, +X east). A wall centred between two facing "
    "walls runs from the face of one to the face of the other at the middle of their overlap (h.add_wall, h.connect_walls to both). "
    "A column on top of a column: same X / Y centre and section, base at the lower column's top, IfcRelConnectsElements between "
    "them. Moving a column towards the nearest exterior wall: perpendicular to that wall. Keep each turn short: write the edit "
    "code directly in the tool call instead of drafting the whole program in your reasoning first."
)
PROTOCOLS = {"v1": (SYSTEM_PROMPT, TOOLS), "v2": (SYSTEM_PROMPT_V2, TOOLS_V2), "v2r": (SYSTEM_PROMPT_V2 + CASCADE_RECIPE, TOOLS_V2),
             "v3": (SYSTEM_PROMPT_V2, TOOLS_V2)}   # v3 (2026-10-28) = v2 + sandbox jail (runners) + BUDGET_NOTE near the call limit
BUDGET_NOTE = ("[budget] {left} tool round{s} left. The result is the last commit(): if the edit is done or as good as you can make it, "
               "commit() now; do not start new explorations.")
BUDGET_NOTE_AT = (3, 1)     # rounds left when the note is appended to the round's last tool output
PRUNE_KEEP = 6          # v2: tool outputs older than the last 6 are shortened in the REQUEST (the stored trajectory keeps them in full)
PRUNE_HEAD = 400
PRUNE_BUDGET_CHARS = 150_000   # ~40k tokens: beyond this the oldest rounds are shortened further (code too) - 64k-context teachers


def _short(text: str, head: int, note: str) -> str:
    return text if len(text) <= head else text[:head] + "\n... [" + note + "]"


def prune_messages(messages: list[dict], keep: int = PRUNE_KEEP, head: int = PRUNE_HEAD, budget: int = PRUNE_BUDGET_CHARS) -> list[dict]:
    """Request view of a long episode. (1) all but the last `keep` tool outputs are cut to `head` characters; (2) if the view is
    still above `budget` characters, the oldest rounds (assistant text + code and their tool outputs, never the system/user turn
    or the last `keep` tool rounds) are shortened too. Applied identically when the student is trained (build_sft_data) and when
    it is served. Inspect outputs are 5-12k characters; a 40-call teacher episode otherwise overflows a 64k context."""
    idx = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    cut = set(idx[:-keep]) if len(idx) > keep else set()
    out = []
    for i, m in enumerate(messages):
        if i in cut and len(m.get("content", "")) > head:
            out.append({**m, "content": _short(m["content"], head, "earlier output shortened; call the tool again if you need it")})
        else:
            out.append(m)
    def size(ms):
        return sum(len(m.get("content") or "") + sum(len(tc["function"]["arguments"]) for tc in (m.get("tool_calls") or [])) for m in ms)
    protect = set(idx[-keep:]) if idx else set()
    first_kept_round = min(protect) if protect else len(out)
    for i in range(2, len(out)):
        if size(out) <= budget:
            break
        m = out[i]
        if i >= first_kept_round - 1:
            break
        if m.get("role") == "assistant":
            nm = {**m, "content": _short(m.get("content") or "", 300, "earlier reasoning shortened")}
            if m.get("tool_calls"):
                tcs = []
                for tc in m["tool_calls"]:
                    try:
                        args = json.loads(tc["function"]["arguments"])
                        if isinstance(args, dict) and isinstance(args.get("code"), str):
                            args["code"] = _short(args["code"], 400, "earlier code shortened")
                        tcs.append({**tc, "function": {**tc["function"], "arguments": json.dumps(args, ensure_ascii=False)}})
                    except json.JSONDecodeError:
                        tcs.append(tc)
                nm["tool_calls"] = tcs
            out[i] = nm
        elif m.get("role") == "tool":
            out[i] = {**m, "content": _short(m.get("content") or "", 200, "earlier output shortened")}
    return out


def find_code(args: dict) -> str:
    """The find_elements tool runs as code in the sandbox worker."""
    kw = {k: args[k] for k in ("ifc_class", "storey", "name_contains", "predefined_type", "near", "radius", "limit") if args.get(k) not in (None, "", [])}
    if "limit" in kw:
        try:
            kw["limit"] = int(kw["limit"])
        except (TypeError, ValueError):
            kw.pop("limit")
    if "radius" in kw:
        try:
            kw["radius"] = float(kw["radius"])
        except (TypeError, ValueError):
            kw.pop("radius")
    return "result = h.find(ifc, " + ", ".join(f"{k}={v!r}" for k, v in kw.items()) + ")"


CHECK_CODE = "result = h.check_model(ifc, open('original.txt').read().strip())"


def inspect_code(args: dict) -> str:
    """The inspect_element tool is executed as code in the same sandbox worker (same committed state, same process isolation)."""
    rel = args.get("related_to"); ps = args.get("psets")
    ps = ps if isinstance(ps, bool) else str(ps).lower() in ("true", "1", "yes")
    return f"result = h.inspect(ifc, {str(args.get('guid', ''))!r}, psets={ps}" + (f", related_to={str(rel)!r}" if rel else "") + ")"


@dataclass
class Episode:
    task_id: str
    model: str
    messages: list = field(default_factory=list)
    tool_calls: int = 0
    text_tool_calls: int = 0        # tool calls recovered from plain text (server parser missed them)
    commits: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    seconds: float = 0.0
    outcome: str = "executed"      # executed | budget | crash | api | timeout | other   (BIM-Edit Table 3 buckets)
    error: str | None = None
    final_text: str = ""


def run_episode(client: OpenAI, model: str, task_id: str, prompt: str, execute: Callable[[str], object], budget: int = 20,
                temperature: float = 0.0, top_p: float = 1.0, max_tokens: int = 8192, feedback: Callable[[object], str] | None = None,
                api_retries: int = 1, retry_delay_s: float = 30.0, extra_body: dict | None = None, protocol: str = "v1") -> Episode:
    ep = Episode(task_id=task_id, model=model)
    system_prompt, tools = PROTOCOLS[protocol]
    ep.messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    t0 = time.time()
    rounds = 0
    try:
        while True:
            resp = None
            for attempt in range(api_retries + 1):
                try:
                    resp = client.chat.completions.create(model=model, messages=(prune_messages(ep.messages) if protocol != "v1" else ep.messages), tools=tools, tool_choice="auto",
                                                          temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                                                          extra_body=extra_body or {})
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt >= api_retries:
                        ep.outcome = "api"; ep.error = f"{type(e).__name__}: {e}"[:2000]
                        ep.seconds = time.time() - t0
                        return ep
                    time.sleep(retry_delay_s)
            u = getattr(resp, "usage", None)
            if u:
                ep.tokens_in += int(getattr(u, "prompt_tokens", 0) or 0); ep.tokens_out += int(getattr(u, "completion_tokens", 0) or 0)
            msg = resp.choices[0].message
            calls = list(msg.tool_calls or [])
            if not calls:
                for k, (tname, targs) in enumerate(_text_tool_calls(msg.content or "")):
                    calls.append(type("TC", (), {"id": f"text_{rounds}_{k}", "function": type("F", (), {"name": tname, "arguments": targs})()})())
                if calls:
                    ep.text_tool_calls += len(calls)
            rc = getattr(msg, "reasoning_content", None) or (getattr(msg, "model_extra", None) or {}).get("reasoning_content") or (getattr(msg, "model_extra", None) or {}).get("reasoning")
            content = msg.content or ""
            if not rc and "</think>" in content:     # thinking mode served without a reasoning parser (DeepSeek V4: the <think> open tag is a prompt token)
                rc, content = content.rsplit("</think>", 1); rc = rc.replace("<think>", "", 1).strip(); content = content.lstrip()
                if not calls:
                    calls = []   # tool calls may follow </think> as DSML text: re-run the fallback on the answer part
                    for k, (tname, targs) in enumerate(_text_tool_calls(content)):
                        calls.append(type("TC", (), {"id": f"text_{rounds}_{k}", "function": type("F", (), {"name": tname, "arguments": targs})()})())
                    if calls:
                        ep.text_tool_calls += len(calls)
            if calls and any(str(c.id).startswith("text_") for c in calls):
                content = _DSML_BLOCK.sub("", _TEXT_CALL.sub("", content)).rstrip()   # the parsed calls are stored as tool_calls, not twice
            def _norm(a: str) -> str:   # the history shows normalised arguments, otherwise a model that wrapped them once nests deeper every turn
                try:
                    u = _unwrap_args(json.loads(a or "{}"))
                    return json.dumps(u, ensure_ascii=False) if isinstance(u, dict) else a
                except json.JSONDecodeError:
                    return a
            ep.messages.append({"role": "assistant", "content": content,
                                **({"reasoning_content": rc} if rc else {}),      # thinking traces (kept for distillation; passed back for preserved-thinking models)
                                **({"tool_calls": [{"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": _norm(c.function.arguments)}} for c in calls]} if calls else {})})
            if not calls:
                ep.final_text = msg.content or ""
                break
            rounds += 1
            for c in calls:
                malformed = None
                try:
                    args = _unwrap_args(json.loads(c.function.arguments or "{}"))
                    if c.function.name == "inspect_element" and isinstance(args, dict):
                        code = inspect_code(args) if args.get("guid") else None
                        malformed = None if code else "inspect_element needs {\"guid\": \"<GlobalId>\"}"
                    elif c.function.name == "find_elements" and isinstance(args, dict):
                        code = find_code(args)
                    elif c.function.name == "check_model":
                        code = CHECK_CODE
                    elif isinstance(args, dict):
                        code = args.get("code")
                        if not isinstance(code, str) or not code.strip():
                            malformed = "execute_ifc_code needs {\"code\": \"<python source>\"}; got keys " + str(list(args))
                    else:
                        code = str(args)
                except json.JSONDecodeError:
                    code = c.function.arguments or ""
                if malformed:
                    ep.tool_calls += 1
                    ep.messages.append({"role": "tool", "tool_call_id": c.id, "name": c.function.name, "content": "Error: malformed tool call - " + malformed})
                    continue
                ep.tool_calls += 1
                r = execute(code)
                if getattr(r, "committed", False):
                    ep.commits += 1
                out = r.output if hasattr(r, "output") else str(r)
                if getattr(r, "crashed", False):
                    ep.outcome = "crash"; ep.error = out[-1000:]
                if feedback is not None and getattr(r, "committed", False):
                    fb = feedback(r)
                    if fb:
                        out = out + "\n\n[checker report]\n" + fb
                ep.messages.append({"role": "tool", "tool_call_id": c.id, "name": c.function.name, "content": out})
            if ep.outcome == "crash":
                break
            if protocol == "v3" and budget - rounds in BUDGET_NOTE_AT and ep.messages[-1].get("role") == "tool":
                left = budget - rounds
                ep.messages[-1]["content"] += "\n\n" + BUDGET_NOTE.format(left=left, s="" if left == 1 else "s")
            if rounds >= budget:
                ep.outcome = "budget"
                break
    except Exception as e:  # noqa: BLE001
        ep.outcome = "other"; ep.error = f"{type(e).__name__}: {e}"[:2000]
    ep.seconds = time.time() - t0
    return ep
