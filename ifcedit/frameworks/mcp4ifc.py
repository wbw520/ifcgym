"""MCP4IFC (Nithyanantham et al., arXiv 2511.05533; code Show2Instruct/ifc-bonsai-mcp @ 6215493, MIT) on an OpenAI-compatible backbone.

The paper's "agent" is Claude Desktop's generic tool-use loop over the FastMCP server; the repository has no client, system prompt
or step cap. This module is that loop, kept minimal: system prompt = the FastMCP `instructions` string + the repository's only prompt
(prompts.py, verbatim) + one framing sentence of ours (FRAMING); the task prompt is the user message; tools = the 39 vendored tools
(mcp4ifc_vendor/README.md: Route-2 port, Blender/Bonsai replaced by plain IfcOpenShell); tool_choice auto; the episode ends when the
assistant answers without a tool call, after `budget` tool-call rounds (outcome "budget"), when the worker process dies ("crash") or on
an API error after API_RETRIES retries ("api"). At the end the in-memory IFC is always written to ifc_path (the generous reading of
the original's save semantics, where only predefined tools and an explicit save_and_load_ifc() persisted edits).
"""
from __future__ import annotations

import json
import os
import time

from openai import OpenAI

from ifcedit.agent import _DSML_BLOCK, _TEXT_CALL, _text_tool_calls, _unwrap_args
from ifcedit.frameworks.mcp4ifc_vendor.blender_mcp import api_tools  # noqa: F401  (registers the tools)
from ifcedit.frameworks.mcp4ifc_vendor.blender_mcp.mcp_instance import INSTRUCTIONS
from ifcedit.frameworks.mcp4ifc_vendor.blender_mcp.prompts import ifc_building_element_creation_strategy
from ifcedit.frameworks.mcp4ifc_vendor.blender_mcp.runtime import WorkerConnection, bind_connection, mcp
from ifcedit.frameworks.mcp4ifc_vendor.blender_mcp.schema import build, validate_args

FRAMING = ("The IFC model is loaded. Apply the instruction to it using the tools; predefined tools save the file; if you edit with "
           "execute_ifc_code_tool, the model in memory is written at the end. When finished, answer in text.")
SYSTEM_PROMPT = INSTRUCTIONS + "\n\n" + ifc_building_element_creation_strategy() + "\n\n" + FRAMING
TOOLS, ARG_MODELS = build(mcp.tools)
TOOL_NAMES = list(mcp.tools)
API_RETRIES = 3
RETRY_DELAY_S = 30.0
TOOL_OUTPUT_CAP = int(os.environ.get("MCP4IFC_TOOL_OUTPUT_CAP", "200000"))   # characters kept of one tool result (env override, 65,536-token serving window) (client-side cap, ours: the original Claude Desktop client has no documented limit; get_scene_info
                            # on a real 440-product building returns ~1 MB, more than any context window; the cut is marked in the transcript)
MIN_MAX_TOKENS = 1024   # on a context-length rejection max_tokens is halved down to this before the episode is given up (vLLM checks prompt + max_tokens <= max_model_len)
_CONTEXT_ERR = ("context length", "maximum context", "context_length", "max_model_len", "too long", "exceeds the model", "reduce the length")


def call_tool(name: str, arguments: str) -> str:
    """Executes one tool call the way the FastMCP server would: unknown tool / invalid arguments -> error text, else the tool's JSON."""
    fn = mcp.tools.get(name)
    if fn is None:
        return f"Error executing tool {name}: Unknown tool: {name}"
    try:
        args = _unwrap_args(json.loads(arguments or "{}"))
    except json.JSONDecodeError as e:
        return f"Error executing tool {name}: invalid JSON arguments ({e})"
    if not isinstance(args, dict):
        return f"Error executing tool {name}: arguments must be a JSON object"
    try:
        model, str_params = ARG_MODELS[name]
        kwargs = validate_args(name, model, str_params, args)
    except ValueError as e:
        return f"Error executing tool {name}: {e}"
    try:
        return fn(**kwargs)
    except Exception as e:  # noqa: BLE001  (the wrappers catch everything themselves; this is a belt)
        return f"Error executing tool {name}: {type(e).__name__}: {e}"


def _norm(a: str) -> str:
    """History view of the arguments (ifcedit.agent): a model that wrapped them once ({"arguments": "{...}"}) otherwise nests deeper every turn."""
    try:
        u = _unwrap_args(json.loads(a or "{}"))
        return json.dumps(u, ensure_ascii=False) if isinstance(u, dict) else a
    except json.JSONDecodeError:
        return a


def run(prompt: str, ifc_path: str, workdir: str, client: OpenAI, model: str, budget: int, max_tokens: int, temperature: float) -> dict:
    r = {"output_ifc": ifc_path, "final_text": None, "tool_calls": 0, "tokens_in": 0, "tokens_out": 0, "outcome": "executed", "error": None,
         "messages": [], "saves": 0, "first_prompt_tokens": None, "max_tokens_final": max_tokens, "context_clamps": 0}
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    r["messages"] = messages
    t0 = time.time()
    conn = WorkerConnection(ifc_path, workdir)
    bind_connection(conn)
    try:
        rounds = 0
        while True:
            resp = None
            attempt = 0
            while True:
                try:
                    resp = client.chat.completions.create(model=model, messages=messages, tools=TOOLS, tool_choice="auto",
                                                          temperature=temperature, max_tokens=max_tokens)
                    break
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {e}"
                    if any(k in err.lower() for k in _CONTEXT_ERR):   # tool schema (~29k tokens) + history + max_tokens exceeds the served context
                        if max_tokens // 2 >= MIN_MAX_TOKENS:          # shrink the requested completion first (no tool pruning); the history only grows, so keep it
                            max_tokens //= 2; r["max_tokens_final"] = max_tokens; r["context_clamps"] += 1
                            continue
                        r["outcome"] = "api"; r["error"] = ("context_overflow: " + err)[:2000]
                        return r
                    attempt += 1
                    if attempt > API_RETRIES:
                        r["outcome"] = "api"; r["error"] = err[:2000]
                        return r
                    time.sleep(RETRY_DELAY_S)
            u = getattr(resp, "usage", None)
            if u:
                pt = int(getattr(u, "prompt_tokens", 0) or 0)
                r["tokens_in"] += pt; r["tokens_out"] += int(getattr(u, "completion_tokens", 0) or 0)
                if r["first_prompt_tokens"] is None:
                    r["first_prompt_tokens"] = pt
            msg = resp.choices[0].message
            calls = list(msg.tool_calls or [])
            rc = getattr(msg, "reasoning_content", None) or (getattr(msg, "model_extra", None) or {}).get("reasoning_content") or (getattr(msg, "model_extra", None) or {}).get("reasoning")
            content = msg.content or ""
            if not rc and "</think>" in content:     # thinking served without a reasoning parser (ifcedit.agent convention)
                rc, content = content.rsplit("</think>", 1); rc = rc.replace("<think>", "", 1).strip(); content = content.lstrip()
            if not calls:                            # server parser missed a DSML / text tool call (ifcedit.agent fallback, any tool name)
                for k, (tname, targs) in enumerate(_text_tool_calls(content)):
                    calls.append(type("TC", (), {"id": f"text_{rounds}_{k}", "function": type("F", (), {"name": tname, "arguments": targs})()})())
                if calls:
                    content = _DSML_BLOCK.sub("", _TEXT_CALL.sub("", content)).rstrip()
            messages.append({"role": "assistant", "content": content, **({"reasoning_content": rc} if rc else {}),
                             **({"tool_calls": [{"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": _norm(c.function.arguments)}} for c in calls]} if calls else {})})
            if not calls:
                r["final_text"] = content
                break
            rounds += 1
            for c in calls:
                r["tool_calls"] += 1
                out = call_tool(c.function.name, c.function.arguments)
                if len(out) > TOOL_OUTPUT_CAP:
                    out = out[:TOOL_OUTPUT_CAP] + f"\n[tool output truncated by the client: {len(out) - TOOL_OUTPUT_CAP} more characters not shown]"
                messages.append({"role": "tool", "tool_call_id": c.id, "name": c.function.name, "content": out})
                if conn.dead:
                    break
            if conn.dead:
                r["outcome"] = "crash"; r["error"] = f"worker process exited (see {conn.log_path})"
                break
            if rounds >= budget:
                r["outcome"] = "budget"
                break
    except Exception as e:  # noqa: BLE001
        r["outcome"] = "other"; r["error"] = f"{type(e).__name__}: {e}"[:2000]
    finally:
        try:
            if not conn.dead:
                w = conn.write_ifc()
                if isinstance(w, dict) and "error" in w:
                    r["error"] = (r["error"] or "") + f" | final write failed: {w['error']}"
                elif isinstance(w, dict):
                    r["saves"] = int(w.get("saves", 0))
        except Exception as e:  # noqa: BLE001
            r["error"] = (r["error"] or "") + f" | final write failed: {e}"
        r["timeouts"] = conn.timeouts
        conn.close()
        bind_connection(None)
        r["seconds"] = round(time.time() - t0, 1)
    return r
