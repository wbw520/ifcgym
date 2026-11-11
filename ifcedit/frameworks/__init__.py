"""Prior IFC-agent frameworks run as comparison rows on the IfcEdit held-out tasks (baselines B007 MCP4IFC, B008 IFC-Agent, 2026-11-03).

Every framework module exposes
    run(prompt: str, ifc_path: str, workdir: str, client: OpenAI, model: str, budget: int, max_tokens: int, temperature: float) -> dict
returning {"output_ifc": <path of the framework's final IFC file (may equal ifc_path if it edits in place)>, "final_text": <last assistant text or None>,
           "tool_calls": int, "tokens_in": int, "tokens_out": int, "outcome": "executed"|"budget"|"api"|"crash"|"other", "error": str|None, "messages": list}
The framework never sees the task specification; ifcedit.run_external copies the input IFC into workdir and scores output_ifc with the four checkers.
"""
