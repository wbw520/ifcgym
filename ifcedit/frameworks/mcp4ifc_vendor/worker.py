"""IfcOpenShell worker of the Route-2 MCP4IFC port: the stand-in for the Blender process of ifc-bonsai-mcp.

Started by runtime.WorkerConnection (one per task), it reads JSON commands {"id", "type", "params"} line by line from stdin and
answers {"id", "status", "result" | "message"} on stdout - the same {status, result} envelope core.py's socket server produced -
dispatching to blender_addon.commands.execute_command (the authors' registry). Commands run on this process's main thread, so the
sandbox's SIGALRM 60 s limit in blender_addon/api/code.py works unchanged. Special commands: __open__ {path} (ifcopenshell.open),
__write__ (write the in-memory file back to its path), __quit__. Everything printed by the add-on code goes to stderr (the
worker log), stdout is reserved for the protocol.
"""
import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)   # makes `blender_addon` importable under its original name (the sandbox whitelists `blender_addon.api.*`)


def _default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    return str(o)


def main() -> None:
    out = os.fdopen(os.dup(1), "w", buffering=1)
    sys.stdout = sys.stderr
    from blender_addon import commands
    from blender_addon.api import ifc_utils
    session = None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            out.write(json.dumps({"id": None, "status": "error", "message": f"bad request: {e}"}) + "\n"); out.flush()
            continue
        rid = req.get("id"); ctype = req.get("type", ""); params = req.get("params") or {}
        try:
            if ctype == "__open__":
                session = ifc_utils.IfcSession(params["path"])
                ifc_utils.bind_session(session)
                resp = {"status": "success", "result": {"opened": params["path"], "schema": session.ifc.schema}}
            elif ctype == "__write__":
                if session is None:
                    resp = {"status": "error", "message": "No IFC file open"}
                else:
                    session.write()
                    resp = {"status": "success", "result": {"saved": session.path, "saves": session.saves}}
            elif ctype == "__quit__":
                break
            elif ctype == "ping":
                resp = {"status": "success", "result": "pong"}
            else:
                resp = commands.execute_command(ctype, params)
        except BaseException as e:  # noqa: BLE001  (SystemExit / KeyboardInterrupt from sandboxed code must not kill the worker)
            traceback.print_exc()
            resp = {"status": "error", "message": str(e)}
        resp["id"] = rid
        try:
            out.write(json.dumps(resp, default=_default) + "\n")
        except (TypeError, ValueError) as e:
            out.write(json.dumps({"id": rid, "status": "error", "message": f"result not serialisable: {e}"}) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
