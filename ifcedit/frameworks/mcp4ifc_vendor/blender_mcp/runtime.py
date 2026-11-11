"""Route-2 replacement for the FastMCP instance and the Blender socket client of ifc-bonsai-mcp (IfcEdit, 2026-11-03).

Original: api_tools.py did `@mcp.tool()` (FastMCP registration) and `get_blender_connection().send_command(cmd, params)` over a TCP
socket to the Blender add-on, whose core.py ran the command on Blender's main thread with a 30 s poll timeout.
Here: `mcp.tool()` records the function in a registry (the OpenAI schemas are built from it in schema.py) and
`get_blender_connection()` returns the WorkerConnection bound to the *current thread* - one worker subprocess per task, started by
ifcedit.frameworks.mcp4ifc.run(), so nothing about the "current IFC file" is module-level and several tasks run concurrently.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("BlenderMCPServer")

COMMAND_TIMEOUT_S = 30.0   # core.py L153: Blender main-thread poll timeout for every command
CODE_TIMEOUT_S = 60.0      # code.py execution_timeout(60): SIGALRM inside the sandbox (fires in the worker's main thread)
CODE_WAIT_MARGIN_S = 10.0  # the parent waits a little longer than the sandbox so the sandbox's own timeout message is returned
WORKER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "worker.py")


class _Registry:
    """Stand-in for FastMCP's `mcp.tool()` decorator: keeps name -> function in definition order."""

    def __init__(self):
        self.tools: Dict[str, Any] = {}

    def tool(self):
        def decorator(fn):
            self.tools[fn.__name__] = fn
            return fn
        return decorator


mcp = _Registry()
_local = threading.local()


def bind_connection(conn: Optional["WorkerConnection"]) -> None:
    _local.conn = conn


def get_blender_connection() -> "WorkerConnection":
    conn = getattr(_local, "conn", None)
    if conn is None:
        raise Exception("Could not connect to Blender. Make sure the Blender addon is running.")
    return conn


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    return str(o)


class WorkerConnection:
    """One IfcOpenShell worker process (worker.py) holding the IFC file of one task; JSON lines over stdin/stdout.

    send_command(type, params) mirrors server.py BlenderConnection.send_command: {"error": msg} on a command error, the command's
    result otherwise; a timed-out command returns {"error": "Command execution timed out"} like core.py (the worker keeps running
    it, later commands queue behind it, and its late response is discarded by id)."""

    def __init__(self, ifc_path: str, workdir: str, python: str = sys.executable):
        self.ifc_path = ifc_path
        self.log_path = os.path.join(workdir, "mcp4ifc_worker.log")
        self._log = open(self.log_path, "w")
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "2")}
        self.proc = subprocess.Popen([python, WORKER], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log, text=True,
                                     bufsize=1, cwd=workdir, env=env)
        self._q: "queue.Queue[Optional[str]]" = queue.Queue()
        self._lock = threading.Lock()
        self._n = 0
        self.dead = False
        self.timeouts = 0
        threading.Thread(target=self._reader, daemon=True).start()
        r = self.send_command("__open__", {"path": ifc_path}, timeout=180.0)
        if isinstance(r, dict) and "error" in r:
            self.close()
            raise RuntimeError(f"could not open IFC in worker: {r['error']}")
        self.schema = r.get("schema") if isinstance(r, dict) else None

    def _reader(self) -> None:
        try:
            for line in self.proc.stdout:
                self._q.put(line)
        finally:
            self._q.put(None)

    def send_command(self, command_type: str, params: Dict[str, Any] = None, timeout: Optional[float] = None) -> Any:
        if self.dead:
            raise Exception("Connection to Blender lost: worker process exited")
        if timeout is None:
            timeout = CODE_TIMEOUT_S + CODE_WAIT_MARGIN_S if command_type == "execute_ifc_code" else COMMAND_TIMEOUT_S
        with self._lock:
            self._n += 1
            rid = self._n
            try:
                self.proc.stdin.write(json.dumps({"id": rid, "type": command_type, "params": params or {}}, default=_json_default) + "\n")
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError, ValueError) as e:
                self.dead = True
                raise Exception(f"Connection to Blender lost: {e}")
            deadline = time.monotonic() + timeout
            while True:
                try:
                    line = self._q.get(timeout=max(0.0, deadline - time.monotonic()))
                except queue.Empty:
                    self.timeouts += 1
                    logger.error(f"command {command_type} timed out after {timeout} s")
                    return {"error": "Command execution timed out"}
                if line is None:
                    self.dead = True
                    raise Exception("Connection to Blender lost: worker process exited")
                try:
                    resp = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if resp.get("id") != rid:
                    continue   # late answer of a command that already timed out
                if resp.get("status") == "error":
                    return {"error": resp.get("message", "Unknown error")}
                return resp.get("result", {})

    def write_ifc(self) -> Any:
        """Final write of the in-memory file to ifc_path (IfcEdit's generous reading: edits made with execute_ifc_code_tool
        without save_and_load_ifc() are not lost)."""
        return self.send_command("__write__", {}, timeout=180.0)

    def close(self) -> None:
        try:
            if not self.dead and self.proc.poll() is None:
                try:
                    self.proc.stdin.write(json.dumps({"id": 0, "type": "__quit__", "params": {}}) + "\n"); self.proc.stdin.flush()
                except (BrokenPipeError, OSError, ValueError):
                    pass
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
            elif self.proc.poll() is None:
                self.proc.kill()
        finally:
            self.dead = True
            try:
                self.proc.stdin.close()
            except Exception:  # noqa: BLE001
                pass
            self._log.close()
