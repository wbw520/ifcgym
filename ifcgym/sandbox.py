"""IFC CodeAct sandbox: the single tool `execute_ifc_code(code)` of BIM-Edit (arXiv:2606.20146, Appendix F.4).

Semantics (as published): the code runs in a subprocess with the IFC model pre-loaded as `ifc` (ifcopenshell.file); also available
`ifcopenshell`, `api` (ifcopenshell.api), `util` (ifcopenshell.util), `element_util` (ifcopenshell.util.element), `guid`
(ifcopenshell.guid). Assign to `result` to return data. `commit()` saves modifications. 120 s timeout per call.
Our choice (DESIGN.md, 2026-10-19): every call starts from the last COMMITTED file in a fresh namespace, so uncommitted changes are
discarded (BIM-Edit Appendix H describes discarded rounds); stdout and the repr of `result` are returned, exceptions are returned as
text. Each commit writes a numbered snapshot next to the working file so trajectories can be replayed and rolled back.
isolate=True (protocol v3, 2026-10-28): the worker runs in its own user / mount / pid / net namespaces and sees only its work dir,
the helper package and the interpreter (read-only) and the input model as original.ifc (read-only) - E010 / E011 students had
walked the file system and opened other tasks' BIM-Edit ground-truth files.
"""
from __future__ import annotations

import json
import os
import select
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field

WORKER = r'''
import sys, json, io, traceback, contextlib
import ifcopenshell, ifcopenshell.api, ifcopenshell.util, ifcopenshell.util.element, ifcopenshell.guid
work = sys.argv[1]; code = open(sys.argv[2]).read(); max_out = int(sys.argv[3])
pending = work + ".pending"            # working state carried from call to call (2026-10-23: a call continues from the previous call;
import os                              # a call that raises is rolled back; commit() saves the working state as the result)
h = None
if len(sys.argv) > 4 and sys.argv[4]:          # tools v2: the helper module (ifcedit/ifctools.py) as `h`
    sys.path.insert(0, sys.argv[4])
    import ifcgym.ifctools as h
ifc = ifcopenshell.open(pending if os.path.exists(pending) else work)
state = {"committed": False}
def commit():
    ifc.write(work); ifc.write(pending); state["committed"] = True
if h is not None:
    h.commit = commit                      # models keep guessing h.commit() / ifc.commit(): accept both
try:
    ifc.commit = commit
except Exception:
    pass
ns = {"ifc": ifc, "ifcopenshell": ifcopenshell, "api": ifcopenshell.api, "util": ifcopenshell.util,
      "element_util": ifcopenshell.util.element, "guid": ifcopenshell.guid, "commit": commit, "result": None, "h": h}
buf = io.StringIO(); err = None
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    try:
        exec(compile(code, "<agent>", "exec"), ns)
    except BaseException as e:  # noqa: BLE001
        err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
out = buf.getvalue()
if err is None:                        # successful call: its edits persist for the next call (not yet the result until commit())
    try:
        ifc.write(pending)
    except Exception as e:  # noqa: BLE001
        err = "could not persist the working state: %r" % (e,)
res = ns.get("result")
try:
    res_s = "" if res is None else (json.dumps(res, ensure_ascii=False, default=str) if not isinstance(res, str) else res)
except Exception:  # noqa: BLE001
    res_s = repr(res)
text = out
if res_s: text += ("\n" if text and not text.endswith("\n") else "") + "result: " + res_s
if err: text += ("\n" if text and not text.endswith("\n") else "") + err
if len(text) > max_out: text = text[: max_out // 2] + "\n... [truncated %d chars] ...\n" % (len(text) - max_out) + text[-max_out // 2 :]
print(json.dumps({"output": text, "committed": state["committed"], "error": err is not None}))
'''


# Jail: bind the allowed paths aside, hide every shared file system under an empty tmpfs, bind the allowed paths back (read-only except
# the work dir), mount the input over work/original.ifc, then exec the worker. Args: work_dir interpreter_root helper_pkg input cmd...
ISOLATE = r'''
set -e
W=$1; PY=$2; PKG=$3; IN=$4; shift 4
S=$(mktemp -d /tmp/sbx.XXXXXX); mount -t tmpfs tmpfs $S; mkdir $S/py $S/pkg $S/work; touch $S/input
mount --bind $PY $S/py; mount --bind $W $S/work; mount --bind $IN $S/input; [ -n "$PKG" ] && mount --bind $PKG $S/pkg
for d in /store /lustre /dev/shm /raid /home /mnt /scratch; do [ -d $d ] && mount -t tmpfs tmpfs $d || true; done
mkdir -p $PY $W; mount --bind $S/py $PY; mount -o remount,bind,ro $PY
if [ -n "$PKG" ]; then mkdir -p $PKG; mount --bind $S/pkg $PKG; mount -o remount,bind,ro $PKG; fi
mount --bind $S/work $W; mount --bind $S/input $W/original.ifc; mount -o remount,bind,ro $W/original.ifc
cd $W; exec "$@"
'''


# Zygote (2026-10-28, user: use the CPUs): one process per episode imports ifcopenshell / numpy / the helpers once, then forks a child
# per tool call that runs WORKER exactly as the old fresh `python -I _worker.py` did (fresh copy of the interpreter state, the model
# re-opened from disk, roll back on error, crash and timeout confined to the child). Saves ~1 s of interpreter start + imports per call.
# Protocol on stdin / stdout: one JSON request per line {"argv": [...], "out": path, "timeout": s} -> one JSON reply {"rc": int, "timeout": bool}.
ZYGOTE = r'''
import sys, os, json, time, signal
import ifcopenshell, ifcopenshell.api, ifcopenshell.util, ifcopenshell.util.element, ifcopenshell.guid, numpy
code_dir = sys.argv[1] if len(sys.argv) > 1 else ""
if code_dir:
    sys.path.insert(0, code_dir)
    import ifcgym.ifctools
    sys.path.remove(code_dir)
src = open(sys.argv[2]).read(); worker_path = sys.argv[2]
for line in sys.stdin:
    req = json.loads(line)
    pid = os.fork()
    if pid == 0:
        try:
            fd = os.open(req["out"], os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o660); os.dup2(fd, 1); os.dup2(fd, 2)
            import random; random.seed(); numpy.random.seed()
            sys.argv = [worker_path] + req["argv"]
            exec(compile(src, worker_path, "exec"), {"__name__": "__main__", "__file__": worker_path})
            sys.stdout.flush(); sys.stderr.flush()
        except BaseException:
            import traceback; traceback.print_exc(); sys.stderr.flush(); os._exit(1)
        os._exit(0)
    t_end = time.time() + req["timeout"]; rc = None; timeout = False
    while True:
        r, status = os.waitpid(pid, os.WNOHANG)
        if r:
            rc = os.waitstatus_to_exitcode(status); break
        if time.time() > t_end:
            os.kill(pid, signal.SIGKILL); os.waitpid(pid, 0); timeout = True; break
        time.sleep(0.005)
    sys.stdout.write(json.dumps({"rc": rc, "timeout": timeout}) + "\n"); sys.stdout.flush()
'''


@dataclass
class ToolResult:
    output: str
    committed: bool
    error: bool
    seconds: float
    timeout: bool = False
    crashed: bool = False


@dataclass
class Sandbox:
    """One task: a working copy of the input IFC, executed by `execute(code)`."""

    input_ifc: str
    workdir: str
    python: str = sys.executable
    timeout_s: float = 120.0
    max_output_chars: int = 20000
    helpers: bool = False          # tools v2: expose ifcgym.ifctools as `h` inside the worker
    isolate: bool = False          # protocol v3: namespace jail (ISOLATE below)
    snapshots: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        os.makedirs(self.workdir, exist_ok=True)
        self.work = os.path.join(self.workdir, "work.ifc")
        shutil.copyfile(self.input_ifc, self.work)
        self._worker = os.path.join(self.workdir, "_worker.py")
        with open(self._worker, "w") as f:
            f.write(WORKER)
        with open(os.path.join(self.workdir, "original.txt"), "w") as f:   # for the check_model tool (h.check_model vs the original)
            f.write("original.ifc" if self.isolate else os.path.abspath(self.input_ifc))
        if self.isolate:
            open(os.path.join(self.workdir, "original.ifc"), "w").close()     # mount point of the read-only input inside the jail
        self._zygote = os.path.join(self.workdir, "_zygote.py")
        with open(self._zygote, "w") as f:
            f.write(ZYGOTE)
        self._z = None
        self.calls = 0

    def _start(self) -> None:
        code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if self.helpers else ""
        cmd = [self.python, "-I", self._zygote, code_dir, self._worker]
        if self.isolate:
            cmd = ["unshare", "--user", "--map-root-user", "--mount", "--pid", "--fork", "--kill-child", "--mount-proc", "--net",
                   "bash", "-c", ISOLATE, "_", os.path.abspath(self.workdir), os.path.dirname(os.path.dirname(os.path.abspath(self.python))),
                   os.path.join(code_dir, "ifcedit") if code_dir else "", os.path.abspath(self.input_ifc)] + cmd
        self._z = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                   cwd=self.workdir, start_new_session=True)
        self._code_dir = code_dir

    def close(self) -> None:
        z, self._z = self._z, None
        if z is None:
            return
        try:
            z.stdin.close(); z.wait(timeout=10)
        except Exception:  # noqa: BLE001
            try:
                os.killpg(z.pid, 9)
            except Exception:  # noqa: BLE001
                pass
            z.wait()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    def execute(self, code: str) -> ToolResult:
        self.calls += 1
        code_path = os.path.join(self.workdir, f"call_{self.calls:03d}.py")
        with open(code_path, "w") as f:
            f.write(textwrap.dedent(code))
        t0 = time.time()
        if self._z is None or self._z.poll() is not None:
            self._start()
        out_path = os.path.join(self.workdir, f"call_{self.calls:03d}.out")
        req = {"argv": [self.work, code_path, str(self.max_output_chars), self._code_dir], "out": out_path, "timeout": self.timeout_s}
        reply = None
        try:
            self._z.stdin.write(json.dumps(req) + "\n"); self._z.stdin.flush()
            r, _, _ = select.select([self._z.stdout], [], [], self.timeout_s + 60)
            line = self._z.stdout.readline() if r else ""
            reply = json.loads(line) if line else None
        except (OSError, ValueError):
            reply = None
        if reply is None:               # zygote dead or hung: restart it for the next call
            dead = self._z.poll() is not None
            self.close()
            if dead:
                return ToolResult(output=f"Error: worker crashed (sandbox process exited)", committed=False, error=True,
                                  seconds=time.time() - t0, crashed=True)
            return ToolResult(output=f"Error: tool call exceeded the {self.timeout_s:.0f} s timeout", committed=False, error=True,
                              seconds=time.time() - t0, timeout=True)
        if reply["timeout"]:
            return ToolResult(output=f"Error: tool call exceeded the {self.timeout_s:.0f} s timeout", committed=False, error=True,
                              seconds=time.time() - t0, timeout=True)
        try:
            text = open(out_path, errors="replace").read()
        except OSError:
            text = ""
        p = type("P", (), {"stdout": text, "stderr": "", "returncode": reply["rc"]})()
        dt = time.time() - t0
        line = (p.stdout.strip().splitlines() or [""])[-1]
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            tail = (p.stdout + p.stderr)[-self.max_output_chars:]
            return ToolResult(output=(tail + "\n" if tail else "") + f"Error: worker crashed (exit code {p.returncode})", committed=False,
                              error=True, seconds=dt, crashed=True)
        if d["committed"]:
            snap = os.path.join(self.workdir, f"snapshot_{len(self.snapshots) + 1:03d}.ifc")
            shutil.copyfile(self.work, snap)
            self.snapshots.append(snap)
        return ToolResult(output=d["output"], committed=d["committed"], error=d["error"], seconds=dt)

    def final_path(self) -> str:
        return self.work
