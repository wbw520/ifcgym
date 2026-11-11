"""IfcEdit episodes for verl's agent-loop rollout: multi-turn GRPO of the IFC-editing student.

One sample of the RL dataset = one generated task (extra_info.task = the task record). The loop plays the evaluation episode of
protocol v3 (ifcedit.agent.run_episode) token by token inside verl's rollout:

    prompt   = SYSTEM_PROMPT_V2 + task prompt, rendered with the TOOLS_V2 schemas by the model's chat template
    response = assistant turn (mask 1) | tool outputs (mask 0) | assistant turn (mask 1) | ... | final answer (mask 1)

Same tools (execute_ifc_code / inspect_element / find_elements / check_model as code in the sandbox), budget of 20 tool rounds,
budget note at 3 and 1 rounds left, temperature from the trainer. Differences to the evaluation: the context
is the full token stream (no pruning of old tool outputs - verl trains one sequence per episode), the episode also ends when the
trajectory token budget or EPISODE_MAX_S wall time is used up, and the sandbox jail hides /proc instead of mounting a private one
(the verl container does not allow a new procfs).

Reward = 1 if the frozen checkers call the committed result strictly correct, else 0 (train/rl/score_one.py, IFC venv). A
scorer failure is an environment failure: no gradient (mask 0) and reward 0.
Reward ablation: IFCGYM_REWARD=intent_validity rewards intent AND validity only (the dependency and preservation gates
are dropped from the reward; the checkers still run and every term is logged per episode). Default IFCGYM_REWARD=strict.

Environment variables (read in every rollout worker):
  IFCGYM_IFC_PYTHON  interpreter with ifcopenshell for the sandbox and the scorer (default .venvs/ifc-verl/bin/python)
  IFCGYM_RL_WORK     scratch root for the episode work dirs and the episode log (episodes.jsonl)
  IFCGYM_REWARD      strict (default) | intent_validity (reward ablation, see above)
  IFCGYM_EPISODE_MAX_S  wall-time cap per episode (default 600 s; the pre-filter's longest episode took 599 s)
  IFCGYM_POOL_THREADS   threads for the blocking sandbox / scorer calls (default 512)
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import shutil
import subprocess
import tempfile
import time
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.tools.schemas import OpenAIFunctionToolSchema

from ifcedit.agent import BUDGET_NOTE, BUDGET_NOTE_AT, CHECK_CODE, TOOLS_V2, _unwrap_args, find_code, inspect_code
from ifcgym.sandbox import ISOLATE, ZYGOTE, Sandbox

CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
IFC_PY = os.environ.get("IFCGYM_IFC_PYTHON", "${IFCGYM_ROOT}/.venvs/ifc-verl/bin/python")
WORK = os.environ.get("IFCGYM_RL_WORK", os.path.join(CODE, "outputs/rl/rl_work"))
EPISODE_MAX_S = float(os.environ.get("IFCGYM_EPISODE_MAX_S", "600"))
REWARD_MODE = os.environ.get("IFCGYM_REWARD", "strict")
assert REWARD_MODE in ("strict", "intent_validity"), REWARD_MODE
BUDGET = 20
TURN_MAX_TOKENS = 16384
_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=int(os.environ.get("IFCGYM_POOL_THREADS", "512")))

# jail without a pid namespace (the verl container refuses a new procfs): /proc is hidden under an empty tmpfs instead
ISOLATE_NOPROC = ISOLATE.replace('cd $W; exec "$@"', 'mount -t tmpfs tmpfs /proc; cd $W; exec "$@"')
# without a pid namespace the call children would outlive a killed zygote: they get PR_SET_PDEATHSIG = SIGKILL
ZYGOTE_RL = ZYGOTE.replace('            fd = os.open(req["out"]', '            import ctypes; ctypes.CDLL("libc.so.6").prctl(1, 9)\n            fd = os.open(req["out"]')
assert ISOLATE_NOPROC != ISOLATE and ZYGOTE_RL != ZYGOTE


class RLSandbox(Sandbox):
    """Sandbox of protocol v3 with the verl-container jail (no pid namespace) and the IFC venv interpreter."""

    def __post_init__(self) -> None:
        super().__post_init__()
        with open(self._zygote, "w") as f:
            f.write(ZYGOTE_RL)

    def _start(self) -> None:
        code_dir = CODE
        cmd = ["unshare", "--user", "--map-root-user", "--mount", "--net", "--fork", "--kill-child",
               "bash", "-c", ISOLATE_NOPROC, "_", os.path.abspath(self.workdir), os.path.dirname(os.path.dirname(os.path.abspath(self.python))),
               os.path.join(code_dir, "ifcedit"), os.path.abspath(self.input_ifc), self.python, "-I", self._zygote, code_dir, self._worker]
        self._z = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                   cwd=self.workdir, start_new_session=True)
        self._code_dir = code_dir


def _tool_code(name: str, arguments: str) -> tuple[str | None, str | None]:
    """(code, error) for one parsed tool call - the mapping of ifcedit.agent.run_episode."""
    try:
        args = _unwrap_args(json.loads(arguments or "{}"))
    except json.JSONDecodeError:
        return arguments or "", None
    if name == "inspect_element" and isinstance(args, dict):
        return (inspect_code(args), None) if args.get("guid") else (None, 'inspect_element needs {"guid": "<GlobalId>"}')
    if name == "find_elements" and isinstance(args, dict):
        return find_code(args), None
    if name == "check_model":
        return CHECK_CODE, None
    if isinstance(args, dict):
        code = args.get("code")
        if not isinstance(code, str) or not code.strip():
            return None, 'execute_ifc_code needs {"code": "<python source>"}; got keys ' + str(list(args))
        return code, None
    return str(args), None


def _score(task: dict, wd: str, edited: str, final_text: str) -> dict | None:
    tp = os.path.join(wd, "task.json"); fp = os.path.join(wd, "final.txt")
    json.dump(task, open(tp, "w")); open(fp, "w").write(final_text or "")
    try:
        p = subprocess.run([IFC_PY, "-I", os.path.join(CODE, "train/rl/score_one.py"), tp, edited, fp], capture_output=True, text=True, timeout=600)
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001 - scorer crash / timeout: environment failure
        return None


@register("ifcedit_agent")
class IfcEditAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.response_length = self.rollout_config.response_length
        self.im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.parser = ToolParser.get_tool_parser("qwen3_coder", self.tokenizer)
        self.schemas = [OpenAIFunctionToolSchema.model_validate(t) for t in TOOLS_V2]
        dummy = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
        pre = self.tokenizer.apply_chat_template(dummy, tokenize=False)
        self._dummy, self._dummy_pre = dummy, pre[: pre.rfind("<|im_end|>") + len("<|im_end|>")]

    def _enc(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _prompt_ids(self, messages: list[dict]) -> list[int]:
        """The evaluation server's rendering (vLLM chat template, tools, generation prompt) - text path, not verl's processor path."""
        return self._enc(self.tokenizer.apply_chat_template(messages, tools=TOOLS_V2, add_generation_prompt=True, tokenize=False))

    def _tool_turn_ids(self, tool_msgs: list[dict]) -> list[int]:
        """Tokens between the assistant's <|im_end|> and the next assistant turn, exactly as the chat template renders a whole
        conversation (Qwen3.5 opens the tool block with <|im_start|>user only after a non-tool message, so the tool messages are
        rendered behind a dummy assistant turn and cut off after its <|im_end|>)."""
        full = self.tokenizer.apply_chat_template(self._dummy + tool_msgs, add_generation_prompt=True, tokenize=False)
        assert full.startswith(self._dummy_pre)
        return self._enc(full[len(self._dummy_pre):])

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        task = json.loads(kwargs["extra_info"]["task"]); t0 = time.time()
        os.makedirs(os.path.join(WORK, "tmp"), exist_ok=True)
        wd = tempfile.mkdtemp(prefix=f"ep_{task['task_id'][-40:]}_", dir=os.path.join(WORK, "tmp"))
        sb = RLSandbox(input_ifc=task["ifc"], workdir=os.path.join(wd, "sbx"), python=IFC_PY, timeout_s=120.0, helpers=True, isolate=True)
        request_id = uuid4().hex
        metrics = {"generate_sequences": 0.0, "tool_calls": 0.0}
        prompt_ids = await self.loop.run_in_executor(None, self._prompt_ids, list(kwargs["raw_prompt"]))
        ids = list(prompt_ids); mask: list[int] = []; logps: list[float] = []; have_lp = False

        def add(tokens: list[int], m: int, lp=None):
            nonlocal have_lp
            ids.extend(tokens); mask.extend([m] * len(tokens))
            if lp is not None and len(lp) == len(tokens):
                logps.extend(lp); have_lp = True
            else:
                logps.extend([0.0] * len(tokens))

        rounds = commits = 0; end = "answer"; final_text = ""; extra_fields: dict[str, Any] = {}
        while True:
            room = self.response_length - len(mask)
            if room < 1024:
                end = "tokens"; break
            if time.time() - t0 > EPISODE_MAX_S:
                end = "time"; break
            g0 = time.time()
            out = await self.server_manager.generate(request_id=request_id, prompt_ids=ids,
                                                     sampling_params={**sampling_params, "max_tokens": min(TURN_MAX_TOKENS, room - 512)})
            metrics["generate_sequences"] += time.time() - g0
            if not extra_fields:
                extra_fields.update(out.extra_fields)
            toks = list(out.token_ids)
            add(toks, 1, out.log_probs)
            if not toks or toks[-1] != self.im_end:            # cut at max_tokens: the evaluation ends such an episode too
                end = "truncated"; break
            content, calls = await self.parser.extract_tool_calls(toks, self.schemas)
            if not calls:
                text = self.tokenizer.decode(toks, skip_special_tokens=True)
                final_text = text.rsplit("</think>", 1)[-1].strip()
                break
            rounds += 1; outs = []; crashed = False
            for c in calls:
                code, err = _tool_code(c.name, c.arguments)
                if err:
                    outs.append("Error: malformed tool call - " + err); continue
                c0 = time.time()
                r = await self.loop.run_in_executor(_POOL, sb.execute, code)
                metrics["tool_calls"] += time.time() - c0
                commits += bool(r.committed); crashed |= bool(r.crashed)
                outs.append(r.output)
            left = BUDGET - rounds
            if left in BUDGET_NOTE_AT and not crashed:
                outs[-1] += "\n\n" + BUDGET_NOTE.format(left=left, s="" if left == 1 else "s")
            tool_ids = await self.loop.run_in_executor(None, self._tool_turn_ids, [{"role": "tool", "content": o} for o in outs])
            if len(mask) + len(tool_ids) >= self.response_length - 1024:
                end = "tokens"; break
            add(tool_ids, 0)
            if crashed:
                end = "crash"; break
            if rounds >= BUDGET:
                end = "budget"; break
        await self.loop.run_in_executor(_POOL, sb.close)
        edited = os.path.join(wd, "edited.ifc"); shutil.copyfile(sb.final_path(), edited)
        s0 = time.time()
        rep = await self.loop.run_in_executor(_POOL, _score, task, wd, edited, final_text)
        score_s = time.time() - s0
        if rep is None:                       # scorer failure: no learning signal from this episode
            reward = 0.0; mask = [0] * len(mask)
        else:
            ok = rep["strict"] if REWARD_MODE == "strict" else (rep["intent"] and rep["validity"])
            reward = 1.0 if ok else 0.0
        # "strict" carries the training reward (it is the group-filter metric); checker_strict is the frozen strict verdict itself
        info = {"strict": reward, "checker_strict": float(bool(rep and rep["strict"])), "rounds": float(rounds), "commits": float(commits), "no_commit": float(commits == 0),
                "end_budget": float(end == "budget"), "end_time": float(end == "time"), "end_tokens": float(end in ("tokens", "truncated")),
                "end_crash": float(end == "crash"), "env_fail": float(rep is None), "episode_s": time.time() - t0}
        try:
            with open(os.path.join(WORK, "episodes.jsonl"), "a") as f:
                f.write(json.dumps({"task_id": task["task_id"], "template": task["template"], "reward": reward, "reward_mode": REWARD_MODE, "end": end, "rounds": rounds,
                                    "commits": commits, "tokens": len(mask), "seconds": round(time.time() - t0, 1), "score_s": round(score_s, 1),
                                    **({k: rep[k] for k in ("intent", "dependency", "preservation", "validity")} if rep else {"env_fail": True})}) + "\n")
        except OSError:
            pass
        shutil.rmtree(wd, ignore_errors=True)
        if not mask:
            add([self.im_end], 0)
        response_ids = ids[len(prompt_ids):][: self.response_length]
        output = AgentLoopOutput(prompt_ids=prompt_ids, response_ids=response_ids, response_mask=mask[: self.response_length],
                                 response_logprobs=logps[: self.response_length] if have_lp else None,
                                 reward_score=reward, num_turns=2 * rounds + 2, metrics=metrics, extra_fields=extra_fields)
        output.extra_fields["reward_extra_info"] = info
        return output
