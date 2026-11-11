"""Score one RL episode with the frozen checkers. Runs under the IFC venv of the verl container (.venvs/ifc-verl):
    python -I train/rl/score_one.py <task.json> <edited.ifc> <final_text.txt>    -> one JSON line on stdout"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from ifcgym.checkers import check  # noqa: E402

spec = json.load(open(sys.argv[1]))
final_text = open(sys.argv[3]).read() if len(sys.argv) > 3 and os.path.exists(sys.argv[3]) else None
rep = check(spec["ifc"], sys.argv[2], spec, final_text)
print(json.dumps({"strict": bool(rep.strict), "validity": bool(rep.validity), "intent": bool(rep.intent), "dependency": bool(rep.dependency),
                  "preservation": bool(rep.preservation), "side_effects": int(rep.side_effects)}))
