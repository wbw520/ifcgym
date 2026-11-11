"""Re-run the gated checkers on saved task outputs (edited/<task>.ifc) and refresh the trajectory summaries - used after a checker
rule changes during the pilot. Usage: python scripts/rescore_tasks.py <run_dir> [<run_dir> ...]  (task specs are found by task_id in outputs/tasks/*.jsonl, or in --tasks=<glob>[,<glob>] - REQUIRED for runs on other task sets:
task ids are only unique within one generator seed, so tasks_train (seed 1) ids collide with the dev ids of outputs/tasks (seed 0))"""
import glob, json, os, sys
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ifcgym.checkers import check
ROOT = "${IFCGYM_ROOT}"
SPECS = {}
AMBIGUOUS = set()

def load_specs(globs):
    for g in globs:
        for p in glob.glob(g if g.startswith("/") else f"{ROOT}/{g}"):
            for l in open(p):
                if l.strip():
                    t = json.loads(l)
                    if t["task_id"] in SPECS and SPECS[t["task_id"]]["target"] != t["target"]:
                        AMBIGUOUS.add(t["task_id"])   # same id, different task: its trajectory cannot be attributed - left untouched
                    SPECS[t["task_id"]] = t
    for i in sorted(AMBIGUOUS):
        SPECS.pop(i, None); print(f"ambiguous task id {i}: skipped")

def one(args):
    traj, edited = args
    try:
        d = json.load(open(traj))
    except Exception:  # noqa: BLE001
        return traj, "unreadable (still being written?)"
    t = SPECS.get(d["summary"]["task_id"])
    if t is None or not (os.path.exists(edited) or os.path.exists(edited + ".gz")):
        return traj, "no spec/edited"
    rep = check(t["ifc"], edited, t, d.get("final_text"))
    s = d["summary"]; old = s.get("strict")
    s.update({"strict": rep.strict, "validity": rep.validity, "intent": rep.intent, "dependency": rep.dependency, "preservation": rep.preservation,
              "under_edit": rep.under_edit, "side_effects": rep.side_effects, "new_errors": rep.new_errors, "details": rep.details, "rescored": True})
    json.dump(d, open(traj, "w"), ensure_ascii=False, indent=1, default=str)
    return traj, f"strict {old} -> {rep.strict}"

if __name__ == "__main__":
    args = sys.argv[1:]; tiers = None; force = False; task_globs = ["outputs/tasks/*.jsonl"]; projects = None
    while args and args[0].startswith("--"):
        if args[0].startswith("--projects="):
            projects = set(args[0][11:].split(","))
        elif args[0].startswith("--tasks="):
            task_globs = args[0][8:].split(",")
        elif args[0].startswith("--tiers="):
            tiers = {int(x) for x in args[0][8:].split(",")}
        elif args[0] == "--force":
            force = True
        args = args[1:]
    load_specs(task_globs); print(f"specs: {len(SPECS)} from {task_globs}")
    jobs = []
    for rd in args:
        for traj in sorted(glob.glob(os.path.join(rd, "trajectories", "*.json"))):
            tid = os.path.basename(traj)[:-5]
            if tiers is not None and (SPECS.get(tid) or {}).get("tier") not in tiers:
                continue
            if projects is not None and (SPECS.get(tid) or {}).get("project") not in projects:
                continue
            if not force and json.load(open(traj)).get("summary", {}).get("rescored"):
                continue
            jobs.append((traj, os.path.join(rd, "edited", tid + ".ifc")))
    changed = 0
    with ProcessPoolExecutor(max_workers=30) as ex:
        for traj, msg in ex.map(one, jobs):
            if "->" in msg and msg.split("->")[0].strip().split()[-1] != msg.split("->")[1].strip():
                changed += 1; print(os.path.basename(traj), msg)
    print(f"rescored {len(jobs)} trajectories, {changed} strict flags changed")
