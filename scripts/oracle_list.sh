#!/bin/bash
# Full-list solvability gate (oracles had run on samples only). Shards a task list by building and runs the three
# scripted solvers (oracle_v4 positive + negative, oracle_extra, oracle_cascade) on every shard; prints totals and the templates
# no solver checked. Run inside the python:3.12 container.   bash scripts/oracle_list.sh <tasks.jsonl> <out_dir> [n_shards=24]
set -u
cd ${IFCGYM_ROOT}
LIST=$1; OUT=$2; N=${3:-24}; BY=${4:-building}; mkdir -p $OUT     # BY=task: round-robin over tasks (one big building no longer serialises a shard)
python - "$LIST" "$OUT" "$N" "$BY" <<'PY'
import json, sys, collections
lst, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3]); by = collections.defaultdict(list)
for l in open(lst):
    if l.strip():
        by[json.loads(l)["ifc"]].append(l)
sh = [[] for _ in range(n)]
if sys.argv[4] == "task":
    for i, l in enumerate(l for k in sorted(by) for l in by[k]): sh[i % n].append(l)
else:
    for i, k in enumerate(sorted(by)): sh[i % n].extend(by[k])
for i, s in enumerate(sh): open(f"{out}/shard_{i:02d}.jsonl", "w").writelines(s)
print(f"{sum(map(len, sh))} tasks, {len(by)} files, {n} shards")
PY
for f in $OUT/shard_*.jsonl; do b=${f%.jsonl}
  (python -u scripts/oracle_v4.py $f > $b.v4.log 2>$b.v4.err; python -u scripts/oracle_v4.py $f --negative > $b.v4neg.log 2>$b.v4neg.err;
   python -u scripts/oracle_extra.py $f > $b.extra.log 2>$b.extra.err; python -u scripts/oracle_extra.py $f --negative > $b.extraneg.log 2>$b.extraneg.err; python -u scripts/oracle_cascade.py $f > $b.cascade.log 2>$b.cascade.err) &
done; wait
python - "$OUT" "$LIST" <<'PY'
import glob, json, re, sys, collections
out, lst = sys.argv[1], sys.argv[2]
tpl_of = {}
for l in open(lst):
    if l.strip():
        t = json.loads(l); tpl_of[t["task_id"]] = t["template"]
tot = {k: collections.Counter() for k in ("v4", "v4neg", "extra", "extraneg", "cascade")}; checked = collections.Counter()
for k in tot:
    for p in glob.glob(f"{out}/shard_*.{k}.log"):
        for l in open(p):
            if k == "cascade":     # per-task lines '<task_id> strict=True ...' (only the T1-T3 move templates are its job)
                m = re.match(r"^(\S+)\s+(?:strict=(True|False)|EXCEPTION)", l)
                if m and tpl_of.get(m.group(1), "").split("-")[0] in ("T1", "T2", "T3"):
                    t = tpl_of[m.group(1)]; tot[k][(t, "ok")] += m.group(2) == "True"; tot[k][(t, "n")] += 1; checked[t] += 1
                continue
            m = re.match(r"^(\S+)\s+(\d+)/(\d+)\s*$", l.strip())
            if m and m.group(1) != "TOTAL":
                tot[k][(m.group(1), "ok")] += int(m.group(2)); tot[k][(m.group(1), "n")] += int(m.group(3)); checked[m.group(1)] += int(m.group(3))
for k, c in tot.items():
    ok = sum(v for (t, s), v in c.items() if s == "ok"); n = sum(v for (t, s), v in c.items() if s == "n")
    bad = sorted((t, c[(t, "ok")], c[(t, "n")]) for (t, s) in list(c) if s == "n" and c[(t, "ok")] != c[(t, "n")])
    print(f"{k:8s} {ok}/{n}" + "".join(f"\n   {t}: {o}/{m}" for t, o, m in bad))
tpl = collections.Counter(tpl_of.values())
miss = sorted((t, n) for t, n in tpl.items() if checked[t] == 0)
print(f"templates with no solver check: {len(miss)} ({sum(n for _, n in miss)} tasks)" + "".join(f"\n   {t} ({n})" for t, n in miss))
PY
