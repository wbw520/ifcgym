#!/bin/bash
# Full-list reference verification (review 2026-10-24: gates verified samples only; 3 wrong teacher tasks sat in unsampled buildings).
# Sorts a task list by building, shards it, runs scripts/verify_refs_v4.py on every shard in parallel, writes the mismatching task ids
# to <out>/mismatch_ids.txt (exclude them from SFT / RL). Run inside the python:3.12 container.
#   bash scripts/verify_list.sh <tasks.jsonl> <out_dir> [n_shards=24]
set -u
cd ${IFCGYM_ROOT}
LIST=$1; OUT=$2; N=${3:-24}; mkdir -p $OUT
python - "$LIST" "$OUT" "$N" <<'PY'
import json, sys, collections
lst, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3]); by = collections.defaultdict(list)
for l in open(lst):
    if l.strip():
        by[json.loads(l)["ifc"]].append(l)
sh = [[] for _ in range(n)]
for i, k in enumerate(sorted(by)): sh[i % n].extend(by[k])
for i, s in enumerate(sh): open(f"{out}/shard_{i:02d}.jsonl", "w").writelines(s)
print(f"{sum(map(len, sh))} tasks, {len(by)} files, {n} shards")
PY
for f in $OUT/shard_*.jsonl; do (python -u scripts/verify_refs_v4.py $f > ${f%.jsonl}.log 2>&1) & done; wait
grep -h '^TOTAL' $OUT/shard_*.log | awk '{c+=$3; m+=$5; u+=$7} END{print "TOTAL checked", c, "mismatches", m, "unhandled", u}' | tee $OUT/summary.txt
grep -l 'Traceback' $OUT/shard_*.log && echo "WARNING: a shard crashed" | tee -a $OUT/summary.txt
grep -h '^MISMATCH' $OUT/shard_*.log | sed -E 's/^MISMATCH ([^:]+):.*/\1/' | sort -u > $OUT/mismatch_ids.txt; echo "mismatch ids: $(wc -l < $OUT/mismatch_ids.txt) -> $OUT/mismatch_ids.txt"
