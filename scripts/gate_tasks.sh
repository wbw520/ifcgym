#!/bin/bash
# Task-set gate (2026-10-22): layer 1 static lint on ALL files, layer 2 scripted oracles on a sample of buildings, layer 3 independent
# mesh-based reference verifier on the same sample + all dev-val files. Run inside the python:3.12 container.
#   bash scripts/gate_tasks.sh outputs/tasks_synth2k_v7 outputs/tasks_devval_v7 [n_sample_buildings]
set -u
cd ${IFCGYM_ROOT}
SET=$1; DEV=$2; N=${3:-6}; OUT=$SET/_gate; mkdir -p $OUT
echo "== layer 1: lint (all)"; python scripts/lint_tasks.py "$SET/*.jsonl" | tee $OUT/lint_synth.log | tail -15
python scripts/lint_tasks.py "$DEV/*.jsonl" | tee $OUT/lint_devval.log | tail -8
echo "== sample of $N buildings (half tiled)"
ls $SET/synth_tiled_*.jsonl | shuf --random-source=<(yes) | head -$((N/2)) > $OUT/sample.txt
ls $SET/synth_*.jsonl | grep -v tiled | shuf --random-source=<(yes) | head -$((N - N/2)) >> $OUT/sample.txt
for f in $(cat $OUT/sample.txt); do b=$(basename $f .jsonl)
  python - "$f" "$OUT/$b.sub.jsonl" <<'PY'
import json, random, sys, collections
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]; by=collections.defaultdict(list)
for r in rows: by[r["template"]].append(r)
rng=random.Random(0); out=[rng.choice(v) for v in by.values()] + [rng.choice(v) for v in by.values() if len(v) > 1]
open(sys.argv[2],"w").write("".join(json.dumps(r)+"\n" for r in out))
PY
  (python -u scripts/oracle_v4.py $OUT/$b.sub.jsonl > $OUT/$b.oracle.log 2>&1; python -u scripts/oracle_cascade.py $OUT/$b.sub.jsonl > $OUT/$b.cascade.log 2>&1; python -u scripts/oracle_extra.py $OUT/$b.sub.jsonl > $OUT/$b.extra.log 2>&1) &
  (python -u scripts/verify_refs_v4.py $f > $OUT/$b.verify.log 2>&1) &
done
for f in $DEV/*.jsonl; do b=$(basename $f .jsonl); (python -u scripts/verify_refs_v4.py $f > $OUT/dev_$b.verify.log 2>&1; python -u scripts/oracle_v4.py $f --limit 400 > $OUT/dev_$b.oracle.log 2>&1) & done
wait
echo "== layer 2: oracles"; grep -h "^TOTAL" $OUT/*.oracle.log | awk '{print FILENAME": "$0}' FILENAME=oracle_v4; grep -h "^TOTAL\|strict" $OUT/*.cascade.log | tail -n 20 | grep -c "strict=1" ; grep -h "" $OUT/*.extra.log | grep -E "^[A-Z]-.* [0-9]+/[0-9]+" | awk '{s+=$2; split($2,a,"/"); ok+=a[1]; tot+=a[2]} END{print "oracle_extra", ok"/"tot}'
echo "   failures:"; grep -h "^FAIL\|^EXC" $OUT/*.oracle.log $OUT/*.extra.log | cut -c1-200 | head -20
echo "== layer 3: reference verifier"; grep -h "^TOTAL" $OUT/*.verify.log; echo "   mismatches:"; grep -h "^MISMATCH" $OUT/*.verify.log | cut -c1-220 | head -30
echo "== gate done"
