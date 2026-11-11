#!/bin/bash
# Regenerate generator tasks for every synthetic building (generator v3: + relaxed tier-1 wall move; v4: BIM-Edit/BIBIMBAP grid) and the real dev-val lists.
# On a held GPU node: SLURM_JOB_ID=<ac-k3distill id> srun --jobid=$SLURM_JOB_ID --overlap -w <node> -N1 -n1 -c 224 --container-image=/store/sr1/container_cache/python/python:3.12.sqsh --container-mounts=/lustre:/lustre,/store:/store --no-container-mount-home bash -c "export PATH=${IFCGYM_ROOT}/.venvs/ifc-audit/bin:\$PATH; bash studies/ifcedit/code/scripts/regen_synth_tasks.sh outputs/tasks_synth2k_v7 <shard> <nshards>"
#   bash ${IFCGYM_ROOT}/scripts/py312.sh -c 32 -t 03:00:00 bash studies/ifcedit/code/scripts/regen_synth_tasks.sh outputs/tasks_synth2k_v4
set -u
cd ${IFCGYM_ROOT}
OUT=${1:-outputs/tasks_synth2k_v4}; SHARD=${2:-0}; NSHARD=${3:-1}; PAR=${PAR:-32}; mkdir -p "$OUT"
export PYTHONPATH=$PWD
ROOT=${ROOT:-${IFCGYM_ROOT}/data/synth/buildings2k}   # v8: ROOT=.../buildings2k_v8 (label-fixed copies, scripts/fix_external_psets.py)
SEL=${IFCGYM_ROOT}/data/synth/buildings2k/selected.txt   # scripts/select_buildings.py (user: ~3000 buildings)
sed "s#^#$ROOT/#" $SEL | awk -v s=$SHARD -v n=$NSHARD 'NR % n == s' | xargs -P $PAR -I{} bash -c '
  b={}; id=$(basename $b); [ -s '"$OUT"'/$id.jsonl ] && exit 0
  python -B -m ifcedit.tasks.generate --ifc $b/arc.ifc --project $id --out '"$OUT"'/$id.jsonl --per-template 2 --per-template-move 6 --seed 1 --perpendicular --families create retrieve spatial update delete create2 retrieve2 concept > '"$OUT"'/$id.log 2>&1 || echo "FAIL $id"'
DEVOUT=${DEVOUT:-outputs/tasks_devval_v7}; mkdir -p $DEVOUT; [ "$SHARD" = 0 ] && for proj in fzk_house wbdg_office; do for s in 0 2; do
  python -B -m ifcedit.tasks.generate --ifc ${IFCGYM_ROOT}/data/ifc-bench/$proj/arc.ifc --project $proj --out $DEVOUT/${proj}_s$s.jsonl --per-template 40 --seed $s --perpendicular --families create retrieve spatial update delete create2 retrieve2 concept > $DEVOUT/${proj}_s$s.log 2>&1
done; done; wait
echo DONE; ls "$OUT" | grep -c jsonl
