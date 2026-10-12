#!/usr/bin/env bash
# Frozen fixed-pattern noise (DSNU + PRNU) suite on EuRoC MH_01_easy.
#
# Conditions (independent patterns per camera, drawn once per seed):
#   clean           no injection -- fresh baselines in the SAME batch, so the
#                   comparison is not confounded by machine-load drift since
#                   the September baselines
#   realistic_low   dsnu 0.025 gray, prnu 0.5%   modern EMVA-1288-measured
#                   global-shutter parts (DSNU ~1.9 e-, PRNU 0.3-0.5%) mapped to
#                   8-bit at ~78 e-/DN unity gain
#   realistic_high  dsnu 0.5 gray,   prnu 2.0%   older-CMOS upper bound
#                   (Konnik & Welsh PRNU 1-2%), DSNU allowing for analog gain
#   stress          dsnu 10 gray,    prnu 10%    POSITIVE CONTROL -- proves the
#                   injector reaches ORB-SLAM3, so a null at realistic levels
#                   cannot be explained away as a broken pipeline
#
# Runs are interleaved round-robin by seed so slow drift in machine conditions
# spreads evenly across conditions instead of landing on one of them.
# Imagery is ephemeral: generated, consumed, deleted. Patterns + manifest +
# trajectory are kept (a few MB per run).
set -uo pipefail
EXP=~/repos/oasis-experiments
SLAM=~/repos/ORB_SLAM3
SRC=/data/tkocher/datasets/euroc/MH_01_easy
GT="$SRC/mav0/state_groundtruth_estimate0/data.csv"
SCRATCH_ROOT=/data/tkocher/datasets/euroc_noisy
RESULTS=/data/tkocher/oasis_results
CFG=EuRoC_oasis.yaml
LOG="$RESULTS/fpn_suite_results.csv"
SEEDS="${SEEDS:-0 1 2 3 4}"
STRESS_SEEDS="${STRESS_SEEDS:-0 1 2}"

declare -A ARGS=(
  [realistic_low]="--dsnu 0.025 --prnu 0.005"
  [realistic_high]="--dsnu 0.5 --prnu 0.02"
  [stress]="--dsnu 10 --prnu 0.10"
)

[ -f "$LOG" ] || echo "run_id,condition,seed,ate_m,valid,gen_s,slam_s" > "$LOG"

run_one () {   # $1=condition $2=seed
  local cond=$1 seed=$2
  local run_id="fpn_${cond}_seed$(printf '%02d' "$seed")"
  local data="$SRC" gen=0
  local t0 t1
  if [ "$cond" != clean ]; then
    data="$SCRATCH_ROOT/_scratch_${run_id}"
    t0=$(date +%s)
    python3 "$EXP/inject_image_noise.py" --src "$SRC" --dst "$data" \
        --seed "$seed" ${ARGS[$cond]} --link-imu --force > /dev/null || {
      echo "$run_id,$cond,$seed,NA,GEN_FAILED,0,0" >> "$LOG"; return; }
    gen=$(( $(date +%s) - t0 ))
  fi
  t1=$(date +%s)
  ( cd "$SLAM" && ./run_oasis.sh "$data" MH01 "$run_id" "$CFG" > /dev/null 2>&1 )
  local slam=$(( $(date +%s) - t1 ))
  local out="$RESULTS/$run_id" ate=NA valid=0
  # Keep the provenance (manifest + frozen fields) next to the trajectory.
  if [ "$cond" != clean ]; then
    cp "$data/NOISE_MANIFEST.json" "$out/" 2>/dev/null
    cp -r "$data/patterns" "$out/" 2>/dev/null
    rm -rf "$data"
  fi
  local f; f=$(ls "$out"/f_*.txt 2>/dev/null | head -1)
  if [ -n "$f" ] && [ "$(wc -l < "$f")" -gt 100 ]; then
    ate=$(python3 "$EXP/eval_ate.py" "$f" "$GT" --quiet 2>/dev/null) && valid=1
  fi
  echo "$run_id,$cond,$seed,$ate,$valid,$gen,$slam" >> "$LOG"
  echo "$(date +%H:%M:%S)  $run_id  ATE=$ate  valid=$valid  (gen ${gen}s, slam ${slam}s)"
}

for s in $SEEDS; do
  run_one clean "$s"
  run_one realistic_low "$s"
  run_one realistic_high "$s"
  case " $STRESS_SEEDS " in *" $s "*) run_one stress "$s" ;; esac
done
echo "=== fpn suite complete $(date +%H:%M:%S) ==="
