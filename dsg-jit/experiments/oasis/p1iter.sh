#!/usr/bin/env bash
# Phase 1.4b on lambda: re-estimate with the IMU interval gate (solver.py IMU_GATE), then an iterated closed
# loop on the tier-2/3 MH01/MH03 runs:
#   round 1  perturb -> correct with the gated estimate of the x* run -> ORB-SLAM3 with export  -> xd*
#   round 2  re-estimate on the xd* export (same oracle group) -> correct the round-1 sequence again
#            -> ORB-SLAM3 -> xe*
# The exported states of a perturbed run sit in the perturbation's basin (a 30 ms offset estimates as ~20 ms),
# so one correction leaves a residual; round 2 measures how much of it a second pass removes.
# Resumable (skips runs with record.json / estimates). Log: /data/tkocher/p1export/p1iter.log
set -u
L=/data/tkocher/p1export; O=/data/tkocher/oasis_results; X=~/repos/oasis-experiments
E=/data/tkocher/datasets/euroc; SCR=/data/tkocher/datasets/euroc_noisy; BASE=$O/_configs/EuRoC.yaml
V=/data/tkocher/venv-p1; PY="$V/bin/python"
export PYTHONPATH=$L/pkg OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2"
stage() { echo "=== $(date '+%F %T') $*"; }
SPECS="MH_01_easy:MH01:3682 MH_03_medium:MH03:2700"
ARMS="toff rbc tbc isf isfs imis"

cd $X
stage "re-estimate EuRoC + TUM-VI with the IMU gate (v1 estimates kept as calib_estimate_v1.json)"
for d in $(ls -d $O/p1x_* $O/xMH0*_* $O/xV*_* $O/xtv* 2>/dev/null | grep -v '\.'); do
  [ -f $d/calib_estimate.json ] && [ ! -f $d/calib_estimate_v1.json ] && cp $d/calib_estimate.json $d/calib_estimate_v1.json
done
if [ ! -f $L/calib_summary_v2.md ]; then
  ls -d $O/p1x_* $O/xMH0*_* $O/xV*_* $O/xtv* 2>/dev/null | grep -v '\.' | \
    xargs -P 12 -n 4 $PY $X/calib_reestimate.py --force 2>&1 | grep -v Warning > $L/p1iter_est.log
  echo "    $(grep -c . $L/p1iter_est.log) runs logged"
  $PY $X/calib_summary.py --results $O --out $L/calib_summary_v2.md > /dev/null 2>&1
fi

round1() { # name stem nframes arm level seed
  local rid=x$2_L$5_$4_s$(printf %02d $6) out=xd$2_L$5_$4_s$(printf %02d $6)
  [ -f $O/$out/record.json ] && return
  grep -q '"oracle"' $O/$rid/calib_estimate.json 2>/dev/null || { echo "    $rid: no estimate"; return; }
  python3 calib_perturb.py --src $E/$1 --dst $SCR/$out.p --yaml $BASE --level $5 --seed $6 --arm $4 --force > /dev/null 2>&1 &&
  python3 calib_correct.py --src $SCR/$out.p --dst $SCR/$out --estimate $O/$rid/calib_estimate.json --force > /dev/null 2>&1 ||
    { echo "    $out: generation failed"; rm -rf $SCR/$out.p $SCR/$out; return; }
  rm -rf $SCR/$out.p                       # the corrected sequence links the original images; keep it for round 2
  OASIS_EXPORT=1 ./run_oasis.sh $SCR/$out $2 $out $SCR/$out/config.yaml > /dev/null 2>&1
  python3 -I phase2_eval_run.py $O/$out --gt $E/$1/mav0/state_groundtruth_estimate0/data.csv --n-frames $3 > $O/$out/record.json
  cp $SCR/$out/CORRECTION_MANIFEST.json $O/$out/
  gzip -f $O/$out/export/*.csv 2>/dev/null
}
round2() { # name stem nframes arm level seed
  local src=xd$2_L$5_$4_s$(printf %02d $6) out=xe$2_L$5_$4_s$(printf %02d $6)
  [ -f $O/$out/record.json ] && { rm -rf $SCR/$src; return; }
  [ -d $SCR/$src ] || { echo "    $src: round-1 sequence missing"; return; }
  grep -q '"oracle"' $O/$src/calib_estimate.json 2>/dev/null || { echo "    $src: no estimate"; return; }
  python3 calib_correct.py --src $SCR/$src --dst $SCR/$out --estimate $O/$src/calib_estimate.json --key $4 --force > /dev/null 2>&1 ||
    { echo "    $out: generation failed"; rm -rf $SCR/$out; return; }
  ./run_oasis.sh $SCR/$out $2 $out $SCR/$out/config.yaml > /dev/null 2>&1
  python3 -I phase2_eval_run.py $O/$out --gt $E/$1/mav0/state_groundtruth_estimate0/data.csv --n-frames $3 > $O/$out/record.json
  cp $SCR/$out/CORRECTION_MANIFEST.json $O/$out/
  rm -rf $SCR/$out $SCR/$src
}
jobs_list() {
  for s in 0 1 2 3 4 5 6 7 8 9; do for spec in $SPECS; do for arm in $ARMS; do for l in 2 3; do
    echo "${spec//:/ } $arm $l $s"; done; done; done; done
}

stage "round 1: correct with the gated estimate, re-run with export"
n=0; while read -r j; do round1 $j & n=$((n+1)); [ $((n % 3)) -eq 0 ] && wait; done < <(jobs_list); wait

stage "round 1 estimates (oracle group of each arm)"
jobs_list | while read -r name stem nf arm l s; do
  d=$O/xd${stem}_L${l}_${arm}_s$(printf %02d $s)
  [ -d $d/export ] && [ ! -f $d/calib_estimate.json ] && echo "$arm $d"
done | xargs -P 12 -L 1 sh -c '"$0" "$1"/calib_reestimate.py --modes oracle --oracle-key "$2" "$3"' $PY $X 2>&1 \
  | grep -v Warning > $L/p1iter_est_r1.log
echo "    $(grep -c . $L/p1iter_est_r1.log) runs logged"

stage "round 2: correct again, re-run"
n=0; while read -r j; do round2 $j & n=$((n+1)); [ $((n % 3)) -eq 0 ] && wait; done < <(jobs_list); wait

stage summary
python3 - <<'EOF' | tee $L/closed_loop_v2.md
import glob, json, os
import numpy as np
O = "/data/tkocher/oasis_results"
def ates(pat):
    out = []
    for f in glob.glob(os.path.join(O, pat, "record.json")):
        try:
            r = json.load(open(f))
        except Exception:
            out.append(None); continue
        out.append(r.get("ate_m") * 1000 if r.get("maps_in_atlas") == 1 and r.get("ate_m") is not None else None)
    return out
def med(v):
    v = [x for x in v if x is not None]
    return f"{np.median(v):.1f}" if v else "--"
def win(v, cm):
    ok = [x for x in v if x is not None]
    return f"{np.mean([x - cm <= 5 for x in ok]) * 100:.0f} % ({len(ok)}/{len(v)})" if v else "--"
print("# Phase 1.4b iterated closed loop: ATE (mm, median over seeds; share within 5 mm of clean, valid/total)\n")
print("v1 = ungated estimate (closed_loop.md); r1 = gated estimate; r2 = second correction re-estimated on the r1 map\n")
print("| seq | arm | tier | clean | uncorrected | v1 | r1 | r2 | r1 within 5 mm | r2 within 5 mm |")
print("|---|---|---|---|---|---|---|---|---|---|")
for seq in ("MH01", "MH03"):
    clean = [x for x in ates(f"c{seq}_L0_clean_*") + ates(f"x{seq}_L0_clean_*") if x is not None]
    cm = float(np.median(clean))
    for arm in ("toff", "rbc", "tbc", "isf", "isfs", "imis"):
        for l in (2, 3):
            u, v1 = ates(f"x{seq}_L{l}_{arm}_s*"), ates(f"xc{seq}_L{l}_{arm}_s*")
            r1, r2 = ates(f"xd{seq}_L{l}_{arm}_s*"), ates(f"xe{seq}_L{l}_{arm}_s*")
            print(f"| {seq} | {arm} | {l} | {cm:.1f} | {med(u)} | {med(v1)} | {med(r1)} | {med(r2)} | {win(r1, cm)} | {win(r2, cm)} |")
EOF
stage "ALL DONE"
