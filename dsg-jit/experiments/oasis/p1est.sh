#!/usr/bin/env bash
# Phase 1.2/1.3 on lambda: after the SLAM queue (p1q.sh) is done, re-estimate the calibration on every export
# (dsg_jit.calib via calib_reestimate.py) and write the summary. Runs after the queue so the estimator never
# competes with ORB-SLAM3 for CPU (its results depend on thread timing). Log: /data/tkocher/p1export/p1est.log
set -u
L=/data/tkocher/p1export; O=/data/tkocher/oasis_results; X=~/repos/oasis-experiments
V=/data/tkocher/venv-p1; export PIP_CACHE_DIR=/data/tkocher/pipcache   # keep everything off /home
stage() { echo "=== $(date '+%F %T') $*"; }

stage waiting for p1q.sh
until grep -qE "ALL DONE|FAILED" $L/p1q.log 2>/dev/null; do sleep 300; done
grep -q "FAILED" $L/p1q.log && stage "p1q.sh reported a failure; estimating on whatever exports exist"

stage venv
if [ ! -x $V/bin/python ]; then
  python3 -m venv $V && $V/bin/pip install -q --upgrade pip && $V/bin/pip install -q "jax[cpu]" numpy scipy \
    || { stage "venv FAILED"; exit 1; }
fi
$V/bin/python -c "import jax, scipy; print('    jax', jax.__version__, 'scipy', scipy.__version__)"
export PYTHONPATH=$L/pkg
# one process per run, few threads each: this is a shared machine
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2"

stage smoke
$V/bin/python $X/calib_reestimate.py $O/p1x_MH01_clean_r0 --force 2>&1 | grep -v Warning | sed 's/^/    /'
[ -f $O/p1x_MH01_clean_r0/calib_estimate.json ] || { stage "estimator smoke FAILED"; exit 1; }

stage "estimate EuRoC, TUM-VI, other sequences"
ls -d $O/p1x_* $O/xMH0*_* $O/xV*_* $O/xtv* 2>/dev/null | grep -v '\.' | \
  xargs -P 12 -n 4 $V/bin/python $X/calib_reestimate.py 2>&1 | grep -v Warning > $L/p1est_runs.log
echo "    $(grep -c . $L/p1est_runs.log) runs logged"
stage "estimate 4Seasons (large maps: fewer at once)"
ls -d $O/x4D_* 2>/dev/null | grep -v '\.' | \
  xargs -P 4 -n 1 $V/bin/python $X/calib_reestimate.py 2>&1 | grep -v Warning >> $L/p1est_runs.log

stage summary
$V/bin/python $X/calib_summary.py --results $O --out $L/calib_summary.md > /dev/null 2>&1
head -c 4000 $L/calib_summary.md
stage "ALL DONE"
