#!/usr/bin/env bash
# Phase 1.4 closed loop on lambda, after p1est.sh: for every perturbed tier-2/3 EuRoC run, regenerate the
# perturbed sequence (calib_perturb.py, same seed), apply the oracle estimate (calib_correct.py), re-run
# ORB-SLAM3 and compare ATE with clean and with the uncorrected run. Roadmap target: re-run ATE within ~5 mm
# of clean. Resumable (skips runs with record.json). Log: /data/tkocher/p1export/p1corr.log
set -u
L=/data/tkocher/p1export; O=/data/tkocher/oasis_results; X=~/repos/oasis-experiments
E=/data/tkocher/datasets/euroc; SCR=/data/tkocher/datasets/euroc_noisy; BASE=$O/_configs/EuRoC.yaml
stage() { echo "=== $(date '+%F %T') $*"; }

stage waiting for p1est.sh
until grep -q "ALL DONE\|FAILED" $L/p1est.log 2>/dev/null; do sleep 300; done

cd $X
one() { # name stem nframes arm level seed
  local rid=x$2_L$5_$4_s$(printf %02d $6) out=xc$2_L$5_$4_s$(printf %02d $6)
  [ -f $O/$out/record.json ] && return
  grep -q '"oracle"' $O/$rid/calib_estimate.json 2>/dev/null || { echo "    $rid: no estimate"; return; }
  python3 calib_perturb.py --src $E/$1 --dst $SCR/$out.p --yaml $BASE --level $5 --seed $6 --arm $4 --force > /dev/null 2>&1 &&
  python3 calib_correct.py --src $SCR/$out.p --dst $SCR/$out --estimate $O/$rid/calib_estimate.json --force > /dev/null 2>&1 ||
    { echo "    $out: generation failed"; rm -rf $SCR/$out.p $SCR/$out; return; }
  ./run_oasis.sh $SCR/$out $2 $out $SCR/$out/config.yaml > /dev/null 2>&1
  python3 -I phase2_eval_run.py $O/$out --gt $E/$1/mav0/state_groundtruth_estimate0/data.csv --n-frames $3 > $O/$out/record.json
  cp $SCR/$out/CORRECTION_MANIFEST.json $O/$out/
  rm -rf $SCR/$out.p $SCR/$out
}
stage closed loop
jobs=()
for s in 0 1 2 3 4 5 6 7 8 9; do
  for spec in MH_01_easy:MH01:3682 MH_03_medium:MH03:2700; do
    for arm in toff rbc tbc isf isfs imis; do for l in 2 3; do jobs+=("${spec//:/ } $arm $l $s"); done; done
  done
done
n=0; for j in "${jobs[@]}"; do one $j & n=$((n+1)); [ $((n % 3)) -eq 0 ] && wait; done; wait

stage summary
python3 - <<'EOF' | tee $L/closed_loop.md
import glob, json, os, re
import numpy as np
O = "/data/tkocher/oasis_results"
def recs(pat):
    out = {}
    for f in glob.glob(os.path.join(O, pat, "record.json")):
        try:
            out[os.path.basename(os.path.dirname(f))] = json.load(open(f))
        except Exception:
            pass
    return out
def ate(r):
    return r.get("ate_m") if r.get("maps_in_atlas") == 1 and r.get("ate_m") is not None else None
print("# Phase 1.4 closed loop: ATE after applying the oracle estimate (mm, median over seeds)\n")
print("| seq | arm | tier | clean | uncorrected | corrected | corrected within 5 mm of clean | valid corrected |")
print("|---|---|---|---|---|---|---|---|")
for seq in ("MH01", "MH03"):
    clean = [ate(r) for r in list(recs(f"c{seq}_L0_clean_*").values()) + list(recs(f"x{seq}_L0_clean_*").values())]
    clean = [c for c in clean if c is not None]
    cm = np.median(clean) * 1000
    for arm in ("toff", "rbc", "tbc", "isf", "isfs", "imis"):
        for l in (2, 3):
            u = recs(f"x{seq}_L{l}_{arm}_s*"); c = recs(f"xc{seq}_L{l}_{arm}_s*")
            if not c:
                continue
            ua = [ate(r) for r in u.values()]; ca = [ate(r) for r in c.values()]
            cv = [x * 1000 for x in ca if x is not None]; uv = [x * 1000 for x in ua if x is not None]
            win = np.mean([x - cm <= 5 for x in cv]) * 100 if cv else 0
            print(f"| {seq} | {arm} | {l} | {cm:.1f} | {np.median(uv) if uv else float('nan'):.1f} | "
                  f"{np.median(cv) if cv else float('nan'):.1f} | {win:.0f} % | {len(cv)}/{len(ca)} |")
EOF
stage "ALL DONE"
