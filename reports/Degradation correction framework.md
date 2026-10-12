# Degradation-aware correction framework for stereo-inertial SLAM

Working design document. Started 2026-10-12. Platform: ORB-SLAM3 stereo-inertial (brdsdsu fork, OASIS), EuRoC MAV (MT9V034 global-shutter mono8 stereo at 20 Hz, ADIS16448 IMU at 200 Hz). Experiment code is in `dsg-jit/experiments/oasis/`, and results are in `dsg-jit/experiments/oasis/results/`.

## 1. Why "degradation", not "denoising"

The project started as *odometry denoising*. That name no longer fits. Of the problems we now target, only one is noise in the statistical sense:

| Class | What is wrong | Is it noise? |
|---|---|---|
| Sensor noise | Random per-pixel or per-sample fluctuation (shot, read, quantization, IMU white noise) | Yes |
| Exposure / photometric | Too little or uneven light. The signal is deterministically scaled and clipped, and noise is only a side effect. | No |
| Motion blur | Deterministic integration of the image over the exposure, given the motion | No |
| Calibration / timing error | A wrong parameter: extrinsics, time offset, IMU scale or misalignment | No (systematic bias) |
| Trajectory / map error | The output of SLAM carries the residual of all the above | Mixed |

**Working name: degradation-aware correction framework.**
- A *degradation* is anything that makes the sensor data differ from what the SLAM model assumes.
- A *correction* is a module that removes or compensates for one degradation, either before SLAM (pre) or after SLAM (post).
- "Noise" is reserved for the stochastic class.

## 2. Lessons that constrain the design

1. **Validate on real data before trusting a synthetic model.**
   - The earlier paper was dropped because its trajectory-level Gaussian noise had no real counterpart.
   - The low-light study nearly repeated that mistake. Uniform synthetic darkening showed a denoising benefit (Phase 2). Real dark EuRoC footage did not (§4.1), because real darkness is high-contrast, not uniform.
   - Rule: for every degradation, first measure where real data sits, then run synthetic sweeps calibrated to that.
2. **Parameter errors are safer to inject than image corruptions.**
   - A wrong calibration value or a mis-scaled IMU axis *is* the real failure.
   - Only the magnitude is an assumption, and it is taken from published tiers.
3. **Report validity before accuracy.**
   - Order: valid-run rate, then tracking coverage, then resets and startup delay, then ATE.
   - ATE on its own hides tracking loss.
   - The validity rule must ignore stereo-inertial startup. That startup costs 11–21 s on MH03–05 in every arm.
4. **Statistical floor.**
   - Clean MH01 ATE is 39.8 ± 5.1 mm (n = 20).
   - Minimum detectable effect is about 5 mm at 10 runs, so use 10 runs or more per cell.
   - Pair arms on the same random draw.
   - Use Holm correction across levels.
5. **Every correction must leave clean input unchanged,** and must be gated on a measured quality signal.

## 3. Framework architecture

```
            ┌──────────── per-frame quality monitors (<1 ms) ────────────┐
 sensors ──►│ brightness/headroom · noise σ̂ · IMU-predicted blur length │
            │ L/R photometric ratio · calibration-consistency residuals  │
            └──────┬───────────────────────────────────────┬─────────────┘
                   ▼                                       │ quality scores
   PRE  gated corrections (identity on clean input)        │
        photometric: stereo-shared saturation-safe gain    │
        blur: skip-and-bridge with IMU / deblur (TBD)      │
        timing: online time-offset compensation (TBD)      │
                   ▼                                       ▼
              ORB-SLAM3 stereo-inertial  ──►  POST (DSG-JIT, differentiable)
                                             · quality-weighted re-optimisation
                                               (down-weight flagged frames)
                                             · batch calibration re-estimation
                                               (time offset, extrinsics, IMU
                                               scale/misalignment), no GT
                                             · robust fixed-lag smoothing
```

- **Quality scores link pre and post.** The monitors that gate the pre-corrections also set the weights in the post-SLAM optimizer. This extends the information-weighted smoothing idea from exp30.
- **Calibration re-estimation is the natural DSG-JIT contribution.** It is a bilevel problem, solved with implicit differentiation:
  - Outer variables: calibration parameters.
  - Inner problem: trajectory estimation.
  - Objective: self-consistency of the measurements, with no ground truth.
- **Post-SLAM integrity rules from the denoising audit apply:**
  - No ground truth in any outer loss.
  - Any ground-truth anchors are disclosed.
  - Every result is compared against an anchors-only baseline.

## 4. Degradation model cards

Each card gives the physical model, the magnitudes and where they come from, the injection method, the correction, the real-data evidence, and a viability status.

### 4.1 Low light / exposure (photometric). Status: VIABLE as a brightness correction; denoising NOT supported on real data

- **Model.**
  - The scene is dimmed by Δ EV stops.
  - Analog gain g = min(2^Δ, 4) compensates up to 2 EV.
  - Output = clip(a·S + n), with attenuation a = g/2^Δ.
  - Noise variance is Poisson–Gaussian: σ² = K·μ + (K·σ_read)², with K = 0.0186 DN/e⁻ and σ_read ≈ 30 e⁻.
  - Measured on MH01 with a photon-transfer test: the sensor is linear, full scale is about 13,700 e⁻, and σ at 128 DN is 1.65 DN.
- **Severity ladder.** These are noise-equivalent levels. Each value is the noise left after the frame is brightened back to the MH01 median of 76.6 DN, measured with the same estimator on darkened MH01:

| Level | L0 | L1 | L2 | L3 | L4 | L5 | L6 |
|---|---|---|---|---|---|---|---|
| Residual σ (DN) | 2.3 | 2.7 | 3.6 | 5.7 | 9.9 | 17.9 | 31.2 |
| Raw mean (DN) | 77 | 77 | 77 | 38 | 19 | 9.8 | 5.2 |

- **Real-data evidence** (`phase0_real_dark.py`):

| Sequence | Frames ≥ L3 | ≥ L4 | ≥ L5 | ≥ L6 | Longest stretch ≥ L4 |
|---|---|---|---|---|---|
| MH01, MH03 | 0% | 0% | 0% | 0% | – |
| MH04 | 12% | 9.6% | 4.2% | 0% | 9.8 s |
| MH05 | 13% | 8.8% | 2.3% | 0% | 10 s |

  - No sign that the camera raises its gain in the dark: the noise-versus-brightness signature of dark frames matches MH01's.
  - Real dark frames are **high-contrast**: about 93% of pixels are below 7 DN next to bright windows and lit strips. Synthetic darkening is uniform.
- **Synthetic result** (Phase 2, MH01, 10 seeds, `results/phase2_MH01_2026-10-12/`):
  - Gain plus NLM (C4) beats gain alone (C2) at L5 (−11.2 mm, Holm p = 0.029).
  - At L6, C4 is the only arm that tracks (10/10 runs vs 0/10).
  - L3 shows no effect, and L4 is borderline (Holm p = 0.055).
  - Lowering the FAST threshold (C1) fails from L5 onward.
  - Equivalence of L1–L2 to clean is not confirmed: L2 is +3.9 mm (p = 0.04).
- **Real result** (MH03–05, 5 arms × 10 runs, `results/real_MH03-05_2026-10-12/`):
  - Gain cuts 1-second drift inside dark windows by 17–20% compared with no correction (25.3 → 21.2 mm on MH04, 28.8 → 23.0 mm on MH05; Holm p ≤ 0.01).
  - Bright windows are unchanged in every arm, as a control should be.
  - NLM on top of gain adds nothing, and is about +1 mm worse.
  - No arm lost tracking.
- **Correction (pre).**
  - Stereo-shared, saturation-safe gain: g = min(m_ref/m, 240/p99 of not-yet-clipped pixels), capped at 16.
  - A mean-based gain clips the lit regions, which are the only regions carrying features.
- **Open.**
  - Does denoising pay in *uniformly* very dark real scenes (L6)? EuRoC never reaches L6, and ETH3D `*_dark` is unusable (§6).
  - Candidate: 4Seasons night sequences. Their sensor type (global shutter? IMU?) is unverified.

### 4.2 Exposure flicker / left–right photometric mismatch. Status: SMALL in EuRoC; fold into 4.1

- **Model.** I' = clip(a_t·I + b_t), with a_t chosen per camera by independent auto-exposure.
- **Real measurement** (per-frame means, MH01/03/04/05):
  - Left/right ratio is within −0.11 to +0.14 EV for 90% of frames.
  - Frame-to-frame steps above 0.25 EV occur in 0.2–1.9% of frames; the largest is 0.53 EV.
  - On MH04/05 the left and right steps are uncorrelated (r ≈ 0.02); on MH01/03 they are correlated (r ≈ 0.78).
  - These figures are upper bounds: they include brightness changes caused by scene motion.
- **Why the impact is expected to be small:**
  - BRIEF descriptors are invariant to monotonic intensity changes.
  - The stereo SAD refinement subtracts the centre pixel, which removes offsets.
  - Only the absolute FAST threshold sees gain changes.
- **Decision.** Not a separate module. The photometric correction in 4.1 applies one gain to both cameras, smoothed over time.

### 4.3 Motion blur. Status: PENDING (real-blur measurement on V1_03/V2_03)

- **Model.** The image is integrated over the exposure time t_e along the motion path. For rotation, blur length ≈ f·|ω|·t_e px, with f = 458 px.
  - Example: 2 rad/s at 10 ms gives about 9 px. Translation adds about 4.6 px (survey estimate).
- **Unknown.** EuRoC does not publish exposure times. TUM VI records exposure per frame.
- **Real data.**
  - Blur appears during initialization in all Machine Hall sequences, and in V1_03, V2_03 and MH04/05.
  - Downloads in progress: V1_01, V1_03, V2_01, V2_03.
- **Correction options.**
  - Pre: drop the frame and bridge with the IMU (ORB-SLAM3 baseline) vs a gyro-kernel Wiener deblur gated on the IMU-predicted blur length.
  - Post: raise the measurement covariance of frames flagged as blurred.
- **Viability question.** Do real sequences contain enough blur to measurably cut ORB repeatability or matches? Plan:
  - Measure blur per frame: IMU angular rate × f gives a blur rate; compare with image sharpness.
  - Correlate with tracking quality.
  - Only then run synthetic sweeps, calibrated to the real distribution.

### 4.4 Calibration and timing errors. Status: PENDING (sensitivity sweep running)

- **Model.** Errors are injected exactly, by `calib_perturb.py`:
  - Belief errors (YAML): camera–IMU rotation (rbc), camera–IMU translation (tbc), stereo rotation (rst).
  - Sensor errors (IMU data): time offset (toff), IMU scale factor (isf), IMU axis misalignment (imis).
  - Each seed draws one error direction and reuses it at every level, so levels are paired.
- **Tiers** (realistic / mishandled / stress, from the survey):

| Error | Tier 1 | Tier 2 | Tier 3 |
|---|---|---|---|
| toff (ms) | 5 | 15 | 30 |
| rbc (°) | 0.25 | 0.8 | 2.0 |
| tbc (cm) | 1 | 3 | 6 |
| rst (°) | 0.2 | 1.0 | 2.5 |
| isf (%) | 0.5 | 1 | 2 |
| imis (°) | 0.05 | 0.5 | 2.0 |

  The isf and imis tiers are **provisional**: the survey gives no scale-factor tiers, and its datasheet figures are unverified.
- **Why it matters.**
  - ORB-SLAM3 estimates none of these online.
  - The survey flags time offset as the parameter that hurts.
  - EuRoC is hardware-synchronized, so any offset in the data is one we injected.
- **Correction.**
  - Pre/online: a consistency monitor (reprojection and IMU preintegration residuals), plus re-estimation gated on enough motion excitation.
  - Post: batch bilevel re-estimation in DSG-JIT.
- **Sweep.** MH01 and MH03, 6 errors × 3 tiers × 10 seeds plus 10 clean runs each; 380 runs started 2026-10-12.
- **Early observation.** A 30 ms offset on MH01 delays tracking startup: the run tracked 0.76 of the sequence but stayed continuous once started. Startup delay must therefore be reported as its own outcome.

### 4.5 Sensor noise (stochastic). Status: LOW PRIORITY

- **IMU white noise:** no effect below 10× the datasheet value; +26 mm at 20×; +209 mm at 50×. OKVIS and VINS-Mono already ship 4–40× inflated noise settings.
- **IMU bias random walk:** never swept independently. This is a gap in the literature and a cheap future arm for `calib_perturb.py`.
- **Fixed-pattern noise (DSNU/PRNU):** null at realistic levels; +9.5 mm at stress level (p = 0.07).
- **Image temporal noise at normal exposure:** σ ≈ 1–2 DN. It only matters through low light (4.1).

### 4.6 Excluded for this rig

- Rolling shutter: the cameras are global shutter.
- Bayer, demosaicing and chromatic effects: the cameras are mono.
- Weather: the venues are indoors.
- JPEG: EuRoC frames are lossless.
- Vignetting: already baked into the frames.
- Lens dirt and occlusion: physically real, with real data in QueensCAMP, but deferred as a possible fourth module.

## 5. Evaluation protocol

- **Validity** (`phase2_eval_run.py`): coverage after the first pose ≥ 90%. Startup delay, mid-run gaps and resets are recorded separately.
- **Accuracy** (`rescore_real.py`):
  - ATE with SE(3) alignment and scale fixed, on a time window common to all runs of a sequence.
  - 1-second relative drift, split by the per-frame degradation level. This localizes effects to the frames that are actually degraded.
- **Statistics:** paired Wilcoxon with Holm correction for seeded designs; Mann–Whitney for unpaired real-data repeats; Fisher's exact test on valid-run counts; TOST (±5 mm) for equivalence claims.
- **Order of work for each degradation:**
  1. Measure real data.
  2. Run a detector-level study (Phase 1 style).
  3. Run SLAM on synthetic data calibrated to the real distribution.
  4. Re-test on real data.

## 6. Open questions and status

| # | Question | Status |
|---|---|---|
| Q1 | Does real low light reach the levels where corrections matter? | **Answered:** MH04/05 reach L4–L5 for ≤ 10 s; never L6 |
| Q2 | Does denoising help on real dark footage? | **Answered for EuRoC: no.** Gain helps by 17–20% in dark windows |
| Q3 | Is exposure flicker or L/R mismatch big enough to need its own module? | **Answered: no**, ≤ 0.53 EV worst case; folded into 4.1 |
| Q4 | Is there real *uniformly very dark* data with ground truth? | ETH3D `*_dark` is **unusable** (mean 0.4 DN, 99th percentile 2 DN; effectively black). Candidate: 4Seasons night (to verify) |
| Q5 | Which calibration and timing errors hurt ORB-SLAM3 at realistic tiers? | **Pending:** sweep running |
| Q6 | How much real motion blur is in EuRoC, and does it hurt? | **Pending:** V1_03/V2_03 downloading |
| Q7 | Can batch re-estimation recover the injected calibration errors without ground truth? | After Q5, only for the errors that matter |
