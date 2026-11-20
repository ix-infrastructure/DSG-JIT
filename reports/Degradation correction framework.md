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
   - **Measured directly** (`errchar.py`, 2026-10-13): none of the degradations studied adds white noise to the trajectory. The clean run's error and the error each calibration error adds both have:
     - lag-1 (0.05 s) autocorrelation 0.975–0.999 (white noise: about 0);
     - correlation times of 1.5–14 s;
     - 60–99% of their power below 0.1 Hz (white noise: about 1%).
     - In addition, repeated runs with the same time-offset error reproduce the same error curve (cross-seed correlation 0.64–0.88; clean runs 0.88–0.92).
     - Marginal distributions look roughly Gaussian (excess kurtosis −1.1 to +1.5), but the error is structured drift and bias, not independent per-pose noise.
     - **Normality and Gauss–Markov fit** (`errnorm.py`, 2026-10-16).
       - Method: samples spaced 2τ apart, pooled over runs and axes. Shapiro–Wilk was run on 40 configurations: MH01/MH03 clean, 6 errors × 3 tiers, and real-dark MH04/MH05 with saturation-safe gain (C2s).
       - Gaussian not rejected (p > 0.05) in 13 of 40 configurations, nearly all MH03 small-to-mid errors. Rejected in the rest: heavy tails, kurtosis up to +13; most MH01 cells, catastrophic stereo-rotation runs and real-dark runs.
       - A first-order Gauss–Markov (AR(1)) model fails everywhere:
         - innovations are white in 0% of series (Ljung–Box);
         - innovation kurtosis is +8 to +270, so the error moves in jumps;
         - AR(2) is preferred in 38 of 40 configurations (ΔBIC 28–830);
         - the exponential ACF misfits at 1 s or 5 s.
       - The error is multi-timescale drift with impulsive corrections, not a Gaussian process of either kind.
     - Results: `results/error_character_2026-10-13/`.
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
- **Uniformly dark real data (4Seasons night, Q4).** Dark but nearly noise-free, so there is nothing for a denoiser to do; the saturation-safe gain is the only arm that helps (Q4).
- **Gain on other night recordings** (Phase 0.6, 2026-10-21, `results/phase0_closeout_2026-10-21/`). The only other 4Seasons nights are test-split recordings without reference poses, so "good" = 1 map and ≥ 90% coverage, no ATE; 15 seeds per arm:

| Night recording | No correction | Saturation-safe gain | Fisher p |
|---|---|---|---|
| `maximilianeum_2_test` (different route) | 0/15 (2–4 maps, 43% coverage, 729 failures/run) | **15/15** (99% coverage, 0 failures) | 1e-8 |
| `old_town_3_test` (same route, 14 min earlier) | 2/15 | 0/15 | 0.48 |

  On the new route the gain effect generalises completely. On `old_town_3` every gain run tracks almost the whole sequence but breaks once, at the same place (frame ≈ 9,300): the darkest stretch of the drive (mean 4 DN) followed by a stop under a streetlight. The map rebuilt during the stop never merges with the first one, and the larger map holds 59.8% of the frames in every seed. Two uncorrected runs passed that point in one map.
- **Synthetic night calibrated to the real one** (Phase 0.5, `synth_night.py`). Day frames of the same route darkened as y = clip(round(k(x − b) + σn)), k = 0.142 and b = 18.6 DN fitted by matching pooled intensity quantiles to the night, σ = 0.23 DN measured. Pixel statistics match (mean 6.3 vs 7.4 DN, median 4 vs 4, 81% vs 83% below 10 DN); streetlight hot-spots are missing (p95 28 vs 20, 35 vs 81 grey levels in use). 15 seeds per arm:

| | Real night | Synthetic night |
|---|---|---|
| No correction, good runs | 12/30 | **0/15** (13 split into 2 maps, 2 crashed; p = 0.004) |
| Gain, good runs | 20/30, median ATE 26.7 m | **15/15, median 7.4 m** (p = 0.02); real day is 7.0 m |

  **The target was missed in both directions:** without correction the synthetic night is harsher than the real one (no lit patches to track on), and gain undoes it completely, while on the real night gain leaves ATE at 4× the day level. Black floor plus quantization is therefore not the whole night loss. **Motion blur is not the missing piece (0.5b):** blurring each day frame over 63% of its frame-to-frame optical flow (night exposure 21 ms vs 0.12 ms by day) leaves the result unchanged: no correction 0/15 good (all split into 2–3 maps), gain 15/15 at 7.2 m. The likelier difference is the streetlight hot-spots: the synthetic night cannot exceed k(255 − b) ≈ 34 DN, while the real night's lights reach 255 DN and cap the saturation-safe gain at about ×2 where the synthetic night gets ×7. Figures and tables: `Phase0_images/0.5_synthetic_night/`.
- **Open.** Add streetlight hot-spots to the synthetic model (likely what caps the real gain). On `old_town_3` the gain itself jumps 2–3× within a second as lights enter and leave the view (`fig_old_town_3_break`); a temporally smoothed gain is the candidate fix.

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

### 4.3 Motion blur. Status: NOT VIABLE as a stand-alone module on EuRoC (no measurable drift cost)

- **Model.** The image is integrated over the exposure time t_e along the motion path. For rotation, blur length ≈ f·|ω|·t_e px, with f = 458 px.
  - Example: 2 rad/s at 10 ms gives about 9 px. Translation adds about 4.6 px (survey estimate).
- **Unknown.** EuRoC does not publish exposure times. TUM VI records exposure per frame.
- **Real data.**
  - Blur appears during initialization in all Machine Hall sequences, and in V1_03, V2_03 and MH04/05.
  - V1_01, V1_03, V2_01 and V2_03 are on lambda at `/data/shared/V11`, `V13`, `V21`, `V23`.
- **Correction options.**
  - Pre: drop the frame and bridge with the IMU (ORB-SLAM3 baseline) vs a gyro-kernel Wiener deblur gated on the IMU-predicted blur length.
  - Post: raise the measurement covariance of frames flagged as blurred.
- **Real measurement** (`phase0_blur.py`).
  - Per frame: the IMU-predicted image-plane rotation rate, plus the FAST-20 corner count and Crete blur metric, each relative to the surrounding ±1 s.
  - Translation is ignored, so the rate is a lower bound.

| Sequence | Frames 250–500 px/s | FAST corners vs neighbours | Frames > 500 px/s | FAST corners vs neighbours |
|---|---|---|---|---|
| MH01 | 2.8% | −23% | 0.05% | −46% |
| MH03 | 10% | −7% | 0% | – |
| MH04 | 3.5% | −11% | 0% | – |
| MH05 | 2.9% | −16% | 0.09% | −88% |
| V1_01 | 8.8% | −3% | 0% | – |
| V2_01 | 7.9% | −7% | 0.1% | −2% |
| V1_03 | 36% | −1% | 12.6% | −9% |
| V2_03 | 37% | −7% | 12.8% | −16% |

- **Findings.**
  - Fast rotation is common only in V1_03 and V2_03. There it costs at most 16% of corners, and sharpness barely changes (Crete +0–1%), which fits short exposures in the bright Vicon room.
  - In the darker Machine Hall the same rates cost far more: −88% on MH05 above 500 px/s. **Blur is coupled to low light through exposure time,** as the survey predicted.
  - Absolute 1-second drift split by blur rate on MH03–05 is mixed (MH03 +35%, MH04 −6%, MH05 +5–12%). It is confounded with motion magnitude, because fast windows also cover more ground.
  - **Motion-normalized drift on the Vicon sequences** (10 clean runs each; Wilcoxon, blurred vs sharp 1 s windows, paired by run):

| Sequence | Blurred windows (> 500 px/s) | Drift per metre, blurred | Drift per metre, sharp | p |
|---|---|---|---|---|
| V1_01 | 0 | – | 3.8% | – |
| V1_03 | 874 | 2.1% | 2.3% | 0.014 (blurred *better*) |
| V2_01 | 22 | 6.0% | 3.2% | 0.002 |
| V2_03 | 710 | 3.7% | 3.3% | 0.49 |

  - Where blur is common (V1_03, V2_03), it does not raise drift; V1_03's fast windows are slightly *better*, plausibly because fast motion excites the IMU. Only V2_01 shows a cost, and it rests on 22 windows, mostly at initialization.
- **Implication.** A stand-alone deblurring module is unlikely to pay off on EuRoC. The realistic blur case is dark and moving at once (long exposure), so it belongs with the photometric module's exposure–gain trade-off rather than as an independent degradation.

### 4.4 Calibration and timing errors. Status: VIABLE (sweep done; stereo extrinsic, time offset and IMU scale matter)

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
| isfs (%, gyro / accel) | 1 / 1 | 3 / 3 | 9 / 4 |
| imis (°) | 0.05 | 0.5 | 2.0 |

  The original isf tiers were provisional and too mild at the top. **isfs replaces them with sourced tiers**: ADIS16448 1% residual scale spec, BMI160/MPU-6000 sensitivity tolerances, and TUM-VI's uncorrected 9.4% gyro-y scale. The imis tiers held up against the same sources (ADIS16448: 0.05° gyro axis-to-axis). isfs reuses isf's per-axis signs per seed, so its 1%/1% tier is the identical perturbation to isf tier 2.
- **Why it matters.**
  - ORB-SLAM3 estimates none of these online.
  - The survey flags time offset as the parameter that hurts.
  - EuRoC is hardware-synchronized, so any offset in the data is one we injected.
- **Correction.**
  - Pre/online: a consistency monitor (reprojection and IMU preintegration residuals), plus re-estimation gated on enough motion excitation.
  - Post: batch bilevel re-estimation in DSG-JIT.
- **Sweep.** MH01 and MH03, 6 errors × 3 tiers × 10 seeds plus 10 clean runs each; 380 runs, 0 harness errors (2026-10-12). Clean median ATE: MH01 37.8 mm, MH03 27.6 mm. Δ = median ATE change vs clean; * = Mann–Whitney vs clean, Holm-corrected over 18 cells, p < 0.05. Results in `results/calib_MH01_MH03_2026-10-12/`.

| Error | Tier 1 Δ MH01 / MH03 (mm) | Tier 2 Δ | Tier 3 Δ |
|---|---|---|---|
| rst (stereo rotation) | +58 / **+62\*** | **+136\* / +85\*** | **5/10 runs fail** on both (≈300 tracking failures/run) |
| toff (time offset) | +4 / **+5\*** | **+10\* / +18\*** | **+21\* / +44\***; MH01 start delayed 25 s |
| isf (IMU scale) | +4 / **+11\*** | +7 / **+22\*** | **+19\* / +56\*** |
| rbc (cam–IMU rotation) | +2 / +2 | +4 / **+9\*** | **+34\* / +43\*** |
| imis (IMU misalignment) | 0 / +1 | +9 / **+11\*** | **+53\* / +62\*** |
| tbc (cam–IMU translation) | 0 / +1 | +10 / +3 | **+11\* / +11\*** |

- **Findings.**
  - **Stereo rotation is by far the most sensitive parameter.** 0.2° (the realistic tier) already doubles to triples ATE on MH03. The damage depends on the rotation axis: at tiers 1–2, ATE tracks the vertical-axis component (Spearman 0.87–0.99), which biases disparity and therefore depth. At tier 3, the runs that fail are the ones with a large x-axis (pitch) component, which breaks row alignment so stereo matching fails.
  - **Time offset and IMU scale factor hurt at realistic tiers on the faster MH03.** On MH03, 5 ms adds about 5 mm and a 0.5% scale error adds about 11 mm. On MH01 the same tiers are not significant. The sign of the offset matters: at 30 ms, an IMU running early is worse than one running late (median 78 vs 47 mm on MH01, 93 vs 65 mm on MH03).
  - **Camera–IMU extrinsics and IMU misalignment are tolerated at realistic tiers** and only hurt once mishandled (≥ 0.5–0.8°) or at stress level. Translation is the most benign.
  - Apart from rst tier 3, every run stayed valid with no mid-run gaps: these errors make the estimate worse without breaking tracking, so a tracking-health monitor alone will not detect them.
- **Sourced IMU-scale tiers** (Phase 0.3, 60 runs, 2026-10-21, `results/phase0_closeout_2026-10-21/`):

| isfs (gyro / accel) | MH01 median ATE | MH03 median ATE | Good runs |
|---|---|---|---|
| clean | 37.8 mm | 27.6 mm | – |
| 1% / 1% | 39.3 mm (×1.0, p = 0.43) | 49.7 mm (×1.8\*) | 20/20 |
| 3% / 3% | 79.5 mm (×2.1\*) | 111 mm (×4.0\*) | 20/20 |
| 9% / 4% | 332 mm (×8.8\*) | 739 mm (×27\*, up to 2.3 m) | 20/20 |

  (\* Mann–Whitney vs clean, p = 0.0002.) **Every run stays in one map with full coverage, even at 9% gyro scale.** The repeat of the identical 1% perturbation (isf tier 2: 44.9 mm, isfs tier 1: 39.3 mm on MH01) differs only by ORB-SLAM3's run-to-run noise (~5 mm).
- **Real (uninjected) IMU miscalibration, TUM-VI** (`tumvi_imu_variants.py`, `tumvi_to_euroc.py`; 2026-10-19). The TUM-VI exports ship IMU data corrected with the published calibration (x_cal = M x_raw − b; Schubert et al. 2018 eqs. 1–2), so the uncorrected data are recovered exactly. Rooms 1–6 × 3 conditions × 10 seeds, fisheye rectified to pinhole (the OASIS fork segfaults on KannalaBrandt8):

| IMU data | Good runs (1 map, ≥ 90% tracked, ATE < 10 m) | Median ATE (good) |
|---|---|---|
| Calibrated, as shipped | 60/60 | 12.1 mm (9.3–29.0 by room) |
| Scale + misalignment uncorrected (M⁻¹x) | 0/60 | – |
| Fully raw, incl. turn-on bias (M⁻¹(x + b)) | 0/60 | – |

  Every room separately: Fisher p < 0.001, composite p = 0.0001. Failed runs split into 2–4 maps or, in one map, scale-diverge to 40–565 m ATE. Scale/misalignment alone is sufficient (largest term: gyro y scale 1.094). **Real consumer-IMU miscalibration is catastrophic here, not silent.** The same magnitude injected into EuRoC (isfs tier 3) stays silent, so whether a large IMU error breaks tracking or only inflates ATE depends on the platform and motion (TUM-VI handheld in small rooms vs EuRoC MAV), not just on the error size; a tracking-health monitor cannot be relied on in either case. Results: `results/tumvi_imu_2026-10-19/`.
- **Implication for correction (Q7).** Re-estimation should target, in order: stereo extrinsic rotation (especially the yaw and pitch components), camera–IMU time offset, and IMU scale factors. Camera–IMU rotation, translation and misalignment are second priority. Caveats: the ATE here is over the full trajectory, and the MH01 toff tier-3 runs start 25 s late so they cover less of the flight.

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
| Q4 | Is there real *uniformly very dark* data with ground truth? | **Answered: yes, 4Seasons `old_town_4_train` night** (ETH3D `*_dark` is unusable: mean 0.4 DN, effectively black). Median frame mean 6.5 DN, exposure fixed at 21 ms all sequence long. The frames are **dark but nearly noise-free** (σ ≈ 0.2 DN; 65–80% of neighbouring pixels identical), so the loss is quantization and black clipping, not noise. Same route, same pipeline: **day 10/10 valid, median ATE 7.0 m, 0 tracking failures; night C0 12/30 valid, median ATE 43 m among valid runs, median 260 tracking failures**. **Night correction arms** ("good" = 1 map, ≥ 90% coverage, ATE < 1 km):
  - **C2s, saturation-safe gain** (median gain 4.9, 30 runs): 20/30 good, median ATE 27 m, median 46 tracking failures. **Significantly better than C0** at n = 30: composite p = 0.004, failures p = 0.0003, ATE among good runs p = 0.013; valid rate alone p = 0.07. One "good" C2s run has ATE 747 m (single map that diverged); counting it as failed leaves 19/30.
  - **C5, CLAHE** (10 runs): 2/10 good. No help.
  - **C1, lowered FAST threshold** (10 runs): 0/10 good, significantly worse (Fisher p = 0.02). The quantization-contour explanation was **rejected** by `orb_quality.py`: low-threshold corners match across stereo and time about as well as gain-corrected ones. C1 and C5 runs instead show 1–2 "bad imu" map resets each (C0 0.5, C2s 0.1 per run). **Cause found (Phase 0.4): not an IMU fault.** `LocalMapping.cc:138` resets a map with `mbBadImu` when, after IMU initialisation and before inertial BA2, a new map has moved < 2 cm over its last two keyframes within its first 10 s, i.e. a map rebuilt while the car is barely moving. Every reset is preceded by that check's "Not enough motion" message, and those messages track how often maps are rebuilt (per run: C0 17, C1 9, C5 11, C2s 0.7). Visual-only stereo on the same night input (no IMU at all): C0 0/10 good, C2s 5/10 good (p = 0.03), so the loss is visual and gain is the arm that fixes it; but the fork's visual-only path crashed (NaN in `SO3::exp`) in 13/20 runs (C0 8, C2s 5), and its drift without the IMU is ~130 m vs 27 m with it.
  - The day level (7 m) is not reached by any arm. Results: `results/fourseasons_night_2026-10-13/` |
| Q5 | Which calibration and timing errors hurt ORB-SLAM3 at realistic tiers? | **Answered:** stereo rotation (0.2° already +60 mm), time offset (5 ms, +5 mm on MH03) and IMU scale (0.5%, +11 mm on MH03; sourced 3% tier ×2–4, 9% tier ×9–27) hurt at realistic tiers. Camera–IMU rotation and translation and IMU misalignment only hurt once mishandled. Tracking stays valid, so the damage is silent (4.4) |
| Q6 | How much real motion blur is in EuRoC, and does it hurt? | **Answered:** common only in V1_03/V2_03 (13% of frames above 500 px/s), costing ≤ 16% of corners and **no measurable drift per metre** (V1_03 2.1% blurred vs 2.3% sharp; V2_03 3.7% vs 3.3%, p = 0.49). Severe only when dark (MH05), so it belongs with 4.1 |
| Q7 | Can batch re-estimation recover the injected calibration errors without ground truth? | After Q5, only for the errors that matter |
