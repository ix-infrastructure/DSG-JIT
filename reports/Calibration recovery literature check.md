# Calibration recovery: literature check (2026-10-12)

This check asks whether post-hoc recovery of calibration errors in DSG-JIT is new. The errors in question are the ones that hurt in the 4.4 sweep: stereo rotation, camera–IMU time offset and IMU scale.

Every citation was opened at its URL. Entries marked [meta] have confirmed bibliographic data only; read them before citing.

## Already done

| Area | Key prior work |
|---|---|
| Online stereo extrinsic self-calibration | Dang et al. T-IP 2009; Hansen et al. CVPR 2012; Ling & Shen IROS 2016 (arXiv 1903.10705, includes an accuracy-sufficiency test); Mueller & Wuensche IV 2016–18 [meta]; Zhao et al. arXiv 2309.10314 |
| Camera–IMU time offset | Li & Mourikis IJRR 2014 (EKF); Furgale et al. IROS 2013 / Rehder et al. T-RO 2016 (Kalibr, target-based); Qin & Shen IROS 2018 (VINS); Kelly & Sukhatme ISER 2010 [meta] and Qiu et al. T-RO 2021 [meta] (trajectory-level, system-agnostic) |
| IMU intrinsics online | Yang, Geneva, Zuo, Huang RSS 2020 "Online IMU Intrinsic Calibration: Is It Necessary?"; T-RO 2023 (doi 10.1109/TRO.2023.3275878, full self-calibration in OpenVINS, with degeneracy analysis) |
| Targetless batch calibration | Maye et al. IV 2013 / IJRR 2016; Schneider et al. ICRA 2017 / Sensors J. 2019; iKalibr T-RO 2025 (arXiv 2407.11420) |
| Miscalibration detection | CalQNet arXiv 2104.04837 (stereo); StOCaMo IbPRIA 2023 (stereo, evaluated on EuRoC); Moravec & Šára PAA 2024; Xiao et al. RA-L 2022 (camera–IMU extrinsic quality); Nobre et al. ISER 2016 / ICRA 2017 [meta] (change detection) |
| Differentiable optimisation | Theseus NeurIPS 2022 and PyPose CVPR 2023 (includes a learned IMU calibration on EuRoC). No work found that uses IFT or bilevel optimisation for t_d, stereo extrinsics or IMU scale |
| Sensitivity studies | arXiv 2609.18628 (subterranean VIO under miscalibration; no t_d or IMU scale); arXiv 2609.13777 (online extrinsic updates raised APE by 46–52%). No multi-seed ORB-SLAM3 sweep over stereo yaw/pitch, t_d and IMU scale was found |

ORB-SLAM3 estimates none of these parameters (Campos et al. T-RO 2021): T_b_c1 and T_c1_c2 are fixed, and there is no t_d field.

## Gaps (not found, which is not the same as "does not exist")

1. Joint post-hoc, GT-free recovery of stereo rotation, t_d and IMU scale from the output of a black-box SLAM system that does no calibration.
2. Identifiability of stereo yaw (a depth/scale bias) versus accelerometer scale versus t_d. Both stereo yaw and accelerometer scale act like metric-scale errors.
3. Silent-miscalibration detection for stereo-inertial SLAM that attributes the error to a parameter and is calibrated against measured ATE damage.

## Risks

- Re-running full batch BA with calibration states added reads as "Maye/Schneider/iKalibr in JAX".
- Bilevel/IFT is only justified if the outer objective differs from the inner one (held-out residuals, cross-sequence consistency, noise-model learning).
- The "system-agnostic" claim needs a second SLAM system.

## Real miscalibration data

- **TUM-VI.** Raw sequences ship without the IMU scale/misalignment correction, and the correction matrices are published. That gives a real, documented IMU intrinsic error.
- **UMA-VI.** Kalibr time shift of about 9.1 ms (Bumblebee2) and 5.3 ms (uEye), which is in our tested range. Still to check: whether the shipped timestamps are already corrected.
- **OpenVINS RPNG datasets.** The documentation states the calibration is imperfect and some sequences are unsynchronised.
- **OpenLORIS.** Software sync, with a 1.7–7.4 ms offset standard deviation.
- **EuRoC.** No calibration errata. Real stereo error would have to come from re-running Kalibr on the calibration bags.
