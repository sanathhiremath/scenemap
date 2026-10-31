# SceneMap

Version 0.1. A runnable stereo-perception demo with actual pixel detection, camera calibration, triangulation, cross-camera correspondence, and temporal identities.

The inputs are owned generated stereo frames. This is a geometry and failure-analysis prototype, not a real-camera benchmark or a learned perception system. The model reconstructs world coordinates from extracted pixel centroids; it does not receive the world positions or identity labels that rendered the frames.

## Run on Windows or Ubuntu

Python 3.11+ is required. Create an environment with `python -m venv .venv`.

Windows: `.venv\Scripts\python -m pip install -e ".[dev]"`.

Ubuntu: `.venv/bin/python -m pip install -e '.[dev]'`.

Activate that environment, or use its Python executable:

```text
python run.py demo
python -m pytest -q
```

The demo writes `artifacts/demo.json` and 96 PNG images. The JSON contains calibration, estimated 3D positions, temporal track IDs, a comparison baseline, evaluation ground truth, and hashes for the frame files.

## The system

1. Two known pinhole projection matrices render same-colored moving objects into 320×180 stereo frames. They cross in image space at different depths.
2. Thresholding and connected components extract image centroids from actual pixels. Neither world identities nor projected ground-truth coordinates are passed into the detector.
3. A simple baseline matches detections by horizontal image order. The geometry-aware version uses rectified epipolar distance and a positive-disparity range with Hungarian assignment.
4. Linear DLT/SVD triangulates each correspondence. Positive depth and reprojection checks reject invalid geometric estimates.
5. A gated constant-velocity tracker assigns temporal IDs, smooths velocity estimates, and expires tracks after missing observations. Output points are observations, not interpolated predictions.
6. An independent evaluator compares outputs to authored world coordinates after reconstruction. It reports 3D position RMSE, reprojection residual, identity switches, matched observations, and skipped frames.

## Four failure stories

| Scenario | What changes | What to inspect |
| --- | --- | --- |
| Synchronized stereo | Correct calibration and matched timestamps | Compare geometry-aware and image-order matching around the crossing |
| Missing observation | One object is hidden in the right camera for three frames | No fabricated 3D point; identity may fragment after the observation gap |
| Incorrect calibration | Estimated baseline is 0.48m while the rendered baseline is 0.60m | Reprojection remains nearly zero while position RMSE rises |
| Timestamp skew | Right camera renders 80ms later after frame4 | Pairs exceeding the 20ms tolerance are rejected rather than triangulated |

The calibration case is intentional: a scale error can preserve image reprojection while making the reconstructed scene metrically wrong. Low reprojection error alone does not prove correct distances.

## Recorded measurements

The included run has geometry-aware clean-scene position RMSE about **0.294m**, versus **0.415m** for image-order correspondence. Clean geometry has **0 identity switches** versus **2** for the baseline. With the wrong baseline, geometry-aware position RMSE rises to **1.038m** despite near-zero reprojection residual. Seven unsynchronized pairs are withheld.

These numbers concern two objects in twelve authored frames per scenario. The nonzero clean error comes from pixel quantization and the small stereo image. This is not an accuracy claim for real-world cameras. Reported identity switches use a simple position-to-ground-truth assignment and should not be presented as a standard MOT metric.

## Boundaries and next steps

The correspondence method assumes rectified stereo, a bounded disparity range, and vertically distinguishable objects. General appearance ambiguity, lens distortion, learned detection, robust bundle adjustment, uncertainty propagation, camera recalibration, and real image datasets remain extensions. Current camera fixtures have known calibration; the project does not estimate intrinsics from checkerboard images.

Eight tests cover exact triangulation, scale ambiguity, actual pixel detection, image-order reversal, missing-camera handling, timestamp guards, track expiry, invalid inputs, and the exported failure scenarios.

## Image rights

All generated frames are original project fixtures released under CC0 1.0. See [ASSET_LICENSE.md](ASSET_LICENSE.md). No external footage, faces, or camera recordings are included.

## Repository

Source: https://github.com/sanathhiremath/scenemap


This repository contains a local working prototype. Reproduction commands, measured scope, and limitations are documented above.
