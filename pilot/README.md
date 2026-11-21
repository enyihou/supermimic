# First-order teacher-distillation pilot

This is an experiment on an **independent SuperMimic reproduction**, not the official authors' checkpoint. It needs neither BONES-SEED nor a world-model checkpoint. The teacher has trained on the LAFAN1 motion library; the held-out clips are held out from the **student**, not the teacher.

Protocol:

- Pin the simulator/actor source and input revisions in `fetch_inputs.py`. Use 18 walk/run/sprint reference files from the G1 LAFAN1 retarget. Students train on subjects 1–3 and are tested on subjects 4–5, with different start phases and kick directions.
- Verify that the frozen teacher tracks through the 2 s post-kick window (`≤0.30 m` mean root position error in its last 0.2 s, with no failure): at least 90% of nominal and 75% of disturbed scenarios in each split. A failing gate writes the teacher results and does not train students. No claim depends on selecting the best kick settings after viewing student results.
- Give BC, BC plus direct action difference, and BC plus student directional derivative the same labeled endpoints, model, examples, minibatch sequence, update budget and three paired seeds. No teacher encoder latents are given to students. Exclude transitions across episode resets and the immediate kick discontinuity from slope pairs.
- Compare the integrated world-frame root tracking error after the kick, capped at 1 m per step, with failures charged the full 1 m for every remaining step. Report nominal error, failures, recovered counts, every scenario and training times. This is a pilot at matched update count; it is not evidence of a matched-wall-time benefit or robustness outside this locomotion suite.

## Storage and staging

Input artifacts are about 98 MB (45.2 MB of CSVs and a 52.5 MB teacher). The maximum 10,000 paired examples occupy about 296 MB as float32 tensors and are not written to disk. Git source/assets, converted NPZs, result checkpoints, and the staged CUDA environment are additional. The environment and Warp caches are expected to dominate scratch usage; measure them before choosing `--tmp`. Do not put expanded files on Ceph.

Use the current AIR runbook. On gpu01, prepare a local Python 3.11 environment with `conda-pack`, CUDA PyTorch compatible with the allocation, `mjlab==1.2.0`, `mujoco-warp==3.8.1`, `mujoco==3.8.1`, `warp-lang==1.12.0`, and `scipy==1.17.0`. Record resolved package versions. Obtain the pinned [reproduction source](https://github.com/Breezewrf/supermimic_mjlab/tree/aaac8c2077c388dd2fcc29f03862029bebd9f6b1); use an export of `src`, `scripts/data`, and required license files, with no Git history or bundled checkpoints. `code.tar.gz` expects `run.sh`, `experiment.py`, and `reproduction/` at its root. `data.tar.gz` expects `csv/`, `teacher.pt`, and `inputs.json` at its root, created by `python pilot/fetch_inputs.py <local-input-dir>`. Use an owned local cache, verify archive members, and publish using the installed AIR template into an **authorized** `/data` project. The job produces `pilot.json`, logs, and student checkpoints in the output archive.

Apply `reproduction-mjspec.patch` in the reproduction root before packaging. It keeps the MuJoCo `MjSpec` alive while reading its joint names, fixing a GPU-environment setup crash without changing the simulation parameters or teacher.

Local correctness check: `python -m unittest discover -s pilot -v`. These unit tests do not exercise CUDA physics.

For a separate, authorized short test on gpu01, `teacher_smoke.py` runs only the frozen teacher on `walk1_subject1` and `run1_subject2`, with one nominal and two pushed rollouts per clip. Supply a converted manifest containing those two motions, the pinned `teacher.pt`, and a new `--output` JSON path. Use exactly one GPU after the fresh availability and owner checks in the AIR server guide. A successful smoke test verifies the runtime and protocol wiring; it is not the full teacher gate or a student comparison.
