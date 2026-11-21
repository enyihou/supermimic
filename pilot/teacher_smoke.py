"""Run a six-scenario, headless teacher-only CUDA smoke test; never fit a student."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from experiment import SOURCE, Scenario, rollout, summarize, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file")
    if torch.cuda.device_count() != 1:
        parser.error("Expose exactly one authorized GPU via CUDA_VISIBLE_DEVICES")

    from src.tasks.supermimic.config.task_cfg import FoundationConfig
    from src.tasks.supermimic.env import SuperMimicEnv
    from src.tasks.supermimic.rl.inference import load_actor, validate_environment

    device = "cuda:0"
    clips = ("walk1_subject1", "run1_subject2")
    found = {Path(entry["file"]).stem for entry in json.loads(args.manifest.read_text())["clips"]}
    if not set(clips) <= found:
        parser.error(f"The manifest must contain {', '.join(clips)}")
    start = perf_counter()
    teacher, saved, _ = load_actor(args.teacher, args.manifest, device)
    cfg = saved["config"]
    cases = [Scenario(clip, 0.48, 0, kick) for clip in clips for kick in (0, 0.45, -0.45)]
    env = SuperMimicEnv(FoundationConfig(
        manifest=args.manifest, clips=clips, num_envs=len(cases), seed=17,
        episode_length_s=100.0, action_clip=cfg.get("action_clip"),
        relative_orientation=cfg.get("relative_orientation", False),
        motion_anchor_ori_b=cfg.get("motion_anchor_ori_b", False),
    ), device=device)
    try:
        validate_environment(env, saved)
        rows, _ = rollout(env, teacher, cases)
        torch.cuda.synchronize()
        summary = summarize(rows)
        report = {"kind": "teacher_cuda_smoke", "source_revision": SOURCE, "torch": torch.__version__,
                  "gpu": torch.cuda.get_device_name(0), "elapsed_s": perf_counter() - start,
                  "peak_torch_memory_bytes": torch.cuda.max_memory_allocated(),
                  "summary": summary, "scenarios": rows}
        write_report(args.output, report)
        print(json.dumps(report, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
