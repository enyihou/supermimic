"""Small, matched-label HOAS pilot on a pinned SuperMimic reproduction."""

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.func import jvp


ARMS = ("bc", "difference", "hoas")
SEEDS = (13, 37, 73)
PRE, POST = 30, 100  # 50 Hz; identical warmup and recovery horizons.
ERROR_CAP = 1.0  # meters; every step from a fall onward gets this penalty.
SOURCE = "aaac8c2077c388dd2fcc29f03862029bebd9f6b1"


def student_model(n_input, n_output):
    return nn.Sequential(nn.Linear(n_input, 256), nn.GELU(), nn.Linear(256, 256),
                         nn.GELU(), nn.Linear(256, n_output))


def training_loss(model, arm, x0, x1, a0, a1, weight=1.0):
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    p0 = model(x0)
    p1 = model(x1)
    bc = 0.5 * ((p0 - a0).square().mean() + (p1 - a1).square().mean())
    if arm == "bc" or weight == 0:
        return bc
    if arm == "difference":
        change = p1 - p0
    else:
        _, change = jvp(model, (x0,), (x1 - x0,))
    return bc + weight * (change - (a1 - a0)).square().mean()


@dataclass(frozen=True)
class Scenario:
    clip: str
    phase: float
    kick_x: float
    kick_y: float

    @property
    def pushed(self):
        return self.kick_x != 0 or self.kick_y != 0


def scenarios(clips, training):
    phases = (0.20, 0.48, 0.76) if training else (0.34, 0.64)
    kicks = ((0, 0), (0, 0.45), (0, -0.45)) if training else (
        (0, 0), (0.40, 0.40), (-0.40, -0.40),
    )
    return [Scenario(clip, phase, *kick) for clip in clips for phase in phases for kick in kicks]


def split_clips(names):
    selected = sorted(n for n in names if n.startswith(("walk", "run", "sprint")))
    train = [n for n in selected if n.endswith(("subject1", "subject2", "subject3"))]
    test = [n for n in selected if n.endswith(("subject4", "subject5"))]
    if not train or not test:
        raise ValueError("Need locomotion clips for training subjects 1–3 and test subjects 4–5")
    return train, test


def action_labels(features, teacher):
    history = features["history"]
    goal = teacher.reference_features(features["future_reference"], teacher.default_q)
    # Fixed input preprocessing only; do not feed teacher encoder latents to the student.
    x = torch.cat((teacher.normalize_history(history).flatten(-2), teacher.goal_norm(goal)), -1)
    return x, history, goal


@torch.no_grad()
def rollout(env, teacher, cases, *, student=None, stats=None, collect=False):
    device = env.device
    assert len(cases) == env.num_envs
    motion = torch.tensor([env.library.names.index(c.clip) for c in cases], device=device)
    phase = torch.tensor([c.phase for c in cases], device=device)
    available = env.library.lengths[motion] - (PRE + POST + 11)
    if bool((available < 0).any()):
        raise ValueError("Selected reference too short for the full evaluation window")
    frame = (available * phase).to(torch.long)
    inputs, _ = env.reset(options={"motion_id": motion, "frame_id": frame})
    kick = torch.tensor([(c.kick_x, c.kick_y) for c in cases], device=device)
    alive = torch.ones(len(cases), dtype=torch.bool, device=device)
    pushed = kick.abs().sum(-1) > 0
    integral = torch.zeros(len(cases), device=device)
    tail_sum = torch.zeros_like(integral)
    first_failure = torch.full_like(motion, -1)
    pairs = []
    prev = None
    for t in range(PRE + POST):
        x, history, goal = action_labels(inputs, teacher)
        if student is None:
            a = teacher.forward_features(history, goal)
        else:
            a = student((x - stats[0]) / stats[1])
        if env.foundation_config.action_clip is not None:
            limit = env.foundation_config.action_clip
            a = a.clamp(-limit, limit)
        if collect and prev is not None:
            # Do not treat the externally injected discontinuity as a local slope target.
            mask = alive & (~pushed if t == PRE + 1 else True)
            pairs.append(tuple(z[mask].cpu() for z in (*prev, x, a)))
        prev = (x, a)
        if t == PRE and bool((alive & pushed).any()):
            ids = (alive & pushed).nonzero().flatten()
            data = env.robot.data
            v = torch.cat((data.root_link_lin_vel_w[ids], data.root_link_ang_vel_w[ids]), -1)
            v[:, :2] += kick[ids]
            env.robot.write_root_link_velocity_to_sim(v, env_ids=ids)
        inputs, _, terminated, truncated, info = env.step(a)
        if bool((alive & truncated).any()):
            raise RuntimeError("Evaluation window ended before its planned horizon")
        failed = alive & terminated
        first_failure[failed] = t
        alive &= ~terminated
        if t >= PRE:
            e = info["metrics"]["root_position_error"].clamp(0, ERROR_CAP)
            e = torch.where(alive, e, ERROR_CAP)
            integral += e * env.step_dt
            if t >= PRE + POST - 10:
                tail_sum += e
    rows = []
    for c, start, error, tail, failure in zip(
        cases, frame.tolist(), integral.tolist(), (tail_sum / 10).tolist(), first_failure.tolist(), strict=True
    ):
        rows.append(asdict(c) | {"pushed": c.pushed, "frame": start, "error_m_s": error,
                                "tail_error_m": tail, "failed": failure >= 0,
                                "first_failure_step": failure, "recovered": failure < 0 and tail <= 0.30})
    data = tuple(torch.cat(z) for z in zip(*pairs, strict=True)) if collect else None
    return rows, data


def summarize(rows):
    groups = {}
    for name, use_push in (("nominal", False), ("disturbed", True)):
        group = [r for r in rows if r["pushed"] == use_push]
        n = len(group)
        groups[name] = {"n": n, "error_m_s": sum(r["error_m_s"] for r in group) / n,
                        "failures": sum(r["failed"] for r in group),
                        "recovered": sum(r["recovered"] for r in group)}
    return groups


def teacher_passes(summary):
    return all(summary[name]["recovered"] >= math.ceil(fraction * summary[name]["n"])
               for name, fraction in (("nominal", 0.90), ("disturbed", 0.75)))


def write_report(path, report):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def main():
    from src.tasks.supermimic.config.task_cfg import FoundationConfig
    from src.tasks.supermimic.env import SuperMimicEnv
    from src.tasks.supermimic.rl.inference import load_actor, validate_environment

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=10000)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if min(args.pairs, args.updates, args.batch_size) < 1:
        parser.error("Pair count, updates and batch size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Use an AIR job allocation with one visible CUDA GPU")
    device = "cuda:0"
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "pilot.json"
    if report_path.exists():
        raise FileExistsError(report_path)

    motion_files = json.loads(args.manifest.read_text())["clips"]
    train_clips, test_clips = split_clips([Path(c["file"]).stem for c in motion_files])
    teacher, saved, _ = load_actor(args.teacher, args.manifest, device)
    cfg = saved["config"]

    def make_env(clips, cases):
        env = SuperMimicEnv(FoundationConfig(
            manifest=args.manifest, clips=tuple(clips), num_envs=len(cases), seed=17,
            episode_length_s=100.0, action_clip=cfg.get("action_clip"),
            relative_orientation=cfg.get("relative_orientation", False),
            motion_anchor_ori_b=cfg.get("motion_anchor_ori_b", False),
        ), device=device)
        validate_environment(env, saved)
        return env

    test_cases = scenarios(test_clips, False)
    train_cases = scenarios(train_clips, True)
    test_env = make_env(test_clips, test_cases)
    train_env = make_env(train_clips, train_cases)
    test_rows, _ = rollout(test_env, teacher, test_cases)
    train_rows, _ = rollout(train_env, teacher, train_cases)
    teacher_test, teacher_train = summarize(test_rows), summarize(train_rows)
    gate = teacher_passes(teacher_train) and teacher_passes(teacher_test)
    report = {"source_revision": SOURCE, "torch": torch.__version__, "settings": vars(args) | {
        "manifest": str(args.manifest), "teacher": str(args.teacher), "output": str(args.output),
        "seeds": SEEDS, "weight": 1.0, "pre_steps": PRE, "post_steps": POST, "error_cap_m": ERROR_CAP,
    }, "train_clips": train_clips, "test_clips": test_clips, "teacher_gate_passed": gate,
        "teacher_train": teacher_train, "teacher_test": teacher_test,
        "teacher_train_rows": train_rows, "teacher_test_rows": test_rows, "students": []}
    write_report(report_path, report)
    print(f"Teacher gate {'PASSED' if gate else 'FAILED'}: {teacher_train=} {teacher_test=}", flush=True)
    if not gate:
        return  # An explicit scientific no-go, not a failed batch job.

    _, data = rollout(train_env, teacher, train_cases, collect=True)
    x0, a0, x1, a1 = data
    take = torch.randperm(len(x0), generator=torch.Generator().manual_seed(17))[:args.pairs]
    x0, x1, a0, a1 = (a[take].to(device) for a in (x0, x1, a0, a1))
    del data
    if len(take) < args.batch_size:
        raise RuntimeError("Insufficient contiguous teacher pairs")
    total_x = torch.cat((x0, x1))
    stats = total_x.mean(0), total_x.std(0, unbiased=False).clamp_min(0.05)
    del total_x
    x0, x1 = ((x - stats[0]) / stats[1] for x in (x0, x1))
    report["actual_pairs"] = len(take)
    report["pair_tensor_bytes"] = sum(t.numel() * t.element_size() for t in (x0, x1, a0, a1))
    write_report(report_path, report)

    for seed in SEEDS:
        for arm in ARMS:
            torch.manual_seed(seed)
            rng = torch.Generator(device=device).manual_seed(seed)
            model = student_model(x0.shape[-1], a0.shape[-1]).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
            started = perf_counter()
            for update in range(1, args.updates + 1):
                ids = torch.randint(len(x0), (args.batch_size,), generator=rng, device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = training_loss(model, arm, x0[ids], x1[ids], a0[ids], a1[ids])
                loss.backward()
                optimizer.step()
                if update % 500 == 0 or update == args.updates:
                    if not math.isfinite(loss.item()):
                        raise FloatingPointError(f"Nonfinite loss: {arm}, seed {seed}, step {update}")
                    path = args.output / f"student-{seed}-{arm}.pt"
                    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                "stats": stats, "step": update, "rng": rng.get_state()}, path.with_suffix(".tmp"))
                    path.with_suffix(".tmp").replace(path)
            torch.cuda.synchronize()
            seconds = perf_counter() - started
            rows, _ = rollout(test_env, teacher, test_cases, student=model.eval(), stats=stats)
            scores = summarize(rows)
            report["students"].append({"seed": seed, "arm": arm, "training_seconds": seconds,
                                       "scores": scores, "rows": rows})
            write_report(report_path, report)
            print(f"{arm} {seed}: {scores}", flush=True)

    comparisons = []
    for seed in SEEDS:
        scores = {row["arm"]: row["scores"] for row in report["students"] if row["seed"] == seed}
        for baseline in ("bc", "difference"):
            h = scores["hoas"]
            b = scores[baseline]
            comparisons.append({"seed": seed, "baseline": baseline,
                                "disturbed_error_delta_m_s": h["disturbed"]["error_m_s"] - b["disturbed"]["error_m_s"],
                                "nominal_error_delta_m_s": h["nominal"]["error_m_s"] - b["nominal"]["error_m_s"],
                                "disturbed_failure_delta": h["disturbed"]["failures"] - b["disturbed"]["failures"]})
    report["hoas_minus_baseline"] = comparisons
    write_report(report_path, report)


if __name__ == "__main__":
    main()
