# Workspace instructions

Use the AIR cluster's GPUs for compute-intensive work in this workspace, such as model training, large inference or evaluation runs, and experiment sweeps. Keep ordinary edits and lightweight checks local.

Always read the current AIR server guide before starting any work on the cluster:

```sh
ssh air 'cat /etc/cluster/AGENTS_SERVER.md'
```

Read `/etc/cluster/server-guide/RUNBOOK.md` before packaging or submitting jobs. Follow the current server guide for GPU limits, job submission, and storage. Submit batch jobs with `lab-submit` from `gpu01`, use `srun` for interactive allocations, and never SSH directly into compute nodes.

## Planning and progress

Always use the `planning-with-files` skill for this project. Read its `SKILL.md` at the start of each task and follow its workflow. For complex work, create or refresh `task_plan.md` and `findings.md` alongside the existing `progress.md` before implementation.

Read [progress.md](progress.md) when resuming work and update it before sending every project progress or status update. Keep it concise: record the date, verified results or changes, open blockers, and the next step. Replace stale status and omit credentials. Keep the log local unless the user asks to share it.
