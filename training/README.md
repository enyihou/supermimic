# Official SuperMimic reproduction

This directory is a reproducible launcher for the authors' released code. It
does not contain BONES-SEED, checkpoints, or credentials. The source pins are
the commits checked for this workspace.

- `supermimic`: `5fb92297ab4fe3d86629d9f0c635334791bb0d5f`
- `mjlab_research`: `379c55e88b1dec73f93943e524a879b2ebbf45c5`
- Dataset: BONES-SEED G1 archive, filtered to the paper's 129,514 clips

## Prepare on gpu01

```sh
./training/prepare_official.sh /home/$(id -un)/supermimic-official
```

Put the gated `g1.tar.gz` in a private input directory and set
`BONES_G1_ARCHIVE` when submitting. The archive is staged to each allocation's
`$LOCAL_SCRATCH`, where training reads it.

## Smoke and full run

Run the smoke first through `lab-submit` on gpu01:

```sh
lab-submit --kind gpu -- training/run_smoke.sbatch \
  /home/$(id -un)/supermimic-official /private/input/bones/g1.tar.gz
```

The full recipe is a one-GPU, 320,000-iteration run. It requires an approved
private archive directory and a dataset archive path:

```sh
lab-submit --kind gpu -- training/run_full.sbatch \
  /home/$(id -un)/supermimic-official /private/input/bones/g1.tar.gz \
  /approved/private/archive/supermimic
```

The scripts stop before submission if the archive is missing, the destination
is not writable, or the source is not at the pinned commits. Checkpoints are
written every 1,000 iterations by the official runner and copied into a
timestamped private backup archive every 30 minutes. The backup loop also runs
on batch termination so a completed checkpoint is preserved before scratch is
removed.

The full run does not publish weights or motion data. Evaluation belongs in a
separate job after the run is restored and verified.
