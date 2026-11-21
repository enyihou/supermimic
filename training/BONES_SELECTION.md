# Approximate Appendix E selection

The authors' [SuperMimic v3 Appendix E](https://arxiv.org/pdf/2610.28480v3)
describes a name filter followed by a 50 Hz geometry filter. It reports
142,220 → 129,667 → 129,514 clips, but does not publish the exact clip
identities, complete keyword list, or numeric posture/stationarity gates.
Use the authors' exact manifest if one is obtained. A matching clip count
would not establish clip-set equivalence.

The alternative selector uses the authenticated default keywords in
[NVIDIA's released SONIC filter](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/1e1ec4f5f27e6c3e03048f693bd7c60cce04afda/gear_sonic/data_process/filter_and_copy_bones_data.py).
The selector parses the static list from the pinned file; it does not run
NVIDIA code. It preserves potentially feasible flat-ground skills by
omitting SONIC's bans on acrobatics, cartwheels, handstands, shoulder rolls,
Kozak dance, monkey jumps, and safety rolls, plus the unexplained `230m`.
It retains the other SONIC case-insensitive substring checks against the
file's immediate parent and stem. It also matches the paper's `box` and
`sit` examples as complete tokens and matches the phrases `jump off 50cm`
and `look over fence` with common filename separators. This adaptation is
ours, not the authors' list.

Geometry checks use the pinned mjlab G1 model, the same BONES conversion
as training, and a z=0 plane in MuJoCo:

- The minimum z of every robot body origin (excluding the world) stays
  above 0.20 m for at least 30 frames.
- Some body's origin falls below −0.12 m, or the minimum body z is below
  −0.08 m for strictly more than 80% of the frames.
- The minimum signed distance from **each** foot's seven collision capsules
  to the plane is greater than 0.02 m for at least 25 consecutive frames
  where the pelvis is at least 0.70 m high, tilted at most 20°, translating
  no faster than 0.15 m/s, and rotating no faster than 0.40 rad/s. Those four
  extra numeric gates are our predeclared interpretation of the paper's
  qualitative conditions, not tuned to match the reported clip count.

Distances come from MuJoCo's
[`mj_geomDistance`](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-geomdistance);
the selector does not treat ankle positions as foot contact or modify the
root height. The signed distance query is only needed at frames meeting
the floating-feet posture/velocity gate. An invalid CSV, unsafe archive
structure, failed conversion, or inconsistent FK aborts without a keep list.

## Review and run

This is an approximate filter, not authorization to replace the authors'
training set. Review its assumptions and get approval before using its list
for the full reproduction. A full pass is considerable compute and I/O:
use AIR allocated node-local scratch and the current cluster runbook.

Put the pinned SONIC source file and verified BONES archive in the
approved private staging location. The source SHA-256 is
`b6b50d441a4876a382eb3ff33ba940c487d1647230b5ecacaa37cfcf521258da`.
Use the pinned mjlab checkout's
`src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml`, alongside its mesh assets.
The output parent must be owner-only and outside the public repo.

```sh
python3 training/select_bones_clips.py \
  "$NODE_LOCAL_BONES_ARCHIVE" "$NODE_LOCAL_PRIVATE/appendix-e-approx-v1" \
  --sha256 "$BONES_SHA256" --g1-xml "$PINNED_G1_XML" \
  --sonic-source "$PINNED_SONIC_FILTER_SOURCE" --workers 8 --accept-approximate
```

On success the output directory contains `keep.txt` (exact tar member
names, suitable for `pack_bones.py --keep-list`), `summary.json` (counts,
configuration and hashes), and `SHA256SUMS`. Keep the member list private.
Use the verified `keep.txt` digest with `--keep-list-sha256` and use
`--selection-id supermimic-v3-appendix-e-approx-v1` when packing this cohort.
Any published summary must only include aggregates and fingerprints; do not
publish the names, source CSVs, or resulting motion arrays. Keep
`approximate: true` in downstream provenance.
