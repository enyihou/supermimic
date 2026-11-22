/** Fail closed before copying a trained model to GitHub Pages. */
import { createHash } from 'node:crypto'
import { createReadStream } from 'node:fs'
import { lstat, readFile, readdir } from 'node:fs/promises'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { createGunzip } from 'node:zlib'
import { pipeline } from 'node:stream/promises'
import { Writable } from 'node:stream'
import {
  ACTION_DIM, ENGINE_VERSION, GOAL_DIM, SOURCE_PINS, STATE_DIM, parseTrainedManifest,
} from '../src/trained/contract.ts'

const root = fileURLToPath(new URL('..', import.meta.url))
const target = 320_000
const milestone = 32_000
const environments = 4096
const dataset = {
  id: 'bones-studio/seed', revision: '2f59b2077b9da34dd4e43618e705c7cb962c9a66', archive: 'g1.tar.gz',
  sha256: '52580ea8bced72ea9e2ff1e7b68f01c51c7f1099581e9a46b7c87e1dec106d8a',
  csv_files: 142_220,
}
const heldOut = {
  source: 'lvhaidong/LAFAN1_Retargeting_Dataset', revision: 'ce1572906efe6157840e8474d5a0d7aa87481e74', clips: 40,
  clip_manifest_sha256: '545cb292fa1fe99809e7bc289a075727a78ca305898f1a71d57b5dd15a92b27d',
}
const curveIds = ['train-episode-reward', 'train-episode-length', 'train-world-model-loss']
const names = ['ATTRIBUTION.md', 'Unitree-LICENSE.txt', 'actor.onnx', 'mjlab-LICENSE.txt', 'scene.mjbpack']
const allowedPublication = {
  'Unitree-LICENSE.txt': 'a5d73fc4aca9074e3e6fe0b1a0ba763cf9514b2249b7390ed20fe8d53630bf25',
  'mjlab-LICENSE.txt': '8ca04d9dbbccd20f0f1842fb33e13949f9e57064b150dad36af8e7a751ca3321',
}
const metrics = [
  'evaluated_clips', 'completed_clips', 'failed_clips', 'completion_rate', 'evaluated_steps', 'reference_steps',
  'mean_normalized_return', 'mean_root_position_error_m', 'mean_heading_aligned_body_position_error_m',
]
const checks = [
  'slurm_terminal_state', 'job_status', 'success_marker', 'source_pins', 'dataset_checksum', 'packed_motion_checksum',
  'target_step', 'checkpoint_archive_integrity', 'milestone_retention',
]
const auditKeys = [
  'schema_version', 'phase', 'source_commits', 'reproduction_commit', 'dataset_sha256', 'motion_archive_sha256',
  'code_sha256', 'environment_sha256', 'allocations', 'num_envs', 'checkpoint_step', 'checkpoint_job_id',
  'checkpoint_archive_sha256', 'checkpoint_archive_bytes', 'checkpoint_artifact_sha256', 'retained_checkpoints', 'motions',
  'motion_manifest', 'outputs_archive_sha256', 'verified_checks', 'held_out_evaluation_verified',
]
const expectedAttribution = '# SuperMimic browser policy\n\n' +
  'Training data includes Motion Data by [Bones Studio](https://bones.studio/). ' +
  'Use of the underlying dataset is subject to the BONES Motion Capture Dataset License Agreement.\n\n' +
  'The scene is compiled from the pinned [mjlab source](https://github.com/HybridRobotics/mjlab_research), ' +
  'licensed under Apache-2.0; see mjlab-LICENSE.txt. The model contains [Unitree G1 robot geometry]' +
  '(https://github.com/unitreerobotics/unitree_mujoco/tree/1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d/unitree_robots/g1), ' +
  'BSD-3-Clause; see Unitree-LICENSE.txt.\n'
const cohortNotice = (approximate) => approximate ?
  "\nTraining cohort: approximate Appendix E selection, not the authors' exact clip list.\n" :
  "\nTraining cohort: authors' Appendix E clip list.\n"
const hex = (n, value) => typeof value === 'string' && value.length === n && /^[a-f0-9]+$/.test(value) && /[1-9a-f]/.test(value)
const sha256 = (x) => hex(64, x)
const positive = (value) => Number.isSafeInteger(value) && value > 0
const object = (value) => typeof value === 'object' && value !== null && !Array.isArray(value)
const keys = (value, expected) => object(value) && sameList(Object.keys(value), expected)
function sameList(actual, expected) {
  if (!Array.isArray(actual) || actual.length !== expected.length) return false
  const sorted = [...expected].sort()
  return [...actual].sort().every((value, i) => value === sorted[i])
}
const pinned = (value, extra = []) => keys(value, [...Object.keys(SOURCE_PINS), ...extra]) &&
  Object.entries(SOURCE_PINS).every(([name, sha]) => value[name] === sha)
const jobId = (value) => typeof value === 'string' && /^[1-9]\d*$/.test(value)
const jobFields = ['job_id', 'state', 'exit_code', 'allocated_gpus']
const completed = (job) => job && jobId(job.job_id) && job.state === 'COMPLETED' && job.exit_code === '0:0' && job.allocated_gpus === 1

function check(condition, description) {
  if (!condition) throw new Error(description)
}

async function file(path, label, maximum = 100 * 1024 * 1024 - 1) {
  const info = await lstat(path)
  check(info.isFile() && positive(info.size) && info.size <= maximum, `${label}: expected a nonempty bounded regular file`)
  return info.size
}

async function directory(path, label, expected) {
  check((await lstat(path)).isDirectory(), `${label}: expected a real directory (no symlinks)`)
  if (expected) check(sameList(await readdir(path), expected), `${label}: unexpected or missing files`)
}

async function json(path, label) {
  await file(path, label, 256 * 1024)
  return JSON.parse(await readFile(path, 'utf8'))
}

async function hash(path) {
  const digest = createHash('sha256')
  for await (const data of createReadStream(path)) digest.update(data)
  return digest.digest('hex')
}

function checkLearningCurves(curves, checkpoint) {
  check(keys(curves, ['schema_version', 'coverage', 'metrics_jsonl_sha256', 'series']) && curves.schema_version === 1 &&
    Array.isArray(curves.series) && curves.series.length <= curveIds.length,
  'Learning curves must use the allowlisted aggregate schema')
  const coverage = curves.coverage
  check(keys(coverage, ['scope', 'resumed_from_iterations', 'log_interval', 'first_logged_step', 'last_logged_step', 'logged_rows']) &&
    coverage.scope === 'final-allocation-only' && Number.isSafeInteger(coverage.resumed_from_iterations) &&
    coverage.resumed_from_iterations >= 0 && coverage.resumed_from_iterations <= checkpoint && positive(coverage.log_interval) &&
    Number.isSafeInteger(coverage.logged_rows) && coverage.logged_rows >= 0,
  'Learning curve coverage is invalid')
  const { resumed_from_iterations: resumed, log_interval: stride, logged_rows: count,
    first_logged_step: first, last_logged_step: last } = coverage
  const atCadence = (step) => Number.isSafeInteger(step) && step > resumed && step <= checkpoint && (step - 1) % stride === 0
  const firstExpected = Math.ceil(resumed / stride) * stride + 1
  const expectedRows = firstExpected > checkpoint ? 0 : Math.floor((checkpoint - firstExpected) / stride) + 1
  check(count === expectedRows, 'Learning curve coverage row count does not span the final allocation on the logged cadence')
  if (count === 0) {
    check(curves.metrics_jsonl_sha256 === null && first === null && last === null && curves.series.length === 0,
      'Empty learning curves cannot include source hashes, steps or series')
    return
  }
  check(sha256(curves.metrics_jsonl_sha256) && first === firstExpected && last === firstExpected + (count - 1) * stride,
  'Learning curve source hash, coverage steps or row count are inconsistent')
  let previousId = -1
  for (const series of curves.series) {
    check(keys(series, ['id', 'observation_count', 'points']), 'Learning curve series contains unexpected fields')
    const index = curveIds.indexOf(series.id)
    check(index > previousId && positive(series.observation_count) && series.observation_count <= count &&
      Array.isArray(series.points) && series.points.length === Math.min(series.observation_count, 128),
    'Learning curve series ID, ordering or point count is invalid')
    previousId = index
    let previousStep = 0
    for (const point of series.points) {
      check(keys(point, ['step', 'value']) && atCadence(point.step) && point.step >= first && point.step <= last && point.step > previousStep &&
        Number.isFinite(point.value) && Math.abs(point.value) <= Number.MAX_SAFE_INTEGER &&
        (index === 0 || (index === 1 ? point.value > 0 : point.value >= 0)),
      'Learning curve point is not an allowlisted bounded aggregate on the logged step cadence')
      previousStep = point.step
    }
  }
}

function checkAudit(audit, manifest) {
  const attached = audit?.held_out_evaluation_verified === true
  const curves = object(audit) && Object.hasOwn(audit, 'learning_curves')
  check(keys(audit, [...auditKeys, ...(attached ? ['held_out_evaluation'] : []), ...(curves ? ['learning_curves'] : [])]) &&
    audit.schema_version === 1 && audit.phase === 'full' && audit.checkpoint_step === target && audit.num_envs === environments &&
    typeof audit.held_out_evaluation_verified === 'boolean' && sameList(audit.verified_checks, attached ? [...checks, 'held_out_evaluation'] : checks) &&
    (!manifest || attached), 'Full audit must verify the exact 320,000-step, 4,096-environment run (and held-out evaluation before model publication)')
  check(pinned(audit.source_commits) && hex(40, audit.reproduction_commit) && audit.dataset_sha256 === dataset.sha256 &&
    (!manifest || (audit.reproduction_commit === manifest.source_commits.reproduction && audit.dataset_sha256 === manifest.dataset_sha256)),
  'Audit source revisions or dataset hash differ from release requirements')
  check(Array.isArray(audit.allocations) && audit.allocations.length > 0 &&
    audit.allocations.every((job) => keys(job, jobFields) && completed(job)) && jobId(audit.checkpoint_job_id),
  'Missing successful single-GPU Slurm training evidence')
  for (const key of ['motion_archive_sha256', 'code_sha256', 'environment_sha256', 'checkpoint_archive_sha256', 'outputs_archive_sha256'])
    check(sha256(audit[key]), `Missing verified ${key}`)
  const hashes = audit.checkpoint_artifact_sha256
  const artifacts = ['actor.pt', 'world_model.pt', 'training_state.pt', 'run-config.json']
  check((keys(hashes, artifacts) || keys(hashes, [...artifacts, 'critic.pt'])) && Object.values(hashes).every(sha256) &&
    (!manifest || (hashes['actor.pt'] === manifest.checkpoint_actor_sha256 && hashes['world_model.pt'] === manifest.checkpoint_world_model_sha256)),
  'Export is not bound to the audited final checkpoint artifacts')
  check(positive(audit.checkpoint_archive_bytes) && Array.isArray(audit.retained_checkpoints) &&
    audit.retained_checkpoints.length === target / milestone && audit.retained_checkpoints.every((item, i) =>
      keys(item, ['step', 'archive_sha256', 'archive_bytes']) && item.step === milestone * (i + 1) &&
      sha256(item.archive_sha256) && positive(item.archive_bytes)) &&
    audit.retained_checkpoints.at(-1).archive_sha256 === audit.checkpoint_archive_sha256 &&
    audit.retained_checkpoints.at(-1).archive_bytes === audit.checkpoint_archive_bytes,
  'Audit does not verify all milestone checkpoints and the final archive')
  const cohort = audit.motion_manifest
  check(keys(cohort, ['selection', 'sha256', 'clipset_sha256', 'clips', 'frames', 'input_fps', 'fps', 'selection_id', 'approximate', 'allowlist_sha256']) &&
    cohort.selection === 'explicit-allowlist' && [cohort.sha256, cohort.clipset_sha256, cohort.allowlist_sha256].every(sha256) &&
    cohort.input_fps === 120 && cohort.fps === 50 && positive(cohort.clips) && cohort.clips <= dataset.csv_files &&
    positive(cohort.frames) && cohort.frames >= 2 * cohort.clips && (
      (cohort.selection_id === 'supermimic-v3-authors-manifest' && cohort.approximate === false && cohort.clips === 129_514) ||
      (cohort.selection_id === 'supermimic-v3-appendix-e-approx-v1' && cohort.approximate === true)
    ) && keys(audit.motions, ['clips', 'frames', 'fps']) && ['clips', 'frames', 'fps'].every((key) => audit.motions[key] === cohort[key]),
  'Training motion selection, clip counts or sample rates do not match the audited exact or approximate cohort')
  if (manifest) {
    const fields = ['selection_id', 'approximate', 'allowlist_sha256', 'clips']
    check(keys(manifest.training_cohort, [...fields, 'motion_archive_sha256']) &&
      fields.every((key) => manifest.training_cohort[key] === cohort[key]) &&
      manifest.training_cohort.motion_archive_sha256 === audit.motion_archive_sha256,
    'Model package must label and match the verified training cohort')
  }
  if (curves) checkLearningCurves(audit.learning_curves, audit.checkpoint_step)
  if (!attached) return

  const evaluation = audit.held_out_evaluation
  check(keys(evaluation, [...jobFields, 'suite', 'metrics', 'metrics_sha256', 'outputs_archive_sha256']) && completed(evaluation) &&
    evaluation.job_id !== audit.checkpoint_job_id && audit.allocations.every((job) => evaluation.job_id !== job.job_id) &&
    sha256(evaluation.metrics_sha256) && sha256(evaluation.outputs_archive_sha256), 'Missing successful independent held-out Slurm evidence')
  const suite = evaluation.suite
  check(keys(suite, [...Object.keys(heldOut), 'archive_sha256']) &&
    Object.entries(heldOut).every(([key, value]) => suite[key] === value) && sha256(suite.archive_sha256) &&
    ![audit.dataset_sha256, audit.motion_archive_sha256].includes(suite.archive_sha256), 'Held-out suite does not match the predeclared independent corpus')
  const m = evaluation.metrics
  check(keys(m, metrics) && metrics.every((key) => Number.isFinite(m[key])) &&
    (!manifest || (keys(manifest.metrics, metrics) && metrics.every((key) => manifest.metrics[key] === m[key]))),
  'Release metrics must exactly match the audited held-out aggregates')
  check(m.evaluated_clips === heldOut.clips && Number.isSafeInteger(m.completed_clips) &&
    m.completed_clips >= 0 && m.completed_clips <= m.evaluated_clips && m.failed_clips === m.evaluated_clips - m.completed_clips &&
    m.completion_rate === m.completed_clips / m.evaluated_clips &&
    positive(m.evaluated_steps) && positive(m.reference_steps) && m.evaluated_clips <= m.evaluated_steps &&
    m.evaluated_steps <= m.reference_steps && (m.failed_clips !== 0 || m.evaluated_steps === m.reference_steps) &&
    m.mean_normalized_return <= 1 + 1e-5 && m.mean_root_position_error_m >= 0 && m.mean_heading_aligned_body_position_error_m >= 0,
  'Held-out aggregate counts or metrics are inconsistent')
}

async function verifyPublicAudit(publicRoot, manifest) {
  const evidence = join(publicRoot, 'reproduction')
  const auditFile = join(evidence, 'full-audit.json')
  if (!manifest) {
    const present = await lstat(auditFile).then(() => true, (error) => {
      if (error.code === 'ENOENT') return false
      throw error
    })
    if (!present) return
  }
  await directory(evidence, 'Public reproduction evidence')
  const preflight = await json(join(evidence, 'preflight.json'), 'Dataset preflight')
  check(preflight.schema_version === 1 && pinned(preflight.source_commits) &&
    Object.entries(dataset).every(([key, value]) => preflight.dataset?.[key] === value) &&
    preflight.dataset.local_sha256_verified === true && preflight.dataset.air_sha256_verified === true &&
    preflight.dataset.local_archive_members_verified === true, 'The pinned dataset must be verified locally and after AIR transfer')
  const audit = await json(auditFile, 'Full training and held-out audit')
  checkAudit(audit, manifest)
}

async function verifyActor(path, manifest) {
  const ort = await import('onnxruntime-web')
  ort.env.wasm.numThreads = 1
  const policy = await ort.InferenceSession.create(await readFile(path), { executionProviders: ['wasm'] })
  try {
    check(policy.inputNames.join(',') === 'state,goal' && policy.outputNames.join(',') === 'action',
      'Unexpected ONNX inputs or outputs (an actor-only export is required)')
    async function infer(state, goal) {
      const inputs = {
        state: new ort.Tensor('float32', Float32Array.from(state), [1, STATE_DIM]),
        goal: new ort.Tensor('float32', Float32Array.from(goal), [1, GOAL_DIM]),
      }
      try {
        const result = (await policy.run(inputs)).action
        try {
          check(result.dims.join(',') === `1,${ACTION_DIM}` && result.data instanceof Float32Array && result.data.every(Number.isFinite),
          'ONNX actor must emit finite float32 actions in the pinned shape')
          return result.data.slice()
        } finally { result.dispose() }
      } finally { Object.values(inputs).forEach((tensor) => tensor.dispose()) }
    }
    let baseline
    for (const { state, goal, action } of manifest.policy.parity) {
      const output = await infer(state, goal)
      check(output.every((value, i) => Math.abs(value - action[i]) <= 2e-5 + 2e-4 * Math.abs(action[i])),
        'ONNX actor fails its PyTorch action parity vectors')
      baseline ??= output
    }
    // Independently perturb each input. A constant response or identity-only
    // graph can satisfy shapes and supplied vectors but is not a tracking actor.
    const [a, b] = manifest.policy.parity
    const different = (output) => output.some((value, i) => Math.abs(value - baseline[i]) > 1e-6)
    check(different(await infer(b.state, a.goal)) && different(await infer(a.state, b.goal)),
      'ONNX tracking actor must respond to both live state and motion goal')
  } finally { await policy.release() }
}

async function verifyScene(path, robot) {
  const bytes = new Uint8Array(robot.scene_uncompressed_bytes)
  let offset = 0
  await pipeline(createReadStream(path), createGunzip(), new Writable({
    write(chunk, _encoding, done) {
      if (offset + chunk.length > bytes.length) return done(new Error('Scene exceeds its declared decompressed size'))
      bytes.set(chunk, offset)
      offset += chunk.length
      done()
    },
  }))
  check(offset === bytes.length, 'Scene decompressed size does not match manifest')
  const { default: load } = await import('@mujoco/mujoco')
  const mj = await load()
  check(mj.mj_versionString() === ENGINE_VERSION, 'Release gate MuJoCo version mismatch')
  const vfs = new mj.MjVFS()
  let model, data
  try {
    const path = '/verify-trained-g1.mjb'
    mj.FS.writeFile(path, bytes, { canOwn: true })
    try { model = mj.MjModel.from_binary_path(path, vfs) }
    finally { mj.FS.unlink(path) }
    check(model && model.nq === 36 && model.nv === 35 && model.nu === ACTION_DIM && model.nmesh >= 25 &&
      model.opt.timestep === robot.timestep && robot.bodies.length === 30, 'MJB is not the G1 scene specified by its manifest')
    const id = (type, name) => mj.mj_name2id(model, type.value, name)
    const o = mj.mjtObj
    const bodyIds = robot.bodies.map(({ name }) => id(o.mjOBJ_BODY, name))
    const free = id(o.mjOBJ_JOINT, robot.free_joint)
    const key = id(o.mjOBJ_KEY, robot.keyframe)
    check(bodyIds.every((b, i) => b > 0 && model.body_parentid[b] === (i ? bodyIds[robot.bodies[i].parent] : 0)) &&
      robot.root_body === 'robot/pelvis' && free >= 0 && key >= 0 &&
      model.jnt_type[free] === mj.mjtJoint.mjJNT_FREE.value && model.jnt_bodyid[free] === bodyIds[0], 'G1 body tree or base is invalid')
    data = new mj.MjData(model)
    mj.mj_resetDataKeyframe(model, data, key)
    check(robot.joints.every(({ name, actuator }, i) => {
      const j = id(o.mjOBJ_JOINT, name), a = id(o.mjOBJ_ACTUATOR, actuator)
      return j > 0 && a >= 0 && model.jnt_type[j] === mj.mjtJoint.mjJNT_HINGE.value && model.actuator_trnid[2 * a] === j &&
        Math.abs(data.qpos[model.jnt_qposadr[j]] - robot.action_offset[i]) <= 1e-6
    }), 'G1 joints, actuator mappings or reference offsets do not match the manifest')
  } finally {
    data?.delete()
    model?.delete()
    vfs.delete()
  }
}

async function verifyModelRelease(publicRoot, modelRoot, id) {
  const release = join(modelRoot, id)
  await directory(release, 'Versioned model release', ['manifest.json', ...names])
  const manifest = parseTrainedManifest(await json(join(release, 'manifest.json'), 'Model manifest'))
  check(keys(manifest, [
    'format', 'model_id', 'task', 'dataset_sha256', 'checkpoint_step', 'source_commits', 'checkpoint_actor_sha256',
    'checkpoint_world_model_sha256', 'training_cohort', 'metrics', 'artifacts', 'policy', 'robot',
  ]) && manifest.model_id === id && manifest.checkpoint_step === target && manifest.dataset_sha256 === dataset.sha256 &&
    pinned(manifest.source_commits, ['reproduction']) && hex(40, manifest.source_commits.reproduction) &&
    sha256(manifest.checkpoint_actor_sha256) && sha256(manifest.checkpoint_world_model_sha256),
  'Model package does not identify the exact audited reproduction')
  check(keys(manifest.artifacts, names) && keys(manifest.policy, [
    'state_dim', 'goal_dim', 'action_dim', 'goal_frames', 'goal_stride', 'parity', 'export_max_abs_error',
  ]) && manifest.policy.parity.every((sample) => keys(sample, ['state', 'goal', 'action'])) && keys(manifest.robot, [
    'engine_version', 'scene_uncompressed_bytes', 'timestep', 'substeps', 'keyframe', 'root_body', 'free_joint',
    'joints', 'bodies', 'action_scale', 'action_offset',
  ]) && manifest.robot.joints.every((item) => keys(item, ['name', 'actuator'])) &&
    manifest.robot.bodies.every((item) => keys(item, ['name', 'parent'])), 'Model metadata contains unexpected data')

  await verifyPublicAudit(publicRoot, manifest)

  for (const name of names) {
    const artifact = manifest.artifacts[name]
    const path = join(release, name)
    check(keys(artifact, ['sha256', 'bytes']) && sha256(artifact.sha256) && await file(path, name) === artifact.bytes &&
      await hash(path) === artifact.sha256, `${name}: missing or mismatched artifact size/hash`)
    if (Object.hasOwn(allowedPublication, name)) check(allowedPublication[name] === artifact.sha256, `${name}: license text differs from the pinned source`)
  }
  check(await readFile(join(release, 'ATTRIBUTION.md'), 'utf8') === expectedAttribution + cohortNotice(manifest.training_cohort.approximate),
    'Model attribution differs from the expected release notice or training cohort label')
  await verifyActor(join(release, 'actor.onnx'), manifest)
  await verifyScene(join(release, 'scene.mjbpack'), manifest.robot)
  return id
}

export async function verifyRelease(publicRoot = join(root, 'public')) {
  const modelRoot = join(publicRoot, 'models')
  await directory(publicRoot, 'Public tree')
  await directory(modelRoot, 'Public model tree')
  const registry = await json(join(modelRoot, 'releases.json'), 'Model registry')
  const hasModels = object(registry) && Object.hasOwn(registry, 'models')
  check(keys(registry, hasModels ? ['schema_version', 'active', 'models'] : ['schema_version', 'active']) &&
    registry.schema_version === 1, 'Invalid model registry')

  const ids = hasModels ? registry.models : registry.active === null ? [] : [registry.active]
  check(Array.isArray(ids) && ids.length === new Set(ids).size && ids.every((id) =>
    typeof id === 'string' && /^[a-z0-9][a-z0-9-]{0,63}$/.test(id) &&
    !/(?:^|-)(synthetic|fixture|mock|test|smoke|benchmark|untrained|random)(?:-|$)/.test(id)),
  'Registry must identify trained releases, not benchmark or QA fixtures')
  check(registry.active === null || ids.includes(registry.active), 'Active model must be listed in the registry')
  if (ids.length === 0) {
    await directory(modelRoot, 'Empty model registry', ['releases.json'])
    await verifyPublicAudit(publicRoot, null)
    return null
  }

  await directory(modelRoot, 'Public model tree', ['releases.json', ...ids])
  for (const id of ids) await verifyModelRelease(publicRoot, modelRoot, id)
  return registry.active ?? ids[0]
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    check(process.argv.length <= 3, 'Usage: node scripts/verify-release.mjs [PUBLIC_ROOT]')
    const id = await verifyRelease(process.argv[2])
    console.log(id ? `Verified audited browser release: ${id}` : 'No trained release in the public registry (verified empty)')
  } catch (error) {
    console.error(`Release verification failed: ${error.message}`)
    process.exitCode = 1
  }
}
