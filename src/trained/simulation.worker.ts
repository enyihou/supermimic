/// <reference lib="webworker" />

import type { MjData, MjModel } from '@mujoco/mujoco'
import mujocoWasm from '@mujoco/mujoco/mujoco.wasm?url'
import ortWasm from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url'
import { ACTION_DIM, ENGINE_VERSION, GOAL_DIM, STATE_DIM, parseTrainedManifest,
  type TrainedManifest, type TrainedMesh, type TrainedMotion, type WorkerRequest, type WorkerResponse } from './contract'
import { actorGoal, actorState } from './observations'

const worker = self as unknown as DedicatedWorkerGlobalScope
const post = (message: WorkerResponse, transfers: Transferable[] = []) => worker.postMessage(message, transfers)
let started = false
let playing = true
let resetRequested = false
let motion: TrainedMotion = 'stand'

function fail(error: unknown) {
  post({ type: 'error', message: error instanceof Error ? error.message : 'The browser simulation failed' })
}

async function readManifest(url: URL): Promise<TrainedManifest> {
  if (url.origin !== worker.location.origin || !['https:', 'http:'].includes(url.protocol)) {
    throw new Error('The trained-model manifest must be served with this website')
  }
  const response = await fetch(url, { cache: 'no-cache', redirect: 'error' })
  if (!response.ok || !response.headers.get('content-type')?.includes('json')) {
    throw new Error('No verified trained-model release is available at this URL')
  }
  return parseTrainedManifest(await response.json())
}

async function readArtifact(base: URL, manifest: TrainedManifest, name: 'actor.onnx' | 'scene.mjbpack') {
  const response = await fetch(new URL(name, base), { redirect: 'error' })
  if (!response.ok) throw new Error(`Could not load verified ${name}`)
  const bytes = await response.arrayBuffer()
  const want = manifest.artifacts[name]
  if (bytes.byteLength !== want.bytes) throw new Error(`${name} size does not match its release manifest (${bytes.byteLength} vs ${want.bytes} bytes)`)
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))
  const hash = Array.from(digest, (b) => b.toString(16).padStart(2, '0')).join('')
  if (hash !== want.sha256) throw new Error(`${name} hash does not match its release manifest`)
  return bytes
}

async function readScene(base: URL, manifest: TrainedManifest) {
  const compressed = await readArtifact(base, manifest, 'scene.mjbpack')
  const reader = new Blob([compressed]).stream().pipeThrough(new DecompressionStream('gzip')).getReader()
  const data = new Uint8Array(manifest.robot.scene_uncompressed_bytes)
  let offset = 0
  try {
    for (;;) {
      const { value, done } = await reader.read()
      if (done) break
      if (offset + value.length > data.length) throw new Error('Exported scene uncompressed size exceeds its release manifest')
      data.set(value, offset)
      offset += value.length
    }
  } finally {
    await reader.cancel()
  }
  if (offset !== data.length) throw new Error('Exported scene uncompressed size does not match its release manifest')
  return data
}

function renderMeshes(model: MjModel, bodyIds: number[], meshType: number): TrainedMesh[] {
  const bodies = new Map(bodyIds.map((id, i) => [id, i]))
  const meshes: TrainedMesh[] = []
  for (let geom = 0; geom < model.ngeom; geom++) {
    const bodyIndex = bodies.get(model.geom_bodyid[geom])
    // The pinned mjlab G1 uses group 2 for visual-only meshes; don't render the
    // collision proxies, ground plane, or any unrelated geometry.
    if (bodyIndex === undefined || model.geom_type[geom] !== meshType || model.geom_group[geom] !== 2) continue
    const mesh = model.geom_dataid[geom]
    const va = model.mesh_vertadr[mesh] * 3
    const fa = model.mesh_faceadr[mesh] * 3
    const vertices = new Float32Array(model.mesh_vert.subarray(va, va + model.mesh_vertnum[mesh] * 3))
    const triangles = new Uint32Array(model.mesh_face.subarray(fa, fa + model.mesh_facenum[mesh] * 3))
    const p = geom * 3
    const q = geom * 4
    const material = model.geom_matid[geom]
    const colors: ArrayLike<number> = material < 0 ? model.geom_rgba : model.mat_rgba
    const c = material < 0 ? q : 4 * material
    if (mesh < 0 || vertices.length < 9 || triangles.length < 3 || !vertices.every(Number.isFinite) ||
        triangles.some((i) => 3 * i >= vertices.length)) throw new Error('Invalid robot geometry in the exported scene')
    meshes.push({
      name: `g1-mesh-${geom}`, bodyIndex, vertices, triangles,
      position: [model.geom_pos[p], model.geom_pos[p + 1], model.geom_pos[p + 2]],
      quaternion: [model.geom_quat[q], model.geom_quat[q + 1], model.geom_quat[q + 2], model.geom_quat[q + 3]],
      rgba: [colors[c], colors[c + 1], colors[c + 2], colors[c + 3]],
    })
  }
  if (!meshes.length) throw new Error('Exported scene has no G1 visual meshes')
  return meshes
}

async function start(manifestUrl: string) {
  const url = new URL(manifestUrl)
  const manifest = await readManifest(url)
  const [actor, mjb, { default: loadMujoco }, ort] = await Promise.all([
    readArtifact(url, manifest, 'actor.onnx'), readScene(url, manifest),
    import('@mujoco/mujoco'), import('onnxruntime-web/wasm'),
  ])
  ort.env.wasm.numThreads = 1 // GitHub Pages does not guarantee cross-origin isolation.
  ort.env.wasm.wasmPaths = { wasm: new URL(ortWasm, worker.location.href).href }
  const [mujoco, policy] = await Promise.all([
    loadMujoco({ locateFile: (path: string) => path.endsWith('.wasm') ? mujocoWasm : path }),
    ort.InferenceSession.create(actor, { executionProviders: ['wasm'] }),
  ])
  if (mujoco.mj_versionString() !== ENGINE_VERSION) throw new Error('MuJoCo version does not match the trained-model release')
  if (policy.inputNames.join(',') !== 'state,goal' || policy.outputNames.join(',') !== 'action') {
    throw new Error('Trained actor model input/output names do not match the release contract')
  }
  async function infer(state: Float32Array, goal: Float32Array) {
    const stateTensor = new ort.Tensor('float32', state, [1, STATE_DIM])
    const goalTensor = new ort.Tensor('float32', goal, [1, GOAL_DIM])
    try {
      const { action } = await policy.run({ state: stateTensor, goal: goalTensor })
      try {
        if (!(action.data instanceof Float32Array) || action.dims[0] !== 1 || action.dims[1] !== ACTION_DIM ||
            action.dims.length !== 2 || !action.data.every(Number.isFinite)) {
          throw new Error('Invalid action from exported trained policy')
        }
        return action.data.slice()
      } finally {
        action.dispose()
      }
    } finally {
      stateTensor.dispose()
      goalTensor.dispose()
    }
  }
  for (const test of manifest.policy.parity) {
    const observed = await infer(Float32Array.from(test.state), Float32Array.from(test.goal))
    if (observed.some((x, i) => Math.abs(x - test.action[i]) > 2e-5 + 2e-4 * Math.abs(test.action[i]))) {
      throw new Error('Browser policy actions differ from the pinned PyTorch actor')
    }
  }

  let model: MjModel
  const vfs = new mujoco.MjVFS()
  // MjVFS.addBuffer converts the JS array element-by-element in this binding.
  // Emscripten's in-memory file accepts the typed array directly, which avoids
  // a very long main-thread-independent stall for the full mesh-bearing G1.
  const files = (mujoco as typeof mujoco & { FS: {
    writeFile(path: string, bytes: Uint8Array, options: { canOwn: boolean }): void
    unlink(path: string): void
  } }).FS
  const scenePath = '/trained-scene.mjb'
  try {
    files.writeFile(scenePath, mjb, { canOwn: true })
    try { model = mujoco.MjModel.from_binary_path(scenePath, vfs) }
    finally { files.unlink(scenePath) }
  } finally {
    vfs.delete()
  }
  const r = manifest.robot
  if (!model || model.nq !== 36 || model.nv !== 35 || model.nu !== ACTION_DIM || model.opt.timestep !== r.timestep) {
    throw new Error('Exported scene does not match the pinned G1 contract')
  }
  const id = (type: number, name: string) => {
    const found = mujoco.mj_name2id(model, type, name)
    if (found < 0) throw new Error('Exported scene is missing an element specified by the release manifest')
    return found
  }
  const object = mujoco.mjtObj
  const root = id(object.mjOBJ_BODY.value, r.root_body)
  const free = id(object.mjOBJ_JOINT.value, r.free_joint)
  const key = id(object.mjOBJ_KEY.value, r.keyframe)
  if (model.jnt_type[free] !== mujoco.mjtJoint.mjJNT_FREE.value || model.jnt_bodyid[free] !== root) {
    throw new Error('Exported scene is missing the expected G1 floating base')
  }
  const jointNames = r.joints.map(({ name }) => name)
  const rootQpos = model.jnt_qposadr[free]
  const rootDof = model.jnt_dofadr[free]
  const joints = r.joints.map(({ name, actuator }) => {
    const joint = id(object.mjOBJ_JOINT.value, name)
    const control = id(object.mjOBJ_ACTUATOR.value, actuator)
    if (model.jnt_type[joint] !== mujoco.mjtJoint.mjJNT_HINGE.value || model.actuator_trnid[2 * control] !== joint) {
      throw new Error('Exported scene actuator order does not match the release manifest')
    }
    return { qpos: model.jnt_qposadr[joint], dof: model.jnt_dofadr[joint], control }
  })
  const bodies = r.bodies.map(({ name }) => id(object.mjOBJ_BODY.value, name))
  if (bodies[0] !== root || bodies.some((b, i) => model.body_parentid[b] !== (i === 0 ? 0 : bodies[r.bodies[i].parent]))) {
    throw new Error('Exported scene body tree does not match the release manifest')
  }
  const meshes = renderMeshes(model, bodies, mujoco.mjtGeom.mjGEOM_MESH.value)
  const data: MjData = new mujoco.MjData(model)
  const dt = r.timestep * r.substeps
  let step = 0
  let due = performance.now()
  const publishPose = () => {
    const bodyPoses = new Float32Array(bodies.length * 7)
    const pos = data.xpos as Float64Array
    const quat = data.xquat as Float64Array
    for (let i = 0; i < bodies.length; i++) {
      bodyPoses.set(pos.subarray(3 * bodies[i], 3 * bodies[i] + 3), 7 * i)
      bodyPoses.set(quat.subarray(4 * bodies[i], 4 * bodies[i] + 4), 7 * i + 3)
    }
    if (!bodyPoses.every(Number.isFinite)) throw new Error('Non-finite G1 state in browser simulation')
    post({ type: 'frame', frame: { step, bodyPoses } }, [bodyPoses.buffer])
  }
  const reset = () => {
    step = 0
    resetRequested = false
    mujoco.mj_resetDataKeyframe(model, data, key)
    for (let i = 0; i < ACTION_DIM; i++) {
      if (Math.abs(data.qpos[joints[i].qpos] - r.action_offset[i]) > 1e-6) {
        throw new Error('Exported scene rest pose does not match the policy action offsets')
      }
      data.ctrl[joints[i].control] = r.action_offset[i]
    }
    mujoco.mj_forward(model, data)
  }
  reset()
  post({ type: 'ready', manifest, meshes }, meshes.flatMap((mesh) => [mesh.vertices.buffer, mesh.triangles.buffer]))
  publishPose()

  async function tick() {
    try {
      if (resetRequested) { reset(); publishPose() }
      if (playing) {
        const state = actorState(data.qpos, data.qvel, rootQpos, rootDof, joints)
        const goal = actorGoal(data.qpos, rootQpos, step, dt, jointNames, r.action_offset, motion)
        const actions = await infer(state, goal)
        if (playing && !resetRequested) {
          for (let j = 0; j < ACTION_DIM; j++) data.ctrl[joints[j].control] = actions[j] * r.action_scale[j] + r.action_offset[j]
          for (let i = 0; i < r.substeps; i++) mujoco.mj_step(model, data)
          mujoco.mj_forward(model, data)
          step += 1
          publishPose()
        }
      }
      // Limit catch-up for slow devices; simulation step size remains unchanged.
      due = Math.max(due + dt * 1000, performance.now())
      setTimeout(tick, Math.max(due - performance.now(), 1))
    } catch (error) { fail(error) }
  }
  void tick()
}

worker.onmessage = ({ data }: MessageEvent<WorkerRequest>) => {
  if (data.type === 'start' && !started) {
    started = true
    void start(data.manifestUrl).catch(fail)
  } else if (data.type === 'playing') {
    playing = data.playing
  } else if (data.type === 'reset') {
    resetRequested = true
  } else if (data.type === 'motion') {
    motion = data.motion
    resetRequested = true
  }
}
