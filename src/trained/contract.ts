export const TRAINED_FORMAT = 'supermimic-gencoord-browser-v1'
export const SOURCE_PINS = {
  supermimic: '5fb92297ab4fe3d86629d9f0c635334791bb0d5f',
  mjlab_research: '379c55e88b1dec73f93943e524a879b2ebbf45c5',
} as const
export const STATE_DIM = 68
export const GOAL_DIM = 640
export const ACTION_DIM = 29
export const GOAL_FRAMES = 10
export const GOAL_STRIDE = 5
export const ENGINE_VERSION = '4.0.1'

export type TrainedMotion = 'stand' | 'shift' | 'reach'
export const TRAINED_MOTIONS: ReadonlyArray<{ id: TrainedMotion, label: string }> = [
  { id: 'stand', label: 'Stand' },
  { id: 'shift', label: 'Weight shift' },
  { id: 'reach', label: 'Arm reach' },
]

interface Artifact { sha256: string, bytes: number }
interface ParityCase { state: number[], goal: number[], action: number[] }

export interface TrainedManifest {
  format: typeof TRAINED_FORMAT
  model_id: string
  task: 'SuperMimic-GenCoord-G1'
  dataset_sha256: string
  checkpoint_step: number
  checkpoint_actor_sha256: string
  checkpoint_world_model_sha256: string
  source_commits: typeof SOURCE_PINS & { reproduction: string }
  metrics: Record<string, number>
  artifacts: Record<string, Artifact>
  policy: {
    state_dim: typeof STATE_DIM
    goal_dim: typeof GOAL_DIM
    action_dim: typeof ACTION_DIM
    goal_frames: typeof GOAL_FRAMES
    goal_stride: typeof GOAL_STRIDE
    parity: ParityCase[]
    export_max_abs_error: number
  }
  robot: {
    engine_version: typeof ENGINE_VERSION
    scene_uncompressed_bytes: number
    timestep: number
    substeps: number
    keyframe: string
    root_body: string
    free_joint: string
    joints: Array<{ name: string, actuator: string }>
    bodies: Array<{ name: string, parent: number }>
    action_scale: number[]
    action_offset: number[]
  }
}

export interface TrainedFrame {
  step: number
  /** Interleaved xyz,wxyz body transforms in the manifest's z-up frame. */
  bodyPoses: Float32Array
}

export interface TrainedMesh {
  name: string
  /** Index into the manifest body list and the live body pose buffer. */
  bodyIndex: number
  vertices: Float32Array
  triangles: Uint32Array
  /** Mesh offsets relative to the body, in the model's z-up coordinates. */
  position: [number, number, number]
  quaternion: [number, number, number, number] // wxyz
  rgba: [number, number, number, number]
}

export type WorkerRequest =
  | { type: 'start', manifestUrl: string }
  | { type: 'playing', playing: boolean }
  | { type: 'reset' }
  | { type: 'motion', motion: TrainedMotion }
export type WorkerResponse =
  | { type: 'ready', manifest: TrainedManifest, meshes: TrainedMesh[] }
  | { type: 'frame', frame: TrainedFrame }
  | { type: 'error', message: string }

const positiveInteger = (x: unknown): x is number => Number.isSafeInteger(x) && (x as number) > 0
const finite = (x: unknown): x is number => typeof x === 'number' && Number.isFinite(x)
const text = (x: unknown): x is string => typeof x === 'string' && x.length > 0 && x.length < 130
const hex = (x: unknown, length: number) => typeof x === 'string' && x.length === length && /^[a-f0-9]+$/.test(x)
const numbers = (x: unknown, length: number): x is number[] => Array.isArray(x) && x.length === length && x.every(finite)

export function parseTrainedManifest(value: unknown): TrainedManifest {
  const m = value as Partial<TrainedManifest> | null
  if (!m || m.format !== TRAINED_FORMAT || m.task !== 'SuperMimic-GenCoord-G1' ||
      !text(m.model_id) || !/^[a-z0-9][a-z0-9-]{0,63}$/.test(m.model_id) ||
      !positiveInteger(m.checkpoint_step) || !hex(m.dataset_sha256, 64) || !hex(m.checkpoint_actor_sha256, 64) ||
      !hex(m.checkpoint_world_model_sha256, 64) ||
      m.source_commits?.supermimic !== SOURCE_PINS.supermimic ||
      m.source_commits.mjlab_research !== SOURCE_PINS.mjlab_research || !hex(m.source_commits.reproduction, 40) ||
      !m.metrics || Array.isArray(m.metrics) || Object.keys(m.metrics).length === 0 ||
      Object.entries(m.metrics).some(([key, number]) => !/^[a-z][a-z0-9_]{0,63}$/.test(key) || !finite(number))) {
    throw new Error('Unsupported trained-model release manifest')
  }
  for (const name of ['actor.onnx', 'scene.mjbpack']) {
    const artifact = m.artifacts?.[name]
    if (!artifact || !hex(artifact.sha256, 64) || !positiveInteger(artifact.bytes) || artifact.bytes >= 100 * 1024 * 1024) {
      throw new Error(`Missing or invalid artifact metadata for ${name}`)
    }
  }
  const policy = m.policy
  if (!policy || policy.state_dim !== STATE_DIM || policy.goal_dim !== GOAL_DIM || policy.action_dim !== ACTION_DIM ||
      policy.goal_frames !== GOAL_FRAMES || policy.goal_stride !== GOAL_STRIDE ||
      !finite(policy.export_max_abs_error) || policy.export_max_abs_error < 0 ||
      !Array.isArray(policy.parity) || policy.parity.length !== 2 || policy.parity.some((sample) =>
        !sample || !numbers(sample.state, STATE_DIM) || !numbers(sample.goal, GOAL_DIM) || !numbers(sample.action, ACTION_DIM))) {
    throw new Error('Unsupported trained policy input/output contract')
  }
  const r = m.robot
  if (!r || r.engine_version !== ENGINE_VERSION || r.timestep !== 0.005 || r.substeps !== 4 ||
      !positiveInteger(r.scene_uncompressed_bytes) || r.scene_uncompressed_bytes > 128 * 1024 * 1024 ||
      !text(r.keyframe) || !text(r.root_body) || !text(r.free_joint) ||
      !numbers(r.action_scale, ACTION_DIM) || r.action_scale.some((n) => n <= 0) || !numbers(r.action_offset, ACTION_DIM) ||
      !Array.isArray(r.joints) || r.joints.length !== ACTION_DIM ||
      r.joints.some((j) => !j || !text(j.name) || !text(j.actuator)) ||
      new Set(r.joints.map((j) => j.name)).size !== ACTION_DIM || new Set(r.joints.map((j) => j.actuator)).size !== ACTION_DIM ||
      !Array.isArray(r.bodies) || r.bodies.length < 2 || r.bodies.length > 100 ||
      r.bodies.some((b, index) => !b || !text(b.name) || !Number.isInteger(b.parent) || b.parent < -1 || b.parent >= index) ||
      new Set(r.bodies.map((b) => b.name)).size !== r.bodies.length || r.bodies[0].name !== r.root_body ||
      r.bodies[0].parent !== -1 || r.bodies.slice(1).some((b) => b.parent < 0)) {
    throw new Error('Unsupported trained policy robot contract')
  }
  return m as TrainedManifest
}
