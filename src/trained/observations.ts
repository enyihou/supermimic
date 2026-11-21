import { ACTION_DIM, GOAL_DIM, GOAL_FRAMES, GOAL_STRIDE, STATE_DIM, type TrainedMotion } from './contract'

/** Rotate a world vector into the body frame (world-from-body quaternion, wxyz). */
export function rotateInverse(w: number, x: number, y: number, z: number, a: number, b: number, c: number) {
  const tx = 2 * (z * b - y * c)
  const ty = 2 * (x * c - z * a)
  const tz = 2 * (y * a - x * b)
  return [a + w * tx + z * ty - y * tz, b + w * ty + x * tz - z * tx, c + w * tz + y * tx - x * ty]
}

/** First two rotation matrix columns, flattened row-major as in the pinned helper. */
export function inverseRootRot6d(w: number, x: number, y: number, z: number) {
  // All procedural references target an upright root with identity heading.
  return [1 - 2 * (y * y + z * z), 2 * (x * y + w * z),
    2 * (x * y - w * z), 1 - 2 * (x * x + z * z),
    2 * (x * z + w * y), 2 * (y * z - w * x)]
}

/** GenCoord state from MuJoCo's free joint and ordered hinge addresses. */
export function actorState(
  qpos: ArrayLike<number>, qvel: ArrayLike<number>, rootQpos: number, rootDof: number,
  joints: readonly { qpos: number, dof: number }[],
) {
  const [w, x, y, z] = [qpos[rootQpos + 3], qpos[rootQpos + 4], qpos[rootQpos + 5], qpos[rootQpos + 6]]
  const out = new Float32Array(STATE_DIM)
  out[0] = qpos[rootQpos + 2]
  out.set(rotateInverse(w, x, y, z, 0, 0, -1), 1)
  // MuJoCo free joints store world linear velocity but body angular velocity.
  out.set(rotateInverse(w, x, y, z, qvel[rootDof], qvel[rootDof + 1], qvel[rootDof + 2]), 4)
  for (let i = 0; i < 3; i++) out[7 + i] = qvel[rootDof + 3 + i]
  for (let j = 0; j < ACTION_DIM; j++) {
    out[10 + j] = qpos[joints[j].qpos]
    out[10 + ACTION_DIM + j] = qvel[joints[j].dof]
  }
  return out
}

function amplitude(name: string, motion: TrainedMotion) {
  if (motion === 'shift') {
    if (name.endsWith('hip_roll_joint')) return 0.065
    if (name.endsWith('ankle_roll_joint')) return -0.055
    if (name.endsWith('waist_roll_joint')) return -0.025
  } else if (motion === 'reach') {
    if (name.endsWith('left_shoulder_pitch_joint')) return -0.32
    if (name.endsWith('left_elbow_joint')) return -0.24
    if (name.endsWith('left_shoulder_roll_joint')) return 0.12
  }
  return 0
}

export function actorGoal(
  qpos: ArrayLike<number>, rootQpos: number, step: number, dt: number,
  jointNames: readonly string[], nominal: readonly number[], motion: TrainedMotion,
) {
  const [w, x, y, z] = [qpos[rootQpos + 3], qpos[rootQpos + 4], qpos[rootQpos + 5], qpos[rootQpos + 6]]
  const rotation = inverseRootRot6d(w, x, y, z)
  const out = new Float32Array(GOAL_DIM)
  const amplitudes = jointNames.map((name) => amplitude(name, motion))
  const omega = 2 * Math.PI / 3
  for (let frame = 0; frame < GOAL_FRAMES; frame++) {
    const theta = (step + frame * GOAL_STRIDE) * dt * omega
    const pos = (1 - Math.cos(theta)) / 2 // Start in the keyframe with zero goal velocity.
    const vel = Math.sin(theta) * omega / 2
    const base = frame * 2 * ACTION_DIM
    for (let j = 0; j < ACTION_DIM; j++) {
      out[base + j] = nominal[j] + amplitudes[j] * pos
      out[base + ACTION_DIM + j] = amplitudes[j] * vel
    }
    out.set(rotation, 2 * ACTION_DIM * GOAL_FRAMES + frame * 6)
  }
  return out
}
