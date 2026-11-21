export type MotionCategory = 'acrobatics' | 'contact rich' | 'manipulation' | 'locomotion' | 'dance' | 'unseen'

export type Motion = {
  label: string
  category: MotionCategory
  frames: number
}

export type SimulationSettings = {
  playing: boolean
  policy: boolean
  goal: boolean
  mujoco: boolean
  noise: number
  friction: number
}
