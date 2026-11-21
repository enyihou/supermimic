import type { Motion } from './types'

const motionRows: Array<[string, Motion['category'], number]> = [
  ['Limb swing illustration', 'locomotion', 200],
  ['Arm shift illustration', 'manipulation', 200],
]

export const motions: Motion[] = motionRows.map(([label, category, frames]) => ({ label, category, frames }))
