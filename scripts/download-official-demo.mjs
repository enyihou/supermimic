import { mkdir, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'

const base = 'https://supermimic.github.io/demo/'
const root = join(process.cwd(), 'public', 'official-demo')
const files = [
  'index.html', 'main.js', 'worker.mjs', 'robot.json', 'robot.bin', 'weights.bin', 'engine.wasm', 'engine.cfg', 'scene.xml', 'constants.json',
  'vendor/three.module.js', 'vendor/three.core.js', 'vendor/OrbitControls.js', 'vendor/RoomEnvironment.js', 'vendor/mujoco/mujoco.js', 'vendor/mujoco/mujoco.wasm',
]
const constants = JSON.parse(await (await fetch(base + 'constants.json')).text())
files.push(...constants.motions.map(motion => motion.file))

await mkdir(root, { recursive: true })
async function download(path) {
  const response = await fetch(base + path)
  if (!response.ok) throw new Error(`${path}: ${response.status}`)
  let body = Buffer.from(await response.arrayBuffer())
  if (path === 'index.html') {
    body = Buffer.from(body.toString().replace('if (window.top !== window.self)', "if (window.top !== window.self && !location.search.includes('standalone'))"))
  }
  const output = join(root, path)
  await mkdir(dirname(output), { recursive: true })
  await writeFile(output, body)
  console.log(path, body.byteLength)
}

for (let index = 0; index < files.length; index += 4) {
  await Promise.all(files.slice(index, index + 4).map(download))
}
console.log(`Downloaded ${files.length} official demo files to ${root}`)
