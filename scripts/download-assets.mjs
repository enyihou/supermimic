import { mkdir, writeFile } from 'node:fs/promises'
import { basename, dirname, join } from 'node:path'

const base = 'https://supermimic.github.io/demo/'
const assets = [
  'constants.json',
  'robot.json',
  'robot.bin',
  'weights.bin',
  'engine.wasm',
  'engine.cfg',
  'scene.xml',
  'motions/00.bin',
  'motions/02.bin',
  'motions/06.bin',
  'motions/11.bin',
  'motions/12.bin',
  'motions/29.bin',
]

const root = join(process.cwd(), 'public', 'model')
await mkdir(root, { recursive: true })

async function download(path) {
  const response = await fetch(base + path)
  if (!response.ok) throw new Error(`${path}: ${response.status}`)
  const output = join(root, path)
  await mkdir(dirname(output), { recursive: true })
  await writeFile(output, Buffer.from(await response.arrayBuffer()))
  console.log(`${path} ${response.headers.get('content-length') ?? 'unknown'} bytes`)
}

const concurrency = 4
for (let index = 0; index < assets.length; index += concurrency) {
  await Promise.all(assets.slice(index, index + concurrency).map(download))
}

console.log(`Downloaded ${assets.length} published model assets to ${root}`)
