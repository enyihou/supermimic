// Browser integration test for the trained-policy worker. All model data below
// is synthetic QA material. There are no files written to the deployed site.
import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { mkdir, readFile, realpath, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, sep } from 'node:path'
import { gzipSync } from 'node:zlib'
import loadMujoco from '@mujoco/mujoco'
import { chromium } from 'playwright'

const [origin, fixtureDir] = process.argv.slice(2)
assert(origin && fixtureDir, 'usage: node scripts/test-browser-runtime.mjs VITE_DEV_ORIGIN SYNTHETIC_FIXTURE_DIR')
const { parity, export_max_abs_error } = JSON.parse(await readFile(join(fixtureDir, 'parity.json'), 'utf8'))
const actor = await readFile(join(fixtureDir, 'actor.onnx'))
const count = 29
const joints = Array.from({ length: count }, (_, i) => ({ name: `j${i}`, actuator: `a${i}` }))
const bodies = [{ name: 'robot', parent: -1 }, ...joints.map((_, i) => ({ name: `b${i}`, parent: 0 }))]
const goal = await loadMujoco()
const qpos = `0 0 1 1 0 0 0 ${Array(count).fill(0).join(' ')}`
const xml = `<mujoco><option timestep="0.005"/><asset>
  <mesh name="tetra" vertex="0 0 0  .08 0 0  0 .08 0  0 0 .08" face="0 2 1  0 1 3  0 3 2  1 2 3"/>
  </asset><worldbody>
  <geom type="plane" size="2 2 .1"/><body name="robot" pos="0 0 1"><freejoint name="root"/>
  <geom type="box" size=".03 .05 .07" mass="1"/>
  <geom type="mesh" mesh="tetra" group="2" density="0" contype="0" conaffinity="0"/>
  ${joints.map(({ name }, i) => `<body name="b${i}" pos="${0.08 + 0.003 * i} 0 ${0.04 * i}">
  <joint name="${name}" type="hinge"/><geom size=".005" contype="0" conaffinity="0" mass=".01"/></body>`).join('')}
  </body></worldbody><actuator>${joints.map(({name, actuator}) => `<position joint="${name}" name="${actuator}" kp="1"/>`).join('')}</actuator>
  <keyframe><key name="init_state" qpos="${qpos}"/></keyframe></mujoco>`
const model = goal.MjModel.from_xml_string(xml)
goal.mj_saveModel(model, 'fixture.mjb', null)
const scene = Buffer.from(goal.FS.readFile('fixture.mjb'))
const compressed = gzipSync(scene)
model.delete()

const metadata = (buffer) => ({ sha256: createHash('sha256').update(buffer).digest('hex'), bytes: buffer.length })
const manifest = {
  format: 'supermimic-gencoord-browser-v1', model_id: 'synthetic-not-trained', task: 'SuperMimic-GenCoord-G1',
  dataset_sha256: '0'.repeat(64), checkpoint_step: 1, checkpoint_actor_sha256: '0'.repeat(64), checkpoint_world_model_sha256: '0'.repeat(64),
  source_commits: { supermimic: '5fb92297ab4fe3d86629d9f0c635334791bb0d5f', mjlab_research: '379c55e88b1dec73f93943e524a879b2ebbf45c5', reproduction: '0'.repeat(40) },
  metrics: { synthetic_fixture_only: 1 }, artifacts: { 'actor.onnx': metadata(actor), 'scene.mjbpack': metadata(compressed) },
  policy: { state_dim: 68, goal_dim: 640, action_dim: 29, goal_frames: 10, goal_stride: 5, parity, export_max_abs_error },
  robot: { engine_version: '4.0.1', scene_uncompressed_bytes: scene.length, timestep: 0.005, substeps: 4, keyframe: 'init_state', root_body: 'robot', free_joint: 'root',
    joints, bodies, action_scale: Array(count).fill(0.1), action_offset: Array(count).fill(0) },
}
const g1 = process.env.G1_SCENE === '1'
const sceneBytes = g1 ? await readFile(join(fixtureDir, 'scene.mjbpack')) : compressed
if (g1) {
  manifest.robot = JSON.parse(await readFile(join(fixtureDir, 'robot.json'), 'utf8'))
  manifest.artifacts['scene.mjbpack'] = metadata(sceneBytes)
}
const registryData = JSON.stringify({ schema_version: 1, active: manifest.model_id })
async function stageTemporaryFixture() {
  if (!process.env.STATIC_PREVIEW_DIR) return null
  const preview = await realpath(process.env.STATIC_PREVIEW_DIR)
  const temporary = await realpath(tmpdir())
  assert(preview.startsWith(temporary + sep), 'Fixture may only be staged in a temporary preview directory')
  const registry = join(preview, 'models', 'releases.json')
  const previous = await readFile(registry)
  assert.equal(JSON.parse(previous).active, null, 'Test must not overwrite an active release registry')
  const modelPath = join(preview, 'models', manifest.model_id)
  await mkdir(modelPath, { mode: 0o700 })
  const cleanup = async () => {
    await writeFile(registry, previous)
    await rm(modelPath, { recursive: true })
  }
  try {
    await Promise.all([
      writeFile(join(modelPath, 'manifest.json'), JSON.stringify(manifest)),
      writeFile(join(modelPath, 'actor.onnx'), actor),
      writeFile(join(modelPath, 'scene.mjbpack'), sceneBytes),
    ])
    await writeFile(registry, registryData)
    return cleanup
  } catch (error) { await cleanup(); throw error }
}
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH || undefined,
  args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
})
let cleanupFixture
try {
  const mobile = process.env.MOBILE === '1'
  const context = await browser.newContext({
    viewport: mobile ? { width: 390, height: 844 } : { width: 1440, height: 900 },
    deviceScaleFactor: mobile ? 2 : 1, isMobile: mobile, hasTouch: mobile,
  })
  const page = await context.newPage()
  const dev = new URL(origin)
  const errors = []
  const requests = []
  page.on('pageerror', (error) => errors.push(error.message))
  page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()) })
  context.on('request', (request) => requests.push(request))
  await context.route('**/__trained-runtime-test', (route) => route.fulfill({
    contentType: 'text/html', body: '<!doctype html><title>Browser policy runtime test</title><main id="status">Starting</main>',
  }))
  const supplyFixture = async (route) => {
    const path = new URL(route.request().url()).pathname
    if (path.endsWith('/manifest.json')) {
      const data = structuredClone(manifest)
      if (path.includes('/bad/')) data.artifacts['actor.onnx'].sha256 = '0'.repeat(64)
      return route.fulfill({ json: data })
    }
    if (path.endsWith('/actor.onnx')) return route.fulfill({ body: actor, contentType: 'application/octet-stream' })
    if (path.endsWith('/scene.mjbpack')) return route.fulfill({ body: sceneBytes, contentType: 'application/octet-stream' })
    return route.abort()
  }
  await context.route('**/__synthetic/**', supplyFixture)
  if (process.env.APP_VIEW === '1') {
    cleanupFixture = await stageTemporaryFixture()
    if (!cleanupFixture) {
      await context.route('**/models/releases.json', (route) => route.fulfill({ contentType: 'application/json', body: registryData }))
      await context.route(`**/models/${manifest.model_id}/**`, supplyFixture)
    }
    await page.goto(new URL('?view=trained', dev).href)
    await page.getByRole('heading', { name: 'Trained model' }).waitFor()
    const ready = page.getByRole('link', { name: 'Release manifest' })
    const unavailable = page.getByRole('alert')
    try { await ready.or(unavailable).waitFor({ timeout: 45_000 }) }
    catch (error) {
      console.error('Rendered content:', await page.locator('main').innerText())
      console.error('Requests:', requests.map((request) => new URL(request.url()).pathname))
      console.error('Browser errors:', errors)
      throw error
    }
    if (await unavailable.isVisible()) throw new Error(await unavailable.textContent())
    assert.equal(await page.locator('canvas').count(), 1)
    await page.waitForFunction(() => +document.querySelector('.trained-controls output').value >= 8)
    await page.getByRole('button', { name: 'Pause' }).click()
    await page.getByRole('button', { name: 'Reset' }).click()
    await page.waitForFunction(() => +document.querySelector('.trained-controls output').value === 0)
    await page.getByLabel('Reference').selectOption('reach')
    await page.getByRole('button', { name: 'Play', exact: true }).click()
    await page.waitForFunction(() => +document.querySelector('.trained-controls output').value >= 8)
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), 'No horizontal viewport overflow')
    assert.deepEqual(errors, [], 'No unexplained app console errors')
    assert(requests.every((request) => new URL(request.url()).origin === dev.origin && request.method() === 'GET'))
    if (process.env.QA_SCREENSHOT) await page.screenshot({ path: process.env.QA_SCREENSHOT })
    console.log(`Rendered trained route, mesh canvas, controls, and viewport fit passed (${mobile ? 'mobile' : 'desktop'}, synthetic actor${g1 ? ', pinned G1 scene' : ''}).`)
    process.exitCode = 0
  } else {
  await page.goto(new URL('__trained-runtime-test', dev).href)
  assert.equal(await page.title(), 'Browser policy runtime test')
  // Dynamic import intentionally tests this module through Vite's worker URL
  // transform. Using an eval with no import statement bypasses the app shell.
  const moduleUrl = new URL('src/trained/simulation.ts', dev).href
  const outcome = await page.evaluate(async ({ moduleUrl, g1 }) => {
    const { startTrainedSimulation } = await import(moduleUrl)
    const status = document.getElementById('status')
    return new Promise((resolve, reject) => {
      const timeout = window.setTimeout(() => { controller.dispose(); reject(new Error(`Timed out: ${status.textContent}`)) }, 45_000)
      let initialZ, beforeReset
      const controller = startTrainedSimulation('/__synthetic/good/manifest.json', {
        onReady(release, meshes) {
          if (g1 ? meshes.length < 25 : meshes.length !== 1 || meshes[0].vertices.length !== 12 || meshes[0].triangles.length !== 12) {
            controller.dispose()
            window.clearTimeout(timeout)
            reject(new Error('Visual mesh data not delivered'))
            return
          }
          status.textContent = `Loaded ${release.model_id}`
        },
        onFrame(frame) {
          if (initialZ === undefined) initialZ = frame.bodyPoses[2]
          if (!beforeReset && frame.step >= 8) {
            beforeReset = frame.bodyPoses[2]
            controller.setPlaying(false)
            controller.setMotion('reach')
          } else if (beforeReset && frame.step === 0) {
            status.textContent = 'Policy simulated, paused and reset'
            controller.dispose()
            window.clearTimeout(timeout)
            resolve({ changed: Math.abs(beforeReset - initialZ) > 1e-4, reset: frame.bodyPoses[2] === initialZ })
          }
        },
        onError(error) { window.clearTimeout(timeout); reject(error) },
      })
    })
  }, { moduleUrl, g1 })
  assert(outcome.changed && outcome.reset, 'Pose must change under physics and return on reset')
  assert.equal(await page.locator('#status').textContent(), 'Policy simulated, paused and reset')

  const refused = await page.evaluate(async (moduleUrl) => {
    const { startTrainedSimulation } = await import(moduleUrl)
    return new Promise((resolve, reject) => {
      const timeout = window.setTimeout(() => { controller.dispose(); reject(new Error('Corrupted artifact was not reported')) }, 10_000)
      const controller = startTrainedSimulation('/__synthetic/bad/manifest.json', {
        onReady() { controller.dispose(); window.clearTimeout(timeout); reject(new Error('Corrupted model accepted')) },
        onFrame() {}, onError(error) { window.clearTimeout(timeout); resolve(error.message) },
      })
    })
  }, moduleUrl)
  assert.match(refused, /actor.onnx hash does not match/)
  assert.deepEqual(errors, [], 'No unexplained browser console or runtime errors')
  assert(requests.every((request) => new URL(request.url()).origin === dev.origin && request.method() === 'GET'), 'Only local static GETs are allowed')
  console.log(`Browser WASM actor parity, mesh transfer, closed-loop stepping, reset and hash rejection passed (${mobile ? 'mobile' : 'desktop'}, synthetic actor${g1 ? ', pinned G1 scene' : ''}).`)
  }
} finally {
  await browser.close()
  await cleanupFixture?.()
}
