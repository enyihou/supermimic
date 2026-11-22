import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { startTrainedSimulation, TRAINED_MOTIONS } from '../trained/simulation'
import type { TrainedFrame, TrainedManifest, TrainedMesh, TrainedMotion } from '../trained/simulation'

const base = import.meta.env.BASE_URL
const validModelId = /^[a-z0-9][a-z0-9-]{0,63}$/

type Release = { state: 'loading' | 'empty' | 'error' } | { state: 'available', id: string }

export function TrainedStage({ modelId }: { modelId: string }) {
  const mount = useRef<HTMLDivElement>(null)
  const output = useRef<HTMLOutputElement>(null)
  const simulation = useRef<ReturnType<typeof startTrainedSimulation> | null>(null)
  const [manifest, setManifest] = useState<TrainedManifest>()
  const [error, setError] = useState('')
  const [playing, setPlaying] = useState(true)
  const [motion, setMotion] = useState<TrainedMotion>('stand')

  useEffect(() => {
    const element = mount.current
    if (!element) return
    let renderer: THREE.WebGLRenderer
    try { renderer = new THREE.WebGLRenderer({ antialias: true }) }
    catch { setError('WebGL is unavailable on this device.'); return }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setClearColor('#f5f5f7')
    element.appendChild(renderer.domElement)
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(40, 1, 0.02, 50)
    camera.position.set(2.35, 1.3, 2.25)
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.target.set(0, 0.75, 0)
    controls.enableDamping = true
    controls.minDistance = 1.4
    controls.maxDistance = 5
    scene.add(new THREE.HemisphereLight(0xffffff, 0xaaaab4, 2.1), new THREE.GridHelper(8, 16, '#aab0bb', '#d6d8de'))
    const key = new THREE.DirectionalLight(0xffffff, 2.1)
    key.position.set(3, 5, 2)
    scene.add(key)
    const robot = new THREE.Group()
    robot.rotation.x = -Math.PI / 2
    scene.add(robot)
    let bodies: THREE.Group[] = []
    let latest: TrainedFrame | null = null
    let drawn: TrainedFrame | null = null
    let raf = 0

    const resize = new ResizeObserver(() => {
      const width = element.clientWidth
      const height = element.clientHeight
      if (!width || !height) return
      camera.aspect = width / height
      camera.updateProjectionMatrix()
      renderer.setSize(width, height, false)
    })
    resize.observe(element)

    const addMesh = (part: TrainedMesh) => {
      const geometry = new THREE.BufferGeometry()
      geometry.setAttribute('position', new THREE.BufferAttribute(part.vertices, 3))
      geometry.setIndex(new THREE.BufferAttribute(part.triangles, 1))
      geometry.computeVertexNormals()
      const [r, g, b, a] = part.rgba
      const material = new THREE.MeshStandardMaterial({
        color: new THREE.Color(r, g, b), roughness: .48, metalness: .07,
        opacity: a, transparent: a < 1,
      })
      const mesh = new THREE.Mesh(geometry, material)
      mesh.position.set(...part.position)
      const [w, x, y, z] = part.quaternion
      mesh.quaternion.set(x, y, z, w)
      bodies[part.bodyIndex].add(mesh)
    }

    const render = () => {
      if (latest && latest !== drawn) {
        drawn = latest
        const pose = latest.bodyPoses
        for (let i = 0; i < bodies.length; i++) {
          const o = i * 7
          bodies[i].position.set(pose[o], pose[o + 1], pose[o + 2])
          bodies[i].quaternion.set(pose[o + 4], pose[o + 5], pose[o + 6], pose[o + 3])
        }
        if (output.current) output.current.value = String(latest.step)
      }
      controls.update()
      renderer.render(scene, camera)
      raf = requestAnimationFrame(render)
    }

    try {
      simulation.current = startTrainedSimulation(`${base}models/${modelId}/manifest.json`, {
        onReady: (release, meshes) => {
          if (release.model_id !== modelId) {
            simulation.current?.dispose()
            setError('Model registry and manifest disagree.')
            return
          }
          bodies = release.robot.bodies.map(() => new THREE.Group())
          robot.add(...bodies)
          for (const part of meshes) addMesh(part)
          setManifest(release)
          render()
        },
        onFrame: (frame) => { latest = frame },
        onError: (failure) => { cancelAnimationFrame(raf); setError(failure.message) },
      })
    } catch {
      setError('Unable to start the on-device model worker.')
    }
    return () => {
      simulation.current?.dispose()
      simulation.current = null
      cancelAnimationFrame(raf)
      resize.disconnect()
      controls.dispose()
      robot.traverse((node) => {
        if (node instanceof THREE.Mesh) {
          node.geometry.dispose()
          ;(node.material as THREE.Material).dispose()
        }
      })
      scene.traverse((node) => {
        if (node instanceof THREE.GridHelper) node.dispose()
      })
      renderer.dispose()
      renderer.domElement.remove()
    }
  }, [modelId])

  const ready = Boolean(manifest) && !error
  return <>
    <div className="trained-stage-shell">
      <div className="trained-viewport" ref={mount} role="img" aria-label="Browser-simulated Unitree G1 robot" />
      {!ready ? <p className="trained-stage-message" role={error ? 'alert' : 'status'}>
        {error ? `Simulation unavailable: ${error}` : 'Verifying the model package and starting browser physics…'}
      </p> : null}
    </div>
    <div className="trained-controls">
      <button disabled={!ready} onClick={() => {
        simulation.current?.setPlaying(!playing)
        setPlaying(!playing)
      }}>{playing ? 'Pause' : 'Play'}</button>
      <button disabled={!ready} onClick={() => simulation.current?.reset()}>Reset</button>
      <label>Reference
        <select disabled={!ready} value={motion} onChange={(event) => {
          const next = event.target.value as TrainedMotion
          simulation.current?.setMotion(next)
          setMotion(next)
        }}>
          {TRAINED_MOTIONS.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}
        </select>
      </label>
      <label>Step <output ref={output} aria-live="off">0</output></label>
    </div>
    {manifest ? <p className="trained-version">
      Model <code>{manifest.model_id}</code> · checkpoint {manifest.checkpoint_step.toLocaleString()} ·{' '}
      <a href={`${base}models/${modelId}/manifest.json`}>Release manifest</a>
    </p> : null}
    {manifest ? <p className="trained-version trained-provenance">
      Actor <code>{manifest.checkpoint_actor_sha256.slice(0, 12)}…</code> · world model <code>{manifest.checkpoint_world_model_sha256.slice(0, 12)}…</code>
    </p> : null}
  </>
}

export default function TrainedView() {
  const [release, setRelease] = useState<Release>({ state: 'loading' })
  useEffect(() => {
    const request = new AbortController()
    void (async () => {
      try {
        const response = await fetch(`${base}models/releases.json`, { signal: request.signal })
        if (!response.ok) throw new Error('Model registry unavailable')
        const registry: unknown = await response.json()
        if (typeof registry !== 'object' || registry === null ||
            !('schema_version' in registry) || registry.schema_version !== 1 ||
            !('active' in registry)) throw new Error('Invalid model registry')
        const id = registry.active
        if (id === null) setRelease({ state: 'empty' })
        else if (typeof id === 'string' && validModelId.test(id)) setRelease({ state: 'available', id })
        else throw new Error('Invalid model ID')
      } catch {
        if (!request.signal.aborted) setRelease({ state: 'error' })
      }
    })()
    return () => request.abort()
  }, [])

  return <section className="simulation-card" aria-labelledby="trained-title">
    <p className="simulation-kicker">Reproduced policy</p>
    <h1 id="trained-title">Trained model</h1>
    {release.state === 'available' ?
      <><p>The trained actor and MuJoCo physics both run on this device. Choose a procedural reference to try the policy.</p>
        <TrainedStage key={release.id} modelId={release.id} /></> :
      <p className="trained-empty" role="status">{release.state === 'empty'
        ? 'No trained model has been released. This view will load the browser export after full training and held-out evaluation are verified.'
        : release.state === 'error' ? 'The model registry could not be loaded. Try refreshing this page.'
          : 'Checking for a verified release…'}</p>}
  </section>
}
