import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js'
import type { Motion } from '../types'

type SimulationStageProps = {
  playing: boolean
  goal: boolean
  mujoco: boolean
  noise: number
  selectedMotion?: Motion | null
  resetToken?: number
  onStep?: (step: number) => void
  onDragStart?: () => void
}

type RobotMeta = {
  bodies?: Record<string, Array<{ pos: number[]; quat: number[]; rgba: number[]; vert_offset: number; num_verts: number; face_offset: number; num_faces: number }>>
}

type Rig = THREE.Group & { userData: { bodyGroups?: Record<string, THREE.Group>; fallback?: boolean } }

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value))

function makeMaterial(color: THREE.ColorRepresentation, opacity = 1) {
  return new THREE.MeshStandardMaterial({ color, roughness: 0.48, metalness: 0.05, transparent: opacity < 1, opacity })
}

function makeFallbackRig(): Rig {
  const root = new THREE.Group() as Rig
  root.userData.fallback = true
  root.position.y = 0.05
  const white = makeMaterial('#d7d7d8')
  const dark = makeMaterial('#242529')
  const joint = makeMaterial('#8f9094')
  const add = (geometry: THREE.BufferGeometry, material: THREE.Material, position: [number, number, number], parent = root) => {
    const mesh = new THREE.Mesh(geometry, material)
    mesh.position.set(...position)
    parent.add(mesh)
    return mesh
  }
  add(new RoundedBoxGeometry(0.5, 0.56, 0.36, 6, 0.08), white, [0, 1.38, 0])
  add(new RoundedBoxGeometry(0.31, 0.22, 0.27, 6, 0.08), dark, [0, 1.77, 0])
  add(new RoundedBoxGeometry(0.24, 0.075, 0.025, 4, 0.02), makeMaterial('#111216'), [0, 1.79, 0.135])
  add(new RoundedBoxGeometry(0.34, 0.1, 0.25, 5, 0.04), dark, [0, 1.56, 0.02])
  add(new THREE.BoxGeometry(0.17, 0.2, 0.2), joint, [0, 1.08, 0])
  const limbs: Array<{ side: number; x: number; upperY: number; lowerY: number; arm: boolean }> = [
    { side: -1, x: -0.22, upperY: 1.33, lowerY: 1.08, arm: true },
    { side: 1, x: 0.22, upperY: 1.33, lowerY: 1.08, arm: true },
    { side: -1, x: -0.13, upperY: 0.94, lowerY: 0.56, arm: false },
    { side: 1, x: 0.13, upperY: 0.94, lowerY: 0.56, arm: false },
  ]
  for (const limb of limbs) {
    const group = new THREE.Group()
    group.position.set(limb.x, limb.upperY, 0)
    root.add(group)
    add(new THREE.CapsuleGeometry(limb.arm ? 0.06 : 0.075, limb.arm ? 0.27 : 0.34, 8, 16), white, [0, -0.15, 0], group)
    add(new THREE.SphereGeometry(limb.arm ? 0.07 : 0.08, 12, 8), joint, [0, -0.32, 0], group)
    add(new THREE.CapsuleGeometry(limb.arm ? 0.055 : 0.07, limb.arm ? 0.24 : 0.35, 8, 16), white, [0, -0.53, 0], group)
    if (!limb.arm) add(new THREE.BoxGeometry(0.2, 0.06, 0.32), dark, [0, -0.76, 0.05], group)
    else add(new THREE.SphereGeometry(0.09, 12, 8), dark, [0, -0.73, 0], group)
    group.userData.side = limb.side
    group.userData.arm = limb.arm
    group.userData.baseRotation = limb.arm ? limb.side * 0.16 : limb.side * 0.04
  }
  return root
}

function applyStyle(root: THREE.Object3D, color: THREE.ColorRepresentation, opacity: number) {
  root.traverse((object) => {
    if (!(object instanceof THREE.Mesh)) return
    const material = object.material.clone()
    material.color.set(color)
    material.transparent = opacity < 1
    material.opacity = opacity
    material.depthWrite = opacity >= 1
    object.material = material
  })
}

function buildOfficialRig(meta: RobotMeta, buffer: ArrayBuffer): Rig | null {
  if (!meta.bodies) return null
  const root = new THREE.Group() as Rig
  root.rotation.x = -Math.PI / 2
  const bodyGroups: Record<string, THREE.Group> = {}
  for (const [name, meshes] of Object.entries(meta.bodies)) {
    const body = new THREE.Group()
    bodyGroups[name] = body
    root.add(body)
    for (const part of meshes) {
      const position = new Float32Array(buffer, part.vert_offset, part.num_verts * 3)
      const index = new Uint32Array(buffer, part.face_offset, part.num_faces * 3)
      const geometry = new THREE.BufferGeometry()
      geometry.setAttribute('position', new THREE.BufferAttribute(position, 3))
      geometry.setIndex(new THREE.BufferAttribute(index, 1))
      geometry.computeVertexNormals()
      const rgba = part.rgba ?? [0.7, 0.7, 0.7, 1]
      const material = new THREE.MeshStandardMaterial({ color: new THREE.Color(rgba[0], rgba[1], rgba[2]), roughness: 0.5, metalness: 0.03, transparent: rgba[3] < 1, opacity: rgba[3] })
      const mesh = new THREE.Mesh(geometry, material)
      const pos = part.pos ?? [0, 0, 0]
      mesh.position.set(pos[0], pos[1], pos[2])
      const q = part.quat ?? [1, 0, 0, 0]
      mesh.quaternion.set(q[1], q[2], q[3], q[0])
      mesh.castShadow = true
      mesh.receiveShadow = true
      body.add(mesh)
    }
  }
  root.userData.bodyGroups = bodyGroups
  return root
}

function cloneRig(source: Rig, color: THREE.ColorRepresentation, opacity: number): Rig {
  const clone = source.clone(true) as Rig
  clone.userData = { ...source.userData }
  applyStyle(clone, color, opacity)
  return clone
}

export default function SimulationStage({ playing, goal, mujoco, noise, selectedMotion, resetToken, onStep, onDragStart }: SimulationStageProps) {
  const mountRef = useRef<HTMLDivElement>(null)
  const propsRef = useRef({ playing, goal, mujoco, noise, selectedMotion, resetToken })
  propsRef.current = { playing, goal, mujoco, noise, selectedMotion, resetToken }

  useEffect(() => {
    const mount = mountRef.current
    if (!mount) return
    let disposed = false
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(window.innerWidth, window.innerHeight)
    renderer.setClearColor(0x000000, 0)
    renderer.outputColorSpace = THREE.SRGBColorSpace
    renderer.shadowMap.enabled = true
    renderer.shadowMap.type = THREE.PCFShadowMap
    renderer.domElement.setAttribute('aria-label', 'Interactive SuperMimic robot simulation')
    mount.appendChild(renderer.domElement)

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 0.05, 100)
    camera.position.set(2.8, 1.5, 2.8)
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.enablePan = false
    controls.minDistance = 1.3
    controls.maxDistance = 6
    controls.target.set(0, 0.86, 0)

    scene.add(new THREE.HemisphereLight(0xffffff, 0xd4d4d7, 1.35))
    const key = new THREE.DirectionalLight(0xffffff, 2.1)
    key.position.set(2.5, 6, 3)
    key.castShadow = true
    key.shadow.mapSize.set(1024, 1024)
    key.shadow.camera.left = -3
    key.shadow.camera.right = 3
    key.shadow.camera.top = 3
    key.shadow.camera.bottom = -3
    scene.add(key)
    const fill = new THREE.DirectionalLight(0xffffff, 0.55)
    fill.position.set(-4, 2, 1.5)
    scene.add(fill)

    const floor = new THREE.Mesh(new THREE.PlaneGeometry(80, 80), new THREE.MeshStandardMaterial({ color: '#f5f5f7', roughness: 0.98, metalness: 0, transparent: true, opacity: 0.72 }))
    floor.rotation.x = -Math.PI / 2
    floor.receiveShadow = true
    scene.add(floor)
    const grid = new THREE.GridHelper(80, 80, '#d2d2d7', '#e6e6e9')
    ;(grid.material as THREE.Material).transparent = true
    ;(grid.material as THREE.Material).opacity = 0.58
    scene.add(grid)
    const shadowCanvas = document.createElement('canvas')
    shadowCanvas.width = shadowCanvas.height = 128
    const context = shadowCanvas.getContext('2d')!
    const gradient = context.createRadialGradient(64, 64, 0, 64, 64, 64)
    gradient.addColorStop(0, 'rgba(0,0,0,.3)')
    gradient.addColorStop(.45, 'rgba(0,0,0,.12)')
    gradient.addColorStop(1, 'rgba(0,0,0,0)')
    context.fillStyle = gradient
    context.fillRect(0, 0, 128, 128)
    const shadow = new THREE.Mesh(new THREE.PlaneGeometry(1.1, 1.1), new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(shadowCanvas), transparent: true, depthWrite: false }))
    shadow.rotation.x = -Math.PI / 2
    shadow.position.y = 0.006
    scene.add(shadow)

    const world = new THREE.Group()
    const goalRig = new THREE.Group()
    const simRig = new THREE.Group()
    scene.add(world, goalRig, simRig)
    let modelRoot: Rig = makeFallbackRig()
    let worldRig: Rig = modelRoot
    let goalReady = false
    let simReady = false
    world.add(worldRig)

    const setupModel = async () => {
      try {
        const [metaResponse, modelResponse] = await Promise.all([fetch('/model/robot.json'), fetch('/model/robot.bin')])
        if (metaResponse.ok && modelResponse.ok) {
          const official = buildOfficialRig(await metaResponse.json() as RobotMeta, await modelResponse.arrayBuffer())
          if (official && !disposed) {
            // The published mesh is hosted locally for reproducibility. The
            // local fallback rig remains the visible presentation because the
            // published pose stream is produced by its WASM worker.
            official.visible = false
            modelRoot = official
          }
        }
      } catch (error) {
        console.warn('Official robot mesh unavailable, using fallback rig.', error)
        // The fallback keeps the simulation useful when the optional mesh files are unavailable.
      }
      if (disposed) return
      const goalSource = cloneRig(worldRig, '#8e8e93', 0.22)
      const simSource = cloneRig(worldRig, '#616267', 0.5)
      goalRig.add(goalSource)
      simRig.add(simSource)
      goalReady = true
      simReady = true
    }
    void setupModel()

    const raycaster = new THREE.Raycaster()
    const pointer = new THREE.Vector2()
    const dragPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0)
    const dragPoint = new THREE.Vector3()
    const dragLine = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
      new THREE.LineBasicMaterial({ color: '#2f7fd0', transparent: true, opacity: .6, depthTest: false }),
    )
    dragLine.visible = false
    dragLine.renderOrder = 999
    scene.add(dragLine)
    let dragging = false
    const setPointer = (event: PointerEvent) => {
      const rect = renderer.domElement.getBoundingClientRect()
      pointer.set(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1)
    }
    const onPointerDown = (event: PointerEvent) => {
      if (event.button !== 0) return
      setPointer(event)
      const hits = raycaster.intersectObject(worldRig, true)
      if (!hits.length) return
      dragging = true
      controls.enabled = false
      renderer.domElement.setPointerCapture(event.pointerId)
      const positions = dragLine.geometry.getAttribute('position') as THREE.BufferAttribute
      positions.setXYZ(0, hits[0].point.x, hits[0].point.y, hits[0].point.z)
      positions.setXYZ(1, hits[0].point.x, hits[0].point.y, hits[0].point.z)
      positions.needsUpdate = true
      dragLine.visible = true
      onDragStart?.()
    }
    const onPointerMove = (event: PointerEvent) => {
      if (!dragging) return
      setPointer(event)
      raycaster.setFromCamera(pointer, camera)
      if (!raycaster.ray.intersectPlane(dragPlane, dragPoint)) return
      const positions = dragLine.geometry.getAttribute('position') as THREE.BufferAttribute
      positions.setXYZ(1, dragPoint.x, 0.01, dragPoint.z)
      positions.needsUpdate = true
    }
    const onPointerUp = (event: PointerEvent) => {
      dragging = false
      controls.enabled = true
      dragLine.visible = false
      if (renderer.domElement.hasPointerCapture(event.pointerId)) renderer.domElement.releasePointerCapture(event.pointerId)
    }
    renderer.domElement.addEventListener('pointerdown', onPointerDown)
    renderer.domElement.addEventListener('pointermove', onPointerMove)
    renderer.domElement.addEventListener('pointerup', onPointerUp)
    renderer.domElement.addEventListener('pointercancel', onPointerUp)

    let frame = 0
    let last = performance.now()
    let elapsed = 0
    let step = 0
    let lastResetToken = resetToken
    const animate = (now: number) => {
      if (disposed) return
      const dt = Math.min((now - last) / 1000, 0.08)
      last = now
      const current = propsRef.current
      if (current.resetToken !== lastResetToken) {
        lastResetToken = current.resetToken
        elapsed = 0
        step = 0
        onStep?.(0)
      }
      if (current.playing) {
        elapsed += dt
        step += dt * 50
        onStep?.(Math.floor(step))
      }
      const motionPhase = (current.selectedMotion?.label.length ?? 0) * 0.07
      const swing = Math.sin(elapsed * 5.2 + motionPhase) * (0.08 + current.noise * 0.25)
      worldRig.position.y = 0.03 + Math.max(0, Math.sin(elapsed * 2.6 + motionPhase)) * 0.015
      worldRig.rotation.z = swing * 0.08
      if (worldRig.userData.fallback) {
        const limbs = worldRig.children.filter((child) => child instanceof THREE.Group)
        limbs.forEach((limb, index) => {
          const side = Number(limb.userData.side ?? (index % 2 ? 1 : -1))
          const arm = Boolean(limb.userData.arm)
          const base = Number(limb.userData.baseRotation ?? 0)
          limb.rotation.z = base + swing * (arm ? 1 : 0.65) * side
          limb.rotation.x = Math.sin(elapsed * 4 + index) * 0.03
        })
      } else {
        const bodyGroups = worldRig.userData.bodyGroups ?? {}
        for (const [name, body] of Object.entries(bodyGroups)) {
          const side = name.includes('left') ? -1 : 1
          if (name.includes('shoulder') || name.includes('hip')) body.rotation.z = swing * side
          if (name.includes('elbow') || name.includes('knee')) body.rotation.x = Math.sin(elapsed * 4 + side) * 0.06
        }
      }
      if (goalReady) {
        goalRig.visible = current.goal
        goalRig.position.copy(worldRig.position)
        goalRig.position.y += 0.015
        goalRig.rotation.z = -swing * 0.3
      }
      if (simReady) {
        simRig.visible = current.mujoco
        simRig.position.copy(worldRig.position)
        simRig.position.x += current.mujoco ? 0.02 : 0
        simRig.rotation.z = swing * 0.06
      }
      shadow.position.x = worldRig.position.x
      shadow.position.z = worldRig.position.z
      shadow.scale.setScalar(0.9 + 0.6 * clamp(worldRig.position.y, 0, 1))
      controls.update()
      renderer.render(scene, camera)
      frame = requestAnimationFrame(animate)
    }
    frame = requestAnimationFrame(animate)

    const onResize = () => {
      const width = mount.clientWidth || window.innerWidth
      const height = mount.clientHeight || window.innerHeight
      camera.aspect = width / height
      camera.updateProjectionMatrix()
      renderer.setSize(width, height)
    }
    window.addEventListener('resize', onResize)
    onResize()

    return () => {
      disposed = true
      cancelAnimationFrame(frame)
      window.removeEventListener('resize', onResize)
      renderer.domElement.removeEventListener('pointerdown', onPointerDown)
      renderer.domElement.removeEventListener('pointermove', onPointerMove)
      renderer.domElement.removeEventListener('pointerup', onPointerUp)
      renderer.domElement.removeEventListener('pointercancel', onPointerUp)
      controls.dispose()
      scene.traverse((object) => {
        if (object instanceof THREE.Mesh) {
          object.geometry.dispose()
          const material = object.material
          if (Array.isArray(material)) material.forEach((item) => item.dispose())
          else material.dispose()
        }
      })
      renderer.dispose()
      mount.removeChild(renderer.domElement)
    }
  }, [onDragStart, onStep])

  return <div ref={mountRef} className="simulation-stage" aria-hidden="true" />
}
