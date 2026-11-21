import type { TrainedFrame, TrainedManifest, TrainedMesh, TrainedMotion, WorkerRequest, WorkerResponse } from './contract'

export type { TrainedFrame, TrainedManifest, TrainedMesh, TrainedMotion } from './contract'
export { TRAINED_MOTIONS } from './contract'

interface SimulationCallbacks {
  onReady: (manifest: TrainedManifest, meshes: TrainedMesh[]) => void
  onFrame: (frame: TrainedFrame) => void
  onError: (error: Error) => void
}

export function startTrainedSimulation(manifestUrl: string, callbacks: SimulationCallbacks) {
  const worker = new Worker(new URL('./simulation.worker.ts', import.meta.url), { type: 'module' })
  let disposed = false
  const send = (message: WorkerRequest) => { if (!disposed) worker.postMessage(message) }
  const fail = (message: string) => {
    if (disposed) return
    worker.terminate()
    disposed = true
    callbacks.onError(new Error(message))
  }
  worker.onmessage = ({ data }: MessageEvent<WorkerResponse>) => {
    if (disposed) return
    if (data.type === 'ready') callbacks.onReady(data.manifest, data.meshes)
    if (data.type === 'frame') callbacks.onFrame(data.frame)
    if (data.type === 'error') fail(data.message)
  }
  worker.onerror = (event) => fail(event.message || 'The trained-model worker could not start')
  worker.onmessageerror = () => fail('The trained-model worker sent an unreadable response')
  send({ type: 'start', manifestUrl: new URL(manifestUrl, document.baseURI).href })
  return {
    setPlaying: (playing: boolean) => send({ type: 'playing', playing }),
    setMotion: (motion: TrainedMotion) => send({ type: 'motion', motion }),
    reset: () => send({ type: 'reset' }),
    dispose() {
      if (!disposed) worker.terminate()
      disposed = true
    },
  }
}
