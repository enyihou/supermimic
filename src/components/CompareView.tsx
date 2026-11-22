import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import SimulationStage from './SimulationStage'
import { motions } from '../motions'
import { BUILT_IN_CHOICES, ORIGINAL_SIMULATION_ID, loadModelChoices, type ModelChoice } from '../trained/catalog'
import { TrainedStage } from './TrainedView'
import type { Motion } from '../types'

const base = import.meta.env.BASE_URL
const originalMinWidth = 390

function OriginalPane({ side }: { side: string }) {
  const viewport = useRef<HTMLDivElement>(null)
  const iframe = useRef<HTMLIFrameElement>(null)

  useLayoutEffect(() => {
    const container = viewport.current
    const frame = iframe.current
    if (!container || !frame) return

    // Preserve the existing control dock at its normal phone width.
    // Only the comparison frame is scaled; the simulator itself is untouched.
    const fit = () => {
      const width = container.clientWidth
      if (!width) return
      const scale = Math.min(1, width / originalMinWidth)
      frame.style.width = `${Math.max(width, originalMinWidth)}px`
      frame.style.height = `${container.clientHeight / scale}px`
      frame.style.transform = scale === 1 ? '' : `scale(${scale})`
    }
    fit()
    const resize = new ResizeObserver(fit)
    resize.observe(container)
    return () => resize.disconnect()
  }, [])

  return <div ref={viewport} className="comparison-viewport">
    <iframe ref={iframe} className="comparison-original-viewport" title={`${side} original simulation`} src={`${base}?native=1`} />
  </div>
}

type NativePaneProps = {
  side: string
}

function NativePane({ side }: NativePaneProps) {
  const [playing, setPlaying] = useState(true)
  const [motionIndex, setMotionIndex] = useState(0)
  const [step, setStep] = useState(0)
  const [resetToken, setResetToken] = useState(0)
  const activeMotion = motions[motionIndex] ?? motions[0]
  const onStep = useCallback((next: number) => setStep(next), [])
  const onDragStart = useCallback(() => undefined, [])
  const reset = useCallback(() => {
    setStep(0)
    setResetToken(token => token + 1)
  }, [])

  return <>
    <div className="comparison-viewport comparison-native-viewport">
      <SimulationStage
        playing={playing}
        goal={false}
        mujoco={false}
        noise={0}
        selectedMotion={activeMotion}
        resetToken={resetToken}
        onStep={onStep}
        onDragStart={onDragStart}
      />
    </div>
    <div className="comparison-controls">
      <button type="button" onClick={() => setPlaying(value => !value)}>{playing ? 'Pause' : 'Play'}</button>
      <button type="button" onClick={reset}>Reset</button>
      <label>Motion
        <select value={motionIndex} onChange={event => {
          setMotionIndex(Number(event.target.value))
          reset()
        }}>
          {motions.slice(0, 12).map((motion: Motion, index) => <option value={index} key={`${motion.label}-${index}`}>{motion.label}</option>)}
        </select>
      </label>
      <span className="comparison-step">{side} · step {step.toLocaleString('en-US')}</span>
    </div>
  </>
}

type ComparisonPaneProps = {
  side: 'Left' | 'Right'
  choice: ModelChoice
  choices: ModelChoice[]
  onChoice: (id: string) => void
}

function ComparisonPane({ side, choice, choices, onChoice }: ComparisonPaneProps) {
  return <article className="comparison-pane" aria-labelledby={`comparison-${side.toLowerCase()}-title`}>
    <header className="comparison-pane-header">
      <div>
        <p className="simulation-kicker">{side} view</p>
        <h2 id={`comparison-${side.toLowerCase()}-title`}>{choice.label}</h2>
      </div>
      <label className="comparison-model-select">Model
        <select aria-label={`${side} model`} value={choice.id} onChange={event => onChoice(event.target.value)}>
          {choices.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}
        </select>
      </label>
    </header>
    {choice.kind === 'original' ? <OriginalPane side={side} /> :
      choice.kind === 'native' ? <NativePane side={side} /> : <TrainedStage key={choice.id} modelId={choice.id} />}
    <p className="comparison-provenance">{choice.kind === 'original' ? 'The original simulation with its full controls, running independently in this pane.' :
      choice.kind === 'native' ? 'Local illustration path from the existing browser simulation.' :
      'This export binds its actor and world-model checkpoint hashes in the release manifest.'}</p>
  </article>
}

export default function CompareView() {
  const [choices, setChoices] = useState(BUILT_IN_CHOICES)
  const [leftId, setLeftId] = useState(ORIGINAL_SIMULATION_ID)
  const [rightId, setRightId] = useState('native')
  const [error, setError] = useState('')

  useEffect(() => {
    const request = new AbortController()
    void loadModelChoices(base).then(
      next => { if (!request.signal.aborted) setChoices(next) },
      reason => { if (!request.signal.aborted) setError(reason instanceof Error ? reason.message : 'Model registry unavailable') },
    )
    return () => request.abort()
  }, [])

  const choiceById = useMemo(() => new Map(choices.map(choice => [choice.id, choice])), [choices])
  const left = choiceById.get(leftId) ?? choices[0]
  const right = choiceById.get(rightId) ?? choices[0]

  return <section className="comparison-card" aria-labelledby="comparison-title">
    <header className="comparison-intro">
      <p className="simulation-kicker">Model comparison</p>
      <h1 id="comparison-title">Two simulations, one workspace</h1>
      <p>Choose the original simulation, the local illustration, or a released model independently on each side. Each trained export carries both checkpoint identities in its manifest.</p>
    </header>
    {error ? <p className="comparison-registry-note" role="status">{error}. The local simulation remains available.</p> : null}
    {!error && choices.length === BUILT_IN_CHOICES.length ? <p className="comparison-registry-note" role="status">No trained browser exports are published yet. Both built-in simulations are available on either side.</p> : null}
    <div className="comparison-grid">
      <ComparisonPane side="Left" choice={left} choices={choices} onChoice={setLeftId} />
      <ComparisonPane side="Right" choice={right} choices={choices} onChoice={setRightId} />
    </div>
  </section>
}
