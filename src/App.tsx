import { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react'
import ControlChrome from './components/ControlChrome'
import ReproductionProgress from './components/ReproductionProgress'
import { motions } from './motions'
import type { MotionCategory, SimulationSettings } from './types'

const TrainedView = lazy(() => import('./components/TrainedView'))
const CompareView = lazy(() => import('./components/CompareView'))
const SimulationStage = lazy(() => import('./components/SimulationStage'))
const base = import.meta.env.BASE_URL
const reference = 'https://supermimic.github.io/demo/'
const siteLinks = [
  { id: 'progress', label: 'Progress', href: base },
  { id: 'official', label: 'Official reference', href: `${base}?view=official` },
  { id: 'trained', label: 'Trained model', href: `${base}?view=trained` },
  { id: 'compare', label: 'Compare', href: `${base}?view=compare` },
] as const

function NativeApp() {
  const [settings, setSettings] = useState<SimulationSettings>({ playing: true, policy: true, goal: false, mujoco: false, noise: 0, friction: .5 })
  const [step, setStep] = useState(0)
  const [resetToken, setResetToken] = useState(0)
  const [activeIndex, setActiveIndex] = useState(0)
  const [boot, setBoot] = useState({ visible: true, label: 'loading model config…', progress: 8 })
  const [hint, setHint] = useState('')

  const activeMotion = motions[activeIndex] ?? motions[0]
  const updateSettings = useCallback((patch: Partial<SimulationSettings>) => setSettings(current => ({ ...current, ...patch })), [])
  const reset = useCallback(() => { setStep(0); setResetToken(token => token + 1) }, [])
  const selectMotion = useCallback((index: number) => { setActiveIndex(index); setStep(0); updateSettings({ playing: true }) }, [updateSettings])
  const shuffleCategory = useCallback((category: MotionCategory) => {
    const options = motions.map((motion, index) => ({ motion, index })).filter(item => item.motion.category === category)
    if (options.length) selectMotion(options[Math.floor(Math.random() * options.length)].index)
  }, [selectMotion])

  useEffect(() => {
    const timer = window.setTimeout(() => setBoot({ visible: false, label: '', progress: 100 }), 1500)
    const headline = window.setTimeout(() => {
      setHint('Illustration only — no trained policy is running.')
      window.setTimeout(() => setHint(''), 5000)
    }, 500)
    return () => { window.clearTimeout(timer); window.clearTimeout(headline) }
  }, [])

  const lastReportedStep = useRef(0)
  const handleStep = useCallback((next: number) => {
    if (next === 0 || next - lastReportedStep.current >= 5) {
      lastReportedStep.current = next
      setStep(next)
    }
  }, [])
  const handleDrag = useCallback(() => setHint('Illustration only — no trained policy is running.'), [])

  return <main className="stage">
    <Suspense fallback={null}><SimulationStage playing={settings.playing} goal={settings.goal} mujoco={settings.mujoco} noise={settings.noise} selectedMotion={activeMotion} resetToken={resetToken} onStep={handleStep} onDragStart={handleDrag} /></Suspense>
    <ControlChrome settings={settings} step={step} activeMotion={activeMotion} motions={motions} boot={boot} hint={hint} onChange={updateSettings} onReset={reset} onSelectMotion={selectMotion} onShuffleCategory={shuffleCategory} />
  </main>
}

export default function App() {
  const params = new URLSearchParams(window.location.search)
  if (params.get('native') === '1') return <NativeApp />
  const requested = params.get('view')
  const view = requested === 'official' || requested === 'trained' || requested === 'compare' ? requested : 'progress'
  return <div className="reproduction-site">
    <header className="reproduction-header">
      <a className="reproduction-brand" href={base}>SuperMimic <span>REPRODUCTION</span></a>
      <nav aria-label="Project views">{siteLinks.map(({ id, label, href }) =>
        <a key={id} href={href} aria-current={view === id ? 'page' : undefined}>{label}</a>,
      )}</nav>
    </header>
    <main className="reproduction-content">
      {view === 'progress' ? <ReproductionProgress /> : view === 'official' ?
        <section className="simulation-card" aria-labelledby="official-title">
          <p className="simulation-kicker">Authors’ demo</p>
          <h1 id="official-title">Official reference</h1>
          <p>The authors host the original live simulation. It opens at its official location and is separate from this reproduction.</p>
          <a className="simulation-action" href={reference} target="_blank" rel="noopener noreferrer">Open official simulation ↗</a>
        </section> :
        view === 'trained' ? <Suspense fallback={<p role="status">Loading trained-model view…</p>}><TrainedView /></Suspense> :
        <Suspense fallback={<p role="status">Loading comparison view…</p>}><CompareView /></Suspense>}
      <footer className="reproduction-footer">
        <a href="https://arxiv.org/abs/2610.28480">Paper</a>
        <a href="https://github.com/enyihou/supermimic/tree/source">Reproduction source</a>
        <a href="https://bones.studio/">Motion Data by Bones Studio</a>
        {import.meta.env.PROD ? <a href={`${base}licenses/dependencies.md`}>Third-party notices</a> : null}
      </footer>
    </main>
  </div>
}
