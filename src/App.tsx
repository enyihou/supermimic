import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import ControlChrome from './components/ControlChrome'
import SimulationStage from './components/SimulationStage'
import ClientGate from './ClientGate'
import { motions } from './motions'
import type { MotionCategory, SimulationSettings } from './types'

function NativeApp() {
  const [settings, setSettings] = useState<SimulationSettings>({ playing: true, policy: true, goal: false, mujoco: false, noise: 0, friction: .5 })
  const [step, setStep] = useState(0)
  const [resetToken, setResetToken] = useState(0)
  const [activeIndex, setActiveIndex] = useState(15)
  const [boot, setBoot] = useState({ visible: true, label: 'loading model config…', progress: 8 })
  const [hint, setHint] = useState('')

  const activeMotion = motions[activeIndex] ?? motions[15]
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
      setHint('Your browser is dreaming this robot.')
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
  const hintText = useMemo(() => hint, [hint])

  return <main className="stage">
    <SimulationStage playing={settings.playing} goal={settings.goal} mujoco={settings.mujoco} noise={settings.noise} selectedMotion={activeMotion} resetToken={resetToken} onStep={handleStep} onDragStart={() => setHint('drag the robot — it has to catch itself')} />
    <ControlChrome settings={settings} step={step} activeMotion={activeMotion} motions={motions} boot={boot} hint={hintText} onChange={updateSettings} onReset={reset} onSelectMotion={selectMotion} onShuffleCategory={shuffleCategory} />
  </main>
}

export default function App() {
  const native = new URLSearchParams(window.location.search).get('native') === '1'
  const officialPath = `${import.meta.env.BASE_URL}official-demo/index.html?standalone`
  return <ClientGate>{native ? <NativeApp /> : <main className="official-frame"><iframe title="SuperMimic live imagination" src={officialPath} /></main>}</ClientGate>
}
