import { useEffect, useMemo, useState, type CSSProperties } from 'react'
import type { Motion, MotionCategory, SimulationSettings } from '../types'

type BootState = {
  visible: boolean
  label: string
  progress: number
}

export type ControlChromeProps = {
  settings: SimulationSettings
  step: number
  activeMotion: Motion
  motions: Motion[]
  boot?: BootState
  hint?: string
  onChange: (patch: Partial<SimulationSettings>) => void
  onReset: () => void
  onSelectMotion: (index: number) => void
  onShuffleCategory: (category: MotionCategory) => void
}

const categoryOrder: MotionCategory[] = ['acrobatics', 'contact rich', 'manipulation', 'locomotion', 'dance', 'unseen']

function ArrowLeftIcon() {
  return <svg viewBox="0 0 12 12" aria-hidden="true"><path d="M7.5 2.5 4 6l3.5 3.5" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" /></svg>
}

function ResetIcon() {
  return <svg viewBox="0 0 16 16" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"><path d="M3.4 8a4.6 4.6 0 1 0 1.6-3.5" /><path d="M2.1 2.3v3.4h3.4" /></svg>
}

function PlayIcon({ playing }: { playing: boolean }) {
  return playing
    ? <svg viewBox="0 0 16 16" width="14" height="14" fill="currentColor" aria-hidden="true"><rect x="3" y="2" width="3.6" height="12" rx="1.2" /><rect x="9.4" y="2" width="3.6" height="12" rx="1.2" /></svg>
    : <svg viewBox="0 0 16 16" width="14" height="14" fill="currentColor" aria-hidden="true"><path d="M4.2 2.3 13 8l-8.8 5.7Z" /></svg>
}

function ControlsIcon() {
  return <svg viewBox="0 0 16 16" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" aria-hidden="true"><path d="M2 4.5h12M2 11.5h12" /><circle cx="6" cy="4.5" r="2" fill="#fff" /><circle cx="10.5" cy="11.5" r="2" fill="#fff" /></svg>
}

function ListIcon() {
  return <svg viewBox="0 0 16 16" width="16" height="16" fill="currentColor" aria-hidden="true"><rect x="2" y="3" width="12" height="1.6" rx=".8" /><rect x="2" y="7.2" width="12" height="1.6" rx=".8" /><rect x="2" y="11.4" width="12" height="1.6" rx=".8" /></svg>
}

function formatCategory(category: MotionCategory) {
  return category
}

export default function ControlChrome({
  settings,
  step,
  activeMotion,
  motions,
  boot = { visible: false, label: '', progress: 0 },
  hint,
  onChange,
  onReset,
  onSelectMotion,
  onShuffleCategory,
}: ControlChromeProps) {
  const [sheet, setSheet] = useState<'motions' | 'ctx' | null>(null)
  const [filter, setFilter] = useState('')
  const [expanded, setExpanded] = useState<Set<MotionCategory>>(new Set(['acrobatics']))

  const grouped = useMemo(() => categoryOrder.map(category => ({
    category,
    items: motions.map((motion, index) => ({ motion, index })).filter(({ motion }) => motion.category === category),
  })).filter(group => group.items.length > 0), [motions])

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null
      if (target?.tagName === 'INPUT' || target?.tagName === 'TEXTAREA') return
      if (event.key === 'Escape') setSheet(null)
      if (event.code === 'Space') {
        event.preventDefault()
        onChange({ playing: !settings.playing })
      }
      if (event.key.toLowerCase() === 'r') onReset()
      if (event.key.toLowerCase() === 'g') onChange({ goal: !settings.goal })
      if (event.key.toLowerCase() === 's') onChange({ mujoco: !settings.mujoco })
      if (event.key.toLowerCase() === 'p') onChange({ policy: !settings.policy })
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [onChange, onReset, settings])

  const updateNoise = (value: number) => onChange({ noise: value })
  const updateFriction = (value: number) => onChange({ friction: value })
  const filteredGroups = grouped.map(group => ({
    ...group,
    items: group.items.filter(({ motion }) => motion.label.toLowerCase().includes(filter.toLowerCase())),
  })).filter(group => group.items.length > 0)

  const toggleFolder = (category: MotionCategory) => setExpanded(current => {
    const next = new Set(current)
    if (next.has(category)) next.delete(category)
    else next.add(category)
    return next
  })

  return <>
    <a className="panel glass home" href="https://supermimic.github.io/" aria-label="SuperMimic home"><ArrowLeftIcon />SuperMimic</a>

    <div className={`panel legend ${settings.mujoco ? 'sim-active' : ''}`}>
      <h1>Live imagination</h1>
      <div className="legend-track glass">
        <span className="legend-key"><span className="dot wm" />World model</span>
        <button type="button" className={`legend-toggle ${settings.goal ? 'active' : ''}`} title="toggle goal pose (g)" onClick={() => onChange({ goal: !settings.goal })}><span className="dot ref" />Goal</button>
        <button type="button" className={`legend-toggle ${settings.mujoco ? 'active' : ''}`} title="toggle MuJoCo ground truth (s)" onClick={() => onChange({ mujoco: !settings.mujoco })}><span className="dot sim" />MuJoCo</button>
      </div>
      <div className="depth">step {step.toLocaleString('en-US')} since the last reset</div>
    </div>

    <div className={`hint ${hint ? 'visible' : ''}`}>{hint || 'drag the robot — it has to catch itself'}</div>

    <div className="panel glass dock">
      <button type="button" className="icon" title="reset state (r)" aria-label="reset" onClick={onReset}><ResetIcon /></button>
      <button type="button" className="icon" title="space" aria-label="play or pause" onClick={() => onChange({ playing: !settings.playing })}><PlayIcon playing={settings.playing} /></button>
      <span className="sep ctx-sep" />

      <div className={`context glass dense ${sheet === 'ctx' ? 'open' : ''}`}>
        <div className="control-row"><span>Policy</span><button type="button" className={`switch ${settings.policy ? 'on' : ''}`} aria-label="toggle policy (p)" title="toggle policy (p)" onClick={() => onChange({ policy: !settings.policy })}>{settings.policy ? 'on' : 'off'}</button></div>
        <div className="slider"><label htmlFor="noise">Action noise<b>{settings.noise.toFixed(2)}</b></label><input id="noise" type="range" min="0" max=".25" step=".01" value={settings.noise} onChange={event => updateNoise(Number(event.target.value))} style={{ '--p': `${settings.noise / .25 * 100}%` } as CSSProperties} /></div>
        <div className="slider"><label htmlFor="friction">Friction<b>{boot.visible ? '–' : `${Math.round(settings.friction * 100)}%`}</b></label><input id="friction" type="range" min="0" max="100" step="1" disabled={boot.visible} value={Math.round(settings.friction * 100)} onChange={event => updateFriction(Number(event.target.value) / 100)} style={{ '--p': `${settings.friction * 100}%` } as CSSProperties} /></div>
      </div>

      <button type="button" className={`icon ctx-trigger ${sheet === 'ctx' ? 'on' : ''}`} title="controls" aria-label="controls" onClick={() => setSheet(current => current === 'ctx' ? null : 'ctx')}><ControlsIcon /></button>
      <span className="sep" />
      <button type="button" className={`motions-button ${sheet === 'motions' ? 'on' : ''}`} title="motions" onClick={() => setSheet(current => current === 'motions' ? null : 'motions')}><ListIcon /><span className="motion-label"><span className="motion-cat">{activeMotion.category}</span>{activeMotion.category ? ' · ' : ''}{activeMotion.label}</span></button>
      <div className={`sheet glass dense ${sheet === 'motions' ? 'open' : ''}`}>
        <input className="filter" type="text" placeholder="Filter motions" aria-label="Filter motions" autoComplete="off" value={filter} onChange={event => setFilter(event.target.value)} />
        <div className="motion-list">
          {filteredGroups.map(group => <div className={`folder ${expanded.has(group.category) ? 'expanded' : ''}`} key={group.category}>
            <button type="button" className="folder-head" title="play this folder, shuffled" onClick={event => { if ((event.target as HTMLElement).closest('.chev')) toggleFolder(group.category); else onShuffleCategory(group.category) }}><span className="chev" />{formatCategory(group.category)}<span className="folder-count">{filter ? group.items.length : motions.filter(motion => motion.category === group.category).length}</span></button>
            {expanded.has(group.category) && <div className="folder-items">{group.items.map(({ motion, index }) => <button type="button" className={`motion-item ${motion.label === activeMotion.label ? 'active' : ''}`} key={`${motion.label}-${index}`} onClick={() => { onSelectMotion(index); setSheet(null) }}>{motion.label}</button>)}</div>}
          </div>)}
        </div>
        <a className="legal" href="https://supermimic.github.io/LICENSE.html" target="_blank" rel="noreferrer">© 2026 SuperMimic · License</a>
      </div>
    </div>

    {boot.visible && <div className="panel glass dense boot"><div className="boot-message">{boot.label || 'loading…'}</div><div className="boot-bar"><div className="boot-fill" style={{ width: `${Math.max(0, Math.min(100, boot.progress))}%` }} /></div></div>}
  </>
}
