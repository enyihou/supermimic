import { useEffect, useState } from 'react'
import './reproduction-progress.css'

const states = {
  verified: 'Verified',
  'in-progress': 'In progress',
  blocked: 'Blocked',
  waiting: 'Waiting',
} as const

type Phase = {
  id: string
  label: string
  state: keyof typeof states
  detail: string
}

type Point = { step: number, value: number }
type Series = { id: string, label: string, unit: string, points: Point[] }
type Report = {
  schemaVersion: 1
  updatedAt: string
  summary: string
  training: { numEnvironments: number, completedIterations: number, targetIterations: number }
  phases: Phase[]
  series: Series[]
}

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function iteration(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0
}

function parseReport(value: unknown): Report {
  if (!record(value) || value.schemaVersion !== 1 || typeof value.updatedAt !== 'string' ||
      !/^\d{4}-\d{2}-\d{2}$/.test(value.updatedAt) || typeof value.summary !== 'string' ||
      !record(value.training) || !Array.isArray(value.phases) || !Array.isArray(value.series)) {
    throw new Error('Invalid reproduction status')
  }
  const training = value.training
  if (!iteration(training.completedIterations) || !iteration(training.targetIterations) ||
      training.targetIterations === 0 || training.completedIterations > training.targetIterations ||
      !iteration(training.numEnvironments) || training.numEnvironments === 0) {
    throw new Error('Invalid reproduction training counters')
  }
  const phases: Phase[] = []
  const phaseIds = new Set<string>()
  for (const phase of value.phases) {
    if (!record(phase) || typeof phase.id !== 'string' || phaseIds.has(phase.id) ||
        typeof phase.label !== 'string' || typeof phase.detail !== 'string' ||
        typeof phase.state !== 'string' || !Object.hasOwn(states, phase.state)) {
      throw new Error('Invalid reproduction phase')
    }
    phaseIds.add(phase.id)
    phases.push({ id: phase.id, label: phase.label, detail: phase.detail, state: phase.state as Phase['state'] })
  }
  const series: Series[] = []
  const seriesIds = new Set<string>()
  for (const metric of value.series) {
    if (!record(metric) || typeof metric.id !== 'string' || seriesIds.has(metric.id) ||
        typeof metric.label !== 'string' || typeof metric.unit !== 'string' || !Array.isArray(metric.points)) {
      throw new Error('Invalid reproduction metric')
    }
    let previous = -1
    const points: Point[] = []
    for (const point of metric.points) {
      if (!record(point) || !iteration(point.step) || point.step <= previous ||
          point.step > training.targetIterations || typeof point.value !== 'number' || !Number.isFinite(point.value)) {
        throw new Error('Invalid reproduction metric point')
      }
      previous = point.step
      points.push({ step: point.step, value: point.value })
    }
    seriesIds.add(metric.id)
    if (points.length > 0) series.push({ id: metric.id, label: metric.label, unit: metric.unit, points })
  }
  return {
    schemaVersion: 1,
    updatedAt: value.updatedAt,
    summary: value.summary,
    training: {
      numEnvironments: training.numEnvironments,
      completedIterations: training.completedIterations,
      targetIterations: training.targetIterations,
    },
    phases,
    series,
  }
}

let request: Promise<Report> | undefined
function loadReport() {
  request ??= fetch(`${import.meta.env.BASE_URL}reproduction/status.json`, { cache: 'no-cache' })
    .then(async response => {
      if (!response.ok) throw new Error('Reproduction status is unavailable')
      return parseReport(await response.json())
    })
    .catch(error => {
      request = undefined
      throw error
    })
  return request
}

const whole = new Intl.NumberFormat('en-US')
const scalar = new Intl.NumberFormat('en-US', { maximumSignificantDigits: 5 })

function MetricChart({ series }: { series: Series }) {
  const { points } = series
  const first = points[0]
  const last = points[points.length - 1]
  let min = first.value
  let max = first.value
  for (const point of points) {
    min = Math.min(min, point.value)
    max = Math.max(max, point.value)
  }
  const plotWidth = 624
  const plotHeight = 112
  const plotted = points.map(point => {
    const x = first.step === last.step ? 320 : 8 + plotWidth * (point.step - first.step) / (last.step - first.step)
    const y = min === max ? 64 : 8 + plotHeight * (1 - (point.value - min) / (max - min))
    return [x.toFixed(1), y.toFixed(1)]
  })

  return <figure className="repro-chart">
    <figcaption>
      <h3>{series.label}</h3>
      <span>{scalar.format(last.value)} {series.unit}</span>
    </figcaption>
    <div className="repro-chart-body">
      <div className="repro-y-axis" aria-hidden="true">
        {min === max ? <span>{scalar.format(min)}</span> : <><span>{scalar.format(max)}</span><span>{scalar.format(min)}</span></>}
      </div>
      <svg viewBox="0 0 640 128" preserveAspectRatio="none" role="img"
        aria-label={`${series.label}: ${points.length} measured ${points.length === 1 ? 'point' : 'points'}; latest ${scalar.format(last.value)} ${series.unit} at iteration ${whole.format(last.step)}.`}>
        <path className="repro-chart-grid" d="M8 8H632M8 64H632M8 120H632" vectorEffect="non-scaling-stroke" />
        {points.length > 1 ? <polyline className="repro-chart-series" points={plotted.map(point => point.join(',')).join(' ')} vectorEffect="non-scaling-stroke" /> : null}
        <circle className="repro-chart-dot" cx={plotted[plotted.length - 1][0]} cy={plotted[plotted.length - 1][1]} r="4" />
      </svg>
    </div>
    <div className="repro-x-axis" aria-hidden="true">
      <span>{whole.format(first.step)}</span>{first.step !== last.step ? <span>{whole.format(last.step)} iterations</span> : <span>iteration</span>}
    </div>
    <details className="repro-chart-data">
      <summary>View measured values</summary>
      <div className="repro-data-table">
        <table>
          <thead><tr><th scope="col">Iteration</th><th scope="col">{series.unit || 'Value'}</th></tr></thead>
          <tbody>{points.map(point => <tr key={point.step}><td>{whole.format(point.step)}</td><td>{scalar.format(point.value)}</td></tr>)}</tbody>
        </table>
      </div>
    </details>
  </figure>
}

export default function ReproductionProgress() {
  const [report, setReport] = useState<Report | null>(null)
  const [failed, setFailed] = useState(false)
  useEffect(() => {
    let active = true
    void loadReport().then(
      current => { if (active) setReport(current) },
      () => { if (active) setFailed(true) },
    )
    return () => { active = false }
  }, [])

  return <section className="repro-progress" aria-labelledby="repro-title">
    <header className="repro-intro">
      <p className="repro-kicker">SuperMimic · G1</p>
      <h1 id="repro-title">Reproduction progress</h1>
      {report ? <p className="repro-summary">{report.summary}</p> : null}
    </header>
    {report ? <>
      <div className="repro-counter">
        <div className="repro-counter-head">
          <span id="repro-iteration-label">Completed training iterations</span>
          <span className="repro-iteration-count">{whole.format(report.training.completedIterations)} / {whole.format(report.training.targetIterations)}</span>
        </div>
        <progress aria-labelledby="repro-iteration-label" value={report.training.completedIterations} max={report.training.targetIterations} />
        <div className="repro-counter-meta">
          <span>1 GPU · {whole.format(report.training.numEnvironments)} environments</span>
          <span>Updated <time dateTime={report.updatedAt}>{report.updatedAt}</time></span>
        </div>
      </div>
      <div className="repro-columns">
        <div className="repro-phases">
          <h2>Milestones</h2>
          <ol>{report.phases.map(phase => <li key={phase.id}>
            <div className="repro-phase-heading">
              <strong>{phase.label}</strong>
              <span className="repro-tag" data-state={phase.state}>{states[phase.state]}</span>
            </div>
            <p>{phase.detail}</p>
          </li>)}</ol>
        </div>
        <div className="repro-metrics">
          <h2>Measured results</h2>
          {report.series.length > 0 ? report.series.map(series => <MetricChart key={series.id} series={series} />) :
            <div className="repro-empty">
              <p>No training or evaluation results published yet.</p>
              <p>Checkpoint and metric charts will appear here when measurements are verified.</p>
            </div>}
        </div>
      </div>
    </> : <p role="status" className="repro-summary">
      {failed ? 'The reproduction report is unavailable. Reload this page to try again.' : 'Loading reproduction report…'}
    </p>}
  </section>
}
