const modelIdPattern = /^[a-z0-9][a-z0-9-]{0,63}$/
const qaModelPattern = /(?:^|-)(?:synthetic|fixture|mock|test|smoke|benchmark|untrained|random)(?:-|$)/

export type ModelChoice = {
  id: string
  label: string
  kind: 'original' | 'native' | 'official' | 'trained'
}

// IDs containing colons cannot collide with server-provided model release IDs.
export const ORIGINAL_SIMULATION_ID = 'original:simulation'
export const BUILT_IN_CHOICES: ModelChoice[] = [
  { id: ORIGINAL_SIMULATION_ID, label: 'Original local illustration', kind: 'original' },
  { id: 'native', label: 'Compact local illustration', kind: 'native' },
  { id: 'official:demo', label: 'Official demo (external)', kind: 'official' },
]

function validModelId(value: unknown): value is string {
  return typeof value === 'string' && modelIdPattern.test(value) && !qaModelPattern.test(value)
}

function uniqueModelIds(ids: unknown[]) {
  return [...new Set(ids.filter(validModelId))]
}

export async function loadModelChoices(base: string): Promise<ModelChoice[]> {
  const response = await fetch(`${base}models/releases.json`, { cache: 'no-cache' })
  if (!response.ok) throw new Error('Model registry unavailable')
  const registry: unknown = await response.json()
  if (typeof registry !== 'object' || registry === null ||
      !('schema_version' in registry) || registry.schema_version !== 1 ||
      !('active' in registry)) {
    throw new Error('Invalid model registry')
  }

  if (registry.active !== null && !validModelId(registry.active)) throw new Error('Invalid active model ID')
  if ('models' in registry && (!Array.isArray(registry.models) || registry.models.some(id => !validModelId(id)))) {
    throw new Error('Invalid model release list')
  }
  const extra = 'models' in registry && Array.isArray(registry.models) ? registry.models : []
  const ids = uniqueModelIds([registry.active, ...extra])
  return [
    ...BUILT_IN_CHOICES,
    ...ids.map(id => ({ id, label: id, kind: 'trained' as const })),
  ]
}
