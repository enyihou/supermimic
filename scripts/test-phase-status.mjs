import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { copyFile, mkdir, mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('..', import.meta.url))
const verify = (publicRoot) => spawnSync('bash', ['scripts/verify-phase-status.sh', publicRoot], {
  cwd: root, encoding: 'utf8',
})

test('source and detached build evidence must reproduce the exact public status', async () => {
  const fixture = await mkdtemp(join(tmpdir(), 'phase-status-test-'))
  try {
    const reproduction = join(fixture, 'reproduction')
    const models = join(fixture, 'models')
    await mkdir(reproduction)
    await mkdir(models)
    const reports = (await readdir(join(root, 'public', 'reproduction')))
      .filter((name) => /^(preflight|status|smoke-audit|benchmark-audit|full-audit)\.json$/.test(name))
    for (const input of [...reports.map((name) => `reproduction/${name}`), 'models/releases.json']) {
      await copyFile(join(root, 'public', input), join(fixture, input))
    }

    const valid = verify(fixture)
    assert.equal(valid.status, 0, valid.stderr)

    const status = join(reproduction, 'status.json')
    const generated = await readFile(status, 'utf8')
    const report = JSON.parse(generated)
    const release = report.phases.find(({ id }) => id === 'release')
    release.state = release.state === 'verified' ? 'waiting' : 'verified'
    await writeFile(status, JSON.stringify(report))
    const inconsistent = verify(fixture)
    assert.equal(inconsistent.status, 1)
    assert.match(inconsistent.stderr, /does not match the verified phase evidence/)

    await writeFile(status, generated.replace('{', '{\n  "summary": "Duplicate hidden by ordinary JSON parsing",'))
    assert.equal(verify(fixture).status, 1)
  } finally {
    await rm(fixture, { recursive: true, force: true })
  }
})
