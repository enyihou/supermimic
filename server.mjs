import { createHash, createHmac, randomBytes, scryptSync, timingSafeEqual } from 'node:crypto'
import { createReadStream, existsSync, readFileSync, writeFileSync } from 'node:fs'
import { extname, join, normalize, sep } from 'node:path'
import { fileURLToPath } from 'node:url'
import http from 'node:http'

const root = join(fileURLToPath(new URL('.', import.meta.url)), 'dist')
const authFile = join(fileURLToPath(new URL('.', import.meta.url)), '.supermimic-auth.json')
const port = Number(process.env.PORT || 5173)
const sessionSecret = process.env.SUPERMIMIC_SESSION_SECRET || 'supermimic-local-session-secret'
const mime = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8',
  '.json': 'application/json', '.wasm': 'application/wasm', '.bin': 'application/octet-stream', '.xml': 'application/xml',
  '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.svg': 'image/svg+xml', '.ico': 'image/x-icon',
}

function readAuth() {
  if (!existsSync(authFile)) return null
  try { return JSON.parse(readFileSync(authFile, 'utf8')) } catch { return null }
}

function sessionToken() {
  return createHmac('sha256', sessionSecret).update('supermimic-authenticated').digest('hex')
}

function isAuthenticated(request) {
  const cookie = request.headers.cookie || ''
  const match = cookie.match(/(?:^|; )supermimic_session=([^;]+)/)
  if (!match) return false
  const expected = sessionToken()
  const received = decodeURIComponent(match[1])
  return received.length === expected.length && timingSafeEqual(Buffer.from(received), Buffer.from(expected))
}

function page(title, body) {
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${title}</title><style>body{margin:0;min-height:100vh;display:grid;place-items:center;background:#f5f5f7;color:#1d1d1f;font:16px/1.4 Inter,system-ui,sans-serif}.card{width:min(360px,calc(100vw - 40px));padding:28px;border-radius:24px;background:rgba(255,255,255,.84);box-shadow:0 0 0 .5px rgba(0,0,0,.08),0 20px 60px rgba(0,0,0,.1);backdrop-filter:blur(24px)}h1{font-size:22px;margin:0 0 8px}p{color:#6e6e73;margin:0 0 22px}label{display:block;font-size:13px;font-weight:600;margin:14px 0 6px}input{width:100%;box-sizing:border-box;padding:12px;border:1px solid #d2d2d7;border-radius:12px;font:inherit}button{width:100%;margin-top:18px;padding:12px;border:0;border-radius:999px;background:#1d1d1f;color:#fff;font:600 14px inherit;cursor:pointer}.error{color:#b42318;font-size:13px;margin-top:12px}</style></head><body><main class="card">${body}</main></body></html>`
}

function setupPage(error = '') {
  return page('Set a password', `<h1>Private demo</h1><p>Create the password you’ll use to open this simulation.</p><form method="post" action="/__auth/setup"><label for="password">Password</label><input id="password" name="password" type="password" minlength="8" required autofocus><label for="confirm">Confirm password</label><input id="confirm" name="confirm" type="password" minlength="8" required><button type="submit">Create password</button>${error ? `<div class="error">${error}</div>` : ''}</form>`)
}

function loginPage(error = '') {
  return page('Sign in', `<h1>Private demo</h1><p>Enter the password to continue.</p><form method="post" action="/__auth/login"><label for="password">Password</label><input id="password" name="password" type="password" required autofocus><button type="submit">Enter demo</button>${error ? `<div class="error">${error}</div>` : ''}</form>`)
}

function redirect(response, location) {
  response.writeHead(303, { Location: location, 'Cache-Control': 'no-store' })
  response.end()
}

function formBody(request) {
  return new Promise((resolve) => {
    let body = ''
    request.on('data', chunk => { body += chunk })
    request.on('end', () => resolve(new URLSearchParams(body)))
  })
}

function savePassword(password) {
  const salt = randomBytes(16).toString('hex')
  const hash = scryptSync(password, salt, 64).toString('hex')
  writeFileSync(authFile, JSON.stringify({ algorithm: 'scrypt', salt, hash }, null, 2), { mode: 0o600 })
}

function verifyPassword(password, auth) {
  try {
    const hash = scryptSync(password, auth.salt, 64).toString('hex')
    return hash.length === auth.hash.length && timingSafeEqual(Buffer.from(hash), Buffer.from(auth.hash))
  } catch { return false }
}

const server = http.createServer(async (request, response) => {
  const url = new URL(request.url || '/', `http://${request.headers.host || 'localhost'}`)
  const auth = readAuth()

  if (url.pathname === '/__auth/setup' && request.method === 'POST') {
    if (auth) return redirect(response, '/')
    const body = await formBody(request)
    const password = body.get('password') || ''
    const confirm = body.get('confirm') || ''
    if (password.length < 8) { response.writeHead(400, { 'Content-Type': 'text/html; charset=utf-8' }); return response.end(setupPage('Use at least 8 characters.')) }
    if (password !== confirm) { response.writeHead(400, { 'Content-Type': 'text/html; charset=utf-8' }); return response.end(setupPage('Passwords do not match.')) }
    savePassword(password)
    return redirect(response, '/')
  }

  if (url.pathname === '/__auth/login' && request.method === 'POST') {
    const body = await formBody(request)
    if (!auth || !verifyPassword(body.get('password') || '', auth)) {
      response.writeHead(401, { 'Content-Type': 'text/html; charset=utf-8', 'Cache-Control': 'no-store' })
      return response.end(loginPage('Incorrect password.'))
    }
    response.writeHead(303, { Location: '/', 'Set-Cookie': `supermimic_session=${encodeURIComponent(sessionToken())}; HttpOnly; SameSite=Lax; Path=/; Max-Age=2592000`, 'Cache-Control': 'no-store' })
    return response.end()
  }

  if (!auth) {
    response.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8', 'Cache-Control': 'no-store' })
    return response.end(setupPage())
  }
  if (!isAuthenticated(request)) {
    response.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8', 'Cache-Control': 'no-store' })
    return response.end(loginPage())
  }

  const requested = decodeURIComponent(url.pathname === '/' ? '/index.html' : url.pathname)
  const candidate = normalize(join(root, requested))
  if (!candidate.startsWith(root + sep) || !existsSync(candidate)) {
    response.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' })
    return response.end('Not found')
  }
  response.writeHead(200, { 'Content-Type': mime[extname(candidate)] || 'application/octet-stream', 'Cache-Control': extname(candidate) === '.html' ? 'no-store' : 'public, max-age=31536000, immutable' })
  createReadStream(candidate).pipe(response)
})

server.listen(port, '0.0.0.0', () => console.log(`SuperMimic private demo listening on ${port}`))
