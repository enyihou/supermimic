import { FormEvent, useEffect, useState, type ReactNode } from 'react'

const STORAGE_KEY = 'supermimic-password-hash'

async function digest(value: string) {
  const bytes = new TextEncoder().encode(value)
  const hash = await crypto.subtle.digest('SHA-256', bytes)
  return [...new Uint8Array(hash)].map(byte => byte.toString(16).padStart(2, '0')).join('')
}

export default function ClientGate({ children }: { children: ReactNode }) {
  const [storedHash, setStoredHash] = useState<string | null | undefined>(undefined)
  const [unlocked, setUnlocked] = useState(false)
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [error, setError] = useState('')

  useEffect(() => setStoredHash(window.localStorage.getItem(STORAGE_KEY)), [])
  if (storedHash === undefined) return null
  if (unlocked) return children

  const firstLogin = storedHash === null
  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setError('')
    if (password.length < 8) return setError('Use at least 8 characters.')
    if (firstLogin && password !== confirm) return setError('Passwords do not match.')
    const hash = await digest(password)
    if (firstLogin) {
      window.localStorage.setItem(STORAGE_KEY, hash)
      setStoredHash(hash)
      setUnlocked(true)
    } else if (hash === storedHash) {
      setUnlocked(true)
    } else {
      setError('Incorrect password.')
    }
  }

  return <main className="client-gate"><section className="client-gate-card">
    <h1>Private demo</h1>
    <p>{firstLogin ? 'Create the password you’ll use to open this simulation.' : 'Enter the password to continue.'}</p>
    <form onSubmit={submit}>
      <label htmlFor="client-password">Password</label>
      <input id="client-password" type="password" value={password} onChange={event => setPassword(event.target.value)} autoFocus required />
      {firstLogin && <><label htmlFor="client-confirm">Confirm password</label><input id="client-confirm" type="password" value={confirm} onChange={event => setConfirm(event.target.value)} required /></>}
      <button type="submit">{firstLogin ? 'Create password' : 'Enter demo'}</button>
      {error && <div className="client-gate-error">{error}</div>}
    </form>
  </section></main>
}
