import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'
import { shouldUseLegacyDesktopCommandFallback } from '@/lib/gateway-rpc'
import { $connection } from '@/store/session'

import {
  $gateway,
  closeSecondaryGateways,
  ensureActiveGatewayOpen,
  ensureGatewayForProfile,
  pruneSecondaryGateways,
  reconnectSecondaryGateways,
  setPrimaryGateway
} from './gateway'

type Listener = (event: unknown) => void

class FakeWebSocket {
  static OPEN = 1
  static CLOSED = 3
  static instances: FakeWebSocket[] = []

  readyState = 0
  private listeners: Record<string, Set<Listener>> = {}

  constructor(public url: string) {
    FakeWebSocket.instances.push(this)
    queueMicrotask(() => {
      this.readyState = FakeWebSocket.OPEN
      this.emit('open', {})
    })
  }

  addEventListener(type: string, listener: Listener): void {
    ;(this.listeners[type] ??= new Set()).add(listener)
  }

  removeEventListener(type: string, listener: Listener): void {
    this.listeners[type]?.delete(listener)
  }

  close(): void {
    this.readyState = FakeWebSocket.CLOSED
    this.emit('close', {})
  }

  drop(): void {
    this.close()
  }

  private emit(type: string, event: unknown): void {
    for (const listener of this.listeners[type] ?? []) {
      listener(event)
    }
  }
}

const connection = (overrides: Partial<HermesConnection>): HermesConnection => ({
  baseUrl: '',
  isFullscreen: false,
  logs: [],
  nativeOverlayWidth: 0,
  token: 'operator-token',
  windowButtonPosition: null,
  wsUrl: 'ws://127.0.0.1:8642/api/ws?token=operator-token',
  ...overrides
})

const missingDesktopCommand = new Error('unknown method: desktop.command')

let currentConnection: HermesConnection
let getConnection: ReturnType<typeof vi.fn<(profile?: string | null) => Promise<HermesConnection>>>

beforeEach(async () => {
  closeSecondaryGateways()
  setPrimaryGateway(null, 'default')
  await ensureGatewayForProfile('default')

  FakeWebSocket.instances = []
  vi.stubGlobal('WebSocket', FakeWebSocket)
  getConnection = vi.fn(async () => currentConnection)
  ;(window as { hermesDesktop?: unknown }).hermesDesktop = {
    getConnection,
    getGatewayWsUrl: vi.fn(async () => currentConnection.wsUrl),
    touchBackend: vi.fn(async () => ({ ok: true }))
  }
})

afterEach(async () => {
  closeSecondaryGateways()
  setPrimaryGateway(null, 'default')
  await ensureGatewayForProfile('default')
  $connection.set(null)
  delete (window as { hermesDesktop?: unknown }).hermesDesktop
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

describe('active secondary reconnect connection sync', () => {
  it('replaces a stale local/token descriptor with OAuth before legacy fallback can run', async () => {
    const local = connection({ authMode: 'token', mode: 'local', profile: 'work' })
    const oauth = connection({
      authMode: 'oauth',
      baseUrl: 'https://public.example.com',
      mode: 'remote',
      profile: 'work',
      token: '',
      wsUrl: 'wss://public.example.com/api/ws?ticket=fresh'
    })

    currentConnection = local
    $connection.set(local)
    await ensureGatewayForProfile('work')

    FakeWebSocket.instances.at(-1)!.drop()
    currentConnection = oauth

    expect(shouldUseLegacyDesktopCommandFallback(missingDesktopCommand, $connection.get())).toBe(true)

    const recovered = await ensureActiveGatewayOpen()

    expect(recovered).toBe($gateway.get())
    expect($connection.get()).toBe(oauth)
    expect($connection.get()).toMatchObject({ authMode: 'oauth', mode: 'remote', profile: 'work' })
    expect(shouldUseLegacyDesktopCommandFallback(missingDesktopCommand, $connection.get())).toBe(false)
    expect(getConnection).toHaveBeenNthCalledWith(2, 'work')
  })

  it('keeps fallback available when the reconnected secondary is genuinely local/token', async () => {
    const local = connection({ authMode: 'token', mode: 'local', profile: 'work' })

    currentConnection = local
    await ensureGatewayForProfile('work')
    FakeWebSocket.instances.at(-1)!.drop()
    $connection.set(connection({ authMode: 'oauth', mode: 'remote', profile: 'work' }))

    await ensureActiveGatewayOpen()

    expect($connection.get()).toBe(local)
    expect(shouldUseLegacyDesktopCommandFallback(missingDesktopCommand, $connection.get())).toBe(true)
  })

  it('publishes OAuth metadata after an automatic reconnect timer opens the active secondary', async () => {
    const local = connection({ authMode: 'token', mode: 'local', profile: 'work' })
    const oauth = connection({
      authMode: 'oauth',
      baseUrl: 'https://public.example.com',
      mode: 'remote',
      profile: 'work',
      token: '',
      wsUrl: 'wss://public.example.com/api/ws?ticket=fresh'
    })

    currentConnection = local
    $connection.set(local)
    await ensureGatewayForProfile('work')

    vi.useFakeTimers()
    FakeWebSocket.instances.at(-1)!.drop()
    currentConnection = oauth

    expect(shouldUseLegacyDesktopCommandFallback(missingDesktopCommand, $connection.get())).toBe(true)

    await vi.advanceTimersByTimeAsync(1_000)

    expect($connection.get()).toBe(oauth)
    expect(shouldUseLegacyDesktopCommandFallback(missingDesktopCommand, $connection.get())).toBe(false)
    expect(getConnection).toHaveBeenNthCalledWith(2, 'work')
  })

  it('does not let a wake reconnect for an inactive secondary overwrite the active connection', async () => {
    const work = connection({ authMode: 'token', mode: 'local', profile: 'work' })
    const refreshedWork = connection({
      authMode: 'oauth',
      baseUrl: 'https://public.example.com',
      mode: 'remote',
      profile: 'work',
      token: '',
      wsUrl: 'wss://public.example.com/api/ws?ticket=fresh'
    })
    const activeDefault = connection({ authMode: 'token', mode: 'local', profile: 'default' })

    currentConnection = work
    await ensureGatewayForProfile('work')
    FakeWebSocket.instances.at(-1)!.drop()

    let resolveReconnect!: (value: HermesConnection) => void
    const reconnectDescriptor = new Promise<HermesConnection>(resolve => {
      resolveReconnect = resolve
    })
    getConnection.mockImplementationOnce(() => reconnectDescriptor)

    reconnectSecondaryGateways()
    await vi.waitFor(() => expect(getConnection).toHaveBeenCalledTimes(2))

    await ensureGatewayForProfile('default')
    $connection.set(activeDefault)
    currentConnection = refreshedWork
    resolveReconnect(refreshedWork)

    await vi.waitFor(() => expect(FakeWebSocket.instances.at(-1)?.readyState).toBe(FakeWebSocket.OPEN))
    await Promise.resolve()

    expect($connection.get()).toBe(activeDefault)
  })

  it('closes a secondary pruned while its reconnect descriptor is pending', async () => {
    const work = connection({ authMode: 'token', mode: 'local', profile: 'work' })
    const activeDefault = connection({ authMode: 'token', mode: 'local', profile: 'default' })

    currentConnection = work
    await ensureGatewayForProfile('work')
    const originalSocket = FakeWebSocket.instances.at(-1)!
    originalSocket.drop()

    await ensureGatewayForProfile('default')
    $connection.set(activeDefault)

    let resolveReconnect!: (value: HermesConnection) => void
    const reconnectDescriptor = new Promise<HermesConnection>(resolve => {
      resolveReconnect = resolve
    })
    getConnection.mockImplementationOnce(() => reconnectDescriptor)

    reconnectSecondaryGateways()
    await vi.waitFor(() => expect(getConnection).toHaveBeenCalledTimes(2))

    pruneSecondaryGateways(new Set())
    resolveReconnect(work)
    await new Promise(resolve => setTimeout(resolve, 0))

    expect(FakeWebSocket.instances).toHaveLength(1)
    expect(originalSocket.readyState).toBe(FakeWebSocket.CLOSED)
    expect($connection.get()).toBe(activeDefault)

    currentConnection = work
    await ensureGatewayForProfile('work')

    expect(FakeWebSocket.instances).toHaveLength(2)
    expect(FakeWebSocket.instances.at(-1)?.readyState).toBe(FakeWebSocket.OPEN)
    expect($gateway.get()?.connectionState).toBe('open')
  })
})
