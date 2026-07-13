import { describe, expect, it } from 'vitest'

import { isMissingRpcMethod, shouldUseLegacyDesktopCommandFallback } from './gateway-rpc'

describe('isMissingRpcMethod', () => {
  it('detects JSON-RPC method-not-found errors', () => {
    expect(isMissingRpcMethod(new Error('unknown method: projects.create'))).toBe(true)
    expect(isMissingRpcMethod(new Error('Method not found'))).toBe(true)
    expect(isMissingRpcMethod(new Error('RPC failed: -32601'))).toBe(true)
  })

  it('ignores unrelated failures', () => {
    expect(isMissingRpcMethod(new Error('Hermes gateway is not connected'))).toBe(false)
    expect(isMissingRpcMethod(new Error('no such project'))).toBe(false)
  })
})

describe('shouldUseLegacyDesktopCommandFallback', () => {
  const local = { mode: 'local' as const }
  const oauth = { authMode: 'oauth' as const, mode: 'remote' as const }
  const token = { authMode: 'token' as const, mode: 'remote' as const }

  it('allows compatibility errors only for trusted local or token connections', () => {
    const missingMethod = new Error('unknown method: desktop.command')
    const unsupportedExtension = new Error('desktop command is not allowed: customer-skill')

    expect(shouldUseLegacyDesktopCommandFallback(missingMethod, local)).toBe(true)
    expect(shouldUseLegacyDesktopCommandFallback(missingMethod, token)).toBe(true)
    expect(shouldUseLegacyDesktopCommandFallback(unsupportedExtension, local)).toBe(true)
    expect(shouldUseLegacyDesktopCommandFallback(unsupportedExtension, token)).toBe(true)

    expect(shouldUseLegacyDesktopCommandFallback(missingMethod, oauth)).toBe(false)
    expect(shouldUseLegacyDesktopCommandFallback(unsupportedExtension, oauth)).toBe(false)
    expect(shouldUseLegacyDesktopCommandFallback(missingMethod, null)).toBe(false)
  })

  it('does not fall back for terminal policy, usage, or ownership errors', () => {
    expect(
      shouldUseLegacyDesktopCommandFallback(new Error('public desktop command is not authorized: debug'), token)
    ).toBe(false)
    expect(shouldUseLegacyDesktopCommandFallback(new Error('usage: /queue <prompt>'), token)).toBe(false)
    expect(shouldUseLegacyDesktopCommandFallback(new Error('session belongs to another principal'), token)).toBe(false)
  })
})
