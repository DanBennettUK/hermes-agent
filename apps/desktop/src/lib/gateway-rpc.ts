/** True when a JSON-RPC call failed because the backend predates the method. */
export function isMissingRpcMethod(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error)

  return /method not found|-32601|unknown method|no such method/i.test(message)
}

interface DesktopCommandFallbackConnection {
  authMode?: 'oauth' | 'token'
  mode?: 'local' | 'remote'
}

/** Use legacy dispatch only for a trusted operator connection and a compatibility error. */
export function shouldUseLegacyDesktopCommandFallback(
  error: unknown,
  connection: DesktopCommandFallbackConnection | null
): boolean {
  const message = error instanceof Error ? error.message : String(error)
  const trustedOperator = connection?.mode === 'local' || connection?.authMode === 'token'

  return trustedOperator && (isMissingRpcMethod(error) || /desktop command is not allowed/i.test(message))
}
