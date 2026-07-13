class PrimaryStartupCancelledError extends Error {
  readonly startupCause: unknown

  constructor(startupCause?: unknown) {
    super('Hermes primary startup was cancelled.')
    this.name = 'PrimaryStartupCancelledError'
    this.startupCause = startupCause
  }
}

interface PrimaryStartupWork {
  readonly generation: number
  bootstrapAbortController: AbortController | null
  bootstrapPromise: Promise<unknown> | null
  startPromise: Promise<unknown> | null
}

async function abortAndWaitForPrimaryStartup(startup: PrimaryStartupWork | null): Promise<void> {
  if (!startup) {
    return
  }

  const controller = startup.bootstrapAbortController
  const promises = [...new Set([startup.bootstrapPromise, startup.startPromise].filter(Boolean))] as Promise<unknown>[]

  try {
    controller?.abort()
  } catch {
    void 0
  }

  await Promise.allSettled(promises)
}

// A cancelled bootstrap may still finish its final marker write while its
// process is winding down. Join that work before best-effort marker removal
// so reset is the last writer and cannot leave a stale completion marker.
async function removeBootstrapMarkerAfterTeardown(
  teardown: Promise<void>,
  removeMarker: () => void,
  onRemoveError: (error: unknown) => void
): Promise<void> {
  await teardown

  try {
    removeMarker()
  } catch (error) {
    onRemoveError(error)
  }
}

/**
 * Generation guard for the one primary Desktop backend startup.
 *
 * Reset/profile re-home cancels the current generation but permits another
 * start. App quit stops the lifecycle permanently so a late renderer event
 * cannot begin a replacement startup while Electron is shutting down.
 */
class PrimaryStartupLifecycle {
  private generation = 0
  private startup: PrimaryStartupWork | null = null
  private stopped = false

  begin(): number {
    if (this.stopped) {
      throw new PrimaryStartupCancelledError()
    }

    this.generation += 1
    this.startup = {
      bootstrapAbortController: null,
      bootstrapPromise: null,
      generation: this.generation,
      startPromise: null
    }
    return this.generation
  }

  captureBootstrap(generation: number, controller: AbortController, promise: Promise<unknown>): PrimaryStartupWork {
    const startup = this.requireCurrent(generation)
    startup.bootstrapAbortController = controller
    startup.bootstrapPromise = promise
    return startup
  }

  captureStart(generation: number, promise: Promise<unknown>): PrimaryStartupWork {
    const startup = this.requireCurrent(generation)
    startup.startPromise = promise
    return startup
  }

  clearBootstrap(generation: number, controller: AbortController, promise: Promise<unknown>): void {
    const startup = this.startup
    if (!startup || startup.generation !== generation) {
      return
    }
    if (startup.bootstrapAbortController === controller) {
      startup.bootstrapAbortController = null
    }
    if (startup.bootstrapPromise === promise) {
      startup.bootstrapPromise = null
    }
  }

  cancel(): PrimaryStartupWork | null {
    const startup = this.startup
    this.startup = null
    this.generation += 1

    try {
      startup?.bootstrapAbortController?.abort()
    } catch {
      void 0
    }

    return startup
  }

  stop(): PrimaryStartupWork | null {
    this.stopped = true
    return this.cancel()
  }

  isCurrent(generation: number): boolean {
    return !this.stopped && generation === this.generation
  }

  assertCurrent(generation: number): void {
    if (!this.isCurrent(generation)) {
      throw new PrimaryStartupCancelledError()
    }
  }

  private requireCurrent(generation: number): PrimaryStartupWork {
    this.assertCurrent(generation)
    return this.startup as PrimaryStartupWork
  }
}

export {
  abortAndWaitForPrimaryStartup,
  PrimaryStartupCancelledError,
  PrimaryStartupLifecycle,
  removeBootstrapMarkerAfterTeardown,
  type PrimaryStartupWork
}
