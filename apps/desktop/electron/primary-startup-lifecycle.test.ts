import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import test from 'node:test'

import {
  abortAndWaitForPrimaryStartup,
  PrimaryStartupCancelledError,
  PrimaryStartupLifecycle,
  removeBootstrapMarkerAfterTeardown
} from './primary-startup-lifecycle'

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (error: Error) => void
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })
  return { promise, reject, resolve }
}

test('reset during primary startup prevents stale bridge ownership and sidecar startup', async () => {
  const lifecycle = new PrimaryStartupLifecycle()
  const remoteReady = deferred<void>()
  const generation = lifecycle.begin()
  let bridgeOwner: string | null = null
  let sidecarStarts = 0
  let connection: string | null = null

  const startup = (async () => {
    await remoteReady.promise
    lifecycle.assertCurrent(generation)
    bridgeOwner = 'primary'
    sidecarStarts += 1
    lifecycle.assertCurrent(generation)
    connection = 'old'
  })()

  lifecycle.cancel()
  remoteReady.resolve()

  await assert.rejects(startup, PrimaryStartupCancelledError)
  assert.equal(bridgeOwner, null)
  assert.equal(sidecarStarts, 0)
  assert.equal(connection, null)
})

test('profile apply cancels an in-flight start and permits only the replacement to attach', async () => {
  const lifecycle = new PrimaryStartupLifecycle()
  const oldRemoteReady = deferred<void>()
  const oldGeneration = lifecycle.begin()
  let bridgeOwner: string | null = null
  let sidecarOwner: string | null = null
  let connection: string | null = null

  const oldStartup = (async () => {
    await oldRemoteReady.promise
    lifecycle.assertCurrent(oldGeneration)
    bridgeOwner = 'old'
    sidecarOwner = 'old'
    connection = 'old'
  })()

  // Profile apply/re-home uses the same cancellation semantics as reset.
  lifecycle.cancel()
  bridgeOwner = null
  sidecarOwner = null

  const newGeneration = lifecycle.begin()
  lifecycle.assertCurrent(newGeneration)
  bridgeOwner = 'new'
  sidecarOwner = 'new'
  connection = 'new'

  oldRemoteReady.resolve()
  await assert.rejects(oldStartup, PrimaryStartupCancelledError)
  assert.equal(bridgeOwner, 'new')
  assert.equal(sidecarOwner, 'new')
  assert.equal(connection, 'new')
})

test('quit cancels an in-flight primary start and permanently rejects late restarts', async () => {
  const lifecycle = new PrimaryStartupLifecycle()
  const runtimeReady = deferred<void>()
  const generation = lifecycle.begin()
  let spawned = false

  const startup = (async () => {
    await runtimeReady.promise
    lifecycle.assertCurrent(generation)
    spawned = true
  })()

  lifecycle.stop()
  runtimeReady.resolve()

  await assert.rejects(startup, PrimaryStartupCancelledError)
  assert.equal(spawned, false)
  assert.throws(() => lifecycle.begin(), PrimaryStartupCancelledError)
})

for (const reason of ['bootstrap reset', 'profile re-home']) {
  test(`${reason} aborts and joins bootstrap before replacement startup`, async () => {
    const lifecycle = new PrimaryStartupLifecycle()
    const installerCanExit = deferred<void>()
    const abortObserved = deferred<void>()
    const oldGeneration = lifecycle.begin()
    const oldController = new AbortController()
    const events: string[] = []

    oldController.signal.addEventListener(
      'abort',
      () => {
        events.push('old-aborted')
        abortObserved.resolve()
      },
      { once: true }
    )

    const oldBootstrap = (async () => {
      events.push('old-write-start')
      await abortObserved.promise
      await installerCanExit.promise
      events.push('old-write-end')
    })()
    const oldStart = oldBootstrap.then(() => {
      lifecycle.assertCurrent(oldGeneration)
    })

    lifecycle.captureBootstrap(oldGeneration, oldController, oldBootstrap)
    lifecycle.captureStart(oldGeneration, oldStart)

    const teardown = abortAndWaitForPrimaryStartup(lifecycle.cancel())
    await abortObserved.promise
    let teardownFinished = false
    void teardown.then(() => {
      teardownFinished = true
    })
    await Promise.resolve()

    assert.equal(oldController.signal.aborted, true)
    assert.equal(teardownFinished, false, 'teardown must still be joining the installer')

    installerCanExit.resolve()
    await teardown
    events.push('teardown-finished')

    const replacementGeneration = lifecycle.begin()
    events.push('replacement-start')
    lifecycle.assertCurrent(replacementGeneration)

    assert.deepEqual(events, [
      'old-write-start',
      'old-aborted',
      'old-write-end',
      'teardown-finished',
      'replacement-start'
    ])
  })
}

test('stale bootstrap cleanup cannot clear a newer controller and promise', () => {
  const lifecycle = new PrimaryStartupLifecycle()
  const oldGeneration = lifecycle.begin()
  const oldController = new AbortController()
  const oldBootstrap = Promise.resolve()

  lifecycle.captureBootstrap(oldGeneration, oldController, oldBootstrap)
  lifecycle.cancel()

  const replacementGeneration = lifecycle.begin()
  const replacementController = new AbortController()
  const replacementBootstrap = Promise.resolve()
  const replacement = lifecycle.captureBootstrap(replacementGeneration, replacementController, replacementBootstrap)

  lifecycle.clearBootstrap(oldGeneration, oldController, oldBootstrap)

  assert.equal(replacement.bootstrapAbortController, replacementController)
  assert.equal(replacement.bootstrapPromise, replacementBootstrap)
  assert.equal(replacementController.signal.aborted, false)
})

test('joined bootstrap teardown prevents overlapping installer writes', async () => {
  const lifecycle = new PrimaryStartupLifecycle()
  const oldCanExit = deferred<void>()
  const oldAborted = deferred<void>()
  let activeWrites = 0
  let maxActiveWrites = 0

  const oldGeneration = lifecycle.begin()
  const oldController = new AbortController()
  oldController.signal.addEventListener('abort', () => oldAborted.resolve(), { once: true })
  const oldBootstrap = (async () => {
    activeWrites += 1
    maxActiveWrites = Math.max(maxActiveWrites, activeWrites)
    await oldAborted.promise
    await oldCanExit.promise
    activeWrites -= 1
  })()
  lifecycle.captureBootstrap(oldGeneration, oldController, oldBootstrap)
  lifecycle.captureStart(oldGeneration, oldBootstrap)

  const teardown = abortAndWaitForPrimaryStartup(lifecycle.cancel())
  const replacement = (async () => {
    await teardown
    const replacementGeneration = lifecycle.begin()
    lifecycle.assertCurrent(replacementGeneration)
    activeWrites += 1
    maxActiveWrites = Math.max(maxActiveWrites, activeWrites)
    activeWrites -= 1
  })()

  await oldAborted.promise
  await Promise.resolve()
  assert.equal(activeWrites, 1)

  oldCanExit.resolve()
  await replacement

  assert.equal(activeWrites, 0)
  assert.equal(maxActiveWrites, 1)
})

test('bootstrap reset removes the marker after joined teardown so a stale bootstrap cannot recreate it', async t => {
  const lifecycle = new PrimaryStartupLifecycle()
  const bootstrapCanExit = deferred<void>()
  const abortObserved = deferred<void>()
  const markerDir = fs.mkdtempSync(path.join(import.meta.dirname, '.bootstrap-reset-test-'))
  const markerPath = path.join(markerDir, '.hermes-bootstrap-complete')
  const events: string[] = []

  t.after(() => fs.rmSync(markerDir, { force: true, recursive: true }))

  const generation = lifecycle.begin()
  const controller = new AbortController()
  controller.signal.addEventListener('abort', () => abortObserved.resolve(), { once: true })
  const bootstrap = (async () => {
    await abortObserved.promise
    await bootstrapCanExit.promise
    fs.writeFileSync(markerPath, 'stale bootstrap marker')
    events.push('stale-marker-written')
  })()
  const startup = bootstrap.then(() => lifecycle.assertCurrent(generation))

  lifecycle.captureBootstrap(generation, controller, bootstrap)
  lifecycle.captureStart(generation, startup)

  const teardown = abortAndWaitForPrimaryStartup(lifecycle.cancel()).then(() => {
    events.push('teardown-joined')
  })
  const reset = removeBootstrapMarkerAfterTeardown(
    teardown,
    () => {
      fs.rmSync(markerPath, { force: true })
      events.push('marker-removed')
    },
    error => assert.fail(error)
  )

  await abortObserved.promise
  assert.equal(fs.existsSync(markerPath), false)

  bootstrapCanExit.resolve()
  await reset

  assert.deepEqual(events, ['stale-marker-written', 'teardown-joined', 'marker-removed'])
  assert.equal(fs.existsSync(markerPath), false)
})

test('bootstrap marker removal errors remain best-effort after teardown', async () => {
  const removalError = new Error('marker is locked')
  let observedError: unknown = null

  await removeBootstrapMarkerAfterTeardown(
    Promise.resolve(),
    () => {
      throw removalError
    },
    error => {
      observedError = error
    }
  )

  assert.equal(observedError, removalError)
})

test('older failure cleanup cannot clobber a newer primary connection, bridge owner, or sidecar', async () => {
  const lifecycle = new PrimaryStartupLifecycle()
  const oldFailure = deferred<void>()
  let currentPromise: Promise<string> | null = null
  let currentChild: { id: string; stopped: boolean } | null = null
  let connection: string | null = null
  let bridgeOwner: string | null = null
  let sidecarOwner: string | null = null

  function start(id: string, phase: Promise<void>, fail: boolean) {
    const generation = lifecycle.begin()
    const child = { id, stopped: false }
    let promise!: Promise<string>

    const operation = (async () => {
      await phase
      lifecycle.assertCurrent(generation)
      if (fail) {
        throw new Error(`${id} failed`)
      }
      connection = id
      bridgeOwner = id
      sidecarOwner = id
      return id
    })()

    promise = operation.catch(error => {
      if (!lifecycle.isCurrent(generation)) {
        child.stopped = true
        if (currentChild === child) {
          currentChild = null
        }
        if (currentPromise === promise) {
          currentPromise = null
        }
        throw new PrimaryStartupCancelledError()
      }
      throw error
    })
    currentChild = child
    currentPromise = promise
    return { child, promise }
  }

  const old = start('old', oldFailure.promise, true)

  // Reset/profile apply detaches the old globals before starting replacement.
  lifecycle.cancel()
  currentPromise = null
  currentChild = null
  bridgeOwner = null
  sidecarOwner = null

  const replacementPhase = Promise.resolve()
  const replacement = start('new', replacementPhase, false)
  assert.equal(await replacement.promise, 'new')

  oldFailure.reject(new Error('old failed late'))
  await assert.rejects(old.promise, PrimaryStartupCancelledError)

  assert.equal(currentPromise, replacement.promise)
  assert.equal(currentChild, replacement.child)
  assert.equal(connection, 'new')
  assert.equal(bridgeOwner, 'new')
  assert.equal(sidecarOwner, 'new')
  assert.equal(replacement.child.stopped, false)
  assert.equal(old.child.stopped, true)
})

test('main wires cancellation and identity guards into every primary lifecycle path', () => {
  const source = fs.readFileSync(path.join(import.meta.dirname, 'main.ts'), 'utf8').replace(/\r\n/g, '\n')
  const startIndex = source.indexOf('async function startHermes()')
  const endIndex = source.indexOf('// Shared navigation guards', startIndex)
  assert.notEqual(startIndex, -1)
  assert.notEqual(endIndex, -1)
  const startBody = source.slice(startIndex, endIndex)

  assert.match(startBody, /const startupGeneration = primaryStartupLifecycle\.begin\(\)/)
  assert.match(startBody, /const teardown = primaryTeardownPromise[\s\S]*?await teardown/)
  assert.match(
    startBody,
    /awaitPrimaryStartupPhase\(startupGeneration, waitForHermes\(remote\.baseUrl, remote\.token\)\)/
  )
  assert.match(
    startBody,
    /primaryStartupLifecycle\.assertCurrent\(startupGeneration\)[\s\S]*?computerUseBridgeLifecycle\.acquire/
  )
  assert.match(startBody, /awaitPrimaryStartupPhase\([\s\S]*?ensureRemoteComputerUseBridge/)
  assert.match(
    startBody,
    /awaitPrimaryStartupPhase\([\s\S]*?ensureRuntime\(resolveHermesBackend\(backendArgs\), startupGeneration\)/
  )
  assert.match(startBody, /awaitPrimaryStartupPhase\([\s\S]*?waitForDashboardPortAnnouncement\(child/)
  assert.match(startBody, /awaitPrimaryStartupPhase\([\s\S]*?adoptServedDashboardToken/)
  assert.match(
    startBody,
    /const ownsPrimaryChild = \(\) =>[\s\S]*?connectionPromise === startupPromise[\s\S]*?hermesProcess === child/
  )
  assert.match(
    startBody,
    /if \(!primaryStartupLifecycle\.isCurrent\(startupGeneration\)\)[\s\S]*?hermesProcess === startupChild[\s\S]*?connectionPromise === startupPromise/
  )
  assert.match(startBody, /primaryStartupLifecycle\.captureStart\(startupGeneration, startupPromise\)/)

  const runtimeIndex = source.indexOf('async function ensureRuntime(')
  const runtimeBody = source.slice(runtimeIndex, source.indexOf('function fetchJson(', runtimeIndex))
  assert.match(
    runtimeBody,
    /primaryStartupLifecycle\.captureBootstrap\([\s\S]*?ownedBootstrapAbortController[\s\S]*?ownedBootstrapPromise/
  )
  assert.match(
    runtimeBody,
    /if \(bootstrapAbortController === ownedBootstrapAbortController\) \{\s*bootstrapAbortController = null/
  )

  const resetIndex = source.indexOf('function resetHermesConnection()')
  const resetBody = source.slice(resetIndex, resetIndex + 700)
  assert.match(resetBody, /primaryStartupLifecycle\.cancel\(\)/)

  const teardownIndex = source.indexOf('function teardownPrimaryBackendAndWait()')
  const teardownBody = source.slice(teardownIndex, teardownIndex + 1200)
  assert.match(teardownBody, /resetHermesConnection\(\)/)
  assert.match(teardownBody, /abortAndWaitForPrimaryStartup\(startup\)/)
  assert.match(teardownBody, /waitForBackendExit\(dying\)/)

  const resetPathIndex = source.indexOf("ipcMain.handle('hermes:bootstrap:reset'")
  assert.notEqual(resetPathIndex, -1)
  assert.match(source.slice(resetPathIndex, resetPathIndex + 1400), /await teardownPrimaryBackendAndWait\(\)/)

  const repairIndex = source.indexOf("ipcMain.handle('hermes:bootstrap:repair'")
  assert.notEqual(repairIndex, -1)
  const repairBody = source.slice(repairIndex, repairIndex + 1600)
  assert.match(
    repairBody,
    /removeBootstrapMarkerAfterTeardown\([\s\S]*?teardownPrimaryBackendAndWait\(\)[\s\S]*?fs\.rmSync\(BOOTSTRAP_COMPLETE_MARKER/
  )

  for (const marker of ["ipcMain.handle('hermes:connection-config:apply'", "ipcMain.handle('hermes:profile:set'"]) {
    const pathIndex = source.indexOf(marker)
    assert.notEqual(pathIndex, -1)
    assert.match(source.slice(pathIndex, pathIndex + 1100), /await teardownPrimaryBackendAndWait\(\)/)
  }

  const quitIndex = source.indexOf("app.on('before-quit'")
  assert.notEqual(quitIndex, -1)
  assert.match(source.slice(quitIndex, quitIndex + 1100), /primaryStartupLifecycle\.stop\(\)/)
})
