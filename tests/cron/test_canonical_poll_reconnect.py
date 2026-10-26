"""The remote canonical firer must observe one fire through an owner restart's transport drop.

A draining owner cuts non-waiting /api/ws clients (close 1013) the moment draining starts,
while its deferred stop keeps the admitted turn alive across the drain — so a healthy fire
can outlive its observer's socket. Dropping to ``unknown`` there amputated a successful
updater run (worker exit 1, no durable terminal state) on every update that restarted the
gateway mid-turn. The status poll must reconnect and finish the observation instead.
"""
from contextlib import asynccontextmanager

import pytest


def _install(connect_script, monkeypatch):
    """One list entry per connect_gateway() attempt: 'refuse' (connect raises) or a
    (op, action) step list the yielded peer consumes in order ('drop' = transport loss)."""
    from hermes_cli import gateway_client

    class ScriptedPeer:
        def __init__(self, script):
            self.script = list(script)

        async def rpc(self, method, **params):
            op = method.split('.', 1)[1]
            step = self.script.pop(0)
            assert step[0] == op, (op, step)
            action = step[1]
            if action == 'drop':
                raise gateway_client.GatewayClientError(
                    'Gateway disconnected; turn outcome is unknown.')
            return action

    connect_attempts = []

    @asynccontextmanager
    async def scripted():
        connect_attempts.append('connect')
        step = connect_script.pop(0)
        if step == 'refuse':
            raise gateway_client.GatewayClientError('draining: socket refused')
        yield ScriptedPeer(step)

    monkeypatch.setattr(gateway_client, 'connect_gateway', scripted)
    return connect_attempts


def test_status_poll_reconnects_across_owner_drain(monkeypatch):
    from cron.scheduler_authority import run_canonical_job

    _install([
        [('submit', {'session_id': 'cron-fire', 'admission_id': 'admit'}),
         ('status', 'drop')],   # admitted, then the drain cuts the poller's socket
        'refuse',               # reconnect hits the drain gate...
        'refuse',               # ...twice
        [('status', {'status': 'terminal',
                     'result': [True, 'document', 'answer', None]})],
    ], monkeypatch)
    result = run_canonical_job({'id': 'job'}, execution_id='fire')
    assert result == (True, 'document', 'answer', None)


def test_submit_answer_loss_stays_fail_closed(monkeypatch):
    from cron.scheduler_authority import run_canonical_job, CronExecutionUnknown

    _install([[('submit', 'drop')]], monkeypatch)
    with pytest.raises(CronExecutionUnknown):
        run_canonical_job({'id': 'job'}, execution_id='fire')


def test_reconnect_budget_expiry_reports_unknown(monkeypatch):
    from cron import scheduler_authority
    from cron.scheduler_authority import run_canonical_job, CronExecutionUnknown

    monkeypatch.setattr(scheduler_authority, 'RECONNECT_BUDGET_SECONDS', 0.0)
    _install([
        [('submit', {'session_id': 'cron-fire', 'admission_id': 'admit'})],
        [('status', 'drop')],
    ], monkeypatch)
    with pytest.raises(CronExecutionUnknown):
        run_canonical_job({'id': 'job'}, execution_id='fire')
