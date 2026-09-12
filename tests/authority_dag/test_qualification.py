"""Explicit trust transition over tiny persisted artifacts, never production replay."""
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from agent.orchestration import AgentRuntime, AgentRequest, FileRunStore
from agent.orchestration.verification_authority import (
    accepted_authorities, load_fragment_authority, qualify_legacy_authorities,
)
from agent.schemas.verification_authority import AuthorityError
from agent.tools.data.authority_context import VerificationContext, current
from agent.tools.data.fragments_authority_contract import CONTRACTS
from test_propagation import chain, completed, Planner


class LegacyStore(FileRunStore):
    """Persist fixture execution without the authority field introduced in .2."""
    def update(self, state, *, expected_revision):
        steps = tuple(replace(s, verification=replace(s.verification, artifact_authority=None))
                      if s.verification else s for s in state.steps)
        return super().update(replace(state, steps=steps), expected_revision=expected_revision)


@pytest.fixture
def legacy(chain):
    kind, args, plan, original = chain
    store = LegacyStore(original.root)
    run = AgentRuntime(planner=Planner(plan), run_store=store).run(
        AgentRequest('authority-request', 'Tiny legacy DAG.', {}))
    assert run.status.value == 'SUCCEEDED', run
    state = store.load(run.run_id)
    assert all(s.verification.artifact_authority is None for s in state.steps)
    return kind, args, state, store


@pytest.fixture
def calls(monkeypatch):
    counts = Counter()
    original = VerificationContext.verify
    def verify(self, kind, function, args, kwargs):
        def deep(*a, **kw):
            counts[kind] += 1
            return function(*a, **kw)
        return original(self, kind, deep, args, kwargs)
    monkeypatch.setattr(VerificationContext, 'verify', verify)
    from agent.orchestration.executor import PlanExecutor
    def forbidden(*a, **kw):
        pytest.fail('Qualification must not execute production')
    monkeypatch.setattr(PlanExecutor, 'execute', forbidden)
    return counts


def sidecar(store, run_id):
    return store.state_path(run_id).with_suffix('.authority.json')


def test_current_authority_is_reused_without_creating_sidecar(completed, calls):
    _, args, run, store = completed
    state = store.load(run.run_id)
    source = Path(args['source_path'])
    source.rename(source.with_suffix('.archived'))
    authorities = qualify_legacy_authorities(store, run.run_id)
    assert all(a.record == state.steps[i].verification.artifact_authority
               for i, a in enumerate(authorities.values()))
    assert not calls
    assert not sidecar(store, run.run_id).exists()
    assert store.load(run.run_id) == state


def test_explicit_qualification_then_reuse_is_idempotent_and_nonmutating(legacy, calls, tmp_path):
    kind, args, state, store = legacy
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in tmp_path.rglob('*') if p.is_file()}
    with pytest.raises(AuthorityError):
        load_fragment_authority(store, state.run_id, 'fragments')
    with accepted_authorities(store, state.run_id):
        pass
    assert not sidecar(store, state.run_id).exists()
    authorities = qualify_legacy_authorities(store, state.run_id)
    assert calls == {kind: 1, 'generic_fragments': 1, 'qc': 1, 'selection': 1, 'matrix': 1}
    assert all(a.record['schema_version'] == 2 for a in authorities.values())
    assert authorities['fragments'].record['producer_qualification']['kind'] == kind
    for path, sha in before.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
    assert store.load(state.run_id) == state
    payload = sidecar(store, state.run_id).read_bytes()
    counts = dict(calls)
    assert qualify_legacy_authorities(store, state.run_id) == authorities
    assert sidecar(store, state.run_id).read_bytes() == payload
    assert dict(calls) == counts
    source = Path(args['source_path'])
    source.rename(source.with_suffix('.archived'))
    from agent.tools.data.scatac_matrix import verify_public_result
    with accepted_authorities(store, state.run_id):
        step = state.steps[-1]
        verify_public_result(step.resolved_arguments, step.result)
    assert dict(calls) == counts
    handle = load_fragment_authority(store, state.run_id, 'fragments')
    for other in CONTRACTS:
        if other != kind:
            with pytest.raises(AuthorityError):
                handle.validate(state.steps[0].result['manifest_path'],
                                state.steps[0].result['manifest_sha256'], producer_kind=other)
    with pytest.raises((AuthorityError, OSError)):
        with accepted_authorities(store, state.run_id, source_policy='current_source_freshness.v1'):
            pass
    assert current() is None


@pytest.mark.parametrize('mutation', ['matrix', 'receipt', 'reference', 'source', 'qc', 'selection'])
def test_corrupt_legacy_science_never_publishes_partial_qualification(legacy, mutation):
    _, args, state, store = legacy
    matrix = state.steps[-1].result
    paths = dict(matrix=Path(matrix['matrix_path']),
        receipt=Path(matrix['manifest_path']).parent.parent/'receipt.json',
        reference=Path(args['reference_bundle_path']), source=Path(args['source_path']),
        qc=Path(state.steps[1].result['manifest_path']),
        selection=Path(state.steps[2].result['manifest_path']))
    path = paths[mutation]
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises((ValueError, OSError)):
        qualify_legacy_authorities(store, state.run_id)
    assert not sidecar(store, state.run_id).exists()
    assert store.load(state.run_id) == state
    assert current() is None


@pytest.mark.parametrize('boundary', ['owner', 'publication'])
def test_interruption_discards_all_pending_authorities(legacy, monkeypatch, boundary):
    _, _, state, store = legacy
    if boundary == 'owner':
        from agent.orchestration import verifier
        original = verifier.verify_step
        def interrupted(step, *a, **kw):
            if step.step_id == 'matrix':
                raise KeyboardInterrupt()
            return original(step, *a, **kw)
        monkeypatch.setattr(verifier, 'verify_step', interrupted)
    else:
        import os
        original = os.replace
        def interrupted(source, destination):
            if Path(destination) == sidecar(store, state.run_id):
                raise KeyboardInterrupt()
            return original(source, destination)
        monkeypatch.setattr(os, 'replace', interrupted)
    with pytest.raises(KeyboardInterrupt):
        qualify_legacy_authorities(store, state.run_id)
    assert not sidecar(store, state.run_id).exists()
    assert not list(store.root.glob('*.tmp'))
    assert store.load(state.run_id) == state
    assert current() is None


@pytest.mark.parametrize('field', ['scope', 'verifier', 'upstream', 'science_profile', 'producer_qualification'])
def test_corrupt_qualified_authority_cannot_be_requalified_or_reused(legacy, field):
    _, _, state, store = legacy
    qualify_legacy_authorities(store, state.run_id)
    path = sidecar(store, state.run_id)
    value = json.loads(path.read_bytes())
    a = value['record']['authorities']['fragments']
    changes = dict(scope='presentation_validation.v1', verifier={'id':'wrong','compatibility_version':'99'},
        upstream={'wrong':'0'*64}, science_profile='0'*64, producer_qualification={})
    a[field] = changes[field]
    from agent.orchestration.run_store import _record_digest
    value['integrity']['digest'] = _record_digest(value['record'])
    path.write_text(json.dumps(value))
    for operation in (lambda: qualify_legacy_authorities(store, state.run_id),
                      lambda: load_fragment_authority(store, state.run_id, 'fragments')):
        with pytest.raises(ValueError):
            operation()


@pytest.mark.parametrize('fault', ['policy', 'verification', 'arguments', 'execution'])
def test_missing_or_mismatched_execution_anchor_fails_closed(legacy, monkeypatch, fault):
    _, _, state, store = legacy
    if fault == 'policy':
        from types import SimpleNamespace
        changed = SimpleNamespace(source_schema_version=3, lifecycle_status=state.lifecycle_status,
                                  plan=state.plan, recovery_policy_snapshot=None)
    else:
        step = state.steps[0]
        if fault == 'verification':
            from types import SimpleNamespace
            step = replace(step, verification=None)
            changed = SimpleNamespace(**vars(state))
            changed.steps = (step, *state.steps[1:])
        elif fault == 'arguments':
            step = replace(step, resolved_arguments=dict(step.resolved_arguments, output_dir='/tmp/wrong-output'))
            changed = replace(state, steps=(step, *state.steps[1:]))
        else:
            from agent.schemas import fingerprint_plan
            plan = replace(state.plan, planner_name='different execution')
            changed = replace(state, plan=plan, plan_fingerprint=fingerprint_plan(plan))
    monkeypatch.setattr(store, 'load', lambda _: changed)
    with pytest.raises((ValueError, RuntimeError)):
        qualify_legacy_authorities(store, state.run_id)
    assert not sidecar(store, state.run_id).exists()


def test_failed_publication_after_write_is_not_reusable(legacy, monkeypatch):
    _, _, state, store = legacy
    original = store._write_record_atomic
    def failure(*a, **kw):
        original(*a, **kw)
        raise OSError('publication acknowledgement failed')
    monkeypatch.setattr(store, '_write_record_atomic', failure)
    with pytest.raises(OSError):
        qualify_legacy_authorities(store, state.run_id)
    assert not sidecar(store, state.run_id).exists()
    assert store.load(state.run_id) == state


def test_qualification_sidecar_checks_bytes_and_exact_execution(legacy):
    _, _, state, store = legacy
    qualify_legacy_authorities(store, state.run_id)
    path = sidecar(store, state.run_id)
    original = path.read_bytes()
    for raw in (b'{}', original.replace(b'"digest":"', b'"digest":"0', 1)):
        path.write_bytes(raw)
        with pytest.raises(RuntimeError):
            with accepted_authorities(store, state.run_id):
                pass
    value = json.loads(original)
    value['record']['source_state_sha256'] = '0'*64
    from agent.orchestration.run_store import _record_digest
    value['integrity']['digest'] = _record_digest(value['record'])
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError):
        qualify_legacy_authorities(store, state.run_id)


def test_cancellation_before_publication_discards_pending_proofs(legacy, monkeypatch):
    _, _, state, store = legacy
    from agent.tools._cancellation import cancellation_scope, ToolWorkCancelled
    publish = store._publish_qualified_authorities
    def cancel(*a, **kw):
        with cancellation_scope(lambda: True):
            publish(*a, **kw)
    monkeypatch.setattr(store, '_publish_qualified_authorities', cancel)
    with pytest.raises(ToolWorkCancelled):
        qualify_legacy_authorities(store, state.run_id)
    assert not sidecar(store, state.run_id).exists()
    assert not list(store.root.glob('*.tmp'))
    assert current() is None


@pytest.mark.parametrize('fault', ['missing_source', 'backend'])
def test_required_producer_evidence_and_runtime_cannot_be_skipped(legacy, monkeypatch, fault):
    _, args, state, store = legacy
    if fault == 'missing_source':
        path = Path(args['source_path'])
        path.rename(path.with_suffix('.archived'))
    else:
        monkeypatch.setenv('AGENT_MATRIX_BEDTOOLS', '/nonexistent/bedtools')
    with pytest.raises((ValueError, OSError)):
        qualify_legacy_authorities(store, state.run_id)
    assert not sidecar(store, state.run_id).exists()
    assert store.load(state.run_id) == state


def test_cancellation_callback_can_read_run_store_during_qualification(legacy):
    import fcntl
    import os
    from agent.tools._cancellation import cancellation_scope
    _, _, state, store = legacy
    checked = []
    def cancellation_check():
        # Nonblocking probe turns a recursive-lock regression into an immediate
        # failure rather than hanging this test or an operator qualification.
        descriptor = os.open(store.lock_path(state.run_id), os.O_RDONLY)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        finally:
            os.close(descriptor)
        checked.append(True)
        return store.load_cancellation(state.run_id) is not None
    with cancellation_scope(cancellation_check):
        qualify_legacy_authorities(store, state.run_id)
    assert checked and sidecar(store, state.run_id).exists()
    assert store.load(state.run_id) == state
