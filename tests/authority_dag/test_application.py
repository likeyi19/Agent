"""Tiny real artifacts exercise the application authority boundary, not producers at scale."""
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import sys

import pytest

from agent.application import ResearchAgentApplication
from agent.orchestration import AgentRequest
from agent.report import (
    ANALYSIS_REPORT_MANIFEST_FILENAME, verify_analysis_evidence, verify_analysis_report,
)
from agent.tools.data.authority_context import current
from test_propagation import chain, Planner


@contextmanager
def owner_calls():
    # Observe the actual undecorated bodies, including strict no-context calls.
    from agent.tools.data import (
        barcode_qc_verifier, cell_selection_verifier, cell_by_ccre_verifier,
        bam_fragments_verifier, external_fragments_verifier,
        fastq_fragments_verifier, scatac_fragments_v2_verifier,
    )
    functions = {
        'qc': barcode_qc_verifier.verify_barcode_qc,
        'selection': cell_selection_verifier.verify_cell_selection,
        'matrix': cell_by_ccre_verifier.verify_cell_by_ccre,
        'bam_fragment_production': bam_fragments_verifier.verify_bam_fragments,
        'external_fragment_adoption': external_fragments_verifier.verify_external_fragments,
        'fastq_fragment_production': fastq_fragments_verifier.verify_fragments,
        'generic_fragments': scatac_fragments_v2_verifier.verify_fragments_v2,
    }
    codes = {f.__wrapped__.__code__: kind for kind, f in functions.items()}
    counts = Counter()
    previous = sys.getprofile()
    def profile(frame, event, arg):
        if event == 'call' and frame.f_code in codes:
            counts[codes[frame.f_code]] += 1
    sys.setprofile(profile)
    try:
        yield counts
    finally:
        sys.setprofile(previous)


@pytest.fixture
def application_chain(chain, tmp_path):
    kind, args, plan, _ = chain
    app = ResearchAgentApplication(tmp_path / 'application', planner=Planner(plan))
    return kind, args, app


def run_application(app):
    result = app.run(AgentRequest('authority-request', 'Tiny DAG.', {}))
    assert result.status.value == 'SUCCEEDED', result
    assert result.evidence and result.report and result.visualization is None
    assert current() is None
    return result


def test_application_reuses_science_and_retains_standalone_deep_verification(application_chain):
    kind, args, app = application_chain
    with owner_calls() as counts:
        result = run_application(app)
    assert counts == {kind: 1, 'generic_fragments': 1, 'qc': 1, 'selection': 1, 'matrix': 1}
    state = app.run_store.load(result.run_id)
    evidence_bytes = Path(result.evidence.path).read_bytes()
    report_bytes = Path(result.report.path).read_bytes()
    assert b'fresh_independent_' not in evidence_bytes

    # Historical qualification is not a claim about present raw-source bytes.
    source = Path(args['source_path'])
    archived = source.with_suffix('.archived')
    source.rename(archived)
    with owner_calls() as counts:
        resumed = app.resume(result.run_id)
    assert resumed == result
    assert not counts
    assert app.run_store.load(result.run_id) == state
    assert Path(result.evidence.path).read_bytes() == evidence_bytes
    assert Path(result.report.path).read_bytes() == report_bytes
    assert current() is None

    # A deserialized result alone grants no authority to standalone reporting.
    with owner_calls() as counts:
        verification = verify_analysis_evidence(
            result.run_result, result.evidence.path, registry=app.registry)
    assert not verification.passed
    assert counts[kind] > 0
    archived.rename(source)
    with owner_calls() as counts:
        verification = verify_analysis_report(
            result.run_result, result.evidence.path,
            Path(result.report.path).parent / ANALYSIS_REPORT_MANIFEST_FILENAME,
            registry=app.registry)
    assert verification.passed, verification
    assert all(counts[name] > 0 for name in (kind, 'generic_fragments', 'qc', 'selection', 'matrix'))


@pytest.mark.parametrize('mutation', ['payload', 'receipt', 'resource', 'authority', 'evidence', 'report'])
def test_application_rejects_corruption_without_reconstructing_science(application_chain, monkeypatch, mutation):
    _, args, app = application_chain
    result = run_application(app)
    state = app.run_store.load(result.run_id)
    matrix = Path(result.run_result.steps[-1].result['manifest_path'])
    if mutation == 'authority':
        step = state.steps[-1]
        authority = dict(step.verification.artifact_authority)
        authority['verifier'] = {'id': 'unsupported', 'compatibility_version': '99'}
        changed = replace(state, steps=(*state.steps[:-1], replace(
            step, verification=replace(step.verification, artifact_authority=authority))))
        monkeypatch.setattr(app.run_store, 'load', lambda _: changed)
    else:
        paths = {
            'payload': Path(result.run_result.steps[-1].result['matrix_path']),
            'receipt': matrix.parent.parent / 'receipt.json',
            'resource': Path(args['reference_bundle_path']),
            'evidence': Path(result.evidence.path),
            'report': Path(result.report.path),
        }
        target = paths[mutation]
        target.write_bytes(target.read_bytes() + b'\n')
    with owner_calls() as counts:
        resumed = app.resume(result.run_id)
    assert resumed.status.value == 'FAILED'
    assert resumed.run_status.value == 'SUCCEEDED'
    assert resumed.error.code == ('APP_REPORT_FAILED' if mutation == 'report' else 'APP_EVIDENCE_FAILED')
    assert resumed.report is None
    assert not counts
    assert current() is None


def test_missing_authority_takes_deep_path_without_persisting_upgrade(application_chain, monkeypatch):
    kind, _, app = application_chain
    # Remove authorities only in the loader's view of accepted persistence.
    # The real store and source result remain unchanged.
    original_load = app.run_store.load
    from agent.application import service
    original_loader = service.accepted_authorities
    @contextmanager
    def legacy_loader(store, run_id):
        class LegacyStore:
            def load(self, key):
                state = original_load(key)
                return replace(state, steps=tuple(replace(s, verification=replace(
                    s.verification, artifact_authority=None)) for s in state.steps))
        with original_loader(LegacyStore(), run_id) as context:
            yield context
    monkeypatch.setattr(service, 'accepted_authorities', legacy_loader)
    with owner_calls() as counts:
        result = run_application(app)
    # One owner pass during execution and one independent fallback in composition.
    assert counts == {kind: 2, 'generic_fragments': 2, 'qc': 2, 'selection': 2, 'matrix': 2}
    state = original_load(result.run_id)
    with owner_calls() as counts:
        assert app.resume(result.run_id) == result
    assert counts == {kind: 1, 'generic_fragments': 1, 'qc': 1, 'selection': 1, 'matrix': 1}
    assert original_load(result.run_id) == state


@pytest.mark.parametrize('error', [ValueError('private authority error'), KeyboardInterrupt()])
def test_authority_scope_exit_failure_cannot_return_success(application_chain, monkeypatch, error):
    _, _, app = application_chain
    from agent.application import service
    original = service.accepted_authorities
    @contextmanager
    def failed_exit(*args, **kwargs):
        with original(*args, **kwargs) as context:
            yield context
            raise error
    monkeypatch.setattr(service, 'accepted_authorities', failed_exit)
    if isinstance(error, KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            app.run(AgentRequest('authority-request', 'Tiny DAG.', {}))
    else:
        result = app.run(AgentRequest('authority-request', 'Tiny DAG.', {}))
        assert result.status.value == 'FAILED'
        assert result.error.code == 'APP_EVIDENCE_FAILED'
        assert result.report is None
        assert 'private authority error' not in str(result.to_dict())
    assert current() is None
