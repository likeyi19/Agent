"""Shared resource choices over genuine, explicitly synthetic-only owner fixtures."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

from fastapi.testclient import TestClient
import pytest

from agent.application import InteractiveAgentApplication
from agent.application.local_resources import ResourceAdmissionError
from agent.application.uploads import H5ADUploadAdmission
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas.orchestration import AgentPlan, PlanStep, StepOutputRef, _serialize
from agent.web.app import create_app
from agent.web.config import ScientificInputSet

from test_external_fragments_execution import (
    DownstreamModel, PROFILE, downstream_inputs, fragment_body,
    fragment_companion, upload_fragments, web_fragments,
)
from helpers import wait_turn
from rich_helpers import ScientificWork


@pytest.fixture
def synthetic_runtime(monkeypatch):
    """Existing owner-approved tiny synthetic scope, never production attestation."""
    monkeypatch.setenv('AGENT_QC_BEDTOOLS', '/usr/bin/bedtools')
    monkeypatch.setenv('AGENT_MATRIX_BEDTOOLS', '/usr/bin/bedtools')
    monkeypatch.setenv('AGENT_QC_ALLOW_SYNTHETIC', '1')
    monkeypatch.delenv('AGENT_QC_RESOURCE_CATALOG', raising=False)


def execute(client, body, *, timeout=90):
    response = client.post('/api/v1/sessions/analysis/turns', json=body)
    assert response.status_code == 202, response.text
    return wait_turn(client, body['turn_id'], session='analysis', timeout=timeout)


@pytest.mark.parametrize('default', [False, True])
def test_qualified_synthetic_owner_chain_captures_exact_explicit_or_default_choice(
        tmp_path, synthetic_runtime, default):
    """Four real owners with established synthetic qualification, not lab biology."""
    arguments = downstream_inputs(tmp_path)
    choice = fragment_companion(arguments, input_set_id='reviewed', default=default)
    backend = web_fragments(tmp_path, arguments, companions=(choice,), model_type=DownstreamModel)
    work = ScientificWork()
    try:
        with TestClient(backend.app) as client:
            selected = upload_fragments(client, arguments)
            assert work.snapshot() == {} and backend.models == []
            assert client.post('/api/v1/sessions', json={'session_id': 'analysis'}).status_code == 201
            body = fragment_body(selected, companion=None if default else 'reviewed',
                utterance='Import these external fragments, compute QC, select candidates using '
                'the explicit supplied thresholds, and construct the cell-by-cCRE matrix.')
            view = execute(client, body)
            assert view['status'] == 'succeeded', (view, backend.errors)
            run = backend.service._application.run_store.load(view['run_id'])
            state = backend.service._application.sessions.load('analysis')
            captured = _serialize(state.interactions[0].submission)
            assert captured['registered_input']['resource_choice'] == choice.resource_choice()
            source = backend.service.resources.load(selected['resource_id'])
            assert captured['execution_inputs']['source_path'] == source.source_path
            assert captured['execution_inputs']['source_sha256'] == source.source_sha256
            assert [s.tool_name for s in run.steps] == ['import_scATAC_fragments',
                'compute_scATAC_qc', 'select_scATAC_cells', 'build_scATAC_cell_by_ccre']
            assert run.steps[1].result['resource_qualification'] == 'synthetic_only'
            for kind in ('fragments', 'qc', 'selection', 'matrix'):
                assert work.snapshot()['production.' + kind] == 1
                assert work.snapshot()['owner.' + kind] == 1
            assert all(s.verification.artifact_authority['schema_version'] == 2 for s in run.steps)
            assert run.steps[0].resolved_arguments['reference_bundle_sha256'] == arguments['reference_bundle_sha256']
            assert run.steps[1].resolved_arguments['qc_reference_manifest_sha256'] == arguments['qc_reference_manifest_sha256']
            assert run.steps[3].resolved_arguments['reference_manifest_sha256'] == arguments['reference_manifest_sha256']
            path = '/api/v1/sessions/analysis/revisions/' + view['revision_id']
            evidence = client.get(path + '/evidence', params={'output_name': 'matrix'})
            assert evidence.status_code == 200 and evidence.json()['status'] == 'available'
            delivered = []
            for artifact in client.get('/api/v1/sessions/analysis/turns/import/scientific-artifacts').json()['artifacts']:
                payload = client.get(path + '/scientific-artifacts/' + artifact['handle'])
                assert payload.status_code == 200
                assert hashlib.sha256(payload.content).hexdigest() == artifact['sha256']
                delivered.append(artifact['filename'])
            assert set(delivered) == {'matrix.h5ad', 'barcodes.tsv.gz', 'lengths.tsv.gz',
                'decisions.tsv.gz', 'selected.tsv.gz'}
            before = work.snapshot()
            assert execute(client, body) == view and work.snapshot() == before
            identical_values = replace(choice, input_set_id='another-choice')
            switched = create_app(backend.service, uploads=backend.uploads,
                input_sets=(choice, identical_values))
            with TestClient(switched) as changed_client:
                retry = changed_client.post('/api/v1/sessions/analysis/turns',
                    json=body | {'input_set_id': 'another-choice'})
                assert retry.status_code == 409
            assert work.snapshot() == before
            assert len(backend.models) == 1 and len(backend.models[0].calls) == 4
        Path(source.source_path).unlink()
        def forbidden(profile):
            raise AssertionError('Completed resource handoff reconstructed a provider.')
        fresh = InteractiveAgentApplication(tmp_path / 'workspace', model_profiles=(PROFILE,),
            default_profile_id=PROFILE.profile_id, approved_source_roots=(tmp_path,),
            planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': forbidden}))
        fresh_uploads = H5ADUploadAdmission(fresh.resources, backend.uploads.root)
        with TestClient(create_app(fresh, uploads=fresh_uploads, input_sets=(choice,))) as client:
            assert client.get('/api/v1/sessions/analysis').status_code == 200
            assert execute(client, body) == view
            assert client.get(path + '/evidence', params={'output_name': 'matrix'}).status_code == 200
        assert work.snapshot() == before
        assert fresh._application.sessions.load('analysis') == state
        (tmp_path / 'synthetic-qualified-handoff-witness.json').write_text(json.dumps(dict(
            scope='explicitly synthetic-only owner qualification; not biological or operator production qualification',
            resource_choice=choice.resource_choice(), captured_submission=captured,
            run_id=view['run_id'], revision_id=view['revision_id'], invocation_counts=before,
            prompt_calls=len(backend.models[0].calls), delivered=delivered,
            fresh_instance_science_delta={}), indent=2) + '\n')
    finally:
        work.close()


@pytest.mark.parametrize('field', ['source_path', 'source_sha256', 'reference_bundle_path',
    'reference_bundle_sha256', 'source_profile', 'namespace'])
def test_actual_compiled_source_resource_arguments_cannot_substitute_captured_choice(tmp_path, field):
    arguments = downstream_inputs(tmp_path)
    choice = fragment_companion(arguments, input_set_id='reviewed')
    backend = web_fragments(tmp_path, arguments, companions=(choice,))
    registered = backend.service.resources.register('selected', arguments['source_path'],
        input_type='external_fragments', label='Explicit fixture source', attribution='Synthetic test source')
    binding = backend.service.resources.compose_fragments(registered.resource_id,
        choice.inputs(), resource_choice=choice.resource_choice())
    captured = dict(registered_input=binding.attribution(),
        execution_inputs=_serialize(binding.execution_inputs))
    values = _serialize(binding.execution_inputs)
    keys = ('source_path', 'source_sha256', 'reference_bundle_path',
        'reference_bundle_sha256', 'source_profile', 'namespace')
    selected = {key: values[key] for key in keys}
    good = AgentPlan('plan', 'request', 'fixture', (PlanStep('import',
        'import_scATAC_fragments', selected),))
    backend.service.resources.validate_plan(captured, good)
    changed = replace(good, steps=(replace(good.steps[0], arguments=selected | {field:'substituted'}),))
    with pytest.raises(ResourceAdmissionError) as failed:
        backend.service.resources.validate_plan(captured, changed)
    assert failed.value.code == 'FRAGMENTS_SOURCE_MISMATCH'
    assert not backend.models and not tuple(backend.service._application.run_store.root.glob('*.json'))


def test_source_mutation_and_unsupported_profile_fail_without_successful_revision(tmp_path):
    arguments = downstream_inputs(tmp_path)
    choice = fragment_companion(arguments, input_set_id='reviewed')
    backend = web_fragments(tmp_path, arguments, companions=(choice,))
    with TestClient(backend.app) as client:
        selected = upload_fragments(client, arguments)
        client.post('/api/v1/sessions', json={'session_id': 'analysis'})
        source = backend.service.resources.load(selected['resource_id'])
        Path(source.source_path).write_bytes(b'replaced registered source')
        response = client.post('/api/v1/sessions/analysis/turns',
            json=fragment_body(selected, companion='reviewed'))
        assert response.status_code == 400
        assert response.json()['error']['code'] == 'LOCAL_RESOURCE_INTEGRITY_INVALID'
        assert not backend.models
        assert not backend.service._application.sessions.load('analysis').revisions
    with pytest.raises(ResourceAdmissionError) as failed:
        backend.service.resources.compose_fragments(source.resource_id,
            choice.inputs() | {'source_profile':'unsupported-fragments-history'}, verify_source=False,
            resource_choice=choice.resource_choice())
    assert failed.value.code == 'LOCAL_RESOURCE_BINDING_INVALID'


def test_configured_choice_does_not_promote_unqualified_synthetic_qc_resource(
        tmp_path, synthetic_runtime, monkeypatch):
    """Configured identity is captured; genuine QC owner still owns qualification."""
    arguments = downstream_inputs(tmp_path)
    choice = fragment_companion(arguments, input_set_id='reviewed')
    backend = web_fragments(tmp_path, arguments, companions=(choice,), model_type=DownstreamModel)
    monkeypatch.delenv('AGENT_QC_ALLOW_SYNTHETIC')
    work = ScientificWork()
    try:
        with TestClient(backend.app) as client:
            selected = upload_fragments(client, arguments)
            client.post('/api/v1/sessions', json={'session_id': 'analysis'})
            view = execute(client, fragment_body(selected, companion='reviewed',
                utterance='Import the selected fragments and compute barcode QC.'))
        assert view['status'] == 'failed' and view['revision_id'] is None
        assert view['error']['code'] == 'QC_RESOURCE_UNQUALIFIED'
        state = backend.service._application.sessions.load('analysis')
        assert state.generation == 0 and not state.revisions
        assert state.interactions[0].submission['registered_input']['resource_choice'] == choice.resource_choice()
        assert work.snapshot()['production.fragments'] == work.snapshot()['owner.fragments'] == 1
        assert work.snapshot().get('production.qc', 0) == 0
        assert work.snapshot().get('owner.qc', 0) == 0
        assert work.snapshot().get('production.selection', 0) == 0
        assert work.snapshot().get('production.matrix', 0) == 0
    finally:
        work.close()


@pytest.mark.parametrize('tool,field', [
    ('compute_scATAC_qc', 'qc_reference_manifest_path'),
    ('compute_scATAC_qc', 'qc_reference_manifest_sha256'),
    ('build_scATAC_cell_by_ccre', 'reference_manifest_path'),
    ('build_scATAC_cell_by_ccre', 'reference_manifest_sha256'),
])
def test_captured_choice_guards_exact_downstream_registered_resource_members(tmp_path, tool, field):
    arguments = downstream_inputs(tmp_path)
    choice = fragment_companion(arguments, input_set_id='reviewed')
    backend = web_fragments(tmp_path, arguments, companions=(choice,))
    registered = backend.service.resources.register('selected', arguments['source_path'],
        input_type='external_fragments', label='Explicit fixture source', attribution='Synthetic test source')
    binding = backend.service.resources.compose_fragments(registered.resource_id,
        choice.inputs(), resource_choice=choice.resource_choice())
    captured = dict(registered_input=binding.attribution(),
        execution_inputs=_serialize(binding.execution_inputs))
    values = _serialize(binding.execution_inputs)
    pair = lambda step, prefix: {prefix + suffix: StepOutputRef(step, 'manifest_' + suffix)
        for suffix in ('path', 'sha256')}
    steps = (
        PlanStep('import', 'import_scATAC_fragments', {key: values[key] for key in
            ('source_path', 'source_sha256', 'reference_bundle_path',
                'reference_bundle_sha256', 'source_profile', 'namespace')}),
        PlanStep('qc', 'compute_scATAC_qc', pair('import', 'fragments_manifest_') | {
            key: values[key] for key in ('qc_reference_manifest_path', 'qc_reference_manifest_sha256')},
            depends_on=('import',)),
        PlanStep('selection', 'select_scATAC_cells', pair('qc', 'barcode_qc_manifest_') | {
            key: values[key] for key in ('min_qc_fragment_records', 'min_tss_enrichment')},
            depends_on=('qc',)),
        PlanStep('matrix', 'build_scATAC_cell_by_ccre', pair('import', 'fragments_manifest_') |
            pair('selection', 'selected_cells_manifest_') | {key: values[key] for key in
                ('reference_manifest_path', 'reference_manifest_sha256')},
            depends_on=('import', 'selection')),
    )
    plan = AgentPlan('plan', 'request', 'fixture', steps)
    backend.service.resources.validate_plan(captured, plan)
    substituted = replace(plan, steps=tuple(replace(step,
        arguments=dict(step.arguments) | {field: 'substituted'}) if step.tool_name == tool
        else step for step in steps))
    with pytest.raises(ResourceAdmissionError) as failed:
        backend.service.resources.validate_plan(captured, substituted)
    assert failed.value.code == 'LOCAL_RESOURCE_CONSUMPTION_MISMATCH'
    assert not backend.models and not tuple(backend.service._application.run_store.root.glob('*.json'))
