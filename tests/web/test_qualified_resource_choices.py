"""Configured choices are captured separately from scientific qualification."""
from dataclasses import replace
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from agent.application import InteractiveBoundaryError
from agent.application.local_resources import LocalResourceAdmission, ResourceAdmissionError
from agent.application.uploads import H5ADUploadAdmission
from agent.orchestration import (AgentRequest, PlanExecutor, SemanticPlanCandidate,
    SemanticPlanStep, SemanticRequestInputSource, build_m92_semantic_compiler_contract,
    compile_semantic_plan)
from agent.schemas.orchestration import _serialize
from agent.tools.data import _bam_fragment_io, bam_fragment_manifest, raw_scatac
from agent.web.app import (_bam_selection, _fastq_selection, _fragments_selection,
    _input_set_choice, create_app)
from agent.web.config import ScientificInputSet

from helpers import harness, submission, wait_turn
from test_bam_execution import (bam_factory, bam_body, qualified_companion, upload_bam,
    web_bam)


def registered_h5ad(tmp_path):
    backend = harness(tmp_path)
    root = backend.service._application._workspace._ensure_directory(tmp_path / 'workspace' / 'uploads')
    backend.service.resources = LocalResourceAdmission(backend.service._application._workspace,
        approved_source_roots=(tmp_path,), registry=backend.service._application.registry)
    uploads = H5ADUploadAdmission(backend.service.resources, root)
    with TestClient(create_app(backend.service, uploads=uploads)) as client:
        response = client.put('/api/v1/uploads/selected', params={'filename': 'Selected cells.h5ad'},
            content=backend.source.read_bytes(), headers={'Content-Type': 'application/octet-stream'})
        assert response.status_code == 201, response.text
    record = backend.service.resources.load(response.json()['resource_id'])
    return backend, uploads, record


def matrix_choice(identifier='first', **changes):
    return ScientificInputSet(identifier, 'Approved ' + identifier + ' declarations',
        dict(species='human') | changes, h5ad_companion=True)


def test_identical_values_from_distinct_choices_conflict_and_history_does_not_reopen_source(tmp_path):
    backend, uploads, record = registered_h5ad(tmp_path)
    first, second = matrix_choice(), matrix_choice('second')
    assert first.inputs() == second.inputs() and first.resource_choice() != second.resource_choice()
    body = submission(resource_id=record.resource_id, input_set_id=first.input_set_id)
    with TestClient(create_app(backend.service, uploads=uploads, input_sets=(first, second))) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        public = client.get('/api/v1/input-sets')
        assert public.json() == {'choices': [first.choice(), second.choice()]}
        assert 'definition_sha256' not in public.text and str(tmp_path) not in public.text
        response = client.post('/api/v1/sessions/session/turns', json=body)
        assert response.status_code == 202, response.text
        accepted = wait_turn(client)
        assert accepted['status'] == 'succeeded' and accepted['revision_id'], accepted
        interaction, = backend.service._application.sessions.load('session').interactions
        assert _serialize(interaction.submission['registered_input']['resource_choice']) == first.resource_choice()
        assert 'resource_choice' not in interaction.submission['execution_inputs']
        assert all('resource_choice' not in prompt and 'definition_sha256' not in json.dumps(prompt)
                   for model in backend.models for prompt in model.calls)
        counts = (len(backend.models), list(backend.science_calls))
        altered = dict(body, input_set_id=second.input_set_id)
        denied = client.post('/api/v1/sessions/session/turns', json=altered)
        assert denied.status_code == 409 and denied.json()['error']['code'] == 'INTERACTIVE_TURN_CONFLICT'
        Path(record.source_path).unlink()
        assert client.post('/api/v1/sessions/session/turns', json=body).status_code == 202
        assert wait_turn(client) == accepted
        assert (len(backend.models), backend.science_calls) == counts
        assert str(tmp_path) not in json.dumps(accepted) and 'definition_sha256' not in json.dumps(accepted)


def test_changed_definition_under_same_choice_id_conflicts_before_replay(tmp_path):
    backend, uploads, record = registered_h5ad(tmp_path)
    selected = matrix_choice()
    body = submission(resource_id=record.resource_id, input_set_id=selected.input_set_id)
    with TestClient(create_app(backend.service, uploads=uploads, input_sets=(selected,))) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        assert client.post('/api/v1/sessions/session/turns', json=body).status_code == 202
        assert wait_turn(client)['status'] == 'succeeded'
    counts = (len(backend.models), list(backend.science_calls))
    changed = replace(selected, label='Changed approved-choice definition')
    assert changed.inputs() == selected.inputs()
    with TestClient(create_app(backend.service, uploads=uploads, input_sets=(changed,))) as client:
        denied = client.post('/api/v1/sessions/session/turns', json=body)
        assert denied.status_code == 409 and denied.json()['error']['code'] == 'INTERACTIVE_TURN_CONFLICT'
    assert (len(backend.models), backend.science_calls) == counts


@pytest.mark.parametrize('family,selection', [
    ('fragments', _fragments_selection), ('bam', _bam_selection), ('fastq', _fastq_selection),
])
def test_capture_uses_exact_explicit_or_unique_configured_default(family, selection):
    inputs = dict(reference_bundle_path='/private/reference.json', reference_bundle_sha256='a' * 64,
                  source_profile='10x-atac-fragments.v1', namespace='declared') if family == 'fragments' else {}
    selected = ScientificInputSet('approved', 'Approved configured candidate', inputs,
        **{family + '_companion': True, family + '_default': True})
    other = replace(selected, input_set_id='other', label='Other configured candidate')
    assert selection(None, (selected,))
    assert _input_set_choice(None, (selected,), family) == selected.resource_choice()
    assert _input_set_choice(selected, (selected, other), family) == selected.resource_choice()
    assert _input_set_choice(None, (selected, other), family) is None
    assert _input_set_choice(None, (replace(selected, **{family + '_default': False}),), family) is None
    if family == 'fragments':
        with pytest.raises(ResourceAdmissionError) as denied:
            selection(None, (selected, other))
        assert denied.value.code == 'FRAGMENTS_RESOURCE_AMBIGUOUS'
    else:
        assert selection(None, (selected, other)) == ({}, family.upper() + '_RESOURCE_AMBIGUOUS')
    with pytest.raises(InteractiveBoundaryError):
        selection(matrix_choice(), (selected,))


def test_bam_default_and_explicit_selection_capture_the_same_choice(tmp_path, bam_factory):
    arguments = bam_factory()
    selected = ScientificInputSet('inspection', 'Explicit BAM inspection declarations',
        dict(species='human', raw_assay='SCATAC', source_genome_assembly='hg38'),
        bam_companion=True, bam_default=True)
    backend = web_bam(tmp_path, companions=(selected,))
    with TestClient(backend.app) as client:
        resource = upload_bam(client, arguments)
        client.post('/api/v1/sessions', json={'session_id': 'analysis'})
        body = bam_body(resource)
        assert client.post('/api/v1/sessions/analysis/turns', json=body).status_code == 202
        accepted = wait_turn(client, 'analysis', session='analysis')
        assert accepted['status'] == 'succeeded', accepted
        captured = backend.service._application.sessions.load('analysis').interactions[0].submission
        assert _serialize(captured['registered_input']['resource_choice']) == selected.resource_choice()
        completions = sum(len(model.calls) for model in backend.models)
        explicit = dict(body, input_set_id=selected.input_set_id)
        assert client.post('/api/v1/sessions/analysis/turns', json=explicit).status_code == 202
        assert wait_turn(client, 'analysis', session='analysis') == accepted
        assert sum(len(model.calls) for model in backend.models) == completions


def test_bam_existing_owner_rejects_old_context_for_new_upload_with_identical_bytes(tmp_path, bam_factory):
    """Second family witness: compilation/admission, no BAM producer execution."""
    arguments = bam_factory()
    backend = web_bam(tmp_path)
    with TestClient(backend.app) as client:
        first = upload_bam(client, arguments, upload_id='first')
        second = upload_bam(client, arguments, upload_id='second')
    records = [backend.service.resources.load(source['resource_id']) for source in (first, second)]
    assert records[0].source_sha256 == records[1].source_sha256
    assert records[0].source_path != records[1].source_path
    selected = qualified_companion(backend.service, first, arguments)
    context_root = Path(selected.inputs()['library_context_path']).parent
    from agent.tools.data import scatac_library_context
    _, context, _ = scatac_library_context.load_scatac_library_processing_context(
        selected.inputs()['library_context_path'], expected_sha256=selected.inputs()['library_context_sha256'])
    inputs = selected.inputs() | dict(intake_manifest_path=context.intake.manifest_path,
                                     intake_manifest_sha256=context.intake.manifest_sha256)
    selected = replace(selected, execution_inputs=inputs)
    binding = backend.uploads.resolve(first['resource_id'], scientific_inputs=inputs,
        resource_choice=selected.resource_choice())
    registry = backend.service._application.registry
    candidate = SemanticPlanCandidate((SemanticPlanStep('prepare', 'prepare_scATAC_bam_fragments',
        sources=tuple(SemanticRequestInputSource(port, name) for port, name in (
            ('source', 'source_path'), ('intake', 'intake_manifest_path'),
            ('library_context', 'library_context_path'), ('reference', 'reference_bundle_path'),
            ('source_profile', 'source_profile'), ('output_dir', 'output_dir')))),))
    request = AgentRequest('bound-bam', 'Prepare this selected BAM with the selected approved context.',
        _serialize(binding.execution_inputs) | dict(output_dir=str(context_root / 'withheld')))
    compiled = compile_semantic_plan(request, candidate, registry, build_m92_semantic_compiler_contract(registry))
    plan = compiled
    assert PlanExecutor(registry).preflight(plan).passed
    submission_value = dict(registered_input=binding.attribution(), execution_inputs=_serialize(binding.execution_inputs))
    backend.service.resources.validate_plan(submission_value, plan)
    owner_binding = _bam_fragment_io.bind(dict(plan.steps[0].arguments) | dict(output_dir=str(context_root / 'withheld')))
    assert owner_binding['source']['path'] == records[0].source_path
    altered_inputs = inputs | dict(source_path=records[1].source_path, source_sha256=records[1].source_sha256)
    with pytest.raises(ResourceAdmissionError) as denied:
        backend.service.resources.resolve(second['resource_id'], tool_name='prepare_scATAC_bam_fragments',
            scientific_inputs={key: value for key, value in altered_inputs.items()
                               if key not in {'source_path', 'source_sha256'}})
    assert denied.value.code == 'LOCAL_RESOURCE_BINDING_INVALID'
    new_intake = raw_scatac.inspect_raw_scATAC(records[1].source_path, str(context_root / 'new-intake'),
        species='human', raw_assay='SCATAC', source_genome_assembly='hg38')
    wrong_arguments = dict(plan.steps[0].arguments) | dict(source_path=records[1].source_path,
        source_sha256=records[1].source_sha256, intake_manifest_path=new_intake['manifest_path'],
        intake_manifest_sha256=new_intake['manifest_sha256'], output_dir=str(context_root / 'withheld-new'))
    with pytest.raises(bam_fragment_manifest.BamFragmentsError) as denied:
        _bam_fragment_io.bind(wrong_arguments)
    assert denied.value.code == 'BAM_FRAGMENTS_CONTEXT_MISMATCH'
    assert not backend.models and not list(backend.service._application.run_store.root.iterdir())
    witness = dict(kind='bam-binding-contract-only', resource_choice=selected.resource_choice(),
        sources=[dict(resource_id=record.resource_id, record_sha256=record.record_sha256,
                      source_sha256=record.source_sha256, source_path=record.source_path) for record in records],
        compiled_arguments=_serialize(plan.steps[0].arguments), preflight_passed=True,
        valid_owner_binding=True, new_upload_old_context_rejection=denied.value.code,
        scientific_producer_executions=0, accepted_runs=0, accepted_revisions=0)
    (tmp_path / 'bam-resource-binding-witness.json').write_text(json.dumps(witness, indent=2))
