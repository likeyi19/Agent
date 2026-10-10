"""Uploaded BAMs use ordinary semantic planning and the existing scientific owners."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from agent.application import InteractiveAgentApplication
from agent.application.uploads import H5ADUploadAdmission
from agent.providers import PlanningModelFactoryRegistry
from agent.report import ANALYSIS_EVIDENCE_FILENAME
from agent.schemas.orchestration import _serialize
from agent.tools.data import bam_fragment_manifest, raw_scatac, raw_scatac_manifest
from agent.tools.data import scatac_fragments_v2, scatac_library_context as library_context
from agent.web.app import create_app
from agent.web.config import ScientificInputSet

from helpers import wait_turn

sys.path.insert(0, str(Path(__file__).parents[1] / 'application'))
from test_registered_bam_integration import (
    DECLARATIONS, INSPECT, PREPARE, PROFILE, bam_factory, instrument, pair,
)


UTTERANCE = 'Inspect this scATAC-seq BAM and report its supported preprocessing readiness.'


class BAMModel:
    """Only provider-authored intent, tool selection and exact semantic edges are scripted."""
    model_id = PROFILE.model_id

    def __init__(self, mode='inspect', callback=None):
        self.mode, self.callback = mode, callback
        self.calls = []

    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        self.calls.append(value)
        if 'turn_schema_version' in value:
            decision = (dict(kind='clarify', reason='unsupported_intent') if self.mode == 'unsupported'
                        else dict(kind='execute_plan', target=INSPECT if self.mode == 'inspect' else PREPARE))
            return json.dumps(dict(turn_schema_version=1, decision=decision))
        if 'selection_schema_version' in response_schema.get('properties', {}):
            return json.dumps(dict(selection_schema_version=1,
                decision=dict(kind='select', capability_ids=['raw_preprocessing'])))
        if 'output_selection_schema_version' in value:
            if self.callback is not None:
                self.callback()
            step = 'inspect' if self.mode == 'inspect' else 'matrix' if self.mode == 'downstream' else 'prepare'
            return json.dumps(dict(outputs=[dict(name='intake' if step == 'inspect' else 'matrix' if step == 'matrix' else 'fragments',
                step_id=step, output_key='manifest_path')]))
        def inp(port, key):
            return dict(target=port, source=dict(kind='input', input=key))
        def upstream(port, step):
            return dict(target=port, source=dict(kind='step', step=step))
        steps = [dict(step_id='inspect', tool=INSPECT, control_dependencies=[],
                      sources=[inp('raw_input', 'raw_input_paths')])]
        if self.mode != 'inspect':
            steps.append(dict(step_id='prepare', tool=PREPARE, control_dependencies=[], sources=[
                inp('source', 'source_path'), upstream('intake', 'inspect'),
                inp('library_context', 'library_context_path'), inp('reference', 'reference_bundle_path'),
                inp('source_profile', 'source_profile')]))
        if self.mode == 'downstream':
            steps.extend([
                dict(step_id='qc', tool='compute_scATAC_qc', control_dependencies=[], sources=[
                    upstream('fragments', 'prepare'), inp('qc_reference', 'qc_reference_manifest_path')]),
                dict(step_id='selection', tool='select_scATAC_cells', control_dependencies=[], sources=[
                    upstream('barcode_qc', 'qc'), inp('min_qc_fragment_records', 'min_qc_fragment_records'),
                    inp('min_tss_enrichment', 'min_tss_enrichment')]),
                dict(step_id='matrix', tool='build_scATAC_cell_by_ccre', control_dependencies=[], sources=[
                    upstream('fragments', 'prepare'), upstream('selected_cells', 'selection'),
                    inp('reference', 'reference_manifest_path')]),
            ])
        offered = value['catalog']['request_inputs']
        for step in steps:
            step['sources'] = [binding for binding in step['sources']
                if binding['source']['kind'] != 'input' or binding['source']['input'] in offered]
        return json.dumps(dict(schema_version=4, decision=dict(kind='plan', steps=steps)))


def web_bam(tmp_path, *, modes=('inspect',), companions=(), registry=None, callback=None):
    models = []
    def factory(profile):
        assert profile == PROFILE
        model = BAMModel(modes[min(len(models), len(modes) - 1)], callback)
        models.append(model)
        return model
    service = InteractiveAgentApplication(tmp_path / 'workspace', model_profiles=(PROFILE,),
        default_profile_id=PROFILE.profile_id, approved_source_roots=(tmp_path,), registry=registry,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': factory}))
    root = service._application._workspace._ensure_directory(tmp_path / 'workspace' / 'uploads')
    uploads = H5ADUploadAdmission(service.resources, root)
    return SimpleNamespace(service=service, models=models, uploads=uploads, input_sets=companions,
        app=create_app(service, uploads=uploads, input_sets=companions))


def upload_bam(client, arguments, *, upload_id='selected-bam', filename='Research BAM.bam'):
    payload = Path(arguments['source_path']).read_bytes()
    response = client.put('/api/v1/uploads/' + upload_id,
        params={'filename': filename, 'input_type': 'bam'}, content=payload,
        headers={'Content-Type': 'application/octet-stream'})
    assert response.status_code == 201, response.text
    resource = response.json()
    assert resource['input_type'] == 'bam' and resource['status'] == 'registered'
    return resource


def qualified_companion(service, resource, arguments, *, corrected=True, input_set_id='qualified'):
    """Explicit synthetic context binds the uploaded locator via the established owner."""
    record = service.resources.load(resource['resource_id'])
    root = Path(arguments['source_path']).parent / ('uploaded-' + input_set_id)
    root.mkdir()
    species = arguments.get('species', 'human')
    declarations = dict(species=species, raw_assay='SCATAC',
                        source_genome_assembly='hg38' if species == 'human' else 'mm10')
    intake = raw_scatac.inspect_raw_scATAC(record.source_path, root / 'intake', **declarations)
    _, manifest, _ = raw_scatac_manifest.load_raw_intake_manifest(intake['manifest_path'])
    context = library_context.build_scatac_library_processing_context(
        intake_manifest_path=intake['manifest_path'], expected_intake_sha256=intake['manifest_sha256'],
        selection_mode=library_context.SelectionMode.ALL_GROUPS,
        libraries=(library_context.LibraryDeclaration(namespace='explicit_library',
            group_ids=(manifest.groups[0].id,), membership_basis=library_context.MembershipBasis.SINGLE_GROUP,
            barcode_interpretation=(library_context.BarcodeInterpretation.CORRECTED_IDENTIFIER if corrected
                                    else library_context.BarcodeInterpretation.RAW_SEQUENCE),
            correction_policy=(library_context.CorrectionPolicy.ALREADY_CORRECTED if corrected
                               else library_context.CorrectionPolicy.UNQUALIFIED_RAW_BAM)),))
    pointer = library_context.publish_scatac_library_processing_context(context, root / 'context.json')
    values = {key: value for key, value in arguments.items()
              if key not in {'source_path', 'source_sha256', 'output_dir', 'intake_manifest_path',
                             'intake_manifest_sha256', 'library_context_path', 'library_context_sha256', 'namespace'}}
    values.update(declarations, library_context_path=pointer['manifest_path'],
                  library_context_sha256=pointer['manifest_sha256'])
    return ScientificInputSet(input_set_id, 'Qualified synthetic paired ATAC context and reference',
                              values, bam_companion=True)


def bam_body(resource, *, turn='analysis', generation=0, companion=None, **changes):
    value = dict(turn_id=turn, expected_generation=generation, utterance=UTTERANCE,
                 resource_id=resource['resource_id'])
    if companion is not None:
        value['input_set_id'] = companion
    return value | changes


def execute(client, body, *, timeout=20):
    response = client.post('/api/v1/sessions/analysis/turns', json=body)
    assert response.status_code == 202, response.text
    return wait_turn(client, body['turn_id'], session='analysis', timeout=timeout)


@pytest.mark.parametrize('case,readiness', [('missing', 'NEEDS_USER_INPUT'), ('valid', 'READY'), ('invalid', 'INVALID')])
def test_uploaded_bam_inspection_does_not_require_producer_context(
        bam_factory, tmp_path, monkeypatch, case, readiness):
    arguments = bam_factory()
    if case == 'invalid':
        Path(arguments['source_path']).write_bytes(b'Not BAM bytes: registration remains a byte transfer.\n')
    companions = (() if case == 'missing' else
        (ScientificInputSet('inspection', 'Human hg38 scATAC declarations', DECLARATIONS, bam_companion=True),))
    registry, calls = instrument(monkeypatch)
    backend = web_bam(tmp_path, companions=companions, registry=registry)
    with TestClient(backend.app) as client:
        resource = upload_bam(client, arguments)
        other = bam_factory(pair(name='different', a={'cb': 'OTHER'}, b={'cb': 'OTHER'}))
        upload_bam(client, other, upload_id='unselected-bam')
        record = backend.service.resources.load(resource['resource_id'])
        assert record.input_type == 'bam' and record.source_sha256 == hashlib.sha256(Path(arguments['source_path']).read_bytes()).hexdigest()
        assert Path(record.source_path).suffix == '.bam' and record.source_index is None
        assert not backend.models and calls == {'intake': 0, 'production': 0, 'verification': 0}
        assert not list(backend.service._application.sessions._store.root.glob('*.json'))
        client.post('/api/v1/sessions', json={'session_id': 'analysis'})
        body = bam_body(resource, companion='inspection' if companions else None)
        view = execute(client, body)
        assert view['status'] == 'succeeded' and view['revision_id'], view
        step, = backend.service._application.run_store.load(view['run_id']).steps
        assert step.tool_name == INSPECT and _serialize(step.resolved_arguments['raw_input_paths']) == [record.source_path]
        assert step.result['readiness'] == readiness and step.verification.passed
        assert step.verification.artifact_authority is None
        _, intake, _ = raw_scatac_manifest.load_raw_intake_manifest(step.result['manifest_path'])
        assert [item.path for item in intake.files] == [record.source_path]
        assert calls == {'intake': 1, 'production': 0, 'verification': 0}
        assert len(backend.models) == 1 and len(backend.models[0].calls) == 4
        path = '/api/v1/sessions/analysis/revisions/' + view['revision_id']
        assert client.get(path + '/evidence', params={'output_name': 'intake'}).json()['status'] == 'available'
        assert client.get(path + '/scientific-artifacts').json() == {'artifacts': []}
        assert str(tmp_path) not in json.dumps(view)
        assert execute(client, body) == view
        assert calls == {'intake': 1, 'production': 0, 'verification': 0}
        state = backend.service._application.sessions.load('analysis')
        assert state.generation == 1 and len(state.revisions) == 1


@pytest.mark.parametrize('species', ['human', 'mouse'])
def test_uploaded_qualified_bam_fresh_intake_handoff_and_multiturn_exact_source(
        bam_factory, tmp_path, monkeypatch, species):
    arguments = bam_factory(species=species)
    arguments['species'] = species
    registry, calls = instrument(monkeypatch)
    backend = web_bam(tmp_path, modes=('prepare', 'inspect'), registry=registry)
    with TestClient(backend.app) as client:
        resource = upload_bam(client, arguments)
    companion = qualified_companion(backend.service, resource, arguments)
    backend.input_sets = (companion,)
    app = create_app(backend.service, uploads=backend.uploads, input_sets=backend.input_sets)
    with TestClient(app) as client:
        client.post('/api/v1/sessions', json={'session_id': 'analysis'})
        body = bam_body(resource, companion='qualified', utterance='Prepare fragments from this BAM using the qualified paired ATAC context.')
        view = execute(client, body)
        assert view['status'] == 'succeeded' and view['revision_id'], view
        record = backend.service.resources.load(resource['resource_id'])
        inspected, prepared = backend.service._application.run_store.load(view['run_id']).steps
        assert prepared.resolved_arguments['source_path'] == record.source_path
        assert prepared.resolved_arguments['source_sha256'] == record.source_sha256
        assert prepared.resolved_arguments['intake_manifest_path'] == inspected.result['manifest_path']
        assert prepared.resolved_arguments['intake_manifest_sha256'] == inspected.result['manifest_sha256']
        for key in ('library_context_path', 'library_context_sha256', 'reference_bundle_path', 'reference_bundle_sha256', 'source_profile'):
            assert prepared.resolved_arguments[key] == companion.inputs()[key]
        assert prepared.resolved_arguments['source_profile'] == 'agent-cb-paired-atac.v1'
        authority = prepared.verification.artifact_authority
        assert authority['schema_version'] == 2 and authority['verifier']['id'] == 'agent.bam-fragments-independent'
        assert authority['producer_qualification']['scope'] == 'bam_read_pair_transformation.v1'
        assert {'path': record.source_path, 'sha256': record.source_sha256,
                'size_bytes': record.size_bytes} in authority['historical_sources']
        fragments = scatac_fragments_v2.load_fragments_manifest_v2(prepared.result['manifest_path'],
            expected_sha256=prepared.result['manifest_sha256'])
        producer = fragments['libraries'][0]['provenance']['producer_record']
        provenance = bam_fragment_manifest.load_record(producer['path'], producer['sha256'])
        assert provenance['source_history'] == bam_fragment_manifest.HISTORY
        assert provenance['source']['path'] == record.source_path and provenance['source']['sha256'] == record.source_sha256
        assert prepared.result['n_fragment_records'] == prepared.result['eligible_pairs'] == 1
        evidence = backend.service._application._workspace.run_paths(view['run_id']).evidence / ANALYSIS_EVIDENCE_FILENAME
        facts = json.loads(evidence.read_bytes())['steps'][-1]['facts']
        assert facts['route'] == 'bam_fragment_production' and facts['bam_transformation_verification'] == 'independently_recomputed'
        path = '/api/v1/sessions/analysis/revisions/' + view['revision_id']
        assert client.get(path + '/evidence', params={'output_name': 'fragments'}).json()['status'] == 'available'
        report, = [a for a in client.get(path + '/artifacts').json()['artifacts'] if a['artifact_type'] == 'analysis_report']
        delivered = client.get(path + '/artifacts/' + report['handle'], params={'download': True})
        assert delivered.status_code == 200 and str(tmp_path) not in delivered.text
        assert client.get(path + '/scientific-artifacts').json() == {'artifacts': []}
        before = dict(calls)
        assert execute(client, body) == view and calls == before
        followup = execute(client, bam_body(resource, turn='inspect-again', generation=1, companion='qualified'))
        assert followup['status'] == 'succeeded'
        state = backend.service._application.sessions.load('analysis')
        assert state.generation == 2 and len(state.revisions) == 2
        assert state.revisions[-1].parent_revision_id == view['revision_id']
        assert calls == {'intake': 2, 'production': 1, 'verification': 1}
        assert len(backend.models) == 2


@pytest.mark.parametrize('failure,code', [('correction', 'BAM_FRAGMENTS_BARCODE_POLICY_UNSUPPORTED'),
                                        ('context', 'BAM_FRAGMENTS_CONTEXT_MISMATCH'),
                                        ('reference', 'REFERENCE_DIGEST_MISMATCH')])
def test_uploaded_bam_preserves_qualified_owner_failures(bam_factory, tmp_path, monkeypatch, failure, code):
    arguments = bam_factory()
    registry, calls = instrument(monkeypatch)
    backend = web_bam(tmp_path, modes=('prepare',), registry=registry)
    with TestClient(backend.app) as client:
        resource = upload_bam(client, arguments)
    companion = qualified_companion(backend.service, resource, arguments, corrected=failure != 'correction')
    values = companion.inputs()
    if failure == 'context':
        values.update({key: arguments[key] for key in ('library_context_path', 'library_context_sha256')})
    if failure == 'reference':
        values['reference_bundle_sha256'] = '0' * 64
    companion = ScientificInputSet('qualified', companion.label, values, bam_companion=True)
    with TestClient(create_app(backend.service, uploads=backend.uploads, input_sets=(companion,))) as client:
        client.post('/api/v1/sessions', json={'session_id': 'analysis'})
        body = bam_body(resource, companion='qualified', utterance='Prepare fragments from this BAM using the supplied qualified context.')
        view = execute(client, body)
        assert view['status'] == 'failed' and view['revision_id'] is None, view
        run = backend.service._application.run_store.load(view['run_id'])
        assert view['error']['code'] == run.errors[0].code == code
        assert str(tmp_path) not in json.dumps(view)
        state = backend.service._application.sessions.load('analysis')
        assert state.generation == 0 and not state.revisions
        before = dict(calls)
        assert execute(client, body) == view and calls == before
        assert calls['intake'] == 1 and calls['verification'] == 0


def test_bam_upload_does_not_route_unsupported_intent_or_start_science(bam_factory, tmp_path, monkeypatch):
    arguments = bam_factory()
    registry, calls = instrument(monkeypatch)
    backend = web_bam(tmp_path, modes=('unsupported',), registry=registry)
    with TestClient(backend.app) as client:
        resource = upload_bam(client, arguments)
        client.post('/api/v1/sessions', json={'session_id': 'analysis'})
        view = execute(client, bam_body(resource, utterance='Correct the barcodes with an unsupported method.'))
        assert view['status'] == 'clarification' and view['revision_id'] is None
        assert view['response']['kind'] == 'clarify'
        assert len(backend.models) == 1 and len(backend.models[0].calls) == 1
        assert calls == {'intake': 0, 'production': 0, 'verification': 0}
        state = backend.service._application.sessions.load('analysis')
        assert state.generation == 0 and not state.revisions and state.turn('analysis').run_id is None


def test_bam_restart_completed_retry_reads_history_but_new_consumption_checks_bytes(
        bam_factory, tmp_path, monkeypatch):
    arguments = bam_factory()
    registry, calls = instrument(monkeypatch)
    backend = web_bam(tmp_path, registry=registry)
    with TestClient(backend.app) as client:
        resource = upload_bam(client, arguments)
        client.post('/api/v1/sessions', json={'session_id': 'analysis'})
        body = bam_body(resource)
        accepted = execute(client, body)
    initial = backend.service._application.sessions.load('analysis')
    Path(backend.service.resources.load(resource['resource_id']).source_path).write_bytes(b'Changed registered source.\n')
    def forbidden(*args, **kwargs):
        raise AssertionError('Completed retries must not construct providers or execute science.')
    fresh = InteractiveAgentApplication(tmp_path / 'workspace', model_profiles=(PROFILE,),
        default_profile_id=PROFILE.profile_id, approved_source_roots=(tmp_path,), registry=registry,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': forbidden}))
    uploads = H5ADUploadAdmission(fresh.resources, backend.uploads.root)
    with TestClient(create_app(fresh, uploads=uploads)) as client:
        assert any({key: choice[key] for key in resource} == resource
                   for choice in client.get('/api/v1/resources').json()['choices'])
        assert client.get('/api/v1/sessions/analysis').status_code == 200
        assert execute(client, body) == accepted
        new = client.post('/api/v1/sessions/analysis/turns',
            json=bam_body(resource, turn='new-consumption', generation=1))
        assert new.status_code == 400 and new.json()['error']['code'] == 'LOCAL_RESOURCE_INTEGRITY_INVALID'
        assert str(tmp_path) not in new.text
    assert fresh._application.sessions.load('analysis') == initial
    assert calls == {'intake': 1, 'production': 0, 'verification': 0} and len(backend.models) == 1


@pytest.mark.parametrize('missing,code,partial', [
    ('library_context_path', 'BAM_LIBRARY_CONTEXT_REQUIRED', False),
    ('reference_bundle_path', 'BAM_REFERENCE_REQUIRED', False),
    ('source_profile', 'BAM_PROFILE_REQUIRED', False),
    ('library_context_path', 'UNAUTHORIZED_REQUEST_INPUT', True),
    ('reference_bundle_path', 'UNAUTHORIZED_REQUEST_INPUT', True),
])
def test_bam_producer_missing_typed_prerequisite_has_safe_operation_specific_feedback(
        bam_factory, tmp_path, monkeypatch, missing, code, partial):
    arguments = bam_factory()
    registry, calls = instrument(monkeypatch)
    backend = web_bam(tmp_path, modes=('prepare',), registry=registry)
    with TestClient(backend.app) as client:
        resource = upload_bam(client, arguments)
    companion = qualified_companion(backend.service, resource, arguments)
    values = companion.inputs()
    values.pop(missing)
    if missing.endswith('_path') and not partial:
        values.pop(missing[:-4] + 'sha256')
    companion = ScientificInputSet('incomplete', 'Incomplete approved BAM context', values, bam_companion=True)
    with TestClient(create_app(backend.service, uploads=backend.uploads, input_sets=(companion,))) as client:
        client.post('/api/v1/sessions', json={'session_id': 'analysis'})
        body = bam_body(resource, companion='incomplete', utterance='Prepare fragments from this selected BAM using the supplied context.')
        view = execute(client, body)
        assert view['status'] == 'failed' and view['revision_id'] is None, view
        assert view['error']['code'] == code
        assert str(tmp_path) not in json.dumps(view) and 'sha256' not in view['error']['message']
        assert calls == {'intake': 0, 'production': 0, 'verification': 0}
        state = backend.service._application.sessions.load('analysis')
        assert state.generation == 0 and not state.revisions
        before = len(backend.models)
        assert execute(client, body) == view and len(backend.models) == before


@pytest.mark.parametrize('during_provider', [False, True])
def test_bam_changed_source_blocks_new_scientific_consumption(
        bam_factory, tmp_path, monkeypatch, during_provider):
    arguments = bam_factory()
    registry, calls = instrument(monkeypatch)
    selected = []
    def changed():
        selected[0].write_bytes(b'Changed during interpretation of the registered BAM.\n')
    backend = web_bam(tmp_path, registry=registry, callback=changed if during_provider else None)
    with TestClient(backend.app) as client:
        resource = upload_bam(client, arguments)
        selected.append(Path(backend.service.resources.load(resource['resource_id']).source_path))
        client.post('/api/v1/sessions', json={'session_id': 'analysis'})
        if during_provider:
            view = execute(client, bam_body(resource))
            assert view['status'] == 'failed' and view['error']['code'] == 'LOCAL_RESOURCE_INTEGRITY_INVALID', view
        else:
            changed()
            response = client.post('/api/v1/sessions/analysis/turns', json=bam_body(resource))
            assert response.status_code == 400 and response.json()['error']['code'] == 'LOCAL_RESOURCE_INTEGRITY_INVALID'
            assert not backend.models
        assert calls == {'intake': 0, 'production': 0, 'verification': 0}
        state = backend.service._application.sessions.load('analysis')
        assert state.generation == 0 and not state.revisions


def downstream_bam_inputs(tmp_path):
    """An explicitly constructed synthetic BAM with the accepted tiny QC design."""
    import pysam
    from test_external_fragments_execution import downstream_inputs
    arguments = downstream_inputs(tmp_path)
    path = Path(arguments['source_path']).with_name('paired-atac.bam')
    header = {'HD': {'VN': '1.6', 'SO': 'unsorted'},
              'SQ': [{'SN': name, 'LN': 6001} for name in ('chr2', 'chr1')]}
    records = (pair('central', a={'pos': 2950, 'mpos': 3030, 'tlen': 100},
                    b={'pos': 3030, 'mpos': 2950, 'tlen': -100}) +
               pair('flank', a={'pos': 1000, 'mpos': 1040, 'tlen': 60},
                    b={'pos': 1040, 'mpos': 1000, 'tlen': -60}))
    with pysam.AlignmentFile(str(path), 'wb', header=header) as output:
        for value in records:
            row = pysam.AlignedSegment(output.header)
            row.query_name, row.flag = value['name'], value['flag']
            row.reference_id, row.reference_start = value['rid'], value['pos']
            row.cigarstring, row.mapping_quality = value['cigar'], value['mapq']
            row.next_reference_id, row.next_reference_start = value['mrid'], value['mpos']
            row.template_length, row.query_sequence = value['tlen'], 'A' * 20
            row.set_tag('CB', value['cb'])
            output.write(row)
    arguments.update(source_path=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     source_profile='agent-cb-paired-atac.v1')
    return arguments


def test_bam_provider_plan_downstream_handoff_preserves_all_existing_downloads(tmp_path, monkeypatch):
    from rich_helpers import ScientificWork
    arguments = downstream_bam_inputs(tmp_path)
    registry, calls = instrument(monkeypatch)
    backend = web_bam(tmp_path, modes=('downstream',), registry=registry)
    with TestClient(backend.app) as client:
        resource = upload_bam(client, arguments)
    companion = qualified_companion(backend.service, resource, arguments)
    monkeypatch.setenv('AGENT_QC_BEDTOOLS', '/usr/bin/bedtools')
    monkeypatch.setenv('AGENT_MATRIX_BEDTOOLS', '/usr/bin/bedtools')
    monkeypatch.setenv('AGENT_QC_ALLOW_SYNTHETIC', '1')
    monkeypatch.delenv('AGENT_QC_RESOURCE_CATALOG', raising=False)
    work = ScientificWork()
    try:
        with TestClient(create_app(backend.service, uploads=backend.uploads, input_sets=(companion,))) as client:
            client.post('/api/v1/sessions', json={'session_id': 'analysis'})
            body = bam_body(resource, companion='qualified', utterance='Use this BAM to prepare fragments, perform QC, select cells using supplied thresholds, and build a cell-by-cCRE matrix.')
            view = execute(client, body, timeout=60)
            assert view['status'] == 'succeeded', view
            run = backend.service._application.run_store.load(view['run_id'])
            assert [step.tool_name for step in run.steps] == [INSPECT, PREPARE, 'compute_scATAC_qc', 'select_scATAC_cells', 'build_scATAC_cell_by_ccre']
            assert run.steps[1].resolved_arguments['intake_manifest_sha256'] == run.steps[0].result['manifest_sha256']
            assert run.steps[2].resolved_arguments['fragments_manifest_sha256'] == run.steps[1].result['manifest_sha256']
            assert run.steps[3].resolved_arguments['barcode_qc_manifest_sha256'] == run.steps[2].result['manifest_sha256']
            assert run.steps[4].resolved_arguments['selected_cells_manifest_sha256'] == run.steps[3].result['manifest_sha256']
            before = work.snapshot()
            for name in ('qc', 'selection', 'matrix'):
                assert before['production.' + name] == before['owner.' + name] == 1
            assert calls == {'intake': 1, 'production': 1, 'verification': 1}
            artifacts = client.get('/api/v1/sessions/analysis/turns/analysis/scientific-artifacts').json()['artifacts']
            assert {item['filename'] for item in artifacts} == {'matrix.h5ad', 'barcodes.tsv.gz', 'lengths.tsv.gz', 'decisions.tsv.gz', 'selected.tsv.gz'}
            path = '/api/v1/sessions/analysis/revisions/' + view['revision_id']
            for item in artifacts:
                response = client.get(path + '/scientific-artifacts/' + item['handle'])
                assert response.status_code == 200 and len(response.content) == item['size_bytes']
                assert hashlib.sha256(response.content).hexdigest() == item['sha256']
            assert execute(client, body) == view and work.snapshot() == before
            assert calls == {'intake': 1, 'production': 1, 'verification': 1}
            state = backend.service._application.sessions.load('analysis')
            assert state.generation == 1 and len(state.revisions) == 1
            assert {item.name for item in state.revisions[0].outputs} == {'matrix'}
    finally:
        work.close()
