"""Uploaded fragments enter the ordinary interpreted, independently verified path."""
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
from agent.tools.data import external_fragment_manifest, scatac_fragments_v2
from agent.web.app import create_app
from agent.web.config import ScientificInputSet

from helpers import wait_turn

sys.path.insert(0, str(Path(__file__).parents[1] / 'application'))
from test_registered_resource_integration import (
    FRAGMENTS_UTTERANCE, IMPORT_FRAGMENTS, PROFILE, ScriptedModel,
    count_fragment_science, fragment_source_factory, fragments_service,
)


def fragment_companion(arguments, *, input_set_id='declared', default=False, **changes):
    values = {key: value for key, value in arguments.items()
              if key not in {'source_path', 'source_sha256', 'source_index_path',
                             'source_index_sha256', 'output_dir'}}
    return ScientificInputSet(input_set_id, 'Declared fragments reference and library',
        values | changes, fragments_companion=True, fragments_default=default)


def web_fragments(tmp_path, arguments, *, companions=None, source_index=False, model_type=None):
    """Real tiny owners, server-approved reference, and the existing upload transport."""
    if model_type is None:
        service, models = fragments_service(tmp_path, source_index=source_index)
    else:
        models = []
        def factory(profile):
            model = model_type(IMPORT_FRAGMENTS, source_index=source_index)
            models.append(model)
            return model
        service = InteractiveAgentApplication(tmp_path / 'workspace', model_profiles=(PROFILE,),
            default_profile_id=PROFILE.profile_id,
            planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': factory}),
            approved_source_roots=(tmp_path,))
    root = service._application._workspace._ensure_directory(tmp_path / 'workspace' / 'uploads')
    errors = []
    original_submit = service.submit_turn
    def observed_submit(*args, **kwargs):
        try:
            return original_submit(*args, **kwargs)
        except BaseException:
            import traceback
            errors.append(traceback.format_exc())
            raise
    service.submit_turn = observed_submit
    uploads = H5ADUploadAdmission(service.resources, root)
    companions = ((fragment_companion(arguments),) if companions is None else companions)
    app = create_app(service, uploads=uploads, input_sets=companions)
    return SimpleNamespace(service=service, models=models, uploads=uploads,
                           input_sets=companions, app=app, errors=errors)


def upload_fragments(client, arguments, *, upload_id='selected-source', filename='Fragments.tsv', index_bytes=None):
    source = Path(arguments['source_path']).read_bytes()
    params = {'filename': filename, 'input_type': 'external_fragments'}
    if 'source_index_path' in arguments:
        index_bytes = Path(arguments['source_index_path']).read_bytes() if index_bytes is None else index_bytes
        params.update(index_filename=filename + '.tbi', source_size=str(len(source)))
        source += index_bytes
    response = client.put('/api/v1/uploads/' + upload_id,
        params=params, content=source,
        headers={'Content-Type': 'application/octet-stream'})
    assert response.status_code == 201, response.text
    value = response.json()
    assert value['input_type'] == 'external_fragments' and value['status'] == 'registered'
    return value


@pytest.mark.parametrize('different_source_index', [False, True])
def test_explicit_uploaded_bgzf_tbi_pair_keeps_exact_identity_and_owner_checks_compatibility(
        fragment_source_factory, tmp_path, monkeypatch, different_source_index):
    arguments = fragment_source_factory(encoding='bgzf', indexed=True)
    supplied_index = Path(arguments['source_index_path']).read_bytes()
    if different_source_index:
        other = fragment_source_factory(encoding='bgzf', indexed=True,
                                        rows=[b'chr2\t10\t30\tOTHER\t9\n'])
        supplied_index = Path(other['source_index_path']).read_bytes()
    backend = web_fragments(tmp_path, arguments, source_index=True)
    calls = count_fragment_science(monkeypatch)
    with TestClient(backend.app) as client:
        resource = upload_fragments(client, arguments, filename='Fragments.tsv.gz', index_bytes=supplied_index)
        record = backend.service.resources.load(resource['resource_id'])
        assert record.source_sha256 == arguments['source_sha256']
        assert record.source_index['sha256'] == hashlib.sha256(supplied_index).hexdigest()
        assert record.source_index['size_bytes'] == len(supplied_index)
        assert Path(record.source_index['path']).read_bytes() == supplied_index
        assert not backend.models and calls == {'production': 0, 'verification': 0}
        client.post('/api/v1/sessions', json={'session_id': 'analysis'})
        body = fragment_body(resource)
        view = execute(client, body)
        run = backend.service._application.run_store.load(view['run_id'])
        step, = run.steps
        assert step.resolved_arguments['source_index_path'] == record.source_index['path']
        assert step.resolved_arguments['source_index_sha256'] == record.source_index['sha256']
        if different_source_index:
            assert view['status'] == 'failed' and view['revision_id'] is None
            assert view['error']['code'] == run.errors[0].code == 'EXTERNAL_FRAGMENTS_INDEX_MISMATCH'
            assert calls == {'production': 1, 'verification': 0}
        else:
            assert view['status'] == 'succeeded' and view['revision_id']
            manifest = scatac_fragments_v2.load_fragments_manifest_v2(step.result['manifest_path'],
                expected_sha256=step.result['manifest_sha256'])
            producer_binding = manifest['libraries'][0]['provenance']['producer_record']
            adoption = external_fragment_manifest.load_adoption_record(producer_binding['path'], producer_binding['sha256'])
            assert adoption['source_index']['sha256'] == record.source_index['sha256']
            assert calls == {'production': 1, 'verification': 1}
        assert execute(client, body) == view


def fragment_body(resource, *, turn='import', generation=0, companion='declared', **changes):
    body = dict(turn_id=turn, expected_generation=generation,
                utterance=FRAGMENTS_UTTERANCE, resource_id=resource['resource_id'])
    if companion is not None:
        body['input_set_id'] = companion
    return body | changes


def execute(client, body):
    response = client.post('/api/v1/sessions/analysis/turns', json=body)
    assert response.status_code == 202, response.text
    return wait_turn(client, body['turn_id'], session='analysis')


@pytest.mark.parametrize('encoding,species', [('plain', 'human'), ('gzip', 'mouse'), ('bgzf', 'human')])
def test_uploaded_selected_source_imports_exact_bytes_and_publishes_accepted_history(
        fragment_source_factory, tmp_path, monkeypatch, encoding, species):
    arguments = fragment_source_factory(encoding=encoding, species=species, selection='subset_export', strand=True)
    backend = web_fragments(tmp_path, arguments)
    calls = count_fragment_science(monkeypatch)
    with TestClient(backend.app) as client:
        resource = upload_fragments(client, arguments, filename='Research fragments.tsv' + ('.gz' if encoding != 'plain' else ''))
        other = fragment_source_factory(rows=[b'chr1\t20\t40\tUNSELECTED\t9\n'])
        upload_fragments(client, other, upload_id='neighbor')
        assert not backend.models and calls == {'production': 0, 'verification': 0}
        assert not list(backend.service._application.sessions._store.root.glob('*.json'))
        client.post('/api/v1/sessions', json={'session_id': 'analysis'})
        body = fragment_body(resource)
        view = execute(client, body)
        assert view['status'] == 'succeeded' and view['response']['status'] == 'activated', (view, backend.errors)
        assert calls == {'production': 1, 'verification': 1}
        assert len(backend.models) == 1 and len(backend.models[0].calls) == 4
        state = backend.service._application.sessions.load('analysis')
        assert state.generation == 1 and len(state.revisions) == 1
        revision = state.revisions[0]
        assert revision.revision_id == view['revision_id']
        assert [(o.name, o.step_id, o.output_key) for o in revision.outputs] == [('fragments', 'import', 'manifest_path')]
        record = backend.service.resources.load(resource['resource_id'])
        assert Path(record.source_path).read_bytes() == Path(arguments['source_path']).read_bytes()
        assert record.source_sha256 == arguments['source_sha256']
        captured = state.interactions[0].submission
        assert captured['registered_input']['resource_id'] == resource['resource_id']
        inputs = _serialize(captured['execution_inputs'])
        assert inputs['source_path'] == record.source_path and inputs['source_sha256'] == record.source_sha256
        assert inputs['namespace'] == arguments['namespace']
        run = backend.service._application.run_store.load(view['run_id'])
        step, = run.steps
        assert step.tool_name == IMPORT_FRAGMENTS
        assert step.resolved_arguments['source_path'] == record.source_path
        assert step.resolved_arguments['source_sha256'] == record.source_sha256
        assert (step.result['n_fragment_records'], step.result['total_support']) == (2, 557)
        authority = step.verification.artifact_authority
        assert authority['schema_version'] == 2 and authority['verifier']['id'] == 'agent.external-fragments-independent'
        assert {'path': record.source_path, 'sha256': record.source_sha256,
                'size_bytes': record.size_bytes} in authority['historical_sources']
        manifest = scatac_fragments_v2.load_fragments_manifest_v2(step.result['manifest_path'],
            expected_sha256=step.result['manifest_sha256'])
        provenance = manifest['libraries'][0]['provenance']
        producer = external_fragment_manifest.load_adoption_record(provenance['producer_record']['path'],
            provenance['producer_record']['sha256'])
        assert producer['namespace'] == arguments['namespace'] and producer['source_selection'] == 'subset_export'
        evidence_file = backend.service._application._workspace.run_paths(view['run_id']).evidence / ANALYSIS_EVIDENCE_FILENAME
        facts = json.loads(evidence_file.read_bytes())['steps'][0]['facts']
        assert facts['route'] == 'external_fragment_adoption' and facts['conservation_verification'] == 'verified'
        path = '/api/v1/sessions/analysis/revisions/' + view['revision_id']
        evidence = client.get(path + '/evidence', params={'output_name': 'fragments'})
        assert evidence.status_code == 200 and evidence.json()['status'] == 'available'
        report, = [a for a in client.get(path + '/artifacts').json()['artifacts'] if a['artifact_type'] == 'analysis_report']
        delivery = client.get(path + '/artifacts/' + report['handle'], params={'download': True})
        assert delivery.status_code == 200 and delivery.headers['X-Artifact-Source-SHA256'] == report['sha256']
        assert str(tmp_path) not in delivery.text
        assert client.get(path + '/scientific-artifacts').json() == {'artifacts': []}
        assert client.get('/api/v1/sessions/analysis/turns/import/scientific-artifacts').json() == {'artifacts': []}
        assert str(tmp_path) not in json.dumps(view)
        assert execute(client, body) == view
        assert calls == {'production': 1, 'verification': 1} and len(backend.models) == 1


class UnsupportedModel(ScriptedModel):
    def complete(self, *, prompt, response_schema):
        if 'turn_schema_version' in json.loads(prompt):
            self.calls.append(json.loads(prompt))
            return json.dumps(dict(turn_schema_version=1,
                decision=dict(kind='clarify', reason='unsupported_intent')))
        raise AssertionError('Unsupported typed intent entered Planner or output selection.')


def test_fragments_file_does_not_route_unsupported_intent_or_start_science(
        fragment_source_factory, tmp_path, monkeypatch):
    arguments = fragment_source_factory()
    backend = web_fragments(tmp_path, arguments, model_type=UnsupportedModel)
    calls = count_fragment_science(monkeypatch)
    with TestClient(backend.app) as client:
        resource = upload_fragments(client, arguments)
        client.post('/api/v1/sessions', json={'session_id': 'analysis'})
        view = execute(client, fragment_body(resource, utterance='Perform an unsupported experiment.'))
        assert view['status'] == 'clarification' and view['revision_id'] is None
        assert view['response']['kind'] == 'clarify'
        assert len(backend.models) == 1 and len(backend.models[0].calls) == 1
        assert calls == {'production': 0, 'verification': 0}
        state = backend.service._application.sessions.load('analysis')
        assert state.generation == 0 and not state.revisions and state.turn('import').run_id is None


def test_fragments_http_and_fresh_facade_retries_preserve_history_without_science(
        fragment_source_factory, tmp_path, monkeypatch):
    arguments = fragment_source_factory()
    backend = web_fragments(tmp_path, arguments)
    calls = count_fragment_science(monkeypatch)
    with TestClient(backend.app) as client:
        resource = upload_fragments(client, arguments)
        client.post('/api/v1/sessions', json={'session_id': 'analysis'})
        body = fragment_body(resource)
        accepted = execute(client, body)
    initial = backend.service._application.sessions.load('analysis')
    old_source = Path(backend.service.resources.load(resource['resource_id']).source_path)
    old_source.unlink()
    def forbidden(*args, **kwargs):
        raise AssertionError('Completed retries constructed a provider or executed science.')
    fresh = InteractiveAgentApplication(tmp_path / 'workspace', model_profiles=(PROFILE,),
        default_profile_id=PROFILE.profile_id,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': forbidden}),
        approved_source_roots=(tmp_path,))
    uploads = H5ADUploadAdmission(fresh.resources, backend.uploads.root)
    with TestClient(create_app(fresh, uploads=uploads, input_sets=backend.input_sets)) as client:
        assert any({k: choice[k] for k in resource} == resource
                   for choice in client.get('/api/v1/resources').json()['choices'])
        assert client.get('/api/v1/sessions/analysis').status_code == 200
        assert execute(client, body) == accepted
        new = client.post('/api/v1/sessions/analysis/turns',
            json=fragment_body(resource, turn='fresh-consumption', generation=1))
        assert new.status_code == 400 and new.json()['error']['code'] == 'LOCAL_RESOURCE_INTEGRITY_INVALID'
        assert str(tmp_path) not in new.text
    assert fresh._application.sessions.load('analysis') == initial
    assert calls == {'production': 1, 'verification': 1} and len(backend.models) == 1


@pytest.mark.parametrize('problem,code', [
    ('record', 'EXTERNAL_FRAGMENTS_RECORD_INVALID'),
    ('reference', 'REFERENCE_DIGEST_MISMATCH'),
])
def test_upload_byte_registration_preserves_scientific_owner_findings(
        fragment_source_factory, tmp_path, monkeypatch, problem, code):
    arguments = fragment_source_factory(rows=[b'chr2\t0\t1\tB\t0\n'] if problem == 'record' else None)
    if problem == 'reference':
        arguments['reference_bundle_sha256'] = '0' * 64
    backend = web_fragments(tmp_path, arguments)
    calls = count_fragment_science(monkeypatch)
    with TestClient(backend.app) as client:
        resource = upload_fragments(client, arguments)
        assert not backend.models and calls == {'production': 0, 'verification': 0}
        client.post('/api/v1/sessions', json={'session_id': 'analysis'})
        body = fragment_body(resource)
        view = execute(client, body)
        assert view['status'] == 'failed' and view['revision_id'] is None
        run = backend.service._application.run_store.load(view['run_id'])
        assert view['error']['code'] == run.errors[0].code == code
        assert str(tmp_path) not in json.dumps(view)
        state = backend.service._application.sessions.load('analysis')
        assert state.generation == 0 and not state.revisions
        assert execute(client, body) == view
        assert calls == {'production': int(problem != 'reference'), 'verification': 0}


@pytest.mark.parametrize('missing,code', [
    ('reference_bundle_path', 'FRAGMENTS_REFERENCE_REQUIRED'),
    ('reference_bundle_sha256', 'FRAGMENTS_REFERENCE_REQUIRED'),
    ('source_profile', 'FRAGMENTS_PROFILE_REQUIRED'),
    ('namespace', 'FRAGMENTS_NAMESPACE_REQUIRED'),
])
def test_missing_typed_fragments_prerequisite_has_safe_specific_feedback_before_provider(
        fragment_source_factory, tmp_path, monkeypatch, missing, code):
    arguments = fragment_source_factory()
    values = fragment_companion(arguments).inputs()
    values.pop(missing)
    companion = ScientificInputSet('incomplete', 'Incomplete operator declarations', values,
                                  fragments_companion=True)
    backend = web_fragments(tmp_path, arguments, companions=(companion,))
    calls = count_fragment_science(monkeypatch)
    with TestClient(backend.app) as client:
        resource = upload_fragments(client, arguments)
        client.post('/api/v1/sessions', json={'session_id': 'analysis'})
        response = client.post('/api/v1/sessions/analysis/turns',
            json=fragment_body(resource, companion='incomplete'))
        assert response.status_code == 400 and response.json()['error']['code'] == code
        assert str(tmp_path) not in response.text and 'sha256' not in response.json()['error']['message']
        assert not backend.models and calls == {'production': 0, 'verification': 0}
        state = backend.service._application.sessions.load('analysis')
        assert state.generation == 0 and not state.turns and not state.interactions


class DownstreamModel(ScriptedModel):
    """The provider explicitly selects every operation and scientific edge."""
    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        if 'output_selection_schema_version' in value:
            self.calls.append(value)
            return json.dumps(dict(outputs=[dict(name='matrix', step_id='matrix', output_key='manifest_path')]))
        if ('turn_schema_version' in value or
                'selection_schema_version' in response_schema.get('properties', {})):
            return super().complete(prompt=prompt, response_schema=response_schema)
        self.calls.append(value)
        def inp(port, key):
            return dict(target=port, source=dict(kind='input', input=key))
        def upstream(port, step):
            return dict(target=port, source=dict(kind='step', step=step))
        return json.dumps(dict(schema_version=4, decision=dict(kind='plan', steps=[
            dict(step_id='import', tool='import_scATAC_fragments', control_dependencies=[], sources=[
                inp('source', 'source_path'), inp('reference', 'reference_bundle_path'),
                inp('source_profile', 'source_profile'), inp('namespace', 'namespace')]),
            dict(step_id='qc', tool='compute_scATAC_qc', control_dependencies=[], sources=[
                upstream('fragments', 'import'), inp('qc_reference', 'qc_reference_manifest_path')]),
            dict(step_id='selection', tool='select_scATAC_cells', control_dependencies=[], sources=[
                upstream('barcode_qc', 'qc'), inp('min_qc_fragment_records', 'min_qc_fragment_records'),
                inp('min_tss_enrichment', 'min_tss_enrichment')]),
            dict(step_id='matrix', tool='build_scATAC_cell_by_ccre', control_dependencies=[], sources=[
                upstream('fragments', 'import'), upstream('selected_cells', 'selection'),
                inp('reference', 'reference_manifest_path')]),
        ])))


def downstream_inputs(tmp_path):
    """Same qualified tiny reference design as the accepted rich-Web fixtures."""
    import pysam
    from agent.tools.data import scatac_reference as reference, scatac_qc_reference as qc_reference
    root = tmp_path / 'scientific-inputs'
    root.mkdir()
    fasta = root / 'reference.fa'
    fasta.write_text('>chr2\n' + 'A' * 6001 + '\n>chr1\n' + 'C' * 6001 + '\n')
    pysam.faidx(str(fasta))
    bed = root / 'features.bed'
    bed.write_text('chr2\t0\t6001\nchr1\t0\t6001\n')
    bundle = reference.build_scatac_reference_bundle(species='human', target_assembly='hg38',
        fasta_path=fasta, fai_path=Path(str(fasta) + '.fai'), ccre_bed_path=bed)
    pointer = reference.publish_scatac_reference_bundle(bundle, root / 'reference.json')
    annotation = root / 'annotation.tsv'
    annotation.write_text('chr2\t3000\t3100\t+\tg1\tt1\tprotein_coding\n')
    qc_reference.build_scatac_qc_reference_bundle(parent_manifest_path=pointer['manifest_path'],
        parent_manifest_sha256=pointer['manifest_sha256'], annotation_path=annotation,
        annotation_source='synthetic', annotation_release='1',
        classifications=tuple(qc_reference.QCContig(n, 6001, 'primary_nuclear_qc') for n in ('chr2', 'chr1')),
        classification_source='explicit-test', output_dir=root / 'qc-reference')
    qc_pointer = root / 'qc-reference' / 'manifest.json'
    source = root / 'external-fragments.tsv'
    source.write_text('chr2\t2950\t3050\tA\t1\nchr2\t1000\t1001\tA\t2\nchr1\t0\t100\tB\t3\n')
    return dict(source_path=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_profile='10x-atac-fragments.v1', reference_bundle_path=pointer['manifest_path'],
        reference_bundle_sha256=pointer['manifest_sha256'], namespace='explicit_library',
        qc_reference_manifest_path=str(qc_pointer), qc_reference_manifest_sha256=hashlib.sha256(qc_pointer.read_bytes()).hexdigest(),
        reference_manifest_path=pointer['manifest_path'], reference_manifest_sha256=pointer['manifest_sha256'],
        min_qc_fragment_records=0, min_tss_enrichment='0')


def test_ordinary_provider_plan_composes_explicit_downstream_resources_and_preserves_downloads(tmp_path, monkeypatch):
    from rich_helpers import ScientificWork
    arguments = downstream_inputs(tmp_path)
    backend = web_fragments(tmp_path, arguments, model_type=DownstreamModel)
    monkeypatch.setenv('AGENT_QC_BEDTOOLS', '/usr/bin/bedtools')
    monkeypatch.setenv('AGENT_MATRIX_BEDTOOLS', '/usr/bin/bedtools')
    monkeypatch.setenv('AGENT_QC_ALLOW_SYNTHETIC', '1')
    monkeypatch.delenv('AGENT_QC_RESOURCE_CATALOG', raising=False)
    work = ScientificWork()
    try:
        with TestClient(backend.app) as client:
            resource = upload_fragments(client, arguments)
            assert not work.snapshot() and not backend.models
            client.post('/api/v1/sessions', json={'session_id': 'analysis'})
            body = fragment_body(resource, utterance='Import these fragments, perform QC, select cells using the supplied thresholds, and construct a cell-by-cCRE matrix.')
            view = execute(client, body)
            assert view['status'] == 'succeeded', view
            run = backend.service._application.run_store.load(view['run_id'])
            assert [step.tool_name for step in run.steps] == ['import_scATAC_fragments', 'compute_scATAC_qc', 'select_scATAC_cells', 'build_scATAC_cell_by_ccre']
            assert run.steps[1].resolved_arguments['fragments_manifest_sha256'] == run.steps[0].result['manifest_sha256']
            assert run.steps[2].resolved_arguments['barcode_qc_manifest_sha256'] == run.steps[1].result['manifest_sha256']
            assert run.steps[3].resolved_arguments['selected_cells_manifest_sha256'] == run.steps[2].result['manifest_sha256']
            assert run.steps[3].resolved_arguments['reference_manifest_sha256'] == arguments['reference_manifest_sha256']
            before = work.snapshot()
            for scientific in ('fragments', 'qc', 'selection', 'matrix'):
                assert before['production.' + scientific] == 1
                assert before['owner.' + scientific] == 1
            artifacts = client.get('/api/v1/sessions/analysis/turns/import/scientific-artifacts').json()['artifacts']
            assert {a['filename'] for a in artifacts} == {'matrix.h5ad', 'barcodes.tsv.gz', 'lengths.tsv.gz', 'decisions.tsv.gz', 'selected.tsv.gz'}
            path = '/api/v1/sessions/analysis/revisions/' + view['revision_id']
            for artifact in artifacts:
                delivered = client.get(path + '/scientific-artifacts/' + artifact['handle'])
                assert delivered.status_code == 200, delivered.text
                assert len(delivered.content) == artifact['size_bytes']
                assert hashlib.sha256(delivered.content).hexdigest() == artifact['sha256']
            assert execute(client, body) == view
            assert work.snapshot() == before
            state = backend.service._application.sessions.load('analysis')
            assert state.generation == 1 and len(state.revisions) == 1
            assert {o.name for o in state.revisions[0].outputs} == {'matrix'}
    finally:
        work.close()
