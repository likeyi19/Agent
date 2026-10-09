"""Explicitly uploaded FASTQ libraries execute through ordinary scientific owners.

Only provider decisions are scripted. Guarded production cases use the existing
qualified Chromap runtime and generated tiny resources, not biological fixtures.
"""
import hashlib
import gzip
import json
import os
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
from agent.tools.data import _chromap as chromap, chromap_reference_index
from agent.tools.data import raw_scatac, raw_scatac_manifest
from agent.tools.data import scatac_fragments_v2, scatac_library_context as library
from agent.web.app import create_app
from agent.web.config import ScientificInputSet

from helpers import wait_turn

sys.path.insert(0, str(Path(__file__).parents[1] / 'application'))
from test_registered_fastq_production import (
    BC, INSPECT, PREPARE, PROFILE, backend as qualified_backend,
    executables, instrument, reads, tiny,
)


UTTERANCE = 'Inspect this exact supplied scATAC-seq FASTQ library and report its supported preprocessing readiness.'


class FASTQModel:
    """Interpreter and Planner choose every operation and explicit semantic edge."""
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
            return json.dumps(dict(outputs=[dict(name='intake' if step == 'inspect' else
                'matrix' if step == 'matrix' else 'fragments', step_id=step, output_key='manifest_path')]))
        def inp(port, name):
            return dict(target=port, source=dict(kind='input', input=name))
        def upstream(port, name):
            return dict(target=port, source=dict(kind='step', step=name))
        steps = [dict(step_id='inspect', tool=INSPECT, control_dependencies=[],
            sources=[inp('raw_input', 'raw_input_paths')])]
        if self.mode != 'inspect':
            steps.append(dict(step_id='prepare', tool=PREPARE, control_dependencies=[], sources=[
                upstream('intake', 'inspect'), inp('library_context', 'library_context_path'),
                inp('reference', 'reference_bundle_path')]))
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


def web_fastq(tmp_path, *, modes=('inspect',), companions=(), registry=None, callback=None):
    models = []
    def factory(profile):
        assert profile == PROFILE
        model = FASTQModel(modes[min(len(models), len(modes) - 1)], callback)
        models.append(model)
        return model
    service = InteractiveAgentApplication(tmp_path / 'workspace', model_profiles=(PROFILE,),
        default_profile_id=PROFILE.profile_id, approved_source_roots=(tmp_path,), registry=registry,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': factory}))
    root = service._application._workspace._ensure_directory(tmp_path / 'workspace' / 'uploads')
    uploads = H5ADUploadAdmission(service.resources, root)
    return SimpleNamespace(service=service, models=models, uploads=uploads, input_sets=companions,
        app=create_app(service, uploads=uploads, input_sets=companions))


def upload_members(client, group, *, optional_index=True, library_id='selected-library',
                   lane='001', chunk='001', compression='plain'):
    records = []
    for role, path in group.files:
        if role.value == 'I1' and not optional_index:
            continue
        # Deliberately identical misleading names cannot determine scientific roles.
        response = client.put('/api/v1/uploads/' + library_id + '-' + lane + '-' + chunk + '-' + role.value,
            params=dict(filename='misleading_R1.fastq', input_type='fastq', library_id=library_id,
                        fastq_layout=group.layout.value, role=role.value, lane=lane, chunk=chunk,
                        compression=compression),
            content=(gzip.compress(Path(path).read_bytes(), mtime=0) if compression == 'gzip'
                else Path(path).read_bytes()), headers={'Content-Type': 'application/octet-stream'})
        assert response.status_code == 201, response.text
        records.append(response.json())
    return records


def complete_library(client, records, *, collection_id='selected-collection'):
    response = client.post('/api/v1/uploads/fastq-collections', json=dict(
        collection_id=collection_id, label='Selected single scATAC library',
        member_ids=[item['resource_id'] for item in records]))
    assert response.status_code == 201, response.text
    return response.json()


def upload_library(client, group, *, collection_id='selected-collection', **kwargs):
    records = upload_members(client, group, **kwargs)
    return complete_library(client, records, collection_id=collection_id), records


def turn_body(resource, *, turn='analysis', generation=0, companion=None, **changes):
    value = dict(turn_id=turn, expected_generation=generation, utterance=UTTERANCE,
                 resource_id=resource['resource_id'])
    if companion is not None:
        value['input_set_id'] = companion
    return value | changes


def execute(client, body):
    response = client.post('/api/v1/sessions/analysis/turns', json=body)
    assert response.status_code == 202, response.text
    return wait_turn(client, body['turn_id'], session='analysis', timeout=60)


def uploaded_records(service, records):
    return [service.resources.load(item['resource_id']) for item in records]


def qualified_companion(service, records, tiny, whitelist, *, input_set_id='qualified'):
    """An operator provisions context against these exact synthetic uploaded locators."""
    source_records = uploaded_records(service, records)
    root = tiny['root'] / ('uploaded-context-' + input_set_id)
    root.mkdir()
    intake = raw_scatac.inspect_raw_scATAC([record.source_path for record in source_records], root / 'intake',
        species='human', raw_assay='TENX_ATAC', fastq_layout=source_records[0].fastq_member['fastq_layout'])
    _, manifest, _ = raw_scatac_manifest.load_raw_intake_manifest(intake['manifest_path'])
    context = library.build_scatac_library_processing_context(
        intake_manifest_path=intake['manifest_path'], expected_intake_sha256=intake['manifest_sha256'],
        selection_mode=library.SelectionMode.ALL_GROUPS,
        libraries=(library.LibraryDeclaration(namespace='explicit_library',
            group_ids=tuple(group.id for group in manifest.groups),
            membership_basis=library.MembershipBasis.SINGLE_GROUP if len(manifest.groups) == 1
                else library.MembershipBasis.CALLER_SHARED_LIBRARY,
            barcode_interpretation=library.BarcodeInterpretation.RAW_SEQUENCE,
            correction_policy=library.CorrectionPolicy.WHITELIST_REQUIRED, whitelist=whitelist),))
    pointer = library.publish_scatac_library_processing_context(context, root / 'context.json')
    return ScientificInputSet(input_set_id, 'Qualified synthetic FASTQ context and human reference',
        dict(species='human', library_context_path=pointer['manifest_path'],
            library_context_sha256=pointer['manifest_sha256'],
            reference_bundle_path=tiny['pointer']['manifest_path'],
            reference_bundle_sha256=tiny['pointer']['manifest_sha256']), fastq_companion=True)


def qualified_runtime(tiny, executables, monkeypatch):
    executable = executables['candidate']['path']
    indexes = tiny['root'] / 'indexes'
    indexes.mkdir()
    chromap_reference_index.prepare_chromap_reference_index(
        reference_manifest_path=tiny['pointer']['manifest_path'],
        expected_reference_sha256=tiny['pointer']['manifest_sha256'], executable=executable,
        backend=qualified_backend(), output_dir=tiny['root'] / 'prepared-index')
    (tiny['root'] / 'prepared-index').rename(indexes / 'accepted')
    monkeypatch.setenv('AGENT_CHROMAP_BIN', executable)
    monkeypatch.setenv('AGENT_CHROMAP_INDEX_ROOT', str(indexes))


@pytest.mark.parametrize('layout', [chromap.FastqLayout.A, chromap.FastqLayout.B])
@pytest.mark.parametrize('optional_index', [False, True])
@pytest.mark.parametrize('compression', ['plain', 'gzip'])
def test_uploaded_fastq_layouts_exact_typed_roles_inspect_without_producer_context(
        tiny, reads, monkeypatch, layout, optional_index, compression):
    group, _ = reads([(100, 200, BC, 2)], layout=layout)
    registry, calls = instrument(monkeypatch)
    companion = ScientificInputSet('inspection', 'Declared human species', dict(species='human'), fastq_companion=True)
    web = web_fastq(tiny['root'], companions=(companion,), registry=registry)
    with TestClient(web.app) as client:
        resource, records = upload_library(client, group, optional_index=optional_index, compression=compression)
        sources = uploaded_records(web.service, records)
        assert not web.models and calls == dict(intake=0, production=0, verification=0, alignment=0)
        assert not list(web.service._application.sessions._store.root.glob('*.json'))
        assert resource['input_type'] == 'fastq' and resource['fastq_collection'] is True
        client.post('/api/v1/sessions', json=dict(session_id='analysis'))
        body = turn_body(resource, companion='inspection')
        view = execute(client, body)
        assert view['status'] == 'succeeded' and view['revision_id'], view
        step, = web.service._application.run_store.load(view['run_id']).steps
        assert step.result['readiness'] == 'READY' and step.result['n_files'] == 3 + optional_index
        assert step.result['n_groups'] == 1 and step.verification.passed
        assert _serialize(step.resolved_arguments['raw_input_paths']) == sorted(record.source_path for record in sources)
        assert step.resolved_arguments['fastq_layout'] == layout.value
        assert step.resolved_arguments['raw_assay'] == 'TENX_ATAC'
        _, manifest, _ = raw_scatac_manifest.load_raw_intake_manifest(step.result['manifest_path'])
        facts = [json.loads(evidence.value) for evidence in manifest.evidence
                 if evidence.value.startswith('{') and json.loads(evidence.value).get('fact') == 'read-meaning']
        assert {fact['role'] for fact in facts} == ({'R1', 'R2', 'R3'} if layout is chromap.FastqLayout.A else {'R1', 'R2', 'I2'}) | ({'I1'} if optional_index else set())
        assert {source.path for source in manifest.files} == {record.source_path for record in sources}
        prefix = '/api/v1/sessions/analysis/revisions/' + view['revision_id']
        assert client.get(prefix + '/evidence', params=dict(output_name='intake')).json()['status'] == 'available'
        assert client.get(prefix + '/scientific-artifacts').json() == {'artifacts': []}
        assert str(tiny['root']) not in json.dumps(view)
        assert execute(client, body) == view
        assert calls == dict(intake=1, production=0, verification=0, alignment=0)
        assert len(web.models) == 1 and len(web.models[0].calls) == 4


@pytest.mark.parametrize('layout', [chromap.FastqLayout.A, chromap.FastqLayout.B])
def test_uploaded_qualified_fastq_fresh_intake_exact_members_and_multiturn(
        tiny, reads, executables, monkeypatch, layout):
    group, whitelist = reads([(100, 200, BC, 3)], layout=layout)
    registry, calls = instrument(monkeypatch)
    web = web_fastq(tiny['root'], modes=('prepare', 'inspect'), registry=registry)
    with TestClient(web.app) as client:
        resource, records = upload_library(client, group)
    companion = qualified_companion(web.service, records, tiny, whitelist)
    qualified_runtime(tiny, executables, monkeypatch)
    with TestClient(create_app(web.service, uploads=web.uploads, input_sets=(companion,))) as client:
        client.post('/api/v1/sessions', json=dict(session_id='analysis'))
        body = turn_body(resource, companion='qualified', utterance='Prepare fragments from this library using its selected qualified context.')
        view = execute(client, body)
        assert view['status'] == 'succeeded' and view['revision_id'], view
        inspected, prepared = web.service._application.run_store.load(view['run_id']).steps
        assert prepared.resolved_arguments['intake_manifest_path'] == inspected.result['manifest_path']
        assert prepared.resolved_arguments['intake_manifest_sha256'] == inspected.result['manifest_sha256']
        for key in ('library_context_path', 'library_context_sha256', 'reference_bundle_path', 'reference_bundle_sha256'):
            assert prepared.resolved_arguments[key] == companion.inputs()[key]
        assert prepared.result['total_support'] == 3 and prepared.result['n_fragment_records'] == 1
        assert prepared.result['n_libraries'] == 1 and prepared.verification.passed
        authority = prepared.verification.artifact_authority
        assert authority['schema_version'] == 2 and authority['verifier']['id'] == 'agent.fastq-fragments-independent'
        assert authority['producer_qualification']['scope'] == 'fastq_source_and_producer_record.v1'
        manifest = scatac_fragments_v2.load_fragments_manifest_v2(prepared.result['manifest_path'],
            expected_sha256=prepared.result['manifest_sha256'])
        actual = [source['resource'] for source in manifest['libraries'][0]['provenance']['sources'] if source['role'] == 'fastq']
        sources = uploaded_records(web.service, records)
        assert {source['path'] for source in actual} == {record.source_path for record in sources}
        for record in sources:
            identity = dict(path=record.source_path, sha256=record.source_sha256, size_bytes=record.size_bytes)
            assert identity in actual and identity in authority['historical_sources']
        prefix = '/api/v1/sessions/analysis/revisions/' + view['revision_id']
        assert client.get(prefix + '/evidence', params=dict(output_name='fragments')).json()['status'] == 'available'
        report, = [item for item in client.get(prefix + '/artifacts').json()['artifacts'] if item['artifact_type'] == 'analysis_report']
        delivered = client.get(prefix + '/artifacts/' + report['handle'], params=dict(download=True))
        assert delivered.status_code == 200 and str(tiny['root']) not in delivered.text
        assert client.get(prefix + '/scientific-artifacts').json() == {'artifacts': []}
        before = dict(calls)
        assert execute(client, body) == view and calls == before
        following = execute(client, turn_body(resource, turn='inspect-again', generation=1, companion='qualified'))
        assert following['status'] == 'succeeded'
        state = web.service._application.sessions.load('analysis')
        assert state.generation == 2 and len(state.revisions) == 2
        assert state.revisions[-1].parent_revision_id == view['revision_id']
        assert calls == dict(intake=2, production=1, verification=1, alignment=1)


def test_fastq_library_preserves_declared_lane_and_chunk_members(tiny, reads, monkeypatch):
    first, _ = reads([(100, 200, BC, 1)], name='lane-one')
    second, _ = reads([(1100, 1200, BC, 1)], name='lane-two')
    registry, calls = instrument(monkeypatch)
    companion = ScientificInputSet('inspection', 'Human species', dict(species='human'), fastq_companion=True)
    web = web_fastq(tiny['root'], companions=(companion,), registry=registry)
    with TestClient(web.app) as client:
        records = (upload_members(client, first) +
            upload_members(client, second, lane='002') +
            upload_members(client, second, lane='002', chunk='002', optional_index=False))
        resource = complete_library(client, records)
        client.post('/api/v1/sessions', json=dict(session_id='analysis'))
        view = execute(client, turn_body(resource, companion='inspection'))
        assert view['status'] == 'succeeded', view
        step, = web.service._application.run_store.load(view['run_id']).steps
        assert step.result['readiness'] == 'READY'
        assert (step.result['n_files'], step.result['n_groups']) == (11, 3)
        _, manifest, _ = raw_scatac_manifest.load_raw_intake_manifest(step.result['manifest_path'])
        assert {(group.lane, group.chunk) for group in manifest.groups} == {('001', '001'), ('002', '001'), ('002', '002')}
        assert {source.path for source in manifest.files} == {record.source_path for record in uploaded_records(web.service, records)}
        assert calls == dict(intake=1, production=0, verification=0, alignment=0)


def test_fastq_upload_collection_does_not_route_unsupported_intent(tiny, reads, monkeypatch):
    group, _ = reads([(100, 200, BC, 1)])
    registry, calls = instrument(monkeypatch)
    web = web_fastq(tiny['root'], modes=('unsupported',), registry=registry)
    with TestClient(web.app) as client:
        resource, _ = upload_library(client, group)
        client.post('/api/v1/sessions', json=dict(session_id='analysis'))
        view = execute(client, turn_body(resource, utterance='Use an unsupported barcode correction algorithm.'))
        assert view['status'] == 'clarification' and view['revision_id'] is None
        assert view['response']['kind'] == 'clarify'
        assert len(web.models) == 1 and len(web.models[0].calls) == 1
        assert calls == dict(intake=0, production=0, verification=0, alignment=0)
        state = web.service._application.sessions.load('analysis')
        assert state.generation == 0 and not state.revisions and state.turn('analysis').run_id is None


@pytest.mark.parametrize('during_provider', [False, True])
def test_fastq_changed_member_blocks_new_scientific_consumption(tiny, reads, monkeypatch, during_provider):
    group, _ = reads([(100, 200, BC, 1)])
    registry, calls = instrument(monkeypatch)
    selected = []
    def changed():
        selected[0].write_bytes(b'Changed during interpretation of the registered member.\n')
    web = web_fastq(tiny['root'], registry=registry, callback=changed if during_provider else None)
    with TestClient(web.app) as client:
        resource, records = upload_library(client, group)
        selected.append(Path(uploaded_records(web.service, records)[0].source_path))
        client.post('/api/v1/sessions', json=dict(session_id='analysis'))
        if during_provider:
            view = execute(client, turn_body(resource))
            assert view['status'] == 'failed' and view['error']['code'] == 'LOCAL_RESOURCE_INTEGRITY_INVALID', view
        else:
            changed()
            response = client.post('/api/v1/sessions/analysis/turns', json=turn_body(resource))
            assert response.status_code == 400 and response.json()['error']['code'] == 'LOCAL_RESOURCE_INTEGRITY_INVALID'
            assert not web.models
        assert calls == dict(intake=0, production=0, verification=0, alignment=0)
        state = web.service._application.sessions.load('analysis')
        assert state.generation == 0 and not state.revisions


def test_fastq_restart_history_and_completed_retry_ignore_deleted_members(tiny, reads, monkeypatch):
    group, _ = reads([(100, 200, BC, 1)])
    registry, calls = instrument(monkeypatch)
    companion = ScientificInputSet('inspection', 'Human species', dict(species='human'), fastq_companion=True)
    web = web_fastq(tiny['root'], companions=(companion,), registry=registry)
    with TestClient(web.app) as client:
        resource, records = upload_library(client, group)
        client.post('/api/v1/sessions', json=dict(session_id='analysis'))
        body = turn_body(resource, companion='inspection')
        accepted = execute(client, body)
        assert accepted['status'] == 'succeeded', accepted
    initial = web.service._application.sessions.load('analysis')
    for record in uploaded_records(web.service, records):
        Path(record.source_path).unlink()
    def forbidden(*args, **kwargs):
        raise AssertionError('Completed history must not create providers or execute science.')
    fresh = InteractiveAgentApplication(tiny['root'] / 'workspace', model_profiles=(PROFILE,),
        default_profile_id=PROFILE.profile_id, approved_source_roots=(tiny['root'],), registry=registry,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': forbidden}))
    uploads = H5ADUploadAdmission(fresh.resources, web.uploads.root)
    with TestClient(create_app(fresh, uploads=uploads, input_sets=(companion,))) as client:
        assert any(item['resource_id'] == resource['resource_id'] and item['fastq_collection']
                   for item in client.get('/api/v1/resources').json()['choices'])
        assert client.get('/api/v1/sessions/analysis').status_code == 200
        assert execute(client, body) == accepted
        prefix = '/api/v1/sessions/analysis/revisions/' + accepted['revision_id']
        assert client.get(prefix + '/evidence', params=dict(output_name='intake')).json()['status'] == 'available'
        response = client.post('/api/v1/sessions/analysis/turns',
            json=turn_body(resource, turn='new-consumption', generation=1, companion='inspection'))
        assert response.status_code == 400 and response.json()['error']['code'] == 'LOCAL_RESOURCE_INTEGRITY_INVALID'
        assert str(tiny['root']) not in response.text
    assert fresh._application.sessions.load('analysis') == initial
    assert calls == dict(intake=1, production=0, verification=0, alignment=0)
    assert len(web.models) == 1


@pytest.mark.parametrize('missing,code', [
    ('library_context', 'FASTQ_LIBRARY_CONTEXT_REQUIRED'),
    ('reference_bundle', 'FASTQ_REFERENCE_REQUIRED'),
])
def test_fastq_prepare_missing_typed_prerequisites_fail_before_scientific_intake(
        tiny, reads, monkeypatch, missing, code):
    group, whitelist = reads([(100, 200, BC, 1)])
    registry, calls = instrument(monkeypatch)
    web = web_fastq(tiny['root'], modes=('prepare',), registry=registry)
    with TestClient(web.app) as client:
        resource, records = upload_library(client, group)
    qualified = qualified_companion(web.service, records, tiny, whitelist)
    values = qualified.inputs()
    for suffix in ('_path', '_sha256'):
        values.pop(missing + suffix)
    companion = ScientificInputSet('incomplete', 'Incomplete approved FASTQ context', values, fastq_companion=True)
    with TestClient(create_app(web.service, uploads=web.uploads, input_sets=(companion,))) as client:
        client.post('/api/v1/sessions', json=dict(session_id='analysis'))
        body = turn_body(resource, companion='incomplete', utterance='Prepare fragments from this FASTQ library.')
        view = execute(client, body)
        assert view['status'] == 'failed' and view['revision_id'] is None, view
        assert view['error']['code'] == code
        assert str(tiny['root']) not in json.dumps(view) and 'sha256' not in view['error']['message']
        assert calls == dict(intake=0, production=0, verification=0, alignment=0)
        before = len(web.models)
        assert execute(client, body) == view and len(web.models) == before


def test_fastq_explicit_incompatible_context_cannot_fall_back(tiny, reads, executables, monkeypatch):
    group, whitelist = reads([(100, 200, BC, 1)])
    registry, calls = instrument(monkeypatch)
    web = web_fastq(tiny['root'], modes=('prepare',), registry=registry)
    with TestClient(web.app) as client:
        resource, records = upload_library(client, group)
        _, other_records = upload_library(client, group, library_id='other-library', collection_id='other-collection')
    good = qualified_companion(web.service, records, tiny, whitelist, input_set_id='good')
    bad = qualified_companion(web.service, other_records, tiny, whitelist, input_set_id='bad')
    good = ScientificInputSet('good', good.label, good.inputs(), fastq_companion=True, fastq_default=True)
    qualified_runtime(tiny, executables, monkeypatch)
    with TestClient(create_app(web.service, uploads=web.uploads, input_sets=(good, bad))) as client:
        client.post('/api/v1/sessions', json=dict(session_id='analysis'))
        body = turn_body(resource, companion='bad', utterance='Prepare fragments using my explicitly selected context.')
        view = execute(client, body)
        assert view['status'] == 'failed' and view['revision_id'] is None, view
        assert view['error']['code'] == 'FRAGMENTS_CONTEXT_MISMATCH'
        assert str(tiny['root']) not in json.dumps(view)
        run = web.service._application.run_store.load(view['run_id'])
        prepared = run.steps[-1]
        assert prepared.resolved_arguments['library_context_path'] == bad.inputs()['library_context_path']
        assert prepared.resolved_arguments['library_context_path'] != good.inputs()['library_context_path']
        assert calls == dict(intake=1, production=1, verification=0, alignment=0)


def test_fastq_provider_downstream_semantic_handoff_preserves_five_downloads(
        tiny, reads, executables, monkeypatch):
    from agent.tools.data import scatac_qc_reference as qc_reference
    from rich_helpers import ScientificWork
    group, whitelist = reads([(100, 200, BC, 3), (2100, 2200, BC, 2)])
    registry, calls = instrument(monkeypatch)
    web = web_fastq(tiny['root'], modes=('downstream',), registry=registry)
    with TestClient(web.app) as client:
        resource, records = upload_library(client, group)
    qualified = qualified_companion(web.service, records, tiny, whitelist)
    qualified_runtime(tiny, executables, monkeypatch)
    annotation = tiny['root'] / 'annotation.tsv'
    annotation.write_text('chrTiny\t150\t250\t+\tg1\tt1\tprotein_coding\n'
                          'chrTiny\t2104\t2204\t+\tg2\tt2\tprotein_coding\n')
    qc_reference.build_scatac_qc_reference_bundle(
        parent_manifest_path=tiny['pointer']['manifest_path'],
        parent_manifest_sha256=tiny['pointer']['manifest_sha256'], annotation_path=annotation,
        annotation_source='synthetic', annotation_release='1',
        classifications=(qc_reference.QCContig('chrTiny', 12000, 'primary_nuclear_qc'),),
        classification_source='explicit-test', output_dir=tiny['root'] / 'qc-reference')
    qc_pointer = tiny['root'] / 'qc-reference' / 'manifest.json'
    values = qualified.inputs() | dict(qc_reference_manifest_path=str(qc_pointer),
        qc_reference_manifest_sha256=hashlib.sha256(qc_pointer.read_bytes()).hexdigest(),
        reference_manifest_path=tiny['pointer']['manifest_path'],
        reference_manifest_sha256=tiny['pointer']['manifest_sha256'],
        min_qc_fragment_records=0, min_tss_enrichment='0')
    companion = ScientificInputSet('qualified', qualified.label, values, fastq_companion=True)
    monkeypatch.setenv('AGENT_QC_BEDTOOLS', '/usr/bin/bedtools')
    monkeypatch.setenv('AGENT_MATRIX_BEDTOOLS', '/usr/bin/bedtools')
    monkeypatch.setenv('AGENT_QC_ALLOW_SYNTHETIC', '1')
    monkeypatch.delenv('AGENT_QC_RESOURCE_CATALOG', raising=False)
    work = ScientificWork()
    try:
        with TestClient(create_app(web.service, uploads=web.uploads, input_sets=(companion,))) as client:
            client.post('/api/v1/sessions', json=dict(session_id='analysis'))
            body = turn_body(resource, companion='qualified', utterance='Process this FASTQ library, perform QC, select cells using the supplied thresholds, and construct a cell-by-cCRE matrix.')
            view = execute(client, body)
            assert view['status'] == 'succeeded', view
            run = web.service._application.run_store.load(view['run_id'])
            assert [step.tool_name for step in run.steps] == [INSPECT, PREPARE, 'compute_scATAC_qc',
                                                           'select_scATAC_cells', 'build_scATAC_cell_by_ccre']
            for before, after, prefix in ((0, 1, 'intake_manifest_'), (1, 2, 'fragments_manifest_'),
                (2, 3, 'barcode_qc_manifest_'), (3, 4, 'selected_cells_manifest_')):
                assert run.steps[after].resolved_arguments[prefix + 'path'] == run.steps[before].result['manifest_path']
                assert run.steps[after].resolved_arguments[prefix + 'sha256'] == run.steps[before].result['manifest_sha256']
            snapshot = work.snapshot()
            for name in ('qc', 'selection', 'matrix'):
                assert snapshot['production.' + name] == snapshot['owner.' + name] == 1
            assert calls == dict(intake=1, production=1, verification=1, alignment=1)
            artifacts = client.get('/api/v1/sessions/analysis/turns/analysis/scientific-artifacts').json()['artifacts']
            assert {item['filename'] for item in artifacts} == {'matrix.h5ad', 'barcodes.tsv.gz',
                'lengths.tsv.gz', 'decisions.tsv.gz', 'selected.tsv.gz'}
            prefix = '/api/v1/sessions/analysis/revisions/' + view['revision_id']
            for item in artifacts:
                response = client.get(prefix + '/scientific-artifacts/' + item['handle'])
                assert response.status_code == 200 and len(response.content) == item['size_bytes']
                assert hashlib.sha256(response.content).hexdigest() == item['sha256']
            assert execute(client, body) == view and work.snapshot() == snapshot
            assert calls == dict(intake=1, production=1, verification=1, alignment=1)
            evidence = web.service._application._workspace.run_paths(view['run_id']).evidence / ANALYSIS_EVIDENCE_FILENAME
            assert evidence.is_file()
            state = web.service._application.sessions.load('analysis')
            assert state.generation == 1 and len(state.revisions) == 1
            assert {output.name for output in state.revisions[0].outputs} == {'matrix'}
    finally:
        work.close()


@pytest.mark.parametrize('mutation', ['deleted', 'mtime'])
def test_fastq_unfinished_compound_presentation_preserves_source_lifetime_boundary(
        tiny, reads, executables, monkeypatch, mutation):
    from agent.application.sessions import AnalysisSessions
    from agent.tools.data import fastq_fragments, fastq_fragments_verifier
    class Interrupted(BaseException):
        pass
    group, whitelist = reads([(100, 200, BC, 2)])
    registry, calls = instrument(monkeypatch)
    web = web_fastq(tiny['root'], modes=('prepare',), registry=registry)
    with TestClient(web.app) as client:
        resource, records = upload_library(client, group)
    companion = qualified_companion(web.service, records, tiny, whitelist)
    qualified_runtime(tiny, executables, monkeypatch)
    binding = web.uploads.resolve(resource['resource_id'], scientific_inputs=companion.inputs())
    web.service.create_session('analysis')
    utterance = 'Prepare fragments from this selected qualified FASTQ library.'
    with monkeypatch.context() as stopped:
        def interrupt(*args, **kwargs):
            raise Interrupted()
        stopped.setattr(AnalysisSessions, '_record_result', interrupt)
        with pytest.raises(Interrupted):
            web.service.submit_turn('analysis', 'prepare', utterance, expected_generation=0,
                registered_input=binding)
    state = web.service._application.sessions.load('analysis')
    assert state.generation == 0 and not state.revisions and state.turn('prepare').status == 'linked'
    run = web.service._application.run_store.load(state.turn('prepare').run_id)
    assert run.lifecycle_status.value == 'SUCCEEDED'
    assert run.steps[-1].verification.artifact_authority['source_policy'] == 'historical_verified_sources.v1'
    assert calls == dict(intake=1, production=1, verification=1, alignment=1)
    source = Path(uploaded_records(web.service, records)[0].source_path)
    original_bytes, original_stat = source.read_bytes(), source.stat()
    if mutation == 'deleted':
        source.unlink()
    else:
        os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 1_000_000_000))
    def forbidden(*args, **kwargs):
        raise AssertionError('Presentation recovery must not rerun FASTQ production or reconstruction.')
    monkeypatch.setattr(fastq_fragments, 'prepare_fastq_fragments', forbidden)
    monkeypatch.setattr(fastq_fragments_verifier, 'verify_stream', forbidden)
    recovered = web.service.recover_turn('analysis', 'prepare', complete_presentation=True)
    assert recovered.status == 'finalizing' and recovered.revision_id is None
    pending = web.service._application.sessions.load('analysis')
    assert pending.generation == 0 and not pending.revisions and pending.turn('prepare').status == 'run_succeeded'
    if mutation == 'deleted':
        source.write_bytes(original_bytes)
    os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    recovered = web.service.recover_turn('analysis', 'prepare', complete_presentation=True)
    assert recovered.status == 'succeeded' and recovered.revision_id
    for record in uploaded_records(web.service, records):
        Path(record.source_path).unlink()
    assert web.service.recover_turn('analysis', 'prepare', complete_presentation=True) == recovered
    assert web.service.submit_turn('analysis', 'prepare', utterance, expected_generation=0,
        registered_input=binding) == recovered
    assert web.service.evidence('analysis', recovered.revision_id, 'fragments').status == 'available'
    assert calls == dict(intake=1, production=1, verification=1, alignment=1) and len(web.models) == 1
