"""Neutral chat sources reach the existing real FASTQ owners without per-upload config."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from agent.application.raw_handoff import FastqDeclaration, FastqMember
from agent.application import InteractiveAgentApplication
from agent.application.uploads import H5ADUploadAdmission
from agent.providers import PlanningModelFactoryRegistry
from agent.web.app import create_app
from agent.web.config import ScientificInputSet
from test_fastq_execution import (FASTQModel, PROFILE, BC, tiny, reads, executables,
    qualified_runtime, instrument)
from test_chat_attachments import upload
from helpers import wait_turn

OBJECTIVE = ('Preprocess my human 10x ATAC library sample, layout A, with raw barcode sequences requiring whitelist correction. '
             'The first two files are R1 and R2, lane 001 chunk 001 of this single library. '
             'Perform QC, select candidates with minimum fragment records 0 and minimum TSS enrichment 0, and build a cell-by-cCRE matrix.')


def raw(members=(), **kwargs):
    return asdict(FastqDeclaration(members=tuple(FastqMember(f'source.{index}', role, '001', '001')
        for index, role in members), **kwargs))


class ChatModel(FASTQModel):
    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        if 'turn_schema_version' in value:
            self.calls.append(value)
            if value['utterance'] == OBJECTIVE:
                arguments = []
                for name, prefix in [('min_qc_fragment_records', 'minimum fragment records '),
                                     ('min_tss_enrichment', 'minimum TSS enrichment ')]:
                    start = OBJECTIVE.index(prefix) + len(prefix)
                    arguments.append(dict(tool='select_scATAC_cells', argument=name, literal='0', start=start, end=start+1))
                decision = dict(kind='request_roles', target='prepare_scATAC_fragments', required_roles=[], roles=[],
                    arguments=arguments, fastq=raw(((0, 'R1'), (1, 'R2')), assay='TENX_ATAC', layout='tenx-atac-r1-r2-r3.v1',
                    namespace='sample', species='human', barcode_policy='raw_sequence_whitelist'))
            else:
                decision = dict(kind='answer_roles', pending='@roles', roles=[], arguments=[], fastq=raw(((3, 'R3'),)))
            return json.dumps(dict(turn_schema_version=1, decision=decision))
        return super().complete(prompt=prompt, response_schema=response_schema)


def chat_web(root, companions=(), registry=None, model_type=ChatModel):
    models = []
    def factory(profile):
        model = model_type('downstream')
        models.append(model)
        return model
    service = InteractiveAgentApplication(root / 'workspace', model_profiles=(PROFILE,),
        default_profile_id=PROFILE.profile_id, approved_source_roots=(root,), registry=registry,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': factory}))
    directory = service._application._workspace._ensure_directory(root / 'workspace' / 'uploads')
    uploads = H5ADUploadAdmission(service.resources, directory)
    return service, uploads, models, create_app(service, uploads=uploads, input_sets=companions)


def send(client, ids, *, turn='first', utterance=OBJECTIVE):
    body = dict(turn_id=turn, expected_generation=0, utterance=utterance, attachment_ids=ids)
    response = client.post('/api/v1/sessions/session/turns', json=body)
    assert response.status_code == 202, response.text
    return wait_turn(client, turn, timeout=120), body


def stage(client, group):
    paths = {r.value: Path(p) for r, p in group.files}
    ids = []
    for role in ('R1', 'R2'):
        response = upload(client, role + '.fastq', paths[role].read_bytes(), token=role)
        assert response.status_code == 201, response.text
        ids.append(response.json()['resource_id'])
    ids.append(upload(client, 'notes.txt', b'unrelated notes', token='notes').json()['resource_id'])
    return ids, paths


def test_incomplete_chat_library_keeps_objective_and_no_science(tiny, reads, monkeypatch):
    group, _ = reads([(100, 200, BC, 1)])
    registry, counts = instrument(monkeypatch)
    service, uploads, models, app = chat_web(tiny['root'], registry=registry)
    with TestClient(app) as client:
        client.post('/api/v1/sessions', json=dict(session_id='session'))
        ids, _ = stage(client, group)
        view, body = send(client, ids)
        assert view['status'] == 'clarification', view
        assert 'R3' in view['response']['text'], view
        state = service._application.sessions.load('session')
        assert state.interactions[0].prerequisite['field'] == 'fastq'
        assert not state.revisions and not state.turns
        assert not any(counts.values())
        assert not (tiny['root'] / 'workspace' / 'fastq_handoffs').exists()
        assert client.post('/api/v1/sessions/session/turns', json=body).status_code == 202
        assert len(models) == 1 and not any(counts.values())


def companion(tiny, whitelist):
    from agent.tools.data import scatac_qc_reference as qc
    annotation = tiny['root'] / 'annotation.tsv'
    annotation.write_text('chrTiny\t150\t250\t+\tg1\tt1\tprotein_coding\n'
                          'chrTiny\t2104\t2204\t+\tg2\tt2\tprotein_coding\n')
    qc.build_scatac_qc_reference_bundle(parent_manifest_path=tiny['pointer']['manifest_path'],
        parent_manifest_sha256=tiny['pointer']['manifest_sha256'], annotation_path=annotation,
        annotation_source='synthetic', annotation_release='1',
        classifications=(qc.QCContig('chrTiny', 12000, 'primary_nuclear_qc'),),
        classification_source='explicit-test', output_dir=tiny['root'] / 'qc-reference')
    pointer = tiny['root'] / 'qc-reference' / 'manifest.json'
    return ScientificInputSet('synthetic', 'Synthetic test reference and whitelist', dict(species='human',
        reference_bundle_path=tiny['pointer']['manifest_path'], reference_bundle_sha256=tiny['pointer']['manifest_sha256'],
        reference_manifest_path=tiny['pointer']['manifest_path'], reference_manifest_sha256=tiny['pointer']['manifest_sha256'],
        qc_reference_manifest_path=str(pointer), qc_reference_manifest_sha256=hashlib.sha256(pointer.read_bytes()).hexdigest()),
        fastq_companion=True, fastq_default=True, fastq_whitelist=asdict(whitelist))


def test_chat_missing_member_real_context_producer_matrix_download_restart(tiny, reads, executables, monkeypatch, record_property):
    from functools import wraps
    from time import perf_counter
    from agent.tools.data import raw_scatac, scatac_library_context as library
    from agent.orchestration import raw_scatac_verifier, verifier
    from agent.orchestration import registry as registry_module
    timings = dict(intake=[], intake_verification=[], context=[])
    def timed(name, function):
        @wraps(function)
        def call(*args, **kwargs):
            start = perf_counter()
            result = function(*args, **kwargs)
            timings[name].append(perf_counter() - start)
            return result
        return call
    inspect = timed('intake', raw_scatac.inspect_raw_scATAC)
    verify = timed('intake_verification', raw_scatac_verifier.verify_raw_scatac_intake)
    # Capture both preparation and Registry aliases, not just one call site.
    monkeypatch.setattr(raw_scatac, 'inspect_raw_scATAC', inspect)
    monkeypatch.setattr(registry_module, 'inspect_raw_scATAC', inspect)
    monkeypatch.setattr(raw_scatac_verifier, 'verify_raw_scatac_intake', verify)
    monkeypatch.setattr(verifier, 'verify_raw_scatac_intake', verify)
    monkeypatch.setattr(library, 'build_scatac_library_processing_context',
        timed('context', library.build_scatac_library_processing_context))
    group, whitelist = reads([(100, 200, BC, 3), (2100, 2200, BC, 2)])
    # The qualified native reader must consume opaque original paths for both
    # gzip and plain inputs; user-visible names do not supply compression facts.
    import gzip
    for role, path in group.files:
        if role.value in ('R1', 'R3'):
            source = Path(path)
            source.write_bytes(gzip.compress(source.read_bytes(), mtime=0))
    resource = companion(tiny, whitelist)
    qualified_runtime(tiny, executables, monkeypatch)
    for key in ('AGENT_QC_BEDTOOLS', 'AGENT_MATRIX_BEDTOOLS'):
        monkeypatch.setenv(key, '/usr/bin/bedtools')
    monkeypatch.setenv('AGENT_QC_ALLOW_SYNTHETIC', '1')
    monkeypatch.delenv('AGENT_QC_RESOURCE_CATALOG', raising=False)
    registry, counts = instrument(monkeypatch)
    from agent.tools.data import fastq_fragments
    native = fastq_fragments.run_stage
    alignment_inputs = []
    def capture_native(argv, **kwargs):
        if '--preset' in argv:
            alignment_inputs.append([argv[argv.index(flag) + 1] for flag in ('-1', '-2', '-b')])
        return native(argv, **kwargs)
    monkeypatch.setattr(fastq_fragments, 'run_stage', capture_native)
    service, uploads, models, app = chat_web(tiny['root'], (resource,), registry)
    with TestClient(app) as client:
        client.post('/api/v1/sessions', json=dict(session_id='session'))
        ids, paths = stage(client, group)
        before = {rid: service.resources.load(rid).to_dict() for rid in ids}
        view, _ = send(client, ids)
        assert view['status'] == 'clarification' and 'R3' in view['response']['text'], view
        assert not any(counts.values())
        assert not any(timings.values())
        final = upload(client, 'R3.fastq', paths['R3'].read_bytes(), token='R3').json()['resource_id']
        view, body = send(client, [final], turn='answer', utterance='The new file is genomic R3, lane 001 chunk 001 of the same library. Continue.')
        record_property('execution_owner_timings_seconds', json.dumps(timings, sort_keys=True))
        assert view['status'] == 'succeeded', view
        assert [a['use'] for a in view['attachments']] == ['consumed', 'consumed', 'staged', 'consumed']
        state = service._application.sessions.load('session')
        assert state.generation == 1 and len(state.revisions) == 1
        run = service._application.run_store.load(view['run_id'])
        assert run.request.prompt == OBJECTIVE
        assert [s.tool_name for s in run.steps] == ['inspect_raw_scATAC', 'prepare_scATAC_fragments',
            'compute_scATAC_qc', 'select_scATAC_cells', 'build_scATAC_cell_by_ccre']
        assert counts == dict(intake=1, production=1, verification=1, alignment=1)
        originals = [service.resources.load(rid) for rid in (ids[0], ids[1], final)]
        raw_sources = run.steps[0].resolved_arguments['raw_input_paths']
        assert {row['path'] for row in raw_sources} == {r.source_path for r in originals}
        assert len(alignment_inputs) == 1 and set(alignment_inputs[0]) == {r.source_path for r in originals}
        assert not list((tiny['root'] / 'workspace' / 'fastq_handoffs').rglob('sources'))
        assert all(p.suffix == '.json' for p in (tiny['root'] / 'workspace' / 'fastq_handoffs').rglob('*') if p.is_file())
        # Includes the existing evidence/report source-step revalidations.
        assert {k: len(v) for k, v in timings.items()} == dict(intake=2, intake_verification=9, context=1)
        assert before == {rid: service.resources.load(rid).to_dict() for rid in ids}
        context_path = Path(run.request.inputs['library_context_path'])
        saved_context = context_path.read_bytes()
        prefix = '/api/v1/sessions/session/turns/answer'
        revision_prefix = '/api/v1/sessions/session/revisions/' + view['revision_id']
        report, = [a for a in client.get(revision_prefix + '/artifacts').json()['artifacts']
                   if a['artifact_type'] == 'analysis_report']
        delivered = client.get(revision_prefix + '/artifacts/' + report['handle'], params=dict(download=True))
        assert delivered.status_code == 200
        assert delivered.headers['X-Artifact-Source-SHA256'] == report['sha256']
        artifacts = client.get(prefix + '/scientific-artifacts').json()['artifacts']
        assert {a['filename'] for a in artifacts} == {'matrix.h5ad', 'barcodes.tsv.gz', 'lengths.tsv.gz',
                                                      'decisions.tsv.gz', 'selected.tsv.gz'}
        for artifact in artifacts:
            response = client.get('/api/v1/sessions/session/revisions/' + view['revision_id']
                + '/scientific-artifacts/' + artifact['handle'])
            assert response.status_code == 200
            assert hashlib.sha256(response.content).hexdigest() == artifact['sha256']
        receipt = state.interactions[-1].admitted['fastq_context']
        assert run.steps[0].result['manifest_sha256'] == receipt['intake_sha256']
        assert run.steps[1].resolved_arguments['library_context_path'] == str(context_path)
        assert run.steps[1].resolved_arguments['intake_manifest_path'] == run.steps[0].result['manifest_path']
        assert run.steps[1].resolved_arguments['intake_manifest_sha256'] == receipt['intake_sha256']
        from agent.tools.data import scatac_library_context as library
        _, context, _ = library.load_scatac_library_processing_context(context_path,
            expected_sha256=receipt['context_sha256'])
        assert context.intake.manifest_sha256 == receipt['intake_sha256']
        from agent.orchestration.verification_authority import _accepted
        for step in run.steps[1:]:
            _, authority, _, _ = _accepted(service._application.run_store, run.run_id, step.step_id)
            assert authority.record['schema_version'] == 2
        assert all(step.verification.passed for step in run.steps)
        # Durable replay rejects forged context/source coupling without reopening science.
        from dataclasses import replace
        from agent.schemas.orchestration import _serialize
        from agent.application.session_state import SessionError
        for field, wrong in [('identity', 'f'*64), ('intake_sha256', 'e'*64),
                             ('context_path', str(context_path.parent.parent / 'other' / 'context.json'))]:
            forged = _serialize(state.interactions[-1].admitted)
            forged['fastq_context'][field] = wrong
            with pytest.raises(SessionError):
                replace(state, interactions=state.interactions[:-1] + (replace(state.interactions[-1], admitted=forged),))
        assert client.post('/api/v1/sessions/session/turns', json=body).status_code == 202
        assert counts == dict(intake=1, production=1, verification=1, alignment=1)
    def forbidden(_):
        raise AssertionError('History read constructed a provider')
    restored = InteractiveAgentApplication(tiny['root'] / 'workspace', model_profiles=(PROFILE,),
        default_profile_id=PROFILE.profile_id, approved_source_roots=(tiny['root'],),
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': forbidden}))
    with TestClient(create_app(restored, uploads=H5ADUploadAdmission(restored.resources, uploads.root), input_sets=(resource,))) as client:
        assert client.get('/api/v1/sessions/session/turns/answer').json() == view
        assert client.post('/api/v1/sessions/session/turns', json=body).status_code == 202
        assert context_path.read_bytes() == saved_context
    assert counts == dict(intake=1, production=1, verification=1, alignment=1)
    assert {k: len(v) for k, v in timings.items()} == dict(intake=2, intake_verification=9, context=1)
    record_property('owner_timings_seconds', json.dumps(timings, sort_keys=True))


@pytest.mark.parametrize('case', ['changed_source', 'foreign_session', 'stale_generation', 'missing_resource', 'forged_role'])
def test_pending_fastq_rejects_unsafe_completion_without_science(tiny, reads, monkeypatch, case):
    group, _ = reads([(100, 200, BC, 1)])
    registry, counts = instrument(monkeypatch)
    class BadAnswer(ChatModel):
        def complete(self, *, prompt, response_schema):
            response = super().complete(prompt=prompt, response_schema=response_schema)
            value = json.loads(response)
            if case == 'forged_role' and value.get('decision', {}).get('kind') == 'answer_roles':
                value['decision']['fastq']['members'][0]['source'] = 'source.15'
            return json.dumps(value)
    service, uploads, models, app = chat_web(tiny['root'], registry=registry, model_type=BadAnswer)
    with TestClient(app) as client:
        client.post('/api/v1/sessions', json=dict(session_id='session'))
        ids, paths = stage(client, group)
        view, _ = send(client, ids)
        assert view['status'] == 'clarification'
        if case == 'foreign_session':
            client.post('/api/v1/sessions', json=dict(session_id='other'))
            reply_id = upload(client, 'R3.fastq', paths['R3'].read_bytes(), token='R3', session='other').json()['resource_id']
        else:
            reply_id = upload(client, 'R3.fastq', paths['R3'].read_bytes(), token='R3').json()['resource_id']
        if case == 'changed_source':
            Path(service.resources.load(ids[0]).source_path).write_bytes(b'changed after capture')
        body = dict(turn_id='answer', expected_generation=1 if case == 'stale_generation' else 0,
            utterance='The new file is genomic R3, lane 001 chunk 001 of the same library. Continue.', attachment_ids=[reply_id])
        response = client.post('/api/v1/sessions/session/turns', json=body)
        if case in {'foreign_session', 'stale_generation'}:
            assert response.status_code in (400, 404, 409, 422), response.text
            assert len(models) == 1
        else:
            assert response.status_code == 202, response.text
            view = wait_turn(client,'answer')
            assert view['status'] in {'failed','clarification'}
            if case == 'missing_resource':
                assert 'must be provisioned' in view['response']['text']
                state = service._application.sessions.load('session')
                assert state.interactions[-1].prerequisite['origin_turn_id'] == 'first'
                assert len(state.interactions[-1].admitted['fastq']['members']) == 3
        assert not service._application.sessions.load('session').revisions
        assert not any(counts.values())
        assert not (tiny['root'] / 'workspace' / 'fastq_handoffs').exists()


def test_fastq_completed_members_preserve_historical_parameter_clarification(tiny, reads, monkeypatch):
    """Historical required ports survive generated context admission.

    Current omission defaults are exercised separately through real selection;
    this fixture preserves the pre-default pending-task compatibility contract.
    """
    from agent.orchestration import ToolRegistry
    from test_scientific_parameter_continuation import legacy_selection_registry
    group, whitelist = reads([(100, 200, BC, 1)])
    resource = companion(tiny, whitelist)
    class MissingThresholds(ChatModel):
        def complete(self, *, prompt, response_schema):
            response = super().complete(prompt=prompt, response_schema=response_schema)
            value = json.loads(response)
            if value.get('decision', {}).get('kind') == 'request_roles':
                value['decision']['arguments'] = []
            return json.dumps(value)
    registry, counts = instrument(monkeypatch)
    legacy = legacy_selection_registry().get('select_scATAC_cells')
    registry = ToolRegistry(tuple(legacy if name == 'select_scATAC_cells' else registry.get(name)
                                  for name in registry.names()))
    service, uploads, models, app = chat_web(tiny['root'], (resource,), registry, MissingThresholds)
    with TestClient(app) as client:
        client.post('/api/v1/sessions', json=dict(session_id='session'))
        ids, paths = stage(client, group)
        view, _ = send(client, ids)
        assert view['status'] == 'clarification'
        final = upload(client,'R3.fastq',paths['R3'].read_bytes(),token='R3').json()['resource_id']
        view, body = send(client,[final],turn='answer',utterance='The new file is R3 of the same library, lane 001 chunk 001. Continue.')
        assert view['status'] == 'clarification', view
        assert view['response']['clarification']['reason'] == 'missing_parameter_value', view
        state = service._application.sessions.load('session')
        assert state.interactions[-1].prerequisite['field'] == 'parameters'
        assert not state.revisions and not any(counts.values())
        assert client.post('/api/v1/sessions/session/turns',json=body).status_code == 202
        assert not any(counts.values())


def test_missing_layout_precedes_missing_read_and_partial_answer_retains_task(tiny, reads, monkeypatch):
    group, _ = reads([(100, 200, BC, 1)])
    registry, counts = instrument(monkeypatch)
    class LayoutModel(ChatModel):
        def complete(self, *, prompt, response_schema):
            value = json.loads(prompt)
            response = json.loads(super().complete(prompt=prompt, response_schema=response_schema))
            if response.get('decision', {}).get('kind') == 'request_roles':
                response['decision']['fastq']['layout'] = None
            elif value.get('utterance') == 'Use layout A with genomic R1/R3 and barcode R2.':
                response['decision']['fastq'] = raw(layout='tenx-atac-r1-r2-r3.v1')
            return json.dumps(response)
    service, uploads, models, app = chat_web(tiny['root'], registry=registry, model_type=LayoutModel)
    with TestClient(app) as client:
        client.post('/api/v1/sessions', json=dict(session_id='session'))
        ids, paths = stage(client, group)
        view, _ = send(client, ids)
        assert 'read layout' in view['response']['text'] and 'R3' not in view['response']['text']
        view, _ = send(client, [], turn='layout', utterance='Use layout A with genomic R1/R3 and barcode R2.')
        assert 'R3' in view['response']['text']
        final = upload(client,'R3.fastq',paths['R3'].read_bytes(),token='R3').json()['resource_id']
        view, _ = send(client,[final],turn='member',utterance='The new file is genomic R3, lane 001 chunk 001 of the same library.')
        assert 'must be provisioned' in view['response']['text'] and 'missing required FASTQ roles' not in view['response']['text']
        state = service._application.sessions.load('session')
        assert state.interactions[-1].prerequisite['origin_turn_id'] == 'first'
        assert state.interactions[-1].prerequisite['binding_turn_id'] == 'member'
        assert not state.revisions and not any(counts.values())
