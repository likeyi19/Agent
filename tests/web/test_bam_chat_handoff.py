"""Original chat BAM bytes, declared history, real owners and durable continuation."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from agent.application import InteractiveAgentApplication
from agent.application.raw_handoff import BamDeclaration
from agent.application.uploads import H5ADUploadAdmission
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas.orchestration import _serialize
from agent.tools.data import raw_scatac_manifest, scatac_library_context as library
from agent.web.app import create_app
from agent.web.config import ScientificInputSet
from test_bam_execution import BAMModel, PROFILE, downstream_bam_inputs, instrument
from test_chat_attachments import upload
from helpers import wait_turn

OBJECTIVE = ('Preprocess the first attached BAM from my human paired scATAC library sample. '
             'Perform QC, select candidates with minimum fragment records 0 and minimum TSS enrichment 0, '
             'and build a cell-by-cCRE matrix. The second attachment is unrelated notes.')
ANSWER = ('This sample was aligned to hg38. Its CB tags are already corrected cell-barcode identifiers. '
          'The paired ATAC alignment coordinates are unshifted and duplicate pairs were retained.')


class ChatBAMModel(BAMModel):
    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        if 'turn_schema_version' not in value:
            return super().complete(prompt=prompt, response_schema=response_schema)
        self.calls.append(value)
        if value['utterance'] == OBJECTIVE:
            arguments = []
            for name, prefix in [('min_qc_fragment_records', 'minimum fragment records '),
                                 ('min_tss_enrichment', 'minimum TSS enrichment ')]:
                start = OBJECTIVE.index(prefix) + len(prefix)
                arguments.append(dict(tool='select_scATAC_cells', argument=name, literal='0', start=start, end=start+1))
            decision = dict(kind='request_roles', target='prepare_scATAC_bam_fragments', required_roles=[], roles=[],
                arguments=arguments, bam=asdict(BamDeclaration(source='source.0', assay='SCATAC', namespace='sample', species='human')))
        else:
            decision = dict(kind='answer_roles', pending='@roles', roles=[], arguments=[],
                bam=asdict(BamDeclaration(assembly='hg38', barcode_policy='corrected_cb_already_corrected',
                                         source_profile='agent-cb-paired-atac.v1')))
        return json.dumps(dict(turn_schema_version=1, decision=decision))


def reusable(arguments):
    values = {k: v for k, v in arguments.items() if k in {
        'species', 'reference_bundle_path', 'reference_bundle_sha256', 'reference_manifest_path',
        'reference_manifest_sha256', 'qc_reference_manifest_path', 'qc_reference_manifest_sha256'}}
    values['species'] = 'human'
    return ScientificInputSet('synthetic', 'Synthetic test reference and QC resources', values,
                               bam_companion=True, bam_default=True)


def backend(root, companions=(), registry=None, model_type=ChatBAMModel):
    models = []
    def factory(profile):
        model = model_type('downstream'); models.append(model); return model
    service = InteractiveAgentApplication(root / 'workspace', model_profiles=(PROFILE,),
        default_profile_id=PROFILE.profile_id, approved_source_roots=(root,), registry=registry,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': factory}))
    uploads = H5ADUploadAdmission(service.resources,
        service._application._workspace._ensure_directory(root / 'workspace' / 'uploads'))
    return service, uploads, models, create_app(service, uploads=uploads, input_sets=companions)


def send(client, ids=(), *, turn='first', utterance=OBJECTIVE, generation=0):
    body = dict(turn_id=turn, expected_generation=generation, utterance=utterance, attachment_ids=list(ids))
    response = client.post('/api/v1/sessions/session/turns', json=body)
    assert response.status_code == 202, response.text
    return wait_turn(client, turn, timeout=120), body


def stage(client, path):
    client.post('/api/v1/sessions', json=dict(session_id='session'))
    return [upload(client, 'research.bam', path.read_bytes(), token='bam').json()['resource_id'],
            upload(client, 'notes.txt', b'unrelated notes', token='notes').json()['resource_id']]


def test_bam_chat_real_context_science_delivery_restart(tmp_path, monkeypatch, record_property):
    from functools import wraps
    from time import perf_counter
    from agent.tools.data import raw_scatac
    from agent.orchestration import raw_scatac_verifier, verifier
    args = downstream_bam_inputs(tmp_path)
    companion = reusable(args)
    timings = dict(intake=[], verification=[], context=[])
    for module, name, key in ((raw_scatac, 'inspect_raw_scATAC', 'intake'),
                              (raw_scatac_verifier, 'verify_raw_scatac_intake', 'verification'),
                              (verifier, 'verify_raw_scatac_intake', 'verification'),
                              (library, 'build_scatac_library_processing_context', 'context')):
        original = getattr(module, name)
        def wrap(original, key):
            @wraps(original)
            def call(*a, **kw):
                start = perf_counter()
                try: return original(*a, **kw)
                finally: timings[key].append(perf_counter()-start)
            return call
        monkeypatch.setattr(module, name, wrap(original, key))
    registry, counts = instrument(monkeypatch)
    monkeypatch.setenv('AGENT_QC_BEDTOOLS', '/usr/bin/bedtools')
    monkeypatch.setenv('AGENT_MATRIX_BEDTOOLS', '/usr/bin/bedtools')
    monkeypatch.setenv('AGENT_QC_ALLOW_SYNTHETIC', '1')
    monkeypatch.delenv('AGENT_QC_RESOURCE_CATALOG', raising=False)
    service, uploads, models, app = backend(tmp_path, (companion,), registry)
    with TestClient(app) as client:
        ids = stage(client, Path(args['source_path']))
        original = service.resources.load(ids[0]); path = Path(original.source_path)
        before = (path.stat().st_ino, path.stat().st_blocks, path.read_bytes())
        question, first_body = send(client, ids)
        assert question['status'] == 'clarification', question
        assert all(word in question['response']['text'] for word in ('hg38', 'corrected', 'unshifted', 'duplicate'))
        assert not any(counts.values()) and not any(timings.values())
        assert not (tmp_path / 'workspace' / 'bam_handoffs').exists()
        # Recreate the facade with the same pending Session before answering.
    service, uploads, models, app = backend(tmp_path, (companion,), registry)
    with TestClient(app) as client:
        assert client.get('/api/v1/sessions/session').status_code == 200
        view, answer_body = send(client, turn='answer', utterance=ANSWER)
        record_property('owner_timings', json.dumps(timings))
        assert view['status'] == 'succeeded', view
        assert [a['use'] for a in view['attachments']] == ['consumed', 'staged']
        assert service.resources.load(ids[0]) == original
        assert before == (path.stat().st_ino, path.stat().st_blocks, path.read_bytes())
        state = service._application.sessions.load('session')
        assert state.generation == 1 and len(state.revisions) == 1
        admitted = _serialize(state.interactions[-1].admitted)
        receipt = admitted['bam_context']
        assert admitted['bam']['barcode_policy'] == 'corrected_cb_already_corrected'
        assert admitted['inputs']['raw_input_paths'] == [dict(path=str(path), input_kind='BAM')]
        assert all(p.suffix == '.json' for p in (tmp_path/'workspace'/'bam_handoffs').rglob('*') if p.is_file())
        run = service._application.run_store.load(view['run_id'])
        assert [s.tool_name for s in run.steps] == ['inspect_raw_scATAC', 'prepare_scATAC_bam_fragments',
            'compute_scATAC_qc', 'select_scATAC_cells', 'build_scATAC_cell_by_ccre']
        assert run.request.prompt == OBJECTIVE
        intake, fragments, qc, selection, matrix = run.steps
        assert (matrix.result['n_cells'], matrix.result['n_features'], matrix.result['nnz'], matrix.result['total_count']) == (1, 2, 1, 2)
        assert matrix.result['logical_matrix_sha256'] == '5b621373bb2175896becdd9f0f8e8c90eab8755e978f071d460c779a0a16d41f'
        assert receipt['intake_sha256'] == intake.result['manifest_sha256']
        _, context, _ = library.load_scatac_library_processing_context(receipt['context_path'])
        assert context.intake.manifest_sha256 == receipt['intake_sha256']
        assert context.libraries[0].namespace == 'sample'
        assert context.libraries[0].barcode_interpretation is library.BarcodeInterpretation.CORRECTED_IDENTIFIER
        assert fragments.resolved_arguments['source_path'] == str(path)
        assert fragments.resolved_arguments['source_sha256'] == original.source_sha256
        for step in run.steps[1:]:
            assert step.verification.passed and step.verification.artifact_authority['schema_version'] == 2
        assert fragments.result['n_fragment_records'] == fragments.result['total_support'] == 2
        route = '/api/v1/sessions/session/revisions/' + view['revision_id']
        artifacts = client.get(route + '/scientific-artifacts').json()['artifacts']
        assert {a['filename'] for a in artifacts} == {'matrix.h5ad','barcodes.tsv.gz','lengths.tsv.gz','decisions.tsv.gz','selected.tsv.gz'}
        for a in artifacts:
            data = client.get(route + '/scientific-artifacts/' + a['handle'])
            assert data.status_code == 200 and hashlib.sha256(data.content).hexdigest() == a['sha256']
        reports = client.get(route + '/artifacts').json()['artifacts']
        report, = [a for a in reports if a['artifact_type'] == 'analysis_report']
        assert client.get(route + '/artifacts/' + report['handle']).status_code == 200
        assert counts == dict(intake=1, production=1, verification=1)
        observed = {k: len(v) for k, v in timings.items()}
        assert observed['intake'] == observed['context'] == 1
        assert observed['verification'] >= 2
        assert client.post('/api/v1/sessions/session/turns', json=answer_body).status_code == 202
        assert wait_turn(client, 'answer') == view
        assert {k: len(v) for k, v in timings.items()} == observed
        (tmp_path/'witness.json').write_text(json.dumps(dict(view=view, first=first_body, answer=answer_body,
            companion=companion.definition(), source=original.source_path, timings=timings), indent=2))
    def forbidden(*a, **kw): raise AssertionError('History must not construct providers or science.')
    fresh = InteractiveAgentApplication(tmp_path/'workspace', model_profiles=(PROFILE,), default_profile_id=PROFILE.profile_id,
        approved_source_roots=(tmp_path,), registry=registry,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': forbidden}))
    with TestClient(create_app(fresh, uploads=H5ADUploadAdmission(fresh.resources, uploads.root), input_sets=(companion,))) as client:
        assert client.get('/api/v1/sessions/session').status_code == 200
        assert client.get('/api/v1/sessions/session/turns/answer').json() == view
        assert client.post('/api/v1/sessions/session/turns', json=answer_body).status_code == 202
        assert wait_turn(client, 'answer') == view
        assert counts == dict(intake=1, production=1, verification=1)


@pytest.mark.parametrize('failure', ['missing_resource', 'mutation', 'stale', 'foreign'])
def test_bam_pending_fail_closed(tmp_path, monkeypatch, failure):
    args = downstream_bam_inputs(tmp_path)
    registry, counts = instrument(monkeypatch)
    service, uploads, models, app = backend(tmp_path, () if failure == 'missing_resource' else (reusable(args),), registry)
    with TestClient(app) as client:
        ids = stage(client, Path(args['source_path']))
        question, _ = send(client, ids)
        assert question['status'] == 'clarification'
        if failure == 'mutation': Path(service.resources.load(ids[0]).source_path).write_bytes(b'changed')
        if failure == 'foreign':
            client.post('/api/v1/sessions', json=dict(session_id='other'))
            ids = [upload(client,'foreign.bam', b'foreign', session='other').json()['resource_id']]
        response = client.post('/api/v1/sessions/session/turns', json=dict(turn_id='answer',
            expected_generation=1 if failure == 'stale' else 0, utterance=ANSWER,
            attachment_ids=ids if failure == 'foreign' else []))
        if failure in ('stale', 'foreign'):
            assert response.status_code in (400, 404, 409)
        else:
            assert response.status_code == 202, response.text
            view = wait_turn(client, 'answer')
            assert view['status'] in ('failed', 'clarification') and view['revision_id'] is None
        assert not any(counts.values())
        assert not service._application.sessions.load('session').revisions


@pytest.mark.parametrize('changes', [dict(source='source.1'), dict(assembly='mm10'), dict(resource='resource.forged')])
def test_bam_answer_cannot_replace_captured_scientific_declarations(tmp_path, monkeypatch, changes):
    class Changed(ChatBAMModel):
        def complete(self, *, prompt, response_schema):
            value=json.loads(super().complete(prompt=prompt,response_schema=response_schema))
            if value.get('decision',{}).get('kind')=='answer_roles':value['decision']['bam'].update(changes)
            return json.dumps(value)
    args=downstream_bam_inputs(tmp_path);registry,counts=instrument(monkeypatch)
    service,uploads,models,app=backend(tmp_path,(reusable(args),),registry,Changed)
    with TestClient(app) as client:
        ids=stage(client,Path(args['source_path']))
        assert send(client,ids)[0]['status']=='clarification'
        view,_=send(client,turn='answer',utterance=ANSWER)
        assert view['status'] in ('failed','clarification') and view['revision_id'] is None
        assert not any(counts.values()) and not service._application.sessions.load('session').revisions
