"""Scripted language decisions; actual uploaded bytes, owners and scientific outputs."""
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pytest
from fastapi.testclient import TestClient

from agent.application.raw_handoff import FragmentsDeclaration
from agent.schemas.orchestration import _serialize
from agent.tools.data import regulatory_feature_reference
from agent.web.config import ScientificInputSet
from test_external_fragments_execution import downstream_inputs, web_fragments, DownstreamModel
from test_chat_attachments import upload
from test_bam_chat_handoff import send

OBJECTIVE = ('Use my external human hg38 fragments from library sample, with 10x ATAC adjusted '
             'zero-based half-open intervals, corrected barcode identifiers and read-pair support. '
             'Perform QC, select with minimum fragment records 0 and minimum TSS enrichment 0, '
             'and construct a cell-by-peak matrix using my custom BED in the same assembly and convention. '
             'The first attachment is fragments; the second is my BED; notes are unrelated.')
MISSING_BED_OBJECTIVE = OBJECTIVE.replace(
    'The first attachment is fragments; the second is my BED; notes are unrelated.',
    'The first attachment is fragments; the second is unrelated notes. I will supply my BED later.')
BED = b'chr2\t3000\t3010\nchr2\t1000\t1001\nchr1\t500\t600\n'


class ChatModel(DownstreamModel):
    missing_bed = False
    objective = OBJECTIVE
    make_matrix = True
    source_handle = 'source.0'
    feature_handle = 'source.1'
    index_handle = None

    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        if 'turn_schema_version' not in value:
            result = json.loads(super().complete(prompt=prompt, response_schema=response_schema))
            if result.get('schema_version') == 4:
                for step in result['decision']['steps']:
                    if step['tool'] == 'build_scATAC_cell_by_ccre':
                        step['tool'] = 'build_scATAC_cell_by_features'
            return json.dumps(result)
        self.calls.append(value)
        if value['utterance'] == self.objective:
            arguments = []
            for name, prefix in [('min_qc_fragment_records', 'minimum fragment records '),
                                 ('min_tss_enrichment', 'minimum TSS enrichment ')] if self.make_matrix else []:
                start = self.objective.index(prefix) + len(prefix)
                arguments.append(dict(tool='select_scATAC_cells', argument=name, literal='0', start=start, end=start+1))
            declared = FragmentsDeclaration(source=self.source_handle, index_source=self.index_handle,
                species='human', assembly='hg38', namespace='sample', source_profile='10x-atac-fragments.v1',
                feature_source=None if self.missing_bed or not self.make_matrix else self.feature_handle,
                feature_category='peak_set' if self.make_matrix else None)
            decision = dict(kind='request_roles', target='import_scATAC_fragments', required_roles=[], roles=[],
                fragments=asdict(declared), arguments=arguments)
        else:
            decision = dict(kind='answer_roles', pending='@roles', roles=[], arguments=[],
                fragments=asdict(FragmentsDeclaration(feature_source='source.2')))
        return json.dumps(dict(turn_schema_version=1, decision=decision))


def companion(args):
    return ScientificInputSet('reviewed', 'Simulated qualified reference and TSS',
        {k: v for k, v in args.items() if k.startswith(('reference_', 'qc_reference_'))},
        fragments_companion=True)


def configured(tmp_path, monkeypatch, model=ChatModel):
    args = downstream_inputs(tmp_path)
    monkeypatch.setenv('AGENT_QC_BEDTOOLS', '/usr/bin/bedtools')
    monkeypatch.setenv('AGENT_MATRIX_BEDTOOLS', '/usr/bin/bedtools')
    monkeypatch.setenv('AGENT_QC_ALLOW_SYNTHETIC', '1')
    monkeypatch.delenv('AGENT_QC_RESOURCE_CATALOG', raising=False)
    return args, web_fragments(tmp_path, args, companions=(companion(args),), model_type=model)


def stage(client, args, *, bed=BED, encoding='plain', missing=False):
    client.post('/api/v1/sessions', json=dict(session_id='session'))
    data = Path(args['source_path']).read_bytes()
    if encoding == 'gzip':
        data = gzip.compress(data, mtime=0)
    ids = [upload(client, 'fragments.tsv.gz' if encoding == 'gzip' else 'fragments.tsv', data, token='fragments').json()['resource_id']]
    if not missing:
        ids.append(upload(client, 'peaks.bed', bed, token='bed').json()['resource_id'])
    ids.append(upload(client, 'notes.txt', b'unrelated notes', token='notes').json()['resource_id'])
    return ids


@pytest.mark.parametrize('supplement', [False, True])
def test_external_fragments_custom_bed_qualified_execution_and_continuation(tmp_path, monkeypatch, supplement):
    from rich_helpers import ScientificWork
    class Model(ChatModel):
        missing_bed = supplement
        objective = MISSING_BED_OBJECTIVE if supplement else OBJECTIVE
    args, backend = configured(tmp_path, monkeypatch, Model)
    work = ScientificWork()
    try:
        with TestClient(backend.app) as client:
            ids = stage(client, args, encoding='gzip', missing=supplement)
            originals = [backend.service.resources.load(i) for i in ids]
            before = [(Path(r.source_path).stat().st_ino, Path(r.source_path).read_bytes()) for r in originals]
            first, body = send(client, ids, utterance=Model.objective)
            if supplement:
                assert first['status'] == 'clarification', (first, backend.errors)
                assert 'BED' in first['response']['text'] and not work.snapshot()
                state = backend.service._application.sessions.load('session')
                assert state.interactions[0].admitted['raw_resource_choice'] == 'resource.reviewed'
                assert not state.turns and not state.revisions
            else:
                view = first
        if supplement:
            backend = web_fragments(tmp_path, args, companions=(companion(args),), model_type=Model)
            with TestClient(backend.app) as client:
                assert client.get('/api/v1/sessions/session').status_code == 200
                bed = upload(client, 'peaks.bed', BED, token='bed').json()['resource_id']
                view, body = send(client, [bed], turn='answer', utterance='The new attachment is my custom peaks BED, in hg38 zero-based half-open coordinates.')
        assert view['status'] == 'succeeded', (view, backend.errors)
        assert [a['use'] for a in view['attachments']] == (['consumed', 'staged', 'consumed'] if supplement else ['consumed', 'consumed', 'staged'])
        assert [(Path(r.source_path).stat().st_ino, Path(r.source_path).read_bytes()) for r in originals] == before
        state = backend.service._application.sessions.load('session')
        admitted = _serialize(state.interactions[-1].admitted)
        run = backend.service._application.run_store.load(view['run_id'])
        assert run.request.prompt == Model.objective
        assert [s.tool_name for s in run.steps] == ['import_scATAC_fragments', 'compute_scATAC_qc', 'select_scATAC_cells', 'build_scATAC_cell_by_features']
        imported, qc, selected, matrix = run.steps
        assert all(s.verification.passed and s.verification.artifact_authority['schema_version'] == 2 for s in run.steps)
        assert imported.resolved_arguments['source_path'] == originals[0].source_path
        assert imported.result['source_encoding'] == 'gzip'
        assert imported.result['n_fragment_records'] == 3 and imported.result['total_support'] == 6
        assert selected.resolved_arguments['min_qc_fragment_records'] == 0
        assert selected.resolved_arguments['min_tss_enrichment'] == 0
        pointer = admitted['fragments_context']['feature_reference']
        _, reference, _ = regulatory_feature_reference.load_regulatory_feature_reference(pointer['manifest_path'], expected_sha256=pointer['manifest_sha256'])
        assert reference.features.category == 'peak_set'
        feature_member = admitted['fragments_sources']['members'][2 if supplement else 1]
        assert reference.features.bed.path == feature_member['source_path']
        assert reference.features.bed.sha256 == hashlib.sha256(BED).hexdigest()
        assert matrix.resolved_arguments['reference_manifest_sha256'] == pointer['manifest_sha256']
        assert matrix.resolved_arguments['fragments_manifest_sha256'] == imported.result['manifest_sha256']
        assert matrix.resolved_arguments['selected_cells_manifest_sha256'] == selected.result['manifest_sha256']
        data = ad.read_h5ad(matrix.result['matrix_path'])
        assert data.var_names.tolist() == ['chr2:3000-3010', 'chr2:1000-1001', 'chr1:500-600']
        assert data.obs['barcode_identifier'].tolist() == ['A']
        np.testing.assert_array_equal(data.X.toarray(), [[1, 1, 0]])
        assert data.X.dtype == np.int64 and matrix.result['n_features'] == 3
        assert data.uns['matrix_semantics'] == 'fragment_counts'
        from dataclasses import replace
        from agent.application.dialogue_execution import execution_submission
        from agent.application.local_resources import ResourceAdmissionError
        substituted = replace(matrix, resolved_arguments=dict(matrix.resolved_arguments,
            reference_manifest_path=args['reference_manifest_path'],
            reference_manifest_sha256=args['reference_manifest_sha256']))
        with pytest.raises(ResourceAdmissionError):
            backend.service.resources.validate_result(execution_submission(state.interactions[-1]),
                (*run.steps[:-1], substituted))

        counts = work.snapshot()
        # Shared existing engine remains independently reconstructed.
        assert counts['production.fragments'] == counts['owner.fragments'] == 1
        assert counts['production.qc'] == counts['owner.qc'] == 1
        assert counts['production.selection'] == counts['owner.selection'] == 1
        assert counts['production.matrix'] == counts['owner.matrix'] == 1
        with TestClient(backend.app) as client:
            route = '/api/v1/sessions/session/revisions/' + view['revision_id']
            artifacts = client.get(route + '/scientific-artifacts').json()['artifacts']
            assert {a['filename'] for a in artifacts} == {'matrix.h5ad', 'barcodes.tsv.gz', 'lengths.tsv.gz', 'decisions.tsv.gz', 'selected.tsv.gz'}
            for artifact in artifacts:
                delivered = client.get(route + '/scientific-artifacts/' + artifact['handle'])
                assert delivered.status_code == 200 and hashlib.sha256(delivered.content).hexdigest() == artifact['sha256']
            report, = [a for a in client.get(route + '/artifacts').json()['artifacts'] if a['artifact_type'] == 'analysis_report']
            assert client.get(route + '/artifacts/' + report['handle']).status_code == 200
            assert client.post('/api/v1/sessions/session/turns', json=body).status_code == 202
        Path(originals[0].source_path).unlink()
        Path(feature_member['source_path']).unlink()
        # Terminal reads after restart cannot construct providers or science.
        class Forbidden(Model):
            def __init__(self, *a, **kw):
                raise AssertionError('Historical read constructed a provider.')
        fresh = web_fragments(tmp_path, args, companions=(companion(args),), model_type=Forbidden)
        with TestClient(fresh.app) as client:
            assert client.get('/api/v1/sessions/session').status_code == 200
            assert client.get('/api/v1/sessions/session/turns/' + body['turn_id']).json() == view
        assert work.snapshot() == counts
    finally:
        work.close()


@pytest.mark.parametrize('bad_bed', [b'chr2\t0\t6002\n', BED + BED.splitlines(keepends=True)[0], b'2\t0\t10\n'])
def test_invalid_custom_bed_never_falls_back_to_configured_ccres(tmp_path, monkeypatch, bad_bed):
    args, backend = configured(tmp_path, monkeypatch)
    with TestClient(backend.app) as client:
        ids = stage(client, args, bed=bad_bed)
        view, _ = send(client, ids, utterance=OBJECTIVE)
        assert view['status'] != 'succeeded' and view['run_id'] is None
        state = backend.service._application.sessions.load('session')
        assert not state.revisions and all(t.run_id is None for t in state.turns)
        assert view['response']['clarification']['reason'] == 'invalid_reference'
        assert not list((tmp_path / 'workspace' / 'fragments_handoffs').rglob('reference.json'))


def test_invalid_fragment_records_fail_independent_scientific_admission(tmp_path, monkeypatch):
    args, backend = configured(tmp_path, monkeypatch)
    Path(args['source_path']).write_text('chr2\t100\t200\tA\t0\n')
    with TestClient(backend.app) as client:
        ids = stage(client, args)
        view, _ = send(client, ids, utterance=OBJECTIVE)
        assert view['status'] == 'failed' and view['error']['code'] == 'EXTERNAL_FRAGMENTS_RECORD_INVALID'
        assert not backend.service._application.sessions.load('session').revisions


@pytest.mark.parametrize('mutation', ['source', 'reference'])
def test_pending_task_rejects_changed_sources_and_pinned_reference(tmp_path, monkeypatch, mutation):
    class Model(ChatModel):
        missing_bed = True
        objective = MISSING_BED_OBJECTIVE
    args, backend = configured(tmp_path, monkeypatch, Model)
    with TestClient(backend.app) as client:
        ids = stage(client, args, missing=True)
        view, _ = send(client, ids, utterance=Model.objective)
        assert view['status'] == 'clarification'
        path = (Path(backend.service.resources.load(ids[0]).source_path) if mutation == 'source'
                else Path(args['reference_bundle_path']))
        path.write_bytes(b'changed')
        bed = upload(client, 'peaks.bed', BED, token='bed').json()['resource_id']
        view, _ = send(client, [bed], turn='answer', utterance='Use the new BED for my preserved task.')
        assert view['status'] != 'succeeded' and view['run_id'] is None
        state = backend.service._application.sessions.load('session')
        assert not state.revisions
        if mutation == 'reference':
            assert 'selected' in view['response']['text'] or 'qualified' in view['response']['text']
            assert state.interactions[-1].admitted['raw_resource_choice'] == 'resource.reviewed'


def test_foreign_supplement_is_rejected_before_interpretation(tmp_path, monkeypatch):
    class Model(ChatModel):
        missing_bed = True
        objective = MISSING_BED_OBJECTIVE
    args, backend = configured(tmp_path, monkeypatch, Model)
    with TestClient(backend.app) as client:
        ids = stage(client, args, missing=True)
        view, _ = send(client, ids, utterance=Model.objective)
        assert view['status'] == 'clarification'
        client.post('/api/v1/sessions', json=dict(session_id='other'))
        foreign = upload(client, 'private.bed', BED, token='private', session='other').json()['resource_id']
        before = len(backend.models)
        response = client.post('/api/v1/sessions/session/turns', json=dict(turn_id='answer', expected_generation=0,
            utterance='Use this BED', attachment_ids=[foreign]))
        assert response.status_code == 404 and len(backend.models) == before
        assert not backend.service._application.sessions.load('session').revisions


class ImportOnlyModel(ChatModel):
    """An explicit one-tool plan; no automatic downstream completion."""
    objective = ('Import my independent human hg38 fragments from library sample. '
                 'These are 10x ATAC adjusted zero-based half-open intervals, corrected '
                 'barcodes and read-pair support. The first attachment is fragments; '
                 'use the second as its source index if attached. No downstream analysis is requested.')
    make_matrix = False
    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        result = json.loads(super().complete(prompt=prompt, response_schema=response_schema))
        if 'turn_schema_version' in value:
            result['decision']['fragments'].update(feature_source=None, feature_category=None)
        elif 'output_selection_schema_version' in value:
            result = dict(outputs=[dict(name='fragments', step_id='import', output_key='manifest_path')])
        elif result.get('schema_version') == 4:
            result['decision']['steps'] = result['decision']['steps'][:1]
            if self.index_handle is not None:
                result['decision']['steps'][0]['sources'].append(dict(target='source_index',
                    source=dict(kind='input', input='source_index_path')))
        return json.dumps(result)


@pytest.mark.parametrize('indexed', [False, True])
def test_bgzf_external_source_optional_index_and_no_agent_ancestry(tmp_path, monkeypatch, indexed):
    import pysam
    class Model(ImportOnlyModel):
        index_handle = 'source.1' if indexed else None
    args, backend = configured(tmp_path, monkeypatch, Model)
    # The existing owner accepts BGZF directly; an optional TBI is explicit.
    raw = tmp_path / 'ordered.tsv'
    raw.write_text('chr2\t1000\t1001\tA\t2\nchr2\t2950\t3050\tA\t1\n')
    compressed = tmp_path / 'ordered.tsv.gz'
    pysam.tabix_compress(str(raw), str(compressed))
    pysam.tabix_index(str(compressed), preset='bed')
    with TestClient(backend.app) as client:
        client.post('/api/v1/sessions', json=dict(session_id='session'))
        ids = [upload(client, 'source.tsv.gz', compressed.read_bytes(), token='source').json()['resource_id']]
        if indexed:
            ids.append(upload(client, 'source.tsv.gz.tbi', Path(str(compressed) + '.tbi').read_bytes(), token='index').json()['resource_id'])
        view, _ = send(client, ids, utterance=Model.objective)
        assert view['status'] == 'succeeded', (view, backend.errors)
        run = backend.service._application.run_store.load(view['run_id'])
        step, = run.steps
        assert step.tool_name == 'import_scATAC_fragments' and step.result['source_encoding'] == 'bgzf'
        assert step.result['n_fragment_records'] == 2 and step.result['total_support'] == 3
        assert bool(step.resolved_arguments.get('source_index_path')) == indexed
        assert all(a['use'] == 'consumed' for a in view['attachments'])
        assert not (tmp_path / 'workspace' / 'bam_handoffs').exists()
        assert not (tmp_path / 'workspace' / 'fastq_handoffs').exists()


class GenomeModel(ImportOnlyModel):
    objective = ('Import my external human hg38 fragments from library sample. These are 10x ATAC '
                 'adjusted zero-based half-open intervals, corrected barcodes and read-pair support. '
                 'The first attachment is fragments and the second is my peak BED in the same assembly '
                 'and coordinate convention. Qualify this source with my reference; do not run QC or select cells.')
    supplied_fai = False

    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        result = json.loads(super().complete(prompt=prompt, response_schema=response_schema))
        if 'turn_schema_version' in value:
            if value['utterance'] == self.objective:
                result['decision']['fragments'].update(feature_source='source.1', feature_category='peak_set')
            else:
                result['decision']['fragments'] = asdict(FragmentsDeclaration(genome_source='source.3',
                    genome_index_source='source.4' if self.supplied_fai else None))
        return json.dumps(result)


@pytest.mark.parametrize('fai', [False, True])
def test_missing_genome_fasta_supplement_qualifies_and_resumes_real_import(tmp_path, monkeypatch, fai):
    class Model(GenomeModel):
        supplied_fai = fai
    args = downstream_inputs(tmp_path)
    backend = web_fragments(tmp_path, args, companions=(), model_type=Model)
    with TestClient(backend.app) as client:
        ids = stage(client, args)
        question, _ = send(client, ids, utterance=Model.objective)
        assert question['status'] == 'clarification' and 'plain genome FASTA' in question['response']['text']
        assert not backend.service._application.sessions.load('session').turns
    backend = web_fragments(tmp_path, args, companions=(), model_type=Model)
    fasta = tmp_path / 'scientific-inputs' / 'reference.fa'
    with TestClient(backend.app) as client:
        assert client.get('/api/v1/sessions/session').status_code == 200
        extra = [upload(client, 'genome.fa', fasta.read_bytes(), token='genome').json()['resource_id']]
        if fai:
            extra.append(upload(client, 'genome.fa.fai', Path(str(fasta) + '.fai').read_bytes(), token='fai').json()['resource_id'])
            monkeypatch.setattr('pysam.faidx', lambda *a, **k: pytest.fail('Supplied FAI must not be rebuilt.'))
        view, _ = send(client, extra, turn='answer', utterance='The new FASTA is my human hg38 genome; use its supplied FAI if present.')
        assert view['status'] == 'succeeded', (view, backend.errors)
        state = backend.service._application.sessions.load('session')
        admitted = _serialize(state.interactions[-1].admitted)
        assert admitted['raw_resource_choice'] is None and admitted['resource_catalog'] == []
        run = backend.service._application.run_store.load(view['run_id'])
        assert run.request.prompt == Model.objective
        step, = run.steps
        assert step.tool_name == 'import_scATAC_fragments'
        assert step.verification.passed and step.verification.artifact_authority['schema_version'] == 2
        assert step.result['species'] == {'scientific_name': 'Homo sapiens', 'taxonomy_id': 9606}
        assert step.result['n_fragment_records'] == 3 and step.result['total_support'] == 6
        pointer = admitted['fragments_context']['feature_reference']
        _, reference, _ = regulatory_feature_reference.load_regulatory_feature_reference(pointer['manifest_path'], expected_sha256=pointer['manifest_sha256'])
        assert reference.features.category == 'peak_set'
        assert reference.genome.fasta.path == backend.service.resources.load(extra[0]).source_path
        assert reference.features.bed.path == backend.service.resources.load(ids[1]).source_path
        if fai:
            assert reference.genome.fai.path == backend.service.resources.load(extra[1]).source_path
        else:
            assert Path(reference.genome.fai.path).parent == Path(pointer['manifest_path']).parent
        assert [a['use'] for a in view['attachments']] == ['consumed', 'consumed', 'staged', 'consumed'] + (['consumed'] if fai else [])
        assert client.get('/api/v1/resources').json()['choices'] == []
        route = '/api/v1/sessions/session/revisions/' + view['revision_id']
        report, = [a for a in client.get(route + '/artifacts').json()['artifacts'] if a['artifact_type'] == 'analysis_report']
        response = client.get(route + '/artifacts/' + report['handle'])
        assert response.status_code == 200 and str(tmp_path) not in response.text


@pytest.mark.parametrize('bad', ['compressed_fasta', 'fai'])
def test_invalid_genome_supplement_is_not_qualified_or_promoted(tmp_path, monkeypatch, bad):
    class Model(GenomeModel):
        supplied_fai = bad == 'fai'
    args = downstream_inputs(tmp_path)
    backend = web_fragments(tmp_path, args, companions=(), model_type=Model)
    with TestClient(backend.app) as client:
        ids = stage(client, args)
        view, _ = send(client, ids, utterance=Model.objective)
        assert view['status'] == 'clarification'
        data = (tmp_path / 'scientific-inputs' / 'reference.fa').read_bytes()
        if bad == 'compressed_fasta':
            data = gzip.compress(data)
        ids = [upload(client, 'genome.fa', data, token='genome').json()['resource_id']]
        if bad == 'fai':
            ids.append(upload(client, 'genome.fa.fai', b'chr2\t6001\t6\t100\t101\n', token='fai').json()['resource_id'])
        view, _ = send(client, ids, turn='answer', utterance='Use my genome and its FAI if supplied.')
        assert view['status'] == 'clarification' and view['run_id'] is None
        assert view['response']['clarification']['reason'] == 'invalid_reference'
        assert not backend.service._application.sessions.load('session').revisions
        assert not list((tmp_path / 'workspace' / 'fragments_handoffs').rglob('reference.json'))
        assert client.get('/api/v1/resources').json()['choices'] == []
