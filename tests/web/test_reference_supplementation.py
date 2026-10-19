"""Reference supplementation uses existing owners and the original pending turn."""
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from fastapi.testclient import TestClient
import helpers
from helpers import submission, wait_turn
from test_h5ad_composition_api import web_composition
from test_upload_api import attach
from agent.application.input_sets import ScientificInputSet
from agent.application import reference_supplementation as refs
from agent.schemas.orchestration import _serialize


def install_shared(resources, record):
    from agent.application.shared_resources import ScientificResourceLibrary, install_private
    if resources.library is None:
        resources.library = ScientificResourceLibrary(resources._workspace.root.parent / 'scientific-library')
    return install_private(resources, record.reference_qualification['session_id'], record.resource_id, shared_use=True)


@pytest.fixture
def reference_web(tmp_path, monkeypatch):
    class Model(helpers.ScriptedModel):
        def complete(self, *, prompt, response_schema):
            p = json.loads(prompt); self.calls.append(p)
            if 'turn_schema_version' in p:
                pending = p['context']['dialogue'].get('pending_prerequisite') if 'context' in p else p['dialogue'].get('pending_prerequisite')
                d = (dict(kind='answer_resources', pending='@resources', choice=pending['choices'][0]['handle'], arguments=[])
                     if p['utterance'].startswith('Use') and pending and pending['choices'] else
                     dict(kind='request_resources', target='adapt_epizoo_species', arguments=[], reference_category='peak_set'))
                return json.dumps(dict(turn_schema_version=1, decision=d))
            if 'selection_schema_version' in response_schema.get('properties', {}):
                return json.dumps(dict(selection_schema_version=1, decision=dict(kind='select', capability_ids=['species_adaptation'])))
            if 'output_selection_schema_version' in p:
                return json.dumps(dict(outputs=[dict(name='model',step_id='adapt',output_key='manifest_path')]))
            return json.dumps(dict(schema_version=4,decision=dict(kind='plan',steps=[
                dict(step_id='adopt',tool='adopt_scATAC_cell_by_features',sources=[],control_dependencies=[]),
                dict(step_id='adapt',tool='adapt_epizoo_species',sources=[dict(kind='step',target='matrix',step='adopt')],control_dependencies=[])])))
    monkeypatch.setattr(helpers,'ScriptedModel',Model)
    declarations = ScientificInputSet('fish','Test fish declarations',dict(species=dict(scientific_name='Danio rerio',taxonomy_id=7955),
        assembly='danRer11',matrix_semantics='binary_accessibility',strategy='de_novo',
        adaptation_spec_path='/operator/spec.json',adaptation_spec_sha256='a'*64),h5ad_companion=True)
    backend, uploads, app = web_composition(tmp_path,companions=(declarations,))
    ad.AnnData(sparse.csr_matrix(np.array([[1,0],[0,1]],dtype=np.int64)),
        obs=pd.DataFrame(index=['a','b']),var=pd.DataFrame(index=['chr1:0-4','chr1:4-8'])).write_h5ad(backend.source)
    admitted=[]
    from agent.orchestration.executor import PlanExecutor
    def stop(self, plan, **kwargs):
        check=self.preflight(plan)
        assert check.passed, check
        admitted.append(plan)
        raise RuntimeError('Acceptance stop after normal preflight; no training authorized.')
    monkeypatch.setattr(PlanExecutor,'execute',stop)
    with TestClient(app, raise_server_exceptions=False) as client:
        source=attach(client,backend.source.read_bytes()).json()
        def start(session='session'):
            client.post('/api/v1/sessions',json={'session_id':session})
            body=submission(resource_id=source['resource_id'],input_set_id='fish',utterance='Adapt EpiZoo to this Danio rerio danRer11 peak matrix using de_novo.')
            r=client.post(f'/api/v1/sessions/{session}/turns',json=body);assert r.status_code==202,r.text
            r=wait_turn(client,session=session)
            assert r['status']=='clarification',r
            return client.get(f'/api/v1/sessions/{session}/reference-prerequisite').json()['pending']
        yield client, backend, uploads, source, start, admitted


def upload(client, data=b'>chr1\nACGTACGT\n', index=None):
    params=dict(filename='genome.fa',input_type='fasta')
    if index is not None:params.update(index_filename='genome.fa.fai',source_size=str(len(data)))
    r=client.put('/api/v1/uploads/genome',params=params,content=data+(index or b''),headers={'content-type':'application/octet-stream'})
    assert r.status_code==201,r.text
    return r.json()


def qualify(client,pending,record,session='session'):
    return client.post(f'/api/v1/sessions/{session}/reference-prerequisite',json=dict(resource_id=record['resource_id'],
        expected_generation=pending['expected_generation'],origin_turn_id=pending['origin_turn_id']))


def test_generate_qualify_continue_share_and_reopen(reference_web,monkeypatch):
    client,backend,uploads,source,start,admitted=reference_web
    pending=start();assert not pending['choices']
    from agent.tools.data import regulatory_feature_reference as owner
    original=owner.build_regulatory_feature_reference;calls=[]
    def counted(**kwargs):calls.append(1);return original(**kwargs)
    monkeypatch.setattr(owner,'build_regulatory_feature_reference',counted)
    record=upload(client)
    r=qualify(client,pending,record);assert r.status_code==200,r.text
    private=uploads.resources.load(r.json()['resource_id']);assert len(calls)==1
    choices=client.get('/api/v1/sessions/session/references').json()['choices']
    assert choices == [dict(resource_id=private.resource_id, label=private.label)]
    assert qualify(client,pending,record).json()==r.json();assert len(calls)==1
    assert not refs.choices(uploads.resources,'another',private.reference_qualification['requirement'])
    next_pending=start('second');assert not next_pending['choices']
    assert client.get('/api/v1/sessions/second/references').json()['choices'] == []
    promoted=install_shared(uploads.resources,private)
    assert promoted['resource']['genome']['fasta']['sha256'] == refs.check_record(uploads.resources,private).genome.fasta.sha256
    assert Path(promoted['resource']['genome']['fasta']['path']).is_relative_to(uploads.resources.library.root)
    assert client.get('/api/v1/sessions/second/reference-prerequisite').json()['pending']['choices']
    body=submission(turn='answer',resource_id=source['resource_id'],input_set_id='fish',utterance='Use the qualified reference and continue.')
    assert client.post('/api/v1/sessions/session/turns',json=body).status_code==202
    result=wait_turn(client,'answer');assert result['status']=='failed',result
    assert len(admitted)==1
    state=backend.service._application.sessions.load('session')
    resolved=state.interactions[-1].admitted
    assert resolved['inputs']['input_path']==uploads.resources.load(source['resource_id']).source_path
    assert resolved['inputs']['reference_manifest_path'] != private.source_path  # Exact task feature binding to library genome.
    assert backend.service._application.run_store.load(result['run_id']).request.prompt == state.interactions[0].utterance
    assert len(calls)==2  # Initial task reference plus new task feature binding, not repeated genome installation.
    from agent.application.local_resources import LocalResourceAdmission
    reopened=LocalResourceAdmission(backend.service._application._workspace,approved_source_roots=(uploads.root,), scientific_resource_root=uploads.resources.library.root)
    assert refs.choices(reopened,'new',private.reference_qualification['requirement'])==(promoted,)
    reopened.library.check(promoted);assert len(calls)==2
    wrong=_serialize(private.reference_qualification['requirement']);wrong['assembly']='danRer10'
    assert not refs.choices(reopened,'new',wrong)
    Path(promoted['resource']['genome']['fasta']['path']).write_text('corrupt')
    with pytest.raises(ValueError):reopened.library.check(promoted)


@pytest.mark.parametrize('data,index',[(b'not fasta\n',None),(b'>chr1\nACGTACGT\n',b'chr1\t8\t7\t8\t9\n'),(b'>chr2\nACGTACGT\n',None),(b'>chr1\nACGT\n',None),(b'\x1f\x8bbad',None)])
def test_invalid_reference_is_not_published(reference_web,data,index):
    client,backend,uploads,source,start,_=reference_web
    pending=start();record=upload(client,data,index)
    r=qualify(client,pending,record);assert r.status_code in (400,422),r.text
    assert not refs.choices(uploads.resources,'session',dict(species=dict(scientific_name='Danio rerio',taxonomy_id=7955),assembly='danRer11',feature_category='peak_set'),origin_turn_id='first')


def test_stale_task_private_scope_and_changed_content(reference_web):
    client,backend,uploads,source,start,_=reference_web
    pending=start();record=upload(client);r=qualify(client,pending,record);assert r.status_code==200,r.text
    private=uploads.resources.load(r.json()['resource_id'])
    assert qualify(client,dict(pending,expected_generation=9),record).status_code==400
    assert qualify(client,dict(pending,origin_turn_id='other'),record).status_code==400
    assert client.post('/api/v1/resources/promote',json={'resource_id':private.resource_id}).status_code in (404,405)
    from agent.application.shared_resources import install_private
    with pytest.raises(ValueError):install_private(uploads.resources,'second',private.resource_id,shared_use=True)
    start('second')
    origin=backend.service._application.sessions.load('second').interactions[0]
    with pytest.raises(Exception):refs.select(uploads.resources,'second',origin,'resource.'+refs.input_set(private).input_set_id)
    fasta=uploads.resources.load(record['resource_id']);Path(fasta.source_path).write_bytes(b'>chr1\nAAAAAAAA\n')
    with pytest.raises(Exception):refs.check_record(uploads.resources,private)


def test_supplied_fai_is_verified_without_index_generation(reference_web,monkeypatch):
    import pysam
    client,backend,uploads,source,start,_=reference_web
    pending=start()
    monkeypatch.setattr(pysam,'faidx',lambda *a,**k:pytest.fail('An explicitly supplied FAI must not be regenerated.'))
    record=upload(client,index=b'chr1\t8\t6\t8\t9\n')
    r=qualify(client,pending,record);assert r.status_code==200,r.text
    qualified=uploads.resources.load(r.json()['resource_id'])
    original=uploads.resources.load(record['resource_id'])
    reference=refs.check_record(uploads.resources,qualified)
    assert reference.genome.fai.path==original.source_index['path']


def test_catalog_mutation_and_changed_feature_order_rejected(reference_web):
    client,backend,uploads,source,start,_=reference_web
    pending=start();r=qualify(client,pending,upload(client));assert r.status_code==200,r.text
    private=uploads.resources.load(r.json()['resource_id'])
    promoted=install_shared(uploads.resources,private)
    # Mutation of a catalog payload without its original content binding fails closed.
    path=uploads.resources.library.path;original=path.read_bytes()
    d=json.loads(original);d['resources'][0]['label']='substituted';path.write_text(json.dumps(d))
    with pytest.raises(ValueError):uploads.resources.library.check(promoted)
    path.write_bytes(original)
    start('second');state=backend.service._application.sessions.load('second');origin=state.interactions[0]
    source_record=uploads.resources.load(source['resource_id'])
    # Different content cannot retain the captured H5AD identity, even if dimensions match.
    matrix=ad.read_h5ad(source_record.source_path);matrix.var_names=matrix.var_names[::-1];matrix.write_h5ad(source_record.source_path)
    with pytest.raises(Exception):refs.select(uploads.resources,'second',origin,'resource.'+refs.input_set(promoted).input_set_id)


def test_conflicting_declarations_and_source_are_not_merged(reference_web):
    from dataclasses import replace
    from agent.application.resource_continuation import _completed
    client,backend,uploads,source,start,_=reference_web
    pending=start();r=qualify(client,pending,upload(client));assert r.status_code==200,r.text
    reference=uploads.resources.load(r.json()['resource_id']);source_record=uploads.resources.load(source['resource_id'])
    origin=backend.service._application.sessions.load('session').interactions[0]
    changed=_serialize(origin.submission);changed['execution_inputs']['assembly']='danRer10'
    reply=replace(origin,utterance='Use the qualified reference.',submission=changed)
    with pytest.raises(Exception):_completed(origin,origin,reply,'resource.'+refs.input_set(reference).input_set_id,(),backend.service._application.registry,reference,source_record)


@pytest.mark.parametrize('same_vocabulary', [True, False])
def test_deployment_reuse_checks_a_new_registered_matrix(reference_web,tmp_path,same_vocabulary):
    client,backend,uploads,source,start,_=reference_web
    pending=start();response=qualify(client,pending,upload(client));assert response.status_code==200,response.text
    reference=uploads.resources.load(response.json()['resource_id'])
    shared=install_shared(uploads.resources,reference)
    matrix=ad.read_h5ad(backend.source);matrix.obs_names=['new-a','new-b']
    if not same_vocabulary:matrix.var_names=matrix.var_names[::-1]
    path=tmp_path/'new-matrix.h5ad';matrix.write_h5ad(path)
    new_source=attach(client,path.read_bytes(),upload_id='new-matrix').json()
    source.update(new_source)  # start captures this explicitly selected new source.
    start('second');origin=backend.service._application.sessions.load('second').interactions[0]
    handle='resource.'+refs.input_set(shared).input_set_id
    selected,record=refs.select(uploads.resources,'second',origin,handle)
    assert record.resource_id==new_source['resource_id']
    bound=refs.check_record(uploads.resources,selected)
    from agent.tools.data.scatac_reference import ordered_identity_sha256
    assert bound.features.ordered_feature_sha256 == ordered_identity_sha256(matrix.var_names)
    from dataclasses import replace
    expected = refs.check_record(uploads.resources,reference).genome
    assert replace(bound.genome, fasta=replace(bound.genome.fasta, path=expected.fasta.path),
                   fai=replace(bound.genome.fai, path=expected.fai.path)) == expected
    assert Path(bound.genome.fasta.path).is_relative_to(uploads.resources.library.root)


def test_private_genome_sharing_choices_remain_session_scoped(reference_web):
    from agent.application.shared_genome import persist_private
    client, backend, uploads, source, start, admitted = reference_web
    response = qualify(client, start(), upload(client))
    private = uploads.resources.load(response.json()['resource_id'])
    genome = persist_private(uploads.resources, private.resource_id)
    from dataclasses import replace
    from agent.application import shared_genome
    qualification = _serialize(genome.reference_qualification)
    qualification.pop('session_id')
    qualification['review'] = dict(operator='Historical fixture', provenance='Simulated', reuse_basis='historical')
    historical = replace(genome, session_id=None, reference_qualification=qualification)
    assert shared_genome.check_content(historical) == _serialize(genome.reference_qualification['genome'])
    with pytest.raises(ValueError):
        shared_genome.bind(uploads.resources, historical, None, 'session', None)
    choices = client.get('/api/v1/sessions/session/references').json()['choices']
    assert {r['resource_id'] for r in choices} == {private.resource_id, genome.resource_id}
    start('second')
    assert client.get('/api/v1/sessions/second/references').json()['choices'] == []


def test_genome_reuse_binds_distinct_features_and_replays(reference_web, tmp_path):
    from agent.application import shared_genome
    from agent.application.local_resources import LocalResourceAdmission
    client, backend, uploads, source, start, admitted = reference_web
    pending = start()
    response = qualify(client, pending, upload(client))
    original = uploads.resources.load(response.json()['resource_id'])
    genome = install_shared(uploads.resources, original)
    response = client.get('/api/v1/resources')
    assert response.status_code == 200, response.text
    listed = {row['resource_id'] for row in response.json()['choices']}
    assert source['resource_id'] in listed and genome['id'] not in listed
    old = refs.check_record(uploads.resources, original)
    for component in (old.genome.fasta, old.genome.fai):
        path = Path(component.path)
        path.rename(path.with_name(path.name + '.unavailable'))
    before = sorted(uploads.root.rglob('*.fa'))
    matrix = ad.read_h5ad(backend.source)
    matrix.var_names = ['chr1:1-3', 'chr1:5-7']
    path = tmp_path / 'different.h5ad'; matrix.write_h5ad(path)
    source.update(attach(client, path.read_bytes(), upload_id='different').json())
    start('second')
    body = submission(turn='answer', resource_id=source['resource_id'], input_set_id='fish',
                      utterance='Use the qualified genome and continue.')
    assert client.post('/api/v1/sessions/second/turns', json=body).status_code == 202
    result = wait_turn(client, 'answer', session='second')
    assert result['status'] == 'failed' and len(admitted) == 1, result
    state = backend.service._application.sessions.load('second')  # pure persisted replay
    resolved = state.interactions[-1].admitted
    from agent.tools.data.regulatory_feature_reference import load_regulatory_feature_reference
    new = load_regulatory_feature_reference(resolved['inputs']['reference_manifest_path'])[1]
    assert new.genome.ordered_contig_sha256 == old.genome.ordered_contig_sha256
    assert new.genome.fasta.sha256 == old.genome.fasta.sha256
    assert new.reference_identity_sha256 != old.reference_identity_sha256
    assert new.features.ordered_feature_sha256 != old.features.ordered_feature_sha256
    assert sorted(uploads.root.rglob('*.fa')) == before
    reopened = LocalResourceAdmission(backend.service._application._workspace, approved_source_roots=(uploads.root,), scientific_resource_root=uploads.resources.library.root)
    assert refs.choices(reopened, 'another', original.reference_qualification['requirement']) == (genome,)
    reopened.library.check(genome)
    wrong = _serialize(original.reference_qualification['requirement']); wrong['assembly'] = 'danRer10'
    assert not refs.choices(reopened, 'another', wrong)
    # A dataset BED is not part of the independent genome's content closure.
    Path(old.features.bed.path).write_text('dataset-specific binding no longer available\n')
    reopened.library.check(genome)
    Path(genome['resource']['genome']['fasta']['path']).write_bytes(b'>chr1\nAAAAAAAA\n')
    with pytest.raises(Exception): reopened.library.check(genome)


@pytest.mark.parametrize('names', [['chr2:0-4','chr2:4-8'], ['chr1:0-4','chr1:4-9']])
def test_shared_genome_does_not_authorize_incompatible_regions(reference_web, tmp_path, names):
    from agent.application import shared_genome
    client, backend, uploads, source, start, _ = reference_web
    pending = start(); response = qualify(client, pending, upload(client))
    reference = uploads.resources.load(response.json()['resource_id'])
    genome = install_shared(uploads.resources, reference)
    matrix = ad.read_h5ad(backend.source); matrix.var_names = names
    path = tmp_path/'invalid-regions.h5ad'; matrix.write_h5ad(path)
    source.update(attach(client, path.read_bytes(), upload_id='invalid-regions').json())
    start('second'); origin = backend.service._application.sessions.load('second').interactions[0]
    with pytest.raises(ValueError):
        refs.select(uploads.resources, 'second', origin, 'resource.'+refs.input_set(genome).input_set_id)
