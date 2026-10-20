"""Scripted interpretation with real annotation authority/publication and stub native backend."""
import hashlib
import io
import json
from pathlib import Path
import time
from dataclasses import replace
import anndata as ad
import h5py
import pytest
from fastapi.testclient import TestClient
from agent.application import InteractiveAgentApplication
from agent.application.input_sets import ScientificInputSet
from agent.application.session_state import AnalysisSession, SessionError
from agent.orchestration import PlanningModelProfile
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas.orchestration import _serialize
from agent.web.app import create_app
from test_annotation import case, wire, TOOL

OBJECTIVE = 'Please annotate my accepted scATAC-seq dataset using MAESTRO and preserve unresolved groups.'
ANSWER = 'Use the reviewed independent groups and resources for the declared synthetic context.'


class Model:
    model_id = 'scripted-maestro-workflow'
    def __init__(self, prompts): self.prompts = prompts
    def complete(self, *, prompt, response_schema):
        p = json.loads(prompt); self.prompts.append(p)
        if 'turn_schema_version' in p:
            if p['utterance'] == OBJECTIVE:
                d = dict(kind='request_resources', target=TOOL, arguments=[])
            elif p['utterance'] == 'Still deciding.':
                d = dict(kind='clarify', reason='ambiguous_subject')
            else:
                d = dict(kind='answer_resources', pending='@resources', choice='resource.reviewed', arguments=[])
            return json.dumps(dict(turn_schema_version=1, decision=d))
        if 'selection_schema_version' in p:
            return json.dumps(dict(selection_schema_version=1, decision=dict(kind='select', capability_ids=['marker_annotation'])))
        if 'output_selection_schema_version' in p:
            return json.dumps(dict(outputs=[dict(name='annotation', step_id='annotation', output_key='manifest_path')]))
        return json.dumps(wire())


def harness(case, tmp_path, choices=None):
    args, calls, _, _ = case
    prompts=[]
    profile=PlanningModelProfile('scripted', 'scripted', Model.model_id)
    facade=InteractiveAgentApplication(tmp_path/'web', model_profiles=(profile,),default_profile_id='scripted',
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':lambda _:Model(prompts)}))
    source=ScientificInputSet('matrix','Accepted synthetic canonical matrix',
        {k:v for k,v in args.items() if k.startswith('matrix_')})
    resources=ScientificInputSet('reviewed','Reviewed independent groups, synthetic context, gene and marker resources',
        {k:v for k,v in args.items() if k.startswith('annotation_')})
    choices = (source,resources) if choices is None else choices(source,resources)
    return facade, choices, prompts


def submit(client, turn, text, selection='matrix', generation=0):
    body=dict(turn_id=turn,utterance=text,expected_generation=generation,profile_id='scripted')
    if selection is not None:body['input_set_id']=selection
    response=client.post('/api/v1/sessions/analysis/turns',json=body)
    assert response.status_code == 202,response.text
    deadline=time.monotonic()+60
    while time.monotonic()<deadline:
        value=client.get('/api/v1/sessions/analysis/turns/'+turn).json()
        if value.get('response') is not None and value['status'] not in ('pending','running','processing','submitted','admitted','interpreting'):
            return value
        time.sleep(.02)
    pytest.fail(str(value))


def start(client):
    assert client.post('/api/v1/sessions',json={'session_id':'analysis'}).status_code==201
    first=submit(client,'request',OBJECTIVE)
    assert first['status']=='clarification',first
    return first


def test_resource_continuation_annotation_download_and_reopen(case,tmp_path):
    facade,choices,prompts=harness(case,tmp_path)
    with TestClient(create_app(facade,input_sets=choices)) as client:
        first=start(client)
        assert 'Reviewed independent groups' in first['response']['text']
        assert submit(client,'wait','Still deciding.')['status']=='clarification'
        result=submit(client,'answer',ANSWER,selection=None)
        assert result['status']=='succeeded',result
        state=facade._application.sessions.load('analysis')
        assert len(state.revisions)==1 and state.generation==1
        run=facade._application.run_store.load(result['run_id'])
        assert run.request.prompt==OBJECTIVE
        assert run.steps[0].verification.artifact_authority['schema_version']==2
        assert len(case[1])==1
        base='/api/v1/sessions/analysis/revisions/'+result['revision_id']
        evidence=client.get(base+'/evidence?output_name=annotation').json()
        assert evidence['field_labels']['contract_version']=='Annotation contract'
        response=client.get(base+'/scientific-artifacts')
        assert response.status_code==200,response.text
        artifacts=response.json()['artifacts'];assert len(artifacts)==1
        artifact=artifacts[0];assert artifact['filename']=='annotated.h5ad'
        download=client.get(base+'/scientific-artifacts/'+artifact['handle'])
        assert download.status_code==200,download.text
        assert hashlib.sha256(download.content).hexdigest()==download.headers['x-artifact-content-sha256']
        assert artifact['sha256']==download.headers['x-artifact-source-sha256']
        assert download.headers['x-artifact-content-sha256']!=artifact['sha256']
        with h5py.File(io.BytesIO(download.content),'r') as h:
            assert 'matrix_manifest_path' not in h['uns/cell_type_annotation']
        saved=tmp_path/'download.h5ad';saved.write_bytes(download.content)
        original=ad.read_h5ad(run.steps[0].result['annotated_h5ad_path']);exported=ad.read_h5ad(saved)
        assert original.obs.equals(exported.obs) and original.var.equals(exported.var)
        assert (original.X!=exported.X).nnz==0
        assert len(case[1])==1
        before=len(prompts)
        assert submit(client,'answer',ANSWER,selection=None)==result
        assert len(prompts)==before
        stored=state.to_dict()
    fresh,_,fresh_prompts=harness(case,tmp_path)
    assert fresh._application.sessions.load('analysis').to_dict()==stored
    assert not fresh_prompts and len(case[1])==1


@pytest.mark.parametrize('problem',['source','spec','catalog','stale','invented'])
def test_resource_substitution_and_stale_rejected(case,tmp_path,problem):
    facade,choices,prompts=harness(case,tmp_path)
    with TestClient(create_app(facade,input_sets=choices)) as client:
        start(client)
    changed=choices
    if problem=='source':changed=(replace(choices[0],execution_inputs=dict(choices[0].inputs(),matrix_manifest_sha256='a'*64)),choices[1])
    elif problem=='spec':changed=(choices[0],replace(choices[1],execution_inputs=dict(choices[1].inputs(),annotation_spec_sha256='b'*64)))
    elif problem=='catalog':changed=(choices[0],replace(choices[1],label='Changed definition'))
    elif problem=='invented':changed=(choices[0],replace(choices[1],input_set_id='another'))
    with TestClient(create_app(facade,input_sets=changed)) as client:
        if problem=='stale':
            response=client.post('/api/v1/sessions/analysis/turns',json=dict(turn_id='answer',utterance=ANSWER,expected_generation=1))
            assert response.status_code==409
        else:
            result=submit(client,'answer',ANSWER)
            assert result['status']=='clarification',result
    assert not facade._application.sessions.load('analysis').revisions
    assert not case[1]


@pytest.mark.parametrize('field',['inputs','catalog','digest'])
def test_captured_resource_tampering_fails(case,tmp_path,field):
    facade,choices,_=harness(case,tmp_path)
    with TestClient(create_app(facade,input_sets=choices)) as client:start(client)
    state=_serialize(facade._application.sessions.load('analysis').to_dict());record=state['interactions'][0]
    if field=='inputs':record['admitted']['inputs']['matrix_manifest_sha256']='0'*64
    elif field=='catalog':record['admitted']['resource_catalog'][-1]['execution_inputs']['annotation_spec_sha256']='0'*64
    else:record['prerequisite']['resource_context_sha256']='0'*64
    with pytest.raises(SessionError):AnalysisSession.from_dict(state)


def test_structured_offers_do_not_replace_captured_roles():
    from agent.application.resource_continuation import offers
    original={'execution_inputs':{'matrix_manifest_path':'/exact/matrix','matrix_manifest_sha256':'a'*64}}
    changed=ScientificInputSet('changed','Another source',{'matrix_manifest_path':'/other/matrix','annotation_spec_path':'/spec'})
    good=ScientificInputSet('good','Reviewed specification',{'annotation_spec_path':'/spec','annotation_spec_sha256':'b'*64})
    assert offers(original,(changed,good))==[dict(handle='resource.good',label='Reviewed specification')]


@pytest.mark.parametrize('mutation',['private_obs','private_category','extra_uns','link','bad_code','changed_provenance'])
def test_annotation_export_rejects_unreviewed_layout(case,tmp_path,mutation):
    from agent.tools.analysis import scatac_annotation as science, annotation_contract as contract
    from agent.application.annotation_delivery import create_annotation_export
    from agent.application.matrix_delivery import MatrixPrivacyError
    from agent.tools.analysis.neighbors_delivery import NeighborsPrivacyError
    args,*_=case
    result=science.execute_annotation(args)
    manifest=contract.load_manifest(result['manifest_path'],result['manifest_sha256'])
    path=Path(result['annotated_h5ad_path'])
    with h5py.File(path,'r+') as h:
        if mutation=='private_obs': h['obs'].create_dataset('private_path',data=['/private/key']*2,dtype=h5py.string_dtype())
        elif mutation=='private_category': h['obs/annotation_group/categories'][0]='/private/group'
        elif mutation=='extra_uns': h['uns'].create_dataset('private',data='secret')
        elif mutation=='link': h['layers']['alias']=h['X']
        elif mutation=='bad_code': h['obs/annotation_status/codes'][0]=-1
        else: h['uns/cell_type_annotation/matrix_authority_sha256'][()]='f'*64
    pin=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),size_bytes=path.stat().st_size)
    # A caller's rehash cannot make unreviewed fields eligible for delivery.
    manifest['sidecars']['annotated.h5ad']=pin
    with path.open('rb') as source,pytest.raises((MatrixPrivacyError,NeighborsPrivacyError)):
        create_annotation_export(source,io.BytesIO(),manifest,source_sha256=pin['sha256'],
            source_size_bytes=pin['size_bytes'],manifest_sha256=result['manifest_sha256'],max_bytes=pin['size_bytes']+65536)


@pytest.mark.parametrize('value',['cell', 'x'*255, 'x'*256, 'x'*65536, '界'*100])
def test_annotation_export_bounded_string_probe_preserves_exact_values(tmp_path,value):
    from agent.application.annotation_delivery import _string_block
    with h5py.File(tmp_path/'strings.h5','w') as h:
        item=h.create_dataset('names',data=[value],dtype=h5py.string_dtype())
        assert _string_block(item,0,1)==[value]


def test_annotation_export_string_probe_rejects_excessive_identifier(tmp_path):
    from agent.application.annotation_delivery import _string_block
    from agent.application.matrix_delivery import MatrixPrivacyError
    with h5py.File(tmp_path/'strings.h5','w') as h:
        item=h.create_dataset('names',data=['x'*65537],dtype=h5py.string_dtype())
        with pytest.raises(MatrixPrivacyError):_string_block(item,0,1)
