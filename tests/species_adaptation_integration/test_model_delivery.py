"""Simulated model bytes, real application/Session/authority and HTTP delivery."""
from dataclasses import replace
from pathlib import Path
import json,hashlib
import pytest
from fastapi.testclient import TestClient
from agent.application import ResearchAgentApplication,InteractiveAgentApplication
from agent.application.session_state import OutputSelection
from agent.orchestration import AgentRequest,LLMPlanner
from agent.application import model_delivery as delivery
from agent.application import matrix_delivery as transport
from agent.web.app import create_app
from test_public import Model,wire
from agent.orchestration import PlanningModelProfile
from agent.providers import PlanningModelFactoryRegistry

def facade(root):
    return InteractiveAgentApplication(root,model_profiles=(PlanningModelProfile("test","scripted","test"),),default_profile_id="test",
        planning_model_factory_registry=PlanningModelFactoryRegistry({"scripted":lambda p:pytest.fail("Provider during delivery")}))


@pytest.fixture
def accepted(integration,tmp_path,monkeypatch):
    from agent.tools.models.epizoo_adaptation import publication as p,contract as c
    inputs,counts,_=integration
    monkeypatch.setattr(p.resources,'runtime',lambda:dict(python='test',packages={},cuda='test',cudnn=0))
    def build(args,root):
        counts['production']+=1
        root.mkdir();(root/'training').mkdir()
        matrix=json.loads(Path(args['matrices'][0]['manifest_path']).read_text())
        reference=json.loads(Path(inputs['reference_manifest_path']).read_text())
        names=('checkpoint.pth','sequence_embeddings.npy','sequences.txt','document_frequency.npy','sentence_tokens.npy',
            'sentence_indptr.npy','cells.jsonl','features.txt','preprocessing.json','execution.json','training/training_log.csv')
        for name in names:(root/name).write_text('SIMULATED model-delivery test bytes')
        value=dict(artifact_type=c.ARTIFACT,contract_version=c.CONTRACT,schema_version=1,
            profile_sha256=c.profile_sha256(args['profile']),arguments=args,reference=reference,
            source_identity='a'*64,seam_identity='b'*64,
            preprocessing=dict(n_cells=matrix['shape'][0],n_features=matrix['shape'][1],
                ordered_feature_sha256=matrix['ordered_feature_sha256']),
            execution=dict(completed_step=10,optimizer_steps=7,amp_skipped_steps=3),
            files={name:{k:v for k,v in c.file_record(root/name).items() if k!='path'} for name in names},
            runtime=p.resources.runtime())
        record=c.write_json(root/'manifest.json',value)
        p.verify_target(root/'manifest.json',expected_sha256=record['sha256'])
        return dict(manifest_path=record['path'],manifest_sha256=record['sha256'])
    monkeypatch.setattr(p,'_build',build)
    app=ResearchAgentApplication(tmp_path/'app',planner=LLMPlanner(Model(wire())))
    app.sessions.create('model')
    state=app.sessions.run('model','train',AgentRequest('model-request','Post-train.',inputs),
        (OutputSelection('model','adapt','manifest_path'),),expected_generation=0)
    assert state.active_revision_id, (state, app.run_store.load('model-request:run').errors)
    service=facade(app.workspace_root)
    rid=state.active_revision_id
    inventory=service.scientific_artifact_handles('model',rid)
    entries={x.filename:x for x in inventory}
    checkpoint=entries['epizoox-checkpoint.pth']
    artifact=service._scientific_artifact('model',rid,checkpoint.handle)
    monkeypatch.setattr(p,'_build',lambda *a:pytest.fail('Retraining during delivery'))
    monkeypatch.setattr(p,'verify_target',lambda *a,**k:pytest.fail('Scientific reconstruction during delivery'))
    return app,service,rid,entries,artifact


def base(rid,session='model'):return f'/api/v1/sessions/{session}/revisions/{rid}/scientific-artifacts'


def test_exact_delivery_metadata_and_fresh_process(accepted,monkeypatch):
    import torch
    app,service,rid,entries,artifact=accepted
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('Deserialized checkpoint for delivery'))
    for current in (service,facade(app.workspace_root)):
        with TestClient(create_app(current)) as client:
            cp=client.get(base(rid)+'/'+entries['epizoox-checkpoint.pth'].handle)
            assert cp.status_code==200,cp.text
            assert cp.content==artifact.path.read_bytes()
            assert hashlib.sha256(cp.content).hexdigest()==cp.headers['x-artifact-content-sha256']
            assert int(cp.headers['content-length'])==artifact.size_bytes
            md=client.get(base(rid)+'/'+entries['epizoox-model.json'].handle)
            assert md.status_code==200,md.text
            assert md.json()['portable_bundle'] is False
            assert set(md.json()['required_not_included'])=={'features.txt','document_frequency.npy'}
            assert str(app.workspace_root) not in md.text and 'cells.jsonl' not in md.text
            assert 'sentence_tokens' not in md.text
        assert current._matrix_downloads.active_count==0


@pytest.mark.parametrize('attack',['missing','tamper','symlink','manifest','receipt','wrong_session','wrong_revision','wrong_handle'])
def test_rejects_bad_file_or_context(accepted,attack):
    app,service,rid,entries,artifact=accepted
    url=base(rid)+'/'+entries['epizoox-checkpoint.pth'].handle
    if attack=='missing':artifact.path.unlink()
    if attack=='tamper':artifact.path.write_bytes(b'unaccepted model replacement')
    if attack=='symlink':
        elsewhere=artifact.path.with_name('private.pth');artifact.path.rename(elsewhere);artifact.path.symlink_to(elsewhere)
    if attack=='manifest':(artifact.path.parent/'manifest.json').write_bytes(b'{}')
    if attack=='receipt':(artifact.path.parent.parent/'receipt.json').write_bytes(b'{}')
    if attack=='wrong_session':
        app.sessions.create('other');url=base(rid,'other')+'/'+entries['epizoox-checkpoint.pth'].handle
    if attack=='wrong_revision':url=base('wrong')+'/'+entries['epizoox-checkpoint.pth'].handle
    if attack=='wrong_handle':url=base(rid)+'/'+'0'*64
    with TestClient(create_app(service)) as client:
        response=client.get(url)
        assert response.status_code==404,response.text
        assert 'content-disposition' not in response.headers and str(app.workspace_root) not in response.text
    assert service._matrix_downloads.active_count==0


@pytest.mark.parametrize('attack',['failed','cancelled','running','missing_authority','wrong_output','wrong_step'])
def test_unaccepted_execution_never_delivered(accepted,monkeypatch,attack):
    app,service,rid,entries,artifact=accepted
    from agent.schemas.run_state import RunLifecycleStatus
    if attack in ('wrong_output','wrong_step'):
        load=service._load;state=load('model');rev=state.revisions[0]
        output=replace(rev.outputs[0],**({'output_key':'not_selected_field'} if attack=='wrong_output' else {'step_id':'adopt'}))
        turn=state.turn(rev.turn_id)
        selection=replace(turn.selections[0],step_id=output.step_id,output_key=output.output_key)
        wrong=replace(state,revisions=(replace(rev,outputs=(output,)),),turns=(replace(turn,selections=(selection,)),))
        monkeypatch.setattr(service,'_load',lambda sid:wrong)
    else:
        store=service._application.run_store;load=store.load
        run=load(service._load('model').revisions[0].run_id)
        if attack=='missing_authority':
            steps=tuple(replace(s,verification=replace(s.verification,artifact_authority=None)) if s.tool_name=='adapt_epizoo_species' else s for s in run.steps)
            wrong=replace(run,steps=steps)
        else:
            from agent.schemas import AgentError,ErrorCategory
            errors=() if attack=='running' else (AgentError(
                ErrorCategory.CANCELLATION if attack=='cancelled' else ErrorCategory.VERIFICATION_ERROR,
                'RUN_CANCELLED' if attack=='cancelled' else 'TEST_FAILED','Simulated terminal state.'),)
            wrong=replace(run,lifecycle_status=RunLifecycleStatus(attack.upper()),run_verification=None,errors=errors)
        monkeypatch.setattr(store,'load',lambda run_id:wrong if run_id==run.run_id else load(run_id))
    with TestClient(create_app(service)) as client:
        r=client.get(base(rid)+'/'+entries['epizoox-checkpoint.pth'].handle)
        assert r.status_code==404,r.text


def test_shared_transfer_capacity_and_role_limits(accepted,monkeypatch):
    _,service,_,_,artifact=accepted;p=service._matrix_downloads
    a=p.prepare(artifact);b=p.prepare(artifact)
    with pytest.raises(transport.MatrixDeliveryError):p.prepare(artifact)
    assert p.active_count==2
    iterator=a.iter_chunks();next(iterator);iterator.close();a.close();a.close();b.close()
    assert p.active_count==p._reserved_bytes==0
    with pytest.raises(transport.MatrixDeliveryError) as e:p.prepare(replace(artifact,size_bytes=delivery.MAX_CHECKPOINT_BYTES+1))
    assert e.value.code=='MODEL_DOWNLOAD_LIMIT'
    with pytest.raises(transport.MatrixDeliveryError):p.prepare(replace(artifact,role='other'))
    monkeypatch.setattr(transport.shutil,'disk_usage',lambda p:type('Disk',(),{'free':0})())
    with pytest.raises(transport.MatrixDeliveryError) as e:p.prepare(artifact)
    assert e.value.code=='SCIENTIFIC_DOWNLOAD_BUSY' and p.active_count==0


@pytest.mark.parametrize('stage',['http.response.start','http.response.body'])
def test_model_http_disconnect_releases_snapshot(accepted,stage):
    import asyncio
    from agent.web.app import _ScientificFileResponse
    _,service,_,_,artifact=accepted
    snapshot=service._matrix_downloads.prepare(artifact)
    async def receive():return {'type':'http.disconnect'}
    async def send(message):
        if message['type']==stage:raise OSError('simulated disconnected client')
    with pytest.raises(BaseException):
        asyncio.run(_ScientificFileResponse(snapshot)({'type':'http','asgi':{'spec_version':'2.4'}},receive,send))
    assert service._matrix_downloads.active_count==service._matrix_downloads._reserved_bytes==0


@pytest.mark.parametrize('size',[4018926698,4938458542,5*1024**3])
def test_full_vocabulary_checkpoint_reservation_and_other_role_bounds(accepted,monkeypatch,size):
    _,service,_,_,artifact=accepted
    p=service._matrix_downloads
    # Exercise real role admission and exact disk reservation without creating
    # multi-GiB test payloads. A source-open failure must release that reservation.
    monkeypatch.setattr(transport.shutil,'disk_usage',lambda _:type('Disk',(),{'free':size+transport._DISK_MARGIN_BYTES})())
    def unavailable(*args):
        assert p.active_count==1 and p._reserved_bytes==size
        raise OSError('Simulated missing source after capacity admission')
    monkeypatch.setattr(transport,'_open_matrix_source',unavailable)
    with pytest.raises(transport.MatrixDeliveryError) as error:
        p.prepare(replace(artifact,size_bytes=size))
    assert error.value.code=='SCIENTIFIC_DOWNLOAD_UNAVAILABLE'
    assert p.active_count==p._reserved_bytes==0
    monkeypatch.setattr(transport.shutil,'disk_usage',lambda _:type('Disk',(),{'free':size+transport._DISK_MARGIN_BYTES-1})())
    with pytest.raises(transport.MatrixDeliveryError) as error:
        p.prepare(replace(artifact,size_bytes=size))
    assert error.value.code=='SCIENTIFIC_DOWNLOAD_BUSY'
    assert p.active_count==p._reserved_bytes==0
    for contract,role in [('scatac-cell-by-ccre.v1','matrix'),('scatac-barcode-qc.v1','table')]:
        with pytest.raises(transport.MatrixDeliveryError) as error:
            p.prepare(replace(artifact,contract=contract,role=role,size_bytes=1024**3+1))
        assert error.value.code.endswith('_DOWNLOAD_LIMIT')
    assert p.active_count==p._reserved_bytes==0


def test_private_metadata_text_is_not_exported(accepted):
    from copy import deepcopy
    _,_,_,_,artifact=accepted
    value=deepcopy(artifact.manifest);value['reference']['target_assembly']='/private/research/path'
    with pytest.raises(ValueError,match='Private text'):delivery.metadata(value,artifact.manifest_sha256)


def test_default_request_reaches_native_owner_without_running_default_steps(integration,tmp_path,monkeypatch):
    from agent.tools.models.epizoo_adaptation import publication as p,contract as c
    from agent.tools.models import native_posttraining as native
    inputs,_,_=integration
    spec=json.loads(Path(inputs['adaptation_spec_path']).read_text())
    spec.update(contract_version='epizoo-adaptation-inputs.v2',execution_profile='native.v2')
    record=c.write_json(tmp_path/'native-spec.json',spec)
    inputs.update(adaptation_spec_path=record['path'],adaptation_spec_sha256=record['sha256'])
    admitted=[];dispatched=[]
    monkeypatch.setattr(native,'admit_resources',lambda profile,n,root:admitted.append(profile))
    def stop(**args):
        dispatched.append(args)
        raise ValueError('Explicit controlled stop before any scientific execution.')
    monkeypatch.setattr(p,'adapt_epizoo_species',stop)
    app=ResearchAgentApplication(tmp_path/'default-app',planner=LLMPlanner(Model(wire())))
    result=app.run(AgentRequest('defaults','Post-train the explicitly supplied matrix.',inputs))
    assert len(dispatched)==len(admitted)==1
    assert admitted[0]['trainer']['max_steps']==500000
    assert dispatched[0]['profile']==admitted[0]
    assert result.status.value=='FAILED'  # deliberate harness stop, never successful training


def test_model_stream_copy_reads_bounded_chunks(accepted,monkeypatch):
    _,service,_,_,artifact=accepted
    original=transport._open_matrix_source;reads=[]
    class Observed:
        def __init__(self,stream):self.stream=stream
        def __enter__(self):return self
        def __exit__(self,*args):self.stream.close()
        def fileno(self):return self.stream.fileno()
        def read(self,size):reads.append(size);return self.stream.read(size)
    monkeypatch.setattr(transport,'_open_matrix_source',lambda *a:Observed(original(*a)))
    with service._matrix_downloads.prepare(artifact) as snap:
        assert b''.join(snap.iter_chunks())==artifact.path.read_bytes()
    assert reads and all(0<size<=transport.MATRIX_CHUNK_BYTES for size in reads)


def test_unretained_model_does_not_appear_on_later_revision(accepted):
    from agent.orchestration import AgentPlan,PlanStep
    app,service,rid,entries,artifact=accepted
    manifest=json.loads(Path(artifact.manifest['arguments']['matrices'][0]['manifest_path']).read_text())
    matrix_path=Path(artifact.manifest['arguments']['matrices'][0]['manifest_path']).parent/manifest['matrix']['path']
    class Inspection:
        def plan(self,request,registry):
            return AgentPlan('inspect-plan',request.request_id,'Inspect this matrix.',
                (PlanStep('inspect','inspect_scATAC',dict(path=str(matrix_path))),))
    child_app=ResearchAgentApplication(app.workspace_root,planner=Inspection())
    child=child_app.sessions.run('model','inspect-turn',AgentRequest('inspect-request','Inspect matrix.',{}),
        (OutputSelection('inspection','inspect','n_cells'),),expected_generation=1)
    assert child.active_revision_id!=rid and child.active_revision_id
    with TestClient(create_app(service)) as client:
        assert client.get(base(child.active_revision_id)).json()['artifacts']==[]
        assert client.get(base(child.active_revision_id)+'/'+entries['epizoox-checkpoint.pth'].handle).status_code==404
        assert client.get(base(rid)+'/'+entries['epizoox-checkpoint.pth'].handle).status_code==200


@pytest.mark.parametrize('state',['initialization','posttraining'])
def test_lifecycle_metadata_has_effective_strategy_and_truthful_training(accepted,tmp_path,state):
    from copy import deepcopy
    from agent.tools.models.epizoo_adaptation import contract as c
    _,_,_,_,artifact=accepted
    manifest=deepcopy(artifact.manifest)
    manifest['schema_version']=2
    manifest['arguments']['profile']['purpose']=state
    manifest['arguments']['strategy']='auto'
    manifest['arguments']['parent']=None if state=='initialization' else {'manifest_sha256':'b'*64}
    manifest['preprocessing']['df_scope']=state+'-input-corpus'
    if state=='initialization':manifest['execution'].update(completed_step=0,optimizer_steps=0,amp_skipped_steps=0)
    decision=dict(selected='de_novo',requested='auto',candidates=[],reason='mapping-unavailable;not-evaluated')
    record=c.write_json(tmp_path/'transfer-decision.json',decision)
    manifest['files']['transfer-decision.json']={k:v for k,v in record.items() if k!='path'}
    result=json.loads(delivery.metadata(manifest,'a'*64,root=tmp_path))
    assert result['format']=='epizoox-download-metadata.v2'
    assert result['strategy']=='de_novo' and result['requested_strategy']=='auto'
    assert result['transfer_assessment']==decision
    assert result['model_state']==state and result['required_not_included']=={}
    assert 'training_cell_count' not in result
    if state=='initialization':assert result['optimizer_updates']==0 and result['training_resume']=='not-trained;parameter-transfer-only'
    else:assert result['parent_initialized_manifest_sha256']=='b'*64
    (tmp_path/'transfer-decision.json').write_text('{}')
    with pytest.raises(ValueError):delivery.metadata(manifest,'a'*64,root=tmp_path)


def test_scientific_answer_projects_the_existing_model_download_inventory(accepted):
    from agent.application.analysis_delivery import export_descriptions
    app,_,rid,entries,_=accepted
    state=app.sessions.load('model');revision=next(r for r in state.revisions if r.revision_id==rid)
    output=revision.outputs[0]
    descriptions=export_descriptions(app.sessions,'model',[dict(revision_id=rid,output_name=output.name,
        accepted_step_sha256=output.accepted_step_sha256,subject=None)])
    assert {r['filename'] for r in descriptions}=={'epizoox-checkpoint.pth','epizoox-model.json'}
    assert all(r['actual_download_status']=='NOT_RECORDED' for r in descriptions)
    assert all(r['status']=='available_for_preparation' for r in descriptions)


@pytest.mark.parametrize('state',['Initialized','Post-trained'])
def test_lifecycle_download_labels_preserve_closed_artifact_contract(state):
    from agent.application.interactive_schemas import ScientificArtifactHandle
    from agent.application.model_delivery import FILES
    for filename,label in FILES.values():
        item=ScientificArtifactHandle('a'*64,'scientific_model','b'*64,'revision','turn',100,
            filename,state+': '+label)
        assert item.title==state+': '+label
    with pytest.raises(ValueError,match='Unsupported scientific artifact'):
        ScientificArtifactHandle('a'*64,'scientific_model','b'*64,'revision','turn',100,
            'epizoox-checkpoint.pth','Biologically validated: EpiZooX model checkpoint')
