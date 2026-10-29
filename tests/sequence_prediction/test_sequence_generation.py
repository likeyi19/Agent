"""Simulated generation exercises contracts, not biological model qualification."""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
from agent.tools.analysis import sequence_generation as owner, sequence_prediction as shared
from agent.application import ResearchAgentApplication, OutputSelection
from agent.orchestration import AgentPlan, AgentRequest, PlanStep


@pytest.fixture
def native(monkeypatch):
    calls=[]
    resource=dict(family='DNA-Diffusion',release='simulated',condition_mapping=owner.MAPPING,
        conditions=[r['condition'] for r in owner.MAPPING],sequence_length=200,training_domain='simulated',
        license='simulated',source='simulated',checkpoint={},software={},schedule=owner.SCHEDULE,
        metadata={'path':'simulated-label-metadata'})
    monkeypatch.setattr(owner,'resolve_model',lambda:(resource,dict(files=[],resource=resource)))
    read=shared.read
    monkeypatch.setattr(shared,'read',lambda p,*a: {'condition_mapping':owner.MAPPING} if str(p)=='simulated-label-metadata' else read(p,*a))
    def run(resource,effective,stage):
        calls.append(effective)
        return dict(sequences=[('ACGT'*50 if i%2 else 'A'*200) for i in range(effective['number_of_sequences'])],
                    receipt=dict(completion='succeeded',sampling=effective))
    monkeypatch.setattr(owner,'native_generate',run)
    return calls


def args(tmp_path,**changes):
    return dict(output_dir=str(tmp_path/'output'),condition='K562',number_of_sequences=5,seed=17,**changes)


def test_descriptors_duplicates_retained_and_fresh_seed(tmp_path,native):
    a=args(tmp_path);a.pop('seed')
    output=owner.execute(a)
    assert output['n_sequences']==5 and output['unique_sequences']==2 and output['duplicate_sequences']==3
    assert output['gc_fraction']==dict(min=0.,mean=.2,max=.5)
    assert output['entropy_bits']==dict(min=0.,mean=.8,max=2.)
    assert output['longest_homopolymer']==200 and output['pairwise_hamming']['pairs']==10
    assert output['pairwise_hamming']['mean']==90
    assert 0<=output['seed']<2**32
    sequences=[s for s in Path(output['sequences_path']).read_text().splitlines() if not s.startswith('>')]
    assert len(sequences)==5 and sequences.count('A'*200)==3
    with pytest.raises(ValueError,match='not empty'):owner.execute(a)
    assert len(native)==1


@pytest.mark.parametrize('change',[{'condition':'brain'},{'number_of_sequences':0},{'number_of_sequences':1001},
    {'number_of_sequences':True},{'seed':-1},{'seed':2**32},{'seed':True},{'guidance_scale':float('nan')},
    {'guidance_scale':-1},{'guidance_scale':True}])
def test_invalid_sampling_is_not_executed(tmp_path,native,change):
    a=args(tmp_path);a.update(change)
    with pytest.raises(ValueError):owner.execute(a)
    assert not native


@pytest.mark.parametrize('sequences',[[],['A'*199],['N'*200],['A'*201]])
def test_native_count_alphabet_length_are_verified(sequences):
    with pytest.raises(ValueError):owner.characteristics(sequences,owner.MAPPING[2],1)


def test_singleton_diversity_is_undefined():
    _,s=owner.characteristics(['ACGT'*50],owner.MAPPING[2],1)
    assert s['pairwise_hamming']['pairs']==0 and s['pairwise_hamming']['mean'] is None


def test_generation_publication_reuse_downloads_and_restart(tmp_path,native,monkeypatch):
    from agent.application.workspace import ManagedWorkspace
    from agent.application.dialogue_evidence import figure_inventory
    from agent.application import sequence_delivery
    from agent.application.matrix_delivery import MatrixDownloadPreparer
    from agent.orchestration.prior_outputs import bind_output
    root=tmp_path/'workspace';a=args(tmp_path)
    a['output_dir']=str(root/'runs'/ManagedWorkspace.run_digest('first:run')/'scientific'/'generate')
    planner_tool=owner.TOOL
    class Planner:
        def plan(self,request,registry):
            return AgentPlan('test',request.request_id,'Simulated generation',(PlanStep('generate',planner_tool,a),))
    app=ResearchAgentApplication(root,planner=Planner());app.sessions.create('generation')
    state=app.sessions.run('generation','first',AgentRequest('first','Generate five K562 candidates',{}),
        (OutputSelection('candidates','generate','sequences_path'),OutputSelection('batch','generate','manifest_path')),expected_generation=0)
    assert state.turn('first').status=='activated',app.run_store.load('first:run').errors
    revision=state.revisions[-1];turn=state.turn('first')
    binding=bind_output(app.run_store,app.registry,turn.run_id,'generate','sequences')
    assert binding.semantic_type=='dna_sequences.v1'
    figures=figure_inventory(app.sessions,turn,revision)
    assert len(figures)==1 and figures[0]['metadata']['kind']=='sequence_generation_composition'
    step=app.run_store.load(turn.run_id).steps[0];result=step.result
    manifest,_=shared.load(result['manifest_path'],result['manifest_sha256'])
    preparer=MatrixDownloadPreparer()
    for pin in manifest['files']:
        name=Path(pin['path']).name
        if name not in sequence_delivery.FILES:continue
        item=SimpleNamespace(path=Path(pin['path']),managed_root=root,sha256=pin['sha256'],size_bytes=pin['size_bytes'],
            contract=owner.CONTRACT,role=name,manifest=manifest,manifest_sha256=result['manifest_sha256'])
        prepared=preparer.prepare(item)
        try:assert b''.join(prepared.iter_chunks())
        finally:prepared.close()
    reopened=ResearchAgentApplication(root,planner=Planner()).sessions.load('generation')
    assert reopened.to_dict()==state.to_dict() and len(native)==1
    assert shared.recover(a,manifest['execution_identity'])['seed']==17
    assert len(native)==1
    # The explicitly retained accepted sequence port enters the existing DHS64
    # consumer without generation replay or any invented context bases.
    from agent.schemas.prior_output import PriorOutputRef
    import numpy as np
    prediction_calls=[]
    resource=dict(conditions=['K562']+[f'biosample-{i}' for i in range(63)],release='simulated',family='DHS64',
        measurements=shared.MEASUREMENTS,context_bases=500,padding='left',training_assembly='hg38',
        training_domain='simulated',license='simulated',source='simulated',checkpoints=[],software={},metadata=dict(sha256='0'*64))
    monkeypatch.setattr(shared,'resolve_model',lambda:(resource,dict(resource=resource,files=[])))
    def predict(resource,rows,stage):
        assert len(rows)==5 and all(len(r['sequence'])==200 and r['native_zero_padding']==300 for r in rows)
        prediction_calls.append(len(rows))
        return np.zeros((3,2,len(rows),64),dtype=np.float32),dict(completion='succeeded')
    monkeypatch.setattr(shared,'native_predict',predict)
    planner_tool=shared.TOOL
    a=dict(sequence_path=PriorOutputRef(binding,'sequences_path'),sequence_sha256=PriorOutputRef(binding,'sequences_sha256'),
        output_dir=str(root/'runs'/ManagedWorkspace.run_digest('prediction:run')/'scientific'/'predict'),condition='K562')
    second=app.sessions.run('generation','prediction',AgentRequest('prediction','Predict the selected generated DNA',{}),
        (OutputSelection('prediction','generate','manifest_path'),),expected_generation=1,use_active_context=True)
    assert second.turn('prediction').status=='activated',app.run_store.load('prediction:run').errors
    assert len(native)==1 and prediction_calls==[5]
    assert second.revisions[0]==revision
    Path(result['sequences_path']).write_text('corrupt')
    with pytest.raises(ValueError):owner.verify_generation(result['manifest_path'],expected_sha256=result['manifest_sha256'])


@pytest.mark.parametrize('bad_loading',[False,True])
def test_native_batch_remainder_calls_no_discard(tmp_path,monkeypatch,bad_loading):
    """Exercise real adapter control flow with a fake native API, including strict loading."""
    import sys,types,json
    from agent.tools.models import diffusion_runtime
    calls=[]
    class Pretrained:
        @staticmethod
        def from_pretrained(*a,**k):return SimpleNamespace(model=None),dict(missing_keys=['unloaded.weight'] if bad_loading else [],unexpected_keys=[],mismatched_keys=[],error_msgs=[])
    class Diffusion:
        def __init__(self,*a,**k):pass
        def to(self,device):self.device=SimpleNamespace(type='cpu');return self
        def eval(self):return self
        def parameters(self):return ()
    def sample(model,**kw):
        calls.append((kw['number_of_samples'],kw['sample_bs']))
        Path('data/outputs/K562_ENCLB843GMH.txt').write_text('\n'.join(['ACGT'*50]*kw['number_of_samples']))
    for module,values in [('dnadiffusion.models.pretrained_unet',{'PretrainedUNet':Pretrained}),('dnadiffusion.models.diffusion',{'Diffusion':Diffusion}),('dnadiffusion.utils.sample_util',{'create_sample':sample})]:
        m=types.ModuleType(module);m.__dict__.update(values);monkeypatch.setitem(sys.modules,module,m)
    monkeypatch.setattr(diffusion_runtime.importlib.metadata,'version',lambda _: 'simulated')
    request=tmp_path/'native.json'
    request.write_text(json.dumps(dict(settings=dict(threads=1,device='cpu',batch_size=4),
        sampling=dict(seed=1,condition_index=3,number_of_sequences=7,guidance_scale=1.),
        schedule=owner.SCHEDULE,condition_mapping=owner.MAPPING,model_directory='simulated')))
    if bad_loading:
        with pytest.raises(ValueError,match='did not load exactly'):diffusion_runtime.main(request)
        assert not calls and not request.with_suffix('.result.json').exists()
        return
    diffusion_runtime.main(request)
    result=json.loads(request.with_suffix('.result.json').read_text())
    assert calls==[(4,4),(3,3)] and len(result['sequences'])==7


def test_cancellation_terminates_native_process_without_publication(tmp_path,monkeypatch):
    from agent.tools._cancellation import cancellation_scope, ToolWorkCancelled
    calls=[]
    class Process:
        def __init__(self,*a,**k):calls.append('start');self.code=None
        def poll(self):return self.code
        def terminate(self):calls.append('terminate');self.code=-15
        def wait(self,timeout=None):return self.code
    monkeypatch.setattr(owner.subprocess,'Popen',Process)
    resource=dict(checkpoint=dict(path=str(tmp_path/'model.safetensors')),
        runtime=dict(python='/simulated/python',settings=dict(device='cpu',batch_size=4,threads=1)),
        schedule=owner.SCHEDULE,condition_mapping=owner.MAPPING)
    with cancellation_scope(lambda:True),pytest.raises(ToolWorkCancelled):
        owner.native_generate(resource,dict(seed=1,number_of_sequences=1,condition_index=3,guidance_scale=1.),tmp_path)
    assert calls==['start','terminate'] and not (tmp_path/'manifest.json').exists()


def test_resource_condition_labels_and_checkpoint_config_are_exact(tmp_path):
    import json
    metadata=tmp_path/'mapping.json';metadata.write_text(json.dumps(dict(condition_mapping=owner.MAPPING,training_labels_sha256='a'*64)))
    config=tmp_path/'config.json';config.write_text(json.dumps(dict(dim=200,num_classes=10,channels=1)))
    resource=dict(schema=owner.RESOURCE,family='DNA-Diffusion',sequence_length=200,condition_mapping=copy.deepcopy(owner.MAPPING),
        conditions=[r['condition'] for r in owner.MAPPING],schedule=owner.SCHEDULE,metadata=dict(path=str(metadata)),
        config=dict(path=str(config)),checkpoint=dict(path=str(tmp_path/'model.safetensors')),
        runtime=dict(python=__import__('sys').executable,settings=dict(device='cuda:2',batch_size=3,threads=1)))
    assert owner.qualify_resource(resource)==resource
    resource['condition_mapping'][2]['index']=1
    with pytest.raises(ValueError,match='mapping'):owner.qualify_resource(resource)
