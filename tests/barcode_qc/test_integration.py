from dataclasses import replace
import json
from pathlib import Path
import pytest
from agent.application import ResearchAgentApplication
from agent.orchestration import (AgentRequest,AgentRuntime,AgentPlan,PlanStep,LLMPlanner,PlanningWireMode,
    RunMode,FileRunStore,StepStatus,ToolRegistry,StepOutputRef,build_default_tool_registry)
from agent.tools.data import scatac_barcode_qc as public,_barcode_qc_production as production,_barcode_qc_binding as binding


class Model:
    model_id='scripted-qc'
    def __init__(self,payload):self.payload=payload
    def complete(self,*,prompt,response_schema):
        self.prompt=prompt;self.schema=response_schema;return json.dumps(self.payload)


def wire():
    return dict(schema_version=4,decision=dict(kind='plan',steps=[dict(step_id='qc',tool='compute_scATAC_qc',
        sources=[dict(target=port,source=dict(kind='input',input=key)) for port,key in (
            ('fragments','fragments_manifest_path'),('qc_reference','qc_reference_manifest_path'))],control_dependencies=[])]))


def request(args,mode=RunMode.EXECUTE):
    return AgentRequest('qc','Compute observed-barcode QC without selecting cells.',{k:v for k,v in args.items() if k!='output_dir'},mode)


class FixedPlanner:
    def __init__(self,args): self.value=AgentPlan('qc-plan','qc','Compute QC.',(PlanStep('qc','compute_scATAC_qc',args),))
    def plan(self,request,registry):return self.value


def test_plan_only_has_zero_scientific_io(tmp_path,monkeypatch):
    args={k:('/PRIVATE/'+k if k.endswith('_path') else '1'*64) for k in (
        'fragments_manifest_path','fragments_manifest_sha256','qc_reference_manifest_path','qc_reference_manifest_sha256')}
    def forbidden(*a,**k):pytest.fail('Scientific IO during PLAN_ONLY')
    monkeypatch.setattr(binding,'bind',forbidden);monkeypatch.setattr(binding,'backend_runtime',forbidden)
    monkeypatch.setattr(public,'load_scatac_qc_reference_bundle',forbidden)
    registry=build_default_tool_registry();registry=ToolRegistry(tuple(replace(registry.get(n),function=forbidden) for n in registry.names()))
    model=Model(wire());app=ResearchAgentApplication(tmp_path/'app',planner=LLMPlanner(model),registry=registry)
    result=app.run(request(args,RunMode.PLAN_ONLY))
    assert result.status.value=='PLANNED',result
    assert not result.run_result.steps and result.evidence is result.report is None
    assert '/PRIVATE' not in model.prompt and '/PRIVATE' not in str(model.schema)
    assert not list(app._workspace.run_paths(result.run_id).scientific.iterdir())


def test_v3_v4_and_registry(qc_case,monkeypatch):
    args,*_=qc_case;registry=build_default_tool_registry()
    req=AgentRequest('qc','QC.',args,RunMode.PLAN_ONLY)
    assert dict(LLMPlanner(Model(wire())).plan(req,registry).steps[0].arguments)==args
    payload=dict(schema_version=3,status='plan',reason=None,steps=[dict(step_id='qc',tool_name='compute_scATAC_qc',
        arguments={k:dict(binding_type='input',input_name=k) for k in args},depends_on=[],description='QC.')])
    assert dict(LLMPlanner(Model(payload),wire_mode=PlanningWireMode.V3).plan(req,registry).steps[0].arguments)==args
    assert len(registry.names())==17
    result=public.compute_scATAC_qc(**args)
    monkeypatch.setattr(public,'verify_public_result',lambda *a:pytest.fail('IO-free result validation'))
    registry.validate_result('compute_scATAC_qc',result)
    with pytest.raises(ValueError):registry.validate_result('compute_scATAC_qc',result|{'n_observed_barcodes':True})


def test_external_upstream_semantics(qc_case):
    args,_,_,_,adoption=qc_case;payload=wire();step=payload['decision']['steps'][0]
    step['sources'][0]['source']=dict(kind='step',step='import')
    payload['decision']['steps'].insert(0,dict(step_id='import',tool='import_scATAC_fragments',control_dependencies=[],
        sources=[dict(target=port,source=dict(kind='input',input=key)) for port,key in (
            ('source','source_path'),('reference','reference_bundle_path'),('namespace','namespace'),('source_profile','source_profile'))]))
    inputs={k:v for k,v in (adoption|args).items() if not k.startswith('fragments_manifest')}
    plan=LLMPlanner(Model(payload)).plan(AgentRequest('qc','Adopt then QC.',inputs,RunMode.PLAN_ONLY),build_default_tool_registry())
    assert plan.steps[1].arguments['fragments_manifest_path']==StepOutputRef('import','manifest_path')
    assert plan.steps[1].arguments['fragments_manifest_sha256']==StepOutputRef('import','manifest_sha256')
    assert plan.steps[1].depends_on==('import',)


def test_application_upstream_execute(qc_case,tmp_path):
    args,_,_,_,adoption=qc_case
    payload=wire();payload['decision']['steps'][0]['sources'][0]['source']=dict(kind='step',step='import')
    payload['decision']['steps'].insert(0,dict(step_id='import',tool='import_scATAC_fragments',control_dependencies=[],
        sources=[dict(target=port,source=dict(kind='input',input=key)) for port,key in (
            ('source','source_path'),('reference','reference_bundle_path'),('namespace','namespace'),('source_profile','source_profile'))]))
    inputs={k:v for k,v in (adoption|args).items() if k!='output_dir' and not k.startswith('fragments_manifest')}
    app=ResearchAgentApplication(tmp_path/'composed-app',planner=LLMPlanner(Model(payload)))
    result=app.run(AgentRequest('qc','Adopt then compute QC without selecting cells.',inputs))
    assert result.status.value=='SUCCEEDED',result
    assert result.evidence and result.report and result.visualization is None
    assert result.run_result.steps[1].resolved_arguments['fragments_manifest_sha256']==result.run_result.steps[0].result['manifest_sha256']


@pytest.mark.parametrize('mutation',[None,'fragments','reference','output'])
def test_application_report_resume(qc_case,tmp_path,monkeypatch,mutation):
    args,*_=qc_case;root=tmp_path/'app';app=ResearchAgentApplication(root,planner=LLMPlanner(Model(wire())))
    result=app.run(request(args))
    assert result.status.value=='SUCCEEDED',result
    assert result.evidence and result.report and result.visualization is None
    text=Path(result.report.path).read_text()
    assert 'not automatically called cells' in text and 'No QC selection' in text and 'No FRiP' in text
    assert 'zero:background' not in text
    if mutation:
        target=args['fragments_manifest_path'] if mutation=='fragments' else args['qc_reference_manifest_path'] if mutation=='reference' else str(Path(result.run_result.steps[0].result['manifest_path']).parent/'barcodes.tsv.gz')
        with open(target,'ab') as f:f.write(b'changed')
    monkeypatch.setattr(production,'produce',lambda *a:pytest.fail('Production repeated'))
    resumed=ResearchAgentApplication(root).resume(result.run_id)
    assert resumed.status.value==('FAILED' if mutation else 'SUCCEEDED'),resumed


class Crash(BaseException):pass


class StopStore:
    def __init__(self,store,before=True,cancel=None):self.store=store;self.before=before;self.cancel=cancel
    def __getattr__(self,name):return getattr(self.store,name)
    def update(self,state,*,expected_revision):
        done=any(s.tool_name=='compute_scATAC_qc' and s.status is StepStatus.SUCCEEDED for s in state.steps)
        if done and self.before:raise Crash()
        saved=self.store.update(state,expected_revision=expected_revision)
        if done:
            if self.cancel:self.cancel(saved.run_id)
            else:raise Crash()
        return saved


def test_application_publication_before_checkpoint(qc_case,tmp_path,monkeypatch):
    args,*_=qc_case;root=tmp_path/'recovery-app'
    app=ResearchAgentApplication(root,planner=LLMPlanner(Model(wire())))
    app.runtime._run_store=StopStore(app.run_store)
    with pytest.raises(Crash):app.run(request(args))
    monkeypatch.setattr(production,'produce',lambda *a:pytest.fail('Production repeated'))
    result=ResearchAgentApplication(root).resume('qc:run')
    assert result.status.value=='SUCCEEDED',result
    assert result.evidence and result.report and result.visualization is None


def test_cancellation_terminates_active_intersection(qc_case,tmp_path,monkeypatch):
    from types import SimpleNamespace
    import subprocess
    args,*_=qc_case;store=FileRunStore(tmp_path/'active-store')
    runtime=AgentRuntime(planner=FixedPlanner(args),run_store=store);events=[]
    class ActiveProcess:
        def __init__(self,*a,**k):self.stopped=False;events.append('started')
        def wait(self,timeout=None):
            if self.stopped:return -15
            runtime.cancel('qc:run');raise subprocess.TimeoutExpired('fixture-intersection',timeout)
        def poll(self):return -15 if self.stopped else None
        def terminate(self):self.stopped=True;events.append('terminated')
    monkeypatch.setattr(production,'subprocess',SimpleNamespace(Popen=ActiveProcess,TimeoutExpired=subprocess.TimeoutExpired))
    result=runtime.run(AgentRequest('qc','QC.',{}))
    assert result.status.value=='CANCELLED',result.errors
    assert events==['started','terminated']
    assert not list(Path(args['output_dir']).glob('.qc-attempt-*'))
    assert not list(Path(args['output_dir']).glob('barcode-qc-*'))


@pytest.mark.parametrize('before',[True,False])
@pytest.mark.parametrize('mutation',[None,'receipt','source','policy'])
def test_exact_receipt_recovery(qc_case,tmp_path,monkeypatch,before,mutation):
    args,*_=qc_case;store=FileRunStore(tmp_path/'store')
    with pytest.raises(Crash):AgentRuntime(planner=FixedPlanner(args),run_store=StopStore(store,before)).run(AgentRequest('qc','QC.',{}))
    if mutation:
        receipt=next(Path(args['output_dir']).glob('barcode-qc-*/receipt.json'))
        if mutation=='source':Path(args['qc_reference_manifest_path']).write_bytes(b'changed')
        elif mutation=='receipt':receipt.unlink()
        else:
            r=json.loads(receipt.read_bytes());r['policy']='wrong';receipt.write_text(json.dumps(r))
    monkeypatch.setattr(production,'produce',lambda *a:pytest.fail('Production repeated in recovery'))
    result=AgentRuntime(run_store=store).resume('qc:run')
    assert result.status.value==('FAILED' if mutation else 'SUCCEEDED'),result.errors


@pytest.mark.parametrize('boundary',['before','during','after'])
def test_cancellation(qc_case,tmp_path,monkeypatch,boundary):
    args,*_=qc_case;store=FileRunStore(tmp_path/'store');runtime=AgentRuntime(planner=FixedPlanner(args),run_store=store)
    if boundary=='before':
        class CancelCreate:
            def __getattr__(self,name):return getattr(store,name)
            def create(self,state):
                saved=store.create(state);store.request_cancellation(state.run_id);return saved
        runtime._run_store=CancelCreate()
        monkeypatch.setattr(production,'produce',lambda *a:pytest.fail('Production after cancellation'))
    elif boundary=='during':
        original=production._intersection
        def cancel(*a,**k):
            runtime.cancel('qc:run');return original(*a,**k)
        monkeypatch.setattr(production,'_intersection',cancel)
    else:runtime._run_store=StopStore(store,False,lambda rid:runtime.cancel(rid))
    result=runtime.run(AgentRequest('qc','QC.',{}))
    assert result.status.value=='CANCELLED',result.errors
    manifests=list(Path(args['output_dir']).glob('barcode-qc-*/manifest.json'))
    assert bool(manifests)==(boundary=='after')
    assert not list(Path(args['output_dir']).glob('.qc-attempt-*'))
    if boundary=='after':assert result.steps[0].status is StepStatus.SUCCEEDED
