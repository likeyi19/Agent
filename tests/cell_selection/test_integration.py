from dataclasses import replace
import json
from pathlib import Path
import pytest
from agent.application import ResearchAgentApplication
from agent.orchestration import (AgentRequest,AgentRuntime,AgentPlan,PlanStep,LLMPlanner,PlanningWireMode,
    RunMode,FileRunStore,StepStatus,ToolRegistry,StepOutputRef,build_default_tool_registry)
from agent.tools.data import scatac_cell_selection as public,_cell_selection_production as production


class Model:
    model_id='scripted-qc'
    def __init__(self,payload):self.payload=payload
    def complete(self,*,prompt,response_schema):
        self.prompt=prompt;self.schema=response_schema;return json.dumps(self.payload)


def wire():
    return dict(schema_version=4,decision=dict(kind='plan',steps=[dict(step_id='qc',tool='select_scATAC_cells',
        sources=[dict(target=port,source=dict(kind='input',input=key)) for port,key in (
            ('barcode_qc','barcode_qc_manifest_path'),('min_qc_fragment_records','min_qc_fragment_records'),('min_tss_enrichment','min_tss_enrichment'))],control_dependencies=[])]))


@pytest.mark.parametrize('include_import',[False,True])
def test_application_composition(qc_case,tmp_path,include_import):
    from barcode_qc.test_integration import wire as qc_wire
    args,_,_,_,adoption=qc_case
    payload=wire();selection=payload['decision']['steps'][0];selection['step_id']='selection'
    selection['sources'][0]['source']=dict(kind='step',step='qc')
    payload['decision']['steps'].insert(0,qc_wire()['decision']['steps'][0])
    inputs={k:v for k,v in args.items() if k!='output_dir'}|dict(min_qc_fragment_records=0,min_tss_enrichment='0')
    if include_import:
        payload['decision']['steps'][0]['sources'][0]['source']=dict(kind='step',step='import')
        payload['decision']['steps'].insert(0,dict(step_id='import',tool='import_scATAC_fragments',control_dependencies=[],
            sources=[dict(target=port,source=dict(kind='input',input=key)) for port,key in (
                ('source','source_path'),('reference','reference_bundle_path'),('namespace','namespace'),('source_profile','source_profile'))]))
        inputs={k:v for k,v in (adoption|inputs).items() if k!='output_dir' and not k.startswith('fragments_manifest')}
    app=ResearchAgentApplication(tmp_path/'composed',planner=LLMPlanner(Model(payload)))
    result=app.run(AgentRequest('qc','Compute QC and explicitly select candidates.',inputs))
    assert result.status.value=='SUCCEEDED',result
    assert result.evidence and result.report and result.visualization is None
    assert result.run_result.steps[-1].resolved_arguments['barcode_qc_manifest_sha256']==result.run_result.steps[-2].result['manifest_sha256']
    assert result.run_result.steps[-1].result['resource_qualification']=='synthetic_only'


@pytest.mark.parametrize('change',[{'min_tss_enrichment':0.1},{'min_tss_enrichment':'NaN'},{'min_qc_fragment_records':-1},{'min_qc_fragment_records':True}])
def test_preflight_threshold_rejection(tmp_path,change):
    args=dict(barcode_qc_manifest_path='/PRIVATE/qc.json',barcode_qc_manifest_sha256='1'*64,
        min_qc_fragment_records=1,min_tss_enrichment='0.1',output_dir=str(tmp_path/'out'))|change
    with pytest.raises(ValueError):build_default_tool_registry().validate_arguments('select_scATAC_cells',args)


def test_missing_threshold_and_optional_parameters(selection_case):
    args,_=selection_case;registry=build_default_tool_registry()
    for missing in ('min_qc_fragment_records','min_tss_enrichment'):
        with pytest.raises(ValueError):registry.validate_arguments('select_scATAC_cells',{k:v for k,v in args.items() if k!=missing})
    payload=wire();extra=dict(min_tss_flank_evidence=1,max_qc_fragment_records=100,max_nucleosome_signal=None)
    payload['decision']['steps'][0]['sources'].extend(dict(target=k,source=dict(kind='input',input=k)) for k in extra)
    plan=LLMPlanner(Model(payload)).plan(AgentRequest('qc','Explicit selection.',args|extra,RunMode.PLAN_ONLY),registry)
    for k,v in extra.items():assert plan.steps[0].arguments[k]==v


def test_empty_selection_application(selection_case,tmp_path):
    args,_=selection_case;args=args|dict(min_qc_fragment_records=10**18)
    app=ResearchAgentApplication(tmp_path/'empty-app',planner=LLMPlanner(Model(wire())))
    result=app.run(request(args))
    assert result.status.value=='SUCCEEDED',result
    assert result.run_result.steps[0].result['n_selected']==0
    assert result.run_result.steps[0].result['readiness']=='no_selected_cells'
    assert result.evidence and result.report


def request(args,mode=RunMode.EXECUTE):
    return AgentRequest('qc','Select candidate cells using the explicit supplied QC thresholds.',{k:v for k,v in args.items() if k!='output_dir'},mode)


class FixedPlanner:
    def __init__(self,args): self.value=AgentPlan('selection-plan','qc','Select candidate cells.',(PlanStep('qc','select_scATAC_cells',args),))
    def plan(self,request,registry):return self.value


def test_plan_only_has_zero_scientific_io(tmp_path,monkeypatch):
    args=dict(barcode_qc_manifest_path='/PRIVATE/qc.json',barcode_qc_manifest_sha256='1'*64,
        min_qc_fragment_records=100,min_tss_enrichment='0.1')
    def forbidden(*a,**k):pytest.fail('Scientific IO during PLAN_ONLY')
    monkeypatch.setattr(public,'verify_barcode_qc',forbidden)
    monkeypatch.setattr(production,'produce',forbidden)
    registry=build_default_tool_registry();registry=ToolRegistry(tuple(replace(registry.get(n),function=forbidden) for n in registry.names()))
    model=Model(wire());app=ResearchAgentApplication(tmp_path/'app',planner=LLMPlanner(model),registry=registry)
    result=app.run(request(args,RunMode.PLAN_ONLY))
    assert result.status.value=='PLANNED',result
    assert not result.run_result.steps and result.evidence is result.report is None
    assert '/PRIVATE' not in model.prompt and '/PRIVATE' not in str(model.schema)
    assert not list(app._workspace.run_paths(result.run_id).scientific.iterdir())


def test_v3_v4_and_registry(selection_case,monkeypatch):
    args,*_=selection_case;registry=build_default_tool_registry()
    req=AgentRequest('qc','Select.',args,RunMode.PLAN_ONLY)
    assert dict(LLMPlanner(Model(wire())).plan(req,registry).steps[0].arguments)==args
    payload=dict(schema_version=3,status='plan',reason=None,steps=[dict(step_id='qc',tool_name='select_scATAC_cells',
        arguments=({k:dict(binding_type='input',input_name=k) for k in args}|{k:None for k in ('min_tss_flank_evidence','max_qc_fragment_records','max_nucleosome_signal')}),depends_on=[],description='Select.')])
    assert dict(LLMPlanner(Model(payload),wire_mode=PlanningWireMode.V3).plan(req,registry).steps[0].arguments)==args
    assert len(registry.names())==18
    result=public.select_scATAC_cells(**args)
    monkeypatch.setattr(public,'verify_public_result',lambda *a:pytest.fail('IO-free result validation'))
    registry.validate_result('select_scATAC_cells',result)
    with pytest.raises(ValueError):registry.validate_result('select_scATAC_cells',result|{'n_observed_barcodes':True})


@pytest.mark.parametrize('mutation',[None,'qc','output'])
def test_application_report_resume(selection_case,tmp_path,monkeypatch,mutation):
    args,*_=selection_case;root=tmp_path/'app';app=ResearchAgentApplication(root,planner=LLMPlanner(Model(wire())))
    result=app.run(request(args))
    assert result.status.value=='SUCCEEDED',result
    assert result.evidence and result.report and result.visualization is None
    text=Path(result.report.path).read_text()
    assert 'QC-selected candidate cells' in text and 'not assessed' in text and 'FRiP' in text
    assert 'zero:background' not in text
    if mutation:
        target=args['barcode_qc_manifest_path'] if mutation=='qc' else str(Path(result.run_result.steps[0].result['manifest_path']).parent/'decisions.tsv.gz')
        with open(target,'ab') as f:f.write(b'changed')
    monkeypatch.setattr(production,'produce',lambda *a:pytest.fail('Production repeated'))
    resumed=ResearchAgentApplication(root).resume(result.run_id)
    assert resumed.status.value==('FAILED' if mutation else 'SUCCEEDED'),resumed


class Crash(BaseException):pass


class StopStore:
    def __init__(self,store,before=True,cancel=None):self.store=store;self.before=before;self.cancel=cancel
    def __getattr__(self,name):return getattr(self.store,name)
    def update(self,state,*,expected_revision):
        done=any(s.tool_name=='select_scATAC_cells' and s.status is StepStatus.SUCCEEDED for s in state.steps)
        if done and self.before:raise Crash()
        saved=self.store.update(state,expected_revision=expected_revision)
        if done:
            if self.cancel:self.cancel(saved.run_id)
            else:raise Crash()
        return saved


def test_application_publication_before_checkpoint(selection_case,tmp_path,monkeypatch):
    args,*_=selection_case;root=tmp_path/'recovery-app'
    app=ResearchAgentApplication(root,planner=LLMPlanner(Model(wire())))
    app.runtime._run_store=StopStore(app.run_store)
    with pytest.raises(Crash):app.run(request(args))
    monkeypatch.setattr(production,'produce',lambda *a:pytest.fail('Production repeated'))
    result=ResearchAgentApplication(root).resume('qc:run')
    assert result.status.value=='SUCCEEDED',result
    assert result.evidence and result.report and result.visualization is None


@pytest.mark.parametrize('before',[True,False])
@pytest.mark.parametrize('mutation',[None,'receipt','source','policy'])
def test_exact_receipt_recovery(selection_case,tmp_path,monkeypatch,before,mutation):
    args,*_=selection_case;store=FileRunStore(tmp_path/'store')
    with pytest.raises(Crash):AgentRuntime(planner=FixedPlanner(args),run_store=StopStore(store,before)).run(AgentRequest('qc','Select.',{}))
    if mutation:
        receipt=next(Path(args['output_dir']).glob('cell-selection-*/receipt.json'))
        if mutation=='source':Path(args['barcode_qc_manifest_path']).write_bytes(b'changed')
        elif mutation=='receipt':receipt.unlink()
        else:
            r=json.loads(receipt.read_bytes());r['policy']='wrong';receipt.write_text(json.dumps(r))
    monkeypatch.setattr(production,'produce',lambda *a:pytest.fail('Production repeated in recovery'))
    result=AgentRuntime(run_store=store).resume('qc:run')
    assert result.status.value==('FAILED' if mutation else 'SUCCEEDED'),result.errors


@pytest.mark.parametrize('boundary',['before','during','after'])
def test_cancellation(selection_case,tmp_path,monkeypatch,boundary):
    args,*_=selection_case;store=FileRunStore(tmp_path/'store');runtime=AgentRuntime(planner=FixedPlanner(args),run_store=store)
    if boundary=='before':
        class CancelCreate:
            def __getattr__(self,name):return getattr(store,name)
            def create(self,state):
                saved=store.create(state);store.request_cancellation(state.run_id);return saved
        runtime._run_store=CancelCreate()
        monkeypatch.setattr(production,'produce',lambda *a:pytest.fail('Production after cancellation'))
    elif boundary=='during':
        original=production.produce
        def cancel(*a,**k):
            runtime.cancel('qc:run');return original(*a,**k)
        monkeypatch.setattr(production,'produce',cancel)
    else:runtime._run_store=StopStore(store,False,lambda rid:runtime.cancel(rid))
    result=runtime.run(AgentRequest('qc','Select.',{}))
    assert result.status.value=='CANCELLED',result.errors
    manifests=list(Path(args['output_dir']).glob('cell-selection-*/manifest.json'))
    assert bool(manifests)==(boundary=='after')
    assert not list(Path(args['output_dir']).glob('.selection-attempt-*'))
    if boundary=='after':assert result.steps[0].status is StepStatus.SUCCEEDED
