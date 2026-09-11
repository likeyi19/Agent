from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pytest
from agent.application import ResearchAgentApplication
from agent.orchestration import (AgentRequest,AgentRuntime,AgentPlan,PlanStep,LLMPlanner,PlanningWireMode,
    RunMode,FileRunStore,StepStatus,ToolRegistry,StepOutputRef,build_default_tool_registry)
from agent.tools.data import scatac_matrix as public, scatac_matrix_contract as m
from cell_selection.test_integration import Model, Crash

TOOL='build_scATAC_cell_by_ccre'


def wire():
    return dict(schema_version=4,decision=dict(kind='plan',steps=[dict(step_id='matrix',tool=TOOL,
        sources=[dict(target=port,source=dict(kind='input',input=key)) for port,key in (
            ('fragments','fragments_manifest_path'),('selected_cells','selected_cells_manifest_path'),('reference','reference_manifest_path'))],control_dependencies=[])]))


def request(args,mode=RunMode.EXECUTE):
    return AgentRequest('matrix','Build the full ordered cell-by-cCRE matrix from these fragments and QC-selected cells.',
        {k:v for k,v in args.items() if k!='output_dir'},mode)


class FixedPlanner:
    def __init__(self,args): self.value=AgentPlan('matrix-plan','matrix','Build full matrix.',(PlanStep('matrix',TOOL,args),))
    def plan(self,request,registry): return self.value


@pytest.mark.parametrize('species',['human','mouse'])
@pytest.mark.parametrize('empty',[False,True])
def test_public_and_application(matrix_case,tmp_path,species,empty):
    import anndata as ad
    import numpy as np
    from scipy import sparse
    args=matrix_case(species=species,empty=empty)
    result=public.build_scATAC_cell_by_ccre(**args)
    manifest=public.verify_public_result(args,result)
    build_default_tool_registry().validate_result(TOOL,result)
    assert result['n_cells']==(0 if empty else 3)
    assert result['n_features']==5 and result['zero_row_count']==(0 if empty else 1)
    assert result['readiness']==('no_selected_cells' if empty else 'matrix_available')
    assert result['matrix_semantics']=='fragment_counts' and result['matrix_profile_id']==m.PROFILE.profile_id
    a=ad.read_h5ad(result['matrix_path']); assert sparse.isspmatrix_csr(a.X) and a.X.dtype==np.int64
    assert a.shape==((0 if empty else 3),5)
    if not empty:
        assert a.X[2].nnz==0 and a.X.nnz==8
        assert list(a.X.data)==[1,2,3,2,1,1,2,1]
    assert result['ordered_feature_sha256']==manifest['ordered_feature_sha256']
    assert set(p.name for p in Path(result['manifest_path']).parent.iterdir())=={'manifest.json','matrix.h5ad'}
    app=ResearchAgentApplication(tmp_path/'app',planner=LLMPlanner(Model(wire())))
    run=app.run(request(args)); assert run.status.value=='SUCCEEDED',run
    assert run.evidence and run.report and run.visualization is None
    out=run.run_result.steps[0].result
    assert (out['logical_matrix_sha256'],out['ordered_selected_sha256'],out['ordered_feature_sha256'])==tuple(result[k] for k in ('logical_matrix_sha256','ordered_selected_sha256','ordered_feature_sha256'))
    text=Path(run.report.path).read_text()
    assert 'canonical fragment-record counts' in text and 'full reference vocabulary' in text
    assert 'FRiP' not in text and 'unique molecules' not in text
    assert 'does not establish model readiness' in text
    assert 'source.bed' not in text
    assert len(Path(run.evidence.path).read_bytes())<50000


@pytest.mark.parametrize('change',[{'fragments_manifest_sha256':'bad'}, {'reference_manifest_path':'relative.json'},
    {'selected_cells_manifest_path':'/a/../b'}, {'output_dir':'relative'}, {'count_mode':'support'}, {'fragments_manifest_sha256':True}])
def test_mechanical_preflight_no_io(tmp_path,change):
    args={k:('/PRIVATE/'+k if not k.endswith('_sha256') else '1'*64) for k in public.ARGUMENTS}|change
    with pytest.raises(ValueError): build_default_tool_registry().validate_arguments(TOOL,args)


def test_registry_and_wires(matrix_case,monkeypatch):
    args=matrix_case(); registry=build_default_tool_registry()
    assert len(registry.names())==18 and registry.names().count(TOOL)==1
    spec=registry.get(TOOL)
    assert set(spec.required_arguments)==set(public.ARGUMENTS) and not spec.optional_arguments
    assert tuple(p.name for p in spec.semantic_planning.consumer_ports)==('fragments','selected_cells','reference','output_dir')
    assert tuple(p.name for p in spec.semantic_planning.producer_ports)==('matrix','dataset')
    assert not spec.retryable_error_codes
    req=AgentRequest('matrix','Build the requested matrix.',args,RunMode.PLAN_ONLY)
    plan=LLMPlanner(Model(wire())).plan(req,registry)
    assert dict(plan.steps[0].arguments)==args
    payload=dict(schema_version=3,status='plan',reason=None,steps=[dict(step_id='matrix',tool_name=TOOL,
        arguments={k:dict(binding_type='input',input_name=k) for k in args},depends_on=[],description='Build.')])
    assert dict(LLMPlanner(Model(payload),wire_mode=PlanningWireMode.V3).plan(req,registry).steps[0].arguments)==args
    result=public.build_scATAC_cell_by_ccre(**args)
    monkeypatch.setattr(public,'verify_public_result',lambda *a:pytest.fail('Scientific result validation IO'))
    registry.validate_result(TOOL,result)
    for change in ({'n_cells':True},{'species':'mouse'},{'nnz':0},{'matrix_semantics':'binary_accessibility'},
                   {'diagnostic':result['diagnostic']|{'unknown':'payload'}},{'unexpected':'payload'}):
        with pytest.raises(ValueError):registry.validate_result(TOOL,result|change)


def test_plan_only_has_zero_scientific_io(tmp_path,monkeypatch):
    import subprocess
    import h5py
    from agent.tools.data import _cell_by_ccre_io as io
    args={k:('/PRIVATE/'+k if not k.endswith('_sha256') else '1'*64) for k in public.ARGUMENTS if k!='output_dir'}
    def forbidden(*a,**k):pytest.fail('Scientific IO during PLAN_ONLY')
    monkeypatch.setattr(public.scientific,'build_cell_by_ccre',forbidden)
    monkeypatch.setattr(public,'verify_cell_by_ccre',forbidden)
    monkeypatch.setattr(io,'bind',forbidden)
    monkeypatch.setattr(h5py,'File',forbidden)
    monkeypatch.setattr(subprocess,'Popen',forbidden)
    registry=build_default_tool_registry(); registry=ToolRegistry(tuple(replace(registry.get(n),function=forbidden) for n in registry.names()))
    model=Model(wire());app=ResearchAgentApplication(tmp_path/'app',planner=LLMPlanner(model),registry=registry)
    result=app.run(request(args,RunMode.PLAN_ONLY))
    assert result.status.value=='PLANNED',result
    assert not result.run_result.steps and result.evidence is result.report is None
    assert '/PRIVATE' not in model.prompt and '/PRIVATE' not in str(model.schema)
    assert not list(app._workspace.run_paths(result.run_id).scientific.iterdir())


@pytest.mark.parametrize('mutation',['lineage','hash','contract','backend'])
def test_failure_before_construction(matrix_case,tmp_path,monkeypatch,mutation):
    from agent.tools.data import _cell_by_ccre_production as production
    args=matrix_case()
    if mutation=='lineage':
        other=matrix_case(species='mouse')
        args|={k:other[k] for k in ('reference_manifest_path','reference_manifest_sha256')}
    elif mutation=='hash':args['fragments_manifest_sha256']='0'*64
    elif mutation=='contract':
        path=tmp_path/'unsupported.json';path.write_text('{"contract_version":"unsupported"}')
        args['selected_cells_manifest_path']=str(path);args['selected_cells_manifest_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    else:monkeypatch.delenv('AGENT_MATRIX_BEDTOOLS')
    monkeypatch.setattr(production,'construct_counts',lambda *a,**k:pytest.fail('Production after invalid authorities'))
    result=AgentRuntime(planner=FixedPlanner(args),run_store=FileRunStore(tmp_path/'store')).run(AgentRequest('matrix','Build.',{}))
    assert result.status.value=='FAILED',result
    assert not list(Path(args['output_dir']).glob('cell-by-ccre-*'))
    assert not list(Path(args['output_dir']).glob('.matrix-attempt-*'))


def test_output_conflict(matrix_case):
    args=matrix_case();r=public.build_scATAC_cell_by_ccre(**args)
    before=Path(r['matrix_path']).read_bytes()
    with pytest.raises(ValueError,match='MATRIX_OUTPUT_CONFLICT'):public.build_scATAC_cell_by_ccre(**args)
    assert Path(r['matrix_path']).read_bytes()==before


class StopStore:
    def __init__(self,store,before=True,cancel=None):self.store=store;self.before=before;self.cancel=cancel
    def __getattr__(self,name):return getattr(self.store,name)
    def update(self,state,*,expected_revision):
        done=any(s.tool_name==TOOL and s.status is StepStatus.SUCCEEDED for s in state.steps)
        if done and self.before:raise Crash()
        saved=self.store.update(state,expected_revision=expected_revision)
        if done:
            if self.cancel:self.cancel(saved.run_id)
            else:raise Crash()
        return saved


@pytest.mark.parametrize('before',[True,False])
@pytest.mark.parametrize('mutation',[None,'receipt','source','policy','matrix','result','arguments'])
def test_exact_recovery(matrix_case,tmp_path,monkeypatch,before,mutation):
    args=matrix_case();store=FileRunStore(tmp_path/'store')
    with pytest.raises(Crash):AgentRuntime(planner=FixedPlanner(args),run_store=StopStore(store,before)).run(AgentRequest('matrix','Build.',{}))
    receipt=next(Path(args['output_dir']).glob('cell-by-ccre-*/receipt.json'))
    if mutation:
        if mutation=='source':Path(args['fragments_manifest_path']).write_bytes(b'changed')
        elif mutation=='receipt':receipt.unlink()
        elif mutation=='matrix':
            with (receipt.parent/'artifact/matrix.h5ad').open('ab') as f:f.write(b'changed')
        else:
            r=json.loads(receipt.read_bytes())
            r[{'policy':'policy','result':'manifest_sha256','arguments':'arguments_sha256'}[mutation]]='0'*64
            receipt.write_text(json.dumps(r))
    monkeypatch.setattr(public.scientific,'build_cell_by_ccre',lambda *a,**k:pytest.fail('Production repeated in recovery'))
    result=AgentRuntime(run_store=store).resume('matrix:run')
    assert result.status.value==('FAILED' if mutation else 'SUCCEEDED'),result.errors


def test_application_publication_before_checkpoint(matrix_case,tmp_path,monkeypatch):
    args=matrix_case();root=tmp_path/'app'
    app=ResearchAgentApplication(root,planner=LLMPlanner(Model(wire())))
    app.runtime._run_store=StopStore(app.run_store)
    with pytest.raises(Crash):app.run(request(args))
    monkeypatch.setattr(public.scientific,'build_cell_by_ccre',lambda *a,**k:pytest.fail('Production repeated'))
    result=ResearchAgentApplication(root).resume('matrix:run')
    assert result.status.value=='SUCCEEDED',result
    assert result.evidence and result.report and result.visualization is None


@pytest.mark.parametrize('boundary',['before','during','verification','after'])
def test_cancellation(matrix_case,tmp_path,monkeypatch,boundary):
    args=matrix_case();store=FileRunStore(tmp_path/'store');runtime=AgentRuntime(planner=FixedPlanner(args),run_store=store)
    if boundary=='before':
        class CancelCreate:
            def __getattr__(self,name):return getattr(store,name)
            def create(self,state):
                saved=store.create(state);store.request_cancellation(state.run_id);return saved
        runtime._run_store=CancelCreate()
        monkeypatch.setattr(public.scientific,'build_cell_by_ccre',lambda *a,**k:pytest.fail('Production after cancellation'))
    elif boundary in ('during','verification'):
        from agent.tools.data import _cell_by_ccre_production as production, cell_by_ccre_verifier as verifier
        module,fn=(production,'construct_counts') if boundary=='during' else (public.scientific,'verify_cell_by_ccre')
        original=getattr(module,fn)
        def cancel(*a,**k):runtime.cancel('matrix:run');return original(*a,**k)
        monkeypatch.setattr(module,fn,cancel)
    else:runtime._run_store=StopStore(store,False,lambda rid:runtime.cancel(rid))
    result=runtime.run(AgentRequest('matrix','Build.',{}))
    assert result.status.value=='CANCELLED',result.errors
    manifests=list(Path(args['output_dir']).glob('cell-by-ccre-*/artifact/manifest.json'))
    assert bool(manifests)==(boundary=='after')
    assert not list(Path(args['output_dir']).glob('.matrix-attempt-*'))
    if boundary=='after':
        assert result.steps[0].status is StepStatus.SUCCEEDED
        assert runtime.resume('matrix:run').status.value=='CANCELLED'


@pytest.mark.parametrize('include_import',[False,True])
def test_full_application_composition(matrix_case,tmp_path,include_import):
    from barcode_qc.test_integration import wire as qc_wire
    from cell_selection.test_integration import wire as selection_wire
    from agent.tools.data import external_fragment_manifest as em
    args=matrix_case();root=Path(args['reference_manifest_path']).parent
    matrix=wire()['decision']['steps'][0]
    matrix['sources'][1]['source']=dict(kind='step',step='selection')
    selection=selection_wire()['decision']['steps'][0];selection['step_id']='selection'
    selection['sources'][0]['source']=dict(kind='step',step='qc')
    steps=[qc_wire()['decision']['steps'][0],selection,matrix]
    qc_ref=root/'qc-reference/manifest.json'
    inputs={k:v for k,v in args.items() if k!='output_dir' and not k.startswith('selected_cells_manifest')}
    inputs|=dict(qc_reference_manifest_path=str(qc_ref),qc_reference_manifest_sha256=hashlib.sha256(qc_ref.read_bytes()).hexdigest(),
                 min_qc_fragment_records=0,min_tss_enrichment='0')
    if include_import:
        for step in (steps[0],matrix):step['sources'][0]['source']=dict(kind='step',step='adopt')
        steps.insert(0,dict(step_id='adopt',tool='import_scATAC_fragments',control_dependencies=[],
            sources=[dict(target=port,source=dict(kind='input',input=key)) for port,key in (
                ('source','source_path'),('reference','reference_bundle_path'),('namespace','namespace'),('source_profile','source_profile'))]))
        inputs={k:v for k,v in inputs.items() if not k.startswith('fragments_manifest')}
        source=root/'source.bed'
        inputs|=dict(source_path=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_profile=em.PROFILE_ID,
                     reference_bundle_path=args['reference_manifest_path'],reference_bundle_sha256=args['reference_manifest_sha256'],namespace='library_0')
    payload=dict(schema_version=4,decision=dict(kind='plan',steps=steps))
    app=ResearchAgentApplication(tmp_path/'composed',planner=LLMPlanner(Model(payload)))
    result=app.run(AgentRequest('composed','Compute QC, select candidates at the supplied thresholds, and build the full matrix.',inputs))
    assert result.status.value=='SUCCEEDED',result
    assert result.evidence and result.report and result.visualization is None
    final=result.run_result.steps[-1]
    assert final.resolved_arguments['selected_cells_manifest_sha256']==result.run_result.steps[-2].result['manifest_sha256']
    if include_import:assert final.resolved_arguments['fragments_manifest_sha256']==result.run_result.steps[0].result['manifest_sha256']
    assert final.result['n_cells']==3 and final.result['zero_row_count']==1
    assert not any(s.tool_name=='epizoo_embed_cells' for s in result.run_result.steps)


@pytest.mark.parametrize('empty',[False,True])
def test_downstream_feature_space(matrix_case,tmp_path,empty):
    args=matrix_case(empty=empty)
    mapping=dict(matrix_source='X',matrix_semantics='fragment_counts',coordinate_source='var_columns',
        feature_chrom_key='chrom',feature_start_key='start',feature_end_key='end',coordinate_system='zero_based_half_open',
        semantics_metadata_key='matrix_semantics',species='human',genome_assembly='hg38')
    payload=wire();payload['decision']['steps'].append(dict(step_id='feature_space',tool='validate_scATAC_feature_space',control_dependencies=[],
        sources=[dict(target='input_path',source=dict(kind='step',step='matrix'))]+
            [dict(target=k,source=dict(kind='input',input=k)) for k in mapping]))
    app=ResearchAgentApplication(tmp_path/'downstream',planner=LLMPlanner(Model(payload)))
    result=app.run(AgentRequest('matrix','Build the matrix and validate its declared raw fragment-count feature space.',
        {k:v for k,v in args.items() if k!='output_dir'}|mapping))
    assert result.status.value==('FAILED' if empty else 'SUCCEEDED'),result
    assert result.run_result.steps[0].status is StepStatus.SUCCEEDED
    assert result.run_result.steps[1].resolved_arguments['input_path']==result.run_result.steps[0].result['matrix_path']
    if not empty:
        out=result.run_result.steps[1].result
        assert out['n_cells']==3 and out['n_features']==5 and out['matrix_semantics']=='fragment_counts'
        assert out['pseudobulk_eligible'] is True
        assert result.evidence and result.report


@pytest.mark.parametrize('change',['wrong_upstream','missing_pair','unknown_target','static_knob'])
def test_invalid_semantic_plan(tmp_path,change):
    args={k:('/PRIVATE/'+k if not k.endswith('_sha256') else '1'*64) for k in public.ARGUMENTS}
    payload=wire()
    if change=='missing_pair':args.pop('fragments_manifest_sha256')
    elif change=='unknown_target':payload['decision']['steps'][0]['sources'][0]['target']='support_weighting'
    elif change=='static_knob':
        payload['decision']['steps'][0]['sources'].append(dict(target='dtype',source=dict(kind='input',input='dtype')));args['dtype']='int32'
    else:
        payload['decision']['steps'][0]['sources'][0]['source']=dict(kind='step',step='inspect')
        payload['decision']['steps'].insert(0,dict(step_id='inspect',tool='inspect_scATAC',control_dependencies=[],
            sources=[dict(target='dataset',source=dict(kind='input',input='input_path'))]))
        args['input_path']='/PRIVATE/input.h5ad'
    with pytest.raises(ValueError):LLMPlanner(Model(payload)).plan(AgentRequest('matrix','Build.',args,RunMode.PLAN_ONLY),build_default_tool_registry())


def test_active_subprocess_cancellation(matrix_case,tmp_path,monkeypatch):
    import subprocess
    import sys
    from types import SimpleNamespace
    from agent.tools.data import _cell_by_ccre_production as production
    args=matrix_case();store=FileRunStore(tmp_path/'store');runtime=AgentRuntime(planner=FixedPlanner(args),run_store=store)
    children=[]
    def active(*a,**k):
        process=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        children.append(process);runtime.cancel('matrix:run');return process
    monkeypatch.setattr(production,'subprocess',SimpleNamespace(Popen=active,PIPE=subprocess.PIPE,TimeoutExpired=subprocess.TimeoutExpired))
    result=runtime.run(AgentRequest('matrix','Build.',{}))
    assert result.status.value=='CANCELLED',result.errors
    assert len(children)==1 and children[0].poll() is not None
    assert not list(Path(args['output_dir']).glob('.matrix-attempt-*'))
    assert not list(Path(args['output_dir']).glob('cell-by-ccre-*'))
