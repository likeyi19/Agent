from dataclasses import asdict
import json
from pathlib import Path
import shutil
import pandas as pd
from scipy import sparse
import pytest

from agent.tools.analysis import annotation_contract as c, scatac_annotation as public, marker_annotation as science
from agent.orchestration import AgentRequest, AgentPlan, PlanStep, RunMode, LLMPlanner, PlanningWireMode, build_default_tool_registry
from agent.application import ResearchAgentApplication
from test_application import owner_calls

TOOL='annotate_scATAC_cell_types'


class Planner:
    def __init__(self,args): self.args=args
    def plan(self,request,registry):
        return AgentPlan('annotation-plan',request.request_id,'Explicit primary annotation.',(PlanStep('annotation',TOOL,self.args),))


class Model:
    model_id='scripted-annotation'
    def __init__(self,payload):self.payload=payload
    def complete(self,*,prompt,response_schema):
        # This fixture scripts detailed planning; selection is separately recorded.
        if "selection_schema_version" in response_schema.get("properties", {}):
            self.scope_calls = getattr(self, "scope_calls", []) + [(prompt, response_schema)]
            return json.dumps({"selection_schema_version": 1, "decision": {
                "kind": "select", "capability_ids": list(json.loads(prompt)["capabilities"])}})
        return json.dumps(self.payload)


def wire():
    return dict(schema_version=4,decision=dict(kind='plan',steps=[dict(step_id='annotation',tool=TOOL,
        sources=[dict(target=target,source=dict(kind='input',input=key)) for target,key in
                 (('matrix','matrix_manifest_path'),('specification','annotation_spec_path'))],control_dependencies=[])]))


@pytest.fixture
def case(completed,tmp_path,monkeypatch):
    _,_,run,store=completed
    result=run.steps[-1].result
    import anndata as ad
    a=ad.read_h5ad(result['matrix_path'])
    assert a.n_obs>=2
    group=tmp_path/'groups.tsv'
    pd.DataFrame({'cell_id':a.obs_names,'group':['group/a']+['NA']*(a.n_obs-1)}).to_csv(group,sep='\t',index=False)
    gene=tmp_path/'gene.tsv';gene.write_text('explicit synthetic gene dependency\n')
    signatures=tmp_path/'signatures.tsv';signatures.write_text('candidate\tgene\nA\tG\nA\tMISSING\nB\tH\nB\tMISSING\n')
    def resource(p,sem,norm):
        return asdict(science.Resource(str(p),science.sha256(p),'synthetic-test','r1','human','hg38','synthetic context','symbol',norm,sem))
    value=dict(schema=c.SPEC,source_run_id=run.run_id,source_step_id='matrix',species='human',assembly='hg38',context='synthetic context',
        groups=dict(path=str(group),sha256=science.sha256(group),provenance='explicit independent synthetic mapping'),
        gene_resource=resource(gene,'maestro-refgenes-transcripts-exons.v1','none'),
        signature_resource=resource(signatures,'candidate-gene-list.v1','uppercase'))
    spec=tmp_path/'annotation-inputs.json';spec.write_bytes(c.matrix.canonical(value))
    monkeypatch.setenv('AGENT_ANNOTATION_SOURCE_STORE',str(tmp_path/'state'))
    upstream=tmp_path/'native';upstream.mkdir()
    for name in science.SOURCES:
        p=upstream/name;p.parent.mkdir(exist_ok=True,parents=True);p.write_text('synthetic pinned backend fixture\n')
    rscript=tmp_path/'Rscript';rscript.write_text('synthetic runtime fixture\n')
    monkeypatch.setenv('AGENT_ANNOTATION_MAESTRO_ROOT',str(upstream))
    monkeypatch.setenv('AGENT_ANNOTATION_RSCRIPT',str(rscript))
    monkeypatch.setenv('AGENT_ANNOTATION_R_LIBRARY',str(tmp_path/'Rlib'))
    calls=[]
    def component(**kw):
        # Stub only the accepted native backend; exercise real M13 group/scoring,
        # publication, replay, derivative and accepted matrix authority machinery.
        calls.append(1)
        a,_=science.load_matrix(kw['matrix'])
        group_path=science.checked_file(kw['groups_path'],kw['groups_sha256'])
        groups=list(pd.read_csv(group_path,sep='\t',dtype=str,keep_default_na=False).itertuples(index=False,name=None))
        if len(science.exact_groups(a.obs_names.tolist(),groups))<2:raise ValueError('Two groups required')
        sig=science.read_signatures(kw['signature_resource'].path)
        scores=science.score_candidates(a.obs_names.tolist(),groups,['G','H'],[(groups[0][1],'G',1.)],sig)
        output=Path(kw['output_dir']);output.mkdir()
        for key,rows in scores.items():(output/(key+'.json')).write_text(json.dumps(rows))
        sparse.save_npz(output/'rp.npz',sparse.csr_matrix([[1.,0.]]*a.n_obs))
        for name in ('markers-native.tsv.gz','markers-maestro-filtered.tsv','markers-signature-input.tsv','group-sizes.tsv'):
            pd.DataFrame({'group':[groups[0][1]],'gene':['G'],'effect':[1.]}).to_csv(output/name,sep='\t',index=False)
        (output/'genes.tsv').write_text('G\nH\n');(output/'cells.tsv').write_text('\n'.join(a.obs_names)+'\n')
        shutil.copyfile(group_path,output/'groups.tsv');(output/'marker-runtime.txt').write_text('synthetic backend only\n')
        provenance=dict(matrix=asdict(kw['matrix']),gene_resource=asdict(kw['gene_resource']),
                        signature_resource=asdict(kw['signature_resource']),profile=kw['profile'],
                        sidecars={p.name:science.sha256(p) for p in output.iterdir()})
        (output/'result.json').write_text(json.dumps(provenance))
        return provenance
    monkeypatch.setattr(science,'annotate_cell_groups',component)
    args=dict(matrix_manifest_path=result['manifest_path'],matrix_manifest_sha256=result['manifest_sha256'],
        annotation_spec_path=str(spec),annotation_spec_sha256=science.sha256(spec),output_dir=str(tmp_path/'annotation'))
    return args,calls,store,run


def rewrite_spec(args,change):
    path=Path(args['annotation_spec_path']);value=json.loads(path.read_bytes());change(value)
    path.write_bytes(c.matrix.canonical(value))
    return args|{'annotation_spec_sha256':science.sha256(path)}


def test_application_and_terminal_reuse(case,tmp_path,monkeypatch):
    args,calls,store,source_run=case
    before=store.load(source_run.run_id)
    source_file=Path(source_run.steps[-1].result['matrix_path']);source_sha=science.sha256(source_file)
    app=ResearchAgentApplication(tmp_path/'app',planner=LLMPlanner(Model(wire())))
    request=AgentRequest('annotation','Annotate the supplied exact groups and context.',{k:v for k,v in args.items() if k!='output_dir'})
    with owner_calls() as upstream:
        result=app.run(request)
    assert result.status.value=='SUCCEEDED',result
    assert not upstream
    assert len(calls)==2  # one production call, one owner replay; no presentation replay
    assert result.evidence and result.report and result.visualization is None
    out=result.run_result.steps[0].result
    from agent.schemas.orchestration import _serialize
    build_default_tool_registry().validate_result(TOOL,_serialize(out))
    assert out['assigned_cells']==1 and out['unassigned_cells']==out['n_cells']-1
    assert out['group_summary'][1]['group']=='NA' and out['group_summary'][1]['primary_annotation'] is None
    authority=result.run_result.steps[0].verification.artifact_authority
    assert authority['schema_version']==2 and authority['upstream']['matrix']['authority_sha256']==out['source_authority_sha256']
    text=Path(result.report.path).read_text()
    assert 'Primary marker-based annotation' in text and 'Validation was not assessed' in text
    assert science.sha256(source_file)==source_sha and store.load(source_run.run_id)==before
    monkeypatch.setattr(science,'annotate_cell_groups',lambda **kw:pytest.fail('Terminal replayed annotation science'))
    with owner_calls() as upstream:
        resumed=app.resume(result.run_id)
    assert resumed==result and not upstream


@pytest.mark.parametrize('mutation',['context','assembly','species','source_step','source_run','group_duplicate','group_missing','group_unknown','resource_hash'])
def test_invalid_inputs_fail_closed(case,mutation):
    args,calls,_,_=case
    def change(v):
        if mutation in ('context','assembly','species'):v[mutation]={'context':'guessed','assembly':'mm10','species':'mouse'}[mutation]
        elif mutation=='source_step':v['source_step_id']='selection'
        elif mutation=='source_run':v['source_run_id']='nonexistent'
        elif mutation=='resource_hash':v['gene_resource']['sha256']='0'*64
        else:
            p=Path(v['groups']['path']);rows=p.read_text().splitlines()
            if mutation=='group_duplicate':rows.append(rows[1])
            elif mutation=='group_missing':rows.pop()
            else:rows[1]='unknown\tgroup/a'
            p.write_text('\n'.join(rows)+'\n');v['groups']['sha256']=science.sha256(p)
    args=rewrite_spec(args,change)
    with pytest.raises((ValueError,FileNotFoundError)):
        public.execute_annotation(args)
    assert not list(Path(args['output_dir']).glob('annotation-*'))
    assert not list(Path(args['output_dir']).glob('.matrix-attempt-*'))


@pytest.mark.parametrize('mutation',['evidence','h5ad','groups','matrix_binding','resource'])
def test_rehashed_tampering_fails(case,mutation):
    args,_,_,_=case
    result=public.execute_annotation(args)
    path=Path(result['manifest_path']);value=c.load_manifest(path,result['manifest_sha256'])
    if mutation=='evidence':
        p=path.parent/'primary/groups.json';rows=json.loads(p.read_bytes());rows[0]['primary_annotation']='forged';p.write_text(json.dumps(rows))
        value['sidecars']['primary/groups.json']=dict(sha256=science.sha256(p),size_bytes=p.stat().st_size)
    elif mutation=='h5ad':
        import anndata as ad
        p=path.parent/'annotated.h5ad';a=ad.read_h5ad(p);a.X.data[0]+=1;a.write_h5ad(p)
        value['sidecars']['annotated.h5ad']=dict(sha256=science.sha256(p),size_bytes=p.stat().st_size)
    elif mutation=='groups':value['inputs']['groups']['sha256']='0'*64
    elif mutation=='matrix_binding':value['matrix_binding']['ordered_cells_sha256']='0'*64
    else:Path(value['inputs']['signature_resource']['path']).write_text('tampered')
    if mutation=='evidence':
        primary=path.parent/'primary/result.json';provenance=json.loads(primary.read_bytes())
        provenance['sidecars']['groups.json']=value['sidecars']['primary/groups.json']['sha256']
        primary.write_text(json.dumps(provenance))
        value['sidecars']['primary/result.json']=dict(sha256=science.sha256(primary),size_bytes=primary.stat().st_size)
    path.write_bytes(c.matrix.canonical(value))
    with public._operation(),pytest.raises((ValueError,AssertionError)):
        public.verify_annotation(path,expected_sha256=science.sha256(path))


@pytest.mark.parametrize('version',[3,4])
def test_planner_and_plan_only(tmp_path,monkeypatch,version):
    args={k:'1'*64 if k.endswith('_sha256') else '/PRIVATE/'+k for k in c.ARGUMENTS}
    payload=wire() if version==4 else dict(schema_version=3,status='plan',reason=None,steps=[dict(step_id='annotation',tool_name=TOOL,
        arguments={k:dict(binding_type='input',input_name=k) for k in args},depends_on=[],description='Annotate.')])
    planner=LLMPlanner(Model(payload),wire_mode=PlanningWireMode.V4 if version==4 else PlanningWireMode.V3)
    request=AgentRequest('annotation','Annotate explicit groups.',args,RunMode.PLAN_ONLY)
    assert dict(planner.plan(request,build_default_tool_registry()).steps[0].arguments)==args
    monkeypatch.setattr(c,'specification',lambda *a:pytest.fail('PLAN_ONLY opened biological inputs'))
    app=ResearchAgentApplication(tmp_path/'app',planner=planner)
    request=AgentRequest('annotation','Annotate explicit groups.',{k:v for k,v in args.items() if k!='output_dir'},RunMode.PLAN_ONLY)
    result=app.run(request)
    assert result.status.value=='PLANNED',result
    assert result.evidence is None and result.report is None


def test_missing_specification_not_invented():
    request=AgentRequest('missing','Annotate.',{'matrix_manifest_path':'/matrix.json','matrix_manifest_sha256':'a'*64},RunMode.PLAN_ONLY)
    with pytest.raises(ValueError):LLMPlanner(Model(wire())).plan(request,build_default_tool_registry())


def test_recovery_without_production(case,monkeypatch):
    args,calls,_,_=case
    result=public.execute_annotation(args,'execution')
    monkeypatch.setattr(public,'_build',lambda *a:pytest.fail('Recovery reran publication production'))
    with owner_calls() as upstream:
        assert public.recover_annotation(args,'execution')==result
    assert not upstream and len(calls)==3  # strict recovery verifies M13; never M11
    with pytest.raises(ValueError):public.recover_annotation(args,'different-execution')


def test_crash_recovery_issues_and_preserves_authority(case,tmp_path,monkeypatch):
    from agent.orchestration import AgentRuntime, FileRunStore, StepStatus
    args,calls,_,_=case
    store=FileRunStore(tmp_path/'interrupted-store')
    class Crash(BaseException):pass
    class StopStore:
        def __getattr__(self,name):return getattr(store,name)
        def update(self,state,*,expected_revision):
            if any(s.status is StepStatus.SUCCEEDED for s in state.steps):raise Crash()
            return store.update(state,expected_revision=expected_revision)
    with pytest.raises(Crash):
        AgentRuntime(planner=Planner(args),run_store=StopStore()).run(AgentRequest('recover','Annotate.',{}))
    assert len(calls)==2
    monkeypatch.setattr(public,'_build',lambda *a:pytest.fail('Recovery republished annotation'))
    with owner_calls() as upstream:
        result=AgentRuntime(run_store=store).resume('recover:run')
    assert result.status.value=='SUCCEEDED',result
    assert len(calls)==3 and not upstream
    assert result.steps[0].verification.artifact_authority['schema_version']==2
    from agent.orchestration.verification_authority import accepted_authorities
    monkeypatch.setattr(science,'annotate_cell_groups',lambda **kw:pytest.fail('Recovered authority was lost'))
    with accepted_authorities(store,result.run_id):
        public.verify_public_result(args,dict(result.steps[0].result)|{'group_summary':
            [dict(r) for r in result.steps[0].result['group_summary']]})


def test_ambiguous_cells_have_null_primary_labels(case):
    args,_,_,_=case
    def change(v):
        path=Path(v['signature_resource']['path'])
        path.write_text('candidate\tgene\nA\tG\nA\tMISSING\nB\tG\nB\tMISSING\n')
        v['signature_resource']['sha256']=science.sha256(path)
    args=rewrite_spec(args,change)
    result=public.execute_annotation(args)
    assert result['assigned_cells']==0 and result['ambiguous_cells']==1
    import anndata as ad
    a=ad.read_h5ad(result['annotated_h5ad_path'])
    assert a.obs['primary_annotation'].isna().all()
    assert list(a.obs['annotation_status'])==['ambiguous','insufficient_evidence']


def test_untrusted_or_unsupported_matrix_owner_rejected(case,monkeypatch):
    from agent.orchestration import verification_authority as owner
    from dataclasses import replace
    args,calls,_,_=case
    original=owner._accepted
    def unsupported(*a):
        stored,*rest=original(*a)
        return (replace(stored,tool_name='adopt_scATAC_cell_by_ccre'),*rest)
    monkeypatch.setattr(owner,'_accepted',unsupported)
    with pytest.raises(ValueError,match='canonical matrix owner'):
        public.execute_annotation(args)
    assert not calls


def test_cancel_before_work(case):
    from agent.tools._cancellation import cancellation_scope,ToolWorkCancelled
    args,calls,_,_=case
    with cancellation_scope(lambda:True),pytest.raises(ToolWorkCancelled):
        public.execute_annotation(args)
    assert not calls and not Path(args['output_dir']).exists()


def test_missing_matrix_authority_never_triggers_qualification(case,monkeypatch):
    from dataclasses import replace
    from agent.orchestration import FileRunStore
    args,calls,_,_=case
    original=FileRunStore.load
    def legacy(store,run_id):
        state=original(store,run_id)
        return replace(state,steps=tuple(replace(s,verification=replace(s.verification,artifact_authority=None))
            if s.tool_name=='build_scATAC_cell_by_ccre' else s for s in state.steps))
    monkeypatch.setattr(FileRunStore,'load',legacy)
    with owner_calls() as upstream,pytest.raises(ValueError,match='legacy qualification is not implicit'):
        public.execute_annotation(args)
    assert not calls and not upstream
