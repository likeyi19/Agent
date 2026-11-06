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
            pd.DataFrame({'cluster':[groups[0][1]],'gene':['G'],'avg_logFC':[1.]}).to_csv(output/name,sep='\t',index=False)
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




@pytest.mark.parametrize('version',[3,4])
def test_planner_and_plan_only(tmp_path,monkeypatch,version):
    args={k:'1'*64 if k.endswith('_sha256') else '/PRIVATE/'+k for k in c.ARGUMENTS}
    payload=wire() if version==4 else dict(schema_version=3,status='plan',reason=None,steps=[dict(step_id='annotation',tool_name=TOOL,
        arguments={k:dict(binding_type='input',input_name=k) for k in args},depends_on=[],description='Annotate.')])
    if version==3:
        for key in build_default_tool_registry().get(TOOL).optional_arguments:
            payload['steps'][0]['arguments'].setdefault(key,None)
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
