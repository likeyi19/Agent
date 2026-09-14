from pathlib import Path
import pytest
from agent.tools.data import scatac_matrix_adoption as public
from agent.application import ResearchAgentApplication
from agent.orchestration import AgentRequest, AgentPlan, PlanStep, RunMode, build_default_tool_registry


class Planner:
    def __init__(self,args): self.args=args
    def plan(self,request,registry):
        return AgentPlan('external-plan',request.request_id,'Adopt external matrix.',(PlanStep('adopt','adopt_scATAC_cell_by_ccre',self.args),))


def test_public_recovery(case,monkeypatch):
    result=public.execute_matrix(case,'test-execution')
    public.verify_public_result(case,result)
    build_default_tool_registry().validate_result('adopt_scATAC_cell_by_ccre',result)
    monkeypatch.setattr(public.scientific,'build',lambda *a: pytest.fail('Recovery reran adoption'))
    assert public.recover_matrix(case,'test-execution')==result
    with pytest.raises(ValueError): public.recover_matrix(case,'other-execution')
    with pytest.raises(ValueError,match='MATRIX_OUTPUT_CONFLICT'): public.execute_matrix(case,'test-execution')


def test_application_authority_reuse(case,tmp_path,monkeypatch):
    from agent.tools.data import cell_by_ccre_verifier as owner
    calls=[]; original=owner._verify_external
    def observed(*a,**kw):
        calls.append(1)
        return original(*a,**kw)
    monkeypatch.setattr(owner,'_verify_external',observed)
    app=ResearchAgentApplication(tmp_path/'app',planner=Planner(case))
    result=app.run(AgentRequest('external','Adopt the external matrix.',{}))
    assert calls==[1]
    assert result.status.value=='SUCCEEDED',result
    assert result.evidence and result.report and result.visualization is None
    text=Path(result.report.path).read_text()
    assert 'externally declared' in text and 'exact source value' in text
    assert 'QC-selected candidates' not in text
    authority=result.run_result.steps[0].verification.artifact_authority
    assert authority['schema_version']==2
    assert authority['verifier']['id']=='agent.cell-by-ccre-independent'
    monkeypatch.setattr(owner,'_verify_external',lambda *a: pytest.fail('Repeated source conservation'))
    Path(case['source_path']).unlink()  # historical source policy after accepted proof
    resumed=app.resume(result.run_id)
    assert resumed.status.value=='SUCCEEDED',resumed
    from agent.orchestration.verification_authority import accepted_authorities
    with pytest.raises((ValueError,OSError)):
        with accepted_authorities(app.run_store,result.run_id,source_policy='current_source_freshness.v1'): pass
    with Path(result.run_result.steps[0].result['matrix_path']).open('ab') as stream: stream.write(b'changed')
    assert app.resume(result.run_id).status.value=='FAILED'


def test_plan_only(case,tmp_path,monkeypatch):
    import h5py
    monkeypatch.setattr(h5py,'File',lambda *a,**k: pytest.fail('Scientific IO during PLAN_ONLY'))
    app=ResearchAgentApplication(tmp_path/'app',planner=Planner(case))
    result=app.run(AgentRequest('external','Adopt.',{},RunMode.PLAN_ONLY))
    assert result.status.value=='PLANNED',result
    assert result.evidence is None and result.report is None


class Model:
    model_id="scripted-external-matrix"
    def __init__(self,payload): self.payload=payload
    def complete(self,*,prompt,response_schema):
        import json
        return json.dumps(self.payload)


@pytest.mark.parametrize('version',[3,4])
def test_semantic_wires(case,tmp_path,version):
    from agent.orchestration import LLMPlanner, PlanningWireMode
    if version==4:
        payload=dict(schema_version=4,decision=dict(kind='plan',steps=[dict(step_id='adopt',tool='adopt_scATAC_cell_by_ccre',
            sources=[dict(target=target,source=dict(kind='input',input=key)) for target,key in (
                ('source','source_path'),('reference','reference_manifest_path'),('species','species'),
                ('assembly','assembly'),('matrix_semantics','matrix_semantics'))],control_dependencies=[])]))
    else:
        payload=dict(schema_version=3,status='plan',reason=None,steps=[dict(step_id='adopt',tool_name='adopt_scATAC_cell_by_ccre',
            arguments={k:dict(binding_type='input',input_name=k) for k in case},depends_on=[],description='Adopt.')])
    planner=LLMPlanner(Model(payload),wire_mode=PlanningWireMode.V4 if version==4 else PlanningWireMode.V3)
    plan=planner.plan(AgentRequest('external','Adopt external matrix.',case,RunMode.PLAN_ONLY),build_default_tool_registry())
    assert dict(plan.steps[0].arguments)==case


def test_recovery_before_success_checkpoint(case,tmp_path,monkeypatch):
    from agent.orchestration import AgentRuntime,FileRunStore,StepStatus
    class Crash(BaseException): pass
    store=FileRunStore(tmp_path/'store')
    class StopStore:
        def __getattr__(self,name): return getattr(store,name)
        def update(self,state,*,expected_revision):
            if any(s.status is StepStatus.SUCCEEDED for s in state.steps): raise Crash()
            return store.update(state,expected_revision=expected_revision)
    with pytest.raises(Crash):
        AgentRuntime(planner=Planner(case),run_store=StopStore()).run(AgentRequest('external','Adopt.',{}))
    monkeypatch.setattr(public.scientific,'build',lambda *a: pytest.fail('Recovery reran adoption'))
    result=AgentRuntime(run_store=store).resume('external:run')
    assert result.status.value=='SUCCEEDED',result


def test_cancellation_cleans_private_stage(case,monkeypatch):
    from agent.tools._cancellation import cancellation_scope, ToolWorkCancelled
    calls=0
    def cancel():
        nonlocal calls
        calls+=1
        return calls>=10
    with cancellation_scope(cancel),pytest.raises(ToolWorkCancelled): public.execute_matrix(case)
    assert not list(Path(case['output_dir']).glob('cell-by-ccre-*'))
    assert not list(Path(case['output_dir']).glob('.matrix-attempt-*'))

