"""M11.2d production orchestration with generated data and fake external tools."""
from dataclasses import replace
import errno
import json
from pathlib import Path
from unittest.mock import Mock
import pytest

from agent.orchestration import (AgentPlan, AgentRequest, AgentRuntime, FileRunStore, PlanStep,
    RunMode, RunStatus, RunLifecycleStatus, StepStatus, StepOutputRef, ToolRegistry,
    build_default_tool_registry, LLMPlanner, PlanningWireMode)
from agent.orchestration.registry import DurableToolHooks
from agent.tools.data import scatac_fragments as public
from agent.tools.data._fragments_common import FragmentsError
from test_fragments_contracts import bound, fake_runtime


class FixedPlanner:
    def __init__(self, plan):self.value=plan;self.calls=0
    def plan(self, request, registry):self.calls+=1;return self.value


class Model:
    model_id='scripted-fragments'
    def __init__(self,payload):self.payload=payload
    def complete(self,*,prompt,response_schema):
        self.prompt=prompt;self.schema=response_schema
        return json.dumps(self.payload)


def public_args(inputs,output):
    return dict(intake_manifest_path=inputs.intake_path,intake_manifest_sha256=inputs.intake_sha256,
        library_context_path=inputs.context_path,library_context_sha256=inputs.context_sha256,
        reference_bundle_path=inputs.reference_path,reference_bundle_sha256=inputs.reference_sha256,
        output_dir=str(output))


@pytest.fixture
def configured(tiny,bound,fake_runtime,monkeypatch):
    inputs,group,white=bound;runtime,control=fake_runtime
    root=tiny['root']/'indexes';root.mkdir();Path(inputs.index_path).parent.rename(root/'candidate')
    monkeypatch.setenv('AGENT_CHROMAP_BIN',runtime.chromap)
    monkeypatch.setenv('AGENT_CHROMAP_INDEX_ROOT',str(root))
    return public_args(inputs,tiny['root']/'output'),control,group,white,root


def plan_for(args,*,downstream=False):
    steps=[PlanStep('fragments','prepare_scATAC_fragments',args)]
    if downstream:
        steps.append(PlanStep('after','inspect_raw_scATAC',{'raw_input_paths':'/must-not-execute',
            'output_dir':str(Path(args['output_dir'])/'downstream')},('fragments',)))
    return AgentPlan('fragments-plan','fragments-request','Prepare canonical fragments.',tuple(steps))


def registry_after():
    registry=build_default_tool_registry()
    after=Mock(side_effect=RuntimeError('sentinel downstream'))
    return ToolRegistry(tuple(replace(registry.get(n),function=after) if n=='inspect_raw_scATAC'
        else registry.get(n) for n in registry.names())),after


class ProcessExit(BaseException):pass


class StopStore:
    def __init__(self,store,*,before_success=False,cancel=None):self.store=store;self.before=before_success;self.cancel=cancel
    def __getattr__(self,name):return getattr(self.store,name)
    def update(self,state,*,expected_revision):
        completed=any(s.tool_name=='prepare_scATAC_fragments' and s.status is StepStatus.SUCCEEDED for s in state.steps)
        if completed and self.before:raise ProcessExit()
        saved=self.store.update(state,expected_revision=expected_revision)
        if completed:
            if self.cancel:self.cancel(saved.run_id)
            else:raise ProcessExit()
        return saved


def test_public_wrapper_and_verifier(configured):
    args,control,*_=configured
    result=public.prepare_scATAC_fragments(**args)
    build_default_tool_registry().validate_result('prepare_scATAC_fragments',result)
    public.verify_public_result(args,result)
    assert result['total_support']==300 and control['calls'].count('chromap')==1
    with pytest.raises(FragmentsError,match='FRAGMENTS_ARTIFACT_CONFLICT'):public.prepare_scATAC_fragments(**args)


def test_runtime_success_terminal_immutability(configured,tiny,monkeypatch):
    args,control,group,*_=configured;planner=FixedPlanner(plan_for(args));store=FileRunStore(tiny['root']/'store')
    runtime=AgentRuntime(planner=planner,run_store=store)
    result=runtime.run(AgentRequest('fragments-request','prepare',{}))
    assert result.status is RunStatus.SUCCEEDED,result.errors
    assert result.steps[0].verification.passed
    assert store.load(result.run_id).steps[0].result==result.steps[0].result
    Path(group.files[0][1]).write_bytes(b'terminal result remains immutable')
    monkeypatch.delenv('AGENT_CHROMAP_BIN');monkeypatch.delenv('AGENT_CHROMAP_INDEX_ROOT')
    assert runtime.resume(result.run_id)==result
    assert control['calls'].count('chromap')==planner.calls==1


@pytest.mark.parametrize('crash_window',[False,True])
@pytest.mark.parametrize('drift',[None,'bgzf','tabix','whitelist','reference','raw','receipt'])
def test_durable_resume_and_exact_publication_recovery(configured,tiny,monkeypatch,crash_window,drift):
    args,control,group,white,_=configured;store=FileRunStore(tiny['root']/'store')
    registry,after=registry_after();planner=FixedPlanner(plan_for(args,downstream=True))
    with pytest.raises(ProcessExit):
        AgentRuntime(registry=registry,planner=planner,run_store=StopStore(store,before_success=crash_window)).run(
            AgentRequest('fragments-request','prepare',{}))
    state=store.load('fragments-request:run')
    assert state.steps[0].status is (StepStatus.RUNNING if crash_window else StepStatus.SUCCEEDED)
    manifest_path=next(Path(args['output_dir']).glob('scatac-fragments-*/fragments/manifest.json'))
    value=json.loads(manifest_path.read_bytes())
    if drift:
        paths={'bgzf':manifest_path.parent/value['libraries'][0]['bgzf']['path'],
            'tabix':manifest_path.parent/value['libraries'][0]['tabix']['path'],
            'whitelist':Path(white.resource.path),'reference':tiny['bed'],
            'raw':Path(group.files[0][1]),'receipt':manifest_path.parent.parent/'receipt.json'}
        with paths[drift].open('ab') as out:out.write(b'changed')
    monkeypatch.delenv('AGENT_CHROMAP_BIN');monkeypatch.delenv('AGENT_CHROMAP_INDEX_ROOT')
    forbidden=Mock();forbidden.plan.side_effect=AssertionError('Planner called on resume')
    result=AgentRuntime(registry=registry,planner=forbidden,run_store=store).resume(state.run_id)
    assert control['calls'].count('chromap')==1
    assert forbidden.plan.call_count==0
    if drift:
        assert result.status is RunStatus.FAILED
        assert after.call_count==0
    else:
        assert result.steps[0].status is StepStatus.SUCCEEDED,result.errors
        assert after.call_count==1  # Pending sentinel only; no alignment rerun.
        if crash_window:assert any(e.details.get('decision')=='recover_verified_publication' for e in result.trace)


def test_unpublished_running_step_never_reruns(configured,tiny):
    args,control,*_=configured;store=FileRunStore(tiny['root']/'store');registry=build_default_tool_registry()
    spec=registry.get('prepare_scATAC_fragments')
    def crash(*args):raise ProcessExit()
    registry=ToolRegistry(tuple(replace(spec,durable_hooks=DurableToolHooks(crash,spec.durable_hooks.recover))
        if n==spec.name else registry.get(n) for n in registry.names()))
    with pytest.raises(ProcessExit):AgentRuntime(registry=registry,planner=FixedPlanner(plan_for(args)),run_store=store).run(
        AgentRequest('fragments-request','prepare',{}))
    result=AgentRuntime(registry=registry,run_store=store).resume('fragments-request:run')
    assert result.status is RunStatus.FAILED
    assert any(e.code=='FRAGMENTS_RECOVERY_UNAVAILABLE' for e in result.errors)
    assert control['calls']==[]


def test_recovery_checkpoint_failure_remains_resumable(configured, tiny):
    from agent.orchestration.run_store import RunStoreError
    args, control, *_ = configured
    store = FileRunStore(tiny['root'] / 'store')
    with pytest.raises(ProcessExit):
        AgentRuntime(planner=FixedPlanner(plan_for(args)),
            run_store=StopStore(store, before_success=True)).run(
                AgentRequest('fragments-request', 'prepare', {}))

    class FailingRecoveryStore:
        def __getattr__(self, name):
            return getattr(store, name)

        def update(self, state, *, expected_revision):
            raise RunStoreError('Injected recovery checkpoint failure.')

    run_id = 'fragments-request:run'
    with pytest.raises(RunStoreError):
        AgentRuntime(run_store=FailingRecoveryStore()).resume(run_id)
    assert store.load(run_id).steps[0].status is StepStatus.RUNNING
    result = AgentRuntime(run_store=store).resume(run_id)
    assert result.status is RunStatus.SUCCEEDED
    assert result.steps[0].verification.passed
    assert control['calls'].count('chromap') == 1


@pytest.mark.parametrize('code',['FRAGMENTS_FASTQ_MALFORMED','FRAGMENTS_WHITELIST_MISMATCH',
    'FRAGMENTS_SOURCE_CHANGED_DURING_EXECUTION','CHROMAP_UNAVAILABLE','CHROMAP_INDEX_INVALID',
    'FRAGMENTS_ARTIFACT_CONFLICT','FRAGMENTS_OUTPUT_MALFORMED'])
def test_no_same_step_retry(configured,code):
    args,control,*_=configured;registry=build_default_tool_registry();call=Mock(side_effect=FragmentsError(code))
    registry=ToolRegistry(tuple(replace(registry.get(n),function=call) if n=='prepare_scATAC_fragments'
        else registry.get(n) for n in registry.names()))
    result=AgentRuntime(registry=registry,planner=FixedPlanner(plan_for(args))).run(AgentRequest('fragments-request','prepare',{}))
    assert result.status is RunStatus.FAILED and call.call_count==1
    assert result.errors[0].code==code and not result.errors[0].recoverable
    assert '/PRIVATE' not in result.errors[0].message


def test_backend_failure_one_alignment_attempt(configured):
    args,control,*_=configured;control['fail']='chromap'
    result=AgentRuntime(planner=FixedPlanner(plan_for(args))).run(AgentRequest('fragments-request','prepare',{}))
    assert result.status is RunStatus.FAILED and control['calls'].count('chromap')==1


@pytest.mark.parametrize('code,category', [
    ('CHROMAP_SOURCE_CHANGED', 'VERIFICATION_ERROR'),
    ('CHROMAP_WHITELIST_REQUIRED', 'USER_INPUT_ERROR'),
    ('CHROMAP_OUTPUT_CONFLICT', 'RESOURCE_ERROR'),
    ('CHROMAP_INDEX_CONTRACT_INVALID', 'ENVIRONMENT_ERROR'),
])
def test_chromap_domain_classification(code, category):
    from agent.orchestration.fragments_registry import classify_fragments_exception
    from agent.schemas import ErrorCategory
    from agent.tools.data._chromap import ChromapError
    classified = classify_fragments_exception(ChromapError(code, '/PRIVATE raw stderr'))
    assert classified.code == code
    assert classified.category is ErrorCategory[category]


def test_index_resolution_zero_multiple_and_missing(configured,monkeypatch):
    args,_,_,_,root=configured
    import shutil
    shutil.copytree(root/'candidate',root/'duplicate')
    with pytest.raises(FragmentsError,match='CHROMAP_INDEX_AMBIGUOUS'):public.resolve_execution(args['reference_bundle_path'],args['reference_bundle_sha256'])
    monkeypatch.setenv('AGENT_CHROMAP_INDEX_ROOT',str(root/'missing'))
    with pytest.raises(FragmentsError,match='CHROMAP_INDEX_UNAVAILABLE'):public.resolve_execution(args['reference_bundle_path'],args['reference_bundle_sha256'])


@pytest.mark.parametrize('field,value',[('status','failed'),('manifest_sha256','bad'),('artifact_schema_version',True),
    ('assembly','mm10'),('n_libraries',0),('n_fragment_records',0),('total_support',2**128),('extra',[])])
def test_result_contract_closed(configured,field,value):
    args,*_=configured;result=public.prepare_scATAC_fragments(**args);result[field]=value
    from agent.orchestration.registry import ToolResultContractError
    with pytest.raises(ToolResultContractError):build_default_tool_registry().validate_result('prepare_scATAC_fragments',result)


def test_cancel_before_step(configured,tiny):
    args,control,*_=configured;store=FileRunStore(tiny['root']/'store')
    class CancellingPlanner(FixedPlanner):
        def plan(self,request,registry):
            AgentRuntime(run_store=store).cancel('fragments-request:run')
            return super().plan(request,registry)
    result=AgentRuntime(run_store=store,planner=CancellingPlanner(plan_for(args))).run(AgentRequest('fragments-request','prepare',{}))
    assert result.status is RunStatus.CANCELLED and not control['calls']


@pytest.mark.parametrize('crash_window',[False,True])
def test_cancel_after_verified_fragments_preserves_result(configured,tiny,crash_window):
    args,control,*_=configured;store=FileRunStore(tiny['root']/'store');registry,after=registry_after()
    planner=FixedPlanner(plan_for(args,downstream=True))
    if crash_window:
        with pytest.raises(ProcessExit):AgentRuntime(registry=registry,planner=planner,run_store=StopStore(store,before_success=True)).run(
            AgentRequest('fragments-request','prepare',{}))
        AgentRuntime(run_store=store).cancel('fragments-request:run')
        runtime=AgentRuntime(registry=registry,run_store=store)
        result=runtime.resume('fragments-request:run')
    else:
        runtime=AgentRuntime(registry=registry,planner=planner,run_store=StopStore(store,
            cancel=lambda run_id:AgentRuntime(run_store=store).cancel(run_id)))
        result=runtime.run(AgentRequest('fragments-request','prepare',{}))
    assert result.status is RunStatus.CANCELLED,result.errors
    assert result.steps[0].status is StepStatus.SUCCEEDED
    assert result.steps[1].status is StepStatus.SKIPPED
    assert control['calls'].count('chromap')==1 and after.call_count==0
    assert runtime.resume(result.run_id)==result
    runtime.cancel(result.run_id);runtime.cancel(result.run_id)
    assert runtime.resume(result.run_id)==result


def test_inspection_to_fragments_runtime_dag(configured):
    args,control,group,*_=configured
    raw_args={'raw_input_paths':[p for _,p in group.files],'species':'human','raw_assay':'TENX_ATAC',
              'fastq_layout':group.layout.value,'output_dir':args['output_dir']}
    fragment_args=dict(args,intake_manifest_path=StepOutputRef('inspect','manifest_path'),
                        intake_manifest_sha256=StepOutputRef('inspect','manifest_sha256'))
    plan=AgentPlan('p','fragments-request','Prepare canonical fragments.',(
        PlanStep('fragments','prepare_scATAC_fragments',fragment_args,('inspect',)),
        PlanStep('inspect','inspect_raw_scATAC',raw_args)))
    result=AgentRuntime(planner=FixedPlanner(plan)).run(AgentRequest('fragments-request','prepare',{}))
    assert result.status is RunStatus.SUCCEEDED,result.errors
    step=next(s for s in result.steps if s.tool_name=='prepare_scATAC_fragments')
    assert step.verification.passed and control['calls'].count('chromap')==1


def test_cancel_requested_inside_tool_waits_for_safe_checkpoint(configured,tiny):
    args,control,*_=configured;store=FileRunStore(tiny['root']/'store');registry,after=registry_after()
    def request_cancel(kind):
        if kind=='chromap':AgentRuntime(run_store=store).cancel('fragments-request:run')
    control['after']=request_cancel
    result=AgentRuntime(registry=registry,planner=FixedPlanner(plan_for(args,downstream=True)),run_store=store).run(
        AgentRequest('fragments-request','prepare',{}))
    assert result.status is RunStatus.CANCELLED,result.errors
    assert result.steps[0].status is StepStatus.SUCCEEDED and result.steps[0].verification.passed
    assert after.call_count==0 and control['calls']==['chromap','sort','bgzip','tabix']


def test_recovery_rejects_unrelated_publication_and_other_policy(configured):
    import shutil
    args,control,*_=configured
    result=public.prepare_scATAC_fragments(**args)
    original=Path(result['manifest_path']).parent.parent
    destination,_,_=public._publication(args,'other-run-identity')
    shutil.copytree(original,destination)
    with pytest.raises(FragmentsError,match='FRAGMENTS_RECOVERY_MISMATCH'):
        public.recover_fragments(args,'other-run-identity')
    assert control['calls'].count('chromap')==1


def test_policy_drift_blocks_crash_recovery(configured,tiny):
    args,control,*_=configured;store=FileRunStore(tiny['root']/'store')
    with pytest.raises(ProcessExit):
        AgentRuntime(planner=FixedPlanner(plan_for(args)),run_store=StopStore(store,before_success=True)).run(
            AgentRequest('fragments-request','prepare',{}))
    registry=build_default_tool_registry()
    registry=ToolRegistry(tuple(replace(registry.get(n),recovery_policy_version='changed-policy')
        if n=='prepare_scATAC_fragments' else registry.get(n) for n in registry.names()))
    from agent.orchestration.run_store import RecoveryPolicyIncompatibleError
    with pytest.raises(RecoveryPolicyIncompatibleError):
        AgentRuntime(registry=registry,run_store=store).resume('fragments-request:run')
    assert control['calls'].count('chromap')==1


def test_disk_full_requires_external_correction():
    from agent.schemas import ErrorCategory,RecoveryDisposition
    error=build_default_tool_registry().classify_exception('prepare_scATAC_fragments',OSError(errno.ENOSPC,'private path'))
    assert error.code=='DISK_FULL' and error.category is ErrorCategory.RESOURCE_ERROR
    assert error.recovery_disposition is RecoveryDisposition.USER_ACTION_REQUIRED and not error.recoverable
    assert 'private path' not in error.message


def test_verifier_dependency_identity_disagreement(configured):
    from agent.orchestration.verifier import verify_step
    args,*_=configured;result=public.prepare_scATAC_fragments(**args)
    step=PlanStep('fragments','prepare_scATAC_fragments',dict(args,
        intake_manifest_path=StepOutputRef('inspect','manifest_path'),
        intake_manifest_sha256=StepOutputRef('inspect','manifest_sha256')),('inspect',))
    verified=verify_step(step,args,result,build_default_tool_registry(),dependency_results={
        'inspect':{'manifest_path':args['intake_manifest_path'],'manifest_sha256':'0'*64}})
    assert not verified.passed
