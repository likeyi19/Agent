"""Guarded real Chromap execution through registry, durable runtime and verifier."""
from pathlib import Path
from unittest.mock import Mock
import pytest

from agent.orchestration import (AgentPlan,AgentRequest,AgentRuntime,FileRunStore,PlanStep,
    RunStatus,StepStatus,StepOutputRef)
from agent.tools.data import fastq_fragments as private, _chromap as c
from fragments_helpers import BC,inputs_for
from test_fragments_orchestration import public_args,plan_for,FixedPlanner,StopStore,ProcessExit,registry_after


@pytest.mark.parametrize('route',['direct','inspect_dag','completed_resume','publication_recovery'])
def test_real_runtime_fragments(tiny,reads,executables,monkeypatch,route):
    group,white=reads([(100,200,BC,300),(1100,1200,BC,1)])
    exe=executables['candidate']['path'];inputs=inputs_for(tiny,[group],white,executable=exe)
    root=tiny['root']/'indexes';root.mkdir();Path(inputs.index_path).parent.rename(root/'accepted')
    monkeypatch.setenv('AGENT_CHROMAP_BIN',exe);monkeypatch.setenv('AGENT_CHROMAP_INDEX_ROOT',str(root))
    args=public_args(inputs,tiny['root']/'output')
    original=private.run_stage;calls=[]
    def counting(argv,**kwargs):
        if '--preset' in argv:calls.append(tuple(argv))
        return original(argv,**kwargs)
    monkeypatch.setattr(private,'run_stage',counting)
    store=FileRunStore(tiny['root']/'store')
    if route in ('direct','inspect_dag'):
        plan=plan_for(args)
        if route=='inspect_dag':
            plan=AgentPlan('p','fragments-request','Prepare canonical fragments.',(
                PlanStep('inspect','inspect_raw_scATAC',{'raw_input_paths':[p for _,p in group.files],
                    'species':'human','raw_assay':'TENX_ATAC','fastq_layout':group.layout.value,'output_dir':args['output_dir']}),
                PlanStep('fragments','prepare_scATAC_fragments',dict(args,
                    intake_manifest_path=StepOutputRef('inspect','manifest_path'),
                    intake_manifest_sha256=StepOutputRef('inspect','manifest_sha256')),('inspect',))))
        result=AgentRuntime(planner=FixedPlanner(plan),run_store=store).run(AgentRequest('fragments-request','prepare',{}))
        assert result.status is RunStatus.SUCCEEDED,result.errors
    else:
        registry,after=registry_after()
        with pytest.raises(ProcessExit):
            AgentRuntime(registry=registry,planner=FixedPlanner(plan_for(args,downstream=True)),
                run_store=StopStore(store,before_success=route=='publication_recovery')).run(AgentRequest('fragments-request','prepare',{}))
        monkeypatch.delenv('AGENT_CHROMAP_BIN');monkeypatch.delenv('AGENT_CHROMAP_INDEX_ROOT')
        monkeypatch.setattr(c,'identify_backend',lambda *a,**k:pytest.fail('Chromap required on resume'))
        planner=Mock();planner.plan.side_effect=AssertionError('Planner called on resume')
        result=AgentRuntime(registry=registry,planner=planner,run_store=store).resume('fragments-request:run')
        assert after.call_count==1 and planner.plan.call_count==0,result.errors
    step=next(s for s in result.steps if s.tool_name=='prepare_scATAC_fragments')
    assert step.status is StepStatus.SUCCEEDED and step.verification.passed,result.errors
    assert step.result['total_support']==301 and len(calls)==1
    assert calls[0][calls[0].index('--num-threads')+1]=='1'
