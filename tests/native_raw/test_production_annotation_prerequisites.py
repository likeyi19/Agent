"""Small genuine same-plan source handoff and truthful failed-child qualification."""
import json
import os
import shutil
from pathlib import Path
import pytest
from agent.orchestration import AgentRuntime, FileRunStore
from agent.orchestration.registry import build_default_tool_registry
from agent.schemas import AgentRequest, AgentPlan, PlanStep, StepOutputRef, PriorOutputRef
from agent.schemas.prior_output import PriorOutputBinding
from agent.orchestration.prior_outputs import bind_output
from agent.tools.analysis import hierarchical_annotation as owner


def test_same_plan_initial_grouping_and_failed_native_child(tmp_path):
    if os.environ.get('RUN_RAW_NATIVE_QUALIFICATION')!='1' or not os.environ.get('ICTA4B_ACCEPTED_FIXTURE'):
        pytest.skip('Explicit native qualification and accepted small fixture required')
    source=Path(os.environ['ICTA4B_ACCEPTED_FIXTURE']);store=FileRunStore(tmp_path/'run_state')
    for path in (source/'run_state').glob('*.json'):shutil.copyfile(path,store.root/path.name)
    registry=build_default_tool_registry();runtime=AgentRuntime(registry=registry,run_store=store)
    old=store.load('continued:run');result=old.steps[0].result
    h=json.loads(Path(result['hierarchy_path']).read_text())
    args=dict(old.plan.steps[0].arguments)
    for key in ('activity_method','activity_kwargs','gene_anno_path','gene_anno_sha256'):
        args.pop(key,None)
    args.update(activity_path=PriorOutputRef(PriorOutputBinding(**h['activity']),'matrix_path'),
        grouping_path=StepOutputRef('grouping','matrix_path'),maximum_levels=1,output_dir=str(tmp_path/'annotation'))
    request=AgentRequest('same-plan',None,{})
    plan=AgentPlan('same-plan:plan',request.request_id,'Actual native prerequisite in the same production plan',(
        PlanStep('grouping','run_snapatac2',dict(input_path=args['input_path'],function_name='tl.leiden',
            kwargs={'resolution':.05},output_dir=str(tmp_path/'grouping'))),
        PlanStep('annotation',owner.TOOL,args,depends_on=('grouping',))))
    complete=runtime.run_prepared(request,plan)
    assert complete.status.value=='SUCCEEDED',complete.to_dict()
    assert complete.steps[-1].verification.artifact_authority['schema_version']==2
    binding=bind_output(store,registry,complete.run_id,'grouping','dataset')
    args.update(grouping_path=PriorOutputRef(binding,'matrix_path'),maximum_levels=2,
        local_kwargs={'pp.select_features':{'n_features':1.5}},output_dir=str(tmp_path/'failed'))
    args['previous_annotation_path']=PriorOutputRef(bind_output(store,registry,complete.run_id,'annotation','annotation'),'manifest_path')
    request=AgentRequest('native-failure',None,{})
    failed=runtime.run_prepared(request,AgentPlan('native-failure:plan',request.request_id,'Real native rejection',(
        PlanStep('annotation',owner.TOOL,args),)))
    assert failed.status.value=='FAILED'
    assert not (tmp_path/'failed'/'manifest.json').exists()
    children=[store.load(e.details['request']['request_id']+':run') for e in failed.trace
        if e.details.get('event')=='reserved']
    assert any(c.lifecycle_status.value=='SUCCEEDED' for c in children)
    assert children[-1].lifecycle_status.value=='FAILED'
    assert any(e.code=='NATIVE_FUNCTION_REJECTED' for e in children[-1].errors)
