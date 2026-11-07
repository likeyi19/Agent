"""Native raw source authority, independent of retired bundle/profile contracts."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
from agent.application.local_resources import LocalResourceAdmission, ResourceAdmissionError
from agent.schemas.orchestration import AgentPlan, PlanStep, _serialize


@pytest.fixture(params=['bam','external_fragments'])
def registered(request,tmp_path):
    path=tmp_path/'source';path.write_bytes(b'Byte registration is not scientific qualification')
    owner=LocalResourceAdmission(tmp_path/'workspace',approved_source_roots=(tmp_path,))
    return owner,owner.register('source',path,input_type=request.param,label='Source',attribution='Technical fixture')


def submission(binding):
    return dict(registered_input=binding.attribution(),execution_inputs=_serialize(binding.execution_inputs))


def test_source_capture_needs_no_unrelated_scientific_context(registered,tmp_path):
    owner,record=registered
    bound=owner.compose_native(record.resource_id,{})
    assert dict(bound.execution_inputs)==dict(input_path=record.source_path,input_sha256=record.source_sha256)
    step=PlanStep('native','run_snapatac2',dict(bound.execution_inputs,function_name='pp.make_fragment_file',output_dir=str(tmp_path/'out')))
    plan=AgentPlan('plan','request','scripted',(step,))
    owner.validate_plan(submission(bound),plan)
    owner.validate_result(submission(bound),[SimpleNamespace(tool_name=step.tool_name,resolved_arguments=step.arguments)])
    for key,value in [('input_path','/foreign.bam'),('input_sha256','f'*64)]:
        changed=replace(step,arguments=dict(step.arguments,**{key:value}))
        with pytest.raises(ResourceAdmissionError):owner.validate_plan(submission(bound),replace(plan,steps=(changed,)))
        with pytest.raises(ResourceAdmissionError):owner.validate_result(submission(bound),[SimpleNamespace(tool_name=step.tool_name,resolved_arguments=changed.arguments)])
    with pytest.raises(ResourceAdmissionError):
        owner.validate_plan(submission(bound),replace(plan,steps=(replace(step,tool_name='inspect_scATAC'),)))


@pytest.mark.parametrize('values',[dict(input_path='/foreign.bam'),dict(input_sha256='f'*64),dict(output_dir='/foreign'),dict(unknown='x'),dict(source_profile='agent-cb-paired-atac.v2')])
def test_source_and_destination_cannot_be_forged_by_companions(registered,values):
    owner,record=registered
    with pytest.raises(ResourceAdmissionError):owner.compose_native(record.resource_id,values)


def test_changed_source_blocks_new_consumption_but_not_exact_metadata_read(registered):
    owner,record=registered
    bound=owner.compose_native(record.resource_id,{})
    from pathlib import Path
    Path(record.source_path).write_bytes(b'changed')
    assert owner.validate_binding(bound,verify_source=False)==_serialize(bound.execution_inputs)
    with pytest.raises(ResourceAdmissionError):owner.validate_binding(bound)
