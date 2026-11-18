"""Task recheck is nonexecuting; only exact current READY authorizes admission."""
from dataclasses import replace
import pytest
import anndata as ad
import numpy as np
from agent.orchestration import build_default_tool_registry,ToolRegistry
from agent.orchestration.scientific_inputs import ScientificInputRequired,ScientificInputRequirement
from agent.application import ResearchAgentApplication
from test_preparation_owner_contracts import Model,setup


def test_ready_recheck_requires_a_separate_exact_authorization(tmp_path):
    model=Model('run_scanpy','pp.log1p',roles=False)
    path=tmp_path/'source.h5ad';ad.AnnData(np.eye(3)).write_h5ad(path)
    ready=False
    def owner(args,context):
        if not ready:raise ScientificInputRequired(ScientificInputRequirement('count_slot','count_slot','scalar'))
        return args
    original=build_default_tool_registry()
    registry=ToolRegistry(tuple(replace(original.get(n),preparation_owner=owner) if n=='run_scanpy' else original.get(n) for n in original.names()))
    app,executor=setup(tmp_path,model,registry)
    first=app.submit_turn('session','task','Transform this dataset.',expected_generation=0,execution_inputs={'input_path':str(path)})
    assert first.status=='clarification',first.to_dict()
    ready=True;calls=len(model.calls)
    checked=app.submit_turn('session','recheck','',expected_generation=0,resume_task_id=first.preparation_task_id)
    assert checked.status=='answered',checked.to_dict()
    assert checked.preparation_recheck and checked.clarification_choices[0]['action']=='execute_preparation'
    assert not executor.plans and not app._application.sessions.load('session').turns
    assert len(model.calls)==calls
    done=app.submit_turn('session','authorize','',expected_generation=0,
        clarification_choice={'question_turn_id':'recheck','handle':checked.clarification_choices[0]['handle']})
    assert done.error.code=='TEST_SCIENCE_WITHHELD',done.to_dict()
    assert len(executor.plans)==1
    assert sum('turn_schema_version' in c for c in model.calls)==1
    assert sum('semantic_prompt_version' in c for c in model.calls)==1
    stale=app.submit_turn('session','stale','',expected_generation=0,
        clarification_choice={'question_turn_id':'recheck','handle':checked.clarification_choices[0]['handle']})
    assert stale.status=='failed' and len(executor.plans)==1
    assert ResearchAgentApplication(app._application.workspace_root,registry=registry).sessions.load('session')==app._application.sessions.load('session')


def test_uploaded_missing_file_is_offered_on_recheck_without_replanning(tmp_path):
    model=Model();app,executor=setup(tmp_path,model)
    treatment=tmp_path/'treatment.tsv';treatment.write_text('chr1\t0\t10\t1\n')
    rec=app.resources.register('treatment',treatment,input_type='attachment',label='treatment',session_id='session',attribution='fixture')
    first=app.submit_turn('session','task','Compare these tracks.',expected_generation=0,
        registered_input=app.resources.compose_group([rec.resource_id],attachment_candidates=True))
    assert any(c['action']=='upload_source' for c in first.clarification_choices if c.get('action'))
    control=tmp_path/'control.tsv';control.write_text('chr1\t0\t10\t2\n')
    added=app.resources.register('control',control,input_type='attachment',label='control',session_id='session',attribution='fixture')
    before=sum('semantic_prompt_version' in c for c in model.calls)
    checked=app.submit_turn('session','recheck','',expected_generation=0,resume_task_id=first.preparation_task_id,attachment_ids=[added.resource_id])
    assert checked.status=='clarification',checked.to_dict()
    choice=next(c for c in checked.clarification_choices if c.get('source_input') and c['label']=='control')
    assert not executor.plans and sum('semantic_prompt_version' in c for c in model.calls)==before
    done=app.submit_turn('session','answer','',expected_generation=0,clarification_choice=dict(question_turn_id='recheck',handle=choice['handle']))
    assert done.error.code=='TEST_SCIENCE_WITHHELD',done.to_dict()
    assert executor.plans[0].steps[0].arguments['input_path']==str(treatment)
    assert executor.plans[0].steps[0].arguments['control_path']==str(control)
