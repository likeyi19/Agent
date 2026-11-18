"""Ordinary source Requirements follow Candidate acquisition; declaration intake stays separate."""
import json
from dataclasses import replace

import pytest

from agent.application import InteractiveAgentApplication, ResearchAgentApplication
from agent.application.session_state import AnalysisSession, SessionError
from agent.orchestration import PlanningModelProfile, build_default_tool_registry
from agent.providers import PlanningModelFactoryRegistry
from test_human_epizoo_mapping import WithholdScience


@pytest.mark.parametrize('empty_intake', [False, True])
@pytest.mark.parametrize('textual', [False, True])
def test_missing_source_exact_answer_reuses_candidate_under_original_task(tmp_path, empty_intake, textual):
    import anndata as ad
    import numpy as np
    calls = []
    class Model:
        model_id = 'canonical-intake'
        def complete(self, *, prompt, response_schema):
            value = json.loads(prompt); calls.append(value)
            if 'turn_schema_version' in value:
                if sum('turn_schema_version' in c for c in calls) > 1:
                    choices=value['dialogue']['pending_prerequisite']['resource_answer_choices']
                    return json.dumps(dict(turn_schema_version=1, decision=dict(kind='answer_resources',
                        pending='@resources',choice=choices[0]['handle'],arguments=[])))
                return json.dumps(dict(turn_schema_version=1, decision=dict(kind='execute_plan',
                    target='inspect_scATAC',roles=[])))
            if 'semantic_prompt_version' in value:
                return json.dumps(dict(schema_version=4, decision=dict(kind='plan', steps=[dict(
                    step_id='inspect', tool='inspect_scATAC', control_dependencies=[],
                    sources=[])])))
            if 'output_selection_schema_version' in value:
                return json.dumps(dict(outputs=[dict(name='inspection',step_id='inspect',output_key='input_path')]))
            if 'dialogue_schema_version' in value:
                return json.dumps(dict(support='supported', paragraphs=[dict(
                    text='科学执行尚未开始。请指定所需输入。',kind='limitation',citations=[],comparisons=[])]))
            if 'grounding_review_version' in value:
                return json.dumps(dict(verdicts={row['id']:dict(verdict='entailed',independent=True)
                    for row in value['paragraphs']}))
            raise AssertionError(list(value))
    registry=build_default_tool_registry(); executor=WithholdScience(registry)
    app=InteractiveAgentApplication(tmp_path/'workspace', registry=registry, executor=executor,
        model_profiles=(PlanningModelProfile('offline','scripted',Model.model_id),),default_profile_id='offline',
        approved_source_roots=(tmp_path,),
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':lambda _:Model()}))
    app.create_session('session')
    path=tmp_path/'matrix.h5ad';ad.AnnData(np.ones((3,4))).write_h5ad(path)
    source=None if empty_intake else app.resources.register('matrix',path,input_type='h5ad',session_id='session',label='Counts',attribution='Fixture')
    group=None if source is None else app.resources.compose_group([source.resource_id],attachment_candidates=True)
    first=app.submit_turn('session','task','请检查这份数据',expected_generation=0,registered_input=None if empty_intake else group)
    assert first.status=='clarification',first.to_dict()
    before=app._application.sessions.load('session');origin=before.interactions[-1]
    assert origin.preparation['candidate_id'] is not None and 'candidate' in origin.preparation
    assert origin.prerequisite is None and not before.turns
    forged=before.to_dict()
    forged['interactions'][0]['admitted']['inputs']['input_path']='/unselected/source.h5ad'
    with pytest.raises((SessionError,ValueError)):AnalysisSession.from_dict(forged)
    if not empty_intake:
        unoffered=before.to_dict()
        unoffered['interactions'][0]['submission']['registered_input']['attachment_candidates']=False
        with pytest.raises((SessionError,ValueError)):AnalysisSession.from_dict(unoffered)
    assert '科学执行尚未开始' in first.response.text
    assert sum('semantic_prompt_version' in c for c in calls)==1
    question=first
    if empty_intake:
        assert first.clarification_choices[0]['action']=='upload_source'
        source=app.resources.register('matrix',path,input_type='h5ad',session_id='session',label='Counts',attribution='Fixture')
        question=app.submit_turn('session','recapture','',expected_generation=0,
            resume_task_id=origin.preparation['task_id'],attachment_ids=[source.resource_id])
        assert question.status=='clarification',question.to_dict()
        assert not executor.plans and sum('semantic_prompt_version' in c for c in calls)==1
    choice=next(c for c in question.clarification_choices if c.get('source_input'))
    submission = (dict() if textual else dict(
        clarification_choice=dict(question_turn_id=question.turn_id,handle=choice['handle'])))
    utterance='Use the uploaded file as the dataset.' if textual else ''
    result=app.submit_turn('session','answer',utterance,expected_generation=0,**submission)
    assert result.status=='failed' and result.error.code=='TEST_SCIENCE_WITHHELD',result.to_dict()
    after=app._application.sessions.load('session')
    assert all(i.prerequisite is None for i in after.interactions)
    assert len({i.preparation['task_id'] for i in after.interactions})==1
    assert after.interactions[0]==origin
    assert after.interactions[-1].preparation['candidate_origin']=='task'
    assert sum('candidate' in i.preparation for i in after.interactions)==1
    assert sum('semantic_prompt_version' in c for c in calls)==1
    assert sum('turn_schema_version' in c for c in calls)==(2 if textual else 1)
    assert len(executor.plans)==1 and executor.plans[0].steps[0].arguments['path']==str(path)
    assert ResearchAgentApplication(app._application.workspace_root).sessions.load('session')==after
    replay=app.submit_turn('session','answer',utterance,expected_generation=0,**submission)
    assert replay==result and len(executor.plans)==1
    changed=after.to_dict();changed['interactions'][-1]['admitted']['inputs']['input_path']='/another/source.h5ad'
    with pytest.raises((SessionError,ValueError)):
        AnalysisSession.from_dict(changed)


@pytest.mark.parametrize('field', ['bam', 'fragments', 'fastq', 'epizoo', 'annotation'])
def test_scientific_declaration_intake_without_files_has_public_task_and_no_candidate(tmp_path, field):
    from agent.application.raw_handoff import empty, declaration_dict
    calls = []
    class Model:
        model_id = 'empty-scientific-intake'
        def complete(self, *, prompt, response_schema):
            value = json.loads(prompt); calls.append(value)
            if value.get('focus')=='outcome' or 'grounding_review_version' in value:
                from test_response_contracts import OutcomeModel
                return OutcomeModel('Please supply the scientific source.').complete(prompt=prompt,response_schema=response_schema)
            assert 'turn_schema_version' in value, list(value)
            return json.dumps(dict(turn_schema_version=1, decision=dict(kind='request_roles',
                target={'epizoo':'epizoo_embed_cells', 'fastq':'run_chromap', 'annotation':'annotate_scATAC_cell_types'}.get(field, 'run_snapatac2'),
                required_roles=[], roles=[], **{field:declaration_dict(replace(empty(field), operation='inference') if field == 'epizoo' else empty(field))})))
    registry = build_default_tool_registry(); executor = WithholdScience(registry)
    app = InteractiveAgentApplication(tmp_path/'workspace', registry=registry, executor=executor,
        model_profiles=(PlanningModelProfile('offline','scripted',Model.model_id),), default_profile_id='offline',
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':lambda _:Model()}))
    app.create_session('session')
    first = app.submit_turn('session','task','Analyze my scientific source.', expected_generation=0)
    assert first.status == 'clarification', first.to_dict()
    state = app._application.sessions.load('session'); origin = state.interactions[-1]
    assert first.preparation_task_id == origin.preparation['task_id']
    assert origin.preparation['candidate_origin'] is None
    assert origin.preparation['task']['context']['intake_owner'] == field
    assert origin.prerequisite is None and 'choices' not in origin.admitted and 'resource_options' not in origin.admitted
    requirement, = origin.admitted['preparation_result']['requirements']
    assert requirement['kind'] == 'declaration' and requirement['step_id'] == '@task'
    assert not state.turns and not state.revisions and not executor.plans
    assert sum('turn_schema_version' in c for c in calls) == 1
    assert sum('semantic_prompt_version' in c for c in calls) == 0
    assert len(calls) == 3
    assert ResearchAgentApplication(app._application.workspace_root).sessions.load('session') == state
    resumed = app.submit_turn('session','recapture','', expected_generation=0,
        resume_task_id=first.preparation_task_id)
    assert resumed.status == 'clarification', resumed.to_dict()
    latest = app._application.sessions.load('session')
    assert latest.interactions[-1].preparation['task_id'] == first.preparation_task_id
    assert latest.interactions[-1].preparation['candidate_origin'] is None
    assert latest.interactions[:len(state.interactions)] == state.interactions
    assert sum('turn_schema_version' in c for c in calls) == 1
    assert sum('semantic_prompt_version' in c for c in calls) == 0
    assert len(calls) == 3 and not executor.plans
    assert ResearchAgentApplication(app._application.workspace_root).sessions.load('session') == latest
