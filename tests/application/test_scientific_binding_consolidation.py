"""Value authority and local questions through the production Session owners."""
import json
import pytest
from agent.application.turn_decisions import parse_decision, IntentError
from agent.application.dialogue_execution import _bind_arguments
from agent.orchestration import build_default_tool_registry
from agent.schemas.orchestration import _serialize
from test_structured_interaction import setup, setting


def decision(text, arguments):
    raw = json.dumps(dict(turn_schema_version=1, decision=dict(kind='execute_plan',
        target=arguments[0]['tool'], arguments=arguments)))
    return parse_decision(raw, utterance=text)


@pytest.mark.parametrize('text,literal,evidence,want', [
    ('请使用 Leiden 聚类，分辨率设置为0.8。', '0.8', '分辨率设置为0.8', .8),
    ('Please set the clustering resolution to 0.8.', '1.2', 'resolution to 0.8', .8),
    ('请把聚类分辨率设置为0.8。', '1.2', '分辨率设置为0.8', .8),
    ('Use resolution 2.4.', '0.4', 'resolution 2.4', 2.4),
])
def test_original_numeric_value_owns_the_binding(text,literal,evidence,want):
    row = dict(tool='cluster_cells',argument='resolution',literal=literal,evidence=evidence)
    parsed = decision(text,[row])
    values = _bind_arguments(build_default_tool_registry(),text,{},parsed.arguments)
    assert list(values.values()) == [want]
    assert parsed.arguments[0].literal == text[parsed.arguments[0].start:parsed.arguments[0].end]


@pytest.mark.parametrize('text,evidence', [('resolution 0.8 or 1.2','resolution 0.8 or 1.2'),
    ('resolution 0.8; resolution 0.8','resolution 0.8')])
def test_recovery_does_not_guess_among_grounded_alternatives(text,evidence):
    with pytest.raises(IntentError, match='ambiguous_parameter'):
        decision(text,[dict(tool='cluster_cells',argument='resolution',literal='2.4',evidence=evidence)])


def test_same_named_parameters_in_different_functions_keep_distinct_values():
    text='Use five components for PCA and eight components for Spectral.'
    rows=[dict(tool=tool,argument='kwargs',native_function=fn,members=[dict(name='n_comps',kind='number',literal=n,evidence=e)])
          for tool,fn,n,e in [('run_scanpy','pp.pca','5','five components for PCA'),('run_snapatac2','tl.spectral','8','eight components for Spectral')]]
    # Number words require a local question, never a guessed numeric conversion.
    with pytest.raises(IntentError): decision(text,rows)
    text='PCA uses 5 components; Spectral uses 8 components.'
    for row,e in zip(rows,('PCA uses 5','Spectral uses 8')):row['members'][0]['evidence']=e
    values=_bind_arguments(build_default_tool_registry(),text,{},decision(text,rows).arguments)
    assert values['run_scanpy__pp_pca__kwargs']=={'n_comps':5}
    assert values['run_snapatac2__tl_spectral__kwargs']=={'n_comps':8}


def test_boolean_string_list_and_choice_keep_researcher_values():
    text='标准化时关闭中心化，将输出名设为“精确名字”；显式列表为[1, 2]。使用pp.scale。'
    rows=[dict(tool='run_scanpy',argument='kwargs',native_function='pp.scale',members=[
        dict(name='zero_center',kind='boolean',literal='关闭',evidence='关闭中心化'),
        dict(name='max_value',kind='json',literal='[3, 4]',evidence='显式列表为[1, 2]')]),
        dict(tool='run_scanpy',argument='function_name',native_function='pp.scale',literal='pp.scale',evidence='使用pp.scale',value='pp.log1p')]
    values=_bind_arguments(build_default_tool_registry(),text,{},decision(text,rows).arguments)
    assert values['run_scanpy__pp_scale__kwargs']=={'zero_center':False,'max_value':[1,2]}
    assert values['run_scanpy__pp_scale__function_name']=='pp.scale'
    row=dict(tool='run_scanpy',argument='kwargs',native_function='pp.scale',members=[dict(name='mask_obs',kind='string',literal='wrong',evidence='将输出名设为“精确名字”')])
    # Grounding preserves arbitrary strings; native member validity remains separate.
    assert decision(text,[row]).arguments[0].members[0].literal=='精确名字'


def test_bad_member_clarification_preserves_sibling_and_task_after_reload(tmp_path):
    text='请标准化并继续对数处理，关闭中心化，上限再小一点。'
    row=dict(tool='run_scanpy',argument='kwargs',native_function='pp.scale',members=[
        dict(name='zero_center',kind='boolean',literal='关闭',evidence='关闭中心化'),
        dict(name='max_value',kind='number',literal='3',evidence='上限再小一点')])
    app, executor, model, inputs=setup(tmp_path,declaration=row)
    first=app.submit_turn('session','task',text,expected_generation=0,execution_inputs=inputs)
    assert first.status=='clarification', first.to_dict()
    state=app._application.sessions.load('session')
    assert dict(state.interactions[0].admitted['inputs']['run_scanpy__pp_scale__kwargs'])=={'zero_center':False}
    option,=first.clarification_choices
    assert option['member']=='max_value' and option['native_function']=='pp.scale'
    assert not executor.plans
    model.interpret=False
    second=app.submit_turn('session','answer','',expected_generation=0,
        parameter_overrides=[setting('kwargs',{'max_value':4})],
        clarification_choice=dict(question_turn_id='task',handle=option['handle']))
    assert second.error.code=='TEST_SCIENCE_WITHHELD',second.to_dict()
    assert _serialize(executor.plans[0].steps[0].arguments['kwargs'])=={'zero_center':False,'max_value':4}
    assert len(executor.plans[0].steps)==2
    assert len([c for c in model.calls if 'turn_schema_version' in c])==1
    app._application.sessions.load('session')


def test_qualitative_field_and_omitted_optional_have_different_behavior(tmp_path):
    request=dict(tool='run_scanpy',argument='kwargs',native_function='pp.scale',member='max_value',evidence='上限小一些')
    app,executor,model,inputs=setup(tmp_path,missing=(request,))
    first=app.submit_turn('session','task','请标准化，上限小一些。',expected_generation=0,execution_inputs=inputs)
    assert first.status=='clarification',first.to_dict()
    assert not executor.plans
    assert first.clarification_choices[0]['member']=='max_value'
    model.interpret=False
    bad=app.submit_turn('session','bad','',expected_generation=0,parameter_overrides=[setting('kwargs',{'base':2},'pp.log1p')])
    assert bad.status=='failed' and not executor.plans


def test_structured_authority_removes_bad_model_duplicate_before_grounding(tmp_path):
    row=dict(tool='run_scanpy',argument='kwargs',native_function='pp.scale',members=[
        dict(name='max_value',kind='number',literal='999',evidence='not in utterance')])
    app,ex,model,inputs=setup(tmp_path,declaration=row)
    result=app.submit_turn('session','task','标准化输入。',expected_generation=0,execution_inputs=inputs,
        parameter_overrides=[setting('kwargs',{'max_value':4})])
    assert result.error.code=='TEST_SCIENCE_WITHHELD',result.to_dict()
    assert dict(ex.plans[0].steps[0].arguments['kwargs'])=={'max_value':4}


def test_two_values_are_unresolved_even_if_candidate_chooses_one():
    text='Set resolution to 0.8 or 1.2.'
    with pytest.raises(IntentError, match='ambiguous_parameter'):
        decision(text,[dict(tool='cluster_cells',argument='resolution',literal='0.8',evidence='resolution to 0.8 or 1.2')])


def test_rejected_members_of_distinct_functions_have_independent_questions(tmp_path):
    requests=[dict(tool='run_scanpy',argument='kwargs',native_function=fn,member=member)
              for fn,member in [('pp.scale','max_value'),('pp.log1p','base')]]
    app,executor,model,inputs=setup(tmp_path,missing=requests)
    first=app.submit_turn('session','task','标准化并做对数处理。',expected_generation=0,execution_inputs=inputs)
    assert first.status=='clarification',first.to_dict()
    assert {row['native_function'] for row in first.clarification_choices}=={'pp.scale','pp.log1p'}
    model.interpret=False
    second=app.submit_turn('session','scale','',expected_generation=0,parameter_overrides=[setting('kwargs',{'max_value':4})])
    assert second.status=='clarification',second.to_dict()
    assert len(second.clarification_choices)==1 and second.clarification_choices[0]['member']=='base'
    final=app.submit_turn('session','log','',expected_generation=0,parameter_overrides=[setting('kwargs',{'base':2},'pp.log1p')])
    assert final.error.code=='TEST_SCIENCE_WITHHELD',final.to_dict()
    assert dict(executor.plans[0].steps[0].arguments['kwargs'])=={'max_value':4}
    assert dict(executor.plans[0].steps[1].arguments['kwargs'])=={'base':2}
    app._application.sessions.load('session')


def test_natural_value_for_repeated_operation_uses_existing_plan_selection(tmp_path):
    text='把上限设置为5，并标准化两种表示。'
    row=dict(tool='run_scanpy',argument='kwargs',native_function='pp.scale',members=[
        dict(name='max_value',kind='number',literal='5',evidence='上限设置为5')])
    app,executor,model,inputs=setup(tmp_path,declaration=row,repeated=True)
    first=app.submit_turn('session','task',text,expected_generation=0,execution_inputs=inputs)
    assert first.status=='clarification',first.to_dict()
    assert [c['handle'] for c in first.clarification_choices]==['step:scale','step:again']
    model.interpret=False
    count=len(model.calls)
    final=app.submit_turn('session','chosen','',expected_generation=0,
        clarification_choice=dict(question_turn_id='task',handle='step:again'))
    assert final.error.code=='TEST_SCIENCE_WITHHELD',final.to_dict()
    assert 'kwargs' not in executor.plans[0].steps[0].arguments
    assert dict(executor.plans[0].steps[1].arguments['kwargs'])=={'max_value':5}
    assert len(model.calls)==count+1  # Output selection, never a second plan/interpretation.
    app._application.sessions.load('session')


def test_same_parameter_on_two_sources_keeps_exact_scope_and_values():
    from dataclasses import asdict
    from agent.application.structured_parameters import bound_records, bind_plan, check_plan
    from agent.orchestration import AgentPlan, PlanStep
    text='第一个矩阵上限为5，第二个矩阵上限为9。'
    rows=[dict(tool='run_scanpy',argument='kwargs',native_function='pp.scale',port='dataset',source='source.'+str(i),
               members=[dict(name='max_value',kind='number',literal=n,evidence=e)])
          for i,n,e in [(0,'5','第一个矩阵上限为5'),(1,'9','第二个矩阵上限为9')]]
    registry=build_default_tool_registry();decl=decision(text,rows).arguments
    values=_bind_arguments(registry,text,{},decl)
    admitted=dict(inputs=values,arguments=[asdict(d) for d in decl],
        roles=[dict(tool='run_scanpy',port='dataset',source='source.'+str(i)) for i in range(2)],
        registered_input=dict(members=[dict(source_path='/captured/a.h5ad'),dict(source_path='/captured/b.h5ad')]))
    bindings=bound_records(admitted,(),registry,current_utterance=text)
    plan=AgentPlan('request','plan','Two selected matrices',tuple(PlanStep('step'+str(i),'run_scanpy',
        dict(input_path='/captured/'+name+'.h5ad',output_dir='/captured/out'+str(i),function_name='pp.scale')) for i,name in enumerate(('a','b'))))
    bound=bind_plan(plan,bindings,values,registry,admitted)
    assert [dict(s.arguments['kwargs']) for s in bound.steps]==[{'max_value':5},{'max_value':9}]
    check_plan(bound,bindings,values,registry,admitted)


def test_retained_native_settings_do_not_become_package_wide_defaults():
    from types import SimpleNamespace
    from agent.application.dialogue_execution import _research_intent, _with_retained
    from agent.application.turn_decisions import Execute, ExplicitArgument, RetainedParameter
    group=dict(handle='g0',tool='run_snapatac2',parameters=[dict(argument='kwargs',value={'n_comps':5},
        native_function='tl.spectral',origin='accepted_argument')])
    capture=SimpleNamespace(utterance='Reuse the selected Spectral settings and use 8 neighbors.',
        snapshot={'dialogue':{'results':[group]}})
    request=Execute('run_snapatac2',arguments=(ExplicitArgument('run_snapatac2','kwargs',{'n_neighbors':8},
        native_function='pp.knn'),),input_results=('g0',),retain_parameters=(RetainedParameter('g0','kwargs'),))
    retained=_research_intent(capture,request)['retained_parameters']
    assert _with_retained(build_default_tool_registry(),{},retained)=={'run_snapatac2__tl_spectral__kwargs':{'n_comps':5}}
    from dataclasses import replace
    with pytest.raises(IntentError,match='ambiguous_parameter'):
        _research_intent(capture,replace(request,arguments=(ExplicitArgument('run_snapatac2','kwargs',{'n_comps':8},native_function='tl.spectral'),)))


def test_complete_structured_value_is_recovered_from_the_unique_original_object():
    text='Scale using {"zero_center": false, "max_value": 4}.'
    parsed=decision(text,[dict(tool='run_scanpy',argument='kwargs',native_function='pp.scale',
        literal='{"zero_center": true, "max_value": 9}',evidence='using {"zero_center": false, "max_value": 4}')])
    assert _bind_arguments(build_default_tool_registry(),text,{},parsed.arguments)=={
        'run_scanpy__pp_scale__kwargs':{'zero_center':False,'max_value':4}}
