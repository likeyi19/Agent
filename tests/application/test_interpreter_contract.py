"""SPP-1I current-context schema and clarification continuity, no native science."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from agent.application.turn_decisions import decision_schema, interpret, _interpretation_public
from test_interpretation_protocol import context, execute, registry, accepts


def wire(value):
    return dict(turn_schema_version=1, decision=value)


def valid(schema, value):
    return accepts(schema, wire(value))


def strict_structure(schema):
    counts = dict(properties=0, enums=0, strings=0)
    def visit(node):
        if isinstance(node, dict):
            if '$ref' in node:
                assert node['$ref'].rsplit('/',1)[-1] in schema['$defs']
            if node.get('type') == 'object':
                assert node['additionalProperties'] is False
                assert set(node['required']) == set(node['properties'])
                counts['properties'] += len(node['properties'])
                counts['strings'] += sum(map(len,node['properties']))
            if 'enum' in node:
                assert node['enum'] and len(node['enum']) == len(set(node['enum']))
                counts['enums'] += len(node['enum'])
                counts['strings'] += sum(len(str(v)) for v in node['enum'])
            if 'anyOf' in node:
                assert node['anyOf']
            assert not {'allOf','not','if','then','else','dependentRequired','patternProperties'} & node.keys()
            for value in node.values(): visit(value)
        elif isinstance(node,list):
            for value in node: visit(value)
    visit(schema)
    def depth(node, level=0):
        if isinstance(node, dict):
            if '$ref' in node: return depth(schema['$defs'][node['$ref'].rsplit('/',1)[-1]],level)
            next_level = level + (node.get('type') in ('object','array'))
            return max([next_level]+[depth(v,next_level) for k,v in node.items() if k != '$defs'])
        if isinstance(node,list): return max([level]+[depth(v,level) for v in node])
        return level
    counts['depth'] = depth(schema)
    assert counts['properties'] <= 5000 and counts['enums'] <= 1000
    assert counts['strings'] + sum(map(len,schema['$defs'])) <= 120000
    assert counts['depth'] <= 10
    return counts


@pytest.mark.parametrize('sources', [0,1,2,16])
@pytest.mark.parametrize('shape', ['plain','parameters','resources','results'])
def test_current_schema_strict_validity_and_empty_offer_branches(registry,sources,shape):
    public=context(registry,sources)
    d=public['dialogue']
    if shape=='parameters':
        d['pending_prerequisite']=dict(handle='@parameters',field='parameters',tool='run_scanpy',
            missing=[dict(tool_name='run_scanpy',native_function='pp.scale',argument='kwargs',member='max_value',step_id='scale'),
                     dict(tool_name='run_scanpy',native_function='pp.scale',argument='count_slot',member=None,step_id='scale')])
    if shape=='resources':
        d['pending_prerequisite']=dict(handle='@resources',field='resources',tool='run_scanpy',
            choices=[dict(handle=f'spp.{i}',label=f'Choice {i}') for i in range(128)])
    if shape=='results':
        d['outputs']=[dict(handle='r0',subjects=dict(candidates=[dict(id='branch.1',references=['branch.1','cluster branch.1'])]))]
        d['result_groups']=[dict(handle='g0',tool='run_scanpy',outputs=['r0'],canonical_output='r0')]
    schema=decision_schema(public,utterance='Explore these data.')
    strict_structure(schema)
    assert valid(schema,dict(kind='clarify',reason='ambiguous_subject',choices=[]))
    assert not valid(schema,dict(kind='clarify',reason='ambiguous_parameter',choices=['plausible.parameter']))
    assert valid(schema,dict(kind='clarify',reason='ambiguous_parameter',choices=['run_scanpy.kwargs']))==(shape=='parameters')
    assert valid(schema,dict(kind='answer_resources',pending='@resources',choice='spp.127',arguments=[]))==(shape=='resources')
    assert not valid(schema,dict(kind='answer_resources',pending='@resources',choice='spp.stale',arguments=[]))
    if shape=='plain':
        kinds={v['properties']['kind']['enum'][0] for v in schema['properties']['decision']['anyOf']}
        assert not {'answer_parameters','answer_roles','answer_resources','execute_candidate','answer_scientific','navigate'} & kinds
    if shape=='results':
        target=schema['$defs']['target']
        assert accepts(target,dict(output='g0',subject='cluster branch.1'),schema)


def test_explicit_scale_function_member_schema_and_grounding(registry):
    text='Run run_scanpy pp.scale with max_value=4.'
    public=context(registry)
    value=execute([
        dict(tool='run_scanpy',argument='function_name',native_function='pp.scale',value='pp.scale',literal='pp.scale',evidence='pp.scale'),
        dict(tool='run_scanpy',argument='kwargs',native_function='pp.scale',members=[dict(name='max_value',kind='number',literal='4',evidence='max_value=4')])])
    schema=decision_schema(public,utterance=text)
    assert valid(schema,value)
    for field,bad in [('tool','fake'),('argument','made_up'),('native_function','pp.fabricated')]:
        changed=deepcopy(value);changed['arguments'][1][field]=bad
        assert not valid(schema,changed)
    changed=deepcopy(value);changed['arguments'][1]['members'][0]['name']='invented'
    assert not valid(schema,changed)
    changed=deepcopy(value);changed['roles'][0]['source']='source.9'
    assert not valid(schema,changed)
    class Model:
        def complete(self,**kwargs):
            assert kwargs['response_schema']==schema
            return json.dumps(wire(value))
    from agent.application.dialogue_execution import _bind_arguments
    decision=interpret(Model(),text,public)
    assert _bind_arguments(registry,text,{},decision.arguments)['run_scanpy__pp_scale__kwargs']=={'max_value':4}


def test_registry_method_names_expand_without_package_rank(registry):
    public=context(registry)
    d=_interpretation_public(public,'Use PCA with n_comps=10.')['dialogue']['capabilities']
    for tool, metadata in public['dialogue']['capabilities'].items():
        for function in metadata.get('functions',{}):
            if function.rsplit('.',1)[-1].lower()=='pca':
                assert function in d[tool]['function_parameters']
    assert set(d)==set(registry.names())
    for word in ('epizoo','archr','chromvar','cicero','macs3','monalisa'):
        assert any(word in tool.lower() for tool in d)
    assert public==context(registry)


@pytest.mark.parametrize('invalid_middle',[False,True])
@pytest.mark.parametrize('unrelated',[False,True])
@pytest.mark.parametrize('scoped_reply',[False,True])
def test_ordinary_clarification_upload_precise_command_and_replay(tmp_path,invalid_middle,unrelated,scoped_reply):
    import anndata as ad
    import numpy as np
    from agent.application import InteractiveAgentApplication, ResearchAgentApplication
    from agent.orchestration import PlanningModelProfile,build_default_tool_registry
    from agent.providers import PlanningModelFactoryRegistry
    from test_human_epizoo_mapping import WithholdScience
    from test_response_contracts import OutcomeModel
    calls=[]; schemas=[]
    outcome=OutcomeModel('请说明希望执行的具体操作。尚未开始分析。')
    class Model:
        model_id='offline-interpreter-contract'
        def complete(self,*,prompt,response_schema):
            data=json.loads(prompt);calls.append(data)
            if 'turn_schema_version' in data:
                schemas.append(response_schema)
                if data['utterance']=='开始一个无关的新任务：为另一个查询文件转移标签，我稍后上传参考和查询文件。':
                    value=execute([],target='transfer_cell_labels',roles=[])
                elif data['utterance'] in ('请检查上传的数据结构。','只做前一步，后面的暂缓。'):
                    value=execute([],target='inspect_scATAC',roles=[dict(tool='inspect_scATAC',port='dataset',source='source.0')])
                elif invalid_middle and 'H5AD' in data['utterance']:
                    value=dict(kind='clarify',reason='ambiguous_parameter',choices=['run_scanpy.kwargs'])
                else:
                    value=dict(kind='clarify',reason='ambiguous_subject',choices=[])
                return json.dumps(wire(value))
            if 'semantic_prompt_version' in data:
                if unrelated:
                    assert data['user_request']=='开始一个无关的新任务：为另一个查询文件转移标签，我稍后上传参考和查询文件。'
                    return json.dumps(dict(schema_version=4,decision=dict(kind='plan',steps=[dict(
                        step_id='transfer',tool='transfer_cell_labels',sources=[],control_dependencies=[])])))
                remembered=data['research_context']['task']['ordinary_clarification']
                assert remembered[0]['utterance']==original.utterance
                assert remembered[0]['question']==first.response.text
                assert 'ordinary_clarification' in ' '.join(data['instructions'])
                return json.dumps(dict(schema_version=4,decision=dict(kind='plan',steps=[dict(step_id='inspect',tool='inspect_scATAC',control_dependencies=[],sources=[dict(target='dataset',source=dict(kind='input',input='input_path'))])])))
            if 'output_selection_schema_version' in data:
                return json.dumps(dict(outputs=[dict(name='inspection',step_id='inspect',output_key='input_path')]))
            return outcome.complete(prompt=prompt,response_schema=response_schema)
    r=build_default_tool_registry();executor=WithholdScience(r);model=Model()
    app=InteractiveAgentApplication(tmp_path/'workspace',registry=r,executor=executor,
        model_profiles=(PlanningModelProfile('offline','scripted',model.model_id),),default_profile_id='offline',
        approved_source_roots=(tmp_path,),planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':lambda _:model}))
    app.create_session('session')
    objective=('先检查数据结构，然后做预处理；预处理方法我还没确定。' if scoped_reply else '我有一个单细胞数据，想先做一下简单的处理，后面还要继续分析。')
    first=app.submit_turn('session','first',objective,expected_generation=0)
    state=app._application.sessions.load('session');original=state.interactions[0]
    assert first.status=='clarification' and original.preparation is None and not state.turns
    path=tmp_path/'counts.h5ad';ad.AnnData(np.ones((3,4))).write_h5ad(path);raw=path.read_bytes()
    source=app.resources.register('counts',path,input_type='h5ad',session_id='session',label='Counts',attribution='Fixture')
    group=app.resources.compose_group([source.resource_id],attachment_candidates=True)
    second=app.submit_turn('session','second','我上传的是 H5AD，里面是一个小的单细胞矩阵。先帮我做预处理就行。',expected_generation=0,registered_input=group,attachment_ids=[source.resource_id])
    state=app._application.sessions.load('session');saved=state.interactions
    assert state.interactions[0]==original and not state.turns and not state.revisions
    assert all(i.preparation is None for i in state.interactions)
    prompts=[p for p in calls if 'turn_schema_version' in p]
    d=prompts[1]['dialogue']
    assert d['predecessor'] is None
    assert d['ordinary_clarification']==[dict(turn_id='first',utterance=original.utterance,question=first.response.text)]
    assert d['selected_sources'][0]['source']=='source.0'
    assert not valid(schemas[1],dict(kind='clarify',reason='ambiguous_parameter',choices=['run_scanpy.kwargs']))
    if invalid_middle:
        assert second.error.code=='INTERACTIVE_INTENT_BINDING_INVALID'
        assert second.response.text=='INTERACTIVE_INTENT_BINDING_INVALID'
        assert not app._application.sessions.load('session').turns
    else: assert second.status=='clarification'
    command=('开始一个无关的新任务：为另一个查询文件转移标签，我稍后上传参考和查询文件。' if unrelated else '只做前一步，后面的暂缓。' if scoped_reply else '请检查上传的数据结构。')
    last=app.submit_turn('session','precise',command,expected_generation=0)
    if unrelated:
        state=app._application.sessions.load('session')
        assert last.status=='clarification',last.to_dict()
        assert state.interactions[:2]==saved and not state.turns and not executor.plans
        current=state.interactions[-1]
        assert current.preparation['task']['objective']==command
        assert not current.admitted.get('roles', ()) and not current.admitted.get('arguments', ())
        assert 'input_path' not in current.admitted['inputs']
        assert current.preparation['candidate_id'] is not None
        assert {r['port'] for r in current.admitted['preparation_result']['requirements']} >= {'reference_dataset','query_dataset'}
        assert sum('semantic_prompt_version' in p for p in calls)==1
        return
    assert last.error.code=='TEST_SCIENCE_WITHHELD',last.to_dict()
    state=app._application.sessions.load('session')
    assert state.interactions[:2]==saved and path.read_bytes()==raw
    assert len(executor.plans)==1 and executor.plans[0].steps[0].arguments['path']==str(path)
    assert sum('semantic_prompt_version' in p for p in calls)==1
    assert sum(i.preparation is not None for i in state.interactions)==1
    task=state.interactions[-1].preparation['task']
    assert task['objective']==command
    assert task['context']['task']['ordinary_clarification'][0]['utterance']==original.utterance
    assert ResearchAgentApplication(app._application.workspace_root).sessions.load('session')==state
    count=len(calls)
    assert app.submit_turn('session','precise',command,expected_generation=0)==last
    assert len(calls)==count


def test_ordinary_context_does_not_cross_other_activity_or_supply_authority():
    from types import SimpleNamespace as N
    from agent.application.turn_context import ordinary_clarification
    def item(turn,previous,kind='clarify'):
        return N(turn_id=turn,utterance='prior objective',preparation=None,prerequisite=None,
            admitted=dict(kind=kind,reason='ambiguous_subject'),status='clarification' if kind=='clarify' else 'answered',
            presentation=dict(text='actual question'),snapshot=dict(dialogue=dict(context=dict(previous_turn_id=previous))))
    state=N(interactions=[item('old',None),item('new','old','answer')])
    captured=dict(dialogue=dict(context=dict(previous_turn_id='new')))
    assert ordinary_clarification(captured,state)==[]
    captured['dialogue']['context']['previous_turn_id']='old'
    result=ordinary_clarification(captured,state)
    assert set(result[0])=={'turn_id','utterance','question'}
    assert not {'source','roles','arguments','task','candidate','predecessor'} & result[0].keys()


def test_many_exact_subjects_keep_membership_without_enum_overflow(registry):
    public=context(registry)
    public['dialogue']['outputs']=[dict(handle=f'r{i}',subjects=dict(candidates=[
        dict(id=f'branch.{i}.{j}',references=[f'branch.{i}.{j}',f'cluster branch.{i}.{j}']) for j in range(32)])) for i in range(20)]
    public['dialogue']['result_groups']=[dict(handle=f'g{i}',tool='run_scanpy',outputs=[f'r{i}'],canonical_output=f'r{i}') for i in range(20)]
    schema=decision_schema(public)
    strict_structure(schema)
    target=schema['$defs']['target']
    assert accepts(target,dict(output='g19',subject='cluster branch.19.31'),schema)
    for bad in ('branch.20.0','cluster branch.19.31\n','prefix branch.19.31'):
        assert not accepts(target,dict(output='g19',subject=bad),schema)


def test_guidance_predecessor_without_scientific_target_is_valid(registry):
    public=context(registry)
    public['dialogue']['predecessor']=dict(target=None,previous=None,focus='question')
    public['dialogue']['guidance_predecessor']=[dict(handle='option 1',references=['option 1','candidate-id'])]
    public['dialogue']['execution_candidates']=[dict(candidate='candidate-id')]
    schema=decision_schema(public)
    strict_structure(schema)
    assert valid(schema,dict(kind='answer_guidance',targets=[],candidate='option 1'))
    assert valid(schema,dict(kind='execute_candidate',candidate='candidate-id',evidence='Run option 1.'))
    assert not valid(schema,dict(kind='execute_candidate',candidate='stale-id',evidence='Run option 1.'))


def test_scoped_resource_tokens_distinguish_same_offer_and_exclude_scalar_choices():
    from dataclasses import replace
    from types import SimpleNamespace as N
    from agent.application import preparation_state as records
    from agent.application.session_state import Interaction
    from agent.application.scientific_preparation import interpreted_resource
    from agent.application.turn_decisions import ResourceAnswer, IntentError
    from agent.orchestration.semantic_compiler import ScientificRequirement, SemanticPlanCandidate, SemanticPlanStep, candidate_identity
    from agent.orchestration import build_default_tool_registry
    candidate=SemanticPlanCandidate(tuple(SemanticPlanStep(s,'annotate_scATAC_cell_types') for s in ('left','right')))
    origin=Interaction('origin','Annotate both sources.',None,0,dict(relations={},bases={}))
    record=records.start('session',origin,origin.utterance,candidate=candidate,contract_identity='registry')
    reqs=[]
    for step in ('left','right'):
        req=ScientificRequirement(candidate_identity(candidate),step,'annotate_scATAC_cell_types',None,'gene_anno','gene_anno_path','resource').to_dict()
        reqs.append(dict(req,offers=[dict(handle='resource.shared',label='Same installed reference')]))
    scalar=ScientificRequirement(candidate_identity(candidate),'left','annotate_scATAC_cell_types',None,'strict','strict','scalar',choices=('yes','no')).to_dict()
    reqs.append(scalar)
    origin=replace(origin,status='clarification',preparation=record,admitted=dict(tool='annotate_scATAC_cell_types',kind='execute',operation='plan',inputs={},preparation_result=records.question(record,origin,reqs)))
    ref=records.capture_intake((origin,),0,None,None)
    answer=Interaction('answer','Use the shared reference for the left source.',None,0,dict(relations={},bases={},preparation=ref))
    public_question=records.public_question(answer,(origin,))
    offered=records.choices(origin)
    resource_tokens={c['handle'] for c in offered if c['choice_identity']=='resource.shared'}
    scalar_tokens={c['handle'] for c in offered if c['choice_identity'] in ('yes','no')}
    assert len(resource_tokens)==2
    assert {c['handle'] for c in public_question['resource_answer_choices']}==resource_tokens
    public=context(build_default_tool_registry(),0);public['dialogue']['pending_prerequisite']=public_question
    public['dialogue']['pending_preparation']=dict(task_id=record['task_id'],choices=offered)
    schema=decision_schema(public,utterance=answer.utterance)
    def permitted(choice):return accepts(schema,dict(turn_schema_version=1,decision=dict(kind='answer_resources',pending='@resources',choice=choice,arguments=[])))
    assert all(permitted(c) for c in resource_tokens)
    assert all(not permitted(c) for c in scalar_tokens)
    assert not permitted('resource.shared') and not permitted('spp.stale')
    state=N(session_id='session',interactions=(origin,answer))
    sessions=N(_interaction_session_id='session',load=lambda _:state)
    before=repr(state.interactions)
    for token in resource_tokens:
        admitted,next_record=interpreted_resource(sessions,answer,ResourceAnswer('@resources',token))
        selected=next(c for c in offered if c['handle']==token)
        assert len(next_record['answers'])==1
        assert next_record['answers'][0]['requirement_id']==selected['requirement_id']
        assert next_record['task_id']==record['task_id'] and next_record['candidate_id']==record['candidate_id']
        assert admitted['inputs']=={} and repr(state.interactions)==before
    for bad in ['resource.shared','spp.stale',*scalar_tokens]:
        with pytest.raises(IntentError):interpreted_resource(sessions,answer,ResourceAnswer('@resources',bad))


def test_global_enum_budget_across_many_offered_sets(registry):
    public=context(registry,16)
    public['dialogue']['outputs']=[dict(handle=f'r{i}',subjects=dict(candidates=[])) for i in range(128)]
    public['dialogue']['result_groups']=[dict(handle=f'g{i}',tool='run_scanpy',outputs=[f'r{i}'],canonical_output=f'r{i}') for i in range(128)]
    schema=decision_schema(public,utterance='run_scanpy run_episcanpy run_snapatac2')
    strict_structure(schema)
    assert accepts(schema['$defs']['target'],dict(output='g127',subject=None),schema)
    assert not accepts(schema['$defs']['target'],dict(output='g128',subject=None),schema)


def test_unrelated_libraries_stay_out_of_intent_projection_and_schema(registry):
    public=context(registry)
    d=public['dialogue']
    d['sequence_models']=[dict(family='model-family',conditions=['condition-a','condition-b'])]
    d['epizoo_resources']=[dict(handle='resource.alpha',label='Alpha resource')]
    projected=_interpretation_public(public,'Explore the uploaded matrix.')['dialogue']
    assert projected['sequence_models']==[] and projected['epizoo_resources']==[]
    assert 'resource.alpha' not in json.dumps(decision_schema(public,utterance='Explore the uploaded matrix.'))
    named=_interpretation_public(public,'Use model-family for condition-a; use resource.alpha.')['dialogue']
    assert named['sequence_models']==d['sequence_models'] and named['epizoo_resources']==d['epizoo_resources']
    d['pending_prerequisite']=dict(handle='@roles',field='epizoo',tool='epizoo_embed_cells')
    assert _interpretation_public(public,'Use that reference.')['dialogue']['epizoo_resources']==d['epizoo_resources']


def test_empty_role_and_declaration_offers_disable_required_selections():
    public=dict(relations=[],dialogue=dict(outputs=[],tools=['inspect_scATAC'],capabilities={
        'inspect_scATAC':dict(description='Inspect a supplied matrix.')},scientific_parameters=[],input_roles=[],selected_sources=[]))
    schema=decision_schema(public)
    strict_structure(schema)
    for variant in schema['properties']['decision']['anyOf']:
        p=variant['properties']
        if p['kind']['enum']==['request_roles']:
            assert p['required_roles'].get('minItems',0)==0
        if 'arguments' in p:assert p['arguments']['maxItems']==0
        if 'roles' in p:assert p['roles']['maxItems']==0


def test_qualitative_active_result_expands_exact_recorded_function_only(registry):
    public=context(registry)
    public['dialogue']['outputs']=[dict(handle='r0'),dict(handle='r1')]
    public['dialogue']['result_groups']=[dict(handle='g0',tool='run_scanpy',is_active=True,outputs=['r0'],canonical_output='r0',
        parameters=[dict(argument='function_name',value='tl.leiden',native_function='tl.leiden')]),
        dict(handle='g1',tool='run_episcanpy',is_active=False,outputs=['r1'],canonical_output='r1',
        parameters=[dict(argument='function_name',value='tl.pca',native_function='tl.pca')])]
    text='Make this clustering finer.'
    projected=_interpretation_public(public,text)['dialogue']['capabilities']
    assert set(projected['run_scanpy']['function_parameters'])=={'tl.leiden'}
    assert 'function_parameters' not in projected['run_episcanpy']
    value=execute([],target='run_scanpy',roles=[],missing_parameters=[dict(tool='run_scanpy',argument='kwargs',
        native_function='tl.leiden',member='resolution',source=None,port=None,evidence=text)])
    assert valid(decision_schema(public,utterance=text),value)
    value['missing_parameters'][0]['native_function']='tl.pca'
    assert not valid(decision_schema(public,utterance=text),value)
