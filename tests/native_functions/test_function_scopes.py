"""Grounded native calls share ordinary interpretation, compilation and admission."""
from dataclasses import asdict, replace
import json
import pytest
from agent.application.turn_decisions import _ground_declaration, _scientific_arguments, IntentError
from agent.application.dialogue_execution import _bind_arguments, _check_parameter_plan
from agent.orchestration import build_default_tool_registry, AgentRequest, AgentPlan, PlanStep
from agent.orchestration.semantic_wire_v4 import parse_semantic_wire_v4
from agent.orchestration.semantic_compiler import compile_semantic_plan, build_semantic_compiler_contract


def declarations(raw_output=False):
    utterance = 'Run scanpy.pp.pca with n_comps=8, then scanpy.pp.neighbors with n_neighbors=12 and use_rep="X_pca".'
    raw=[]
    for fn, member, value in [('pp.pca','n_comps','8'),('pp.neighbors','n_neighbors','12')]:
        raw.append(dict(tool='run_scanpy',argument='function_name',native_function=fn,value=fn,literal=fn,evidence=fn))
        members=[dict(name=member,kind='number',literal=value,evidence=member+'='+value)]
        if fn=='pp.neighbors':members.append(dict(name='use_rep',kind='string',literal='X_pca',evidence='use_rep="X_pca"'))
        raw.append(dict(tool='run_scanpy',argument='kwargs',native_function=fn,members=members))
    return utterance, raw if raw_output else _scientific_arguments([_ground_declaration(x,utterance) for x in raw])


def test_grounded_functions_preserve_distinct_kwargs_and_reject_substitution():
    registry=build_default_tool_registry();utterance,decl=declarations()
    inputs=_bind_arguments(registry,utterance,{},decl)
    assert inputs['run_scanpy__pp_pca__kwargs']=={'n_comps':8}
    assert inputs['run_scanpy__pp_neighbors__kwargs']=={'n_neighbors':12,'use_rep':'X_pca'}
    plan=AgentPlan('test','test','native composition',(
        PlanStep('pca','run_scanpy',dict(function_name='pp.pca',kwargs={'n_comps':8})),
        PlanStep('graph','run_scanpy',dict(function_name='pp.neighbors',kwargs={'n_neighbors':12,'use_rep':'X_pca'}))))
    admitted=dict(inputs=inputs,arguments=[asdict(x) for x in decl])
    assert len(_check_parameter_plan(registry,plan,admitted,()))==4
    for change in ({'function_name':'tl.umap'}, {'kwargs':{'n_neighbors':20,'use_rep':'X_pca'}}):
        step=plan.steps[1]; bad=replace(plan,steps=(plan.steps[0],replace(step,arguments={**step.arguments,**change})))
        with pytest.raises(IntentError):_check_parameter_plan(registry,bad,admitted,())
    assert _scientific_arguments(json.loads(json.dumps(admitted['arguments'])))==decl


def test_scoped_native_sources_compile_with_exact_upstream_h5ad(tmp_path):
    registry=build_default_tool_registry();utterance,decl=declarations()
    inputs=_bind_arguments(registry,utterance,dict(input_path=str(tmp_path/'input.h5ad'),output_dir=str(tmp_path/'out')),decl)
    wire={'schema_version':4,'decision':{'kind':'plan','steps':[
        dict(step_id='pca',tool='run_scanpy',sources=[dict(target='function_name',kind='input',input='run_scanpy__pp_pca__function_name'),dict(target='kwargs',kind='input',input='run_scanpy__pp_pca__kwargs'),dict(target='dataset',kind='input',input='input_path')],control_dependencies=[]),
        dict(step_id='graph',tool='run_scanpy',sources=[dict(target='function_name',kind='input',input='run_scanpy__pp_neighbors__function_name'),dict(target='kwargs',kind='input',input='run_scanpy__pp_neighbors__kwargs'),dict(target='dataset',kind='step_port',step='pca',source_port='dataset')],control_dependencies=[])]}}
    candidate=parse_semantic_wire_v4(json.dumps(wire),AgentRequest('test',utterance,inputs),registry)
    plan=compile_semantic_plan(AgentRequest('test',utterance,inputs),candidate,registry,build_semantic_compiler_contract(registry))
    _check_parameter_plan(registry,plan,dict(inputs=inputs,arguments=[asdict(x) for x in decl]),())
    assert plan.steps[1].arguments['input_path'].step_id=='pca'


def test_task_function_choice_is_registry_owned_and_explicit_function_is_protected(tmp_path):
    from agent.orchestration.semantic_compiler import SemanticPlanCandidate, SemanticPlanStep, SemanticRegisteredChoiceSource
    registry=build_default_tool_registry()
    candidate=SemanticPlanCandidate((SemanticPlanStep('pca','run_scanpy',(SemanticRegisteredChoiceSource('function_name','pp.pca'),)),))
    inputs=dict(input_path=str(tmp_path/'source.h5ad'),output_dir=str(tmp_path/'out'))
    plan=compile_semantic_plan(AgentRequest('task','Use Scanpy for PCA',inputs),candidate,registry,build_semantic_compiler_contract(registry))
    assert plan.steps[0].arguments['function_name']=='pp.pca'
    with pytest.raises(ValueError):
        compile_semantic_plan(AgentRequest('task','Run neighbors',dict(inputs,function_name='pp.neighbors')),candidate,registry,build_semantic_compiler_contract(registry))


def test_native_declarations_survive_persisted_record_and_reject_fanout():
    from types import SimpleNamespace
    from agent.application.dialogue_execution import _validate_argument_record
    from agent.application.session_state import digest
    registry=build_default_tool_registry();utterance,decl=declarations()
    inputs=_bind_arguments(registry,utterance,{},decl)
    admitted=dict(kind='execute',operation='plan',inputs=inputs,arguments=[asdict(d) for d in decl],arguments_received_inputs_sha256=digest({}))
    record=SimpleNamespace(utterance=utterance,submission=None,admitted=admitted)
    _validate_argument_record(record)
    plan=AgentPlan('test','test','fanout',tuple(PlanStep(name,'run_scanpy',dict(function_name=fn,kwargs=kw)) for name,fn,kw in [
        ('pca','pp.pca',{'n_comps':8}),('graph','pp.neighbors',{'n_neighbors':12,'use_rep':'X_pca'}),
        ('extra','tl.umap',{'n_comps':8})]))
    with pytest.raises(IntentError):_check_parameter_plan(registry,plan,admitted,())
    inputs['run_scanpy__pp_pca__kwargs']={'n_comps':9}
    with pytest.raises(ValueError):_validate_argument_record(record)


def test_interactive_mixed_calls_bind_one_upload_and_accept_ordered_outputs(tmp_path,monkeypatch):
    import anndata as ad
    import numpy as np
    from agent.application import InteractiveAgentApplication
    from agent.application.local_resources import LocalResourceAdmission
    from agent.providers import PlanningModelFactoryRegistry
    from agent.orchestration import PlanningModelProfile
    from agent.tools.analysis import native_matrix
    utterance,raw=declarations(True);calls=[]
    source=tmp_path/'source.h5ad';ad.AnnData(np.ones((16,12))).write_h5ad(source)
    def native(self,args,stage,matrix):
        calls.append(args)
        matrix.write_h5ad(stage/'result.h5ad')
        return dict(completion='succeeded',versions={'scanpy':'SIMULATED'},implementation=[],native_output_type='NoneType',result_slots=[],measurement='unchanged fixture')
    monkeypatch.setattr(native_matrix.MatrixPackage,'native',native)
    class Model:
        model_id='scripted/native-scope'
        def complete(self,*,prompt,response_schema):
            p=json.loads(prompt)
            if 'turn_schema_version' in p:
                return json.dumps(dict(turn_schema_version=1,decision=dict(kind='execute_plan',target='run_scanpy',arguments=raw,roles=[dict(tool='run_scanpy',port='dataset',source='source.0')])))
            if 'output_selection_schema_version' in p:return json.dumps(dict(outputs=[dict(name='neighbors',step_id='graph',output_key='matrix_path')]))
            if 'schema_version' in response_schema.get('properties',{}):
                steps=[]
                for step,fn,parent in [('pca','pp_pca',None),('graph','pp_neighbors','pca')]:
                    sources=[dict(target=k,kind='input',input='run_scanpy__'+fn+'__'+k) for k in ('function_name','kwargs')]
                    sources.append(dict(target='dataset',kind='input',input='input_path') if parent is None else dict(target='dataset',kind='step_port',step=parent,source_port='dataset'))
                    steps.append(dict(step_id=step,tool='run_scanpy',sources=sources,control_dependencies=[]))
                return json.dumps(dict(schema_version=4,decision=dict(kind='plan',steps=steps)))
            raise RuntimeError('Generated prose is outside this simulated qualification')
    profile=PlanningModelProfile('test','scripted','scripted/native-scope');root=tmp_path/'workspace'
    app=InteractiveAgentApplication(root,model_profiles=(profile,),default_profile_id='test',approved_source_roots=(tmp_path,),planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':lambda p:Model()}))
    app.create_session('mixed');local=LocalResourceAdmission(root,approved_source_roots=(tmp_path,))
    record=local.register('data',source,input_type='h5ad',label='data',attribution='simulated')
    group=local.compose_group([record.resource_id],attachment_candidates=True)
    done=app.submit_turn('mixed','first',utterance,expected_generation=0,registered_input=group).to_dict()
    state=app._application.sessions.load('mixed')
    assert len(state.revisions)==1 and len(calls)==2,done
    run=app._application.run_store.load(state.turns[-1].run_id)
    assert run.steps[0].resolved_arguments['kwargs']=={'n_comps':8}
    assert run.steps[1].resolved_arguments['kwargs']=={'n_neighbors':12,'use_rep':'X_pca'}
    assert run.steps[1].resolved_arguments['input_path']==run.steps[0].result['matrix_path']
    from agent.application.dialogue_evidence import _explicit_user_parameter_facts, _Unavailable
    run = run.to_run_result()
    for step, planned in zip(run.steps, run.plan.steps):
        facts = _explicit_user_parameter_facts(app, state, run, step, planned)
        assert facts and facts[0].value['parameters'] == {
            'function_name': 'explicit_user_instruction', 'kwargs': 'explicit_user_instruction'}
    # Accepted execution values cannot substitute for the exact user declaration.
    step = run.steps[1]
    changed = replace(step, resolved_arguments={**step.resolved_arguments, 'kwargs': {'n_neighbors': 13}})
    with pytest.raises(_Unavailable):
        _explicit_user_parameter_facts(app, state, run, changed, run.plan.steps[1])
    assert app._application.sessions.load('mixed').to_dict()==state.to_dict()


def test_upload_guard_rejects_literal_substitution_and_changed_upload(tmp_path):
    import anndata as ad
    import numpy as np
    from agent.application.local_resources import LocalResourceAdmission,ResourceAdmissionError
    path=tmp_path/'input.h5ad';ad.AnnData(np.ones((2,3))).write_h5ad(path)
    owner=LocalResourceAdmission(tmp_path/'workspace',approved_source_roots=(tmp_path,))
    entry=owner.register('data',path,input_type='h5ad',label='data',attribution='fixture')
    group=owner.compose_group([entry.resource_id],attachment_candidates=True)
    binding=owner.bind_group(group,[dict(tool='run_scanpy',port='dataset',source='source.0')],verify_source=True)
    submission=dict(registered_input=binding.attribution(),execution_inputs=dict(binding.execution_inputs))
    args=dict(input_path=str(path),function_name='pp.pca')
    step=PlanStep('pca','run_scanpy',args)
    owner.validate_group_consumer(submission,step,args)
    other=dict(args,input_path=str(tmp_path/'other.h5ad'))
    with pytest.raises(ResourceAdmissionError):owner.validate_group_consumer(submission,PlanStep('bad','run_scanpy',other),other)
    path.write_bytes(b'changed upload')
    with pytest.raises(ResourceAdmissionError):owner.validate_group_consumer(submission,step,args)


@pytest.mark.parametrize('tool,function', [('run_scanpy','pp.neighbors'),('run_signac','Jaccard')])
def test_function_scope_does_not_replace_a_grounded_choice_value(tool,function):
    registry=build_default_tool_registry()
    text=f'Run {function}.'
    raw=dict(tool=tool,argument='function_name',native_function=function,literal=function,evidence=function)
    invalid=_scientific_arguments([_ground_declaration(raw,text)])
    with pytest.raises(IntentError) as rejected:
        _bind_arguments(registry,text,{},invalid)
    assert rejected.value.diagnostic_reason=='parameter_form_mismatch'
    assert rejected.value.operand=='parameter.form'
    valid=_scientific_arguments([_ground_declaration(dict(raw,value=function),text)])
    from agent.orchestration.native_parameters import input_name
    assert _bind_arguments(registry,text,{},valid)=={input_name(tool,function,'function_name'):function}
    other=next(fn for fn in registry.get(tool).required_arguments['function_name'].choices if fn!=function)
    wrong=_scientific_arguments([_ground_declaration(dict(raw,value=other),text)])
    with pytest.raises(IntentError):_bind_arguments(registry,text,{},wrong)


def compile_scoped_functions(tmp_path, *, registered_choice=False, crossed=False):
    from agent.orchestration.semantic_compiler import (SemanticPlanCandidate, SemanticPlanStep,
        SemanticRequestInputSource as Input, SemanticStepOutputSource as Output, SemanticRegisteredChoiceSource as Choice)
    registry=build_default_tool_registry()
    inputs=dict(input_path=str(tmp_path/'source.h5ad'),output_dir=str(tmp_path/'out'),
        run_scanpy__pp_pca__function_name='pp.pca',run_scanpy__pp_pca__kwargs={'n_comps':8},
        run_scanpy__pp_neighbors__function_name='pp.neighbors',run_scanpy__pp_neighbors__kwargs={'n_neighbors':12,'use_rep':'X_pca'})
    if not registered_choice:inputs['run_scanpy__tl_umap__function_name']='tl.umap'
    steps=[]
    for name,fn,parent in [('pca','pp_pca',None),('graph','pp_neighbors','pca'),('umap','tl_umap','graph')]:
        sources=[Input('dataset','input_path') if parent is None else Output('dataset',parent,'dataset')]
        sources.append(Choice('function_name','tl.umap') if name=='umap' and registered_choice
            else Input('function_name','run_scanpy__'+fn+'__function_name'))
        if name!='umap' or crossed:
            sources.append(Input('kwargs','run_scanpy__'+('pp_neighbors' if name=='umap' else fn)+'__kwargs'))
        steps.append(SemanticPlanStep(name,'run_scanpy',tuple(sources)))
    return compile_semantic_plan(AgentRequest('task','PCA, graph, UMAP',inputs),SemanticPlanCandidate(tuple(steps)),
        registry,build_semantic_compiler_contract(registry))


@pytest.mark.parametrize('registered_choice',[False,True])
def test_unrequested_native_kwargs_remain_omitted_in_composed_tasks(tmp_path,registered_choice):
    plan=compile_scoped_functions(tmp_path,registered_choice=registered_choice)
    assert plan.steps[0].arguments['kwargs']=={'n_comps':8}
    assert plan.steps[1].arguments['kwargs']=={'n_neighbors':12,'use_rep':'X_pca'}
    assert plan.steps[2].arguments['function_name']=='tl.umap'
    assert 'kwargs' not in plan.steps[2].arguments
    assert plan.steps[2].arguments['input_path'].step_id=='graph'


def test_explicit_cross_function_request_source_is_rejected(tmp_path):
    from agent.orchestration.semantic_compiler import SemanticPlanCompileError
    with pytest.raises(SemanticPlanCompileError) as rejected:
        compile_scoped_functions(tmp_path,crossed=True)
    assert rejected.value.code=='UNAUTHORIZED_REQUEST_INPUT'


def test_prerequisite_projection_preserves_scopes_and_never_mutates_request():
    from agent.orchestration.native_parameters import prerequisite_inputs
    tool=build_default_tool_registry().get('run_scanpy')
    inputs={'input_path':'original.h5ad','run_scanpy__pp_pca__function_name':'pp.pca',
            'run_scanpy__pp_pca__kwargs':{'n_comps':8}}
    projected=prerequisite_inputs(tool,inputs)
    assert projected['function_name']=='pp.pca' and projected['kwargs']=={'n_comps':8}
    assert 'function_name' not in inputs and 'kwargs' not in inputs
    assert prerequisite_inputs(tool,dict(inputs,run_scanpy__tl_umap__function_name='tl.umap')) is None
    assert prerequisite_inputs(tool,dict(inputs,function_name='tl.umap')) is None
    assert prerequisite_inputs(tool,dict(inputs,kwargs={'n_comps':9})) is None
    ordinary={'input_path':'original.h5ad','kwargs':{'n_comps':8}}
    assert prerequisite_inputs(tool,ordinary)==ordinary
    from agent.application.dialogue_execution import _requested_parameters, _missing_parameter_inputs
    registry=build_default_tool_registry()
    admitted={'tool':tool.name,'inputs':inputs}
    assert _missing_parameter_inputs(registry,tool.name,['kwargs'],inputs)==[]
    with pytest.raises(IntentError):_requested_parameters(registry,admitted,['kwargs'])
    # The earlier false question remains readable, not new question authority.
    assert _requested_parameters(registry,admitted,['kwargs'],historical=True)==['kwargs']


def test_function_scope_is_distinct_from_direct_original_source_scope(tmp_path):
    from dataclasses import asdict
    from agent.application.turn_decisions import _scientific_arguments, _ground_declaration
    plan=compile_scoped_functions(tmp_path)
    registry=build_default_tool_registry()
    utterance='Use 8 PCA components then 12 neighbors from X_pca.'
    raw=[dict(tool='run_scanpy',argument='kwargs',native_function='pp.pca',source='source.0',port='dataset',
        members=[dict(name='n_comps',kind='number',literal='8',evidence='8 PCA components')]),
        dict(tool='run_scanpy',argument='kwargs',native_function='pp.neighbors',members=[
            dict(name='n_neighbors',kind='number',literal='12',evidence='12 neighbors'),
            dict(name='use_rep',kind='string',literal='X_pca',evidence='from X_pca')])]
    declarations=_scientific_arguments([_ground_declaration(x,utterance) for x in raw])
    admitted=dict(inputs=_bind_arguments(registry,utterance,{},declarations),arguments=[asdict(d) for d in declarations],
        registered_input=dict(members=[dict(source_path=str(tmp_path/'source.h5ad'))]),
        roles=[dict(tool='run_scanpy',port='dataset',source='source.0')])
    _check_parameter_plan(registry,plan,admitted,())
    # Downstream native output is not the exact original file even with shared ancestry.
    crossed={**admitted,'arguments':[admitted['arguments'][0],dict(admitted['arguments'][1],source='source.0',port='dataset')]}
    with pytest.raises(IntentError):_check_parameter_plan(registry,plan,crossed,())
