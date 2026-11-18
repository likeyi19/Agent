"""Canonical Session task/candidate and exact answer ledger; no science/provider."""
from dataclasses import replace
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from agent.application.session_state import Interaction, SessionError, SessionConflictError
from agent.application.session_store import FileSessionStore
from agent.application import preparation_state as owner
from agent.orchestration.semantic_compiler import (
    SemanticPlanCandidate, SemanticPlanStep, ScientificRequirement, candidate_identity,
)


def pending(store):
    candidate = SemanticPlanCandidate((SemanticPlanStep('peaks','call_scATAC_peaks'),))
    interaction = Interaction('origin','Call peaks',None,0,dict(relations={},bases={}))
    record = owner.start('session', interaction, 'Call peaks', candidate=candidate, contract_identity='registry-v1')
    req = ScientificRequirement(candidate_identity(candidate),'peaks','call_scATAC_peaks',None,
                                'effective_genome_size','effective_genome_size','scalar')
    result = owner.question(record, interaction, (req,))
    interaction = replace(interaction, status='clarification', preparation=record,
                          admitted=dict(preparation_result=result))
    store._update('session', lambda s: replace(s, interactions=(interaction,)))
    return interaction, req


def reply(origin, req, *, name='answer', value=100):
    token = owner.choices(origin)[0]['handle']
    interaction = Interaction(name,'',None,0,dict(relations={},bases={}), submission=dict(
        clarification_choice=dict(question_turn_id=origin.turn_id,handle=token),
        parameter_overrides=[dict(tool='call_scATAC_peaks',step_id='peaks',argument='effective_genome_size',value=value)]))
    return interaction, {req.requirement_id:value}


@pytest.fixture
def store(tmp_path):
    instance = FileSessionStore(tmp_path)
    instance.create('session')
    return instance


def test_candidate_stored_once_restart_exact_answer_chain(store):
    origin, req = pending(store)
    interaction, values = reply(origin,req)
    before = store.load('session')
    record = owner.answer_question(before.interactions,interaction,values=values)
    interaction = replace(interaction,preparation=record)
    store._update('session',lambda s:replace(s,interactions=(*s.interactions,interaction)))
    saved = FileSessionStore(store.root).load('session')
    assert owner.scalar_bindings(saved.interactions,record) == {'peaks':{'effective_genome_size':100}}
    assert owner.candidate_for(saved.interactions,record) == owner.candidate_for(saved.interactions,origin.preparation)
    assert sum('candidate' in i.preparation for i in saved.interactions) == 1
    assert 'task' not in record
    assert not saved.turns and not saved.revisions
    assert saved == before.from_dict(json.loads(json.dumps(saved.to_dict())))


def test_resource_answer_resolves_only_its_exact_parameter_consumer(store):
    from agent.application.structured_parameters import prepared_resource_values, check_plan
    from agent.application.turn_decisions import IntentError
    from agent.orchestration import build_default_tool_registry
    from agent.schemas import AgentPlan, PlanStep
    candidate = SemanticPlanCandidate((SemanticPlanStep('annotation', 'annotate_scATAC_cell_types'),))
    origin = Interaction('origin', '物种是人类', None, 0, dict(relations={}, bases={}))
    record = owner.start('session', origin, origin.utterance, candidate=candidate, contract_identity='registry-v1')
    offer = dict(handle='species:human', label='Human', value='Human')
    req = ScientificRequirement(candidate_identity(candidate), 'annotation', 'annotate_scATAC_cell_types',
                                None, 'species', 'species', 'resource').to_dict()
    req['offers'] = [offer]
    result = owner.question(record, origin, (req,))
    origin = replace(origin, preparation=record, status='clarification',
                     admitted=dict(inputs=dict(species='人类'), preparation_result=result))
    token = owner.choices(origin)[0]['handle']
    answer = Interaction('answer', '', None, 0, dict(relations={}, bases={}), submission=dict(
        clarification_choice=dict(question_turn_id='origin', handle=token)))
    answered = owner.answer_question((origin,), answer, values={req['requirement_id']: offer})
    admitted = owner.continued_admission((origin,), answered, 'continued')
    answer = replace(answer, preparation=answered, admitted=admitted)
    store._update('session', lambda s: replace(s, interactions=(origin, answer)))
    state = FileSessionStore(store.root).load('session')
    origin, answer = state.interactions
    registry = build_default_tool_registry()
    plan = AgentPlan('plan', 'continued', 'fixture', (PlanStep('annotation', 'annotate_scATAC_cell_types',
                                                            dict(species='Human')),))
    values = prepared_resource_values(plan, answer.admitted, state.interactions, registry, interaction=answer)
    row = dict(tool='annotate_scATAC_cell_types', argument='species', value='人类')
    check_plan(plan, [row], answer.admitted['inputs'], registry, answer.admitted, prepared_values=values)
    assert values == {('annotation', 'species'): 'Human'}
    assert answer.admitted['inputs']['species'] == origin.admitted['inputs']['species'] == '人类'
    assert origin == state.interactions[0] and not state.turns and not state.revisions
    with pytest.raises(IntentError, match='planning_failed'):
        check_plan(plan, [row], answer.admitted['inputs'], registry, answer.admitted)
    for changed in (dict(species='Mouse'), dict(species='Human', function_name='unexpected')):
        invalid = replace(plan, steps=(replace(plan.steps[0], arguments=changed),))
        with pytest.raises(IntentError):
            prepared_resource_values(invalid, answer.admitted, state.interactions, registry, interaction=answer)
    sibling = replace(plan, steps=(replace(plan.steps[0], step_id='other'),))
    with pytest.raises(IntentError, match='invalid_reference'):
        prepared_resource_values(sibling, answer.admitted, state.interactions, registry, interaction=answer)
    assert prepared_resource_values(plan, origin.admitted, state.interactions, registry, interaction=origin) == {}
    for key, value in [('question_version', 'stale'), ('step_id', 'other'),
                       ('value', dict(offer, value='Mouse'))]:
        forged = dict(answer.preparation, answers=[dict(answer.preparation['answers'][0], **{key: value})])
        with pytest.raises(SessionError):
            owner.validate_record('session', replace(answer, preparation=forged), (origin,))


def test_consumed_question_and_changed_question_token_rejected(store):
    origin,req=pending(store)
    interaction,values=reply(origin,req)
    record=owner.answer_question(store.load('session').interactions,interaction,values=values)
    store._update('session',lambda s:replace(s,interactions=(*s.interactions,replace(interaction,preparation=record))))
    other,values=reply(origin,req,name='another')
    with pytest.raises(SessionConflictError):
        owner.answer_question(store.load('session').interactions,other,values=values)
    bad=replace(interaction,submission=dict(clarification_choice=dict(question_turn_id='origin',handle='changed')))
    with pytest.raises(SessionConflictError):
        owner.answer_question((origin,),bad,values=values)


def test_unrelated_task_does_not_steal_pending_question(store):
    origin,req=pending(store)
    unrelated=Interaction('unrelated','Another task',None,0,dict(relations={},bases={}))
    unrelated=replace(unrelated,preparation=owner.start('session',unrelated,unrelated.utterance))
    store._update('session',lambda s:replace(s,interactions=(*s.interactions,unrelated)))
    answer,values=reply(origin,req)
    record=owner.answer_question(store.load('session').interactions,answer,values=values)
    assert record['task_id']==origin.preparation['task_id']
    assert record['previous']=='origin'


def test_stale_generation_and_wrong_requirement_rejected(store):
    origin,req=pending(store)
    answer,values=reply(origin,req)
    with pytest.raises(SessionConflictError):
        owner.answer_question((origin,),replace(answer,base_generation=1),values=values)
    with pytest.raises(SessionConflictError):
        owner.answer_question((origin,),answer,values={'other':100})


def test_exact_answer_cannot_bind_an_independent_requirement(store):
    origin, req = pending(store)
    other = replace(req, step_id='independent')
    # Build a fresh captured question with two independent operations.
    origin = replace(origin, admitted=dict(preparation_result=owner.question(
        origin.preparation, origin, (req, other))))
    answer, values = reply(origin, req)
    values[other.requirement_id] = 200
    with pytest.raises(SessionConflictError, match='does not address'):
        owner.answer_question((origin,), answer, values=values)


def test_crash_after_capture_never_reserves_science(store):
    origin,req=pending(store)
    answer,values=reply(origin,req)
    record=owner.answer_question((origin,),answer,values=values)
    store._update('session',lambda s:replace(s,interactions=(*s.interactions,replace(answer,preparation=record))))
    saved=FileSessionStore(store.root).load('session')
    assert saved.interactions[-1].status=='interpreting'
    assert saved.interactions[-1].admitted is None
    assert saved.turns==saved.revisions==()


def test_atomic_competing_answers_and_historical_immutability(store):
    origin,req=pending(store)
    def submit(name):
        answer,values=reply(origin,req,name=name)
        def mutate(state):
            record=owner.answer_question(state.interactions,answer,values=values)
            return replace(state,interactions=(*state.interactions,replace(answer,preparation=record)))
        try:
            store._update('session',mutate)
            return True
        except (SessionConflictError,SessionError):
            return False
    with ThreadPoolExecutor(2) as pool:
        outcomes=list(pool.map(submit,['one','two']))
    assert sorted(outcomes)==[False,True]
    with pytest.raises((SessionConflictError,SessionError)):
        store._update('session',lambda s:replace(s,interactions=(replace(s.interactions[0],
            preparation=dict(s.interactions[0].preparation,task_id='changed')),*s.interactions[1:])))


def test_canonical_question_capture_continues_after_candidate_acquisition():
    origin=Interaction('intake','Inspect the selected data',None,0,dict(relations={},bases={}))
    root=owner.start('session',origin,origin.utterance,context={'intake_owner':'roles'})
    origin=replace(origin,preparation=root)
    candidate=SemanticPlanCandidate((SemanticPlanStep('peaks','call_scATAC_peaks'),))
    current=Interaction('candidate','Selected input',None,0,dict(relations={},bases={}))
    from agent.orchestration.semantic_compiler import candidate_record
    record=dict(version=2,task_id=root['task_id'],task_origin='intake',previous='intake',answers=[],
        candidate_origin='candidate',candidate_id=candidate_identity(candidate),candidate=candidate_record(candidate),contract_identity='registry')
    req=ScientificRequirement(candidate_identity(candidate),'peaks','call_scATAC_peaks',None,
                              'effective_genome_size','effective_genome_size','scalar')
    current=replace(current,preparation=record,status='clarification',
        admitted=dict(preparation_result=owner.question(record,current,(req,))))
    assert owner.capture_intake((origin,current),0,None,{},None)['binding_turn_id'] == 'candidate'


def test_interpreted_role_cannot_claim_one_requirement_and_bind_another_consumer():
    from agent.application.scientific_preparation import role_requirements
    from agent.orchestration import build_default_tool_registry
    member=dict(resource_id='local-'+'a'*64,record_sha256='b'*64,
        source_path='/original/source.h5ad',input_type='h5ad',label='Original')
    admitted=dict(tool='transfer_cell_labels',roles=[],required_roles=[
        dict(tool='transfer_cell_labels',port=p) for p in ('reference_dataset','query_dataset')],
        registered_input=dict(members=[member]))
    initial=Interaction('task','Transfer labels',None,0,dict(relations={},bases={}))
    root=owner.start('session',initial,initial.utterance,context={'intake_owner':'roles'})
    needs=role_requirements(admitted,build_default_tool_registry())
    initial=replace(initial,preparation=root,status='clarification',
        admitted=dict(admitted,preparation_result=owner.question(root,initial,needs)))
    req=needs[0]
    answer=dict(requirement_id=req['requirement_id'],value=dict(handle='source.0',label='Original',source=member,
        role=dict(tool='transfer_cell_labels',port='query_dataset',source='source.0')),
        answering_interaction='answer',question_version=owner.observation(initial)['question_version'],
        question_turn_id='task',step_id=req['step_id'],argument=req['argument'],member=None,
        kind='input_role',provenance='interpreted_role')
    record=dict(version=2,task_id=root['task_id'],task_origin='task',candidate_origin=None,candidate_id=None,
                previous='task',answers=[answer])
    current=Interaction('answer','Use the first file',None,0,dict(relations={},bases={}),preparation=record)
    with pytest.raises(SessionError,match='exact consumer'):
        owner.validate_record('session',current,(initial,))


@pytest.mark.parametrize('variant', ['inputs', 'selection', 'candidate', 'downgrade'])
def test_exact_answer_durable_identity_cannot_be_rewritten(store, variant):
    origin, req = pending(store)
    answer, values = reply(origin, req)
    record = owner.answer_question((origin,), answer, values=values)
    answer = replace(answer, preparation=record)
    owner.validate_record('session', answer, (origin,))
    if variant in {'inputs', 'selection'}:
        submission = dict(answer.submission)
        submission.update(execution_inputs={'input_path': '/changed.h5ad'} if variant == 'inputs' else {})
        if variant == 'selection':
            submission['registered_input'] = {'resource_id': 'changed'}
        answer = replace(answer, submission=submission)
    elif variant == 'candidate':
        record = dict(record, candidate=origin.preparation['candidate'], candidate_origin=answer.turn_id,
                      contract_identity=origin.preparation['contract_identity'])
        answer = replace(answer, preparation=record)
    else:
        answer = replace(answer, preparation=dict(record, version=1))
    with pytest.raises(SessionError):
        owner.validate_record('session', answer, (origin,))


def test_scope_projection_does_not_retarget_an_unanswered_independent_declaration():
    from types import SimpleNamespace
    from agent.application.dialogue_execution import _parameter_records
    a = dict(tool='run_scanpy', native_function='pp.scale', argument='kwargs', value={'max_value': 5})
    b = dict(tool='run_scanpy', native_function='pp.scale', argument='count_slot', value='X')
    previous = SimpleNamespace(turn_id='before', preparation=None, prerequisite=None, admitted=dict(arguments=[a, b],
        preparation_result=dict(requirements=[dict(requirement_id='req', tool_name='run_scanpy',
            native_function='pp.scale', argument='kwargs')])))
    admitted = dict(arguments=[dict(a, step_id='one'), dict(b, step_id='two')])
    current = SimpleNamespace(turn_id='now', admitted=admitted, preparation=dict(previous='before', answers=[
        dict(kind='scope', step_id='@declaration', question_turn_id='before', requirement_id='req', value='one')]))
    projected = _parameter_records(admitted, [previous, current], interaction=current)
    assert projected[1]['arguments'][0]['step_id'] == 'one'
    assert 'step_id' not in projected[1]['arguments'][1]
    assert 'step_id' not in previous.admitted['arguments'][0]


@pytest.mark.parametrize('accepted_unicode',[False,True])
def test_compiled_independent_source_dependencies_invalidate_only_related_answer(accepted_unicode):
    from agent.schemas import AgentRequest
    from agent.orchestration import build_default_tool_registry
    from agent.orchestration.semantic_compiler import (
        SemanticRequestInputSource as Input, SemanticStepOutputSource as Output,
        SemanticRegisteredChoiceSource as Choice, analyze_candidate, AnalysisDisposition,
    )
    registry = build_default_tool_registry()
    from agent.schemas import PriorOutputBinding,PriorOutputRef
    binding=PriorOutputBinding('已接受运行','检查步骤','inspect_scATAC','dataset','raw_scatac_dataset.v1','a'*64,'b'*64,'c'*64,'accepted-file.v1')
    left=PriorOutputRef(binding,'input_path') if accepted_unicode else '/exact/left.h5ad'
    candidate = SemanticPlanCandidate((
        SemanticPlanStep('left', 'inspect_scATAC', (Input('dataset','reference_input_path'),)),
        SemanticPlanStep('right', 'inspect_scATAC', (Input('dataset','query_input_path'),)),
        SemanticPlanStep('left_consumer', 'run_snapatac2',
            (Output('dataset','left','dataset'), Choice('function_name','metrics.tsse'))),
        SemanticPlanStep('right_consumer', 'run_snapatac2',
            (Output('dataset','right','dataset'), Choice('function_name','metrics.tsse'))),
    ))
    analysis = analyze_candidate(AgentRequest('independent','Analyse both sources',
        dict(reference_input_path='/exact/left.h5ad', query_input_path='/exact/right.h5ad', output_dir='/out')), candidate, registry)
    assert analysis.disposition is not AnalysisDisposition.INVALID, analysis.error
    operations=tuple(replace(op,arguments={**op.arguments,'path':left}) if op.step_id=='left' else op for op in analysis.operations)
    requirements=[]
    for side in ('left','right'):
        sid=side+'_consumer'
        paths=owner.operation_sources(sid, operations, registry)
        assert paths == (['accepted:'+owner.digest(binding.to_dict())] if accepted_unicode and side=='left' else ['/exact/'+side+'.h5ad'])
        req=ScientificRequirement(analysis.candidate_id,sid,'run_snapatac2','metrics.tsse',
            'gene_anno','gene_anno_path','resource',dependencies=(sid+':source:bytes',sid+':declaration:assembly')).to_dict()
        requirements.append(dict(req,source_dependencies=paths,
            offers=[dict(handle='resource.'+side,label=side,port='gene_anno')]))
    origin=Interaction('origin','Analyse both sources',None,0,{})
    record=owner.start('session',origin,origin.utterance,candidate=candidate,contract_identity='registry')
    origin=replace(origin,preparation=record,status='clarification',
        admitted=dict(preparation_result=owner.question(record,origin,requirements)))
    chain=[origin]
    for index,req in enumerate(requirements):
        previous=chain[-1]
        choice=next(c for c in owner.choices(previous) if c['requirement_id']==req['requirement_id'])
        turn=Interaction('answer'+str(index),'',None,0,{},submission=dict(
            clarification_choice=dict(question_turn_id=previous.turn_id,handle=choice['handle'])))
        record=owner.answer_question(chain,turn,values={req['requirement_id']:req['offers'][0]})
        turn=replace(turn,preparation=record,status='clarification',
            admitted=dict(preparation_result=owner.question(record,turn,requirements)))
        chain.append(turn)
    before=repr(chain)
    correction=Interaction('correction','Left assembly is hg38.',None,0,{})
    corrected=owner.record_answers(chain[-1],correction,{},provenance='interpreted_source')
    corrected['declarations']=[dict(scope=dict(owner='annotation',sources=[dict(accepted_binding=binding.to_dict()) if accepted_unicode else dict(source_path='/exact/left.h5ad')]),
        member='assembly',old_value='hg19',value='hg38')]
    assert owner.resource_bindings(chain,corrected)=={
        'right_consumer':{'gene_anno_path':requirements[1]['offers'][0]}}
    assert repr(chain)==before
    # A member absent from the scientific requirement's dependency list does
    # not invalidate either selection merely because it shares a source.
    corrected['declarations'][0]['member']='context'
    assert set(owner.resource_bindings(chain,corrected))=={'left_consumer','right_consumer'}


def test_query_output_inherits_query_declarations_but_depends_on_both_inputs():
    from agent.schemas import PlanStep, StepOutputRef, PriorOutputBinding, PriorOutputRef
    from agent.orchestration import build_default_tool_registry
    from agent.application.session_state import digest
    registry=build_default_tool_registry()
    accepted=PriorOutputBinding('accepted-run','inspect','inspect_scATAC','dataset',
        'raw_scatac_dataset.v1','a'*64,'b'*64,'c'*64,'accepted-file.v1')
    operations=(
        PlanStep('transfer','transfer_cell_labels',dict(
            reference_h5ad_path='/reference.h5ad',query_h5ad_path=PriorOutputRef(accepted,'input_path'))),
        PlanStep('consumer','inspect_scATAC',dict(path=StepOutputRef('transfer','annotation_path'))),
    )
    # Projection only: annotation_path is not an H5AD channel and this is not an
    # executable plan. The compiler's actual registered output lineage is tested.
    token='accepted:'+digest(accepted.to_dict())
    assert owner.operation_sources('consumer',operations,registry)==['/reference.h5ad',token]
    assert owner.operation_sources('consumer',operations,registry,lineage=True)==[token]
    assert owner.source_identities([dict(accepted_binding=accepted.to_dict())])=={token}
    changed=replace(accepted,run_id='another-run')
    assert owner.source_identities([dict(accepted_binding=changed.to_dict())])!={token}


def test_reference_answer_invalidates_only_its_dependent_branch():
    from agent.orchestration.semantic_compiler import AnalyzedOperation
    from agent.schemas import StepOutputRef
    ops=tuple(AnalyzedOperation(s,'run_snapatac2',
        {'input_path':StepOutputRef(deps[0],'fragments_path')} if deps else {},deps) for s,deps in [
        ('map_a',()),('map_b',()),('import_a',('map_a',)),('import_b',('map_b',))])
    ops=(*ops[:-1],replace(ops[-1],dependencies=('map_a','map_b')))
    assert owner.operation_dependencies('import_b',ops)==['import_b','map_b']
    origin=Interaction('task','Map one library in two independent branches',None,0,{})
    record=owner.start('session',origin,origin.utterance)
    requirements=[]
    for sid,port in [('map_a','reference'),('import_a','chrom_sizes'),('import_b','chrom_sizes')]:
        req=ScientificRequirement('',sid,'run_snapatac2',None,port,port+'_path','resource',
            dependencies=(sid+':reference:fasta',) if port=='chrom_sizes' else ()).to_dict()
        requirements.append(dict(req,source_dependencies=['/same-library'],source_steps=owner.operation_dependencies(sid,ops),
            offers=[dict(handle='resource.'+sid,label=sid,port=port)]))
    chain=[replace(origin,preparation=record,status='clarification',admitted=dict(preparation_result=owner.question(record,origin,requirements)))]
    for req in [*requirements[1:],requirements[0]]:
        previous=chain[-1]
        choice=next(c for c in owner.choices(previous) if c['requirement_id']==req['requirement_id'])
        turn=Interaction(req['step_id'],'',None,0,{},submission=dict(clarification_choice=dict(question_turn_id=previous.turn_id,handle=choice['handle'])))
        record=owner.answer_question(chain,turn,values={req['requirement_id']:req['offers'][0]})
        chain.append(replace(turn,preparation=record,status='clarification',admitted=dict(preparation_result=owner.question(record,turn,requirements))))
    bindings=owner.resource_bindings(chain,record)
    assert set(bindings)=={'map_a','import_b'}


@pytest.mark.parametrize('member,invalidated', [('feature_source','peaks'),('feature_category','peaks'),
    ('genome_source','chrom_sizes'),('genome_index_source','chrom_sizes')])
def test_native_owner_source_member_dependencies_invalidate_only_affected_component(member,invalidated):
    from agent.application.native_raw_handoff import _component
    from agent.orchestration.semantic_compiler import ScientificInputRequired
    origin=Interaction('task','Use declared fragments and separate components',None,0,{})
    record=owner.start('session',origin,origin.utterance)
    requirements=[]
    for port in ('peaks','chrom_sizes','gene_anno'):
        with pytest.raises(ScientificInputRequired) as exc:
            _component(dict(catalog={},selected=None,custom={},resources={}),port)
        need,=exc.value.requirements
        req=ScientificRequirement('',port,'run_snapatac2',None,port,need.argument,'resource',
            dependencies=tuple(port+':'+d for d in need.dependencies)).to_dict()
        requirements.append(dict(req,source_dependencies=['/fragments'],offers=[dict(handle='resource.'+port,label=port,port=port)]))
    chain=[replace(origin,preparation=record,status='clarification',admitted=dict(preparation_result=owner.question(record,origin,requirements)))]
    for req in requirements:
        previous=chain[-1]
        choice=next(c for c in owner.choices(previous) if c['requirement_id']==req['requirement_id'])
        turn=Interaction(req['port'],'',None,0,{},submission=dict(clarification_choice=dict(question_turn_id=previous.turn_id,handle=choice['handle'])))
        record=owner.answer_question(chain,turn,values={req['requirement_id']:req['offers'][0]})
        chain.append(replace(turn,preparation=record,status='clarification',admitted=dict(preparation_result=owner.question(record,turn,requirements))))
    changed=dict(record,previous=chain[-1].turn_id,answers=[],declarations=[dict(member=member,old_value=None,value='source.1',
        scope=dict(owner='fragments',sources=[dict(source_path='/fragments')]))])
    assert set(owner.resource_bindings(chain,changed))=={'peaks','chrom_sizes','gene_anno'}-{invalidated}


def test_parameter_origin_follows_exact_interaction_when_admissions_are_equal():
    from agent.application.dialogue_execution import parameter_record_owners
    admitted=dict(inputs={'species':'human'},arguments=[])
    old=Interaction('old','unrelated text',None,0,dict(relations={},bases={}),admitted=admitted)
    current=Interaction('current','human',None,0,dict(relations={},bases={}),admitted=dict(admitted))
    rows=parameter_record_owners(current.admitted,(old,current),interaction=current)
    assert rows==[(current.admitted,current)]
    assert parameter_record_owners(admitted,(old,current))==[(admitted,None)]


def test_unicode_accepted_source_anchor_invalidates_only_its_dependent_answer():
    from agent.schemas import PlanStep,PriorOutputBinding,PriorOutputRef
    from agent.orchestration import build_default_tool_registry
    from agent.orchestration.preparation_dependencies import accepted_source_identity
    binding=PriorOutputBinding('已接受运行','检查步骤','inspect_scATAC','dataset','raw_scatac_dataset.v1','a'*64,'b'*64,'c'*64,'accepted-file.v1')
    registry=build_default_tool_registry()
    operations=(PlanStep('left','inspect_scATAC',dict(path=PriorOutputRef(binding,'input_path'))),PlanStep('right','inspect_scATAC',dict(path='/independent.h5ad')))
    anchor='accepted:'+owner.digest(binding.to_dict())
    assert accepted_source_identity(binding)==anchor
    assert owner.operation_sources('left',operations,registry)==[anchor]
    assert owner.source_identities([dict(accepted_binding=binding.to_dict())])=={anchor}


def test_final_assertion_checks_canonical_scalar_answer_without_original_declaration(store):
    from agent.application.structured_parameters import preparation_parameter_values,assert_parameter_projection
    from agent.application.dialogue_execution import _check_parameter_plan
    from agent.application.turn_decisions import IntentError
    from agent.orchestration import build_default_tool_registry,AgentPlan
    from agent.schemas import PlanStep
    origin,req=pending(store)
    answer,values=reply(origin,req)
    record=owner.answer_question((origin,),answer,values=values)
    answer=replace(answer,preparation=record,admitted=dict(inputs={},arguments=[]))
    registry=build_default_tool_registry();chain=(origin,answer)
    rows=preparation_parameter_values(answer,chain,registry)
    assert rows[0]['value']==100 and rows[0]['step_id']=='peaks'
    plan=AgentPlan('plan','request','test',(PlanStep('peaks','call_scATAC_peaks',dict(effective_genome_size=100)),))
    assert assert_parameter_projection(plan,(),registry,answer.admitted,prepared_values=rows)=={('peaks','effective_genome_size')}
    _check_parameter_plan(registry,plan,answer.admitted,chain,interaction=answer)
    bad=replace(plan,steps=(replace(plan.steps[0],arguments=dict(effective_genome_size=999)),))
    with pytest.raises(IntentError):assert_parameter_projection(bad,(),registry,answer.admitted,prepared_values=rows)
    with pytest.raises(IntentError):_check_parameter_plan(registry,bad,answer.admitted,chain,interaction=answer)


def test_canonical_kwargs_member_merges_without_losing_explicit_keys():
    from agent.application.structured_parameters import preparation_parameter_values,assert_parameter_projection
    from agent.orchestration import build_default_tool_registry,AgentPlan
    from agent.schemas import PlanStep
    from agent.application.turn_decisions import IntentError
    candidate=SemanticPlanCandidate((SemanticPlanStep('cluster','run_scanpy'),))
    origin=Interaction('origin','cluster',None,0,{})
    record=owner.start('s',origin,'cluster',candidate=candidate,contract_identity='registry')
    req=ScientificRequirement(record['candidate_id'],'cluster','run_scanpy','tl.leiden','kwargs','kwargs','scalar',member='resolution').to_dict()
    origin=replace(origin,preparation=record,admitted=dict(preparation_result=owner.question(record,origin,[req])))
    answer=Interaction('answer','3',None,0,{})
    # The reducer consumes an already validated canonical answer, independent of UI transport.
    ledger=owner.record_answers(origin,answer,{req['requirement_id']:0.5},provenance='exact_submission')
    answer=replace(answer,preparation=ledger)
    registry=build_default_tool_registry();prepared=preparation_parameter_values(answer,(origin,answer),registry)
    rows=(dict(step_id='cluster',tool='run_scanpy',native_function='tl.leiden',argument='kwargs',value=dict(random_state=0)),)
    plan=AgentPlan('plan','r','test',(PlanStep('cluster','run_scanpy',dict(input_path='/source.h5ad',output_dir='/out',function_name='tl.leiden',kwargs=dict(random_state=0,resolution=0.5))),))
    registry.validate_arguments(plan.steps[0].tool_name,plan.steps[0].arguments)
    assert assert_parameter_projection(plan,rows,registry,dict(inputs={}),prepared_values=prepared)=={('cluster','kwargs')}
    for kwargs in [dict(resolution=0.5),dict(random_state=0,resolution=9)]:
        bad=replace(plan,steps=(replace(plan.steps[0],arguments=dict(function_name='tl.leiden',kwargs=kwargs)),))
        with pytest.raises(IntentError):assert_parameter_projection(bad,rows,registry,dict(inputs={}),prepared_values=prepared)
