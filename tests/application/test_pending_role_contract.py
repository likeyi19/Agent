"""Typed shared role questions and pure captured-source admission contracts."""
from dataclasses import FrozenInstanceError, asdict, replace
import json

import pytest

pytestmark = pytest.mark.usefixtures('isolated_binding_catalog')


from agent.application.local_resources import (
    RegisteredInputGroup, ResourceAdmissionError, group_role_inputs, validate_group_roles,
)
from agent.application.turn_decisions import (
    Execute, InputRole, IntentError, RequestRoles, RoleAnswer, RoleRequirement,
    ScopedArgument, ScalarChoiceArgument, decision_schema, interpret, parse_decision,
)
from agent.orchestration.registry import build_default_tool_registry


def requirements(tool='evaluate_cell_clustering'):
    names = ('analysis', 'ground_truth_dataset') if tool == 'evaluate_cell_clustering' else (
        'reference_dataset', 'query_dataset')
    return [dict(tool=tool, port=port) for port in names]


def roles(tool='evaluate_cell_clustering'):
    return [requirement | dict(source=f'source.{index}')
            for index, requirement in enumerate(requirements(tool))]


def wire(value):
    return json.dumps(dict(turn_schema_version=1, decision=value))


def request(**changes):
    return dict(kind='request_roles', target='evaluate_cell_clustering',
                required_roles=requirements(), roles=[]) | changes


def selection():
    return RegisteredInputGroup([dict(resource_id='local-' + str(index)*64,
        record_sha256=str(index + 2)*64, source_path=f'/unopened/source-{index}.h5ad',
        input_type='h5ad', label=f'Selected {index + 1} 类型 Δ') for index in range(2)], {})


def public(*, selected=True, pending=False):
    value = dict(bases={}, relations=[], dialogue=dict(outputs=[],
        tools=['evaluate_cell_clustering', 'transfer_cell_labels'], scientific_parameters=[
            dict(tool='evaluate_cell_clustering', argument='label_key', required=True,
                 description='Explicit evaluation-only column.', exact_string=True),
            dict(tool='transfer_cell_labels', argument='reference_label_key', required=True,
                 description='Explicit reference column.', exact_string=True),
        ]))
    if selected:
        value['dialogue'].update(selected_sources=[dict(source=f'source.{index}', position=index + 1,
            label=f'Selected {index + 1} 类型 Δ', input_type='h5ad') for index in range(2)],
            input_roles=requirements() + requirements('transfer_cell_labels'))
    if pending:
        value['dialogue']['pending_prerequisite'] = dict(handle='@roles', field='roles',
            objective='Evaluate the selected fixed clustering against the selected ground truth.',
            required_roles=requirements(), roles=[])
    return value



def test_role_request_retains_exact_strings_and_is_not_an_executable_plan():
    utterance = 'Evaluate these datasets with column "Cell 类型 Δ".'
    literal = 'Cell 类型 Δ'
    start = utterance.index(literal)
    argument = ScopedArgument('evaluate_cell_clustering', 'label_key', literal, start, start + len(literal))
    value = parse_decision(wire(request(arguments=[asdict(argument)])))
    assert value == RequestRoles('evaluate_cell_clustering',
        tuple(RoleRequirement(**r) for r in requirements()), arguments=(argument,))
    assert not isinstance(value, Execute)
    with pytest.raises(FrozenInstanceError):
        value.required_roles = ()


@pytest.mark.parametrize('changes', [
    dict(required_roles=[]),
    dict(required_roles=requirements()*5), dict(required_roles=[requirements()[0]]*2),
    dict(required_roles=[requirements()[0] | dict(source='source.0'), requirements()[1]]),
    dict(target='cluster_cells'), dict(context='previous'), dict(roles=roles()),
    dict(roles=[dict(tool='transfer_cell_labels', port='reference_dataset', source='source.0')]),
    dict(required_roles=[dict(tool='illegal/name', port='analysis'), requirements()[1]]),
    dict(inputs={}),
])
def test_role_request_wire_is_closed_and_requires_a_genuinely_missing_association(changes):
    with pytest.raises(IntentError, match='invalid_decision'):
        parse_decision(wire(request(**changes)))


def test_role_answers_allow_partial_associations_or_an_exact_scientific_declaration():
    reply = dict(kind='answer_roles', pending='@roles', roles=roles()[:1])
    assert parse_decision(wire(reply)) == RoleAnswer('@roles', (InputRole(**roles()[0]),))
    declaration = dict(tool='evaluate_cell_clustering', argument='label_key',
                       literal='类型 Δ', start=1, end=5)
    assert parse_decision(wire(reply | dict(roles=[], arguments=[declaration]))) == RoleAnswer(
        '@roles', (), (ScopedArgument(**declaration),))


@pytest.mark.parametrize('changes', [
    dict(roles=[]), dict(roles=None), dict(pending=''), dict(pending='x'*129),
    dict(roles=[dict(tool='evaluate_cell_clustering', port='analysis', source='source.16')]),
    dict(roles=roles(), required_roles=requirements()),
    dict(arguments=[dict(tool='epizoo_embed_cells', argument='species', literal='mouse',
                         start=0, end=5, value={'untrusted': 'mouse'})]),
])
def test_role_answer_has_no_arbitrary_values_or_source_identity_wire(changes):
    with pytest.raises(IntentError, match='invalid_decision'):
        parse_decision(wire(dict(kind='answer_roles', pending='@roles', roles=roles()[:1]) | changes))


def test_role_answer_retains_reviewed_scalar_choice_wire():
    declaration = dict(tool='epizoo_embed_cells', argument='species', literal='mouse',
                       start=0, end=5, value='mouse')
    decision = parse_decision(wire(dict(kind='answer_roles', pending='@roles',
        roles=roles()[:1], arguments=[declaration])))
    assert decision.arguments == (ScalarChoiceArgument(**declaration),)




def test_interpreter_gets_safe_original_roles_and_no_keyword_repair():
    class Model:
        def complete(self, *, prompt, response_schema):
            self.prompt = json.loads(prompt)
            self.schema = response_schema
            return wire(dict(kind='answer_roles', pending='@roles', roles=roles()))
    model = Model()
    answer = 'The first dataset is fixed analysis and the second is ground truth.'
    value = interpret(model, answer, public(pending=True))
    assert value == RoleAnswer('@roles', tuple(InputRole(**r) for r in roles()))
    assert model.prompt['utterance'] == answer
    assert model.prompt['dialogue']['pending_prerequisite'] == public(pending=True)['dialogue']['pending_prerequisite']
    instructions = ' '.join(model.prompt['instructions'])
    assert 'offered question and original objective' in instructions
    assert 'No stale or cross-task identity is valid' in instructions
    assert '/unopened/' not in json.dumps(model.prompt)


@pytest.mark.parametrize('tool', ['evaluate_cell_clustering', 'transfer_cell_labels'])
def test_pure_pending_admission_shares_registry_roles_without_opening_sources(tool, monkeypatch):
    registry, group = build_default_tool_registry(), selection()
    def forbidden(*args, **kwargs):
        raise AssertionError('Pure pending role admission opened a source.')
    monkeypatch.setattr('builtins.open', forbidden)
    required = tuple(RoleRequirement(**r) for r in requirements(tool))
    partial = validate_group_roles(group, roles(tool)[:1], registry, required)
    assert partial == (InputRole(**roles(tool)[0]),)
    completed = validate_group_roles(group, roles(tool), registry, required, complete=True)
    assert completed == tuple(InputRole(**r) for r in sorted(roles(tool), key=lambda r: (r['tool'], r['port'])))
    assert not group.roles and not group.execution_inputs
    assert len(group_role_inputs(group, [asdict(r) for r in completed], registry)) == 2


@pytest.mark.parametrize('attachment_candidates', [False, True])
def test_pending_distinct_consumers_can_explicitly_share_one_exact_source(monkeypatch, attachment_candidates):
    registry = build_default_tool_registry()
    group = selection()
    group = RegisteredInputGroup(group.members if attachment_candidates else group.members[:1], {},
                                 attachment_candidates=attachment_candidates)
    required = [dict(tool=tool, port='dataset') for tool in ('inspect_scATAC', 'mock_embed_cells')]
    assigned = [role | dict(source='source.0') for role in required]
    def forbidden(*args, **kwargs):
        raise AssertionError('Pure pending role admission opened a scientific source.')
    monkeypatch.setattr('builtins.open', forbidden)
    partial = validate_group_roles(group, assigned[:1], registry, required)
    assert partial == (InputRole(**assigned[0]),)
    completed = validate_group_roles(group, assigned, registry, required, complete=True)
    assert completed == tuple(InputRole(**role) for role in assigned)
    assert group_role_inputs(group, [asdict(role) for role in completed], registry) == {
        'input_path': group.members[0]['source_path']}


@pytest.mark.parametrize('complete', [False, True])
def test_pending_explicit_shared_alias_rejects_different_source_bindings(complete):
    registry, group = build_default_tool_registry(), selection()
    required = [dict(tool=tool, port='dataset') for tool in ('inspect_scATAC', 'mock_embed_cells')]
    assigned = [role | dict(source=f'source.{index}') for index, role in enumerate(required)]
    with pytest.raises(ResourceAdmissionError) as rejected:
        validate_group_roles(group, assigned, registry, required, complete=complete)
    assert rejected.value.code == 'MULTI_INPUT_ROLE_INVALID'


@pytest.mark.parametrize('variant', ['missing', 'count', 'duplicate_required', 'unsupported_port',
                                    'shared_alias', 'foreign_role', 'same_source', 'unoffered_source'])
def test_pending_role_compatibility_fails_closed(variant):
    registry, group = build_default_tool_registry(), selection()
    required, assigned = requirements(), roles()
    if variant == 'missing':
        assigned = assigned[:1]
    elif variant == 'count':
        required = required[:1]
    elif variant == 'duplicate_required':
        required = required[:1]*2
    elif variant == 'unsupported_port':
        required[0]['port'] = 'unreviewed'
    elif variant == 'shared_alias':
        required[1] = dict(tool='cluster_cells', port='analysis')
    elif variant == 'foreign_role':
        assigned[0] = dict(tool='transfer_cell_labels', port='reference_dataset', source='source.0')
    elif variant == 'same_source':
        assigned[1]['source'] = 'source.0'
    else:
        assigned[1]['source'] = 'source.7'
    with pytest.raises(ResourceAdmissionError) as rejected:
        validate_group_roles(group, assigned, registry, required, complete=True)
    assert rejected.value.code == 'MULTI_INPUT_ROLE_INVALID'


def test_existing_complete_group_binding_does_not_accept_partial_pending_associations():
    registry, group = build_default_tool_registry(), selection()
    validate_group_roles(group, roles()[:1], registry, requirements())
    with pytest.raises(ResourceAdmissionError) as rejected:
        group_role_inputs(group, roles()[:1], registry)
    assert rejected.value.code == 'MULTI_INPUT_ROLE_INVALID'
    assert parse_decision(wire(dict(kind='execute_plan', target='evaluate_cell_clustering'))) == Execute('evaluate_cell_clustering')


@pytest.mark.parametrize('association_stage', ['initial', 'answer'])
def test_ordinary_preplan_role_arrangements_are_rejected_without_science(tmp_path, association_stage):
    """Historical complete-role wires grant no current pre-Candidate authority."""
    from test_role_declaration_continuation import (
        OBJECTIVE, REFERENCE, transfer_continuation_harness, request, submit,
    )
    decision=request(roles=[REFERENCE] if association_stage=='initial' else [])
    assert isinstance(parse_decision(wire(decision)),RequestRoles)
    app, _, group, executor, calls, _=transfer_continuation_harness(tmp_path,
        decisions={OBJECTIVE:decision})
    result=submit(app,group=group)
    assert result['error']['code']=='INTERACTIVE_INTENT_BINDING_INVALID'
    state=app._application.sessions.load('session')
    assert not state.turns and not state.revisions and not executor.plans
    assert state.interactions[-1].preparation is None
    assert state.interactions[-1].submission['registered_input']['members']==group.members
    assert len(calls)==1


def candidate_transfer_harness(tmp_path):
    from test_multi_input_role_admission import TransferBindingModel, transfer_binding_harness, PROFILE, LABEL
    from test_response_contracts import OutcomeModel
    from agent.application.session_state import AnalysisSession,SessionError
    from agent.schemas.orchestration import _serialize
    original,records,group,executor,calls,companions=transfer_binding_harness(tmp_path)
    token=dict(input_set_id='operator',definition_sha256='a'*64)
    group=replace(group,attachment_candidates=True,resource_choice=token)
    class CandidateModel(TransferBindingModel):
        def complete(self,*,prompt,response_schema):
            value=json.loads(prompt)
            if 'turn_schema_version' in value:
                self.calls.append(dict(value=value))
                return wire(dict(kind='execute_plan',target='transfer_cell_labels',roles=[],arguments=[]))
            if value.get('focus')=='outcome' or 'grounding_review_version' in value:
                return OutcomeModel('Please select the required input.').complete(prompt=prompt,response_schema=response_schema)
            result=json.loads(super().complete(prompt=prompt,response_schema=response_schema))
            if 'semantic_prompt_version' in value:
                result['decision']['steps'][0]['sources']=[s for s in result['decision']['steps'][0]['sources']
                    if s['target'] not in {'reference_dataset','query_dataset'}]
            return json.dumps(result)
    from agent.application import InteractiveAgentApplication
    from agent.providers import PlanningModelFactoryRegistry
    app=InteractiveAgentApplication(original._application.workspace_root,registry=original._application.registry,executor=executor,
        model_profiles=(PROFILE,),default_profile_id=PROFILE.profile_id,approved_source_roots=(tmp_path/'approved',),
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':lambda _:CandidateModel(calls)}))
    return app,records,group,executor,calls,companions


@pytest.mark.parametrize('reselect_original', [False, True])
def test_candidate_sources_then_exact_column_retain_identity_and_resource_choice(tmp_path,reselect_original):
    from agent.application.session_state import AnalysisSession,SessionError
    from agent.schemas.orchestration import _serialize
    from test_multi_input_role_admission import LABEL
    app,records,group,executor,calls,companions=candidate_transfer_harness(tmp_path)
    token=_serialize(group.resource_choice)
    question=app.submit_turn('session','task','Transfer labels; let me choose sources and the reference label field.',
        expected_generation=0,registered_input=group)
    state=app._application.sessions.load('session');origin=state.interactions[-1]
    assert origin.preparation['candidate_id'] is not None and question.status=='clarification',question.to_dict()
    assert origin.admitted['registered_input'] is None
    from agent.application.source_observations import apply_source_choices
    assert apply_source_choices(dict(origin.admitted),[],{},app._application.registry,
        captured_group=group.attribution())==dict(origin.admitted)
    for index,(argument,label) in enumerate([('reference_h5ad_path','Selected reference'),('query_h5ad_path','Selected query')]):
        choice=next(c for c in question.clarification_choices if c.get('source_input') and c['argument']==argument and c['label']==label)
        question=app.submit_turn('session','source'+str(index),'',expected_generation=0,
            clarification_choice=dict(question_turn_id=question.turn_id,handle=choice['handle']))
        if index==0:
            bound=app._application.sessions.load('session').interactions[-1].admitted
            assert bound['registered_input']['resource_choice']==token
            assert len(bound['registered_input']['roles'])==1
            assert bound['registered_input']['roles'][0]['port']=='reference_dataset'
    assert question.status=='clarification',question.to_dict()
    choice=next(c for c in question.clarification_choices if c.get('source_field') and c['value']==LABEL)
    result=app.submit_turn('session','column','',expected_generation=0,
        clarification_choice=dict(question_turn_id=question.turn_id,handle=choice['handle']),
        registered_input=group if reselect_original else None)
    assert result.error.code=='TEST_SCIENCE_WITHHELD',result.to_dict()
    step=executor.plans[0].steps[0]
    assert step.arguments['reference_h5ad_path']==records[0].source_path
    assert step.arguments['query_h5ad_path']==records[1].source_path
    assert step.arguments['reference_label_key']==LABEL
    assert all(step.arguments[k]==v for k,v in companions.items())
    state=app._application.sessions.load('session')
    assert state.interactions[0]==origin
    assert state.interactions[-1].admitted['registered_input']['resource_choice']==token
    assert len({i.preparation['candidate_id'] for i in state.interactions})==1
    assert sum('semantic_prompt_version' in c.get('value',{}) for c in calls)==1
    assert sum('turn_schema_version' in c.get('value',{}) for c in calls)==1
    assert AnalysisSession.from_dict(_serialize(state.to_dict()))==state
    forged=_serialize(state.to_dict());forged['interactions'][-1]['admitted']['roles'][0]['source']='source.1'
    with pytest.raises((SessionError,ValueError)):AnalysisSession.from_dict(forged)


def test_role_request_cannot_bypass_existing_candidate_selection_guard(tmp_path, monkeypatch):
    """A capturable role question remains an execution request at M17.2 admission.

    The existing predicate independently owns candidate-reference language. This
    seam supplies its positive classification and verifies the new decision does
    not evade the already-authoritative guard by delaying executable planning.
    """
    from test_role_declaration_continuation import (
        OBJECTIVE, transfer_continuation_harness, submit,
    )
    seen = []
    def candidate_reference(interaction, prior):
        seen.append((interaction.turn_id, interaction.utterance, prior))
        return True
    monkeypatch.setattr('agent.application.scientific_guidance.references_candidate', candidate_reference)
    app, _, group, executor, calls, _ = transfer_continuation_harness(tmp_path)
    result = submit(app, group=group)
    assert seen == []  # Retired ordinary role wire is rejected before candidate admission.
    assert result['status'] == 'failed'
    assert result['error']['code'] == 'INTERACTIVE_INTENT_BINDING_INVALID'
    state = app._application.sessions.load('session')
    assert state.interactions[-1].prerequisite is None
    assert all(turn.request_id is None and turn.run_id is None for turn in state.turns)
    assert not state.revisions and not executor.plans
    assert sum('turn_schema_version' in (json.loads(c['prompt']) if 'prompt' in c else c) for c in calls) == 1
