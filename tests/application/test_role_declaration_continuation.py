"""A second role family uses the same durable declaration continuation.

Reference/query fixtures compile and preflight through production contracts. The
recording executor deliberately withholds scientific execution and no embedding
or model authority is claimed. Identity/state negatives perform zero science.
"""
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Event, Thread

import pytest

from agent.application import InteractiveAgentApplication, InteractiveBoundaryError
from agent.application.local_resources import ResourceAdmissionError
from agent.application.session_state import AnalysisSession, SessionError
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas.orchestration import _serialize

from test_multi_input_role_admission import (LABEL, PROFILE, ROLES as ORIGINAL_ROLES,
    TransferBindingModel, transfer_binding_harness)


ROLES = sorted(deepcopy(ORIGINAL_ROLES), key=lambda item: (item['tool'], item['port']))
REQUIRED_ROLES = [{key: role[key] for key in ('tool', 'port')} for role in ROLES]
OBJECTIVE = f'Transfer cell labels between these selected datasets using reference column "{LABEL}".'
ROLE_ANSWER = 'The first selected dataset is reference, and the second is query.'
REFERENCE_ANSWER = 'The first selected dataset is reference.'
QUERY_ANSWER = 'The second selected dataset is query.'
CONTRADICTORY = 'Actually the second selected dataset is reference.'
UNRELATED = 'Inspect an unrelated dataset.'
REFERENCE = next(role for role in ROLES if role['port'] == 'reference_dataset')
QUERY = next(role for role in ROLES if role['port'] == 'query_dataset')


def label_declaration(utterance=OBJECTIVE, label=LABEL, **changes):
    start = utterance.index(label)
    return dict(tool='transfer_cell_labels', argument='reference_label_key', literal=label,
                start=start, end=start + len(label)) | changes


def request(*, roles=(), required=REQUIRED_ROLES, declarations=None):
    return dict(kind='request_roles', target='transfer_cell_labels',
        required_roles=deepcopy(required), roles=deepcopy(list(roles)), argument=None,
        arguments=[label_declaration()] if declarations is None else declarations, context='none')


def answer(roles=ROLES, *, pending='@roles', declarations=()):
    return dict(kind='answer_roles', pending=pending, roles=deepcopy(roles),
                arguments=deepcopy(list(declarations)))


class TransferRoleContinuationModel(TransferBindingModel):
    def __init__(self, calls, *, decisions=None, variant=None, before_interpret=None):
        super().__init__(calls, variant=variant, roles=ROLES)
        self.decisions = {OBJECTIVE: request(), ROLE_ANSWER: answer(),
            REFERENCE_ANSWER: answer([REFERENCE]), QUERY_ANSWER: answer([QUERY]),
            CONTRADICTORY: answer([REFERENCE | dict(source='source.1')]),
            UNRELATED: dict(kind='execute_plan', target='inspect_scATAC', context='none')} | (decisions or {})
        self.before_interpret = before_interpret

    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        if 'turn_schema_version' in value and value['utterance'] in self.decisions:
            self.calls.append(dict(value=value, prompt=prompt, schema=response_schema))
            if self.before_interpret:
                self.before_interpret(value['utterance'])
            return json.dumps(dict(turn_schema_version=1,
                decision=self.decisions[value['utterance']]), ensure_ascii=False)
        return super().complete(prompt=prompt, response_schema=response_schema)


def transfer_continuation_harness(tmp_path, *, decisions=None, variant=None, before_interpret=None, uploads=False):
    original, records, group, executor, calls, companions = transfer_binding_harness(tmp_path)
    roots = (tmp_path / 'approved',)
    if uploads:
        upload_root = tmp_path / 'workspace' / 'uploads'
        upload_root.mkdir()
        roots += (upload_root,)
    app = InteractiveAgentApplication(original._application.workspace_root,
        model_profiles=(PROFILE,), default_profile_id=PROFILE.profile_id,
        registry=original._application.registry, executor=executor,
        approved_source_roots=roots,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': lambda _:
            TransferRoleContinuationModel(calls, decisions=decisions, variant=variant,
                                          before_interpret=before_interpret)}))
    return app, records, group, executor, calls, companions


def submit(app, *, turn='first', utterance=OBJECTIVE, group=None, generation=0):
    return app.submit_turn('session', turn, utterance, expected_generation=generation,
                           registered_input=group).to_dict()


def pending(app, view, *, turn='first', missing=REQUIRED_ROLES):
    assert view['response']['kind'] == 'clarify' and view['revision_id'] is None and not view['steps'], view
    state = app._application.sessions.load('session')
    record = next(item for item in state.interactions if item.turn_id == turn)
    assert record.prerequisite['field'] == 'roles'
    assigned = {(role['tool'], role['port']) for role in record.admitted['roles']}
    assert [role for role in _serialize(record.prerequisite['required_roles'])
            if (role['tool'], role['port']) not in assigned] == missing
    assert not state.revisions and state.generation == 0
    return record


def assert_withheld(result, executor, records, companions):
    assert result['error']['code'] == 'TEST_SCIENCE_WITHHELD' and result['revision_id'] is None, result
    step, = executor.plans[-1].steps
    assert step.tool_name == 'transfer_cell_labels'
    assert step.arguments['reference_h5ad_path'] == records[0].source_path
    assert step.arguments['query_h5ad_path'] == records[1].source_path
    assert step.arguments['reference_label_key'] == LABEL
    assert all(step.arguments[key] == value for key, value in companions.items())
    return step


def test_reference_query_continuation_compiles_shared_roles_and_exact_original_objective(tmp_path):
    app, records, group, executor, calls, companions = transfer_continuation_harness(tmp_path)
    token = dict(input_set_id='operator', definition_sha256='a' * 64)
    group = replace(group, resource_choice=token)
    origin = pending(app, submit(app, group=group))
    assert not executor.plans and len(calls) == 1
    assert not app._application.sessions.load('session').turns
    result = submit(app, turn='answer', utterance=ROLE_ANSWER)
    step = assert_withheld(result, executor, records, companions)
    state = app._application.sessions.load('session')
    assert state.interactions[0] == origin and not state.revisions
    resumed = state.interactions[-1]
    assert _serialize(resumed.admitted['roles']) == ROLES
    metadata = _serialize(resumed.admitted['registered_input'])
    assert metadata['members'] == _serialize(origin.submission['registered_input']['members'])
    assert metadata['resource_choice'] == token
    run = app._application.run_store.load(result['run_id'])
    assert run.request.prompt == OBJECTIVE
    original_received = _serialize(origin.submission)
    assert all(run.request.inputs[key] == value for key, value in original_received['execution_inputs'].items())
    answer_prompt = next(call['value'] for call in calls if call['value'].get('utterance') == ROLE_ANSWER)
    assert answer_prompt['dialogue']['pending_prerequisite']['objective'] == OBJECTIVE
    assert answer_prompt['dialogue']['pending_prerequisite']['required_roles'] == REQUIRED_ROLES
    assert [item['source'] for item in answer_prompt['dialogue']['selected_sources']] == ['source.0', 'source.1']
    for call in calls:
        assert str(tmp_path) not in call['prompt']
        assert not any(record.resource_id in call['prompt'] or record.record_sha256 in call['prompt'] for record in records)
    counts = len(calls), len(executor.plans)
    before = state.to_dict()
    assert submit(app, turn='answer', utterance=ROLE_ANSWER) == result
    assert counts == (len(calls), len(executor.plans)) and app._application.sessions.load('session').to_dict() == before
    (tmp_path / 'secondary-role-continuation-witness.json').write_text(json.dumps(dict(
        scope='Reference/query binding, compiler and preflight only; all science deliberately withheld.',
        objective=OBJECTIVE, answer=ROLE_ANSWER, original_turn_id=origin.turn_id,
        original_generation=origin.base_generation, original_revision=origin.base_revision_id,
        selected=metadata['members'], resource_choice=token, required_roles=REQUIRED_ROLES, roles=ROLES,
        compiled_arguments=_serialize(step.arguments), run_id=run.run_id,
        scientific_execution_count=0, revision_count=0), ensure_ascii=False, indent=2), encoding='utf-8')


def test_partial_role_answer_is_durable_and_contradiction_cannot_reassign_it(tmp_path):
    app, records, group, executor, calls, companions = transfer_continuation_harness(tmp_path)
    origin = pending(app, submit(app, group=group))
    partial = pending(app, submit(app, turn='reference', utterance=REFERENCE_ANSWER),
                      turn='reference', missing=[{key: QUERY[key] for key in ('tool', 'port')}])
    assert not executor.plans
    assert _serialize(partial.admitted['roles']) == [REFERENCE]
    bad = submit(app, turn='contradictory', utterance=CONTRADICTORY)
    assert bad['response']['kind'] == 'clarify' and bad['revision_id'] is None
    assert not executor.plans and app._application.sessions.load('session').interactions[0] == origin
    done = submit(app, turn='query', utterance=QUERY_ANSWER)
    assert_withheld(done, executor, records, companions)
    assert _serialize(app._application.sessions.load('session').interactions[-1].admitted['roles']) == ROLES


def test_initial_explicit_role_is_preserved_while_only_missing_association_is_answered(tmp_path):
    app, records, group, executor, _, companions = transfer_continuation_harness(tmp_path,
        decisions={OBJECTIVE: request(roles=[REFERENCE])})
    origin = pending(app, submit(app, group=group), missing=[{key: QUERY[key] for key in ('tool','port')}])
    assert _serialize(origin.admitted['roles']) == [REFERENCE]
    rejected = submit(app, turn='contradictory', utterance=CONTRADICTORY)
    assert rejected['response']['kind'] == 'clarify' and not executor.plans
    result = submit(app, turn='query', utterance=QUERY_ANSWER)
    assert_withheld(result, executor, records, companions)
    assert app._application.sessions.load('session').interactions[0] == origin


@pytest.mark.parametrize('bad_answer', [
    answer(pending='@not-pending'), answer(pending='@parameters'),
    answer([ROLES[0] | dict(source='source.7'), ROLES[1]]),
    answer([ROLES[0], ROLES[1] | dict(source=ROLES[0]['source'])]),
    answer([ROLES[0], ROLES[1] | dict(tool='cluster_cells', port='analysis')]),
    answer([ROLES[0], ROLES[1] | dict(port='unregistered_port')]),
])
def test_invalid_pending_unoffered_and_conflicting_roles_fail_without_execution(tmp_path, bad_answer):
    app, _, group, executor, _, _ = transfer_continuation_harness(tmp_path,
        decisions={ROLE_ANSWER: bad_answer})
    origin = pending(app, submit(app, group=group))
    result = submit(app, turn='answer', utterance=ROLE_ANSWER)
    assert result['response']['status'] in ('failed', 'clarification') and result['revision_id'] is None, result
    assert not executor.plans and not app._application.sessions.load('session').revisions
    assert app._application.sessions.load('session').interactions[0] == origin


@pytest.mark.parametrize('variant', ['swap_datasets', 'swap_embeddings', 'wrong_port', 'repeat_consumer'])
def test_completed_roles_cannot_allow_compiled_consumer_substitution(tmp_path, variant):
    app, _, group, executor, _, _ = transfer_continuation_harness(tmp_path, variant=variant)
    pending(app, submit(app, group=group))
    result = submit(app, turn='answer', utterance=ROLE_ANSWER)
    assert result['revision_id'] is None and not executor.plans, result
    assert not app._application.sessions.load('session').revisions


@pytest.mark.parametrize('variant', ['reordered', 'changed_digest', 'changed_choice', 'changed_bytes'])
def test_pending_role_answer_preserves_exact_selection_and_resource_choice(tmp_path, variant):
    app, records, group, executor, calls, _ = transfer_continuation_harness(tmp_path)
    group = replace(group, resource_choice=dict(input_set_id='operator', definition_sha256='a' * 64))
    origin = pending(app, submit(app, group=group))
    if variant == 'reordered':
        changed = replace(group, members=tuple(reversed(group.members)))
    elif variant == 'changed_digest':
        members = _serialize(group.members)
        members[1]['record_sha256'] = 'f' * 64
        changed = replace(group, members=members)
    elif variant == 'changed_choice':
        changed = replace(group, resource_choice=dict(input_set_id='different', definition_sha256='a' * 64))
    else:
        Path(records[1].source_path).write_bytes(b'Replaced selected current input after role clarification.')
        changed = None
    try:
        result = submit(app, turn='answer', utterance=ROLE_ANSWER, group=changed)
    except ResourceAdmissionError:
        pass
    else:
        assert result['response']['status'] in ('failed', 'clarification') and result['revision_id'] is None, result
    state = app._application.sessions.load('session')
    assert not executor.plans and not state.revisions and state.interactions[0] == origin


def test_role_answer_stale_generation_rejects_before_provider_and_preserves_state(tmp_path):
    app, _, group, executor, calls, _ = transfer_continuation_harness(tmp_path)
    pending(app, submit(app, group=group))
    before = app._application.sessions.load('session').to_dict()
    count = len(calls)
    with pytest.raises(InteractiveBoundaryError) as error:
        submit(app, turn='stale', utterance=ROLE_ANSWER, generation=1)
    assert error.value.error.code == 'INTERACTIVE_GENERATION_CONFLICT'
    assert not executor.plans and len(calls) == count
    assert app._application.sessions.load('session').to_dict() == before


def test_unrelated_new_task_does_not_inherit_role_pending_objective_or_sources(tmp_path):
    app, _, group, executor, _, _ = transfer_continuation_harness(tmp_path)
    pending(app, submit(app, group=group))
    result = submit(app, turn='unrelated', utterance=UNRELATED)
    assert result['revision_id'] is None and not executor.plans, result
    interaction = app._application.sessions.load('session').interactions[-1]
    admitted = interaction.admitted
    assert admitted['inputs'] == {}
    assert not admitted['roles'] and not admitted['registered_input']['members']
    assert 'continuation' not in admitted
    assert interaction.prerequisite['origin_turn_id'] == 'unrelated'
    assert interaction.prerequisite['tool'] == 'inspect_scATAC'


def test_role_answer_without_any_pending_request_cannot_create_a_task(tmp_path):
    app, _, group, executor, _, _ = transfer_continuation_harness(tmp_path)
    result = submit(app, turn='answer', utterance=ROLE_ANSWER, group=group)
    assert result['response']['kind'] == 'clarify' and result['revision_id'] is None, result
    assert not executor.plans and not app._application.sessions.load('session').revisions


@pytest.mark.parametrize('changes', [dict(literal='Invented label'),
                                   dict(argument='unregistered_column'), dict(tool='evaluate_cell_clustering')])
def test_role_reply_cannot_invent_or_redirect_exact_string_declarations(tmp_path, changes):
    utterance = ROLE_ANSWER + f' Use reference column "{LABEL}".'
    declaration = label_declaration(utterance) | changes
    app, _, group, executor, _, _ = transfer_continuation_harness(tmp_path,
        decisions={utterance:answer(declarations=[declaration])})
    origin = pending(app, submit(app, group=group))
    rejected = submit(app, turn='invalid-label', utterance=utterance)
    assert rejected['revision_id'] is None and rejected['response']['status'] in ('failed', 'clarification'), rejected
    assert not executor.plans and not app._application.sessions.load('session').revisions
    assert app._application.sessions.load('session').interactions[0] == origin


def test_role_reply_grounds_unique_exact_label_despite_wrong_provider_offsets(tmp_path):
    # Work Unit B locates a unique exact quote; offsets cannot redirect its value.
    utterance = ROLE_ANSWER + f' Use reference column "{LABEL}".'
    declaration = label_declaration(utterance) | dict(start=0, end=len(LABEL))
    app, _, group, executor, _, _ = transfer_continuation_harness(tmp_path,
        decisions={utterance: answer(declarations=[declaration])})
    origin = pending(app, submit(app, group=group))
    submit(app, turn='exact-label', utterance=utterance)
    assert len(executor.plans) == 1
    assert executor.plans[0].steps[0].arguments['reference_label_key'] == LABEL
    state = app._application.sessions.load('session')
    assert state.interactions[0] == origin and not state.revisions


def test_conflicting_concurrent_role_answers_and_changed_retry_fail_closed(tmp_path):
    started, release = Event(), Event()
    def before_interpret(utterance):
        if utterance == ROLE_ANSWER:
            started.set()
            assert release.wait(20), 'The test did not release the original pending answer.'
    app, records, group, executor, calls, companions = transfer_continuation_harness(tmp_path,
        before_interpret=before_interpret)
    origin = pending(app, submit(app, group=group))
    result, errors = [], []
    def continue_once():
        try:
            result.append(submit(app, turn='answer', utterance=ROLE_ANSWER))
        except Exception as error:
            errors.append(error)
    worker = Thread(target=continue_once)
    worker.start()
    assert started.wait(20), 'The original answer did not reach the Interpreter.'
    count = len(calls)
    try:
        with pytest.raises(InteractiveBoundaryError) as changed:
            submit(app, turn='answer', utterance=CONTRADICTORY)
        assert changed.value.error.code == 'INTERACTIVE_TURN_CONFLICT'
        concurrent = submit(app, turn='other-answer', utterance=CONTRADICTORY)
        assert concurrent['response']['kind'] == 'clarify' and concurrent['revision_id'] is None
        assert not executor.plans
    finally:
        release.set()
        worker.join(30)
    assert not worker.is_alive() and not errors, errors
    assert len(result) == 1
    # The existing leases are per turn. The second answer cannot capture an
    # interpreting predecessor, and the first loses its latest-turn anchor.
    assert result[0]['response']['kind'] == 'clarify' and result[0]['revision_id'] is None
    assert not executor.plans and not app._application.sessions.load('session').revisions
    before = len(calls), len(executor.plans)
    assert submit(app, turn='answer', utterance=ROLE_ANSWER) == result[0]
    assert submit(app, turn='other-answer', utterance=CONTRADICTORY) == concurrent
    with pytest.raises(InteractiveBoundaryError) as changed:
        submit(app, turn='answer', utterance=CONTRADICTORY)
    assert changed.value.error.code == 'INTERACTIVE_TURN_CONFLICT'
    assert before == (len(calls), len(executor.plans))
    assert app._application.sessions.load('session').interactions[0] == origin


def test_partial_role_state_replays_after_actual_process_restart_without_reupload(tmp_path):
    app, records, group, executor, _, companions = transfer_continuation_harness(tmp_path)
    pending(app, submit(app, group=group))
    pending(app, submit(app, turn='reference', utterance=REFERENCE_ANSWER), turn='reference',
            missing=[{key: QUERY[key] for key in ('tool', 'port')}])
    state = app._application.sessions.load('session')
    decoded = AnalysisSession.from_dict(_serialize(state.to_dict()))
    assert decoded == state and not executor.plans
    script = '''
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'tests/application'))
from agent.application import InteractiveAgentApplication
from agent.orchestration import build_default_tool_registry
from agent.providers import PlanningModelFactoryRegistry
from test_human_epizoo_mapping import WithholdScience
from test_role_declaration_continuation import PROFILE, TransferRoleContinuationModel, QUERY_ANSWER, OBJECTIVE, ROLES, submit
root = Path(sys.argv[1])
registry = build_default_tool_registry()
executor, calls = WithholdScience(registry), []
app = InteractiveAgentApplication(root / 'workspace', model_profiles=(PROFILE,), default_profile_id=PROFILE.profile_id,
    registry=registry, executor=executor, approved_source_roots=(root / 'approved',),
    planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':lambda _:TransferRoleContinuationModel(calls)}))
state = app._application.sessions.load('session')
assert state.interactions[-1].prerequisite['field'] == 'roles' and not calls and not executor.plans
result = submit(app, turn='fresh-process', utterance=QUERY_ANSWER)
assert result['error']['code'] == 'TEST_SCIENCE_WITHHELD', result
assert len(executor.plans) == 1 and not result['revision_id']
run = app._application.run_store.load(result['run_id'])
assert run.request.prompt == OBJECTIVE
counts = len(calls), len(executor.plans)
assert submit(app, turn='fresh-process', utterance=QUERY_ANSWER) == result
assert counts == (len(calls), len(executor.plans))
print(json.dumps(dict(result=result, arguments=dict(executor.plans[0].steps[0].arguments), counts=counts)))
'''
    child = subprocess.run([sys.executable, '-B', '-c', script, str(tmp_path)],
        capture_output=True, text=True, env=os.environ.copy(), timeout=90)
    assert child.returncode == 0, child.stdout + child.stderr
    observed = json.loads(child.stdout)
    arguments = observed['arguments']
    assert arguments['reference_h5ad_path'] == records[0].source_path
    assert arguments['query_h5ad_path'] == records[1].source_path
    assert arguments['reference_label_key'] == LABEL
    assert all(arguments[key] == value for key, value in companions.items())


@pytest.mark.parametrize('variant', ['pending_id', 'origin', 'role_source', 'resource_choice'])
def test_persisted_role_binding_corruption_rejects_during_session_decode(tmp_path, variant):
    app, _, group, executor, _, _ = transfer_continuation_harness(tmp_path)
    group = replace(group, resource_choice=dict(input_set_id='operator', definition_sha256='a' * 64))
    pending(app, submit(app, group=group))
    pending(app, submit(app, turn='reference', utterance=REFERENCE_ANSWER), turn='reference',
            missing=[{key: QUERY[key] for key in ('tool', 'port')}])
    value = _serialize(app._application.sessions.load('session').to_dict())
    record = value['interactions'][-1]
    if variant == 'pending_id':
        record['snapshot']['prerequisite']['binding_turn_id'] = 'not-an-interaction'
    elif variant == 'origin':
        record['prerequisite']['origin_turn_id'] = 'not-an-interaction'
    elif variant == 'role_source':
        record['admitted']['roles'][0]['source'] = 'source.1'
    else:
        record['admitted']['registered_input']['resource_choice']['definition_sha256'] = 'f' * 64
    with pytest.raises(SessionError):
        AnalysisSession.from_dict(value)
    assert not executor.plans


def test_pending_roles_accept_only_appended_required_registration(tmp_path):
    """A further registered file can fill a missing role in the same objective.

    This is bounded admission coverage; no new scientific qualification is claimed.
    """
    from fastapi.testclient import TestClient
    from agent.application.uploads import H5ADUploadAdmission
    from agent.web.app import create_app
    from agent.application.dialogue_execution import _role_binding
    from types import SimpleNamespace
    extra_required = dict(tool='evaluate_cell_clustering', port='ground_truth_dataset')
    decisions = {OBJECTIVE: request(required=REQUIRED_ROLES + [extra_required])}
    app, records, group, executor, calls, companions = transfer_continuation_harness(
        tmp_path, decisions=decisions, uploads=True)
    origin = pending(app, submit(app, group=group), missing=REQUIRED_ROLES + [extra_required])
    # Upload after the objective is pending; original registrations remain exact.
    uploads = H5ADUploadAdmission(app.resources, tmp_path / 'workspace' / 'uploads')
    with TestClient(create_app(app, uploads=uploads)) as client:
        response = client.put('/api/v1/uploads/additional',
            params={'filename': 'simulated-additional-ground-truth.h5ad'},
            content=Path(records[0].source_path).read_bytes(),
            headers={'Content-Type': 'application/octet-stream'})
        assert response.status_code == 201, response.text
        extra = app.resources.load(response.json()['resource_id'])
    extended = app.resources.compose_group([*(r.resource_id for r in records), extra.resource_id], companions)
    current = SimpleNamespace(submission=dict(execution_inputs=_serialize(extended.execution_inputs),
        registered_input=extended.attribution()))
    binding = _role_binding(origin, origin, current)
    assert len(binding.members) == 3 and binding.members[:2] == group.members
    # The ordinary capture projects the appended file with a new opaque position.
    reply = submit(app, turn='answer', utterance=REFERENCE_ANSWER, group=extended)
    latest = pending(app, reply, turn='answer', missing=[{k: QUERY[k] for k in ('tool', 'port')}, extra_required])
    assert len(latest.admitted['registered_input']['members']) == 3
    prompt = next(c['value'] for c in reversed(calls) if 'turn_schema_version' in c['value'])
    assert [s['source'] for s in prompt['dialogue']['selected_sources']] == ['source.0', 'source.1', 'source.2']
    assert not executor.plans
    reordered = app.resources.compose_group([extra.resource_id, *(r.resource_id for r in records)], companions)
    with pytest.raises(ValueError):
        _role_binding(origin, latest, SimpleNamespace(submission=dict(
            execution_inputs=_serialize(reordered.execution_inputs), registered_input=reordered.attribution())))
