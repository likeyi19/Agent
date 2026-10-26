"""Durable role answers continue ordinary uploaded-source scientific requests.

Only typed model responses are scripted. The primary witness reuses the genuine
UA3.6.3a numerical fixture and the normal upload, Session, Planner, compiler,
registered Leiden/evaluation owners, verification, evidence and Web worker.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from agent.application import InteractiveAgentApplication
from agent.application.uploads import H5ADUploadAdmission
from agent.orchestration import ToolRegistry, build_default_tool_registry
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas import RunLifecycleStatus, StepOutputRef
from agent.schemas.orchestration import _serialize
from agent.web.app import create_app
from agent.web.config import ScientificInputSet

from test_multi_input_scientific_roles import (
    ANSWER as LABEL_ANSWER, LABEL, PROFILE, ROLES, MultiInputModel, assert_no_owner, declaration,
    execute, multi_body, multi_input_harness, multi_input_work, snapshots,
    upload_sources,
)


OBJECTIVE = ('Cluster one of these selected datasets, then evaluate its fixed '
             'clustering against the other as ground truth. '
             f'Use ground-truth column "{LABEL}".')
ROLE_ANSWER = ('The first dataset is the fixed analysis input, '
               'and the second is ground truth.')
REQUIRED_ROLES = [{key: role[key] for key in ('tool', 'port')} for role in ROLES]


def role_request(utterance=OBJECTIVE, *, roles=(), label=True):
    return dict(kind='request_roles', target='evaluate_cell_clustering',
        required_roles=deepcopy(REQUIRED_ROLES), roles=deepcopy(list(roles)),
        argument=None, arguments=[declaration(utterance)] if label else [], context='none')


def role_answer(roles=ROLES, *, pending='@roles', arguments=()):
    return dict(kind='answer_roles', pending=pending, roles=deepcopy(roles),
                arguments=deepcopy(list(arguments)))


class RoleContinuationModel(MultiInputModel):
    def __init__(self, *, decisions=None, variant=None, before_interpret=None):
        super().__init__(variant=variant, before_interpret=before_interpret)
        self.decisions = {OBJECTIVE: role_request(), ROLE_ANSWER: role_answer()} | (decisions or {})

    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        if 'turn_schema_version' in value and value['utterance'] in self.decisions:
            self.calls.append(dict(value=value, prompt=prompt, schema=response_schema))
            if self.before_interpret:
                self.before_interpret()
            return json.dumps(dict(turn_schema_version=1,
                decision=self.decisions[value['utterance']]), ensure_ascii=False)
        return super().complete(prompt=prompt, response_schema=response_schema)


def role_continuation_harness(tmp_path, *, decisions=None, content=None,
                              variant=None, before_interpret=None):
    """The browser can use this same genuine fixture without a second workflow."""
    harness = multi_input_harness(tmp_path, content=content)
    def factory(profile):
        model = RoleContinuationModel(decisions=decisions, variant=variant,
                                      before_interpret=before_interpret)
        harness.models.append(model)
        return model
    harness.service = InteractiveAgentApplication(harness.service._application.workspace_root,
        model_profiles=(PROFILE,), default_profile_id=PROFILE.profile_id,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': factory}),
        approved_source_roots=(harness.uploads.root,))
    harness.uploads = H5ADUploadAdmission(harness.service.resources, harness.uploads.root)
    return harness


def assert_role_pending(harness, view, *, turn='first', missing=REQUIRED_ROLES):
    assert view['response']['kind'] == 'clarify' and view['revision_id'] is None, view
    assert not view['steps']
    state = harness.service._application.sessions.load('session')
    record = next(item for item in state.interactions if item.turn_id == turn)
    assert record.prerequisite['field'] == 'roles'
    assigned = {(role['tool'], role['port']) for role in record.admitted['roles']}
    assert [role for role in _serialize(record.prerequisite['required_roles'])
            if (role['tool'], role['port']) not in assigned] == missing
    assert not state.revisions and state.generation == 0
    return record


def test_uploaded_role_answer_executes_real_science_and_preserves_exact_history(tmp_path):
    harness = role_continuation_harness(tmp_path)
    choice = ScientificInputSet('captured', 'Explicit captured settings', {})
    original_science = snapshots(harness.source.parent.parent)
    with TestClient(create_app(harness.service, uploads=harness.uploads, input_sets=(choice,))) as client:
        with multi_input_work() as uploads:
            records = upload_sources(client, harness)
        assert not uploads and not harness.models
        selected = [row['resource_id'] for row in records]
        initial = multi_body(selected, utterance=OBJECTIVE) | dict(input_set_id=choice.input_set_id)
        with multi_input_work() as first:
            pending = execute(client, initial)
        assert_no_owner(first)
        assert first['interpreter'] == 1 and first['planner'] == first['compiler'] == 0
        origin = assert_role_pending(harness, pending)
        assert not harness.service._application.sessions.load('session').turns
        assert _serialize(origin.submission['registered_input']['resource_choice']) == choice.resource_choice()
        answer_payload = multi_body(turn='answer', utterance=ROLE_ANSWER)
        with multi_input_work() as resumed:
            result = execute(client, answer_payload)
        assert result['status'] == 'succeeded' and result['revision_id'], result
        assert resumed['production.leiden'] == resumed['production.evaluation'] == 1
        assert resumed['production.neighbors'] == resumed['production.epizoo'] == 0
        assert resumed['interpreter'] == resumed['planner'] == resumed['compiler'] == 1
        assert resumed['verification.clustering'] > 0 and resumed['verification.evaluation'] > 0
        run = harness.service._application.run_store.load(result['run_id'])
        assert run.lifecycle_status is RunLifecycleStatus.SUCCEEDED
        assert run.request.prompt == OBJECTIVE
        cluster, evaluation = run.steps
        assert [item.tool_name for item in run.steps] == ['cluster_cells', 'evaluate_cell_clustering']
        pins = [harness.service.resources.load(identifier) for identifier in selected]
        assert cluster.resolved_arguments['analysis_path'] == pins[0].source_path
        assert 'reference_h5ad_path' not in cluster.resolved_arguments
        assert evaluation.resolved_arguments['reference_h5ad_path'] == pins[1].source_path
        assert evaluation.resolved_arguments['label_key'] == LABEL
        assert run.plan.steps[1].arguments['analysis_path'] == StepOutputRef('clustering', 'analysis_path')
        assert all(item.verification.passed for item in run.steps)
        assert all(evaluation.result[key] == pytest.approx(1.)
                   for key in ('nmi', 'ari', 'ami', 'homogeneity'))
        state = harness.service._application.sessions.load('session')
        assert state.interactions[0] == origin and state.generation == 1 and len(state.revisions) == 1
        resumed_record = state.interactions[-1]
        assert _serialize(resumed_record.admitted['roles']) == ROLES
        attribution = _serialize(resumed_record.admitted['registered_input'])
        assert attribution['members'] == _serialize(origin.submission['registered_input']['members'])
        assert attribution['resource_choice'] == choice.resource_choice()
        assert resumed_record.base_generation == origin.base_generation == 0
        assert resumed_record.base_revision_id == origin.base_revision_id is None
        answer_call = next(call for model in harness.models for call in model.calls
                           if call['value'].get('utterance') == ROLE_ANSWER)
        offered = answer_call['value']['dialogue']['pending_prerequisite']
        assert offered['handle'] == '@roles' and offered['objective'] == OBJECTIVE
        assert offered['required_roles'] == REQUIRED_ROLES
        assert [row['source'] for row in answer_call['value']['dialogue']['selected_sources']] == ['source.0', 'source.1']
        for model in harness.models:
            for call in model.calls:
                assert str(tmp_path) not in call['prompt']
                assert not any(row.record_sha256 in call['prompt'] or row.resource_id in call['prompt'] for row in pins)
        path = f"/api/v1/sessions/session/revisions/{result['revision_id']}"
        calls_before = sum(len(model.calls) for model in harness.models)
        with multi_input_work() as historical:
            evidence = client.get(path + '/evidence', params={'output_name': 'evaluation'})
            assert evidence.status_code == 200
            facts = {row['field']: row for row in evidence.json()['facts']}
            assert facts['label_key']['value'] == LABEL and facts['nmi']['value'] == pytest.approx(1.)
            assert facts['explicit_user_parameters']['value']['parameters']['label_key'] == 'explicit_user_instruction'
            artifacts = client.get(path + '/artifacts').json()['artifacts']
            report = next(row for row in artifacts if row['artifact_type'] == 'analysis_report')
            assert client.get(path + '/artifacts/' + report['handle']).status_code == 200
            assert execute(client, answer_payload) == result
            history = client.get('/api/v1/sessions/session').json()
        assert not historical and calls_before == sum(len(model.calls) for model in harness.models)
        before = snapshots(harness.service._application.workspace_root)
        for pin in pins:
            source = Path(pin.source_path)
            source.rename(source.with_suffix('.archived'))
        with multi_input_work() as archived:
            assert execute(client, answer_payload) == result
            assert client.get('/api/v1/sessions/session').json() == history
            assert client.get(path + '/evidence', params={'output_name': 'evaluation'}).json() == evidence.json()
        assert not archived
    def forbidden(*args, **kwargs):
        raise AssertionError('Completed role continuation entered a provider or scientific owner.')
    default = build_default_tool_registry()
    registry = ToolRegistry(tuple(replace(default.get(name), function=forbidden) for name in default.names()))
    reopened = InteractiveAgentApplication(harness.service._application.workspace_root,
        model_profiles=(PROFILE,), default_profile_id=PROFILE.profile_id, registry=registry,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': forbidden}),
        approved_source_roots=(harness.uploads.root,))
    with multi_input_work() as restored:
        with TestClient(create_app(reopened, uploads=H5ADUploadAdmission(reopened.resources, harness.uploads.root))) as client:
            assert client.get('/api/v1/sessions/session').json() == history
            assert execute(client, answer_payload) == result
            assert client.get(path + '/evidence', params={'output_name': 'evaluation'}).json() == evidence.json()
    assert not restored and snapshots(harness.source.parent.parent) == original_science
    # Only the explicitly archived current-input names differ after acceptance.
    after = snapshots(harness.service._application.workspace_root)
    assert {key: value for key, value in before.items() if key not in {pin.source_path for pin in pins}} == {
        key: value for key, value in after.items() if not key.endswith('.archived')}
    (tmp_path / 'role-continuation-scientific-witness.json').write_text(json.dumps(dict(
        scope='Synthetic scientific/software acceptance; no foundation inference or biological qualification.',
        objective=OBJECTIVE, answer=ROLE_ANSWER, original_turn_id=origin.turn_id,
        original_generation=origin.base_generation, original_revision=origin.base_revision_id,
        resource_choice=choice.resource_choice(), selected=[dict(resource_id=pin.resource_id,
            record_sha256=pin.record_sha256, source_sha256=pin.source_sha256, source_path=pin.source_path) for pin in pins],
        required_roles=REQUIRED_ROLES, roles=ROLES, initial_counts=dict(first), resumed_counts=dict(resumed),
        historical_counts=dict(historical), archived_counts=dict(archived), restored_counts=dict(restored),
        run_id=run.run_id, revision_id=result['revision_id'], label_key=LABEL,
        metrics={key: evaluation.result[key] for key in ('nmi', 'ari', 'ami', 'homogeneity')},
        resolved_arguments=[_serialize(item.resolved_arguments) for item in run.steps],
        prompt_sizes=[dict(prompt_bytes=len(call['prompt'].encode('utf-8')),
            schema_bytes=len(json.dumps(call['schema'], sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')))
            for model in harness.models for call in model.calls]), ensure_ascii=False, indent=2), encoding='utf-8')


@pytest.mark.parametrize('content', ['wrong_cell_order', 'missing_provenance'])
def test_role_completion_does_not_replace_scientific_owner_eligibility(tmp_path, content):
    harness = role_continuation_harness(tmp_path, content=content)
    with TestClient(create_app(harness.service, uploads=harness.uploads)) as client:
        records = upload_sources(client, harness)
        with multi_input_work() as initial:
            assert_role_pending(harness, execute(client, multi_body([row['resource_id'] for row in records], utterance=OBJECTIVE)))
        assert_no_owner(initial)
        with multi_input_work() as calls:
            result = execute(client, multi_body(turn='answer', utterance=ROLE_ANSWER))
        assert result['revision_id'] is None and not harness.service._application.sessions.load('session').revisions
        assert calls['production.neighbors'] == calls['production.epizoo'] == 0
        if content == 'wrong_cell_order':
            assert calls['production.leiden'] == calls['production.evaluation'] == 1
            assert calls['verification.evaluation'] == 0
        else:
            assert calls['owner.clustering'] == 1 and calls['production.leiden'] == calls['production.evaluation'] == 0


def test_ambiguous_role_subject_without_captured_target_remains_plain_clarification(tmp_path):
    utterance = 'Evaluate these datasets somehow.'
    harness = role_continuation_harness(tmp_path, decisions={utterance:
        dict(kind='clarify', reason='ambiguous_subject')})
    with TestClient(create_app(harness.service, uploads=harness.uploads)) as client:
        records = upload_sources(client, harness)
        with multi_input_work() as calls:
            result = execute(client, multi_body([row['resource_id'] for row in records], utterance=utterance))
        assert result['response']['kind'] == 'clarify' and result['revision_id'] is None
        assert_no_owner(calls)
        assert calls['planner'] == calls['compiler'] == 0
        assert harness.service._application.sessions.load('session').interactions[0].prerequisite is None


def test_changed_web_input_set_cannot_retarget_pending_role_answer(tmp_path):
    harness = role_continuation_harness(tmp_path)
    first = ScientificInputSet('first', 'Original exact choice', {})
    second = ScientificInputSet('second', 'Different exact choice', {})
    with TestClient(create_app(harness.service, uploads=harness.uploads, input_sets=(first, second))) as client:
        records = upload_sources(client, harness)
        selected = [row['resource_id'] for row in records]
        initial = multi_body(selected, utterance=OBJECTIVE) | dict(input_set_id=first.input_set_id)
        with multi_input_work() as start:
            origin = assert_role_pending(harness, execute(client, initial))
        assert_no_owner(start)
        changed = multi_body(selected, turn='changed', utterance=ROLE_ANSWER) | dict(input_set_id=second.input_set_id)
        with multi_input_work() as calls:
            result = execute(client, changed)
        assert result['revision_id'] is None and result['response']['status'] == 'failed'
        assert result['response']['error']['code'] == 'INTERACTIVE_INTENT_BINDING_INVALID'
        assert_no_owner(calls)
        assert not harness.service._application.sessions.load('session').revisions
        assert harness.service._application.sessions.load('session').interactions[0] == origin


def test_role_then_exact_string_clarification_uses_original_objective_and_same_sources(tmp_path):
    objective = ('Cluster one of these selected datasets, then evaluate its fixed '
                 'clustering against the other as ground truth.')
    harness = role_continuation_harness(tmp_path, decisions={objective:
        role_request(objective, label=False)})
    with TestClient(create_app(harness.service, uploads=harness.uploads)) as client:
        records = upload_sources(client, harness)
        selected = [row['resource_id'] for row in records]
        with multi_input_work() as initial:
            origin = assert_role_pending(harness, execute(client, multi_body(selected, utterance=objective)))
        assert_no_owner(initial)
        with multi_input_work() as roles:
            unresolved = execute(client, multi_body(turn='roles', utterance=ROLE_ANSWER))
        assert_no_owner(roles)
        assert unresolved['response']['kind'] == 'clarify'
        assert unresolved['response']['clarification']['choices'] == ['label_key']
        middle = harness.service._application.sessions.load('session').interactions[-1]
        assert middle.prerequisite['field'] == 'parameters'
        assert _serialize(middle.admitted['roles']) == ROLES
        with multi_input_work() as completed:
            result = execute(client, multi_body(turn='label', utterance=LABEL_ANSWER))
        assert result['status'] == 'succeeded' and result['revision_id'], result
        assert completed['production.leiden'] == completed['production.evaluation'] == 1
        assert completed['production.neighbors'] == completed['production.epizoo'] == 0
        state = harness.service._application.sessions.load('session')
        assert state.interactions[0] == origin and state.interactions[1] == middle
        assert state.generation == 1 and len(state.revisions) == 1
        final = state.interactions[-1]
        assert _serialize(final.admitted['roles']) == ROLES
        assert _serialize(final.admitted['registered_input']['members']) == _serialize(origin.submission['registered_input']['members'])
        run = harness.service._application.run_store.load(result['run_id'])
        assert run.request.prompt == objective
        assert run.steps[1].resolved_arguments['label_key'] == LABEL
        label_prompt = next(call['value'] for model in harness.models for call in model.calls
                            if call['value'].get('utterance') == LABEL_ANSWER)
        assert label_prompt['dialogue']['pending_prerequisite']['objective'] == objective
        path = f"/api/v1/sessions/session/revisions/{result['revision_id']}/evidence"
        facts = {row['field']: row for row in client.get(path, params={'output_name':'evaluation'}).json()['facts']}
        assert facts['explicit_user_parameters']['value']['parameters']['label_key'] == 'explicit_user_instruction'
        before = sum(len(model.calls) for model in harness.models)
        with multi_input_work() as duplicate:
            assert execute(client, multi_body(turn='label', utterance=LABEL_ANSWER)) == result
        assert not duplicate and before == sum(len(model.calls) for model in harness.models)
