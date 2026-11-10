"""Candidate capture is not consumption; selected uploads stay strict. No science."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from agent.application import annotation_preparation
from agent.application.dialogue_execution import admit, execution_submission, _validate_argument_record
from agent.application.local_resources import LocalResourceAdmission, ResourceAdmissionError
from agent.application.raw_handoff import AnnotationDeclaration, AnnotationUploads
from agent.application.session_state import Interaction
from agent.application.turn_decisions import Execute, InputRole, RequestRoles
from agent.orchestration import build_default_tool_registry
from agent.schemas import AgentPlan, PlanStep
from agent.schemas.orchestration import _serialize


@pytest.fixture
def sources(tmp_path):
    registry = build_default_tool_registry()
    owner = LocalResourceAdmission(tmp_path/'workspace', approved_source_roots=(tmp_path,), registry=registry)
    records = []
    for i in range(3):
        p = tmp_path/f'file{i}'
        p.write_bytes(f'Explicit registration byte fixture {i}'.encode())
        records.append(owner.register(f'file{i}', p, input_type='attachment', label=f'File {i}', attribution='Software fixture'))
    pool = owner.compose_group([r.resource_id for r in records], attachment_candidates=True)
    item = Interaction('new', 'Inspect a previously uploaded file.', None, 0, {},
        submission=dict(execution_inputs={}, registered_input=pool.attribution()))
    app = SimpleNamespace(_workspace=owner._workspace, registry=registry)
    sessions = SimpleNamespace(_application=app, _interaction_session_id='session')
    sessions.load = lambda _: SimpleNamespace(session_id='session', generation=0, active_revision_id=None, interactions=(item,))
    return owner, records, pool, item, sessions


def selected(sources):
    owner, records, pool, item, sessions = sources
    roles = [dict(tool='inspect_scATAC', port='dataset', source='source.1')]
    binding = owner.bind_group(pool, roles)
    submission = dict(registered_input=binding.execution_attribution(), execution_inputs=_serialize(binding.execution_inputs))
    plan = AgentPlan('plan', 'request', 'Inspect selected bytes',
        (PlanStep('inspect', 'inspect_scATAC', dict(path=records[1].source_path)),))
    return binding, submission, plan


def test_generic_candidate_only_admission_explicitly_clears_received_binding(sources):
    owner, records, pool, item, sessions = sources
    value = admit(sessions, item, Execute('inspect_scATAC'), {})
    assert value['registered_input'] is None and not value['inputs']
    current = replace(item, admitted=value)
    assert 'registered_input' not in execution_submission(current)
    assert _serialize(item.submission['registered_input']) == pool.attribution()
    _validate_argument_record(current)


def test_reuse_of_existing_upload_establishes_the_new_consumer_role(sources):
    owner, records, pool, item, sessions = sources
    value = admit(sessions, item, Execute('inspect_scATAC',
        roles=(InputRole('inspect_scATAC', 'dataset', 'source.1'),)), {})
    submission = execution_submission(replace(item, admitted=value))
    assert submission['registered_input']['roles'] == [dict(tool='inspect_scATAC', port='dataset', source='source.1')]
    assert submission['execution_inputs']['input_path'] == records[1].source_path


def test_candidate_metadata_and_bytes_not_consumed_are_not_revalidated(sources, monkeypatch):
    owner, records, pool, item, sessions = sources
    binding, submission, plan = selected(sources)
    original = owner.load
    reads = []
    def only_selected(resource):
        assert resource == records[1].resource_id
        reads.append(resource)
        return original(resource)
    monkeypatch.setattr(owner, 'load', only_selected)
    owner.validate_plan(submission, plan)
    reads.clear()
    owner.validate_group_consumer(submission, plan.steps[0], plan.steps[0].arguments)
    assert reads == [records[1].resource_id]


def test_selected_source_is_checked_again_at_actual_consumption(sources):
    owner, records, pool, item, sessions = sources
    binding, submission, plan = selected(sources)
    owner.validate_plan(submission, plan)
    from pathlib import Path
    Path(records[1].source_path).write_bytes(b'Changed after plan acceptance')
    with pytest.raises(ResourceAdmissionError):
        owner.validate_group_consumer(submission, plan.steps[0], plan.steps[0].arguments)


def test_admission_hashes_newly_selected_members_after_assigning_roles(sources):
    from pathlib import Path
    owner, records, pool, _, _ = sources
    Path(records[1].source_path).write_bytes(b'Changed after candidate capture')
    with pytest.raises(ResourceAdmissionError):
        owner.bind_group(pool, [dict(tool='inspect_scATAC', port='dataset', source='source.1')], verify_source=True)
    assert owner.bind_group(pool, [], verify_source=True).execution_attribution() is None


@pytest.mark.parametrize('variant', ['unused', 'wrong_path', 'duplicate_consumer'])
def test_selected_upload_must_be_consumed_by_its_declared_role(sources, variant):
    owner, records, pool, item, sessions = sources
    binding, submission, plan = selected(sources)
    if variant == 'unused':
        step = PlanStep('other', 'inspect_scATAC', dict(path=records[0].source_path))
        plan = replace(plan, steps=(step,))
    elif variant == 'wrong_path':
        plan = replace(plan, steps=(replace(plan.steps[0], arguments=dict(path='/not/selected')),))
    else:
        plan = replace(plan, steps=plan.steps+(replace(plan.steps[0], step_id='repeat'),))
    with pytest.raises(ResourceAdmissionError, match='exact selected inputs'):
        owner.validate_plan(submission, plan)


def test_candidate_empty_roles_never_pass_selected_group_validator(sources):
    owner, _, pool, _, _ = sources
    _, _, plan = selected(sources)
    assert pool.execution_attribution() is None
    with pytest.raises(ResourceAdmissionError, match='distinct, supported scientific roles'):
        owner.validate_plan(dict(registered_input=pool.attribution(), execution_inputs={}), plan)


@pytest.mark.parametrize('bad', ['checksum', 'role_duplicate', 'role_unknown', 'path_companion'])
def test_invalid_selected_registration_and_roles_fail_closed(sources, bad):
    owner, records, pool, item, sessions = sources
    binding, submission, plan = selected(sources)
    with pytest.raises(ResourceAdmissionError):
        if bad == 'checksum':
            members = _serialize(binding.members)
            members[1]['source_sha256'] = 'f'*64
            owner.validate_binding(replace(binding, members=members))
        elif bad == 'path_companion':
            owner.validate_binding(replace(pool, execution_inputs=dict(input_path=records[0].source_path)))
        else:
            roles = _serialize(binding.roles)
            roles = roles+roles if bad == 'role_duplicate' else [dict(tool='inspect_scATAC', port='not_a_port', source='source.1')]
            owner.bind_group(pool, roles)


def test_reference_only_selection_keeps_exact_reference_checks_without_dataset_guard(sources, monkeypatch):
    owner, records, pool, item, sessions = sources
    declared = AnnotationDeclaration(uploaded=AnnotationUploads(gene_source='source.0', signature_source='source.2'))
    seen = []
    original = LocalResourceAdmission.validate_members
    def checked(self, members, **kwargs):
        seen.extend(m['resource_id'] for m in members)
        return original(self, members, **kwargs)
    monkeypatch.setattr(LocalResourceAdmission, 'validate_members', checked)
    decision = RequestRoles('annotate_scATAC_cell_types', (), annotation=declared)
    value, pending = annotation_preparation.admit(sessions, item, decision, ())
    assert value['registered_input'] is None and not value['inputs'] and pending is None
    assert value['annotation_sources'] == pool.attribution()
    assert seen == [records[0].resource_id, records[2].resource_id]
    assert 'registered_input' not in execution_submission(replace(item, admitted=value))
    from pathlib import Path
    Path(records[2].source_path).write_bytes(b'Changed selected reference')
    with pytest.raises(ResourceAdmissionError):
        annotation_preparation.admit(sessions, item, decision, ())


@pytest.mark.parametrize('variant', ['selected', 'wrong_consumer_path', 'changed_before_consumption'])
def test_real_execution_acceptance_installs_only_the_selected_upload_guard(sources, monkeypatch, variant):
    """Use the real callback, reservation and executor; stop before its tool body."""
    import json
    from pathlib import Path
    from agent.application import ResearchAgentApplication, dialogue_execution
    from agent.application.turn_context import snapshot
    from agent.orchestration.executor import PlanExecutor, _execution_input_guard
    from agent.tools import _native_process
    owner, records, pool, item, _ = sources
    app = ResearchAgentApplication(owner._workspace.root)
    sessions = app.sessions
    sessions._interaction_session_id = 'session'
    state = sessions.create('session')
    item = replace(item, snapshot=snapshot(sessions, state))
    sessions._store._write(replace(state, interactions=(item,)))
    value = admit(sessions, item, Execute('inspect_scATAC',
        roles=(InputRole('inspect_scATAC', 'dataset', 'source.1'),)), {})
    path = records[0].source_path if variant == 'wrong_consumer_path' else records[1].source_path
    value['native_plan'] = AgentPlan('plan', value['request_id'], item.utterance,
        (PlanStep('inspect', 'inspect_scATAC', dict(path=path)),)).to_dict()
    item = replace(item, admitted=value, status='admitted')
    sessions._store._write(replace(state, interactions=(item,)))
    events = []
    validate = LocalResourceAdmission.validate_plan
    consume = LocalResourceAdmission.validate_group_consumer
    def check_plan(self, submission, plan):
        events.append('validate_plan')
        result = validate(self, submission, plan)
        if variant == 'changed_before_consumption':
            Path(records[1].source_path).write_bytes(b'Changed after acceptance check')
        return result
    def check_consumer(self, submission, step, arguments):
        events.append('validate_group_consumer')
        try:
            consume(self, submission, step, arguments)
        except ResourceAdmissionError as exc:
            events.append('consumption_rejected:' + exc.code)
            raise
        assert _execution_input_guard.get() is not None
        assert arguments['path'] == records[1].source_path
        saved = sessions.load('session')
        assert len(saved.turns) == 1 and saved.interactions[-1].status == 'submitted'
        events.append('reserved_and_verified_before_dispatch')
        raise StoppedBeforeScience()
    class StoppedBeforeScience(BaseException): pass
    class Model:
        def complete(self, *, prompt, response_schema):
            assert 'output_selection_schema_version' in json.loads(prompt)
            events.append('output_selection')
            return json.dumps(dict(outputs=[dict(name='inspected cells', step_id='inspect', output_key='n_cells')]))
    monkeypatch.setattr(LocalResourceAdmission, 'validate_plan', check_plan)
    monkeypatch.setattr(LocalResourceAdmission, 'validate_group_consumer', check_consumer)
    monkeypatch.setattr(_native_process, 'call', lambda *a, **k: pytest.fail('Native dispatch prohibited'))
    def run():
        return dialogue_execution.execute(sessions, item, value, Model())
    if variant == 'selected':
        with pytest.raises(StoppedBeforeScience):
            result = run()
            pytest.fail(repr((result, events)))
        assert events == ['output_selection', 'validate_plan', 'validate_group_consumer', 'reserved_and_verified_before_dispatch']
    else:
        result = run()
        assert result.status == 'failed' and not sessions.load('session').revisions
        if variant == 'wrong_consumer_path':
            assert result.error.code == 'MULTI_INPUT_SOURCE_MISMATCH'
            assert events == ['output_selection', 'validate_plan'] and not sessions.load('session').turns
        else:
            assert 'validate_group_consumer' in events
            assert 'consumption_rejected:LOCAL_RESOURCE_INTEGRITY_INVALID' in events
            assert 'reserved_and_verified_before_dispatch' not in events
