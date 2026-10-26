"""Pure captured-source identity contracts; no files, providers or science."""
from dataclasses import replace

import pytest

from agent.application.dialogue_execution import (
    _role_binding, captured_input_group, validate_source_fields,
)
from agent.application.local_resources import (
    LocalResourceAdmission, RegisteredInput, RegisteredInputGroup, available_input_view, group_role_inputs,
)
from agent.application.session_state import Interaction, SessionError
from agent.orchestration import build_default_tool_registry
from agent.schemas.orchestration import _serialize


def member(n):
    return dict(resource_id='local-' + f'{n:064x}', record_sha256=f'{n:064x}',
                source_sha256=f'{n+10:064x}', source_path=f'/simulated/source-{n}.h5ad',
                input_type='h5ad', label=f'Simulated source {n}')


def submission(binding, **extra):
    return dict(execution_inputs=_serialize(binding.execution_inputs),
                registered_input=binding.attribution(), **extra)


def pending_case(field='roles', *, original=None, current_inputs=None):
    a, b = member(1), member(2)
    old = original or RegisteredInputGroup([a], {}, attachment_candidates=True)
    received = RegisteredInputGroup([b], current_inputs or {}, attachment_candidates=True,
                                    resource_choice=old.resource_choice)
    available = RegisteredInputGroup([b, *old.members], current_inputs or {}, attachment_candidates=True,
                                     resource_choice=old.resource_choice)
    admitted = dict(inputs=_serialize(old.execution_inputs), registered_input=old.attribution(),
                    roles=_serialize(old.roles), required_roles=[dict(tool='pca_embed_cells', port='dataset')])
    if field != 'roles':
        admitted[field + '_sources'] = old.attribution()
        # Raw method declarations hold scalar settings separately from source membership.
        admitted[field + '_arguments'] = {}
    origin = Interaction('origin', 'Continue the original scientific objective.', None, 0,
                         dict(dialogue={}), admitted=admitted, submission=submission(old))
    current = Interaction('reply', 'Use this newly attached file.', None, 0,
        dict(dialogue={}, prerequisite=dict(field=field, origin_turn_id='origin', binding_turn_id='origin')),
        submission=submission(received, available_input=available.attribution()))
    return origin, current, old, available


def captured(interaction, binding):
    snapshot = _serialize(interaction.snapshot)
    snapshot['dialogue']['source_view'] = submission(binding)
    snapshot['dialogue']['source_fields'] = [dict(source=f'source.{i}',
        resource_id=m['resource_id'], source_sha256=m['source_sha256'],
        fields=[f'cohort_{i}'], complete=True) for i, m in enumerate(binding.members)]
    return replace(interaction, snapshot=snapshot)


@pytest.mark.parametrize('field', ['roles', 'annotation', 'epizoo'])
def test_pending_prefix_and_current_attachment_share_one_offered_and_admitted_order(field):
    origin, current, _, received = pending_case(field)
    # Received candidates are current-first; pending meanings retain their prefix.
    assert [m['resource_id'] for m in received.members] == [member(2)['resource_id'], member(1)['resource_id']]
    binding = captured_input_group(current, (origin,))
    assert binding.members == (member(1), member(2))
    frozen = captured(current, binding)
    validate_source_fields(frozen, (origin,))
    assert captured_input_group(frozen, (origin,)) == binding
    viewed = available_input_view(frozen)
    actual = LocalResourceAdmission._submission_binding(viewed.submission)
    assert actual.members == binding.members
    roles = [dict(tool='pca_embed_cells', port='dataset', source='source.1')]
    assert group_role_inputs(actual, roles, build_default_tool_registry())['input_path'] == member(2)['source_path']
    assert 'available_input' not in viewed.submission


def test_neutral_view_keeps_only_current_received_companions():
    original = RegisteredInputGroup([member(1)], dict(n_components=7, pca_random_seed=4),
        roles=[dict(tool='pca_embed_cells', port='dataset', source='source.0')], attachment_candidates=True)
    origin, current, _, _ = pending_case(original=original, current_inputs=dict(n_components=7))
    current = replace(current, submission=submission(RegisteredInputGroup([member(1), member(2)],
        dict(n_components=7), attachment_candidates=True)))
    view = captured_input_group(current, (origin,))
    assert dict(view.execution_inputs) == dict(n_components=7)
    assert not view.roles and view.resource_choice is None
    assert view.members == (member(1), member(2))


def test_neutral_view_does_not_inherit_fixed_task_parameters_roles_or_resource_choice():
    resource = dict(input_set_id='fixed', definition_sha256='a'*64)
    original = RegisteredInputGroup([member(1)], dict(n_components=7),
        roles=[dict(tool='pca_embed_cells', port='dataset', source='source.0')], resource_choice=resource)
    origin, current, _, _ = pending_case(original=original)
    current = replace(current, submission=None)
    view = captured_input_group(current, (origin,))
    assert view.members == original.members and not view.execution_inputs and not view.roles
    assert view.resource_choice is None and view.attachment_candidates
    # Typed continuation still uses the fixed original scientific selection.
    typed = _role_binding(origin, origin, current)
    assert not typed.attachment_candidates and typed.resource_choice == original.resource_choice
    assert typed.execution_inputs == original.execution_inputs


@pytest.mark.parametrize('change', ['record_identity', 'source_identity', 'captured_reorder', 'fixed_selection'])
def test_candidate_extension_cannot_reinterpret_captured_members(change):
    old = RegisteredInputGroup([member(1), member(3)], {}, attachment_candidates=True)
    origin, current, _, _ = pending_case(original=old)
    incoming = [member(2), member(1), member(3)]
    if change == 'record_identity': incoming[1] = dict(incoming[1], record_sha256='f'*64)
    elif change == 'source_identity': incoming[1] = dict(incoming[1], source_sha256='f'*64)
    elif change == 'captured_reorder': incoming = [member(2), member(3), member(1)]
    else:
        fixed = RegisteredInputGroup([member(1)], {})
        origin = replace(origin, submission=submission(fixed), admitted=dict(
            inputs={}, registered_input=fixed.attribution(), roles=[],
            required_roles=[dict(tool='pca_embed_cells', port='dataset')]))
        current = replace(current, submission=submission(RegisteredInputGroup([member(2)], {})))
    if change != 'fixed_selection':
        current = replace(current, submission=submission(RegisteredInputGroup([member(2)], {}, attachment_candidates=True),
            available_input=RegisteredInputGroup(incoming, {}, attachment_candidates=True).attribution()))
    with pytest.raises(ValueError):
        captured_input_group(current, (origin,))


@pytest.mark.parametrize('change', ['member_order', 'resource_identity', 'extra_scalar', 'field_identity'])
def test_captured_view_replay_rejects_forged_members_or_values_without_source_io(monkeypatch, change):
    origin, current, _, _ = pending_case()
    frozen = captured(current, captured_input_group(current, (origin,)))
    monkeypatch.setattr('agent.tools.analysis.categorical.fields',
                        lambda *a, **k: pytest.fail('Captured view replay reopened a scientific source.'))
    monkeypatch.setattr(LocalResourceAdmission, '_verify_records',
                        lambda *a, **k: pytest.fail('Captured view replay rehashed an input.'))
    validate_source_fields(frozen, (origin,))
    snapshot = _serialize(frozen.snapshot)
    view = snapshot['dialogue']['source_view']
    if change == 'member_order': view['registered_input']['members'].reverse()
    elif change == 'resource_identity': view['registered_input']['members'][0]['record_sha256'] = 'f'*64
    elif change == 'extra_scalar': view['execution_inputs']['n_components'] = 99
    else: snapshot['dialogue']['source_fields'][0]['source_sha256'] = 'f'*64
    with pytest.raises(SessionError, match='captured observation'):
        validate_source_fields(replace(frozen, snapshot=snapshot), (origin,))


def single_candidate(*, extra=False, pending=None):
    selected = member(2)
    registered = RegisteredInput(selected['resource_id'], selected['record_sha256'], 'inspect_scATAC',
        dict(input_path=selected['source_path']), composition='h5ad-science.v1')
    candidates = RegisteredInputGroup([selected, member(1)] if extra else [selected], {},
                                      attachment_candidates=True)
    return Interaction('single', 'Use this supplied matrix and its annotations.', None, 0,
        dict(dialogue={}, **({} if pending is None else dict(prerequisite=pending))),
        submission=submission(registered, candidate_input=candidates.attribution()))


@pytest.mark.parametrize('extra', [False, True])
def test_ordinary_single_h5ad_candidate_uses_the_same_group_field_and_role_owner(extra):
    current = single_candidate(extra=extra)
    original = _serialize(current.submission)
    view = captured_input_group(current)
    assert view.members[0] == member(2)
    assert len(view.members) == (2 if extra else 1)
    assert not view.execution_inputs and not view.roles and view.attachment_candidates
    frozen = captured(current, view)
    validate_source_fields(frozen)
    actual = LocalResourceAdmission._submission_binding(available_input_view(frozen).submission)
    assert actual.members == view.members
    assert group_role_inputs(actual,
        [dict(tool='pca_embed_cells', port='dataset', source='source.0')],
        build_default_tool_registry())['input_path'] == member(2)['source_path']
    # The original primary selection remains available for its scientific owner.
    assert _serialize(frozen.submission) == original
    assert frozen.submission['registered_input']['resource_id'] == member(2)['resource_id']


@pytest.mark.parametrize('field', ['roles', 'annotation', 'epizoo'])
def test_pending_prefix_and_single_primary_selection_retain_exact_resource_identity(field):
    origin, _, _, _ = pending_case(field)
    current = single_candidate(extra=True, pending=dict(field=field,
        origin_turn_id='origin', binding_turn_id='origin'))
    view = captured_input_group(current, (origin,))
    assert view.members == (member(1), member(2))
    frozen = captured(current, view)
    validate_source_fields(frozen, (origin,))
    assert frozen.submission['registered_input']['resource_id'] == view.members[1]['resource_id']
    actual = LocalResourceAdmission._submission_binding(available_input_view(frozen).submission)
    assert actual.members[1] == member(2)


@pytest.mark.parametrize('change', ['primary_identity', 'primary_path', 'extra_scalar'])
def test_single_candidate_projection_rejects_substituted_primary_identity_or_inputs(change):
    current = single_candidate(extra=True)
    values = _serialize(current.submission)
    if change == 'primary_identity':
        values['registered_input']['resource_id'] = member(1)['resource_id']
    elif change == 'primary_path':
        values['execution_inputs']['input_path'] = member(1)['source_path']
    else:
        values['execution_inputs']['n_components'] = 3
    with pytest.raises(ValueError):
        captured_input_group(replace(current, submission=values))
