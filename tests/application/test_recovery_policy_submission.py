"""Explicit planning policy metadata uses its existing canonical representation."""
from dataclasses import asdict

import pytest

from agent.application import InteractiveAgentApplication, InteractiveBoundaryError
from agent.application.session_state import digest
from agent.orchestration import PlanningModelProfile, PlanningRecoveryPolicy
from agent.orchestration.planning_recovery import SCOPED_PLANNING_RECOVERY_POLICY_VERSION
from agent.providers import PlanningModelFactoryRegistry
from test_interactive_boundary import ScriptedModel
from test_service import _counting_registry, _tiny_h5ad


PROFILE = PlanningModelProfile('policy-test', 'scripted', 'model/policy')
# Captured from the existing pre-UA3.5.4b None-policy configuration envelope.
NONE_POLICY_DIGEST = '30e7ee9655c99086f814876d44d33733135b592be8d18e5e172029ad41763d98'


def explicit_policy(*, codes=None):
    arguments = dict(
        policy_version=SCOPED_PLANNING_RECOVERY_POLICY_VERSION,
        max_transport_retries=0, max_repairs=0, max_profile_failovers=0,
        max_primary_local_recovery_actions=0, max_total_provider_calls=2,
        max_retry_delay_seconds=0,
    )
    if codes is not None:
        arguments['retryable_provider_codes'] = frozenset(codes)
    return PlanningRecoveryPolicy(**arguments)


def application(workspace, policy, *, registry=None):
    models = []

    def factory(profile):
        model = ScriptedModel(profile)
        models.append(model)
        return model

    service = InteractiveAgentApplication(
        workspace, model_profiles=(PROFILE,), default_profile_id=PROFILE.profile_id,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': factory}),
        planning_recovery_policy=policy, registry=registry,
    )
    service.create_session('session')
    return service, models


@pytest.mark.parametrize('policy', [PlanningRecoveryPolicy(), explicit_policy()])
def test_explicit_policy_validates_then_executes_and_duplicate_replays_without_models(tmp_path, policy):
    source = _tiny_h5ad(tmp_path / 'input.h5ad')
    scientific_calls = []
    service, models = application(tmp_path / 'workspace', policy,
                                  registry=_counting_registry(scientific_calls))
    arguments = dict(expected_generation=0, execution_inputs={'input_path': str(source)})
    before = service._application.sessions.load('session').to_dict()
    first = service.validate_submission('session', 'inspect', 'Inspect the supplied matrix.', **arguments)
    repeated = service.validate_submission('session', 'inspect', 'Inspect the supplied matrix.', **arguments)
    assert first == repeated and len(first) == 64
    assert service._application.sessions.load('session').to_dict() == before
    assert not models and not scientific_calls

    view = service.submit_turn('session', 'inspect', 'Inspect the supplied matrix.', **arguments)
    assert view.status == 'succeeded' and view.response.status == 'activated'
    assert scientific_calls == ['inspect_scATAC']
    assert len(models) == 1 and len(models[0].calls) == 4
    assert service._planning['planning_recovery_policy'] is policy
    submission = service._application.sessions.load('session').interactions[-1].submission
    assert submission['configuration_sha256'] == digest(dict(
        profile=asdict(PROFILE), wire_mode=None, recovery_profile=None,
        recovery_policy=policy.to_dict(),
    ))
    assert service.submit_turn('session', 'inspect', 'Inspect the supplied matrix.', **arguments) == view
    assert len(models) == 1 and len(models[0].calls) == 4
    assert scientific_calls == ['inspect_scATAC']


def test_explicit_policy_digest_is_deterministic_from_owner_serialization(tmp_path):
    codes = ('PROVIDER_RATE_LIMITED', 'PROVIDER_CONNECTION_FAILED')
    policies = (explicit_policy(codes=codes), explicit_policy(codes=reversed(codes)))
    submissions, fingerprints = [], []
    for index, policy in enumerate(policies):
        service, models = application(tmp_path / f'workspace-{index}', policy)
        before = service._application.sessions.load('session').to_dict()
        submissions.append(service._submission(PROFILE, 0, {}, None))
        fingerprints.append(service.validate_submission(
            'session', 'inspect', 'Inspect the supplied matrix.', expected_generation=0))
        assert policy.to_dict()['retryable_provider_codes'] == tuple(sorted(codes))
        assert service._application.sessions.load('session').to_dict() == before
        assert not models
    assert submissions[0] == submissions[1]
    assert fingerprints[0] == fingerprints[1]


def test_none_policy_preserves_exact_existing_configuration_digest(tmp_path):
    service, models = application(tmp_path / 'workspace', None)
    submission = service._submission(PROFILE, 0, {}, None)
    assert submission['configuration_sha256'] == NONE_POLICY_DIGEST
    assert service.validate_submission(
        'session', 'inspect', 'Inspect the supplied matrix.', expected_generation=0)
    assert not models


@pytest.mark.parametrize('action', ['validate_submission', 'submit_turn'])
@pytest.mark.parametrize('change,expected', [
    ({'expected_generation': True}, 'INTERACTIVE_INPUT_INVALID'),
    ({'expected_generation': 1}, 'INTERACTIVE_GENERATION_CONFLICT'),
    ({'profile_id': 'unknown'}, 'INTERACTIVE_MODEL_UNAVAILABLE'),
    ({'execution_inputs': {'output_dir': '/private/forbidden'}}, 'INTERACTIVE_INPUT_INVALID'),
])
def test_invalid_explicit_policy_submission_is_rejected_before_model_dispatch(
        tmp_path, action, change, expected):
    scientific_calls = []
    service, models = application(tmp_path / 'workspace', explicit_policy(),
                                  registry=_counting_registry(scientific_calls))
    before = service._application.sessions.load('session').to_dict()
    with pytest.raises(InteractiveBoundaryError) as caught:
        getattr(service, action)('session', 'inspect', 'Inspect the supplied matrix.',
                                **(dict(expected_generation=0) | change))
    assert caught.value.error.code == expected
    assert not models and not scientific_calls
    assert service._application.sessions.load('session').to_dict() == before
