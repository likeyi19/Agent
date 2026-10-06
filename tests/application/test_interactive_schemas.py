"""Client projections are closed, immutable, bounded, and JSON-safe."""
from dataclasses import FrozenInstanceError
import json

import pytest

from agent.application.interactive_schemas import (
    ArtifactHandle, ClientError, ModelChoice, PresentedResponse, RevisionView,
    SessionView, StepView, TurnView,
)


def test_exact_presentation_roundtrip_freezes_and_copies_structured_display():
    clarification = dict(reason='missing_parameter_value', choices=['min_tss_enrichment'],
                         value_required=True)
    shown = PresentedResponse('clarify', 'clarification', 'Which threshold?',
                              clarification=clarification)
    clarification['choices'].append('untrusted')
    serialized = shown.to_dict()
    assert set(serialized) == {'version', 'kind', 'status', 'text', 'clarification',
                               'scientific', 'guidance', 'error'}
    assert serialized['clarification']['choices'] == ['min_tss_enrichment']
    assert PresentedResponse.from_dict(json.loads(json.dumps(serialized))) == shown
    serialized['clarification']['choices'].append('client mutation')
    assert shown.clarification['choices'] == ('min_tss_enrichment',)
    with pytest.raises(TypeError):
        shown.clarification['reason'] = 'other'
    with pytest.raises(FrozenInstanceError):
        shown.text = 'changed'


def test_scientific_presentation_retains_metadata_without_duplicate_claims():
    shown = PresentedResponse('answer', 'answered', 'Accepted evidence explanation.',
        scientific=dict(support='supported', limitations=['Bounded evidence.'],
                        evidence_scope='accepted_persisted_summary'))
    assert PresentedResponse.from_dict(shown.to_dict()) == shown
    assert shown.to_dict()['scientific'] == {
        'support': 'supported', 'limitations': ['Bounded evidence.'],
        'evidence_scope': 'accepted_persisted_summary',
    }
    with pytest.raises(ValueError, match='fields'):
        PresentedResponse('answer', 'answered', 'text', scientific={
            **shown.to_dict()['scientific'], 'claims': [{'value': 100}],
        })


def test_guidance_preserves_candidate_reference_and_original_option():
    candidate = dict(candidate_id='c' * 64, origin_turn_id='guidance', option=3,
                     capability='cluster_cells', text='Conditional explanation.',
                     support='insufficient_evidence', limitations=['Readiness is unassessed.'])
    shown = PresentedResponse('answer', 'answered', 'Advice.',
        guidance=dict(candidates=[candidate], limitations=['No execution authorization.']))
    candidate['text'] = 'mutated'
    assert shown.guidance['candidates'][0]['text'] == 'Conditional explanation.'
    assert shown.to_dict()['guidance']['candidates'][0]['option'] == 3
    assert PresentedResponse.from_dict(shown.to_dict()) == shown


@pytest.mark.parametrize('extra', ['path', 'authority', 'provider_payload', 'facts'])
def test_presentation_storage_shape_rejects_unknown_fields(extra):
    data = PresentedResponse('execute', 'activated', 'Finished.').to_dict()
    data[extra] = 'unreviewed'
    with pytest.raises(ValueError, match='fields'):
        PresentedResponse.from_dict(data)


@pytest.mark.parametrize('change', [
    {'version': True}, {'version': 2}, {'kind': 'raw_provider_response'},
    {'text': 'a' * 131_073}, {'status': ''},
    {'scientific': {'support': 'guessed', 'limitations': [], 'evidence_scope': 'summary'}},
    {'scientific': {'support': 'supported', 'limitations': ['a' * 4_097],
                    'evidence_scope': 'summary'}},
    {'clarification': {'reason': 'required', 'choices': [], 'value_required': True}},
])
def test_invalid_presentation_fails_closed(change):
    data = dict(kind='answer', status='answered', text='text') | change
    with pytest.raises((TypeError, ValueError)):
        PresentedResponse(**data)


def test_utf8_presentation_byte_limit_and_exact_required_null_keys():
    with pytest.raises(ValueError, match='byte limit'):
        PresentedResponse('answer', 'answered', '字' * 100_000)
    data = PresentedResponse('answer', 'answered', '').to_dict()
    del data['scientific']
    with pytest.raises(ValueError, match='fields'):
        PresentedResponse.from_dict(data)


def test_client_error_shape_is_not_an_internal_exception():
    error = ClientError('STALE_GENERATION', 'The observed generation is stale.')
    shown = PresentedResponse('answer', 'unavailable', error.message, error=error)
    assert PresentedResponse.from_dict(shown.to_dict()) == shown
    with pytest.raises(ValueError, match='fields'):
        ClientError.from_dict(error.to_dict() | {'stack_trace': 'internal'})


def test_client_views_serialize_only_reviewed_fields_and_keep_interruption():
    response = PresentedResponse('execute', 'activated', 'Finished.')
    turn = TurnView('session', 'turn', 'Run analysis.', 0, profile_id='offered',
                    run_id='request:run', revision_id='revision', status='interrupted',
                    response=response, steps=(StepView('step', 'inspect_scATAC', 'RUNNING', 1),),
                    state_revision=4)
    revision = RevisionView('revision', None, 'turn', True, ('result',))
    view = SessionView('session', 1, 'revision', (revision,), (turn,))
    serialized = json.loads(json.dumps(view.to_dict(), allow_nan=False))
    assert serialized['turns'][0]['status'] == 'interrupted'
    assert serialized['turns'][0]['steps'] == [
        dict(step_id='step', tool_name='inspect_scATAC', status='RUNNING', attempts=1),
    ]
    assert serialized['revisions'][0]['outputs'] == ['result']
    assert set(serialized['turns'][0]) == {
        'session_id', 'turn_id', 'utterance', 'base_generation', 'profile_id',
        'run_id', 'revision_id', 'status', 'response', 'error', 'steps', 'state_revision',
        'base_revision_id',
    }
    assert 'liveness' not in json.dumps(serialized) and 'percentage' not in json.dumps(serialized)


def test_legacy_turn_view_has_explicit_unavailable_display_without_invented_utterance():
    turn = TurnView('session', 'legacy', '', 0,
                    error=ClientError('PRESENTATION_UNAVAILABLE', 'No stored display is available.'))
    assert turn.utterance == '' and turn.response is None
    assert turn.to_dict()['error']['code'] == 'PRESENTATION_UNAVAILABLE'


@pytest.mark.parametrize('build', [
    lambda: ModelChoice('p', 'Display', 1),
    lambda: ModelChoice('p\ninternal', 'Display', False),
    lambda: StepView('s', 'inspect_scATAC', 'INTERRUPTED', 1),
    lambda: StepView('s', 'inspect_scATAC', 'RUNNING', True),
    lambda: SessionView('session', True, None),
    lambda: SessionView('session', 0, None, turns=(TurnView('other', 't', '', 0),)),
    lambda: SessionView('session', 0, None, revisions=(RevisionView('r', None, 't', True),)),
    lambda: RevisionView('r', None, 't', False, ('output', 'output')),
    lambda: TurnView('session', 't', '', 0, steps=(StepView('s', 'tool', 'PENDING', 0),) * 2),
    lambda: SessionView('session', 0, None, turns=(TurnView('session', 't', '', 0),) * 1001),
])
def test_invalid_client_view_fails_closed(build):
    with pytest.raises((ValueError, TypeError)):
        build()


def test_model_and_artifact_metadata_do_not_carry_server_configuration_or_paths():
    choice = ModelChoice('operator-admitted', 'Scientific model', True, 'openai', 'model')
    artifact = ArtifactHandle('a' * 64, 'report', 'b' * 64, 'revision', 'turn')
    assert set(choice.to_dict()) == {
        'profile_id', 'display_label', 'is_default', 'provider_id', 'model_id',
    }
    assert set(artifact.to_dict()) == {'handle', 'artifact_type', 'sha256', 'revision_id', 'turn_id'}
    assert 'path' not in artifact.to_dict() and 'endpoint' not in choice.to_dict()
    with pytest.raises(ValueError, match='digest'):
        ArtifactHandle('handle', 'report', 'A' * 64, 'revision', 'turn')
