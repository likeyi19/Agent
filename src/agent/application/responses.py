"""Presentation dispatch; natural-language ownership is scientific_dialogue."""
from dataclasses import replace

CONVERSATION_LANGUAGE = ('Respond in the conversational language of the researcher question and existing conversation; a short value-only reply keeps that language. '
    'Respect an explicit request for another response language. Keep scientific identifiers, '
    'resource labels, function names, parameter names and citation identifiers exact; '
    'Scientific identifiers are not a conversational language preference. A numeric or value-only '
    'clarification reply inherits the pending objective and conversation language. This applies '
    'to every language supported by the selected model; do not default to English because a reply is short.')


def conversation_utterances(state, turn_id):
    """Bound language context to captured user utterances through this turn."""
    rows = []
    for interaction in state.interactions:
        rows.append(interaction.utterance)
        if interaction.turn_id == turn_id:
            break
    return rows[-4:]


def summarize(sessions, session_id, turn_id, outcome, *, model=None):
    """Do not rewrite reviewed answers, saved presentation or deterministic actions."""
    if outcome.presentation is not None or outcome.scientific is not None or outcome.guidance is not None:
        return outcome
    from . import scientific_dialogue as dialogue
    state = sessions.load(session_id)
    interaction = next((i for i in state.interactions if i.turn_id == turn_id), None)
    if interaction is not None and interaction.presentation is not None:
        from .turns import _outcome
        return _outcome(sessions, interaction)
    if outcome.kind == 'answer':
        # The common owner has already attempted generation; never recurse.
        return dialogue.status_outcome(outcome, outcome.error.code) if outcome.error is not None else outcome
    if outcome.kind == 'navigate':
        return replace(outcome, text='')
    if outcome.kind == 'execute' and outcome.status in {'activated', 'stale'}:
        if interaction is not None and (interaction.admitted or {}).get('result_edit'):
            return replace(outcome, text='')
        return dialogue.completion(sessions, session_id, turn_id, outcome, model=model)
    if outcome.kind == 'execute' and outcome.status == 'failed' and outcome.error is None:
        turn = next((t for t in state.turns if t.turn_id == turn_id), None)
        if turn is not None and turn.run_id is not None:
            run = sessions._application.run_store.load(turn.run_id)
            if run.errors:
                from .interactive_schemas import ClientError
                from agent.orchestration.error_policy import public_error_message
                outcome = replace(outcome, error=ClientError(run.errors[0].code, public_error_message(run.errors[0])))
    if interaction is not None and (interaction.submission or {}).get('resume_task_id'):
        return replace(outcome, text='')
    if interaction is not None and any((interaction.submission or {}).get(k) for k in (
            'clarification_choice', 'parameter_overrides', 'candidate_selection', 'rerun_revision_id', 'resume_task_id')):
        # Exact buttons/forms are deterministic transport. Their structured
        # outcome remains visible without a new conversational completion.
        from .preparation_state import choices
        if outcome.kind != 'clarify' or interaction.preparation is not None and choices(interaction):
            return replace(outcome, text='')
    if outcome.error is not None and dialogue.language_failure(outcome.error.code) and outcome.error.code != 'INTERACTIVE_REFERENCE_INVALID':
        return dialogue.status_outcome(outcome, outcome.error.code)
    if interaction is not None and (outcome.kind == 'clarify' or outcome.status == 'failed'):
        return dialogue.explain_outcome(sessions, session_id, interaction, outcome, model)
    return replace(outcome, text='')
