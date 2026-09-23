"""Deterministic text downstream of reviewed facts; no model or science calls."""
from dataclasses import replace
import re

from .turn_decisions import Answer, Clarify, IntentError, resolve_relation
from .response_facts import LABELS, project

UNAVAILABLE = 'I cannot establish that from the current verified state.'
UNSUPPORTED = ('This conversational layer supports analysis state, parameters, execution and provenance. '
               'It does not yet have reviewed evidence for biological or marker-based explanations.')


def _joined(values):
    values = tuple(dict.fromkeys(values))
    return ' and '.join(values) if len(values) <= 2 else ', '.join(values[:-1]) + ', and ' + values[-1]


def _value(value):
    return 'disabled' if value is None else str(value)


def render(facts, *, technical=False):
    """Render only reviewed facts; utterance and raw artifacts are deliberately absent."""
    intent = facts.intent
    roles = {o.role for o in facts.outputs}
    active = 'active' if facts.is_active else 'historical (not active)'
    if intent == 'version':
        text = f'Analysis version {facts.version} is {active}.'
    elif intent in {'matrix', 'selection'}:
        role = 'matrix' if intent == 'matrix' else 'cell selection'
        text = (f'Analysis version {facts.version} is {active}; it contains the accepted {role}.'
                if role in roles else f'Analysis version {facts.version} is {active}; it has no {role}.')
        if role == 'matrix' and role not in roles: text += ' No historical matrix is being treated as active.'
    elif intent == 'thresholds':
        text = ('; '.join(f'{LABELS[k]}: {_value(v)}' for k, v in facts.parameters) + '.'
                if facts.parameters else 'There is no selection in this revision, so no active selection thresholds can be reported.')
    elif intent in {'comparison', 'changes'}:
        if facts.comparison_revision_id is None:
            text = 'This revision has no parent to compare with.'
        else:
            parts = [f'Compared with version {facts.comparison_version}:']
            parts += [f'{LABELS[k]} changed from {_value(a)} to {_value(b)}.' for k, a, b in facts.parameter_changes]
            parts += [f'{role.capitalize()}: {status}.' for role, status in facts.output_changes]
            text = ' '.join(parts)
    elif intent in {'execution', 'reuse'}:
        ran = [w.role for w in facts.work if w.tool_attempts]
        reused = [r.output.role for r in facts.reuse]
        text = ('The originating run recorded tool attempts for ' + _joined(ran) + '.'
                if ran else 'The originating run records no new tool attempts.')
        if reused:
            text += ' It consumed prior accepted ' + _joined(reused) + ' through exact authority-bound references.'
            absent = [r.output.role for r in facts.reuse if r.production_step_absent]
            if absent: text += ' No ' + _joined(absent) + ' production step was included.'
        else: text += ' It records no cross-run prior-output consumption.'
        if facts.retained_outputs:
            text += ' The revision retains prior ' + _joined(o.role for o in facts.retained_outputs) + ' in its explicit output set.'
        if any(w.recovery_events for w in facts.work):
            text += ' The run also records same-run recovery events; its attempt history includes earlier work.'
        text += ' Exact production and owner-reconstruction call totals were not recorded.'
    elif intent == 'provenance':
        if 'matrix' not in roles:
            text = 'This revision has no matrix. '
        else:
            text = 'The matrix uses the exact accepted cell selection and fragments, with its recorded reference. '
        text += ' '.join(f'{e.consumer_role.capitalize()} used the accepted {e.source_role}.' for e in facts.lineage)
        if not facts.lineage: text += 'No supported lineage edges are present.'
    elif intent == 'verification':
        text = ('The outputs have recorded successful verification and exact accepted scientific-authority.v2 bindings. '
                f'The {facts.evidence_files_validated} pinned evidence file(s) match their accepted digests. '
                'This answer checks persisted acceptance, not current scientific validity or the full artifact contents. '
                'Authority-bound reuse requires integrity validation during execution; historical reconstruction counts were not recorded.')
    else:
        text = UNSUPPORTED
    if not facts.is_active and intent not in {'version', 'matrix', 'selection'}:
        text = f'For historical version {facts.version} (not active): ' + text
    if technical:
        text += f' Revision ID: {facts.revision_id}.'
        text += ''.join(f' {o.role}: run={o.run_id}, step={o.step_id}, manifest SHA-256={o.manifest_sha256}, authority SHA-256={o.authority_sha256}.' for o in facts.outputs)
        if facts.reference_sha256: text += f' Reference SHA-256: {facts.reference_sha256}.'
    return text


def clarification_text(clarification):
    if clarification.reason == 'requires_execution':
        return 'This request requires the scientific execution path and its required inputs; it cannot be answered as an existing result.'
    if clarification.reason == 'ambiguous_subject':
        return 'Which result or subject do you mean? Please name it explicitly.'
    if clarification.reason == 'ambiguous_predecessor':
        return 'There are multiple or unfinished discussions. Please identify the discussion to continue.'
    if clarification.reason == 'incompatible_comparison':
        return 'The requested subjects do not have an established compatible comparison.'
    if clarification.reason == 'ambiguous_revision':
        if not clarification.choices: return UNAVAILABLE
        names = {'parent':'parent version', 'previous_active':'previously active version'}
        return 'Do you mean the ' + ' or the '.join(names[c] for c in clarification.choices) + '?'
    if clarification.reason in {'missing_parameter_value', 'ambiguous_parameter'}:
        if not clarification.choices: return 'There is no editable selection threshold in the captured state.'
        if len(clarification.choices) == 1:
            return f'What value should I use for {LABELS[clarification.choices[0]].lower()}?'
        return 'Which threshold would you like to change, and what value should I use?'
    if clarification.reason == 'unsupported_intent': return UNSUPPORTED
    if clarification.reason == 'unavailable_context': return UNAVAILABLE
    return 'I could not establish an exact supported change. Please specify the threshold and numerical value, or an explicit version.'


def answer_outcome(sessions, session_id, request, *, revision_id=None, comparison_id=None):
    from .turns import TurnOutcome
    if not isinstance(request, Answer): raise TypeError('A structured Answer is required.')
    if request.intent == 'unsupported':
        return TurnOutcome('answer', 'unsupported', text=UNSUPPORTED)
    try:
        state = sessions.load(session_id)
        if revision_id is None:
            current = next((r for r in state.revisions if r.revision_id == state.active_revision_id), None)
            if current is None: raise IntentError('unavailable_context')
            relations = {'current': current.revision_id, 'parent': current.parent_revision_id,
                         'previous_active': next((e.from_revision_id for e in reversed(state.navigation)
                             if e.to_revision_id != e.from_revision_id), None)}
            captured = {'relations': relations}
            if request.relation == 'previous':
                candidates = {captured['relations'].get(k) for k in ('parent', 'previous_active')} - {None}
                if not candidates: raise IntentError('unavailable_context')
                if len(candidates) != 1: raise IntentError('ambiguous_revision')
                target = candidates.pop()
            else:
                target = captured['relations'].get(request.relation)
                if target is None: raise IntentError('unavailable_context')
            revision_id = state.active_revision_id if request.intent == 'comparison' else target
            if request.intent == 'comparison': comparison_id = target
        facts = project(sessions, state, request, revision_id=revision_id, comparison_id=comparison_id)
        return TurnOutcome('answer', 'answered', text=render(facts, technical=request.technical), facts=facts)
    except IntentError as exc:
        clarification = Clarify(exc.reason, ('parent', 'previous_active') if exc.reason == 'ambiguous_revision' else ())
        return TurnOutcome('clarify', 'clarification', clarification, clarification_text(clarification))
    except (ValueError, RuntimeError, OSError, KeyError, StopIteration):
        return TurnOutcome('answer', 'unavailable', text=UNAVAILABLE)


def admit_answer(interaction, decision):
    from agent.schemas.orchestration import _serialize
    captured = _serialize(interaction.snapshot)
    relation_text = re.sub(r'\b(previous|earlier|parent) (?:result|revision)\b', r'\1 version', interaction.utterance, flags=re.I)
    target = resolve_relation(decision.relation, captured, relation_text)
    if decision.technical and not re.search(r'\b(?:technical|ids?|hashes|sha-?256)\b', interaction.utterance, re.I):
        raise IntentError('unsupported_intent')
    return dict(kind='answer', intent=decision.intent, relation=decision.relation, technical=decision.technical,
                revision_id=interaction.base_revision_id if decision.intent == 'comparison' else target,
                comparison_id=target if decision.intent == 'comparison' else None)


def admitted_answer(sessions, session_id, admitted):
    request = Answer(admitted['intent'], admitted['relation'], admitted['technical'])
    return answer_outcome(sessions, session_id, request, revision_id=admitted['revision_id'],
                          comparison_id=admitted['comparison_id'])


def summarize(sessions, session_id, turn_id, outcome):
    """Add text only to the high-level interface; low-level run results are unchanged."""
    if outcome.text: return outcome
    if outcome.kind == 'clarify':
        return replace(outcome, text=clarification_text(outcome.clarification))
    if outcome.kind == 'navigate' and outcome.status == 'activated':
        state = sessions.load(session_id)
        target = state.turn(turn_id).revision_id
        version = next(i + 1 for i, r in enumerate(state.revisions) if r.revision_id == target)
        current = 'is active' if state.active_revision_id == target else 'was activated; another version is now active'
        return replace(outcome, text=render_navigation(version, current))
    if outcome.kind != 'execute' or outcome.status not in {'activated', 'stale'}:
        return replace(outcome, text='The turn has not produced a newly activated analysis revision.')
    try:
        state = sessions.load(session_id)
        turn = state.turn(turn_id)
        facts = project(sessions, state, Answer('changes'), revision_id=turn.revision_id)
    except (ValueError, RuntimeError, OSError, KeyError, StopIteration):
        return replace(outcome, text=UNAVAILABLE)
    return replace(outcome, text=render_execution(facts), facts=facts)


def render_execution(facts):
    """One deterministic rendering of a completed execution's exact facts."""
    parts = [f'{LABELS[k]} is {_value(v)}.' for k, _, v in facts.parameter_changes if k in dict(facts.parameters)]
    ran = [w.role for w in facts.work if w.tool_attempts]
    if ran: parts.append('The run completed tool execution for ' + _joined(ran) + '.')
    reused = [r.output.role for r in facts.reuse]
    if reused: parts.append('Prior accepted ' + _joined(reused) + ' were consumed through authority-bound references.')
    retained_only = [o.role for o in facts.retained_outputs if o.role not in reused]
    if retained_only: parts.append('The output set also retains prior ' + _joined(retained_only) + '.')
    if not any(o.role == 'matrix' for o in facts.outputs):
        parts.append('This revision has no matrix; the historical matrix is not part of this output set.')
    parts.append('The new revision is active.' if facts.is_active else 'This result is historical; it is not the active revision.')
    return ' '.join(parts)


def render_navigation(version, current):
    return f'Analysis version {version} {current}. Navigation ran no scientific computation.'
