"""Exact guidance selection admission; execution remains in dialogue_execution.

No DAG, prerequisites, readiness solver or scientific authority is constructed.
"""
import json
import re

from agent.orchestration.active_context import ActiveContextItem, ActivePlanningContext
from agent.orchestration.prior_outputs import binding_for_locator, validate_binding
from agent.schemas.orchestration import _serialize, freeze_json_mapping
from .session_state import digest
from .turn_decisions import IntentError, _shape, clauses
from . import scientific_guidance as guidance
from . import scientific_dialogue as dialogue


def candidates(interactions, captured):
    """Follow the captured discussion chain, not completion order or prose."""
    by_id = {i.turn_id:i for i in interactions}
    key = captured['dialogue']['predecessor']
    seen = set()
    while key is not None:
        if key in seen: raise IntentError('ambiguous_predecessor')
        seen.add(key)
        prior = by_id.get(key)
        if prior is None or prior.status != 'answered': return ()
        refs = guidance._references(prior)
        if refs:
            # A rationale follow-up may retain only one option. Its origin still
            # owns the original stable ordinal set.
            origin = by_id.get(refs[0]['origin_turn_id'])
            return guidance._references(origin)
        key = prior.admitted.get('predecessor') if prior.admitted else None
    return ()


def public_candidates(state, captured):
    refs = candidates(state.interactions, captured)
    by_id = {i.turn_id:i for i in state.interactions}
    return [dict(candidate=r['candidate_id'], option=n+1, capability=r['capability'],
        objective=by_id[r['objective_turn_id']].utterance,
        current=r['base_revision_id'] == state.active_revision_id,
        subjects=[t['subject'] for t in r['targets']]) for n,r in enumerate(refs)]


def command_evidence(utterance, evidence):
    # Reuse M15's rejection of conditional/negated command clauses. Splitting
    # command boundaries never selects a capability or interprets a referent.
    clauses(utterance)
    parts = re.split(r'(?<=[.!?])\s+|,\s*(?:but\s+)?', utterance.strip(), flags=re.I)
    if evidence not in parts or parts.count(evidence) != 1:
        raise IntentError('unsupported_intent')
    from .dialogue_execution import is_execution_command
    if not (is_execution_command(evidence) or re.match(r'^(?:please\s+)?(?:do|execute|compare)\b', evidence, re.I)):
        raise IntentError('unsupported_intent')


def validate_selection(state, interaction, preceding):
    """Validate durable selection metadata on the ordinary session load path."""
    a = _serialize(interaction.admitted)
    _shape(a, ('kind','operation','tool','revision_id','request_id','inputs',
               'selected_candidate','command_evidence','catalog_sha256'))
    r = a['selected_candidate']
    if (a['kind'] != 'execute' or a['operation'] != 'plan'
            or r not in [_serialize(v) for v in candidates(preceding, interaction.snapshot)]
            or r['base_revision_id'] != interaction.base_revision_id
            or a['revision_id'] != interaction.base_revision_id or a['tool'] != r['capability']):
        raise ValueError('Candidate selection context changed.')
    origin = next(i for i in preceding if i.turn_id == r['origin_turn_id'])
    if a['catalog_sha256'] != origin.admitted['catalog_sha256']:
        raise ValueError('Selected catalog changed.')
    if a['request_id'] != 'turn-' + digest(dict(session=state.session_id, turn=interaction.turn_id)):
        raise ValueError('Selection request identity changed.')
    command_evidence(interaction.utterance, a['command_evidence'])


def _check(sessions, interaction, selected, catalog_sha256):
    state = sessions.load(sessions._interaction_session_id)
    if state.active_revision_id != interaction.base_revision_id or state.generation != interaction.base_generation:
        raise IntentError('ambiguous_revision')
    if selected['base_revision_id'] != interaction.base_revision_id:
        raise IntentError('ambiguous_revision')
    registry = sessions._application.registry
    if selected['capability'] not in registry.names() or digest(guidance.catalog(registry)) != catalog_sha256:
        raise IntentError('unavailable_context')
    for target in selected['targets']:
        view = dialogue._load(sessions, state.session_id, target)
        if target['subject'] is not None and target['subject'] not in dialogue._subjects(view)[0]:
            raise IntentError('ambiguous_subject')
    return state


def _bindings(state, app, selected):
    for target in selected['targets']:
        revision = next(r for r in state.revisions if r.revision_id == target['revision_id'])
        locator = next(o for o in revision.outputs if o.name == target['output_name'])
        yield target, binding_for_locator(locator, app.run_store, app.registry)


def execution_context(sessions, interaction, admitted):
    """Readable discussion evidence must independently pass M15.3 authority."""
    selected = admitted['selected_candidate']
    state = _check(sessions, interaction, selected, admitted['catalog_sha256'])
    app = sessions._application
    items, bindings = [], set()
    for _, binding in _bindings(state, app, selected):
        validate_binding(binding, store=app.run_store, registry=app.registry, integrity=True)
        if binding not in bindings:
            items.append(ActiveContextItem(f'ctx.{len(items)}', binding)); bindings.add(binding)
    return ActivePlanningContext(state.session_id, interaction.base_revision_id,
                                 interaction.base_generation, tuple(items))


def admit(sessions, interaction, decision, inputs):
    command_evidence(interaction.utterance, decision.evidence)
    state = sessions.load(sessions._interaction_session_id)
    preceding = state.interactions[:next(n for n,i in enumerate(state.interactions) if i.turn_id == interaction.turn_id)]
    matches = [r for r in candidates(preceding, interaction.snapshot) if r['candidate_id'] == decision.candidate]
    if len(matches) != 1: raise IntentError('ambiguous_predecessor')
    selected = _serialize(matches[0])
    origin = next(i for i in preceding if i.turn_id == selected['origin_turn_id'])
    admitted = dict(kind='execute', operation='plan', tool=selected['capability'],
        revision_id=interaction.base_revision_id,
        request_id='turn-' + digest(dict(session=state.session_id, turn=interaction.turn_id)),
        inputs={} if inputs is None else _serialize(freeze_json_mapping(inputs, 'execution_inputs')),
        selected_candidate=selected, command_evidence=decision.evidence,
        catalog_sha256=origin.admitted['catalog_sha256'])
    _check(sessions, interaction, selected, admitted['catalog_sha256'])
    try: execution_context(sessions, interaction, admitted)
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        raise IntentError('unavailable_context') from exc
    return admitted


def planning_intent(sessions, interaction, admitted, context):
    state = sessions.load(sessions._interaction_session_id)
    r = admitted['selected_candidate']
    objective = next(i.utterance for i in state.interactions if i.turn_id == r['objective_turn_id'])
    subjects = [dict(context_handle=next(i.handle for i in context.items if i.binding == binding),
                     output=target['output_name'], subject=target['subject'])
                for target, binding in _bindings(state, sessions._application, r)]
    return json.dumps(dict(objective=objective, selected_capability=r['capability'],
        captured_subjects=subjects,
        current_user_request=interaction.utterance,
        instructions='Plan the explicitly selected objective using normal Registry contracts and offered inputs/context. '
        'Candidate identity is not a plan or execution authority. Preserve captured subjects; apply compatible current-turn '
        'refinements without replacing captured identity. Determine required steps through normal Registry semantics; '
        'do not invent missing scientific parameter values or unoffered sources. '
        'If the objective/refinement cannot be satisfied, return unsupported.'), ensure_ascii=False)
