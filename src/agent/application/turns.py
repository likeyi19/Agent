"""Small interaction adapter above the existing Planner, sessions and executor."""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
import re
import logging

from agent.schemas.orchestration import _serialize
from agent.orchestration.planning_model import PlanningModelError
from agent.orchestration.error_policy import BUILTIN_ERROR_CODES, safe_message_for
from .session_state import Interaction, SessionConflictError, digest
from .turn_context import snapshot, public_context
from agent.orchestration.planning_diagnostics import log_operation_event
from .turn_decisions import (Execute, ExecuteCandidate, Navigate, Clarify, Answer, SpeciesAnswer, ParameterAnswer, RequestRoles, RoleAnswer, RequestResources, ResourceAnswer, IntentError, interpret, clauses,
                             resolve_relation)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TurnOutcome:
    kind: str
    status: str
    clarification: Clarify | None = None
    text: str = ""
    scientific: object = None
    guidance: object = None
    error: object = None
    presentation: object = None
    artifacts: tuple = ()


def _update(sessions, turn_id, **changes):
    def change(state):
        return replace(state, interactions=tuple(replace(i, **changes) if i.turn_id == turn_id else i
                                                  for i in state.interactions))
    return sessions._store._update(sessions_id(sessions), change)


def sessions_id(sessions):
    return sessions._interaction_session_id


def _failed_admission(sessions, interaction, decision, *, preserve_pending=False):
    changes = dict(status='failed')
    reference = interaction.snapshot.get('prerequisite')
    if reference and (preserve_pending or isinstance(decision, (Clarify, SpeciesAnswer, ParameterAnswer, RoleAnswer, ResourceAnswer))):
        # Failed binding is not authority to retry. A later explicit reply can
        # still address the same pending objective and last valid association.
        changes['prerequisite'] = {k: _serialize(v) for k, v in reference.items()
                                   if k != 'predecessor_turn_id'}
    _update(sessions, interaction.turn_id, **changes)


def _terminal(sessions, interaction, clarification):
    captured = _serialize(interaction.snapshot)
    choices = clarification.choices
    prerequisite = None
    species_reason = clarification.reason in {'missing_species', 'ambiguous_species', 'unsupported_species', 'conflicting_species'}
    parameter_reason = clarification.reason in {'conflicting_scientific_parameter', 'invalid_parameter_value', 'ungrounded_operand'}
    reference = captured.get('prerequisite')
    if reference is not None and reference['field'] in {'parameters', 'roles', 'resources', 'fastq', 'bam', 'fragments', 'annotation', 'epizoo'}:
        prerequisite = {k: _serialize(v) for k, v in reference.items() if k != 'predecessor_turn_id'}
        if reference['field'] == 'parameters':
            choices = tuple(reference['missing'])
        elif reference['field'] in {'fastq', 'bam', 'fragments', 'annotation', 'epizoo'}:
            from .raw_handoff import public
            choices = tuple(public(interaction, sessions.load(sessions_id(sessions)).interactions)['missing'])
        elif reference['field'] == 'resources':
            from .resource_continuation import public
            choices = tuple(c['label'] for c in public(interaction, sessions.load(sessions_id(sessions)).interactions)['choices'])
        else:
            from .dialogue_execution import prerequisite_public
            pending = prerequisite_public(interaction, sessions.load(sessions_id(sessions)).interactions)
            choices = tuple(r['tool'] + '.' + r['port'] for r in pending['missing_roles'])
    elif (species_reason or parameter_reason) and reference is not None:
        prerequisite = {k: reference[k] for k in ('origin_turn_id', 'field', 'resource_context_sha256')}
        choices = ('human', 'mouse') if species_reason else ('Leiden resolution',)
    elif species_reason:
        if clarification.reason == 'missing_species':
            # Only compiler diagnosis creates a new durable missing declaration.
            clarification = Clarify('invalid_prerequisite')
        else:
            # A fresh semantic refusal owns no pending request or execution.
            choices = ('human', 'mouse')
    if clarification.reason == 'ambiguous_revision':
        choices = tuple(k for k in ('parent', 'previous_active') if k in captured['relations'])
    clarification = replace(clarification, choices=choices,
                            value_required=clarification.reason in {'missing_parameter_value', 'missing_species', 'ambiguous_species',
                                                                   'conflicting_scientific_parameter', 'invalid_parameter_value', 'ungrounded_operand', 'missing_input_roles', 'missing_resources', 'missing_fastq_declarations', 'missing_bam_declarations', 'missing_fragments_declarations', 'missing_annotation_declarations', 'missing_epizoo_declarations'})
    admitted = dict(kind='clarify', **asdict(clarification))
    def change(state):
        # A clarification is an interaction, not an execution checkpoint.
        # Existing navigation-only SessionTurn records remain readable history.
        return replace(state, interactions=tuple(
            replace(i, status='clarification', admitted=admitted, prerequisite=prerequisite) if i.turn_id == interaction.turn_id else i
            for i in state.interactions))
    sessions._store._update(sessions_id(sessions), change)
    return TurnOutcome('clarify', 'clarification', clarification)


def _outcome(sessions, interaction, answerer=None):
    if interaction.presentation is not None:
        from .interactive_schemas import PresentedResponse
        shown = PresentedResponse.from_dict(_serialize(interaction.presentation))
        clarification = None if shown.clarification is None else Clarify(**shown.clarification)
        return TurnOutcome(shown.kind, shown.status, clarification, shown.text,
                           error=shown.error, presentation=shown)
    admitted = {} if interaction.admitted is None else _serialize(interaction.admitted)
    if (interaction.submission is not None and admitted.get('kind') == 'answer'):
        # Interactive retries never regenerate a missing historical response.
        return TurnOutcome('answer', 'unavailable', text='Response unavailable: HISTORICAL_PRESENTATION_MISSING')
    if admitted.get('kind') == 'answer':
        if admitted.get('intent') == 'guidance':
            from .scientific_guidance import answer
            if interaction.status == 'failed':
                return TurnOutcome('answer', 'unavailable', text='Response unavailable: PREVIOUS_RESPONSE_FAILED')
            return answer(sessions, sessions_id(sessions), interaction, answerer)
        if admitted.get('intent') == 'scientific':
            from .scientific_dialogue import answer
            if interaction.status == 'failed':
                return TurnOutcome('answer', 'unavailable', text='Response unavailable: PREVIOUS_RESPONSE_FAILED')
            result = answer(sessions, sessions_id(sessions), interaction, answerer)
            if interaction.status == 'admitted':
                _update(sessions, interaction.turn_id, status='answered' if result.status == 'answered' else 'failed')
            return result
        return TurnOutcome('answer', 'unavailable', text='Response unavailable: HISTORICAL_PRESENTATION_MISSING')
    if interaction.status == 'clarification':
        if admitted.get('kind') == 'request_resources':
            from .resource_continuation import offers
            from .input_sets import ScientificInputSet
            choices = tuple(c['label'] for c in offers(interaction.submission,
                tuple(ScientificInputSet(**_serialize(r)) for r in admitted['resource_catalog'])))
            return TurnOutcome('clarify', 'clarification', Clarify('missing_resources', choices, True))
        if admitted.get('kind') == 'request_roles':
            assigned = {(r['tool'], r['port']) for r in admitted['roles']}
            choices = tuple(r['tool'] + '.' + r['port'] for r in admitted['required_roles']
                            if (r['tool'], r['port']) not in assigned)
            return TurnOutcome('clarify', 'clarification', Clarify('missing_input_roles', choices, True))
        if admitted.get('kind') == 'execute' and interaction.prerequisite is None:
            if 'parameter_scope' in admitted:
                return TurnOutcome('clarify', 'clarification', Clarify('ambiguous_parameter'))
            from .dialogue_execution import parameter_scope_choices
            state = sessions.load(sessions_id(sessions))
            run = sessions._application.run_store.load(admitted['request_id'] + ':run')
            if not any(error.code == 'AMBIGUOUS_PARAMETER_SCOPE' for error in run.errors):
                raise IntentError('invalid_prerequisite')
            return TurnOutcome('clarify', 'clarification', Clarify('ambiguous_parameter',
                parameter_scope_choices(admitted, state.interactions)))
        if interaction.prerequisite is not None and admitted.get('kind') == 'execute':
            if interaction.prerequisite['field'] == 'resources':
                from .resource_continuation import offers
                from .input_sets import ScientificInputSet
                choices = tuple(c['label'] for c in offers(interaction.submission,
                    tuple(ScientificInputSet(**_serialize(r)) for r in admitted['resource_catalog'])))
                return TurnOutcome('clarify', 'clarification', Clarify('missing_resources', choices, True))
            if interaction.prerequisite['field'] == 'parameters':
                return TurnOutcome('clarify', 'clarification', Clarify('missing_parameter_value',
                    tuple(interaction.prerequisite['missing']), True))
            return TurnOutcome('clarify', 'clarification', Clarify('missing_species', ('human', 'mouse'), True))
        return TurnOutcome('clarify', 'clarification', Clarify(admitted['reason'],
            tuple(admitted.get('choices', ())), admitted.get('value_required', False)))
    state = sessions.load(sessions_id(sessions))
    turn = next((t for t in state.turns if t.turn_id == interaction.turn_id), None)
    if turn is not None and turn.request_id is not None:
        state = sessions.recover(state.session_id, interaction.turn_id)
        return TurnOutcome('execute', state.turn(interaction.turn_id).status)
    if turn is not None and turn.request_id is None and turn.status == 'activated':
        return TurnOutcome('navigate', 'activated')
    return TurnOutcome(admitted.get('kind', 'pending'), interaction.status)


NAVIGATION = re.compile(r'(?:use|go back to|switch back to|switch to) (?:the )?(?:parent|previous|earlier|previously active) version|go back one version|(?:use|switch back to) the version i was using before', re.I)


def admit(sessions, interaction, decision):
    captured = _serialize(interaction.snapshot)
    parts = clauses(interaction.utterance)
    rid = resolve_relation(decision.relation, captured, interaction.utterance)
    if not parts or not all(NAVIGATION.fullmatch(p) for p in parts):
        raise IntentError('unsupported_intent')
    return dict(kind='navigate', revision_id=rid)


class _AdmittedPlanner:
    """Delegate unchanged planning/recovery, then check effects before execution."""
    def __init__(self, delegate, accept, prepare=lambda plan: plan):
        self.delegate, self.accept, self.prepare = delegate, accept, prepare

    def plan(self, request, registry):
        plan = self.prepare(self.delegate.plan(request, registry))
        self.accept(request, plan)
        return plan

    def __getattr__(self, name):
        if name not in ('plan_with_recovery', 'plan_with_diagnostics'): raise AttributeError(name)
        method = getattr(self.delegate, name)  # Preserve absence for legacy planners.
        def delegated(request, registry, **kwargs):
            result = method(request, registry, **kwargs)
            if name == 'plan_with_recovery':
                plan = self.prepare(result.attempt.plan)
                result = replace(result, attempt=replace(result.attempt, plan=plan))
            else:
                plan = self.prepare(result.plan)
                result = replace(result, plan=plan)
            self.accept(request, plan)
            return result
        return delegated


def respond(sessions, session_id, turn_id, utterance, *, interpreter, expected_generation=None,
            answerer=None, predecessor_turn_id=None, execution_inputs=None, submission=None,
            epizoo_resources=(), input_sets=(), reference_resources=None):
    # AnalysisSessions.respond supplies a per-call facade over the shared store.
    sessions._interaction_session_id = session_id
    answerer = interpreter if answerer is None else answerer
    if type(utterance) is not str or (not utterance.strip() and not (submission or {}).get('rerun_revision_id') and not (submission or {}).get('parameter_overrides') and not (submission or {}).get('clarification_choice')) or len(utterance) > 4096:
        raise ValueError('A bounded nonempty utterance is required.')
    try:
        state = sessions.load(session_id)
    except (ValueError, RuntimeError, OSError):
        return TurnOutcome('answer', 'unavailable', text='Response unavailable: EVIDENCE_UNAVAILABLE')
    existing = next((i for i in state.interactions if i.turn_id == turn_id), None)
    if existing is not None:
        if existing.utterance != utterance: raise SessionConflictError('Interaction identity changed.')
        if _serialize(existing.submission) != _serialize(submission):
            raise SessionConflictError('Interaction submission changed.')
        if predecessor_turn_id is not None and existing.snapshot.get('dialogue', {}).get('predecessor') != predecessor_turn_id:
            raise SessionConflictError('Interaction predecessor changed.')
        if (execution_inputs is not None and existing.admitted is not None
                and (existing.admitted.get('operation') == 'plan' or 'parameter_continuation' in existing.admitted)):
            fingerprint = existing.admitted.get('parameter_received_inputs_sha256',
                existing.admitted.get('arguments_received_inputs_sha256', existing.admitted.get('argument_received_inputs_sha256')))
            if existing.submission is None and fingerprint is not None:
                matches = digest(_serialize(execution_inputs)) == fingerprint
            else:
                expected_inputs = ((existing.submission or {}).get('execution_inputs', {})
                    if any(k in existing.admitted for k in ('continuation', 'argument', 'arguments', 'roles'))
                    else existing.admitted['inputs'])
                matches = _serialize(execution_inputs) == _serialize(expected_inputs)
            if not matches:
                raise SessionConflictError('Execution inputs changed on retry.')
        return _outcome(sessions, existing, answerer)
    if any(t.turn_id == turn_id for t in state.turns): raise SessionConflictError('Turn identity already used.')
    generation = state.generation if expected_generation is None else expected_generation
    if generation != state.generation: raise SessionConflictError('Session generation changed.')
    try:
        tool_names = {}
        captured = snapshot(sessions, state, tool_names=tool_names)
        from .scientific_dialogue import capture as dialogue_capture
        captured['dialogue'] = dialogue_capture(sessions, state, captured, predecessor_turn_id, tool_names=tool_names)
        from .dialogue_execution import capture_prerequisite
        pending = capture_prerequisite(state.interactions, state.active_revision_id, generation,
                                      submission, predecessor_turn_id, execution_inputs=execution_inputs,
                                      canonical_sources=True)
        if pending is not None:
            captured['prerequisite'] = pending
        from .dialogue_execution import captured_input_group
        from .local_resources import source_field_choices
        selected = captured_input_group(Interaction(turn_id, utterance, state.active_revision_id,
            generation, captured, submission=submission), state.interactions)
        if selected is not None:
            captured['dialogue']['source_view'] = dict(
                execution_inputs=_serialize(selected.execution_inputs), registered_input=selected.attribution())
            if selected.members:
                captured['dialogue']['source_fields'] = source_field_choices(selected)
    except SessionConflictError:
        raise
    except (ValueError, RuntimeError, OSError, KeyError):
        return TurnOutcome('answer', 'unavailable', text='Response unavailable: EVIDENCE_UNAVAILABLE')
    interaction = Interaction(turn_id, utterance, state.active_revision_id, generation, captured,
                              submission=submission)
    def capture(current):
        if (current.generation != generation or current.interactions != state.interactions
                or current.revisions != state.revisions or current.turns != state.turns):
            raise SessionConflictError('Interaction capture raced with another turn.')
        return replace(current, interactions=current.interactions+(interaction,))
    sessions._store._update(session_id, capture)
    if (submission or {}).get('rerun_revision_id'):
        from .structured_parameters import prepare_result_edit, ParameterEditUnavailable
        from .dialogue_execution import execute as execute_request
        try:
            admitted = prepare_result_edit(sessions, interaction)
            _update(sessions, turn_id, admitted=admitted, status='admitted')
            return execute_request(sessions, interaction, admitted, interpreter)
        except ParameterEditUnavailable as exc:
            _update(sessions, turn_id, status='failed')
            return TurnOutcome('execute', 'failed', text=str(exc))
    decision = None
    diagnostic_context = (session_id, turn_id)
    diagnostic_stage = 'preparation'
    try:
        from .epizoo_handoff import captured_models
        input_sets = tuple(input_sets) + captured_models(sessions, state)
        if reference_resources is not None:
            from .shared_resources import discover
            input_sets = tuple(input_sets) + discover(reference_resources, session_id)
        visible = public_context(captured)
        context, _ = sessions.planning_context_for_interaction(session_id, turn_id, integrity=False)
        from .scientific_dialogue import public as dialogue_public
        visible['dialogue'] = dialogue_public(captured, state, sessions._application.registry,
                                              sessions=sessions, utterance=utterance,
                                              execution_inputs=execution_inputs, context=context)
        from .dialogue_execution import prerequisite_public
        pending_public = prerequisite_public(interaction, state.interactions)
        if pending_public is not None and reference_resources is not None:
            from .reference_supplementation import pending_choices
            pending_choices(pending_public, interaction, state, reference_resources)
        if pending_public is not None:
            visible['dialogue']['pending_prerequisite'] = pending_public
        from .resource_continuation import offers, source_kind, catalog
        catalog(input_sets)
        visible['dialogue']['resource_choices'] = offers(submission, input_sets)
        visible['dialogue']['resource_source'] = source_kind(submission)
        visible['dialogue']['reference_supplementation'] = (reference_resources is not None
            and source_kind(submission) == 'h5ad')
        from .dialogue_execution import scientific_parameters, scientific_capabilities
        source_fields = captured['dialogue'].get('source_fields', ())
        visible['dialogue']['source_fields'] = [dict(source=row['source'], fields=list(row['fields']),
                                                   complete=row['complete']) for row in source_fields]
        visible['dialogue']['scientific_parameters'] = scientific_parameters(sessions._application.registry)
        visible['dialogue']['capabilities'] = scientific_capabilities(sessions._application.registry)
        if reference_resources is not None and reference_resources.library is not None:
            visible['dialogue']['sequence_models'] = reference_resources.library.sequence_models()
        from .local_resources import available_input_view
        role_submission = available_input_view(interaction, allow_empty=True).submission
        if (submission or {}).get('available_input_overflow'):
            visible['dialogue']['session_source_candidates_overflow'] = True
        if (role_submission is not None
                and role_submission.get('registered_input', {}).get('composition') == 'scientific-role-selection.v1'):
            from .local_resources import LocalResourceAdmission, group_choices
            binding = LocalResourceAdmission._submission_binding(role_submission)
            choices = group_choices(binding, sessions._application.registry)
            visible['dialogue']['selected_sources'] = choices['sources']
            visible['dialogue']['input_roles'] = choices['roles']
            current = (submission or {}).get('registered_input', {})
            current_ids = {member['resource_id'] for member in current.get('members', ())}
            current_ids.update(item['resource_id'] for item in (submission or {}).get('attachments', ()))
            primary_id = current.get('resource_id') if current.get('tool_name') == 'inspect_scATAC' else None
            if primary_id is not None:
                current_ids.add(primary_id)
            visible['dialogue']['current_turn_sources'] = [f'source.{i}' for i, member in enumerate(binding.members)
                if member['resource_id'] in current_ids]
            if any(member['resource_id'] not in current_ids for member in binding.members):
                visible['dialogue']['session_source_candidates'] = True
            if primary_id is not None:
                primary = [f'source.{i}' for i, member in enumerate(binding.members) if member['resource_id'] == primary_id]
                if len(primary) != 1:
                    raise IntentError('invalid_reference')
                visible['dialogue']['selected_primary_source'] = primary[0]
            if binding.attachment_candidates:
                visible['dialogue']['attachment_candidates'] = True
                from .raw_handoff import offers as raw_offers
                visible['dialogue']['fastq_resources'] = raw_offers(input_sets)
                visible['dialogue']['bam_resources'] = raw_offers(input_sets, 'bam')
                visible['dialogue']['fragments_resources'] = raw_offers(input_sets, 'fragments')
                visible['dialogue']['annotation_resources'] = raw_offers(input_sets, 'annotation')
                visible['dialogue']['epizoo_resources'] = raw_offers(input_sets, 'epizoo')
        from .annotation_groups import public_choices
        visible['dialogue']['annotation_group_results'] = public_choices(sessions, interaction)
        if context is not None and context.items:
            visible['dialogue']['planning_context'] = list(context.public_items())
        # Project owner observations of the exact offered inputs, never a workflow.
        # Registered bindings retain source integrity; no observation becomes authority.
        offered = (role_submission or {}).get('registered_input', {})
        members = offered.get('members', ())
        if members and (context is None or not context.items or (submission or {}).get('attachments')):
            from .local_resources import LocalResourceAdmission
            owner = LocalResourceAdmission(sessions._application._workspace, registry=sessions._application.registry)
            motif_requirements = []
            for index, member in enumerate(members):
                record = owner.load(member['resource_id'])
                if record.input_type != 'h5ad':
                    continue
                try:
                    owner._verify_records([record])
                except (ValueError, OSError):
                    continue
                from agent.tools.analysis.chromvar import observe_input
                try:
                    motif_observation = observe_input(record.source_path)
                    from .dialogue_evidence import _unsafe_fact_value
                    if not _unsafe_fact_value(motif_observation):
                        motif_requirements.append(dict(source=f'source.{index}', **motif_observation))
                except (ValueError, OSError, KeyError, TypeError):
                    pass
            if motif_requirements:
                visible['dialogue']['motif_input_declarations'] = motif_requirements
        settings = (submission or {}).get('parameter_overrides', ())
        if settings:
            visible['dialogue']['structured_parameters'] = [
                {k: _serialize(row[k]) for k in ('tool', 'argument', 'native_function', 'port', 'source') if row.get(k) is not None}
                for row in settings]
            visible['dialogue']['structured_parameter_instruction'] = (
                'These are explicit researcher settings already captured by the application. '
                'Use the selected tool/function scope to understand the task. Do not reconstruct '
                'these settings as text declarations or ask for their values; application binding owns them.')
        from .structured_parameters import selected_groups
        chosen_groups = selected_groups(captured, state, (submission or {}).get('selected_results', ()))
        if chosen_groups:
            visible['dialogue']['selected_input_results'] = [row['handle'] for row in chosen_groups]
            visible['dialogue']['selected_input_instruction'] = (
                'The researcher selected these exact accepted results through the interface. '
                'Their identities are already resolved. Use them for the current task; do not guess another result.')
        diagnostic_stage = 'interpretation'
        from .structured_parameters import selected_scope
        scoped_admitted = selected_scope(sessions, interaction, state.interactions)
        from .resource_continuation import selected_answer
        from .guidance_selection import selected_candidate
        decision = (Execute(scoped_admitted['tool']) if scoped_admitted is not None else
                    selected_candidate(interaction) or selected_answer(interaction, state.interactions))
        if decision is None:
            from .structured_parameters import resolve_sources
            from .dialogue_execution import captured_input_group
            authoritative = resolve_sources(settings, captured_input_group(interaction))
            decision = interpret(interpreter, utterance, visible, diagnostic_context=diagnostic_context,
                                 authoritative=authoritative)
        from .structured_parameters import merge_decision
        if scoped_admitted is None:
            decision = merge_decision(decision, interaction, sessions._application.registry)
        if chosen_groups:
            if isinstance(decision, (Execute, RequestRoles, RequestResources)):
                decision = replace(decision, input_results=tuple(g['handle'] for g in chosen_groups), context='current')
            elif isinstance(decision, Answer) and decision.intent == 'scientific' and len(chosen_groups) == 1:
                from .turn_decisions import ScientificTarget
                decision = replace(decision, scientific=replace(decision.scientific,
                    target=ScientificTarget(chosen_groups[0]['canonical']['handle'], decision.scientific.target.subject)))
            elif isinstance(decision, Answer) and decision.intent == 'guidance':
                from .turn_decisions import ScientificTarget
                decision = replace(decision, guidance=replace(decision.guidance,
                    targets=tuple(ScientificTarget(g['canonical']['handle']) for g in chosen_groups)))
            else:
                raise IntentError('unsupported_intent')
        diagnostic_stage = 'admission'
        if interaction.base_revision_id is None:
            if not (isinstance(decision, (ExecuteCandidate, Clarify, SpeciesAnswer, ParameterAnswer, RequestRoles, RoleAnswer, RequestResources, ResourceAnswer))
                    or isinstance(decision, Execute)
                    or isinstance(decision, Answer) and decision.intent == 'guidance'):
                raise IntentError('unavailable_context')
        if isinstance(decision, (Execute, RequestRoles, RequestResources)):
            from .scientific_guidance import references_candidate
            from .guidance_selection import candidates
            refs = candidates(state.interactions, captured)
            origin = refs[0]['origin_turn_id'] if refs else captured['dialogue']['predecessor']
            prior = next((i for i in state.interactions if i.turn_id == origin), None)
            if references_candidate(interaction, prior):
                raise IntentError('unsupported_intent')
        if isinstance(decision, Clarify):
            from .turn_decisions import PARAMETER_CLARIFICATION_REASONS
            if decision.choices and decision.reason in PARAMETER_CLARIFICATION_REASONS:
                offered_parameters = {p['tool']+'.'+p['argument'] for p in visible['dialogue'].get('scientific_parameters',())}
                if any('.' in name and name not in offered_parameters for name in decision.choices):
                    raise IntentError('invalid_decision')
            if decision.reason == 'unavailable_model_condition':
                conditions = tuple(model['family'] + ': ' + ', '.join(model['conditions'])
                                   for model in visible['dialogue'].get('sequence_models', ()))
                decision = Clarify(decision.reason, conditions)
            return _terminal(sessions, interaction, decision)
        if scoped_admitted is not None:
            admitted = scoped_admitted
        elif isinstance(decision, RequestResources):
            from .resource_continuation import request, offers
            admitted, prerequisite = request(sessions, interaction, decision, input_sets)
            options = offers(submission, input_sets)
            reason = 'missing_resources'
            if 'reference_requirement' in admitted:
                from .reference_supplementation import choices as reference_choices, input_set as reference_input_set
                options = [dict(handle='resource.' + choice.input_set_id, label=choice.label)
                    for choice in (reference_input_set(r) for r in reference_choices(reference_resources, session_id,
                        admitted['reference_requirement'], origin_turn_id=turn_id))] if reference_resources else []
                reason = 'missing_reference'
            admitted['resource_options'] = [dict(option, description=option['label']) for option in options]
            _update(sessions, turn_id, status='clarification', admitted=admitted, prerequisite=prerequisite)
            choices = tuple(option['label'] for option in options)
            return TurnOutcome('clarify', 'clarification', Clarify(reason, choices, True))
        elif isinstance(decision, ResourceAnswer):
            from .resource_continuation import answer
            admitted = answer(sessions, interaction, decision, input_sets, reference_resources=reference_resources)
        elif isinstance(decision, (RequestRoles, RoleAnswer)) and (
                decision.fastq is not None or decision.bam is not None or decision.fragments is not None or decision.annotation is not None or decision.epizoo is not None or isinstance(decision, RoleAnswer) and (captured.get('prerequisite') or {}).get('field') in {'fastq', 'bam', 'fragments', 'annotation', 'epizoo'}):
            from .raw_handoff import admit as admit_raw
            admitted, prerequisite = admit_raw(sessions, interaction, decision, input_sets,
                                               reference_resources=reference_resources)
            if prerequisite is not None:
                _update(sessions, turn_id, status='clarification', admitted=admitted, prerequisite=prerequisite)
                return TurnOutcome('clarify', 'clarification', Clarify(admitted['reason'], tuple(admitted['choices']), True))
        elif isinstance(decision, RequestRoles):
            from .dialogue_execution import admit_role_request
            admitted, prerequisite, choices = admit_role_request(sessions, interaction, decision, epizoo_resources)
            _update(sessions, turn_id, status='clarification', admitted=admitted, prerequisite=prerequisite)
            return TurnOutcome('clarify', 'clarification', Clarify('missing_input_roles', choices, True))
        elif isinstance(decision, RoleAnswer):
            from .dialogue_execution import admit_roles
            admitted, prerequisite = admit_roles(sessions, interaction, decision, epizoo_resources)
            if prerequisite is not None:
                _update(sessions, turn_id, status='clarification', admitted=admitted, prerequisite=prerequisite)
                return TurnOutcome('clarify', 'clarification', Clarify('missing_input_roles', tuple(admitted['choices']), True))
        elif isinstance(decision, SpeciesAnswer):
            from .dialogue_execution import admit_species
            admitted = admit_species(sessions, interaction, decision, epizoo_resources)
        elif isinstance(decision, ParameterAnswer):
            from .dialogue_execution import admit_parameters
            admitted = admit_parameters(sessions, interaction, decision, epizoo_resources, execution_inputs=execution_inputs)
            if admitted['kind'] == 'clarify':
                reference = captured['prerequisite']
                prerequisite = {k: _serialize(v) for k, v in reference.items() if k != 'predecessor_turn_id'}
                prerequisite.update(binding_turn_id=interaction.turn_id, missing=admitted['choices'])
                _update(sessions, turn_id, status='clarification', admitted=admitted, prerequisite=prerequisite)
                return TurnOutcome('clarify', 'clarification', Clarify('missing_parameter_value', tuple(admitted['choices']), True))
        elif isinstance(decision, Answer):
            if decision.intent == 'guidance':
                from .scientific_guidance import admit as admit_guidance
                admitted = admit_guidance(sessions, interaction, decision.guidance)
            elif decision.intent == 'scientific':
                from .scientific_dialogue import admit as admit_scientific
                admitted = admit_scientific(sessions, interaction, decision.scientific)
        elif isinstance(decision, ExecuteCandidate):
            from .guidance_selection import admit as admit_candidate
            admitted = admit_candidate(sessions, interaction, decision, execution_inputs)
        elif isinstance(decision, Execute):
            from .dialogue_execution import admit as admit_execution, missing_file_request, admit_role_request
            pending_request = missing_file_request(sessions._application.registry, interaction, decision, execution_inputs)
            if pending_request is not None:
                admitted, prerequisite, choices = admit_role_request(sessions, interaction, pending_request, epizoo_resources)
                _update(sessions, turn_id, status='clarification', admitted=admitted, prerequisite=prerequisite)
                return TurnOutcome('clarify', 'clarification', Clarify('missing_input_roles', choices, True))
            admitted = admit_execution(sessions, interaction, decision, execution_inputs,
                                       epizoo_resources=epizoo_resources)
            selected_tool = sessions._application.registry.get(admitted['tool'])
            from .shared_resources import scientific_resource_scope
            from agent.orchestration.native_parameters import prerequisite_inputs
            prerequisite_values = prerequisite_inputs(selected_tool, admitted['inputs'])
            reference_owner = getattr(sessions, '_reference_resources', None)
            with scientific_resource_scope(None if reference_owner is None else reference_owner.library):
                owner_missing = (() if selected_tool.parameter_prerequisites is None or prerequisite_values is None else
                                 selected_tool.parameter_prerequisites(prerequisite_values))
            from .dialogue_execution import _missing_parameter_inputs
            missing_parameters = tuple(_missing_parameter_inputs(sessions._application.registry, admitted['tool'],
                tuple(dict.fromkeys(owner_missing)), admitted['inputs'])) + decision.missing_parameters
            if missing_parameters:
                from .dialogue_execution import request_parameters
                admitted, prerequisite = request_parameters(sessions, interaction, admitted,
                    missing_parameters, epizoo_resources)
                _update(sessions, turn_id, status='clarification', admitted=admitted, prerequisite=prerequisite)
                choices = tuple(prerequisite['missing'])
                return TurnOutcome('clarify', 'clarification', Clarify('missing_parameter_value', choices, True))
            if visible['dialogue'].get('resource_source'):
                from .resource_continuation import catalog
                admitted['resource_catalog'] = catalog(input_sets)
        else:
            admitted = admit(sessions, interaction, decision)
        from .structured_parameters import bound_records
        if (any(admitted.get(f + '_context', {}).get('adapter') == 'native-inputs.v1' for f in ('fastq','fragments','bam'))
                or admitted.get('kind') == 'execute' and (admitted.get('tool') == 'annotate_scATAC_cell_types'
                    or bound_records(admitted, state.interactions, sessions._application.registry, current_utterance=interaction.utterance))):
            from .dialogue_execution import prepare_native_task
            admitted, pending, error = prepare_native_task(sessions, interaction, admitted)
            if pending is not None:
                _update(sessions, turn_id, admitted=admitted, status='clarification', prerequisite=pending)
                return TurnOutcome('clarify', 'clarification', Clarify('missing_resources', tuple(admitted['choices']), True))
            if 'parameter_scope' in admitted:
                _update(sessions, turn_id, admitted=admitted, status='clarification')
                return TurnOutcome('clarify', 'clarification', Clarify('ambiguous_parameter'))
            if 'native_plan' not in admitted:
                _update(sessions, turn_id, admitted=admitted, status='failed')
                return TurnOutcome('execute', 'failed', error=error, text='' if error is None else error.message)
    except Exception as exc:
        log_operation_event(logger, 'turn.boundary_failed', context=diagnostic_context,
            stage=diagnostic_stage, error=exc, kind=None if decision is None else type(decision).__name__,
            reason=getattr(exc, 'diagnostic_reason', None) or getattr(exc, 'reason', None),
            operand=getattr(exc, 'operand', None), proposal=decision,
            registry=sessions._application.registry)
        from .local_resources import ResourceAdmissionError
        if isinstance(exc, ResourceAdmissionError):
            _failed_admission(sessions, interaction, decision)
            return TurnOutcome('execute', 'failed', text=exc.message, error=exc.error)
        if isinstance(exc, PlanningModelError):
            from .interactive_schemas import ClientError
            code = exc.code if exc.code in BUILTIN_ERROR_CODES else 'PLANNING_PROVIDER_ERROR'
            error = ClientError(code, safe_message_for(code))
            _failed_admission(sessions, interaction, decision,
                preserve_pending=diagnostic_stage == 'interpretation')
            return TurnOutcome('execute', 'failed', text=error.message, error=error)
        if not isinstance(exc, IntentError):
            # Preparation and application failures are not model decisions.
            # The interactive boundary owns their sanitized presentation.
            _failed_admission(sessions, interaction, decision)
            raise
        reason = exc.reason
        from .turn_decisions import DeclarationGroundingError
        raw_rejection = exc.raw if isinstance(exc, DeclarationGroundingError) else None
        candidate = decision
        if (reason in {'invalid_parameter_value', 'ungrounded_operand', 'ambiguous_parameter'}
                and isinstance(candidate, Execute)):
            import json
            value = dict(kind='execute_plan', **_serialize(asdict(candidate)))
            raw_rejection = json.dumps(dict(turn_schema_version=1, decision=value))
        if raw_rejection is not None and captured.get('prerequisite') is None:
            from .dialogue_execution import request_parameter_repair
            try:
                admitted, pending = request_parameter_repair(sessions, interaction,
                    raw_rejection, execution_inputs, epizoo_resources)
            except (ValueError, TypeError, KeyError):
                pass  # An unsafe or unsupported objective stays an ordinary clarification.
            else:
                _update(sessions, turn_id, status='clarification', admitted=admitted, prerequisite=pending)
                return TurnOutcome('clarify', 'clarification',
                    Clarify(reason, tuple(pending['missing']), True))
        if (reason in {'invalid_decision', 'invalid_reference', 'invalid_prerequisite',
                       'unsupported_intent', 'unavailable_context'}
                and getattr(exc, 'diagnostic_reason', None) != 'incompatible_scientific_reference'):
            from .interactive_schemas import ClientError
            _failed_admission(sessions, interaction, decision,
                preserve_pending=diagnostic_stage == 'interpretation')
            if reason == 'unavailable_context' and (isinstance(decision, Answer) or diagnostic_stage == 'preparation'):
                error = ClientError('INTERACTIVE_REFERENCE_INVALID',
                    'The saved evidence required for this answer is unavailable or invalid.')
                return TurnOutcome('answer', 'unavailable', text=error.message, error=error)
            error = ClientError('INTERACTIVE_INTENT_BINDING_INVALID',
                'The interpreted request could not be matched to the offered data or results. '
                'No new analysis was started.')
            return TurnOutcome('answer' if isinstance(decision, Answer) else 'execute',
                               'failed', text=error.message, error=error)
        return _terminal(sessions, interaction, Clarify(reason, exc.parameters))
    changes = dict(admitted=admitted, status='admitted')
    if isinstance(decision, Answer) and decision.intent == 'guidance' and captured.get('prerequisite'):
        changes['prerequisite'] = {k: _serialize(v) for k, v in captured['prerequisite'].items()
                                   if k != 'predecessor_turn_id'}
    _update(sessions, turn_id, **changes)
    try:
        if isinstance(decision, Answer):
            if decision.intent == 'guidance':
                from .scientific_guidance import answer
                result = answer(sessions, session_id, replace(interaction, admitted=admitted), answerer)
                return result
            elif decision.intent == 'scientific':
                from .scientific_dialogue import answer
                result = answer(sessions, session_id, replace(interaction, admitted=admitted), answerer)
            _update(sessions, turn_id, status='answered' if result.status in {'answered','unsupported'} else 'failed')
            return result
        if isinstance(decision, Navigate):
            sessions.switch(session_id, turn_id, admitted['revision_id'], expected_generation=generation)
            _update(sessions, turn_id, status='navigated')
            return TurnOutcome('navigate', 'activated')
        if admitted.get('operation') == 'plan':
            from .dialogue_execution import execute as execute_request
            if epizoo_resources:
                return execute_request(sessions, interaction, admitted, interpreter, epizoo_resources=epizoo_resources)
            return execute_request(sessions, interaction, admitted, interpreter)
        raise IntentError('unsupported_intent')
    except Exception:
        _update(sessions, turn_id, status='failed')
        raise
