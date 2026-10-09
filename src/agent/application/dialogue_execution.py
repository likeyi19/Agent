"""Route explicit new-science commands to the existing Planner/Application.

There is no scientific implementation or implicit workflow/output completion
here. Structured inputs come from the caller; output selection is explicit and
validated against the actual plan before any execution is allowed.
"""
from contextlib import nullcontext
from dataclasses import replace
import json
import re

from agent.schemas import AgentRequest
from agent.schemas.orchestration import _serialize, freeze_json_mapping
from .session_state import OutputSelection, SessionTurn, SessionConflictError, digest
from .turn_decisions import IntentError, _object, _shape


def is_execution_command(utterance):
    # Retained only for M17.2 candidate command evidence. Generic admission
    # follows the typed interpreter decision without classifying language again.
    command = re.sub(r'^(?:please\s+|(?:can|could|would) you\s+)', '', utterance.strip(), flags=re.I)
    return bool(re.match(r'^(?:run|compute|perform|test|reannotate|annotate|inspect|build|prepare|evaluate|adapt|transfer|select)\b', command, re.I))


def admit(sessions, interaction, decision, inputs):
    if (decision.base != 'current' or decision.delta is not None
            or decision.target not in sessions._application.registry.names()):
        raise IntentError('unsupported_intent')
    values = {} if inputs is None else _serialize(freeze_json_mapping(inputs, 'execution_inputs'))
    return dict(kind='execute', operation='plan', tool=decision.target, revision_id=interaction.base_revision_id,
                request_id='turn-' + digest(dict(session=sessions._interaction_session_id, turn=interaction.turn_id)),
                inputs=values)


def execute(sessions, interaction, admitted, model):
    from .turns import _AdmittedPlanner, _update, TurnOutcome
    from .service import ResearchAgentApplication
    from .scientific_dialogue import _json
    from .local_resources import LocalResourceAdmission, ResourceAdmissionError
    app = sessions._application
    sid = sessions._interaction_session_id
    selected = 'selected_candidate' in admitted
    intent, context = interaction.utterance, None
    if selected:
        from .guidance_selection import execution_context, planning_intent
        context = execution_context(sessions, interaction, admitted)
        intent = planning_intent(sessions, interaction, admitted, context)
    request = AgentRequest(admitted['request_id'], intent, admitted['inputs'])
    registered = interaction.submission is not None and 'registered_input' in interaction.submission
    resources = LocalResourceAdmission(app._workspace, registry=app.registry) if registered else None
    resource_failure = None

    def accept(effective, plan):
        nonlocal resource_failure
        if admitted['tool'] not in {s.tool_name for s in plan.steps}:
            raise IntentError('planning_failed')
        offered = [dict(step_id=s.step_id, tool=s.tool_name,
                        outputs=list(app.registry.get(s.tool_name).result_contract.required_fields)) for s in plan.steps]
        schema = _object(dict(outputs={'type':'array','minItems':1,'maxItems':32,
            'items':_object(dict(name={'type':'string'}, step_id={'type':'string'}, output_key={'type':'string'}))}))
        choice = _json(model.complete(prompt=json.dumps(dict(output_selection_schema_version=1,
            question=intent, steps=offered,
            instructions='Select exact named outputs of this actual plan to retain in the new revision. Do not infer or append scientific steps.')),
            response_schema=schema))
        _shape(choice, ('outputs',))
        if type(choice['outputs']) is not list or not 1 <= len(choice['outputs']) <= 32:
            raise IntentError('planning_failed')
        selections = []
        for item in choice['outputs']:
            _shape(item, ('name','step_id','output_key'))
            options = [o for o in offered if o['step_id'] == item['step_id']]
            if len(options) != 1 or item['output_key'] not in options[0]['outputs']:
                raise IntentError('planning_failed')
            selections.append(OutputSelection(**item))
        if len({o.name for o in selections}) != len(selections): raise IntentError('planning_failed')
        if selected:
            execution_context(sessions, interaction, admitted)
        if registered:
            # The last provider call has completed. Check application byte identity
            # before execution; scientific compatibility still belongs to the tools.
            try:
                resources.validate_plan(interaction.submission, plan)
                if any(effective.inputs.get(key) != value
                       for key, value in interaction.submission['execution_inputs'].items()):
                    raise ResourceAdmissionError('LOCAL_RESOURCE_BINDING_INVALID')
            except ResourceAdmissionError as exc:
                from agent.orchestration.planner import PlannerError
                from agent.schemas import ErrorCategory
                resource_failure = exc.error
                raise PlannerError(exc.code, exc.message, category=ErrorCategory.RESOURCE_ERROR) from exc
        def link(state):
            if selected and (state.active_revision_id != interaction.base_revision_id
                             or state.generation != interaction.base_generation):
                raise SessionConflictError('Selected candidate state changed before submission.')
            if any(t.turn_id == interaction.turn_id for t in state.turns):
                raise SessionConflictError('Turn already submitted.')
            turn = SessionTurn(interaction.turn_id, interaction.base_revision_id, interaction.base_generation,
                request.request_id, digest(effective.to_dict()), request.request_id + ':run', 'linked', tuple(selections))
            return replace(state, turns=state.turns+(turn,))
        sessions._store._update(sid, link)
        _update(sessions, interaction.turn_id, status='submitted')

    execution_app = ResearchAgentApplication(app.workspace_root,
        planner=_AdmittedPlanner(app.runtime.planner, accept), registry=app.registry, executor=app.runtime.executor)
    from agent.orchestration.active_context import planning_context
    with planning_context(context) if context is not None else nullcontext():
        result = execution_app.run(request)
    state = sessions.load(sid)
    if any(t.turn_id == interaction.turn_id for t in state.turns):
        try:
            sessions._record_result(sid, interaction.turn_id, result)
        except ResourceAdmissionError as exc:
            return TurnOutcome('execute', 'failed', text=exc.message, error=exc.error)
        state = sessions.recover(sid, interaction.turn_id)
        return completion_outcome(state, interaction.turn_id, error=result.error)
    _update(sessions, interaction.turn_id, status='failed')
    error = resource_failure or result.error or next(iter(result.run_result.errors), None)
    if (registered and interaction.submission['registered_input'].get('composition') == 'h5ad-science.v1'
            and error is not None and error.code in {'MISSING_REQUIRED_SOURCE', 'MISSING_REQUIRED_BINDING'}
            and error.details.get('tool_name') == 'epizoo_embed_cells'
            and error.details.get('target_port') == 'species'
            and 'species' not in interaction.submission['execution_inputs']):
        # Presentation of a compiler-authored missing-port fact, not a second
        # interpretation of the utterance or a repaired scientific plan.
        error = ResourceAdmissionError('H5AD_SPECIES_REQUIRED').error
    return TurnOutcome('execute', 'failed', error=error,
        text=error.message if error is not None else
        'The scientific execution request could not form a valid plan from the supplied inputs.')


def completion_outcome(state, turn_id, *, error=None):
    """Render the persisted completion, including explicit post-crash recovery."""
    from .turns import TurnOutcome
    turn = state.turn(turn_id)
    status = turn.status
    active = status == 'activated' and state.active_revision_id == turn.revision_id
    return TurnOutcome('execute', status,
        text='The request followed the registered scientific execution path. '
             + ('Its accepted outputs form the active revision.' if active
                else 'Its accepted revision is historical; another revision is active.' if status == 'activated'
                else 'Its persisted execution status is ' + status + '.'), error=error)
