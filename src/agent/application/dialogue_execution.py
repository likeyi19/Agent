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
    # An execution route requires an explicit command, not a result question.
    # No tool/parameter/step is inferred from this command-language check.
    command = re.sub(r'^(?:please\s+|(?:can|could|would) you\s+)', '', utterance.strip(), flags=re.I)
    return bool(re.match(r'^(?:run|compute|perform|test|reannotate|annotate|inspect|build|prepare|evaluate|adapt|transfer|select)\b', command, re.I))


def admit(sessions, interaction, decision, inputs):
    if (decision.base != 'current' or decision.delta is not None
            or decision.target not in sessions._application.registry.names()):
        raise IntentError('unsupported_intent')
    if not is_execution_command(interaction.utterance):
        raise IntentError('unsupported_intent')
    values = {} if inputs is None else _serialize(freeze_json_mapping(inputs, 'execution_inputs'))
    return dict(kind='execute', operation='plan', tool=decision.target, revision_id=interaction.base_revision_id,
                request_id='turn-' + digest(dict(session=sessions._interaction_session_id, turn=interaction.turn_id)),
                inputs=values)


def execute(sessions, interaction, admitted, model):
    from .turns import _AdmittedPlanner, _update, TurnOutcome
    from .service import ResearchAgentApplication
    from .scientific_dialogue import _json
    app = sessions._application
    sid = sessions._interaction_session_id
    selected = 'selected_candidate' in admitted
    intent, context = interaction.utterance, None
    if selected:
        from .guidance_selection import execution_context, planning_intent
        context = execution_context(sessions, interaction, admitted)
        intent = planning_intent(sessions, interaction, admitted, context)
    request = AgentRequest(admitted['request_id'], intent, admitted['inputs'])

    def accept(effective, plan):
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
        sessions._record_result(sid, interaction.turn_id, result)
        state = sessions.recover(sid, interaction.turn_id)
        return TurnOutcome('execute', state.turn(interaction.turn_id).status,
            text='The request followed the registered scientific execution path. '
                 + ('Its accepted outputs form the active revision.' if state.turn(interaction.turn_id).status == 'activated'
                    else 'Its persisted execution status is ' + state.turn(interaction.turn_id).status + '.'))
    _update(sessions, interaction.turn_id, status='failed')
    return TurnOutcome('execute', 'failed', text='The scientific execution request could not form a valid plan from the supplied inputs.')
