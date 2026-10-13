"""Bind existing operator InputSets to the existing Session pending objective.

The captured catalog is configuration, never scientific authority. Only opaque
handles and display labels reach the interpreter. Compilation and scientific
qualification still occur through the ordinary execution path.
"""
from dataclasses import replace
from agent.schemas.orchestration import _serialize
from .session_state import digest, SessionError
from .turn_decisions import IntentError
from .input_sets import ScientificInputSet

COMPOSITIONS = {'external-fragments-science.v1': 'fragments',
                'bam-science.v1': 'bam', 'fastq-science.v1': 'fastq'}


def catalog(input_sets):
    if (type(input_sets) is not tuple or len(input_sets) > 64
            or any(not isinstance(s, ScientificInputSet) for s in input_sets)
            or len({s.input_set_id for s in input_sets}) != len(input_sets)):
        raise IntentError('invalid_prerequisite')
    return [{key: _serialize(getattr(s, key)) for key in s.__dataclass_fields__}
            for s in input_sets if any(getattr(s, family + '_companion') for family in COMPOSITIONS.values())]


def offers(submission, input_sets):
    family = COMPOSITIONS.get((submission or {}).get('registered_input', {}).get('composition'))
    return [dict(handle='resource.' + s.input_set_id, label=s.label)
            for s in input_sets if family and getattr(s, family + '_companion')]


def _source(reference, preceding):
    return (next(i for i in preceding if i.turn_id == reference['origin_turn_id']),
            next(i for i in preceding if i.turn_id == reference['binding_turn_id']))


def capture(prior, origin, current):
    """A new InputSet may be selected; an existing source cannot be replaced."""
    from .dialogue_execution import execution_submission
    original = execution_submission(origin)
    selected = current.get('registered_input')
    old = original.get('registered_input', {})
    if selected is not None and any(selected.get(k) != old.get(k)
            for k in ('resource_id', 'record_sha256', 'collection_id', 'tool_name', 'composition', 'members')):
        return None
    # Resource values are checked against the captured catalog at admission.
    # Source values must still be exactly the captured registration.
    for k in ('source_path', 'source_sha256', 'source_index_path', 'source_index_sha256', 'raw_input_paths'):
        if k in current.get('execution_inputs', {}) and current['execution_inputs'][k] != original['execution_inputs'].get(k):
            return None
    return dict(**_serialize(prior.prerequisite), predecessor_turn_id=prior.turn_id)


def public(interaction, preceding):
    from .dialogue_execution import _execution_objective
    reference = interaction.snapshot['prerequisite']
    origin, source = _source(reference, preceding)
    rows = source.admitted.get('resource_catalog', origin.admitted['resource_catalog'])
    choices = tuple(ScientificInputSet(**_serialize(row)) for row in rows)
    return dict(handle='@resources', field='resources', tool=reference['tool'],
                objective=_execution_objective(origin, preceding),
                choices=offers(origin.submission, choices))


def request(sessions, interaction, decision, input_sets):
    from .dialogue_execution import admit
    from .turn_decisions import Execute
    if COMPOSITIONS.get((interaction.submission or {}).get('registered_input', {}).get('composition')) is None:
        raise IntentError('invalid_prerequisite')
    admitted = admit(sessions, interaction, Execute('current', 'plan', decision.target, None,
        arguments=decision.arguments), interaction.submission['execution_inputs'])
    rows = catalog(input_sets)
    admitted.update(kind='request_resources', resource_catalog=rows)
    pending = dict(origin_turn_id=interaction.turn_id, binding_turn_id=interaction.turn_id,
        field='resources', tool=decision.target, resource_context_sha256=digest(_serialize(rows)))
    return admitted, pending


def _completed(origin, source, interaction, choice_handle, arguments, registry):
    from .dialogue_execution import _bind_arguments, execution_submission
    from .local_resources import LocalResourceAdmission
    original = execution_submission(origin)
    binding = LocalResourceAdmission._submission_binding(original)
    values = _serialize(source.admitted['inputs'])
    rows = origin.admitted['resource_catalog']
    choices = tuple(ScientificInputSet(**_serialize(row)) for row in rows)
    family = COMPOSITIONS[binding.composition]
    offered = {('resource.' + s.input_set_id): s for s in choices if getattr(s, family + '_companion')}
    if choice_handle not in offered:
        raise IntentError('invalid_prerequisite')
    chosen = offered[choice_handle]
    original_choice = binding.resource_choice
    if original_choice is not None and _serialize(original_choice) != chosen.resource_choice():
        raise IntentError('invalid_prerequisite')
    for key, value in chosen.inputs().items():
        if key in values and values[key] != value:
            raise IntentError('conflicting_scientific_parameter')
        values[key] = value
    current = _serialize(interaction.submission or {})
    received_choice = current.get('registered_input', {}).get('resource_choice')
    if received_choice is not None and received_choice != chosen.resource_choice():
        raise IntentError('invalid_prerequisite')
    # Received inputs may repeat captured declarations or the exact selected set.
    # They cannot inject another path or change a previous declaration.
    if any(key not in values or value != values[key]
           for key, value in current.get('execution_inputs', {}).items()):
        raise IntentError('invalid_prerequisite')
    values = _bind_arguments(registry, interaction.utterance, values, arguments)
    completed = replace(binding, execution_inputs=values, resource_choice=chosen.resource_choice(),
                        resource_selection_error=None)
    return completed


def answer(sessions, interaction, decision, input_sets):
    from .dialogue_execution import validate_prerequisite
    from .local_resources import LocalResourceAdmission
    from dataclasses import asdict
    state = sessions.load(sessions._interaction_session_id)
    reference = interaction.snapshot.get('prerequisite')
    if (decision.pending != '@resources' or not reference or reference['field'] != 'resources'
            or state.generation != interaction.base_generation or state.active_revision_id != interaction.base_revision_id
            or state.interactions[-1].turn_id != interaction.turn_id
            or digest(catalog(input_sets)) != reference['resource_context_sha256']):
        raise IntentError('invalid_prerequisite')
    validate_prerequisite(interaction, state.interactions[:-1])
    origin, source = _source(reference, state.interactions[:-1])
    if origin.admitted['kind'] == 'execute':
        run = sessions._application.run_store.load(origin.admitted['request_id'] + ':run')
        if run.steps or not run.errors or run.request.prompt != origin.utterance:
            raise IntentError('invalid_prerequisite')
    completed = _completed(origin, source, interaction, decision.choice, decision.arguments,
                           sessions._application.registry)
    owner = LocalResourceAdmission(sessions._application._workspace, registry=sessions._application.registry)
    owner.validate_binding(completed)
    return dict(kind='execute', operation='plan', tool=origin.admitted['tool'],
        revision_id=interaction.base_revision_id,
        request_id='turn-' + digest(dict(session=state.session_id, turn=interaction.turn_id)),
        inputs=_serialize(completed.execution_inputs), registered_input=completed.attribution(),
        arguments=[asdict(d) for d in decision.arguments], resource_handle=decision.choice,
        continuation={k: reference[k] for k in ('origin_turn_id', 'binding_turn_id', 'field')})


def validate(interaction, preceding):
    """Replay exact captured configuration and declarations without scientific IO."""
    from .dialogue_execution import capture_prerequisite, _bind_arguments
    from .turn_decisions import _scientific_arguments
    from agent.orchestration.registry import build_default_tool_registry
    admitted = interaction.admitted or {}
    reference = interaction.snapshot.get('prerequisite')
    pending = interaction.prerequisite
    try:
        expected = capture_prerequisite(preceding, interaction.base_revision_id,
            interaction.base_generation, interaction.submission,
            interaction.snapshot.get('dialogue', {}).get('requested_predecessor'))
        if reference != expected:
            raise ValueError('Resource prerequisite capture changed.')
        if (admitted.get('kind') == 'request_resources' or
                pending is not None and pending['field'] == 'resources'
                and pending['origin_turn_id'] == interaction.turn_id):
            rows = admitted['resource_catalog']
            if catalog(tuple(ScientificInputSet(**_serialize(r)) for r in rows)) != _serialize(rows):
                raise ValueError('Invalid captured InputSets.')
            if pending != dict(origin_turn_id=interaction.turn_id, binding_turn_id=interaction.turn_id,
                field='resources', tool=admitted['tool'], resource_context_sha256=digest(_serialize(rows))):
                raise ValueError('Changed resource request.')
            registry = build_default_tool_registry()
            registry.get(admitted['tool'])
            values = _bind_arguments(registry, interaction.utterance,
                _serialize(interaction.submission['execution_inputs']),
                _scientific_arguments(_serialize(admitted.get('arguments', []))))
            if values != _serialize(admitted['inputs']):
                raise ValueError('Changed resource request inputs.')
            return
        if reference is None:
            raise ValueError('Missing resource origin.')
        if admitted.get('kind') == 'clarify':
            if pending != {k: v for k, v in reference.items() if k != 'predecessor_turn_id'}:
                raise ValueError('Lost pending resource request.')
            return
        if not admitted:  # captured, before interpretation
            return
        if admitted.get('continuation', {}).get('field') != 'resources':
            if pending is not None:
                raise ValueError('Unexpected resource pending state.')
            return  # an unrelated ordinary decision does not answer this request
        origin, source = _source(reference, preceding)
        completed = _completed(origin, source, interaction, admitted['resource_handle'],
            _scientific_arguments(_serialize(admitted['arguments'])), build_default_tool_registry())
        if (admitted['kind'] != 'execute' or admitted['operation'] != 'plan'
                or admitted['tool'] != origin.admitted['tool']
                or admitted['revision_id'] != interaction.base_revision_id
                or _serialize(admitted['inputs']) != _serialize(completed.execution_inputs)
                or _serialize(admitted['registered_input']) != completed.attribution()
                or admitted['continuation'] != {k: reference[k] for k in ('origin_turn_id', 'binding_turn_id', 'field')}):
            raise ValueError('Changed resource continuation.')
    except (ValueError, TypeError, KeyError, StopIteration, AttributeError) as exc:
        raise SessionError('Invalid pending scientific resources.') from exc
