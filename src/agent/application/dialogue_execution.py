"""Route explicit new-science commands to the existing Planner/Application.

There is no scientific implementation or implicit workflow/output completion
here. Structured inputs come from the caller; output selection is explicit and
validated against the actual plan before any execution is allowed.
"""
from contextlib import nullcontext
from dataclasses import asdict, replace
import json
import re

from agent.schemas import AgentRequest
from agent.schemas.orchestration import _serialize, freeze_json_mapping
from .session_state import OutputSelection, SessionTurn, SessionConflictError, digest
from .turn_decisions import (IntentError, LeidenResolution, ScalarChoiceArgument,
    _scientific_arguments, admit_leiden_resolution, _object, _shape)


def _resolution_input(registry):
    """Resolve the one reviewed argument through its existing semantic port."""
    return _parameter_input(registry, 'cluster_cells', 'resolution')


def _parameter_input(registry, tool_name, argument_name):
    """Resolve a canonical scalar identity using its registered request port."""
    if tool_name not in registry.names():
        raise IntentError('unsupported_intent')
    tool = registry.get(tool_name)
    argument = tool.required_arguments.get(argument_name, tool.optional_arguments.get(argument_name))
    choice = (argument is not None and argument.planning is not None
        and argument.planning.conversational_choice and argument.accepted_types == (str,)
        and bool(argument.choices) and all(type(v) is str and v.strip() for v in argument.choices)
        and not argument.planning.accepted_artifact_kinds)
    if (argument is None or argument.planning is None or not argument.planning.scientific_parameter
            or not (choice or any(t in argument.accepted_types for t in (int, float)))
            or tool.semantic_planning is None):
        raise IntentError('unsupported_intent')
    inputs = {member.input_name for port in tool.semantic_planning.consumer_ports
              for source in port.request_sources for member in source.members
              if len(source.members) == 1 and any(field.name == member.name
                  and field.field_name == argument_name for field in port.members)}
    if len(inputs) != 1:
        raise IntentError('ambiguous_parameter')
    return inputs.pop(), argument


def scientific_parameters(registry):
    """Offer registered scalar contracts; never copy defaults or input values."""
    offered = []
    for tool_name in registry.names():
        tool = registry.get(tool_name)
        for name, spec in {**tool.required_arguments, **tool.optional_arguments}.items():
            try:
                _parameter_input(registry, tool_name, name)
            except IntentError:
                continue
            item = dict(tool=tool_name, argument=name, required=name in tool.required_arguments,
                        description=spec.planning.description)
            if spec.planning.conversational_choice:
                item['choices'] = list(spec.choices)
            offered.append(item)
    return offered


def _arguments(decision):
    argument = getattr(decision, 'argument', None)
    return (() if argument is None else (argument,)) + decision.arguments


def _bind_arguments(registry, utterance, inputs, declarations, *, replace_existing=False):
    from .turn_decisions import admit_scientific_argument
    values = dict(inputs)
    names = set()
    for declaration in declarations:
        name, specification = _parameter_input(registry, declaration.tool, declaration.argument)
        if specification.planning.conversational_choice != isinstance(declaration, ScalarChoiceArgument):
            raise IntentError('unsupported_intent')
        if name in names:
            raise IntentError('ambiguous_parameter')
        names.add(name)
        value = admit_scientific_argument(declaration, utterance, specification)
        if name in values:
            try:
                specification.validate(declaration.argument, values[name])
            except (ValueError, TypeError) as exc:
                raise IntentError('invalid_parameter_value') from exc
            if values[name] != value and not replace_existing:
                raise IntentError('conflicting_scientific_parameter')
        values[name] = value
    return values


def _select_declared_resource(values, resources):
    """Use the existing catalog owner without replacing received declarations."""
    from .local_resources import ResourceAdmissionError, select_epizoo_resource
    selected_id = values.get('expected_resource_identity', {}).get('resource_id')
    completed, error = select_epizoo_resource(values, selected_id, resources)
    if any(completed.get(key) != value for key, value in values.items()):
        raise ResourceAdmissionError('EPIZOO_RESOURCE_SELECTION_INVALID')
    return completed, error


def _replay_initial_inputs(registry, interaction, declarations):
    """Replay captured application policy; it grants no scientific authority."""
    from .local_resources import QualifiedEpiZooResource, RegisteredInput, qualified_epizoo_resources
    admitted = _serialize(interaction.admitted)
    received = _serialize(interaction.submission)['execution_inputs']
    values = _bind_arguments(registry, interaction.utterance, received, declarations)
    original = interaction.submission.get('registered_input', {})
    choice = any(isinstance(d, ScalarChoiceArgument) for d in declarations)
    needs_resource = (choice and original.get('tool_name') == 'inspect_scATAC'
        and original.get('composition') in (None, 'h5ad-science.v1'))
    receipt = admitted.get('resource_selection')
    if needs_resource:
        _shape(receipt, ('catalog', 'resource_context_sha256'))
        if type(receipt['catalog']) not in (list, tuple) or len(receipt['catalog']) > 64:
            raise ValueError('Invalid captured resource catalog.')
        resources = qualified_epizoo_resources(tuple(QualifiedEpiZooResource(**_serialize(row))
            for row in receipt['catalog']))
        if resource_context_digest(resources) != receipt['resource_context_sha256']:
            raise ValueError('Captured resource catalog changed.')
        values, error = _select_declared_resource(values, resources)
        binding = RegisteredInput(original['resource_id'], original['record_sha256'], original['tool_name'],
            values, 'h5ad-science.v1', error)
        if _serialize(admitted.get('registered_input')) != binding.attribution():
            raise ValueError('Resource selection changed registered attribution.')
    elif receipt is not None:
        raise ValueError('Unexpected initial resource selection.')
    return values


def _bind_resolution(registry, utterance, inputs, declaration):
    return _bind_arguments(registry, utterance, inputs, (declaration,))


def _validate_resolution_record(interaction):
    """Check the single persisted declaration against its received source text."""
    from .session_state import SessionError
    admitted = interaction.admitted or {}
    if 'argument' not in admitted:
        received = {} if interaction.submission is None else interaction.submission.get('execution_inputs', {})
        if ('argument_received_inputs_sha256' in admitted
                or (interaction.submission is not None and admitted.get('operation') == 'plan'
                    and 'continuation' not in admitted and 'selected_candidate' not in admitted and 'arguments' not in admitted
                    and 'resolution' in admitted.get('inputs', {}) and 'resolution' not in received)):
            raise SessionError('Missing persisted Leiden resolution declaration.')
        return
    try:
        from agent.orchestration.registry import build_default_tool_registry
        registry = build_default_tool_registry()
        declaration = LeidenResolution(**_serialize(admitted['argument']))
        name, specification = _resolution_input(registry)
        value = admit_leiden_resolution(declaration, interaction.utterance, specification)
        specification.validate(name, admitted.get('inputs', {}).get(name))
        if (admitted.get('kind') != 'execute' or admitted.get('operation') != 'plan'
                or admitted['inputs'].get(name) != value):
            raise ValueError('Declaration differs from admitted input.')
        received = {} if interaction.submission is None else _serialize(interaction.submission)['execution_inputs']
        if name in received:
            specification.validate(name, received[name])
            if received[name] != value:
                raise ValueError('Conflicting received declaration.')
        if 'continuation' not in admitted:
            if interaction.submission is not None:
                declarations = (declaration,) + _scientific_arguments(_serialize(admitted.get('arguments', ())))
                expected = _replay_initial_inputs(registry, interaction, declarations)
                if expected != _serialize(admitted['inputs']):
                    raise ValueError('Declaration changed other received inputs.')
                original = interaction.submission.get('registered_input')
                if original is not None and any(admitted.get('registered_input', {}).get(k) != original.get(k)
                        for k in ('resource_id', 'record_sha256', 'tool_name')):
                    raise ValueError('Declaration changed the registered source.')
            else:
                values = _serialize(admitted['inputs'])
                declarations = (declaration,) + _scientific_arguments(_serialize(admitted.get('arguments', ())))
                names = {_parameter_input(registry, d.tool, d.argument)[0] for d in declarations}
                without = {k: v for k, v in values.items() if k not in names}
                if 'arguments_added_inputs' in admitted:
                    without = {k: v for k, v in values.items() if k not in admitted['arguments_added_inputs']}
                if admitted.get('argument_received_inputs_sha256') not in (digest(values), digest(without)):
                    raise ValueError('Declaration changed received direct inputs.')
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise SessionError('Invalid persisted Leiden resolution declaration.') from exc


def _validate_argument_record(interaction):
    """Recheck current-turn typed declarations without providers or science."""
    from .session_state import SessionError
    admitted = interaction.admitted or {}
    if 'arguments' not in admitted:
        if 'arguments_received_inputs_sha256' in admitted:
            raise SessionError('Missing persisted scientific arguments.')
        if (interaction.submission is not None and 'execution_inputs' in interaction.submission and admitted.get('kind') == 'execute'
                and admitted.get('operation') == 'plan' and 'continuation' not in admitted
                and 'argument' not in admitted and 'selected_candidate' not in admitted
                and _serialize(admitted.get('inputs')) != _serialize(interaction.submission['execution_inputs'])):
            raise SessionError('Missing persisted scientific argument declaration.')
        return
    try:
        from agent.orchestration.registry import build_default_tool_registry
        registry = build_default_tool_registry()
        declarations = _scientific_arguments(_serialize(admitted['arguments']))
        if 'argument' in admitted:
            declarations += (LeidenResolution(**_serialize(admitted['argument'])),)
        if not 1 <= len(declarations) <= 8 or len({(d.tool, d.argument) for d in declarations}) != len(declarations):
            raise ValueError('Invalid scoped declarations.')
        values = _serialize(admitted['inputs'])
        expected = _bind_arguments(registry, interaction.utterance, values, declarations)
        if expected != values:
            raise ValueError('Arguments differ from admitted values.')
        if 'continuation' not in admitted and 'parameter_continuation' not in admitted:
            if interaction.submission is not None:
                received = _serialize(interaction.submission)['execution_inputs']
                if _replay_initial_inputs(registry, interaction, declarations) != values:
                    raise ValueError('Arguments changed unrelated received inputs.')
                if ('registered_input' in admitted and admitted['registered_input'] != interaction.submission.get('registered_input')):
                    # H5AD composition changes its binding digest, never source identities.
                    before = interaction.submission.get('registered_input', {})
                    after = admitted['registered_input']
                    if any(before.get(k) != after.get(k) for k in ('resource_id', 'record_sha256', 'tool_name')):
                        raise ValueError('Argument changed registered source.')
            else:
                names = {_parameter_input(registry, d.tool, d.argument)[0] for d in declarations}
                without = {k: v for k, v in values.items() if k not in names}
                if 'arguments_added_inputs' in admitted:
                    added = admitted['arguments_added_inputs']
                    if len(set(added)) != len(added) or not set(added) <= names:
                        raise ValueError('Invalid added argument identities.')
                    without = {k: v for k, v in values.items() if k not in added}
                if admitted.get('arguments_received_inputs_sha256') not in (digest(values), digest(without)):
                    raise ValueError('Arguments changed received direct inputs.')
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise SessionError('Invalid persisted scientific arguments.') from exc


def _missing_parameters(error, admitted, registry):
    if error is None or error.code not in {'MISSING_REQUIRED_SOURCE', 'MISSING_REQUIRED_BINDING'}:
        return None
    tool_name = error.details.get('tool_name')
    names = error.details.get('argument_names')
    if tool_name not in registry.names() or not isinstance(names, (list, tuple)) or not 1 <= len(names) <= 8:
        return None
    tool = registry.get(tool_name)
    try:
        for name in names:
            if name not in tool.required_arguments:
                return None
            input_name, _ = _parameter_input(registry, tool_name, name)
            if tool.required_arguments[name].planning.conversational_choice:
                return None
            if input_name in admitted['inputs']:
                return None
    except IntentError:
        return None
    return tool_name, list(names)


def _parameter_source(reference, preceding):
    """Resolve the exact last accepted pending binding, never scan for a value."""
    origin = next(i for i in preceding if i.turn_id == reference['origin_turn_id'])
    source = next(i for i in preceding if i.turn_id == reference['binding_turn_id'])
    return origin, source, _serialize(source.admitted['inputs'])


def _validate_parameter_prerequisite(interaction, preceding):
    from .session_state import SessionError
    from agent.orchestration.registry import build_default_tool_registry
    registry = build_default_tool_registry()
    reference = interaction.snapshot.get('prerequisite')
    pending = interaction.prerequisite
    admitted = interaction.admitted or {}
    try:
        if reference is None or pending is not None and pending['origin_turn_id'] == interaction.turn_id:
            if (pending is None or pending['origin_turn_id'] != interaction.turn_id
                    or pending['binding_turn_id'] != interaction.turn_id
                    or admitted.get('kind') != 'execute' or admitted.get('operation') != 'plan'):
                raise ValueError('No originating execution.')
            values = admitted['inputs']
            tool = registry.get(pending['tool'])
            required = {name for name in tool.required_arguments
                        if tool.required_arguments[name].planning.scientific_parameter}
            missing = sorted(name for name in required if _parameter_input(registry, tool.name, name)[0] not in values)
            if list(pending['missing']) != missing:
                raise ValueError('Missing requirements changed.')
            return
        origin, source, original = _parameter_source(reference, preceding)
        continuation = admitted.get('parameter_continuation', admitted.get('continuation'))
        if continuation is None:
            if pending is None:
                return
            if (pending is None or _serialize(pending) != {k: _serialize(v) for k, v in reference.items()
                    if k != 'predecessor_turn_id'} or admitted.get('kind') != 'clarify'):
                raise ValueError('Pending binding lost.')
            return
        if (set(continuation) != {'origin_turn_id', 'binding_turn_id', 'field'}
                or continuation['field'] != 'parameters'
                or any(continuation[k] != reference[k] for k in ('origin_turn_id', 'binding_turn_id'))
                or admitted.get('tool', reference['tool']) != origin.admitted['tool']
                or admitted.get('revision_id', interaction.base_revision_id) != interaction.base_revision_id):
            raise ValueError('Parameter continuation changed scope.')
        declarations = _scientific_arguments(_serialize(admitted['arguments']))
        if any(d.tool != reference['tool'] for d in declarations):
            raise ValueError('Parameter changed operation.')
        expected = _bind_arguments(registry, interaction.utterance, original, declarations, replace_existing=True)
        if expected != _serialize(admitted['inputs']):
            raise ValueError('Parameter continuation changed inputs.')
        if 'registered_input' in admitted and admitted['registered_input'] != (origin.submission or {}).get('registered_input'):
            raise ValueError('Parameter continuation changed registered source.')
        missing = [name for name in origin.prerequisite['missing']
                   if _parameter_input(registry, reference['tool'], name)[0] not in expected]
        if admitted.get('kind') == 'clarify':
            wanted = {k: _serialize(v) for k, v in reference.items() if k != 'predecessor_turn_id'}
            wanted.update(binding_turn_id=interaction.turn_id, missing=missing)
            if not missing or _serialize(pending) != wanted or list(admitted['choices']) != missing:
                raise ValueError('Partial clarification changed requirements.')
        elif admitted.get('kind') != 'execute' or missing or pending is not None:
            raise ValueError('Incomplete execution continuation.')
    except (ValueError, TypeError, KeyError, AttributeError, StopIteration) as exc:
        raise SessionError('Invalid pending scientific parameter binding.') from exc


def admit_parameters(sessions, interaction, decision, resources, *, execution_inputs=None):
    """Bind an answer atomically to the exact pending operation and inputs."""
    state = sessions.load(sessions._interaction_session_id)
    reference = interaction.snapshot.get('prerequisite')
    if (decision.pending != '@parameters' or reference is None or reference['field'] != 'parameters'
            or state.generation != interaction.base_generation or state.active_revision_id != interaction.base_revision_id
            or state.interactions[-1].turn_id != interaction.turn_id):
        raise IntentError('invalid_prerequisite')
    validate_prerequisite(interaction, state.interactions[:-1])
    origin, source, values = _parameter_source(reference, state.interactions[:-1])
    if origin.admitted['request_id'] != 'turn-' + digest(dict(session=state.session_id, turn=origin.turn_id)):
        raise IntentError('invalid_prerequisite')
    if any(d.tool != reference['tool'] for d in decision.arguments):
        raise IntentError('ambiguous_parameter')
    if resource_context_digest(resources) != reference['resource_context_sha256']:
        raise IntentError('invalid_prerequisite')
    stored = sessions._application.run_store.load(origin.admitted['request_id'] + ':run')
    original_error = next((e for e in stored.errors
        if _missing_parameters(e, origin.admitted, sessions._application.registry) ==
           (reference['tool'], list(origin.prerequisite['missing']))), None)
    if (stored.steps or stored.request.prompt != origin.utterance or original_error is None
            or any(stored.request.inputs.get(k) != v for k, v in origin.admitted['inputs'].items())):
        raise IntentError('invalid_prerequisite')
    current = (_serialize(interaction.submission)['execution_inputs'] if interaction.submission is not None
               else _serialize(execution_inputs or {}))
    declared_inputs = {_parameter_input(sessions._application.registry, d.tool, d.argument)[0]
                       for d in decision.arguments}
    for parameter in scientific_parameters(sessions._application.registry):
        if parameter['tool'] != reference['tool']:
            continue
        name, spec = _parameter_input(sessions._application.registry, reference['tool'], parameter['argument'])
        if name in current:
            try:
                spec.validate(parameter['argument'], current[name])
            except (ValueError, TypeError) as exc:
                raise IntentError('invalid_parameter_value') from exc
            if name not in declared_inputs and (name not in values or values[name] != current[name]):
                raise IntentError('conflicting_scientific_parameter')
    # Current received values must agree with the answer; prior accepted values
    # may be corrected only by an exact scoped declaration in this utterance.
    _bind_arguments(sessions._application.registry, interaction.utterance, current, decision.arguments)
    values = _bind_arguments(sessions._application.registry, interaction.utterance, values,
                            decision.arguments, replace_existing=True)
    if origin.submission is not None and 'registered_input' in origin.submission:
        from .local_resources import LocalResourceAdmission
        LocalResourceAdmission(sessions._application._workspace, registry=sessions._application.registry).validate_submission(
            execution_submission(origin))
    missing = [name for name in origin.prerequisite['missing']
               if _parameter_input(sessions._application.registry, reference['tool'], name)[0] not in values]
    continuation = dict(origin_turn_id=origin.turn_id, binding_turn_id=source.turn_id, field='parameters')
    admitted = dict(kind='execute', operation='plan', tool=origin.admitted['tool'],
        revision_id=interaction.base_revision_id,
        request_id='turn-' + digest(dict(session=state.session_id, turn=interaction.turn_id)),
        inputs=values, arguments=[asdict(d) for d in decision.arguments], continuation=continuation)
    if origin.submission is not None and 'registered_input' in origin.submission:
        admitted['registered_input'] = _serialize(origin.submission['registered_input'])
    if missing:
        admitted = dict(kind='clarify', reason='missing_parameter_value', choices=missing, value_required=True,
            inputs=values, arguments=admitted['arguments'], parameter_continuation=continuation)
    if interaction.submission is None:
        admitted['parameter_received_inputs_sha256'] = digest(current)
    return admitted


def _parameter_identities(admitted, interactions):
    """Read only explicitly linked declarations and originally missing identities."""
    identities = set()
    records = [admitted]
    continuation = admitted.get('continuation', {})
    if continuation:
        by_id = {i.turn_id: i for i in interactions}
        origin = by_id[continuation['origin_turn_id']]
        records.append(origin.admitted)
        if continuation.get('field') == 'parameters':
            identities.update((origin.prerequisite['tool'], name) for name in origin.prerequisite['missing'])
            pointer = continuation['binding_turn_id']
            visited = set()
            while pointer != origin.turn_id:
                if pointer in visited:
                    raise IntentError('invalid_prerequisite')
                visited.add(pointer)
                record = by_id[pointer].admitted
                records.append(record)
                pointer = record['parameter_continuation']['binding_turn_id']
    for record in records:
        declarations = list(record.get('arguments', ()))
        if 'argument' in record:
            declarations.append(record['argument'])
        identities.update((d['tool'], d['argument']) for d in declarations)
    return identities


def parameter_scope_choices(admitted, interactions):
    return tuple(sorted(tool + '.' + name for tool, name in _parameter_identities(admitted, interactions)))


def _check_parameter_plan(registry, plan, admitted, interactions):
    """Verify exact consumers of scoped declarations; never alter a plan."""
    identities = _parameter_identities(admitted, interactions)
    for tool_name, argument_name in identities:
        name, _ = _parameter_input(registry, tool_name, argument_name)
        consumers = [s for s in plan.steps if s.tool_name == tool_name]
        if len(consumers) != 1:
            raise IntentError('ambiguous_parameter')
        if consumers[0].arguments.get(argument_name) != admitted['inputs'][name]:
            raise IntentError('planning_failed')
        for step in plan.steps:
            if step is consumers[0]:
                continue
            tool = registry.get(step.tool_name)
            for candidate in {**tool.required_arguments, **tool.optional_arguments}:
                try:
                    other_input, _ = _parameter_input(registry, step.tool_name, candidate)
                except IntentError:
                    continue
                if other_input == name and candidate in step.arguments:
                    raise IntentError('ambiguous_parameter')


def resource_context_digest(resources):
    # Exact operator policy, including pins and default designations. Changing
    # the catalog during a clarification cannot silently select another model.
    return digest(sorted((asdict(r) for r in resources), key=lambda r: r['resource_id']))


def _missing_species(error, submission):
    return (submission is not None
        and submission.get('registered_input', {}).get('composition') == 'h5ad-science.v1'
        and error is not None and error.code in {'MISSING_REQUIRED_SOURCE', 'MISSING_REQUIRED_BINDING'}
        and error.details.get('tool_name') == 'epizoo_embed_cells'
        and error.details.get('target_port') == 'species'
        and 'species' not in submission['execution_inputs'])


def capture_prerequisite(preceding, revision_id, generation, submission, predecessor=None, *, execution_inputs=None):
    """Offer only the immediately pending request in the same captured scope."""
    if not preceding or predecessor is not None:
        return None
    prior = preceding[-1]
    if (prior.prerequisite is None or prior.base_revision_id != revision_id
            or prior.base_generation != generation):
        return None
    pending = _serialize(prior.prerequisite)
    origins = [i for i in preceding if i.turn_id == pending['origin_turn_id']]
    if len(origins) != 1:
        raise IntentError('invalid_prerequisite')
    origin = origins[0]
    current = dict(execution_inputs=_serialize(execution_inputs or {})) if submission is None else _serialize(submission)
    selected = current.get('registered_input')
    if pending['field'] == 'parameters':
        from agent.orchestration.registry import build_default_tool_registry
        registry = build_default_tool_registry()
        original = dict(execution_inputs=_serialize(origin.admitted['inputs']))
        if origin.submission is not None:
            original.update(_serialize(origin.submission))
        if selected is not None and selected != original.get('registered_input'):
            return None
        permitted = {_parameter_input(registry, pending['tool'], item['argument'])[0]
                     for item in scientific_parameters(registry) if item['tool'] == pending['tool']}
        for key, value in current.get('execution_inputs', {}).items():
            if key not in permitted and original['execution_inputs'].get(key) != value:
                return None
        return dict(**pending, predecessor_turn_id=prior.turn_id)
    original = _serialize(execution_submission(origin))
    if selected is not None and any(selected.get(k) != original['registered_input'][k]
                                   for k in ('resource_id', 'record_sha256')):
        return None
    values = current.get('execution_inputs', {})
    for key, value in values.items():
        if key in {'species', 'resolution'} or (key in {'checkpoint_path', 'expected_resource_identity'}
                                and key not in original['execution_inputs']):
            continue
        if key not in original['execution_inputs'] or original['execution_inputs'][key] != value:
            return None
    return dict(**pending, predecessor_turn_id=prior.turn_id)


def validate_prerequisite(interaction, preceding):
    """Validate optional Session references without reading providers or science."""
    from .session_state import SessionError
    _validate_resolution_record(interaction)
    _validate_argument_record(interaction)
    captured = _serialize(interaction.snapshot)
    expected = capture_prerequisite(preceding, interaction.base_revision_id,
        interaction.base_generation, interaction.submission,
        captured.get('dialogue', {}).get('requested_predecessor'))
    reference = captured.get('prerequisite')
    if reference is not None and reference != expected:
        raise SessionError('Pending declaration capture changed.')
    pending = interaction.prerequisite
    if ((reference or {}).get('field') == 'parameters' or (pending or {}).get('field') == 'parameters'
            or 'parameter_continuation' in (interaction.admitted or {})
            or (interaction.admitted or {}).get('continuation', {}).get('field') == 'parameters'):
        _validate_parameter_prerequisite(interaction, preceding)
        return
    if pending is not None:
        if pending['origin_turn_id'] == interaction.turn_id:
            admitted = interaction.admitted or {}
            submission = execution_submission(interaction) or {}
            if (admitted.get('kind') != 'execute' or admitted.get('operation') != 'plan'
                    or submission.get('registered_input', {}).get('composition') != 'h5ad-science.v1'
                    or 'species' in submission.get('execution_inputs', {})
                    or _serialize(admitted.get('inputs')) != _serialize(submission.get('execution_inputs'))):
                raise SessionError('Pending declaration has no original H5AD request.')
        elif (reference is None or _serialize(pending) != {k: reference[k]
                for k in ('origin_turn_id', 'field', 'resource_context_sha256')}
                or interaction.admitted is None or interaction.admitted.get('kind') != 'clarify'
                or interaction.admitted.get('reason') not in
                    {'missing_species', 'ambiguous_species', 'unsupported_species', 'conflicting_species',
                     'conflicting_scientific_parameter', 'invalid_parameter_value', 'ungrounded_operand'}):
            raise SessionError('Pending declaration lost its originating request.')
    admitted = interaction.admitted or {}
    continuation = admitted.get('continuation')
    if continuation is not None:
        if (reference is None or set(continuation) != {'origin_turn_id', 'species'}
                or continuation['origin_turn_id'] != reference['origin_turn_id']
                or continuation['species'] not in ('human', 'mouse')
                or admitted.get('kind') != 'execute' or admitted.get('operation') != 'plan'
                or admitted.get('inputs', {}).get('species') != continuation['species']):
            raise SessionError('Invalid scientific clarification continuation.')
        origin = next(i for i in preceding if i.turn_id == continuation['origin_turn_id'])
        original = execution_submission(origin)
        metadata = admitted.get('registered_input', {})
        if ('resolution' in admitted.get('inputs', {}) and 'argument' not in admitted and 'arguments' not in admitted
                and 'resolution' not in original['execution_inputs']):
            from agent.orchestration.registry import build_default_tool_registry
            name, specification = _resolution_input(build_default_tool_registry())
            received = (interaction.submission or {}).get('execution_inputs', {})
            try:
                specification.validate(name, received.get(name))
                specification.validate(name, admitted['inputs'][name])
                if received[name] != admitted['inputs'][name]:
                    raise ValueError('Continuation changed received resolution.')
            except (ValueError, TypeError) as exc:
                raise SessionError('Missing or invalid continuation resolution declaration.') from exc
        if 'argument' in origin.admitted:
            from agent.orchestration.registry import build_default_tool_registry
            name, specification = _resolution_input(build_default_tool_registry())
            try:
                specification.validate(name, admitted.get('inputs', {}).get(name))
            except (ValueError, TypeError) as exc:
                raise SessionError('Invalid retained Leiden resolution.') from exc
        from agent.orchestration.registry import build_default_tool_registry
        registry = build_default_tool_registry()
        changed = {_parameter_input(registry, d['tool'], d['argument'])[0] for d in admitted.get('arguments', ())}
        if (admitted.get('tool') != origin.admitted['tool']
                or admitted.get('revision_id') != interaction.base_revision_id
                or metadata.get('composition') != 'h5ad-science.v1'
                or metadata.get('tool_name') != original['registered_input']['tool_name']
                or any(metadata.get(k) != original['registered_input'][k]
                       for k in ('resource_id', 'record_sha256'))
                or any(admitted['inputs'].get(k) != _serialize(v)
                       for k, v in original['execution_inputs'].items() if k not in changed)):
            raise SessionError('Continuation changed its original scientific request.')


def prerequisite_public(interaction, preceding):
    reference = interaction.snapshot.get('prerequisite')
    if reference is None:
        return None
    origin = next(i for i in preceding if i.turn_id == reference['origin_turn_id'])
    if reference['field'] == 'parameters':
        return dict(handle='@parameters', field='parameters', tool=reference['tool'],
                    missing=list(reference['missing']), objective=origin.utterance)
    return dict(handle='@species', field='species', choices=['human', 'mouse'],
                objective=origin.utterance)


def admit_species(sessions, interaction, decision, resources):
    """Bind one semantic species answer to exact captured inputs and policy."""
    from .local_resources import LocalResourceAdmission, ResourceAdmissionError, select_epizoo_resource
    state = sessions.load(sessions._interaction_session_id)
    reference = interaction.snapshot.get('prerequisite')
    if (decision.pending != '@species' or reference is None
            or state.generation != interaction.base_generation
            or state.active_revision_id != interaction.base_revision_id
            or state.interactions[-1].turn_id != interaction.turn_id):
        raise IntentError('invalid_prerequisite')
    validate_prerequisite(interaction, state.interactions[:-1])
    origin = next(i for i in state.interactions if i.turn_id == reference['origin_turn_id'])
    original = _serialize(execution_submission(origin))
    request_id = 'turn-' + digest(dict(session=state.session_id, turn=origin.turn_id))
    if origin.admitted['request_id'] != request_id:
        raise IntentError('invalid_prerequisite')
    try:
        run = sessions._application.run_store.load(request_id + ':run')
    except (ValueError, RuntimeError, OSError) as exc:
        raise IntentError('invalid_prerequisite') from exc
    if (run.request.prompt != origin.utterance or run.steps
            or any(run.request.inputs.get(k) != v for k, v in original['execution_inputs'].items())
            or not any(_missing_species(error, original) for error in run.errors)):
        raise IntentError('invalid_prerequisite')
    current = {} if interaction.submission is None else _serialize(interaction.submission)
    typed_species = current.get('execution_inputs', {}).get('species')
    if typed_species is not None and typed_species != decision.species:
        raise IntentError('conflicting_species')
    if resource_context_digest(resources) != reference['resource_context_sha256']:
        raise ResourceAdmissionError('EPIZOO_RESOURCE_SELECTION_INVALID')
    owner = LocalResourceAdmission(sessions._application._workspace, registry=sessions._application.registry)
    # Verify the original registration before composing; never substitute the
    # current UI selection for the pending source.
    binding = owner._submission_binding(original)
    owner.validate_binding(binding)
    declarations = {k: v for k, v in original['execution_inputs'].items() if k != 'input_path'}
    argument = sessions._application.registry.get('epizoo_embed_cells').required_arguments['species']
    argument.validate('species', decision.species)
    declarations['species'] = decision.species
    original_pins = declarations.get('expected_resource_identity')
    current_inputs = current.get('execution_inputs', {})
    resolution_name, resolution_spec = _resolution_input(sessions._application.registry)
    if resolution_name in current_inputs:
        try:
            resolution_spec.validate(resolution_name, current_inputs[resolution_name])
        except (ValueError, TypeError) as exc:
            raise IntentError('invalid_parameter_value') from exc
        if resolution_name in declarations and declarations[resolution_name] != current_inputs[resolution_name]:
            raise IntentError('conflicting_scientific_parameter')
        declarations[resolution_name] = current_inputs[resolution_name]
    if decision.argument is not None:
        declarations = _bind_resolution(sessions._application.registry, interaction.utterance,
                                       declarations, decision.argument)
    if decision.arguments:
        _bind_arguments(sessions._application.registry, interaction.utterance, current_inputs, decision.arguments)
        declarations = _bind_arguments(sessions._application.registry, interaction.utterance,
                                       declarations, decision.arguments, replace_existing=True)
    changed = {_parameter_input(sessions._application.registry, d.tool, d.argument)[0] for d in decision.arguments}
    current_pins = current_inputs.get('expected_resource_identity')
    selected_id = (original_pins or current_pins or {}).get('resource_id')
    declarations, selection_error = select_epizoo_resource(declarations, selected_id, resources)
    # Selection cannot replace an already validated declaration, including an
    # explicit model chosen before the species was supplied.
    if (any(declarations.get(k) != value for k, value in original['execution_inputs'].items()
            if k != 'input_path' and k not in changed)
            or any(declarations.get(k) != current_inputs[k]
                   for k in ('checkpoint_path', 'expected_resource_identity') if k in current_inputs)):
        raise ResourceAdmissionError('EPIZOO_RESOURCE_SELECTION_INVALID')
    completed = owner.compose_h5ad(binding.resource_id, declarations,
                                  resource_selection_error=selection_error)
    admitted = dict(kind='execute', operation='plan', tool=origin.admitted['tool'],
        revision_id=interaction.base_revision_id,
        request_id='turn-' + digest(dict(session=state.session_id, turn=interaction.turn_id)),
        inputs=_serialize(completed.execution_inputs), registered_input=completed.attribution(),
        continuation=dict(origin_turn_id=origin.turn_id, species=decision.species))
    if decision.argument is not None:
        admitted['argument'] = asdict(decision.argument)
    if decision.arguments:
        admitted['arguments'] = [asdict(d) for d in decision.arguments]
    return admitted


def execution_submission(interaction):
    """Use the admitted binding for continuation, preserving the received turn."""
    admitted = interaction.admitted or {}
    if 'continuation' not in admitted and 'argument' not in admitted and 'arguments' not in admitted:
        return interaction.submission
    submission = dict(_serialize(interaction.submission) or {}, execution_inputs=_serialize(admitted['inputs']))
    if 'registered_input' in admitted:
        submission['registered_input'] = _serialize(admitted['registered_input'])
    return submission


def is_execution_command(utterance):
    # Retained only for M17.2 candidate command evidence. Generic admission
    # follows the typed interpreter decision without classifying language again.
    command = re.sub(r'^(?:please\s+|(?:can|could|would) you\s+)', '', utterance.strip(), flags=re.I)
    return bool(re.match(r'^(?:run|compute|perform|test|reannotate|annotate|inspect|build|prepare|evaluate|adapt|transfer|select)\b', command, re.I))


def admit(sessions, interaction, decision, inputs, *, epizoo_resources=()):
    if (decision.base != 'current' or decision.delta is not None
            or decision.target not in sessions._application.registry.names()):
        raise IntentError('unsupported_intent')
    values = {} if inputs is None else _serialize(freeze_json_mapping(inputs, 'execution_inputs'))
    admitted = dict(kind='execute', operation='plan', tool=decision.target, revision_id=interaction.base_revision_id,
                request_id='turn-' + digest(dict(session=sessions._interaction_session_id, turn=interaction.turn_id)),
                inputs=values)
    if _arguments(decision):
        values = _bind_arguments(sessions._application.registry, interaction.utterance, values, _arguments(decision))
        admitted.update(inputs=values)
        if decision.argument is not None:
            admitted['argument'] = asdict(decision.argument)
        if decision.arguments:
            admitted['arguments'] = [asdict(d) for d in decision.arguments]
        if interaction.submission is None:
            names = {_parameter_input(sessions._application.registry, d.tool, d.argument)[0]
                     for d in _arguments(decision)}
            if decision.arguments:
                admitted['arguments_added_inputs'] = sorted(names - set(inputs or {}))
            for field in ('argument', 'arguments'):
                if field in admitted:
                    admitted[field + '_received_inputs_sha256'] = digest({} if inputs is None else _serialize(inputs))
        elif 'registered_input' in interaction.submission:
            from .local_resources import LocalResourceAdmission, ResourceAdmissionError
            owner = LocalResourceAdmission(sessions._application._workspace, registry=sessions._application.registry)
            original = owner._submission_binding(interaction.submission)
            owner.validate_binding(original)
            if original.tool_name == 'inspect_scATAC' and original.composition in (None, 'h5ad-science.v1'):
                selection_error = original.resource_selection_error
                if any(isinstance(d, ScalarChoiceArgument) for d in decision.arguments):
                    values, selection_error = _select_declared_resource(values, epizoo_resources)
                    admitted['resource_selection'] = dict(catalog=[asdict(r) for r in epizoo_resources],
                        resource_context_sha256=resource_context_digest(epizoo_resources))
                completed = owner.compose_h5ad(original.resource_id,
                    {k: v for k, v in values.items() if k != 'input_path'},
                    resource_selection_error=selection_error)
                admitted.update(inputs=_serialize(completed.execution_inputs), registered_input=completed.attribution())
            else:
                # Other registered-input categories keep their exact resource
                # attribution; scientific scalar declarations change no identity.
                admitted['registered_input'] = _serialize(interaction.submission['registered_input'])
    return admitted


def execute(sessions, interaction, admitted, model, *, epizoo_resources=()):
    from .turns import _AdmittedPlanner, _update, TurnOutcome
    from .service import ResearchAgentApplication
    from .scientific_dialogue import _json
    from .local_resources import LocalResourceAdmission, ResourceAdmissionError
    app = sessions._application
    sid = sessions._interaction_session_id
    selected = 'selected_candidate' in admitted
    intent, context = interaction.utterance, None
    if 'continuation' in admitted:
        state = sessions.load(sid)
        origin = next(i for i in state.interactions if i.turn_id == admitted['continuation']['origin_turn_id'])
        intent = origin.utterance
    interaction = replace(interaction, admitted=admitted, submission=execution_submission(
        replace(interaction, admitted=admitted)))
    if selected:
        from .guidance_selection import execution_context, planning_intent
        context = execution_context(sessions, interaction, admitted)
        intent = planning_intent(sessions, interaction, admitted, context)
    request = AgentRequest(admitted['request_id'], intent, admitted['inputs'])
    registered = interaction.submission is not None and 'registered_input' in interaction.submission
    resources = LocalResourceAdmission(app._workspace, registry=app.registry) if registered else None
    resource_failure = None
    parameter_failure = None

    def accept(effective, plan):
        nonlocal resource_failure, parameter_failure
        if admitted['tool'] not in {s.tool_name for s in plan.steps}:
            raise IntentError('planning_failed')
        try:
            _check_parameter_plan(app.registry, plan, admitted, sessions.load(sid).interactions)
        except IntentError as exc:
            if exc.reason == 'ambiguous_parameter':
                parameter_failure = exc
                from agent.orchestration.planner import PlannerError
                from agent.schemas import ErrorCategory
                raise PlannerError('AMBIGUOUS_PARAMETER_SCOPE',
                    'Please identify one unambiguous operation for the scoped scientific parameters.',
                    category=ErrorCategory.USER_INPUT_ERROR) from exc
            raise
        offered = [dict(step_id=s.step_id, tool=s.tool_name,
                        outputs=list(app.registry.get(s.tool_name).result_contract.required_fields)) for s in plan.steps]
        schema = _object(dict(outputs={'type':'array','minItems':1,'maxItems':32,
            'items':_object(dict(name={'type':'string'}, step_id={'type':'string'}, output_key={'type':'string'}))}))
        raw_choice = model.complete(prompt=json.dumps(dict(output_selection_schema_version=1,
            question=intent, steps=offered,
            instructions=(
                'Select a relevant subset of 1 to 32 outputs of this actual plan to retain in the new revision. '
                'Choose results that satisfy the user request; do not automatically enumerate every field from every step. '
                'Preserve multiple legitimate outputs when the user explicitly requests them. '
                'For each selection, copy step_id exactly from an offered step and output_key exactly from that same step\'s outputs. '
                'Never invent or repair step identities, output keys, scientific products or output relationships. '
                'Each name is a nonblank string label and must be unique across all selections for the new revision, '
                'including when different steps expose the same output_key. Names need not equal output keys; '
                'use meaningful labels that distinguish the selected results. Do not infer or append scientific steps.'))),
            response_schema=schema)
        try:
            choice = _json(raw_choice)
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
        except ValueError as exc:
            from agent.orchestration.planner import PlannerError
            from agent.schemas import ErrorCategory
            raise PlannerError('PLANNER_OUTPUT_INVALID', 'The model returned an invalid output selection.',
                               category=ErrorCategory.INTERNAL_AGENT_ERROR) from exc
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
            if 'continuation' in admitted and (
                    state.active_revision_id != interaction.base_revision_id
                    or state.generation != interaction.base_generation
                    or state.interactions[-1].turn_id != interaction.turn_id):
                raise SessionConflictError('Pending request changed before submission.')
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
    error = resource_failure or result.error or next(iter(result.run_result.errors), None)
    if parameter_failure is not None:
        from .turn_decisions import Clarify
        _update(sessions, interaction.turn_id, status='clarification')
        return TurnOutcome('clarify', 'clarification', Clarify('ambiguous_parameter',
            parameter_scope_choices(admitted, sessions.load(sid).interactions)))
    # Selected guidance candidates retain M17.2's missing-input failure contract;
    # prerequisite continuation captures an ordinary utterance, not candidate context.
    if not selected and _missing_species(error, interaction.submission):
        # Presentation of a compiler-authored missing-port fact, not a second
        # interpretation of the utterance or a repaired scientific plan.
        from .turn_decisions import Clarify
        pending = dict(origin_turn_id=interaction.turn_id, field='species',
                       resource_context_sha256=resource_context_digest(epizoo_resources))
        _update(sessions, interaction.turn_id, status='clarification', prerequisite=pending)
        return TurnOutcome('clarify', 'clarification', Clarify('missing_species', ('human', 'mouse'), True))
    missing = None if selected else _missing_parameters(error, admitted, app.registry)
    if missing is not None:
        from .turn_decisions import Clarify
        tool_name, names = missing
        pending = dict(origin_turn_id=interaction.turn_id, field='parameters',
            binding_turn_id=interaction.turn_id, tool=tool_name, missing=names,
            resource_context_sha256=resource_context_digest(epizoo_resources))
        _update(sessions, interaction.turn_id, status='clarification', prerequisite=pending)
        return TurnOutcome('clarify', 'clarification', Clarify('missing_parameter_value', tuple(names), True))
    if (registered and interaction.submission['registered_input'].get('composition') == 'bam-science.v1'
            and error is not None and error.code in {'MISSING_REQUIRED_SOURCE', 'MISSING_REQUIRED_BINDING'}
            and error.details.get('tool_name') == 'prepare_scATAC_bam_fragments'):
        # Present only the compiler's exact missing prerequisite. Inspection has
        # no producer requirements, and this does not add or repair plan inputs.
        prerequisites = {
            'library_context': (('library_context_path', 'library_context_sha256'), 'BAM_LIBRARY_CONTEXT_REQUIRED'),
            'reference': (('reference_bundle_path', 'reference_bundle_sha256'), 'BAM_REFERENCE_REQUIRED'),
            'source_profile': (('source_profile',), 'BAM_PROFILE_REQUIRED'),
            'intake': (('intake_manifest_path', 'intake_manifest_sha256'), 'BAM_INTAKE_REQUIRED'),
        }
        missing = prerequisites.get(error.details.get('target_port'))
        if missing is not None and any(key not in interaction.submission['execution_inputs'] for key in missing[0]):
            selection_error = interaction.submission['registered_input'].get('resource_selection_error')
            code = 'BAM_RESOURCE_AMBIGUOUS' if selection_error == 'BAM_RESOURCE_AMBIGUOUS' else missing[1]
            error = ResourceAdmissionError(code).error
    if (registered and interaction.submission['registered_input'].get('composition') == 'fastq-science.v1'
            and error is not None and error.code in {'MISSING_REQUIRED_SOURCE', 'MISSING_REQUIRED_BINDING'}
            and error.details.get('tool_name') == 'prepare_scATAC_fragments'):
        prerequisites = {
            'library_context': (('library_context_path', 'library_context_sha256'), 'FASTQ_LIBRARY_CONTEXT_REQUIRED'),
            'reference': (('reference_bundle_path', 'reference_bundle_sha256'), 'FASTQ_REFERENCE_REQUIRED'),
            'intake': (('intake_manifest_path', 'intake_manifest_sha256'), 'FASTQ_INTAKE_REQUIRED'),
        }
        missing = prerequisites.get(error.details.get('target_port'))
        if missing is not None and any(key not in interaction.submission['execution_inputs'] for key in missing[0]):
            selection_error = interaction.submission['registered_input'].get('resource_selection_error')
            code = 'FASTQ_RESOURCE_AMBIGUOUS' if selection_error == 'FASTQ_RESOURCE_AMBIGUOUS' else missing[1]
            error = ResourceAdmissionError(code).error
    _update(sessions, interaction.turn_id, status='failed')
    return TurnOutcome('execute', 'failed', error=error,
        text=error.message if error is not None else
        'The scientific execution request could not form a valid plan from the supplied inputs.')


def completion_outcome(state, turn_id, *, error=None):
    """Render the persisted completion, including explicit post-crash recovery."""
    from .turns import TurnOutcome
    turn = state.turn(turn_id)
    status = turn.status
    active = status == 'activated' and state.active_revision_id == turn.revision_id
    interaction = next((i for i in state.interactions if i.turn_id == turn_id), None)
    continuation = {} if interaction is None or interaction.admitted is None else interaction.admitted.get('continuation', {})
    declaration = '' if 'species' not in continuation else 'The dataset is declared as ' + continuation['species'] + '. '
    return TurnOutcome('execute', status,
        text=declaration + 'The request followed the registered scientific execution path. '
             + ('Its accepted outputs form the active revision.' if active
                else 'Its accepted revision is historical; another revision is active.' if status == 'activated'
                else 'Its persisted execution status is ' + status + '.'), error=error)
