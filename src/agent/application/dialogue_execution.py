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
    _input_roles, _scientific_arguments, admit_leiden_resolution, is_exact_string_argument, _object, _shape)


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
    exact_string = argument is not None and is_exact_string_argument(argument)
    if (argument is None or argument.planning is None or not argument.planning.scientific_parameter
            or not (choice or exact_string or any(t in argument.accepted_types for t in (int, float)))
            or tool.semantic_planning is None):
        raise IntentError('unsupported_intent')
    if exact_string:
        ports = [port for port in tool.semantic_planning.consumer_ports
                 if any(member.field_name == argument_name for member in port.members)]
        if (len(ports) != 1 or len(ports[0].members) != 1
                or ports[0].accepted_upstream_types or len(ports[0].request_sources) != 1):
            raise IntentError('ambiguous_parameter')
        source = ports[0].request_sources[0]
        if len(source.members) != 1 or source.members[0].name != ports[0].members[0].name:
            raise IntentError('ambiguous_parameter')
        return source.members[0].input_name, argument
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
            elif is_exact_string_argument(spec):
                item.update(exact_string=True, provenance_role=(spec.planning.provenance_role.value if spec.planning.provenance_role is not None else None))
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
    from .local_resources import (LocalResourceAdmission, QualifiedEpiZooResource, RegisteredInput,
        RegisteredInputGroup, group_role_inputs, qualified_epizoo_resources)
    admitted = _serialize(interaction.admitted)
    received = _serialize(interaction.submission)['execution_inputs']
    original = interaction.submission.get('registered_input', {})
    if original.get('composition') == 'scientific-role-selection.v1':
        binding = LocalResourceAdmission._submission_binding(interaction.submission)
        roles = [asdict(item) for item in _input_roles(admitted.get('roles', []))]
        values = _bind_arguments(registry, interaction.utterance,
            group_role_inputs(binding, roles, registry), declarations)
        expected = RegisteredInputGroup(binding.members, values, roles=roles,
            resource_choice=binding.resource_choice).attribution()
        if _serialize(admitted.get('registered_input')) != expected:
            raise ValueError('Scientific input role attribution changed.')
        return values
    values = _bind_arguments(registry, interaction.utterance, received, declarations)
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
            values, 'h5ad-science.v1', error, original.get('resource_choice'))
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
                        for k in ('resource_id', 'record_sha256', 'tool_name', 'resource_choice')):
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
            if 'roles' not in admitted:
                raise SessionError('Missing persisted scientific argument declaration.')
            try:
                from agent.orchestration.registry import build_default_tool_registry
                if _replay_initial_inputs(build_default_tool_registry(), interaction, ()) != _serialize(admitted['inputs']):
                    raise ValueError('Role binding changed inputs.')
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                raise SessionError('Invalid persisted scientific input roles.') from exc
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
                    if any(before.get(k) != after.get(k)
                           for k in ('resource_id', 'record_sha256', 'tool_name', 'resource_choice')):
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


def _validate_input_role_record(interaction):
    """Replay captured associations without current sources or scientific reads."""
    from .session_state import SessionError
    admitted = interaction.admitted or {}
    received = (interaction.submission or {}).get('registered_input', {})
    grouped = received.get('composition') == 'scientific-role-selection.v1'
    if not grouped and 'roles' not in admitted:
        return
    if 'continuation' in admitted or 'parameter_continuation' in admitted:
        # Exact origin/last-binding checks are owned by parameter continuation.
        return
    if admitted.get('operation') != 'plan':
        if 'roles' in admitted:
            raise SessionError('Scientific input roles require admitted planning.')
        return
    try:
        from agent.orchestration.registry import build_default_tool_registry
        if not grouped or not _input_roles(_serialize(admitted.get('roles', []))):
            raise ValueError('Missing captured scientific input roles.')
        declarations = _scientific_arguments(_serialize(admitted.get('arguments', [])))
        if 'argument' in admitted:
            declarations += (LeidenResolution(**_serialize(admitted['argument'])),)
        expected = _replay_initial_inputs(build_default_tool_registry(), interaction, declarations)
        if expected != _serialize(admitted['inputs']):
            raise ValueError('Scientific role inputs differ from captured source identities.')
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise SessionError('Invalid persisted scientific input roles.') from exc


def _parameter_source(reference, preceding):
    """Resolve the exact last accepted pending binding, never scan for a value."""
    origin = next(i for i in preceding if i.turn_id == reference['origin_turn_id'])
    source = next(i for i in preceding if i.turn_id == reference['binding_turn_id'])
    return origin, source, _serialize(source.admitted['inputs'])


def _repair_declarations(raw, utterance, registry):
    """Retain only valid declarations from a structurally valid new objective."""
    import json
    from .turn_decisions import (parse_decision, _ground_declaration,
        DeclarationGroundingError, Execute)
    try:
        parse_decision(raw, utterance=utterance)
    except DeclarationGroundingError:
        pass  # The parser has checked the complete decision shape.
    value = json.loads(raw)['decision']
    if value['kind'] not in {'execute_plan', 'request_resources'}:
        raise IntentError('invalid_prerequisite')
    good, rejected = [], []
    declarations = ([value['argument']] if value.get('argument') is not None else []) + value.get('arguments', [])
    for item in declarations:
        # Registry identity is checked even when the literal cannot be located.
        _parameter_input(registry, item['tool'], item['argument'])
        try:
            grounded = _ground_declaration(item, utterance)
            _bind_arguments(registry, utterance, {}, _scientific_arguments([grounded]))
            good.append(grounded)
        except IntentError as exc:
            if exc.reason not in {'ungrounded_operand', 'ambiguous_parameter', 'invalid_parameter_value'}:
                raise
            rejected.append((item['tool'], item['argument']))
    if not rejected or len({tool for tool, _ in rejected}) != 1:
        raise IntentError('invalid_prerequisite')
    clean = dict(value, arguments=good)
    clean.pop('argument', None)
    clean['kind'] = 'execute_plan'
    safe = parse_decision(json.dumps(dict(turn_schema_version=1, decision=clean)))
    if not isinstance(safe, Execute):
        raise IntentError('invalid_prerequisite')
    return safe, rejected


def request_parameter_repair(sessions, interaction, raw, inputs, resources):
    """Capture a question, never execute a rejected interpreter decision."""
    if interaction.submission is None:
        raise IntentError('invalid_prerequisite')
    import json
    from .resource_continuation import source_kind
    from .scientific_guidance import references_candidate
    from .guidance_selection import candidates
    state = sessions.load(sessions._interaction_session_id)
    refs = candidates(state.interactions, interaction.snapshot)
    predecessor = refs[0]['origin_turn_id'] if refs else interaction.snapshot['dialogue']['predecessor']
    prior = next((i for i in state.interactions if i.turn_id == predecessor), None)
    if references_candidate(interaction, prior):
        raise IntentError('unsupported_intent')
    if (json.loads(raw)['decision']['kind'] == 'request_resources'
            and source_kind(interaction.submission) is None):
        raise IntentError('invalid_prerequisite')
    registry = sessions._application.registry
    safe, rejected = _repair_declarations(raw, interaction.utterance, registry)
    admitted = admit(sessions, interaction, safe, inputs, epizoo_resources=resources)
    tool = rejected[0][0]
    missing = sorted(name for _, name in rejected)
    # A rejected declaration must not quietly fall back to a received/default value.
    if any(_parameter_input(registry, tool, name)[0] in admitted['inputs'] for name in missing):
        raise IntentError('conflicting_scientific_parameter')
    admitted['parameter_rejection'] = raw
    pending = dict(origin_turn_id=interaction.turn_id, binding_turn_id=interaction.turn_id,
        field='parameters', tool=tool, missing=missing,
        resource_context_sha256=resource_context_digest(resources))
    return admitted, pending


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
            if 'parameter_rejection' in admitted:
                safe, rejected = _repair_declarations(admitted['parameter_rejection'], interaction.utterance, registry)
                if (safe.target != admitted['tool'] or any(t != tool.name for t, _ in rejected)
                        or sorted(n for _, n in rejected) != list(pending['missing'])
                        or any(_parameter_input(registry, t, n)[0] in values for t, n in rejected)
                        or [asdict(d) for d in safe.arguments] != _serialize(admitted.get('arguments', []))
                        or _replay_initial_inputs(registry, interaction, safe.arguments) != _serialize(values)):
                    raise ValueError('Rejected declaration capture changed.')
                return
            required = {item['argument'] for item in scientific_parameters(registry)
                        if item['tool'] == tool.name and item['required'] and 'choices' not in item}
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
        expected_metadata = (execution_submission(origin) or {}).get('registered_input')
        require_metadata = (admitted.get('kind') == 'execute' or 'registered_input' in admitted
            or (_received_origin(origin, preceding).admitted or {}).get('kind') == 'request_roles')
        if require_metadata and _serialize(admitted.get('registered_input')) != _serialize(expected_metadata):
            raise ValueError('Parameter continuation changed registered source.')
        if admitted.get('roles', ()) != (origin.admitted or {}).get('roles', ()):
            raise ValueError('Parameter continuation changed scientific input roles.')
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
    if 'parameter_rejection' not in origin.admitted:
        stored = sessions._application.run_store.load(origin.admitted['request_id'] + ':run')
        original_error = next((e for e in stored.errors
            if _missing_parameters(e, origin.admitted, sessions._application.registry) ==
               (reference['tool'], list(origin.prerequisite['missing']))), None)
        if (stored.steps or stored.request.prompt != _execution_objective(origin, state.interactions) or original_error is None
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
    original_submission = execution_submission(origin) or {}
    if 'registered_input' in original_submission:
        from .local_resources import LocalResourceAdmission
        original_metadata = (origin.submission or {}).get('registered_input')
        if original_metadata is None:
            original_metadata = _received_origin(origin, state.interactions).submission['registered_input']
        current_metadata = (interaction.submission or {}).get('registered_input')
        if (current_metadata is not None
                and (original_metadata.get('composition') == 'scientific-role-selection.v1'
                     or 'resource_choice' in original_metadata or 'resource_choice' in current_metadata)
                and _serialize(current_metadata) != _serialize(original_metadata)):
            raise IntentError('invalid_prerequisite')
        LocalResourceAdmission(sessions._application._workspace, registry=sessions._application.registry).validate_submission(
            execution_submission(origin))
    missing = [name for name in origin.prerequisite['missing']
               if _parameter_input(sessions._application.registry, reference['tool'], name)[0] not in values]
    continuation = dict(origin_turn_id=origin.turn_id, binding_turn_id=source.turn_id, field='parameters')
    admitted = dict(kind='execute', operation='plan', tool=origin.admitted['tool'],
        revision_id=interaction.base_revision_id,
        request_id='turn-' + digest(dict(session=state.session_id, turn=interaction.turn_id)),
        inputs=values, arguments=[asdict(d) for d in decision.arguments], continuation=continuation)
    if origin.admitted.get('context') == 'current':
        admitted['context'] = 'current'
    if 'registered_input' in original_submission:
        admitted['registered_input'] = _serialize(original_submission['registered_input'])
    if origin.admitted.get('roles'):
        admitted['roles'] = _serialize(origin.admitted['roles'])
    if missing:
        admitted = dict(kind='clarify', reason='missing_parameter_value', choices=missing, value_required=True,
            inputs=values, arguments=admitted['arguments'], parameter_continuation=continuation)
        if origin.admitted.get('context') == 'current':
            admitted['context'] = 'current'
        if origin.admitted.get('roles'):
            admitted['roles'] = _serialize(origin.admitted['roles'])
        if 'registered_input' in original_submission:
            admitted['registered_input'] = _serialize(original_submission['registered_input'])
    if interaction.submission is None:
        admitted['parameter_received_inputs_sha256'] = digest(current)
    return admitted


def _parameter_identities(admitted, interactions):
    """Read only explicitly linked declarations and originally missing identities."""
    identities = set()
    records = []
    queue = [admitted]
    by_id = {i.turn_id: i for i in interactions}
    visited = set()
    while queue:
        record = queue.pop()
        records.append(record)
        continuation = record.get('parameter_continuation', record.get('continuation'))
        if continuation is None:
            continue
        origin = by_id[continuation['origin_turn_id']]
        if continuation.get('field') == 'parameters':
            identities.update((origin.prerequisite['tool'], name) for name in origin.prerequisite['missing'])
        for pointer in dict.fromkeys(continuation.get(k) for k in ('origin_turn_id', 'binding_turn_id')):
            if pointer is not None and pointer not in visited:
                visited.add(pointer)
                queue.append(by_id[pointer].admitted)
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


def _role_source(reference, preceding):
    origin = next(i for i in preceding if i.turn_id == reference['origin_turn_id'])
    source = next(i for i in preceding if i.turn_id == reference['binding_turn_id'])
    return origin, source


def _role_binding(origin, source, interaction):
    """Only explicit append of registered files may fill captured missing roles."""
    from .local_resources import LocalResourceAdmission
    captured = origin.submission if source is None else dict(
        execution_inputs=_serialize(source.admitted['inputs']),
        registered_input=_serialize(source.admitted['registered_input']))
    original = replace(LocalResourceAdmission._submission_binding(captured), roles=())
    selected = (interaction.submission or {}).get('registered_input')
    if selected is None:
        return original
    current = LocalResourceAdmission._submission_binding(interaction.submission)
    required = origin.admitted['required_roles']
    if (type(current) is not type(original) or current.composition != original.composition
            or not len(original.members) <= len(current.members) <= len(required)
            or current.members[:len(original.members)] != original.members
            or current.resource_choice != original.resource_choice
            or current.roles):
        raise IntentError('invalid_prerequisite')
    if any(key not in original.execution_inputs or original.execution_inputs[key] != value
           for key, value in current.execution_inputs.items()):
        raise IntentError('invalid_prerequisite')
    return replace(original, members=current.members)


def _role_completion(registry, origin, source, interaction, roles, declarations):
    """Replay explicit associations and declarations against one captured group."""
    from .local_resources import (LocalResourceAdmission, RegisteredInputGroup,
        validate_group_roles, group_role_inputs)
    binding = _role_binding(origin, source, interaction)
    required = _serialize(origin.admitted['required_roles'])
    previous = () if source is None else _input_roles(_serialize(source.admitted['roles']))
    merged = { (r.tool, r.port): r for r in previous }
    for role in roles:
        key = (role.tool, role.port)
        if key in merged and merged[key] != role:
            raise IntentError('invalid_prerequisite')
        merged[key] = role
    captured_roles = validate_group_roles(binding, tuple(merged.values()), registry, required)
    base = (_serialize(binding.execution_inputs) if source is None else _serialize(source.admitted['inputs']))
    values = _bind_arguments(registry, interaction.utterance, base, declarations)
    missing = [r for r in required if (r['tool'], r['port']) not in merged]
    metadata = binding.attribution()
    metadata['roles'] = [asdict(r) for r in captured_roles]
    if not missing:
        validate_group_roles(binding, captured_roles, registry, required, complete=True)
        values = dict(group_role_inputs(binding, [asdict(r) for r in captured_roles], registry), **values)
        metadata = RegisteredInputGroup(binding.members, values, roles=[asdict(r) for r in captured_roles],
            resource_choice=binding.resource_choice).attribution()
    return values, [asdict(r) for r in captured_roles], missing, metadata


def _validate_role_prerequisite(interaction, preceding):
    """Check durable pending role bindings using captured bytes only."""
    from .session_state import SessionError
    from agent.orchestration.registry import build_default_tool_registry
    registry = build_default_tool_registry()
    admitted = interaction.admitted or {}
    pending = interaction.prerequisite if (interaction.prerequisite or {}).get('field') == 'roles' else None
    reference = interaction.snapshot.get('prerequisite')
    try:
        initial = admitted.get('kind') == 'request_roles'
        if initial:
            if (reference is not None and pending is None or pending is None
                    or pending['field'] != 'roles' or pending['origin_turn_id'] != interaction.turn_id
                    or pending['binding_turn_id'] != interaction.turn_id
                    or admitted['tool'] not in registry.names()
                    or pending['tool'] != admitted['tool']
                    or _serialize(pending['required_roles']) != _serialize(admitted['required_roles'])
                    or admitted['revision_id'] != interaction.base_revision_id):
                raise ValueError('No captured role request.')
            required = admitted['required_roles']
            if admitted['tool'] not in {r['tool'] for r in required}:
                raise ValueError('Role request has no registered target.')
            origin, source = interaction, None
            additions = _input_roles(_serialize(admitted['roles']))
        else:
            if reference is None or reference['field'] != 'roles':
                raise ValueError('No captured pending role request.')
            origin, source = _role_source(reference, preceding)
            continuation = admitted.get('continuation')
            if continuation is None:
                wanted = {k: _serialize(v) for k, v in reference.items() if k != 'predecessor_turn_id'}
                if pending is not None and (admitted.get('kind') != 'clarify' or _serialize(pending) != wanted):
                    raise ValueError('Role question lost its captured task.')
                return
            if (set(continuation) != {'origin_turn_id', 'binding_turn_id', 'field'}
                    or any(continuation[k] != reference[k] for k in continuation)
                    or admitted.get('tool') != origin.admitted['tool']
                    or admitted.get('revision_id') != interaction.base_revision_id
                    or admitted.get('context') != origin.admitted.get('context')
                    or _serialize(admitted['required_roles']) != _serialize(origin.admitted['required_roles'])):
                raise ValueError('Role continuation changed scope.')
            additions = _input_roles(_serialize(admitted['roles']))
        declarations = _scientific_arguments(_serialize(admitted.get('arguments', [])))
        if 'argument' in admitted:
            declarations += (LeidenResolution(**_serialize(admitted['argument'])),)
        from .turn_decisions import _validate_arguments
        _validate_arguments(declarations)
        values, roles, missing, metadata = _role_completion(registry, origin, source,
            interaction, additions, declarations)
        if (values != _serialize(admitted['inputs']) or roles != _serialize(admitted['roles'])
                or metadata != _serialize(admitted['registered_input'])):
            raise ValueError('Captured scientific role binding changed.')
        if admitted.get('context') == 'current' and (interaction.base_revision_id is None
                or interaction.snapshot['relations'].get('current') != interaction.base_revision_id):
            raise ValueError('Captured current context changed.')
        if missing:
            wanted = (dict(origin_turn_id=interaction.turn_id, binding_turn_id=interaction.turn_id,
                field='roles', tool=admitted['tool'], required_roles=_serialize(admitted['required_roles']),
                resource_context_sha256=pending['resource_context_sha256']) if initial else
                {k: _serialize(v) for k, v in reference.items() if k != 'predecessor_turn_id'})
            wanted['binding_turn_id'] = interaction.turn_id
            choices = [r['tool'] + '.' + r['port'] for r in missing]
            if (_serialize(pending) != wanted or (not initial and
                    (admitted.get('kind') != 'clarify' or admitted.get('reason') != 'missing_input_roles'
                     or list(admitted.get('choices', ())) != choices or admitted.get('value_required') is not True))):
                raise ValueError('Partial role answer changed requirements.')
        elif initial or admitted.get('kind') != 'execute' or admitted.get('operation') != 'plan' or pending is not None:
            raise ValueError('Invalid completed scientific role request.')
    except (ValueError, TypeError, KeyError, AttributeError, StopIteration) as exc:
        raise SessionError('Invalid pending scientific role binding.') from exc


def admit_role_request(sessions, interaction, decision, resources):
    """Capture a known registered objective without creating a Run or Plan."""
    from .local_resources import LocalResourceAdmission
    registry = sessions._application.registry
    if decision.target not in registry.names() or interaction.submission is None:
        raise IntentError('unsupported_intent')
    owner = LocalResourceAdmission(sessions._application._workspace, registry=registry)
    binding = owner._submission_binding(interaction.submission)
    owner.validate_binding(binding)
    admitted = dict(kind='request_roles', tool=decision.target, revision_id=interaction.base_revision_id,
        required_roles=[asdict(r) for r in decision.required_roles], roles=[asdict(r) for r in decision.roles])
    prototype = replace(interaction, admitted=admitted)
    values, roles, missing, metadata = _role_completion(registry, prototype, None, interaction,
        decision.roles, _arguments(decision))
    if not missing:
        raise IntentError('invalid_prerequisite')
    admitted.update(inputs=values, roles=roles, registered_input=metadata)
    if decision.argument is not None:
        admitted['argument'] = asdict(decision.argument)
    if decision.arguments:
        admitted['arguments'] = [asdict(d) for d in decision.arguments]
    if decision.context == 'current':
        context, _ = sessions.planning_context_for_interaction(sessions._interaction_session_id,
            interaction.turn_id, integrity=False)
        if context is None:
            raise IntentError('unavailable_context')
        admitted['context'] = 'current'
    pending = dict(origin_turn_id=interaction.turn_id, binding_turn_id=interaction.turn_id,
        field='roles', tool=decision.target, required_roles=admitted['required_roles'],
        resource_context_sha256=resource_context_digest(resources))
    return admitted, pending, tuple(r['tool'] + '.' + r['port'] for r in missing)


def admit_roles(sessions, interaction, decision, resources):
    """Continue only the exact offered pending task and ordered registered inputs."""
    from .local_resources import LocalResourceAdmission
    state = sessions.load(sessions._interaction_session_id)
    reference = interaction.snapshot.get('prerequisite')
    if (decision.pending != '@roles' or reference is None or reference['field'] != 'roles'
            or state.generation != interaction.base_generation or state.active_revision_id != interaction.base_revision_id
            or state.interactions[-1].turn_id != interaction.turn_id
            or resource_context_digest(resources) != reference['resource_context_sha256']):
        raise IntentError('invalid_prerequisite')
    validate_prerequisite(interaction, state.interactions[:-1])
    origin, source = _role_source(reference, state.interactions[:-1])
    owner = LocalResourceAdmission(sessions._application._workspace, registry=sessions._application.registry)
    owner.validate_binding(_role_binding(origin, source, interaction))
    values, roles, missing, metadata = _role_completion(sessions._application.registry, origin,
        source, interaction, decision.roles, decision.arguments)
    admitted = dict(kind='clarify' if missing else 'execute', tool=origin.admitted['tool'],
        revision_id=interaction.base_revision_id, inputs=values, roles=roles,
        required_roles=_serialize(origin.admitted['required_roles']), registered_input=metadata,
        continuation={k: reference[k] for k in ('origin_turn_id', 'binding_turn_id', 'field')})
    if origin.admitted.get('context') == 'current':
        admitted['context'] = 'current'
    if decision.arguments:
        admitted['arguments'] = [asdict(d) for d in decision.arguments]
    pending = None
    if missing:
        choices = [r['tool'] + '.' + r['port'] for r in missing]
        admitted.update(reason='missing_input_roles', choices=choices, value_required=True)
        pending = {k: _serialize(v) for k, v in reference.items() if k != 'predecessor_turn_id'}
        pending['binding_turn_id'] = interaction.turn_id
    else:
        admitted.update(operation='plan', request_id='turn-' + digest(dict(session=state.session_id,
            turn=interaction.turn_id)))
    return admitted, pending


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
    if pending['field'] == 'resources':
        from .resource_continuation import capture
        return capture(prior, origin, current)
    if pending['field'] == 'roles':
        _, source = _role_source(pending, preceding)
        try:
            from types import SimpleNamespace
            _role_binding(origin, source, SimpleNamespace(submission=current))
        except (ValueError, TypeError, KeyError, AttributeError):
            return None
        return dict(**pending, predecessor_turn_id=prior.turn_id)
    if pending['field'] == 'parameters':
        from agent.orchestration.registry import build_default_tool_registry
        registry = build_default_tool_registry()
        original = dict(execution_inputs=_serialize(origin.admitted['inputs']))
        metadata = (origin.submission or {}).get('registered_input')
        if metadata is None:
            metadata = (_received_origin(origin, preceding).submission or {}).get('registered_input')
        if metadata is not None:
            original['registered_input'] = _serialize(metadata)
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
    admitted = interaction.admitted or {}
    resource_continuation = admitted.get('continuation', {}).get('field') == 'resources'
    if (admitted.get('kind') == 'request_resources' or resource_continuation
            or (interaction.prerequisite or {}).get('field') == 'resources'):
        from .resource_continuation import validate
        validate(interaction, preceding)
        if (interaction.prerequisite or {}).get('field') == 'parameters':
            _validate_parameter_prerequisite(interaction, preceding)
        return
    role_continuation = admitted.get('continuation', {}).get('field') == 'roles'
    if (admitted.get('kind') == 'request_roles' or role_continuation
            or (interaction.prerequisite or {}).get('field') == 'roles'
            or (interaction.snapshot.get('prerequisite') or {}).get('field') == 'roles'):
        captured = _serialize(interaction.snapshot)
        expected = capture_prerequisite(preceding, interaction.base_revision_id,
            interaction.base_generation, interaction.submission,
            captured.get('dialogue', {}).get('requested_predecessor'))
        if captured.get('prerequisite') != expected:
            raise SessionError('Pending declaration capture changed.')
        _validate_role_prerequisite(interaction, preceding)
        if (interaction.prerequisite or {}).get('field') == 'parameters':
            _validate_parameter_prerequisite(interaction, preceding)
        elif not role_continuation and admitted.get('kind') != 'request_roles':
            _validate_input_role_record(interaction)
            _validate_resolution_record(interaction)
            _validate_argument_record(interaction)
        return
    if 'context' in admitted:
        captured = interaction.snapshot
        if (admitted['context'] != 'current' or interaction.base_revision_id is None
                or captured['relations'].get('current') != interaction.base_revision_id
                or not (admitted.get('operation') == 'plan' or 'parameter_continuation' in admitted)):
            raise SessionError('Invalid captured execution context.')
        continuation = admitted.get('parameter_continuation', admitted.get('continuation'))
        if continuation is not None:
            origin = next((i for i in preceding if i.turn_id == continuation['origin_turn_id']), None)
            if origin is None or (origin.admitted or {}).get('context') != 'current':
                raise SessionError('Pending execution context changed.')
    continuation = admitted.get('parameter_continuation', admitted.get('continuation'))
    if continuation is not None:
        origin = next((i for i in preceding if i.turn_id == continuation['origin_turn_id']), None)
        if origin is not None and (origin.admitted or {}).get('context') != admitted.get('context'):
            raise SessionError('Pending execution context changed.')
    _validate_input_role_record(interaction)
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
        original_choice = original['registered_input'].get('resource_choice')
        received_choice = (interaction.submission or {}).get('registered_input', {}).get('resource_choice')
        expected_choice = received_choice if received_choice is not None else original_choice
        if metadata.get('resource_choice') != expected_choice:
            raise SessionError('Continuation changed its captured resource choice.')
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
    if reference['field'] == 'resources':
        from .resource_continuation import public
        return public(interaction, preceding)
    if reference['field'] == 'roles':
        _, source = _role_source(reference, preceding)
        roles = _serialize(source.admitted['roles'])
        assigned = {(r['tool'], r['port']) for r in roles}
        required = _serialize(reference['required_roles'])
        return dict(handle='@roles', field='roles', tool=reference['tool'],
            required_roles=required, roles=roles,
            missing_roles=[r for r in required if (r['tool'], r['port']) not in assigned],
            objective=origin.utterance)
    if reference['field'] == 'parameters':
        return dict(handle='@parameters', field='parameters', tool=reference['tool'],
                    missing=list(reference['missing']), objective=_execution_objective(origin, preceding))
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
    original_choice = original['registered_input'].get('resource_choice')
    received_choice = current.get('registered_input', {}).get('resource_choice')
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
                                  resource_selection_error=selection_error,
                                  resource_choice=(received_choice if received_choice is not None
                                                   else binding.resource_choice))
    admitted = dict(kind='execute', operation='plan', tool=origin.admitted['tool'],
        revision_id=interaction.base_revision_id,
        request_id='turn-' + digest(dict(session=state.session_id, turn=interaction.turn_id)),
        inputs=_serialize(completed.execution_inputs), registered_input=completed.attribution(),
        continuation=dict(origin_turn_id=origin.turn_id, species=decision.species))
    if origin.admitted.get('context') == 'current':
        admitted['context'] = 'current'
    if decision.argument is not None:
        admitted['argument'] = asdict(decision.argument)
    if decision.arguments:
        admitted['arguments'] = [asdict(d) for d in decision.arguments]
    return admitted


def execution_submission(interaction):
    """Use the admitted binding for continuation, preserving the received turn."""
    admitted = interaction.admitted or {}
    if not any(key in admitted for key in ('continuation', 'argument', 'arguments', 'roles')):
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


def _received_origin(interaction, interactions):
    """Follow exact persisted links to the original received request."""
    by_id = {i.turn_id: i for i in interactions}
    visited = set()
    while True:
        if interaction.turn_id in visited:
            raise IntentError('invalid_prerequisite')
        visited.add(interaction.turn_id)
        admitted = interaction.admitted or {}
        continuation = admitted.get('parameter_continuation', admitted.get('continuation'))
        if continuation is None:
            return interaction
        interaction = by_id[continuation['origin_turn_id']]


def _execution_objective(interaction, interactions):
    return _received_origin(interaction, interactions).utterance


def admit(sessions, interaction, decision, inputs, *, epizoo_resources=()):
    if (decision.base != 'current' or decision.delta is not None
            or decision.target not in sessions._application.registry.names()):
        raise IntentError('unsupported_intent')
    values = {} if inputs is None else _serialize(freeze_json_mapping(inputs, 'execution_inputs'))
    admitted = dict(kind='execute', operation='plan', tool=decision.target, revision_id=interaction.base_revision_id,
                request_id='turn-' + digest(dict(session=sessions._interaction_session_id, turn=interaction.turn_id)),
                inputs=values)
    grouped = (interaction.submission is not None
        and interaction.submission.get('registered_input', {}).get('composition') == 'scientific-role-selection.v1')
    if grouped:
        from .local_resources import LocalResourceAdmission, ResourceAdmissionError
        if not decision.roles:
            raise IntentError('ambiguous_subject')
        owner = LocalResourceAdmission(sessions._application._workspace, registry=sessions._application.registry)
        original = owner._submission_binding(interaction.submission)
        if _serialize(original.execution_inputs) != values:
            raise ResourceAdmissionError('LOCAL_RESOURCE_BINDING_INVALID')
        roles = [asdict(item) for item in decision.roles]
        completed = owner.bind_group(original, roles, verify_source=True)
        values = _serialize(completed.execution_inputs)
        admitted.update(inputs=values, roles=roles, registered_input=completed.attribution())
    elif decision.roles:
        raise IntentError('unsupported_intent')
    if decision.context == 'current':
        context, _ = sessions.planning_context_for_interaction(
            sessions._interaction_session_id, interaction.turn_id, integrity=False)
        if context is None:
            raise IntentError('unavailable_context')
        admitted['context'] = 'current'
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
            if grouped:
                # Scalar declarations add values; exact selected membership and
                # Registry consumer associations remain the admitted group.
                pass
            elif original.tool_name == 'inspect_scATAC' and original.composition in (None, 'h5ad-science.v1'):
                selection_error = original.resource_selection_error
                if any(isinstance(d, ScalarChoiceArgument) for d in decision.arguments):
                    values, selection_error = _select_declared_resource(values, epizoo_resources)
                    admitted['resource_selection'] = dict(catalog=[asdict(r) for r in epizoo_resources],
                        resource_context_sha256=resource_context_digest(epizoo_resources))
                completed = owner.compose_h5ad(original.resource_id,
                    {k: v for k, v in values.items() if k != 'input_path'},
                    resource_selection_error=selection_error,
                    resource_choice=original.resource_choice)
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
        intent = _execution_objective(origin, state.interactions)
    interaction = replace(interaction, admitted=admitted, submission=execution_submission(
        replace(interaction, admitted=admitted)))
    if selected:
        from .guidance_selection import execution_context, planning_intent
        context = execution_context(sessions, interaction, admitted)
        intent = planning_intent(sessions, interaction, admitted, context)
    elif admitted.get('context') == 'current':
        context, context_locators = sessions.planning_context_for_interaction(sid, interaction.turn_id)
        if context is None:
            raise IntentError('unavailable_context')
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
        retention = {}
        retained_offer = []
        if not selected and context is not None:
            for item in context.items:
                for locator in context_locators[item.handle]:
                    handle = 'retain.' + str(len(retention))
                    retention[handle] = locator
                    retained_offer.append(dict(handle=handle, context_handle=item.handle,
                        semantic_type=item.binding.semantic_type, producer_tool=item.binding.tool_name,
                        source_port=item.binding.source_port))
            schema = _object(dict(**schema['properties'], retain={'type':'array','maxItems':32,
                'items':{'type':'string','enum':list(retention)}}))
        selection_prompt = dict(output_selection_schema_version=2 if retained_offer else 1,
            question=intent, steps=offered,
            instructions=(
                'Select a relevant subset of 1 to 32 outputs of this actual plan to retain in the new revision. '
                'Choose results that satisfy the user request; do not automatically enumerate every field from every step. '
                'Preserve multiple legitimate outputs when the user explicitly requests them. '
                'For each selection, copy step_id exactly from an offered step and output_key exactly from that same step\'s outputs. '
                'Never invent or repair step identities, output keys, scientific products or output relationships. '
                'Each name is a nonblank string label and must be unique across all selections for the new revision, '
                'including when different steps expose the same output_key. Names need not equal output keys; '
                'use meaningful labels that distinguish the selected results. Do not infer or append scientific steps.'))
        if retained_offer:
            selection_prompt['retained_outputs'] = retained_offer
            selection_prompt['instructions'] += (
                ' Select retain handles explicitly for applicable captured upstream results needed in the continued analysis. '
                'Do not retain everything, stale descendants or unrelated results. Return [] when no historical result should be kept. '
                'Multiple handles for one context artifact are distinct named selections; do not choose an alias by first match. '
                'Retention does not grant scientific authority to a new result. New and retained selections total at most 32.')
        raw_choice = model.complete(prompt=json.dumps(selection_prompt), response_schema=schema)
        try:
            choice = _json(raw_choice)
            _shape(choice, ('outputs', 'retain') if retained_offer else ('outputs',))
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
            retained = ()
            if retained_offer:
                handles = choice['retain']
                if (type(handles) is not list
                        or any(type(h) is not str or h not in retention for h in handles)
                        or len(set(handles)) != len(handles)
                        or len(handles) + len(selections) > 32):
                    raise IntentError('planning_failed')
                retained = tuple(retention[h] for h in handles)
                if ({o.name for o in selections}.intersection(o.name for o in retained)
                        or len({o.name for o in retained}) != len(retained)):
                    raise IntentError('planning_failed')
        except ValueError as exc:
            from agent.orchestration.planner import PlannerError
            from agent.schemas import ErrorCategory
            raise PlannerError('PLANNER_OUTPUT_INVALID', 'The model returned an invalid output selection.',
                               category=ErrorCategory.INTERNAL_AGENT_ERROR) from exc
        if selected:
            execution_context(sessions, interaction, admitted)
        elif context is not None:
            checked_context, checked_locators = sessions.planning_context_for_interaction(sid, interaction.turn_id)
            if checked_context != context or checked_locators != context_locators:
                raise IntentError('unavailable_context')
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
            if context is not None and not selected and (
                    state.active_revision_id != interaction.base_revision_id
                    or state.generation != interaction.base_generation):
                raise SessionConflictError('Captured planning state changed before submission.')
            if any(t.turn_id == interaction.turn_id for t in state.turns):
                raise SessionConflictError('Turn already submitted.')
            turn = SessionTurn(interaction.turn_id, interaction.base_revision_id, interaction.base_generation,
                request.request_id, digest(effective.to_dict()), request.request_id + ':run', 'linked', tuple(selections),
                retained_outputs=retained)
            return replace(state, turns=state.turns+(turn,))
        sessions._store._update(sid, link)
        _update(sessions, interaction.turn_id, status='submitted')

    execution_app = ResearchAgentApplication(app.workspace_root,
        planner=_AdmittedPlanner(app.runtime.planner, accept), registry=app.registry, executor=app.runtime.executor)
    from agent.orchestration.active_context import planning_context
    from agent.orchestration.executor import execution_input_guard
    grouped = (registered and interaction.submission['registered_input'].get('composition')
        == 'scientific-role-selection.v1')
    guard = (lambda tool, arguments: resources.validate_group_consumer(
        interaction.submission, tool, arguments)) if grouped else None
    with execution_input_guard(guard), planning_context(context) if context is not None else nullcontext():
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
    raw_resource_missing = (registered and interaction.submission['registered_input'].get('composition')
        == 'external-fragments-science.v1' and any(k not in admitted['inputs']
        for k in ('reference_bundle_path', 'reference_bundle_sha256', 'source_profile')))
    missing = None if selected or raw_resource_missing else _missing_parameters(error, admitted, app.registry)
    if missing is not None:
        from .turn_decisions import Clarify
        tool_name, names = missing
        pending = dict(origin_turn_id=interaction.turn_id, field='parameters',
            binding_turn_id=interaction.turn_id, tool=tool_name, missing=names,
            resource_context_sha256=resource_context_digest(epizoo_resources))
        _update(sessions, interaction.turn_id, status='clarification', prerequisite=pending)
        return TurnOutcome('clarify', 'clarification', Clarify('missing_parameter_value', tuple(names), True))
    if (registered and error is not None
            and error.code in {'MISSING_REQUIRED_SOURCE', 'MISSING_REQUIRED_BINDING'}):
        # Present only compiler-owned missing-port facts. This neither supplies
        # scientific inputs nor creates pending authority for a resource choice.
        context = {
            'bam-science.v1': ('prepare_scATAC_bam_fragments', 'BAM', {
                'library_context': (('library_context_path', 'library_context_sha256'), 'LIBRARY_CONTEXT_REQUIRED'),
                'reference': (('reference_bundle_path', 'reference_bundle_sha256'), 'REFERENCE_REQUIRED'),
                'source_profile': (('source_profile',), 'PROFILE_REQUIRED'),
                'intake': (('intake_manifest_path', 'intake_manifest_sha256'), 'INTAKE_REQUIRED'),
            }),
            'fastq-science.v1': ('prepare_scATAC_fragments', 'FASTQ', {
                'library_context': (('library_context_path', 'library_context_sha256'), 'LIBRARY_CONTEXT_REQUIRED'),
                'reference': (('reference_bundle_path', 'reference_bundle_sha256'), 'REFERENCE_REQUIRED'),
                'intake': (('intake_manifest_path', 'intake_manifest_sha256'), 'INTAKE_REQUIRED'),
            }),
            'external-fragments-science.v1': ('import_scATAC_fragments', 'FRAGMENTS', {
                'reference': (('reference_bundle_path', 'reference_bundle_sha256'), 'REFERENCE_REQUIRED'),
                'source_profile': (('source_profile',), 'PROFILE_REQUIRED'),
                'namespace': (('namespace',), 'NAMESPACE_REQUIRED'),
            }),
        }.get(interaction.submission['registered_input'].get('composition'))
        if context is not None and error.details.get('tool_name') == context[0]:
            missing = context[2].get(error.details.get('target_port'))
            if missing is not None and any(key not in interaction.submission['execution_inputs'] for key in missing[0]):
                selection_error = interaction.submission['registered_input'].get('resource_selection_error')
                ambiguous = context[1] + '_RESOURCE_AMBIGUOUS'
                code = ambiguous if selection_error == ambiguous else context[1] + '_' + missing[1]
                error = ResourceAdmissionError(code).error
    if (not selected and 'resource_catalog' in admitted and 'continuation' not in admitted
            and error is not None and error.code in {
                'FRAGMENTS_REFERENCE_REQUIRED', 'FRAGMENTS_PROFILE_REQUIRED', 'FRAGMENTS_NAMESPACE_REQUIRED',
                'FRAGMENTS_RESOURCE_AMBIGUOUS', 'BAM_REFERENCE_REQUIRED', 'BAM_PROFILE_REQUIRED',
                'BAM_LIBRARY_CONTEXT_REQUIRED', 'BAM_RESOURCE_AMBIGUOUS', 'FASTQ_REFERENCE_REQUIRED',
                'FASTQ_LIBRARY_CONTEXT_REQUIRED', 'FASTQ_RESOURCE_AMBIGUOUS'}):
        from .turn_decisions import Clarify
        from .input_sets import ScientificInputSet
        from .resource_continuation import offers
        rows = _serialize(admitted['resource_catalog'])
        pending = dict(origin_turn_id=interaction.turn_id, binding_turn_id=interaction.turn_id,
            field='resources', tool=admitted['tool'], resource_context_sha256=digest(rows))
        _update(sessions, interaction.turn_id, status='clarification', prerequisite=pending)
        choices = tuple(c['label'] for c in offers(interaction.submission,
            tuple(ScientificInputSet(**r) for r in rows)))
        return TurnOutcome('clarify', 'clarification', Clarify('missing_resources', choices, True))
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
