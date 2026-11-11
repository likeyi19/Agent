"""Structured researcher settings on the existing declaration and plan owners.

Registry ports identify arguments. Native catalogs identify kwargs. Recorded execution values may be displayed; native defaults and scientific
validity remain with the tools. There is no separate settings history.
"""
from dataclasses import replace

from agent.schemas.orchestration import _serialize
from agent.orchestration.native_parameters import declaration_input
from .turn_decisions import ExplicitArgument, IntentError, argument_identity


def pending_slots(interaction, registry=None):
    """The same field identities for pre-plan and native-discovered questions."""
    failure = (interaction.prerequisite or {}).get('native_failure')
    if failure is not None:
        if registry is None:
            from agent.orchestration.registry import build_default_tool_registry
            registry = build_default_tool_registry()
        return parameter_slots(registry, interaction.admitted['tool'], [dict(tool='run_episcanpy',
            native_function=failure['function_name'], argument='kwargs', member='num_clusters')])
    return (interaction.admitted or {}).get('parameter_slots')


def parameter_slots(registry, target, requests):
    """Project pending fields from the same Registry and declaration scopes.

    These are question identities, not parameter values or a second catalog.
    """
    from .dialogue_execution import _parameter_input
    from .session_state import digest
    slots = {}
    for request in requests:
        row = dict(tool=target, argument=request) if isinstance(request, str) else {
            k: v for k, v in _serialize(request).items() if v is not None and k != 'evidence'}
        if set(row) - {'tool', 'argument', 'native_function', 'port', 'source', 'member'}:
            raise IntentError('invalid_parameter_value')
        member = row.pop('member', None)
        d = ExplicitArgument(**row, value=None)
        ordinary, spec = _parameter_input(registry, d.tool, d.argument)
        declaration_input(registry, d, ordinary)
        if member is not None:
            native = registry.get(d.tool).native_functions
            allowed = (dict(native.functions).get(d.native_function, ()) if native is not None
                       else dict(spec.planning.conversational_members))
            if d.argument != 'kwargs' or member not in allowed:
                raise IntentError('invalid_parameter_value')
            row['member'] = member
        key = 'parameter.value_' + digest(row)[:16]
        slots[key] = row
    if not 1 <= len(slots) <= 8:
        raise IntentError('invalid_prerequisite')
    return slots


def missing_slots(registry, slots, values):
    missing = []
    for handle, row in slots.items():
        key = input_key(registry, {k:v for k,v in row.items() if k != 'member'} | {'value': None})
        effective = values
        if row.get('native_function') is None:
            from agent.orchestration.native_parameters import prerequisite_inputs
            effective = prerequisite_inputs(registry.get(row['tool']), values) or values
        if key not in effective or 'member' in row and row['member'] not in effective[key]:
            missing.append(handle)
    return sorted(missing)


def validate_slot_answers(slots, arguments):
    """A local answer may fill only its captured operation/source/member."""
    from dataclasses import asdict
    for declaration in arguments:
        row = asdict(declaration)
        candidates = [slot for slot in slots.values() if all(slot.get(k) == row.get(k)
            for k in ('tool', 'argument', 'port', 'source'))
            and (slot.get('native_function') is None or slot['native_function'] == row.get('native_function'))]
        if not candidates:
            raise IntentError('invalid_reference')
        if any('member' not in slot for slot in candidates):
            continue
        supplied = (set(declaration.value) if isinstance(declaration, ExplicitArgument)
                    else {m.name for m in getattr(declaration, 'members', ())})
        if not supplied or supplied - {slot['member'] for slot in candidates}:
            raise IntentError('invalid_reference')


def shadowed_arguments(arguments, explicit):
    """Discard model duplicates before grounding; exact UI input owns the value."""
    remaining = []
    for row in arguments:
        matches = [s for s in explicit if all(row.get(k) == s.get(k)
                   for k in ('tool', 'argument', 'native_function', 'port', 'source'))]
        if not matches:
            remaining.append(row)
        elif row.get('members'):
            keys = {k for s in matches for k in s['value']} if row['argument'] == 'kwargs' else set()
            members = [m for m in row['members'] if m['name'] not in keys]
            if members:
                remaining.append(dict(row, members=members))
    return remaining


def declarations(rows, registry):
    from .dialogue_execution import _bind_arguments
    result = tuple(ExplicitArgument(**_serialize(row)) for row in rows)
    from .turn_decisions import _validate_arguments
    _validate_arguments(result)
    _bind_arguments(registry, '', {}, result)
    for d in result:
        if registry.get(d.tool).native_functions is not None and d.native_function is None:
            raise ValueError('Select the native function for this parameter.')
    return result


def resolve_sources(rows, binding):
    """Lower exact browser file identities into this capture's existing source roles."""
    resolved = []
    for row in rows:
        row = _serialize(row)
        if row.get('resource_id') is not None:
            matches = [] if binding is None else [i for i, member in enumerate(binding.members)
                                                  if member['resource_id'] == row['resource_id']]
            if len(matches) != 1:
                raise IntentError('invalid_reference')
            source = 'source.' + str(matches[0])
            if row.get('source') is not None and row['source'] != source:
                raise IntentError('invalid_reference')
            row['source'] = source
        resolved.append(row)
    return resolved


def merge_decision(decision, interaction, registry):
    """Explicit values enter as declarations, never model-produced text evidence."""
    rows = (interaction.submission or {}).get('parameter_overrides', ())
    if not rows:
        return decision
    from .dialogue_execution import captured_input_group
    binding = captured_input_group(interaction)
    rows = resolve_sources(rows, binding)
    explicit = declarations(rows, registry)
    if not hasattr(decision, 'arguments'):
        raise IntentError('unsupported_intent')
    previous = tuple(d for d in decision.arguments if not isinstance(d, ExplicitArgument))
    return replace(decision, arguments=previous + explicit)


def explicit_records(admitted, interactions):
    from .dialogue_execution import _parameter_records
    found = {}
    for record in _parameter_records(admitted, interactions):
        for row in record.get('arguments', ()):
            if row.get('origin') == 'structured':
                key = argument_identity(ExplicitArgument(**_serialize(row)))
                found.setdefault(key, _serialize(row))
    return list(found.values())


def bound_records(admitted, interactions, registry, *, current_utterance=None):
    """One plan binding projection for UI values and admitted language values.

    Language evidence stays in its original declaration; this transient view is
    consumed by the existing prepared-plan binder, never saved as UI input.
    """
    from .dialogue_execution import _parameter_records, _parameter_input
    from .turn_decisions import _scientific_arguments, admit_scientific_argument
    rows = explicit_records(admitted, interactions)
    found = {tuple(row.get(k) for k in ('tool','argument','native_function','port','source','step_id')) for row in rows}
    for record in _parameter_records(admitted, interactions):
        original = next((i for i in reversed(interactions) if i.admitted == record), None)
        source = record
        while source.get('scope_selection'):
            original = next(i for i in interactions if i.turn_id == source['scope_selection'])
            source = original.admitted
        for declaration in _scientific_arguments(_serialize(record.get('arguments', ()))):
            if isinstance(declaration, ExplicitArgument) or declaration.argument == 'function_name':
                continue
            scope = {k:getattr(declaration,k) for k in ('tool','argument','native_function','port','source','step_id')
                     if getattr(declaration,k,None) is not None}
            _, spec = _parameter_input(registry, declaration.tool, declaration.argument)
            if scope.get('source') and not scope.get('port'):
                scope['port'] = spec.planning.observation_source
            identity = tuple(scope.get(k) for k in ('tool','argument','native_function','port','source','step_id'))
            if identity in found:
                continue
            utterance = original.utterance if original is not None else current_utterance
            if utterance is None:
                # A pure plan assertion has no utterance owner; its already
                # admitted inputs still provide the exact function-scoped value.
                value = record['inputs'][input_key(registry, scope | {'value': None})]
            elif spec.planning.observation_source:
                value = getattr(declaration, 'value', declaration.literal)
            else:
                value = admit_scientific_argument(declaration, utterance, spec)
            rows.append(scope | {'value': _serialize(value)})
            found.add(identity)
    return rows


def validate_capture(interaction):
    """Every new structured declaration is the exact captured researcher input."""
    if interaction.admitted is None or interaction.admitted.get('kind') == 'clarify' and 'arguments' not in interaction.admitted:
        return
    rows = [row for row in (interaction.admitted or {}).get('arguments', ())
            if row.get('origin') == 'structured']
    captured = (interaction.submission or {}).get('parameter_overrides', ())
    if any(row.get('resource_id') is not None for row in captured):
        from .dialogue_execution import captured_input_group
        captured = resolve_sources(captured, captured_input_group(interaction))
    if _serialize(rows) != _serialize(captured):
        raise ValueError('Structured parameter declaration differs from its submission.')


def input_key(registry, row):
    from .dialogue_execution import _parameter_input
    d = ExplicitArgument(**_serialize(row))
    return declaration_input(registry, d, _parameter_input(registry, d.tool, d.argument)[0])


def matches(plan, row, registry, admitted):
    selected = [s for s in plan.steps if s.tool_name == row['tool']
                and (row.get('native_function') is None or s.arguments.get('function_name') == row['native_function'])]
    if row.get('port') is not None:
        from .local_resources import _group_port, plan_source_path
        members = (admitted.get('registered_input') or {}).get('members', ())
        index = int(row['source'].split('.')[1])
        role = {k: row[k] for k in ('tool', 'port', 'source')}
        if index >= len(members) or role not in admitted.get('roles', ()):
            raise IntentError('invalid_reference')
        port, _, _ = _group_port(registry, row['tool'], row['port'])
        selected = [s for s in selected if plan_source_path(s.arguments.get(port.members[0].field_name), plan.steps)
                    == members[index]['source_path']]
    if row.get('step_id') is not None:
        selected = [s for s in selected if s.step_id == row['step_id']]
    failure = admitted.get('native_failure')
    if (failure is not None and row['tool'] == 'run_episcanpy' and row['argument'] == 'kwargs'
            and row.get('native_function') == failure['function_name'] and 'num_clusters' in row['value']):
        selected = [s for s in selected if s.step_id == failure['step_id']]
    return selected


class ScopeRequired(IntentError):
    def __init__(self, plan, row, choices):
        super().__init__('ambiguous_parameter')
        self.plan, self.row, self.choices = plan, row, choices


def effective_value(row, rows, inputs, registry):
    key = input_key(registry, row)
    value = _serialize(inputs[key])
    same_input = [r for r in rows if input_key(registry, r) == key]
    if len(same_input) > 1:
        # One Registry source can name several explicitly selected consumers.
        # Keep each consumer's values; never forward a sibling's override.
        if row['argument'] == 'kwargs':
            owned = {k for r in same_input for k in r['value']}
            value = {**{k: v for k, v in value.items() if k not in owned}, **_serialize(row['value'])}
        else:
            value = _serialize(row['value'])
    return value


def bind_plan(plan, rows, inputs, registry, admitted):
    """Bind actual selected operations using their registered semantic arguments."""
    steps = list(plan.steps)
    for row in rows:
        selected = matches(plan, row, registry, admitted)
        if len(selected) != 1:
            if selected:
                raise ScopeRequired(plan, row, selected)
            raise IntentError('invalid_reference')
        step = selected[0]
        value = effective_value(row, rows, inputs, registry)
        arguments = dict(step.arguments)
        if row['argument'] == 'kwargs':
            value = {**_serialize(arguments.get('kwargs', {})), **value}
        arguments[row['argument']] = value
        registry.validate_arguments(step.tool_name, arguments)
        steps[steps.index(step)] = replace(step, arguments=arguments)
        plan = replace(plan, steps=tuple(steps))
    return plan


def check_plan(plan, rows, inputs, registry, admitted):
    for row in rows:
        selected = matches(plan, row, registry, admitted)
        if len(selected) != 1:
            raise IntentError('ambiguous_parameter')
        expected = effective_value(row, rows, inputs, registry)
        actual = _serialize(selected[0].arguments.get(row['argument']))
        if row['argument'] == 'kwargs':
            if not isinstance(actual, dict) or any(actual.get(k) != v for k, v in expected.items()):
                raise IntentError('planning_failed')
        elif actual != expected:
            raise IntentError('planning_failed')


def catalog(registry, tool_name=None, function=None):
    """Transient view of Registry descriptions and installed native signatures."""
    from .dialogue_execution import scientific_parameters
    names = [tool_name] if tool_name else registry.names()
    tools = []
    parameters = scientific_parameters(registry)
    for name in names:
        tool = registry.get(name)
        rows = [dict(p, types=[t.__name__ for t in {**tool.required_arguments, **tool.optional_arguments}[p['argument']].accepted_types])
                for p in parameters if p['tool'] == name and p['argument'] not in ('kwargs', 'function_name')]
        functions = [] if tool.native_functions is None else list(dict(tool.native_functions.functions))
        item = dict(tool=name, functions=functions, parameters=rows)
        if function is not None:
            if function not in functions:
                raise ValueError('Choose a registered native function.')
            spec = tool.optional_arguments['kwargs']
            descriptions = dict(spec.planning.conversational_members)
            metadata = ({} if tool.native_functions.parameter_metadata is None else
                        tool.native_functions.parameter_metadata(function))
            kwargs = []
            for key in dict(tool.native_functions.functions)[function]:
                row = dict(argument=key, description=descriptions.get(key, 'Native ' + key), types=[])
                row.update(metadata.get(key, {}))
                kwargs.append(row)
            item['kwargs'] = kwargs
        if rows or functions:
            tools.append(item)
    return tools


def scope_choices(interaction):
    pending = (interaction.admitted or {}).get('parameter_scope')
    if pending is None:
        return ()
    return tuple(dict(handle='step:' + step['step_id'], label=step['step_id'],
        description=step['step_id'] + ' · ' + step['tool_name'] + ' · '
        + step['arguments'].get('function_name', '') + ' · '
        + (step.get('description') or '')) for step in pending['plan']['steps']
        if step['step_id'] in pending['choices'])


def _selected_declaration(rows, scope):
    candidates = [r for r in rows if all(r.get(k) == scope.get(k)
        for k in ('tool','argument','native_function','source','step_id'))
        and (r.get('port') == scope.get('port') or 'mode' in r)]
    if len(candidates) != 1:
        raise IntentError('invalid_reference')
    return candidates[0]


def selected_scope(sessions, interaction, preceding):
    choice = (interaction.submission or {}).get('clarification_choice')
    if choice is None or not choice['handle'].startswith('step:'):
        return None
    source = next((i for i in preceding if i.turn_id == choice['question_turn_id']), None)
    if (source is None or source != preceding[-1] or source.status != 'clarification'
            or source.base_generation != interaction.base_generation
            or source.base_revision_id != interaction.base_revision_id
            or choice['handle'] not in {r['handle'] for r in scope_choices(source)}):
        raise IntentError('invalid_reference')
    from .session_state import digest
    admitted = _serialize(source.admitted)
    pending = admitted.pop('parameter_scope')
    rows = admitted['arguments']
    row = _selected_declaration(rows, pending['argument'])
    row['step_id'] = choice['handle'][5:]
    admitted.update(request_id='turn-' + digest(dict(session=sessions._interaction_session_id,
        turn=interaction.turn_id)), scope_selection=source.turn_id, native_plan=pending['plan'])
    admitted['native_plan']['request_id'] = admitted['request_id']
    return admitted


def validate_scope(interaction, preceding):
    source = next(i for i in preceding if i.turn_id == interaction.admitted['scope_selection'])
    choice = interaction.submission['clarification_choice']
    pending = source.admitted['parameter_scope']
    expected = _serialize(source.admitted)
    expected.pop('parameter_scope')
    row = _selected_declaration(expected['arguments'], pending['argument'])
    row['step_id'] = choice['handle'][5:]
    expected.update(request_id=interaction.admitted['request_id'], scope_selection=source.turn_id,
                    native_plan=_serialize(pending['plan']))
    expected['native_plan']['request_id'] = expected['request_id']
    prepared_fields = {'native_plan', 'native_planning_trace', 'native_resource_bindings', 'parameter_scope'}
    actual_core = {k: v for k, v in _serialize(interaction.admitted).items() if k not in prepared_fields}
    expected_core = {k: v for k, v in expected.items() if k not in prepared_fields}
    if (source != preceding[-1] or source.base_generation != interaction.base_generation
            or source.base_revision_id != interaction.base_revision_id
            or choice['question_turn_id'] != source.turn_id
            or choice['handle'] not in {r['handle'] for r in scope_choices(source)}
            or actual_core != expected_core):
        raise ValueError('Selected parameter step differs from its captured plan.')


def interaction_options(sessions, state):
    """Offer exact existing result/source identities for an ordinary request."""
    from .turn_context import snapshot, result_groups
    from .scientific_dialogue import capture
    captured = snapshot(sessions, state)
    captured['dialogue'] = capture(sessions, state, captured, None)
    results = []
    for group in result_groups(captured, state):
        identity = {k: group['canonical'][k] for k in ('revision_id', 'output_name', 'accepted_step_sha256')}
        results.append(dict(identity=identity, label=group['canonical']['output_name'],
            tool=group['canonical']['tool'], current=identity['revision_id'] == state.active_revision_id))
    return dict(generation=state.generation, results=results)


def selected_groups(captured, state, selections):
    from .turn_context import result_groups
    groups = result_groups(captured, state)
    result = []
    for selection in selections:
        matches = [g for g in groups if all(g['canonical'][k] == value for k, value in selection.items())]
        if len(matches) != 1 or matches[0] in result:
            raise IntentError('invalid_reference')
        result.extend(matches)
    return result


def accepted_analysis(sessions, session_id, revision_id):
    """Read the selected Revision's actual accepted plan, never Registry inventory."""
    state = sessions.load(session_id)
    revision = next((r for r in state.revisions if r.revision_id == revision_id), None)
    if revision is None:
        raise ValueError('Unknown accepted analysis.')
    sessions._validate_revision(revision)
    run = sessions._application.run_store.load(revision.run_id)
    return state, revision, run


def recorded_native_kwargs(step):
    """Actual accepted kwargs, including defaults recorded by the native worker."""
    recorded = _serialize(step.result.get('parameters', {}))
    effective = recorded.get('effective_native_parameters', {}) if isinstance(recorded, dict) else {}
    return {**_serialize(step.resolved_arguments.get('kwargs', {})), **effective}


def result_parameters(sessions, session_id, revision_id):
    state, revision, run = accepted_analysis(sessions, session_id, revision_id)
    registry = sessions._application.registry
    steps = []
    for step in run.steps:
        if step.result is None or step.verification is None or not step.verification.passed:
            continue
        function = step.resolved_arguments.get('function_name')
        spec = catalog(registry, step.tool_name, function)
        spec = spec[0] if spec else dict(parameters=[])
        recorded = _serialize(step.result.get('parameters', {}))
        fields = []
        for kwarg, rows in ((False, spec['parameters']), (True, spec.get('kwargs', ()))):
            for field in rows:
                name = field['argument']
                values = (recorded_native_kwargs(step) if kwarg
                          else {**_serialize(step.resolved_arguments),
                                **{k:v for k,v in recorded.items() if k == name and k != 'parameter_selection'},
                                **{k:_serialize(v) for k,v in step.result.items() if k == name}})
                if function and not kwarg and name not in values:
                    continue
                # Current installed defaults are not evidence of a past execution.
                field = {k:v for k,v in field.items() if k != 'default_when_omitted'}
                value = values.get(name)
                editable = value is None or isinstance(value, (str, int, float, bool)) or (
                    isinstance(value, list) and (all(isinstance(v, str) for v in value) or
                    all(type(v) in (int, float) for v in value)))
                fields.append(dict(field, kwarg=kwarg, recorded=name in values,
                    value=value, editable=editable))
        steps.append(dict(step_id=step.step_id, tool=step.tool_name, native_function=function,
            parameters=fields, recorded=recorded))
    return dict(revision_id=revision_id, generation=state.generation, steps=steps)


class ParameterEditUnavailable(ValueError):
    """A recorded operation cannot accept this structured edit."""


def prepare_result_edit(sessions, interaction):
    """Bind edits to an accepted plan; existing DAG references supply unchanged work."""
    from pathlib import Path
    from agent.schemas import AgentRequest, StepOutputRef, PriorOutputRef
    from agent.orchestration.prior_outputs import bind_output, channel_for
    from .session_state import digest
    from .dialogue_execution import _bind_arguments
    sid = sessions._interaction_session_id
    rid = interaction.submission['rerun_revision_id']
    _, revision, run = accepted_analysis(sessions, sid, rid)
    if rid != interaction.base_revision_id:
        raise ParameterEditUnavailable('Continue from the selected result before running its changes.')
    rows = _serialize(interaction.submission.get('parameter_overrides', ()))
    if not rows or any(not row.get('step_id') or row.get('source') or row.get('resource_id') for row in rows):
        raise ParameterEditUnavailable('Select parameters on exact recorded steps.')
    registry = sessions._application.registry
    offered = result_parameters(sessions, sid, rid)
    for row in rows:
        step = next((s for s in offered['steps'] if s['step_id'] == row['step_id']
                     and s['tool'] == row['tool'] and s['native_function'] == row.get('native_function')), None)
        if step is None:
            raise ParameterEditUnavailable('The selected operation is not an accepted step of this analysis.')
        names = set(row['value']) if row['argument'] == 'kwargs' else {row['argument']}
        if names - {p['argument'] for p in step['parameters'] if p['editable'] and p['kwarg'] == (row['argument'] == 'kwargs')}:
            raise ParameterEditUnavailable('The selected value is not an adjustable scientific parameter.')
    values = _bind_arguments(registry, '', {}, declarations(rows, registry))
    # The accepted selected value is an ordinary effective parameter. Pin it on
    # every recorded clustering step before edits, including downstream reruns.
    # Do not materialize unrelated native defaults into a second parameter system.
    stored = {s.step_id:s for s in run.steps}
    baseline = []
    for step in run.plan.steps:
        recorded = stored[step.step_id].result.get('parameters', {})
        selection = recorded.get('parameter_selection')
        args = dict(step.arguments)
        if selection is not None:
            parameter = selection['parameter']
            effective = recorded.get('effective_native_parameters', {})
            if parameter in effective:
                args['kwargs'] = {**_serialize(args.get('kwargs', {})), parameter: _serialize(effective[parameter])}
        baseline.append(replace(step, arguments=args))
    baseline = replace(run.plan, steps=tuple(baseline))
    plan = bind_plan(baseline, rows, values, registry, {})
    # An explicit selection-policy edit requests a new selection/default, rather
    # than merely changing a sibling kwarg. A simultaneous numeric edit prevails.
    for row in rows:
        if row['argument'] != 'parameter_selection':
            continue
        selected = stored[row['step_id']].result['parameters'].get('parameter_selection')
        if selected is None:
            continue
        parameter = selected['parameter']
        if any(r['step_id'] == row['step_id'] and r['argument'] == 'kwargs' and parameter in r['value'] for r in rows):
            continue
        if row['value'] == 'native_defaults' and parameter == 'num_clusters':
            raise ParameterEditUnavailable('This native function has no num_clusters default; supply an explicit cluster count.')
        steps = []
        for step in plan.steps:
            args = dict(step.arguments)
            if step.step_id == row['step_id']:
                args['kwargs'] = {k:v for k,v in args.get('kwargs', {}).items() if k != parameter}
            steps.append(replace(step, arguments=args))
        plan = replace(plan, steps=tuple(steps))
    affected = {s.step_id for s, old in zip(plan.steps, baseline.steps) if s.arguments != old.arguments}
    if not affected:
        raise ParameterEditUnavailable('Change at least one recorded parameter before running.')
    while True:
        expanded = affected | {s.step_id for s in plan.steps if affected.intersection(s.depends_on)
            or any(isinstance(v, StepOutputRef) and v.step_id in affected for v in s.arguments.values())}
        if expanded == affected:
            break
        affected = expanded
    request_id = 'turn-' + digest(dict(session=sid, turn=interaction.turn_id))
    # The existing application owner allocates this run's output location.
    effective_request, _ = sessions._application._prepare_request(AgentRequest(request_id, run.request.prompt, {}))
    old_root, new_root = Path(run.request.inputs['output_dir']), Path(effective_request.inputs['output_dir'])
    stored = {s.step_id:s for s in run.steps}
    planned = {s.step_id:s for s in plan.steps}
    steps = []
    for step in plan.steps:
        if step.step_id not in affected:
            continue
        args = dict(step.arguments)
        semantic = registry.get(step.tool_name).semantic_planning
        for name, value in args.items():
            if isinstance(value, StepOutputRef) and value.step_id not in affected:
                producer = registry.get(planned[value.step_id].tool_name).semantic_planning
                consumers = [p for p in semantic.consumer_ports if any(m.field_name == name for m in p.members)]
                ports = [p for p in producer.producer_ports if any(m.field_name == value.output_key for m in p.members)]
                compatible = []
                for port in ports:
                    try:
                        binding = bind_output(sessions._application.run_store, registry, run.run_id, value.step_id, port.name)
                        if len(consumers) != 1:
                            continue
                        channel_for(binding, step.tool_name, consumers[0].name, registry)
                        compatible.append(binding)
                    except (ValueError, RuntimeError):
                        continue
                if len(compatible) != 1:
                    raise ParameterEditUnavailable('This recorded dependency has no unambiguous supported accepted-output binding.')
                args[name] = PriorOutputRef(compatible[0], value.output_key)
            elif name == 'output_dir' and isinstance(value, str):
                try:
                    args[name] = str(new_root / Path(value).relative_to(old_root))
                except ValueError as exc:
                    raise ParameterEditUnavailable('This analysis used an output location outside its managed run. It cannot be reused for an edit.') from exc
        source = stored[step.step_id].result.get('source_identity', {})
        if (isinstance(source, dict) or hasattr(source, 'get')) and source.get('sha256') and 'input_sha256' in registry.get(step.tool_name).optional_arguments:
            if not isinstance(args.get('input_path'), (StepOutputRef, PriorOutputRef)):
                args['input_sha256'] = source['sha256']
        steps.append(replace(step, arguments=args, depends_on=tuple(d for d in step.depends_on if d in affected)))
    plan = replace(plan, plan_id=request_id + ':accepted-edit', request_id=request_id, steps=tuple(steps))
    outputs = [dict(name=o.name, step_id=o.step_id, output_key=o.output_key)
               for o in revision.outputs if o.run_id == run.run_id and o.step_id in affected]
    retained = [dict(name=o.name, run_id=o.run_id, step_id=o.step_id, output_key=o.output_key,
                     accepted_step_sha256=o.accepted_step_sha256) for o in revision.outputs
                if not (o.run_id == run.run_id and o.step_id in affected)]
    if not outputs:
        raise ParameterEditUnavailable('The edited steps do not supply a selected output of this result.')
    return dict(kind='execute', operation='plan', tool=steps[0].tool_name, request_id=request_id,
        revision_id=rid, result_edit=dict(revision_id=rid, run_id=run.run_id, run_result_sha256=revision.run_result_sha256),
        inputs=values, arguments=rows, native_plan=plan.to_dict(), outputs=outputs, retained_outputs=retained,
        objective=run.request.prompt)


def validate_result_edit(interaction):
    """Replay the structured capture; accepted RunStore checks remain at execution."""
    record = interaction.admitted
    validate_capture(interaction)
    if (record['result_edit']['revision_id'] != interaction.base_revision_id
            or record['revision_id'] != interaction.submission.get('rerun_revision_id')
            or record['result_edit']['revision_id'] != record['revision_id']):
        raise ValueError('Parameter edits differ from their captured result.')
