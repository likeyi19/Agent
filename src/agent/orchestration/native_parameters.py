"""Function-qualified scalar sources for existing package-level Registry tools.

Function names are reviewed choices, not arbitrary callables or step identifiers.
The ordinary compiler binds these request sources; admission checks their consumers.
"""
from dataclasses import replace


def input_name(tool, function, argument):
    return tool + '__' + function.replace('.', '_') + '__' + argument


def function_choices(tool):
    spec = tool.required_arguments.get('function_name')
    return tuple(spec.choices) if spec and spec.planning and spec.planning.planner_choice else ()


def request_source_applies(tool, selector, function):
    """A registered function-qualified scalar belongs only to that function.

    An unresolved function does not authorize guessing: keep the ordinary
    compiler ambiguity checks until a function has actually been selected.
    Unqualified historical/request sources retain their existing scope.
    """
    if function is None:
        return True
    for candidate in function_choices(tool):
        for argument in (*tool.required_arguments, *tool.optional_arguments):
            if selector == input_name(tool.name, candidate, argument):
                return candidate == function
    return True


def prerequisite_inputs(tool, inputs):
    """Project one unambiguous native call for pre-plan prerequisite advice.

    These local aliases never enter the request or grant execution authority.
    Multiple function scopes must wait for planning and native input validation;
    a tool-level prerequisite callback cannot assign them to individual steps.
    """
    qualified = {
        function: {argument: inputs[name]
                   for argument in (*tool.required_arguments, *tool.optional_arguments)
                   if (name := input_name(tool.name, function, argument)) in inputs}
        for function in function_choices(tool)
    }
    qualified = {function: values for function, values in qualified.items() if values}
    if not qualified:
        return inputs
    if len(qualified) != 1:
        return None
    function, values = next(iter(qualified.items()))
    if ('function_name' in inputs and inputs['function_name'] != function
            or any(name in inputs and inputs[name] != value for name, value in values.items())):
        return None
    return {**inputs, **values}


def scalar_port(tool, functions, argument, *, required):
    from .registry import _semantic_argument_port, _semantic_request, _semantic_request_member
    port = _semantic_argument_port(argument, required=required)
    return replace(port, request_sources=port.request_sources + tuple(
        _semantic_request(input_name(tool, fn, argument),
            _semantic_request_member('value', input_name(tool, fn, argument))) for fn in functions))


def declaration_input(registry, declaration, ordinary):
    function = getattr(declaration, 'native_function', None)
    if function is None:
        return ordinary
    tool = registry.get(declaration.tool)
    if function not in function_choices(tool):
        raise ValueError('Unreviewed native function scope')
    name = input_name(tool.name, function, declaration.argument)
    if not any(source.selector == name for port in tool.semantic_planning.consumer_ports
               for source in port.request_sources):
        raise ValueError('Native function scope requires a registered scalar port')
    return name


def parameter_matches(steps, row, registry, admitted):
    from agent.application.turn_decisions import IntentError
    selected = [s for s in steps if s.tool_name == row['tool']
                and (row.get('native_function') is None or s.arguments.get('function_name') == row['native_function'])]
    if row.get('port') is not None:
        from agent.application.local_resources import _group_port, plan_source_path
        members = (admitted.get('registered_input') or {}).get('members', ())
        index = int(row['source'].split('.')[1])
        role = {k: row[k] for k in ('tool', 'port', 'source')}
        roles = admitted.get('roles', ())
        # Raw declarations retain their original attachment identities outside
        # the derived native-science registration. They select the native input
        # port without creating H5AD group roles or granting derived-file scope.
        if (admitted.get('registered_input') or {}).get('composition') == 'native-science.v1':
            raw = [field for field in ('bam', 'fragments')
                if admitted.get(field + '_context', {}).get('adapter') == 'native-inputs.v1'
                and admitted.get(field, {}).get('source') == row['source']]
            if len(raw) == 1 and row['tool'] == 'run_snapatac2' and row['port'] == 'dataset':
                members = admitted.get(raw[0] + '_sources', {}).get('members', ())
                roles = (role,)
        if index >= len(members) or role not in roles:
            raise IntentError('invalid_reference')
        port, _, _ = _group_port(registry, row['tool'], row['port'])
        selected = [s for s in selected if plan_source_path(s.arguments.get(port.members[0].field_name), steps)
                    == members[index]['source_path']]
    if row.get('step_id') is not None:
        selected = [s for s in selected if s.step_id == row['step_id']]
    failure = admitted.get('native_failure')
    if (failure is not None and row['tool'] == 'run_episcanpy' and row['argument'] == 'kwargs'
            and row.get('native_function') == failure['function_name'] and 'num_clusters' in row['value']):
        selected = [s for s in selected if s.step_id == failure['step_id']]
    return selected


def resolved_parameter(steps, row, registry, admitted, value, *, prepared_values=()):
    """One exact consumer/effective-value projection shared by bind and assert."""
    from agent.application.turn_decisions import IntentError
    from agent.schemas.orchestration import _serialize
    selected=parameter_matches(steps,row,registry,admitted)
    if len(selected)!=1:
        raise IntentError('ambiguous_parameter' if selected else 'invalid_reference')
    step=selected[0]
    overrides=[r for r in prepared_values if r['step_id']==step.step_id and r['tool']==step.tool_name
        and r.get('native_function')==step.arguments.get('function_name') and r['argument']==row['argument']]
    if len(overrides)>1:raise IntentError('invalid_reference')
    if overrides:
        value=({**value,**overrides[0]['value']} if row['argument']=='kwargs' else overrides[0]['value'])
    return step, _serialize(value)


def assert_native_parameter_scopes(steps, rows, registry, admitted):
    """Protect explicitly scoped kwarg members from unrelated native calls.

    Omission preserves native defaults. A separately owned consumer retains its
    own assertion; a supplied unowned member needs the actual installed default.
    Same names in different registered tools confer no shared scope.
    """
    from agent.application.turn_decisions import IntentError
    from agent.schemas.orchestration import _serialize
    protected={};owned=set()
    for row in rows:
        tool=registry.get(row['tool'])
        if row['argument']!='kwargs' or tool.native_functions is None:continue
        selected=parameter_matches(steps,row,registry,admitted)
        if len(selected)!=1:raise IntentError('ambiguous_parameter')
        protected.setdefault(tool.name,set()).update(row['value'])
        owned.update((selected[0].step_id,key) for key in row['value'])
    for step in steps:
        keys=protected.get(step.tool_name,set()) & set(step.arguments.get('kwargs',{}))
        missing=[key for key in keys if (step.step_id,key) not in owned]
        if not missing:continue
        native=registry.get(step.tool_name).native_functions
        metadata={} if native.parameter_metadata is None else native.parameter_metadata(step.arguments.get('function_name'))
        from .scientific_inputs import OperationPreparationContext
        context=OperationPreparationContext(registry.get(step.tool_name),step.step_id,registry,
            {s.step_id:(s.tool_name,s.arguments) for s in steps},admitted,{},[],None,None,{},observe_existing=False)
        for key in missing:
            item=metadata.get(key,{})
            actual=_serialize(step.arguments['kwargs'][key])
            if 'default_when_omitted' in item and actual==_serialize(item['default_when_omitted']):continue
            promised=context.planned_parameter_value(step.arguments,item)
            if promised is not None and actual==_serialize(promised):continue
            raise IntentError('ambiguous_parameter')


def bind_structured_parameters(selected, settings, registry, request_inputs):
    """Bind captured scalar values before selected-method input preparation.

    This edits arguments of already selected operations, never selects operations.
    Repeated unresolved consumers defer materialization to the application's
    existing captured-plan scope choice; they grant no execution authority.
    """
    bound = {step: (tool, dict(arguments)) for step, (tool, arguments) in selected.items()}
    applied, ambiguous = {}, False
    from .semantic_compiler import AnalyzedOperation
    from .active_context import current_context
    context=current_context()
    scope={} if context is None else context.task.get('parameter_sources',{})
    prepared=() if context is None else context.task.get('prepared_parameter_values',())
    for row in settings:
        operations = tuple(AnalyzedOperation(step, tool, args, ()) for step, (tool, args) in bound.items())
        candidates = [op.step_id for op in parameter_matches(operations, row, registry, scope)]
        if len(candidates) != 1:
            if not candidates:
                from .semantic_compiler import SemanticPlanCompileError
                raise SemanticPlanCompileError('INVALID_REQUEST_BINDING', 'Explicit parameter has no selected consumer.')
            ambiguous = True
            continue
        operation, effective = resolved_parameter(operations,row,registry,scope,row['value'],prepared_values=() if context is None else context.task.get('prepared_parameter_values',()))
        step = operation.step_id
        tool_name, arguments = bound[step]
        tool = registry.get(tool_name)
        name = row['argument']
        spec = {**tool.required_arguments, **tool.optional_arguments}[name]
        if not spec.planning.scientific_parameter or not any(
                len(port.members) == 1 and port.members[0].field_name == name
                for port in tool.semantic_planning.consumer_ports):
            raise ValueError('Structured values require a registered scalar port.')
        from agent.schemas.orchestration import _serialize
        value = effective
        if name == 'kwargs':
            value = {**_serialize(arguments.get(name, {})), **value}
            for key in value:
                scalar = tool.optional_arguments.get(key)
                explicit = key in request_inputs or input_name(tool_name, row.get('native_function') or '', key) in request_inputs
                if (scalar and scalar.planning and scalar.planning.planner_choice and not explicit
                        and key in arguments and arguments[key] != value[key]):
                    arguments.pop(key)
        spec.validate(name, value)
        arguments[name] = value
        applied.setdefault(step, set()).add(name)
    if (settings or prepared) and not ambiguous:
        operations=tuple(AnalyzedOperation(step,tool,args,()) for step,(tool,args) in bound.items())
        assert_native_parameter_scopes(operations,(*settings,*prepared),registry,scope)
    return bound, applied, ambiguous


def missing_clustering_parameters(tool, inputs):
    """The two count-based APIs have no default when ASW is explicitly disabled."""
    if tool.name != 'run_episcanpy':
        return ()
    missing = []
    for function in ('tl.kmeans', 'tl.hc'):
        selection = inputs.get(input_name(tool.name, function, 'parameter_selection'))
        if inputs.get('function_name') == function:
            selection = selection or inputs.get('parameter_selection')
        kwargs = inputs.get(input_name(tool.name, function, 'kwargs'),
                            inputs.get('kwargs', {}) if inputs.get('function_name') == function else {})
        if selection == 'native_defaults' and 'num_clusters' not in kwargs:
            missing.append(dict(tool=tool.name, native_function=function, argument='kwargs', member='num_clusters'))
    return tuple(missing)
