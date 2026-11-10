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


def bind_structured_parameters(selected, settings, registry, request_inputs):
    """Bind captured scalar values before selected-method input preparation.

    This edits arguments of already selected operations, never selects operations.
    Repeated unresolved consumers defer materialization to the application's
    existing captured-plan scope choice; they grant no execution authority.
    """
    bound = {step: (tool, dict(arguments)) for step, (tool, arguments) in selected.items()}
    applied, ambiguous = {}, False
    for row in settings:
        candidates = [step for step, (tool, arguments) in bound.items() if tool == row['tool']
            and (row.get('native_function') is None or arguments.get('function_name') == row['native_function']
                 or row['argument'] == 'function_name')
            and (row.get('step_id') is None or step == row['step_id'])]
        if len(candidates) != 1:
            ambiguous = ambiguous or bool(candidates)
            continue
        step = candidates[0]
        tool_name, arguments = bound[step]
        tool = registry.get(tool_name)
        name = row['argument']
        spec = {**tool.required_arguments, **tool.optional_arguments}[name]
        if not spec.planning.scientific_parameter or not any(
                len(port.members) == 1 and port.members[0].field_name == name
                for port in tool.semantic_planning.consumer_ports):
            raise ValueError('Structured values require a registered scalar port.')
        from agent.schemas.orchestration import _serialize
        value = _serialize(row['value'])
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
    return bound, applied, ambiguous
