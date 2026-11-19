"""Research preparation before the existing execution reservation boundary.

Semantic acquisition returns a candidate. Scientific input owners resolve only
selected operations. This coordinator never runs a Runtime to ask a question.
"""
from dataclasses import asdict, replace
from types import SimpleNamespace

from agent.schemas import AgentRequest
from agent.schemas.orchestration import _serialize
from agent.orchestration.active_context import ActivePlanningContext, planning_context
from agent.orchestration.semantic_compiler import (
    analyze_candidate, compile_analysis, AnalysisDisposition, ScientificRequirement, ScientificInputRequired, ScientificInputRequirement,
)
from . import preparation_state as records
from .session_state import digest
from agent.orchestration.scientific_inputs import ScientificDiagnostic


def _direct_source_capture(path):
    """Capture bytes without allowing disappearing files to escape the turn."""
    from agent.tools._file_identity import file_sha256
    source = dict(source_path=path)
    try:
        source['source_sha256'] = file_sha256(path)
    except OSError:
        pass
    return source


def validate_direct_source(interactions, record):
    """Recheck the captured direct source at live preparation/admission only."""
    if record is None:
        return
    source = (record if 'task' in record else records.origin_record(interactions, record))['task']['sources']
    pins = [source] if source.get('source_path') and source.get('source_sha256') else []
    candidate_origin = (record if 'candidate' in record else next(
        (i.preparation for i in interactions if i.turn_id == record['candidate_origin']), {}))
    pins += list((candidate_origin or {}).get('source_pins', ()))
    for answer, _ in records.active_answers(interactions, record):
        value = answer['value']
        if answer['kind']=='resource' and hasattr(value, 'get'):
            if value.get('component'):
                pins.append(dict(source_path=value['component']['path'],source_sha256=value['component']['sha256']))
            pins.extend(dict(source_path=pin['path'],source_sha256=pin['sha256'])
                for pin in value.get('qualified_resource', {}).get('files', ()))
    from agent.tools._file_identity import file_sha256
    from .local_resources import ResourceAdmissionError
    # An acquisition that could not capture a consumed file never gains byte
    # authority by observing that file later, including a lane in a path list.
    if any('source_sha256' not in pin for pin in pins):
        raise ResourceAdmissionError('LOCAL_RESOURCE_INTEGRITY_INVALID')
    pins = list({(pin['source_path'], pin['source_sha256']): pin for pin in pins}.values())
    try:
        unchanged = all(file_sha256(pin['source_path']) == pin['source_sha256'] for pin in pins)
    except OSError:
        unchanged = False
    if not unchanged:
        raise ResourceAdmissionError('LOCAL_RESOURCE_INTEGRITY_INVALID')


def _model_catalogs(interactions, record, *, addition=None):
    """Append explicit recaptures while keeping earlier model decisions exact."""
    from .dialogue_execution import resource_context_digest
    from .local_resources import QualifiedEpiZooResource, qualified_epizoo_resources
    original = record if 'task' in record else records.origin_record(interactions, record)
    receipt = original['task'].get('context', {}).get('resource_selection')
    if receipt is None:
        return ()
    original_models = qualified_epizoo_resources(tuple(QualifiedEpiZooResource(**_serialize(row)) for row in receipt['catalog']))
    if resource_context_digest(original_models) != receipt['resource_context_sha256']:
        raise ValueError('Captured qualified model catalog changed.')
    batches = [i.snapshot['preparation_resources']['models'] for i in interactions
        if i.preparation is not None and i.preparation['task_id']==record['task_id'] and 'resume' in i.preparation
        and 'models' in i.snapshot.get('preparation_resources', {})
        and not any(d.get('code') == 'PREPARATION_RESOURCE_CHANGED'
                    for d in (records.observation(i) or {}).get('diagnostics', ()))]
    if addition is not None:
        batches.append(addition)
    models = {row.resource_id:row for row in original_models}
    catalogs = [original_models]
    for batch in batches:
        additions = qualified_epizoo_resources(tuple(QualifiedEpiZooResource(**_serialize(row)) for row in batch))
        for row in additions:
            if row.resource_id in models and models[row.resource_id] != row:
                raise ValueError('Captured qualified model definition changed.')
            models[row.resource_id] = row
        catalogs.append(tuple(models.values()))
    return tuple(catalogs)


def _context(sessions, interaction, admitted):
    from .dialogue_execution import planning_task, _execution_objective
    state = sessions.load(sessions._interaction_session_id)
    current = replace(interaction, admitted=admitted)
    state = replace(state, interactions=tuple(current if i.turn_id == current.turn_id else i
                                               for i in state.interactions))
    task = planning_task(current, state, sessions._application.registry)
    objective = _execution_objective(current, state.interactions)
    if 'selected_candidate' in admitted:
        from .guidance_selection import execution_context, planning_intent
        context = execution_context(sessions, current, admitted)
        objective = planning_intent(sessions, current, admitted, context)
        context = replace(context, task=task)
    else:
        context, _ = sessions.execution_context_for_interaction(sessions._interaction_session_id,
                                                              interaction.turn_id, admitted=admitted)
        context = (replace(context, task=task) if context is not None else ActivePlanningContext(
            sessions._interaction_session_id, interaction.base_revision_id, interaction.base_generation, (), task=task))
    return state, current, context, objective


def _owner_requirements(candidate_id, step_id, tool, arguments, exc, *, operations=(), registry=None):
    ports = {port.name: port for port in tool.semantic_planning.consumer_ports}
    for requirement in exc.requirements:
        if (requirement.port not in ports or requirement.argument not in
                {member.field_name for member in ports[requirement.port].members}):
            raise ValueError('Scientific input owner reported an unregistered operand.')
    return [dict(ScientificRequirement(candidate_id, step_id, tool.name,
            arguments.get('function_name'), requirement.port, requirement.argument,
            requirement.kind, member=requirement.member, choices=requirement.choices,
            dependencies=tuple(step_id + ':' + key for key in requirement.dependencies)).to_dict(),
        source_dependencies=records.operation_sources(step_id, operations, registry),
        source_steps=records.operation_dependencies(step_id, operations),
        applicability=_serialize(requirement.applicability), offers=_serialize(requirement.offers),
        messages=list(requirement.messages)) for requirement in exc.requirements]


def scoped_source_admission(admitted, intake_owner, captured_source, paths):
    """Bind source declarations only through the selected Registry lineage."""
    scoped = dict(admitted)
    applies = True
    if intake_owner in {'fastq','bam','fragments','annotation','epizoo'}:
        declared_paths = {member['source_path'] for member in
                          records.declaration_members(admitted, intake_owner, captured_source)}
        if not declared_paths and captured_source.get('source_path'):
            declared_paths.add(captured_source['source_path'])
        if not declared_paths:
            declared_paths.update('accepted:' + digest(binding) for binding in captured_source.get('accepted_results', ()))
        applies = bool(set(paths) & declared_paths)
        if not applies:
            for key in tuple(scoped):
                if key == intake_owner or key.startswith(intake_owner + '_') or key.startswith('downstream_annotation') or key == 'raw_resource_choice':
                    scoped.pop(key, None)
    return scoped, applies


def source_field_path(tool, argument, arguments, selected, registry=None, *, store=None):
    from .source_observations import existing_object
    spec = tool.required_arguments.get(argument, tool.optional_arguments.get(argument))
    if spec is None or spec.planning is None or spec.planning.observation_source is None:
        return None
    port = next(p for p in tool.semantic_planning.consumer_ports if p.name == spec.planning.observation_source)
    # Existing object resolution retains accepted publication authority. Future
    # outputs remain symbolic except the registered byte-transparent adapter.
    value = existing_object(arguments.get(port.members[0].field_name), selected,
        registry or __import__('agent.orchestration.registry', fromlist=['build_default_tool_registry']).build_default_tool_registry(), store=store)
    return None if value is None else value.path


def source_field_offers(registry, interaction, preceding, tool, argument, arguments, selected, *, store=None):
    """Project real captured fields for one exact consumer's dataset port.

    Text declarations and UI offers share source_field_inputs. This projection
    adds question/consumer identities, without choosing a scientific grouping.
    """
    from .dialogue_execution import captured_input_group
    from .local_resources import source_field_inputs
    spec = tool.required_arguments.get(argument, tool.optional_arguments.get(argument))
    if spec is None or spec.planning is None or spec.planning.observation_source is None:
        return []
    binding = captured_input_group(interaction, preceding)
    port = next(p for p in tool.semantic_planning.consumer_ports if p.name == spec.planning.observation_source)
    path = source_field_path(tool, argument, arguments, selected, registry, store=store)
    offers = []
    for index, member in enumerate(() if binding is None else binding.members):
        if member['source_path'] != path:
            continue
        role = dict(tool=tool.name, port=port.name, source='source.' + str(index))
        fields = source_field_inputs(registry, binding, (role,),
            interaction.snapshot.get('dialogue', {}).get('source_fields', ())).get((tool.name, argument), ())
        for name in fields:
            identity = dict(tool=tool.name, argument=argument, port=port.name,
                axis=spec.planning.observation_axis, source=role['source'], resource_id=member['resource_id'],
                source_sha256=member['source_sha256'])
            offers.append(dict(handle='field.' + digest(dict(identity, value=name)),
                label=identity['axis'] + '.' + name, argument=argument, value=name, source_field=identity,
                component=dict(path=member['source_path'], sha256=member['source_sha256'])))
    if not offers:
        from .source_observations import existing_object
        observed = existing_object(arguments.get(port.members[0].field_name), selected, registry, store=store)
        if observed is not None:
            identity = dict(tool=tool.name, argument=argument, port=port.name,
                axis=spec.planning.observation_axis, **observed.identity())
            for name in observed.fields(spec.planning.observation_axis):
                offers.append(dict(handle='field.' + digest(dict(identity, value=name)),
                    label=identity['axis'] + '.' + name, argument=argument, value=name,
                    source_field=identity, component=dict(path=observed.path, sha256=observed.sha256)))
    return offers


def prepare(sessions, interaction, admitted, *, record=None, intake_needs=None, intake_owner=None):
    """Return a strictly preflighted plan or a persisted non-executable question."""
    from .turns import _update
    from .shared_resources import scientific_resource_scope
    from agent.orchestration.prior_outputs import prior_output_store
    from agent.orchestration.planner import PlannerError
    from .interactive_schemas import ClientError
    admitted = {k: v for k, v in admitted.items() if k not in {'choices', 'resource_options'}}
    def invalid_source(record):
        from .local_resources import ResourceAdmissionError
        failure = ResourceAdmissionError('LOCAL_RESOURCE_INTEGRITY_INVALID')
        error = ClientError(failure.code, str(failure))
        result = records.question(record, interaction, (), status='INVALID',
            diagnostics=(dict(code=error.code),))
        _update(sessions, interaction.turn_id, preparation=record)
        return dict(admitted, preparation_result=result), None, error, None

    app = sessions._application
    resources = getattr(sessions, '_epizoo_resources', ())
    from .dialogue_execution import resource_context_digest
    resource_context = dict(resource_selection=dict(catalog=[asdict(r) for r in resources],
        resource_context_sha256=resource_context_digest(resources)))
    state = sessions.load(sessions._interaction_session_id)
    record = record or interaction.preparation
    if intake_owner is not None and record is None:
        sources = admitted.get(intake_owner + '_sources') or admitted.get('registered_input') or admitted.get('inputs', {})
        if not admitted.get('registered_input') and isinstance(admitted.get('inputs', {}).get('input_path'), str):
            sources = _direct_source_capture(admitted['inputs']['input_path'])
        if admitted.get('input_results') or admitted.get('context') == 'current':
            accepted, _ = sessions.execution_context_for_interaction(state.session_id, interaction.turn_id, admitted=admitted)
            if accepted is not None:
                from agent.orchestration.prior_outputs import source_port
                datasets = [item.binding for item in accepted.items if any(m.name == 'dataset' for m in
                    source_port(app.registry, item.binding.tool_name, item.binding.source_port).members)]
                # Context visibility is not a declaration of every visible source.
                # Multiple datasets require consumer-scoped answers after selection.
                sources = dict(sources, accepted_results=[datasets[0].to_dict()] if len(datasets) == 1 else [])
        record = records.start(state.session_id, interaction, interaction.utterance,
            sources=sources, context=dict(intake_owner=intake_owner, **resource_context))
        if sources.get('source_path') and not sources.get('source_sha256'):
            return invalid_source(record)
    if intake_needs:
        if record is None:
            raise ValueError('Scientific intake needs a canonical task.')
        result = records.question(record, interaction, intake_needs)
        _update(sessions, interaction.turn_id, preparation=record,
                admitted=dict(admitted, preparation_result=result), status='clarification')
        return dict(admitted, preparation_result=result), None, None, None
    if admitted.get('kind') == 'request_resources':
        if record is None:
            record = records.start(state.session_id, interaction, interaction.utterance,
                sources=admitted.get('registered_input', admitted.get('inputs', {})), context=dict(intake_owner='resources', **resource_context))
        origin = (interaction if record['task_origin'] == interaction.turn_id else
                  next(i for i in state.interactions if i.turn_id == record['task_origin']))
        requirements = resource_requirements(sessions, origin, admitted)
        result = records.question(record, interaction, requirements)
        _update(sessions, interaction.turn_id, preparation=record,
                admitted=dict(admitted, preparation_result=result), status='clarification')
        return dict(admitted, preparation_result=result), None, None, None
    if admitted.get('kind') == 'request_roles':
        if record is None:
            record = records.start(state.session_id, interaction, interaction.utterance,
                sources=admitted['registered_input'], context=dict(intake_owner='roles', **resource_context))
        requirements = role_requirements(admitted, app.registry)
        result = records.question(record, interaction, requirements)
        _update(sessions, interaction.turn_id, preparation=record,
                admitted=dict(admitted, preparation_result=result), status='clarification')
        return dict(admitted, preparation_result=result), None, None, None
    if record is not None:
        if record['candidate_origin'] is not None:
            _update(sessions, interaction.turn_id, preparation=record)
        interaction = replace(interaction, preparation=record)
    state, current, context, objective = _context(sessions, interaction, admitted)
    request = AgentRequest(admitted['request_id'], objective, admitted['inputs'])
    prepared, _ = app._prepare_request(request)
    reference_owner = getattr(sessions, '_reference_resources', None)
    record = interaction.preparation
    if record is None:
        latest = next(i for i in state.interactions if i.turn_id == interaction.turn_id)
        record = latest.preparation
    with scientific_resource_scope(None if reference_owner is None else reference_owner.library), planning_context(context), prior_output_store(app.run_store):
        candidate = None if record is None else records.candidate_for(state.interactions, record)
        if candidate is None:
            planner = app.runtime.planner
            if not callable(getattr(planner, 'acquire_candidate', None)):
                raise ValueError('Research preparation requires a wire-v4 candidate planner.')
            try:
                acquired = planner.acquire_candidate(prepared, app.registry)
            except PlannerError as exc:
                return admitted, None, ClientError(exc.code, str(exc)), None
            analysis = acquired.attempt.analysis
            candidate = analysis.candidate
            contract_identity = acquired.attempt.context.catalog_fingerprint
            if record is None:
                sources = admitted.get('registered_input') or admitted.get('inputs', {})
                if not admitted.get('registered_input') and isinstance(admitted.get('inputs', {}).get('input_path'), str):
                    sources = _direct_source_capture(admitted['inputs']['input_path'])
                record = records.start(sessions._interaction_session_id, interaction, objective,
                    candidate=candidate, contract_identity=contract_identity,
                    sources=sources,
                    context=dict(base_revision_id=interaction.base_revision_id, task=_serialize(context.task), **resource_context))
                if sources.get('source_path') and not sources.get('source_sha256'):
                    return invalid_source(record)
            else:
                from agent.orchestration.semantic_compiler import candidate_record, candidate_identity
                record = dict(record, candidate=candidate_record(candidate), candidate_id=candidate_identity(candidate),
                              candidate_origin=interaction.turn_id, contract_identity=contract_identity)
            # The compiler identifies consumed direct paths. Offered attachments
            # and symbolic accepted outputs are not additional consumed files.
            from pathlib import Path
            captured = (record if 'task' in record else records.origin_record(state.interactions, record))['task']['sources']
            registered = admitted.get('registered_input') or {}
            known = [captured, *captured.get('members', ()), registered, *registered.get('members', ())]
            origin = interaction if record['task_origin'] == interaction.turn_id else next(
                i for i in state.interactions if i.turn_id == record['task_origin'])
            received = origin.submission or {}
            paths = {pin['source_path'] for pin in known if pin.get('source_path')}
            if not received.get('registered_input'):
                from collections.abc import Mapping
                def leaves(value):
                    if isinstance(value, str):
                        return {value}
                    children = value.values() if isinstance(value, Mapping) else value if isinstance(value, (list, tuple)) else ()
                    return set().union(*(leaves(v) for v in children)) if children else set()
                paths.update(leaves(received.get('execution_inputs', {})))
            consumed = set() if analysis.disposition is AnalysisDisposition.INVALID else {
                path for op in analysis.operations
                for path in records.operation_sources(op.step_id, analysis.operations, app.registry)}
            # Auto-bound references/models retain their resource owner's closure.
            # Pin only actual consumed user sources, including secondary inputs.
            paths = sorted(paths & consumed)
            record['source_pins'] = []
            for path in paths:
                # Directory/research-object closures retain their native owner.
                if Path(path).is_dir():
                    continue
                hashes = {pin['source_sha256'] for pin in known
                    if pin.get('source_path') == path and pin.get('source_sha256')}
                hashes.update(admitted['inputs'][key.removesuffix('_path') + '_sha256']
                    for key, value in admitted['inputs'].items() if key.endswith('_path') and value == path
                    and key.removesuffix('_path') + '_sha256' in admitted['inputs'])
                if len(hashes) > 1:
                    raise ValueError('Consumed source has conflicting captured byte identities.')
                pin = dict(source_path=path, source_sha256=next(iter(hashes))) if hashes else _direct_source_capture(path)
                record['source_pins'].append(pin)
            if any('source_sha256' not in pin for pin in record['source_pins']):
                return invalid_source(record)
            _update(sessions, interaction.turn_id, preparation=record)
        else:
            bindings = records.scalar_bindings(state.interactions, record)
            analysis = analyze_candidate(prepared, candidate, app.registry, bindings=bindings)
        if interaction.preparation is not None and record is not None:
            _update(sessions, interaction.turn_id, preparation=record)
        from .local_resources import ResourceAdmissionError
        try:
            validate_direct_source(state.interactions, record)
        except ResourceAdmissionError:
            return invalid_source(record)
        # Parsed invalidity always wins; no scientific owner or prompt hides it.
        owner_requirements, bindings_receipt = [], []
        if analysis.disposition is not AnalysisDisposition.INVALID:
            from .native_raw_handoff import coordinates
            history = records.active_answers(state.interactions, record)
            scoped_resources = records.resource_bindings(state.interactions, record)
            source_scopes = {op.step_id: set(records.operation_sources(op.step_id, analysis.operations,
                             app.registry, lineage=True)) for op in analysis.operations}
            declarations = [(source_scopes[op.step_id], {key: op.arguments.get(key)
                             for key in ('species', 'assembly')}) for op in analysis.operations
                            if any(op.arguments.get(key) is not None for key in ('species', 'assembly'))]
            coordinate_choices = [(answer, requirement) for answer, requirement in history
                                  if answer['kind'] == 'resource' and answer['value'].get('coordinates')]
            task = (record if 'task' in record else records.origin_record(state.interactions, record))['task']
            intake_owner = task['context'].get('intake_owner')
            captured_source = task['sources']
            field_questions = {(answer['step_id'], answer['argument']): next(
                i for i in state.interactions if i.turn_id == answer['question_turn_id'])
                for answer, _ in records.active_answers(state.interactions, record)
                if answer['kind'] == 'resource' and 'source_field' in answer['value']}
            def resolve(tool, arguments, selected, *, step_id):
                original = dict(arguments)
                scoped = dict(admitted)
                selected_resources = scoped_resources.get(step_id, {})
                paths = source_scopes[step_id]
                scoped, _ = scoped_source_admission(admitted, intake_owner, captured_source, paths)
                raw = scoped.get(str(intake_owner) + '_context', {})
                if raw.get('adapter') != 'native-inputs.v1':
                    raw = {}
                ids = raw.get('selected') or ()
                if isinstance(ids, str):
                    ids = (ids,)
                selected_coordinates = coordinates([raw['catalog'].get(key) for key in ids])
                matched = []
                own = {}
                references = {}
                for answer, requirement in coordinate_choices:
                    if paths and paths == source_scopes.get(answer['step_id']):
                        value = answer['value']
                        matched.append(value['coordinates'])
                        references.setdefault(requirement['port'], set()).add(value['handle'].removeprefix('resource.'))
                        if step_id == answer['step_id']:
                            own.update(value['coordinates'])
                scoped['source_coordinates'] = coordinates(matched) | own | selected_coordinates | coordinates(
                    [row for scope, row in declarations if paths and paths == scope]) | {
                    key: raw[key] for key in ('species', 'assembly') if raw.get(key) is not None}
                scoped['coordinate_resources'] = {port: next(iter(ids)) for port, ids in references.items()
                                                  if len(ids) == 1}
                try:
                    from pathlib import Path
                    from agent.tools.analysis.categorical import has_field
                    for name, spec in (*tool.required_arguments.items(), *tool.optional_arguments.items()):
                        if name not in arguments or spec.planning is None or spec.planning.observation_source is None:
                            continue
                        source = source_field_path(tool, name, arguments, selected, app.registry)
                        if isinstance(source, (str, Path)) and not has_field(source, arguments[name], axis=spec.planning.observation_axis):
                            raise ScientificInputRequired(ScientificInputRequirement(
                                next(p.name for p in tool.semantic_planning.consumer_ports if any(m.field_name == name for m in p.members)),
                                name, dependencies=('source:bytes',), messages=('The selected field is absent from the required source axis. Select an actual field for this consumer.',)))
                    for name, answer in selected_resources.items():
                        if 'source_field' not in answer:
                            continue
                        # An exact answer retains its question's source identity.
                        # A task recheck may offer only newly uploaded files; it
                        # must not reidentify the selected original field from
                        # that unrelated current attachment view. Current graph
                        # operands and the live field/byte checks still apply.
                        question = field_questions[(step_id, name)]
                        offers = source_field_offers(app.registry, question, state.interactions, tool,
                            name, arguments, selected)
                        if _serialize(answer) not in offers:
                            raise ValueError('Selected source field no longer matches its exact consumer, axis or source.')
                        from agent.tools._file_identity import checked_file
                        checked_file(answer['component']['path'], answer['component']['sha256'])
                    if tool.preparation_owner is not None:
                        from agent.orchestration.scientific_inputs import OperationPreparationContext
                        owner_context = OperationPreparationContext(tool, step_id, app.registry, selected, scoped,
                            selected_resources, bindings_receipt, sessions, current, record,
                            _model_catalogs(sessions.load(sessions._interaction_session_id).interactions, record))
                        arguments = tool.preparation_owner(arguments, owner_context)
                    from agent.orchestration.scientific_inputs import OperationPreparationContext, validate_input_properties
                    validate_input_properties(arguments,OperationPreparationContext(tool,step_id,app.registry,selected,
                        scoped,selected_resources,bindings_receipt,sessions,current,record))
                except ScientificInputRequired as exc:
                    owner_requirements.extend(_owner_requirements(analysis.candidate_id, step_id, tool, original, exc, operations=analysis.operations, registry=app.registry))
                    return original if exc.bindings is None else exc.bindings
                return arguments
            bindings = {} if record is None else records.scalar_bindings(state.interactions, record)
            from .local_resources import ResourceAdmissionError
            try:
                analysis = analyze_candidate(prepared, candidate, app.registry, bindings=bindings, prepare_inputs=resolve)
            except (ResourceAdmissionError, ScientificDiagnostic) as exc:
                result = records.question(record, interaction, (), status='INVALID', diagnostics=(dict(code=exc.code, message=str(exc)),))
                return dict(admitted, preparation_result=result), None, (exc.error if isinstance(exc, ResourceAdmissionError) else ClientError(exc.code, str(exc))), None
        if analysis.disposition is AnalysisDisposition.INVALID:
            result = records.question(record, interaction, (), status='INVALID', diagnostics=(dict(
                code=analysis.error.code, fields=analysis.error.diagnostic_fields),))
            return dict(admitted, preparation_result=result), None, ClientError(analysis.error.code, str(analysis.error)), None
        requirements = [r.to_dict() for r in analysis.requirements]
        from .dialogue_execution import _parameter_input
        for index, requested in enumerate(admitted.get('requested_parameters', ())):
            row = dict(tool=admitted['tool'], argument=requested) if isinstance(requested, str) else requested
            matches = [op for op in analysis.operations if op.tool_name == row['tool']
                and (row.get('native_function') is None or op.arguments.get('function_name') == row['native_function'])
                and (row.get('step_id') is None or op.step_id == row['step_id'])]
            if len(matches) > 1:
                requirements.append(ScientificRequirement(analysis.candidate_id, '@request:' + str(index), row['tool'],
                    row.get('native_function'), row['argument'], row['argument'], 'scope', member=row.get('member'),
                    choices=tuple(op.step_id for op in matches)).to_dict())
            elif len(matches) == 1:
                op = matches[0]
                name, spec = _parameter_input(app.registry, op.tool_name, row['argument'])
                value = op.arguments.get(row['argument'])
                if row['argument'] not in op.arguments or row.get('member') and (not isinstance(value, dict) and not hasattr(value, 'keys') or row['member'] not in value):
                    requirements.append(ScientificRequirement(analysis.candidate_id, op.step_id, op.tool_name,
                        op.arguments.get('function_name'), row['argument'], row['argument'], 'scalar',
                        member=row.get('member'), choices=tuple(spec.choices)).to_dict())
            else:
                raise ValueError('Requested scientific parameter has no selected operation.')
        owner_steps = {r['step_id'] for r in owner_requirements}
        requirements = [r for r in requirements if not (r['kind'] == 'application' and r['step_id'] in owner_steps)]
        owner_keys = {(r['step_id'],r['argument'],r.get('member')) for r in owner_requirements}
        requirements = [r for r in requirements if (r['step_id'],r['argument'],r.get('member')) not in owner_keys]
        requirements.extend(owner_requirements)
        selected = {op.step_id:(op.tool_name, dict(op.arguments)) for op in analysis.operations}
        for index, req in enumerate(requirements):
            operation = next((op for op in analysis.operations if op.step_id == req['step_id']), None)
            if operation is None:
                continue
            if req['kind']=='external_input':
                from .source_observations import source_offers
                offers = source_offers(current, state.interactions, app.registry.get(operation.tool_name), req['port'])
                if offers:
                    addressed = ScientificRequirement(analysis.candidate_id, req['step_id'], req['tool_name'],
                        req['native_function'],req['port'],req['argument'],'resource',
                        dependencies=tuple(req.get('dependencies', ())))
                    requirements[index]=dict(req,**addressed.to_dict(),offers=offers)
                continue
            offers = source_field_offers(app.registry, current, state.interactions,
                app.registry.get(operation.tool_name), req['argument'], operation.arguments, selected)
            if offers:
                field_req = ScientificRequirement(analysis.candidate_id, req['step_id'], req['tool_name'],
                    req['native_function'], req['port'], req['argument'], 'resource',
                    member=req.get('member'), dependencies=tuple(req.get('dependencies', ())))
                requirements[index] = dict(req, **field_req.to_dict(), offers=offers,
                    messages=list(req.get('messages', ())) + ['Choose an exact observed field. Observed fields do not establish biological identity; bounded offers may omit private or unsupported metadata.'])
        requirements = [dict(req,
            source_dependencies=records.operation_sources(req['step_id'], analysis.operations, app.registry),
            source_steps=records.operation_dependencies(req['step_id'], analysis.operations)) for req in requirements]
        if requirements:
            result = records.question(record, interaction, requirements, blocked=analysis.blocked_checks)
            return dict(admitted, preparation_result=result, native_resource_bindings=bindings_receipt), None, None, None
        from .local_resources import ResourceAdmissionError
        try:
            validate_direct_source(state.interactions, record)
        except ResourceAdmissionError as exc:
            result = records.question(record, interaction, (), status='INVALID',
                diagnostics=(dict(code=exc.code),))
            return dict(admitted, preparation_result=result), None, ClientError(exc.code, exc.message), None
        plan = compile_analysis(prepared, analysis, app.registry, planner_name=app.runtime.planner.name)
        verification = app.runtime.executor.preflight(plan)
        if not verification.passed:
            error = verification.error or ClientError('PLAN_PREFLIGHT_FAILED', 'Scientific plan preflight failed; execution has not started.')
            result = records.question(record, interaction, (), status='INVALID',
                diagnostics=(dict(code=error.code),))
            return dict(admitted, preparation_result=result), None, error, None
        result = records.question(record, interaction, (), status='READY')
        return dict(admitted, preparation_result=result, native_resource_bindings=bindings_receipt), None, None, plan


def typed_answer(sessions, interaction):
    """Reduce every exact preparation answer once, before language dispatch."""
    selection = (interaction.submission or {}).get('clarification_choice') or {}
    if not selection.get('handle', '').startswith('spp.'):
        return None
    state = sessions.load(sessions._interaction_session_id)
    preceding = tuple(i for i in state.interactions if i.turn_id != interaction.turn_id)
    previous = next((i for i in preceding if i.turn_id == selection.get('question_turn_id')), None)
    if previous is None:
        from .session_state import SessionConflictError
        raise SessionConflictError('Unknown preparation question.')
    offered = next((c for c in records.choices(previous) if c['handle'] == selection['handle']), None)
    if offered is None:
        from .session_state import SessionConflictError
        raise SessionConflictError('Preparation choice was not offered.')
    if offered.get('action')=='upload_source':
        from .session_state import SessionConflictError
        raise SessionConflictError('Upload a file and explicitly recheck this task before choosing its exact source.')
    if offered.get('action')=='execute_preparation':
        from .session_state import SessionConflictError
        task_turns={i.turn_id for i in preceding if i.preparation is not None and i.preparation['task_id']==previous.preparation['task_id']}
        if (records.head(preceding,previous.preparation['task_id'])!=previous or previous.status!='answered'
                or previous.base_generation!=interaction.base_generation or previous.base_revision_id!=interaction.base_revision_id
                or any(t.turn_id in task_turns for t in state.turns)):
            raise SessionConflictError('Prepared execution authorization is stale or already submitted.')
        record=records.record_answers(previous,interaction,{},provenance='exact_submission')
        record['authorization']=dict(question_turn_id=previous.turn_id,question_version=records.observation(previous)['question_version'])
        return records.continued_admission(preceding,record,'turn-'+digest(dict(session=state.session_id,turn=interaction.turn_id))),record,(),None
    if offered.get('action') != 'parameter':
        req = next(r for r in records.observation(previous)['requirements']
                   if r['requirement_id'] == offered['requirement_id'])
        index = int(selection['handle'].rsplit('.', 1)[1])
        value = (req['choices'][index] if req['kind'] in ('scope', 'scalar') else _serialize(req['offers'][index]))
        record = records.answer_question(preceding, interaction, values={req['requirement_id']: value})
        if req['kind'] == 'declaration':
            from .raw_handoff import declaration
            from .turn_decisions import RoleAnswer
            decision = RoleAnswer(pending='@roles', roles=(), **{req['argument']:declaration(value['value'], req['argument'])})
            admitted, needs = source_admission(sessions, interaction, decision)
            return (admitted, record, needs, req['argument'])
        if req['kind'] == 'input_resource':
            return resource_answer_admission(sessions, interaction, record), record, (), None
        if req['kind'] == 'input_role':
            return role_answer_admission(sessions, interaction, record), record, (), None
        return records.continued_admission(preceding, record,
            'turn-' + digest(dict(session=sessions._interaction_session_id, turn=interaction.turn_id))), record, (), None
    settings = (interaction.submission or {}).get('parameter_overrides', ())
    values = {}
    for setting in settings:
        matching = [r for r in records.observation(previous)['requirements'] if r['kind'] == 'scalar'
            and r['requirement_id'] == offered['requirement_id']
            and r['tool_name'] == setting['tool'] and r['argument'] == setting['argument']
            and r['native_function'] == setting.get('native_function')
            and r['step_id'] == setting.get('step_id')]
        if len(matching) != 1:
            from .session_state import SessionConflictError
            raise SessionConflictError('Parameter form changed its selected scientific operation.')
        req = matching[0]
        value = setting['value'] if req.get('member') is None else setting['value'][req['member']]
        values[req['requirement_id']] = value
    record = records.answer_question(preceding, interaction, values=values)
    if (previous.prerequisite or {}).get('native_failure'):
        from .turn_decisions import ParameterAnswer,ExplicitArgument
        from .dialogue_execution import admit_parameters
        admitted=admit_parameters(sessions,interaction,ParameterAnswer('@parameters',tuple(
            ExplicitArgument(**_serialize(row)) for row in settings)),getattr(sessions,'_epizoo_resources',()))
        return admitted,record,(),None
    return records.continued_admission(preceding, record,
        'turn-' + digest(dict(session=sessions._interaction_session_id, turn=interaction.turn_id))), record, (), None


def role_requirements(admitted, registry):
    """Task-scoped input needs, before a semantic candidate exists."""
    assigned = {(r['tool'], r['port']) for r in admitted['roles']}
    requirements = []
    for role in admitted['required_roles']:
        if (role['tool'], role['port']) in assigned:
            continue
        tool = registry.get(role['tool'])
        port = next(p for p in tool.semantic_planning.consumer_ports if p.name == role['port'])
        identity = dict(step_id='@task', tool_name=role['tool'], native_function=None,
                        port=role['port'], argument=port.members[0].field_name, member=None, kind='input_role')
        offers = [dict(handle='source.' + str(index), label=member['label'],
                       role=dict(tool=role['tool'], port=role['port'], source='source.' + str(index)),
                       source=_serialize(member))
                  for index, member in enumerate(admitted['registered_input']['members'])]
        requirements.append(dict(identity, requirement_id=digest(identity), candidate_id=None,
                                 dependencies=[], choices=[], offers=offers))
    return requirements


def role_answer_admission(sessions, interaction, record, *, arguments=()):
    """Apply associations through the original source/role legality owner."""
    from .dialogue_execution import _role_completion, _role_binding
    from .local_resources import LocalResourceAdmission
    from .turn_decisions import InputRole
    state = sessions.load(sessions._interaction_session_id)
    original = next(i for i in state.interactions if i.turn_id == record['task_origin'])
    previous = next(i for i in state.interactions if i.turn_id == record['previous'])
    roles = tuple(InputRole(**_serialize(a['value']['role'])) for a in record['answers'])
    owner = LocalResourceAdmission(sessions._application._workspace, registry=sessions._application.registry)
    owner.validate_binding(_role_binding(original, previous, interaction))
    values, roles, missing, metadata = _role_completion(sessions._application.registry,
        original, previous, interaction, roles, arguments)
    admitted = {k: _serialize(v) for k, v in previous.admitted.items()
                if k not in {'preparation_result', 'native_resource_bindings', 'choices', 'resource_options', 'arguments'}}
    admitted.update(kind='request_roles' if missing else 'execute', operation='plan',
        revision_id=interaction.base_revision_id, inputs=values, roles=roles, registered_input=metadata,
        arguments=[_serialize(vars(a)) for a in arguments],
        request_id='turn-' + digest(dict(session=state.session_id, turn=interaction.turn_id)))
    return admitted


def interpreted_roles(sessions, interaction, decision):
    """Reduce an interpreted role answer against the captured canonical task."""
    from .dialogue_execution import _role_binding
    from .session_state import SessionConflictError
    from .turn_decisions import IntentError
    if decision.pending != '@roles':
        raise IntentError('invalid_prerequisite')
    state = sessions.load(sessions._interaction_session_id)
    if state.interactions[-1].turn_id != interaction.turn_id:
        raise IntentError('invalid_prerequisite')
    preceding = tuple(i for i in state.interactions if i.turn_id != interaction.turn_id)
    original, previous = records.intake_context(interaction, preceding)
    binding = _role_binding(original, previous, interaction)
    requirements = role_requirements(previous.admitted, sessions._application.registry)
    answers = []
    for role in decision.roles:
        matches = [r for r in requirements if (r['tool_name'], r['port']) == (role.tool, role.port)]
        if not matches and _serialize(vars(role)) in _serialize(previous.admitted['roles']):
            continue
        if len(matches) != 1 or not role.source.startswith('source.'):
            raise IntentError('invalid_prerequisite')
        index = int(role.source.split('.')[1])
        if index >= len(binding.members):
            raise IntentError('invalid_prerequisite')
        req = matches[0]
        value = dict(handle=role.source, label=binding.members[index]['label'],
                     role=_serialize(vars(role)), source=_serialize(binding.members[index]))
        answers.append(dict(requirement_id=req['requirement_id'], value=value,
            answering_interaction=interaction.turn_id, question_version=records.observation(previous)['question_version'],
            question_turn_id=previous.turn_id, step_id=req['step_id'], argument=req['argument'],
            member=req['member'], kind=req['kind'], provenance='interpreted_role'))
    record = records.record_answers(previous, interaction,
        {a['requirement_id']:a['value'] for a in answers}, provenance='interpreted_role')
    return role_answer_admission(sessions, interaction, record, arguments=decision.arguments), record


def validate_role_admission(interaction, preceding):
    """Historical replay of canonical role science, without opening files."""
    from .dialogue_execution import _role_completion, _role_binding
    from .turn_decisions import _scientific_arguments, InputRole
    from .session_state import SessionError
    from agent.orchestration.registry import build_default_tool_registry
    admitted, record = interaction.admitted, interaction.preparation
    if not admitted:
        return
    if record['previous'] is None:
        original, previous = interaction, None
        roles = tuple(InputRole(**_serialize(r)) for r in admitted['roles'])
    else:
        original = next(i for i in preceding if i.turn_id == record['task_origin'])
        previous = next(i for i in preceding if i.turn_id == record['previous'])
        if any(a['kind'] != 'input_role' for a in record['answers']):
            # Once intake finishes, normal exact candidate answers retain its
            # admitted sources; verify the entire admission, not a new intake.
            expected = records.continued_admission(preceding, record, admitted['request_id'])
            actual = {k: _serialize(v) for k, v in admitted.items()
                      if k not in {'preparation_result','native_resource_bindings','choices','resource_options'}}
            if actual != expected:
                raise SessionError('Canonical role task changed after intake.')
            return
        roles = tuple(InputRole(**_serialize(a['value']['role'])) for a in record['answers'])
        binding = _role_binding(original, previous, interaction)
        for answer in record['answers']:
            index = int(answer['value']['role']['source'].split('.')[1])
            if index >= len(binding.members) or _serialize(binding.members[index]) != _serialize(answer['value']['source']):
                raise SessionError('Canonical role answer changed its exact source identity.')
        if admitted['required_roles'] != original.admitted['required_roles'] or admitted['tool'] != original.admitted['tool']:
            raise SessionError('Canonical role task changed its requested operations.')
    values, roles, missing, metadata = _role_completion(build_default_tool_registry(), original, previous,
        interaction, roles, _scientific_arguments(_serialize(admitted.get('arguments', []))))
    if (values != _serialize(admitted['inputs']) or roles != _serialize(admitted['roles'])
            or metadata != _serialize(admitted['registered_input']) or interaction.prerequisite is not None):
        raise SessionError('Canonical role admission changed its scientific binding.')
    if missing:
        result = records.observation(interaction)
        if result is None or _serialize(result['requirements']) != role_requirements(admitted, build_default_tool_registry()):
            raise SessionError('Canonical intake lost its unresolved roles.')


def resume_resource_conflict(previous, captured):
    if captured is None:
        return False
    current = {row['input_set_id']: row for row in captured['catalog']}
    old = previous.admitted or {}
    return (any(row['input_set_id'] in current and row != current[row['input_set_id']]
                for row in old.get('resource_catalog', ()))
            or any(key in old.get('raw_resource_evidence', {}) and value != old['raw_resource_evidence'][key]
                   for key, value in captured.get('evidence', {}).items()))


def resume_admission(interaction, preceding, record):
    """Retain exact admission, recapturing only explicit unbound role offers."""
    previous = next(i for i in preceding if i.turn_id == record['previous'])
    admitted = {k: _serialize(v) for k, v in previous.admitted.items()
                if k not in {'preparation_result','native_resource_bindings','choices','resource_options'}}
    admitted.update(revision_id=interaction.base_revision_id,
        request_id='turn-' + digest(dict(session=record['task_id'], turn=interaction.turn_id, action='resume')))
    captured = interaction.snapshot.get('preparation_resources')
    if captured is not None and 'resource_catalog' in admitted and not resume_resource_conflict(previous, captured):
        from .resource_continuation import catalog
        from .input_sets import ScientificInputSet
        from .raw_handoff import _extend_catalog
        current = catalog(tuple(ScientificInputSet(**_serialize(row)) for row in captured['catalog']))
        old = admitted['resource_catalog']
        current_ids = {row['input_set_id'] for row in current}
        admitted['resource_catalog'] = _extend_catalog(old,
            [row for row in old if row['input_set_id'] not in current_ids] + current)
        if 'evidence' in captured:
            old_evidence = admitted['raw_resource_evidence']
            additions = _serialize(captured['evidence'])
            admitted['raw_resource_evidence'] = {**old_evidence, **additions}
            field = records.intake_task(preceding, record)
            if field + '_context' in admitted:
                admitted[field + '_context']['catalog'] = admitted['raw_resource_evidence']
    if record['candidate_origin'] is None and records.intake_task(preceding, record) == 'roles':
        from .dialogue_execution import _role_binding
        original = next(i for i in preceding if i.turn_id == record['task_origin'])
        binding = _role_binding(original, previous, interaction)
        metadata = binding.attribution()
        metadata['roles'] = _serialize(previous.admitted['roles'])
        admitted['registered_input'] = metadata
    elif record['candidate_origin'] is None and records.intake_task(preceding, record) in {'fastq','bam','fragments','annotation','epizoo'}:
        from .raw_handoff import source_binding
        field = records.intake_task(preceding, record)
        original = next(i for i in preceding if i.turn_id == record['task_origin'])
        binding = source_binding(original, previous, interaction, field)
        if binding is not None:
            old = admitted.get(field + '_sources')
            metadata = binding.attribution()
            if admitted.get('registered_input') == old:
                admitted['registered_input'] = metadata
            admitted[field + '_sources'] = metadata
    return admitted


def resume(sessions, interaction, task_id):
    """Explicit fresh-context reanalysis. This operation never reserves science."""
    from .session_state import SessionConflictError
    from .turns import _update, TurnOutcome
    from .interactive_schemas import ClientError
    state = sessions.load(sessions._interaction_session_id)
    preceding = tuple(i for i in state.interactions if i.turn_id != interaction.turn_id)
    previous = records.head(preceding, task_id)
    if previous is None or records.observation(previous) is None:
        raise SessionConflictError('Unknown scientific preparation task.')
    task_turns = {i.turn_id for i in preceding if i.preparation is not None
                  and i.preparation['task_id'] == task_id}
    if any(t.turn_id in task_turns for t in state.turns):
        raise SessionConflictError('A submitted scientific task cannot be retried by preparation resume.')
    if previous.status not in {'clarification','failed','answered'}:
        raise SessionConflictError('Scientific preparation is still processing.')
    old = previous.preparation
    record = dict(version=records.VERSION, task_id=task_id, task_origin=old['task_origin'],
        candidate_id=old['candidate_id'], candidate_origin=old['candidate_origin'],
        previous=previous.turn_id, answers=[],
        resume=dict(base_generation=previous.base_generation, base_revision_id=previous.base_revision_id))
    admitted = resume_admission(interaction, preceding, record)
    original = next(i for i in preceding if i.turn_id == record['task_origin'])
    def invalid(code, message):
        result = records.question(record, interaction, (), status='INVALID', diagnostics=({'code':code},))
        _update(sessions, interaction.turn_id, preparation=record,
            admitted=dict(admitted, preparation_result=result), status='failed')
        return TurnOutcome('execute','failed',text=message,error=ClientError(code,message))
    if resume_resource_conflict(previous, interaction.snapshot.get('preparation_resources')):
        return invalid('PREPARATION_RESOURCE_CHANGED',
                       'A captured scientific resource definition changed; its original identity is preserved.')
    try:
        _model_catalogs(preceding, record, addition=interaction.snapshot.get('preparation_resources', {}).get('models'))
    except ValueError:
        return invalid('PREPARATION_RESOURCE_CHANGED',
                       'A captured qualified model definition changed; its original identity is preserved.')
    candidate = records.candidate_for(preceding, record)
    from agent.orchestration.semantic_compiler import SemanticContextSource
    if (interaction.base_revision_id != original.base_revision_id
            and ((original.admitted or {}).get('context') == 'current'
                 or (original.admitted or {}).get('input_results')
                 or candidate is not None and any(isinstance(source, SemanticContextSource)
                    for step in candidate.steps for source in step.sources))):
        return invalid('PREPARATION_CONTEXT_CHANGED',
                       'The original accepted-result context changed; select the original scientific context before continuing.')
    try:
        validate_direct_source(preceding, record)
        from .local_resources import LocalResourceAdmission
        owner = LocalResourceAdmission(sessions._application._workspace, registry=sessions._application.registry)
        if admitted.get('registered_input'):
            owner.validate_binding(owner._submission_binding(dict(
                registered_input=admitted['registered_input'], execution_inputs=admitted['inputs'])))
        elif (original.submission or {}).get('registered_input'):
            from .dialogue_execution import execution_submission
            selected_submission = execution_submission(original)
            if selected_submission.get('registered_input') is not None:
                owner.validate_submission(selected_submission)
        else:
            from agent.tools._file_identity import file_sha256
            for key, value in admitted.get('inputs', {}).items():
                if key.endswith('_path') and isinstance(value, str):
                    pin = admitted['inputs'].get(key.removesuffix('_path') + '_sha256')
                    captured = records.origin_record(preceding, record)['task']['sources']
                    if pin is None and captured.get('source_path') == value:
                        pin = captured.get('source_sha256')
                    if pin is None and record['candidate_origin'] is not None:
                        acquisition = next(i.preparation for i in preceding if i.turn_id == record['candidate_origin'])
                        pin = next((p['source_sha256'] for p in acquisition.get('source_pins', ())
                                    if p['source_path'] == value), None)
                    if pin is None or file_sha256(value) != pin:
                        return invalid('PREPARATION_SOURCE_CHANGED',
                                       'The original input has no verifiable unchanged byte identity.')
        from .raw_handoff import declaration, source_handles
        for field in ('fastq','bam','fragments','annotation','epizoo'):
            if field in admitted and admitted.get(field + '_sources'):
                members = admitted[field + '_sources']['members']
                handles = source_handles(declaration(_serialize(admitted[field]), field))
                owner.validate_members([members[int(handle.split('.')[1])] for handle in handles])
        from agent.tools._file_identity import file_sha256
        for bindings in records.resource_bindings(preceding, record).values():
            for answer in bindings.values():
                component = answer.get('component')
                if component is not None and file_sha256(component['path']) != component['sha256']:
                    return invalid('PREPARATION_RESOURCE_CHANGED', 'A selected scientific reference changed.')
    except (ValueError, OSError, KeyError):
        return invalid('PREPARATION_SOURCE_CHANGED', 'A captured scientific source or resource is unavailable or changed.')
    # Candidate-less declaration tasks recapture their unresolved observations;
    # answering the refreshed question owns first semantic acquisition.
    intake_needs = None
    if candidate is None and admitted.get('kind') not in {'request_roles', 'request_resources'}:
        if records.intake_task(preceding, record) not in {'fastq','bam','fragments','annotation','epizoo'}:
            return invalid('PREPARATION_INTAKE_UNAVAILABLE', 'This task has no reusable canonical intake.')
        observation_turn = previous
        while not records.observation(observation_turn)['requirements'] and observation_turn.preparation['previous'] is not None:
            observation_turn = next(i for i in preceding if i.turn_id == observation_turn.preparation['previous'])
        intake_needs = records.observation(observation_turn)['requirements']
        if not intake_needs:
            return invalid('PREPARATION_INTAKE_UNAVAILABLE', 'This task has no captured intake requirements.')
    try:
        admitted, _, error, plan = prepare(sessions, interaction, admitted, record=record, intake_needs=intake_needs)
    except (ValueError, OSError, KeyError) as exc:
        # A record may already be published by preparation; retain that exact
        # identity while reporting failed revalidation, never fallback planning.
        return invalid('PREPARATION_REVALIDATION_FAILED', 'Scientific preparation could not revalidate the original context.')
    result = admitted['preparation_result']
    status = 'clarification' if result['status'] == 'NEEDS_INPUT' else 'failed' if result['status'] == 'INVALID' else 'answered'
    _update(sessions, interaction.turn_id, preparation=record, admitted=admitted, status=status)
    if status == 'clarification':
        from .turn_decisions import Clarify
        return TurnOutcome('clarify', status, Clarify('missing_parameter_value',
            tuple(r['argument'] for r in result['requirements']), True))
    return TurnOutcome('answer' if status == 'answered' else 'execute', status,
        text='Preparation rechecked; scientific execution has not started.', error=error)


def resource_requirements(sessions, interaction, admitted):
    """Typed operator configuration need before selecting scientific operations."""
    from .resource_continuation import offers
    from .input_sets import ScientificInputSet
    options = offers(interaction.submission, tuple(ScientificInputSet(**_serialize(r))
        for r in admitted['resource_catalog']))
    if 'reference_requirement' in admitted:
        from .reference_supplementation import choices, input_set
        resource_owner = getattr(sessions, '_reference_resources', None)
        options = ([] if resource_owner is None else [dict(handle='resource.'+s.input_set_id, label=s.label)
            for s in (input_set(r) for r in choices(resource_owner, sessions._interaction_session_id,
                admitted['reference_requirement'], origin_turn_id=interaction.turn_id))])
    requirement = ScientificRequirement('', '@task', admitted['tool'], None,
        'configuration', 'resource', 'input_resource').to_dict()
    return [dict(requirement, offers=[_serialize(o) for o in options],
                 messages=['Select the configured scientific resource for this captured source.'])]


def resource_answer_admission(sessions, interaction, record, *, arguments=()):
    from .resource_continuation import complete
    state = sessions.load(sessions._interaction_session_id)
    preceding = tuple(i for i in state.interactions if i.turn_id != interaction.turn_id)
    origin, previous = records.intake_context(interaction, preceding)
    answer, = record['answers']
    return complete(sessions, interaction, origin, previous, answer['value']['handle'], arguments,
        reference_resources=getattr(sessions, '_reference_resources', None))


def interpreted_resource(sessions, interaction, decision):
    from .turn_decisions import IntentError
    state = sessions.load(sessions._interaction_session_id)
    if decision.pending != '@resources' or state.interactions[-1].turn_id != interaction.turn_id:
        raise IntentError('invalid_prerequisite')
    preceding = tuple(i for i in state.interactions if i.turn_id != interaction.turn_id)
    _, previous = records.intake_context(interaction, preceding)
    selected = next((c for c in records.choices(previous) if c['handle'] == decision.choice), None)
    matches = [(r, o) for r in records.observation(previous)['requirements'] if r['kind'] in ('input_resource','resource')
               for o in r.get('offers', ()) if (selected is not None and selected['requirement_id'] == r['requirement_id']
                   and selected['choice_identity'] == o['handle']) or (selected is None and o['handle'] == decision.choice)]
    if len(matches)!=1:
        raise IntentError('invalid_prerequisite')
    req, option = matches[0]
    record = records.answer_question(preceding, interaction,
        values={req['requirement_id']:_serialize(option)}, interpreted=True)
    if req['kind'] == 'input_resource':
        return resource_answer_admission(sessions, interaction, record, arguments=decision.arguments), record
    if decision.arguments:
        raise IntentError('invalid_prerequisite')
    return records.continued_admission(preceding, record,
        'turn-' + digest(dict(session=state.session_id, turn=interaction.turn_id))), record


def source_admission(sessions, interaction, decision):
    """Invoke the existing scientific declaration owner without pending storage."""
    from .raw_handoff import admit
    return admit(sessions, interaction, decision, getattr(sessions, '_input_sets', ()),
                 reference_resources=getattr(sessions, '_reference_resources', None))


def interpreted_source(sessions, interaction, decision):
    from .turn_decisions import IntentError
    state = sessions.load(sessions._interaction_session_id)
    preceding = tuple(i for i in state.interactions if i.turn_id != interaction.turn_id)
    original, previous = records.intake_context(interaction, preceding)
    field = records.intake_task(preceding, previous.preparation)
    if decision.pending != '@roles' or state.interactions[-1].turn_id != interaction.turn_id:
        raise IntentError('invalid_prerequisite')
    if getattr(decision, field) is None and not decision.arguments and not decision.unknown_fields:
        raise IntentError('invalid_prerequisite')
    admitted, needs = source_admission(sessions, interaction, decision)
    # Interpret resource intent through the scientific owner, then bind the
    # selected component to one captured requirement through the same reducer.
    addition = getattr(decision, field)
    if (previous.preparation['candidate_origin'] is not None and addition is not None
            and (getattr(addition, 'resource', None) is not None
                 or getattr(addition, 'resource_preference', None) is not None)):
        chosen = admitted.get('raw_resource_choice') or admitted.get(field, {}).get('resource')
        matches = [(r, o) for r in records.observation(previous)['requirements'] if r['kind']=='resource'
                   for o in r.get('offers', ()) if o['handle'] == chosen]
        if len(matches) > 1:
            raise IntentError('invalid_prerequisite')
        if not matches:
            record = records.record_answers(previous, interaction, {}, provenance='interpreted_source')
            record['declarations'] = records.declaration_changes(previous, interaction, admitted, field,
                records.origin_record(preceding, record)['task']['sources'])
            return admitted, record, needs, field
        from .turn_decisions import ResourceAnswer
        resource_admitted, record = interpreted_resource(sessions, interaction, ResourceAnswer('@resources', chosen))
        changes = records.declaration_changes(previous, interaction, admitted, field,
            records.origin_record(preceding, record)['task']['sources'])
        changes = [c for c in changes if c['member'] not in {'resource','resource_preference'}]
        if not changes and not decision.arguments and not decision.unknown_fields:
            return resource_admitted, record, (), field
        record['declarations'] = records.declaration_changes(previous, interaction, admitted, field,
            records.origin_record(preceding, record)['task']['sources'])
        return admitted, record, needs, field
    record = records.record_answers(previous, interaction, {}, provenance='interpreted_source')
    required = [r for r in records.observation(previous)['requirements']
                if r['kind']=='declaration' and r['argument']==field]
    if required:
        value = dict(addition=_serialize(admitted[field+'_addition']),
                     unknown_fields=_serialize(admitted.get('unknown_fields', ())),
                     arguments=_serialize(admitted.get('arguments', ())))
        record = records.record_answers(previous, interaction, {required[0]['requirement_id']:value},
                                         provenance='interpreted_source')
    else:
        record['declarations'] = records.declaration_changes(previous, interaction, admitted, field,
            records.origin_record(preceding, record)['task']['sources'])
        from .turn_decisions import SourceFieldArgument
        values = {}
        for argument in decision.arguments:
            if not isinstance(argument, SourceFieldArgument):
                continue
            matches = [(req, offer) for req in records.observation(previous)['requirements']
                if req['tool_name'] == argument.tool and req['argument'] == argument.argument
                for offer in req.get('offers', ()) if offer.get('source_field', {}).get('source') == argument.source
                and offer.get('value') == argument.value]
            if len(matches) > 1:
                raise IntentError('ambiguous_parameter')
            if matches:
                req, offer = matches[0]
                values[req['requirement_id']] = _serialize(offer)
        if values:
            record = dict(records.record_answers(previous, interaction, values, provenance='interpreted_resource'),
                          declarations=record['declarations'])
    return admitted, record, needs, field


def validate_source_admission(interaction, preceding):
    """Use the scientific owner's pure provenance replay, with canonical ancestry."""
    from .raw_handoff import validate, declaration, declaration_dict
    from .session_state import SessionError
    validate(interaction, preceding)
    if not interaction.admitted:
        return
    record = interaction.preparation
    field = records.intake_task((*preceding, interaction), record)
    for answer in record['answers']:
        if answer['kind'] != 'declaration':
            continue
        if answer['provenance'] == 'exact_submission':
            expected = declaration_dict(declaration(_serialize(answer['value']['value']), field))
            if _serialize(interaction.admitted[field+'_addition']) != expected:
                raise SessionError('Exact declaration differs from its captured value.')
        else:
            expected = dict(addition=_serialize(interaction.admitted[field+'_addition']),
                unknown_fields=_serialize(interaction.admitted.get('unknown_fields', ())),
                arguments=_serialize(interaction.admitted.get('arguments', ())))
            if _serialize(answer['value']) != expected:
                raise SessionError('Interpreted declaration differs from its recorded evidence.')


def interpreted_parameters(sessions, interaction, decision):
    from .turn_decisions import IntentError
    from dataclasses import asdict
    state = sessions.load(sessions._interaction_session_id)
    preceding = tuple(i for i in state.interactions if i.turn_id != interaction.turn_id)
    _, previous = records.intake_context(interaction, preceding)
    if decision.pending != '@parameters' or state.interactions[-1].turn_id != interaction.turn_id:
        raise IntentError('invalid_prerequisite')
    records.validate_exact_submission(preceding, interaction, previous)
    values, grounding = {}, {}
    for declaration in decision.arguments:
        from .dialogue_execution import _bind_arguments
        bound = _bind_arguments(sessions._application.registry, interaction.utterance, {}, (declaration,))
        if len(bound) != 1:
            raise IntentError('invalid_parameter_value')
        declared_value, = bound.values()
        matches = [r for r in records.observation(previous)['requirements'] if r['kind']=='scalar'
            and r['tool_name'] == declaration.tool and r['argument'] == declaration.argument
            and r['native_function'] == getattr(declaration,'native_function',None)
            and (r.get('member') is None or isinstance(declared_value, dict) and r['member'] in declared_value)
            and getattr(declaration,'step_id',r['step_id']) == r['step_id']]
        if len({r['step_id'] for r in matches}) != 1:
            raise IntentError('ambiguous_parameter')
        for req in matches:
            key=req['requirement_id']
            if key in values:
                raise IntentError('conflicting_scientific_parameter')
            grounding[key] = asdict(declaration)
            values[key] = records.parameter_value(interaction, req, grounding[key], sessions._application.registry)
    if not values:
        raise IntentError('invalid_parameter_value')
    record = records.record_answers(previous, interaction, values, provenance='interpreted_parameter')
    for answer in record['answers']:
        answer['grounding'] = grounding[answer['requirement_id']]
    admitted = records.continued_admission(preceding, record,
        'turn-' + digest(dict(session=state.session_id,turn=interaction.turn_id)))
    return admitted, record
