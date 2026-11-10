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
                'bam-science.v1': 'bam', 'fastq-science.v1': 'fastq',
                'h5ad-science.v1': 'h5ad'}


def pending_choices(reference, preceding):
    """Read this existing pending question's captured, authoritative options."""
    if reference is None:
        return ()
    if reference['field'] == 'species':
        return tuple(dict(handle='species:' + value, label=value, description=label)
            for value, label in (('human', '人类 · human'), ('mouse', '小鼠 · mouse')))
    if reference['field'] == 'parameters':
        origin = next(i for i in preceding if i.turn_id == reference['binding_turn_id'])
        record = origin.admitted or {}
        functions = {s['arguments']['function_name'] for s in record.get('native_plan', {}).get('steps', ())
                     if s['tool_name'] == reference['tool'] and s['arguments'].get('function_name')}
        functions.update(d['native_function'] for d in record.get('arguments', ())
                         if d['tool'] == reference['tool'] and d.get('native_function'))
        scope = {'native_function': next(iter(functions))} if len(functions) == 1 else {}
        return tuple(dict(**scope, handle='parameter:' + reference['tool'] + ':' + name,
            label=reference['tool'] + '.' + name, action='parameter', tool=reference['tool'], argument=name)
            for name in reference['missing'])
    if reference['field'] == 'roles':
        from .dialogue_execution import _role_source
        origin, source = _role_source(reference, preceding)
        metadata = (source.admitted or {}).get('registered_input', {})
        assigned = {(r['tool'], r['port']) for r in source.admitted.get('roles', ())}
        return tuple(dict(handle='role:' + role['tool'] + ':' + role['port'],
            label=role['tool'] + '.' + role['port'], action='input_role', tool=role['tool'], port=role['port'],
            sources=[dict(resource_id=member['resource_id'], label=member['label'])
                     for member in metadata.get('members', ())])
            for role in reference['required_roles'] if (role['tool'], role['port']) not in assigned)
    if reference['field'] not in ('resources', 'fastq', 'bam', 'fragments', 'annotation', 'epizoo'):
        return ()
    source = next(i for i in preceding if i.turn_id == reference['binding_turn_id'])
    return tuple(_serialize((source.admitted or {}).get('resource_options', ())))


def selected_answer(interaction, preceding):
    """A Web choice is typed intent, not another language interpretation."""
    selection = (interaction.submission or {}).get('clarification_choice')
    reference = interaction.snapshot.get('prerequisite')
    if selection is None:
        settings = (interaction.submission or {}).get('parameter_overrides', ())
        if reference is not None and reference['field'] == 'parameters' and settings:
            from .turn_decisions import ParameterAnswer, ExplicitArgument
            return ParameterAnswer('@parameters', tuple(ExplicitArgument(**_serialize(row)) for row in settings))
        return None
    if reference is None or selection['question_turn_id'] != reference.get('predecessor_turn_id'):
        raise IntentError('invalid_reference')
    if reference['field'] == 'roles':
        from .dialogue_execution import captured_input_group
        from .turn_decisions import RoleAnswer, InputRole
        parts = selection['handle'].split(':', 3)
        if len(parts) != 4 or parts[0] != 'role':
            raise IntentError('invalid_reference')
        _, tool, port, resource_id = parts
        offered = pending_choices(reference, preceding)
        if not any(row['tool'] == tool and row['port'] == port for row in offered):
            raise IntentError('invalid_reference')
        binding = captured_input_group(interaction, preceding)
        matches = [] if binding is None else [i for i, member in enumerate(binding.members)
                                              if member['resource_id'] == resource_id]
        if len(matches) != 1:
            raise IntentError('invalid_reference')
        return RoleAnswer('@roles', (InputRole(tool, port, 'source.' + str(matches[0])),))
    choices = pending_choices(reference, preceding)
    if not any(option['handle'] == selection['handle'] for option in choices):
        raise IntentError('invalid_reference')
    from .turn_decisions import RoleAnswer, ResourceAnswer, SpeciesAnswer, InputRole
    if reference['field'] == 'species':
        return SpeciesAnswer('@species', selection['handle'].partition(':')[2])
    if reference['field'] == 'resources':
        return ResourceAnswer('@resources', selection['handle'])
    from .raw_handoff import empty
    declared = replace(empty(reference['field']), resource=selection['handle'])
    return RoleAnswer('@roles', (), **{reference['field']: declared})


def preference_schema():
    from .turn_decisions import _object
    text = dict(type='string', minLength=1, maxLength=512)
    return {'anyOf': [_object(dict(literal=text, evidence=text)), {'type': 'null'}],
        'description': 'Explicit reference choice and verbatim evidence; otherwise null.'}


def validate_preference(value):
    if value is not None and (type(value) is not dict or set(value) != {'literal','start','end'}
            or type(value['literal']) is not str or not value['literal'].strip() or len(value['literal']) > 512
            or type(value['start']) is not int or type(value['end']) is not int
            or not 0 <= value['start'] < value['end'] <= 4096
            or value['end'] - value['start'] != len(value['literal'])):
        raise ValueError('Invalid grounded resource preference')


def preference_matches(preference, choices, *, label=lambda s:s.label,
                       identity=lambda s:'resource.' + s.input_set_id, aliases=lambda s:()):
    """Textual identification only; scientific consumers own applicability."""
    if preference is None:
        return []
    name = ' '.join(preference['literal'].casefold().split())
    exact = [s for s in choices if name in (' '.join(label(s).casefold().split()), identity(s),
        *(' '.join(value.casefold().split()) for value in aliases(s)))]
    return exact or [s for s in choices if name in ' '.join(label(s).casefold().split())]


def match_preference(preference, choices, *, aliases=lambda s:()):
    """Resolve a user identity for a consumer requiring one whole resource."""
    if preference is None:
        return [], None
    matches = preference_matches(preference, choices, aliases=aliases)
    if len(matches) != 1:
        return [('The requested reference “' + preference['literal'] + '” is '
                 + ('ambiguous: ' + '; '.join(s.label for s in matches) if matches else 'not available in this Session')
                 + '. Please identify the reference you want; no alternative was selected.')], None
    return [], matches[0]


def distinct_choices(candidates, *, default, identity, order):
    """Existing unique/default/alias policy over already applicable resources."""
    defaults = [choice for choice in candidates if default(choice)]
    if len(defaults) == 1:
        return defaults
    groups = {}
    for choice in candidates:
        key = identity(choice)
        if key not in groups or order(choice) < order(groups[key]):
            groups[key] = choice
    return sorted(groups.values(), key=order)


def source_kind(submission):
    submission = submission or {}
    if 'registered_input' in submission:
        if submission['registered_input'].get('composition') == 'scientific-role-selection.v1':
            return 'roles'
        return COMPOSITIONS.get(submission['registered_input'].get('composition'))
    # An accepted-result continuation can have no new file/parameter inputs.
    # Its empty captured mapping still supports explicit resource selection;
    # it grants no dataset or execution authority.
    return 'structured' if 'execution_inputs' in submission else None


def _offered(submission, input_sets):
    family = source_kind(submission)
    if family in ('structured', 'roles'):
        captured = submission['execution_inputs']
        # Exact configuration supplementation, not scientific compatibility or
        # inferred source matching. Existing values can never be replaced.
        return [s for s in input_sets
                if not any(getattr(s, f + '_companion') for f in ('h5ad', *COMPOSITIONS.values()))
                and set(s.execution_inputs) - set(captured)
                and all(k not in captured or captured[k] == v for k, v in s.execution_inputs.items())]
    return [s for s in input_sets if family and getattr(s, family + '_companion')
            and (family != 'h5ad' or 'target_model_manifest_path' in s.execution_inputs)]


def catalog(input_sets):
    if (type(input_sets) is not tuple or len(input_sets) > 64
            or any(not isinstance(s, ScientificInputSet) for s in input_sets)
            or len({s.input_set_id for s in input_sets}) != len(input_sets)):
        raise IntentError('invalid_prerequisite')
    return [s.definition()
            for s in input_sets if not s.h5ad_companion
            or any(k in s.execution_inputs for k in ('target_model_manifest_path','adaptation_spec_path',
                'library_epizoo_transfer','library_genome','qualified_genome'))]


def offers(submission, input_sets):
    return [dict(handle='resource.' + s.input_set_id, label=s.label)
            for s in _offered(submission, input_sets)]


def _source(reference, preceding):
    return (next(i for i in preceding if i.turn_id == reference['origin_turn_id']),
            next(i for i in preceding if i.turn_id == reference['binding_turn_id']))


def capture(prior, origin, current):
    """A new InputSet may be selected; an existing source cannot be replaced."""
    from .dialogue_execution import execution_submission
    original = execution_submission(origin)
    if source_kind(original) == 'structured':
        if ('registered_input' in current or any(k in original['execution_inputs']
                and original['execution_inputs'][k] != v
                for k, v in current.get('execution_inputs', {}).items())):
            return None
        return dict(**_serialize(prior.prerequisite), predecessor_turn_id=prior.turn_id)
    selected = current.get('registered_input')
    old = original.get('registered_input', {})
    if selected is not None and any(selected.get(k) != old.get(k)
            for k in ('resource_id', 'record_sha256', 'collection_id', 'tool_name', 'composition', 'members')):
        return None
    # Resource values are checked against the captured catalog at admission.
    # Source values must still be exactly the captured registration.
    for k in ('input_path', 'source_path', 'source_sha256', 'source_index_path', 'source_index_sha256', 'raw_input_paths'):
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
    if source_kind(interaction.submission) is None:
        raise IntentError('invalid_prerequisite')
    admitted = admit(sessions, interaction, Execute(decision.target,
        arguments=decision.arguments, roles=decision.roles, context=decision.context,
        input_results=decision.input_results), interaction.submission['execution_inputs'])
    rows = catalog(input_sets)
    admitted.update(kind='request_resources', resource_catalog=rows)
    if decision.reference_category is not None:
        from .reference_supplementation import requirement
        if source_kind(interaction.submission) != 'h5ad':
            raise IntentError('invalid_prerequisite')
        admitted['reference_requirement'] = requirement(admitted['inputs'], decision.reference_category)
    pending = dict(origin_turn_id=interaction.turn_id, binding_turn_id=interaction.turn_id,
        field='resources', tool=decision.target, resource_context_sha256=digest(_serialize(rows)))
    return admitted, pending


def _completed(origin, source, interaction, choice_handle, arguments, registry, reference_record=None, reference_source=None):
    from .dialogue_execution import _bind_arguments, execution_submission
    from .local_resources import LocalResourceAdmission
    original = execution_submission(origin)
    if ('reference_requirement' in origin.admitted) != (reference_record is not None):
        raise IntentError('invalid_prerequisite')
    binding = (None if source_kind(original) == 'structured'
               else LocalResourceAdmission._submission_binding(dict(execution_inputs=source.admitted['inputs'],
                    registered_input=source.admitted.get('registered_input', original['registered_input']))))
    values = _serialize(source.admitted['inputs'])
    rows = origin.admitted['resource_catalog']
    choices = tuple(ScientificInputSet(**_serialize(row)) for row in rows)
    offered = {('resource.' + s.input_set_id): s for s in _offered(original, choices)}
    if reference_record is not None:
        from .reference_supplementation import input_set, validate_completion
        validate_completion(reference_record, reference_source, origin, binding)
        supplementary = input_set(reference_record)
        offered = {'resource.' + supplementary.input_set_id: supplementary}
        values['input_sha256'] = reference_source.source_sha256
    if choice_handle not in offered:
        raise IntentError('invalid_prerequisite')
    chosen = offered[choice_handle]
    original_choice = None if binding is None else binding.resource_choice
    if reference_record is None and original_choice is not None and _serialize(original_choice) != chosen.resource_choice():
        raise IntentError('invalid_prerequisite')
    for key, value in chosen.inputs().items():
        if key in values and values[key] != value:
            raise IntentError('conflicting_scientific_parameter')
        values[key] = value
    current = _serialize(interaction.submission or {})
    if binding is None and 'registered_input' in current:
        raise IntentError('invalid_prerequisite')
    received_choice = current.get('registered_input', {}).get('resource_choice')
    if received_choice is not None and received_choice != (_serialize(original_choice) if reference_record is not None else chosen.resource_choice()):
        raise IntentError('invalid_prerequisite')
    # Received inputs may repeat captured declarations or the exact selected set.
    # They cannot inject another path or change a previous declaration.
    if any(key not in values or value != values[key]
           for key, value in current.get('execution_inputs', {}).items()):
        raise IntentError('invalid_prerequisite')
    values = _bind_arguments(registry, interaction.utterance, values, arguments)
    if binding is None:
        return dict(execution_inputs=values)
    from .local_resources import RegisteredInputGroup
    completed = replace(binding, execution_inputs=values, resource_choice=original_choice if reference_record is not None else chosen.resource_choice(),
                        **({} if type(binding) is RegisteredInputGroup else dict(resource_selection_error=None)))
    return dict(execution_inputs=_serialize(completed.execution_inputs), registered_input=completed.attribution())


def answer(sessions, interaction, decision, input_sets, *, reference_resources=None):
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
    reference_record = reference_source = None
    if 'reference_requirement' in origin.admitted:
        from .reference_supplementation import select
        reference_record, reference_source = select(reference_resources, state.session_id, origin, decision.choice)
    completed = _completed(origin, source, interaction, decision.choice, decision.arguments,
                           sessions._application.registry, reference_record, reference_source)
    if reference_resources is not None:
        from .shared_resources import check_choice
        check_choice(reference_resources, state.session_id, decision.choice)
    if 'registered_input' in completed:
        owner = LocalResourceAdmission(sessions._application._workspace, registry=sessions._application.registry)
        owner.validate_binding(LocalResourceAdmission._submission_binding(completed))
    return dict(kind='execute', operation='plan', tool=origin.admitted['tool'],
        revision_id=interaction.base_revision_id,
        request_id='turn-' + digest(dict(session=state.session_id, turn=interaction.turn_id)),
        inputs=completed['execution_inputs'],
        **({'context': origin.admitted['context']} if 'context' in origin.admitted else {}),
        **({'registered_input': completed['registered_input']} if 'registered_input' in completed else {}),
        arguments=[asdict(d) for d in decision.arguments], resource_handle=decision.choice,
        **({'reference_record': reference_record.to_dict(), 'reference_source': reference_source.to_dict()}
           if reference_record is not None else {}),
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
        if 'context' in admitted and (admitted['context'] != 'current'
                or interaction.base_revision_id is None
                or interaction.snapshot['relations'].get('current') != interaction.base_revision_id):
            raise ValueError('Invalid captured resource execution context.')
        expected = capture_prerequisite(preceding, interaction.base_revision_id,
            interaction.base_generation, interaction.submission,
            interaction.snapshot.get('dialogue', {}).get('requested_predecessor'),
            canonical_sources='source_view' in interaction.snapshot.get('dialogue', {}))
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
            base = _serialize(interaction.submission['execution_inputs'])
            if source_kind(interaction.submission) == 'roles':
                from .local_resources import LocalResourceAdmission, group_role_inputs
                base = group_role_inputs(LocalResourceAdmission._submission_binding(interaction.submission),
                    _serialize(admitted.get('roles', [])), registry)
            values = _bind_arguments(registry, interaction.utterance, base,
                _scientific_arguments(_serialize(admitted.get('arguments', []))))
            if 'reference_requirement' in admitted:
                from .reference_supplementation import requirement
                required = _serialize(admitted['reference_requirement'])
                if requirement(values, required['feature_category']) != required:
                    raise ValueError('Changed reference requirement.')
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
        from .local_resources import LocalResourceRecord
        completed = _completed(origin, source, interaction, admitted['resource_handle'],
            _scientific_arguments(_serialize(admitted['arguments'])), build_default_tool_registry(),
            LocalResourceRecord(**_serialize(admitted['reference_record'])) if 'reference_record' in admitted else None,
            LocalResourceRecord(**_serialize(admitted['reference_source'])) if 'reference_source' in admitted else None)
        if (admitted['kind'] != 'execute' or admitted['operation'] != 'plan'
                or admitted['tool'] != origin.admitted['tool']
                or admitted['revision_id'] != interaction.base_revision_id
                or _serialize(admitted['inputs']) != completed['execution_inputs']
                or _serialize(admitted.get('registered_input')) != completed.get('registered_input')
                or admitted.get('context') != origin.admitted.get('context')
                or admitted['continuation'] != {k: reference[k] for k in ('origin_turn_id', 'binding_turn_id', 'field')}):
            raise ValueError('Changed resource continuation.')
    except (ValueError, TypeError, KeyError, StopIteration, AttributeError) as exc:
        raise SessionError('Invalid pending scientific resources.') from exc
