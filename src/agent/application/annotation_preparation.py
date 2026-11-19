"""Materialize selected annotation inputs at the application/compiler boundary.

The planner selects operations and sources. This owner binds existing qualified
resources to those selections; the scientific owner still qualifies and executes.
"""
from agent.orchestration.scientific_inputs import ScientificDiagnostic
from dataclasses import replace
from pathlib import Path

from agent.schemas import PriorOutputRef, StepOutputRef
from agent.schemas.orchestration import _serialize
from .session_state import digest
from agent.orchestration.semantic_compiler import ScientificInputRequired, ScientificInputRequirement


def capture_sources(origin, previous, interaction):
    """Reuse exact file capture, allowing an accepted-result-only objective."""
    from .raw_handoff import attachment_binding
    captured = previous if previous is not None and 'annotation_sources' in previous.admitted else None
    try:
        return attachment_binding(origin, captured, interaction, 'annotation')
    except ValueError:
        if (captured is not None or any((item.submission or {}).get(key) is not None
                for item in (origin, interaction) for key in ('registered_input', 'candidate_input'))):
            raise
        return None


def _capture(interaction, origin, previous, declared, arguments, registry, input_roles=()):
    """Capture explicit file roles and declarations; select no method/resources."""
    from .raw_handoff import source_handles
    from .dialogue_execution import _bind_arguments, _source_field_roles, _input_roles, _argument_fields
    from .local_resources import RegisteredInputGroup, group_role_inputs
    values = (_serialize((origin.submission or {}).get('execution_inputs', {})) if previous is None
              else _serialize(previous.admitted['annotation_arguments']))
    input_roles = [vars(role) for role in _source_field_roles(registry, _input_roles(input_roles), arguments)]
    candidates = capture_sources(origin, previous, interaction)
    if candidates is None and (source_handles(declared) or input_roles):
        raise ValueError('Annotation roles require captured files.')
    if declared.source is not None and (origin.submission or {}).get('candidate_input') is not None:
        selected = origin.submission['registered_input']
        position = int(declared.source.split('.')[1])
        if (candidates is None or position >= len(candidates.members)
                or any(candidates.members[position][key] != selected[key] for key in ('resource_id', 'record_sha256'))):
            from .turn_decisions import IntentError
            raise IntentError('invalid_prerequisite')
    roles = [_serialize(role) for role in input_roles]
    for port, source in (('source', declared.source), ('groups', declared.groups_source)):
        if source is not None:
            if candidates is None or int(source.split('.')[1]) >= len(candidates.members):
                raise ValueError('Annotation source is outside captured files.')
            role = dict(tool='annotate_scATAC_cell_types', port=port, source=source)
            if role not in roles:
                roles.append(role)
    field_binding = None if candidates is None else RegisteredInputGroup(candidates.members, values, roles, attachment_candidates=True)
    values = _bind_arguments(registry, interaction.utterance, values, arguments, replace_existing=True,
        field_choices=_argument_fields(registry, interaction, field_binding, roles, declarations=arguments))
    if candidates is None:
        return values, None, None
    bound = RegisteredInputGroup(candidates.members, values, roles, attachment_candidates=True)
    bound = replace(bound, execution_inputs=group_role_inputs(bound, roles, registry))
    return values, candidates, bound


def admit(sessions, interaction, decision, input_sets):
    from .raw_handoff import AnnotationDeclaration, declaration, declaration_dict, merge, _captured, _extend_catalog, validate_arguments
    from .resource_continuation import catalog
    from .dialogue_execution import _research_intent
    from .turn_decisions import RequestRoles, Execute
    state = sessions.load(sessions._interaction_session_id)
    initial = isinstance(decision, (RequestRoles, Execute))
    from .dialogue_execution import IntentError
    if (state.generation != interaction.base_generation or state.active_revision_id != interaction.base_revision_id
            or state.interactions[-1].turn_id != interaction.turn_id):
        raise IntentError('invalid_prerequisite')
    validate_arguments(decision.arguments, 'annotation')
    from .preparation_state import captured_owner
    if not initial and (decision.pending != '@roles' or captured_owner(interaction, state.interactions[:-1]) != 'annotation'):
        raise IntentError('invalid_prerequisite')
    origin, previous = (interaction, None) if initial else _captured(interaction, state.interactions[:-1])
    rows = catalog(input_sets)
    if previous is not None:
        old_rows = _serialize(previous.admitted['resource_catalog'])
        rows = _extend_catalog(old_rows, rows)
    addition = getattr(decision, 'annotation', None) or AnnotationDeclaration()
    declared = merge(AnnotationDeclaration() if previous is None else declaration(_serialize(previous.admitted['annotation']), 'annotation'),
                     addition, pending=True, unknown_fields=() if initial else decision.unknown_fields)
    input_roles = ([vars(role) for role in decision.roles] if initial
                   else _serialize(previous.admitted.get('roles', ())))
    from .dialogue_execution import _source_field_roles, _input_roles
    input_roles = [vars(role) for role in _source_field_roles(sessions._application.registry, _input_roles(input_roles), decision.arguments)]
    values, candidates, bound = _capture(interaction, origin, previous, declared, decision.arguments,
        sessions._application.registry, input_roles)
    if candidates is not None:
        from .local_resources import LocalResourceAdmission
        owner = LocalResourceAdmission(sessions._application._workspace, registry=sessions._application.registry)
        owner.validate_binding(bound, verify_source=True)
        from .raw_handoff import source_handles
        if any(int(handle.split('.')[1]) >= len(candidates.members) for handle in source_handles(declared)):
            raise ValueError('Annotation role is outside captured files.')
        # Reference attachments are not dataset roles. Their exact registered
        # bytes still must survive continuation before resource interpretation.
        dataset_sources = {role['source'] for role in bound.roles}
        owner.validate_members([candidates.members[int(handle.split('.')[1])]
            for handle in source_handles(declared) if handle not in dataset_sources])
    result = dict(kind='execute', operation='plan', tool='annotate_scATAC_cell_types',
        revision_id=interaction.base_revision_id, resource_catalog=rows,
        annotation=declaration_dict(declared), annotation_addition=declaration_dict(addition),
        annotation_arguments=values, annotation_preparation='selected-method.v3',
        inputs=values if bound is None else _serialize(bound.execution_inputs), roles=input_roles,
        registered_input=None if bound is None else bound.execution_attribution(),
        arguments=[_serialize(vars(argument)) for argument in decision.arguments],
        request_id='turn-' + digest(dict(session=state.session_id, turn=interaction.turn_id)))
    if candidates is not None:
        result['annotation_sources'] = candidates.attribution()
        if declared.source is not None:
            from agent.tools.analysis.annotation_input import observe
            member = candidates.members[int(declared.source.split('.')[1])]
            result['annotation_source_evidence'] = observe(member['source_path'], member['source_sha256'])
    if initial:
        result.update(_research_intent(interaction, decision))
        result['context'] = decision.context
    else:
        if decision.unknown_fields:
            result['unknown_fields'] = list(decision.unknown_fields)
        result['context'] = (origin.admitted or {}).get('context', 'none')
    observed = result.get('annotation_source_evidence')
    questions, options = [], []
    if (observed is None and not result['inputs'].get('input_path') and result.get('context') != 'current'
            and not result.get('input_results')):
        questions = ['Supply and identify the original count dataset to annotate.']
    if observed is not None:
        unknown = [name for name in ('species', 'assembly')
            if getattr(declared, name) is None and observed['declarations'].get(name) is None]
        if unknown:
            questions = ['Please specify the sample ' + ' and '.join(unknown)
                + ' so the native gene annotation and CellMarker evidence can be matched. '
                'You may also identify the original grouping column to reuse.']
        elif declared.generate_umap is None and not umap_available(sessions, interaction, result):
            questions = ['Generate UMAP? No compatible UMAP coordinates were found. Would you like to compute UMAP for visualization? This will run an additional scientific analysis.']
            options = [dict(handle='annotation-umap:' + action, label=label, description=label)
                for action, label in (('generate', 'Generate UMAP'), ('skip', 'Skip UMAP'))]
    if questions:
        result.update(kind='clarify', reason='missing_annotation_declarations')
    from .raw_handoff import declaration_need
    options = [dict(option, value=declaration_dict(AnnotationDeclaration(generate_umap=option['handle'].endswith(':generate')))) for option in options]
    return result, declaration_need('annotation', result['tool'], questions, options)


def umap_available(sessions, interaction, admitted):
    """Read coordinate availability from exact source/accepted native evidence."""
    observed = admitted['annotation_source_evidence']
    if observed.get('compatible_umap'):
        return True
    if interaction.base_revision_id is None:
        return False
    from .result_reading import handoff
    from agent.tools.analysis.local_subdivision import _metadata
    from agent.tools.analysis.embedding_analysis import _cell_order_digest
    context, _ = sessions.execution_context_for_interaction(
        sessions._interaction_session_id, interaction.turn_id, admitted=admitted)
    for item in (() if context is None else context.items):
        binding = item.binding
        from agent.orchestration.prior_outputs import validate_binding
        validate_binding(binding, store=sessions._application.run_store,
            registry=sessions._application.registry, integrity=True)
        result = handoff(sessions._application.run_store, binding.run_id, binding.step_id)
        source=None if result is None else result.get('cell_source') or result['source']
        if (result is not None and result['coordinates'] and source is not None
                and source['sha256'] == observed['source_sha256']):
            obs, = _metadata(dict(matrix_path=result['file']['path']), 'obs')
            if _cell_order_digest(obs.index.tolist()) == observed['cell_order_sha256']:
                return True
    return False


def validate(interaction, preceding):
    """Pure current capture replay; prepared files never authorize execution."""
    from .raw_handoff import AnnotationDeclaration, declaration, declaration_dict, merge, _captured
    from .turn_decisions import _scientific_arguments
    from agent.orchestration.registry import build_default_tool_registry
    value = _serialize(interaction.admitted)
    if value['annotation_preparation'] == 'selected-method.v3' and 'registered_input' not in value:
        raise ValueError('Annotation upload selection is missing.')
    from .dialogue_execution import capture_prerequisite
    from .resource_continuation import catalog
    from .input_sets import ScientificInputSet
    from .raw_handoff import _extend_catalog
    from .preparation_state import intake_task, observation
    canonical = interaction.preparation is not None and intake_task((*preceding, interaction), interaction.preparation) == 'annotation'
    captured = _serialize(interaction.snapshot)
    expected = None if canonical else capture_prerequisite(preceding, interaction.base_revision_id,
        interaction.base_generation, interaction.submission, captured.get('dialogue', {}).get('requested_predecessor'),
        canonical_sources='source_view' in captured.get('dialogue', {}),
            utterance=interaction.utterance if captured.get('dialogue', {}).get('prerequisite_capture_version') == 2 else None)
    if captured.get('prerequisite') != expected:
        raise ValueError('Annotation prerequisite capture changed.')
    origin, previous = _captured(interaction, preceding)
    rows = value['resource_catalog']
    if catalog(tuple(ScientificInputSet(**r) for r in rows)) != rows:
        raise ValueError('Annotation resource catalog changed.')
    if previous is not None:
        old_rows = _serialize(previous.admitted['resource_catalog'])
        if (not canonical and digest(old_rows) != expected['resource_context_sha256']
                or _extend_catalog(old_rows, rows) != rows
                or not canonical and value.get('continuation') != {key: expected[key] for key in ('origin_turn_id', 'binding_turn_id', 'field')}):
            raise ValueError('Annotation continuation changed.')
    declared = merge(AnnotationDeclaration() if previous is None else declaration(_serialize(previous.admitted['annotation']), 'annotation'),
        declaration(value['annotation_addition'], 'annotation'), pending=True, unknown_fields=value.get('unknown_fields', ()))
    arguments, candidates, bound = _capture(interaction, origin, previous, declared,
        _scientific_arguments(value['arguments']), build_default_tool_registry(), value['roles'])
    # Older publications captured even unselected candidates as registered input.
    # Replay their exact attribution; they acquire no new execution authority.
    attribution = (None if bound is None else bound.execution_attribution()
        if value['annotation_preparation'] == 'selected-method.v3' else bound.attribution())
    if (value['annotation'] != declaration_dict(declared) or value['annotation_arguments'] != arguments
            or value['inputs'] != (arguments if bound is None else _serialize(bound.execution_inputs))
            or value.get('registered_input') != attribution
            or value.get('annotation_sources') != (None if candidates is None else candidates.attribution())
            or value['tool'] != 'annotate_scATAC_cell_types' or value.get('operation') != 'plan'
            or value['revision_id'] != interaction.base_revision_id):
        raise ValueError('Selected-method annotation capture changed.')
    if value['kind'] == 'clarify' or interaction.status == 'clarification':
        expected = dict(origin_turn_id=origin.turn_id, binding_turn_id=interaction.turn_id,
            field='annotation', tool=value['tool'], resource_context_sha256=digest(value['resource_catalog']))
        if (_serialize(interaction.prerequisite) != (None if canonical else expected)
                or not ((observation(interaction) or {}).get('requirements') if canonical else value.get('choices'))):
            raise ValueError('Annotation prerequisite lost its exact pending scope.')

def _native_grouping_reference(reference, selected):
    """Resolve retained groups to their actual native producer; no field inference."""
    from agent.tools.analysis.cell_result import NATIVE, grouping_key, planned_count_preservation
    seen = set()
    while isinstance(reference, StepOutputRef):
        # Deferred preparation may not yet have this selected producer. Its
        # ordinary compiled/runtime source checks remain authoritative.
        if reference.step_id not in (selected or {}):
            return reference
        if reference.output_key != 'matrix_path' or reference.step_id in seen:
            return None
        seen.add(reference.step_id)
        tool, arguments = selected[reference.step_id]
        if (tool in NATIVE and arguments.get('kwargs', {}).get('inplace', True) is not False
                and grouping_key(tool.removeprefix('run_'), arguments.get('function_name'), arguments.get('kwargs', {}))):
            return reference
        if not planned_count_preservation(tool, arguments):
            return None
        reference = arguments.get('input_path')
    # Accepted grouping selections already use annotation_groups' shared
    # lineage handoff and keep their captured binding/integrity owner.
    return reference if isinstance(reference, PriorOutputRef) else None


def preparer(sessions, interaction, admitted, *, requirement_bindings=None, bindings=None):
    """Bind installed evidence and exact selected sources; planning owns methods."""
    def prepare(tool, arguments, selected):
        if tool.name != 'annotate_scATAC_cell_types':
            return arguments
        from agent.tools.analysis.sequence_prediction import library
        from agent.tools.data.cellmarker import CellMarkerEvidence, resolve_species
        from .raw_handoff import AnnotationDeclaration, declaration
        lib = library()
        values = dict(arguments)
        def missing(port, argument, messages, options=(), *, kind='resource'):
            return ScientificInputRequired(ScientificInputRequirement(port, argument, kind,
                dependencies=(('source:bytes', 'declaration:groups_source', 'declaration:groups_result',
                               'declaration:grouping_strategy', 'declaration:grouping_provenance') if argument in ('group_column', 'grouping_path') else
                              ('source:bytes', 'declaration:species') if argument == 'species' else
                              ('source:bytes', 'declaration:species', 'declaration:assembly')),
                applicability=dict(source=_serialize(values.get('input_path')),
                    species=values.get('species'), cellmarker_identity=values.get('cellmarker_identity')),
                offers=tuple(options), messages=tuple(messages)), bindings=values)
        annotation = admitted.get('annotation', admitted.get('downstream_annotation'))
        declared = declaration(_serialize(annotation), 'annotation') if annotation else AnnotationDeclaration()
        # Exact requirement answers are already bound to this operation. This
        # scientific owner applies their meanings without changing task-wide
        # raw declarations or invoking a second answer dispatcher.
        for argument, answer in (requirement_bindings or {}).items():
            handle = answer['handle']
            if argument == 'species' and handle.startswith('annotation-species:'):
                declared = replace(declared, species=answer['value'])
                values['species'] = answer['value']
            elif argument == 'group_column' and handle.startswith('annotation-column:'):
                from agent.tools._file_identity import file_sha256
                source = values.get('input_path')
                if not isinstance(source, (str, Path)) or file_sha256(source) != answer['source_sha256']:
                    raise ValueError('Selected annotation column source bytes changed.')
                values['group_column'] = answer['value']
            elif argument == 'gene_anno_path' and handle.startswith('resource.'):
                declared = replace(declared, resource=handle)
        def species_name(value):
            return value.casefold()
        if declared.species is not None:
            if species_name(values.get('species', declared.species)) != species_name(declared.species):
                raise ValueError('Selected annotation species differs from the captured declaration')
            values['species'] = declared.species
        entries = [entry for entry in lib.entries() if entry['kind'] == 'cellmarker'
                   and ('cellmarker_id' not in values or entry['id'] == values['cellmarker_id'])]
        if len(entries) != 1:
            raise missing('cellmarker_id', 'cellmarker_id', ('Select the installed CellMarker 3.0 evidence resource.',))
        evidence = CellMarkerEvidence(lib, entries[0]['id'])
        if values.get('cellmarker_identity', evidence.identity) != evidence.identity:
            raise ValueError('Captured CellMarker evidence identity changed')
        values.update(cellmarker_id=entries[0]['id'], cellmarker_identity=evidence.identity)
        if values.get('species') is not None:
            try:
                # Preserve the declaration in the captured context and plan;
                # the evidence owner resolves the exact query and records both.
                resolve_species(values['species'], evidence.metadata['species'])
            except ValueError as exc:
                options = [dict(handle='annotation-species:' + str(i), label=species,
                    description='Use the exact CellMarker species value ' + species,
                    value=species) for i, species in enumerate(evidence.metadata['species'])]
                raise missing('species', 'species', (str(exc) + '. Select the sample species from the pinned CellMarker reference.',), options) from exc
        if declared.groups_result is not None:
            from .annotation_groups import choices
            from agent.orchestration.prior_outputs import bind_output
            rows = [row for row in choices(sessions, interaction) if row['handle'] == declared.groups_result]
            if len(rows) != 1:
                raise missing('groups', 'grouping_path', ('Select the compatible accepted initial clustering.',))
            row = rows[0]
            binding = bind_output(sessions._application.run_store, sessions._application.registry,
                                  row['run_id'], row['step_id'], 'dataset')
            ref = PriorOutputRef(binding, 'matrix_path')
            if 'grouping_path' in values and values['grouping_path'] != ref:
                raise ValueError('Selected initial clustering conflicts with the planned source')
            values['grouping_path'] = ref
        supplied = values.get('group_column')
        if supplied is not None and values.get('grouping_path') == values.get('input_path'):
            values.pop('grouping_path', None)
        if supplied is not None and 'grouping_path' in values:
            raise ScientificDiagnostic('ANNOTATION_GROUP_SOURCE_CONFLICT', 'An explicit source grouping must be reused directly; do not plan another initial clustering')
        if supplied is None and isinstance(values.get('grouping_path'), (StepOutputRef, PriorOutputRef)):
            grouping = _native_grouping_reference(values['grouping_path'], selected)
            if grouping is None:
                raise missing('groups', 'grouping_path', ('Select compatible accepted native clustering or plan its registered prerequisites.',))
            values['grouping_path'] = grouping
        if supplied is None and 'grouping_path' not in values:
            raise missing('group_column', 'group_column', ('Select an original obs grouping column, compatible initial clustering, or plan the necessary registered native clustering steps.',))
        if declared.grouping_strategy == 'reuse_only' and supplied is None:
            if not isinstance(values.get('grouping_path'), PriorOutputRef):
                raise missing('groups', 'grouping_path', ('Select existing source groups or compatible accepted clustering to honor the request to reuse existing groups.',))
        for name, label in (('grouping_path','initial clustering'),('activity_path','Gene Activity'),('activity_source_path','native Gene Activity source')):
            if name in values and not isinstance(values[name],(PriorOutputRef,StepOutputRef)):
                raise missing({'grouping_path': 'groups', 'activity_path': 'activity', 'activity_source_path': 'activity_source'}[name], name, ('Select compatible accepted ' + label + '; a file locator alone does not establish its scientific source identity.',))
        if 'activity_path' not in values:
            if 'activity_method' not in values:
                raise missing('activity_method', 'activity_method', ('Select compatible accepted Gene Activity or its applicable native construction.',))
            from .input_sets import ScientificInputSet
            from .native_raw_handoff import resource_evidence
            catalog = resource_evidence(tuple(ScientificInputSet(**_serialize(row))
                for row in admitted.get('resource_catalog', ())))
            reference_species=values.get('species', declared.species)
            if reference_species is not None:
                reference_species=next((key for key in ('human','mouse')
                    if species_name(reference_species)==species_name(key)),reference_species)
            receipt = dict(catalog=catalog, custom={}, species=reference_species,
                assembly=declared.assembly or admitted.get('source_coordinates', {}).get('assembly'),
                selected=None if declared.resource is None else declared.resource.removeprefix('resource.'))
            raw = next((admitted.get(field + '_context', {}) for field in ('fastq', 'fragments', 'bam')
                if admitted.get(field + '_context', {}).get('adapter') == 'native-inputs.v1'), {})
            receipt['source_assembly'] = raw.get('assembly')
            if raw.get('source_reference') is not None:
                receipt['source_reference'] = _serialize(raw['source_reference'])
                receipt['resources'] = _serialize(raw.get('resources', {}))
            if receipt['assembly'] is None:
                receipt['assembly'] = raw.get('assembly')
            if declared.resource is None:
                receipt['selected'] = raw.get('selected')
                if raw.get('component_selections'):
                    receipt['component_selections'] = _serialize(raw['component_selections'])
            # Read claims only from this consumer's exact source (or its
            # identity-preserving inspection handoff), never another upload.
            source = values.get('input_path')
            seen = set()
            while isinstance(source, StepOutputRef) and source.step_id in (selected or {}) and source.step_id not in seen:
                seen.add(source.step_id)
                producer, arguments = selected[source.step_id]
                if producer != 'inspect_scATAC' or source.output_key != 'input_path':
                    break
                source = arguments.get('path')
            if isinstance(source, PriorOutputRef):
                from agent.orchestration.prior_outputs import resolve, validate_binding
                validate_binding(source.binding, integrity=True)
                source = resolve(source)
            from .native_raw_handoff import gene_annotation_component
            resource, chosen, reason, observations, assembly = gene_annotation_component(receipt, source,
                source_sha256=values.get('input_sha256'), supplied=values.get('gene_anno_path'),
                supplied_sha256=values.get('gene_anno_sha256'), explicit=declared.resource is not None)
            values.update(gene_anno_path=resource['path'], gene_anno_sha256=resource['sha256'])
            if bindings is not None:
                bindings.append(dict(tool=tool.name, port='gene_anno', resource=chosen, reason=reason,
                    path=resource['path'], sha256=resource['sha256'], assembly=assembly,
                    source_declared_assembly=raw.get('assembly'),
                    source_observations=observations))
            if (values['activity_method'] == 'snapatac2.pp.make_gene_matrix'
                    and not values.get('activity_kwargs', {}).get('use_x') and 'activity_source_path' not in values):
                raise missing('activity_source', 'activity_source_path', ('Select the accepted fragment/insertion source for native SnapATAC2 Gene Activity.',))
        return values
    return prepare


def prepare_operation(arguments, context):
    source = arguments.get('input_path')
    admitted = dict(context.admitted, source_coordinates=context.coordinate_context(source))
    arguments = context.coordinate_arguments(arguments, source)
    return preparer(context.sessions, context.interaction, admitted,
        requirement_bindings=context.answers, bindings=context.receipts)(context.tool, arguments, context.selected)
