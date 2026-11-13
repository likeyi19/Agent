"""Materialize selected annotation inputs at the application/compiler boundary.

The planner selects operations and sources. This owner binds existing qualified
resources to those selections; the scientific owner still qualifies and executes.
"""
from dataclasses import replace
from pathlib import Path

from agent.schemas import PriorOutputRef, StepOutputRef
from agent.schemas.orchestration import _serialize
from .session_state import digest


class AnnotationPrerequisite(ValueError):
    def __init__(self, questions, options=()):
        self.questions = tuple(questions)
        self.options = tuple(options)
        super().__init__(' '.join(questions))


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
    reference = interaction.snapshot.get('prerequisite')
    if not initial and (decision.pending != '@roles' or reference is None or reference['field'] != 'annotation'):
        raise IntentError('invalid_prerequisite')
    origin, previous = (interaction, None) if initial else _captured(interaction, state.interactions[:-1])
    rows = catalog(input_sets)
    if previous is not None:
        old_rows = _serialize(previous.admitted['resource_catalog'])
        if digest(old_rows) != reference['resource_context_sha256']:
            raise IntentError('invalid_prerequisite')
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
        reference = interaction.snapshot['prerequisite']
        result['continuation'] = {key: reference[key] for key in ('origin_turn_id', 'binding_turn_id', 'field')}
        if decision.unknown_fields:
            result['unknown_fields'] = list(decision.unknown_fields)
        result['context'] = (origin.admitted or {}).get('context', 'none')
    return result, None


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
    captured = _serialize(interaction.snapshot)
    expected = capture_prerequisite(preceding, interaction.base_revision_id,
        interaction.base_generation, interaction.submission, captured.get('dialogue', {}).get('requested_predecessor'),
        canonical_sources='source_view' in captured.get('dialogue', {}))
    if captured.get('prerequisite') != expected:
        raise ValueError('Annotation prerequisite capture changed.')
    origin, previous = _captured(interaction, preceding)
    rows = value['resource_catalog']
    if catalog(tuple(ScientificInputSet(**r) for r in rows)) != rows:
        raise ValueError('Annotation resource catalog changed.')
    if previous is not None:
        old_rows = _serialize(previous.admitted['resource_catalog'])
        if (digest(old_rows) != expected['resource_context_sha256']
                or _extend_catalog(old_rows, rows) != rows
                or value.get('continuation') != {key: expected[key] for key in ('origin_turn_id', 'binding_turn_id', 'field')}):
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
        if _serialize(interaction.prerequisite) != expected or not value.get('choices'):
            raise ValueError('Annotation prerequisite lost its exact pending scope.')



def preparer(sessions, interaction, admitted):
    """Bind installed evidence and exact selected sources; planning owns methods."""
    def prepare(tool, arguments, selected):
        if tool.name != 'annotate_scATAC_cell_types':
            return arguments
        from agent.tools.analysis.sequence_prediction import library
        from agent.tools.data.cellmarker import CellMarkerEvidence
        from .raw_handoff import AnnotationDeclaration, declaration
        lib = library()
        values = dict(arguments)
        annotation = admitted.get('annotation', admitted.get('downstream_annotation'))
        declared = declaration(_serialize(annotation), 'annotation') if annotation else AnnotationDeclaration()
        if declared.species is not None:
            if values.get('species', declared.species) != declared.species:
                raise ValueError('Selected annotation species differs from the captured declaration')
            values['species'] = declared.species
        entries = [entry for entry in lib.entries() if entry['kind'] == 'cellmarker'
                   and ('cellmarker_id' not in values or entry['id'] == values['cellmarker_id'])]
        if len(entries) != 1:
            raise AnnotationPrerequisite(('Select the installed CellMarker 3.0 evidence resource.',))
        evidence = CellMarkerEvidence(lib, entries[0]['id'])
        if values.get('cellmarker_identity', evidence.identity) != evidence.identity:
            raise ValueError('Captured CellMarker evidence identity changed')
        values.update(cellmarker_id=entries[0]['id'], cellmarker_identity=evidence.identity)
        if declared.groups_result is not None:
            from .annotation_groups import choices
            from agent.orchestration.prior_outputs import bind_output
            rows = [row for row in choices(sessions, interaction) if row['handle'] == declared.groups_result]
            if len(rows) != 1:
                raise AnnotationPrerequisite(('Select the compatible accepted initial clustering.',))
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
            raise ValueError('An explicit source grouping must be reused directly; do not plan another initial clustering')
        if supplied is None and not isinstance(values.get('grouping_path'), PriorOutputRef):
            source = values.get('input_path')
            observed = admitted.get('annotation_source_evidence')
            field_offers = () if interaction is None else interaction.snapshot.get('dialogue', {}).get('source_fields', ())
            if isinstance(source, (str, Path)) and (observed is not None or any(len(row['fields']) > 1 for row in field_offers)):
                from agent.tools.analysis.annotation_input import observe
                from agent.tools._file_identity import file_sha256
                sha = file_sha256(source)
                if observed is None or observed['source_sha256'] != sha:
                    observed = observe(source, sha)
                columns = observed['grouping_columns']
                if len(columns) > 1:
                    options = [dict(handle='annotation-column:' + str(i), label=column,
                        description='Use original obs.' + column + ' as the supplied initial grouping',
                        argument='group_column', value=column, source_sha256=sha)
                        for i, column in enumerate(columns)]
                    raise AnnotationPrerequisite(('Select the original observation grouping column to reuse at level 1.',), options)
        if supplied is None and 'grouping_path' not in values:
            raise AnnotationPrerequisite(('Select an original obs grouping column, compatible initial clustering, or plan the necessary registered native clustering steps.',))
        if declared.grouping_strategy == 'reuse_only' and supplied is None:
            reference = values.get('grouping_path')
            # Same-plan inspection/identity projection can preserve an existing
            # grouping. A newly selected clustering operation cannot satisfy reuse.
            def reuses_groups(ref, seen=()):
                if isinstance(ref, PriorOutputRef):
                    return True
                if not isinstance(ref, StepOutputRef) or ref.step_id in seen or ref.step_id not in (selected or {}):
                    return False
                _, step_args = selected[ref.step_id]
                if step_args.get('function_name') in {'tl.leiden', 'tl.louvain', 'tl.kmeans', 'tl.hc'}:
                    return False
                return reuses_groups(step_args.get('input_path'), (*seen, ref.step_id))
            if not reuses_groups(reference):
                raise AnnotationPrerequisite(('Select existing source groups or compatible accepted clustering to honor the request to reuse existing groups.',))
        for name, label in (('grouping_path','initial clustering'),('activity_path','Gene Activity'),('activity_source_path','native Gene Activity source')):
            if name in values and not isinstance(values[name],(PriorOutputRef,StepOutputRef)):
                raise AnnotationPrerequisite(('Select compatible accepted ' + label + '; a file locator alone does not establish its scientific source identity.',))
        if 'activity_path' not in values:
            if 'activity_method' not in values:
                raise AnnotationPrerequisite(('Select compatible accepted Gene Activity or its applicable native construction.',))
            if 'gene_anno_path' not in values:
                refs = [e for e in lib.entries() if e['kind'] == 'gene_annotation'
                        and (declared.species is None or e['resource'].get('species') == declared.species)
                        and (declared.assembly is None or e['resource'].get('assembly') == declared.assembly)]
                if len(refs) != 1:
                    raise AnnotationPrerequisite(('Select the applicable gene annotation reference for native Gene Activity.',))
                resource = lib.check(refs[0])['file']
                values.update(gene_anno_path=resource['path'], gene_anno_sha256=resource['sha256'])
            if (values['activity_method'] == 'snapatac2.pp.make_gene_matrix'
                    and not values.get('activity_kwargs', {}).get('use_x') and 'activity_source_path' not in values):
                raise AnnotationPrerequisite(('Select the accepted fragment/insertion source for native SnapATAC2 Gene Activity.',))
        return values
    return prepare
