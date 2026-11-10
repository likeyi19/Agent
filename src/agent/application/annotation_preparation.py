"""Materialize selected annotation inputs at the application/compiler boundary.

The planner selects operations and sources. This owner binds existing qualified
resources to those selections; the scientific owner still qualifies and executes.
"""
from dataclasses import replace
from pathlib import Path
import os
import tempfile

from agent.schemas import PriorOutputRef, StepOutputRef
from agent.schemas.orchestration import _serialize
from .session_state import digest


class AnnotationPrerequisite(ValueError):
    def __init__(self, questions):
        self.questions = tuple(questions)
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


def _capture(interaction, origin, previous, declared, arguments, registry):
    """Capture explicit file roles and declarations; select no method/resources."""
    from .raw_handoff import source_handles
    from .dialogue_execution import _bind_arguments
    from .local_resources import RegisteredInputGroup, group_role_inputs
    values = _bind_arguments(registry, interaction.utterance,
        {} if previous is None else _serialize(previous.admitted['annotation_arguments']),
        arguments, replace_existing=True)
    candidates = capture_sources(origin, previous, interaction)
    if candidates is None and source_handles(declared):
        raise ValueError('Annotation roles require captured files.')
    if declared.source is not None and (origin.submission or {}).get('candidate_input') is not None:
        selected = origin.submission['registered_input']
        position = int(declared.source.split('.')[1])
        if (candidates is None or position >= len(candidates.members)
                or any(candidates.members[position][key] != selected[key] for key in ('resource_id', 'record_sha256'))):
            from .turn_decisions import IntentError
            raise IntentError('invalid_prerequisite')
    roles = []
    for port, source in (('source', declared.source), ('groups', declared.groups_source)):
        if source is not None:
            if candidates is None or int(source.split('.')[1]) >= len(candidates.members):
                raise ValueError('Annotation source is outside captured files.')
            roles.append(dict(tool='annotate_scATAC_cell_types', port=port, source=source))
    if candidates is None:
        return values, None, None
    bound = RegisteredInputGroup(candidates.members, values, roles, attachment_candidates=True)
    bound = replace(bound, execution_inputs=group_role_inputs(bound, roles, registry))
    return values, candidates, bound


def admit(sessions, interaction, decision, input_sets):
    from .raw_handoff import AnnotationDeclaration, declaration, declaration_dict, merge, _captured, _extend_catalog, validate_arguments
    from .resource_continuation import catalog
    from .dialogue_execution import _research_intent
    from .turn_decisions import RequestRoles
    state = sessions.load(sessions._interaction_session_id)
    initial = isinstance(decision, RequestRoles)
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
    addition = decision.annotation or AnnotationDeclaration()
    declared = merge(AnnotationDeclaration() if previous is None else declaration(_serialize(previous.admitted['annotation']), 'annotation'),
                     addition, pending=True, unknown_fields=() if initial else decision.unknown_fields)
    values, candidates, bound = _capture(interaction, origin, previous, declared, decision.arguments, sessions._application.registry)
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
        inputs=values if bound is None else _serialize(bound.execution_inputs), roles=[],
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
        _scientific_arguments(value['arguments']), build_default_tool_registry())
    # Older publications captured even unselected candidates as registered input.
    # Replay their exact attribution; they acquire no new execution authority.
    attribution = (None if bound is None else bound.execution_attribution()
        if value['annotation_preparation'] == 'selected-method.v3' else bound.attribution())
    if (value['annotation'] != declaration_dict(declared) or value['annotation_arguments'] != arguments
            or value['inputs'] != (arguments if bound is None else _serialize(bound.execution_inputs))
            or value.get('registered_input') != attribution
            or value.get('annotation_sources') != (None if candidates is None else candidates.attribution())
            or value['tool'] != 'annotate_scATAC_cell_types' or value.get('operation') != 'plan'
            or value['revision_id'] != interaction.base_revision_id or value['roles']):
        raise ValueError('Selected-method annotation capture changed.')
    if value['kind'] == 'clarify' or interaction.status == 'clarification':
        expected = dict(origin_turn_id=origin.turn_id, binding_turn_id=interaction.turn_id,
            field='annotation', tool=value['tool'], resource_context_sha256=digest(value['resource_catalog']))
        if _serialize(interaction.prerequisite) != expected or not value.get('choices'):
            raise ValueError('Annotation prerequisite lost its exact pending scope.')


def publish(workspace, content, name):
    """Private immutable adapter bytes; never an uploaded or scientific result."""
    import hashlib
    sha = hashlib.sha256(content).hexdigest()
    directory = workspace._ensure_directory(workspace.root / 'annotation_inputs')
    root = workspace._ensure_directory(directory / sha)
    path = root / name
    fd, temporary = tempfile.mkstemp(dir=root, prefix='.preparing-')
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise ValueError('Prepared annotation input changed.')
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return str(path), sha


def preparer(sessions, interaction, admitted):
    """One scoped binder for selected uploaded or accepted count sources."""
    store, workspace = sessions._application.run_store, sessions._application._workspace

    def exact(value):
        if isinstance(value, PriorOutputRef):
            from agent.orchestration.prior_outputs import validate_binding
            from agent.orchestration.verification_authority import _accepted_step
            validate_binding(value.binding, store=store)
            row, _, _ = _accepted_step(store, value.binding.run_id, value.binding.step_id)
            return row.result[value.output_key]
        return value

    def planned_counts(producer, values, selected):
        """Project a selected native count construction, never qualify its output.

        Execution still checks actual source metadata, counts and ordered groups.
        No reference choice, counting choice or biological context is inferred here.
        """
        if producer != 'run_snapatac2' or values.get('function_name') not in ('pp.make_peak_matrix', 'pp.add_tile_matrix'):
            return None
        kwargs = values.get('kwargs', {})
        if kwargs.get('use_x') or values.get('count_slot', 'X') != 'X':
            return None
        source = values.get('input_path')
        seen = set()
        while isinstance(source, StepOutputRef):
            if source.step_id in seen:
                raise ValueError('Cyclic planned fragment source.')
            seen.add(source.step_id)
            tool, upstream = selected[source.step_id]
            if tool != 'run_snapatac2':
                return None
            function = upstream.get('function_name')
            if function == 'pp.import_fragments':
                from agent.tools.analysis.snapatac_raw_worker import peak_measurement
                storage = 'paired_fragments' if upstream.get('kwargs', {}).get('is_paired', True) else 'single_end_insertions'
                strategy = values.get('counting_strategy', kwargs.get('counting_strategy', 'paired-insertion'))
                if 'counting_strategy' in kwargs and kwargs['counting_strategy'] != strategy:
                    raise ValueError('Explicit native kwargs and scalar have conflicting counting_strategy values')
                return dict(measurement=peak_measurement(storage, strategy), count_slot='X',
                    **{key: upstream[key] for key in ('species', 'assembly') if upstream.get(key) is not None})
            if function not in ('metrics.tsse', 'metrics.frag_size_distr', 'pp.filter_cells', 'pp.call_cells'):
                return None
            source = upstream.get('input_path')
        return None

    def prepare(tool, arguments, selected):
        if tool.name != 'annotate_scATAC_cell_types' or 'annotation_spec_path' in arguments:
            return arguments  # Explicit pinned specifications keep their meaning.
        from .raw_handoff import AnnotationDeclaration, declaration, merge, resources, resource_evidence, _resolve_resource
        from .input_sets import ScientificInputSet
        from agent.tools.analysis import annotation_input, annotation_contract as c
        from .result_reading import handoff
        annotation = admitted.get('annotation', admitted.get('downstream_annotation'))
        declared = declaration(_serialize(annotation), 'annotation') if annotation is not None else AnnotationDeclaration()
        source = arguments.get('source_path')
        sha = arguments.get('source_sha256')
        visited = set()
        projected = None
        # A planned count-preserving operation may use an already available
        # input for declarations. This does not qualify the future output.
        while isinstance(source, StepOutputRef):
            if source.step_id in visited:
                raise ValueError('Cyclic selected annotation source.')
            visited.add(source.step_id)
            producer, values = selected[source.step_id]
            from agent.tools.analysis.cell_result import planned_count_preservation
            if not planned_count_preservation(producer, values):
                projected = planned_counts(producer, values, selected)
                if projected is not None:
                    break
                raise AnnotationPrerequisite(('Select an available count matrix with declared measurement semantics.',))
            source, sha = values.get('input_path'), values.get('input_sha256')
        observation = None
        if projected is None:
            source, sha = exact(source), exact(sha)
            if not isinstance(source, (str, Path)) or not isinstance(sha, str):
                raise AnnotationPrerequisite(('Select the count H5AD to annotate.',))
            observation = annotation_input.observe(source, sha)
            supplied = dict(observation['declarations'])
        else:
            supplied = projected
            # Raw biological declarations bind import later, after compilation.
            for field in ('fastq', 'bam', 'fragments'):
                for key in ('species', 'assembly'):
                    value = admitted.get(field, {}).get(key)
                    if value is not None:
                        if key in supplied and supplied[key] != value:
                            raise ValueError('Planned count source conflicts with raw ' + key + '.')
                        supplied[key] = value
        metadata_slot = supplied.pop('count_slot', 'X')
        if arguments.get('count_slot', 'X') != metadata_slot:
            supplied.pop('measurement', None)
        declared = merge(declared, AnnotationDeclaration(**supplied))
        missing = ['Specify the annotation ' + key + '.' for key in ('species', 'assembly', 'measurement', 'context')
                   if getattr(declared, key) is None]
        missing += c.input_problems(declared.species, declared.assembly, declared.measurement)
        if missing:
            raise AnnotationPrerequisite(missing)
        groups = arguments.get('group_analysis_path')
        grouping_locator = None
        if declared.groups_result is not None:
            from .annotation_groups import choices
            matches = [row for row in choices(sessions, interaction) if row['handle'] == declared.groups_result]
            if len(matches) != 1 or 'group_column' in arguments or 'groups_path' in arguments:
                raise ValueError('Accepted grouping selection conflicts with the planned grouping source.')
            grouping_locator = matches[0]
            if groups is not None:
                if not isinstance(groups, PriorOutputRef):
                    raise ValueError('Selected accepted grouping requires its qualified prior-output binding.')
                planned_view = handoff(store, groups.binding.run_id, groups.binding.step_id)
                planned_groups = None if planned_view is None else planned_view['groups']
                if planned_groups is None or any(planned_groups[k] != grouping_locator[k]
                        for k in ('run_id', 'step_id', 'accepted_step_sha256')):
                    raise ValueError('Accepted grouping selection conflicts with the planned grouping source.')
                if (exact(groups) != planned_view['file']['path']
                        or exact(arguments['group_analysis_sha256']) != planned_view['file']['sha256']):
                    raise ValueError('Accepted grouping input members changed.')
        elif isinstance(groups, PriorOutputRef):
            grouping_locator = dict(run_id=groups.binding.run_id, step_id=groups.binding.step_id)
        if grouping_locator is not None:
            if observation is None:
                raise AnnotationPrerequisite(('Select clustering computed from the planned count source.',))
            view = handoff(store, grouping_locator['run_id'], grouping_locator['step_id'])
            grouping = None if view is None else view['groups']
            source_view = (handoff(store, arguments['source_path'].binding.run_id, arguments['source_path'].binding.step_id)
                           if isinstance(arguments.get('source_path'), PriorOutputRef) else None)
            source_identity = sha if source_view is None or source_view['source'] is None else source_view['source']['sha256']
            if grouping is None or view['source'] is None or view['source']['sha256'] != source_identity:
                raise AnnotationPrerequisite(('Select clustering with the exact count-source ancestry.',))
            from agent.tools.analysis.categorical import observations, axis, require_same_cells
            ids, labels = observations(grouping['path'], grouping['sha256'], grouping['field'])
            require_same_cells(ids, axis(source, sha))
            if labels.isna().any():
                raise ValueError('Accepted grouping contains missing labels.')
            if declared.groups_result is not None and groups is None:
                # Keep grouping in the existing scientific input owner. Native
                # groups carry their accepted proof DAG; dedicated groups retain
                # the existing exact-source analysis-file contract.
                from agent.tools.analysis.cell_result import NATIVE
                from agent.orchestration.verification_authority import _accepted_step
                grouped, _, _ = _accepted_step(store, grouping['run_id'], grouping['step_id'])
                if grouped.tool_name in NATIVE:
                    from agent.orchestration.prior_outputs import bind_output
                    binding = bind_output(store, sessions._application.registry,
                        grouping['run_id'], grouping['step_id'], 'dataset')
                    arguments['group_analysis_path'] = PriorOutputRef(binding, 'matrix_path')
                    arguments['group_analysis_sha256'] = PriorOutputRef(binding, 'matrix_sha256')
                else:
                    arguments['group_analysis_path'] = grouping['path']
                    arguments['group_analysis_sha256'] = grouping['sha256']
            declared = replace(declared, grouping_provenance='Exact ordered accepted clustering ' + grouping['accepted_step_sha256'])
        elif groups is not None:
            declared = replace(declared, grouping_provenance=declared.grouping_provenance or
                'Exact-source independent clustering selected by the scientific plan')
        elif 'groups_path' not in arguments and 'group_column' not in arguments:
            qualified = [] if observation is None else observation['qualified_groupings']
            if len(qualified) == 1:
                arguments['group_column'] = qualified[0]['column']
                declared = replace(declared, grouping_provenance=qualified[0]['provenance'])
            else:
                columns = '' if observation is None else ', '.join(observation['grouping_columns'])
                raise AnnotationPrerequisite(('Select independently derived clustering'
                    + (' from ' + columns if columns else '') + ' or plan clustering for these counts.',))
        if observation is not None and 'group_column' in arguments and declared.grouping_provenance is None:
            offered = [row for row in observation['qualified_groupings'] if row['column'] == arguments['group_column']]
            if len(offered) == 1:
                declared = replace(declared, grouping_provenance=offered[0]['provenance'])
        if declared.grouping_provenance is None:
            raise AnnotationPrerequisite(('Describe how the selected groups were derived independently of marker inspection.',))
        choices = tuple(ScientificInputSet(**_serialize(row)) for row in admitted.get('resource_catalog', ()))
        if not choices:
            from .shared_resources import current_scientific_library
            library = current_scientific_library()
            choices = () if library is None else library.choices()
        choices = resources(choices, 'annotation')
        evidence = resource_evidence(choices, 'annotation')
        questions, choice = _resolve_resource(declared, choices, evidence)
        if questions:
            raise AnnotationPrerequisite(questions)
        members = admitted.get('annotation_sources', {}).get('members', ())
        spec, problems = annotation_input.resource_specification(
            evidence[choice.input_set_id] if choice else None, declared, members)
        if problems:
            raise AnnotationPrerequisite(problems)
        qualification = None
        if declared.uploaded is not None:
            import csv
            try:
                qualification = annotation_input.qualify_resources(spec)
            except (ValueError, OSError, TypeError, KeyError, csv.Error) as exc:
                from .interactive import _unsafe_fact_value
                detail = str(exc)
                if len(detail) > 512 or _unsafe_fact_value(detail):
                    detail = 'Check the supplied content and declared biological compatibility.'
                raise AnnotationPrerequisite(('The supplied annotation reference is incompatible: ' + detail,)) from exc
        method = spec['method']
        if 'annotation_method' in arguments and arguments['annotation_method'] != method:
            raise ValueError('Explicit annotation method conflicts with selected measurement.')
        arguments['annotation_method'] = method
        arguments['annotation_spec_path'], arguments['annotation_spec_sha256'] = publish(workspace, c.matrix.canonical(spec), 'resources.json')
        reference_owner = getattr(sessions, '_reference_resources', None)
        if qualification is not None and reference_owner is not None:
            from .shared_resources import persist
            uploaded_source = declared.uploaded.gene_source or declared.uploaded.signature_source
            persist(reference_owner, {key: arguments[key] for key in ('annotation_spec_path', 'annotation_spec_sha256')},
                session_id=sessions._interaction_session_id,
                origin_turn_id=admitted.get('continuation', {}).get('origin_turn_id', interaction.turn_id),
                label='Qualified annotation references', qualification=qualification,
                publication_source=(members[int(uploaded_source.split('.')[1])]['source_path']
                                    if uploaded_source is not None else None))
        return arguments
    return prepare
