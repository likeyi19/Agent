"""Production deterministic iterative annotation coordinator.

Native science belongs to registered ICTA-1 children. Re-entering this function
reconstructs the tree from their accepted outcomes; no branch database is needed.
"""
from collections import deque
from pathlib import Path

from agent.orchestration.annotation_runs import annotation_invocation
from agent.orchestration.verification_authority import accepted_authorities
from agent.schemas.prior_output import PriorOutputRef
from agent.schemas.verification_authority import AuthorityError, digest
from agent.tools._file_identity import file_closure
from agent.tools.data.cellmarker import CellMarkerEvidence
from .cell_result import describe
from .local_subdivision import (_read, _metadata, _peak_source, assess_subdivision,
                                differential_kwargs, LOCAL_METHODS)
from .marker_candidates import annotate_markers, marker_kwargs
from .native_gene_activity import corresponding_positions
from .sequence_prediction import library

CHILDREN = frozenset({'run_snapatac2', 'run_scanpy', 'run_episcanpy'})


def branch_id(parent, group):
    return 'branch-' + digest(dict(parent=parent, native_group=group))


def groups(native, binding):
    if binding is None:
        obs, = _metadata({'matrix_path': native['source']['path']}, 'obs')
        column = native['groupby']
        return column, obs, list(dict.fromkeys(obs[column].tolist()))
    descriptor = describe(binding.tool_name, native)
    if not descriptor or not descriptor['grouping']:
        raise ValueError('An accepted native clustering is required; initial planning owns its prerequisites')
    obs, = _metadata(native, 'obs')
    column = descriptor['grouping']
    if obs[column].isna().any():
        raise ValueError('Native grouping has missing labels without explicit population identity')
    # First appearance on the accepted scientific cell axis, never display sorting.
    return column, obs, list(dict.fromkeys(obs[column].tolist()))


def coordinate(arguments, execution_identity):
    """Durable hook for the single production annotation ToolSpec."""
    from . import hierarchical_annotation as publication
    args = publication.arguments(arguments)
    inv = annotation_invocation()
    readers = inv.evidence_context()
    maximum = args.get('maximum_levels', 5)
    if type(maximum) is not int or maximum < 1:
        raise ValueError('maximum_levels is a positive analysis budget, including initial level 1')
    local_kwargs = args.get('local_kwargs', {})
    if set(local_kwargs) - {m.removeprefix('snapatac2.') for m in LOCAL_METHODS}:
        raise ValueError('Only the approved native local methods can be configured')
    differential = dict(args.get('differential_kwargs', {}))
    if set(differential) & {'cell_group1', 'cell_group2', 'direction'}:
        raise ValueError('Child-versus-parent-rest positive comparison is derived from accepted membership')
    source = file_closure([args['input_path']])[0]
    if args.get('input_sha256', source['sha256']) != source['sha256']:
        raise AuthorityError('Original annotation source changed')
    if 'group_column' in args:
        if 'grouping_path' in args:
            raise ValueError('Select the original source column or an accepted native grouping, not both')
        grouping = None
        initial = dict(origin='user_supplied', source=source, groupby=args['group_column'],
            qualification='Intended grouping role supplied by researcher; biological identity and derivation not validated.')
        groupby, original_obs, initial_groups = groups(initial, grouping)
        positions = list(range(len(original_obs)))
    else:
        grouping = inv.input_binding('grouping_path')
        _, initial = _read(grouping, **readers)
        groupby, original_obs, initial_groups = groups(initial, grouping)
        with accepted_authorities(readers['store'], grouping.run_id) as context:
            positions = corresponding_positions(source['path'], source['sha256'],
                initial['matrix_path'], initial['matrix_sha256'], context)
    if positions != list(range(len(original_obs))) or len(_metadata({'matrix_path': source['path']}, 'obs')[0]) != len(positions):
        raise AuthorityError('Initial grouping must retain every original ordered cell exactly once')
    evidence = CellMarkerEvidence(library(), args['cellmarker_id'])
    if args.get('cellmarker_identity') is not None and args['cellmarker_identity'] != evidence.identity:
        raise AuthorityError('Captured CellMarker resource identity changed')
    operations, annotations, subdivisions = [], {}, {}
    overrides = args.get('operation_overrides', {})
    applied = set()
    def progress(**details):
        if hasattr(inv, 'progress'):
            inv.progress(**details)


    def invoke(branch, operation, tool, native_args, parent_inputs=None):
        for key, override in overrides.items():
            if (override['branch_id'], override['operation_id']) != (branch, operation):
                continue
            if override['tool_name'] != tool or override['function_name'] != native_args['function_name']:
                raise AuthorityError('Edited branch operation changed its native function')
            changes = override['arguments']
            if set(changes) - {'kwargs','count_slot','parameter_selection'}:
                raise ValueError('Only native parameters of the captured operation can be edited')
            protected = {'groupby','groups','reference','method','corr_method','use_raw','layer','n_genes'} if operation=='markers' else {'cell_group1','cell_group2','direction'} if operation.startswith('diff-') else set()
            if set(changes.get('kwargs',{})) & protected:
                raise ValueError('The approved comparison and marker method remain coordinator-owned')
            native_args = dict(native_args)
            for name, value in changes.items():
                native_args[name] = dict(native_args.get(name,{}), **value) if name=='kwargs' else value
            applied.add(key)
        progress(branch_id=branch, operation_id=operation, function_name=tool.removeprefix('run_')+'.'+native_args['function_name'])
        inv.invoke(branch_id=branch, operation_id=operation, tool_name=tool,
                   arguments=native_args, parent_inputs=parent_inputs)
        binding = inv.output(branch, operation, 'dataset')
        _, native = _read(binding, **readers)
        operations.append(dict(branch_id=branch, operation_id=operation,
            binding=binding.to_dict(), native_result=native))
        return binding, native

    if 'activity_path' in args:
        activity = inv.input_binding('activity_path')
        _, activity_result = _read(activity, **readers)
        if 'gene_activity' not in activity_result['parameters']:
            raise ValueError('Selected activity requires its existing accepted Gene Activity declaration')
    else:
        method = args.get('activity_method')
        if method not in {'snapatac2.pp.make_gene_matrix', 'episcanpy.tl.geneactivity'}:
            raise ValueError('Capture a compatible ICTA-2 activity route or supply accepted activity')
        package, function = method.split('.', 1)
        activity_source = inv.input_binding('activity_source_path') if 'activity_source_path' in args else grouping
        activity, activity_result = invoke('initial', 'activity', 'run_' + package,
            dict(**({'input_path': PriorOutputRef(activity_source, 'matrix_path')} if activity_source is not None else {}), function_name=function,
                 kwargs=args.get('activity_kwargs', {}), **({'count_slot': args['activity_count_slot']}
                    if 'activity_count_slot' in args else {})),
            parent_inputs={**{key: key for key in ('gene_anno_path', 'gene_anno_sha256') if key in args},
                           **({'input_path': 'input_path'} if activity_source is None else {})})

    def annotate(round_id, binding, native):
        column, obs, labels = groups(native, binding)
        selected = [g for g in labels if g != '-1']
        if not selected:
            return {}
        kwargs = marker_kwargs(column)
        kwargs['groups'] = selected
        markers, _ = invoke(round_id, 'markers', 'run_scanpy', dict(
            input_path=PriorOutputRef(activity, 'matrix_path'),
            **({'grouping_path': PriorOutputRef(binding, 'matrix_path')} if binding is not None else {'group_column': column}),
            function_name='tl.rank_genes_groups', kwargs=kwargs),
            parent_inputs={'grouping_path': 'input_path'} if binding is None else None)
        result = annotate_markers(markers, **readers, evidence=evidence, species=args.get('species'),
            context=args.get('marker_context'), identity_kind=args.get('identity_kind', 'name'),
            identity_column=args.get('identity_column'), marker_alpha=args.get('marker_alpha', .05),
            ora_alpha=args.get('ora_alpha', .05))
        annotations[round_id] = result
        return {row['cluster']: row for row in result['clusters']}

    initial_annotation = annotate('initial', grouping, initial)
    branches, queue = [], deque()

    def add(parent, label, depth, positions, binding, native, annotation, round_id):
        if annotation is not None:
            annotation = {k: annotation[k] for k in ('cluster', 'groupby', 'status', 'preferred', 'significant', 'reasons')}
        row = dict(branch_id=branch_id(parent, label), parent=parent, cluster=label, level=depth,
            positions=positions, n_cells=len(positions), residual_positions=list(positions),
            annotation=annotation, annotation_round=round_id, stopping_reason=None,
            grouping_binding=None if binding is None else binding.to_dict())
        branches.append(row)
        queue.append((row, binding, native))
        return row

    for label in initial_groups:
        add(None, label, 1, [i for i, g in enumerate(original_obs[groupby]) if g == label],
            grouping, initial, initial_annotation.get(label), 'initial')
    while queue:
        inv.check_cancelled()
        branch, parent_binding, parent_native = queue.popleft()
        progress(level=branch['level'], branch_id=branch['branch_id'], pending_branches=len(queue), outcome=None)
        if branch['cluster'] == '-1':
            branch['stopping_reason'] = 'native_unassigned'
            progress(outcome=branch['stopping_reason'], function_name=None, operation_id=None)
            continue
        if _peak_source(parent_binding, parent_native, readers['store']) is None:
            branch['stopping_reason'] = 'missing_required_peak_space'
            progress(outcome=branch['stopping_reason'], function_name=None, operation_id=None)
            continue
        if branch['level'] >= maximum:
            branch['stopping_reason'] = 'maximum_levels'
            progress(outcome=branch['stopping_reason'], function_name=None, operation_id=None)
            continue
        column, parent_obs, _ = groups(parent_native, parent_binding)
        previous, local = parent_binding, []
        for i, method in enumerate(LOCAL_METHODS):
            function = method.removeprefix('snapatac2.')
            native_args = dict(**({'input_path': PriorOutputRef(previous, 'matrix_path')} if previous is not None else {}), function_name=function,
                               kwargs=local_kwargs.get(function, {}))
            if i == 0:
                native_args['cell_subset'] = dict(groupby=column, groups=[branch['cluster']])
            if i == 3:
                native_args['parameter_selection'] = args.get('parameter_selection', 'asw')
            previous, local_native = invoke(branch['branch_id'], function, 'run_snapatac2', native_args,
                parent_inputs={'input_path': 'input_path'} if previous is None else None)
            local.append(previous)
        local_column, local_obs, candidates = groups(local_native, previous)
        candidates = [g for g in candidates if g != '-1']
        diffs = {}
        if len(candidates) > 1 or (candidates and '-1' in local_obs[local_column].tolist()):
            for label in candidates:
                kwargs = dict(differential, **differential_kwargs(previous, label, **readers))
                diff, _ = invoke(branch['branch_id'], 'diff-' + digest(label), 'run_snapatac2',
                    dict(input_path=PriorOutputRef(previous, 'matrix_path'), function_name='tl.diff_test', kwargs=kwargs))
                diffs[label] = diff
        decision = assess_subdivision(parent_binding, local, diffs, **readers,
            source_grouping=parent_native if parent_binding is None else None,
            adjusted_p_threshold=args.get('adjusted_p_threshold', .01),
            minimum_peak_features=args.get('minimum_peak_features', 5))
        subdivisions[branch['branch_id']] = decision
        outcome = decision['outcome']
        if outcome not in {'no_candidate_subdivision', 'no_supported_subdivision',
                           'partial_supported_subdivision', 'supported_subdivision'}:
            raise AuthorityError('Subdivision evidence incomplete: ' + outcome)
        branch['stopping_reason'] = outcome
        progress(outcome=outcome, function_name=None, operation_id=None)
        retained = [c for c in decision['children'] if c['retained']]
        if not retained:
            continue
        local_annotations = annotate(branch['branch_id'], previous, local_native)
        # The local axis is the exact parent branch in original order. The ICTA-3C
        # reader has already verified this selection against the accepted parent.
        selection = decision['parent_membership']['positions']
        if len(selection) != len(branch['positions']):
            raise AuthorityError('Local selection differs from branch membership')
        root_positions = dict(zip(selection, branch['positions']))
        branch['residual_positions'] = [root_positions[p] for p in decision['residual_parent']['positions']]
        for child in retained:
            add(branch['branch_id'], child['cluster'], branch['level'] + 1,
                [root_positions[p] for p in child['source_positions']], previous, local_native,
                local_annotations.get(child['cluster']), branch['branch_id'])
    if set(overrides) != applied:
        raise ValueError('The selected edited operation was not reached in this hierarchy')
    evidence._check()
    hierarchy = dict(schema='hierarchical-annotation.v1', completion='complete', maximum_levels=maximum,
        source=source, grouping=initial if grouping is None else grouping.to_dict(), activity=activity.to_dict(),
        input_results=([initial] if grouping is not None else []) + [activity_result],
        cellmarker=evidence.identity, branches=branches, operations=operations,
        interpretation=dict(marker_alpha=args.get('marker_alpha', .05), ora_alpha=args.get('ora_alpha', .05),
            adjusted_p_threshold=args.get('adjusted_p_threshold', .01), minimum_peak_features=args.get('minimum_peak_features', 5)),
        limitations='Current-round computational labels and exploratory branches; not independently validated biological identities.')
    result = publication.publish(args, execution_identity, hierarchy, annotations, subdivisions,
                                 resource_files=evidence.entry['files'])
    progress(pending_branches=0, outcome='complete', function_name=None, operation_id=None)
    return result


def annotate_scATAC_cell_types(input_path, cellmarker_id, output_dir, *,
        grouping_path=None, group_column=None, input_sha256=None, activity_path=None, activity_source_path=None,
        gene_anno_path=None, gene_anno_sha256=None, cellmarker_identity=None,
        activity_method=None, activity_kwargs=None, activity_count_slot=None, maximum_levels=5,
        local_kwargs=None, differential_kwargs=None, parameter_selection='asw', species=None,
        marker_alpha=.05, ora_alpha=.05, adjusted_p_threshold=.01, minimum_peak_features=5,
        marker_context=None, identity_kind='name', identity_column=None, operation_overrides=None, previous_annotation_path=None):
    """Production execution requires the owning durable Runtime."""
    raise RuntimeError('Annotation requires its owning durable Runtime and accepted scientific inputs')
