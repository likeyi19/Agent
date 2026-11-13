"""Shared description for native activity construction and explicit input reuse.

This is metadata on existing H5AD publications, not another scientific object
or calculator. Workers capture native axes; the publication owner checks them.
"""
import hashlib
import json

import h5py
from anndata._io.specs import read_elem, write_elem


METHODS = {('snapatac2', 'pp.make_gene_matrix'), ('episcanpy', 'tl.geneactivity')}


def construction_record(result):
    return dict(origin='recorded_native_execution', method=result['function_name'],
        source_sha256=result['source_identity']['sha256'], measurement=result['measurement'],
        effective_parameters=result['parameters']['effective_native_parameters'])


def axis_identity(path):
    with h5py.File(path, 'r') as handle:
        obs, var = read_elem(handle['obs']), read_elem(handle['var'])
    def sha(text):
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    return dict(cells_sha256=sha(json.dumps(obs.index.tolist(), ensure_ascii=False)),
        features_sha256=sha(json.dumps(var.index.tolist(), ensure_ascii=False)),
        feature_index_name=var.index.name,
        feature_attributes_sha256=sha(var.to_json(orient='split', force_ascii=False)),
        feature_attributes=list(var.columns), n_cells=len(obs), n_features=len(var))


def source_metadata(path, args, *, slot='X'):
    """Read declarations and small history fields without loading fragments/X."""
    with h5py.File(path, 'r') as handle:
        uns = handle.get('uns', {})
        declarations = {}
        for key in ('species', 'assembly'):
            saved = read_elem(uns[key]) if key in uns else None
            declarations[key] = saved if saved is not None else args.get(key)
        measurement = args.get('matrix_semantics')
        if slot == 'X' and 'matrix_semantics' in uns:
            saved = read_elem(uns['matrix_semantics'])
            measurement = saved
        if slot == 'raw.X':
            measurement = None  # AnnData.raw has no independent uns measurement.
        history = {}
        if slot == 'X' and 'agent_native_invocation' in uns:
            invocation = read_elem(uns['agent_native_invocation'])
            history = {k: invocation[k] for k in ('function', 'source_sha256', 'parameters_json') if k in invocation}
        if slot == 'X' and 'log1p' in uns:
            history['log1p'] = read_elem(uns['log1p'])
        inherited = [group + '.' + key for group in ('obs', 'uns', 'obsm', 'obsp')
            for key in (read_elem(handle['obs']).columns if group == 'obs' else handle.get(group, {}))]
    return dict(slot=slot, measurement=measurement or 'unspecified_retained_measurement',
        declarations=declarations, known_history=history, history_completeness='unknown', slots=inherited)


def publish_description(path, args, receipt, summary, source):
    """Check the worker's actual new axis, then expose one shared data contract."""
    from agent.schemas.verification_authority import AuthorityError
    if (receipt['package'], receipt['function_name']) not in METHODS:
        raise AuthorityError('Unregistered gene-axis constructor')
    axes = axis_identity(path)
    if axes != receipt.get('gene_activity_axes'):
        raise AuthorityError('Native gene axis or attributes differ from the completed invocation')
    original = receipt['gene_activity_source']
    effective = receipt['effective_parameters']
    feature_type = effective['id_type'] if receipt['package'] == 'snapatac2' else effective['feature_type']
    description = dict(interface='gene-activity.v1', matrix_slot='X', feature_type=feature_type,
        axes=axes, source_axis_relation='identical ordered cells; native new feature axis',
        source_slot=original['slot'], source_measurement=original['measurement'],
        source_history=original['known_history'], source_history_completeness=original['history_completeness'],
        construction_transformations='native domain aggregation only; no added normalization',
        declarations=original['declarations'], inherited_slots=sorted(set(receipt.get('inherited_slots', []))
            - {'uns.matrix_semantics', 'uns.agent_native_invocation', 'uns.agent_gene_activity'}),
        inherited_slot_meaning='retained source metadata, graphs and embeddings; not recomputed in gene space')
    description['construction'] = construction_record(summary)
    if args.get('cell_subset') is not None:
        description['source_axis_relation'] = 'exact selected source cells; native new feature axis'
    if 'raw_current_features_equal' in original:
        description['raw_current_features_equal'] = original['raw_current_features_equal']
        description['raw_feature_interpretation'] = (
            'Native epiScanpy indexes raw.X by current var positions; independent feature correspondence is not verified')
        description['construction']['raw_feature_interpretation'] = description['raw_feature_interpretation']
        description['construction']['raw_current_features_equal'] = original['raw_current_features_equal']
    summary['parameters']['gene_activity'] = description
    summary['source_identity'].update(measurement=original['measurement'],
        declarations={k: v for k, v in original['declarations'].items() if v is not None},
        supplied_declarations={k: args[k] for k in ('species', 'assembly', 'matrix_semantics') if k in args})
    accepted = accepted_source(args, source)
    if accepted is not None:
        summary['source_identity']['accepted_publication'] = {
            k: accepted[k] for k in ('manifest_path', 'manifest_sha256')}
    write_description(path, summary)


def accepted_source(args, source):
    """Only an existing exact authority proof can establish accepted provenance."""
    from agent.tools.data.authority_context import current
    context = current()
    if context is not None:
        for proof in context.proofs.values():
            result = proof.result_metadata
            if result.get('matrix_path') == args['input_path'] and result.get('matrix_sha256') == source['sha256']:
                return result
    return None


def reuse_description(path, args, summary, source):
    """Describe the selected activity input; never infer authority from H5AD uns."""
    from agent.schemas.orchestration import _serialize
    accepted = accepted_source(args, source)
    if accepted is not None:
        summary['source_identity']['accepted_publication'] = {
            k: accepted[k] for k in ('manifest_path', 'manifest_sha256')}
    prior = accepted['parameters'].get('gene_activity') if accepted is not None else None
    slot = summary['parameters']['count_slot']
    # A constructor's X description does not identify an arbitrary selected layer.
    prior = prior if slot == 'X' else None
    if not args.get('gene_activity') and prior is None:
        return
    original = source_metadata(args['input_path'], args, slot=slot)
    description = dict(interface='gene-activity.v1', matrix_slot='X', feature_type='gene',
        axes=axis_identity(path), source_slot=slot, source_measurement=original['measurement'],
        source_history=original['known_history'], source_history_completeness='unknown',
        declarations=original['declarations'],
        source_axis_relation='native AnnData input adaptation; exact positions recorded when selected',
        construction_transformations='existing activity reused without reconstruction; requested native operation recorded separately',
        construction=dict(origin='supplied_declaration', method=None, normalization_history='unknown'),
        inherited_slots=sorted(set(original['slots']) - set(summary['result_slots'])
            - {'uns.matrix_semantics', 'uns.agent_native_invocation', 'uns.agent_gene_activity'}),
        inherited_slot_meaning='source metadata and sliced global graphs/coordinates; not recomputed locally')
    if prior is not None:
        description['feature_type'] = prior['feature_type']
        # Construction facts come from the accepted native result even when its
        # optional description did not yet carry a separate construction record.
        construction = prior.get('construction')
        if construction is None and accepted['function_name'] in {p + '.' + f for p, f in METHODS}:
            construction = construction_record(accepted)
        description['construction'] = _serialize(construction or dict(
            origin='unknown', method=None, normalization_history='unknown'))
    else:
        with h5py.File(args['input_path'], 'r') as handle:
            if 'uns/agent_gene_activity' in handle:
                description['supplied_gene_activity_metadata'] = 'unverified source data, retained in the exact source file'
    summary['parameters']['gene_activity'] = description
    write_description(path, summary)


def write_description(path, summary):
    # The same description travels with a delivered H5AD. Authority still comes
    # from NativePublication, never this uns field or an uploaded label.
    portable_source = dict(summary['source_identity'])
    if 'accepted_publication' in portable_source:
        portable_source['accepted_publication'] = dict(manifest_sha256=portable_source['accepted_publication']['manifest_sha256'])
    with h5py.File(path, 'r+') as handle:
        write_elem(handle.require_group('uns'), 'agent_gene_activity', json.dumps(dict(
            description=summary['parameters']['gene_activity'], method=summary['function_name'], measurement=summary['measurement'],
            effective_parameters=summary['parameters'].get('effective_native_parameters', {}),
            source_identity=portable_source), ensure_ascii=False))


def population_ancestry(path, sha, context):
    """Exact row maps through accepted native publications, nearest source first.

    Cell names are checked after the join, never used to establish correspondence.
    Source bytes without an accepted producer terminate the chain.
    """
    from agent.schemas.verification_authority import AuthorityError
    from .cell_result import describe
    context.validate_anchors()
    proofs = {(p.result_metadata.get('matrix_path'), p.result_metadata.get('matrix_sha256')): p
              for p in context.proofs.values() if p.result_metadata.get('matrix_path')}
    with h5py.File(path, 'r') as handle:
        positions = list(range(len(read_elem(handle['obs']))))
    chain, seen = {}, set()
    location = (str(path), sha)
    while location not in seen:
        seen.add(location)
        chain[location] = positions
        proof = proofs.get(location)
        if proof is None:
            break
        native = proof.result_metadata
        parameters = native['parameters']
        method = native['function_name']
        descriptor = describe('run_' + method.split('.')[0], native)
        if parameters.get('input_orientation'):
            break
        if not (descriptor and descriptor['preserves_counts'] or
                method in {'scanpy.tl.rank_genes_groups', 'scanpy.pp.log1p', 'scanpy.pp.normalize_total',
                           'episcanpy.pp.log1p', 'episcanpy.pp.normalize_total',
                           'snapatac2.pp.make_gene_matrix', 'snapatac2.pp.make_peak_matrix',
                           'episcanpy.tl.geneactivity'}):
            break
        projection = parameters.get('grouping_projection')
        selection = parameters.get('cell_subset')
        if selection is not None:
            positions = [selection['positions'][i] for i in positions]
        if projection is not None:
            positions = [projection['activity_positions'][i] for i in positions]
        location = (proof.description['manifest']['arguments']['input_path'], native['source_identity']['sha256'])
        previous = native['source_identity'].get('accepted_publication')
        if previous is not None:
            parent = proofs.get(location)
            if parent is None or any(parent.result_metadata[k] != previous[k] for k in ('manifest_path', 'manifest_sha256')):
                raise AuthorityError('Population ancestry lacks the exact accepted source publication')
    return chain


def corresponding_positions(source_path, source_sha, target_path, target_sha, context):
    """Map the target's ordered cells to exact source positions via shared ancestry."""
    from agent.schemas.verification_authority import AuthorityError
    source = population_ancestry(source_path, source_sha, context)
    target = population_ancestry(target_path, target_sha, context)
    for identity, target_positions in target.items():
        if identity not in source:
            continue
        source_positions = source[identity]
        if len(set(source_positions)) != len(source_positions) or len(set(target_positions)) != len(target_positions):
            raise AuthorityError('Population ancestry duplicates source observations')
        reverse = {p: i for i, p in enumerate(source_positions)}
        if not set(target_positions) <= set(reverse):
            raise AuthorityError('Selected activity does not cover the exact grouping population')
        positions = [reverse[p] for p in target_positions]
        with h5py.File(source_path, 'r') as handle:
            names = read_elem(handle['obs']).index[positions].tolist()
        with h5py.File(target_path, 'r') as handle:
            if names != read_elem(handle['obs']).index.tolist():
                raise AuthorityError('Accepted population ancestry changed ordered cell identities')
        return positions
    raise AuthorityError('Activity and grouping have no exact accepted common cell source')


def project_grouping(matrix, args, source):
    """Small metadata adaptation before registered Scanpy markers; no statistics."""
    from agent.schemas.verification_authority import AuthorityError
    from agent.tools.data.authority_context import current
    from agent.tools._file_identity import file_closure
    from .cell_result import describe
    context = current()
    if context is None or args['function_name'] != 'tl.rank_genes_groups':
        raise AuthorityError('Grouping projection requires accepted sources for native markers')
    if args.get('cell_subset') is not None or args.get('input_orientation', 'cells_by_features') != 'cells_by_features':
        raise AuthorityError('Grouping projection selects its exact accepted population; select or orient inputs upstream')
    pin = file_closure([args['grouping_path']])[0]
    native = next((p.result_metadata for p in context.proofs.values()
                   if p.result_metadata.get('matrix_path') == pin['path']
                   and p.result_metadata.get('matrix_sha256') == pin['sha256']), None)
    groupby = args.get('kwargs', {}).get('groupby')
    if args.get('group_column') is not None:
        if args['group_column'] != groupby:
            raise AuthorityError('Marker grouping differs from the explicitly selected source column')
        provenance = dict(origin='user_supplied', qualification='Intended grouping role supplied by researcher; biological identity and derivation not validated.')
    else:
        if native is None:
            raise AuthorityError('Grouping projection requires an accepted native publication')
        descriptor = describe('run_' + native['function_name'].split('.')[0], native)
        if not descriptor or descriptor['grouping'] != groupby:
            raise AuthorityError('Marker grouping differs from the selected native clustering')
        provenance = dict(accepted_publication={k: native[k] for k in ('manifest_path', 'manifest_sha256')})
    positions = corresponding_positions(args['input_path'], source['sha256'], pin['path'], pin['sha256'], context)
    matrix = matrix[positions, :].copy()
    with h5py.File(pin['path'], 'r') as handle:
        matrix.obs[groupby] = read_elem(handle['obs'])[groupby]
    return matrix, dict(source=pin, activity_positions=positions, groupby=groupby,
        **provenance)
