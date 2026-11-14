"""Final hierarchical H5AD owner using the existing NativePublication lifecycle.

Copies the original encoded scientific slots. Only new observation columns and
one descriptive uns field are written; neither is independent execution authority.
"""
import os
from copy import deepcopy
from pathlib import Path
import shutil
import tempfile

import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem, write_elem

from agent.schemas.orchestration import _serialize
from agent.schemas.verification_authority import AuthorityError, digest
from agent.tools._cancellation import cancellation_checkpoint
from agent.tools._file_identity import file_closure, read_json
from agent.tools.data.authority_context import current
from agent.tools.data.scatac_matrix_contract import canonical
from .native_publication import NativePublication
from .sequence_prediction import arguments as path_arguments, read, write

TOOL = 'annotate_scATAC_cell_types'
KIND = 'hierarchical_annotation'
FILES = ('annotated.h5ad', 'hierarchy.json', 'annotations.json', 'subdivisions.json')
UNS = 'agent_hierarchical_annotation'
RESULT_FIELDS = dict(status=(str,), contract_version=(str,), n_cells=(int,), n_branches=(int,),
    maximum_levels=(int,), levels_reached=(int,), terminal_populations=(int,),
    population_summary=(list,), inherited_label_cells=(int,), branch_summary=(list,), branches_omitted=(int,), limitations=(str,),
    **{name + suffix: (str,) for name in ('manifest','annotated_h5ad','hierarchy','annotations','subdivisions')
       for suffix in ('_path','_sha256')})


# Marker tables retain every tested gene in every accepted round. These
# scientific sidecars can exceed the generic small-metadata reader's 64 MiB.
EVIDENCE_MAX_BYTES = 512 * 1024**2
MARKER_COLUMNS = ('feature', 'identity', 'exclusion_reason', 'score', 'pvalue',
                  'pvalue_adj', 'positive', 'detection_fractions')
DIFFERENTIAL_COLUMNS = ('feature name', 'log2(fold_change)', 'p-value', 'adjusted p-value')
BACKGROUND_FIELDS = ('tested_features', 'universe', 'reference', 'candidates')


def _statistical_table(rows, columns):
    fields = set(columns)
    if any(set(row) != fields for row in rows):
        raise AuthorityError('Unexpected statistical table fields')
    return dict(columns=list(columns), rows=[[row[k] for k in columns] for row in rows])


def _statistical_records(table, columns=None):
    if columns is None and isinstance(table, dict):
        columns = table.get('columns')
    if (not isinstance(table, dict) or set(table) != {'columns', 'rows'}
            or not isinstance(columns, (tuple, list)) or len(set(columns)) != len(columns)
            or table['columns'] != list(columns) or not isinstance(table['rows'], list)
            or any(not isinstance(row, list) or len(row) != len(columns) for row in table['rows'])):
        raise AuthorityError('Invalid annotation statistical table')
    return [dict(zip(columns, row)) for row in table['rows']]


def annotation_tables(annotations):
    """Self-contained marker tables: name each column once, retain every row.

    This changes storage only. Rankings, repeated feature names, fraction arrays
    and all native statistical values keep their original order and precision.
    No external publication is needed to interpret the downloaded tables.
    """
    rounds, backgrounds = {}, []
    for key, value in annotations.items():
        background = {k: value[k] for k in BACKGROUND_FIELDS if k in value}
        encoded = canonical(background)
        index = next((i for i, row in enumerate(backgrounds) if canonical(row) == encoded), None)
        if index is None:
            index = len(backgrounds)
            backgrounds.append(background)
        clusters = []
        for cluster in value['clusters']:
            markers = cluster['markers']
            table = _statistical_table(markers, MARKER_COLUMNS)
            clusters.append(dict(cluster, markers=table))
        rounds[key] = dict({k: v for k, v in value.items() if k not in BACKGROUND_FIELDS},
            schema='marker-candidates.v2', background=index, clusters=clusters)
    return dict(schema='annotation-evidence.v1', shared_backgrounds=backgrounds, rounds=rounds)


def annotation_records(annotations):
    """Read current column tables and the preserved accepted v1 row evidence."""
    if annotations.get('schema') == 'annotation-evidence.v1':
        backgrounds = annotations['shared_backgrounds']
        restored = {}
        for key, row in annotations['rounds'].items():
            index = row['background']
            if type(index) is not int or not 0 <= index < len(backgrounds):
                raise AuthorityError('Invalid annotation background index')
            restored[key] = dict(deepcopy(backgrounds[index]), **{k: v for k, v in row.items() if k != 'background'})
        annotations = restored
    rounds = {}
    for key, value in annotations.items():
        if not isinstance(value, dict) or value.get('schema') != 'marker-candidates.v2':
            rounds[key] = value  # Historical accepted row representation.
            continue
        clusters = []
        for cluster in value['clusters']:
            table = cluster['markers']
            clusters.append(dict(cluster, markers=_statistical_records(table, MARKER_COLUMNS)))
        rounds[key] = dict(value, schema='marker-candidates.v1', clusters=clusters)
    return rounds


def _native_parameters(native, *, encode):
    """Keep repeated explicit arrays once in this native record's kwargs.

    Effective parameters name the inherited arrays and retain their other
    values directly. Both requested and effective values remain in this record.
    """
    parameters = native.get('parameters', {})
    effective = parameters.get('effective_native_parameters', {})
    kwargs = parameters.get('kwargs', {})
    if encode:
        shared = [key for key, value in effective.items()
                  if isinstance(value, list) and key in kwargs and canonical(value) == canonical(kwargs[key])]
        if not shared:
            return native
        effective = dict(schema='effective-parameters.from-kwargs.v1', from_kwargs=shared,
                         values={k: v for k, v in effective.items() if k not in shared})
    elif effective.get('schema') == 'effective-parameters.from-kwargs.v1':
        effective = dict(effective['values'], **{k: deepcopy(kwargs[k]) for k in effective['from_kwargs']})
    else:
        return native
    return dict(native, parameters=dict(parameters, effective_native_parameters=effective))


def scientific_evidence(value, name, *, encode=False):
    """The annotation owner's three self-contained sidecar representations."""
    if name == 'annotations.json':
        return annotation_tables(value) if encode else annotation_records(value)
    def native(row):
        return dict(row, native_result=_native_parameters(row['native_result'], encode=encode)) if 'native_result' in row else row
    if name == 'hierarchy.json':
        updates = {}
        if 'input_results' in value:
            updates['input_results'] = [_native_parameters(r, encode=encode) for r in value['input_results']]
        if 'operations' in value:
            updates['operations'] = [native(r) for r in value['operations']]
        return dict(value, **updates)
    decisions = {}
    for key, decision in value.items():
        if not isinstance(decision, dict) or 'children' not in decision:
            decisions[key] = decision
            continue
        children = []
        for child in decision['children']:
            if 'differential' in child:
                differential = native(child['differential'])
                rows = differential['rows']
                rows = _statistical_table(rows, tuple(rows[0]) if rows else DIFFERENTIAL_COLUMNS) if encode else (
                    _statistical_records(rows) if isinstance(rows, dict) else rows)
                child = dict(child, differential=dict(differential, rows=rows))
            children.append(child)
        decisions[key] = dict(decision, parent=native(decision['parent']),
            local_steps=[native(r) for r in decision['local_steps']], children=children)
    return decisions


def read_evidence(path, sha):
    """Read one pinned hierarchical sidecar under its explicit storage bound."""
    if Path(path).name not in FILES[1:]:
        raise ValueError('Select a hierarchical annotation evidence sidecar')
    value = read_json(path, sha, limit=EVIDENCE_MAX_BYTES)
    return scientific_evidence(value, Path(path).name)


def annotation_column(name):
    return name in {'cell_type_final','annotation_status','annotation_branch_id','annotation_depth'} or name.startswith(('cell_type_level_', 'annotation_branch_level_'))



def arguments(values):
    return path_arguments(_serialize(values))


def observation_columns(obs, hierarchy):
    """Validate conservation and project labels separately from branch identity."""
    if hierarchy['completion'] != 'complete':
        raise AuthorityError('Partial annotation cannot be published as complete')
    maximum = hierarchy['maximum_levels']
    if type(maximum) is not int or maximum < 1:
        raise AuthorityError('Invalid maximum-level budget')
    columns = {}
    for level in range(1, maximum + 1):
        columns['cell_type_level_' + str(level)] = [None] * len(obs)
        columns['annotation_branch_level_' + str(level)] = [None] * len(obs)
    columns.update(cell_type_final=[None] * len(obs), annotation_status=[None] * len(obs),
        annotation_branch_id=[None] * len(obs), annotation_depth=[0] * len(obs))
    rows = hierarchy['branches']
    by_id = {r['branch_id']: r for r in rows}
    if len(by_id) != len(rows):
        raise AuthorityError('Duplicate global branch identity')
    roots, leaves = [], []
    seen = set()
    for row in rows:
        branch, level, positions = row['branch_id'], row['level'], row['positions']
        from .iterative_annotation import branch_id
        if branch != branch_id(row['parent'], row['cluster']):
            raise AuthorityError('Branch identity differs from its exact parent and native group')
        if (type(level) is not int or not 1 <= level <= maximum or not positions
                or len(set(positions)) != len(positions) or row['n_cells'] != len(positions)
                or any(type(i) is not int or not 0 <= i < len(obs) for i in positions)):
            raise AuthorityError('Invalid branch membership or depth')
        if row['parent'] is None:
            if level != 1:
                raise AuthorityError('Initial annotation must be level 1')
            roots.extend(positions)
        else:
            parent = by_id.get(row['parent'])
            if parent is None or row['parent'] not in seen or level != parent['level'] + 1 or not set(positions) <= set(parent['positions']):
                raise AuthorityError('Branch has invalid accepted parent membership')
        children = [r for r in rows if r['parent'] == branch]
        partition = [i for child in children for i in child['positions']] + row['residual_positions']
        if sorted(partition) != sorted(positions):
            raise AuthorityError('Children and residual lose or duplicate parent cells')
        stop = row['stopping_reason']
        if stop not in {'native_unassigned', 'maximum_levels',
                        'no_candidate_subdivision', 'no_supported_subdivision',
                        'partial_supported_subdivision', 'supported_subdivision'}:
            raise AuthorityError('Incomplete or failed branch cannot become a biological stop')
        if stop == 'maximum_levels' and level != maximum:
            raise AuthorityError('Budget stop differs from configured maximum')
        if bool(children) != (stop in {'partial_supported_subdivision', 'supported_subdivision'}):
            raise AuthorityError('Subdivision stop differs from retained branches')
        if children and (bool(row['residual_positions']) != (stop == 'partial_supported_subdivision')):
            raise AuthorityError('Partial subdivision differs from residual membership')
        annotation = row['annotation']
        label, status = None, 'unassigned' if row['cluster'] == '-1' else 'insufficient_evidence'
        if annotation is not None:
            status = annotation['status']
            if status not in {'assigned', 'ambiguous', 'unresolved', 'insufficient_evidence'}:
                raise AuthorityError('Unknown annotation evidence status')
            if status == 'assigned':
                if len(annotation['preferred']) != 1:
                    raise AuthorityError('Assigned annotation requires a unique preferred candidate')
                label = annotation['preferred'][0]['cell_name']
        for i in positions:
            columns['cell_type_level_' + str(level)][i] = label
            columns['annotation_branch_level_' + str(level)][i] = branch
            if label is not None:
                columns['cell_type_final'][i] = label
            columns['annotation_status'][i] = status
            columns['annotation_branch_id'][i] = branch
            columns['annotation_depth'][i] = level
        leaves.extend(row['residual_positions'])
        seen.add(branch)
    if sorted(roots) != list(range(len(obs))) or sorted(leaves) != list(range(len(obs))):
        raise AuthorityError('Hierarchy does not preserve every original observation exactly once')
    overlap = set(columns) & set(obs.columns)
    if overlap:
        raise AuthorityError('Original observation columns collide with annotation outputs: ' + ', '.join(sorted(overlap)))
    return pd.DataFrame({key: value if key == 'annotation_depth' else pd.Categorical(value)
                         for key, value in columns.items()}, index=obs.index)


def _equal_node(left, right):
    """Bounded encoded-slot comparison; never densify the peak matrix."""
    if set(left.attrs) != set(right.attrs):
        return False
    if any(not np.array_equal(left.attrs[k], right.attrs[k]) for k in left.attrs):
        return False
    if isinstance(left, h5py.Group):
        return (isinstance(right, h5py.Group) and set(left) == set(right)
                and all(_equal_node(left[k], right[k]) for k in left))
    if not isinstance(right, h5py.Dataset) or left.shape != right.shape or left.dtype != right.dtype:
        return False
    if not left.shape:
        return np.array_equal(left[()], right[()], equal_nan=left.dtype.kind in 'fc')
    width = max(1, 1024 * 1024 // max(1, left.dtype.itemsize * int(np.prod(left.shape[1:]))))
    for start in range(0, left.shape[0], width):
        cancellation_checkpoint()
        if not np.array_equal(left[start:start+width], right[start:start+width], equal_nan=left.dtype.kind in 'fc'):
            return False
    return True


def portable_hierarchy(hierarchy):
    """Ordinary AnnData metadata; full scientific evidence stays in pinned sidecars."""
    rows = hierarchy['branches']
    branches = pd.DataFrame(dict(
        parent_branch_id=[row['parent'] or '' for row in rows],
        level=np.asarray([row['level'] for row in rows], dtype=np.int64),
        n_cells=np.asarray([row['n_cells'] for row in rows], dtype=np.int64),
        n_residual_cells=np.asarray([len(row['residual_positions']) for row in rows], dtype=np.int64),
        stopping_reason=[row['stopping_reason'] for row in rows]),
        index=pd.Index([row['branch_id'] for row in rows], name='branch_id'))
    return dict(maximum_levels=hierarchy['maximum_levels'], branches=branches,
                evidence_files=dict(hierarchy='hierarchy.json', annotations='annotations.json',
                                    subdivisions='subdivisions.json'))


def verify_h5ad(path, hierarchy):
    with h5py.File(hierarchy['source']['path'], 'r') as source, h5py.File(path, 'r') as output:
        original, obs = read_elem(source['obs']), read_elem(output['obs'])
        expected = observation_columns(original, hierarchy)
        if (not obs.index.equals(original.index) or list(obs.columns) != [*original.columns, *expected.columns]
                or not obs[list(original.columns)].equals(original) or not obs[list(expected.columns)].equals(expected)):
            raise AuthorityError('Hierarchical observation annotations or original cell axis changed')
        if set(source) != set(output) or any(not _equal_node(source[k], output[k]) for k in source if k not in {'obs', 'uns'}):
            raise AuthorityError('Hierarchical publication changed original scientific slots')
        if set(output['uns']) != set(source['uns']) | {UNS}:
            raise AuthorityError('Hierarchical publication changed original uns membership')
        if any(not _equal_node(source['uns'][k], output['uns'][k]) for k in source['uns']):
            raise AuthorityError('Hierarchical publication changed original scientific metadata')
        portable = read_elem(output['uns'][UNS])
        expected_portable = portable_hierarchy(hierarchy)
        if (not isinstance(portable, dict) or set(portable) != set(expected_portable)
                or not isinstance(portable['branches'], pd.DataFrame)
                or not portable['branches'].equals(expected_portable['branches'])
                or any(portable[key] != expected_portable[key] for key in expected_portable if key != 'branches')):
            raise AuthorityError('Portable hierarchy differs from its verified metadata')


class HierarchicalPublication(NativePublication):
    def verify_publication(self, path, *, expected_sha256):
        result = super().verify_publication(path, expected_sha256=expected_sha256)
        hierarchy = read_evidence(result['hierarchy_path'], result['hierarchy_sha256'])
        annotations = read_evidence(result['annotations_path'], result['annotations_sha256'])
        subdivisions = read_evidence(result['subdivisions_path'], result['subdivisions_sha256'])
        with h5py.File(hierarchy['source']['path'], 'r') as source:
            obs = read_elem(source['obs'])
        if (result['n_cells'] != len(obs) or result['n_branches'] != len(hierarchy['branches'])
                or result['maximum_levels'] != hierarchy['maximum_levels']
                or any(_serialize(result[k]) != v for k,v in summary(hierarchy).items())):
            raise AuthorityError('Publication summary differs from the complete hierarchy')
        grouping = hierarchy['grouping']
        if grouping.get('origin') == 'user_supplied':
            if grouping['source'] != hierarchy['source'] or read(path, expected_sha256)['arguments'].get('group_column') != grouping['groupby']:
                raise AuthorityError('Supplied grouping differs from the original source selection')
            for row in hierarchy['branches']:
                if row['parent'] is None and (row['grouping_binding'] is not None or
                        row['positions'] != [i for i, label in enumerate(obs[grouping['groupby']]) if label == row['cluster']]):
                    raise AuthorityError('Initial branch differs from the supplied observation grouping')
        for round_id, annotation in annotations.items():
            markers = [op for op in hierarchy['operations']
                       if op['branch_id'] == round_id and op['operation_id'] == 'markers']
            if (len(markers) != 1 or annotation['provenance']['binding'] != markers[0]['binding']
                    or annotation['provenance']['native_result'] != markers[0]['native_result']
                    or annotation['reference']['resource'] != hierarchy['cellmarker']):
                raise AuthorityError('Annotation evidence differs from its exact native or CellMarker source')
        for row in hierarchy['branches']:
            if row['annotation'] is not None:
                round_result = annotations[row['annotation_round']]
                if (not any(all(c[k] == v for k, v in row['annotation'].items()) for c in round_result['clusters'])
                        or row['annotation']['cluster'] != row['cluster']):
                    raise AuthorityError('Branch annotation differs from accepted current-round evidence')
            if row['branch_id'] in subdivisions:
                decision = subdivisions[row['branch_id']]
                if decision['outcome'] != row['stopping_reason']:
                    raise AuthorityError('Branch stopping reason differs from differential evidence')
                children = [c for c in hierarchy['branches'] if c['parent'] == row['branch_id']]
                if {c['cluster'] for c in children} != {c['cluster'] for c in decision['children'] if c['retained']}:
                    raise AuthorityError('Published children differ from supported subdivision')
                selection = decision['parent_membership']
                if (selection['cell_ids'] != obs.index[row['positions']].tolist()
                        or selection['groups'] != [row['cluster']]
                        or decision['parent']['binding'] != row['grouping_binding']
                        or row['grouping_binding'] is None and decision['parent'].get('source_grouping') != hierarchy['grouping']):
                    raise AuthorityError('Subdivision does not describe the exact original parent cells')
                positions = dict(zip(selection['positions'], row['positions']))
                if row['residual_positions'] != [positions[i] for i in decision['residual_parent']['positions']]:
                    raise AuthorityError('Published residual differs from tested native populations')
                for child in children:
                    accepted = next(c for c in decision['children'] if c['cluster'] == child['cluster'])
                    if (child['positions'] != [positions[i] for i in accepted['source_positions']]
                            or obs.index[child['positions']].tolist() != accepted['cell_ids']
                            or child['grouping_binding'] != decision['local_steps'][-1]['binding']):
                        raise AuthorityError('Published child differs from its supported native population')
        verify_h5ad(result['annotated_h5ad_path'], hierarchy)
        return result


publication = HierarchicalPublication(tool=TOOL, kind=KIND, contract='hierarchical-annotation.v1',
    profile=digest(dict(contract='hierarchical-annotation.v1', science='ICTA-2/3B/3C', levels=5)),
    verifier=dict(id='agent.hierarchical-annotation-completion', compatibility_version='1'), files=FILES,
    arguments=arguments, outputs=dict(annotated_h5ad='annotated.h5ad', hierarchy='hierarchy.json',
                                     annotations='annotations.json', subdivisions='subdivisions.json'))


def publish(args, execution_identity, hierarchy, annotations, subdivisions, *, resource_files):
    root = Path(args['output_dir'])
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError('Final annotation destination is not empty; no implicit retry')
    hierarchy = _serialize(hierarchy)
    sources = [hierarchy['source']['path'], *(r['path'] for r in resource_files), __file__,
               str(Path(__file__).with_name('iterative_annotation.py')),
               str(Path(__file__).with_name('marker_candidates.py')),
               str(Path(__file__).with_name('local_subdivision.py'))]
    expected = {r['path']: r for r in resource_files}
    for native in [*hierarchy['input_results'], *(op['native_result'] for op in hierarchy['operations'])]:
        manifest = read(native['manifest_path'], native['manifest_sha256'])
        sources.extend([native['manifest_path'], *(r['path'] for r in manifest['files'] + manifest['sources'])])
        expected.update({r['path']: r for r in manifest['files'] + manifest['sources']})
    pins = file_closure(sources)
    if any(r['path'] in expected and r != expected[r['path']] for r in pins):
        raise AuthorityError('Accepted native evidence or resource bytes changed before final publication')
    if next(r for r in pins if r['path'] == hierarchy['source']['path']) != hierarchy['source']:
        raise AuthorityError('Original scientific source changed during annotation')
    with tempfile.TemporaryDirectory(prefix='.hierarchy-', dir=root) as temporary:
        stage = Path(temporary)
        target = stage / 'annotated.h5ad'
        shutil.copyfile(hierarchy['source']['path'], target)
        with h5py.File(target, 'r+') as handle:
            obs = read_elem(handle['obs'])
            columns = observation_columns(obs, hierarchy)
            if UNS in handle.get('uns', {}):
                raise AuthorityError('Original data already contains hierarchy metadata; select its original source')
            write_elem(handle, 'obs', pd.concat([obs, columns], axis=1))
            write_elem(handle.require_group('uns'), UNS, portable_hierarchy(hierarchy))
        verify_h5ad(target, hierarchy)
        for name, value in [('hierarchy.json', hierarchy), ('annotations.json', annotations), ('subdivisions.json', subdivisions)]:
            write(stage / name, scientific_evidence(value, name, encode=True))
        if file_closure(sources) != pins:
            raise AuthorityError('Annotation scientific sources changed during publication')
        for name in FILES:
            with (stage / name).open('rb') as stream:
                os.fsync(stream.fileno())
            os.replace(stage / name, root / name)
    value = dict(contract=publication.contract, profile=publication.profile, arguments=args,
        execution_identity=execution_identity, native_completion='succeeded',
        result=dict(status='success', contract_version=publication.contract, n_cells=sum(r['n_cells'] for r in hierarchy['branches'] if r['parent'] is None),
                    n_branches=len(hierarchy['branches']), maximum_levels=hierarchy['maximum_levels'], limitations=hierarchy['limitations'],
                    **summary(hierarchy)),
        files=file_closure(root / name for name in FILES), sources=pins)
    sha = write(root / 'manifest.json', value)
    if current() is not None:
        current().register_execution(KIND, root, execution_identity)
    return publication.verify_publication(root / 'manifest.json', expected_sha256=sha)


def recover(arguments, execution_identity):
    return publication.recover(arguments, execution_identity)


# Existing generic native delivery consumes the same authoritative publication.
CONTRACT = publication.contract
VERIFIER = publication.verifier
DELIVERY_FILES = {
    'annotated.h5ad': ('Hierarchical annotated cells (H5AD)', 'application/x-hdf5'),
    'hierarchy.json': ('Complete annotation hierarchy', 'application/json'),
    'annotations.json': ('Current-round marker and CellMarker ORA evidence', 'application/json'),
    'subdivisions.json': ('Subdivision and differential peak evidence', 'application/json'),
}
load = publication.load
result = publication.result


def summary(hierarchy):
    """Bounded actual hierarchy populations; final labels may be inherited."""
    from collections import Counter
    branches = hierarchy['branches']
    parents = {row['branch_id']:row for row in branches}
    cells, populations = Counter(), Counter()
    inherited = 0
    terminal = 0
    rows = []
    for row in branches:
        annotation = row['annotation']
        status = annotation['status'] if annotation else ('unassigned' if row['cluster']=='-1' else 'insufficient_evidence')
        label = annotation['preferred'][0]['cell_name'] if annotation and status=='assigned' else None
        n = len(row['residual_positions'])
        if n:
            terminal += 1
            cells[status] += n
            populations[status] += 1
            if label is None:
                parent = row['parent']
                while parent is not None:
                    ancestor = parents[parent]
                    if ancestor['annotation'] and ancestor['annotation']['status']=='assigned':
                        inherited += n
                        break
                    parent = ancestor['parent']
        if len(rows) < 50:
            rows.append(dict(branch_id=row['branch_id'], parent=row['parent'], level=row['level'],
                n_cells=row['n_cells'], residual_cells=n, label=label, status=status, stopping_reason=row['stopping_reason']))
    return dict(levels_reached=max((r['level'] for r in branches),default=0), terminal_populations=terminal,
        population_summary=[dict(status=k,populations=populations[k],cells=cells[k]) for k in
            ('assigned','ambiguous','unresolved','insufficient_evidence','unassigned')],
        inherited_label_cells=inherited, branch_summary=rows, branches_omitted=max(0,len(branches)-len(rows)))


def presentation_data(path, sha, *, level=None):
    """Bounded activity columns from accepted evidence; no tests, rankings or fitting."""
    from collections import Counter
    import anndata as ad
    from agent.tools._file_identity import checked_file
    manifest, actual = publication.load(path, sha)
    pins = {Path(p['path']).name:p for p in manifest['files']}
    for name, pin in pins.items():
        if name not in ('hierarchy.json', 'annotations.json'):
            checked_file(pin['path'],pin['sha256'])
    hierarchy = read_evidence(pins['hierarchy.json']['path'],pins['hierarchy.json']['sha256'])
    annotations = read_evidence(pins['annotations.json']['path'],pins['annotations.json']['sha256'])
    with h5py.File(pins['annotated.h5ad']['path'], 'r') as f:
        obs = read_elem(f['obs'])
    reached = max((r['level'] for r in hierarchy['branches']),default=0)
    if level is not None and (type(level) is not int or not 1 <= level <= reached):
        raise ValueError('Select a hierarchy level actually reached')
    label_column = 'cell_type_final' if level is None else 'cell_type_level_' + str(level)
    branch_column = 'annotation_branch_id' if level is None else 'annotation_branch_level_' + str(level)
    labels = [None if pd.isna(x) else str(x) for x in obs[label_column]]
    groups = [None if pd.isna(x) else str(x) for x in obs[branch_column]]
    group_order = list(dict.fromkeys(g for g in groups if g is not None))[:30]
    selected, genes = [], []
    for row in hierarchy['branches']:
        if row['branch_id'] not in group_order:
            continue
        round_evidence = annotations.get(row['annotation_round'], {})
        cluster = next((r for r in round_evidence.get('clusters',[]) if r['cluster']==row['cluster']),None)
        if cluster is None:
            continue
        # First positive accepted native markers, preserving recorded rank order.
        for marker in [m for m in cluster['markers'] if m['positive'] and m['identity'] is not None][:2]:
            gene = marker['feature']
            if gene not in genes and len(genes) < 12:
                genes.append(gene)
            if gene in genes:
                selected.append(dict(group=row['branch_id'],gene=gene,round=row['annotation_round'],
                    native_cluster=row['cluster']))
    activity = hierarchy['input_results'][-1]
    checked_file(activity['matrix_path'],activity['matrix_sha256'])
    matrix = ad.read_h5ad(activity['matrix_path'], backed='r')
    try:
        # Coordinator/marker handoff proved source correspondence; keep its exact order.
        if matrix.obs_names.tolist() != obs.index.tolist():
            raise AuthorityError('Accepted activity axis differs from the original ordered cell axis')
        feature_names = matrix.var_names.tolist()
        feature_counts = Counter(feature_names)
        ambiguous = [g for g in genes if feature_counts[g] > 1]
        genes = [g for g in genes if feature_counts[g]==1]
        selected = [r for r in selected if r['gene'] in genes]
        # AnnData label indexing requires the entire axis to be unique. The
        # selected names already have exactly one source column; use its position.
        positions = [feature_names.index(g) for g in genes]
        values = matrix[:,positions].X if genes else np.empty((len(obs),0))
        if hasattr(values,'toarray'): values=values.toarray()
        values=np.asarray(values,dtype=float)
    finally:
        matrix.file.close()
    mean, fraction = [], []
    for group in group_order:
        rows = values[np.asarray([g==group for g in groups])]
        mean.append(rows.mean(axis=0).tolist());fraction.append((rows>0).mean(axis=0).tolist())
    counts=Counter(labels)
    status_cells = Counter(obs['annotation_status']) if 'annotation_status' in obs else Counter()
    status_populations = Counter(status for _, status in dict.fromkeys(
        zip(obs['annotation_branch_id'], obs['annotation_status']))) if status_cells else Counter()
    sources=[dict(path=str(path),sha256=actual),*pins.values(),
        dict(path=activity['matrix_path'],sha256=activity['matrix_sha256'])]
    branch_views = []
    for row in hierarchy['branches'][:60]:
        annotation = row.get('annotation')
        status = annotation['status'] if annotation else ('unassigned' if row.get('cluster') == '-1' else 'insufficient_evidence')
        label = annotation['preferred'][0]['cell_name'] if status == 'assigned' else None
        branch_views.append(dict({k: row[k] for k in ('branch_id','parent','level','n_cells','stopping_reason')},
                                 label=label, status=status))
    return dict(cells=obs.index.tolist(),labels=labels,groups=groups,group_order=group_order,
        group_counts=[groups.count(g) for g in group_order],groups_omitted=max(0,len(set(groups)-{None})-len(group_order)),
        genes=genes,mean_activity=mean,fraction_positive=fraction,per_cell_activity=values.tolist(),
        selected_marker_sources=selected,marker_selection='First two positive accepted native markers per displayed branch in recorded rank order; at most 12 distinct genes and 30 branches.',
        ambiguous_marker_names=ambiguous,
        measurement='Accepted Gene Activity X; accessibility-derived, not measured RNA expression',
        composition=[dict(label=k,n_cells=v) for k,v in counts.items()],
        population_summary=[dict(status=status, cells=status_cells[status], populations=status_populations[status])
            for status in ('assigned','ambiguous','unresolved','insufficient_evidence','unassigned')] if status_cells else [],
        inherited_label_cells=int(sum(pd.notna(label) and status != 'assigned'
            for label, status in zip(obs['cell_type_final'], obs['annotation_status']))) if 'annotation_status' in obs else 0,
        source_sha256=hierarchy['source']['sha256'],
        sources=sources,level=level,levels_reached=reached,
        coordinate_source_sha256s=[hierarchy['source']['sha256'],*(r['matrix_sha256'] for r in hierarchy['input_results'][:-1])],
        branches=branch_views,
        branches_omitted=max(0,len(hierarchy['branches'])-60))
