"""Final hierarchical H5AD owner using the existing NativePublication lifecycle.

Copies the original encoded scientific slots. Only new observation columns and
one descriptive uns field are written; neither is independent execution authority.
"""
import os
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
from agent.tools._file_identity import file_closure
from agent.tools.data.authority_context import current
from .native_publication import NativePublication
from .sequence_prediction import arguments as path_arguments, read, write

# Internal test opt-in only. ICTA-4B will bind this owner to the existing public
# annotation operation while retiring its old science and delivery contracts.
TOOL = 'icta_annotation_internal'
KIND = 'hierarchical_annotation'
FILES = ('annotated.h5ad', 'hierarchy.json', 'annotations.json', 'subdivisions.json')
UNS = 'agent_hierarchical_annotation'


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
        if stop not in {'native_unassigned', 'missing_required_peak_space', 'maximum_levels',
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
        hierarchy = read(result['hierarchy_path'], result['hierarchy_sha256'])
        annotations = read(result['annotations_path'], result['annotations_sha256'])
        subdivisions = read(result['subdivisions_path'], result['subdivisions_sha256'])
        with h5py.File(hierarchy['source']['path'], 'r') as source:
            obs = read_elem(source['obs'])
        if (result['n_cells'] != len(obs) or result['n_branches'] != len(hierarchy['branches'])
                or result['maximum_levels'] != hierarchy['maximum_levels']):
            raise AuthorityError('Publication summary differs from the complete hierarchy')
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
                        or decision['parent']['binding'] != row['grouping_binding']):
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
            write(stage / name, value)
        if file_closure(sources) != pins:
            raise AuthorityError('Annotation scientific sources changed during publication')
        for name in FILES:
            with (stage / name).open('rb') as stream:
                os.fsync(stream.fileno())
            os.replace(stage / name, root / name)
    value = dict(contract=publication.contract, profile=publication.profile, arguments=args,
        execution_identity=execution_identity, native_completion='succeeded',
        result=dict(status='success', contract_version=publication.contract, n_cells=sum(r['n_cells'] for r in hierarchy['branches'] if r['parent'] is None),
                    n_branches=len(hierarchy['branches']), maximum_levels=hierarchy['maximum_levels'], limitations=hierarchy['limitations']),
        files=file_closure(root / name for name in FILES), sources=pins)
    sha = write(root / 'manifest.json', value)
    if current() is not None:
        current().register_execution(KIND, root, execution_identity)
    return publication.verify_publication(root / 'manifest.json', expected_sha256=sha)


def recover(arguments, execution_identity):
    return publication.recover(arguments, execution_identity)
