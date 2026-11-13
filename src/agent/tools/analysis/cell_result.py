"""Transient cell-artifact capabilities, derived from accepted owner metadata.

Callers authenticate publication/acceptance first. A file's field names alone
never grant these capabilities. No catalog, authority or scientific work lives here.
"""
from collections.abc import Mapping

DEDICATED = ('annotate_scATAC_cell_types',)
NATIVE = ('run_snapatac2', 'run_scanpy', 'run_episcanpy')
TOOLS = DEDICATED + NATIVE
_PRESERVE = frozenset(('pp.select_features', 'tl.spectral', 'pp.knn',
                      'tl.leiden', 'tl.louvain', 'tl.kmeans', 'tl.hc', 'tl.dbscan',
                      'tl.umap', 'pp.pca', 'tl.pca', 'pp.neighbors'))


def grouping_key(package, function, kwargs):
    """Installed native grouping slots, including Scanpy's restricted suffix."""
    supported = {'snapatac2': ('tl.leiden', 'tl.dbscan'),
                 'scanpy': ('tl.leiden',),
                 'episcanpy': ('tl.leiden', 'tl.louvain', 'tl.kmeans', 'tl.hc')}
    if function not in supported.get(package, ()):
        return None
    default = function.split('.')[1]
    key = kwargs.get('key_added', default)
    if package != 'snapatac2' and kwargs.get('restrict_to') is not None and key == default:
        key += '_R'
    return key


def describe(tool, result):
    """Describe actual produced slots; retained fields are not new computations."""
    if tool not in TOOLS:
        return None
    if tool == 'annotate_scATAC_cell_types':
        return dict(path_field='annotated_h5ad_path', sha_field='annotated_h5ad_sha256',
                    coordinates=None, grouping=None, preserves_counts=True, input_argument='input_path')
    package = tool.removeprefix('run_')
    if result.get('contract_version') != 'native-' + package + '-matrix.v1' or not result.get('matrix_path'):
        return None
    full = result.get('function_name', '')
    if not full.startswith(package + '.'):
        return None
    function = full[len(package)+1:]
    parameters = result.get('parameters', {})
    kwargs = parameters.get('kwargs', {})
    slots = result.get('result_slots', ())
    if not isinstance(kwargs, Mapping):
        return None
    # Independent native returns have their own slots; inherited H5AD fields
    # cannot be offered as newly computed UMAP/clustering.
    inplace = kwargs.get('inplace', True) is not False
    coordinate = ('X_' + kwargs.get('key_added', 'umap')) if package == 'snapatac2' else 'X_umap'
    grouping = grouping_key(package, function, kwargs)
    return dict(path_field='matrix_path', sha_field='matrix_sha256',
        coordinates=coordinate if inplace and function == 'tl.umap' and 'obsm.' + coordinate in slots else None,
        grouping=grouping if inplace and grouping and 'obs.' + grouping in slots else None,
        preserves_counts=function in _PRESERVE and parameters.get('count_slot', 'X') == 'X',
        input_argument='input_path')


def file(tool, result):
    descriptor = describe(tool, result)
    if descriptor is None:
        return None
    return dict(path=result[descriptor['path_field']], sha256=result[descriptor['sha_field']])


def planned_coordinates(tool, arguments):
    """Selection offer only; actual written slots are checked after acceptance."""
    return (tool in NATIVE
        and arguments.get('function_name') == 'tl.umap'
        and arguments.get('kwargs', {}).get('inplace', True) is not False)


def planned_count_preservation(tool, arguments):
    return (tool in NATIVE and arguments.get('function_name') in _PRESERVE
            and arguments.get('count_slot', 'X') == 'X')


def native_coordinate_source(context, path, sha):
    """Exact accepted native written-slot projection, never H5AD-name inference."""
    if context is None:
        return None
    context.validate_anchors()
    for proof in context.proofs.values():
        for tool in NATIVE:
            if proof.kind != 'native_' + tool.removeprefix('run_') + '_matrix':
                continue
            result = proof.result_metadata
            if result.get('matrix_sha256') != sha or str(result.get('matrix_path')) != str(path):
                continue
            descriptor = describe(tool, result)
            if descriptor is None or not descriptor['coordinates'] or not descriptor['preserves_counts']:
                raise ValueError('Selected native publication has no qualified written UMAP coordinates')
            return dict(field=descriptor['coordinates'], source_sha256=result['source_identity']['sha256'])
    return None


def native_group_rows(context, path, sha, source_sha, cell_ids):
    """Consume verified native groups in the existing invocation authority DAG.

    Return None only when this is not a native publication in that DAG. A loose
    file, inherited field or independent return cannot acquire native authority.
    """
    if context is None:
        return None
    context.validate_anchors()
    rows = {}
    for proof in context.proofs.values():
        for tool in NATIVE:
            if proof.kind == 'native_' + tool.removeprefix('run_') + '_matrix':
                result = proof.result_metadata
                descriptor = describe(tool, result)
                if descriptor is not None:
                    rows[result['matrix_sha256']] = (tool, result, descriptor)
    selected = rows.get(sha)
    if selected is None:
        return None
    if str(selected[1]['matrix_path']) != str(path):
        raise ValueError('Selected native grouping locator differs from its verified publication.')
    def lineage(identity):
        found, seen = [], set()
        while identity in rows:
            if identity in seen:
                raise ValueError('Cyclic native scientific dependency.')
            seen.add(identity)
            tool, result, descriptor = rows[identity]
            found.append((tool, result, descriptor))
            if not descriptor['preserves_counts']:
                return identity, found
            identity = result['source_identity']['sha256']
        return identity, found
    root, ancestors = lineage(sha)
    if root != lineage(source_sha)[0]:
        raise ValueError('Clustering must retain the exact original count-file identity')
    groups = next(((result, desc) for _, result, desc in ancestors if desc['grouping']), None)
    if groups is None:
        raise ValueError('Selected native publication has no verified written clustering slot.')
    from .categorical import observations, require_same_cells
    result, descriptor = groups
    ids, labels = observations(result['matrix_path'], result['matrix_sha256'], descriptor['grouping'])
    require_same_cells(tuple(map(str, cell_ids)), ids)
    if labels.isna().any():
        raise ValueError('Selected grouping contains missing labels.')
    context.validate_anchors()
    return list(zip(ids, map(str, labels)))


def source_bound_cluster_rows(cell_ids, analysis_path, analysis_sha256, source_sha256):
    """Project accepted native labels only for the exact source and ordered cells."""
    from agent.tools.data.authority_context import current
    rows = native_group_rows(current(), analysis_path, analysis_sha256, source_sha256, cell_ids)
    if rows is None:
        raise ValueError('Grouping requires an accepted native clustering of the exact source')
    return rows
