"""Bounded, path-free download of an accepted MAESTRO annotation.

Scientific authority belongs to the annotation owner. Export copies the exact
sparse arrays, axes and categorical annotations into a new HDF5 file; the one
private matrix locator is omitted. Source and delivered digests stay distinct.
"""
import h5py
import numpy as np
from agent.tools.analysis import annotation_contract as owner
from agent.tools.analysis import neighbors_delivery as io
from . import matrix_delivery as matrix
from .matrix_delivery import _string_block, _private_metadata, _redactions, _copy

CONTRACT = owner.CONTRACT
ROLE = 'annotation_export'
FILENAME = 'annotated.h5ad'
LABEL = 'Annotated cells — safe export'
EXPORT_CONTRACT = 'scatac-cell-type-annotation-export.v1'
DESTINATION_OVERHEAD_BYTES = 65536


def _strings(item):
    matrix._reject(h5py.check_string_dtype(item.dtype) is None or item.ndim != 1)
    for left in range(0, len(item), 4096):
        for text in _string_block(item, left, min(left + 4096, len(item))):
            matrix._reject(not text or any(ord(c) < 32 or ord(c) == 127 for c in text)
                           or matrix._PATH_TEXT.search(text) is not None)
            yield text


def _feature_axis(group, length):
    columns = ('_index', 'chrom', 'start', 'end', 'matrix_column_index')
    matrix._reject(not isinstance(group, h5py.Group) or set(group) != set(columns))
    matrix._attrs(group, {'encoding-type': 'dataframe', 'encoding-version': '0.2.0',
                         '_index': '_index', 'column-order': columns[1:]})
    for name in columns:
        item = group[name]
        matrix._reject(not isinstance(item, h5py.Dataset) or item.shape != (length,))
        strings = name in {'_index', 'chrom'}
        matrix._attrs(item, {'encoding-type': 'string-array' if strings else 'array', 'encoding-version': '0.2.0'})
        if strings:
            for _ in _strings(item): pass
        else:
            matrix._reject(item.dtype != np.dtype('int64'))


def _inspect(file, manifest, *, exported=False):
    if owner.external(manifest):
        from agent.tools.analysis import peak_h5ad
        from agent.tools.analysis.scatac_annotation import _provenance
        peak_h5ad.inspect(file, privacy=exported or manifest['schema_version'] == 1,
            private_paths=() if exported else ('/uns/cell_type_annotation/source_path',))
        provenance = _provenance(manifest['arguments'], manifest['source_authority_sha256'], manifest.get('feature_participation'), version=manifest['schema_version'])
        if exported: provenance.pop('source_path')
        io._group(file['uns/cell_type_annotation'],provenance)
        for name,value in provenance.items():
            io._scalar(file['uns/cell_type_annotation'][name],value,private=name=='source_path')
        return
    reject = matrix._reject
    reject(file.userblock_size != 0 or set(file) != matrix._TOP_LEVEL)
    matrix._safe_objects(file, maximum_depth=3)
    matrix._attrs(file, {'encoding-type': 'anndata', 'encoding-version': '0.1.0'})
    for name in ('layers', 'obsm', 'obsp', 'varm', 'varp'):
        io._group(file[name], ())
    x = file['X']
    reject(not isinstance(x, h5py.Group) or set(x) != {'data', 'indices', 'indptr'}
           or set(x.attrs) != {'encoding-type', 'encoding-version', 'shape'})
    dims = np.asarray(x.attrs['shape'])
    reject(dims.shape != (2,) or dims.dtype != np.dtype('int64')
           or dims[0] != manifest['summary']['n_cells'] or dims[1] < 1
           or x.attrs['encoding-type'] != 'csr_matrix' or x.attrs['encoding-version'] != '0.1.0')
    n, p = map(int, dims)
    for name in x:
        d = x[name]
        reject(not isinstance(d, h5py.Dataset) or d.ndim != 1 or len(d.attrs) or d.dtype != np.dtype('int64'))
    reject(len(x['data']) != len(x['indices']) or len(x['indptr']) != n + 1)
    base = ('_index', 'namespace', 'barcode_identifier', 'matrix_row_index')
    columns = (*base, *owner.OBS_COLUMNS)
    obs = file['obs']
    reject(not isinstance(obs, h5py.Group) or set(obs) != set(columns))
    matrix._attrs(obs, {'encoding-type': 'dataframe', 'encoding-version': '0.2.0',
                       '_index': '_index', 'column-order': columns[1:]})
    for name in base:
        d = obs[name]
        reject(not isinstance(d, h5py.Dataset) or d.shape != (n,))
        strings = name != 'matrix_row_index'
        matrix._attrs(d, {'encoding-type': 'string-array' if strings else 'array', 'encoding-version': '0.2.0'})
        if strings:
            for _ in _strings(d): pass
        else:
            reject(d.dtype != np.dtype('int64'))
    for name in owner.OBS_COLUMNS:
        group = obs[name]
        reject(not isinstance(group, h5py.Group) or set(group) != {'codes', 'categories'}
               or set(group.attrs) != {'encoding-type', 'encoding-version', 'ordered'}
               or group.attrs['encoding-type'] != 'categorical' or group.attrs['encoding-version'] != '0.2.0'
               or not isinstance(group.attrs['ordered'], (bool, np.bool_)) or group.attrs['ordered'])
        codes, categories = group['codes'], group['categories']
        reject(not isinstance(codes, h5py.Dataset) or codes.shape != (n,) or codes.dtype.kind != 'i'
               or codes.dtype.itemsize not in (1, 2, 4, 8)
               or not isinstance(categories, h5py.Dataset) or categories.ndim != 1 or len(categories) > n)
        matrix._attrs(codes, {'encoding-type': 'array', 'encoding-version': '0.2.0'})
        matrix._attrs(categories, {'encoding-type': 'string-array', 'encoding-version': '0.2.0'})
        for _ in _strings(categories): pass
        for left in range(0, n, 65536):
            values = codes[left:left+65536]
            reject(np.any(values < (-1 if name == 'primary_annotation' else 0)) or np.any(values >= len(categories)))
    _feature_axis(file['var'], p)
    source_uns = matrix._CREATED_UNS
    io._group(file['uns'], source_uns | {'cell_type_annotation'} | ({io.EXPORT_KEY} if exported else set()))
    for name in source_uns:
        d = file['uns'][name]
        reject(not isinstance(d, h5py.Dataset) or d.shape != ())
        matrix._attrs(d, {'encoding-type': 'string', 'encoding-version': '0.2.0'})
        value = next(matrix._strings(d))
        expected = {'species': manifest['inputs']['species'], 'assembly': manifest['inputs']['assembly'],
                    'matrix_semantics': 'fragment_counts',
                    'logical_matrix_sha256': manifest['summary']['logical_matrix_sha256'],
                    'ordered_feature_sha256': manifest['matrix_binding']['ordered_features_sha256']}
        reject(name in expected and value != expected[name])
    from agent.tools.analysis.scatac_annotation import _provenance
    provenance = _provenance(manifest['arguments'], manifest['source_authority_sha256'], version=manifest['schema_version'])
    if exported: provenance.pop('matrix_manifest_path')
    io._group(file['uns/cell_type_annotation'], provenance)
    for name, value in provenance.items():
        io._scalar(file['uns/cell_type_annotation'][name], value, private=name == 'matrix_manifest_path')


def create_annotation_export(source_stream, destination_stream, manifest, *, source_sha256,
                             source_size_bytes, manifest_sha256, max_bytes):
    matrix._reject(manifest['contract_version'] != CONTRACT
        or manifest['sidecars']['annotated.h5ad'] != dict(sha256=source_sha256, size_bytes=source_size_bytes)
        or not 0 < max_bytes <= matrix.MAX_MATRIX_BYTES)
    destination_stream.seek(0, 2)
    matrix._reject(destination_stream.tell() != 0)
    metadata = dict(contract_version=EXPORT_CONTRACT, source_contract=CONTRACT,
        source_sha256=source_sha256, source_size_bytes=source_size_bytes,
        source_manifest_sha256=manifest_sha256,
        provenance_policy='private-source-locator-omitted' if owner.external(manifest) else 'historical-matrix-locator-omitted',
        authority_scope='export-does-not-issue-scientific-authority')
    with h5py.File(source_stream, 'r') as source:
        _inspect(source, manifest)
        if owner.external(manifest):
            from agent.tools.analysis.peak_h5ad import inspect
            numeric = inspect(source)
        else:
            numeric = sum(source[n].size * source[n].dtype.itemsize for n in ('X/data', 'X/indices', 'X/indptr'))
        matrix._reject(numeric > max_bytes)
        omitted = _redactions(source) if owner.external(manifest) and manifest['schema_version'] == 2 else set()
        if manifest['schema_version'] == 2:
            metadata['omitted_private_metadata_fields'] = len(omitted)
        with h5py.File(io._BoundedDestination(destination_stream, max_bytes), 'w') as destination:
            _copy(source, destination, omitted)
            io._write_metadata(destination['uns'].create_group(io.EXPORT_KEY), metadata)
    destination_stream.flush(); destination_stream.seek(0)
    with h5py.File(destination_stream, 'r') as exported:
        _inspect(exported, manifest, exported=True)
        io._group(exported['uns'][io.EXPORT_KEY], metadata)
        for name, value in metadata.items(): io._scalar(exported['uns'][io.EXPORT_KEY][name], value)
    destination_stream.seek(0)


def accepted_annotation(app, session_id, revision, origin, output, *, step_id=None):
    """Inventory accepted metadata only; do not reopen upstream scientific inputs."""
    import hashlib
    from dataclasses import asdict
    from pathlib import Path
    from agent.schemas.orchestration import _serialize
    from agent.orchestration.prior_outputs import accepted_step_digest
    from agent.orchestration.verification_authority import _accepted
    from agent.tools.analysis import scatac_annotation as science
    from .interactive import _ScientificMatrixArtifact
    from .interactive_schemas import ScientificArtifactHandle
    from .session_state import digest
    store = app._application.run_store
    run_id = revision.run_id if output is None else output.run_id
    step_id = step_id if output is None else output.step_id
    step, authority, execution, anchor = _accepted(store, run_id, step_id)
    result, record = _serialize(step.result), authority.to_dict()
    if (step.tool_name != 'annotate_scATAC_cell_types'
            or output is not None and accepted_step_digest(step) != output.accepted_step_sha256
            or record['schema_version'] != 2 or record['completion'] != 'succeeded'
            or record['scope'] != 'scientific_correctness.v1'
            or record['artifact_contract'] != CONTRACT or record['artifact_type'] != owner.ARTIFACT
            or record['science_profile'] != owner.profile_sha(step.resolved_arguments, result['artifact_schema_version'])
            or record['verifier'] != owner.verifier(result['artifact_schema_version'])):
        raise ValueError('Unsupported annotation authority.')
    spec = app._application.registry.get(step.tool_name)
    spec.result_contract.validate(result)
    key = 'manifest_path' if output is None else output.output_key
    if key not in {'manifest_path', 'manifest_sha256', 'annotated_h5ad_path'}:
        raise ValueError('Unreviewed annotation output.')
    workspace = app._application._workspace
    root = workspace.runs / workspace.run_digest(run_id) / 'scientific'
    args, destination, token = science._publication(step.resolved_arguments, execution, version=result['artifact_schema_version'])
    path = Path(result['manifest_path'])
    if path != destination / 'artifact/manifest.json' or path != path.resolve():
        raise ValueError('Unmanaged annotation publication.')
    path.relative_to(root)
    with matrix._open_matrix_source(path, root) as stream: raw = stream.read(owner.MAX_BYTES + 1)
    if len(raw) > owner.MAX_BYTES or hashlib.sha256(raw).hexdigest() != result['manifest_sha256']:
        raise ValueError('Changed annotation manifest.')
    # Closed owner loader, no upstream source verification or backend replay.
    manifest = owner.load_manifest(path, result['manifest_sha256'])
    receipt_path = destination / 'receipt.json'
    with matrix._open_matrix_source(receipt_path, root) as stream: receipt = stream.read(16385)
    expected_receipt = science._make_receipt(args, token, result['manifest_sha256'], version=manifest['schema_version'])
    import json
    payload = path.parent / FILENAME
    pin = manifest['sidecars'][FILENAME]
    files = {f['path']: f for f in record['files']}
    expected_files = [dict(path=str(path), sha256=result['manifest_sha256'], size_bytes=len(raw)),
        dict(path=str(receipt_path), sha256=hashlib.sha256(receipt).hexdigest(), size_bytes=len(receipt)),
        dict(path=str(payload), **pin)]
    if (len(receipt) > 16384 or json.loads(receipt) != expected_receipt
            or record['receipt_sha256'] != expected_files[1]['sha256']
            or record['arguments_sha256'] != owner.digest(args)
            or record['publication_path'] != str(path) or record['manifest_sha256'] != result['manifest_sha256']
            or record['resources']['manifest_identity'] != manifest or manifest['arguments'] != args
            or {k: v for k, v in record['resources']['result_metadata'].items() if k not in {'manifest_path', 'annotated_h5ad_path'}}
               != {k: v for k, v in result.items() if k not in {'manifest_path', 'annotated_h5ad_path'}}
            or result != science._summary(manifest, path, result['manifest_sha256'])
            or any(files.get(f['path']) != f for f in expected_files)
            or _accepted(store, run_id, step_id)[3] != anchor):
        raise ValueError('Annotation result, receipt and authority differ.')
    identity = dict(session_id=session_id, revision_id=revision.revision_id, turn_id=origin.turn_id,
        accepted_step=dict(run_id=run_id, step_id=step_id, accepted_step_sha256=accepted_step_digest(step)),
        artifact_role=ROLE, export_contract=EXPORT_CONTRACT, authority_sha256=authority.identity_sha256,
        manifest_sha256=result['manifest_sha256'], payload_sha256=pin['sha256'], size_bytes=pin['size_bytes'])
    reference = ScientificArtifactHandle(digest(identity), 'scientific_annotation', pin['sha256'],
        revision.revision_id, origin.turn_id, pin['size_bytes'], FILENAME, LABEL)
    return _ScientificMatrixArtifact(reference, payload, root, pin['sha256'], pin['size_bytes'],
        CONTRACT, manifest, ROLE, result['manifest_sha256'])
