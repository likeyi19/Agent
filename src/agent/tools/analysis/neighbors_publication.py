"""Owner-qualified neighbor publications using the shared scientific authority DAG.

The graph is produced by the existing Scanpy owner. Qualification checks its
published structure, parameters and exact aligned source representation; it
never repeats neighbor construction. Accepted proofs permit integrity-only
historical consumption without opening the original embedding or ID sources.
"""

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from agent.schemas.orchestration import _serialize
from agent.schemas.verification_authority import (
    AuthorityError, VerifiedArtifactAuthority, canonical, digest,
)
from agent.tools.data.authority_context import current, owned_verification
from agent.tools.data._fragments_common import snapshots, unchanged
from agent.tools.data.fastq_verification_authority import file_closure
from . import embedding_analysis as science


KIND = 'cell_neighbors'
ARTIFACT = 'agent.cell-neighbors'
CONTRACT = 'cell-neighbors-publication.v1'
MAX_BYTES = 65536
PROFILE = dict(representation='X_epizoo', embedding_dim=512, embedding_dtype='float32',
               storage='compact-embedding-only-anndata', backend='Scanpy', method='umap',
               transformer='none', verification='structure-exact-source-alignment.v1')
PROFILE_SHA256 = digest(PROFILE)
VERIFIER = dict(id='agent.cell-neighbors-source-alignment', compatibility_version='1')
ARGUMENTS = ('embedding_path', 'cell_ids_path', 'output_dir', 'n_neighbors',
             'metric', 'random_seed', 'overwrite')
SUMMARY = tuple(name for name in science.CellNeighborsToolResult.__annotations__
                if name not in ('artifact_type', 'contract_version', 'manifest_path', 'manifest_sha256'))


def _shape(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise AuthorityError('Invalid neighbor publication fields.')


def _sha(value):
    from agent.schemas.verification_authority import _sha as validate
    validate(value)


def _path(value, *, current=True):
    if type(value) is not str or not Path(value).is_absolute() or str(Path(value)) != value:
        raise AuthorityError('Neighbor publication paths must be canonical absolute paths.')
    if '..' in Path(value).parts or any(ord(c) < 32 for c in value) or (current and Path(value) != Path(value).resolve()):
        raise AuthorityError('Neighbor publication paths must be canonical absolute paths.')


def arguments(value):
    args = dict(value)
    if set(args) - set(ARGUMENTS) or not {'embedding_path', 'cell_ids_path', 'output_dir'} <= set(args):
        raise AuthorityError('Invalid neighbor publication arguments.')
    for name, default in (('n_neighbors', 15), ('metric', 'euclidean'),
                          ('random_seed', 0), ('overwrite', False)):
        args.setdefault(name, default)
    for name in ('embedding_path', 'cell_ids_path', 'output_dir'):
        if not isinstance(args[name], (str, Path)):
            raise AuthorityError('Invalid neighbor publication path argument.')
        args[name] = (str(Path(args[name]).expanduser().resolve()) if name == 'output_dir'
                      else os.path.abspath(Path(args[name]).expanduser()))
        _path(args[name], current=name == 'output_dir')
    if Path(args['embedding_path']).suffix.casefold() != '.npy' or Path(args['cell_ids_path']).suffix.casefold() != '.txt':
        raise AuthorityError('Neighbor publication source types are incompatible.')
    science._validate_seed(args['random_seed'])
    science._validate_overwrite(args['overwrite'])
    if type(args['n_neighbors']) is not int or args['n_neighbors'] < 2:
        raise AuthorityError('Invalid neighbor publication neighbor count.')
    if args['metric'] not in ('euclidean', 'cosine'):
        raise AuthorityError('Invalid neighbor publication metric.')
    return args


def _publication(value, execution_identity, *, sources):
    args = arguments(value)
    if execution_identity is not None:
        _sha(execution_identity)
    root = Path(args['output_dir'])
    stem = Path(sources['embedding']['path']).stem + '.neighbors'
    token = digest(dict(arguments=args, execution_identity=execution_identity,
                        contract=CONTRACT, profile=PROFILE_SHA256))
    return args, root, stem, token


def _file(entry, *, absolute=True):
    _shape(entry, ('path', 'sha256', 'size_bytes'))
    _sha(entry['sha256'])
    if type(entry['size_bytes']) is not int or entry['size_bytes'] < 0:
        raise AuthorityError('Invalid neighbor publication file size.')
    if absolute:
        _path(entry['path'], current=False)
    elif type(entry['path']) is not str or Path(entry['path']).name != entry['path']:
        raise AuthorityError('Neighbor payload must be a single owned filename.')


def read_json(path, sha=None):
    path = Path(path)
    _path(str(path))
    with path.open('rb') as handle:
        raw = handle.read(MAX_BYTES + 1)
    return read_json_bytes(raw, sha)


def read_json_bytes(raw, sha=None):
    """Parse bounded owner metadata from a caller's securely opened snapshot."""
    if type(raw) is not bytes:
        raise AuthorityError('Neighbor publication metadata must be bytes.')
    if len(raw) > MAX_BYTES:
        raise AuthorityError('Neighbor publication metadata exceeds its bound.')
    actual = hashlib.sha256(raw).hexdigest()
    if sha is not None:
        _sha(sha)
        if actual != sha:
            raise AuthorityError('Neighbor publication metadata identity changed.')
    from agent.tools.data.scatac_fragments_v2 import _pairs
    def invalid(_):
        raise AuthorityError('Nonfinite neighbor publication metadata.')
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=invalid), actual


def load_manifest(path, sha=None):
    value, actual = read_json(path, sha)
    return _validate_manifest(value, actual, path)


def load_manifest_bytes(raw, path, sha=None):
    """Apply the same closed owner contract without reopening a locator."""
    path = Path(path)
    _path(str(path))
    value, actual = read_json_bytes(raw, sha)
    return _validate_manifest(value, actual, path)


def _validate_manifest(value, actual, path):
    _shape(value, ('artifact_type', 'schema_version', 'contract_version', 'profile_sha256',
                   'arguments', 'execution_identity', 'sources', 'analysis',
                   'cell_order_sha256', 'summary'))
    if (value['artifact_type'], value['schema_version'], value['contract_version'],
            value['profile_sha256']) != (ARTIFACT, 1, CONTRACT, PROFILE_SHA256):
        raise AuthorityError('Unsupported neighbor publication contract.')
    if type(value['schema_version']) is not int:
        raise AuthorityError('Unsupported neighbor publication contract.')
    if value['arguments'] != arguments(value['arguments']):
        raise AuthorityError('Noncanonical neighbor publication arguments.')
    if value['execution_identity'] is not None:
        _sha(value['execution_identity'])
    _shape(value['sources'], ('embedding', 'cell_ids'))
    for key in ('embedding', 'cell_ids'):
        _file(value['sources'][key])
    _file(value['analysis'], absolute=False)
    expected = Path(value['sources']['embedding']['path']).stem + '.neighbors.h5ad'
    if value['analysis']['path'] != expected or Path(path).parent != Path(value['arguments']['output_dir']):
        raise AuthorityError('Neighbor publication has incorrect artifact membership.')
    _sha(value['cell_order_sha256'])
    _shape(value['summary'], SUMMARY)
    if value['summary']['analysis_path'] != str(Path(path).parent / expected):
        raise AuthorityError('Neighbor publication result has incorrect artifact membership.')
    return value, actual


def capture_inputs(embedding_path, cell_ids_path):
    paths = (str(embedding_path), str(cell_ids_path))
    before = snapshots(paths)
    records = file_closure(paths)
    unchanged(before)
    by_path = {entry['path']: entry for entry in records}
    return dict(embedding=by_path[paths[0]], cell_ids=by_path[paths[1]]), before


def _write_json(path, value, *, overwrite):
    if path.exists() and not overwrite:
        raise FileExistsError(f'Neighbor publication already exists: {path}. Use overwrite=True to replace it.')
    raw = canonical(value)
    if len(raw) > MAX_BYTES:
        raise AuthorityError('Neighbor publication metadata exceeds its bound.')
    descriptor, name = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() and not overwrite:
            raise FileExistsError(f'Neighbor publication already exists: {path}.')
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)
    return hashlib.sha256(raw).hexdigest()


def _receipt_value(args, identity, token, sha):
    return dict(artifact_type=ARTIFACT + '-receipt', schema_version=1,
                contract_version=CONTRACT, execution_identity=identity, publication_token=token,
                arguments_sha256=digest(args), manifest_sha256=sha)


def publish(value, result, sources, before):
    context = current()
    execution = None if context is None else context.execution
    identity = None
    if execution is not None and execution[0] == KIND and execution[1] == Path(value['output_dir']):
        identity = execution[2]
    args, root, stem, token = _publication(value, identity, sources=sources)
    unchanged(before)
    artifact = file_closure([result['analysis_path']])[0]
    artifact['path'] = Path(artifact['path']).name
    cell_ids = science._load_cell_ids(Path(sources['cell_ids']['path']))
    manifest = dict(artifact_type=ARTIFACT, schema_version=1, contract_version=CONTRACT,
                    profile_sha256=PROFILE_SHA256, arguments=args, execution_identity=identity,
                    sources=sources, analysis=artifact, cell_order_sha256=science._cell_order_digest(cell_ids),
                    summary=dict(result))
    path = root / (stem + '.manifest.json')
    sha = _write_json(path, manifest, overwrite=args['overwrite'])
    _write_json(root / (stem + '.receipt.json'), _receipt_value(args, identity, token, sha),
                overwrite=args['overwrite'])
    verify_neighbors_publication(path, expected_sha256=sha)
    unchanged(before)
    descriptor = os.open(root, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return _summary(manifest, path, sha)


def _summary(value, path, sha):
    return deepcopy(value['summary']) | dict(artifact_type=ARTIFACT, contract_version=CONTRACT,
                                           manifest_path=str(path), manifest_sha256=sha)


def reuse_result(kind, description, proof, kwargs):
    """Return reviewed metadata after shared accepted-proof integrity checks."""
    return _summary(description['manifest'], description['manifest_path'], description['manifest_sha256'])


def proof_metadata(kind, result, context):
    return dict(result)


def check_sources(value):
    expected = sorted(value['sources'].values(), key=lambda entry: entry['path'])
    for key, name in (('embedding', 'embedding_path'), ('cell_ids', 'cell_ids_path')):
        if str(Path(value['arguments'][name]).resolve()) != value['sources'][key]['path']:
            raise AuthorityError('Neighbor publication source locator changed during production.')
    if file_closure(entry['path'] for entry in expected) != expected:
        raise AuthorityError('Neighbor source bytes differ from their production identity.')


def _verify(path, *, expected_sha256=None):
    path = Path(path)
    before = snapshots([path])
    value, sha = load_manifest(path, expected_sha256)
    args = value['arguments']
    analysis_path = path.parent / value['analysis']['path']
    before.update(snapshots([analysis_path, *(entry['path'] for entry in value['sources'].values())]))
    if file_closure([analysis_path])[0] != dict(value['analysis'], path=str(analysis_path)):
        raise AuthorityError('Neighbor scientific artifact bytes changed.')
    check_sources(value)
    artifact = science._read_analysis(analysis_path)
    cell_ids, distances, connectivities = science._validate_neighbors_artifact(
        artifact, expected_stages=frozenset({'neighbors'}))
    source_ids = science._load_cell_ids(Path(value['sources']['cell_ids']['path']))
    embeddings = science._load_embeddings(Path(value['sources']['embedding']['path']))
    if cell_ids != source_ids or value['cell_order_sha256'] != science._cell_order_digest(cell_ids):
        raise AuthorityError('Neighbor published ordered cell identity differs from its source.')
    stored = np.asarray(artifact.obsm[science.EPIZOO_REPRESENTATION_KEY])
    if embeddings.shape != stored.shape or any(
            not np.array_equal(embeddings[start:start + 8192], stored[start:start + 8192])
            for start in range(0, len(cell_ids), 8192)):
        raise AuthorityError('Neighbor published representation differs from its aligned source.')
    parameters = dict(n_neighbors=args['n_neighbors'], metric=args['metric'], method='umap',
                      transformer='none', random_seed=args['random_seed'], use_rep='X_epizoo')
    provenance = science._provenance(artifact)
    if (provenance.get('source_embedding_path') != value['sources']['embedding']['path'] or
            provenance.get('source_cell_ids_path') != value['sources']['cell_ids']['path'] or
            not isinstance(provenance.get('parameters'), Mapping) or
            dict(provenance['parameters'].get('neighbors', {})) != parameters):
        raise AuthorityError('Neighbor scientific provenance or parameters are invalid.')
    if provenance.get('software_versions') != science._software_versions():
        raise AuthorityError('Neighbor initial production software provenance is invalid.')
    scanpy_parameters = artifact.uns['neighbors'].get('params')
    if (not isinstance(scanpy_parameters, Mapping) or
            any(scanpy_parameters.get(name) != expected for name, expected in
                (('n_neighbors', args['n_neighbors']), ('metric', args['metric']), ('method', 'umap'),
                 ('random_state', args['random_seed']), ('use_rep', 'X_epizoo')))):
        raise AuthorityError('Neighbor graph parameters disagree with its production provenance.')
    expected_summary = dict(status='success', embedding_path=value['sources']['embedding']['path'], cell_ids_path=value['sources']['cell_ids']['path'],
        analysis_path=str(analysis_path), n_cells=len(cell_ids), embedding_dim=512, n_neighbors=args['n_neighbors'],
        metric=args['metric'], neighbors_method='umap', transformer='none', random_seed=args['random_seed'],
        connectivities_nnz=int(connectivities.nnz), distances_nnz=int(distances.nnz), finite=True,
        cell_order_preserved=True, backend='Scanpy', software_versions=dict(provenance.get('software_versions', {})))
    if value['summary'] != expected_summary or not 2 <= args['n_neighbors'] < len(cell_ids):
        raise AuthorityError('Neighbor publication summary differs from its scientific artifact.')
    unchanged(before)
    return _summary(value, path, sha)


@owned_verification(KIND)
def verify_neighbors_publication(path, *, expected_sha256=None):
    return _verify(path, expected_sha256=expected_sha256)


def verify_result(resolved_arguments, result):
    """Verify the exact registered result with the current operation's proof."""
    metadata = {'artifact_type', 'contract_version', 'manifest_path', 'manifest_sha256'}
    if not metadata <= result.keys():
        raise AuthorityError('Incomplete neighbor publication result.')
    path = result['manifest_path']
    _path(path)
    value, sha = load_manifest(path, result['manifest_sha256'])
    if value['arguments'] != arguments(resolved_arguments):
        raise AuthorityError('Neighbor publication differs from declared execution arguments.')
    reviewed = verify_neighbors_publication(path, expected_sha256=sha)
    if dict(result) != reviewed:
        raise AuthorityError('Neighbor result differs from its owner-qualified publication.')
    return reviewed


def describe(path, sha, context):
    path = Path(path)
    value, actual = load_manifest(path, sha)
    files = context.files([path, path.parent / value['analysis']['path']])
    expected = dict(value['analysis'], path=str(path.parent / value['analysis']['path']))
    if expected not in files:
        raise AuthorityError('Neighbor scientific artifact bytes changed.')
    historical = sorted(value['sources'].values(), key=lambda entry: entry['path'])
    identity = dict(kind=KIND, manifest=value,
        files=[dict(entry, path='@owned/' + Path(entry['path']).name) for entry in files if entry['path'] != str(path)],
        execution_identity=context.execution_for(KIND, path), upstream={}, historical_sources=historical,
        profile=PROFILE_SHA256, verifier=VERIFIER, qualification={}, policy='historical_verified_sources.v1')
    return dict(kind=KIND, manifest_path=str(path), manifest_sha256=actual, manifest=value,
                proof_identity=identity, files=files, historical_sources=historical, upstream={},
                profile=PROFILE_SHA256, verifier=VERIFIER, qualification={})


def publication_record(context, tool_name, arguments_value, result, execution_identity, result_metadata):
    value, sha = load_manifest(result['manifest_path'], result['manifest_sha256'])
    args, root, stem, token = _publication(arguments_value, execution_identity, sources=value['sources'])
    path = root / (stem + '.manifest.json')
    receipt_path = root / (stem + '.receipt.json')
    if tool_name != 'build_cell_neighbors' or result['manifest_path'] != str(path):
        raise AuthorityError('Neighbor publication belongs to a different producing execution.')
    if (value['arguments'] != args or value['execution_identity'] != execution_identity or
            dict(result) != _summary(value, path, sha)):
        raise AuthorityError('Neighbor publication lacks exact producing execution provenance.')
    if _serialize(result_metadata) != dict(result):
        raise AuthorityError('Neighbor proof metadata differs from its owner-qualified result.')
    receipt, _ = read_json(receipt_path)
    if receipt != _receipt_value(args, execution_identity, token, sha):
        raise AuthorityError('Neighbor publication receipt identity changed.')
    description, key = context.describe(KIND, path, sha)
    files = context.files([*(entry['path'] for entry in description['files']), receipt_path])
    return VerifiedArtifactAuthority(dict(schema_version=2, artifact_type=ARTIFACT, artifact_contract=CONTRACT,
        publication_path=str(path), manifest_sha256=sha, files=files, execution_identity=execution_identity,
        arguments_sha256=digest(args), receipt_sha256=next(entry['sha256'] for entry in files if entry['path'] == str(receipt_path)),
        upstream={'producer_inputs': digest(description['proof_identity'])},
        resources=dict(scientific_proof_sha256=key, manifest_identity=value, result_metadata=_serialize(result_metadata)),
        science_profile=PROFILE_SHA256, producer_qualification={}, verifier=VERIFIER,
        scope='scientific_correctness.v1', completion='succeeded',
        integrity_scope='artifact_integrity_lineage_historical_sources.v1',
        source_policy='historical_verified_sources.v1', historical_sources=description['historical_sources']))
