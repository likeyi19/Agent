"""Public primary annotation: existing science, authority DAG and publication envelope."""
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
import tempfile
import os
from typing import TypedDict

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from . import annotation_contract as c, marker_annotation as science
from agent.tools.data import scatac_matrix as publication
from agent.tools.data.authority_context import current, VerificationContext, authority_operation, owned_verification
from agent.tools._cancellation import cancellation_checkpoint

RECOVERY_POLICY = c.RECOVERY_POLICY
ARGUMENTS = c.ARGUMENTS


class CellTypeAnnotationResult(TypedDict):
    status: str
    manifest_path: str
    manifest_sha256: str
    artifact_type: str
    artifact_schema_version: int
    contract_version: str
    annotated_h5ad_path: str
    annotated_h5ad_sha256: str
    source_manifest_sha256: str
    source_authority_sha256: str
    annotation_spec_sha256: str
    gene_resource_sha256: str
    signature_resource_sha256: str
    logical_matrix_sha256: str
    ordered_cells_sha256: str
    ordered_feature_sha256: str
    annotation_profile: str
    species: str
    assembly: str
    context: str
    n_cells: int
    n_groups: int
    assigned_cells: int
    unassigned_cells: int
    ambiguous_cells: int
    group_summary: list
    groups_omitted: int
    validation_state: str


def _operation():
    return nullcontext(current()) if current() is not None else authority_operation(VerificationContext())


def source(args, spec):
    from agent.orchestration.verification_authority import bind_annotation_matrix
    authority, manifest = bind_annotation_matrix(spec, args['matrix_manifest_path'], args['matrix_manifest_sha256'], current())
    if (manifest['contract_version'] != c.matrix.CONTRACT or
            (manifest['species'], manifest['assembly']) != (spec['species'], spec['assembly'])):
        raise ValueError('Unsupported matrix semantics or incompatible species/assembly')
    path = Path(args['matrix_manifest_path']).parent / manifest['matrix']['path']
    # Ordered cell IDs are a derived integrity binding, not inferred biology.
    a = ad.read_h5ad(path, backed='r')
    try:
        cells_sha = science.ordered_identity_sha256(a.obs_names.tolist())
    finally:
        a.file.close()
    binding = asdict(science.MatrixInput(args['matrix_manifest_path'], args['matrix_manifest_sha256'],
        manifest['matrix']['sha256'], cells_sha, manifest['ordered_feature_sha256'],
        spec['species'], spec['assembly'], c.matrix.PROFILE.profile_id))
    return authority, manifest, binding


def _publication(args, execution_identity):
    args = c.arguments(args)
    token = c.digest(dict(arguments=args, policy=RECOVERY_POLICY, execution_identity=execution_identity,
                          profile_sha256=c.PROFILE_SHA256))
    return args, Path(args['output_dir']) / ('annotation-' + token), token


def _make_receipt(args, token, sha):
    return dict(artifact_type='agent.annotation-receipt', schema_version=1, policy=RECOVERY_POLICY,
                execution_identity=token, arguments_sha256=c.digest(args), manifest_sha256=sha,
                profile_sha256=c.PROFILE_SHA256)


def _receipt(destination, args, token=None):
    r = c.read_json(destination / 'receipt.json', limit=16384)
    if (r != _make_receipt(args, r['execution_identity'], r['manifest_sha256']) or
            destination.name != 'annotation-' + r['execution_identity'] or
            token is not None and token != r['execution_identity']):
        raise ValueError('Annotation recovery receipt mismatch')
    c.matrix.sha(r['execution_identity']); c.matrix.sha(r['manifest_sha256'])
    return r


def _summary(value, path, sha):
    binding = value['matrix_binding']
    return dict(status='success', manifest_path=str(path), manifest_sha256=sha,
        artifact_type=c.ARTIFACT, artifact_schema_version=1, contract_version=c.CONTRACT,
        annotated_h5ad_path=str(path.parent / 'annotated.h5ad'),
        annotated_h5ad_sha256=value['sidecars']['annotated.h5ad']['sha256'],
        source_manifest_sha256=binding['manifest_sha256'], source_authority_sha256=value['source_authority_sha256'],
        annotation_spec_sha256=value['arguments']['annotation_spec_sha256'],
        gene_resource_sha256=value['inputs']['gene_resource']['sha256'],
        signature_resource_sha256=value['inputs']['signature_resource']['sha256'],
        ordered_cells_sha256=binding['ordered_cells_sha256'], ordered_feature_sha256=binding['ordered_features_sha256'],
        annotation_profile=science.PROFILE, species=binding['species'], assembly=binding['assembly'],
        context=value['inputs']['context'], validation_state='not_assessed', **value['summary'])


def _counts(groups, matrix):
    assigned = sum(g['n_cells'] for g in groups if g['status'] == 'assigned')
    return dict(logical_matrix_sha256=matrix['logical_matrix_sha256'], n_cells=matrix['shape'][0],
        n_groups=len(groups), assigned_cells=assigned, unassigned_cells=matrix['shape'][0]-assigned,
        ambiguous_cells=sum(g['n_cells'] for g in groups if g['status']=='ambiguous'),
        group_summary=[{k:g[k] for k in ('group','n_cells','primary_annotation','status')} for g in groups[:50]],
        groups_omitted=max(0,len(groups)-50))


def _provenance(args, authority):
    return dict(contract=c.CONTRACT, profile_sha256=c.PROFILE_SHA256,
                matrix_manifest_path=args['matrix_manifest_path'], matrix_manifest_sha256=args['matrix_manifest_sha256'],
                matrix_authority_sha256=authority, annotation_spec_sha256=args['annotation_spec_sha256'])


def _build(args, output):
    spec = c.specification(args['annotation_spec_path'], args['annotation_spec_sha256'])
    authority, matrix, binding = source(args, spec)
    output.mkdir()
    science.annotate_cell_groups(**c.component_arguments(args, spec, binding, output/'primary'))
    cells = c.read_json(output/'primary/cells.json', limit=1024**3)
    groups = c.read_json(output/'primary/groups.json', limit=1024**3)
    a, _ = science.load_matrix(science.MatrixInput(**binding))
    if any(k in a.obs for k in c.OBS_COLUMNS) or 'cell_type_annotation' in a.uns:
        raise ValueError('Canonical input already contains annotation-owned metadata')
    if [r['cell_id'] for r in cells] != a.obs_names.tolist():
        raise ValueError('Annotation cell propagation identity mismatch')
    for column, key in zip(c.OBS_COLUMNS, ('group','primary_annotation','status'), strict=True):
        a.obs[column] = pd.Categorical([r[key] for r in cells])
    a.uns['cell_type_annotation'] = _provenance(args, authority)
    a.write_h5ad(output/'annotated.h5ad', convert_strings_to_categoricals=False)
    value = dict(artifact_type=c.ARTIFACT, schema_version=1, contract_version=c.CONTRACT,
        profile=c.PROFILE, profile_sha256=c.PROFILE_SHA256, arguments=args, inputs=spec,
        source_authority_sha256=authority, matrix_binding=binding,
        sidecars={str(p.relative_to(output)):dict(sha256=science.sha256(p),size_bytes=p.stat().st_size)
                  for p in sorted(output.rglob('*')) if p.is_file()}, summary=_counts(groups,matrix))
    path = output/'manifest.json'
    path.write_bytes(c.matrix.canonical(value))
    sha = science.sha256(path)
    verify_annotation(path, expected_sha256=sha)
    for name in (*value['sidecars'], 'manifest.json'):
        with (output/name).open('rb') as stream:
            os.fsync(stream.fileno())
    publication._fsync_dir(output/'primary'); publication._fsync_dir(output)
    return dict(manifest_path=str(path), manifest_sha256=sha)


def _same_sparse(left, right):
    return (left.shape == right.shape and left.dtype == right.dtype and
            all(np.array_equal(getattr(left,k),getattr(right,k)) for k in ('indptr','indices','data')))


def _verify_derivative(path, binding, cells, args, authority):
    original, _ = science.load_matrix(science.MatrixInput(**binding))
    derived = ad.read_h5ad(path)
    if (not sparse.isspmatrix_csr(derived.X) or not _same_sparse(original.X, derived.X)
            or not original.obs_names.equals(derived.obs_names) or not original.var_names.equals(derived.var_names)
            or derived.obs.columns.tolist() != [*original.obs.columns, *c.OBS_COLUMNS]):
        raise ValueError('Annotated H5AD changed canonical matrix values or axes')
    pd.testing.assert_frame_equal(original.obs, derived.obs[list(original.obs.columns)], check_categorical=False)
    pd.testing.assert_frame_equal(original.var, derived.var, check_categorical=False)
    for column, key in zip(c.OBS_COLUMNS, ('group','primary_annotation','status'), strict=True):
        actual = [None if pd.isna(x) else str(x) for x in derived.obs[column]]
        if actual != [r[key] for r in cells]:
            raise ValueError('Annotated H5AD metadata differs from exact cell propagation')
    if derived.uns.pop('cell_type_annotation', None) != _provenance(args, authority):
        raise ValueError('Annotated H5AD matrix/annotation lineage mismatch')
    np.testing.assert_equal(original.uns, derived.uns)
    # The canonical owner publishes no alternate matrices or embeddings.
    if (original.raw is not None or derived.raw is not None or
            any(len(getattr(a,key)) for a in (original,derived)
                for key in ('obsm','varm','obsp','varp','layers'))):
        raise ValueError('Unexpected alternate matrix slots in canonical annotation input/output')


@owned_verification('annotation')
def verify_annotation(path, *, expected_sha256):
    """Replay M13 science only; upstream matrix proof is imported, never rebuilt.

    Uses the qualified backend again with independently loaded immutable inputs,
    not a second implementation of annotation biology. Rehashed evidence must
    reproduce that backend and exact annotation/derivative propagation.
    """
    value = c.load_manifest(path, expected_sha256)
    args = value['arguments']; root = Path(path).parent
    spec = c.specification(args['annotation_spec_path'], args['annotation_spec_sha256'])
    authority, matrix, binding = source(args, spec)
    if value['inputs'] != spec or value['source_authority_sha256'] != authority or value['matrix_binding'] != binding:
        raise ValueError('Annotation source or scientific input binding mismatch')
    for name, entry in value['sidecars'].items():
        p = science.checked_file(root/name,entry['sha256'])
        if p != root/name or p.stat().st_size != entry['size_bytes']:
            raise ValueError('Annotation sidecar size or path mismatch')
    provenance = c.read_json(root/'primary/result.json')
    expected_sidecars = {name.removeprefix('primary/'): entry['sha256']
                        for name,entry in value['sidecars'].items()
                        if name.startswith('primary/') and name != 'primary/result.json'}
    if provenance.get('sidecars') != expected_sidecars:
        raise ValueError('Primary component sidecar identities differ from annotation publication')
    with tempfile.TemporaryDirectory(prefix='.annotation-verification-', dir=root.parent) as tmp:
        replay = Path(tmp)/'primary'
        expected_provenance = science.annotate_cell_groups(**c.component_arguments(args,spec,binding,replay))
        for key in ('sidecars',):
            provenance.pop(key); expected_provenance.pop(key)
        if provenance != expected_provenance:
            raise ValueError('Annotation backend provenance mismatch')
        for name in ('groups.json','cells.json','candidate_evidence.json','signature_coverage.json'):
            if c.read_json(root/'primary'/name,limit=1024**3) != c.read_json(replay/name,limit=1024**3):
                raise ValueError('Annotation evidence or primary assignments fail scientific replay')
        if not _same_sparse(sparse.load_npz(root/'primary/rp.npz'), sparse.load_npz(replay/'rp.npz')):
            raise ValueError('Annotation RP evidence fails pinned backend replay')
        for name in ('markers-native.tsv.gz','markers-maestro-filtered.tsv','markers-signature-input.tsv','group-sizes.tsv'):
            pd.testing.assert_frame_equal(pd.read_csv(root/'primary'/name,sep='\t',keep_default_na=False),
                                          pd.read_csv(replay/name,sep='\t',keep_default_na=False),check_exact=True)
        for name in ('genes.tsv','cells.tsv','groups.tsv'):
            if (root/'primary'/name).read_bytes() != (replay/name).read_bytes():
                raise ValueError('Annotation evidence axes differ from scientific inputs')
    cells = c.read_json(root/'primary/cells.json',limit=1024**3)
    groups = c.read_json(root/'primary/groups.json',limit=1024**3)
    if value['summary'] != _counts(groups,matrix):
        raise ValueError('Annotation summary mismatch')
    _verify_derivative(root/'annotated.h5ad',binding,cells,args,authority)
    cancellation_checkpoint()
    return _summary(value,Path(path),expected_sha256)


def verify_public_result(args, result):
    args = c.arguments(args)
    path = Path(result['manifest_path']); destination = path.parent.parent
    if (path != path.resolve() or path.name != 'manifest.json' or path.parent.name != 'artifact'
            or destination.parent != Path(args['output_dir'])):
        raise ValueError('Annotation publication locator mismatch')
    with _operation():
        receipt = _receipt(destination,args)
        if receipt['manifest_sha256'] != result['manifest_sha256']:
            raise ValueError('Annotation receipt artifact mismatch')
        value = c.load_manifest(path,result['manifest_sha256'])
        if value['arguments'] != args or dict(result) != _summary(value,path,result['manifest_sha256']):
            raise ValueError('Annotation public result mismatch')
        verify_annotation(path,expected_sha256=result['manifest_sha256'])
        return value


def execute_annotation(arguments, execution_identity=None):
    cancellation_checkpoint()
    args,destination,token = _publication(arguments,execution_identity)
    with _operation() as context:
        context.register_execution('annotation',args['output_dir'],execution_identity)
        return publication._execute_publication(args,destination,token,lambda output:_build(args,output),
            _summary,RECOVERY_POLICY,c.PROFILE_SHA256,load_manifest=c.load_manifest,receipt_factory=_make_receipt)


def recover_annotation(arguments, execution_identity):
    args,destination,token = _publication(arguments,execution_identity)
    if not destination.is_dir() or destination.is_symlink():
        raise ValueError('Annotation recovery publication unavailable')
    receipt = _receipt(destination,args,token)
    path = destination/'artifact/manifest.json'
    result = _summary(c.load_manifest(path,receipt['manifest_sha256']),path,receipt['manifest_sha256'])
    verify_public_result(args,result)
    return result


def annotate_scATAC_cell_types(matrix_manifest_path, matrix_manifest_sha256, annotation_spec_path,
                              annotation_spec_sha256, output_dir) -> CellTypeAnnotationResult:
    """Annotate exact canonical groups with explicit pinned tissue/context resources."""
    return execute_annotation(locals())
