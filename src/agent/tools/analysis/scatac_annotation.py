"""Public primary annotation: existing science, authority DAG and publication envelope."""
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
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


def _canonical_source(args, spec):
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


def _publication(args, execution_identity, *, version=2):
    args = c.arguments(args)
    token = c.digest(dict(arguments=args, policy=RECOVERY_POLICY, execution_identity=execution_identity,
                          profile_sha256=c.profile_sha(args, version)))
    return args, Path(args['output_dir']) / ('annotation-' + token), token


def _make_receipt(args, token, sha, *, version=2):
    return dict(artifact_type='agent.annotation-receipt', schema_version=1, policy=RECOVERY_POLICY,
                execution_identity=token, arguments_sha256=c.digest(args), manifest_sha256=sha,
                profile_sha256=c.profile_sha(args, version))


def _receipt(destination, args, token=None):
    r = c.read_json(destination / 'receipt.json', limit=16384)
    version = c.load_manifest(destination/'artifact/manifest.json', r['manifest_sha256'])['schema_version']
    if (r != _make_receipt(args, r['execution_identity'], r['manifest_sha256'], version=version) or
            destination.name != 'annotation-' + r['execution_identity'] or
            token is not None and token != r['execution_identity']):
        raise ValueError('Annotation recovery receipt mismatch')
    c.matrix.sha(r['execution_identity']); c.matrix.sha(r['manifest_sha256'])
    return r


def _summary(value, path, sha):
    binding = value['matrix_binding']
    return dict(status='success', manifest_path=str(path), manifest_sha256=sha,
        artifact_type=c.ARTIFACT, artifact_schema_version=value['schema_version'], contract_version=c.CONTRACT,
        annotated_h5ad_path=str(path.parent / 'annotated.h5ad'),
        annotated_h5ad_sha256=value['sidecars']['annotated.h5ad']['sha256'],
        source_manifest_sha256=binding['manifest_sha256'] if 'manifest_sha256' in binding else binding['source_sha256'],
        source_authority_sha256=value['source_authority_sha256'],
        annotation_spec_sha256=value['arguments']['annotation_spec_sha256'],
        gene_resource_sha256=value['inputs']['gene_resource']['sha256'],
        signature_resource_sha256=value['inputs']['signature_resource']['sha256'],
        ordered_cells_sha256=binding['ordered_cells_sha256'], ordered_feature_sha256=binding['ordered_features_sha256'],
        annotation_profile=value['profile']['primary_profile'], species=binding['species'], assembly=binding['assembly'],
        context=value['inputs']['context'], validation_state='not_assessed', **value['summary'])


def _counts(groups, matrix):
    assigned = sum(g['n_cells'] for g in groups if g['status'] == 'assigned')
    return dict(logical_matrix_sha256=matrix['logical_matrix_sha256'], n_cells=matrix['shape'][0],
        n_groups=len(groups), assigned_cells=assigned, unassigned_cells=matrix['shape'][0]-assigned,
        ambiguous_cells=sum(g['n_cells'] for g in groups if g['status']=='ambiguous'),
        group_summary=[{k:g[k] for k in ('group','n_cells','primary_annotation','status')} for g in groups[:50]],
        groups_omitted=max(0,len(groups)-50))


def _provenance(args, authority, participation=None, *, version=2):
    if args.get('annotation_method') == c.EXTERNAL_PRIMARY:
        return dict(contract=c.CONTRACT, profile_sha256=c.profile_sha(args, version), source_path=args['source_path'],
                    source_sha256=args['source_sha256'], input_qualification_sha256=authority,
                    source_contract='external-peak-annotation-input.v1',
                    counting_history='declared_not_reconstructed', count_slot=args['count_slot'],
                    annotation_spec_sha256=args['annotation_spec_sha256'],
                    **({'rp_' + k: v for k, v in participation.items()} if participation is not None else {}))
    return dict(contract=c.CONTRACT, profile_sha256=c.profile_sha(args, version),
                matrix_manifest_path=args['matrix_manifest_path'], matrix_manifest_sha256=args['matrix_manifest_sha256'],
                matrix_authority_sha256=authority, annotation_spec_sha256=args['annotation_spec_sha256'])


def _build(args, output):
    spec = c.resolved_specification(args)
    if spec['schema'] == c.EXTERNAL_SPEC:
        from .external_marker_annotation import prepare
        a, (authority, matrix, binding) = prepare(args, spec)
        component_input = dict(_matrix=a)
    else:
        authority, matrix, binding = _canonical_source(args, spec)
        a, _ = c.load_source(binding)
        component_input = {}
    output.mkdir()
    primary = c.backend(binding).annotate_cell_groups(
        **c.component_arguments(args, spec, binding, output/'primary'), **component_input)
    participation = primary.get('feature_participation')
    cells = c.read_json(output/'primary/cells.json', limit=1024**3)
    groups = c.read_json(output/'primary/groups.json', limit=1024**3)
    if any(k in a.obs for k in c.OBS_COLUMNS) or 'cell_type_annotation' in a.uns:
        raise ValueError('Canonical input already contains annotation-owned metadata')
    if [r['cell_id'] for r in cells] != a.obs_names.tolist():
        raise ValueError('Annotation cell propagation identity mismatch')
    for column, key in zip(c.OBS_COLUMNS, ('group','primary_annotation','status'), strict=True):
        a.obs[column] = pd.Categorical([r[key] for r in cells])
    a.uns['cell_type_annotation'] = _provenance(args, authority, participation)
    a.write_h5ad(output/'annotated.h5ad', convert_strings_to_categoricals=False)
    value = dict(artifact_type=c.ARTIFACT, schema_version=2, contract_version=c.CONTRACT,
        profile=c.profile(args), profile_sha256=c.profile_sha(args), arguments=args, inputs=spec,
        source_authority_sha256=authority, matrix_binding=binding,
        sidecars={str(p.relative_to(output)):dict(sha256=science.sha256(p),size_bytes=p.stat().st_size)
                  for p in sorted(output.rglob('*')) if p.is_file()}, summary=_counts(groups,matrix))
    if participation is not None:
        value['feature_participation'] = participation
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
    return (left.format == right.format and left.shape == right.shape and left.dtype == right.dtype and
            all(np.array_equal(getattr(left,k),getattr(right,k)) for k in ('indptr','indices','data')))


def _verify_derivative(path, binding, cells, args, authority, participation=None, *, version=2, original=None):
    if original is None:
        original, _ = c.load_source(binding)
    derived = ad.read_h5ad(path)
    direct = binding.get('method') == c.EXTERNAL_PRIMARY
    if (not _same_array(original.X, derived.X)
            or not original.obs_names.equals(derived.obs_names) or not original.var_names.equals(derived.var_names)
            or derived.obs.columns.tolist() != [*original.obs.columns, *c.OBS_COLUMNS]):
        raise ValueError('Annotated H5AD changed canonical matrix values or axes')
    pd.testing.assert_frame_equal(original.obs, derived.obs[list(original.obs.columns)], check_categorical=False)
    pd.testing.assert_frame_equal(original.var, derived.var, check_categorical=False)
    for column, key in zip(c.OBS_COLUMNS, ('group','primary_annotation','status'), strict=True):
        actual = [None if pd.isna(x) else str(x) for x in derived.obs[column]]
        if actual != [r[key] for r in cells]:
            raise ValueError('Annotated H5AD metadata differs from exact cell propagation')
    if derived.uns.pop('cell_type_annotation', None) != _provenance(args, authority, participation, version=version):
        raise ValueError('Annotated H5AD matrix/annotation lineage mismatch')
    np.testing.assert_equal(original.uns, derived.uns)
    if direct:
        for key in ('obsm','varm','obsp','varp','layers'):
            left,right=getattr(original,key),getattr(derived,key)
            if set(left)!=set(right) or any(not _same_array(left[k],right[k]) for k in left):
                raise ValueError('Annotated H5AD changed an ancillary scientific slot')
        if (original.raw is None) != (derived.raw is None):
            raise ValueError('Annotated H5AD changed raw storage')
        if original.raw is not None:
            if not _same_array(original.raw.X,derived.raw.X): raise ValueError('Changed raw values')
            pd.testing.assert_frame_equal(original.raw.var,derived.raw.var)
            if set(original.raw.varm)!=set(derived.raw.varm) or any(not _same_array(original.raw.varm[k],derived.raw.varm[k]) for k in original.raw.varm):
                raise ValueError('Changed raw feature metadata')
        return
    # The canonical owner publishes no alternate matrices or embeddings.
    if (original.raw is not None or derived.raw is not None or
            any(len(getattr(a,key)) for a in (original,derived)
                for key in ('obsm','varm','obsp','varp','layers'))):
        raise ValueError('Unexpected alternate matrix slots in canonical annotation input/output')


def _same_array(left,right):
    if sparse.issparse(left) or sparse.issparse(right):
        return sparse.issparse(left) and sparse.issparse(right) and _same_sparse(left,right)
    if isinstance(left,pd.DataFrame) or isinstance(right,pd.DataFrame):
        return isinstance(left,pd.DataFrame) and isinstance(right,pd.DataFrame) and left.equals(right)
    return (left.shape == right.shape and left.dtype == right.dtype
            and np.array_equal(left, right, equal_nan=left.dtype.kind in 'fc'))


def _verify_completion(root, binding, spec, cells, groups, original=None):
    """Native result structure and exact label propagation, without biological replay."""
    from collections import Counter
    ids = (root/'primary/cells.tsv').read_text().splitlines()
    genes = (root/'primary/genes.tsv').read_text().splitlines()
    if science.ordered_identity_sha256(ids) != binding['ordered_cells_sha256']:
        raise ValueError('Native cell identity differs from the input')
    science.ordered_identity_sha256(genes)
    rows = list(pd.read_csv(root/'primary/groups.tsv', sep='\t', dtype=str,
                           keep_default_na=False).itertuples(index=False, name=None))
    order = science.exact_groups(ids, rows)
    if 'path' in spec['groups']:
        expected = list(pd.read_csv(spec['groups']['path'], sep='\t', dtype=str,
                                    keep_default_na=False).itertuples(index=False, name=None))
    else:
        from .external_marker_annotation import group_rows
        expected = group_rows(original, spec)
    if rows != expected or [g['group'] for g in groups] != order:
        raise ValueError('Native groups differ from the selected exact input')
    sizes = Counter(g for _, g in rows)
    for group in groups:
        if (group['n_cells'] != sizes[group['group']] or
                group['status'] not in ('assigned', 'unresolved', 'insufficient_evidence', 'ambiguous') or
                group['validation_state'] != 'not_assessed'):
            raise ValueError('Invalid native group completion')
        if group['status'] == 'assigned':
            science.identifier(group['primary_annotation'])
        elif group['primary_annotation'] is not None:
            raise ValueError('Unresolved native group has a primary annotation')
    lookup = {g['group']: g for g in groups}
    expected_cells = [dict(cell_id=cell, group=group, **{
        key: lookup[group][key] for key in ('candidate', 'status', 'primary_annotation')})
        for cell, group in rows]
    if cells != expected_cells:
        raise ValueError('Native cell assignments differ from group propagation')
    rp = sparse.load_npz(root/'primary/rp.npz')
    if (not sparse.isspmatrix_csr(rp) or rp.shape != (len(ids), len(genes)) or
            not np.isfinite(rp.data).all() or (rp.data < 0).any() or
            (np.asarray(rp.sum(axis=1)).ravel() <= 0).any()):
        raise ValueError('Invalid native RP result structure')
    rp.check_format(full_check=True)


@owned_verification('annotation')
def verify_annotation(path, *, expected_sha256):
    """Check native completion, exact propagation and pinned publication integrity.

    This does not rerun RP, marker testing or candidate scoring. Historical v1
    replay authority can only be reused from its accepted RunStore record.
    """
    value = c.load_manifest(path, expected_sha256)
    if value['schema_version'] == 1:
        raise ValueError('Historical replay authority is required; standalone legacy requalification is retired')
    args = value['arguments']; root = Path(path).parent
    spec = c.resolved_specification(args)
    original = None
    if spec['schema'] == c.EXTERNAL_SPEC:
        from .external_marker_annotation import prepare
        original, (authority, matrix, binding) = prepare(args, spec)
    else:
        authority, matrix, binding = _canonical_source(args, spec)
    if value['inputs'] != spec or value['source_authority_sha256'] != authority or value['matrix_binding'] != binding:
        raise ValueError('Annotation source or scientific input binding mismatch')
    for name, entry in value['sidecars'].items():
        p = science.checked_file(root/name,entry['sha256'])
        if p != root/name or p.stat().st_size != entry['size_bytes']:
            raise ValueError('Annotation sidecar size or path mismatch')
    provenance = c.read_json(root/'primary/result.json')
    if spec['schema'] == c.EXTERNAL_SPEC:
        if provenance.get('binding') != binding or provenance.get('inputs') != spec:
            raise ValueError('Native completion has a different source or resource binding')
    elif (provenance.get('matrix') != binding or any(
            provenance.get(key) != spec[key] for key in ('gene_resource', 'signature_resource'))):
        raise ValueError('Native completion has a different canonical source or resource binding')
    if value.get('feature_participation') != provenance.get('feature_participation'):
        raise ValueError('Native feature participation metadata differs from the backend')
    expected_sidecars = {name.removeprefix('primary/'): entry['sha256']
                        for name,entry in value['sidecars'].items()
                        if name.startswith('primary/') and name != 'primary/result.json'}
    if provenance.get('sidecars') != expected_sidecars:
        raise ValueError('Primary component sidecar identities differ from annotation publication')
    cells = c.read_json(root/'primary/cells.json',limit=1024**3)
    groups = c.read_json(root/'primary/groups.json',limit=1024**3)
    _verify_completion(root, binding, spec, cells, groups, original)
    if value['summary'] != _counts(groups,matrix):
        raise ValueError('Annotation summary mismatch')
    _verify_derivative(root/'annotated.h5ad',binding,cells,args,authority,value.get('feature_participation'),
                       version=value['schema_version'], original=original)
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
            _summary,RECOVERY_POLICY,c.profile_sha(args),load_manifest=c.load_manifest,receipt_factory=_make_receipt)


def recover_annotation(arguments, execution_identity):
    args,destination,token = _publication(arguments,execution_identity)
    if not destination.exists():
        args,destination,token = _publication(arguments,execution_identity, version=1)
    if not destination.is_dir() or destination.is_symlink():
        raise ValueError('Annotation recovery publication unavailable')
    receipt = _receipt(destination,args,token)
    path = destination/'artifact/manifest.json'
    result = _summary(c.load_manifest(path,receipt['manifest_sha256']),path,receipt['manifest_sha256'])
    verify_public_result(args,result)
    return result


def annotate_scATAC_cell_types(annotation_spec_path, annotation_spec_sha256, output_dir,
                              matrix_manifest_path=None, matrix_manifest_sha256=None,
                              groups_path=None, groups_sha256=None, source_path=None, source_sha256=None,
                              annotation_method=None, count_slot=None, group_column=None) -> CellTypeAnnotationResult:
    """Annotate exact groups on canonical or declared peak counts with pinned resources."""
    return execute_annotation({k: v for k, v in locals().items() if v is not None})
