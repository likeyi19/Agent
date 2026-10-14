"""Closed, bounded M13 annotation inputs and publication metadata."""
import json
from pathlib import Path
from . import marker_annotation as science
from agent.tools.data import scatac_matrix_contract as matrix

CONTRACT = 'scatac-cell-type-annotation.v1'
ARTIFACT = 'agent.scatac-cell-type-annotation'
SPEC = 'scatac-annotation-inputs.v1'
EXTERNAL_SPEC = 'scatac-annotation-peak-resources.v1'
EXTERNAL_PRIMARY = 'maestro-enhanced-count-weighted-peaks.v1'
EXTERNAL_SEMANTICS = 'cellranger-atac-cut-site-counts.v1'
RECOVERY_POLICY = 'annotate-scatac-cell-types-v1'
PROFILE = dict(primary_profile=science.PROFILE, upstream_revision=science.REVISION,
               matrix_semantics=matrix.PROFILE.profile_id,
               assignment='primary_unique_positive_assignment; validation_optional_non_gating',
               verification='pinned_backend_replay_and_exact_propagation.v1')
from agent.schemas.verification_authority import digest
PROFILE_SHA256 = digest(PROFILE)
EXTERNAL_PROFILE = dict(PROFILE, primary_profile=EXTERNAL_PRIMARY,
    matrix_semantics=EXTERNAL_SEMANTICS,
    source='external_peak_h5ad;counting_history_declared_not_verified',
    private_input='exact_float64_counts', rp='Enhanced', decay=10000,
    normalization='sparse_log1p_RP_over_cell_sum_times_10000',
    markers='native_Presto;raw_p_lt_0.01;abs_logFC_gt_0.25;rounded_detection_gt_0.1;adjusted_p_lt_1e-5',
    signature_scoring='signed_marker_sum_with_duplicate_signature_weighting',
    primary_policy='unique_positive_maximum;zero_unresolved;exact_positive_tie_ambiguous')
EXTERNAL_PROFILE_SHA256 = digest(EXTERNAL_PROFILE)
ARGUMENTS = ('matrix_manifest_path', 'matrix_manifest_sha256', 'annotation_spec_path',
             'annotation_spec_sha256', 'output_dir')
GROUP_ARGUMENTS = ('groups_path', 'groups_sha256')
DIRECT_ARGUMENTS = ('source_path', 'source_sha256', 'annotation_method', 'count_slot', 'group_column')
OBS_COLUMNS = ('annotation_group', 'primary_annotation', 'annotation_status')
MAX_BYTES = 65536


def read_json(path, expected=None, *, limit=MAX_BYTES):
    path = Path(path)
    matrix.absolute_path(str(path))
    if path != path.resolve():
        raise ValueError('Annotation paths must be canonical absolute paths')
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('Annotation metadata exceeds its declared bound')
    if expected is not None:
        matrix.sha(expected)
        import hashlib
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('Annotation content identity mismatch')
    from agent.tools.data.scatac_fragments_v2 import _pairs
    def invalid(_):
        raise ValueError('Nonfinite JSON value')
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=invalid)


def arguments(value):
    args = dict(value)
    direct = args.get('annotation_method') == EXTERNAL_PRIMARY
    if direct:
        expected = ('annotation_spec_path', 'annotation_spec_sha256', 'output_dir',
                    'source_path', 'source_sha256', 'annotation_method', 'count_slot')
        matrix.shape(args, (*expected, *(GROUP_ARGUMENTS if 'groups_path' in args else ('group_column',))))
    else:
        matrix.shape(args, ARGUMENTS)
    for key in args:
        if key.endswith('_sha256'):
            matrix.sha(args[key])
        elif key in ('annotation_method', 'count_slot', 'group_column'):
            science.identifier(args[key])
        else:
            args[key] = str(args[key]) if isinstance(args[key], Path) else args[key]
            matrix.absolute_path(args[key])
    return args


def specification(path, sha):
    value = read_json(path, sha)
    external = value.get('schema') == EXTERNAL_SPEC
    matrix.shape(value, ('schema', 'species', 'assembly', 'context', 'gene_resource', 'signature_resource',
        *(('grouping_provenance', 'method', 'measurement', 'coordinate_system', 'contig_scope')
          if external else ('source_run_id', 'source_step_id', 'groups'))))
    if value['schema'] not in (SPEC, EXTERNAL_SPEC) or (value['species'], value['assembly']) not in (('human', 'hg38'), ('mouse', 'mm10')):
        raise ValueError('Unsupported annotation input specification')
    for key in (('context', 'grouping_provenance') if external else ('source_run_id', 'source_step_id', 'context')):
        science.identifier(value[key])
    if not external:
        matrix.shape(value['groups'], ('path', 'sha256', 'provenance'))
        science.identifier(value['groups']['provenance'])
        matrix.absolute_path(value['groups']['path']); matrix.sha(value['groups']['sha256'])
        science.checked_file(value['groups']['path'], value['groups']['sha256'])
    else:
        if ((value['species'], value['assembly']) != ('human', 'hg38')
                or value['method'] != EXTERNAL_PRIMARY or value['measurement'] != EXTERNAL_SEMANTICS
                or value['coordinate_system'] != 'zero-based-half-open'):
            raise ValueError('Only declared human hg38 Cell Ranger ATAC cut-site counts are qualified')
        matrix.shape(value['contig_scope'], ('path', 'sha256'))
        matrix.absolute_path(value['contig_scope']['path']); matrix.sha(value['contig_scope']['sha256'])
        science.checked_file(value['contig_scope']['path'], value['contig_scope']['sha256'])
    for name, semantics, normalization in (
        ('gene_resource', 'maestro-refgenes-transcripts-exons.v1', 'none'),
        ('signature_resource', 'candidate-gene-list.v1', 'uppercase')):
        matrix.shape(value[name], science.Resource.__dataclass_fields__)
        resource = science.Resource(**value[name])
        matrix.absolute_path(resource.path)
        resource.check(value['species'], value['assembly'], semantics, normalization)
    if (value['signature_resource']['context'] != value['context'] or
            value['gene_resource']['gene_namespace'] != value['signature_resource']['gene_namespace']):
        raise ValueError('Explicit signature context and gene namespaces must match')
    return value


def resolved_specification(args):
    """Bind a reusable reviewed resource choice to an explicitly supplied TSV."""
    value = specification(args['annotation_spec_path'], args['annotation_spec_sha256'])
    external = value['schema'] == EXTERNAL_SPEC
    if external != (args.get('annotation_method') == EXTERNAL_PRIMARY):
        raise ValueError('Explicit annotation method and resource specification differ')
    if external:
        if 'groups_path' in args:
            science.checked_file(args['groups_path'], args['groups_sha256'])
            groups = dict(path=args['groups_path'], sha256=args['groups_sha256'])
        else:
            groups = dict(obs_column=args['group_column'])
        value = dict(value, groups=dict(groups, provenance=value['grouping_provenance']))
    return value


def profile(args):
    method = args.get('annotation_method', science.PROFILE)
    if method not in (science.PROFILE, EXTERNAL_PRIMARY):
        raise ValueError('Unknown annotation method')
    return EXTERNAL_PROFILE if method == EXTERNAL_PRIMARY else PROFILE


def profile_sha(args):
    return digest(profile(args))


def backend(binding):
    if binding.get('method') == EXTERNAL_PRIMARY:
        from . import external_marker_annotation
        return external_marker_annotation
    return science


def load_manifest(path, sha=None):
    value = read_json(path, sha)
    matrix.shape(value, ('artifact_type', 'schema_version', 'contract_version', 'profile',
                        'profile_sha256', 'arguments', 'inputs', 'source_authority_sha256',
                        'matrix_binding', 'sidecars', 'summary'))
    if (value['artifact_type'] != ARTIFACT or type(value['schema_version']) is not int or
        value['schema_version'] != 1 or value['contract_version'] != CONTRACT or
        value['profile'] != profile(value['arguments']) or value['profile_sha256'] != profile_sha(value['arguments'])):
        raise ValueError('Unsupported annotation publication contract')
    arguments(value['arguments']); matrix.sha(value['source_authority_sha256'])
    if value['profile'] == PROFILE:
        matrix.shape(value['matrix_binding'], science.MatrixInput.__dataclass_fields__)
    else:
        from .external_marker_annotation import BINDING_FIELDS
        matrix.shape(value['matrix_binding'], BINDING_FIELDS)
    expected = {'annotated.h5ad', 'primary/result.json', 'primary/rp.npz', 'primary/genes.tsv',
                'primary/cells.tsv', 'primary/groups.tsv', 'primary/marker-runtime.txt',
                'primary/markers-native.tsv.gz', 'primary/markers-maestro-filtered.tsv',
                'primary/markers-signature-input.tsv', 'primary/group-sizes.tsv',
                *(f'primary/{key}.json' for key in ('groups','cells','candidate_evidence','signature_coverage'))}
    matrix.shape(value['sidecars'], expected)
    for entry in value['sidecars'].values():
        matrix.shape(entry, ('sha256', 'size_bytes'))
        matrix.sha(entry['sha256']); matrix.integer(entry['size_bytes'])
    return value


def backend_runtime():
    import os
    names = ('AGENT_ANNOTATION_MAESTRO_ROOT', 'AGENT_ANNOTATION_RSCRIPT', 'AGENT_ANNOTATION_R_LIBRARY')
    if any(not os.environ.get(k) for k in names):
        raise ValueError('Pinned annotation operator runtime is not configured')
    return science.Runtime(*(str(Path(os.environ[k]).resolve()) for k in names))


def component_arguments(args, spec, binding, output):
    if spec['schema'] == EXTERNAL_SPEC:
        return dict(binding=binding, specification=spec, runtime=backend_runtime(), output_dir=str(output))
    return dict(matrix=science.MatrixInput(**binding), groups_path=spec['groups']['path'],
                groups_sha256=spec['groups']['sha256'], grouping_provenance=spec['groups']['provenance'],
                gene_resource=science.Resource(**spec['gene_resource']),
                signature_resource=science.Resource(**spec['signature_resource']),
                runtime=backend_runtime(), profile=profile(args)['primary_profile'], output_dir=str(output))


def load_source(binding):
    if binding.get('method') == EXTERNAL_PRIMARY:
        return backend(binding).load_matrix(binding)
    return science.load_matrix(science.MatrixInput(**binding))
