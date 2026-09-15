"""Closed, bounded M13 annotation inputs and publication metadata."""
import json
from pathlib import Path
from . import marker_annotation as science
from agent.tools.data import scatac_matrix_contract as matrix

CONTRACT = 'scatac-cell-type-annotation.v1'
ARTIFACT = 'agent.scatac-cell-type-annotation'
SPEC = 'scatac-annotation-inputs.v1'
RECOVERY_POLICY = 'annotate-scatac-cell-types-v1'
PROFILE = dict(primary_profile=science.PROFILE, upstream_revision=science.REVISION,
               matrix_semantics=matrix.PROFILE.profile_id,
               assignment='primary_unique_positive_assignment; validation_optional_non_gating',
               verification='pinned_backend_replay_and_exact_propagation.v1')
from agent.schemas.verification_authority import digest
PROFILE_SHA256 = digest(PROFILE)
ARGUMENTS = ('matrix_manifest_path', 'matrix_manifest_sha256', 'annotation_spec_path',
             'annotation_spec_sha256', 'output_dir')
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
    matrix.shape(args, ARGUMENTS)
    for key in args:
        if key.endswith('_sha256'):
            matrix.sha(args[key])
        else:
            args[key] = str(args[key]) if isinstance(args[key], Path) else args[key]
            matrix.absolute_path(args[key])
    return args


def specification(path, sha):
    value = read_json(path, sha)
    matrix.shape(value, ('schema', 'source_run_id', 'source_step_id', 'species', 'assembly',
                         'context', 'groups', 'gene_resource', 'signature_resource'))
    if value['schema'] != SPEC or (value['species'], value['assembly']) not in (('human', 'hg38'), ('mouse', 'mm10')):
        raise ValueError('Unsupported annotation input specification')
    for key in ('source_run_id', 'source_step_id', 'context'):
        science.identifier(value[key])
    matrix.shape(value['groups'], ('path', 'sha256', 'provenance'))
    science.identifier(value['groups']['provenance'])
    matrix.absolute_path(value['groups']['path']); matrix.sha(value['groups']['sha256'])
    science.checked_file(value['groups']['path'], value['groups']['sha256'])
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


def load_manifest(path, sha=None):
    value = read_json(path, sha)
    matrix.shape(value, ('artifact_type', 'schema_version', 'contract_version', 'profile',
                        'profile_sha256', 'arguments', 'inputs', 'source_authority_sha256',
                        'matrix_binding', 'sidecars', 'summary'))
    if (value['artifact_type'] != ARTIFACT or type(value['schema_version']) is not int or
        value['schema_version'] != 1 or value['contract_version'] != CONTRACT or
        value['profile'] != PROFILE or value['profile_sha256'] != PROFILE_SHA256):
        raise ValueError('Unsupported annotation publication contract')
    arguments(value['arguments']); matrix.sha(value['source_authority_sha256'])
    matrix.shape(value['matrix_binding'], science.MatrixInput.__dataclass_fields__)
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
    return dict(matrix=science.MatrixInput(**binding), groups_path=spec['groups']['path'],
                groups_sha256=spec['groups']['sha256'], grouping_provenance=spec['groups']['provenance'],
                gene_resource=science.Resource(**spec['gene_resource']),
                signature_resource=science.Resource(**spec['signature_resource']),
                runtime=backend_runtime(), profile=science.PROFILE, output_dir=str(output))
