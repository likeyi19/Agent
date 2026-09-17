"""Closed external variant of the matrix family; no fragment-derived claims."""
import hashlib
import json
import sys
from . import scatac_matrix_contract as m

ARTIFACT = m.ARTIFACT
REFERENCE_CONTRACT = 'scatac-reference-bundle.v1'

CONTRACT = 'scatac-cell-by-ccre.external.v1'
RECOVERY_POLICY = 'adopt-scatac-cell-by-ccre-v1'
PROFILE = 'exact-external-cell-by-ccre-adoption.v1'
PROFILE_SPEC = dict(profile_id=PROFILE, operation='external_adoption',
    input='H5AD-X-canonical-CSR-signed-int64;no-explicit-zeros;no-duplicate-entries',
    values='externally-declared;positive-stored-integers;checked-entry-row-total-int64',
    cells='unique-nonempty-UTF8;exact-source-order;no-namespace-or-selection',
    features='exact-complete-reference-names-coordinates-order;no-repair',
    transformation='int64-index-storage-normalization;axis-only-H5AD;no-value-change',
    verification='independent-source-output-entrywise-conservation',
    history='external-declaration;fragment-QC-selection-history-not-established')
PROFILE_SHA256 = hashlib.sha256(m.canonical(PROFILE_SPEC)).hexdigest()
SEMANTICS = ('fragment_counts', 'insertion_counts', 'binary_accessibility')
FIELDS = ('artifact_type', 'schema_version', 'contract_version', 'operation', 'profile',
          'profile_sha256', 'species', 'assembly', 'source', 'reference', 'matrix_semantics',
          'shape', 'nnz', 'total_count', 'zero_row_count', 'ordered_cells_sha256',
          'ordered_feature_sha256', 'logical_matrix_sha256', 'source_logical_matrix_sha256',
          'matrix', 'readiness', 'identity_sha256')


def identity(value):
    return hashlib.sha256(b'agent.external-cell-by-ccre.v1\0' + m.canonical(
        {k: v for k, v in value.items() if k != 'identity_sha256'})).hexdigest()


def logical_identity(n, p, rows, semantics):
    if semantics not in SEMANTICS: m.fail('MATRIX_SEMANTICS_INVALID')
    m.integer(n); m.integer(p, 1)
    digest = hashlib.sha256(b'agent.external-regulatory-sparse-matrix.v1\0')
    digest.update(m.canonical(dict(shape=[n, p], dtype='int64', semantics=semantics)))
    digest.update(b'\n')
    nnz = total = zeros = count = 0
    for columns, values in rows:
        count += 1
        if count > n or len(columns) != len(values): m.fail('MATRIX_CSR_INVALID')
        digest.update(len(columns).to_bytes(8, 'big'))
        previous = -1
        for col in columns:
            m.integer(col, 0, p - 1)
            if col <= previous: m.fail('MATRIX_CSR_INVALID')
            previous = col; digest.update(col.to_bytes(8, 'big'))
        row_sum = 0
        for value in values:
            m.integer(value, 1, 1 if semantics == 'binary_accessibility' else m.INT64_MAX)
            row_sum = m.checked_add(row_sum, value)
            digest.update(value.to_bytes(8, 'big', signed=True))
        nnz = m.checked_add(nnz, len(columns)); total = m.checked_add(total, row_sum)
        zeros += not columns
    if count != n: m.fail('MATRIX_CSR_INVALID')
    return dict(logical_matrix_sha256=digest.hexdigest(), nnz=nnz,
                total_count=total, zero_row_count=zeros)


def validate(value, *, _profile=None):
    c = _profile or sys.modules[__name__]
    m.shape(value, FIELDS)
    if c is sys.modules[__name__]:
        if (value['species'], value['assembly']) not in (('human', 'hg38'), ('mouse', 'mm10')): m.fail()
    else:
        c.validate_binding(value['species'], value['assembly'])
    if (value['artifact_type'] != c.ARTIFACT or type(value['schema_version']) is not int
            or value['schema_version'] != 1 or value['contract_version'] != c.CONTRACT
            or value['operation'] != 'external_adoption' or value['profile'] != c.PROFILE
            or value['profile_sha256'] != c.PROFILE_SHA256 or value['matrix_semantics'] not in SEMANTICS): m.fail()
    m.shape(value['source'], ('path', 'sha256', 'size_bytes'))
    m.absolute_path(value['source']['path']); m.sha(value['source']['sha256']); m.integer(value['source']['size_bytes'], 1)
    r = value['reference']
    m.shape(r, ('manifest_path', 'manifest_sha256', 'identity_sha256', 'contract_version'))
    m.absolute_path(r['manifest_path']); m.sha(r['manifest_sha256']); m.sha(r['identity_sha256'])
    if r['contract_version'] != c.REFERENCE_CONTRACT: m.fail()
    for k in ('ordered_cells_sha256', 'ordered_feature_sha256', 'logical_matrix_sha256',
              'source_logical_matrix_sha256', 'identity_sha256'): m.sha(value[k])
    dims = value['shape']
    if type(dims) is not list or len(dims) != 2: m.fail()
    n, p = m.integer(dims[0]), m.integer(dims[1], 1)
    nz, total, zeros = m.integer(value['nnz']), m.integer(value['total_count']), m.integer(value['zero_row_count'], 0, n)
    if (not n-zeros <= nz <= (n-zeros)*p or total < nz or (nz == 0) != (total == 0)
            or value['matrix_semantics'] == 'binary_accessibility' and total != nz): m.fail()
    m.shape(value['matrix'], ('path', 'sha256', 'size_bytes', 'format', 'dtype'))
    if tuple(value['matrix'][k] for k in ('path', 'format', 'dtype')) != ('matrix.h5ad', 'csr', 'int64'): m.fail()
    m.sha(value['matrix']['sha256']); m.integer(value['matrix']['size_bytes'], 1)
    if (value['readiness'] != ('matrix_available' if n else 'no_source_cells')
            or value['source_logical_matrix_sha256'] != value['logical_matrix_sha256']
            or value['identity_sha256'] != c.identity(value)): m.fail('MATRIX_IDENTITY_MISMATCH')
    return value


def load(raw, *, _profile=None):
    if len(raw) > m.MAX_MANIFEST_BYTES: m.fail()
    from .scatac_fragments_v2 import _pairs
    try:
        value = json.loads(raw.decode('utf-8'), object_pairs_hook=_pairs,
                           parse_constant=lambda _: m.fail('MATRIX_JSON_INVALID'))
        validate(value, _profile=_profile)
        if m.canonical(value) != raw: m.fail('MATRIX_JSON_INVALID')
        return value
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise m.ScATACMatrixError('MATRIX_CONTRACT_INVALID') from exc


def validate_binding(species, assembly):
    if (species, assembly) not in (('human', 'hg38'), ('mouse', 'mm10')):
        m.fail('MATRIX_SEMANTICS_INVALID')


from .scatac_reference import (load_scatac_reference_bundle as load_reference,
                              reinspect_scatac_reference_bundle_sources as reinspect_reference)


def contract_for(value):
    """Closed matrix-owner dispatch; never infer a profile from species or shape."""
    if value['contract_version'] == CONTRACT:
        return sys.modules[__name__]
    from . import regulatory_matrix_contract as neutral
    if value['contract_version'] == neutral.CONTRACT:
        return neutral
    m.fail('MATRIX_CONTRACT_INVALID')


def is_external(value):
    return value.get('contract_version') in (CONTRACT, 'scatac-cell-by-features.external.v1')
