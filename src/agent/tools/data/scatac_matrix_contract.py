"""M11.5a closed science/artifact contracts; no matrix builder or publication.

Structural validity is not scientific verification. M11.5b must reconstruct
complete counts from freshly verified sources using an independent algorithm.
"""
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import PurePosixPath
import re

INT64_MAX = 2**63 - 1
ARTIFACT = 'agent.scatac-cell-by-ccre'
CONTRACT = 'scatac-cell-by-ccre.v1'
MATRIX_DOMAIN = b'agent.regulatory-sparse-matrix.v1\0'
MAX_MANIFEST_BYTES = 65536


class ScATACMatrixError(ValueError):
    def __init__(self, code='MATRIX_CONTRACT_INVALID'):
        self.code = code
        super().__init__(code)


def fail(code='MATRIX_CONTRACT_INVALID'):
    raise ScATACMatrixError(code)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


@dataclass(frozen=True)
class MatrixScienceProfile:
    profile_id: str = 'canonical-fragment-record-overlap-counts.v1'
    matrix_semantics: str = 'fragment_counts'
    unit: str = 'canonical-scatac-fragments.v2-record'
    support: str = 'ignored;no-expansion'
    strand: str = 'ignored-for-overlap-and-value;record-multiplicity-preserved'
    coordinates: str = 'zero-based-half-open;no-shift;no-projection'
    overlap: str = 'same-contig;max(fragment_start,ccre_start)<min(fragment_end,ccre_end)'
    contribution: str = 'one-per-record-per-distinct-ccre;no-fractional-weighting'
    accumulation: str = 'checked-integer-sum;not-binary;no-new-deduplication'
    duplicate_incidence: str = 'reject-repeated-record-identity-and-column-index'
    record_identity: str = 'namespace-and-zero-based-ordinal-in-complete-canonical-library-stream'
    rows: str = 'selected.tsv.gz-matrix_row_index;contiguous;never-resort'
    columns: str = 'exact-reference-bed-row-index;complete-vocabulary;never-filter'
    zero_rows: str = 'preserve;absent-selected-fragment-identity-is-error'
    zero_columns: str = 'preserve'
    empty_selection: str = 'valid-shape-(0,n_full_ccre);downstream-readiness-separate'
    values: str = 'nonnegative-signed-int64;checked-entry-row-and-total-sums'
    storage: str = 'csr;sorted-unique-columns;no-explicit-zeros;index-dtype-not-scientific'


PROFILE = MatrixScienceProfile()
PROFILE_SHA256 = hashlib.sha256(canonical(asdict(PROFILE))).hexdigest()


def shape(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        fail()


def integer(value, low=0, high=INT64_MAX):
    if type(value) is not int or not low <= value <= high:
        fail('MATRIX_INTEGER_INVALID')
    return value


def checked_add(a, b):
    return integer(integer(a) + integer(b))


def sha(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        fail('MATRIX_DIGEST_INVALID')


def absolute_path(value):
    if (type(value) is not str or not value.startswith('/') or value.startswith('//')
            or str(PurePosixPath(value)) != value or '..' in PurePosixPath(value).parts
            or any(ord(c) < 32 for c in value)):
        fail('MATRIX_PATH_INVALID')


def interval(chrom, start, end):
    if (type(chrom) is not str or not chrom or ':' in chrom
            or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in chrom)):
        fail('MATRIX_INTERVAL_INVALID')
    integer(start); integer(end, 1)
    if start >= end:
        fail('MATRIX_INTERVAL_INVALID')
    return chrom, start, end


def overlaps(a, b):
    ac, start, end = interval(*a)
    bc, left, right = interval(*b)
    return ac == bc and max(start, left) < min(end, right)


def overlap_diagnostic(total, overlapping):
    integer(total); integer(overlapping)
    if overlapping > total:
        fail('MATRIX_DIAGNOSTIC_INVALID')
    ratio = Fraction(overlapping, total) if total else None
    return dict(n_selected_fragment_records_total=total,
                n_selected_fragment_records_overlapping_any_ccre=overlapping,
                fraction_selected_fragment_records_overlapping_any_ccre=(
                    dict(numerator=ratio.numerator, denominator=ratio.denominator)
                    if ratio is not None else None),
                undefined_reason=None if total else 'NO_SELECTED_FRAGMENT_RECORDS')


def logical_matrix_identity(n_rows, n_cols, rows):
    """Stream canonical (columns, values) rows using the unchanged M8 encoding.

    Python integer conversion must occur at the storage reader boundary, after
    checking physical dtype/CSR validity. Never coalesce duplicate sparse entries.
    This validates counts/identity, not fragment-to-cCRE scientific correctness.
    """
    integer(n_rows); integer(n_cols, 1)
    digest = hashlib.sha256(MATRIX_DOMAIN)
    digest.update(canonical(dict(shape=[n_rows, n_cols], dtype='int64', semantics='fragment_counts')))
    digest.update(b'\n')
    nnz = total = zero_rows = observed = 0
    for columns, values in rows:
        observed += 1
        if observed > n_rows or len(columns) != len(values):
            fail('MATRIX_CSR_INVALID')
        digest.update(len(columns).to_bytes(8, 'big'))
        previous = -1
        for column in columns:
            integer(column, 0, n_cols - 1)
            if column <= previous:
                fail('MATRIX_CSR_INVALID')
            previous = column
            digest.update(column.to_bytes(8, 'big'))
        row_sum = 0
        for value in values:
            integer(value, 1)
            row_sum = checked_add(row_sum, value)
            digest.update(value.to_bytes(8, 'big', signed=True))
        nnz = checked_add(nnz, len(columns))
        total = checked_add(total, row_sum)
        zero_rows += len(columns) == 0
    if observed != n_rows:
        fail('MATRIX_CSR_INVALID')
    return dict(logical_matrix_sha256=digest.hexdigest(), nnz=nnz,
                total_count=total, zero_row_count=zero_rows)


def csr_identity(matrix):
    """Strict in-memory CSR reader for contract tests; never repairs storage."""
    if (getattr(matrix, 'format', None) != 'csr' or matrix.dtype.kind != 'i'
            or matrix.dtype.itemsize != 8 or matrix.indices.dtype.kind not in 'iu'
            or matrix.indptr.dtype.kind not in 'iu' or matrix.data.ndim != 1
            or matrix.indices.ndim != 1 or matrix.indptr.ndim != 1):
        fail('MATRIX_CSR_INVALID')
    n, m = map(int, matrix.shape)
    if (len(matrix.indptr) != n + 1 or int(matrix.indptr[0]) != 0
            or len(matrix.data) != len(matrix.indices)
            or int(matrix.indptr[-1]) != len(matrix.data)):
        fail('MATRIX_CSR_INVALID')
    def rows():
        previous = 0
        for endpoint in matrix.indptr[1:]:
            right = int(endpoint)
            if not previous <= right <= len(matrix.data):
                fail('MATRIX_CSR_INVALID')
            yield [int(x) for x in matrix.indices[previous:right]], [int(x) for x in matrix.data[previous:right]]
            previous = right
    return logical_matrix_identity(n, m, rows())


def selected_axis_identity(rows):
    """Validate given authoritative order, without sorting or selecting rows.

    Caller must compare digest/count to freshly verified selection. Presence in
    fragments requires the M11.5b source pass; this helper cannot establish it.
    """
    from .scatac_selection_profile import decode_cell_id
    digest = hashlib.sha256(b'agent.ordered-selected-cells.v1\0')
    seen = set(); count = 0
    for index, namespace, barcode, rendered in rows:
        integer(index)
        if index != count or decode_cell_id(rendered) != (namespace, barcode) or (namespace, barcode) in seen:
            fail('MATRIX_ROW_IDENTITY_INVALID')
        seen.add((namespace, barcode)); count += 1
        digest.update((namespace + '\t' + barcode + '\n').encode('ascii'))
    return count, digest.hexdigest()


def reference_axis_identity(rows, contigs):
    """Check given column positions/names/bounds; never sort or filter features.

    contigs is the freshly verified reference dictionary. Compare returned
    count/digest to that reference; matching dimensions alone is insufficient.
    """
    digest = hashlib.sha256(); seen = set(); count = 0
    for index, name, chrom, start, end in rows:
        integer(index); interval(chrom, start, end)
        if (index != count or name != f'{chrom}:{start}-{end}' or name in seen
                or chrom not in contigs or end > contigs[chrom]):
            fail('MATRIX_COLUMN_IDENTITY_INVALID')
        seen.add(name); count += 1
        digest.update((name + '\n').encode('utf-8'))
    if count == 0:
        fail('MATRIX_COLUMN_IDENTITY_INVALID')
    return count, digest.hexdigest()


H5AD_LAYOUT = (
    ('X', 'csr:int64:full-shape:canonical'),
    ('obs_names', 'exact-selected-rendered-IDs'),
    ('obs', ('namespace', 'barcode_identifier', 'matrix_row_index')),
    ('var_names', 'exact-reference-feature-names'),
    ('var', ('chrom', 'start', 'end', 'matrix_column_index')),
    ('uns', ('matrix_semantics', 'matrix_profile_id', 'matrix_profile_sha256',
             'species', 'assembly', 'coordinate_system', 'ordered_selected_sha256',
             'ordered_feature_sha256', 'logical_matrix_sha256')),
    ('forbidden', ('raw', 'duplicate-counts-layer', 'complete-upstream-manifests', 'full-QC-table')),
)


def manifest_identity(value):
    return hashlib.sha256(b'agent.cell-by-ccre-identity.v1\0' + canonical(
        {k: v for k, v in value.items() if k != 'identity_sha256'})).hexdigest()


MANIFEST_FIELDS = ('artifact_type', 'schema_version', 'contract_version', 'profile',
    'profile_sha256', 'species', 'assembly', 'upstream', 'ordered_selected_sha256',
    'ordered_feature_sha256', 'shape', 'nnz', 'total_count', 'zero_row_count',
    'logical_matrix_sha256', 'matrix', 'backend', 'diagnostic', 'readiness', 'identity_sha256')


def validate_manifest(value):
    """Closed, IO-free outer schema; does not confer a verified claim."""
    shape(value, MANIFEST_FIELDS)
    if (len(canonical(value)) > MAX_MANIFEST_BYTES or value['artifact_type'] != ARTIFACT
            or type(value['schema_version']) is not int or value['schema_version'] != 1
            or value['contract_version'] != CONTRACT or value['profile'] != asdict(PROFILE)
            or value['profile_sha256'] != PROFILE_SHA256
            or (value['species'], value['assembly']) not in (('human', 'hg38'), ('mouse', 'mm10'))):
        fail()
    shape(value['upstream'], ('fragments', 'selection', 'reference'))
    contracts = ('scatac-fragments.v2', 'scatac-cell-selection.v1', 'scatac-reference-bundle.v1')
    for key, contract in zip(('fragments', 'selection', 'reference'), contracts):
        pointer = value['upstream'][key]
        shape(pointer, ('manifest_path', 'manifest_sha256', 'identity_sha256', 'contract_version'))
        absolute_path(pointer['manifest_path'])
        sha(pointer['manifest_sha256']); sha(pointer['identity_sha256'])
        if pointer['contract_version'] != contract:
            fail('MATRIX_LINEAGE_INVALID')
    for key in ('ordered_selected_sha256', 'ordered_feature_sha256', 'logical_matrix_sha256', 'identity_sha256'):
        sha(value[key])
    dims = value['shape']
    if type(dims) is not list or len(dims) != 2:
        fail()
    n, m = integer(dims[0]), integer(dims[1], 1)
    nnz = integer(value['nnz']); total = integer(value['total_count']); zeros = integer(value['zero_row_count'], 0, n)
    if not n - zeros <= nnz <= (n - zeros) * m or total < nnz or (nnz == 0) != (total == 0):
        fail('MATRIX_SUMMARY_INVALID')
    shape(value['matrix'], ('path', 'sha256', 'size_bytes', 'format', 'dtype'))
    if (value['matrix']['path'], value['matrix']['format'], value['matrix']['dtype']) != ('matrix.h5ad', 'csr', 'int64'):
        fail()
    sha(value['matrix']['sha256']); integer(value['matrix']['size_bytes'], 1)
    shape(value['backend'], ('profile_id', 'runtime_sha256'))
    if value['backend']['profile_id'] != 'bedtools-ccre-record-incidence.v1':
        fail('MATRIX_BACKEND_INVALID')
    sha(value['backend']['runtime_sha256'])
    diagnostic = value['diagnostic']
    shape(diagnostic, overlap_diagnostic(0, 0))
    expected = overlap_diagnostic(diagnostic['n_selected_fragment_records_total'],
                                  diagnostic['n_selected_fragment_records_overlapping_any_ccre'])
    if canonical(expected) != canonical(diagnostic):
        fail('MATRIX_DIAGNOSTIC_INVALID')
    records = diagnostic['n_selected_fragment_records_total']
    hits = diagnostic['n_selected_fragment_records_overlapping_any_ccre']
    if (records < n or (n == 0 and records != 0) or not hits <= total <= hits * m
            or hits < n - zeros or (hits == 0) != (total == 0)):
        fail('MATRIX_DIAGNOSTIC_INVALID')
    if n == 0 and (value['ordered_selected_sha256'] != selected_axis_identity([])[1]
            or value['logical_matrix_sha256'] != logical_matrix_identity(0, m, [])['logical_matrix_sha256']):
        fail('MATRIX_IDENTITY_MISMATCH')
    expected_readiness = 'no_selected_cells' if n == 0 else 'matrix_available'
    if value['readiness'] != expected_readiness or value['identity_sha256'] != manifest_identity(value):
        fail('MATRIX_IDENTITY_MISMATCH')
    return value


def load_manifest_bytes(raw):
    if type(raw) is not bytes or len(raw) > MAX_MANIFEST_BYTES:
        fail()
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                fail('MATRIX_JSON_INVALID')
            out[key] = value
        return out
    try:
        value = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs,
                           parse_constant=lambda _: fail('MATRIX_JSON_INVALID'))
        validate_manifest(value)
        if canonical(value) != raw:
            fail('MATRIX_JSON_INVALID')
        return value
    except (UnicodeError, TypeError, ValueError, RecursionError) as exc:
        if isinstance(exc, ScATACMatrixError):
            raise
        raise ScATACMatrixError('MATRIX_JSON_INVALID') from exc
