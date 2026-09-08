"""M10.3 read-only BAM intake via lazy pysam/htslib; no execution-tool export.

Observation is bounded to 256 sequential decoded records. Header limits apply
AFTER backend decoding. Neither these limits nor the record limit guarantee a
pre-allocation byte/memory bound inside htslib. Decoded byte counts are unknown.
No BAM binary decoding, index construction, reference access, or preprocessing
is implemented here. Normalized digests contain no read/barcode/quality vectors.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import Enum
import errno
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import stat
import warnings

from . import raw_scatac_manifest as m

__all__ = ["BamAssay", "inspect_bam_inputs", "observe_bam_source"]
CONTRACT = "raw-bam.v1"
TESTED_PYSAM_VERSION = "0.24.1"
MAX_CANDIDATE_FILES = 128
MAX_SELECTIONS = 128
MAX_DIRECTORY_ENTRIES = 16384
MAX_RECORDS = 256
MAX_HEADER_BYTES = 1024 * 1024  # Post-decode header projection limit, not a backend memory bound.
MAX_REFERENCES = 4096
MAX_READ_GROUPS = 256
MAX_PROGRAMS = 128
MAX_METADATA_VALUES = 4
TAGS = ("CB", "CR", "CY", "BC", "QT", "RG")


class BamInspectionError(ValueError):
    """Sanitized BAM-only operational failure; no backend exception prose."""
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class BamAssay(str, Enum):
    SCATAC = "SCATAC"
    TENX_ATAC = "TENX_ATAC"


class SortDeclaration(str, Enum):
    ABSENT = "absent"
    COORDINATE = "coordinate"
    QUERYNAME = "queryname"
    UNSORTED = "unsorted"
    UNKNOWN = "unknown"
    OTHER = "other"


class ObservedOrder(str, Enum):
    INSUFFICIENT = "insufficient-primary-mapped-records"
    CONSISTENT = "sample-coordinate-consistent"
    INVERSION = "sample-coordinate-inversion"


class IndexState(str, Enum):
    ABSENT = "no-candidate"
    OPENABLE = "one-openable"
    UNUSABLE = "candidate-unusable"
    MULTIPLE = "multiple-candidates"


class ObservationStop(str, Enum):
    EOF = "eof"
    BUDGET = "record-budget"
    ERROR = "content-error"
    HEADER_LIMIT = "header-limit"


class Finding(str, Enum):
    NOT_BAM = "BAM_NOT_BAM"
    CONTENT_INVALID = "BAM_CONTENT_INVALID"
    TRUNCATED = "BAM_TRUNCATION_OBSERVED"
    HEADER_INVALID = "BAM_HEADER_INVALID"
    HEADER_LIMIT = "BAM_HEADER_LIMIT"
    METADATA_UNREVIEWED = "BAM_METADATA_UNREVIEWED"
    TAG_TYPE = "BAM_TAG_TYPE_INVALID"
    TAG_VALUE = "BAM_TAG_VALUE_INVALID"
    TAG_DUPLICATE = "BAM_TAG_DUPLICATE"
    RG_UNDECLARED = "BAM_RG_UNDECLARED"
    RECORD_INVALID = "BAM_RECORD_INVALID"
    EMPTY = "BAM_EMPTY"
    NO_USABLE = "BAM_NO_PRIMARY_MAPPED_OBSERVED"
    ASSAY_UNRESOLVED = "BAM_ASSAY_UNRESOLVED"
    SORT_CONTRADICTION = "BAM_SORT_DECLARATION_CONFLICT"
    SORT_INVERSION = "BAM_SORT_INVERSION_OBSERVED"
    INDEX_UNUSABLE = "BAM_INDEX_UNUSABLE"
    INDEX_MULTIPLE = "BAM_INDEX_MULTIPLE"


_INFORMATION = {Finding.HEADER_LIMIT, Finding.METADATA_UNREVIEWED,
                Finding.NO_USABLE, Finding.ASSAY_UNRESOLVED}
_ADVISORY = {Finding.SORT_CONTRADICTION, Finding.SORT_INVERSION,
             Finding.INDEX_UNUSABLE, Finding.INDEX_MULTIPLE}


class Fact(str, Enum):
    HEADER = "header"
    REFERENCES = "references"
    READ_GROUPS = "read-groups"
    PROGRAMS = "programs"
    PRODUCER_SEMANTICS = "producer-barcode-definitions"
    ASSAY = "assay"
    PARSE = "parse"
    FLAGS = "flags"
    TAG = "tag"
    SORT = "sort"
    INDEX = "index"
    LIMITS = "limits"


class Denominator(str, Enum):
    ALL = "all"
    PRIMARY = "primary"
    MAPPED_PRIMARY = "mapped_primary"


@dataclass(frozen=True)
class TagCounts:
    n: int = 0
    present: int = 0
    usable: int = 0
    wrong_type: int = 0
    invalid_value: int = 0
    duplicate: int = 0


@dataclass(frozen=True)
class TagObservation:
    tag: str
    all: TagCounts
    primary: TagCounts
    mapped_primary: TagCounts


@dataclass(frozen=True)
class RecordCounts:
    records: int = 0
    primary: int = 0
    secondary: int = 0
    supplementary: int = 0
    mapped: int = 0
    unmapped: int = 0
    mapped_primary: int = 0
    paired: int = 0
    unpaired: int = 0
    read1: int = 0
    read2: int = 0
    proper_pair: int = 0
    duplicate: int = 0
    qc_fail: int = 0


@dataclass(frozen=True)
class HeaderObservation:
    hd_present: bool
    sort_declaration: SortDeclaration
    format_version: str | None
    reference_count: int
    reference_sha256: str
    m5_count: int
    m5_sha256: str
    # (normalized value, authoritative for the complete sequence dictionary).
    species: tuple[tuple[str, bool], ...]
    assemblies: tuple[tuple[str, bool], ...]
    read_group_count: int
    library_count: int
    sample_identifier_count: int
    library_id: str | None
    read_group_sha256: str
    program_count: int
    program_sha256: str
    cellranger_atac: bool
    producer_version: str | None


@dataclass(frozen=True)
class IndexObservation:
    state: IndexState
    candidates: tuple[str, ...]  # Local standard suffix kinds only, never arbitrary paths.


@dataclass(frozen=True)
class BamObservation:
    file: m.FileRecord
    pysam_version: str
    htslib_version: str
    header: HeaderObservation | None
    counts: RecordCounts
    tags: tuple[TagObservation, ...]
    order: ObservedOrder
    coordinate_comparisons: int
    coordinate_inversions: int
    index: IndexObservation
    stop: ObservationStop
    findings: tuple[Finding, ...]
    normalized_observation_sha256: str


@dataclass(frozen=True)
class _Selected:
    file: m.FileRecord
    snapshot: tuple[int, ...]


@dataclass(frozen=True)
class _HeaderContext:
    summary: HeaderObservation
    read_group_ids: frozenset[str]


def _error(code: str):
    raise BamInspectionError("RAW_BAM_" + code) from None


def _backend():
    try:
        backend = importlib.import_module("pysam")
    except (ImportError, OSError):
        _error("DEPENDENCY_UNAVAILABLE")
    if backend.__version__ != TESTED_PYSAM_VERSION:
        _error("BACKEND_VERSION_UNSUPPORTED")
    return backend


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _fact(fact: Fact, **values):
    return _json({"contract": CONTRACT, "fact": fact.value, **values})


def _snapshot(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
            value.st_ctime_ns, value.st_mode)


def _stat(path: Path):
    try:
        value = path.stat(follow_symlinks=False)
    except OSError:
        _error("SOURCE_UNAVAILABLE")
    if not stat.S_ISREG(value.st_mode):
        _error("NOT_REGULAR_FILE")
    if any(not 0 <= n <= 2**63 - 1 for n in (value.st_size, value.st_mtime_ns)):
        _error("SOURCE_METADATA_INVALID")
    return value


def _unchanged(path: Path, before):
    try:
        after = _snapshot(path.stat(follow_symlinks=False))
    except OSError:
        _error("SOURCE_CHANGED")
    if after != before:
        _error("SOURCE_CHANGED")


def _resolve(value):
    if not isinstance(value, (str, Path)) or not str(value) or "://" in str(value):
        _error("SELECTION_INVALID")
    try:
        path = Path(value).expanduser().resolve(strict=True)
        m.file_identity(str(path))
        return path
    except (ValueError, OSError, RuntimeError):
        _error("SELECTION_INVALID")


def _discover(inputs):
    if isinstance(inputs, (str, Path)):
        inputs = (inputs,)
    if not isinstance(inputs, (tuple, list)) or not 0 < len(inputs) <= MAX_SELECTIONS:
        _error("SELECTION_INVALID")
    candidates = {}
    for path in sorted({_resolve(v) for v in inputs}, key=str):
        if path.is_dir():
            names = []
            try:
                with os.scandir(path) as entries:
                    for n, entry in enumerate(entries, 1):
                        if n > MAX_DIRECTORY_ENTRIES:
                            _error("DISCOVERY_LIMIT")
                        if entry.name.endswith(".bam"):
                            if entry.is_symlink():
                                _error("DISCOVERY_SYMLINK")
                            names.append(entry.name)
                            if len(names) > MAX_CANDIDATE_FILES:
                                _error("DISCOVERY_LIMIT")
            except OSError:
                _error("DISCOVERY_UNAVAILABLE")
            for name in sorted(names):
                candidates.setdefault(str(path / name), str(path))
        else:
            if not path.name.endswith(".bam"):
                _error("SUFFIX_UNSUPPORTED")
            candidates[str(path)] = None
        if len(candidates) > MAX_CANDIDATE_FILES:
            _error("DISCOVERY_LIMIT")
    if not candidates:
        _error("NO_CANDIDATES")
    result, inodes = [], set()
    for name, root in sorted(candidates.items()):
        value = _stat(Path(name))
        inode = (value.st_dev, value.st_ino)
        if inode in inodes:
            _error("ALIAS_CONFLICT")
        inodes.add(inode)
        try:
            file = m.FileRecord(name, m.InputKind.BAM, value.st_size, value.st_mtime_ns,
                m.SelectionBasis.EXPLICIT if root is None else m.SelectionBasis.DIRECTORY_DISCOVERY, root)
        except m.RawIntakeManifestError:
            _error("SELECTION_INVALID")
        result.append(_Selected(file, _snapshot(value)))
    return tuple(result)


def _token(value, maximum=128):
    return (type(value) is str and len(value) <= maximum
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:+-]*", value) is not None)


def _header_label(value):
    return type(value) is str and bool(value) and all(32 <= ord(c) <= 126 for c in value)


def _species(value):
    if type(value) is not str:
        return None
    return {"human": "human", "homo sapiens": "human", "mouse": "mouse",
            "mus musculus": "mouse"}.get(value.strip().lower())


def _assembly(value):
    if type(value) is not str or not _token(value.strip()):
        return None
    normalized = value.strip().lower()
    return {"grch38": "hg38", "hg38": "hg38", "grcm38": "mm10", "mm10": "mm10"}.get(normalized, normalized)


class _HeaderLimit(Exception):
    pass


class _HeaderInvalid(Exception):
    pass


def _header(bam):
    # htslib has already decoded the header. These are projection/iteration
    # bounds, not a pre-allocation promise. UR and PG CL are never projected.
    text = str(bam.header)
    if len(text.encode('utf-8')) > MAX_HEADER_BYTES or bam.nreferences > MAX_REFERENCES:
        raise _HeaderLimit
    raw = bam.header.to_dict()
    sq, rg, pg, hd = raw.get('SQ', []), raw.get('RG', []), raw.get('PG', []), raw.get('HD', {})
    if len(sq) > MAX_REFERENCES or len(rg) > MAX_READ_GROUPS or len(pg) > MAX_PROGRAMS:
        raise _HeaderLimit
    if len(sq) != bam.nreferences:
        raise _HeaderInvalid
    refs, m5 = [], []
    for entry, name, length in zip(sq, bam.references, bam.lengths, strict=True):
        if entry.get('SN') != name or entry.get('LN') != length or length <= 0:
            raise _HeaderInvalid
        md5 = entry.get('M5')
        if md5 is not None and (type(md5) is not str or re.fullmatch('[0-9a-fA-F]{32}', md5) is None):
            raise _HeaderInvalid
        if md5 is not None:
            m5.append((name, md5.lower()))
        refs.append((name, length, entry.get('AS'), entry.get('SP'), md5.lower() if md5 else None))
    if len({r[0] for r in refs}) != len(refs):
        raise _HeaderInvalid
    notices = set()

    def metadata(key, normalize):
        supplied = [s[key] for s in sq if key in s]
        normalized = [normalize(v) for v in supplied]
        if any(v is None for v in normalized):
            notices.add(Finding.METADATA_UNREVIEWED)
        values = sorted({v for v in normalized if v is not None})
        if len(values) > MAX_METADATA_VALUES:
            raise _HeaderLimit
        complete = bool(sq) and len(supplied) == len(sq) and all(v is not None for v in normalized)
        return tuple((v, complete) for v in values)

    species, assemblies = metadata('SP', _species), metadata('AS', _assembly)
    ids, libs, samples = [], [], []
    for record in rg:
        if not _header_label(record.get('ID')):
            raise _HeaderInvalid
        ids.append(record['ID'])
        for key, values in (('LB', libs), ('SM', samples)):
            if key in record:
                if not _header_label(record[key]):
                    raise _HeaderInvalid
                values.append(record[key])
    if len(set(ids)) != len(ids):
        raise _HeaderInvalid
    unique_library = next(iter(set(libs))) if len(set(libs)) == 1 and len(libs) == len(rg) else None
    if unique_library is not None and (len(unique_library) > 128 or unique_library.strip() != unique_library):
        # Preserve counts/digest even when a valid SAM label cannot fit M10.1.
        notices.add(Finding.METADATA_UNREVIEWED)
        unique_library = None
    programs, versions, producer = [], set(), False
    for record in pg:
        # Exact PN, or exact ID when PN is absent. Generic aligner names and
        # command-line substrings confer no ATAC authority.
        identity = record.get('PN', record.get('ID'))
        reviewed = identity == 'cellranger-atac'
        version = record.get('VN')
        if version is not None and not _token(version):
            notices.add(Finding.METADATA_UNREVIEWED)
            version = None
        programs.append((record.get('ID'), record.get('PN'), version))
        if reviewed:
            producer = True
            if version is not None:
                versions.add(version)
    declaration = hd.get('SO')
    sort = (SortDeclaration.ABSENT if declaration is None else
            SortDeclaration(declaration) if declaration in ('coordinate', 'queryname', 'unsorted', 'unknown')
            else SortDeclaration.OTHER)
    vn = hd.get('VN')
    if vn is not None and (not _token(vn) or re.fullmatch(r'[0-9]+(?:\.[0-9]+)*', vn) is None):
        notices.add(Finding.METADATA_UNREVIEWED)
        vn = None
    summary = HeaderObservation('HD' in raw, sort, vn, len(refs), _digest(refs), len(m5), _digest(m5),
        species, assemblies, len(rg), len(set(libs)), len(set(samples)), unique_library,
        _digest(sorted((r.get('ID'), r.get('LB'), r.get('SM')) for r in rg)),
        len(pg), _digest(programs), producer, next(iter(versions)) if len(versions) == 1 else None)
    return _HeaderContext(summary, frozenset(ids)), notices


def _index_candidates(path):
    return ((Path(str(path) + '.bai'), 'bam.bai'), (path.with_suffix('.bai'), 'bai'),
            (Path(str(path) + '.csi'), 'bam.csi'), (path.with_suffix('.csi'), 'csi'))


def _index_snapshot(path):
    result = []
    for candidate, kind in _index_candidates(path):
        try:
            value = candidate.stat(follow_symlinks=False)
        except FileNotFoundError:
            value = None
        except OSError:
            _error('INDEX_OBSERVATION_FAILED')
        result.append((candidate, kind, _snapshot(value) if value is not None else None))
    return tuple(result)


def _open_raw(path):
    try:
        return os.fdopen(os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW), 'rb')
    except OSError:
        _error('READ_FAILED')


def _index_observation(path, candidates, backend):
    present = [(p, k, s) for p, k, s in candidates if s is not None]
    kinds = tuple(k for p, k, s in present)
    if not present:
        return IndexObservation(IndexState.ABSENT, ())
    if len(present) > 1:
        return IndexObservation(IndexState.MULTIPLE, kinds)
    candidate, _, snapshot = present[0]
    if not stat.S_ISREG(snapshot[-1]):
        return IndexObservation(IndexState.UNUSABLE, kinds)
    try:
        # A separate handle never changes the sequential sampling position.
        with _open_raw(path) as raw:
            with backend.AlignmentFile(raw, 'rb', check_sq=False, index_filename=str(candidate)) as indexed:
                available = indexed.check_index()
    except BamInspectionError:
        raise
    except (ValueError, OSError):
        return IndexObservation(IndexState.UNUSABLE, kinds)
    return IndexObservation(IndexState.OPENABLE if available else IndexState.UNUSABLE, kinds)


def _sample(bam, context):
    counts = dict.fromkeys(RecordCounts.__dataclass_fields__, 0)
    tag_counts = {tag: {scope: dict.fromkeys(TagCounts.__dataclass_fields__, 0) for scope in Denominator} for tag in TAGS}
    findings, previous, comparisons, inversions = set(), None, 0, 0
    stop = ObservationStop.BUDGET
    iterator = bam.fetch(until_eof=True)
    for _ in range(MAX_RECORDS):
        try:
            record = next(iterator)
        except StopIteration:
            stop = ObservationStop.EOF
            break
        except (OSError, ValueError) as exc:
            if isinstance(exc, OSError) and exc.errno not in (None, 0, errno.EINVAL):
                _error('READ_FAILED')
            findings.add(Finding.CONTENT_INVALID)
            stop = ObservationStop.ERROR
            break
        counts['records'] += 1
        primary = not record.is_secondary and not record.is_supplementary
        mapped_primary = primary and not record.is_unmapped
        for key, flag in (
            ('primary', primary), ('secondary', record.is_secondary), ('supplementary', record.is_supplementary),
            ('mapped', not record.is_unmapped), ('unmapped', record.is_unmapped),
            ('mapped_primary', mapped_primary), ('paired', record.is_paired), ('unpaired', not record.is_paired),
            ('read1', record.is_read1), ('read2', record.is_read2), ('proper_pair', record.is_proper_pair),
            ('duplicate', record.is_duplicate), ('qc_fail', record.is_qcfail)):
            counts[key] += int(flag)
        if not record.is_unmapped and (record.reference_id < 0 or record.reference_id >= bam.nreferences
                or record.reference_start < 0 or record.reference_start >= bam.lengths[record.reference_id]):
            findings.add(Finding.RECORD_INVALID)
        if mapped_primary:
            coordinate = (record.reference_id, record.reference_start)
            if previous is not None:
                comparisons += 1
                inversions += int(coordinate < previous)
            previous = coordinate
        # Let pysam decode aux types; do not reinterpret BAM binary structures.
        observed = {tag: [] for tag in TAGS}
        try:
            for tag, value, kind in record.get_tags(with_value_type=True):
                if tag in observed:
                    observed[tag].append((value, kind))
        except (ValueError, UnicodeError):
            findings.add(Finding.CONTENT_INVALID)
            stop = ObservationStop.ERROR
            break
        for tag, values in observed.items():
            present = bool(values)
            wrong = any(kind != 'Z' or type(value) is not str for value, kind in values)
            minimum = 33 if tag in ('CY', 'QT') else 32
            bad = any(type(value) is str and (not value or any(not minimum <= ord(c) <= 126 for c in value))
                      for value, kind in values)
            duplicate = len(values) > 1
            usable = present and not (wrong or bad or duplicate)
            if wrong:
                findings.add(Finding.TAG_TYPE)
            if bad:
                findings.add(Finding.TAG_VALUE)
            if duplicate:
                findings.add(Finding.TAG_DUPLICATE)
            if tag == 'RG' and usable and values[0][0] not in context.read_group_ids:
                findings.add(Finding.RG_UNDECLARED)
            for scope, eligible in ((Denominator.ALL, True), (Denominator.PRIMARY, primary),
                                    (Denominator.MAPPED_PRIMARY, mapped_primary)):
                if eligible:
                    c = tag_counts[tag][scope]
                    for name, number in (('n', 1), ('present', present), ('usable', usable), ('wrong_type', wrong),
                                         ('invalid_value', bad), ('duplicate', duplicate)):
                        c[name] += int(number)
    if stop is ObservationStop.EOF and counts['records'] == 0:
        findings.add(Finding.EMPTY)
    elif counts['mapped_primary'] == 0:
        findings.add(Finding.NO_USABLE)
    order = (ObservedOrder.INVERSION if inversions else ObservedOrder.CONSISTENT
             if comparisons else ObservedOrder.INSUFFICIENT)
    if inversions:
        findings.add(Finding.SORT_CONTRADICTION if context.summary.sort_declaration is SortDeclaration.COORDINATE
                     else Finding.SORT_INVERSION)
    tags = tuple(TagObservation(tag, *(TagCounts(**tag_counts[tag][scope]) for scope in Denominator)) for tag in TAGS)
    return RecordCounts(**counts), tags, order, comparisons, inversions, stop, findings


def _observe(selected, backend):
    path = Path(selected.file.path)
    _unchanged(path, selected.snapshot)
    indexes_before = _index_snapshot(path)
    header, counts, tags = None, RecordCounts(), tuple(TagObservation(t, TagCounts(), TagCounts(), TagCounts()) for t in TAGS)
    order, comparisons, inversions = ObservedOrder.INSUFFICIENT, 0, 0
    findings, stop = set(), ObservationStop.ERROR
    try:
        with _open_raw(path) as raw:
            if _snapshot(os.fstat(raw.fileno())) != selected.snapshot:
                _error('SOURCE_CHANGED')
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='no BGZF EOF marker.*')
                    # File-object opening disables implicit index discovery. Only
                    # the separate explicit index probe may open a sidecar.
                    bam = backend.AlignmentFile(raw, 'rb', check_sq=False, ignore_truncation=True)
                with bam:
                    if not bam.is_bam:
                        findings.add(Finding.NOT_BAM)
                    else:
                        try:
                            bam.check_truncation()
                        except OSError:
                            findings.add(Finding.TRUNCATED)
                        context, notices = _header(bam)
                        header = context.summary
                        findings.update(notices)
                        counts, tags, order, comparisons, inversions, stop, sampled = _sample(bam, context)
                        findings.update(sampled)
                        if Finding.TRUNCATED in findings:
                            stop = ObservationStop.ERROR
            except _HeaderLimit:
                findings.add(Finding.HEADER_LIMIT)
                stop = ObservationStop.HEADER_LIMIT
            except _HeaderInvalid:
                findings.add(Finding.HEADER_INVALID)
            except (ValueError, UnicodeError, OSError) as exc:
                if isinstance(exc, BamInspectionError):
                    raise
                # After a decoded corruption finding, htslib close can raise an
                # OSError carrying stale errno (including ENOENT). Preserve the
                # observed invalid content instead of relabeling it operational.
                corruption_seen = bool(findings & {Finding.CONTENT_INVALID, Finding.TRUNCATED, Finding.HEADER_INVALID})
                if isinstance(exc, OSError) and exc.errno not in (None, 0, errno.EINVAL) and not corruption_seen:
                    _error('READ_FAILED')
                findings.add(Finding.CONTENT_INVALID)
                stop = ObservationStop.ERROR
            if _snapshot(os.fstat(raw.fileno())) != selected.snapshot:
                _error('SOURCE_CHANGED')
    except OSError:
        _error('READ_FAILED')
    if Finding.TRUNCATED in findings:
        stop = ObservationStop.ERROR
    index = _index_observation(path, indexes_before, backend) if header is not None else IndexObservation(
        IndexState.MULTIPLE if sum(s is not None for p, k, s in indexes_before) > 1 else IndexState.UNUSABLE
        if any(s is not None for p, k, s in indexes_before) else IndexState.ABSENT,
        tuple(k for p, k, s in indexes_before if s is not None))
    if index.state is IndexState.UNUSABLE:
        findings.add(Finding.INDEX_UNUSABLE)
    elif index.state is IndexState.MULTIPLE:
        findings.add(Finding.INDEX_MULTIPLE)
    _unchanged(path, selected.snapshot)
    if indexes_before != _index_snapshot(path):
        _error('SOURCE_CHANGED')
    findings = tuple(sorted(findings, key=lambda f: f.value))
    normalized = {'contract': CONTRACT, 'pysam': backend.__version__, 'htslib': backend.version.__htslib_version__,
                  'header': asdict(header) if header else None, 'counts': asdict(counts),
                  'tags': [asdict(t) for t in tags], 'order': order, 'comparisons': comparisons,
                  'inversions': inversions, 'stop': stop,
                  'findings': tuple(f for f in findings if f not in (Finding.INDEX_UNUSABLE, Finding.INDEX_MULTIPLE))}
    # Index availability is independent of sequential source observations and is
    # deliberately excluded from this normalized observed-region digest.
    return BamObservation(selected.file, backend.__version__, backend.version.__htslib_version__, header,
        counts, tags, order, comparisons, inversions, index, stop, findings, _digest(normalized))


def observe_bam_source(source: m.FileRecord) -> BamObservation:
    """Reopen one recorded BAM and reconstruct observations, not readiness.

    Source size/mtime must match. Index availability is freshly observed, not a
    certification that every BAM record is correctly indexed. Header/summary
    digests never hash read/sequence/barcode vectors or physical BAM contents.
    """
    backend = _backend()
    if not isinstance(source, m.FileRecord) or source.input_kind is not m.InputKind.BAM:
        _error('SELECTION_INVALID')
    path = _resolve(source.path)
    if str(path) != source.path or not path.name.endswith('.bam'):
        _error('SELECTION_INVALID')
    value = _stat(path)
    if (source.size_bytes, source.mtime_ns) != (value.st_size, value.st_mtime_ns):
        _error('SOURCE_CHANGED')
    return _observe(_Selected(source, _snapshot(value)), backend)


def inspect_bam_inputs(inputs: Sequence[str | Path] | str | Path, *, assay: BamAssay | None = None,
                       species: str | None = None, source_genome_assembly: str | None = None) -> m.RawIntakeManifest:
    """Populate M10.1 for one group per BAM; no biological grouping or repairs.

    Explicit species tokens are lowercase (human/mouse or another declared taxon).
    Source assembly uses only reviewed GRCh38/hg38 and GRCm38/mm10 equivalences;
    other identifier tokens remain distinct after case normalization.
    """
    if assay is not None and type(assay) is not BamAssay:
        _error('DECLARATION_INVALID')
    if species is not None and (not _token(species) or species != species.lower()):
        _error('DECLARATION_INVALID')
    assembly = _assembly(source_genome_assembly) if source_genome_assembly is not None else None
    if source_genome_assembly is not None and assembly is None:
        _error('DECLARATION_INVALID')
    backend = _backend()
    inventory = _discover(inputs)
    evidence, coverage, groups, issues, prerequisites, index_snapshots = [], [], [], [], [], []
    for selected in inventory:
        index_snapshots.append(_index_snapshot(Path(selected.file.path)))
        observed = _observe(selected, backend)
        file = selected.file
        gid = m.group_identity(m.InputKind.BAM, (file.id,))
        complete = observed.stop is ObservationStop.EOF
        reason = (m.StopReason.EOF if complete else m.StopReason.ERROR if observed.stop is ObservationStop.ERROR
                  else m.StopReason.BUDGET)
        c = m.CoverageRecord(gid, (file.id,), m.CoverageMethod.SEQUENTIAL if complete else m.CoverageMethod.PREFIX,
            m.CoverageScope.COMPLETE if complete else m.CoverageScope.SAMPLE, observed.counts.records,
            None, MAX_RECORDS, None, complete, reason, observed.normalized_observation_sha256)
        coverage.append(c)

        def claim(topic, value, authority=m.EvidenceSource.OBSERVATION):
            e = m.EvidenceRecord(authority, topic, value, group_id=gid, file_id=file.id,
                                 coverage_id=c.id if authority is not m.EvidenceSource.USER_DECLARATION else None)
            evidence.append(e)
            return e.id

        def fact(kind, **values):
            return claim(m.EvidenceClaim.STRUCTURE, _fact(kind, **values))

        grouping = claim(m.EvidenceClaim.GROUPING, 'single-selected-bam.v1', m.EvidenceSource.FILENAME)
        findings = set(observed.findings)
        h = observed.header
        resolved_assay = BamAssay.TENX_ATAC if h and h.cellranger_atac else assay
        if resolved_assay is None:
            findings.add(Finding.ASSAY_UNRESOLVED)
        if assay is not None:
            claim(m.EvidenceClaim.STRUCTURE, _fact(Fact.ASSAY, value=assay.value), m.EvidenceSource.USER_DECLARATION)
        fact(Fact.ASSAY, resolved=resolved_assay.value if resolved_assay else None,
             producer_authority=bool(h and h.cellranger_atac))
        fact(Fact.PARSE, stop=observed.stop.value, observation_digest=observed.normalized_observation_sha256)
        fact(Fact.LIMITS, records=MAX_RECORDS, header_bytes=MAX_HEADER_BYTES, references=MAX_REFERENCES,
             read_groups=MAX_READ_GROUPS, programs=MAX_PROGRAMS, metadata_values=MAX_METADATA_VALUES)
        values = asdict(observed.counts)
        fact(Fact.FLAGS, subset='alignment', **{k: values[k] for k in
            ('records', 'primary', 'secondary', 'supplementary', 'mapped', 'unmapped', 'mapped_primary')})
        fact(Fact.FLAGS, subset='flags', **{k: values[k] for k in
            ('records', 'paired', 'unpaired', 'read1', 'read2', 'proper_pair', 'duplicate', 'qc_fail')})
        tag_refs = {}
        for tag in observed.tags:
            tag_refs[tag.tag] = fact(Fact.TAG, tag=tag.tag,
                columns=','.join(TagCounts.__dataclass_fields__), **{
                    scope.value: list(asdict(getattr(tag, scope.value)).values()) for scope in Denominator})
        fact(Fact.SORT, observed=observed.order.value, comparisons=observed.coordinate_comparisons,
             inversions=observed.coordinate_inversions)
        fact(Fact.INDEX, state=observed.index.state.value, candidates=observed.index.candidates)
        library_refs = ()
        producer_refs = ()
        if h:
            fact(Fact.HEADER, hd=h.hd_present, so=h.sort_declaration.value, vn=h.format_version)
            fact(Fact.REFERENCES, n=h.reference_count, sha256=h.reference_sha256,
                 m5_n=h.m5_count, m5_sha256=h.m5_sha256)
            fact(Fact.READ_GROUPS, n=h.read_group_count, libraries=h.library_count,
                 sample_ids=h.sample_identifier_count, sha256=h.read_group_sha256)
            fact(Fact.PROGRAMS, n=h.program_count, sha256=h.program_sha256,
                 cellranger_atac=h.cellranger_atac, producer_version=h.producer_version)
            if h.library_id:
                library_refs = (claim(m.EvidenceClaim.LIBRARY, h.library_id, m.EvidenceSource.AUTHORITATIVE_METADATA),)
            if h.cellranger_atac:
                producer_refs = (claim(m.EvidenceClaim.BARCODE_PRODUCER, 'cellranger-atac',
                                       m.EvidenceSource.AUTHORITATIVE_METADATA),)
                claim(m.EvidenceClaim.STRUCTURE, _fact(Fact.PRODUCER_SEMANTICS,
                    value='cellranger-atac-cb-corrected-cr-raw-cy-quality.v1'), m.EvidenceSource.AUTHORITATIVE_METADATA)

        def metadata(explicit, entries, topic):
            assertions = []
            if explicit is not None:
                authority = m.EvidenceSource.USER_DECLARATION
                assertions.append(m.MetadataAssertion(explicit, authority, (claim(topic, explicit, authority),)))
            for value, authoritative in entries:
                authority = m.EvidenceSource.AUTHORITATIVE_METADATA if authoritative else m.EvidenceSource.OBSERVATION
                assertions.append(m.MetadataAssertion(value, authority, (claim(topic, value, authority),)))
            return m.MetadataResolution(tuple(assertions))

        barcodes = {t.tag: t for t in observed.tags}
        locator = 'CB' if barcodes['CB'].all.usable else 'CR' if barcodes['CR'].all.usable else None
        barcode = m.BarcodeProvenance(producer='cellranger-atac' if producer_refs else None,
                                      producer_evidence_ids=producer_refs)
        if locator:
            source = m.BarcodeSource.BAM_CELL_IDENTIFIER if locator == 'CB' else m.BarcodeSource.BAM_RAW_SEQUENCE
            refs = (claim(m.EvidenceClaim.BARCODE_SOURCE, source.value + ':' + locator),)
            quality = (claim(m.EvidenceClaim.BARCODE_QUALITIES, 'sampled-CY-string-observed.v1'),) if barcodes['CY'].all.usable else ()
            scope = m.BarcodeIdentityScope.UNRESOLVED if locator == 'CR' and h and h.library_count > 1 else m.BarcodeIdentityScope.GROUP_LOCAL
            barcode = m.BarcodeProvenance(source, locator, refs, quality,
                producer='cellranger-atac' if producer_refs else None, producer_evidence_ids=producer_refs, identity_scope=scope)
            if locator == 'CR':
                prerequisites.append(m.DownstreamPrerequisite(gid, m.PrerequisiteCode.BARCODE_PROCESSING, (tag_refs['CR'],)))
        effects = {m.ReadinessEffect.ADVISORY if f in _ADVISORY else m.ReadinessEffect.INFORMATION_REQUIRED
                   if f in _INFORMATION else m.ReadinessEffect.INVALID for f in findings}
        structure = (m.StructureState.INVALID if m.ReadinessEffect.INVALID in effects else
                     m.StructureState.UNRESOLVED if m.ReadinessEffect.INFORMATION_REQUIRED in effects else m.StructureState.SUPPORTED)
        structure_refs = (claim(m.EvidenceClaim.STRUCTURE, structure.value),)
        for finding in sorted(findings, key=lambda f: f.value):
            effect = (m.ReadinessEffect.ADVISORY if finding in _ADVISORY else m.ReadinessEffect.INFORMATION_REQUIRED
                      if finding in _INFORMATION else m.ReadinessEffect.INVALID)
            refs = tuple(tag_refs.values()) if finding in (Finding.TAG_TYPE, Finding.TAG_VALUE, Finding.TAG_DUPLICATE) else structure_refs
            issues.append(m.IntakeIssue(finding.value, m.Severity.WARNING if effect is not m.ReadinessEffect.INVALID
                else m.Severity.ERROR, gid, effect, refs, (file.id,)))
        groups.append(m.GroupRecord(m.InputKind.BAM, (file.id,), m.IntakeRoute.BAM_TO_CCRE,
            grouping_basis=m.GroupingBasis.SYNTACTIC, grouping_evidence_ids=(grouping,),
            structure=structure, structure_evidence_ids=structure_refs,
            species=metadata(species, h.species if h else (), m.EvidenceClaim.SPECIES),
            source_genome_assembly=metadata(assembly, h.assemblies if h else (), m.EvidenceClaim.SOURCE_ASSEMBLY),
            library_id=h.library_id if h else None, library_evidence_ids=library_refs, barcode=barcode))
    for selected, snapshot in zip(inventory, index_snapshots, strict=True):
        _unchanged(Path(selected.file.path), selected.snapshot)
        if _index_snapshot(Path(selected.file.path)) != snapshot:
            _error('SOURCE_CHANGED')
    if len(evidence) > m.MAX_COLLECTION_ITEMS:
        _error('MANIFEST_LIMIT')
    return m.build_raw_intake_manifest(files=tuple(s.file for s in inventory), groups=tuple(groups),
        evidence=tuple(evidence), coverage=tuple(coverage), issues=tuple(issues), prerequisites=tuple(prerequisites),
        inspection=m.InspectionConfiguration(backends=(m.BackendIdentity('raw-bam', CONTRACT),
            m.BackendIdentity('pysam', backend.__version__), m.BackendIdentity('htslib', backend.version.__htslib_version__))))
