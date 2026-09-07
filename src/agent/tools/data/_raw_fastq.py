"""Read-only, bounded FASTQ intake (M10.2); no registered Agent tool.

Only an explicit TENX_ATAC declaration authorizes the two reviewed role layouts.
Names establish syntactic read sets, never biological samples/libraries. Evidence
values are canonical JSON under ``raw-fastq.v1`` in the existing M10.1 fields.

The narrow parser accepts four LF/CRLF-terminated lines, printable ASCII sequence
and Phred text (no alphabet inference), and optional exactly repeated headers on
``+`` lines. Synchronization compares the first header token, removing only a
terminal /1 or /2. Identifiers are hashed transiently and never enter a manifest.
Budgets and implementation version are repository-owned, not planner inputs.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType
import zlib

from . import raw_scatac_manifest as m

__all__ = ["FastqAssay", "FastqLayout", "inspect_fastq_inputs", "observe_fastq_sources"]

CONTRACT = "raw-fastq.v1"
MAX_CANDIDATE_FILES = 128
MAX_SELECTIONS = 128
MAX_DIRECTORY_ENTRIES = 16384
MAX_RECORDS = 256
MAX_DECODED_BYTES = 2 * 1024 * 1024
MAX_LINE_BYTES = 16 * 1024  # Includes the line terminator.
# Also bound gzip headers, empty members, and low-yield compressed streams.
MAX_ENCODED_BYTES = 4 * 1024 * 1024
SUFFIXES = (".fastq.gz", ".fq.gz", ".fastq", ".fq")


class FastqInspectionError(ValueError):
    """Sanitized operational error; scientific findings belong in the manifest."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class FastqAssay(str, Enum):
    TENX_ATAC = "TENX_ATAC"


class ReadRole(str, Enum):
    R1 = "R1"
    R2 = "R2"
    R3 = "R3"
    I1 = "I1"
    I2 = "I2"


class FastqLayout(str, Enum):
    A = "tenx-atac-r1-r2-r3.v1"
    B = "tenx-atac-r1-i2-r2.v1"
    UNRESOLVED = "unresolved"
    UNSUPPORTED = "unsupported-role-set"


class ReadMeaning(str, Enum):
    GENOMIC_1 = "genomic-read-1"
    GENOMIC_2 = "genomic-read-2"
    BARCODE = "raw-cell-barcode-i5"
    SAMPLE_INDEX = "sample-index"


# I1 is optional in both arrangements; it is never a cell-barcode source.
LAYOUT_ROLES = MappingProxyType({
    FastqLayout.A: MappingProxyType({ReadRole.R1: ReadMeaning.GENOMIC_1,
        ReadRole.R2: ReadMeaning.BARCODE, ReadRole.R3: ReadMeaning.GENOMIC_2,
        ReadRole.I1: ReadMeaning.SAMPLE_INDEX}),
    FastqLayout.B: MappingProxyType({ReadRole.R1: ReadMeaning.GENOMIC_1,
        ReadRole.I2: ReadMeaning.BARCODE, ReadRole.R2: ReadMeaning.GENOMIC_2,
        ReadRole.I1: ReadMeaning.SAMPLE_INDEX}),
})
BARCODE_ROLES = MappingProxyType({FastqLayout.A: ReadRole.R2, FastqLayout.B: ReadRole.I2})


class Compression(str, Enum):
    PLAIN = "plain"
    GZIP = "gzip"


class ParseFinding(str, Enum):
    VALID = "valid-observed-records"
    EMPTY = "empty"
    TRUNCATED = "truncated-record"
    HEADER = "invalid-header"
    SEPARATOR = "invalid-separator"
    REPEATED_HEADER = "repeated-header-mismatch"
    LENGTH = "sequence-quality-length-mismatch"
    CHARACTERS = "invalid-characters"
    GZIP = "gzip-corruption-observed"
    LIMIT = "inspection-limit"


class ParseStop(str, Enum):
    EOF = "eof"
    RECORD_LIMIT = "record-limit"
    BYTE_LIMIT = "decoded-byte-limit"
    LINE_LIMIT = "line-limit"
    ENCODED_LIMIT = "encoded-byte-limit"
    MALFORMED = "malformed-record"
    GZIP_ERROR = "gzip-error"


class SyncState(str, Enum):
    SINGLE = "single-file"
    PREFIX = "synchronized-prefix"
    COMPLETE = "synchronized-complete"
    UNOBSERVED = "no-common-records"
    NAME_MISMATCH = "read-name-mismatch"
    COUNT_MISMATCH = "observed-count-mismatch"


class Fact(str, Enum):
    NAME = "name"
    ASSAY = "assay"
    LAYOUT = "layout"
    PARSE = "parse"
    LENGTH = "length"
    LIMITS = "limits"
    SYNC = "sync"
    READ_MEANING = "read-meaning"
    LAYOUT_DECLARATION = "layout-declaration"
    MISSING_ROLE = "missing-role"


class MissingRoleBasis(str, Enum):
    DECLARED = "declared-layout"
    UNIQUE = "unique-candidate"
    ALTERNATIVE = "alternative-candidate"


@dataclass(frozen=True)
class FastqName:
    sample: str
    sample_number: str
    lane: str | None
    role: ReadRole
    chunk: str


@dataclass(frozen=True)
class SourceObservation:
    """Compact source facts; records counts complete four-line candidates checked.

    Length summaries concern valid records only. Decoded bytes/digest also cover
    observed partial or malformed components, without retaining their contents.
    """
    file: m.FileRecord
    compression: Compression
    suffix_matches: bool
    finding: ParseFinding
    stop: ParseStop
    records: int
    decoded_bytes: int
    eof: bool
    observed_region_sha256: str
    min_length: int | None
    max_length: int | None


@dataclass(frozen=True)
class ReadSetObservation:
    sources: tuple[SourceObservation, ...]
    synchronization: SyncState
    records_compared: int
    synchronized_records: int


@dataclass(frozen=True)
class _Selected:
    file: m.FileRecord
    snapshot: tuple[int, ...]


def _error(code):
    raise FastqInspectionError("RAW_FASTQ_" + code) from None


def _suffix(name: str) -> str | None:
    return next((s for s in SUFFIXES if name.endswith(s)), None)


def _name(name: str) -> FastqName | None:
    suffix = _suffix(name)
    if suffix is None:
        return None
    # Peel fields from the right; an underscore in the sample is not a split.
    prefix, sep, chunk = name[:-len(suffix)].rpartition("_")
    prefix, role_sep, role = prefix.rpartition("_")
    if not sep or not role_sep or not re.fullmatch(r"[0-9]{3}", chunk):
        return None
    if role not in ReadRole._value2member_map_:
        return None
    lane = None
    left, lane_sep, candidate = prefix.rpartition("_L")
    if lane_sep and re.fullmatch(r"[0-9]{3}", candidate):
        prefix, lane = left, candidate
    sample, sample_sep, number = prefix.rpartition("_S")
    if (not sample_sep or not re.fullmatch(r"[0-9]{1,9}", number)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", sample)):
        return None
    return FastqName(sample, number, lane, ReadRole(role), chunk)


def _snapshot(value: os.stat_result) -> tuple[int, ...]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
            value.st_ctime_ns, value.st_mode)


def _stat(path: Path) -> os.stat_result:
    try:
        value = path.stat(follow_symlinks=False)
    except OSError:
        _error("SOURCE_UNAVAILABLE")
    if not stat.S_ISREG(value.st_mode):
        _error("NOT_REGULAR_FILE")
    if any(not 0 <= n <= 2**63 - 1 for n in (value.st_size, value.st_mtime_ns)):
        _error("SOURCE_METADATA_INVALID")
    return value


def _unchanged(path: Path, snapshot: tuple[int, ...]):
    try:
        value = path.stat(follow_symlinks=False)
    except OSError:
        _error("SOURCE_CHANGED")
    if _snapshot(value) != snapshot:
        _error("SOURCE_CHANGED")


def _resolve(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)) or not str(value) or "://" in str(value):
        _error("SELECTION_INVALID")
    try:
        path = Path(value).expanduser().resolve(strict=True)
        # Check M10.1 representability before any observations are made.
        m.file_identity(str(path))
        return path
    except (OSError, RuntimeError, ValueError):
        _error("SELECTION_INVALID")


def _discover(inputs: Sequence[str | Path] | str | Path) -> tuple[_Selected, ...]:
    if isinstance(inputs, (str, Path)):
        inputs = (inputs,)
    if not isinstance(inputs, (tuple, list)) or not 0 < len(inputs) <= MAX_SELECTIONS:
        _error("SELECTION_INVALID")
    candidates: dict[str, tuple[m.SelectionBasis, str | None]] = {}

    def add(path: Path, root: Path | None):
        if _suffix(path.name) is None:
            _error("SUFFIX_UNSUPPORTED")
        key = str(path)
        choice = (m.SelectionBasis.EXPLICIT if root is None else m.SelectionBasis.DIRECTORY_DISCOVERY,
                  str(root) if root else None)
        if key not in candidates or root is None:
            candidates[key] = choice
        if len(candidates) > MAX_CANDIDATE_FILES:
            _error("DISCOVERY_LIMIT")

    for path in sorted({_resolve(v) for v in inputs}, key=str):
        if path.is_dir():
            # No recursive traversal or implicit symlink following. Bound the
            # listing before sorting so arbitrary directory size cannot allocate.
            names = []
            try:
                with os.scandir(path) as entries:
                    for count, entry in enumerate(entries, 1):
                        if count > MAX_DIRECTORY_ENTRIES:
                            _error("DISCOVERY_LIMIT")
                        if _suffix(entry.name) is not None:
                            if entry.is_symlink():
                                _error("DISCOVERY_SYMLINK")
                            names.append(entry.name)
                            if len(names) > MAX_CANDIDATE_FILES:
                                _error("DISCOVERY_LIMIT")
            except OSError:
                _error("DISCOVERY_UNAVAILABLE")
            for name in sorted(names):
                add(path / name, path)
        else:
            add(path, None)
    if not candidates:
        _error("NO_CANDIDATES")
    result, inodes = [], set()
    for name, (basis, root) in sorted(candidates.items()):
        value = _stat(Path(name))
        inode = (value.st_dev, value.st_ino)
        # Canonical path aliases collapse above. Different hard-link names may
        # assign competing roles/roots; reject rather than choose a role.
        if inode in inodes:
            _error("ALIAS_CONFLICT")
        inodes.add(inode)
        try:
            file = m.FileRecord(name, m.InputKind.FASTQ, value.st_size, value.st_mtime_ns, basis, root)
        except m.RawIntakeManifestError:
            _error("SELECTION_INVALID")
        result.append(_Selected(file, _snapshot(value)))
    return tuple(result)


class _EncodedLimit(Exception):
    pass


class _EncodedReader:
    """Bound physical input reads, including pathological gzip header/member work."""

    def __init__(self, raw):
        self.raw = raw
        self.remaining = MAX_ENCODED_BYTES

    def read(self, size=-1):
        if self.remaining == 0:
            raise _EncodedLimit
        size = self.remaining if size < 0 else min(size, self.remaining)
        value = self.raw.read(size)
        self.remaining -= len(value)
        return value


class _Stop(Exception):
    def __init__(self, reason: ParseStop):
        self.reason = reason


class _DecodedReader:
    def __init__(self, stream, compressed: bool):
        self.stream = stream
        self.compressed = compressed
        self.count = 0
        self.digest = hashlib.sha256()

    def line(self) -> bytes:
        value = bytearray()
        while len(value) < MAX_LINE_BYTES:
            if self.count == MAX_DECODED_BYTES:
                raise _Stop(ParseStop.BYTE_LIMIT)
            # read1(1) deliberately avoids decoded read-ahead beyond the record
            # budget: gzip.readline() may check a later member/CRC prematurely.
            byte = self.stream.read1(1) if self.compressed else self.stream.read(1)
            if not byte:
                return bytes(value)
            self.count += 1
            self.digest.update(byte)
            value.extend(byte)
            if byte == b"\n":
                return bytes(value)
        raise _Stop(ParseStop.LINE_LIMIT)


def _content(line: bytes) -> bytes:
    return line[:-2] if line.endswith(b"\r\n") else line[:-1]


def _printable(value: bytes, *, header=False) -> bool:
    return all(33 <= b <= 126 or (header and b in (9, 32)) for b in value)


def _header_id(header: bytes) -> bytes:
    token = header[1:].split(maxsplit=1)[0]
    if token.endswith((b"/1", b"/2")):
        token = token[:-2]
    return hashlib.sha256(token).digest()


def _parse(stream, compressed: bool):
    reader = _DecodedReader(stream, compressed)
    identifiers, lengths = [], []
    complete_records = 0
    finding, stop, eof = ParseFinding.VALID, ParseStop.RECORD_LIMIT, False
    try:
        while len(identifiers) < MAX_RECORDS:
            lines = []
            for index in range(4):
                line = reader.line()
                if not line:
                    eof, stop = True, ParseStop.EOF
                    if index:
                        finding = ParseFinding.TRUNCATED
                    elif not identifiers:
                        finding = ParseFinding.EMPTY
                    break
                if not line.endswith(b"\n"):
                    eof, stop, finding = True, ParseStop.EOF, ParseFinding.TRUNCATED
                    break
                lines.append(_content(line))
            if eof:
                break
            complete_records += 1
            header, sequence, separator, quality = lines
            if (not header.startswith(b"@") or not header[1:] or not 33 <= header[1] <= 126
                    or header[1:].split(maxsplit=1)[0] in (b"/1", b"/2")):
                finding = ParseFinding.HEADER
            elif not separator.startswith(b"+"):
                finding = ParseFinding.SEPARATOR
            elif (not sequence or not _printable(sequence) or not _printable(quality)
                  or not _printable(header, header=True) or not _printable(separator, header=True)):
                finding = ParseFinding.CHARACTERS
            elif len(sequence) != len(quality):
                finding = ParseFinding.LENGTH
            elif separator[1:] and separator[1:] != header[1:]:
                finding = ParseFinding.REPEATED_HEADER
            if finding is not ParseFinding.VALID:
                stop = ParseStop.MALFORMED
                break
            identifiers.append(_header_id(header))
            lengths.append(len(sequence))
    except _Stop as exc:
        stop = exc.reason
        # Byte budget termination is normal sampling, including a partial final
        # record. A component too large for this parser is an unresolved limit.
        if stop is ParseStop.LINE_LIMIT or not identifiers:
            finding = ParseFinding.LIMIT
    except _EncodedLimit:
        finding, stop = ParseFinding.LIMIT, ParseStop.ENCODED_LIMIT
    except (gzip.BadGzipFile, EOFError, zlib.error):
        finding, stop = ParseFinding.GZIP, ParseStop.GZIP_ERROR
    return (finding, stop, eof, reader.count, reader.digest.hexdigest(),
            complete_records, tuple(identifiers), min(lengths, default=None), max(lengths, default=None))


def _observe(selected: _Selected):
    file, before = selected.file, selected.snapshot
    path = Path(file.path)
    _unchanged(path, before)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as raw:
            if _snapshot(os.fstat(raw.fileno())) != before:
                _error("SOURCE_CHANGED")
            magic = raw.read(2)
            raw.seek(0)
            compressed = magic == b"\x1f\x8b"
            if compressed:
                with gzip.GzipFile(fileobj=_EncodedReader(raw), mode="rb") as stream:
                    parsed = _parse(stream, True)
            else:
                parsed = _parse(raw, False)
            if _snapshot(os.fstat(raw.fileno())) != before:
                _error("SOURCE_CHANGED")
    except OSError:
        _error("READ_FAILED")
    _unchanged(path, before)
    finding, stop, eof, count, digest, complete_records, identifiers, minimum, maximum = parsed
    return SourceObservation(file, Compression.GZIP if compressed else Compression.PLAIN,
        file.path.endswith(".gz") == compressed, finding, stop, complete_records,
        count, eof, digest, minimum, maximum), identifiers


def _read_set(selected: tuple[_Selected, ...]) -> ReadSetObservation:
    samples = [_observe(s) for s in selected]
    sources = tuple(s[0] for s in samples)
    identities = [s[1] for s in samples]
    common = min(map(len, identities))
    compared, matching = 0, 0
    for index in range(common):
        compared += 1
        if len({ids[index] for ids in identities}) != 1:
            break
        matching += 1
    count_mismatch = any(s.eof and s.finding in (ParseFinding.VALID, ParseFinding.EMPTY)
                         and s.records < max(map(len, identities)) for s in sources)
    state = (SyncState.SINGLE if len(sources) == 1
             else SyncState.NAME_MISMATCH if compared != matching
             else SyncState.COUNT_MISMATCH if count_mismatch
             else SyncState.COMPLETE if all(s.eof and s.finding is ParseFinding.VALID for s in sources)
             else SyncState.PREFIX if common else SyncState.UNOBSERVED)
    for source in selected:
        _unchanged(Path(source.file.path), source.snapshot)
    return ReadSetObservation(sources, state, compared if len(sources) > 1 else 0,
                              matching if len(sources) > 1 else 0)


def observe_fastq_sources(sources: Sequence[m.FileRecord]) -> ReadSetObservation:
    """Reopen an explicitly supplied read set under the fixed bounded contract.

    Accept one file for source-only reinspection. Reconstruct parse, digest, and
    synchronization observations; never consume a serialized readiness decision.
    Require recorded size/mtime to match, then check inode/ctime as well during
    this observation. Returns compact summaries, no read IDs/sequence vectors.
    Later verifiers can compare this result with canonical manifest evidence.
    """
    if not isinstance(sources, (tuple, list)) or not 0 < len(sources) <= MAX_CANDIDATE_FILES:
        _error("SELECTION_INVALID")
    selected, paths, inodes = [], set(), set()
    for source in sources:
        if not isinstance(source, m.FileRecord) or source.input_kind is not m.InputKind.FASTQ:
            _error("SELECTION_INVALID")
        path = _resolve(source.path)
        if str(path) != source.path or _suffix(path.name) is None:
            _error("SELECTION_INVALID")
        value = _stat(path)
        if (value.st_size, value.st_mtime_ns) != (source.size_bytes, source.mtime_ns):
            _error("SOURCE_CHANGED")
        inode = (value.st_dev, value.st_ino)
        if source.path in paths or inode in inodes:
            _error("ALIAS_CONFLICT")
        paths.add(source.path)
        inodes.add(inode)
        selected.append(_Selected(source, _snapshot(value)))
    return _read_set(tuple(sorted(selected, key=lambda s: s.file.path)))


def _fact(fact: Fact, **values) -> str:
    return json.dumps({"contract": CONTRACT, "fact": fact.value, **values},
                      sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _coverage(gid: str, source: SourceObservation) -> m.CoverageRecord:
    reason = (m.StopReason.EOF if source.eof else m.StopReason.ERROR
              if source.stop in (ParseStop.MALFORMED, ParseStop.GZIP_ERROR) else m.StopReason.BUDGET)
    complete = source.eof and source.finding in (ParseFinding.VALID, ParseFinding.EMPTY)
    return m.CoverageRecord(gid, (source.file.id,), m.CoverageMethod.SEQUENTIAL if complete
        else m.CoverageMethod.PREFIX, m.CoverageScope.COMPLETE if complete else m.CoverageScope.SAMPLE,
        source.records, source.decoded_bytes, MAX_RECORDS, MAX_DECODED_BYTES,
        source.eof, reason, source.observed_region_sha256)


def _layout(names: list[FastqName | None], assay: FastqAssay | None,
            declared_layout: FastqLayout | None = None):
    roles = [n.role for n in names if n is not None]
    duplicate = len(roles) != len(set(roles))
    if duplicate or any(n is None for n in names) or assay is None:
        return FastqLayout.UNRESOLVED, duplicate
    required = set(roles) - {ReadRole.I1}
    a = set(LAYOUT_ROLES[FastqLayout.A]) - {ReadRole.I1}
    b = set(LAYOUT_ROLES[FastqLayout.B]) - {ReadRole.I1}
    if required == a and declared_layout in (None, FastqLayout.A):
        return FastqLayout.A, False
    if required == b and declared_layout in (None, FastqLayout.B):
        return FastqLayout.B, False
    if ((required < a and declared_layout in (None, FastqLayout.A))
            or (required < b and declared_layout in (None, FastqLayout.B))):
        return FastqLayout.UNRESOLVED, False
    return FastqLayout.UNSUPPORTED, False


def _missing_roles(names: list[FastqName | None], assay: FastqAssay | None,
                   declared_layout: FastqLayout | None):
    """Exact absent roles, conditional on each admissible incomplete layout.

    R1/R2 alone cannot distinguish missing R3 (A) from missing I2 (B).
    Preserve both alternatives unless a declaration or unique role set decides.
    """
    if assay is None or any(n is None for n in names):
        return ()
    roles = [n.role for n in names]
    if len(roles) != len(set(roles)):
        return ()
    observed = set(roles) - {ReadRole.I1}
    candidates = (declared_layout,) if declared_layout else (FastqLayout.A, FastqLayout.B)
    return tuple((candidate, tuple(sorted(required - observed, key=lambda r: r.value)))
                 for candidate in candidates
                 if observed < (required := set(LAYOUT_ROLES[candidate]) - {ReadRole.I1}))


def inspect_fastq_inputs(inputs: Sequence[str | Path] | str | Path, *,
                         assay: FastqAssay | None = None, species: str | None = None,
                         declared_layout: FastqLayout | None = None) -> m.RawIntakeManifest:
    """Inspect local files/leaf directories and populate the unchanged M10.1 contract.

    Assay is an explicit enum declaration (no guessed or free-text chemistry).
    Species is an optional normalized lowercase declaration; M10.1 owns target
    mapping and unsupported-species readiness. Selection order is immaterial.
    An optional A/B layout declaration applies to every selected group and can
    distinguish incomplete R1/R2 sets. It never substitutes for declaring assay.
    File symlinks resolve to canonical names; directory-discovered symlinks are
    rejected. Explicit directory aliases resolve once, without subtree traversal.
    """
    if assay is not None and type(assay) is not FastqAssay:
        _error("DECLARATION_INVALID")
    if declared_layout is not None and (type(declared_layout) is not FastqLayout
            or declared_layout not in (FastqLayout.A, FastqLayout.B)):
        _error("DECLARATION_INVALID")
    if species is not None and (type(species) is not str or
            re.fullmatch(r"[a-z0-9][a-z0-9_.:+-]{0,127}", species) is None):
        _error("DECLARATION_INVALID")
    inventory = _discover(inputs)
    grouped = {}
    for selected in inventory:
        file = selected.file
        name = _name(Path(file.path).name)
        # Canonical parent is the leaf scope for both explicit and discovered
        # selection. No same-name merging across independent roots or lanes.
        key = ((str(Path(file.path).parent), name.sample, name.sample_number, name.lane, name.chunk)
               if name else (file.path,))
        grouped.setdefault(key, []).append(selected)
    evidence, coverage, issues, groups = [], [], [], []
    for selected_list in grouped.values():
        observed = _read_set(tuple(selected_list))
        sources = observed.sources
        file_ids = tuple(s.file.id for s in sources)
        gid = m.group_identity(m.InputKind.FASTQ, file_ids)
        covers = tuple(_coverage(gid, s) for s in sources)
        coverage.extend(covers)

        def claim(topic, value, *, source=m.EvidenceSource.OBSERVATION, file_id=None, cov=None):
            e = m.EvidenceRecord(source, topic, value, group_id=gid, file_id=file_id,
                coverage_id=cov.id if cov is not None else None)
            evidence.append(e)
            return e.id

        def finding(code, effect, refs=(), files=()):
            issues.append(m.IntakeIssue("FASTQ_" + code, m.Severity.WARNING
                if effect in (m.ReadinessEffect.ADVISORY, m.ReadinessEffect.INFORMATION_REQUIRED)
                else m.Severity.ERROR, gid, effect, tuple(refs), tuple(files)))

        names = [_name(Path(s.file.path).name) for s in sources]
        grouping_refs = []
        for s, c, name in zip(sources, covers, names, strict=True):
            if name:
                grouping_refs.append(claim(m.EvidenceClaim.GROUPING, _fact(Fact.NAME,
                    sample_number=name.sample_number, lane=name.lane, chunk=name.chunk, role=name.role.value),
                    source=m.EvidenceSource.FILENAME, file_id=s.file.id))
            ref = claim(m.EvidenceClaim.STRUCTURE, _fact(Fact.PARSE, compression=s.compression.value,
                suffix_matches=s.suffix_matches, finding=s.finding.value, stop=s.stop.value),
                file_id=s.file.id, cov=c)
            claim(m.EvidenceClaim.STRUCTURE, _fact(Fact.LENGTH, minimum=s.min_length,
                maximum=s.max_length, constant=s.min_length is not None and s.min_length == s.max_length),
                file_id=s.file.id, cov=c)
            claim(m.EvidenceClaim.STRUCTURE, _fact(Fact.LIMITS,
                line_bytes=MAX_LINE_BYTES, encoded_bytes=MAX_ENCODED_BYTES), file_id=s.file.id, cov=c)
            if not s.suffix_matches:
                finding("COMPRESSION_SUFFIX_MISMATCH", m.ReadinessEffect.ADVISORY, (ref,), (s.file.id,))
            if s.finding is ParseFinding.LIMIT:
                finding("INSPECTION_LIMIT", m.ReadinessEffect.INFORMATION_REQUIRED, (ref,), (s.file.id,))
            elif s.finding is not ParseFinding.VALID:
                finding("CONTENT_" + s.finding.name, m.ReadinessEffect.INVALID, (ref,), (s.file.id,))
        if assay is not None:
            claim(m.EvidenceClaim.STRUCTURE, _fact(Fact.ASSAY, value=assay.value),
                  source=m.EvidenceSource.USER_DECLARATION)
        if declared_layout is not None:
            claim(m.EvidenceClaim.STRUCTURE, _fact(Fact.LAYOUT_DECLARATION, value=declared_layout.value),
                  source=m.EvidenceSource.USER_DECLARATION)
        layout, duplicate = _layout(names, assay, declared_layout)
        missing = _missing_roles(names, assay, declared_layout)
        # Every group-wide assessment is linked to every source's bounded
        # observation, not an arbitrarily chosen first file's coverage.
        layout_refs, sync_refs = [], []
        for c in covers:
            layout_refs.append(claim(m.EvidenceClaim.STRUCTURE,
                _fact(Fact.LAYOUT, value=layout.value), cov=c))
            sync_refs.append(claim(m.EvidenceClaim.STRUCTURE, _fact(Fact.SYNC,
                state=observed.synchronization.value, compared=observed.records_compared,
                synchronized=observed.synchronized_records), cov=c))
        if duplicate:
            finding("DUPLICATE_ROLE", m.ReadinessEffect.INVALID, grouping_refs, file_ids)
        if observed.synchronization in (SyncState.NAME_MISMATCH, SyncState.COUNT_MISMATCH):
            finding(observed.synchronization.name, m.ReadinessEffect.INVALID, sync_refs, file_ids)
        if layout is FastqLayout.UNSUPPORTED:
            finding("ROLE_SET_UNSUPPORTED", m.ReadinessEffect.UNSUPPORTED, layout_refs, file_ids)
        elif layout is FastqLayout.UNRESOLVED:
            if missing:
                basis = (MissingRoleBasis.DECLARED if declared_layout else MissingRoleBasis.UNIQUE
                         if len(missing) == 1 else MissingRoleBasis.ALTERNATIVE)
                missing_refs = tuple(claim(m.EvidenceClaim.STRUCTURE, _fact(Fact.MISSING_ROLE,
                    candidate_layout=candidate.value, role=role.value, basis=basis.value), cov=c)
                    for candidate, roles in missing for role in roles for c in covers)
                finding("REQUIRED_ROLE_MISSING", m.ReadinessEffect.INFORMATION_REQUIRED, missing_refs, file_ids)
            if not missing or len(missing) > 1:
                finding("ASSAY_REQUIRED" if assay is None else "LAYOUT_UNRESOLVED",
                        m.ReadinessEffect.INFORMATION_REQUIRED, layout_refs, file_ids)
        current = [i for i in issues if i.group_id == gid]
        effects = {i.effect for i in current}
        structure = (m.StructureState.INVALID if m.ReadinessEffect.INVALID in effects
            else m.StructureState.UNSUPPORTED if m.ReadinessEffect.UNSUPPORTED in effects
            else m.StructureState.UNRESOLVED if m.ReadinessEffect.INFORMATION_REQUIRED in effects
            else m.StructureState.SUPPORTED)
        structure_refs = tuple(claim(m.EvidenceClaim.STRUCTURE, structure.value, cov=c) for c in covers)
        metadata = m.MetadataResolution()
        if species is not None:
            ref = claim(m.EvidenceClaim.SPECIES, species, source=m.EvidenceSource.USER_DECLARATION)
            metadata = m.MetadataResolution((m.MetadataAssertion(species, m.EvidenceSource.USER_DECLARATION, (ref,)),))
        barcode = m.BarcodeProvenance()
        if structure is m.StructureState.SUPPORTED and layout in (FastqLayout.A, FastqLayout.B):
            for s, c, n in zip(sources, covers, names, strict=True):
                claim(m.EvidenceClaim.STRUCTURE, _fact(Fact.READ_MEANING,
                    role=n.role.value, meaning=LAYOUT_ROLES[layout][n.role].value), file_id=s.file.id, cov=c)
            locator = BARCODE_ROLES[layout].value
            refs = tuple(claim(m.EvidenceClaim.BARCODE_SOURCE, "fastq_read:" + locator,
                file_id=s.file.id, cov=c) for s, c, n in zip(sources, covers, names, strict=True)
                if n.role.value == locator)
            barcode = m.BarcodeProvenance(m.BarcodeSource.FASTQ_READ, locator, refs)
        name = names[0]  # Same reviewed grouping key, or an unresolved singleton.
        groups.append(m.GroupRecord(m.InputKind.FASTQ, file_ids, m.IntakeRoute.FASTQ_TO_CCRE,
            grouping_basis=m.GroupingBasis.SYNTACTIC if name else m.GroupingBasis.UNRESOLVED,
            grouping_evidence_ids=tuple(grouping_refs), syntactic_sample_token=name.sample if name else None,
            lane=name.lane if name else None, chunk=name.chunk if name else None,
            structure=structure, structure_evidence_ids=structure_refs, species=metadata,
            source_genome_assembly=m.MetadataResolution(applicable=False), barcode=barcode))
    for selected in inventory:
        _unchanged(Path(selected.file.path), selected.snapshot)
    return m.build_raw_intake_manifest(files=tuple(s.file for s in inventory), groups=tuple(groups),
        evidence=tuple(evidence), coverage=tuple(coverage), issues=tuple(issues),
        inspection=m.InspectionConfiguration(backends=(m.BackendIdentity("raw-fastq", CONTRACT),)))
