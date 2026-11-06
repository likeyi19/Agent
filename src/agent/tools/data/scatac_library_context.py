"""M11.1b library/barcode decisions, independent of genomic references.

M10 owns group identity, library evidence and barcode extraction facts. This
artifact projects source/locator facts only through exact M10 binding checks;
it defines no FASTQ layout or BAM producer interpretation rules. All namespace
and barcode-interpretation decisions are explicit caller declarations. A valid
context is not an executable-backend qualification or an M10 readiness override.

Group membership and library collections are sets, canonically sorted by group
ID and namespace respectively. Cell identity is the tuple (namespace, barcode
identifier); no rendered cell-ID syntax is defined here. Explicit subset mode
allows omitted groups, including omitted lanes, but never splits a selected
authoritative M10 library across namespaces.

Whitelist v1 accepts plain ASCII uppercase A/C/G/T, one uniform-length token
(1..256 bases) per line. LF is the only separator; the final LF is optional.
Empty/duplicate records, CR, spaces, suffixes and other alphabets fail. File
SHA-256 covers all bytes. Candidate-set SHA-256 covers ASCII tokens sorted
lexically, each followed by LF, with no prefix. This mathematical set identity
does not promise that a future backend is insensitive to input file order.

Canonical JSON: UTF-8, sorted object keys, compact separators, ensure_ascii=False,
allow_nan=False, no final newline. Context SHA-256 covers the bytes after
excluding its own digest, intake path, and whitelist paths/provenance, prefixed
by ASCII 'agent.scatac-library-processing-context.v1' and one NUL. The separate
manifest SHA-256 covers every published byte, including paths/provenance.

Loading/validation read no other artifact or resource. Build reads M10 JSON and
freshly reinspects supplied whitelist identities; binding validation reads only
M10 JSON. Neither opens FASTQ/BAM. Whitelist reinspection is also a separate API.
Publication checks M10 binding and protects declared input/resource paths, then
uses sibling staging, fsync, validation and atomic no-clobber link or replacement.
As in M11.1a, this assumes a trusted local filesystem, not hostile path races.
"""

from dataclasses import asdict, dataclass, fields, replace
from enum import Enum
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from typing import TypedDict

from . import raw_scatac_manifest as intake
from .scatac_reference import ResourceIdentity, SourceProvenance


LIBRARY_CONTEXT_ARTIFACT_TYPE = "agent.scatac-library-processing-context"
LIBRARY_CONTEXT_SCHEMA_VERSION = 1
LIBRARY_CONTEXT_CONTRACT_VERSION = "scatac-library-processing-context.v1"
MAX_CONTEXT_BYTES = 4 * 1024 * 1024
MAX_GROUPS = intake.MAX_COLLECTION_ITEMS
MAX_BARCODE_LENGTH = 256
WHITELIST_SET_ALGORITHM = "sha256-ascii-sorted-lf.v1"
WHITELIST_SYNTAX = "uppercase-acgt-uniform.v1"


class LibraryContextError(ValueError):
    """Stable data/artifact error code; not a runtime retry disposition."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


class MembershipBasis(str, Enum):
    SINGLE_GROUP = "single_group"
    INTAKE_LIBRARY_ID = "intake_library_id"
    CALLER_SHARED_LIBRARY = "caller_shared_library"


class SelectionMode(str, Enum):
    ALL_GROUPS = "all_groups"
    EXPLICIT_SUBSET = "explicit_subset"


class BarcodeInterpretation(str, Enum):
    RAW_SEQUENCE = "raw_sequence"
    CORRECTED_IDENTIFIER = "corrected_identifier"


class CorrectionPolicy(str, Enum):
    WHITELIST_REQUIRED = "whitelist_required"
    ALREADY_CORRECTED = "already_corrected"
    UNQUALIFIED_RAW_BAM = "unqualified_raw_bam"


@dataclass(frozen=True)
class BarcodeWhitelistIdentity:
    resource: ResourceIdentity
    n_barcodes: int
    barcode_length: int
    barcode_set_sha256: str
    set_identity_algorithm: str = WHITELIST_SET_ALGORITHM
    syntax: str = WHITELIST_SYNTAX


@dataclass(frozen=True)
class IntakeBinding:
    manifest_path: str
    manifest_sha256: str
    artifact_type: str = intake.RAW_INTAKE_ARTIFACT_TYPE
    schema_version: int = intake.RAW_INTAKE_SCHEMA_VERSION
    contract_version: str = intake.RAW_INTAKE_CONTRACT_VERSION


@dataclass(frozen=True)
class GroupBarcodeBinding:
    group_id: str
    source: intake.BarcodeSource
    locator: str


@dataclass(frozen=True)
class LibraryDeclaration:
    """Caller choices. Optional source assertions must match M10 exactly."""

    namespace: str
    group_ids: tuple[str, ...]
    membership_basis: MembershipBasis
    barcode_interpretation: BarcodeInterpretation
    correction_policy: CorrectionPolicy
    whitelist: BarcodeWhitelistIdentity | None = None
    declared_barcode_length: int | None = None
    expected_barcode_bindings: tuple[GroupBarcodeBinding, ...] | None = None


@dataclass(frozen=True)
class ProcessingLibrary:
    namespace: str
    groups: tuple[GroupBarcodeBinding, ...]
    input_kind: intake.InputKind
    source_library_id: str | None
    membership_basis: MembershipBasis
    barcode_interpretation: BarcodeInterpretation
    correction_policy: CorrectionPolicy
    whitelist: BarcodeWhitelistIdentity | None = None
    declared_barcode_length: int | None = None
    namespace_basis: str = "caller_declaration"
    interpretation_basis: str = "caller_declaration"


@dataclass(frozen=True)
class LibraryProcessingContext:
    intake: IntakeBinding
    libraries: tuple[ProcessingLibrary, ...]
    selection_mode: SelectionMode
    context_identity_sha256: str
    artifact_type: str = LIBRARY_CONTEXT_ARTIFACT_TYPE
    schema_version: int = LIBRARY_CONTEXT_SCHEMA_VERSION
    contract_version: str = LIBRARY_CONTEXT_CONTRACT_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


class LibraryContextManifestReference(TypedDict):
    manifest_path: str
    manifest_sha256: str
    artifact_schema_version: int


def _fail(message, code="LIBRARY_CONTEXT_INVALID"):
    raise LibraryContextError(code, message)


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _shape(value, cls):
    if type(value) is not dict or set(value) != {f.name for f in fields(cls)}:
        _fail("Missing or unknown context record fields.")
    return dict(value)


def _text(value, maximum=4096):
    if (type(value) is not str or not value or value != value.strip()
            or len(value) > maximum or any(ord(c) < 32 or ord(c) == 127 for c in value)):
        _fail("Invalid context text.")
    value.encode("utf-8")


def _namespace(value):
    # Safe explicit labels, not file/group hashes or a final cell-ID encoding.
    if type(value) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value) is None:
        _fail("Namespace requires an explicit safe label.", "LIBRARY_NAMESPACE_INVALID")


def _path_text(value):
    _text(value)
    if (not value.startswith("/") or value.startswith("//") or ".." in PurePosixPath(value).parts
            or str(PurePosixPath(value)) != value):
        _fail("Artifact/resource path must be normalized and absolute.")


def _sha(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        _fail("Invalid SHA-256 identity.")


def _positive(value, maximum=2**63 - 1):
    if type(value) is not int or not 0 < value <= maximum:
        _fail("Invalid positive integer.")


def _collection(value):
    if type(value) not in (list, tuple) or not 0 < len(value) <= MAX_GROUPS:
        _fail("Expected a nonempty bounded collection.")
    return value


def _group_ids(value):
    ids = _collection(value)
    for gid in ids:
        if type(gid) is not str or re.fullmatch(r"group:[0-9a-f]{64}", gid) is None:
            _fail("Invalid M10 group identity.", "LIBRARY_GROUP_INVALID")
    if len(set(ids)) != len(ids):
        _fail("Duplicate group membership.", "LIBRARY_GROUP_INVALID")
    return tuple(sorted(ids))


def _provenance(value):
    p = SourceProvenance(**_shape(value, SourceProvenance))
    claims = (p.source, p.accession, p.citation)
    for claim in claims:
        if claim is not None:
            _text(claim)
    if not ((p.basis == "unknown" and all(c is None for c in claims))
            or (p.basis == "caller_supplied" and any(c is not None for c in claims))):
        _fail("Historical provenance must be unknown or an explicit caller claim.")
    return p


def _whitelist(value):
    d = _shape(value, BarcodeWhitelistIdentity)
    r = _shape(d["resource"], ResourceIdentity)
    _path_text(r["path"])
    _sha(r["sha256"])
    r["provenance"] = _provenance(r["provenance"])
    d["resource"] = ResourceIdentity(**r)
    _positive(d["n_barcodes"])
    _positive(d["barcode_length"], MAX_BARCODE_LENGTH)
    _sha(d["barcode_set_sha256"])
    if d["syntax"] != WHITELIST_SYNTAX or d["set_identity_algorithm"] != WHITELIST_SET_ALGORITHM:
        _fail("Unsupported whitelist syntax or identity algorithm.")
    return BarcodeWhitelistIdentity(**d)


def _group_binding(value):
    d = _shape(value, GroupBarcodeBinding)
    _group_ids((d["group_id"],))
    d["source"] = intake.BarcodeSource(d["source"])
    if d["source"] is intake.BarcodeSource.UNKNOWN:
        _fail("Barcode source is unresolved.", "LIBRARY_BARCODE_INVALID")
    _text(d["locator"], 128)
    return GroupBarcodeBinding(**d)


def _library(value):
    d = _shape(value, ProcessingLibrary)
    _namespace(d["namespace"])
    groups = tuple(_group_binding(g) for g in _collection(d["groups"]))
    _group_ids(tuple(g.group_id for g in groups))
    d["groups"] = tuple(sorted(groups, key=lambda g: g.group_id))
    d["input_kind"] = intake.InputKind(d["input_kind"])
    d["membership_basis"] = MembershipBasis(d["membership_basis"])
    d["barcode_interpretation"] = BarcodeInterpretation(d["barcode_interpretation"])
    d["correction_policy"] = CorrectionPolicy(d["correction_policy"])
    if d["namespace_basis"] != "caller_declaration" or d["interpretation_basis"] != "caller_declaration":
        _fail("Namespace and interpretation require explicit caller declarations.")
    if d["source_library_id"] is not None:
        _text(d["source_library_id"], 128)
    basis = d["membership_basis"]
    if ((basis is MembershipBasis.SINGLE_GROUP and len(groups) != 1)
            or (basis is MembershipBasis.CALLER_SHARED_LIBRARY and len(groups) < 2)
            or (basis is MembershipBasis.INTAKE_LIBRARY_ID and d["source_library_id"] is None)):
        _fail("Invalid library membership authority.", "LIBRARY_MEMBERSHIP_INVALID")
    if d["whitelist"] is not None:
        d["whitelist"] = _whitelist(d["whitelist"])
    length = d["declared_barcode_length"]
    if length is not None:
        _positive(length, MAX_BARCODE_LENGTH)
        if d["whitelist"] is not None and length != d["whitelist"].barcode_length:
            _fail("Declared barcode length contradicts the whitelist.", "LIBRARY_BARCODE_INVALID")
    raw = d["barcode_interpretation"] is BarcodeInterpretation.RAW_SEQUENCE
    policy = d["correction_policy"]
    whitelist = d["whitelist"]
    if d["input_kind"] is intake.InputKind.FASTQ:
        valid = (all(g.source is intake.BarcodeSource.FASTQ_READ for g in groups)
                 and raw and policy is CorrectionPolicy.WHITELIST_REQUIRED and whitelist is not None)
    elif d["input_kind"] is intake.InputKind.BAM:
        valid = (all(g.source in (intake.BarcodeSource.BAM_CELL_IDENTIFIER,
                                  intake.BarcodeSource.BAM_RAW_SEQUENCE) for g in groups)
                 and ((raw and policy is CorrectionPolicy.UNQUALIFIED_RAW_BAM)
                      or (not raw and policy is CorrectionPolicy.ALREADY_CORRECTED and whitelist is None)))
    else:
        valid = False
    if not valid:
        _fail("Unsupported barcode interpretation/correction combination.", "LIBRARY_BARCODE_INVALID")
    return ProcessingLibrary(**d)


def _context_digest(context):
    value = context.to_dict()
    del value["context_identity_sha256"]
    del value["intake"]["manifest_path"]
    for library in value["libraries"]:
        if library["whitelist"] is not None:
            del library["whitelist"]["resource"]["path"]
            del library["whitelist"]["resource"]["provenance"]
    return hashlib.sha256(b"agent.scatac-library-processing-context.v1\0" + _json_bytes(value)).hexdigest()


def _parse_context(value, *, check_digest):
    d = _shape(value.to_dict() if type(value) is LibraryProcessingContext else value, LibraryProcessingContext)
    if len(_json_bytes(d)) > MAX_CONTEXT_BYTES:
        _fail("Context byte limit exceeded.")
    if (d["artifact_type"] != LIBRARY_CONTEXT_ARTIFACT_TYPE
            or type(d["schema_version"]) is not int or d["schema_version"] != 1
            or d["contract_version"] != LIBRARY_CONTEXT_CONTRACT_VERSION):
        _fail("Unsupported library context version.")
    b = _shape(d["intake"], IntakeBinding)
    _path_text(b["manifest_path"])
    _sha(b["manifest_sha256"])
    if (b["artifact_type"] != intake.RAW_INTAKE_ARTIFACT_TYPE
            or type(b["schema_version"]) is not int or b["schema_version"] != intake.RAW_INTAKE_SCHEMA_VERSION
            or b["contract_version"] != intake.RAW_INTAKE_CONTRACT_VERSION):
        _fail("Unsupported intake artifact version.", "LIBRARY_INTAKE_INVALID")
    d["intake"] = IntakeBinding(**b)
    d["selection_mode"] = SelectionMode(d["selection_mode"])
    libraries = tuple(_library(v) for v in _collection(d["libraries"]))
    namespaces = [lib.namespace for lib in libraries]
    if len(set(namespaces)) != len(namespaces):
        _fail("Independent libraries require unique namespaces.", "LIBRARY_NAMESPACE_INVALID")
    _group_ids(tuple(g.group_id for lib in libraries for g in lib.groups))
    known = [lib.source_library_id for lib in libraries if lib.source_library_id is not None]
    if len(set(known)) != len(known):
        _fail("A known shared library cannot span independent namespaces.", "LIBRARY_MEMBERSHIP_INVALID")
    d["libraries"] = tuple(sorted(libraries, key=lambda lib: lib.namespace))
    context = LibraryProcessingContext(**d)
    if check_digest:
        _sha(context.context_identity_sha256)
        if context.context_identity_sha256 != _context_digest(context):
            _fail("Context identity disagrees with its fields.", "LIBRARY_DIGEST_MISMATCH")
    return context


def validate_scatac_library_processing_context(value: object) -> LibraryProcessingContext:
    """Lightweight domain validation only; source facts require explicit binding."""
    try:
        return _parse_context(value, check_digest=True)
    except LibraryContextError:
        raise
    except (ValueError, TypeError, AttributeError, RecursionError) as exc:
        raise LibraryContextError("LIBRARY_CONTEXT_INVALID", "Invalid library context.") from exc


def _local_path(value):
    if not isinstance(value, (str, Path)):
        _fail("Expected a local path.")
    _text(str(value))
    if "://" in str(value):
        _fail("Expected a local path.")
    p = Path(value).expanduser()
    if p.is_symlink():
        _fail("Symbolic resource/artifact paths are unsupported.")
    p = p.resolve(strict=True)
    _path_text(str(p))
    _snapshot(p)
    return p


def _snapshot(path):
    s = path.stat()
    if not stat.S_ISREG(s.st_mode) or s.st_size <= 0:
        _fail("Expected a nonempty regular file.", "LIBRARY_RESOURCE_INVALID")
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def _duplicate_free(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail("Duplicate JSON key.", "LIBRARY_JSON_INVALID")
        result[key] = value
    return result


def _reject_constant(value):
    _fail("Nonfinite JSON number.", "LIBRARY_JSON_INVALID")


def load_scatac_library_processing_context(path: str | Path, *, expected_sha256: str | None = None
                                         ) -> tuple[Path, LibraryProcessingContext, str]:
    """Read only bounded context JSON; no M10 artifact, whitelist or raw IO."""
    try:
        if expected_sha256 is not None:
            _sha(expected_sha256)
        resolved = _local_path(path)
        with resolved.open("rb") as handle:
            payload = handle.read(MAX_CONTEXT_BYTES + 1)
        if len(payload) > MAX_CONTEXT_BYTES:
            _fail("Context byte limit exceeded.", "LIBRARY_JSON_INVALID")
        digest = hashlib.sha256(payload).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            _fail("Context manifest byte identity changed.", "LIBRARY_DIGEST_MISMATCH")
        data = json.loads(payload.decode("utf-8"), object_pairs_hook=_duplicate_free,
                          parse_constant=_reject_constant)
        return resolved, validate_scatac_library_processing_context(data), digest
    except LibraryContextError:
        raise
    except (OSError, ValueError, TypeError, RecursionError, RuntimeError) as exc:
        raise LibraryContextError("LIBRARY_JSON_INVALID", "Context is not readable strict UTF-8 JSON.") from exc
