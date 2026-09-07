"""M10.1 raw sequencing intake domain contract (no file-format inspection).

Readiness describes entry into a future preprocessing route, never execution
success, QC success, or model compatibility. Constructors are convenient typed
records; ``build_raw_intake_manifest`` / ``validate_raw_intake_manifest`` are the
validation boundaries. All collection fields are unordered sets of records or
references, serialized in canonical order; none may contain sequencing vectors.

Source paths and observations are supplied by future inspectors. This module
does not discover, open, stat, hash, or interpret those source files. Loading
validates domain consistency, not the truth of scientific observations. Source
revalidation will be a separate M10 capability.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from types import UnionType
from typing import TypedDict, get_args, get_origin, get_type_hints
import unicodedata


RAW_INTAKE_ARTIFACT_TYPE = "agent.raw-scatac-intake"
RAW_INTAKE_SCHEMA_VERSION = 1
RAW_INTAKE_CONTRACT_VERSION = "raw-scatac-intake.v1"
INSPECTION_CONTRACT_VERSION = "raw-scatac-inspection.v1"
INSPECTION_BUDGET_VERSION = "raw-scatac-inspection-budget.v1"
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_COLLECTION_ITEMS = 4096
MAX_TREE_DEPTH = 24


class RawIntakeManifestError(ValueError):
    """Stable domain/artifact error; never a runtime recovery disposition."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class InputKind(str, Enum):
    FASTQ = "fastq"
    BAM = "bam"
    UNKNOWN = "unknown"
    OTHER = "other"


class IntakeRoute(str, Enum):
    FASTQ_TO_CCRE = "fastq-to-cell-by-ccre.v1"
    BAM_TO_CCRE = "bam-to-cell-by-ccre.v1"


class Readiness(str, Enum):
    READY = "READY"
    READY_WITH_REPAIRS = "READY_WITH_REPAIRS"
    NEEDS_USER_INPUT = "NEEDS_USER_INPUT"
    UNSUPPORTED = "UNSUPPORTED"
    INVALID = "INVALID"


# Explicit ordering, independent of enum declaration or lexical ordering.
READINESS_PRECEDENCE = (
    Readiness.READY, Readiness.READY_WITH_REPAIRS, Readiness.NEEDS_USER_INPUT,
    Readiness.UNSUPPORTED, Readiness.INVALID,
)


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ReadinessEffect(str, Enum):
    ADVISORY = "advisory"
    REPAIR_REQUIRED = "repair_required"
    INFORMATION_REQUIRED = "information_required"
    UNSUPPORTED = "unsupported"
    INVALID = "invalid"


class EvidenceSource(str, Enum):
    USER_DECLARATION = "user_declaration"
    AUTHORITATIVE_METADATA = "authoritative_metadata"
    OBSERVATION = "observation"
    FILENAME = "filename"


class EvidenceClaim(str, Enum):
    SPECIES = "species"
    SOURCE_ASSEMBLY = "source_genome_assembly"
    HISTORICAL_REFERENCE = "historical_reference"
    INPUT_BINDING = "input_binding"
    GROUPING = "grouping"
    LIBRARY = "library"
    STRUCTURE = "structure"
    BARCODE_SOURCE = "barcode_source"
    BARCODE_QUALITIES = "barcode_qualities"
    BARCODE_NAMESPACE = "barcode_namespace"
    BARCODE_PRODUCER = "barcode_producer"
    PREPARATION_ADMISSIBILITY = "preparation_admissibility"


class ResolutionState(str, Enum):
    RESOLVED = "resolved"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"
    NOT_APPLICABLE = "not_applicable"


class TargetState(str, Enum):
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    UNSUPPORTED = "unsupported"


class AssemblyCompatibility(str, Enum):
    NOT_APPLICABLE = "not_applicable"
    MATCH = "match"
    MISMATCH = "mismatch"
    SOURCE_UNKNOWN = "source_unknown"
    SOURCE_CONFLICT = "source_conflict"
    TARGET_UNRESOLVED = "target_unresolved"
    UNSUPPORTED = "unsupported"


class StructureState(str, Enum):
    UNRESOLVED = "unresolved"
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    INVALID = "invalid"


class SelectionBasis(str, Enum):
    EXPLICIT = "explicit"
    DIRECTORY_DISCOVERY = "directory_discovery"


class GroupingBasis(str, Enum):
    UNRESOLVED = "unresolved"
    EXPLICIT = "explicit"
    SYNTACTIC = "syntactic"
    AUTHORITATIVE_METADATA = "authoritative_metadata"


class BarcodeSource(str, Enum):
    UNKNOWN = "unknown"
    FASTQ_READ = "fastq_read"
    BAM_CELL_IDENTIFIER = "bam_cell_identifier"
    BAM_RAW_SEQUENCE = "bam_raw_sequence"


class BarcodeIdentityScope(str, Enum):
    GROUP_LOCAL = "group_local"
    UNRESOLVED = "unresolved"


class InformationCode(str, Enum):
    SPECIES = "species"
    SOURCE_ASSEMBLY = "source_genome_assembly"
    ROUTE = "route"
    GROUPING = "grouping"
    STRUCTURE = "structure"
    BARCODE_SOURCE = "barcode_source"
    BARCODE_NAMESPACE = "barcode_namespace"
    BARCODE_IDENTITY_SCOPE = "barcode_identity_scope"
    INSPECTION_COVERAGE = "inspection_coverage"
    ROUTE_ADMISSIBILITY = "route_admissibility"


class PreparationCode(str, Enum):
    ASSEMBLY_HARMONIZATION = "assembly_harmonization"
    INPUT_PREPARATION = "input_preparation"


class PreparationAdmissibility(str, Enum):
    UNRESOLVED = "unresolved"
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"


class PrerequisiteCode(str, Enum):
    TARGET_REFERENCE = "target_reference"
    ALIGNMENT = "alignment"
    BARCODE_PROCESSING = "barcode_processing"
    CELL_BY_CCRE = "cell_by_ccre"


class CoverageMethod(str, Enum):
    NOT_INSPECTED = "not_inspected"
    METADATA = "metadata"
    PREFIX = "prefix"
    SEQUENTIAL = "sequential"


class CoverageScope(str, Enum):
    NONE = "none"
    SAMPLE = "sample"
    COMPLETE = "complete"


class StopReason(str, Enum):
    NOT_STARTED = "not_started"
    BUDGET = "budget"
    EOF = "eof"
    ERROR = "error"


def _fail(message: str, code: str = "RAW_INTAKE_CONTRACT_INVALID") -> None:
    raise RawIntakeManifestError(code, message)


def _plain(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        if not all(type(k) is str for k in value):
            _fail("Manifest object keys must be strings.")
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return sorted((_plain(v) for v in value), key=_json_bytes)
    if value is None or type(value) in (str, int, bool):
        return value
    _fail("Manifest contains an unsupported JSON value.")


def _json_bytes(value: object) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False,
                          sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise RawIntakeManifestError(
            "RAW_INTAKE_CONTRACT_INVALID", "Manifest is not canonical JSON."
        ) from exc


def _identity(domain: str, value: object) -> str:
    return domain + ":" + hashlib.sha256(
        (RAW_INTAKE_CONTRACT_VERSION + ":" + domain).encode() + b"\0"
        + _json_bytes(_plain(value))
    ).hexdigest()


def file_identity(path: str) -> str:
    """Identity of an already resolved path; performs no filesystem access."""
    _path(path)
    return _identity("file", path)


def group_identity(input_kind: InputKind, file_ids: tuple[str, ...]) -> str:
    """Membership identity, never a biological sample or cell identity."""
    return _identity("group", (input_kind, tuple(sorted(file_ids))))


@dataclass(frozen=True)
class FileRecord:
    path: str
    input_kind: InputKind
    size_bytes: int | None = None
    mtime_ns: int | None = None
    selection: SelectionBasis = SelectionBasis.EXPLICIT
    selection_root: str | None = None
    binding_evidence_ids: tuple[str, ...] = ()
    id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", file_identity(self.path))


@dataclass(frozen=True)
class EvidenceRecord:
    source: EvidenceSource
    claim: EvidenceClaim
    value: str
    group_id: str | None = None
    file_id: str | None = None
    coverage_id: str | None = None
    id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _identity("evidence", {
            f.name: getattr(self, f.name) for f in fields(self) if f.init
        }))


@dataclass(frozen=True)
class MetadataAssertion:
    value: str
    source: EvidenceSource
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class MetadataResolution:
    assertions: tuple[MetadataAssertion, ...] = ()
    applicable: bool = True
    state: ResolutionState = field(init=False)
    value: str | None = field(init=False)

    def __post_init__(self) -> None:
        values = {a.value for a in self.assertions}
        authoritative = any(a.source in (
            EvidenceSource.USER_DECLARATION, EvidenceSource.AUTHORITATIVE_METADATA
        ) for a in self.assertions)
        # Contradictions remain visible even across authority levels. Filename
        # and unreviewed observations cannot establish exact species/build.
        state = (ResolutionState.NOT_APPLICABLE if not self.applicable
                 else ResolutionState.CONFLICT if len(values) > 1
                 else ResolutionState.RESOLVED if values and authoritative
                 else ResolutionState.UNKNOWN)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "value", next(iter(values))
                           if state is ResolutionState.RESOLVED else None)


@dataclass(frozen=True)
class BarcodeProvenance:
    source: BarcodeSource = BarcodeSource.UNKNOWN
    locator: str | None = None
    evidence_ids: tuple[str, ...] = ()
    quality_evidence_ids: tuple[str, ...] = ()
    producer: str | None = None
    producer_evidence_ids: tuple[str, ...] = ()
    namespace: str | None = None
    namespace_evidence_ids: tuple[str, ...] = ()
    identity_scope: BarcodeIdentityScope = BarcodeIdentityScope.GROUP_LOCAL
    # An absent external namespace label is harmless when identifiers are
    # scoped to their independent group. Cross-group composition must explicitly
    # mark unresolved scope when that separation is not sufficient; labels alone
    # never authorize merging or establish a shared biological cell identity.
    # No corrected/canonical flag: neither raw sequence nor a generic cell
    # identifier proves correction. No barcode sequence or quality is stored.


@dataclass(frozen=True)
class TargetAssembly:
    state: TargetState
    value: str | None


@dataclass(frozen=True)
class GroupRecord:
    input_kind: InputKind
    file_ids: tuple[str, ...]
    route: IntakeRoute | None = None
    grouping_basis: GroupingBasis = GroupingBasis.UNRESOLVED
    grouping_evidence_ids: tuple[str, ...] = ()
    syntactic_sample_token: str | None = None
    lane: str | None = None
    chunk: str | None = None
    library_id: str | None = None
    library_evidence_ids: tuple[str, ...] = ()
    structure: StructureState = StructureState.UNRESOLVED
    structure_evidence_ids: tuple[str, ...] = ()
    species: MetadataResolution = field(default_factory=MetadataResolution)
    source_genome_assembly: MetadataResolution = field(default_factory=MetadataResolution)
    barcode: BarcodeProvenance = field(default_factory=BarcodeProvenance)
    id: str = field(init=False)
    target_genome_assembly: TargetAssembly = field(init=False)
    assembly_compatibility: AssemblyCompatibility = field(init=False)
    harmonization_required: bool = field(init=False)
    barcode_identity_scope_id: str | None = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", group_identity(self.input_kind, self.file_ids))
        object.__setattr__(self, "barcode_identity_scope_id",
                           _identity("barcode-scope", self.id)
                           if self.barcode.identity_scope is BarcodeIdentityScope.GROUP_LOCAL
                           else None)
        species = self.species.value
        target = TargetAssembly(TargetState.UNRESOLVED, None)
        if self.input_kind is InputKind.OTHER:
            target = TargetAssembly(TargetState.UNSUPPORTED, None)
        elif species is not None and species not in ("human", "mouse"):
            target = TargetAssembly(TargetState.UNSUPPORTED, None)
        elif self.route is not None and species is not None:
            target = TargetAssembly(TargetState.RESOLVED,
                                    {"human": "hg38", "mouse": "mm10"}[species])
        source = self.source_genome_assembly
        harmonize = (self.input_kind is InputKind.BAM and source.value is not None
                     and target.value is not None and source.value != target.value)
        if self.input_kind is InputKind.FASTQ:
            # FASTQ has no aligned genomic coordinates. Historical references
            # belong in evidence, never in its source coordinate assembly.
            compatibility = AssemblyCompatibility.NOT_APPLICABLE
        elif source.state is ResolutionState.CONFLICT:
            compatibility = AssemblyCompatibility.SOURCE_CONFLICT
        elif target.state is TargetState.UNSUPPORTED:
            compatibility = AssemblyCompatibility.UNSUPPORTED
        elif target.value is None:
            compatibility = AssemblyCompatibility.TARGET_UNRESOLVED
        elif source.value is None:
            compatibility = AssemblyCompatibility.SOURCE_UNKNOWN
        else:
            compatibility = (AssemblyCompatibility.MATCH if source.value == target.value
                             else AssemblyCompatibility.MISMATCH)
        object.__setattr__(self, "target_genome_assembly", target)
        object.__setattr__(self, "assembly_compatibility", compatibility)
        object.__setattr__(self, "harmonization_required", harmonize)


@dataclass(frozen=True)
class CoverageRecord:
    group_id: str
    file_ids: tuple[str, ...]
    method: CoverageMethod
    scope: CoverageScope
    records_inspected: int
    decoded_bytes_inspected: int | None
    record_limit: int | None
    decoded_byte_limit: int | None
    eof_observed: bool
    stop_reason: StopReason
    observed_region_sha256: str | None = None
    id: str = field(init=False)

    def __post_init__(self) -> None:
        # This digest, if supplied, binds only the declared observed region;
        # even complete record checks are not a hash of physical source bytes.
        object.__setattr__(self, "id", _identity("coverage", {
            f.name: getattr(self, f.name) for f in fields(self) if f.init
        }))


@dataclass(frozen=True)
class RequiredInformation:
    group_id: str
    code: InformationCode
    evidence_ids: tuple[str, ...] = ()
    id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _identity("information", {
            f.name: getattr(self, f.name) for f in fields(self) if f.init
        }))


@dataclass(frozen=True)
class PreparationRequirement:
    group_id: str
    code: PreparationCode
    evidence_ids: tuple[str, ...] = ()
    admissibility: PreparationAdmissibility = PreparationAdmissibility.UNRESOLVED
    route: IntakeRoute | None = None
    admissibility_evidence_ids: tuple[str, ...] = ()
    id: str = field(init=False)

    def __post_init__(self) -> None:
        # A harmonization requirement is not an executable strategy. M11 must
        # establish read/cell recovery, prefer original FASTQ and consider
        # read-level realignment. Direct aligned-BAM liftOver is not a default.
        object.__setattr__(self, "id", _identity("preparation", {
            f.name: getattr(self, f.name) for f in fields(self) if f.init
        }))


@dataclass(frozen=True)
class DownstreamPrerequisite:
    group_id: str
    code: PrerequisiteCode
    evidence_ids: tuple[str, ...] = ()
    id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _identity("prerequisite", {
            f.name: getattr(self, f.name) for f in fields(self) if f.init
        }))


@dataclass(frozen=True)
class IntakeIssue:
    code: str
    severity: Severity
    group_id: str
    effect: ReadinessEffect
    evidence_ids: tuple[str, ...] = ()
    file_ids: tuple[str, ...] = ()
    requirement_ids: tuple[str, ...] = ()
    message: str | None = None
    id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _identity("issue", {
            f.name: getattr(self, f.name) for f in fields(self) if f.init
        }))


@dataclass(frozen=True)
class BackendIdentity:
    name: str
    version: str


@dataclass(frozen=True)
class InspectionConfiguration:
    contract_version: str = INSPECTION_CONTRACT_VERSION
    budget_version: str = INSPECTION_BUDGET_VERSION
    backends: tuple[BackendIdentity, ...] = ()


@dataclass(frozen=True)
class GroupReadiness:
    group_id: str
    readiness: Readiness


_EFFECT_STATES = {
    ReadinessEffect.ADVISORY: Readiness.READY,
    ReadinessEffect.REPAIR_REQUIRED: Readiness.READY_WITH_REPAIRS,
    ReadinessEffect.INFORMATION_REQUIRED: Readiness.NEEDS_USER_INPUT,
    ReadinessEffect.UNSUPPORTED: Readiness.UNSUPPORTED,
    ReadinessEffect.INVALID: Readiness.INVALID,
}


def _maximum(states: tuple[Readiness, ...]) -> Readiness:
    return max(states, key=READINESS_PRECEDENCE.index, default=Readiness.READY)


@dataclass(frozen=True)
class RawIntakeManifest:
    files: tuple[FileRecord, ...]
    groups: tuple[GroupRecord, ...]
    evidence: tuple[EvidenceRecord, ...] = ()
    coverage: tuple[CoverageRecord, ...] = ()
    issues: tuple[IntakeIssue, ...] = ()
    required_information: tuple[RequiredInformation, ...] = ()
    repairs: tuple[PreparationRequirement, ...] = ()
    prerequisites: tuple[DownstreamPrerequisite, ...] = ()
    inspection: InspectionConfiguration = field(default_factory=InspectionConfiguration)
    artifact_type: str = RAW_INTAKE_ARTIFACT_TYPE
    schema_version: int = RAW_INTAKE_SCHEMA_VERSION
    intake_contract_version: str = RAW_INTAKE_CONTRACT_VERSION
    group_readiness: tuple[GroupReadiness, ...] = field(init=False)
    readiness: Readiness = field(init=False)

    def __post_init__(self) -> None:
        summaries = []
        for group in self.groups:
            states = [_EFFECT_STATES[i.effect] for i in self.issues if i.group_id == group.id]
            if any(r.group_id == group.id for r in self.required_information):
                states.append(Readiness.NEEDS_USER_INPUT)
            for repair in self.repairs:
                if repair.group_id == group.id:
                    states.append({
                        PreparationAdmissibility.UNRESOLVED: Readiness.NEEDS_USER_INPUT,
                        PreparationAdmissibility.SUPPORTED: Readiness.READY_WITH_REPAIRS,
                        PreparationAdmissibility.UNSUPPORTED: Readiness.UNSUPPORTED,
                    }[repair.admissibility])
            summaries.append(GroupReadiness(group.id, _maximum(tuple(states))))
        object.__setattr__(self, "group_readiness", tuple(summaries))
        object.__setattr__(self, "readiness", _maximum(tuple(s.readiness for s in summaries)))

    def to_dict(self) -> dict[str, object]:
        """Fresh JSON-compatible canonical projection; does not certify sources."""
        return _plain(self)


class ManifestReference(TypedDict):
    manifest_path: str
    manifest_sha256: str
    artifact_schema_version: int


def _text(value: str, maximum: int = 256) -> None:
    if (not value or value != value.strip() or len(value) > maximum
            or any(unicodedata.category(c).startswith("C") for c in value)):
        _fail("Text must be bounded, nonblank, trimmed, and free of control characters.")


def _token(value: str) -> None:
    _text(value, 128)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:+-]*", value) is None:
        _fail("Expected a bounded scientific identifier.")


def _path(value: str) -> None:
    if type(value) is not str:
        _fail("A source path must be a string.")
    _text(value, 4096)
    path = PurePosixPath(value)
    if not value.startswith("/") or value.startswith("//") or str(path) != value or ".." in path.parts:
        _fail("Source paths must already be normalized absolute local paths.")


def _decode(value: object, annotation: object, depth: int = 0) -> object:
    """Strict decoder for these local closed dataclass records only.

    Derived fields are checked, not deserialized as constructor authority.
    This deliberately supports only the types used by the M10.1 records.
    """
    if depth > MAX_TREE_DEPTH:
        _fail("Manifest nesting limit exceeded.")
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is UnionType:
        if value is None and type(None) in args:
            return None
        return _decode(value, next(a for a in args if a is not type(None)), depth + 1)
    if origin is tuple:
        if not isinstance(value, (tuple, list)) or len(value) > MAX_COLLECTION_ITEMS:
            _fail("Expected a bounded record/reference collection.")
        decoded = tuple(_decode(v, args[0], depth + 1) for v in value)
        return tuple(sorted(decoded, key=lambda v: _json_bytes(_plain(v))))
    if annotation in (str, int, bool):
        if type(value) is not annotation:
            _fail("Manifest contains a wrong primitive type.")
        if annotation is str:
            _text(value, 4096)
        if annotation is int and (value < 0 or value > 2**63 - 1):
            _fail("Manifest counts must be nonnegative signed-64-bit integers.")
        return value
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        if type(value) is not str:
            _fail("Expected a serialized domain vocabulary value.")
        try:
            return annotation(value)
        except ValueError:
            _fail("Unsupported domain vocabulary value.")
    if isinstance(annotation, type) and is_dataclass(annotation):
        if not isinstance(value, Mapping) or set(value) != {f.name for f in fields(annotation)}:
            _fail("Manifest record keys do not match its closed shape.")
        hints = get_type_hints(annotation)
        values = {f.name: _decode(value[f.name], hints[f.name], depth + 1)
                  for f in fields(annotation)}
        instance = annotation(**{f.name: values[f.name] for f in fields(annotation) if f.init})
        for f in fields(annotation):
            if not f.init and _plain(values[f.name]) != _plain(getattr(instance, f.name)):
                _fail("A derived identity, metadata state, assembly, or readiness was forged.")
        return instance
    _fail("Unsupported local manifest field type.")


def _unique(values: tuple[str, ...]) -> None:
    if len(values) != len(set(values)):
        _fail("Duplicate IDs or references are prohibited.")


def _metadata(value: MetadataResolution) -> None:
    if not value.applicable and value.assertions:
        _fail("Inapplicable coordinate metadata cannot contain assertions.")
    _unique(tuple(_json_bytes(_plain(a)) for a in value.assertions))
    for assertion in value.assertions:
        _token(assertion.value)
        if assertion.value != assertion.value.lower():
            _fail("Species and assembly assertions must already be normalized lowercase identifiers.")
        if not assertion.evidence_ids:
            _fail("Metadata assertions require evidence.")


def _intrinsic_requirements(groups: tuple[GroupRecord, ...], coverage: tuple[CoverageRecord, ...],
                            supplied_repairs: tuple[PreparationRequirement, ...] = ()):
    """Only common domain gates; no naming, barcode-tag, sort or index rules."""
    information, repairs, prerequisites, issues = [], [], [], []
    for group in groups:
        gid = group.id
        def missing(code):
            information.append(RequiredInformation(gid, code))

        if group.route is None:
            missing(InformationCode.ROUTE)
        if group.grouping_basis is GroupingBasis.UNRESOLVED:
            missing(InformationCode.GROUPING)
        if group.species.state is ResolutionState.UNKNOWN:
            missing(InformationCode.SPECIES)
        for name, metadata in (("SPECIES", group.species),
                               ("SOURCE_ASSEMBLY", group.source_genome_assembly)):
            if metadata.state is ResolutionState.CONFLICT:
                refs = tuple(sorted({e for a in metadata.assertions for e in a.evidence_ids}))
                issues.append(IntakeIssue(name + "_CONFLICT", Severity.ERROR, gid,
                                          ReadinessEffect.INVALID, refs))
        if group.target_genome_assembly.state is TargetState.UNSUPPORTED:
            issues.append(IntakeIssue("ROUTE_UNSUPPORTED", Severity.ERROR, gid,
                                      ReadinessEffect.UNSUPPORTED))
        if group.input_kind is InputKind.BAM and group.source_genome_assembly.state is ResolutionState.UNKNOWN:
            missing(InformationCode.SOURCE_ASSEMBLY)
        if group.harmonization_required:
            # Mismatch establishes required work, not its feasibility. A future
            # route may supply an evidenced assessment for this same requirement.
            matches = [r for r in supplied_repairs if r.group_id == gid
                       and r.code is PreparationCode.ASSEMBLY_HARMONIZATION]
            repairs.append(matches[0] if len(matches) == 1 else
                           PreparationRequirement(gid, PreparationCode.ASSEMBLY_HARMONIZATION))
        group_repairs = [r for r in supplied_repairs if r.group_id == gid]
        group_repairs += [r for r in repairs if r.group_id == gid and r not in group_repairs]
        if any(r.admissibility is PreparationAdmissibility.UNRESOLVED for r in group_repairs):
            missing(InformationCode.ROUTE_ADMISSIBILITY)
        for repair in group_repairs:
            if repair.admissibility is PreparationAdmissibility.UNSUPPORTED:
                issues.append(IntakeIssue("PREPARATION_UNSUPPORTED", Severity.ERROR, gid,
                    ReadinessEffect.UNSUPPORTED, repair.admissibility_evidence_ids,
                    requirement_ids=(repair.id,)))
        # Only the target mapping is derived. No source assembly or conversion
        # strategy is inferred from species, paths, or contig names.
        if group.target_genome_assembly.value is not None:
            prerequisites.append(DownstreamPrerequisite(gid, PrerequisiteCode.TARGET_REFERENCE))
        if group.route is IntakeRoute.FASTQ_TO_CCRE:
            prerequisites.append(DownstreamPrerequisite(gid, PrerequisiteCode.ALIGNMENT))
        if group.structure is StructureState.UNRESOLVED:
            missing(InformationCode.STRUCTURE)
        elif group.structure in (StructureState.INVALID, StructureState.UNSUPPORTED):
            issues.append(IntakeIssue("STRUCTURE_" + group.structure.value.upper(), Severity.ERROR,
                                      gid, ReadinessEffect(group.structure.value), group.structure_evidence_ids))
        if group.barcode.source is BarcodeSource.UNKNOWN:
            missing(InformationCode.BARCODE_SOURCE)
        if group.barcode.identity_scope is BarcodeIdentityScope.UNRESOLVED:
            missing(InformationCode.BARCODE_IDENTITY_SCOPE)
        usable = [c for c in coverage if c.group_id == gid
                  and c.scope is not CoverageScope.NONE and c.stop_reason is not StopReason.ERROR]
        if not set(group.file_ids) <= {f for c in usable for f in c.file_ids}:
            missing(InformationCode.INSPECTION_COVERAGE)
    return tuple(information), tuple(repairs), tuple(prerequisites), tuple(issues)


def _validate(manifest: RawIntakeManifest) -> None:
    if (manifest.artifact_type != RAW_INTAKE_ARTIFACT_TYPE
            or manifest.schema_version != RAW_INTAKE_SCHEMA_VERSION
            or manifest.intake_contract_version != RAW_INTAKE_CONTRACT_VERSION
            or manifest.inspection.contract_version != INSPECTION_CONTRACT_VERSION
            or manifest.inspection.budget_version != INSPECTION_BUDGET_VERSION):
        _fail("Unsupported raw-intake artifact or contract identity.")
    if not manifest.files or not manifest.groups:
        _fail("An intake manifest requires files and groups.")
    collections = (manifest.files, manifest.groups, manifest.evidence, manifest.coverage,
                   manifest.issues, manifest.required_information, manifest.repairs, manifest.prerequisites)
    _unique(tuple(r.id for collection in collections for r in collection))
    file_map = {f.id: f for f in manifest.files}
    group_map = {g.id: g for g in manifest.groups}
    evidence_map = {e.id: e for e in manifest.evidence}
    coverage_map = {c.id: c for c in manifest.coverage}

    def refs(ids, group=None, claim=None, value=None, source=None):
        _unique(ids)
        for eid in ids:
            e = evidence_map.get(eid)
            if e is None:
                _fail("Dangling evidence reference.")
            if group is not None and (e.group_id not in (None, group.id)
                                     or (e.file_id is not None and e.file_id not in group.file_ids)):
                _fail("Evidence belongs to a different raw-input group.")
            if claim is not None and e.claim is not claim:
                _fail("Evidence does not support the asserted claim.")
            if value is not None and e.value != value:
                _fail("Evidence value contradicts the assertion.")
            if source is not None and e.source is not source:
                _fail("Assertion authority differs from its evidence.")

    for backend in manifest.inspection.backends:
        _token(backend.name)
        _token(backend.version)
    _unique(tuple(b.name for b in manifest.inspection.backends))
    membership = []
    for f in manifest.files:
        _path(f.path)
        if f.selection_root is not None:
            _path(f.selection_root)
            if not PurePosixPath(f.path).is_relative_to(f.selection_root) or f.path == f.selection_root:
                _fail("Discovered file must be within its declared selection root.")
        if (f.selection is SelectionBasis.DIRECTORY_DISCOVERY) != (f.selection_root is not None):
            _fail("Discovery requires an explicit root; explicit files have no discovery root.")
        refs(f.binding_evidence_ids, claim=EvidenceClaim.INPUT_BINDING)
        if any(evidence_map[e].file_id != f.id for e in f.binding_evidence_ids):
            _fail("Input binding evidence must identify its file.")
    for g in manifest.groups:
        _unique(g.file_ids)
        if not g.file_ids or not set(g.file_ids) <= file_map.keys():
            _fail("Empty group or dangling file reference.")
        membership.extend(g.file_ids)
        if any(file_map[f].input_kind is not g.input_kind for f in g.file_ids):
            _fail("Group kind must match every member file kind.")
        expected_route = {InputKind.FASTQ: IntakeRoute.FASTQ_TO_CCRE,
                          InputKind.BAM: IntakeRoute.BAM_TO_CCRE}.get(g.input_kind)
        if g.route is not None and g.route is not expected_route:
            _fail("Route does not accept this input kind.")
        _metadata(g.species)
        _metadata(g.source_genome_assembly)
        if not g.species.applicable:
            _fail("Species resolution cannot be not-applicable.")
        if g.source_genome_assembly.applicable == (g.input_kind is InputKind.FASTQ):
            _fail("FASTQ source coordinates are not applicable; other inputs must retain source resolution.")
        for metadata, claim in ((g.species, EvidenceClaim.SPECIES),
                                (g.source_genome_assembly, EvidenceClaim.SOURCE_ASSEMBLY)):
            for assertion in metadata.assertions:
                refs(assertion.evidence_ids, g, claim, assertion.value, assertion.source)
        for text in (g.syntactic_sample_token, g.lane, g.chunk, g.library_id):
            if text is not None:
                _text(text, 128)
        refs(g.grouping_evidence_ids, g, EvidenceClaim.GROUPING)
        if g.grouping_basis is not GroupingBasis.UNRESOLVED and not g.grouping_evidence_ids:
            _fail("Resolved grouping needs provenance.")
        grouping_sources = {
            GroupingBasis.EXPLICIT: (EvidenceSource.USER_DECLARATION,),
            GroupingBasis.AUTHORITATIVE_METADATA: (EvidenceSource.AUTHORITATIVE_METADATA,),
            GroupingBasis.SYNTACTIC: (EvidenceSource.FILENAME, EvidenceSource.OBSERVATION),
        }
        if g.grouping_basis in grouping_sources and any(
            evidence_map[e].source not in grouping_sources[g.grouping_basis]
            for e in g.grouping_evidence_ids
        ):
            _fail("Grouping basis differs from its evidence authority.")
        if g.grouping_basis is GroupingBasis.UNRESOLVED and any(
            v is not None for v in (g.syntactic_sample_token, g.lane, g.chunk)
        ):
            _fail("Syntactic grouping observations require a grouping basis.")
        refs(g.library_evidence_ids, g, EvidenceClaim.LIBRARY, g.library_id)
        if (g.library_id is None) != (not g.library_evidence_ids):
            _fail("Library identity and its evidence must be present together.")
        if any(evidence_map[e].source not in (EvidenceSource.USER_DECLARATION,
                                              EvidenceSource.AUTHORITATIVE_METADATA)
               for e in g.library_evidence_ids):
            _fail("Syntactic observations cannot establish a library identity.")
        refs(g.structure_evidence_ids, g, EvidenceClaim.STRUCTURE, g.structure.value)
        if g.structure is not StructureState.UNRESOLVED and not g.structure_evidence_ids:
            _fail("A structure finding requires evidence.")
        b = g.barcode
        refs(b.evidence_ids, g, EvidenceClaim.BARCODE_SOURCE,
             b.source.value + ":" + b.locator if b.locator is not None else None)
        refs(b.quality_evidence_ids, g, EvidenceClaim.BARCODE_QUALITIES)
        for value, ids, claim in ((b.producer, b.producer_evidence_ids, EvidenceClaim.BARCODE_PRODUCER),
                                  (b.namespace, b.namespace_evidence_ids, EvidenceClaim.BARCODE_NAMESPACE)):
            if value is not None:
                _text(value, 128)
            refs(ids, g, claim, value)
            if (value is None) != (not ids):
                _fail("Barcode producer/namespace and its evidence must be present together.")
            if any(evidence_map[e].source is EvidenceSource.FILENAME for e in ids):
                _fail("Filename evidence cannot establish barcode producer/namespace.")
        if b.source is BarcodeSource.UNKNOWN:
            if b.locator is not None or b.evidence_ids or b.quality_evidence_ids:
                _fail("Unknown barcode source cannot claim an extraction locator or observations.")
        else:
            if b.locator is None or not b.evidence_ids:
                _fail("Known barcode source requires locator and evidence.")
            _token(b.locator)
            if ((b.source is BarcodeSource.FASTQ_READ and g.input_kind is not InputKind.FASTQ)
                    or (b.source in (BarcodeSource.BAM_CELL_IDENTIFIER, BarcodeSource.BAM_RAW_SEQUENCE)
                        and g.input_kind is not InputKind.BAM)):
                _fail("Barcode source is incompatible with the input kind.")
    _unique(tuple(membership))
    if set(membership) != file_map.keys():
        _fail("Every inventory file must belong to exactly one explicit group.")
    for e in manifest.evidence:
        _text(e.value)
        if e.group_id is None and e.file_id is None:
            _fail("Evidence requires a group or file scope.")
        if e.group_id is not None and e.group_id not in group_map:
            _fail("Dangling evidence group reference.")
        if e.file_id is not None and e.file_id not in file_map:
            _fail("Dangling evidence file reference.")
        if e.group_id is not None and e.file_id is not None and e.file_id not in group_map[e.group_id].file_ids:
            _fail("Evidence file and group scopes disagree.")
        if e.source is EvidenceSource.OBSERVATION and e.coverage_id is None:
            _fail("Deterministic observations require explicit coverage.")
        if e.coverage_id is not None:
            c = coverage_map.get(e.coverage_id)
            if c is None or (e.group_id is not None and c.group_id != e.group_id) or (e.file_id is not None and e.file_id not in c.file_ids):
                _fail("Dangling or incompatible observation coverage reference.")
    for c in manifest.coverage:
        g = group_map.get(c.group_id)
        _unique(c.file_ids)
        if g is None or not c.file_ids or not set(c.file_ids) <= set(g.file_ids):
            _fail("Coverage references an unknown group or nonmember file.")
        for count, limit in ((c.records_inspected, c.record_limit),
                             (c.decoded_bytes_inspected, c.decoded_byte_limit)):
            if limit is not None and (limit == 0 or (count is not None and count > limit)):
                _fail("Coverage exceeds its positive declared inspection limit.")
        if c.method is CoverageMethod.NOT_INSPECTED:
            if (c.scope is not CoverageScope.NONE or c.stop_reason is not StopReason.NOT_STARTED
                    or c.eof_observed or c.records_inspected != 0
                    or c.decoded_bytes_inspected not in (None, 0) or c.observed_region_sha256 is not None):
                _fail("Not-inspected coverage cannot claim observations.")
        elif (c.scope is CoverageScope.NONE or c.stop_reason is StopReason.NOT_STARTED
              or (c.record_limit is None and c.decoded_byte_limit is None)):
            _fail("Inspected coverage needs scope and a declared bound.")
        if c.scope is CoverageScope.COMPLETE and (not c.eof_observed or c.stop_reason is not StopReason.EOF
                                                 or c.method is not CoverageMethod.SEQUENTIAL):
            _fail("Complete coverage requires sequential inspection to observed EOF.")
        if c.eof_observed != (c.stop_reason is StopReason.EOF):
            _fail("EOF observation and stopping reason disagree.")
        if c.observed_region_sha256 is not None and re.fullmatch(r"[0-9a-f]{64}", c.observed_region_sha256) is None:
            _fail("Observed-region digest must be a lowercase SHA-256.")
    requirements = {r.id: r for collection in (manifest.required_information, manifest.repairs, manifest.prerequisites) for r in collection}
    for collection in (manifest.issues, manifest.required_information, manifest.repairs, manifest.prerequisites):
        for r in collection:
            if r.group_id not in group_map:
                _fail("Finding references an unknown group.")
            refs(r.evidence_ids, group_map[r.group_id])
    for issue in manifest.issues:
        if re.fullmatch(r"[A-Z][A-Z0-9_]{0,95}", issue.code) is None:
            _fail("Issue codes must be stable uppercase identifiers.")
        if issue.message is not None:
            _text(issue.message)
        _unique(issue.file_ids)
        _unique(issue.requirement_ids)
        if not set(issue.file_ids) <= set(group_map[issue.group_id].file_ids):
            _fail("Issue files must belong to its affected group.")
        for rid in issue.requirement_ids:
            if rid not in requirements or requirements[rid].group_id != issue.group_id:
                _fail("Dangling or cross-group requirement reference.")
        if issue.effect is ReadinessEffect.REPAIR_REQUIRED and not any(
            isinstance(requirements[r], PreparationRequirement) for r in issue.requirement_ids
        ):
            _fail("Repair effects require an actual linked preparation requirement.")
    # One assessment per requirement: contradictory support decisions must not
    # be resolved by record order or by selecting the most convenient evidence.
    _unique(tuple((r.group_id, r.code) for r in manifest.repairs))
    expected = _intrinsic_requirements(manifest.groups, manifest.coverage, manifest.repairs)
    for required, actual in zip(expected, (manifest.required_information, manifest.repairs,
                                            manifest.prerequisites, manifest.issues), strict=True):
        if not {_json_bytes(_plain(r)) for r in required} <= {_json_bytes(_plain(r)) for r in actual}:
            _fail("Manifest omits a mandatory metadata, assembly, structure, or coverage finding.")
    for repair in manifest.repairs:
        group = group_map[repair.group_id]
        if repair.route is not None and repair.route is not group.route:
            _fail("Preparation admissibility must refer to the selected group route.")
        if repair.admissibility is PreparationAdmissibility.UNRESOLVED:
            if repair.admissibility_evidence_ids:
                _fail("Unresolved preparation cannot claim a resolved admissibility assessment.")
        else:
            if repair.route is None or not repair.admissibility_evidence_ids:
                _fail("Resolved preparation admissibility requires a selected route and evidence.")
            refs(repair.admissibility_evidence_ids, group, EvidenceClaim.PREPARATION_ADMISSIBILITY,
                 repair.code.value + ":" + repair.route.value + ":" + repair.admissibility.value)
            if any(evidence_map[e].source not in (EvidenceSource.AUTHORITATIVE_METADATA,
                                                  EvidenceSource.OBSERVATION)
                   for e in repair.admissibility_evidence_ids):
                _fail("Preparation support requires reviewed route evidence, not a user or filename assertion.")
        if (repair.code is PreparationCode.ASSEMBLY_HARMONIZATION
                and not group_map[repair.group_id].harmonization_required):
            _fail("Assembly harmonization requires a known assembly-bound source/target mismatch.")
    for prerequisite in manifest.prerequisites:
        if (prerequisite.code is PrerequisiteCode.ALIGNMENT
                and group_map[prerequisite.group_id].route is not IntakeRoute.FASTQ_TO_CCRE):
            _fail("Normal direct alignment is a prerequisite of the FASTQ route only.")


def validate_raw_intake_manifest(value: object) -> RawIntakeManifest:
    """Validate closed shapes, derived fields and cross-record domain invariants."""
    try:
        raw = _plain(value)
        if len(_json_bytes(raw)) > MAX_MANIFEST_BYTES:
            _fail("Manifest byte limit exceeded.")
        manifest = _decode(raw, RawIntakeManifest)
        _validate(manifest)
        return manifest
    except RawIntakeManifestError:
        raise
    except (TypeError, ValueError, KeyError, AttributeError, RecursionError) as exc:
        raise RawIntakeManifestError("RAW_INTAKE_CONTRACT_INVALID", "Invalid raw-intake domain record.") from exc


def build_raw_intake_manifest(
    *, files: tuple[FileRecord, ...], groups: tuple[GroupRecord, ...],
    evidence: tuple[EvidenceRecord, ...] = (), coverage: tuple[CoverageRecord, ...] = (),
    issues: tuple[IntakeIssue, ...] = (), required_information: tuple[RequiredInformation, ...] = (),
    repairs: tuple[PreparationRequirement, ...] = (), prerequisites: tuple[DownstreamPrerequisite, ...] = (),
    inspection: InspectionConfiguration = InspectionConfiguration(),
) -> RawIntakeManifest:
    """Derive mandatory findings and readiness without observing any raw input."""
    try:
        # Type-check before deriving common gates. This accepts the same frozen
        # tuple/mapping representations as other repository JSON contracts.
        supplied = _decode(RawIntakeManifest(
            files, groups, evidence, coverage, issues, required_information,
            repairs, prerequisites, inspection,
        ).to_dict(), RawIntakeManifest)
        information, preparation, normal, intrinsic_issues = _intrinsic_requirements(
            supplied.groups, supplied.coverage, supplied.repairs
        )
        # Do not silently deduplicate caller findings.
        manifest = replace(supplied,
                           required_information=supplied.required_information + information,
                           repairs=supplied.repairs + tuple(r for r in preparation if r not in supplied.repairs),
                           prerequisites=supplied.prerequisites + normal,
                           issues=supplied.issues + intrinsic_issues)
    except RawIntakeManifestError:
        raise
    except (TypeError, ValueError, KeyError, AttributeError, RecursionError) as exc:
        raise RawIntakeManifestError("RAW_INTAKE_CONTRACT_INVALID", "Invalid raw-intake domain record.") from exc
    return validate_raw_intake_manifest(manifest)


def canonical_manifest_bytes(value: object) -> bytes:
    return _json_bytes(validate_raw_intake_manifest(value).to_dict())


def _duplicate_free(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail("Duplicate JSON object key.", "RAW_INTAKE_JSON_INVALID")
        result[key] = value
    return result


def _reject_constant(value):
    _fail("Nonfinite JSON number.", "RAW_INTAKE_JSON_INVALID")


def load_raw_intake_manifest(path: str | Path, *, expected_sha256: str | None = None) -> tuple[Path, RawIntakeManifest, str]:
    """Hash actual manifest bytes; never hash any declared raw-source path."""
    resolved = Path(path).expanduser().resolve()
    with resolved.open("rb") as handle:
        payload = handle.read(MAX_MANIFEST_BYTES + 1)
    if len(payload) > MAX_MANIFEST_BYTES:
        _fail("Manifest byte limit exceeded.", "RAW_INTAKE_JSON_INVALID")
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and (type(expected_sha256) is not str
                                      or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
                                      or digest != expected_sha256):
        _fail("Manifest SHA-256 mismatch.", "RAW_INTAKE_DIGEST_MISMATCH")
    try:
        raw = json.loads(payload.decode("utf-8"), object_pairs_hook=_duplicate_free,
                         parse_constant=_reject_constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise RawIntakeManifestError("RAW_INTAKE_JSON_INVALID", "Manifest is not strict UTF-8 JSON.") from exc
    return resolved, validate_raw_intake_manifest(raw), digest


def publish_raw_intake_manifest(value: object, path: str | Path, *, overwrite: bool = False) -> ManifestReference:
    """Publish canonical JSON in a trusted local directory, like M8 artifacts.

    No source reads, directory discovery or hidden repairs occur. Failure before
    replacement preserves an existing destination. After replacement, a directory
    fsync failure may leave a complete valid new artifact (not a partial file).
    This is not a claim of protection against hostile concurrent path changes.
    """
    if type(overwrite) is not bool:
        _fail("Overwrite must be boolean.")
    manifest = validate_raw_intake_manifest(value)
    payload = _json_bytes(manifest.to_dict())
    destination = Path(path).expanduser().absolute()
    if destination.is_symlink() or (destination.exists() and not destination.is_file()):
        _fail("Manifest destination must be a regular file.", "RAW_INTAKE_OUTPUT_CONFLICT")
    destination = destination.resolve()
    if str(destination) in {f.path for f in manifest.files}:
        _fail("Manifest output cannot replace a declared raw input.", "RAW_INTAKE_OUTPUT_CONFLICT")
    if destination.exists() and not overwrite:
        raise FileExistsError("Raw-intake manifest already exists.")
    # The caller owns output directory creation, as no intake tool exists yet.
    temporary = None
    try:
        descriptor, name = tempfile.mkstemp(dir=destination.parent, prefix=".raw-intake-", suffix=".tmp")
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        digest = hashlib.sha256(payload).hexdigest()
        load_raw_intake_manifest(temporary, expected_sha256=digest)
        if destination.exists() and not overwrite:
            raise FileExistsError("Raw-intake manifest already exists.")
        directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.replace(temporary, destination)
            temporary = None
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        load_raw_intake_manifest(destination, expected_sha256=digest)
        return {"manifest_path": str(destination), "manifest_sha256": digest,
                "artifact_schema_version": RAW_INTAKE_SCHEMA_VERSION}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
