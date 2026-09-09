"""M11.1a reference identity; no preprocessing or model resources.

Build/reinspect explicitly read resource files. Validate/load/publish only
operate on the lightweight bundle. Assembly is caller-declared, never inferred
or independently authenticated from sequence content. A supplied FAI defines
the contig dictionary; its correspondence to FASTA sequence is not certified.

BED3+ is plain UTF-8, tab-delimited, without headers, blank lines or CR. Extra
columns are ignored scientifically but included in the BED byte hash. Names
are exactly chrom + ':' + decimal(start) + '-' + decimal(end), in BED row
order. Contig records are name + TAB + decimal(length), in FAI row order.
Both ordered digests use UTF-8 records followed by LF (including the last).
Names/coordinates are never normalized, sorted, deduplicated or repaired.
Coordinates and FAI integers use unsigned decimal without signs, whitespace or
leading zeros (except zero itself), bounded by signed int64. FAI has exactly
five columns and unique names, positive lengths/line sizes, and offsets within
the FASTA file. BED bounds are always checked against this exact dictionary.
FAI row order is preserved even when it differs from sequence-offset order.

Canonical manifest bytes are UTF-8 JSON, sorted object keys, no extra whitespace
or trailing newline, ensure_ascii=False and allow_nan=False. The reference
digest is SHA-256 of b"agent.scatac-reference-identity.v1\\0" followed by these
canonical bytes after excluding the reference digest itself and every resource
path/provenance record. The separately returned manifest digest hashes the
complete artifact bytes, including locations and historical assertions.

This contract identifies the supplied vocabulary, including tiny fixtures;
it does not certify arbitrary BEDs as the project's canonical vocabulary.
Callers bind the expected feature count/order or the complete reference identity.
Historical provenance is a separate, optional caller assertion, not a fact
inferred from paths or authenticated by a content hash.
"""

from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from typing import TypedDict

from ._ordered_identity import ORDERED_IDENTITY_ALGORITHM, ordered_identity_sha256


REFERENCE_ARTIFACT_TYPE = "agent.scatac-reference-bundle"
REFERENCE_SCHEMA_VERSION = 1
REFERENCE_CONTRACT_VERSION = "scatac-reference-bundle.v1"
MAX_MANIFEST_BYTES = 64 * 1024
MAX_LINE_BYTES = 64 * 1024
COORDINATE_CONVENTION = "zero-based-half-open"
FEATURE_NAME_CONVENTION = "chrom:start-end.v1"


class ScATACReferenceError(ValueError):
    """Stable data-layer error; no runtime retry disposition is implied."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def _fail(message: str, code: str = "REFERENCE_CONTRACT_INVALID"):
    raise ScATACReferenceError(code, message)


@dataclass(frozen=True)
class SourceProvenance:
    basis: str = "unknown"
    source: str | None = None
    accession: str | None = None
    citation: str | None = None


@dataclass(frozen=True)
class ResourceIdentity:
    path: str
    sha256: str
    provenance: SourceProvenance = SourceProvenance()


@dataclass(frozen=True)
class GenomeReference:
    fasta: ResourceIdentity
    fai: ResourceIdentity
    contig_count: int
    ordered_contig_sha256: str


@dataclass(frozen=True)
class CCREReference:
    bed: ResourceIdentity
    feature_count: int
    ordered_feature_sha256: str
    coordinate_convention: str = COORDINATE_CONVENTION
    feature_name_convention: str = FEATURE_NAME_CONVENTION
    bounds_checked: bool = True


@dataclass(frozen=True)
class AnnotationReference:
    resource: ResourceIdentity
    format: str = "unknown"


@dataclass(frozen=True)
class ScATACReferenceBundle:
    species: str
    target_assembly: str
    genome: GenomeReference
    ccre: CCREReference
    reference_identity_sha256: str
    annotation: AnnotationReference | None = None
    assembly_binding: str = "explicit_declaration"
    ordered_identity_algorithm: str = ORDERED_IDENTITY_ALGORITHM
    artifact_type: str = REFERENCE_ARTIFACT_TYPE
    schema_version: int = REFERENCE_SCHEMA_VERSION
    contract_version: str = REFERENCE_CONTRACT_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


class ReferenceManifestReference(TypedDict):
    manifest_path: str
    manifest_sha256: str
    artifact_schema_version: int


def _json_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(",", ":")).encode("utf-8")


def _text(value):
    if (type(value) is not str or not value or value != value.strip()
            or len(value) > 4096 or any(ord(c) < 32 or ord(c) == 127 for c in value)):
        _fail("Invalid reference text.")
    value.encode("utf-8")


def _path_text(value):
    _text(value)
    if (not value.startswith("/") or value.startswith("//")
            or ".." in PurePosixPath(value).parts or str(PurePosixPath(value)) != value):
        _fail("Reference paths must be absolute normalized local paths.")


def _sha(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        _fail("Invalid SHA-256 identity.")


def _positive(value):
    if type(value) is not int or not 0 < value <= 2**63 - 1:
        _fail("Reference counts must be positive integers.")


def _shape(value, cls):
    if type(value) is not dict or set(value) != {f.name for f in fields(cls)}:
        _fail("Reference record has missing or unknown fields.")
    return value


def _provenance(value) -> SourceProvenance:
    p = SourceProvenance(**_shape(value, SourceProvenance))
    claims = (p.source, p.accession, p.citation)
    for claim in claims:
        if claim is not None:
            _text(claim)
    if p.basis == "unknown":
        if any(c is not None for c in claims):
            _fail("Unknown provenance cannot contain historical claims.")
    elif p.basis == "caller_supplied":
        if all(c is None for c in claims):
            _fail("Caller-supplied provenance requires a claim.")
    else:
        _fail("Unsupported provenance basis.")
    return p


def _resource(value) -> ResourceIdentity:
    d = _shape(value, ResourceIdentity)
    _path_text(d["path"])
    _sha(d["sha256"])
    return ResourceIdentity(d["path"], d["sha256"], _provenance(d["provenance"]))


def _reference_digest(bundle: ScATACReferenceBundle) -> str:
    # Resource identity is portable: locations and historical assertions do not
    # alter it. The manifest byte digest protects those fields separately.
    value = bundle.to_dict()
    del value["reference_identity_sha256"]
    resources = [value["genome"]["fasta"], value["genome"]["fai"], value["ccre"]["bed"]]
    if value["annotation"] is not None:
        resources.append(value["annotation"]["resource"])
    for resource in resources:
        del resource["path"]
        del resource["provenance"]
    return hashlib.sha256(b"agent.scatac-reference-identity.v1\0" + _json_bytes(value)).hexdigest()


def validate_scatac_reference_bundle(value: object) -> ScATACReferenceBundle:
    """Validate closed JSON/domain contracts without accessing resource paths."""
    try:
        raw = value.to_dict() if type(value) is ScATACReferenceBundle else value
        d = dict(_shape(raw, ScATACReferenceBundle))
        if len(_json_bytes(d)) > MAX_MANIFEST_BYTES:
            _fail("Reference manifest byte limit exceeded.")
        if (type(d["schema_version"]) is not int or d["schema_version"] != 1
                or d["artifact_type"] != REFERENCE_ARTIFACT_TYPE
                or d["contract_version"] != REFERENCE_CONTRACT_VERSION
                or d["assembly_binding"] != "explicit_declaration"
                or d["ordered_identity_algorithm"] != ORDERED_IDENTITY_ALGORITHM):
            _fail("Unsupported reference version or identity convention.")
        if (d["species"], d["target_assembly"]) not in (("human", "hg38"), ("mouse", "mm10")):
            _fail("Unsupported species/assembly pair.")
        g = dict(_shape(d["genome"], GenomeReference))
        g["fasta"], g["fai"] = _resource(g["fasta"]), _resource(g["fai"])
        _positive(g["contig_count"])
        _sha(g["ordered_contig_sha256"])
        d["genome"] = GenomeReference(**g)
        c = dict(_shape(d["ccre"], CCREReference))
        c["bed"] = _resource(c["bed"])
        _positive(c["feature_count"])
        _sha(c["ordered_feature_sha256"])
        if (c["coordinate_convention"] != COORDINATE_CONVENTION
                or c["feature_name_convention"] != FEATURE_NAME_CONVENTION
                or c["bounds_checked"] is not True):
            _fail("Unsupported cCRE convention or unchecked bounds.")
        d["ccre"] = CCREReference(**c)
        if d["annotation"] is not None:
            a = dict(_shape(d["annotation"], AnnotationReference))
            a["resource"] = _resource(a["resource"])
            if a["format"] not in ("unknown", "gtf", "gff3", "bed3"):
                _fail("Unsupported annotation format declaration.")
            d["annotation"] = AnnotationReference(**a)
        _sha(d["reference_identity_sha256"])
        bundle = ScATACReferenceBundle(**d)
        if bundle.reference_identity_sha256 != _reference_digest(bundle):
            _fail("Reference identity disagrees with its fields.", "REFERENCE_DIGEST_MISMATCH")
        return bundle
    except ScATACReferenceError:
        raise
    except (TypeError, ValueError, AttributeError, RecursionError, UnicodeError) as exc:
        raise ScATACReferenceError("REFERENCE_CONTRACT_INVALID", "Invalid reference bundle.") from exc


def canonical_reference_bundle_bytes(value: object) -> bytes:
    return _json_bytes(validate_scatac_reference_bundle(value).to_dict())


def _snapshot(path: Path):
    s = path.stat()
    if not stat.S_ISREG(s.st_mode) or s.st_size <= 0:
        _fail("Reference must be a nonempty regular file.", "REFERENCE_SOURCE_INVALID")
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def _source_path(value) -> Path:
    if not isinstance(value, (str, Path)):
        _fail("Reference source must be a local path.")
    _text(str(value))
    if "://" in str(value):
        _fail("Reference source must be local.")
    path = Path(value).expanduser()
    if path.is_symlink():
        _fail("Reference source must not be a symbolic link.", "REFERENCE_SOURCE_INVALID")
    path = path.resolve(strict=True)
    _path_text(str(path))
    _snapshot(path)
    return path


def _file_hash(path: Path, *, fasta=False):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        first = True
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            if first and fasta and not chunk.startswith(b">"):
                _fail("Expected a plain FASTA resource.", "REFERENCE_SOURCE_INVALID")
            first = False
            digest.update(chunk)
    return digest.hexdigest()


def _lines(path: Path, digest):
    with path.open("rb") as handle:
        while line := handle.readline(MAX_LINE_BYTES + 1):
            if len(line) > MAX_LINE_BYTES or b"\r" in line or b"\0" in line:
                _fail("Invalid reference text record.", "REFERENCE_SOURCE_INVALID")
            digest.update(line)
            yield line.removesuffix(b"\n").decode("utf-8").split("\t")


def _uint(text: str, code: str):
    # Canonical unsigned decimal only: no signs, whitespace, leading zeros.
    if re.fullmatch(r"0|[1-9][0-9]{0,18}", text) is None or int(text) > 2**63 - 1:
        _fail("Invalid reference integer.", code)
    return int(text)


def _contig(name: str, code: str):
    if not name or ":" in name or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in name):
        _fail("Invalid exact contig name.", code)


def _inspect_fai(path, fasta_size):
    digest = hashlib.sha256()
    contigs = {}

    def records():
        for row in _lines(path, digest):
            if len(row) != 5:
                _fail("Expected five-column FASTA index.", "REFERENCE_FAI_INVALID")
            name = row[0]
            _contig(name, "REFERENCE_FAI_INVALID")
            length, offset, bases, width = (_uint(v, "REFERENCE_FAI_INVALID") for v in row[1:])
            if (name in contigs or length <= 0 or bases <= 0 or width < bases
                    or offset <= 0 or offset >= fasta_size):
                _fail("Invalid or duplicate FAI entry.", "REFERENCE_FAI_INVALID")
            contigs[name] = length
            yield f"{name}\t{length}"

    ordered = ordered_identity_sha256(records())
    return contigs, digest.hexdigest(), ordered


def _inspect_bed(path, contigs):
    digest = hashlib.sha256()
    count = 0

    def names():
        nonlocal count
        for row in _lines(path, digest):
            if len(row) < 3:
                _fail("Expected BED3 or more columns.", "REFERENCE_BED_INVALID")
            chrom, start, end = row[:3]
            _contig(chrom, "REFERENCE_BED_INVALID")
            start, end = _uint(start, "REFERENCE_BED_INVALID"), _uint(end, "REFERENCE_BED_INVALID")
            if chrom not in contigs or not 0 <= start < end <= contigs[chrom]:
                _fail("BED coordinates disagree with reference contigs.", "REFERENCE_BED_INVALID")
            count += 1
            yield f"{chrom}:{start}-{end}"
    try:
        ordered = ordered_identity_sha256(names())
    except ScATACReferenceError:
        raise
    except ValueError as exc:
        raise ScATACReferenceError("REFERENCE_BED_INVALID", "Invalid or duplicate BED features.") from exc
    return count, digest.hexdigest(), ordered


def build_scatac_reference_bundle(
    *, species: str, target_assembly: str, fasta_path: str | Path,
    fai_path: str | Path, ccre_bed_path: str | Path,
    annotation_path: str | Path | None = None, annotation_format: str = "unknown",
    genome_provenance: SourceProvenance = SourceProvenance(),
    ccre_provenance: SourceProvenance = SourceProvenance(),
    annotation_provenance: SourceProvenance = SourceProvenance(),
    expected_feature_count: int | None = None,
    expected_ordered_feature_sha256: str | None = None,
) -> ScATACReferenceBundle:
    """Explicitly inspect/hash resources, preserving all BED/FAI rows in order.

    No defaults select resource paths or canonical vocabularies. Optional
    expectations pin the intended complete feature space, without model filters.
    All sources are snapshotted before and after inspection (trusted local FS).
    """
    try:
        if (species, target_assembly) not in (("human", "hg38"), ("mouse", "mm10")):
            _fail("Unsupported species/assembly pair.")
        for p in (genome_provenance, ccre_provenance, annotation_provenance):
            if type(p) is not SourceProvenance:
                _fail("Expected a SourceProvenance record.")
            _provenance(asdict(p))
        if annotation_format not in ("unknown", "gtf", "gff3", "bed3"):
            _fail("Unsupported annotation format declaration.")
        if annotation_path is None and (annotation_format != "unknown" or annotation_provenance != SourceProvenance()):
            _fail("Annotation declarations require an annotation resource.")
        if expected_feature_count is not None:
            _positive(expected_feature_count)
        if expected_ordered_feature_sha256 is not None:
            _sha(expected_ordered_feature_sha256)
        paths = [_source_path(p) for p in (fasta_path, fai_path, ccre_bed_path)]
        if annotation_path is not None:
            paths.append(_source_path(annotation_path))
        before = [_snapshot(p) for p in paths]
        fasta_sha = _file_hash(paths[0], fasta=True)
        contigs, fai_sha, contig_sha = _inspect_fai(paths[1], before[0][2])
        count, bed_sha, feature_sha = _inspect_bed(paths[2], contigs)
        if ((expected_feature_count is not None and count != expected_feature_count)
                or (expected_ordered_feature_sha256 is not None and feature_sha != expected_ordered_feature_sha256)):
            _fail("cCRE vocabulary differs from the expected identity.", "REFERENCE_EXPECTATION_MISMATCH")
        annotation = None if annotation_path is None else AnnotationReference(
            ResourceIdentity(str(paths[3]), _file_hash(paths[3]), annotation_provenance), annotation_format)
        if before != [_snapshot(p) for p in paths]:
            _fail("Reference source changed during inspection.", "REFERENCE_SOURCE_CHANGED")
        bundle = ScATACReferenceBundle(species, target_assembly,
            GenomeReference(ResourceIdentity(str(paths[0]), fasta_sha, genome_provenance),
                            ResourceIdentity(str(paths[1]), fai_sha, genome_provenance), len(contigs), contig_sha),
            CCREReference(ResourceIdentity(str(paths[2]), bed_sha, ccre_provenance), count, feature_sha),
            "", annotation)
        return validate_scatac_reference_bundle(replace(bundle, reference_identity_sha256=_reference_digest(bundle)))
    except ScATACReferenceError:
        raise
    except OSError as exc:
        raise ScATACReferenceError("REFERENCE_SOURCE_UNAVAILABLE", "Reference source could not be read.") from exc
    except (TypeError, ValueError, UnicodeError, RuntimeError) as exc:
        raise ScATACReferenceError("REFERENCE_SOURCE_INVALID", "Invalid reference source.") from exc


def _duplicate_free(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail("Duplicate JSON key.", "REFERENCE_JSON_INVALID")
        result[key] = value
    return result


def _reject_constant(value):
    _fail("Nonfinite JSON number.", "REFERENCE_JSON_INVALID")


def load_scatac_reference_bundle(path: str | Path, *, expected_sha256: str | None = None
                                ) -> tuple[Path, ScATACReferenceBundle, str]:
    """Read only bounded manifest bytes; never stat/open declared resources."""
    try:
        if expected_sha256 is not None:
            _sha(expected_sha256)
        resolved = _source_path(path)
        with resolved.open("rb") as handle:
            payload = handle.read(MAX_MANIFEST_BYTES + 1)
        if len(payload) > MAX_MANIFEST_BYTES:
            _fail("Manifest byte limit exceeded.", "REFERENCE_JSON_INVALID")
        digest = hashlib.sha256(payload).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            _fail("Manifest SHA-256 mismatch.", "REFERENCE_DIGEST_MISMATCH")
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_duplicate_free,
                           parse_constant=_reject_constant)
        return resolved, validate_scatac_reference_bundle(value), digest
    except ScATACReferenceError:
        raise
    except OSError as exc:
        raise ScATACReferenceError("REFERENCE_ARTIFACT_UNAVAILABLE", "Reference manifest could not be read.") from exc
    except (TypeError, ValueError, UnicodeError, RecursionError, RuntimeError) as exc:
        raise ScATACReferenceError("REFERENCE_JSON_INVALID", "Manifest is not strict UTF-8 JSON.") from exc


def reinspect_scatac_reference_bundle_sources(value: object) -> ScATACReferenceBundle:
    """Recompute all byte/order identities; no publication or historical proof."""
    bundle = validate_scatac_reference_bundle(value)
    current = build_scatac_reference_bundle(
        species=bundle.species, target_assembly=bundle.target_assembly,
        fasta_path=bundle.genome.fasta.path, fai_path=bundle.genome.fai.path,
        ccre_bed_path=bundle.ccre.bed.path,
        genome_provenance=bundle.genome.fasta.provenance,
        ccre_provenance=bundle.ccre.bed.provenance,
        annotation_path=bundle.annotation.resource.path if bundle.annotation else None,
        annotation_format=bundle.annotation.format if bundle.annotation else "unknown",
        annotation_provenance=bundle.annotation.resource.provenance if bundle.annotation else SourceProvenance())
    if current.reference_identity_sha256 != bundle.reference_identity_sha256:
        _fail("Reference resources differ from the bundle.", "REFERENCE_SOURCE_CHANGED")
    return bundle


def publish_scatac_reference_bundle(value: object, path: str | Path, *, overwrite: bool = False
                                   ) -> ReferenceManifestReference:
    """Atomically publish canonical JSON in an existing caller-owned directory.

    No source reinspection. No overwrite of resource paths (including hardlink
    aliases). A directory-fsync failure after publication can leave a complete
    artifact; it never makes partial bytes an accepted result. Trusted local FS.
    """
    bundle = validate_scatac_reference_bundle(value)
    if type(overwrite) is not bool:
        _fail("Overwrite must be boolean.")
    temporary = None
    try:
        if not isinstance(path, (str, Path)):
            _fail("Output must be a local path.")
        _text(str(path))
        if "://" in str(path):
            _fail("Output must be local.")
        destination = Path(path).expanduser().absolute()
        if destination.is_symlink() or (destination.exists() and not destination.is_file()):
            _fail("Invalid reference output destination.", "REFERENCE_OUTPUT_CONFLICT")
        destination = destination.resolve()
        resources = [bundle.genome.fasta, bundle.genome.fai, bundle.ccre.bed]
        if bundle.annotation:
            resources.append(bundle.annotation.resource)
        for resource in resources:
            source = Path(resource.path)
            if destination == source or (destination.exists() and source.exists() and destination.samefile(source)):
                _fail("Output cannot replace a reference source.", "REFERENCE_OUTPUT_CONFLICT")
        if destination.exists() and not overwrite:
            _fail("Reference output already exists.", "REFERENCE_OUTPUT_CONFLICT")
        payload = canonical_reference_bundle_bytes(bundle)
        digest = hashlib.sha256(payload).hexdigest()
        fd, name = tempfile.mkstemp(dir=destination.parent, prefix=".scatac-reference-", suffix=".tmp")
        temporary = Path(name)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        load_scatac_reference_bundle(temporary, expected_sha256=digest)
        directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            if overwrite:
                os.replace(temporary, destination)
            else:
                # Atomic no-clobber publication, including a competing writer.
                os.link(temporary, destination)
                temporary.unlink()
            temporary = None
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        load_scatac_reference_bundle(destination, expected_sha256=digest)
        return {"manifest_path": str(destination), "manifest_sha256": digest,
                "artifact_schema_version": REFERENCE_SCHEMA_VERSION}
    except ScATACReferenceError:
        raise
    except FileExistsError as exc:
        raise ScATACReferenceError("REFERENCE_OUTPUT_CONFLICT", "Reference output already exists.") from exc
    except (OSError, ValueError, RuntimeError) as exc:
        raise ScATACReferenceError("REFERENCE_PUBLICATION_FAILED", "Reference manifest publication failed.") from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as exc:
                raise ScATACReferenceError(
                    "REFERENCE_PUBLICATION_FAILED", "Reference staging cleanup failed.") from exc
