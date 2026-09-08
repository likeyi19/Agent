"""Public raw sequencing intake; private inspectors own format/scientific facts."""
from __future__ import annotations

from collections.abc import Sequence
import hashlib
import os
from pathlib import Path
import re
import stat
from typing import TypedDict

from . import _raw_fastq as fastq
from . import raw_scatac_manifest as m


class RawScATACInspection(TypedDict):
    status: str
    manifest_path: str
    manifest_sha256: str
    artifact_type: str
    artifact_schema_version: int
    intake_contract_version: str
    input_kind: str
    readiness: str
    n_files: int
    n_groups: int
    n_issues: int
    n_required_information: int
    n_repairs: int
    n_prerequisites: int


class RawScATACError(ValueError):
    """Stable sanitized public input/publication failure."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _fail(code: str):
    raise RawScATACError(code) from None


def _kind(name: str) -> m.InputKind | None:
    if name.endswith(fastq.SUFFIXES):
        return m.InputKind.FASTQ
    if name.endswith('.bam'):
        return m.InputKind.BAM
    return None


def _select(raw_input_paths):
    """Bounded name/stat dispatch only; never open raw content or load pysam."""
    values = (raw_input_paths,) if isinstance(raw_input_paths, (str, Path)) else raw_input_paths
    if not isinstance(values, (list, tuple)) or not 0 < len(values) <= 128:
        _fail('RAW_INPUT_SELECTION_INVALID')
    roots = set()
    try:
        for value in values:
            if not isinstance(value, (str, Path)) or not str(value) or '://' in str(value):
                _fail('RAW_INPUT_SELECTION_INVALID')
            roots.add(Path(value).expanduser().resolve(strict=True))
        candidates = {}
        for root in sorted(roots):
            if root.is_dir():
                with os.scandir(root) as entries:
                    for n, entry in enumerate(entries, 1):
                        if n > 16384:
                            _fail('RAW_INPUT_DISCOVERY_LIMIT')
                        if _kind(entry.name) is not None:
                            if entry.is_symlink():
                                _fail('RAW_INPUT_SELECTION_INVALID')
                            candidates[root / entry.name] = _kind(entry.name)
                        if len(candidates) > 128:
                            _fail('RAW_INPUT_DISCOVERY_LIMIT')
            else:
                kind = _kind(root.name)
                if kind is None:
                    _fail('RAW_INPUT_SELECTION_INVALID')
                candidates[root] = kind
            if len(candidates) > 128:
                _fail('RAW_INPUT_DISCOVERY_LIMIT')
        if not candidates:
            _fail('RAW_INPUT_NO_CANDIDATES')
        kinds = set(candidates.values())
        if len(kinds) != 1:
            _fail('RAW_INPUT_KIND_MIXED')
        snapshots = []
        for path in sorted(candidates):
            value = path.stat(follow_symlinks=False)
            if not stat.S_ISREG(value.st_mode):
                _fail('RAW_INPUT_SELECTION_INVALID')
            snapshots.append((str(path), fastq._snapshot(value)))
        return tuple(sorted(roots)), next(iter(kinds)), tuple(snapshots)
    except RawScATACError:
        raise
    except PermissionError:
        _fail('RAW_INPUT_ACCESS_FAILED')
    except (OSError, ValueError, RuntimeError):
        _fail('RAW_INPUT_SELECTION_INVALID')


def _declarations(kind, species, raw_assay, source_genome_assembly, fastq_layout):
    if species is not None and (type(species) is not str or
            re.fullmatch(r'[a-z0-9][a-z0-9_.:+-]{0,127}', species) is None):
        _fail('RAW_INPUT_DECLARATION_INVALID')
    if raw_assay is not None and (type(raw_assay) is not str or raw_assay not in ('TENX_ATAC', 'SCATAC')):
        _fail('RAW_INPUT_DECLARATION_INVALID')
    if fastq_layout is not None and (type(fastq_layout) is not str or
            fastq_layout not in (fastq.FastqLayout.A.value, fastq.FastqLayout.B.value)):
        _fail('RAW_INPUT_DECLARATION_INVALID')
    if kind is m.InputKind.FASTQ:
        if source_genome_assembly is not None or raw_assay == 'SCATAC' or (
                fastq_layout is not None and raw_assay != 'TENX_ATAC'):
            _fail('RAW_INPUT_DECLARATION_INCOMPATIBLE')
    elif fastq_layout is not None:
        _fail('RAW_INPUT_DECLARATION_INCOMPATIBLE')


def _reconstruct(raw_input_paths, *, species=None, raw_assay=None,
                 source_genome_assembly=None, fastq_layout=None):
    """Reconstruct from sources without publication or any public-tool call."""
    roots, kind, before = _select(raw_input_paths)
    _declarations(kind, species, raw_assay, source_genome_assembly, fastq_layout)
    if kind is m.InputKind.FASTQ:
        manifest = fastq.inspect_fastq_inputs(roots, species=species,
            assay=fastq.FastqAssay(raw_assay) if raw_assay else None,
            declared_layout=fastq.FastqLayout(fastq_layout) if fastq_layout else None)
    else:
        from . import _raw_bam as bam
        manifest = bam.inspect_bam_inputs(roots, species=species,
            assay=bam.BamAssay(raw_assay) if raw_assay else None,
            source_genome_assembly=source_genome_assembly)
    try:
        after = _select(roots)
    except RawScATACError:
        _fail('RAW_INPUT_SOURCE_CHANGED')
    if after != (roots, kind, before) or tuple(sorted(f.path for f in manifest.files)) != tuple(p for p, _ in before):
        _fail('RAW_INPUT_SOURCE_CHANGED')
    return manifest


def _summary(manifest: m.RawIntakeManifest, path: Path, digest: str) -> RawScATACInspection:
    kinds = {g.input_kind for g in manifest.groups}
    if len(kinds) != 1:
        _fail('RAW_INPUT_KIND_MIXED')
    return RawScATACInspection(status='success', manifest_path=str(path), manifest_sha256=digest,
        artifact_type=manifest.artifact_type, artifact_schema_version=manifest.schema_version,
        intake_contract_version=manifest.intake_contract_version, input_kind=next(iter(kinds)).value,
        readiness=manifest.readiness.value, n_files=len(manifest.files), n_groups=len(manifest.groups),
        n_issues=len(manifest.issues), n_required_information=len(manifest.required_information),
        n_repairs=len(manifest.repairs), n_prerequisites=len(manifest.prerequisites))


def inspect_raw_scATAC(
    raw_input_paths: Sequence[str | Path] | str | Path,
    output_dir: str | Path,
    *,
    species: str | None = None,
    raw_assay: str | None = None,
    source_genome_assembly: str | None = None,
    fastq_layout: str | None = None,
) -> RawScATACInspection:
    """Inspect FASTQ or BAM and publish an authoritative content-addressed manifest.

    Readiness is independent of execution success. Only matching existing
    canonical artifacts may be reused; no overwrite or preprocessing is exposed.
    """
    if (not isinstance(output_dir, (str, Path)) or not str(output_dir)
            or '://' in str(output_dir) or '\0' in str(output_dir)):
        _fail('RAW_INPUT_OUTPUT_INVALID')
    manifest = _reconstruct(raw_input_paths, species=species, raw_assay=raw_assay,
        source_genome_assembly=source_genome_assembly, fastq_layout=fastq_layout)
    payload = m.canonical_manifest_bytes(manifest)
    digest = hashlib.sha256(payload).hexdigest()
    try:
        output = Path(output_dir).expanduser().resolve()
        if output in {Path(f.path).parent for f in manifest.files}:
            _fail('RAW_INTAKE_OUTPUT_CONFLICT')
        output.mkdir(parents=True, exist_ok=True)
        destination = output / f'raw-scatac-intake-{digest}.json'
        if destination.is_symlink() or (destination.exists() and not destination.is_file()):
            _fail('RAW_INTAKE_OUTPUT_CONFLICT')
        if not destination.exists():
            try:
                m.publish_raw_intake_manifest(manifest, destination)
            except FileExistsError:
                pass  # Concurrent identical publication may be safely reused.
        try:
            path, stored, actual_digest = m.load_raw_intake_manifest(destination, expected_sha256=digest)
            if m.canonical_manifest_bytes(stored) != payload:
                _fail('RAW_INTAKE_OUTPUT_CONFLICT')
        except m.RawIntakeManifestError:
            _fail('RAW_INTAKE_OUTPUT_CONFLICT')
    except RawScATACError:
        raise
    except (OSError, ValueError, RuntimeError):
        _fail('RAW_INTAKE_WRITE_FAILED')
    return _summary(stored, path, actual_digest)


__all__ = ['RawScATACInspection', 'inspect_raw_scATAC']
