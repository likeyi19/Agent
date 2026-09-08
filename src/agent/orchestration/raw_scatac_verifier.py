"""Independent source reconstruction for the registered raw-intake capability."""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
from pathlib import Path

from agent.tools.data import raw_scatac_manifest as m
from agent.tools.data.raw_scatac import _reconstruct, _summary
from .registry import ToolRegistry


class RawIntakeVerificationError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def verify_raw_scatac_intake(arguments: Mapping[str, object], result: Mapping[str, object],
                             registry: ToolRegistry) -> None:
    """Reopen sources and compare canonical science, never call/publish the tool."""
    registry.validate_arguments('inspect_raw_scATAC', arguments)
    registry.validate_result('inspect_raw_scATAC', result)
    output = Path(arguments['output_dir']).expanduser().resolve(strict=True)
    declared = Path(result['manifest_path'])
    path = declared.resolve(strict=True)
    digest = result['manifest_sha256']
    if (not output.is_dir() or declared.is_symlink() or path.parent != output
            or path.name != f'raw-scatac-intake-{digest}.json' or not path.is_file()):
        raise RawIntakeVerificationError('RAW_INTAKE_OUTPUT_BINDING_MISMATCH')
    path, stored, actual_digest = m.load_raw_intake_manifest(path, expected_sha256=digest)
    expected = _reconstruct(arguments['raw_input_paths'], **{
        key: arguments.get(key) for key in
        ('species', 'raw_assay', 'source_genome_assembly', 'fastq_layout')})
    # Reconstruction repeats bounded selection, inventory, metadata declarations,
    # source observations and readiness derivation through the accepted layers.
    canonical = m.canonical_manifest_bytes(expected)
    if (canonical != m.canonical_manifest_bytes(stored)
            or hashlib.sha256(canonical).hexdigest() != actual_digest):
        raise RawIntakeVerificationError('RAW_INTAKE_SOURCE_MISMATCH')
    if dict(result) != _summary(stored, path, actual_digest):
        raise RawIntakeVerificationError('RAW_INTAKE_SUMMARY_MISMATCH')
