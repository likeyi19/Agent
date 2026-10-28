"""Accepted statistical H5AD exports through the shared snapshot/privacy owner.

Export preserves arrays and ordered axes, omits private metadata, and records the
accepted source digest. It never issues scientific authority or reruns statistics.
"""
import os
from pathlib import Path

import h5py

from . import matrix_delivery as matrix
from agent.tools.analysis import neighbors_delivery as io

CONTRACT = 'statistical-h5ad-delivery.v1'
ROLE = 'statistical_export'
TOOLS = {
    'build_replicate_pseudobulk': ('pseudobulk_path', 'pseudobulk_sha256',
        'pseudobulk.h5ad', 'Replicate pseudobulk counts'),
    'run_replicate_differential_accessibility': ('da_path', 'da_sha256',
        'differential-accessibility.h5ad', 'Differential accessibility results'),
    'transfer_cell_labels': ('annotation_path', 'annotation_sha256',
        'transferred-labels.h5ad', 'Transferred cell labels'),
}


def accepted_statistical_result(service, session_id, revision, origin, output, *, step_id=None):
    from agent.orchestration.verification_authority import _accepted_step
    from agent.orchestration.prior_outputs import accepted_step_digest
    from .interactive import _ScientificMatrixArtifact
    from .interactive_schemas import ScientificArtifactHandle
    from .session_state import digest
    run_id = revision.run_id if output is None else output.run_id
    step_id = step_id if output is None else output.step_id
    step, _, state = _accepted_step(service._application.run_store, run_id, step_id)
    anchor = accepted_step_digest(step)
    if step.tool_name not in TOOLS or state.lifecycle_status.value != 'SUCCEEDED':
        raise ValueError('No accepted statistical result.')
    key, checksum, filename, label = TOOLS[step.tool_name]
    if output is not None and (output.accepted_step_sha256 != anchor or output.output_key not in (key, checksum)):
        raise ValueError('The selected output does not identify this statistical file.')
    result = step.result
    sha = result[checksum]
    workspace = service._application._workspace
    root = workspace.runs / workspace.run_digest(run_id) / 'scientific'
    path = Path(result[key])
    with matrix._open_matrix_source(path, root) as stream:
        size = os.fstat(stream.fileno()).st_size
    identity = dict(session_id=session_id, revision_id=revision.revision_id,
        turn_id=origin.turn_id, run_id=run_id, step_id=step_id, accepted_step_sha256=anchor,
        role=ROLE, contract=CONTRACT, sha256=sha, size_bytes=size)
    ref = ScientificArtifactHandle(digest(identity), 'scientific_statistics', sha,
        revision.revision_id, origin.turn_id, size, filename, label)
    return _ScientificMatrixArtifact(ref, path, root, sha, size, CONTRACT,
        dict(tool_name=step.tool_name, accepted_step_sha256=anchor), ROLE)


def create_statistical_export(source_stream, destination_stream, manifest, *, source_sha256,
                              source_size_bytes, manifest_sha256, max_bytes):
    from agent.tools.analysis.peak_h5ad import inspect
    matrix._reject(manifest['tool_name'] not in TOOLS or not 0 < max_bytes <= matrix.MAX_MATRIX_BYTES)
    require_matrix = manifest['tool_name'] == 'build_replicate_pseudobulk'
    destination_stream.seek(0, 2)
    matrix._reject(destination_stream.tell() != 0)
    with h5py.File(source_stream, 'r') as source:
        # Shared bounded storage inspection; axis-only DA/transfer artifacts
        # legitimately omit X. Scientific eligibility was accepted upstream.
        inspect(source, require_matrix=require_matrix)
        omitted = matrix._redactions(source, preserve_annotation=False)
        matrix._reject(any(not name.startswith('/uns/') for name in omitted))
        # Preserve safe method/provenance siblings instead of dropping an entire
        # dictionary merely because one nested source locator is private.
        def redact(item):
            if (isinstance(item, h5py.Group) and item.attrs.get('encoding-type') == 'dict'
                    and not matrix._private_metadata(item, recurse=False)):
                for child in item.values():
                    if matrix._private_metadata(child):
                        redact(child)
            else:
                omitted.add(item.name)
        for name in tuple(omitted):
            if name.startswith('/uns/'):
                omitted.remove(name)
                redact(source[name])
        metadata = dict(contract_version=CONTRACT, source_sha256=source_sha256,
            source_size_bytes=source_size_bytes, tool_name=manifest['tool_name'],
            accepted_step_sha256=manifest['accepted_step_sha256'],
            omitted_private_metadata_fields=len(omitted),
            authority_scope='export-does-not-issue-scientific-authority')
        with h5py.File(io._BoundedDestination(destination_stream, max_bytes), 'w') as target:
            matrix._copy(source, target, omitted)
            uns = target.require_group('uns')
            matrix._reject(io.EXPORT_KEY in uns)
            io._write_metadata(uns.create_group(io.EXPORT_KEY), metadata)
    destination_stream.flush(); destination_stream.seek(0)
    inspect(destination_stream, privacy=True, require_matrix=require_matrix)
    destination_stream.seek(0)
