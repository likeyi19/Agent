"""Closed safe export of accepted UMAP cells, embeddings and Leiden labels."""
import os
from pathlib import Path

import anndata as ad
import h5py
import numpy as np

CONTRACT = 'compact-analysis-delivery.v1'
ROLE = 'compact_analysis_export'
FILENAME = 'analysis.h5ad'
LABEL = 'Embeddings, Leiden and UMAP — safe export'


def accepted_analysis(service, session_id, revision, origin, output, *, step_id=None):
    from agent.orchestration.verification_authority import _accepted_step
    from agent.orchestration.prior_outputs import accepted_step_digest
    from agent.application.session_state import digest
    from agent.application.interactive_schemas import ScientificArtifactHandle
    from agent.application.interactive import _ScientificMatrixArtifact
    from agent.application.matrix_delivery import _open_matrix_source
    run_id = revision.run_id if output is None else output.run_id
    step_id = step_id if output is None else output.step_id
    stored, _, state = _accepted_step(service._application.run_store, run_id, step_id)
    result = stored.result
    anchor = accepted_step_digest(stored)
    if (stored.tool_name != 'compute_cell_umap' or state.lifecycle_status.value != 'SUCCEEDED'
            or output is not None and (output.output_key != 'analysis_path' or output.accepted_step_sha256 != anchor)
            or 'analysis_sha256' not in result):
        raise ValueError('No accepted pinned compact analysis is available.')
    sha = result['analysis_sha256']
    if type(sha) is not str or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
        raise ValueError('Invalid compact analysis digest.')
    workspace = service._application._workspace
    root = workspace.runs / workspace.run_digest(run_id) / 'scientific'
    path = Path(result['analysis_path'])
    with _open_matrix_source(path, root) as stream:
        size = os.fstat(stream.fileno()).st_size
    identity = dict(session_id=session_id, revision_id=revision.revision_id,
        turn_id=origin.turn_id, run_id=run_id, step_id=step_id, accepted_step=anchor,
        role=ROLE, sha256=sha, size_bytes=size)
    reference = ScientificArtifactHandle(digest(identity), 'scientific_analysis', sha,
        revision.revision_id, origin.turn_id, size, FILENAME, LABEL)
    return _ScientificMatrixArtifact(reference, path, root, sha, size, CONTRACT,
        dict(n_cells=result['n_cells']), ROLE)


def create_analysis_export(source, destination, manifest, *, source_sha256,
                           source_size_bytes, manifest_sha256, max_bytes):
    from agent.tools.analysis.peak_h5ad import _PATH
    from agent.tools.analysis.neighbors_delivery import _safe_objects, _strings
    from agent.tools.analysis.embedding_analysis import _validate_neighbors_artifact, _validate_cluster_labels
    with h5py.File(source, 'r') as handle:
        _safe_objects(handle, max_bytes, numeric_kinds=frozenset({'i', 'f', 'b'}))
        def bounded_strings(name, value):
            if isinstance(value, h5py.Dataset) and h5py.check_string_dtype(value.dtype) is not None:
                for _ in _strings(value, private=True):
                    pass
        handle.visititems(bounded_strings)
        from anndata.io import read_elem
        data = read_elem(handle)
    _validate_neighbors_artifact(data, expected_stages=frozenset({'umap'}))
    _validate_cluster_labels(data)
    if data.n_obs != manifest['n_cells']:
        raise ValueError('Changed analysis cell count.')
    for text in (*data.obs_names, *data.obs['leiden'].astype(str)):
        if not text or _PATH.search(text) or any(ord(c) < 32 for c in text):
            raise ValueError('Cell identities or labels are ineligible for public delivery.')
    result = ad.AnnData(obs=data.obs[['leiden']].copy())
    for key, dimensions in (('X_epizoo', 512), ('X_umap', 2)):
        values = np.asarray(data.obsm[key])
        if values.shape != (data.n_obs, dimensions) or not np.isfinite(values).all():
            raise ValueError('Invalid compact analysis coordinates.')
        result.obsm[key] = values.copy()
    result.uns['agent_scientific_export'] = dict(contract=CONTRACT, source_sha256=source_sha256)
    with h5py.File(destination, 'w') as handle:
        from anndata.io import write_elem
        write_elem(handle, '/', result)
    destination.seek(0, 2)
    if destination.tell() > max_bytes:
        raise ValueError('Compact analysis export exceeds its size bound.')
    destination.seek(0)
