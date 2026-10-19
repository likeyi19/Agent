"""Closed safe exports of pinned embeddings, Leiden and UMAP results."""
import os
from pathlib import Path

import anndata as ad
import h5py
import numpy as np

CONTRACT = 'compact-analysis-delivery.v1'
ROLE = 'compact_analysis_export'
FILENAME = 'analysis.h5ad'
FILENAMES = {'embedding': 'embeddings.h5ad', 'clustering': 'clusters.h5ad', 'umap': FILENAME}
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
    stages = {'epizoo_embed_cells': 'embedding', 'pca_embed_cells': 'embedding', 'cluster_cells': 'clustering', 'compute_cell_umap': 'umap'}
    if (stored.tool_name not in stages or state.lifecycle_status.value != 'SUCCEEDED'
            or output is not None and output.accepted_step_sha256 != anchor
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
    stage = stages[stored.tool_name]
    filename = FILENAMES[stage]
    label = {'embedding': 'Cell embeddings — safe export',
             'clustering': 'Embeddings and Leiden — safe export',
             'umap': 'Embeddings and UMAP — safe export'
                     if result.get('clustering_present') is False else LABEL}[stage]
    reference = ScientificArtifactHandle(digest(identity), 'scientific_analysis', sha,
        revision.revision_id, origin.turn_id, size, filename, label)
    return _ScientificMatrixArtifact(reference, path, root, sha, size, CONTRACT,
        dict(n_cells=result['n_cells'], **({'stage': stage} if stage != 'umap' else {}), **({'clustering_present': False}
             if result.get('clustering_present') is False else {})), ROLE)


def create_analysis_export(source, destination, manifest, *, source_sha256,
                           source_size_bytes, manifest_sha256, max_bytes):
    from agent.tools.analysis.peak_h5ad import _PATH
    from agent.tools.analysis.neighbors_delivery import _safe_objects, _strings
    from agent.tools.analysis.embedding_analysis import _validate_neighbors_artifact, _validate_umap_labels, _validate_cluster_labels
    with h5py.File(source, 'r') as handle:
        _safe_objects(handle, max_bytes, numeric_kinds=frozenset({'i', 'f', 'b'}), max_depth=6, max_objects=160)
        def bounded_strings(name, value):
            if isinstance(value, h5py.Dataset) and h5py.check_string_dtype(value.dtype) is not None:
                for _ in _strings(value, private=True):
                    pass
        handle.visititems(bounded_strings)
        from anndata.io import read_elem
        data = read_elem(handle)
    stage = manifest.get('stage', 'umap')
    if stage not in ('embedding', 'clustering', 'umap'):
        raise ValueError('Unsupported compact analysis stage.')
    if stage != 'embedding':
        _validate_neighbors_artifact(data, expected_stages=frozenset({stage}))
    elif data.X is not None or data.n_vars or not data.obs_names.is_unique:
        raise ValueError('Invalid compact embedding structure.')
    clustered = (_validate_umap_labels(data) if stage == 'umap' else
                 bool(_validate_cluster_labels(data)) if stage == 'clustering' else False)
    if stage == 'umap' and clustered is not manifest.get('clustering_present', True):
        raise ValueError('Changed UMAP clustering variant.')
    if data.n_obs != manifest['n_cells']:
        raise ValueError('Changed analysis cell count.')
    labels = tuple(data.obs['leiden'].astype(str)) if clustered else ()
    for text in (*data.obs_names, *labels):
        if not text or _PATH.search(text) or any(ord(c) < 32 for c in text):
            raise ValueError('Cell identities or labels are ineligible for public delivery.')
    result = ad.AnnData(obs=data.obs[['leiden'] if clustered else []].copy())
    from agent.tools.analysis.embedding_analysis import representation
    if stage == 'embedding' and 'agent_cell_embedding' in data.uns:
        from agent.tools.analysis.pca_embedding import descriptor
        meta = descriptor(data); key, dimensions = meta['representation'], meta['embedding_dim']
    elif stage == 'embedding':
        key, dimensions = 'X_epizoo', 512
    else:
        key, values = representation(data); dimensions = values.shape[1]
    embedding_key = key
    for key, dimensions in ((key, dimensions),) + ((('X_umap', 2),) if stage == 'umap' else ()):
        values = np.asarray(data.obsm[key])
        if values.shape != (data.n_obs, dimensions) or not np.isfinite(values).all():
            raise ValueError('Invalid compact analysis coordinates.')
        result.obsm[key] = values.copy()
    result.uns['agent_scientific_export'] = dict(contract=CONTRACT, source_sha256=source_sha256)
    if embedding_key == 'X_pca':
        metadata = data.uns.get('agent_cell_embedding', data.uns.get('agent_milestone6', {}).get('embedding'))
        result.uns['agent_scientific_export'].update(method=metadata['method'], representation=embedding_key,
            embedding_dim=metadata['embedding_dim'], parameters=metadata['parameters'], input_sha256=metadata['source_sha256'])
    if stage != 'umap':
        result.uns['agent_scientific_export']['stage'] = stage
    elif not clustered:
        result.uns['agent_scientific_export']['clustering_present'] = False
    with h5py.File(destination, 'w') as handle:
        from anndata.io import write_elem
        write_elem(handle, '/', result)
    destination.seek(0, 2)
    if destination.tell() > max_bytes:
        raise ValueError('Compact analysis export exceeds its size bound.')
    destination.seek(0)
