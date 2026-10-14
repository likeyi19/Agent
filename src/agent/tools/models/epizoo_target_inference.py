"""Target-model eligibility and invocation of native EpiZooX inference.

The adaptation owner establishes model authority. This consumer checks the exact
inference correspondence; it does not transfer parameters or train a model.
"""
from pathlib import Path
import json

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch

from agent.tools.models.epizoo_adaptation import publication, contract as c
from agent.tools.models.epizoo_adaptation.backend import seeded, CheckedLoader
from agent.tools.models import epizoo as legacy


METHOD = 'epizoo-target-inference.v1'
SETTINGS = dict(batch_size=1, max_length=8192, random_sample=True, random_seed=0,
                num_workers=0, dtype='float32', use_amp=True)


def bind_model(path, sha, run_id=None, step_id=None):
    if run_id is not None or step_id is not None:
        if not run_id or not step_id:
            raise ValueError('Both accepted target execution coordinates are required.')
        from agent.orchestration.target_model_binding import accepted_target
        return accepted_target(path, sha, run_id, step_id)
    publication.verify_target(path, expected_sha256=sha)
    return None


def verify_result(arguments, result):
    """Verify consumed model identity and persisted numerical/cell correspondence."""
    p = result['target_model_provenance']
    path = Path(arguments['target_model_manifest_path']).resolve(strict=True)
    sha = arguments['target_model_manifest_sha256']
    authority = bind_model(path, sha, arguments.get('target_model_source_run_id'),
                           arguments.get('target_model_source_step_id'))
    manifest = publication.load_manifest(path, sha)
    expected = dict(method=METHOD, manifest_path=str(path), manifest_sha256=sha,
        checkpoint_sha256=manifest['files']['checkpoint.pth']['sha256'],
        species=manifest['reference']['species'], assembly=arguments['assembly'],
        matrix_semantics=arguments['matrix_semantics'],
        ordered_feature_sha256=manifest['preprocessing']['ordered_feature_sha256'],
        training_purpose=manifest['arguments']['profile']['purpose'],
        training_steps=manifest['execution']['completed_step'],
        settings=dict(SETTINGS, device=arguments.get('device', 'cuda:0')),
        accepted_model_authority=authority)
    if (set(p) != set(expected) | {'input_sha256', 'features_reordered', 'embedding_sha256', 'cell_ids_sha256', 'runtime'}
            or any(p.get(k) != v for k, v in expected.items())
            or p['species'] != dict(arguments['species'])
            or arguments['assembly'] != manifest['reference']['target_assembly']
            or arguments['matrix_semantics'] != manifest['arguments']['profile']['matrix_semantics']
            or result['checkpoint_path'] != str(path.parent / 'checkpoint.pth')
            or type(p['features_reordered']) is not bool):
        raise ValueError('Target inference provenance differs from the resolved scientific inputs.')
    from agent.tools.models.epizoo_adaptation.resources import runtime
    numerical = lambda value: {k: v for k, v in value.items() if k not in ('cpu_threads', 'interop_threads')}
    if (not isinstance(p['runtime'], dict) and not hasattr(p['runtime'], 'items')
            or numerical(p['runtime']) != numerical(runtime())
            or any(type(p['runtime'].get(k)) is not int or p['runtime'][k] < 1
                   for k in ('cpu_threads', 'interop_threads'))):
        raise ValueError('Invalid captured inference runtime.')
    for field, digest in (('input_path', 'input_sha256'), ('embedding_path', 'embedding_sha256'),
                           ('cell_ids_path', 'cell_ids_sha256')):
        if c.file_record(result[field])['sha256'] != p[digest]:
            raise ValueError('Target inference input or output bytes changed.')
    cells = Path(result['cell_ids_path']).read_text().splitlines()
    import h5py
    from anndata.io import read_elem
    with h5py.File(result['input_path'], 'r') as handle:
        source_cells = read_elem(handle['obs']).index.tolist()
        source_features = read_elem(handle['var']).index.tolist()
    vocabulary = (path.parent / 'features.txt').read_text().splitlines()
    if (len(source_features) != len(vocabulary) or len(set(source_features)) != len(source_features)
            or set(source_features) != set(vocabulary)
            or p['features_reordered'] != (source_features != vocabulary)):
        raise ValueError('Target feature correspondence differs from the accepted vocabulary.')
    embedding = np.load(result['embedding_path'], mmap_mode='r', allow_pickle=False)
    if (cells != source_cells or len(cells) != result['n_cells']
            or embedding.shape != (len(cells), 512) or embedding.dtype != np.float32
            or not np.isfinite(embedding).all()):
        raise ValueError('Target embedding shape, values or ordered cells differ.')


def qualify_input(adata, manifest, vocabulary, *, species, taxonomy_id, assembly,
                  matrix_semantics):
    """Return a lossless sparse column permutation into the model vocabulary."""
    reference = manifest['reference']
    declared_species = dict(scientific_name=species, taxonomy_id=taxonomy_id)
    if (type(taxonomy_id) is not int or taxonomy_id < 1
            or declared_species != reference['species']
            or assembly != reference['target_assembly']
            or matrix_semantics != manifest['arguments']['profile']['matrix_semantics']):
        raise ValueError('Target model species, assembly or measurement semantics mismatch.')
    for key, expected in (('species', declared_species), ('assembly', assembly),
                          ('matrix_semantics', matrix_semantics)):
        if key in adata.uns:
            observed = adata.uns[key]
            if key == 'species' and isinstance(observed, str):
                try:
                    observed = json.loads(observed)
                except ValueError:
                    observed = dict(scientific_name=observed, taxonomy_id=taxonomy_id)
            if observed != expected:
                raise ValueError('Conflicting scientific dataset metadata.')
    if (not sp.issparse(adata.X) or adata.X.format not in ('csr', 'csc')
            or not adata.X.has_canonical_format or adata.n_obs < 1
            or not adata.obs_names.is_unique or not adata.var_names.is_unique):
        raise ValueError('Unique axes and a nonempty canonical sparse CSR/CSC input are required.')
    values = adata.X.data
    if (values.dtype.kind not in 'iuf' or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(values > 2**53 - 1)
            or np.any(values != np.floor(values))
            or matrix_semantics == 'binary_accessibility' and np.any((values != 0) & (values != 1))):
        raise ValueError('Invalid raw measurements for the target model profile.')
    if (len(vocabulary) != adata.n_vars or len(set(vocabulary)) != len(vocabulary)
            or set(vocabulary) != set(adata.var_names)):
        raise ValueError('Dataset features must correspond exactly to the target token vocabulary.')
    positions = adata.var_names.get_indexer(vocabulary)
    # No row alignment, subset, count projection, alias repair or feature creation.
    matrix = adata.X[:, positions].tocsr(copy=True)
    matrix.eliminate_zeros()
    matrix.sort_indices()
    result = ad.AnnData(matrix, obs=adata.obs[[]].copy())
    result.var_names = vocabulary
    return result, not np.array_equal(positions, np.arange(len(positions)))


def infer(adata, manifest, root, *, device):
    """Use the native preprocessing, dataset, strict model loader and extractor."""
    from epizoo.data.processing import compute_tfidf, generate_cell_sentences
    from epizoo.data.datasets import InferenceCellDatasetX, inference_collate_fn_x
    from epizoo.models.epizoo_x import EpiZooXConfig
    from epizoo.qualification import strict_target
    from epizoo.inference.embeddings import extract_cell_embeddings
    from torch.utils.data import DataLoader

    df = np.load(root / 'document_frequency.npy', allow_pickle=False)
    if (df.dtype != np.int64 or df.shape != (adata.n_vars,)
            or np.any(df < 0) or np.any(df > manifest['preprocessing']['n_cells'])):
        raise ValueError('Invalid accepted target training-corpus document frequency.')
    transformed = compute_tfidf(adata, df=df,
        cell_number=manifest['preprocessing']['n_cells'], scale_factor=10000,
        dtype=np.float32, verbose=False)
    if not sp.issparse(transformed.X) or not np.isfinite(transformed.X.data).all():
        raise ValueError('Native target TF-IDF returned invalid values.')
    sentences = generate_cell_sentences(transformed, species=None, base_offset=4)
    with seeded(SETTINGS['random_seed']):
        dataset = InferenceCellDatasetX(sentences.obs['cell_indices'].tolist(),
            max_length=SETTINGS['max_length'], random_sample=SETTINGS['random_sample'])
        loader = DataLoader(dataset, batch_size=SETTINGS['batch_size'], shuffle=False,
            num_workers=SETTINGS['num_workers'], collate_fn=inference_collate_fn_x)
        state = torch.load(root / 'checkpoint.pth', map_location='cpu', weights_only=True, mmap=True)
        model = strict_target(state, EpiZooXConfig(vocab_size=adata.n_vars + 4,
            **manifest['arguments']['profile']['architecture']))
        try:
            result = extract_cell_embeddings(model, CheckedLoader(loader), device=device,
                use_amp=SETTINGS['use_amp'], return_numpy=True, show_progress=False)
            return np.asarray(result, dtype=np.float32)
        finally:
            del model, state
            if device.startswith('cuda'):
                torch.cuda.empty_cache()


def embed_target(input_path, output_dir, *, species, taxonomy_id, assembly,
                 matrix_semantics, target_model_manifest_path,
                 target_model_manifest_sha256, device, overwrite,
                 target_model_source_run_id=None, target_model_source_step_id=None):
    from agent.tools.analysis.epizoo_embedding import (
        _artifact_paths, _resolve_output_dir, _ensure_outputs_available,
        _validate_backend_result, _validate_cell_ids_for_text, _write_artifacts,
    )
    if device != 'cuda:0' or type(overwrite) is not bool:
        raise ValueError('Target inference requires explicit cuda:0 and boolean overwrite; no fallback.')
    path = Path(target_model_manifest_path).resolve(strict=True)
    c.checks.sha(target_model_manifest_sha256)
    # Reuses operation-local/accepted owner proof, otherwise invokes the existing
    # independent structural verifier. Never trusts a manifest as authority.
    authority = bind_model(path, target_model_manifest_sha256,
                           target_model_source_run_id, target_model_source_step_id)
    manifest = publication.load_manifest(path, target_model_manifest_sha256)
    source = Path(input_path).resolve(strict=True)
    from agent.tools.analysis.peak_h5ad import inspect as inspect_layout, MAX_BYTES
    from agent.tools.data import inspect_scATAC
    if source.stat().st_size > MAX_BYTES:
        raise ValueError('Target inference source exceeds the 1 GiB admission bound.')
    inspect_layout(source)
    if not inspect_scATAC(source)['x_is_sparse']:
        raise ValueError('Target inference requires sparse X; dense matrices are rejected.')
    source_record = c.file_record(source)
    vocabulary = (path.parent / 'features.txt').read_text().splitlines()
    data = ad.read_h5ad(source)
    ordered_cells = tuple(map(str, data.obs_names))
    _validate_cell_ids_for_text(ordered_cells)
    data, aligned = qualify_input(data, manifest, vocabulary, species=species,
        taxonomy_id=taxonomy_id, assembly=assembly, matrix_semantics=matrix_semantics)
    destination = _resolve_output_dir(output_dir)
    embeddings_path, ids_path = _artifact_paths(source, destination)
    _ensure_outputs_available(embeddings_path, ids_path, overwrite=overwrite)
    resources = [path] + [path.parent / name for name in manifest['files']]
    snapshots = {p: legacy._resource_file_snapshot(p) for p in resources}
    from agent.tools.models.epizoo_adaptation.resources import runtime
    inference_runtime = runtime()
    embeddings = infer(data, manifest, path.parent, device=device)
    _validate_backend_result(embeddings, ordered_cells, ordered_cells)
    for p, snapshot in snapshots.items():
        legacy._check_resource_snapshot(p, snapshot)
    if bind_model(path, target_model_manifest_sha256,
                  target_model_source_run_id, target_model_source_step_id) != authority:
        raise ValueError('Target authority changed during inference.')
    c.check_file(source_record)
    _write_artifacts(embeddings, ordered_cells, embeddings_path, ids_path, overwrite=overwrite)
    return dict(status='success', input_path=str(source), embedding_path=str(embeddings_path),
        cell_ids_path=str(ids_path), n_cells=len(ordered_cells), embedding_dim=512,
        embedding_dtype='float32', finite=True, cell_order_preserved=True, backend='EpiZoo',
        species=species, checkpoint_path=str(path.parent / 'checkpoint.pth'), device=device,
        target_model_provenance=dict(method=METHOD, manifest_path=str(path),
            accepted_model_authority=authority,
            runtime=inference_runtime,
            manifest_sha256=target_model_manifest_sha256,
            checkpoint_sha256=manifest['files']['checkpoint.pth']['sha256'],
            input_sha256=source_record['sha256'], species=manifest['reference']['species'],
            assembly=assembly, matrix_semantics=matrix_semantics,
            ordered_feature_sha256=manifest['preprocessing']['ordered_feature_sha256'],
            features_reordered=aligned, training_purpose=manifest['arguments']['profile']['purpose'],
            training_steps=manifest['execution']['completed_step'],
            settings=dict(SETTINGS, device=device),
            embedding_sha256=c.file_record(embeddings_path)['sha256'],
            cell_ids_sha256=c.file_record(ids_path)['sha256']))
