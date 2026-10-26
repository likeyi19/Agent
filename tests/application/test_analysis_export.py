"""W4 delivery: simulated accepted science, real H5AD/HTTP/snapshot behavior.

No PCA, graph fit, clustering, UMAP, annotation or model inference is executed.
"""
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scipy import sparse

from agent.application import analysis_delivery as delivery
from agent.application.matrix_delivery import MatrixDownloadPreparer, MatrixDeliveryError
from agent.schemas import AgentRequest
from agent.tools.analysis import embedding_analysis as science, pca_embedding as pca
from agent.web.app import create_app
from test_dialogue_evidence import app, accepted, forbid_work, files
from test_interactive_rich import read_only_boundary


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_fixture(root, *, encoding='csr', stage='umap', source=None):
    """Actual AnnData encodings; coordinates/method records are simulated."""
    root.mkdir(parents=True, exist_ok=True)
    ids = pd.Index(['cell-a', 'cell-b', 'cell-c', 'cell-d'])
    if source is None:
        source = root/'original.h5ad'
        x = np.array([[0., 1.5, 0.], [2., 0., 4.], [1., 0., 0.], [0., 3., 2.]])
        matrix = {'csr': sparse.csr_matrix, 'csc': sparse.csc_matrix, 'dense': np.asarray}[encoding](x)
        obs = pd.DataFrame(dict(cohort=pd.Categorical(['treated', 'control', None, 'control']),
            leiden=pd.Categorical(['source-0', 'source-0', 'source-1', 'source-1']),
            private_location=['/private/lab/source']*4), index=ids)
        var = pd.DataFrame(dict(feature_class=['peak']*3, weight=[1., 2., 3.]), index=['peak-a', 'peak-b', 'peak-c'])
        original = ad.AnnData(matrix, obs=obs, var=var, layers={'accessibility': matrix.copy()},
            obsm={'X_pca': np.ones((4, 2)), 'source_embedding': np.arange(12).reshape(4, 3)},
            obsp={'source_graph': sparse.eye(4, format='csr')},
            varm={'source_loadings': np.arange(6).reshape(3, 2)},
            uns={'measurement_semantics': 'supplied accessibility values; not fragment counts',
                 'safe_metadata': {'release': 'fixture', 'dependencies': {'analysis_path': {'sha256': 'a'*64}}},
                 'credentials': 'must-not-leak',
                 'private_metadata': {'location': '/private/lab/source'}})
        original.raw = original
        original.write_h5ad(source, compression='gzip', convert_strings_to_categoricals=False)
    original = ad.read_h5ad(source)
    data = ad.AnnData(obs=pd.DataFrame(index=ids), obsm={'X_pca': np.arange(8, dtype='float32').reshape(4, 2)})
    descriptor = dict(contract=pca.CONTRACT, method='PCA', representation='X_pca', embedding_dim=2,
        embedding_dtype='float32', parameters=pca._parameters(2, 17), publication_path='/private/pca/manifest.json',
        source_path=str(source), source_sha256=sha(source), cell_order_sha256=science._cell_order_digest(tuple(ids)),
        feature_order_sha256=science._cell_order_digest(tuple(original.var_names)),
        software_versions={'scanpy': 'simulated-embedding-version'})
    if stage == 'embedding':
        data.uns[pca.KEY] = descriptor
    else:
        params = dict(neighbors=dict(n_neighbors=3, metric='cosine', method='umap', transformer='none',
            random_seed=23, use_rep='X_pca'), clustering=dict(algorithm='leiden', resolution=.75, flavor='igraph',
            n_iterations=2, directed=False, use_weights=True, random_seed=29, key_added='leiden'))
        data.obs['leiden'] = pd.Categorical(['0', '0', '1', '1'])
        if stage == 'umap':
            data.obsm['X_umap'] = np.array([[0, 1], [1, 0], [2, 3], [3, 2]], dtype='float32')
            params['umap'] = dict(min_dist=.3, spread=1.2, n_components=2, init_pos='spectral', random_seed=31,
                                  key_added='X_umap')
        data.uns[science.PROVENANCE_KEY] = dict(schema_version=1, stage=stage, embedding=descriptor,
            cell_order_sha256=descriptor['cell_order_sha256'], parameters=params,
            software_versions={'scanpy': 'simulated-' + stage + '-version'})
    native = root/(stage + '.h5ad'); data.write_h5ad(native)
    manifest = dict(n_cells=4, stage=stage, provenance=dict(revision_id='fixture-revision', run_id='fixture-run',
        step_id='fixture-step', accepted_step_sha256='a'*64),
        original_source=dict(path=str(source), sha256=sha(source), size_bytes=source.stat().st_size))
    return SimpleNamespace(source=source, native=native, manifest=manifest, original=original, result=data)


def artifact_for(fixture, *, integrated=False):
    return SimpleNamespace(path=fixture.native, managed_root=fixture.native.parent,
        sha256=sha(fixture.native), size_bytes=fixture.native.stat().st_size, contract=delivery.CONTRACT,
        manifest=fixture.manifest, manifest_sha256=None,
        role=delivery.INTEGRATED_ROLE if integrated else delivery.ROLE,
        reference=SimpleNamespace(filename='integrated-' + delivery.FILENAMES[fixture.manifest['stage']]))


def download(artifact):
    preparer = MatrixDownloadPreparer()
    with preparer.prepare(artifact) as snapshot:
        raw = b''.join(snapshot.iter_chunks())
        assert hashlib.sha256(raw).hexdigest() == snapshot.sha256
        assert snapshot.source_sha256 == artifact.sha256
    assert preparer.active_count == preparer._reserved_bytes == 0
    return ad.read_h5ad(io.BytesIO(raw)), raw


@pytest.mark.parametrize('stage', ['embedding', 'clustering', 'umap'])
def test_compact_scope_actual_parameters_and_owner_versions(tmp_path, stage):
    fixture = export_fixture(tmp_path, stage=stage)
    original = fixture.native.read_bytes()
    result, _ = download(artifact_for(fixture))
    assert result.X is None and result.n_vars == 0
    assert result.obs_names.equals(fixture.result.obs_names)
    assert result.obs.equals(fixture.result.obs)
    for key in result.obsm:
        np.testing.assert_array_equal(result.obsm[key], fixture.result.obsm[key])
    meta = result.uns[delivery.EXPORT_KEY]
    assert meta['format'] == 'compact_analysis_result_h5ad' and 'X=None' in meta['scope']
    assert bool(meta['clustering_present']) is (stage != 'embedding')
    methods = meta['methods']
    assert methods['embedding']['parameters']['random_seed'] == 17
    assert methods['embedding']['parameters']['normalization'] == 'none'
    assert methods['embedding']['source_sha256'] == sha(fixture.source)
    assert methods['embedding']['software_versions']['scanpy'] == 'simulated-embedding-version'
    if stage != 'embedding':
        assert methods['clustering']['parameters']['resolution'] == .75
        assert methods['neighbors']['parameters']['metric'] == 'cosine'
        # A later publication's software version must not be attributed to an
        # earlier native method whose original version was not retained there.
        assert 'software_versions' not in methods['neighbors']
    if stage == 'umap':
        assert methods['umap']['parameters']['random_seed'] == 31
        assert methods['umap']['software_versions']['scanpy'] == 'simulated-umap-version'
    assert meta['accepted_result']['revision_id'] == 'fixture-revision'
    assert meta['source_sha256'] == sha(fixture.native)
    assert str(tmp_path) not in str(result.uns) and '/private/' not in str(result.uns)
    assert fixture.native.read_bytes() == original


@pytest.mark.parametrize('encoding', ['csr', 'csc', 'dense'])
def test_integrated_preserves_measurements_axes_source_content_and_privacy(tmp_path, monkeypatch, encoding):
    fixture = export_fixture(tmp_path, encoding=encoding)
    source_bytes = fixture.source.read_bytes()
    def no_densification(*args, **kwargs):
        pytest.fail('Scientific export densified a sparse matrix.')
    monkeypatch.setattr(sparse.csr_matrix, 'toarray', no_densification)
    monkeypatch.setattr(sparse.csc_matrix, 'toarray', no_densification)
    exported, raw = download(artifact_for(fixture, integrated=True))
    original = fixture.original
    assert type(exported.X) is type(original.X)
    if sparse.issparse(original.X):
        for key in ('data', 'indices', 'indptr'):
            np.testing.assert_array_equal(getattr(exported.X, key), getattr(original.X, key))
    else:
        np.testing.assert_array_equal(exported.X, original.X)
    assert exported.obs_names.equals(original.obs_names) and exported.var_names.equals(original.var_names)
    pd.testing.assert_frame_equal(exported.obs[original.obs.columns.drop('private_location')], original.obs.drop(columns='private_location'))
    pd.testing.assert_frame_equal(exported.var, original.var)
    assert exported.raw.var_names.equals(original.raw.var_names)
    for slot in ('layers', 'obsp', 'varm'):
        for key, values in getattr(original, slot).items():
            actual = getattr(exported, slot)[key]
            if sparse.issparse(values):
                assert (values != actual).nnz == 0
            else:
                np.testing.assert_array_equal(values, actual)
    np.testing.assert_array_equal(exported.obsm['X_pca'], original.obsm['X_pca'])
    np.testing.assert_array_equal(exported.obsm['agent_X_pca'], fixture.result.obsm['X_pca'])
    np.testing.assert_array_equal(exported.obsm['X_umap'], fixture.result.obsm['X_umap'])
    pd.testing.assert_series_equal(exported.obs['agent_leiden'], fixture.result.obs['leiden'], check_names=False)
    meta = exported.uns[delivery.EXPORT_KEY]
    assert meta['format'] == 'integrated_annotated_h5ad'
    assert bool(meta['clustering_present']) is True
    assert meta['fields']['obsm'] == {'X_pca': 'agent_X_pca', 'X_umap': 'X_umap'}
    assert meta['fields']['obs'] == {'leiden': 'agent_leiden'}
    assert meta['original_source_sha256'] == sha(fixture.source)
    assert meta['omitted_private_metadata_fields'] == 3
    assert exported.uns['measurement_semantics'] == original.uns['measurement_semantics']
    assert exported.uns['safe_metadata'] == original.uns['safe_metadata']
    assert 'credentials' not in exported.uns and b'must-not-leak' not in raw and b'/private/' not in raw
    with h5py.File(fixture.source) as left, h5py.File(io.BytesIO(raw)) as right:
        key = 'X' if encoding == 'dense' else 'X/data'
        assert left[key].compression == right[key].compression == 'gzip'
    assert fixture.source.read_bytes() == source_bytes


@pytest.mark.parametrize('integrated', [False, True])
def test_unclustered_export_marker_describes_selected_result_not_original_labels(tmp_path, integrated):
    fixture = export_fixture(tmp_path)
    # A simulated label-free selected result; the original data still has its
    # own unrelated Leiden column. Integration must not claim we computed it.
    del fixture.result.obs['leiden']
    del fixture.result.uns[science.PROVENANCE_KEY]['parameters']['clustering']
    fixture.result.uns[science.PROVENANCE_KEY]['clustering_present'] = False
    fixture.result.write_h5ad(fixture.native)
    fixture.manifest['clustering_present'] = False
    before = {p: p.read_bytes() for p in (fixture.source, fixture.native)}
    exported, _ = download(artifact_for(fixture, integrated=integrated))
    metadata = exported.uns[delivery.EXPORT_KEY]
    assert bool(metadata['clustering_present']) is False
    assert 'clustering' not in metadata['methods']
    assert metadata['source_sha256'] == sha(fixture.native)
    assert exported.obs_names.equals(fixture.result.obs_names)
    assert 'agent_leiden' not in exported.obs
    if integrated:
        pd.testing.assert_series_equal(exported.obs['leiden'], fixture.original.obs['leiden'])
        assert metadata['fields']['obs'] == {}
    else:
        assert 'leiden' not in exported.obs and exported.X is None
    for key in ('X_pca', 'X_umap'):
        destination = metadata['fields']['obsm'][key] if integrated else key
        np.testing.assert_array_equal(exported.obsm[destination], fixture.result.obsm[key])
    assert all(p.read_bytes() == value for p, value in before.items())


def test_legacy_anndata_category_references_are_rebound_to_the_export(tmp_path):
    original = export_fixture(tmp_path/'source')
    with h5py.File(original.source, 'r+') as handle:
        obs = handle['obs']; obs.attrs['encoding-version'] = '0.1.0'
        categories = obs.create_group('__categories')
        for key in ('cohort', 'leiden'):
            values = original.original.obs[key]
            del obs[key]
            from anndata.io import write_elem
            write_elem(obs, key, values.cat.codes.to_numpy())
            write_elem(categories, key, np.asarray(values.cat.categories, dtype=object))
            obs[key].attrs['categories'] = categories[key].ref
            categories[key].attrs['ordered'] = values.cat.ordered
    # Construct the simulated accepted descriptor against the actual legacy
    # bytes, rather than pretending a later rewrite retained source identity.
    fixture = export_fixture(tmp_path/'result', source=original.source)
    before = fixture.source.read_bytes()
    exported, raw = download(artifact_for(fixture, integrated=True))
    pd.testing.assert_series_equal(exported.obs['cohort'], fixture.original.obs['cohort'])
    pd.testing.assert_series_equal(exported.obs['leiden'], fixture.original.obs['leiden'])
    pd.testing.assert_series_equal(exported.obs['agent_leiden'], fixture.result.obs['leiden'], check_names=False)
    with h5py.File(io.BytesIO(raw)) as handle:
        target = handle[handle['obs/cohort'].attrs['categories']]
        assert target.name == '/obs/__categories/cohort'
    assert fixture.source.read_bytes() == before


@pytest.mark.parametrize('failure', ['cells', 'source_bytes', 'collision'])
def test_integration_failure_keeps_compact_available_and_releases_capacity(tmp_path, failure):
    fixture = export_fixture(tmp_path)
    if failure == 'cells':
        source = fixture.original.copy(); source.obs_names = list(reversed(source.obs_names)); source.raw = None
        source.write_h5ad(fixture.source)
        fixture.manifest['original_source'].update(sha256=sha(fixture.source), size_bytes=fixture.source.stat().st_size)
    elif failure == 'source_bytes':
        with fixture.source.open('r+b') as stream:
            stream.seek(100); stream.write(b'changed')
    else:
        source = fixture.original.copy(); source.obsm['agent_X_pca'] = np.zeros((4, 2))
        source.write_h5ad(fixture.source)
        fixture.manifest['original_source'].update(sha256=sha(fixture.source), size_bytes=fixture.source.stat().st_size)
    preparer = MatrixDownloadPreparer()
    with pytest.raises(MatrixDeliveryError) as exc:
        preparer.prepare(artifact_for(fixture, integrated=True))
    assert exc.value.code == 'SCIENTIFIC_DOWNLOAD_UNAVAILABLE'
    assert preparer.active_count == preparer._reserved_bytes == 0
    compact, _ = download(artifact_for(fixture))
    assert compact.X is None and compact.n_obs == 4


def accepted_export(app, monkeypatch):
    """Use existing accepted-Run fixtures; the PCA authority anchor is simulated.

    Actual source lineage resolution, digests, privacy, cells, Session/Revision
    binding and transfer code run. This does not qualify a PCA eigenspace.
    """
    _, paths = app._prepare_request(AgentRequest('initial', 'Fixture', {}))
    fixture = export_fixture(paths.scientific, stage='embedding')
    rid = accepted(app, 'pca_embed_cells', overrides=dict(status='success', input_path=str(fixture.source),
        embedding_path=str(fixture.native), analysis_path=str(fixture.native), analysis_sha256=sha(fixture.native),
        source_sha256=sha(fixture.source), n_cells=4, n_features=3, embedding_dim=2, embedding_dtype='float32',
        method='PCA', representation='X_pca', parameters=pca._parameters(2, 17),
        software_versions={'scanpy': 'simulated-embedding-version'}, backend='Scanpy',
        finite=True, cell_order_preserved=True), arguments={'input_path': str(fixture.source)})[0]
    from agent.orchestration import verification_authority as authority
    original = authority._accepted
    def simulated_anchor(store, run_id, step_id):
        if (run_id, step_id) == ('initial:run', 'step0'):
            return (authority._accepted_step(store, run_id, step_id)[0], None, None, 'simulated')
        return original(store, run_id, step_id)
    monkeypatch.setattr(authority, '_accepted', simulated_anchor)
    return fixture, rid


def add_fixture_figures(app):
    """Two simulated scientific identities and a physical report copy of one."""
    from PIL import Image
    from agent.application.session_state import CompletionFile
    state = app.sessions.load('session'); turn = state.turn('initial')
    paths = app._workspace.run_paths(turn.run_id)
    pictures = [paths.visualizations/'first.png', paths.visualizations/'second.png', paths.report/'first.png']
    for path, color in zip(pictures, ('red', 'blue', 'red')):
        Image.new('RGB', (16, 16), color).save(path)
    manifest = paths.visualizations/'visualization_manifest.json'
    manifest.write_text(json.dumps(dict(source=dict(run_id=turn.run_id), figures=[dict(
        figure_id='fixture-' + str(index), relative_path=path.name, png_sha256=sha(path), figure_kind='pca_projection',
        source_step_ids=['step0'], source_artifacts=[dict(producing_step_id='step0', tool_name='pca_embed_cells',
            artifact_kind='simulated_figure', artifact_path='/private/fixture', integrity={})])
        for index, path in enumerate(pictures[:2])])), encoding='utf-8')
    pins = tuple(CompletionFile(str(path), sha(path)) for path in (*pictures, manifest))
    app.sessions._store._write(replace(state, turns=(replace(turn, completion_files=turn.completion_files + pins),)))
    return pictures


def test_exact_revision_handles_streamed_files_and_failed_export_leave_science_unchanged(app, monkeypatch):
    fixture, rid = accepted_export(app, monkeypatch)
    facade = read_only_boundary(forbid_work(app, monkeypatch))
    before = files(app)
    with TestClient(create_app(facade)) as client:
        base = f'/api/v1/sessions/session/revisions/{rid}'
        inventory = client.get(base + '/scientific-artifacts').json()['artifacts']
        compact, integrated = (next(a for a in inventory if a['artifact_type'] == kind)
                               for kind in ('scientific_analysis', 'scientific_integrated'))
        assert compact['handle'] != integrated['handle']
        assert integrated['purpose'] == 'Integrated annotated H5AD'
        assert integrated['size_bytes'] is None
        assert 'prepared when downloaded' in integrated['description']
        assert compact['size_bytes'] == fixture.native.stat().st_size
        # The established handle is independent of presentation-size semantics.
        from agent.application.session_state import digest
        assert integrated['handle'] == digest(dict(compact_handle=compact['handle'], role=delivery.INTEGRATED_ROLE,
            original_sha256=sha(fixture.source), original_size_bytes=fixture.source.stat().st_size))
        artifact = facade._scientific_artifact('session', rid, integrated['handle'])
        assert artifact.size_bytes == fixture.native.stat().st_size
        assert artifact.manifest['original_source']['size_bytes'] == fixture.source.stat().st_size
        assert compact['purpose'] == 'Compact analysis-result H5AD'
        assert 'X=None' in compact['description']
        revision = client.get(base).json()
        assert revision['scientific_artifacts'] == inventory
        for item in inventory:
            url = base + '/scientific-artifacts/' + item['handle']
            result = client.get(url)
            assert result.status_code == 200, result.text
            assert result.headers['content-disposition'] == f'attachment; filename="{item["filename"]}"'
            assert result.headers['x-artifact-content-sha256'] == hashlib.sha256(result.content).hexdigest()
            assert int(result.headers['content-length']) == len(result.content)
            decoded = ad.read_h5ad(io.BytesIO(result.content))
            assert (decoded.X is None) is (item is compact)
            assert client.get(url.replace(rid, 'another-revision')).status_code == 404
        # Force only the optional delivery builder to fail; no state is revoked.
        def unavailable(*args, **kwargs): raise ValueError('Private /server/problem')
        monkeypatch.setattr(delivery, 'create_integrated_export', unavailable)
        url = base + '/scientific-artifacts/' + integrated['handle']
        failed = client.get(url)
        assert failed.status_code == 404 and '/server/' not in failed.text
        assert client.get(base).json()['status'] == 'succeeded'
        assert client.get(base + '/scientific-artifacts/' + compact['handle']).status_code == 200
        busy_page = client.get(url, headers={'Accept': 'text/html'})
        assert busy_page.headers['content-type'].startswith('text/html')
        assert 'Accepted scientific results remain available' in busy_page.text
    assert files(app) == before
    assert facade._matrix_downloads.active_count == 0


def test_unqualified_original_source_does_not_acquire_integration_by_equal_axes(app, monkeypatch):
    # Keep the synthetic result accepted as compact history but do not install
    # its simulated authority. Equal cells/shape cannot qualify an original file.
    with monkeypatch.context() as fixture_patch:
        fixture, rid = accepted_export(app, fixture_patch)
    facade = read_only_boundary(forbid_work(app, monkeypatch))
    inventory = facade.scientific_artifact_handles('session', rid)
    assert [a.artifact_type for a in inventory] == ['scientific_analysis']
    with facade.prepare_scientific_artifact('session', rid, inventory[0].handle) as download:
        assert download.filename == 'embeddings.h5ad'


def test_figure_inventory_keeps_old_handles_while_marking_report_copies(app, monkeypatch):
    _, rid = accepted_export(app, monkeypatch)
    pictures = add_fixture_figures(app)
    facade = read_only_boundary(forbid_work(app, monkeypatch))
    before = files(app)
    figures = facade.artifact_handles('session', rid)
    assert len(figures) == 3 and len({a.handle for a in figures}) == 3
    identities = [(a.figure['manifest_sha256'], a.figure['figure_id']) for a in figures]
    assert len(set(identities)) == 2 and identities[0] == identities[2]
    for artifact, path in zip(figures, pictures):
        assert facade.artifact_content('session', rid, artifact.handle).data == path.read_bytes()
    assert files(app) == before


def test_legacy_epizoo_export_uses_recorded_inference_settings_without_qualifying_a_source(app, monkeypatch):
    _, paths = app._prepare_request(AgentRequest('initial', 'Fixture', {}))
    path = paths.scientific/'embedding.h5ad'
    data = ad.AnnData(obs=pd.DataFrame(index=['a', 'b', 'c', 'd']),
                      obsm={'X_epizoo': np.ones((4, 512), dtype='float32')})
    data.write_h5ad(path)
    settings = dict(batch_size=2, max_length=600, random_sample=True, random_seed=19,
                    num_workers=0, device='cpu', dtype='float32', amp_requested=False, amp_enabled=False,
                    show_progress=False, overwrite=False)
    rid = accepted(app, 'epizoo_embed_cells', overrides=dict(analysis_path=str(path), analysis_sha256=sha(path),
        n_cells=4, embedding_dim=512, embedding_dtype='float32', backend='EpiZoo', species='human', device='cpu',
        resource_provenance=dict(inference_settings=settings, expected_resource_identity={'checkpoint_sha256':'a'*64})),
        arguments={'device':'cpu'})[0]
    facade = read_only_boundary(forbid_work(app, monkeypatch))
    inventory = facade.scientific_artifact_handles('session', rid)
    assert [a.artifact_type for a in inventory] == ['scientific_analysis']
    with facade.prepare_scientific_artifact('session', rid, inventory[0].handle) as snapshot:
        result = ad.read_h5ad(io.BytesIO(b''.join(snapshot.iter_chunks())))
    methods = result.uns[delivery.EXPORT_KEY]['methods']['embedding']
    assert methods['method'] == 'EpiZoo' and methods['parameters'] == settings
    assert methods['resource_identity']['checkpoint_sha256'] == 'a'*64
    assert methods['recorded_execution_arguments'] == {'device':'cpu'}
