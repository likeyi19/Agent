"""Owner-reviewed download derivatives preserve science and omit private locators."""

from copy import deepcopy
import hashlib
import io
from pathlib import Path
import shutil
import tempfile
import time

import anndata as ad
import h5py
import numpy as np
import pytest

from agent.application.matrix_delivery import MatrixPrivacyError, validate_matrix_h5ad
from agent.schemas.verification_authority import AuthorityError
from agent.tools.analysis import embedding_analysis as science
from agent.tools.analysis import neighbors_delivery as delivery
from agent.tools.analysis import neighbors_publication as owner


@pytest.fixture(scope='module')
def original_publication(tmp_path_factory):
    # Genuine production is setup only. Export tests never rebuild its graphs.
    root = tmp_path_factory.mktemp('neighbors-owner')
    embeddings, cells = root / 'private-embedding-source.npy', root / 'private-cells-source.txt'
    np.save(embeddings, np.random.default_rng(17).normal(size=(20, 512)).astype('float32'))
    cells.write_text(''.join(f'细胞-{index:02d}\n' for index in range(20)))
    result = science.build_cell_neighbors(embeddings, cells, root / 'scientific', n_neighbors=5)
    manifest, sha = owner.load_manifest(result['manifest_path'], result['manifest_sha256'])
    return result, manifest, sha


@pytest.fixture
def publication(original_publication, tmp_path):
    result, manifest, manifest_sha = original_publication
    path = tmp_path / 'source.h5ad'
    shutil.copyfile(result['analysis_path'], path)
    return path, deepcopy(manifest), manifest_sha


def export(publication, *, max_bytes=None):
    path, manifest, manifest_sha = publication
    raw = path.read_bytes()
    # Format-negative mutations deliberately receive their new byte identity;
    # accepted authority admission is separately owned and tested by Application.
    manifest['analysis'].update(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    source, destination = io.BytesIO(raw), io.BytesIO()
    maximum = max_bytes or len(raw) + delivery.DESTINATION_OVERHEAD_BYTES
    kwargs = dict(source_sha256=manifest['analysis']['sha256'], source_size_bytes=len(raw),
                  manifest_sha256=manifest_sha, max_bytes=maximum)
    delivery.create_neighbors_export(source, destination, manifest, **kwargs)
    assert not source.closed and not destination.closed
    return destination, kwargs


def test_actual_owner_export_preserves_all_scientific_values_and_explicit_identity(publication):
    path, manifest, manifest_sha = publication
    original = path.read_bytes()
    output, kwargs = export(publication)
    raw = output.getvalue()
    assert raw != original
    assert path.read_bytes() == original
    assert len(raw) <= len(original) + delivery.DESTINATION_OVERHEAD_BYTES
    assert str(Path(manifest['sources']['embedding']['path']).parent).encode() not in raw
    assert b'private-embedding-source' not in raw and b'private-cells-source' not in raw
    source, exported = ad.read_h5ad(path), ad.read_h5ad(output)
    assert tuple(exported.obs_names) == tuple(source.obs_names)
    assert exported.X is None and exported.n_vars == 0
    assert np.array_equal(exported.obsm['X_epizoo'], source.obsm['X_epizoo'])
    for graph in ('connectivities', 'distances'):
        for member in ('data', 'indices', 'indptr'):
            assert np.array_equal(getattr(exported.obsp[graph], member), getattr(source.obsp[graph], member))
    assert exported.uns['neighbors'] == source.uns['neighbors']
    assert exported.uns[science.PROVENANCE_KEY] == {
        name: value for name, value in source.uns[science.PROVENANCE_KEY].items()
        if name not in {'source_embedding_path', 'source_cell_ids_path', 'embedding'}}
    metadata = exported.uns[delivery.EXPORT_KEY]
    assert metadata['embedding_method'] == 'EpiZoo'
    assert metadata['representation'] == 'X_epizoo'
    assert metadata['embedding_dim'] == 512
    assert metadata['source_sha256'] == hashlib.sha256(original).hexdigest()
    assert metadata['source_size_bytes'] == len(original)
    assert metadata['source_manifest_sha256'] == manifest_sha
    assert metadata['source_embedding_sha256'] == manifest['sources']['embedding']['sha256']
    assert metadata['source_cell_ids_sha256'] == manifest['sources']['cell_ids']['sha256']
    assert metadata['authority_scope'] == 'export-does-not-issue-scientific-authority'
    output.seek(0)
    delivery.validate_neighbors_export(output, manifest, **kwargs)


def test_original_remains_ineligible_for_matrix_or_export_delivery(publication):
    path, manifest, sha = publication
    with pytest.raises(MatrixPrivacyError):
        validate_matrix_h5ad(path, owner.CONTRACT, manifest)
    with pytest.raises(delivery.NeighborsPrivacyError):
        delivery.validate_neighbors_export(path, manifest, source_sha256=manifest['analysis']['sha256'],
            source_size_bytes=manifest['analysis']['size_bytes'], manifest_sha256=sha,
            max_bytes=manifest['analysis']['size_bytes'] + delivery.DESTINATION_OVERHEAD_BYTES)


def test_repeat_export_has_identical_bytes_across_wall_clock_seconds(publication):
    first, _ = export(publication)
    time.sleep(1.05)
    second, _ = export(publication)
    assert first.getvalue() == second.getvalue()


def test_export_never_loads_historical_inputs_or_invokes_owner_science(publication, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('download invoked scientific reconstruction or source loads')
    for name in ('_load_embeddings', '_load_cell_ids', '_read_analysis', '_validate_neighbors_artifact',
                 'build_cell_neighbors'):
        monkeypatch.setattr(science, name, forbidden)
    for name in ('_verify', 'capture_inputs', 'check_sources'):
        monkeypatch.setattr(owner, name, forbidden)
    monkeypatch.setattr(ad, 'read_h5ad', forbidden)
    output, _ = export(publication)
    assert output.getvalue()


@pytest.mark.parametrize('target', ['/', '/obs', '/obs/_index', '/obsm/X_epizoo',
                                  '/obsp/connectivities', '/obsp/connectivities/data',
                                  '/uns', '/uns/neighbors/params/metric'])
def test_unreviewed_attributes_rejected(publication, target):
    path, _, _ = publication
    with h5py.File(path, 'r+') as file:
        file[target].attrs['source_path'] = '/srv/private/patient.h5ad'
    with pytest.raises(delivery.NeighborsPrivacyError):
        export(publication)


@pytest.mark.parametrize('target', ['/obs/private', '/raw', '/uns/private', '/uns/neighbors/private',
                                  '/uns/agent_milestone6/private', '/obsm/private'])
def test_extra_sensitive_metadata_is_rejected_not_silently_redacted(publication, target):
    path, _, _ = publication
    with h5py.File(path, 'r+') as file:
        file.create_dataset(target, data='/srv/private/patient.h5ad', dtype=h5py.string_dtype())
    with pytest.raises(delivery.NeighborsPrivacyError):
        export(publication)


@pytest.mark.parametrize('identifier', ['/home/private/person', '~/private', 'file:///srv/private',
                                      'C:\\Users\\person', 'source /srv/private', 'cell\x01', 'x' * 70_000])
def test_unsafe_ordered_identifier_rejected_without_relabeling(publication, identifier):
    path, _, _ = publication
    with h5py.File(path, 'r+') as file:
        file['obs/_index'][0] = identifier
    with pytest.raises(delivery.NeighborsPrivacyError):
        export(publication)


def test_unicode_and_scientific_identifier_punctuation_preserved(publication):
    path, _, _ = publication
    with h5py.File(path, 'r+') as file:
        file['obs/_index'][0] = '病人 甲 : A|B/sample#1'
    output, _ = export(publication)
    with h5py.File(output) as file:
        assert file['obs/_index'].asstr()[0] == '病人 甲 : A|B/sample#1'


@pytest.mark.parametrize('link', ['external', 'soft', 'alias', 'cycle'])
def test_hdf5_links_rejected_before_following(publication, link):
    path, _, _ = publication
    with h5py.File(path, 'r+') as file:
        if link == 'external':
            file['unexpected'] = h5py.ExternalLink('/srv/private.h5ad', '/X')
        elif link == 'soft':
            file['unexpected'] = h5py.SoftLink('/obs/_index')
        elif link == 'alias':
            del file['obsp/connectivities/indices']
            file['obsp/connectivities/indices'] = file['obsp/distances/indices']
        else:
            file['layers/cycle'] = file['layers']
    with pytest.raises(delivery.NeighborsPrivacyError):
        export(publication)


@pytest.mark.parametrize('storage', ['virtual', 'external', 'filter', 'compressed', 'fill'])
def test_indirect_or_unreviewed_storage_rejected(publication, storage, tmp_path):
    path, _, _ = publication
    with h5py.File(path, 'r+') as file:
        group = file['obsp/connectivities']
        values = group['data'][()]
        del group['data']
        if storage == 'virtual':
            layout = h5py.VirtualLayout(shape=values.shape, dtype=values.dtype)
            layout[:] = h5py.VirtualSource('/srv/private.h5ad', '/data', shape=values.shape)
            group.create_virtual_dataset('data', layout)
        elif storage == 'external':
            group.create_dataset('data', shape=values.shape, dtype=values.dtype,
                external=[(str(tmp_path / 'private.raw'), 0, h5py.h5f.UNLIMITED)])
        elif storage in {'filter', 'compressed'}:
            group.create_dataset('data', data=values, compression='lzf' if storage == 'filter' else 'gzip')
        else:
            group.create_dataset('data', data=values, fillvalue=17)
    with pytest.raises(delivery.NeighborsPrivacyError):
        export(publication)


def test_userblock_is_ineligible_even_with_otherwise_reviewed_content(publication, tmp_path):
    path, manifest, sha = publication
    alternative = tmp_path / 'with-userblock.h5ad'
    with h5py.File(path) as source, h5py.File(alternative, 'w', userblock_size=512) as destination:
        destination.attrs.update(dict(source.attrs))
        for name in source:
            source.copy(name, destination)
    with alternative.open('r+b') as stream:
        stream.write(b'/srv/private/patient.h5ad')
    with pytest.raises(delivery.NeighborsPrivacyError):
        export((alternative, manifest, sha))


@pytest.mark.parametrize('mismatch', ['metric', 'software', 'source_locator', 'graph_dtype', 'shape', 'stage'])
def test_closed_owner_provenance_and_representation_rejected(publication, mismatch):
    path, manifest, _ = publication
    with h5py.File(path, 'r+') as file:
        if mismatch == 'metric':
            file['uns/neighbors/params/metric'][()] = 'cosine'
        elif mismatch == 'software':
            value = '/srv/private/version'
            file['uns/agent_milestone6/software_versions/numpy'][()] = value
            manifest['summary']['software_versions']['numpy'] = value
        elif mismatch == 'source_locator':
            file['uns/agent_milestone6/source_embedding_path'][()] = '/srv/different.npy'
        elif mismatch == 'graph_dtype':
            graph = file['obsp/connectivities']
            values = graph['data'][()].astype('int64')
            del graph['data']
            graph.create_dataset('data', data=values)
        elif mismatch == 'shape':
            file['obsp/connectivities'].attrs['shape'] = [20, 21]
        else:
            file['uns/agent_milestone6/stage'][()] = 'umap'
    with pytest.raises(delivery.NeighborsPrivacyError):
        export(publication)


def test_string_reads_are_fixed_width_and_numeric_copy_chunks_bounded(publication, monkeypatch):
    original_read, original_cast = h5py.Dataset.__getitem__, h5py.Dataset.astype
    numeric_sizes, casts = [], []
    def tracked(self, selection, *args, **kwargs):
        value = original_read(self, selection, *args, **kwargs)
        if self.name.startswith(('/obsm/', '/obsp/')):
            numeric_sizes.append(value.nbytes)
        return value
    def cast(self, dtype):
        casts.append(dtype)
        return original_cast(self, dtype)
    monkeypatch.setattr(h5py.Dataset, '__getitem__', tracked)
    monkeypatch.setattr(h5py.Dataset, 'astype', cast)
    export(publication)
    assert numeric_sizes and max(numeric_sizes) <= delivery._COPY_BYTES
    assert casts and set(casts) == {'S65537'}


def test_payload_and_metadata_bounds_fail_closed(publication):
    path, manifest, sha = publication
    raw, destination = path.read_bytes(), io.BytesIO()
    kwargs = dict(source_sha256=manifest['analysis']['sha256'], source_size_bytes=len(raw),
                  manifest_sha256=sha, max_bytes=len(raw) + delivery.DESTINATION_OVERHEAD_BYTES)
    for changed in ({'source_sha256': '0' * 64}, {'source_size_bytes': len(raw) + 1},
                    {'manifest_sha256': 'invalid'}, {'max_bytes': 100},
                    {'max_bytes': delivery.MAX_EXPORT_BYTES + 1},
                    {'max_bytes': len(raw) + delivery.DESTINATION_OVERHEAD_BYTES + 1}):
        with pytest.raises(delivery.NeighborsPrivacyError):
            delivery.create_neighbors_export(io.BytesIO(raw), destination, manifest, **dict(kwargs, **changed))
        assert destination.getvalue() == b''
    path.write_bytes(b'not HDF5')
    with pytest.raises(delivery.NeighborsPrivacyError):
        export(publication)


def test_fresh_destination_required_and_export_metadata_cannot_be_rewritten(publication):
    output, kwargs = export(publication)
    _, manifest, _ = publication
    with pytest.raises(delivery.NeighborsPrivacyError):
        delivery.create_neighbors_export(io.BytesIO(publication[0].read_bytes()), output, manifest, **kwargs)
    with h5py.File(output, 'r+') as file:
        file['uns'][delivery.EXPORT_KEY]['source_sha256'][()] = '0' * 64
    with pytest.raises(delivery.NeighborsPrivacyError):
        delivery.validate_neighbors_export(output, manifest, **kwargs)


def test_hdf5_destination_seeks_cannot_expand_past_transport_reservation():
    with tempfile.TemporaryFile('w+b') as stream:
        bounded = delivery._BoundedDestination(stream, 8)
        bounded.write(b'1234')
        assert bounded.truncate() == 4
        bounded.seek(9)
        with pytest.raises(delivery.NeighborsPrivacyError):
            bounded.truncate()
        with pytest.raises(delivery.NeighborsPrivacyError):
            bounded.write(b'x')
        with pytest.raises(delivery.NeighborsPrivacyError):
            bounded.truncate(9)
        stream.seek(0, 2)
        assert stream.tell() == 4


def test_owner_byte_parsers_preserve_existing_contract_without_reopening(original_publication, monkeypatch):
    result, manifest, sha = original_publication
    raw = Path(result['manifest_path']).read_bytes()
    receipt_path = Path(result['manifest_path'])
    receipt_raw = receipt_path.with_name(receipt_path.name.replace('.manifest.json', '.receipt.json')).read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError('byte parser reopened scientific state')
    monkeypatch.setattr(Path, 'open', forbidden)
    assert owner.load_manifest_bytes(raw, result['manifest_path'], sha) == (manifest, sha)
    assert owner.read_json_bytes(receipt_raw)[0]['execution_identity'] is None
    with pytest.raises(AuthorityError):
        owner.read_json_bytes(b'x' * (owner.MAX_BYTES + 1))
    with pytest.raises(AuthorityError):
        owner.read_json_bytes(b'{"value": NaN}')
    with pytest.raises(ValueError):
        owner.read_json_bytes(b'{"value":1,"value":2}')
    with pytest.raises(AuthorityError):
        owner.load_manifest_bytes(raw, result['manifest_path'], '0' * 64)
