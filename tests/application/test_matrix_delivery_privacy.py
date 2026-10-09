"""Original matrix delivery eligibility uses the actual normalized layouts."""

import io
import json
import sqlite3
from types import SimpleNamespace
from types import MappingProxyType

import h5py
import numpy as np
import pytest

from agent.application.matrix_delivery import MatrixPrivacyError, validate_matrix_h5ad
from agent.tools.data import _cell_by_ccre_io as writer
from agent.tools.data import scatac_matrix_contract as canonical
from agent.tools.data import fragment_feature_matrix_contract as explicit
from agent.tools.data import selected_feature_matrix_contract as selected


CREATED = (canonical.CONTRACT, explicit.CONTRACT, selected.CONTRACT)
ADOPTED = ('scatac-cell-by-ccre.external.v1', 'scatac-cell-by-features.external.v1')
SPECIES = {'scientific_name': 'Macaca fascicularis', 'taxonomy_id': 9541}


@pytest.fixture
def make_matrix(tmp_path):
    serial = 0

    def make(contract=canonical.CONTRACT, *, cell='scatac-cell.v1.bGFi.QUM', assembly=None):
        nonlocal serial
        serial += 1
        path = tmp_path / f'{serial}.h5ad'
        neutral = contract.startswith('scatac-cell-by-features')
        species = SPECIES if neutral else 'human'
        assembly = assembly or ('macFas5' if neutral else 'hg38')
        budget = SimpleNamespace(check=lambda: None)
        with sqlite3.connect(':memory:') as database:
            if contract in CREATED:
                contract_module = {canonical.CONTRACT: canonical, explicit.CONTRACT: explicit,
                                   selected.CONTRACT: selected}[contract]
                database.execute('CREATE TABLE cells(row INTEGER, rendered TEXT, ns TEXT, bc TEXT)')
                database.execute('INSERT INTO cells VALUES(0,?,?,?)', (cell, 'lab', 'AC'))
                database.execute('CREATE TABLE features(col INTEGER,name TEXT,chrom TEXT,start INTEGER,end INTEGER)')
                database.executemany('INSERT INTO features VALUES(?,?,?,?,?)',
                                     [(0, 'chr1:0-2', 'chr1', 0, 2), (1, 'chr1:2-4', 'chr1', 2, 4)])
                bound = SimpleNamespace(contract=contract_module,
                    selection={'selected_count': 1, 'ordered_selected_sha256': 'a' * 64},
                    reference=SimpleNamespace(species=species, target_assembly=assembly,
                                             ccre=SimpleNamespace(feature_count=2, ordered_feature_sha256='b' * 64)))
                summary = writer.write_h5(path, database, bound, [([0, 1], [2, 3])], budget)
                manifest = dict(contract_version=contract, shape=[1, 2], species=species, assembly=assembly,
                                profile={'profile_id': contract_module.PROFILE.profile_id},
                                profile_sha256=contract_module.PROFILE_SHA256,
                                ordered_selected_sha256='a' * 64, ordered_feature_sha256='b' * 64,
                                logical_matrix_sha256=summary['logical_matrix_sha256'])
            else:
                # This is the closed output portion of _external_matrix_io.build;
                # it never performs source admission or scientific reconstruction.
                with h5py.File(path, 'w') as file:
                    file.attrs.update({'encoding-type': 'anndata', 'encoding-version': '0.1.0'})
                    for name in ('layers', 'obsm', 'obsp', 'varm', 'varp', 'uns'):
                        file.create_group(name).attrs.update({'encoding-type': 'dict', 'encoding-version': '0.1.0'})
                    database.execute('CREATE TABLE cells(name TEXT)')
                    database.execute('INSERT INTO cells VALUES(?)', (cell,))
                    database.execute('CREATE TABLE features(name TEXT)')
                    database.executemany('INSERT INTO features VALUES(?)', [('chr1:0-2',), ('chr1:2-4',)])
                    for name, length, query in (('obs', 1, 'SELECT name FROM cells'),
                                                ('var', 2, 'SELECT name FROM features')):
                        writer.write_axis(file, name, ('_index',), ('_index',), database.execute(query), length, budget)
                    matrix = file.create_group('X')
                    matrix.attrs.update({'encoding-type': 'csr_matrix', 'encoding-version': '0.1.0',
                                         'shape': np.array([1, 2], dtype='int64')})
                    for name, values in [('data', [2, 3]), ('indices', [0, 1]), ('indptr', [0, 2])]:
                        matrix.create_dataset(name, data=values, dtype='int64', compression='gzip')
                    metadata = dict(species=json.dumps(species, sort_keys=True, separators=(',', ':')) if neutral else species,
                                    assembly=assembly, matrix_semantics='fragment_counts', reference_identity_sha256='c' * 64)
                    for name, value in metadata.items():
                        file['uns'].create_dataset(name, data=value, dtype=h5py.string_dtype()).attrs.update(
                            {'encoding-type': 'string', 'encoding-version': '0.2.0'})
                manifest = dict(contract_version=contract, shape=[1, 2], species=species, assembly=assembly,
                                matrix_semantics='fragment_counts', reference={'identity_sha256': 'c' * 64})
        return path, manifest

    return make


@pytest.mark.parametrize('contract', CREATED + ADOPTED)
def test_actual_normalized_matrix_layouts_pass_and_remain_unchanged(make_matrix, contract):
    path, manifest = make_matrix(contract)
    original = path.read_bytes()
    validate_matrix_h5ad(path, contract, manifest)
    assert path.read_bytes() == original


def test_seekable_stream_is_supported_and_left_open(make_matrix):
    path, manifest = make_matrix()
    snapshot = io.BytesIO(path.read_bytes())
    validate_matrix_h5ad(snapshot, canonical.CONTRACT, manifest)
    assert not snapshot.closed
    snapshot.seek(0)
    assert snapshot.read() == path.read_bytes()


def test_frozen_accepted_manifest_supported(make_matrix):
    path, manifest = make_matrix(ADOPTED[1])
    manifest['species'] = MappingProxyType(manifest['species'])
    manifest['shape'] = tuple(manifest['shape'])
    validate_matrix_h5ad(path, ADOPTED[1], MappingProxyType(manifest))


@pytest.mark.parametrize('cell', ['patient-A/replicate-1', '病人 甲 : A|B', 'sample#1.ATCG-1'])
def test_scientific_identifiers_are_preserved(make_matrix, cell):
    path, manifest = make_matrix(ADOPTED[1], cell=cell)
    validate_matrix_h5ad(path, ADOPTED[1], manifest)
    with h5py.File(path) as file:
        assert file['obs/_index'].asstr()[0] == cell


@pytest.mark.parametrize('target', ['/', '/obs', '/obs/_index', '/var/chrom', '/X', '/X/data', '/uns', '/layers'])
def test_unreviewed_attributes_rejected(make_matrix, target):
    path, _ = make_matrix()
    with h5py.File(path, 'r+') as file:
        file[target].attrs['source_path'] = '/home/operator/private.h5ad'
    with pytest.raises(MatrixPrivacyError):
        validate_matrix_h5ad(path, canonical.CONTRACT)


@pytest.mark.parametrize('target', ['/uns/source_path', '/obs/private', '/var/private', '/obsm/private', '/raw'])
def test_unreviewed_metadata_and_structures_rejected(make_matrix, target):
    path, _ = make_matrix()
    with h5py.File(path, 'r+') as file:
        file.create_dataset(target, data='/home/operator/private.h5ad', dtype=h5py.string_dtype())
    with pytest.raises(MatrixPrivacyError):
        validate_matrix_h5ad(path, canonical.CONTRACT)


@pytest.mark.parametrize('text', ['/home/operator/private.h5ad', 'C:\\Users\\operator\\private.h5ad',
                                 '~/private.h5ad', 'file:///home/operator/private.h5ad',
                                 'source /srv/science/private.h5ad'])
def test_private_paths_in_reviewed_scientific_metadata_rejected(make_matrix, text):
    path, _ = make_matrix(ADOPTED[1], assembly=text)
    with pytest.raises(MatrixPrivacyError):
        validate_matrix_h5ad(path, ADOPTED[1])


def test_private_path_as_scientific_identifier_requires_a_future_export(make_matrix):
    path, _ = make_matrix(ADOPTED[0], cell='/home/operator/private.h5ad')
    with pytest.raises(MatrixPrivacyError):
        validate_matrix_h5ad(path, ADOPTED[0])


@pytest.mark.parametrize('link', ['external', 'soft', 'hard_alias', 'cycle'])
def test_links_aliases_and_cycles_rejected_without_following(make_matrix, link):
    path, _ = make_matrix()
    with h5py.File(path, 'r+') as file:
        if link == 'external':
            file['unexpected'] = h5py.ExternalLink('/home/operator/private.h5ad', '/X')
        elif link == 'soft':
            file['unexpected'] = h5py.SoftLink('/X')
        elif link == 'hard_alias':
            del file['var/start']
            file['var/start'] = file['var/end']
        else:
            file['layers/cycle'] = file['layers']
    with pytest.raises(MatrixPrivacyError):
        validate_matrix_h5ad(path, canonical.CONTRACT)


@pytest.mark.parametrize('storage', ['virtual', 'external', 'unreviewed_filter', 'private_fill'])
def test_indirect_or_unreviewed_dataset_storage_rejected(make_matrix, storage, tmp_path):
    path, _ = make_matrix()
    with h5py.File(path, 'r+') as file:
        del file['X/data']
        if storage == 'virtual':
            layout = h5py.VirtualLayout(shape=(2,), dtype='int64')
            layout[:] = h5py.VirtualSource('/home/operator/private.h5ad', '/data', shape=(2,))
            file['X'].create_virtual_dataset('data', layout)
        elif storage == 'external':
            file['X'].create_dataset('data', shape=(2,), dtype='int64',
                                     external=[(str(tmp_path / 'external.raw'), 0, h5py.h5f.UNLIMITED)])
        elif storage == 'unreviewed_filter':
            file['X'].create_dataset('data', data=[2, 3], dtype='int64', compression='lzf')
        else:
            file['X'].create_dataset('data', data=[2, 3], dtype='int64', fillvalue=17)
    with pytest.raises(MatrixPrivacyError):
        validate_matrix_h5ad(path, canonical.CONTRACT)


def test_oversized_strings_rejected_with_bounded_fixed_width_reads(make_matrix, monkeypatch):
    path, _ = make_matrix(ADOPTED[0])
    with h5py.File(path, 'r+') as file:
        file['obs/_index'][0] = 'x' * 100_000
    original = h5py.Dataset.astype
    observed = []
    def tracked(self, dtype):
        observed.append(dtype)
        return original(self, dtype)
    monkeypatch.setattr(h5py.Dataset, 'astype', tracked)
    with pytest.raises(MatrixPrivacyError):
        validate_matrix_h5ad(path, ADOPTED[0])
    assert observed and set(observed) == {'S65537'}


def test_privacy_inspection_never_reads_matrix_values(make_matrix, monkeypatch):
    path, manifest = make_matrix()
    original = h5py.Dataset.__getitem__
    def guarded(self, selection, *args, **kwargs):
        assert not self.name.startswith('/X/')
        return original(self, selection, *args, **kwargs)
    monkeypatch.setattr(h5py.Dataset, '__getitem__', guarded)
    validate_matrix_h5ad(path, canonical.CONTRACT, manifest)


@pytest.mark.parametrize('conflict', ['shape', 'metadata', 'contract'])
def test_conflicting_accepted_metadata_rejected(make_matrix, conflict):
    path, manifest = make_matrix()
    if conflict == 'shape':
        manifest['shape'] = [2, 2]
    elif conflict == 'metadata':
        manifest['assembly'] = 'mm10'
    else:
        manifest['contract_version'] = 'unsupported.v1'
    with pytest.raises(MatrixPrivacyError):
        validate_matrix_h5ad(path, canonical.CONTRACT, manifest)


def test_unrecognized_contract_and_non_hdf5_rejected(make_matrix, tmp_path):
    path, _ = make_matrix()
    with pytest.raises(MatrixPrivacyError):
        validate_matrix_h5ad(path, 'compact-neighbors.v1')
    path = tmp_path / 'invalid.h5ad'
    path.write_bytes(b'not HDF5')
    with pytest.raises(MatrixPrivacyError):
        validate_matrix_h5ad(path, canonical.CONTRACT)
