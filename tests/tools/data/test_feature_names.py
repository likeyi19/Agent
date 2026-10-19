"""Vocabulary observation consumes only the exact stored feature index."""
import hashlib

import h5py
import numpy as np
import pytest

from agent.tools.data.scatac import read_feature_names
from agent.tools.models.epizoo_inputs import observe_source


def fixture(path, version='0.2.0', names=('chr1:1-2', 'chr2:3-4')):
    with h5py.File(path, 'w') as f:
        var = f.create_group('var')
        var.attrs.update({'encoding-type': 'dataframe', 'encoding-version': version, '_index': 'peaks'})
        index = var.create_dataset('peaks', data=np.array(names, dtype=object), dtype=h5py.string_dtype())
        if version == '0.2.0':
            index.attrs.update({'encoding-type': 'string-array', 'encoding-version': version})
        # Unconsumed annotations/X must not impose an annotation-owner format gate.
        f['X'] = h5py.ExternalLink('/nonexistent', 'matrix')
        f.create_group('uns').attrs['encoding-type'] = 'unrelated-extension'
    return path


@pytest.mark.parametrize('version', ['0.1.0', '0.2.0'])
def test_exact_order_and_identity_without_loading_matrix(tmp_path, version):
    path = fixture(tmp_path / 'source.h5ad', version)
    assert read_feature_names(path) == ['chr1:1-2', 'chr2:3-4']
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert observe_source(path, digest)['source_sha256'] == digest
    with pytest.raises(ValueError, match='identity'):
        observe_source(path, '0' * 64)


@pytest.mark.parametrize('kind', ['external-axis', 'external-index', 'numeric', 'empty', 'duplicate', 'long'])
def test_consumed_index_rejects_indirection_or_invalid_vocabulary(tmp_path, kind):
    path = fixture(tmp_path / 'source.h5ad')
    with h5py.File(path, 'a') as f:
        if kind == 'external-axis':
            del f['var']; f['var'] = h5py.ExternalLink('/nonexistent', 'var')
        else:
            del f['var/peaks']
            if kind == 'external-index':
                f['var/peaks'] = h5py.ExternalLink('/nonexistent', 'index')
            elif kind == 'numeric':
                f['var/peaks'] = [1, 2]
            else:
                names = [] if kind == 'empty' else ['same', 'same'] if kind == 'duplicate' else ['x' * 65_537]
                f['var'].create_dataset('peaks', data=np.array(names, dtype=object), dtype=h5py.string_dtype())
    with pytest.raises(ValueError):
        read_feature_names(path)
