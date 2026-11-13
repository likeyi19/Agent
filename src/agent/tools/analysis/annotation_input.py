"""Bounded source declarations for the existing direct peak annotation owner.

Observations are source-bound admission evidence, not proof of counting history
or biological validity. Native consumers retain scientific qualification.
"""
import anndata as ad
from pathlib import Path
import pandas as pd
from agent.tools import _file_identity as science
from . import peak_h5ad


def observe(path, sha256):
    from agent.tools._native_process import call
    return call('annotation_observe', dict(path=str(path), sha256=sha256), lambda: _observe(path, sha256))


def _observe(path, sha256):
    path = science.checked_file(path, sha256)
    peak_h5ad.inspect(path, max_logical_bytes=None)
    a = ad.read_h5ad(path, backed='r')
    try:
        declarations = {}
        for name in ('species', 'assembly', 'measurement', 'context', 'count_slot'):
            if name in a.uns:
                science.identifier(a.uns[name])
                declarations[name] = a.uns[name]
        if 'measurement' not in declarations and 'matrix_semantics' in a.uns:
            science.identifier(a.uns['matrix_semantics'])
            declarations['measurement'] = a.uns['matrix_semantics']
        columns = [str(name) for name in a.obs.columns
                   if not a.obs[name].isna().any() and 1 < a.obs[name].nunique() < len(a.obs)
                   and (isinstance(a.obs[name].dtype, pd.CategoricalDtype)
                        or a.obs[name].dtype.kind in 'Obiu')]
        return dict(interface_version=5, source_sha256=sha256, declarations=declarations,
                    grouping_columns=columns)
    finally:
        a.file.close()
