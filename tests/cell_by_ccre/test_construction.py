import json
from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from agent.tools.data.scatac_cell_by_ccre import build_cell_by_ccre, MatrixLimits
from agent.tools.data.cell_by_ccre_verifier import verify_cell_by_ccre, IntervalIndex
from agent.tools.data import scatac_matrix_contract as m


def test_complete_publication(fixture_factory):
    args = fixture_factory()
    result = build_cell_by_ccre(**args)
    path = Path(result['manifest_path']); value = json.loads(path.read_bytes())
    x = ad.read_h5ad(path.parent/'matrix.h5ad')
    assert x.X.dtype == np.dtype('int64')
    assert x.X.toarray().tolist() == [[1,2,3,2,0],[1,1,2,1,0],[0,0,0,0,0]]
    assert x.obs['barcode_identifier'].tolist() == ['A','B','Z']
    assert x.var_names.tolist() == ['chr2:3000-3010','chr2:2950-3051','chr2:0-6000','chr2:2999-3001','chr1:1-2']
    assert value['diagnostic'] == m.overlap_diagnostic(6,5)
    assert result['logical_matrix_sha256'] == m.csr_identity(x.X)['logical_matrix_sha256']
    assert x.raw is None and not x.layers and set(path.parent.iterdir()) == {path,path.parent/'matrix.h5ad'}
    checked = verify_cell_by_ccre(path,expected_sha256=result['manifest_sha256'],bedtools_path=args['bedtools_path'])
    assert checked['logical_matrix_sha256'] == result['logical_matrix_sha256']
    assert not list(path.parent.parent.glob('.matrix-*'))
    with pytest.raises(m.ScATACMatrixError,match='MATRIX_OUTPUT_CONFLICT'): build_cell_by_ccre(**args)


def test_empty_does_not_invoke_intersection(fixture_factory,monkeypatch):
    from agent.tools.data import _cell_by_ccre_production as production
    monkeypatch.setattr(production,'intersect',lambda *a:pytest.fail('Empty selection called BEDTools'))
    args=fixture_factory(empty=True); result=build_cell_by_ccre(**args)
    x=ad.read_h5ad(Path(result['manifest_path']).parent/'matrix.h5ad')
    assert x.shape == (0,5) and x.X.indptr.tolist()==[0] and x.X.nnz==0
    assert result['diagnostic']==m.overlap_diagnostic(0,0)


def test_species_share_identical_science(fixture_factory):
    results=[build_cell_by_ccre(**fixture_factory(species=s)) for s in ('human','mouse')]
    assert results[0]['logical_matrix_sha256']==results[1]['logical_matrix_sha256']
    assert results[0]['diagnostic']==results[1]['diagnostic']


def test_support_invariance(fixture_factory):
    rows=[('chr2',1000,1001,'A',1),('chr2',3000,3001,'A',2)]
    results=[build_cell_by_ccre(**fixture_factory(rows=rows)),
             build_cell_by_ccre(**fixture_factory(rows=[(*r[:4],999999) for r in rows]))]
    assert results[0]['logical_matrix_sha256']==results[1]['logical_matrix_sha256']


def test_independent_index_nested_long_and_touching():
    intervals=[(0,10000,3),(1,2,1),(100,200,0),(101,102,2)]
    index=IntervalIndex(iter(intervals),len(intervals))
    assert set(index.query(102,103))=={0,3}
    assert set(index.query(101,102))=={0,2,3}
    assert set(index.query(10000,10001))==set()
    rng=np.random.default_rng(2)
    raw=sorted((int(a),int(a+b),i) for i,(a,b) in enumerate(rng.integers(1,10000,size=(1000,2))))
    index=IntervalIndex(iter(raw),len(raw))
    for start in range(0,20000,211):
        end=start+251
        assert set(index.query(start,end))=={c for a,b,c in raw if a<end and b>start}


def test_budget_failure_cleanup(fixture_factory):
    args=fixture_factory()
    with pytest.raises(m.ScATACMatrixError): build_cell_by_ccre(**args,limits=MatrixLimits(max_scratch_bytes=4096))
    assert not Path(args['output_dir']).exists()
    assert not list(Path(args['output_dir']).parent.glob('.matrix-*'))


def test_distinct_strand_records_retain_multiplicity(fixture_factory):
    rows=[('chr2',1000,1001,'A',1,'+'),('chr2',3000,3001,'A',200,'+'),('chr2',3000,3001,'A',5,'-')]
    result=build_cell_by_ccre(**fixture_factory(rows=rows))
    x=ad.read_h5ad(Path(result['manifest_path']).parent/'matrix.h5ad')
    assert x.X.toarray().tolist()==[[2,2,3,2,0]]


def test_exact_lineage_paths_and_hashes(fixture_factory):
    args=fixture_factory(); other=fixture_factory()
    for kind in ('fragments','reference'):
        wrong=args | {f'{kind}_manifest_path':other[f'{kind}_manifest_path'],f'{kind}_manifest_sha256':other[f'{kind}_manifest_sha256']}
        with pytest.raises(m.ScATACMatrixError,match='MATRIX_LINEAGE_INVALID'):build_cell_by_ccre(**wrong)


def test_m8_logical_digest_cross_check(fixture_factory):
    from agent.tools.analysis.replicate_pseudobulk import _matrix_digest_and_nnz
    result=build_cell_by_ccre(**fixture_factory())
    x=ad.read_h5ad(Path(result['manifest_path']).parent/'matrix.h5ad')
    assert result['logical_matrix_sha256']==_matrix_digest_and_nnz(x.X,*x.shape,'fragment_counts')[0]


def test_all_selected_rows_can_be_zero(fixture_factory):
    result=build_cell_by_ccre(**fixture_factory(features=[('chr2',1,2)]))
    x=ad.read_h5ad(Path(result['manifest_path']).parent/'matrix.h5ad')
    assert x.shape==(3,1) and x.X.indptr.tolist()==[0,0,0,0] and x.X.nnz==0
    assert result['zero_row_count']==3 and result['diagnostic']==m.overlap_diagnostic(6,0)
