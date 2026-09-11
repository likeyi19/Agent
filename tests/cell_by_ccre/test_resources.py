import json
import os
from pathlib import Path
import resource
import sqlite3

import pytest

from agent.tools._cancellation import cancellation_scope, ToolWorkCancelled
from agent.tools.data.scatac_cell_by_ccre import build_cell_by_ccre, MatrixLimits
from agent.tools.data import _cell_by_ccre_production as production, _cell_by_ccre_io as io
from agent.tools.data import scatac_matrix_contract as m


def assert_clean(args):
    assert not Path(args['output_dir']).exists()
    assert not list(Path(args['output_dir']).parent.glob('.matrix-*'))
    assert not list(Path(args['output_dir']).parent.glob('.*.matrix-lock'))


def test_subprocess_failure_cleanup(fixture_factory,monkeypatch):
    args=fixture_factory()
    monkeypatch.setattr(production.bed,'intersection_argv',lambda *a:('/usr/bin/bedtools','intersect','--not-an-option'))
    with pytest.raises(m.ScATACMatrixError,match='MATRIX_INTERSECTION_FAILED'):build_cell_by_ccre(**args)
    assert_clean(args)


def test_cancellation_during_incidence_cleans_process_and_scratch(fixture_factory,monkeypatch):
    args=fixture_factory();original=production.incidence;cancelled=[False]
    def incident(*a):
        original(*a);cancelled[0]=True
    monkeypatch.setattr(production,'incidence',incident)
    with cancellation_scope(lambda:cancelled[0]):
        with pytest.raises(ToolWorkCancelled):build_cell_by_ccre(**args)
    assert_clean(args)


def test_repeated_backend_incidence_rejected(fixture_factory,monkeypatch):
    args=fixture_factory();original=production.incidence
    def twice(db,line):original(db,line);original(db,line)
    monkeypatch.setattr(production,'incidence',twice)
    with pytest.raises(m.ScATACMatrixError,match='MATRIX_DUPLICATE_INCIDENCE'):build_cell_by_ccre(**args)
    assert_clean(args)


def test_production_overflow_check(tmp_path):
    with io.database(tmp_path/'scratch.sqlite',io.Budget(tmp_path,MatrixLimits())) as db:
        db.execute('CREATE TABLE records(id INTEGER,chrom TEXT,start INTEGER,end INTEGER,row INTEGER)')
        db.execute('INSERT INTO records VALUES(0,"chr2",1,3,0)')
        db.execute('INSERT INTO features VALUES(0,0,"chr2",2,4,"chr2:2-4")')
        db.execute('CREATE TABLE incidences(fid INTEGER,col INTEGER,PRIMARY KEY(fid,col))')
        db.execute('CREATE TABLE counts(row INTEGER,col INTEGER,value INTEGER,PRIMARY KEY(row,col))')
        db.execute('INSERT INTO counts VALUES(0,0,?)',(m.INT64_MAX,))
        with pytest.raises(m.ScATACMatrixError,match='MATRIX_INTEGER_INVALID'):
            production.incidence(db,b'chr2\t1\t3\t0\tchr2\t2\t4\t0\n')


def test_selected_identity_absent_is_error(fixture_factory,monkeypatch):
    from agent.tools.data.scatac_fragment_reader import VerifiedFragments
    args=fixture_factory();original=VerifiedFragments.iter_fragments
    def omit(self,namespace):
        for record in original(self,namespace):
            if record.barcode_identifier!='Z':yield record
    project=production.project
    def omit_during_projection(*a):
        with monkeypatch.context() as patch:
            patch.setattr(VerifiedFragments,'iter_fragments',omit)
            return project(*a)
    monkeypatch.setattr(production,'project',omit_during_projection)
    with pytest.raises(m.ScATACMatrixError,match='MATRIX_SELECTED_CELL_ABSENT'):build_cell_by_ccre(**args)
    assert_clean(args)


@pytest.mark.skipif(os.environ.get('RUN_SCATAC_MATRIX_STRESS')!='1',reason='Explicit bounded synthetic matrix stress opt-in')
def test_larger_synthetic_stress(fixture_factory):
    features=[('chr2',3000+i%256,3001+i%256+i//256) for i in range(8192)]
    rows=[('chr2',1000,1001,f'cell{i:03}',1) for i in range(64)]
    rows += [('chr2',3000+j,3001+j,f'cell{i:03}',1+(i+j)%37) for i in range(64) for j in range(64)]
    args=fixture_factory(rows=rows,features=features)
    result=build_cell_by_ccre(**args,limits=MatrixLimits(max_scratch_bytes=512*1024**2))
    assert result['nnz']>90000
    assert result['diagnostic']==m.overlap_diagnostic(4160,4096)
    evidence={**result,'features':8192,'selected_cells':64,'canonical_records':4160,
              'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    Path('/tmp/agent-m115b-stress.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence,sort_keys=True))


def test_independent_gate_rejects_incomplete_production(fixture_factory,monkeypatch):
    args=fixture_factory();original=production.incidence;calls=[0]
    def omit_one(db,line):
        calls[0]+=1
        if calls[0]!=1:original(db,line)
    monkeypatch.setattr(production,'incidence',omit_one)
    with pytest.raises(m.ScATACMatrixError,match='MATRIX_SCIENCE_MISMATCH'):build_cell_by_ccre(**args)
    assert_clean(args)


def test_verifier_cancellation_cleans_only_private_scratch(fixture_factory,tmp_path,monkeypatch):
    from agent.tools.data.cell_by_ccre_verifier import verify_cell_by_ccre, IntervalIndex
    args=fixture_factory();result=build_cell_by_ccre(**args)
    scratch=tmp_path/'verifier-scratch';scratch.mkdir()
    def cancel(*a):raise ToolWorkCancelled()
    monkeypatch.setattr(IntervalIndex,'query',cancel)
    with pytest.raises(ToolWorkCancelled):
        verify_cell_by_ccre(result['manifest_path'],expected_sha256=result['manifest_sha256'],
                           bedtools_path='/usr/bin/bedtools',scratch_parent=scratch)
    assert not list(scratch.iterdir())
    assert Path(result['manifest_path']).is_file()


def test_qc_change_after_verification_blocks_publication(fixture_factory,monkeypatch):
    from agent.tools.data import scatac_cell_by_ccre as api
    args=fixture_factory();original=api.verify_cell_by_ccre
    selection=json.loads(Path(args['selection_manifest_path']).read_bytes())
    qp=Path(selection['arguments']['barcode_qc_manifest_path']);q=json.loads(qp.read_bytes())
    table=qp.parent/q['table']['path']
    def change_after_verification(*a,**kw):
        result=original(*a,**kw)
        with table.open('ab') as f:f.write(b'changed')
        return result
    monkeypatch.setattr(api,'verify_cell_by_ccre',change_after_verification)
    with pytest.raises(ValueError,match='FRAGMENTS_V2_SOURCE_CHANGED'):build_cell_by_ccre(**args)
    assert_clean(args)
