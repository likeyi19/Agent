import sys

import pytest

from agent.tools.data import _matrix_bedtools as b
from agent.tools.data.scatac_matrix_contract import ScATACMatrixError
from agent.tools._cancellation import cancellation_scope, ToolWorkCancelled


def test_real_boundaries_multiplicity_order_long_and_determinism(tmp_path):
    contigs=[('z',1_000_000_000),('a',100)]
    fragments=[('z',0,10,0),('z',9,11,1),('z',12,15,2),('z',5,25,3),('z',10,20,4),
        ('z',19,21,5),('z',20,25,6),('z',12,15,7),('z',999999999,1000000000,8),('a',1,3,9),
        ('z',0,1000000000,10)]
    features=[('z',10,20,40),('z',12,18,3),('z',17,22,7),('z',0,1000000000,8),('a',2,4,2)]
    expected={(r[3],c[3]) for r in fragments for c in features
              if r[0]==c[0] and r[1]<c[2] and c[1]<r[2]}
    one=b.qualification_intersections('/usr/bin/bedtools',contigs,fragments,features,scratch_parent=tmp_path)
    two=b.qualification_intersections('/usr/bin/bedtools',contigs,fragments[::-1],features[::-1],scratch_parent=tmp_path)
    assert set(one)==expected and len(one)==len(expected)
    assert one==two
    assert not list(tmp_path.iterdir())
    assert b.genome_order_bytes(contigs)==b'z\t1000000000\na\t100\n'


@pytest.mark.parametrize('fragments,features',[([],[]),([],[('a',1,2,0)]),([('a',1,2,0)],[]),
    ([('a',0,10,0)],[('a',10,20,0)])])
def test_real_empty_stream(tmp_path,fragments,features):
    assert b.qualification_intersections('/usr/bin/bedtools',[('a',30)],fragments,features,scratch_parent=tmp_path)==()
    assert not list(tmp_path.iterdir())


def test_duplicate_incidence_and_corrupt_returned_coordinates():
    line=b'a\t1\t3\t0\ta\t2\t4\t1\n'
    args=({0:('a',1,3)},{1:('a',2,4)})
    with pytest.raises(ScATACMatrixError,match='MATRIX_DUPLICATE_INCIDENCE'):
        list(b.validate_incidences([line,line],*args))
    with pytest.raises(ScATACMatrixError):
        list(b.validate_incidences([line.replace(b'\t4\t',b'\t5\t')],*args))


def test_input_and_runtime_limits(tmp_path):
    with pytest.raises(ScATACMatrixError):b.genome_order_bytes([('a',b.MAX_CONTIG_LENGTH+1)])
    with pytest.raises(ScATACMatrixError):b.genome_order_bytes([('a',5),('a',5)])
    with pytest.raises(ScATACMatrixError):b._projection([('a',1,2,0)]*(b.MAX_ROWS+1),[('a',5)])
    with pytest.raises(ScATACMatrixError):b._projection([('a',1,2,0)]*2,[('a',5)])
    with pytest.raises(ScATACMatrixError):b.qualify_runtime('/bin/false')


def test_admitted_projection_limit(tmp_path):
    records=[('a',1,3,i) for i in range(b.MAX_ROWS)]
    hits=b.qualification_intersections('/usr/bin/bedtools',[('a',10)],records,[('a',2,4,0)],scratch_parent=tmp_path)
    assert len(hits)==b.MAX_ROWS and len(set(hits))==b.MAX_ROWS
    assert not list(tmp_path.iterdir())


def test_subprocess_error_cleanup(tmp_path,monkeypatch):
    monkeypatch.setattr(b,'intersection_argv',lambda *a:('/bin/false',))
    with pytest.raises(ScATACMatrixError,match='MATRIX_INTERSECTION_FAILED'):
        b.qualification_intersections('/usr/bin/bedtools',[('a',10)],[],[],scratch_parent=tmp_path)
    assert not list(tmp_path.iterdir())


def test_output_resource_limit_cleanup(tmp_path,monkeypatch):
    monkeypatch.setattr(b,'MAX_OUTPUT_BYTES',1)
    with pytest.raises(ScATACMatrixError,match='MATRIX_BACKEND_LIMIT'):
        b.qualification_intersections('/usr/bin/bedtools',[('a',10)],[('a',1,3,0)],[('a',1,3,0)],scratch_parent=tmp_path)
    assert not list(tmp_path.iterdir())


def test_stderr_limit_cleanup(tmp_path,monkeypatch):
    monkeypatch.setattr(b,'MAX_STDERR_BYTES',1)
    monkeypatch.setattr(b,'intersection_argv',lambda *a:(sys.executable,'-c',
        'import sys; sys.stderr.write("diagnostic")'))
    with pytest.raises(ScATACMatrixError,match='MATRIX_BACKEND_LIMIT'):
        b.qualification_intersections('/usr/bin/bedtools',[('a',10)],[],[],scratch_parent=tmp_path)
    assert not list(tmp_path.iterdir())


def test_active_subprocess_cancellation_cleanup(tmp_path,monkeypatch):
    processes=[];popen=b.subprocess.Popen
    def launch(*a,**k):
        p=popen(*a,**k);processes.append(p);return p
    monkeypatch.setattr(b.subprocess,'Popen',launch)
    monkeypatch.setattr(b,'intersection_argv',lambda *a:(sys.executable,'-c','import time; time.sleep(30)'))
    def cancelled():
        return any(p.poll() is None for p in processes)
    # Pin checks use subprocess.run/Popen too; cancel only after scratch is made.
    with cancellation_scope(lambda: bool(list(tmp_path.iterdir())) and cancelled()):
        with pytest.raises(ToolWorkCancelled):
            b.qualification_intersections('/usr/bin/bedtools',[('a',10)],[],[],scratch_parent=tmp_path)
    assert processes and all(p.poll() is not None for p in processes)
    assert not list(tmp_path.iterdir())


def test_input_byte_limit(tmp_path,monkeypatch):
    monkeypatch.setattr(b,'MAX_INPUT_BYTES',5)
    with pytest.raises(ScATACMatrixError):b.genome_order_bytes([('longname',10)])
    with pytest.raises(ScATACMatrixError):b._projection([('a',1,2,0)],[('a',10)])


def test_timeout_cleanup(tmp_path,monkeypatch):
    monkeypatch.setattr(b,'MAX_SECONDS',0)
    with pytest.raises(ScATACMatrixError,match='MATRIX_BACKEND_TIMEOUT'):
        b.qualification_intersections('/usr/bin/bedtools',[('a',10)],[],[],scratch_parent=tmp_path)
    assert not list(tmp_path.iterdir())
