"""Explicitly gated executable tests; ordinary pytest never builds/downloads."""
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import subprocess
import pytest
from agent.tools.data import _chromap as c
from agent.tools.data import chromap_reference_index as idx
BC = 'ACGTACGTACGTACGT'


def success(result):
    r, rows, raw = result
    assert r.returncode == 0, r.stderr.decode()
    return rows, raw


def test_stock_saturation_and_exact_candidate(reads, run_backend):
    counts = [1,2,254,255,256,300,65536]
    group, white = reads([(100+i*1000,200+i*1000,BC,n) for i,n in enumerate(counts)])
    stock, _ = success(run_backend('stock',group,white,name='stock'))
    fixed, _ = success(run_backend('candidate',group,white,name='fixed'))
    stock.sort(key=lambda r:int(r[1])); fixed.sort(key=lambda r:int(r[1]))
    assert [int(r[4]) for r in stock] == [min(n,255) for n in counts]
    assert [int(r[4]) for r in fixed] == counts
    assert [r[:4] for r in stock] == [r[:4] for r in fixed]
    assert stock[:3] == fixed[:3]


def test_barcode_cases_equivalent(reads, run_backend):
    # Ambiguous raw A... has two neighbors, with 20:1 abundance; tie uses C... family.
    a='A'*16; a1='C'+a[1:]; a2='G'+a[1:]
    t='T'*16; t1='A'+t[1:]; t2='C'+t[1:]
    zero='GGGGCCCCGGGGCCCC'; single='GGGGCCCCGGGGCCCA'
    cases=[(100,200,a1,20),(1100,1200,a2,1),(2100,2200,a,1),
        (3100,3200,t1,5),(4100,4200,t2,5),(5100,5200,t,1),
        (6100,6200,single,1),(7100,7200,'N'+a1[1:],1),
        (8100,8200,'NN'+a1[2:],1),(9100,9200,'ACGT'*4,1)]
    group, white = reads(cases,whitelist=(a1,a2,t1,t2,zero))
    s, sr = success(run_backend('stock',group,white,name='bc_stock'))
    p, pr = success(run_backend('candidate',group,white,name='bc_fixed'))
    assert s == p and sr == pr
    by_start = {int(r[1]):r[3] for r in p}
    assert by_start == {104:a1,1104:a2,2104:a1,3104:t1,4104:t2,6104:zero,7104:a1}

@pytest.mark.parametrize('length',[31,32])
def test_barcode_lengths(reads,run_backend,length):
    barcode=('ACGT'*8)[:length]
    group,white=reads([(100,200,barcode,2)],whitelist=(barcode,))
    for mode in ('stock','candidate'):
        rows,_=success(run_backend(mode,group,white,name=mode))
        assert rows == [('chrTiny','104','195',barcode,'2')]


def test_33_backend_rejection(reads,run_backend,tiny):
    group,white=reads([(100,200,BC,2)])
    bcpath=dict(group.files)[c.ReadRole.R2]
    Path(bcpath).write_text('@one\n'+'A'*33+'\n+\n'+'I'*33+'\n')
    longwhite=tiny['root']/'long.txt';longwhite.write_text('A'*33+'\n')
    def mutate(argv): argv[argv.index('--barcode-whitelist')+1]=str(longwhite)
    for mode in ('stock','candidate'):
        r,_,_=run_backend(mode,group,white,name=mode,mutate=mutate)
        assert r.returncode != 0 and b'greater than 32' in r.stderr


def test_layout_tn5_and_repeatability(reads,run_backend):
    records=[(100,200,BC,2),(0,100,BC,1),(11900,12000,BC,1),
        (1000,1030,BC,1),(2000,2070,BC,1),(3000,3100,BC,1,True)]
    a,wa=reads(records,name='A'); b,wb=reads(records,name='B',layout=c.FastqLayout.B)
    expected=sorted([('chrTiny',str(s+4),str(e-5),BC,str(n)) for s,e,_,n,*_ in records if s != 0 and e != 12000])
    # Stock IsValidCandidate requires an error-threshold flank at each contig edge.
    # Boundary pairs are rejected, not clipped; the width patch must preserve this.
    baseline=None
    for mode in ('stock','candidate'):
        for group,white,label in [(a,wa,'A'),(b,wb,'B')]:
            for repeat,threads in enumerate([1,1,1,4]):
                rows,raw=success(run_backend(mode,group,white,name=f'{mode}{label}{repeat}',threads=threads))
                assert rows == expected
                baseline=raw if baseline is None else baseline
                assert raw == baseline


def test_cross_lane_cell_isolation(reads,run_backend):
    other='TGCATGCATGCATGCA'
    a,w=reads([(100,200,BC,200),(100,200,other,2)],name='laneA',whitelist=(BC,other))
    b,_=reads([(100,200,BC,100)],name='laneB',whitelist=(BC,other))
    for mode in ('stock','candidate'):
        rows,_=success(run_backend(mode,a,w,name=mode,groups=[b,a]))
        assert rows==sorted([('chrTiny','104','195',BC,'255' if mode=='stock' else '300'),('chrTiny','104','195',other,'2')])
        def reverse(argv):
            for flag in ('-1','-2','-b'):
                i=argv.index(flag)+1;argv[i]=','.join(argv[i].split(',')[::-1])
        reversed_rows,_=success(run_backend(mode,a,w,name=mode+'_reversed',groups=[a,b],mutate=reverse))
        assert rows==reversed_rows
    # Separate namespaces imply separate executions even with identical barcode tokens.
    ra,_=success(run_backend('candidate',a,w,name='namespace_A'))
    rb,_=success(run_backend('candidate',b,w,name='namespace_B'))
    assert ('namespace_A',BC)!=('namespace_B',BC)
    assert next(r[4] for r in ra if r[3]==BC)=='200'
    assert rb[0][4]=='100'

@pytest.mark.parametrize('case',['count','truncated','names'])
def test_backend_synchronization_boundary(reads,run_backend,case):
    group,w=reads([(100,200,BC,2)])
    p=Path(dict(group.files)[c.ReadRole.R2]); lines=p.read_text().splitlines(True)
    if case=='count': p.write_text(''.join(lines[:4]))
    elif case=='truncated': p.write_text(''.join(lines[:-1]))
    else: p.write_text(''.join('@different\n' if line.startswith('@') else line for line in lines))
    for mode in ('stock','candidate'):
        result=run_backend(mode,group,w,name=mode)
        if case=='names':
            rows,_=success(result); assert rows[0][4]=='2'
        else: assert result[0].returncode != 0


def test_tiny_index_build_reuse_and_repeatability(tiny,reads,run_backend,executables):
    exe=executables['candidate']['path']
    backend=c.identify_backend(exe,expected_sha256=executables['candidate']['sha256'])
    pointer=tiny['pointer']; args=dict(reference_manifest_path=pointer['manifest_path'],
        expected_reference_sha256=pointer['manifest_sha256'],executable=exe,backend=backend)
    with ThreadPoolExecutor(max_workers=2) as pool:
        concurrent=list(pool.map(lambda _:idx.prepare_chromap_reference_index(
            **args,output_dir=tiny['root']/'one'),range(2)))
    one=concurrent[0]
    assert concurrent[1]==one
    assert idx.prepare_chromap_reference_index(**args,output_dir=tiny['root']/'one')==one
    two=idx.prepare_chromap_reference_index(**args,output_dir=tiny['root']/'two')
    group,w=reads([(100,200,BC,300)])
    r1,_=success(run_backend('candidate',group,w,name='one_map',index=tiny['root']/'one/reference.chromap'))
    r2,_=success(run_backend('candidate',group,w,name='two_map',index=tiny['root']/'two/reference.chromap'))
    assert r1==r2
    print('INDEX_REBUILDS',one.index_sha256,two.index_sha256,one.index_size_bytes,two.index_size_bytes)


def test_wide_storage_spill_representative_and_overflow(tiny,executables):
    source=Path(executables['candidate']['path']).parent
    probe=tiny['root']/'support-probe'
    cpp=Path(__file__).with_name('support_probe.cc')
    command=['g++','-std=c++11','-O2','-fopenmp','-I',str(source/'src'),str(cpp),
        str(source/'objs/mapping_writer.o'),str(source/'objs/sequence_batch.o'),'-lz','-o',str(probe)]
    built=subprocess.run(command,capture_output=True)
    assert built.returncode==0,built.stderr.decode()
    for mode in ('wide','memory','spill','overflow','summary-overflow'):
        out=tiny['root']/(mode+'.bed')
        r=subprocess.run([str(probe),mode,str(tiny['fa']),str(out)],capture_output=True)
        if 'overflow' in mode:
            assert r.returncode!=0 and b'OVERFLOW' in r.stderr
        else:
            assert r.returncode==0,r.stderr.decode()
            rows=[line.split() for line in out.read_text().splitlines()]
            if mode=='wide':assert [int(r[4]) for r in rows]==[255,256,65536,2**32+7,2**64-1]
            else:assert rows==[['chrTiny','104','195','A'*16,'300']]


def test_in_memory_stock_candidate_parity(reads,run_backend):
    group,w=reads([(100,200,BC,2),(1100,1200,BC,300)])
    def memory(argv):
        i=argv.index('--preset');del argv[i:i+2]
        argv.remove('--low-mem')
    s,_=success(run_backend('stock',group,w,name='stock_memory',mutate=memory))
    p,_=success(run_backend('candidate',group,w,name='candidate_memory',mutate=memory))
    assert s[0]==p[0] and s[1][:4]==p[1][:4]
    assert s[1][4]=='255' and p[1][4]=='300'


def test_pinned_patch_application(tmp_path,executables):
    root=Path(executables['candidate']['path']).parent.parent
    patch=Path(__file__).resolve().parents[2]/'third_party/patches/chromap/support-v1.patch'
    fresh=c.prepare_support_source(root/'upstream.tar',patch,tmp_path/'reviewed-source')
    candidate=Path(executables['candidate']['path']).parent
    for source in (fresh/'src').iterdir():
        assert source.read_bytes()==(candidate/'src'/source.name).read_bytes()
    assert not (fresh/'chromap').exists()
