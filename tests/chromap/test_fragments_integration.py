"""Guarded true patched-Chromap/BGZF/TBI synthetic execution."""
import gzip
from pathlib import Path
import subprocess
import pytest

from agent.tools.data import _chromap as c
from agent.tools.data.fastq_fragments import prepare_fastq_fragments
from agent.tools.data._fragments_common import FragmentsRuntime, FragmentsError
from agent.tools.data.scatac_fragments_verifier import verify_fragments
from fragments_helpers import BC, inputs_for


@pytest.mark.parametrize('layout', [c.FastqLayout.A, c.FastqLayout.B])
def test_weighted_layout_e2e(tiny, reads, executables, layout):
    group, white = reads([(100+i*1000, 200+i*1000, BC, n) for i, n in enumerate([1,2,255,256,300])]
        + [(6100,6200,'CCGTACGTACGTACGT',1)], layout=layout)
    # Exercise actual gzip role streams on Layout B using M10-supported filenames.
    if layout is c.FastqLayout.B:
        files = []
        for role, name in group.files:
            path = Path(name); zipped = Path(name + '.gz')
            zipped.write_bytes(gzip.compress(path.read_bytes(), mtime=0)); path.unlink()
            files.append((role, str(zipped)))
        from dataclasses import replace
        group = replace(group, files=tuple(files))
    runtime = FragmentsRuntime(executables['candidate']['path'])
    inputs = inputs_for(tiny, [group], white, executable=runtime.chromap)
    result = prepare_fastq_fragments(inputs=inputs, output_dir=tiny['root']/'published', runtime=runtime)
    value = verify_fragments(result['manifest_path'], expected_sha256=result['manifest_sha256'], runtime=runtime)
    entry = value['libraries'][0]
    path = Path(result['manifest_path']).parent / entry['bgzf']['path']
    rows = [line.split('\t') for line in gzip.decompress(path.read_bytes()).decode().splitlines()]
    assert [int(r[4]) for r in rows] == [1,2,255,256,300,1]
    assert all(r[3] == BC for r in rows)
    assert result['total_support'] == 815 and result['n_fragment_records'] == 6
    assert entry['n_distinct_barcodes'] == 1
    assert rows[0][:3] == ['chrTiny','104','195']
    # Separately rebuild the tiny TBI and compare full-contig functional query.
    rebuilt = tiny['root']/'rebuilt.gz'; rebuilt.write_bytes(path.read_bytes())
    subprocess.run([runtime.tabix,'-p','bed',str(rebuilt)],check=True)
    assert subprocess.check_output([runtime.tabix,str(rebuilt),'chrTiny']) == gzip.decompress(path.read_bytes())
    before = path.read_bytes()
    with pytest.raises(FragmentsError, match='FRAGMENTS_ARTIFACT_CONFLICT'):
        prepare_fastq_fragments(inputs=inputs, output_dir=path.parent.parent, runtime=runtime)
    assert path.read_bytes() == before
    assert set(p.name for p in path.parent.iterdir()) == {'fragments.tsv.gz','fragments.tsv.gz.tbi'}
    # Corrupt TBI and update its declared physical hash: usability still fails.
    from agent.tools.data import scatac_fragments_manifest as fm
    tbi=Path(str(path)+'.tbi');tbi.write_bytes(b'not a TBI index')
    entry['tabix']['sha256']=c.sha256(tbi);entry['tabix']['size_bytes']=tbi.stat().st_size
    Path(result['manifest_path']).write_bytes(fm.canonical_fragments_manifest_bytes(value))
    with pytest.raises(FragmentsError,match='FRAGMENTS_INDEX_MISMATCH'):
        verify_fragments(result['manifest_path'],runtime=runtime)


@pytest.mark.parametrize('shared', [True, False])
def test_library_execution_scope(tiny, reads, executables, shared):
    a, white = reads([(100,200,BC,200)], name='laneA')
    b, _ = reads([(100,200,BC,100)], name='laneB')
    runtime = FragmentsRuntime(executables['candidate']['path'])
    inputs = inputs_for(tiny, [a,b], white, shared=shared, executable=runtime.chromap)
    result = prepare_fastq_fragments(inputs=inputs, output_dir=tiny['root']/'published', runtime=runtime)
    value = verify_fragments(result['manifest_path'], runtime=runtime)
    assert result['total_support'] == 300
    assert len(value['libraries']) == (1 if shared else 2)
    assert sorted(e['sum_support'] for e in value['libraries']) == ([300] if shared else [100,200])
    assert len({(e['namespace'],BC) for e in value['libraries']}) == len(value['libraries'])


def test_real_sort_fai_order_and_bgzf_repeatability(tmp_path,executables):
    from agent.tools.data._fragments_canonical import canonicalize
    from agent.tools.data._fragments_common import verify_packaging
    from agent.tools.data.scatac_fragments_verifier import verify_stream
    runtime=FragmentsRuntime(executables['candidate']['path']);verify_packaging(runtime)
    raw=tmp_path/'raw.bed'
    raw.write_text(f'chrA\t1\t2\t{BC}\t256\nchrZ\t10\t20\t{BC}\t300\n'
        f'chrZ\t2\t3\t{BC}\t255\n')
    contigs=[('chrZ',100),('chrA',100)]
    canonical,summary=canonicalize(raw,directory=tmp_path,contigs=contigs,whitelist={BC},barcode_length=16,runtime=runtime)
    assert canonical.read_text().splitlines()[0]==f'chrZ\t2\t3\t{BC}\t255'
    outputs=[]
    for i in range(2):
        path=tmp_path/f'repeat{i}.gz'
        with path.open('wb') as out:
            subprocess.run([runtime.bgzip,'-c','-l','6','-@','1',str(canonical)],stdout=out,check=True)
        subprocess.run([runtime.tabix,'-p','bed',str(path)],check=True)
        verify_stream(path,dict(summary,barcode_length=16),contigs,{BC})
        assert subprocess.check_output([runtime.tabix,'-l',str(path)])==b'chrZ\nchrA\n'
        outputs.append(path.read_bytes())
    assert outputs[0]==outputs[1]
