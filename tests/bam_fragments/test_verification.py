"""Manufacture self-consistent wrong artifacts; source recomputation must reject."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import pytest
from .conftest import pair, sha
from agent.tools.data import bam_fragments as production, bam_fragment_manifest as m
from agent.tools.data import _bam_fragment_io as io, scatac_fragments_v2 as v2
from agent.tools.data.bam_fragments_verifier import verify_bam_fragments
from agent.tools.data.scatac_fragments_v2_verifier import verify_fragments_v2, FragmentVerificationRuntime
from agent.tools.data._fragments_common import canonical


@pytest.mark.parametrize('fault',['missing_start_shift','missing_end_shift','double_shift','wrong_mate',
    'wrong_cigar_end','wrong_barcode','dropped_pair','added_pair','wrong_support','endpoint_collapse',
    'include_mapq_29','exclude_mapq_30','strict_mapq','include_mapq_255'])
def test_independent_verifier_rejects_forged_consistent_artifacts(bam_factory,tmp_path,fault):
    records=pair('a')+pair('a2')+pair('b',b={'pos':61,'tlen':0},a={'mpos':61,'tlen':0})
    records+=pair('filtered',a={'mapq':29},b={'mapq':29})
    args=bam_factory(records);stage=tmp_path/'stage';stage.mkdir();runtime=FragmentVerificationRuntime()
    value,_=production.prepare_in_stage(args,stage,runtime);entry=value['libraries'][0]
    path=stage/entry['bgzf']['path'];rows=[line.split('\t') for line in gzip.decompress(path.read_bytes()).decode().splitlines()]
    if fault=='missing_start_shift':rows[0][1]='10'
    elif fault=='missing_end_shift':rows[0][2]='80'
    elif fault=='double_shift':rows[0][1]='18';rows[0][2]='70'
    elif fault=='wrong_mate':rows[0][1]='64';rows[0][2]='85'
    elif fault=='wrong_cigar_end':rows[0][2]='74'
    elif fault=='wrong_barcode':rows[0][3]='WRONG-1'
    elif fault in ('dropped_pair','exclude_mapq_30','strict_mapq'):rows[0][4]='1'
    elif fault in ('added_pair','include_mapq_29','include_mapq_255'):rows[0][4]='3'
    elif fault=='wrong_support':rows[0][4]='1';rows[1][4]='2'
    elif fault=='endpoint_collapse':rows=rows[:1];rows[0][4]='3'
    rows.sort(key=lambda r:(int(r[1]),int(r[2]),r[3]))
    data=''.join('\t'.join(row)+'\n' for row in rows).encode()
    path.write_bytes(subprocess.run([runtime.bgzip,'-c','-l','6','-@','1'],input=data,capture_output=True,check=True).stdout)
    Path(str(path)+'.tbi').unlink()
    subprocess.run([runtime.tabix,'-p','bed',str(path)],capture_output=True,check=True)
    summary=dict(canonical_record_stream_sha256=hashlib.sha256(data).hexdigest(),n_fragment_records=len(rows),
        sum_support=sum(int(r[4]) for r in rows),n_distinct_barcodes=len({r[3] for r in rows}),max_support=max(int(r[4]) for r in rows))
    entry.update(summary)
    for key,p in (('bgzf',path),('tabix',Path(str(path)+'.tbi'))):entry[key]=io.resource(p)|{'path':str(p.relative_to(stage))}
    p=entry['provenance'];record=m.load_record(p['producer_record']['path'],p['producer_record']['sha256'])
    record['canonical']=summary
    record['qualification']['eligible_pairs']=summary['sum_support']
    record['qualification']['n_templates']=summary['sum_support']+sum(record['qualification']['exclusions'].values())
    record['qualification']['n_primary_pairs']=record['qualification']['n_templates']
    record['outputs']={k:{f:entry[k][f] for f in ('sha256','size_bytes')} for k in ('bgzf','tabix')}
    producer=stage/'production.json';producer.write_bytes(canonical(m.validate_record(record)))
    entry['provenance']=m.provenance(record,p['profile']['resource'],io.resource(producer))
    value['fragments_identity_sha256']=v2.fragments_identity(value)
    manifest=stage/'manifest.json';manifest.write_bytes(v2.canonical_fragments_manifest_v2_bytes(value))
    # Generic integrity is insufficient for producer science: this forged bundle
    # has valid hashes, canonical rows, support summaries and functional TBI.
    verify_fragments_v2(manifest,expected_sha256=sha(manifest),runtime=runtime)
    with pytest.raises(m.BamFragmentsError,match='CONSERVATION|TRANSFORMATION'):
        verify_bam_fragments(manifest,expected_sha256=sha(manifest),runtime=runtime)


def test_runtime_is_exact_and_tampering_fails(monkeypatch):
    current=io.runtime_identity()
    assert current['versions']=={'pysam':'0.24.1','htslib':'1.24','samtools':'1.24'}
    changed=json.loads(json.dumps(current));changed['files'][next(iter(changed['files']))]='0'*64
    monkeypatch.setattr(io,'_observed_runtime',lambda:changed)
    with pytest.raises(m.BamFragmentsError,match='RUNTIME'):io.runtime_identity()


@pytest.mark.parametrize('component', ['decoder', 'sort'])
def test_verification_qualifies_runtime_before_source_processing(bam_factory, tmp_path, monkeypatch, component):
    from dataclasses import replace
    from agent.tools.data import bam_fragments_verifier as verifier

    stage = tmp_path / 'stage'; stage.mkdir()
    runtime = FragmentVerificationRuntime()
    production.prepare_in_stage(bam_factory(), stage, runtime)
    manifest = stage / 'manifest.json'
    if component == 'decoder':
        changed = json.loads(json.dumps(io.runtime_identity()))
        changed['files'][next(iter(changed['files']))] = '0' * 64
        monkeypatch.setattr(io, '_observed_runtime', lambda: changed)
    else:
        # A real executable with the wrong bytes must never be invoked.
        marker = tmp_path / 'unqualified-sort-ran'
        executable = tmp_path / 'sort'
        executable.write_text('#!/bin/sh\ntouch "' + str(marker) + '"\nexit 1\n')
        executable.chmod(0o700)
        runtime = replace(runtime, sort=str(executable))

    def forbidden(*args, **kwargs):
        pytest.fail('Source processing preceded runtime qualification')
    monkeypatch.setattr(io, 'bind', forbidden)
    monkeypatch.setattr(verifier, 'reconstruct', forbidden)
    with pytest.raises(ValueError, match='RUNTIME_MISMATCH|TOOLCHAIN_MISMATCH'):
        verify_bam_fragments(manifest, expected_sha256=sha(manifest), runtime=runtime)
    if component == 'sort':
        assert not marker.exists()


def test_uint128_overflow_schema(bam_factory,tmp_path):
    stage=tmp_path/'stage';stage.mkdir();value,_=production.prepare_in_stage(bam_factory(),stage,FragmentVerificationRuntime())
    p=value['libraries'][0]['provenance']['producer_record'];record=m.load_record(p['path'],p['sha256'])
    record['canonical']['sum_support']=2**128
    with pytest.raises(ValueError):m.validate_record(record)


def test_source_mutation_during_decode_fails_and_cleans(bam_factory,monkeypatch):
    from agent.tools.data import scatac_bam_fragments as public
    args=bam_factory();original=production.generate
    def mutate(*a,**k):
        result=original(*a,**k);path=Path(args['source_path']);path.write_bytes(path.read_bytes()+b'changed');return result
    monkeypatch.setattr(production,'generate',mutate)
    with pytest.raises(ValueError):public.prepare_scATAC_bam_fragments(**args)
    assert not list(Path(args['output_dir']).glob('.bam-attempt-*'))
    assert not list(Path(args['output_dir']).glob('bam-fragments-*'))
