import gzip
import json
from pathlib import Path
import pytest
from agent.tools.data import scatac_bam_fragments as public, bam_fragment_manifest as m
from .conftest import pair
from agent.tools.data import bam_fragments as production, _bam_fragment_io as io
from agent.tools.data.bam_fragments_verifier import reconstruct
from agent.tools.data.scatac_fragments_v2_verifier import FragmentVerificationRuntime


def output(args):
    result=public.prepare_scATAC_bam_fragments(**args)
    path=Path(result['manifest_path']);manifest=json.loads(path.read_bytes());entry=manifest['libraries'][0]
    rows=gzip.decompress((path.parent/entry['bgzf']['path']).read_bytes()).decode().splitlines()
    record=m.load_record(entry['provenance']['producer_record']['path'],entry['provenance']['producer_record']['sha256'])
    return result,rows,record


def test_hand_derived_pair(bam_factory):
    result,rows,record=output(bam_factory())
    assert rows==['chr2\t14\t75\taCgT-1\t1']
    assert result['eligible_pairs']==1 and result['strand_mode']=='absent'
    assert record['qualification']['n_records']==2
    manifest=json.loads(Path(result['manifest_path']).read_bytes())
    assert manifest['libraries'][0]['provenance']['source_selection']=='unspecified'
    from agent.tools.data.scatac_fragment_reader import open_verified_fragments
    view=open_verified_fragments(result['manifest_path'],expected_sha256=result['manifest_sha256'],runtime=FragmentVerificationRuntime())
    assert [r.strand for r in view.iter_fragments(result['namespace'])]==[None]


@pytest.mark.parametrize('count',[2,255,256,301])
def test_exact_support_and_duplicate_flags(bam_factory,count):
    records=[]
    for i in range(count):records.extend(pair(str(i),a={'flag':99|1024},b={'flag':147|1024}))
    _,rows,_=output(bam_factory(records))
    assert rows==[f'chr2\t14\t75\taCgT-1\t{count}']


def normalize(records):
    """Set reciprocal metadata for deliberately valid geometric test templates."""
    import re
    for a,b in ((records[0],records[1]),(records[1],records[0])):
        a['mrid']=b['rid'];a['mpos']=b['pos'];a['tlen']=0
        a['flag']=(a['flag']&~40)|(8 if b['flag']&4 else 0)|(32 if b['flag']&16 else 0)
    return records


@pytest.mark.parametrize('left',[29,30,31,255])
@pytest.mark.parametrize('right',[29,30,31,255])
def test_both_mate_mapq_before_aggregation(bam_factory,left,right):
    _,rows,record=output(bam_factory(pair('anchor')+pair('candidate',a={'mapq':left},b={'mapq':right})))
    passing=left in (30,31) and right in (30,31)
    assert rows==[f'chr2\t14\t75\taCgT-1\t{1+passing}']
    assert record['qualification']['exclusions']['mapq']==int(not passing)


@pytest.mark.parametrize('changes,reason',[
    ({'a':{'flag':99|4}},'unmapped'),
    ({'b':{'rid':1}},'discordant'),
    ({'a':{'flag':99&~2}},'improper'),
    ({'b':{'flag':147|512}},'qc_failed'),
    ({'a':{'tags':[('SA','chr2,20,+,20M,30,0;','Z')]}},'split'),
    ({'a':{'cigar':'10M1N10M'}},'unsupported_cigar'),
    ({'a':{'cigar':'10M1P10M'}},'unsupported_cigar'),
    ({'a':{'cigar':'5S20M'}},'unsupported_cigar'),
    ({'a':{'cigar':'5H20M'}},'unsupported_cigar'),
    ({'b':{'cigar':'20M5S'}},'unsupported_cigar'),
    ({'b':{'cigar':'20M5H'}},'unsupported_cigar'),
    ({'a':{'cigar':'1I20M'}},'unsupported_cigar'),
    ({'b':{'cigar':'20M1D'}},'unsupported_cigar'),
    ({'a':{'pos':80},'b':{'pos':10}},'geometry'),
    ({'a':{'flag':99|16}},'geometry'),
    ({'a':{'cigar':'80M'}},'geometry'),
    ({'a':{'cb':None}},'missing_cb'),
    ({'a':{'cigar':'4M'},'b':{'pos':14,'cigar':'4M'}},'short_shifted'),
])
def test_exclusions_are_pairwise_and_deterministic(bam_factory,changes,reason):
    candidate=normalize(pair('candidate',**changes))
    _,rows,record=output(bam_factory(pair('anchor')+candidate))
    assert rows==['chr2\t14\t75\taCgT-1\t1']
    assert record['qualification']['exclusions']==dict.fromkeys(m.EXCLUSIONS,0)|{reason:1}


@pytest.mark.parametrize('changes,expected',[
    ({'a':{'cigar':'10M2I10M'}},'chr2\t14\t75\taCgT-1\t1'),
    ({'b':{'cigar':'10M2D10M'}},'chr2\t14\t77\taCgT-1\t1'),
    ({'a':{'cigar':'20M5S'}},'chr2\t14\t75\taCgT-1\t1'),
    ({'b':{'cigar':'5S20M'}},'chr2\t14\t75\taCgT-1\t1'),
    ({'a':{'cigar':'20M5H'}},'chr2\t14\t75\taCgT-1\t1'),
    ({'b':{'cigar':'5H20M'}},'chr2\t14\t75\taCgT-1\t1'),
    ({'b':{'pos':20}},'chr2\t14\t35\taCgT-1\t1'),
    ({'a':{'flag':163},'b':{'flag':83}},'chr2\t14\t75\taCgT-1\t1'),
])
def test_hand_derived_cigar_geometry(bam_factory,changes,expected):
    _,rows,_=output(bam_factory(normalize(pair(**changes))))
    assert rows==[expected]


@pytest.mark.parametrize('changes',[
    {'a':{'cb':'DIFFERENT'}}, {'a':{'mpos':61}}, {'a':{'flag':99&~32}},
    {'a':{'flag':99|8}}, {'a':{'flag':99|128}}, {'a':{'flag':99&~1}},
    {'a':{'tlen':71}}, {'b':{'tlen':0}}, {'a':{'rg':'a'},'b':{'rg':'b'}},
    {'a':{'cb':'with space'}}, {'a':{'tags':[('CB','another','Z')]}},
    {'a':{'tags':[('SA','malformed','Z')]}}, {'a':{'cigar':'10M2S10M'}},
    {'a':{'pos':995}}, {'a':{'name':None}},
])
def test_malformed_templates_fail_not_excluded(bam_factory,changes):
    with pytest.raises(ValueError):
        output(bam_factory(pair('anchor')+pair('bad',**changes),rg=[{'ID':'a','LB':'library'},{'ID':'b','LB':'library'}]))


@pytest.mark.parametrize('kind',['missing','duplicate_r1','duplicate_r2','duplicate_pair','secondary','supplementary'])
def test_primary_multiplicity_and_ancillary_records(bam_factory,kind):
    records=pair('candidate')
    if kind=='missing':records.pop()
    elif kind=='duplicate_r1':records.append(dict(records[0]))
    elif kind=='duplicate_r2':records.append(dict(records[1]))
    elif kind=='duplicate_pair':records+=pair('candidate')
    else:records.append(records[0]|{'flag':99|(256 if kind=='secondary' else 2048)})
    args=bam_factory(pair('anchor')+records)
    if kind not in ('secondary','supplementary'):
        with pytest.raises(ValueError):output(args)
    else:
        _,rows,record=output(args)
        assert rows==[f'chr2\t14\t75\taCgT-1\t{2 if kind=="secondary" else 1}']
        assert record['qualification']['n_'+kind]==1


def test_exact_key_no_endpoint_or_barcode_collapse(bam_factory):
    records=pair('a')+normalize(pair('b',b={'pos':61}))+normalize(pair('c',a={'pos':11}))
    records+=pair('d',a={'cb':'ACGT-1'},b={'cb':'ACGT-1'})
    _,rows,_=output(bam_factory(records[::-1]))
    assert rows==['chr2\t14\t75\tACGT-1\t1','chr2\t14\t75\taCgT-1\t1',
                  'chr2\t14\t76\taCgT-1\t1','chr2\t15\t75\taCgT-1\t1']


@pytest.mark.parametrize('species',['human','mouse'])
@pytest.mark.parametrize('sq',[[{'SN':'chr2','LN':1000}], [{'SN':'chr1','LN':1000},{'SN':'chr2','LN':1000}]])
def test_reference_subset_order_species(bam_factory,species,sq):
    _,rows,_=output(bam_factory(species=species,sq=sq))
    assert rows==[f'{sq[0]["SN"]}\t14\t75\taCgT-1\t1']


@pytest.mark.parametrize('sq',[
    [{'SN':'chr2','LN':999}], [{'SN':'2','LN':1000}],
    [{'SN':'chr2','LN':1000,'M5':'0'*32}],
])
def test_reference_mismatch(bam_factory,sq):
    with pytest.raises(ValueError):output(bam_factory(sq=sq))


def test_md5_match(bam_factory):
    import hashlib
    _,rows,_=output(bam_factory(sq=[{'SN':'chr2','LN':1000,'M5':hashlib.md5(b'A'*1000).hexdigest()}]))
    assert len(rows)==1


def test_explicit_namespace_and_context_policy(bam_factory):
    for namespace in ('library_A','library_B'):
        result,_,record=output(bam_factory(namespace=namespace))
        assert result['namespace']==record['namespace']==namespace
    with pytest.raises(ValueError):output(bam_factory(corrected=False))


@pytest.mark.parametrize('mutation',['sha','profile','truncated','appended','reference','context'])
def test_identity_mismatch(bam_factory,mutation):
    args=bam_factory()
    if mutation=='sha':args['source_sha256']='0'*64
    elif mutation=='profile':args['source_profile']='unknown'
    elif mutation in ('truncated','appended'):
        path=Path(args['source_path']);raw=path.read_bytes();path.write_bytes(raw[:-28] if mutation=='truncated' else raw+b'bad')
    else:args[mutation+'_bundle_sha256' if mutation=='reference' else 'library_context_sha256']='0'*64
    with pytest.raises(ValueError):output(args)


def test_support_bounds():
    assert production.checked_support(2**64-2,1)==2**64-1
    for base,inc in ((2**64-1,1),(0,2**64),(0,True),(0,0)):
        with pytest.raises(ValueError):production.checked_support(base,inc)


@pytest.mark.parametrize('records',[
    pair(a={'rg':'one'},b={'rg':'one'}),
    pair('a',a={'rg':'one'},b={'rg':'one'})+pair('b',a={'rg':'two'},b={'rg':'two'}),
])
def test_read_groups_same_library(bam_factory,records):
    result,_,_=output(bam_factory(records,rg=[{'ID':'one','LB':'library'},{'ID':'two','LB':'library'}]))
    assert result['eligible_pairs']==len(records)//2


@pytest.mark.parametrize('cb,tags',[(None,[('CR','ACGT','Z')]),(None,[('XC','ACGT','Z')])])
def test_cr_and_custom_barcode_sources_are_not_executable(bam_factory,cb,tags):
    with pytest.raises(ValueError):output(bam_factory(pair(a={'cb':cb,'tags':tags},b={'cb':cb,'tags':tags})))


def test_bad_index_is_not_execution_requirement(bam_factory):
    # M10 binds its own advisory index observations; a new unrelated sidecar
    # after intake would rightly invalidate that intake observation. Use a
    # noncandidate index name to prove no additional input is required/consumed.
    args=bam_factory();Path(args['source_path']).with_name('unselected-index').write_bytes(b'bad')
    assert output(args)[0]['eligible_pairs']==1


def test_unmapped_mate_unknown_orientation_and_location_is_excluded(bam_factory):
    records=pair('unmapped',a={'flag':73,'mrid':-1,'mpos':-1,'tlen':0},
                 b={'flag':133,'rid':-1,'pos':-1,'cigar':None,'mrid':0,'mpos':10,'tlen':0})
    _,_,record=output(bam_factory(pair('anchor')+records))
    assert record['qualification']['exclusions']['unmapped']==1


def test_exclusion_priority_is_frozen(bam_factory):
    records=pair('candidate',a={'flag':99|512,'mapq':29,'cb':None},b={'cb':None})
    _,_,record=output(bam_factory(pair('anchor')+records))
    assert record['qualification']['exclusions']['qc_failed']==1
    assert sum(record['qualification']['exclusions'].values())==1


def test_sam_bytes_rejected_by_decoder(tmp_path):
    path=tmp_path/'disguised.bam';path.write_text('@HD\tVN:1.6\n@SQ\tSN:chr2\tLN:1000\n')
    with pytest.raises(ValueError):io.project(str(path),tmp_path/'projection')


def test_aggregate_overflow_is_checked(tmp_path,monkeypatch):
    source=tmp_path/'rows';source.write_text('0\t1\t10\tA\t\t2\n0\t2\t10\tA\t\t2\n')
    monkeypatch.setattr(production,'MAX_TOTAL',3)
    with pytest.raises(ValueError,match='OVERFLOW'):production.aggregate(source,(('chr2',1000),),tmp_path,FragmentVerificationRuntime())


@pytest.mark.parametrize('which',['source','context'])
def test_other_valid_artifact_cannot_replace_selected_binding(bam_factory,which):
    a=bam_factory();b=bam_factory(namespace='other')
    keys=('source_path','source_sha256') if which=='source' else ('library_context_path','library_context_sha256')
    a.update({k:b[k] for k in keys})
    with pytest.raises(ValueError):output(a)


@pytest.mark.parametrize('remove',[1,28,45])
def test_complete_decoder_rejects_truncated_eof(bam_factory,tmp_path,remove):
    args=bam_factory();path=Path(args['source_path']);path.write_bytes(path.read_bytes()[:-remove])
    with pytest.raises(ValueError):io.project(str(path),tmp_path/'projection')


def test_handled_interruption_removes_private_staging(bam_factory,monkeypatch):
    args=bam_factory()
    def interrupt(*a,**k):raise KeyboardInterrupt()
    monkeypatch.setattr(production,'generate',interrupt)
    with pytest.raises(KeyboardInterrupt):public.prepare_scATAC_bam_fragments(**args)
    assert not list(Path(args['output_dir']).glob('.bam-attempt-*'))
    assert not list(Path(args['output_dir']).glob('bam-fragments-*'))


def test_all_exact_contigs_including_mitochondria_and_scaffolds(bam_factory):
    records=pair('scaffold',a={'rid':1,'mrid':1},b={'rid':1,'mrid':1})+pair('mito')
    _,rows,_=output(bam_factory(records,reference_names=('chrM','scaffold42')))
    assert [r.split('\t')[0] for r in rows]==['chrM','scaffold42']


def test_output_uses_fai_order_not_bam_dictionary_or_input_order(bam_factory):
    records=pair('chr1')+pair('chr2',a={'rid':1,'mrid':1},b={'rid':1,'mrid':1})
    _,rows,_=output(bam_factory(records,sq=[{'SN':'chr1','LN':1000},{'SN':'chr2','LN':1000}]))
    assert [r.split('\t')[0] for r in rows]==['chr2','chr1']


def test_existing_context_accepts_identical_relocated_intake(bam_factory,tmp_path):
    args=bam_factory();new=tmp_path/'relocated-intake.json'
    new.write_bytes(Path(args['intake_manifest_path']).read_bytes());args['intake_manifest_path']=str(new)
    assert output(args)[0]['eligible_pairs']==1


def test_rg_string_does_not_inherit_cb_identifier_restrictions(bam_factory):
    args=bam_factory(pair(a={'rg':'read group'},b={'rg':'read group'}),rg=[{'ID':'read group','LB':'library'}])
    assert output(args)[0]['eligible_pairs']==1
