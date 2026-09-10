from fractions import Fraction
import gzip
import json
from pathlib import Path
import pytest
from agent.tools.data import scatac_barcode_qc as public,_barcode_qc_contract as m
from agent.tools.data.barcode_qc_verifier import verify_barcode_qc
from agent.tools.data.scatac_qc_profile import ScATACQCError


def table(result):
    return [line.split('\t') for line in gzip.decompress((Path(result['manifest_path']).parent/'barcodes.tsv.gz').read_bytes()).decode().splitlines()[1:]]


def test_exact_science_and_full_universe(qc_case):
    args,source,*_=qc_case;result=public.compute_scATAC_qc(**args)
    assert result['n_observed_barcodes']==3 and result['n_fragment_records']==5 and result['n_qc_fragment_records']==4
    rows=table(result);assert [r[1] for r in rows]==['A-1','a-1','zero:background']
    assert rows[0][2:10]==['3','3','5','4','5','3','0','0']
    assert rows[0][10:13]==['1000','909','DEFINED']
    assert rows[1][7:10]==['0','1','0']
    assert rows[2][3:]==['0','0','0','0','0','0','0','NA','NA','ZERO_TSS_BACKGROUND','NA','NA','ZERO_NUCLEOSOME_FREE']
    value=verify_barcode_qc(result['manifest_path'],expected_sha256=result['manifest_sha256']).to_dict()
    assert value['summary']['histogram_records']==4
    assert value['resource_qualification']['mode']=='synthetic_only'


@pytest.mark.parametrize('offset',[-2001,-2000,-1901,-1900,-51,-50,0,50,51,1900,1901,2000,2001])
def test_endpoint_boundaries(fixture_factory,offset):
    p=3000+offset
    args,*_=fixture_factory([('chr2',p,p+1,'one',987)])
    row=table(public.compute_scATAC_qc(**args))[0]
    positions=[3000,3000,3001]
    expected=[2*sum(lo<=p-tss<=hi for tss in positions) for lo,hi in [(-50,50),(-2000,-1901),(1901,2000)]]
    assert list(map(int,row[4:7]))==expected


@pytest.mark.parametrize('length,bin_index',[(1,0),(146,0),(147,1),(293,1),(294,2),(1000,2),(1001,2)])
def test_length_histogram(fixture_factory,length,bin_index):
    args,*_=fixture_factory([('chr2',2000,2000+length,'length',100)])
    result=public.compute_scATAC_qc(**args);row=table(result)
    assert list(map(int,row[0][7:10]))==[int(i==bin_index) for i in range(3)]
    h=gzip.decompress((Path(result['manifest_path']).parent/'lengths.tsv.gz').read_bytes()).decode().splitlines()
    assert len(h)==1001 and h[min(length,1001)-1].split('\t')[1]=='1'


def test_support_does_not_change_metrics(fixture_factory):
    rows=[('chr2',1000,3001,'Mixed-1',1)]
    a,*_=fixture_factory(rows);b,*_=fixture_factory([(*rows[0][:4],900)])
    x=public.compute_scATAC_qc(**a);y=public.compute_scATAC_qc(**b)
    assert table(x)==table(y)


def test_no_production_path_in_verifier(qc_case,monkeypatch):
    from agent.tools.data import _barcode_qc_production as production
    args,*_=qc_case;result=public.compute_scATAC_qc(**args)
    monkeypatch.setattr(production,'produce',lambda *a:pytest.fail('Production used in verifier'))
    monkeypatch.setattr(production,'_intersection',lambda *a:pytest.fail('BEDTools used in verifier'))
    public.verify_public_result(args,result)


def test_missing_resource_or_backend_qualification(qc_case,monkeypatch):
    args,*_=qc_case
    monkeypatch.delenv('AGENT_QC_ALLOW_SYNTHETIC')
    with pytest.raises(ScATACQCError,match='UNQUALIFIED'):public.compute_scATAC_qc(**args)
    monkeypatch.setenv('AGENT_QC_ALLOW_SYNTHETIC','1');monkeypatch.delenv('AGENT_QC_BEDTOOLS')
    with pytest.raises(ScATACQCError,match='BEDTOOLS_UNAVAILABLE'):public.compute_scATAC_qc(**args)
    assert not Path(args['output_dir']).exists()


def test_exact_large_integer_format():
    c=2**100;raw=m.row_bytes('ns','bc',(1,1,c,1,0,1,0,0)).decode().split('\t')
    assert Fraction(int(raw[10]),int(raw[11]))==Fraction(200*c,101)


def test_no_qc_contig_records_preserves_observed_barcode(fixture_factory):
    args,*_=fixture_factory([('organelle',1,2,'kept-2',999)])
    result=public.compute_scATAC_qc(**args)
    assert result['n_observed_barcodes']==1 and result['n_qc_fragment_records']==0
    assert table(result)[0][10:13]==['NA','NA','ZERO_TSS_BACKGROUND']
    value=public.verify_public_result(args,result).to_dict()
    assert value['summary']['histogram_records']==0 and value['summary']['tss_min'] is None
