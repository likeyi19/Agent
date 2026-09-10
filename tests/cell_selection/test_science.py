from fractions import Fraction
import hashlib
import pytest
from agent.tools.data import scatac_selection_profile as p, cell_selection_verifier as v, _cell_selection_contract as m


def args(**changes):
    return dict(min_qc_fragment_records=2,min_tss_enrichment='200/101',min_tss_flank_evidence=None,
        max_qc_fragment_records=None,max_nucleosome_signal=None)|changes


@pytest.mark.parametrize('changes,counts,expected',[
    ({},(3,1,1,1,0,1,0,0),('QC_FRAGMENT_COUNT_BELOW_MIN',)),
    ({},(3,2,1,1,0,1,1,0),()),
    ({},(3,3,1,1,0,1,1,1),()),
    ({'max_qc_fragment_records':2},(3,2,1,1,0,1,1,0),()),
    ({'max_qc_fragment_records':2},(3,3,1,1,0,1,1,1),('QC_FRAGMENT_COUNT_ABOVE_MAX',)),
    ({'min_tss_enrichment':'1/10'},(100,100,101,1000,1000,100,0,0),()),
    ({'min_tss_enrichment':'0.100000000000000001'},(100,100,101,1000,1000,100,0,0),('TSS_ENRICHMENT_BELOW_MIN',)),
    ({'min_tss_enrichment':'0.099999999999999999'},(100,100,101,1000,1000,100,0,0),()),
    ({'min_tss_flank_evidence':2},(3,2,1,1,0,1,1,0),('TSS_FLANK_EVIDENCE_BELOW_MIN',)),
    ({'min_tss_flank_evidence':1},(3,2,1,1,0,1,1,0),()),
    ({'min_tss_flank_evidence':0},(3,2,1,1,0,1,1,0),()),
    ({},(3,2,0,0,0,0,2,0),('TSS_ENRICHMENT_UNDEFINED',)),
    ({'max_nucleosome_signal':'1'},(3,2,1,1,0,0,2,0),('NUCLEOSOME_SIGNAL_UNDEFINED',)),
    ({'max_nucleosome_signal':'1/2'},(3,3,1,1,0,2,1,0),()),
    ({'max_nucleosome_signal':'0.49'},(3,3,1,1,0,2,1,0),('NUCLEOSOME_SIGNAL_ABOVE_MAX',)),
    ({'max_nucleosome_signal':'0.51'},(3,3,1,1,0,2,1,0),()),
    ({'min_tss_flank_evidence':1,'max_nucleosome_signal':1},(3,0,0,0,0,0,0,0),
        ('QC_FRAGMENT_COUNT_BELOW_MIN','TSS_ENRICHMENT_UNDEFINED','TSS_FLANK_EVIDENCE_BELOW_MIN','NUCLEOSOME_SIGNAL_UNDEFINED')),
])
def test_exact_boundaries(changes,counts,expected):
    policy=p.thresholds(args(**changes))
    assert policy==v.canonical_thresholds(args(**changes))
    assert p.decide(counts,policy)==v.failures(counts,policy)==expected


@pytest.mark.parametrize('value',[True,False,0.1,float('nan'),float('inf'),'-1','+1','01','1.',' 1','1e2','1/0','1/01','.5','1_000','1\n','0.'+'1'*19,10**18+1])
def test_invalid_thresholds(value):
    with pytest.raises(ValueError):p.thresholds(args(min_tss_enrichment=value))
    with pytest.raises(ValueError):v.canonical_thresholds(args(min_tss_enrichment=value))


@pytest.mark.parametrize('value',[0,1,'0','1.00','2/2','0.125','1/8','1000000000000000000'])
def test_canonical_thresholds(value):
    actual=p.thresholds(args(min_tss_enrichment=value))
    assert actual==v.canonical_thresholds(args(min_tss_enrichment=value))
    expected=Fraction(value)
    assert actual['min_tss_enrichment']=={'numerator':expected.numerator,'denominator':expected.denominator}


@pytest.mark.parametrize('pair',[('library_0','AA-1'),('library_1','AA-1'),('library_0','aa-1'),('ns','!:._-~'),('n'*128,'B'*256)])
def test_identity_roundtrip(pair):
    encoded=p.encode_cell_id(*pair)
    assert '=' not in encoded and len(encoded)<=529
    assert p.decode_cell_id(encoded)==pair


@pytest.mark.parametrize('value',['','scatac-cell.v2.YQ.Yg','scatac-cell.v1..Yg','scatac-cell.v1.YQ==.Yg','scatac-cell.v1.YR.Yg',
    'scatac-cell.v1.YQ.Yg.extra','scatac-cell.v1.YQ./w','scatac-cell.v1.YQ.AA'])
def test_invalid_rendered_ids(value):
    with pytest.raises(ValueError):p.decode_cell_id(value)


@pytest.mark.parametrize('pair',[('','A'),('ns',''),('ns','A B'),('ns','A\n'),('n'*129,'b'),('ns','b'*257)])
def test_invalid_identity(pair):
    with pytest.raises(ValueError):p.encode_cell_id(*pair)


def test_empty_digest_and_large_integers():
    assert hashlib.sha256(m.ORDER_DOMAIN).hexdigest()!=hashlib.sha256(b'').hexdigest()
    a=args(min_tss_enrichment='200/101');counts=(10**9,10**9,2*10**15,10**15,10**15,10**9,0,0)
    assert p.decide(counts,p.thresholds(a))==v.failures(counts,v.canonical_thresholds(a))==()
