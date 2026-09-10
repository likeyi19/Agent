import gzip
import hashlib
import json
from pathlib import Path
import pytest
from agent.tools.data import scatac_cell_selection as public, _cell_selection_contract as m, _barcode_qc_contract as qc
from agent.tools.data.scatac_barcode_qc import compute_scATAC_qc
from agent.tools.data.cell_selection_verifier import verify_cell_selection


@pytest.fixture
def published(fixture_factory,tmp_path):
    case=fixture_factory(rows=[('chr2',1000,1001,b,7) for b in ('A-1','a-1')]+
        [('chr2',2950,3051,b,9) for b in ('A-1','a-1')]+[('organelle',10,12,'zero:background',3)])
    q=compute_scATAC_qc(**case[0]);args=dict(barcode_qc_manifest_path=q['manifest_path'],barcode_qc_manifest_sha256=q['manifest_sha256'],
        min_qc_fragment_records=0,min_tss_enrichment='0',output_dir=str(tmp_path/'selected'))
    result=public.select_scATAC_cells(**args)
    return args,result


@pytest.mark.parametrize('mutation',['boolean','missing_reason','extra_reason','threshold','drop','add','decision_order',
    'selected_order','index','namespace','barcode','rendered','selected_count','reason_summary','digest','lineage'])
def test_rehashed_forgery_fails_scientific_verification(published,mutation):
    args,result=published;path=Path(result['manifest_path']);value=json.loads(path.read_bytes())
    decisions=gzip.decompress((path.parent/'decisions.tsv.gz').read_bytes()).decode().splitlines()
    selected=gzip.decompress((path.parent/'selected.tsv.gz').read_bytes()).decode().splitlines()
    first=decisions[1].split('\t');last=decisions[-1].split('\t')
    if mutation=='boolean': first[4:6]=['false','false']
    elif mutation=='missing_reason':last[-1]='NONE'
    elif mutation=='extra_reason':first[-1]='QC_FRAGMENT_COUNT_BELOW_MIN'
    elif mutation=='threshold':
        value['arguments']['min_tss_enrichment']='1000000';value['thresholds']['min_tss_enrichment']={'numerator':1000000,'denominator':1}
    elif mutation=='namespace':first[0]='different'
    elif mutation=='barcode':first[1]='different'
    elif mutation=='rendered':first[2]=decisions[2].split('\t')[2]
    elif mutation=='selected_count':value['selected_count']-=1;value['rejected_count']+=1
    elif mutation=='reason_summary':value['reason_counts']['QC_FRAGMENT_COUNT_BELOW_MIN']=1
    elif mutation=='digest':value['ordered_selected_sha256']='1'*64
    elif mutation=='lineage':value['qc_identity_sha256']='1'*64
    decisions[1]='\t'.join(first);decisions[-1]='\t'.join(last)
    if mutation=='drop':decisions.pop()
    elif mutation=='add':decisions.append(decisions[-1])
    elif mutation=='decision_order':decisions[1:3]=reversed(decisions[1:3])
    elif mutation=='selected_order':selected[1:3]=reversed(selected[1:3])
    elif mutation=='index':selected[1]='9'+selected[1][1:]
    for key,lines in (('decisions',decisions),('selected',selected)):
        p=path.parent/value[key]['path'];p.write_bytes(gzip.compress(('\n'.join(lines)+'\n').encode(),mtime=0))
        value[key]=qc.resource(p)
    value['identity_sha256']=m.identity(value);path.write_bytes(qc.canonical(value));sha=hashlib.sha256(path.read_bytes()).hexdigest()
    # Every case is a closed, self-consistent manifest with valid sidecar hashes.
    m.load_manifest(path,sha)
    for key in ('decisions','selected'):assert qc.resource(path.parent/value[key]['path'])==value[key]
    with pytest.raises(ValueError,match='SELECTION_.*MISMATCH'):
        verify_cell_selection(path,expected_sha256=sha)


def test_changed_comparison_profile_rejected(published):
    args,result=published;path=Path(result['manifest_path']);value=json.loads(path.read_bytes())
    value['selection_profile']['comparisons']='minimum_exclusive_maximum_inclusive'
    value['selection_profile_sha256']=qc.digest(value['selection_profile']);value['identity_sha256']=m.identity(value)
    with pytest.raises(ValueError,match='SELECTION_PROFILE_INVALID'):m.CellSelectionManifest(qc.canonical(value))


def test_threshold_boolean_alias_is_not_canonical(published):
    _,result=published;value=json.loads(Path(result['manifest_path']).read_bytes())
    value['thresholds']['min_tss_enrichment']['numerator']=False
    value['identity_sha256']=m.identity(value)
    with pytest.raises(ValueError,match='SELECTION_THRESHOLD_MISMATCH'):m.CellSelectionManifest(qc.canonical(value))


@pytest.mark.parametrize('field',['arguments','resource_qualification','producer_authority'])
def test_nested_lineage_is_closed(published,field):
    _,result=published;value=json.loads(Path(result['manifest_path']).read_bytes())
    value['qc_lineage'][field]['invented_science']='unsupported'
    value['identity_sha256']=m.identity(value)
    with pytest.raises(ValueError):m.CellSelectionManifest(qc.canonical(value))


def test_selection_retains_synthetic_gate(selection_case,monkeypatch):
    args,_=selection_case;monkeypatch.delenv('AGENT_QC_ALLOW_SYNTHETIC')
    with pytest.raises(ValueError):public.select_scATAC_cells(**args)
    assert not Path(args['output_dir']).exists()


def test_multiple_namespaces_selected_order(tmp_path,monkeypatch):
    from barcode_qc.test_producer_neutrality import fastq_fixture,compute
    _,fragments,barcode=fastq_fixture(tmp_path,monkeypatch,namespaces=2)
    compute(tmp_path,fragments,0)
    path=next((tmp_path/'qc-0').glob('barcode-qc-*/manifest.json'))
    result=public.select_scATAC_cells(barcode_qc_manifest_path=str(path),barcode_qc_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        min_qc_fragment_records=0,min_tss_enrichment=0,output_dir=str(tmp_path/'selection'))
    rows=gzip.decompress((Path(result['manifest_path']).parent/'selected.tsv.gz').read_bytes()).decode().splitlines()[1:]
    tuples=[line.split('\t') for line in rows]
    assert [(r[0],r[1],r[2]) for r in tuples]==[('0','library_0',barcode),('1','library_1',barcode)]
    assert tuples[0][3]!=tuples[1][3]


def test_exact_qc_pointer_cannot_be_substituted(published,tmp_path):
    args,result=published
    source=Path(args['barcode_qc_manifest_path']);other=tmp_path/'other-qc.json';other.write_bytes(source.read_bytes())
    with pytest.raises(ValueError,match='SELECTION_RECOVERY_MISMATCH'):
        public.verify_public_result(args|dict(barcode_qc_manifest_path=str(other)),result)


def test_threshold_and_execution_change_recovery_identity(published):
    args,_=published
    assert public._publication(args,'run-A')[2]!=public._publication(args,'run-B')[2]
    assert public._publication(args,'run-A')[2]!=public._publication(args|dict(min_tss_enrichment='0.1'),'run-A')[2]


def test_empty_and_deterministic_sidecars(published,tmp_path):
    args,result=published
    repeat=public.select_scATAC_cells(**(args|dict(output_dir=str(tmp_path/'repeat'))))
    for name in ('decisions.tsv.gz','selected.tsv.gz'):
        assert (Path(result['manifest_path']).parent/name).read_bytes()==(Path(repeat['manifest_path']).parent/name).read_bytes()
    empty=public.select_scATAC_cells(**(args|dict(output_dir=str(tmp_path/'empty'),min_qc_fragment_records=10**18)))
    value=public.verify_public_result(args|dict(output_dir=str(tmp_path/'empty'),min_qc_fragment_records=10**18),empty).to_dict()
    assert empty['n_selected']==0 and empty['readiness']=='no_selected_cells'
    assert value['ordered_selected_sha256']==hashlib.sha256(m.ORDER_DOMAIN).hexdigest()
    assert gzip.decompress((Path(empty['manifest_path']).parent/'selected.tsv.gz').read_bytes())==m.SELECTED_HEADER


def test_verifier_does_not_call_production(published,monkeypatch):
    from agent.tools.data import _cell_selection_production as production, scatac_selection_profile as profile
    args,result=published
    def forbidden(*a,**k):pytest.fail('Production decision or rendering called by verifier')
    monkeypatch.setattr(production,'produce',forbidden);monkeypatch.setattr(profile,'decide',forbidden);monkeypatch.setattr(profile,'encode_cell_id',forbidden)
    public.verify_public_result(args,result)


@pytest.mark.parametrize('target',['qc_manifest','qc_table','qc_resource','qc_hash'])
def test_qc_integrity(selection_case,target):
    args,case=selection_case
    if target=='qc_hash':args=args|dict(barcode_qc_manifest_sha256='0'*64)
    else:
        path=Path(args['barcode_qc_manifest_path'])
        if target=='qc_table':path=path.parent/'barcodes.tsv.gz'
        elif target=='qc_resource':path=Path(case[0]['qc_reference_manifest_path'])
        with path.open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError):public.select_scATAC_cells(**args)
    assert not Path(args['output_dir']).exists()
