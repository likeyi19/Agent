import copy
import csv
import json
from pathlib import Path

import pytest
from agent.tools.analysis import marker_validation as v


@pytest.fixture
def case(tmp_path):
    def write(name,value):
        p=tmp_path/name;p.write_text(json.dumps(value));return str(p),v.sha256(p)
    primary={'groups':[dict(group='opaque',candidate='candidate-A',status='assigned',primary_annotation='candidate-A'),
                       dict(group='unresolved',candidate=None,status='unresolved',primary_annotation=None)]}
    p,ph=write('primary.json',primary);g,gh=write('genes.json',['G','H','J'])
    marker=tmp_path/'markers.tsv';marker.write_text('cluster\tgene\tavg_logFC\nopaque\tG\t1\nunresolved\tH\t2\n')
    table=tmp_path/'knowledge.tsv'
    rows=[]
    for typ,markername,symbol in [('Type A','G','G'),('Type A','absent','UNMEASURED'),('Type B','H','H'),('Type A','alias','J'),('Type A','alias','H')]:
        r=dict.fromkeys(v.FIELDS,'None');r.update(species='Human',tissue_type='Tissue',cancer_type='Normal',cell_name=typ,marker=markername,Symbol=symbol,PMID='123')
        rows.append(r)
    with table.open('w') as f:
        w=csv.DictWriter(f,fieldnames=v.FIELDS,delimiter='\t');w.writeheader();w.writerows(rows)
    resource=v.KnowledgeResource(str(table),v.sha256(table),str(table),v.sha256(table),'synthetic','fixture-v1','2026-09-15','synthetic data')
    mapping=dict(schema='external-marker-alignment.v1',species='Human',context='declared',tissues=['Tissue'],cancer_type='Normal',resource_table_sha256=resource.table_sha256,primary_sha256=ph,rationale='synthetic',candidates={'candidate-A':[dict(external_type='Type A',relationship='equivalent',rationale='synthetic'),dict(external_type='Type B',relationship='incompatible',rationale='synthetic alternative')]})
    m,mh=write('mapping.json',mapping)
    return dict(primary_path=p,primary_sha256=ph,genes_path=g,genes_sha256=gh,markers_path=str(marker),markers_sha256=v.sha256(marker),resource=resource,mapping_path=m,mapping_sha256=mh,species='Human',context='declared',profile=v.PROFILE)


def change(args,key,edit):
    p=Path(args[key]);data=json.loads(p.read_text());edit(data);p.write_text(json.dumps(data));args[key.replace('_path','_sha256')]=v.sha256(p)


def test_supported_immutable_and_deterministic(case):
    before=Path(case['primary_path']).read_bytes()
    result=v.corroborate_marker_annotation(**case)
    assert result==v.corroborate_marker_annotation(**case)
    assert result['groups'][0]['corroboration_state']=='corroborating'
    assert result['groups'][1]['primary_state']=='unresolved'
    assert result['groups'][1]['corroboration_state']=='not_applicable'
    assert result['groups'][0]['primary_annotation']=='candidate-A'
    assert result['groups'][1]['primary_annotation'] is None
    assert Path(case['primary_path']).read_bytes()==before
    assert result['groups'][0]['evidence'][0]['unmeasured']==['UNMEASURED']


def test_competing_associations_do_not_fabricate_conflict(case):
    p=Path(case['markers_path']);p.write_text(p.read_text()+'opaque\tH\t1\n');case['markers_sha256']=v.sha256(p)
    result=v.corroborate_marker_annotation(**case)
    assert result['groups'][0]['corroboration_state']=='mixed'
    assert result['groups'][0]['primary_annotation']=='candidate-A'
    assert result['groups'][0]['competing_positive_associations']
    assert result['groups'][0]['conflict_assessment']=='not_established_by_positive_marker_membership'


def test_negative_not_absence_or_conflict(case):
    p=Path(case['markers_path']);p.write_text(p.read_text().replace('opaque\tG\t1','opaque\tG\t-1'));case['markers_sha256']=v.sha256(p)
    r=v.corroborate_marker_annotation(**case)['groups'][0]
    assert r['corroboration_state']=='insufficient_evidence'
    assert r['evidence'][0]['lower_rp_markers']==['G']
    assert r['primary_annotation']=='candidate-A'


def test_ambiguous_aliases(case):
    rows=v.load_knowledge(case['resource'])
    assert [r['mapping_status'] for r in rows][-2:]==['ambiguous','ambiguous']
    assert all(r['normalized'] is None for r in rows[-2:])
    assert rows[0]['original']['marker']=='G'


@pytest.mark.parametrize('key,value',[('species','Mouse'),('context','other'),('profile','other')])
def test_input_mismatch(case,key,value):
    case[key]=value
    with pytest.raises(ValueError):v.corroborate_marker_annotation(**case)


@pytest.mark.parametrize('field', ['primary_path','genes_path','markers_path','mapping_path'])
def test_hash_mismatch(case,field):
    Path(case[field]).write_text('changed')
    with pytest.raises(ValueError):v.corroborate_marker_annotation(**case)


def test_resource_change_identity(case):
    from dataclasses import replace
    old=v.corroborate_marker_annotation(**case)['identity_sha256']
    case['resource']=replace(case['resource'],release='new explicit release')
    assert v.corroborate_marker_annotation(**case)['identity_sha256']!=old
    Path(case['resource'].table_path).write_text('changed')
    with pytest.raises(ValueError):v.corroborate_marker_annotation(**case)


@pytest.mark.parametrize('field,value',[('tissues',['unknown']),('species','Mouse'),('context','elsewhere'),('resource_table_sha256','a'*64),('primary_sha256','a'*64)])
def test_mapping_context_contract(case,field,value):
    change(case,'mapping_path',lambda x:x.update({field:value}))
    with pytest.raises(ValueError):v.corroborate_marker_annotation(**case)


def test_unmapped_and_narrower(case):
    change(case,'mapping_path',lambda x:x['candidates'].update({'candidate-A':[]}))
    assert v.corroborate_marker_annotation(**case)['groups'][0]['corroboration_state']=='unmapped'


def test_unknown_cell_identity(case):
    change(case,'mapping_path',lambda x:x['candidates']['candidate-A'][0].update(external_type='unknown'))
    with pytest.raises(ValueError):v.corroborate_marker_annotation(**case)


def test_case_collision(case):
    change(case,'genes_path',lambda x:x.append('g'))
    with pytest.raises(ValueError):v.corroborate_marker_annotation(**case)


def test_missing_symbol_does_not_poison_known_mapping(case):
    from dataclasses import replace
    p=Path(case['resource'].table_path)
    with p.open(newline='') as f:rows=list(csv.DictReader(f,delimiter='\t'))
    rows.append({**rows[0],'Symbol':''})
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=v.FIELDS,delimiter='\t');w.writeheader();w.writerows(rows)
    r=replace(case['resource'],snapshot_sha256=v.sha256(p),table_sha256=v.sha256(p))
    normalized=v.load_knowledge(r)
    assert normalized[0]['mapping_status']=='exact_symbol'
    assert normalized[-1]['mapping_status']=='unmappable'


def test_narrower_type_not_equivalent_support(case):
    def edit(x):
        x['candidates']['candidate-A']=x['candidates']['candidate-A'][:1]
        x['candidates']['candidate-A'][0]['relationship']='narrower'
    change(case,'mapping_path',edit)
    assert v.corroborate_marker_annotation(**case)['groups'][0]['corroboration_state']=='insufficient_evidence'


def test_duplicate_marker_rejected(case):
    p=Path(case['markers_path']);p.write_text(p.read_text()+'opaque\tG\t2\n');case['markers_sha256']=v.sha256(p)
    with pytest.raises(ValueError):v.corroborate_marker_annotation(**case)


@pytest.mark.parametrize('mutation', [
    {'status':'candidate','accepted_identity':None},
    {'primary_annotation':'different-label'},
    {'status':'unresolved','primary_annotation':None},
])
def test_inconsistent_or_historical_primary_contract_rejected(case,mutation):
    change(case,'primary_path',lambda x:x['groups'][0].update(mutation))
    change(case,'mapping_path',lambda x:x.update(primary_sha256=case['primary_sha256']))
    with pytest.raises(ValueError):v.corroborate_marker_annotation(**case)


def test_missing_validation_resource_does_not_change_primary(case):
    from dataclasses import replace
    before=Path(case['primary_path']).read_bytes()
    case['resource']=replace(case['resource'],snapshot_path=str(Path(case['primary_path']).parent/'unavailable'))
    with pytest.raises(FileNotFoundError):v.corroborate_marker_annotation(**case)
    assert Path(case['primary_path']).read_bytes()==before
    assert json.loads(before)['groups'][0]['primary_annotation']=='candidate-A'
