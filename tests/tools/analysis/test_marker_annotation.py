import copy
import hashlib
from pathlib import Path

import pytest
from agent.tools.analysis import marker_annotation as m


def score(markers, signatures=None):
    return m.score_candidates(['c2','c1'], [('c2','opaque/a'),('c1','other')], ['G','H'],
                              markers, signatures or {'A':['G','H'], 'B':['H','MISSING']})


@pytest.mark.parametrize('rows', [[('c1','x'),('c2','y')],[('c2','x')],[('c2','x'),('c2','y')],
                                  [('c2','x'),('unknown','y')],[('c2',''),('c1','y')]])
def test_exact_mapping(rows):
    with pytest.raises(ValueError): m.exact_groups(['c2','c1'],rows)


@pytest.mark.parametrize('markers,status,candidate', [([], 'insufficient_evidence',None),
    ([('opaque/a','G',0)],'unresolved',None), ([('opaque/a','G',-1)],'unresolved',None),
    ([('opaque/a','H',1)],'ambiguous',None),([('opaque/a','G',.00000001)],'assigned','A')])
def test_states(markers,status,candidate):
    result=score(markers)
    assert result['groups'][0]['status']==status
    assert result['groups'][0]['candidate']==candidate
    assert result['groups'][0]['primary_annotation']==candidate
    assert result['cells'][0]['primary_annotation']==candidate
    assert all('accepted_identity' not in r for r in result['groups']+result['cells'])
    assert result['groups'][0]['validation_state']=='not_assessed'
    assert [r['cell_id'] for r in result['cells']]==['c2','c1']
    assert result==score(markers)


def test_coverage_duplicates_and_competition():
    result=score([('opaque/a','G',2),('opaque/a','H',-1)], {'A':['G','G','H'], 'B':['H','MISSING']})
    assert result['signature_coverage'][0]['duplicate_entries']==1
    assert result['signature_coverage'][1]['missing_genes']==['MISSING']
    assert result['candidate_evidence'][0]['score']==3/m.math.log2(3)
    assert result['candidate_evidence'][1]['negative']==['H']
    assert 'candidate_evidence' not in result['cells'][0]


@pytest.mark.parametrize('markers', [[('unknown','G',1)], [('opaque/a','unknown',1)],
    [('opaque/a','G',1),('opaque/a','G',2)], [('opaque/a','G',float('nan'))]])
def test_invalid_markers(markers):
    with pytest.raises(ValueError): score(markers)


def test_resource_identity(tmp_path):
    p=tmp_path/'signature.tsv';p.write_text('candidate\tgene\nA\tg\nA\th\n')
    r=m.Resource(str(p),m.sha256(p),'test','r1','human','hg38','explicit test','symbol','uppercase','candidate-gene-list.v1')
    assert r.check('human','hg38','candidate-gene-list.v1','uppercase')==p
    assert m.read_signatures(p)=={'A':['G','H']}
    with pytest.raises(ValueError):r.check('mouse','mm10','candidate-gene-list.v1','uppercase')
    p.write_text('changed')
    with pytest.raises(ValueError):r.check('human','hg38','candidate-gene-list.v1','uppercase')


def test_case_collision_and_denominator():
    with pytest.raises(ValueError):m.score_candidates(['a'],[('a','x')],['G','g'],[],{'A':['G','H']})
    with pytest.raises(ValueError):score([],{'A':['G']})


@pytest.fixture
def canonical(tmp_path):
    from dataclasses import asdict
    import anndata as ad
    import pandas as pd
    import numpy as np
    from scipy.sparse import csr_matrix
    from agent.tools.data.scatac_selection_profile import encode_cell_id
    c=m.matrix_contract
    cells=[encode_cell_id('lib',str(i)) for i in range(6)]
    features=['chr1:10490-10510','chr1:20490-20510']
    x=csr_matrix([[2,0]]*3+[[0,5]]*3,dtype=np.int64)
    a=ad.AnnData(x, obs=pd.DataFrame({'namespace':['lib']*6,'barcode_identifier':[str(i) for i in range(6)],'matrix_row_index':range(6)},index=cells),var=pd.DataFrame(index=features))
    p=tmp_path/'matrix.h5ad';a.write_h5ad(p)
    v=dict(artifact_type=c.ARTIFACT,schema_version=1,contract_version=c.CONTRACT, profile=asdict(c.PROFILE),profile_sha256=c.PROFILE_SHA256,species='human',assembly='hg38',
       upstream={k:dict(manifest_path='/explicit/'+k+'.json',manifest_sha256='a'*64,identity_sha256='b'*64,contract_version=t) for k,t in zip(('fragments','selection','reference'),('scatac-fragments.v2','scatac-cell-selection.v1','scatac-reference-bundle.v1'))},
       ordered_selected_sha256=c.selected_axis_identity([(i,'lib',str(i),v) for i,v in enumerate(cells)])[1],ordered_feature_sha256=m.ordered_identity_sha256(features),shape=[6,2],**c.csr_identity(x),
       matrix=dict(path='matrix.h5ad',sha256=m.sha256(p),size_bytes=p.stat().st_size,format='csr',dtype='int64'),backend=dict(profile_id='bedtools-ccre-record-incidence.v1',runtime_sha256='f'*64),diagnostic=c.overlap_diagnostic(21,21),readiness='matrix_available')
    v['identity_sha256']=c.manifest_identity(v)
    manifest=tmp_path/'manifest.json';manifest.write_bytes(c.canonical(v))
    b=m.MatrixInput(str(manifest),m.sha256(manifest),m.sha256(p),m.ordered_identity_sha256(cells),v['ordered_feature_sha256'],'human','hg38','canonical-fragment-record-overlap-counts.v1')
    return b,a


def test_matrix_binding_and_immutability(canonical):
    b,a=canonical
    loaded,path=m.load_matrix(b)
    assert (loaded.X!=a.X).nnz==0
    assert m.sha256(path)==b.matrix_sha256
    from dataclasses import replace
    for field,value in [('matrix_sha256','a'*64),('ordered_cells_sha256','a'*64),('ordered_features_sha256','a'*64),('assembly','mm10'),('value_semantics','binary_accessibility'),('value_semantics','insertion_counts'),('value_semantics','fragment_counts')]:
        with pytest.raises(ValueError):m.load_matrix(replace(b,**{field:value}))


def test_private_rp_immutable(canonical,tmp_path,monkeypatch):
    import numpy as np
    b,a=canonical
    source=tmp_path/'source.py'
    source.write_text("def calculate_RP_score(x,features,cells,annotation,decay,out,method):\n assert (x.data==1).all()\n assert decay==10000 and method=='Enhanced'\n write_10X_h5(out,x,['G','H'],cells)\n")
    monkeypatch.setitem(m.SOURCES,'MAESTRO/scATAC_Genescore.py',m.sha256(source))
    before=a.X.copy()
    rp,genes=m.enhanced_rp(a.X,a.var_names.tolist(),a.obs_names.tolist(),tmp_path/'unused',source)
    assert (a.X!=before).nnz==0 and (rp.data==1).all()
    assert m.sha256(Path(b.manifest_path).parent/'matrix.h5ad')==b.matrix_sha256


def test_sparse_assignment_needs_no_validator():
    result=score([('opaque/a','G',1e-12)])
    group=result['groups'][0]
    assert group['status']=='assigned' and group['primary_annotation']=='A'
    assert group['validation_state']=='not_assessed'
    assert result['candidate_evidence'][0]['matched_entries']==1
    assert result['candidate_evidence'][0]['positive']==['G']
    assert result['cells'][0]['primary_annotation']=='A'
    assert result['cells'][1]['primary_annotation'] is None
