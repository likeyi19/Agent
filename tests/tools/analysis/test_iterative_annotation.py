"""Controlled coordinator scheduling and hierarchy semantics, not native science."""
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from agent.schemas.prior_output import PriorOutputBinding
from agent.schemas.verification_authority import AuthorityError
from agent.tools._file_identity import file_sha256
from agent.tools.analysis import iterative_annotation as c, hierarchical_annotation as p


def annotation(cluster, status, name='broad'):
    return dict(cluster=cluster, groupby='leiden', status=status,
        preferred=[{'cell_name':name,'species':'Mouse'}] if status=='assigned' else
                  [{'cell_name':n,'species':'Mouse'} for n in ('A','B')] if status=='ambiguous' else [],
        significant=[], reasons=[])


@pytest.fixture
def controlled(tmp_path,monkeypatch):
    obs=pd.DataFrame(index=[f'c{i}' for i in range(9)])
    source=tmp_path/'source.h5ad'
    ad.AnnData(np.ones((9,10)),obs=obs).write_h5ad(source)
    values={}; outputs={}; calls=[]
    def binding(labels, positions, function='snapatac2.tl.leiden', **extra):
        n=len(values)
        b=PriorOutputBinding('r'+str(n),'s','run_'+function.split('.')[0],'dataset','native_matrix_h5ad.v1',
            'a'*64,'b'*64,'c'*64)
        values[b]=dict(labels=labels,positions=positions,function_name=function,matrix_path=str(source),
            matrix_sha256=file_sha256(source),parameters={'gene_activity':{}},**extra)
        return b
    group=binding(['A']*6+['B']*2+['-1'],list(range(9)))
    activity=binding(['A']*6+['B']*2+['-1'],list(range(9)),function='episcanpy.tl.geneactivity')
    splits={(0,1,2,3,4,5):['0','0','1','1','2','2'],(0,1):['0','1'],(2,3):['0','1']}
    def invoke(**kw):
        calls.append(kw)
        if (kw['branch_id'],kw['operation_id']) in outputs: return
        args=kw['arguments']; src=values[args['input_path'].binding]; labels=src['labels']; positions=src['positions']
        function=args['function_name']
        if function=='pp.select_features':
            chosen=args['cell_subset']['groups'][0]
            selected=[i for i,g in enumerate(labels) if g==chosen]
            positions=[positions[i] for i in selected]
            labels=[labels[i] for i in selected]
        if function=='tl.leiden':
            labels=splits.get(tuple(positions),['0']*len(positions))
        if function=='tl.rank_genes_groups':
            src=values[args['grouping_path'].binding]; labels,positions=src['labels'],src['positions']
        b=binding(labels,positions,function=kw['tool_name'].removeprefix('run_')+'.'+function)
        outputs[kw['branch_id'],kw['operation_id']]=b
    inv=SimpleNamespace(evidence_context=lambda:dict(store=None,registry=None),
        input_binding=lambda name:group if name=='grouping_path' else activity,
        invoke=invoke,output=lambda branch,operation,port:outputs[branch,operation],check_cancelled=lambda:None)
    monkeypatch.setattr(c,'annotation_invocation',lambda:inv)
    monkeypatch.setattr(c,'_read',lambda b,**kw:(None,values[b]))
    def groups(native,b):
        o=obs.iloc[native['positions']].copy();o['leiden']=native['labels']
        return 'leiden',o,list(dict.fromkeys(native['labels']))
    monkeypatch.setattr(c,'groups',groups)
    monkeypatch.setattr(c,'accepted_authorities',lambda *a: nullcontext(None))
    monkeypatch.setattr(c,'corresponding_positions',lambda *a:list(range(9)))
    monkeypatch.setattr(c,'_peak_source',lambda *a:True)
    monkeypatch.setattr(c,'library',lambda:None)
    monkeypatch.setattr(c,'CellMarkerEvidence',lambda *a:SimpleNamespace(identity={},entry={'files':[]},_check=lambda:None))
    def annotate(b,**kw):
        v=values[b]
        return {'clusters':[annotation(g,'assigned' if g=='A' else 'ambiguous' if g=='1' else 'unresolved')
                            for g in dict.fromkeys(v['labels']) if g!='-1']}
    monkeypatch.setattr(c,'annotate_markers',annotate)
    monkeypatch.setattr(c,'differential_kwargs',lambda *a,**k:{})
    def assess(parent,local,diffs,**kw):
        old,new=values[parent],values[local[-1]]
        source_positions=[old['positions'].index(i) for i in new['positions']]
        kids=[]
        labels=list(dict.fromkeys(new['labels']))
        retained=[]
        if len(labels)>1:
            for label in labels:
                positions=[source_positions[i] for i,g in enumerate(new['labels']) if g==label]
                keep=label!='2' and tuple(new['positions'])!=(2,3)
                kids.append(dict(cluster=label,source_positions=positions,retained=keep))
                if keep:retained+=positions
        residual=[i for i in source_positions if i not in retained]
        outcome=('no_candidate_subdivision' if len(labels)==1 else 'no_supported_subdivision' if not retained
                 else 'partial_supported_subdivision' if residual else 'supported_subdivision')
        return dict(outcome=outcome,children=kids,parent_membership={'positions':source_positions},residual_parent={'positions':residual})
    monkeypatch.setattr(c,'assess_subdivision',assess)
    monkeypatch.setattr(p,'publish',lambda args,identity,hierarchy,annotations,subdivisions,**kw:hierarchy)
    return SimpleNamespace(args=dict(input_path=str(source),input_sha256=file_sha256(source),grouping_path=str(source),
        activity_path=str(source),cellmarker_id='captured',output_dir=str(tmp_path/'output')),obs=obs,calls=calls,inv=inv)


def test_deterministic_independent_branches_partial_residual_and_default_budget(controlled):
    f=controlled; h=c.coordinate(f.args,'execution')
    assert h['maximum_levels']==5
    rows=h['branches']; roots=[r for r in rows if r['parent'] is None]
    a=roots[0]
    assert a['stopping_reason']=='partial_supported_subdivision' and a['residual_positions']==[4,5]
    assert roots[1]['stopping_reason']=='no_candidate_subdivision'
    assert roots[2]['stopping_reason']=='native_unassigned'
    children=[r for r in rows if r['parent']==a['branch_id']]
    assert children[0]['annotation']['status']=='unresolved'
    assert children[0]['stopping_reason']=='supported_subdivision'  # unnamed still eligible
    assert children[1]['annotation']['status']=='ambiguous'
    assert children[1]['stopping_reason']=='no_supported_subdivision'  # ambiguity never forces recursion
    columns=p.observation_columns(f.obs,h)
    assert columns.annotation_depth.tolist()==[3,3,2,2,1,1,1,1,1]
    assert columns.cell_type_final.iloc[:6].tolist()==['broad']*6
    assert columns.annotation_status.iloc[2]=='ambiguous'
    assert pd.isna(columns.cell_type_level_2.iloc[0])
    assert pd.isna(columns.cell_type_level_3.iloc[2])
    assert columns.annotation_status.iloc[8]=='unassigned'
    assert len({r['branch_id'] for r in rows})==len(rows)
    assert [r['cluster'] for r in rows].count('0')>1
    before=[(r['branch_id'],r['operation_id']) for r in f.calls]
    assert len(before)==len(set(before))
    f.calls.clear(); assert c.coordinate(f.args,'execution')==h
    assert [(r['branch_id'],r['operation_id']) for r in f.calls]==before


@pytest.mark.parametrize('maximum',[1,2,3])
def test_controller_enforces_actual_depth_budget(controlled,maximum):
    f=controlled; h=c.coordinate(dict(f.args,maximum_levels=maximum),'execution')
    assert max(r['level'] for r in h['branches'])==maximum
    assert all(r['level']<=maximum for r in h['branches'])
    assert any(r['stopping_reason']=='maximum_levels' for r in h['branches'])
    p.observation_columns(f.obs,h)


def test_gene_activity_only_retains_current_annotation(controlled,monkeypatch):
    monkeypatch.setattr(c,'_peak_source',lambda *a:None)
    h=c.coordinate(controlled.args,'execution')
    assert [x['operation_id'] for x in controlled.calls]==['markers']
    assert h['branches'][0]['stopping_reason']=='missing_required_peak_space'
    assert p.observation_columns(controlled.obs,h).cell_type_final.iloc[0]=='broad'


def test_native_failure_never_publishes_biological_stop(controlled):
    original=controlled.inv.invoke
    def fail(**kw):
        if kw['operation_id']=='tl.spectral':raise RuntimeError('native failed')
        original(**kw)
    controlled.inv.invoke=fail
    with pytest.raises(RuntimeError,match='native failed'):
        c.coordinate(controlled.args,'execution')
    assert any(r['operation_id']=='markers' for r in controlled.calls)


@pytest.mark.parametrize('status',['assigned','ambiguous','unresolved','insufficient_evidence'])
def test_current_leaf_status_never_fabricates_label(controlled,status):
    h=c.coordinate(dict(controlled.args,maximum_levels=1),'execution')
    h['branches'][0]['annotation']=annotation('A',status)
    obs=p.observation_columns(controlled.obs,h)
    assert obs.annotation_status.iloc[0]==status
    assert (obs.cell_type_final.iloc[0]=='broad') if status=='assigned' else pd.isna(obs.cell_type_final.iloc[0])


@pytest.mark.parametrize('mutation',['duplicate','loss','depth','parent','failed','identity'])
def test_publication_rejects_corrupt_hierarchy(controlled,mutation):
    h=c.coordinate(controlled.args,'execution'); h=deepcopy(h)
    r=h['branches'][0]
    if mutation=='duplicate':r['residual_positions'].append(0)
    elif mutation=='loss':r['residual_positions'].pop()
    elif mutation=='depth':r['level']=6
    elif mutation=='parent':r['parent']='unknown'
    elif mutation=='failed':r['stopping_reason']='native_execution_failure'
    else:r['branch_id']='arbitrary'
    with pytest.raises(AuthorityError):p.observation_columns(controlled.obs,h)


def test_sibling_failure_keeps_prior_branch_native_outcomes(controlled):
    original=controlled.inv.invoke
    failed_branch=c.branch_id(None,'B')
    def fail(**kw):
        if kw['branch_id']==failed_branch and kw['operation_id']=='tl.spectral':
            raise RuntimeError('sibling execution failed')
        return original(**kw)
    controlled.inv.invoke=fail
    with pytest.raises(RuntimeError,match='sibling execution failed'):
        c.coordinate(controlled.args,'execution')
    first=c.branch_id(None,'A')
    completed=controlled.inv.output(first,'markers','dataset')
    assert completed.tool_name=='run_scanpy'
    assert any(row['branch_id']==first and row['operation_id'].startswith('diff-') for row in controlled.calls)


@pytest.mark.parametrize('method,selected,kwargs',[
    ('episcanpy.tl.geneactivity',False,{'upstream':0}),
    ('snapatac2.pp.make_gene_matrix',True,{'upstream':0}),
    ('snapatac2.pp.make_gene_matrix',True,{'use_x':True,'counting_strategy':'fragment'}),
])
def test_constructor_forwards_selected_binding_and_native_parameters(controlled,method,selected,kwargs):
    f=controlled
    args={k:v for k,v in f.args.items() if k!='activity_path'}
    args.update(activity_method=method,activity_kwargs=kwargs,maximum_levels=1)
    if selected:args['activity_source_path']=args['input_path']
    expected=f.inv.input_binding('activity_source_path' if selected else 'grouping_path')
    c.coordinate(args,'execution')
    first=f.calls[0]
    assert first['operation_id']=='activity'
    assert first['arguments']['input_path'].binding==expected
    assert first['arguments']['kwargs']==kwargs
    assert first['tool_name']=='run_'+method.split('.')[0]


@pytest.mark.parametrize('damage', [None, 'parent', 'level', 'count', 'residual', 'stop', 'identity', 'missing', 'extra', 'budget', 'path'])
def test_structured_portable_hierarchy_roundtrip_and_integrity(controlled, tmp_path, damage):
    import json
    import h5py
    from anndata.io import write_elem
    from agent.tools.analysis import peak_h5ad
    f=controlled; h=c.coordinate(f.args,'execution')
    # Complete evidence can be arbitrarily larger than the portable branch table.
    h['execution_evidence']={'private_source':'/private/science/source.h5ad','receipt':'x'*70000}
    assert len(json.dumps(h).encode())>65536
    original=ad.read_h5ad(h['source']['path']); value=original.copy()
    value.obs=pd.concat([original.obs,p.observation_columns(original.obs,h)],axis=1)
    metadata=p.portable_hierarchy(h)
    assert 'execution_evidence' not in metadata
    assert metadata['branches'].index.tolist()==[r['branch_id'] for r in h['branches']]
    for row,(_,branch) in zip(h['branches'],metadata['branches'].iterrows()):
        assert branch.to_dict()==dict(parent_branch_id=row['parent'] or '',level=row['level'],
            n_cells=row['n_cells'],n_residual_cells=len(row['residual_positions']),stopping_reason=row['stopping_reason'])
    path=tmp_path/'portable.h5ad'
    value.uns[p.UNS]=metadata;value.write_h5ad(path)
    peak_h5ad.inspect(path)
    p.verify_h5ad(path,h)
    if damage is None:
        assert len(h['execution_evidence']['receipt'])==70000
        return
    table=metadata['branches']
    if damage in {'parent','level','count','residual','stop'}:
        col={'parent':'parent_branch_id','level':'level','count':'n_cells','residual':'n_residual_cells','stop':'stopping_reason'}[damage]
        table.iloc[0,table.columns.get_loc(col)]='wrong' if damage in {'parent','stop'} else 999
    elif damage=='identity':table.index=['wrong',*table.index[1:]]
    elif damage=='missing':metadata['branches']=table.iloc[1:]
    elif damage=='extra':metadata['private_path']='/private/source.h5ad'
    elif damage=='budget':metadata['maximum_levels']+=1
    else:metadata['evidence_files']['hierarchy']='/private/hierarchy.json'
    with h5py.File(path,'r+') as handle:write_elem(handle['uns'],p.UNS,metadata)
    with pytest.raises(AuthorityError,match='Portable hierarchy'):
        p.verify_h5ad(path,h)
