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
    monkeypatch.setattr(c,'library',lambda:None)
    monkeypatch.setattr(c,'CellMarkerEvidence',lambda *a:SimpleNamespace(identity={},entry={'files':[]},_check=lambda:None))
    def annotate(b,**kw):
        v=values[b]
        return {'clusters':[annotation(g,'assigned' if g=='A' else 'ambiguous' if g=='1' else 'unresolved')
                            for g in dict.fromkeys(v['labels']) if g!='-1']}
    monkeypatch.setattr(c,'annotate_markers',annotate)
    monkeypatch.setattr(c,'differential_preparer',lambda binding,**kw:lambda child,kwargs:(kwargs,None))
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


def test_branch_operation_override_is_local_and_preserves_owned_marker_contract(controlled):
    f=controlled
    chosen=c.branch_id(None,'B')
    override=dict(branch_id=chosen,operation_id='tl.leiden',tool_name='run_snapatac2',
        function_name='tl.leiden',arguments={'kwargs':{'resolution':.7}})
    c.coordinate(dict(f.args,operation_overrides={'captured':override}),'execution')
    leiden=[r for r in f.calls if r['operation_id']=='tl.leiden']
    assert len(leiden)>1
    assert [r['branch_id'] for r in leiden if r['arguments']['kwargs'].get('resolution')==.7]==[chosen]
    f.calls.clear()
    override=dict(branch_id='initial',operation_id='markers',tool_name='run_scanpy',
        function_name='tl.rank_genes_groups',arguments={'kwargs':{'method':'t-test'}})
    with pytest.raises(ValueError,match='owned|marker'):
        c.coordinate(dict(f.args,operation_overrides={'captured':override}),'execution')


def test_progress_records_actual_branches_and_stopping_outcomes(controlled):
    rows=[];controlled.inv.progress=lambda **kw:rows.append(kw)
    h=c.coordinate(controlled.args,'execution')
    assert rows[-1]['outcome']=='complete' and rows[-1]['pending_branches']==0
    assert all(r.get('pending_branches',0)>=0 for r in rows)
    stops={r['outcome'] for r in rows if r.get('outcome')}
    assert {b['stopping_reason'] for b in h['branches']}<=stops
    assert any(r.get('function_name')=='snapatac2.tl.leiden' for r in rows)
    assert all('total_steps' not in r for r in rows)


def test_hierarchical_evidence_exceeds_generic_metadata_bound_without_losing_checks(tmp_path, monkeypatch):
    import hashlib
    import json
    from agent.tools.analysis import hierarchical_annotation as owner
    from agent.tools.analysis.sequence_prediction import read as generic_read
    path=tmp_path/'annotations.json'
    # This is the demonstrated production boundary, not native scientific work.
    path.write_bytes(b'{"records":"'+b'x'*(64*1024**2)+b'"}')
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='declared bound'): generic_read(path,sha)
    result=owner.read_evidence(path,sha)
    assert len(result['records'])==64*1024**2
    with pytest.raises(ValueError,match='identity mismatch'): owner.read_evidence(path,'0'*64)
    monkeypatch.setattr(owner,'EVIDENCE_MAX_BYTES',128)
    with pytest.raises(ValueError,match='declared bound'): owner.read_evidence(path,sha)
    for raw in [b'{"repeated":1,"repeated":2}',b'{"invalid":NaN}']:
        path.write_bytes(raw)
        with pytest.raises(ValueError):owner.read_evidence(path,hashlib.sha256(raw).hexdigest())
    other=tmp_path/'unrelated.json';other.write_text('{}')
    with pytest.raises(ValueError,match='evidence sidecar'):owner.read_evidence(other,hashlib.sha256(other.read_bytes()).hexdigest())


def test_marker_column_tables_keep_every_value_and_reject_corrupt_rows(tmp_path):
    from copy import deepcopy
    from agent.tools.analysis.sequence_prediction import write
    markers = [dict(feature='repeat', identity='repeat', exclusion_reason=None,
        score=1.125, pvalue=1e-20, pvalue_adj=2e-20, positive=True,
        detection_fractions=dict(pts=[.1,.9],pts_rest=[.2,.4])),
        dict(feature='repeat',identity='repeat',exclusion_reason=None,
        score=-1.125,pvalue=.95,pvalue_adj=1.,positive=False,detection_fractions={})]
    original = {'initial': dict(schema='marker-candidates.v1',provenance={'source_sha256':'a'*64},
        reference={'resource':{'id':'exact'}}, candidates=[{'genes':['repeat']}],
        clusters=[dict(cluster='g',status='assigned',tests=[{'pvalue':1e-9}],markers=markers)])}
    before = deepcopy(original)
    table = p.annotation_tables(original)
    assert original == before
    assert table['rounds']['initial']['schema'] == 'marker-candidates.v2'
    assert table['rounds']['initial']['clusters'][0]['markers']['columns'] == list(p.MARKER_COLUMNS)
    target = tmp_path/'annotations.json'; sha = write(target, table)
    assert p.read_evidence(target, sha) == before
    assert p.annotation_records(before) == before  # Accepted historical bytes.
    for corrupt in ('columns', 'length', 'extra'):
        broken = deepcopy(table); value = broken['rounds']['initial']['clusters'][0]['markers']
        if corrupt == 'columns': value['columns'][0] = 'wrong'
        elif corrupt == 'length': value['rows'][0].pop()
        else: value['unowned'] = True
        with pytest.raises(AuthorityError,match='statistical table'): p.annotation_records(broken)
    original['initial']['clusters'][0]['markers'][0]['new_field'] = 1
    with pytest.raises(AuthorityError,match='statistical table fields'): p.annotation_tables(original)


def test_owner_publication_keeps_original_counts_and_absent_marker_rounds(tmp_path):
    import json
    from agent.tools.analysis.iterative_annotation import branch_id
    from agent.tools._file_identity import file_closure
    original=tmp_path/'original.h5ad'
    obs=pd.DataFrame({'groups':pd.Categorical(['A','A'])},index=['one','two'])
    ad.AnnData(np.array([[1.,2.],[0.,3.]]),obs=obs).write_h5ad(original)
    source=file_closure([original])[0]
    branch=dict(branch_id=branch_id(None,'A'),parent=None,cluster='A',level=1,
        positions=[0,1],n_cells=2,residual_positions=[0,1],annotation=None,
        annotation_round='initial',grouping_binding=None,stopping_reason='maximum_levels')
    hierarchy=dict(completion='complete',source=source,grouping=dict(origin='user_supplied',source=source,groupby='groups'),
        maximum_levels=1,branches=[branch],input_results=[],operations=[],cellmarker={},limitations='controlled fixture')
    result=p.publish(dict(input_path=str(original),group_column='groups',output_dir=str(tmp_path/'published')),
        'test-publication-only',hierarchy,{}, {},resource_files=[])
    assert p.read_evidence(result['annotations_path'],result['annotations_sha256'])=={}
    assert json.loads((tmp_path/'published/hierarchy.json').read_bytes())==hierarchy
    p.verify_h5ad(result['annotated_h5ad_path'],hierarchy)


def test_backgrounds_are_self_contained_exact_and_restore_independent_rounds():
    original={'one':dict(schema='marker-candidates.v1',universe=['A'],tested_features=[dict(feature='A',identity='A')],
        reference={'resource':{'id':'pinned'}},candidates=[dict(genes=['A'])],clusters=[])}
    original['two']=deepcopy(original['one'])
    packed=p.annotation_tables(original)
    assert len(packed['shared_backgrounds'])==1
    assert packed['rounds']['one']['background']==packed['rounds']['two']['background']==0
    restored=p.annotation_records(packed)
    assert restored==original
    restored['one']['candidates'][0]['genes'].clear()
    assert restored['two']['candidates'][0]['genes']==['A']
    original['two']['universe']=['Different']
    assert len(p.annotation_tables(original)['shared_backgrounds'])==2
    packed['rounds']['one']['background']=True
    with pytest.raises(AuthorityError,match='background index'):p.annotation_records(packed)


def test_native_parameter_arrays_and_differential_tables_keep_all_values():
    native=dict(parameters=dict(kwargs={'features':[0,2], 'cell_group1':[0]},
        effective_native_parameters={'features':[0,2],'cell_group1':[0],'min_pct':.05}))
    rows=[{'feature name':'peak','log2(fold_change)':1.25,'p-value':1e-15,'adjusted p-value':None,'native extra':2}]
    decision=dict(parent={'native_result':native},local_steps=[{'native_result':native}],
        children=[dict(differential=dict(native_result=native,rows=rows))])
    original={'branch':decision};packed=p.scientific_evidence(original,'subdivisions.json',encode=True)
    effective=packed['branch']['children'][0]['differential']['native_result']['parameters']['effective_native_parameters']
    assert effective['from_kwargs']==['features','cell_group1']
    assert 'native extra' in packed['branch']['children'][0]['differential']['rows']['columns']
    restored=p.scientific_evidence(packed,'subdivisions.json')
    assert restored==original
    params=restored['branch']['parent']['native_result']['parameters']
    params['kwargs']['features'].append(1)
    assert params['effective_native_parameters']['features']==[0,2]
    native['parameters']['effective_native_parameters']['features']=[0.,2.]
    typed=p._native_parameters(native,encode=True)
    assert 'features' not in typed['parameters']['effective_native_parameters']['from_kwargs']
    assert p._native_parameters(typed,encode=False)['parameters']['effective_native_parameters']['features']==[0.,2.]


def test_compact_evidence_preserves_path_key_privacy_as_well_as_scientific_values():
    import json
    from agent.application.sequence_delivery import metadata_bytes
    native=dict(parameters=dict(kwargs={'features':[0,2], 'reference_path':['private.h5ad']},
        effective_native_parameters={'features':[0,2],'reference_path':['private.h5ad']}))
    row={'feature name':'peak','log2(fold_change)':1.,'p-value':.001,'adjusted p-value':.01,'source_path':'relative-private.tsv'}
    original={'branch':dict(parent={'native_result':native},local_steps=[],children=[dict(differential=dict(native_result=native,rows=[row]))])}
    packed=p.scientific_evidence(original,'subdivisions.json',encode=True)
    delivered=json.loads(metadata_bytes(packed));expected=json.loads(metadata_bytes(original))
    assert p.scientific_evidence(delivered,'subdivisions.json')==expected
    assert 'private.h5ad' not in json.dumps(delivered) and 'relative-private.tsv' not in json.dumps(delivered)
