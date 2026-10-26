"""Simulated accepted publications, real readers/statistics and scripted R5.

No biological qualification or native analysis is established by these fixtures.
"""
import hashlib
import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from agent.application.dialogue_evidence import DetailRequest
from agent.application import scientific_dialogue as sd
from agent.tools.analysis import categorical as owner
from test_dialogue_evidence import app, accepted, files, forbid_work
from test_scientific_dialogue import Model, question, ask


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publication(app, name='initial', *, source=None, ids=None, labels=None):
    root = app.workspace_root/'fixture'; root.mkdir(exist_ok=True)
    ids = ids or ['a','b','c','d','e']
    if source is None:
        source = root/(name+'-source.h5ad')
        obs = pd.DataFrame(dict(treatment=pd.Categorical(['drug','drug','vehicle',None,'vehicle'],
            categories=['vehicle','drug','unused'])), index=ids)
        ad.AnnData(sparse.eye(5, format='csr'), obs=obs).write_h5ad(source)
    path = root/(name+'-labels.h5ad')
    obs = pd.DataFrame(dict(primary_annotation=pd.Categorical(labels or ['A','A','B','B',None],
        categories=['A','B','unused']), annotation_group=['g1' if i<2 else 'g2' if i<4 else 'g3'
                                                       for i in range(len(ids))]), index=ids)
    ad.AnnData(obs=obs).write_h5ad(path)
    rid = accepted(app, 'annotate_scATAC_cell_types', name=name,
        arguments=dict(source_path=str(source), source_sha256=sha(source)),
        overrides=dict(annotated_h5ad_path=str(path), annotated_h5ad_sha256=sha(path),
            group_summary=[], groups_omitted=0, n_cells=len(ids)))[0]
    return rid, path, source


def detail(app, rid, section='category_counts', field='result:primary_annotation', **kwargs):
    return app.sessions.evidence('session',rid,'output0',fields=(),
        detail=DetailRequest(section, field=field, **kwargs))


def test_general_categorical_semantics_and_pagination():
    series = pd.Series(pd.Categorical(['treated',None,'control','treated'],
        categories=['control','treated','unused'], ordered=True))
    full = owner.counts(series)
    assert full['n_entities'] == 4 and full['n_labeled'] == 3 and full['n_missing'] == 1
    assert full['denominator_labeled'] == 3 and full['denominator_all'] == 4
    assert [r['count'] for r in full['records']] == [1,2,0]
    assert full['records'][1]['proportion_all'] == .5
    assert full['records'][1]['proportion_labeled'] == 2/3
    assert full['ordered_categories'] and full['complete']
    page = owner.counts(series, offset=1, limit=1)
    assert page['records'] == [full['records'][1]] and not page['complete']
    assert page['total_records'] == 3 and page['omitted_records'] == 2
    assert owner.counts(pd.Series(pd.Categorical([None,None],categories=['unused'])))['n_missing'] == 2
    typed = owner.counts(pd.Series(pd.Categorical([1,'1',1],categories=[1,'1',2])))
    assert [(r['category'],r['count']) for r in typed['records']] == [(1,2),('1',1),(2,0)]
    assert [r['count'] for r in owner.counts(pd.Series(pd.Categorical([True,False,True])))['records']] == [1,2]
    with pytest.raises(ValueError): owner.counts(pd.Series([1.5,2.0]))


def test_contingency_denominators_missingness_and_order():
    ids = ('a','b','c','d','e')
    left = pd.Series(pd.Categorical(['A','A','B','B',None],categories=['A','B','unused']))
    right = pd.Series(['drug','drug','vehicle',None,'vehicle'])
    value = owner.association(ids,left,ids,right)
    assert value['n_paired'] == 3 and value['n_left_missing'] == value['n_right_missing'] == 1
    assert sum(r['count'] for r in value['records']) == 3
    assert [(r['left'],r['right'],r['count']) for r in value['records']] == [('A','drug',2),('B','vehicle',1)]
    assert all(r['count'] <= r['row_total'] and r['count'] <= r['column_total'] for r in value['records'])
    assert value['records'][0]['proportion_paired'] == 2/3
    assert value['left_categories']['records'][-1]['count'] == 0
    assert not owner.association(ids,left,ids,right,limit=1)['complete']
    for wrong in (tuple(reversed(ids)), ('a','b','c','d','different')):
        with pytest.raises(ValueError,match='identities'): owner.association(ids,left,wrong,right)


def test_read_only_source_distribution_and_derived_attribution(app, monkeypatch):
    rid,path,source = publication(app)
    before = files(app)
    guarded = forbid_work(app,monkeypatch)
    monkeypatch.setattr(ad,'read_h5ad',lambda *a,**k: pytest.fail('Reader loaded whole AnnData'))
    view = detail(guarded,rid,field='source:treatment')
    assert view.status == 'available',view
    fact = view.detail[0]
    assert [r['count'] for r in fact.value['records']] == [2,2,0]
    assert fact.derivation['read_version'] == owner.READ_VERSION
    assert fact.derivation['sources'][0]['artifact_sha256'] == sha(source)
    assert fact.artifact_sha256 == sha(source)
    assert 'supplied_source' in fact.value['fields'][0]['role']
    assert files(app) == before


@pytest.mark.parametrize('failure',['different_source','reordered_axis','changed_source','missing_source'])
def test_association_requires_actual_source_and_axis(app, failure):
    first,path,source = publication(app)
    other_source = source
    ids = None
    if failure == 'different_source':
        other_source = source.parent/'different.h5ad'
        data = ad.read_h5ad(source); data.X.data[0] += 1; data.write_h5ad(other_source)
    if failure == 'reordered_axis': ids = ['b','a','c','d','e']
    second,_,_ = publication(app,'second',source=other_source,ids=ids)
    state=app.sessions.load('session');output=state.revisions[-1].outputs[0]
    peer=dict(revision_id=second,output_name='output0',accepted_step_sha256=output.accepted_step_sha256)
    if failure == 'changed_source': source.write_bytes(source.read_bytes()+b'changed')
    if failure == 'missing_source': source.unlink()
    view=detail(app,first,'category_association',other_field=(
        'source:treatment' if failure in ('changed_source','missing_source') else 'result:primary_annotation'), comparison=peer)
    assert view.status == 'unavailable' and not view.detail
    if failure=='different_source': assert view.reason=='categorical_source_mismatch'
    if failure=='reordered_axis': assert view.reason=='categorical_axis_mismatch'
    assert detail(app,first).status == 'available'  # Independent counts survive.


def test_two_accepted_revisions_and_same_source_field_association(app, monkeypatch):
    first,_,source=publication(app)
    second,_,_=publication(app,'second',source=source,labels=['B','A','B','B',None])
    output=app.sessions.load('session').revisions[-1].outputs[0]
    peer=dict(revision_id=second,output_name='output0',accepted_step_sha256=output.accepted_step_sha256)
    guarded=forbid_work(app,monkeypatch);before=files(app)
    pair=detail(guarded,first,'category_association',other_field='result:primary_annotation',comparison=peer)
    assert pair.status=='available',pair
    value=pair.detail[0].value
    assert value['n_paired']==4 and sum(r['count'] for r in value['records'])==4
    cross=detail(guarded,first,'category_association',other_field='source:treatment')
    assert cross.status=='available' and cross.detail[0].value['n_paired']==3
    assert files(app)==before


def selected(section='category_counts',field='source:treatment',other_field=None,limit=8,offset=0):
    return dict(section=section,field=field,other_field=other_field,limit=limit,offset=offset)


def test_field_selection_structured_table_and_r5_review(app, monkeypatch):
    rid,path,source=publication(app)
    before=files(app)
    def decision(payload):
        rows=payload['dialogue']['result_groups']
        assert len(rows)==1
        assert 'source:treatment' in {f['field'] for f in rows[0]['categorical']['fields']}
        return dict(question(rows[0]['handle']),detail=selected())
    def reply(payload):
        c=next(c for c in payload['evidence']['claims'] if c['field']=='detail.category_counts')
        assert c['value']['n_missing']==1 and c['value']['complete']
        assert c['value']['records'][0]['category']=='vehicle'
        assert 'source' not in c
        assert str(app.workspace_root) not in json.dumps(payload)
        return dict(support='supported',paragraphs=[dict(kind='finding',text='The supplied treatment labels include two vehicle observations among five entities.',citations=[c['claim_id']],comparisons=[])])
    model=Model(decision,response=reply)
    result=ask(forbid_work(app,monkeypatch),'counts','How many observations have each treatment label?',model)
    assert result.status=='answered',result
    assert result.scientific.evidence_scope=='accepted_summary_and_on_demand_facts'
    assert result.scientific.claims[0].source['derivation']['sources'][0]['artifact_sha256']==sha(source)
    assert all(files(app)[p]==v for p,v in before.items() if not p.startswith('sessions/'))
    assert app.sessions.load('session').generation==1


def test_r5_qualified_association_and_independent_fallback(app, monkeypatch):
    first,_,source=publication(app)
    second,_,_=publication(app,'second',source=source)
    def decision(payload):
        groups=payload['dialogue']['result_groups']
        return dict(question(groups[0]['handle'], comparison=dict(output=groups[1]['handle'],subject=None)),
            detail=selected('category_association','result:primary_annotation','source:treatment'))
    def reply(payload):
        c=next(c for c in payload['evidence']['claims'] if c['field']=='detail.category_association')
        return dict(support='supported',paragraphs=[dict(kind='finding',text='The descriptive table covers three jointly labeled entities; supplied treatment labels are not biological validation.',citations=[c['claim_id']],comparisons=[])])
    model=Model(decision,response=reply)
    result=ask(forbid_work(app,monkeypatch),'association','Compare the computational assignments with the supplied treatment groups.',model)
    assert result.status=='answered',result
    assert result.scientific.claims[0].field=='detail.category_association'
    assert model.calls[-1]['evidence']['comparison']['mode']=='source_qualified_categorical_association'
    assert 'not independent biological validation' in model.calls[-1]['evidence']['meanings']['categorical_reading']


def test_general_structured_summary_is_not_lost(app):
    accepted(app,'inspect_scATAC',extra_facts={'category_summary':[dict(category='exposed',count=4,denominator=5)]})
    result=ask(app,'structured','What does the recorded category summary contain?',Model(question(),['category_summary']))
    assert result.status=='answered' and result.scientific.claims[0].value[0]['count']==4


def test_independent_sources_keep_counts_without_correspondence(app, monkeypatch):
    publication(app)
    publication(app,'second')  # Different H5AD byte identity despite equal axes.
    # Make the source identities genuinely different, not just filenames.
    def decision(payload):
        groups=payload['dialogue']['result_groups']
        return dict(question(groups[0]['handle'],comparison=dict(output=groups[1]['handle'],subject=None)),
            detail=selected('category_association','result:primary_annotation','result:primary_annotation'))
    # Use a third accepted source with a different experimental observation.
    root=app.workspace_root/'fixture';source=root/'third-source.h5ad'
    data=ad.read_h5ad(root/'initial-source.h5ad');data.obs['treatment']=pd.Categorical(['other']*5)
    data.write_h5ad(source)
    publication(app,'third',source=source)
    def reply(payload):
        claims=[c for c in payload['evidence']['claims'] if c['field']=='detail.category_counts']
        assert {c['target'] for c in claims}=={'t0','t1'}
        return dict(support='supported',paragraphs=[dict(kind='finding',text='Each result has five entities and one missing assignment. These are independently attributed facts; cell correspondence is unavailable.',citations=[c['claim_id'] for c in claims],comparisons=[])])
    model=Model(decision,response=reply)
    result=ask(forbid_work(app,monkeypatch),'independent','Describe both results; compare only if their entities correspond.',model)
    # The newest and parent result are explicitly selected by the scripted model.
    assert result.status=='answered',result
    assert {c.target for c in result.scientific.claims}=={'t0','t1'}
    assert all(c.field=='detail.category_counts' for c in result.scientific.claims)
    assert any('correspondence is unavailable' in x for x in result.scientific.limitations)


def test_unoffered_field_fails_binding_and_invalid_requests_reject(app):
    publication(app)
    result=ask(app,'bad','Describe an unknown field.',Model(dict(question('g0'),detail=selected(field='source:invented'))))
    assert result.status == 'failed' and result.error is not None
    for kwargs in (dict(section='category_counts'), dict(section='category_counts',field='x',limit=33),
                   dict(section='category_counts',field='x',other_field='y'),
                   dict(section='category_association',field='x',other_field='y',offset=-1)):
        with pytest.raises(ValueError): DetailRequest(**kwargs)


def test_old_cluster_can_supply_counts_without_original_source(app):
    root=app.workspace_root/'fixture';root.mkdir()
    path=root/'assignments.h5ad'
    ad.AnnData(obs=pd.DataFrame(dict(leiden=pd.Categorical(['west','east','west'])),index=['x','y','z'])).write_h5ad(path)
    rid=accepted(app,'cluster_cells',overrides=dict(analysis_path=str(path),analysis_sha256=sha(path),n_cells=3))[0]
    view=detail(app,rid,field='result:leiden')
    assert view.status=='available',view
    assert [r['count'] for r in view.detail[0].value['records']]==[1,2]
    assert view.detail[0].derivation['sources'][0]['source_sha256'] is None


def test_selected_observation_rejects_indirect_storage_and_bounds(app):
    import h5py
    rid,path,source=publication(app)
    # New simulated accepted identity deliberately contains an indirect field.
    with h5py.File(path,'a') as f:
        del f['obs/primary_annotation']
        f['obs/primary_annotation']=h5py.ExternalLink(str(source),'/obs/treatment')
    with pytest.raises(ValueError,match='Indirect'):
        owner.observations(path,sha(path),'primary_annotation')


def test_native_legacy_anndata_categories_are_read_without_rewriting(tmp_path):
    import h5py
    path=tmp_path/'legacy.h5ad'
    # Simulated AnnData dataframe 0.1 encoding, handled by AnnData's reader.
    with h5py.File(path,'w') as f:
        obs=f.create_group('obs')
        obs.attrs.update({'encoding-type':'dataframe','encoding-version':'0.1.0',
                          '_index':'cells','column-order':np.array(['condition'],dtype=h5py.string_dtype())})
        obs.create_dataset('cells',data=np.array(['x','y','z'],dtype=h5py.string_dtype()))
        levels=obs.create_dataset('__categories/condition',data=np.array(['mock','treated','unused'],dtype=h5py.string_dtype()))
        col=obs.create_dataset('condition',data=[1,-1,0]);col.attrs['categories']=levels.ref
    before=path.read_bytes()
    assert owner.fields(path)==['condition']
    ids,values=owner.observations(path,sha(path),'condition')
    result=owner.counts(values)
    assert ids==('x','y','z')
    assert [r['count'] for r in result['records']]==[1,1,0] and result['n_missing']==1
    assert path.read_bytes()==before


def test_bounded_pages_continue_through_captured_selection(app):
    publication(app)
    first=ask(app,'page-one','Show the first treatment category.',Model(
        dict(question('g0'),detail=selected(limit=1)),['detail.category_counts']))
    assert first.status=='answered',first
    def decision(payload):
        previous=payload['dialogue']['predecessor']['detail']
        assert previous['offset']==0 and previous['limit']==1
        return dict(question('@focus',focus='continue'),detail=dict(previous,offset=1))
    second=ask(app,'page-two','Show the next category.',Model(decision,['detail.category_counts']))
    assert second.status=='answered',second
    fact=second.scientific.claims[0]
    assert fact.value['records'][0]['category']=='drug' and not fact.value['complete']
    assert fact.value['n_entities']==5 and fact.value['offset']==1


@pytest.mark.parametrize('change',['undeclared','different_path'])
def test_recorded_dependency_checks_are_pure_and_exact(monkeypatch, change):
    from types import SimpleNamespace as S
    from agent.application import result_reading as reader
    from agent.schemas import StepOutputRef
    source=S(result={'analysis_path':'/exact/source.h5ad'})
    stored=S(resolved_arguments={'analysis_path':'/exact/source.h5ad'})
    planned=S(arguments={'analysis_path':StepOutputRef('producer','analysis_path')},depends_on=('producer',))
    if change=='undeclared': planned.depends_on=()
    else: stored.resolved_arguments['analysis_path']='/different/source.h5ad'
    def accepted_step(store,run,step):
        return (source,None,S(lifecycle_status=S(value='SUCCEEDED'))) if step=='producer' else (stored,planned,None)
    monkeypatch.setattr(reader,'_accepted_step',accepted_step)
    with pytest.raises(ValueError): reader.dependency(None,'run','consumer','analysis_path')


def test_derived_table_does_not_bypass_r5_consistency_review(app):
    publication(app)
    class RejectingReview(Model):
        reviews=0
        def complete(self, *, prompt, response_schema):
            value=json.loads(prompt)
            if 'grounding_review_version' in value:
                self.reviews+=1
                return json.dumps(dict(integrity='intact', verdicts={row['id']:dict(verdict='unsupported', independent=True) for row in value['paragraphs']}))
            return super().complete(prompt=prompt,response_schema=response_schema)
    model=RejectingReview(dict(question('g0'),detail=selected()),['detail.category_counts'])
    result=ask(app,'review','Describe the supplied treatment distribution.',model)
    assert result.status=='unavailable' and result.scientific is None and model.reviews==1


@pytest.mark.parametrize('shared_graph',[True,False])
def test_existing_qualified_graph_population_does_not_require_original_counts(app,monkeypatch,shared_graph):
    from agent.application import dialogue_evidence as de
    from agent.application.session_state import digest
    # The pre-existing helper owns graph qualification. This bounded contract
    # test supplies its reviewed fact; it does not manufacture native authority.
    def graph_facts(sessions,run,step,planned):
        identity='a'*64 if shared_graph or run.run_id=='first:run' else 'b'*64
        return (de.EvidenceFact('clustering_neighbors_identity_sha256','available',identity,
            artifact_name='accepted_run_result',artifact_sha256=digest(run.to_dict())),)
    monkeypatch.setattr(de,'_clustering_neighbors_facts',graph_facts)
    root=app.workspace_root/'fixture';root.mkdir()
    revisions=[]
    for name,values in (('first',['left','right','left']),('second',['right','left','left'])):
        path=root/(name+'.h5ad')
        ad.AnnData(obs=pd.DataFrame(dict(leiden=pd.Categorical(values)),index=['x','y','z'])).write_h5ad(path)
        revisions.append(accepted(app,'cluster_cells',name=name,overrides=dict(
            analysis_path=str(path),analysis_sha256=sha(path),n_cells=3))[0])
    output=app.sessions.load('session').revisions[-1].outputs[0]
    peer=dict(revision_id=revisions[-1],output_name='output0',accepted_step_sha256=output.accepted_step_sha256)
    view=detail(app,revisions[0],'category_association',field='result:leiden',other_field='result:leiden',comparison=peer)
    assert view.status==('available' if shared_graph else 'unavailable'),view
    if shared_graph:
        assert view.detail[0].value['n_paired']==3
        assert all(s['source_sha256'] is None for s in view.detail[0].derivation['sources'])


def test_later_fields_are_paged_and_exact_user_selection_is_not_first_page_limited(app,monkeypatch):
    source=app.workspace_root/'many-fields.h5ad'
    obs=pd.DataFrame({f'condition_{i:02d}':pd.Categorical(['on','off',None,'on','off']) for i in range(40)},
                     index=['a','b','c','d','e'])
    obs['measurement']=np.arange(5,dtype=float)
    obs['credentials']=pd.Categorical(['private']*5)
    ad.AnnData(sparse.eye(5,format='csr'),obs=obs).write_h5ad(source)
    rid,_,_=publication(app,source=source)
    guarded=forbid_work(app,monkeypatch)
    first=detail(guarded,rid,'categorical_fields',field=None,limit=32).detail[0].value
    assert first['returned_fields']==32 and first['next_offset']==32 and first['total_fields']==42
    assert 'source:condition_39' not in {row['field'] for row in first['fields']}
    page=detail(guarded,rid,'categorical_fields',field=None,offset=32,limit=32).detail[0].value
    assert page['offset']==32 and page['returned_fields']==10 and page['next_offset'] is None
    assert 'source:condition_39' in {row['field'] for row in page['fields']}
    before=files(app)
    def reply(payload):
        claims=[c for c in payload['evidence']['claims'] if c['field'].startswith('detail.categor')]
        assert len(claims)==1
        return dict(support='supported',paragraphs=[dict(kind='finding',text='The requested bounded categorical information is available with its own coverage.',citations=[claims[0]['claim_id']],comparisons=[])])
    direct=ask(guarded,'named-later','Count the source field condition_39.',Model(
        dict(question('g0'),detail=selected(field='source:condition_39')),response=reply))
    assert direct.status=='answered',direct
    assert direct.scientific.claims[0].value['n_entities']==5
    listing=ask(guarded,'field-page','Show the next page of available categorical field names.',Model(
        dict(question('g0'),detail=selected('categorical_fields',field=None,offset=32,limit=32)),response=reply))
    assert listing.status=='answered',listing
    assert listing.scientific.claims[0].value['offset']==32
    def from_page(payload):
        inventory=payload['dialogue']['result_groups'][0]['categorical']
        assert inventory['offset']==32 and 'source:condition_38' in {r['field'] for r in inventory['fields']}
        return dict(question('g0'),detail=selected(field='source:condition_38'))
    follow=ask(guarded,'from-page','Count condition_38 from that page.',Model(from_page,response=reply))
    assert follow.status=='answered',follow
    for i,field in enumerate(('source:invented','source:measurement','source:credentials','source:/X')):
        denied=ask(guarded,f'invalid-field-{i}',f'Count {field}.',Model(
            dict(question('g0'),detail=selected(field=field)),response=reply))
        assert denied.status == 'failed' and denied.scientific is None
        assert denied.error.code == 'INTERACTIVE_INTENT_BINDING_INVALID'
    assert all(files(app)[p]==v for p,v in before.items() if not p.startswith('sessions/'))


@pytest.mark.parametrize('missing_peer',[False,True])
def test_direct_independent_counts_bind_each_field_to_its_result(app,monkeypatch,missing_peer):
    first,_,_=publication(app)
    source=app.workspace_root/'independent-cohort.h5ad'
    ids=['w','x','y','z']
    ad.AnnData(sparse.eye(4,format='csr'),obs=pd.DataFrame(dict(
        cohort=pd.Categorical(['C','C','C',None])),index=ids)).write_h5ad(source)
    second,_,_=publication(app,'independent',source=source,ids=ids,labels=['A','B','B','A'])
    guarded=forbid_work(app,monkeypatch)
    monkeypatch.setattr(owner,'association',lambda *a,**k:pytest.fail('Independent distributions attempted association'))
    if missing_peer:
        from agent.application.sessions import AnalysisSessions
        original=AnalysisSessions.evidence
        def read(self,*args,**kwargs):
            request=kwargs.get('detail')
            if request is not None and request.section=='category_counts' and request.field=='source:cohort':
                source.unlink(missing_ok=True)  # Lost after discovery/admission, before selected reading.
            return original(self,*args,**kwargs)
        monkeypatch.setattr(AnalysisSessions,'evidence',read)
    def decision(payload):
        groups=payload['dialogue']['result_groups']
        def result(field):
            matches=[g['handle'] for g in groups if field in {r['field'] for r in g['categorical']['fields']}]
            assert len(matches)==1
            return matches[0]
        return dict(question(result('source:treatment'),comparison=dict(output=result('source:cohort'),subject=None)),
            detail=selected('category_counts','source:treatment','source:cohort'))
    def reply(payload):
        evidence=payload['evidence']
        assert evidence['comparison']['mode']=='independent_accepted_facts'
        claims=[c for c in evidence['claims'] if c['field']=='detail.category_counts']
        by_field={c['value']['fields'][0]['field']:c for c in claims}
        assert by_field['treatment']['value']['denominator_all']==5
        if missing_peer:
            assert set(by_field)=={'treatment'}
            assert any('requested detail is unavailable' in text for text in evidence['limitations'])
        else:
            assert set(by_field)=={'treatment','cohort'}
            assert by_field['cohort']['value']['denominator_all']==4
            assert by_field['cohort']['value']['records'][0]['count']==3
        return dict(support='insufficient_evidence' if missing_peer else 'supported',paragraphs=[dict(
            kind='finding',text='These distributions are independently attributed and establish no cell correspondence.',
            citations=[c['claim_id'] for c in claims],comparisons=[])])
    before=files(app)
    result=ask(guarded,'direct-counts','Show treatment and cohort distributions for the two independent datasets separately.',
        Model(decision,response=reply))
    assert result.status=='answered',result
    assert all(c.field=='detail.category_counts' for c in result.scientific.claims)
    assert {c.source['revision_id'] for c in result.scientific.claims}==({first} if missing_peer else {first,second})
    if not missing_peer:
        assert len({c.source['derivation']['sources'][0]['artifact_sha256'] for c in result.scientific.claims})==2
    assert app.sessions.load('session').generation==2
    from agent.application.turn_decisions import ScientificQuestion,ScientificTarget
    with pytest.raises(ValueError,match='second accepted target'):
        ScientificQuestion(ScientificTarget('g0'),detail=selected('category_counts','source:treatment','source:cohort'))
    for p,value in before.items():
        if not p.startswith('sessions/') and str(app.workspace_root/p)!=str(source):
            assert files(app)[p]==value
