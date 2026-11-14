"""Numerical/interpretation witnesses, not biological annotation validation."""
from math import comb

import pytest

from agent.tools.analysis.marker_candidates import _interpret, _reference, candidate_evidence
from agent.tools.data.cellmarker import CellMarkerEvidence, COLUMNS
from agent.application.shared_resources import ScientificResourceLibrary
from test_cellmarker import fixture_resource


def candidate(name, genes):
    return dict(candidate=dict(species='Mouse', cell_name=name), genes=genes)


def exact_tail(N, K, n, k):
    return sum(comb(K, j) * comb(N-K, n-j) for j in range(k, min(K, n)+1)
               if 0 <= n-j <= N-K) / comb(N, n)


def independent_bh(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    output = [None] * len(values)
    bound = 1.
    for rank in range(len(values), 0, -1):
        index = order[rank-1]
        bound = min(bound, values[index] * len(values) / rank)
        output[index] = bound
    return output


def test_exact_ora_and_full_bh_family():
    result = _interpret(range(20), range(5), [candidate('hit', range(4)),
        candidate('overlap', range(3)), candidate('zero', range(10, 15)),
        candidate('uncovered', range(21, 25))], .05)
    assert result['testing_family_size'] == 3
    p = [exact_tail(t['N'], t['K'], t['n'], t['k']) for t in result['tests']]
    assert [t['pvalue'] for t in result['tests']] == pytest.approx(p)
    assert [t['pvalue_adj'] for t in result['tests']] == pytest.approx(independent_bh(p))
    assert result['tests'][2]['pvalue'] == 1
    assert result['tests'][0]['pvalue_adj'] > result['tests'][0]['pvalue']
    assert result['status'] == 'assigned'
    assert len(result['significant']) == 2
    assert result['preferred'][0]['cell_name'] == 'hit'
    assert result['untestable_candidates'] == [dict(species='Mouse', cell_name='uncovered')]


@pytest.mark.parametrize('sets,positive,status,preferred', [
    ([('only', [0, 1])], [0, 1], 'assigned', ['only']),
    ([('one', [0, 1]), ('two', [0, 1])], [0, 1], 'ambiguous', ['one', 'two']),
    ([('one', [0, 1]), ('two', [1, 2])], [], 'unresolved', []),
    ([('one', [40, 41])], [0, 1], 'insufficient_evidence', []),
    ([], [0, 1], 'insufficient_evidence', []),
])
def test_outcomes(sets, positive, status, preferred):
    result = _interpret(range(30), positive, [candidate(n, g) for n, g in sets], .05)
    assert result['status'] == status
    assert [r['cell_name'] for r in result['preferred']] == preferred


def test_duplicate_identity_and_zero_draw():
    rows = [candidate('a', ['x', 'x', 'y']), candidate('b', ['z'])]
    result = _interpret(['x', 'x', 'y', 'z', 'unmatched'], ['x', 'x'], rows, .05)
    assert [(r['N'], r['K'], r['n'], r['k']) for r in result['tests']] == [(4,2,1,1),(4,1,1,0)]
    assert all(t['pvalue'] == 1 for t in _interpret(['x', 'y', 'z'], [], rows, .05)['tests'])
    assert _interpret([], [], rows, .05)['status'] == 'insufficient_evidence'


def test_all_sources_exact_context_and_lossless_evidence(tmp_path):
    row = dict.fromkeys(COLUMNS, '')
    row.update(species='Mouse', cell_name='T cell', tissue_type='Blood', disease='Normal',
               marker='Cd3e', symbol='Cd3e', gene_id='12501.0')
    categories = ['Method', 'Single-cell sequencing', 'Experiment', 'Review', 'Company']
    rows = [dict(row, marker_source=c, method_details='COSG' if c == 'Method' else '') for c in categories]
    rows += [rows[0], dict(rows[0], method_details='CelliD'),
             dict(row, species='Human', marker='CD3E', symbol='CD3E'),
             dict(row, cell_name='Other', marker='uncovered', symbol='uncovered')]
    resource = fixture_resource(tmp_path/'reference', rows)
    library = ScientificResourceLibrary(tmp_path/'library')
    entry = library.install('cellmarker', resource, label='synthetic', version='1', provenance='fixture', usage='test')
    lookup = CellMarkerEvidence(library, entry['id'])
    candidates, query = _reference(lookup, ['Cd3e', 'no_match'], 'name', 'Mouse', {})
    cell = next(c for c in candidates if c['candidate']['cell_name'] == 'T cell')
    assert cell['genes'] == ['Cd3e']
    assert cell['evidence']['source_categories'] == sorted(categories)
    assert cell['evidence']['original_rows'] == 7
    assert cell['evidence']['distinct_records'] == 6
    assert cell['evidence']['matched_fields'] == ['marker', 'symbol']
    assert next(c for c in candidates if c['candidate']['cell_name'] == 'Other')['coverage'] == 'no_markers_in_tested_background'
    result = dict(reference=dict(resource=lookup.identity, query=query), candidates=candidates)
    original = [item for page in candidate_evidence(result, cell['candidate'], lookup, page_size=1) for item in page['items']]
    assert sorted(p for item in original for p in item['source_rows']) == list(range(1,8))
    assert original[0]['fields']['pmid'] == ''
    assert original[0]['fields']['tissue_type'] == 'Blood'
    for kind, names, expected in [('gene_id', ['12501.0'], ['12501.0']), ('gene_id', ['12501'], []), ('name', ['cd3e'], [])]:
        selected, _ = _reference(lookup, names, kind, 'Mouse', {'tissue_type':'Blood'})
        assert next(c for c in selected if c['candidate']['cell_name'] == 'T cell')['genes'] == expected
    selected, _ = _reference(lookup, ['Cd3e'], 'name', 'Mouse', {'tissue_type':'missing'})
    assert selected == []
    selected, _ = _reference(lookup, ['Cd3e', 'CD3E'], 'name', None, {})
    assert {c['candidate']['species'] for c in selected} == {'Human', 'Mouse'}


def test_native_evidence_reader_retains_features_and_explicit_identities(tmp_path, monkeypatch):
    """Controlled accepted-reader boundary; genuine acceptance is tested separately."""
    from types import SimpleNamespace
    import anndata as ad
    import numpy as np
    import pandas as pd
    from agent.tools.analysis import marker_candidates as owner
    from agent.schemas.prior_output import PriorOutputBinding
    path = tmp_path/'markers.h5ad'
    a = ad.AnnData(np.ones((4,4)),var=pd.DataFrame({'ids':['1.0','1.0','2.0','']},index=['a','b','c','d']))
    ranked = dict(params=dict(groupby='exact_group',reference='other',use_raw=False),
        names=np.array([('a',),('b',),('c',),('d',)],dtype=[('group','O')]),
        scores=np.array([(5.,),(-2.,),(0.,),(1.,)],dtype=[('group','f8')]),
        pvals=np.array([(.001,),(.01,),(1.,),(.001,)],dtype=[('group','f8')]),
        pvals_adj=np.array([(.004,),(.02,),(1.,),(.004,)],dtype=[('group','f8')]))
    a.uns['custom'] = ranked
    a.write_h5ad(path)
    native = dict(matrix_path=str(path),function_name='scanpy.tl.rank_genes_groups',
        parameters=dict(kwargs=dict(key_added='custom'),effective_native_parameters=owner.marker_kwargs('exact_group'),gene_activity={}))
    monkeypatch.setattr(owner,'validate_binding',lambda *a, **k: SimpleNamespace(result=native))
    binding = PriorOutputBinding('run','step','run_scanpy','dataset','native_matrix_h5ad.v1','0'*64,'1'*64,'2'*64)
    result = owner.annotate_markers(binding,store=None,registry=None,identity_kind='gene_id',identity_column='ids')
    assert result['universe'] == ['1.0','2.0']
    assert result['excluded_features'] == ['d']
    assert result['clusters'][0]['positive_genes'] == ['1.0']
    assert result['clusters'][0]['comparison'] == 'other'
    assert result['clusters'][0]['markers'][0]['detection_fractions'] == {}
    assert result['clusters'][0]['status'] == 'insufficient_evidence'  # no supplied reference
    # Two same-named native ranks cannot identify different explicitly selected IDs.
    a.var_names = ['a','a','c','d']
    a.var['ids'] = ['1.0','3.0','2.0','']
    a.uns['custom']['names']['group'][1] = 'a'
    a.write_h5ad(path)
    result = owner.annotate_markers(binding,store=None,registry=None,identity_column='ids')
    assert result['identity_exclusions']['a'] == 'conflicting identities for the same native feature name'
    assert result['universe'] == ['2.0']
    assert not any('uniquely' in reason for reason in result['clusters'][0]['reasons'])
    a.uns['custom']['names'] = a.uns['custom']['names'][:2]
    for key in ('scores','pvals','pvals_adj'):
        a.uns['custom'][key] = a.uns['custom'][key][:2]
    a.write_h5ad(path)
    result = owner.annotate_markers(binding,store=None,registry=None)
    assert any('full tested' in reason for reason in result['clusters'][0]['reasons'])

    for key in ('scores','pvals','pvals_adj'):
        a.uns['custom'][key]['group'][:] = np.nan
    a.write_h5ad(path)
    result = owner.annotate_markers(binding,store=None,registry=None)
    assert any('finite score' in reason for reason in result['clusters'][0]['reasons'])
    assert result['clusters'][0]['tests'] == []


def reference_fixture(tmp_path, rows):
    resource = fixture_resource(tmp_path/'reference', rows)
    library = ScientificResourceLibrary(tmp_path/'library')
    entry = library.install('cellmarker', resource, label='synthetic', version='1', provenance='fixture', usage='test')
    return CellMarkerEvidence(library, entry['id'])


def source_row(**values):
    row = dict.fromkeys(COLUMNS, '')
    row.update(species='Mouse', cell_name='Target', marker_source='Method', **values)
    return row


@pytest.mark.parametrize('names,positive,symbol,kind,selected', [
    (['Gpr116'], ['Gpr116'], 'Adgrf5', 'name', 'Gpr116'),
    (['Adgrf5'], ['Adgrf5'], 'Adgrf5', 'name', 'Adgrf5'),
    (['Gpr116','Adgrf5'], ['Gpr116','Adgrf5'], 'Adgrf5', 'name', 'Adgrf5'),
    (['Gpr116','Adgrf5'], ['Gpr116'], 'Adgrf5', 'name', 'Adgrf5'),
    (['Gpr116','Adgrf5'], ['Adgrf5'], 'Adgrf5', 'name', 'Adgrf5'),
    (['Gpr116'], ['Gpr116'], '', 'name', 'Gpr116'),
    (['224792.0'], ['224792.0'], 'Adgrf5', 'gene_id', '224792.0'),
])
def test_source_association_selects_one_identity(tmp_path, names, positive, symbol, kind, selected):
    row = source_row(marker='Gpr116', symbol=symbol, gene_id='224792.0')
    lookup = reference_fixture(tmp_path, [row, row, dict(row, method_details='COSG')])
    universe = names + [f'background_{i}' for i in range(20-len(names))]
    candidates, query = _reference(lookup, universe, kind, 'Mouse', {})
    assert len(candidates) == 1 and candidates[0]['genes'] == [selected]
    result = _interpret(universe, positive, candidates, .05)
    test = result['tests'][0]
    expected = (20, 1, len(positive), int(selected in positive))
    assert (test['N'],test['K'],test['n'],test['k']) == expected
    assert test['pvalue'] == pytest.approx(exact_tail(*expected))
    assert test['pvalue_adj'] == test['pvalue']  # one hypothesis
    original = [item for page in candidate_evidence(dict(reference=dict(resource=lookup.identity, query=query),
        candidates=candidates), candidates[0]['candidate'], lookup) for item in page['items']]
    assert sorted(p for item in original for p in item['source_rows']) == [1,2,3]
    assert original[0]['multiplicity'] == 2
    assert {m['value'] for item in original for m in item['matches']} == set(names)
    assert {m['field'] for item in original for m in item['matches']} == set(candidates[0]['evidence']['matched_fields'])
    assert candidates[0]['evidence']['literature_records'] == 0
    assert candidates[0]['evidence']['method_details'] == ['COSG']


def test_partial_literature_and_methods_are_descriptive(tmp_path):
    blank = source_row(marker='Gpr116', symbol='Adgrf5')
    partial = [dict(blank, **{field:value}) for field,value in
               [('pmid','123.0'),('title','Supplied title'),('journal','Supplied journal'),('year','2026.0')]]
    complete = dict(blank, pmid='123.0', title='Supplied title', journal='Supplied journal', year='2026.0', method_details='COSG')
    rows = [blank,blank,*partial,complete,complete,dict(complete,method_details='CelliD')]
    lookup = reference_fixture(tmp_path, rows)
    candidates, query = _reference(lookup, ['Gpr116','Adgrf5'], 'name', 'Mouse', {})
    summary = candidates[0]['evidence']
    assert summary['literature_records'] == 5
    assert summary['method_details'] == ['COSG','CelliD']
    assert summary['original_rows'] == 9 and summary['distinct_records'] == 7
    original = [item for page in candidate_evidence(dict(reference=dict(resource=lookup.identity,query=query),
        candidates=candidates), candidates[0]['candidate'], lookup) for item in page['items']]
    assert original[0]['fields']['pmid'] == original[0]['fields']['method_details'] == ''
    assert [r['fields'] for r in original] == [blank,*partial,complete,dict(complete,method_details='CelliD')]


@pytest.mark.parametrize('ambiguous_names,valid_count', [(1,30), (3,30), (1,1)])
def test_partial_identity_ambiguity_keeps_unaffected_genes(tmp_path, monkeypatch, ambiguous_names, valid_count):
    """Synthetic reader witness; native values and source records stay exact."""
    from types import SimpleNamespace
    from copy import deepcopy
    import hashlib
    import anndata as ad
    import numpy as np
    import pandas as pd
    from agent.tools.analysis import marker_candidates as owner
    from agent.schemas.prior_output import PriorOutputBinding
    valid = [f'gene{i}' for i in range(valid_count)]
    bad = [f'conflict{i}' for i in range(ambiguous_names) for _ in range(2)]
    names = bad + valid + ['duplicate_valid']
    ids = [f'conflicting_id{i}' for i in range(len(bad))] + valid + ['gene0']
    positive_count = min(4,valid_count)
    scores = [7.] * len(bad) + [5.]*positive_count + [-2.]*(valid_count-positive_count) + [5.]
    path = tmp_path/'native.h5ad'
    a = ad.AnnData(np.ones((4,len(names))), var=pd.DataFrame({'ids':ids},index=names))
    ranked = dict(params=dict(groupby='exact',reference='rest',use_raw=False),
        names=np.array([(n,) for n in names],dtype=[('group','O')]),
        scores=np.array([(s,) for s in scores],dtype=[('group','f8')]),
        pvals=np.array([(.001,)]*len(names),dtype=[('group','f8')]),
        pvals_adj=np.array([(.01,)]*len(names),dtype=[('group','f8')]))
    a.uns['rank_genes_groups'] = ranked
    a.write_h5ad(path)
    native = dict(matrix_path=str(path),function_name='scanpy.tl.rank_genes_groups',source_identity=dict(sha256='f'*64),
        parameters=dict(kwargs={},effective_native_parameters=owner.marker_kwargs('exact'),gene_activity={}))
    before_native = deepcopy(native)
    monkeypatch.setattr(owner,'validate_binding',lambda *a,**k: SimpleNamespace(result=native))
    binding = PriorOutputBinding('run','step','run_scanpy','dataset','native_matrix_h5ad.v1','0'*64,'1'*64,'2'*64)
    lookup = reference_fixture(tmp_path,[source_row(marker=g,symbol=g) for g in valid[:4]])
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    result = owner.annotate_markers(binding,store=None,registry=None,evidence=lookup,species='Mouse',identity_column='ids')
    cluster = result['clusters'][0]
    assert cluster['status'] == ('assigned' if valid_count == 30 else 'unresolved')
    assert cluster['reasons'] == []
    assert result['universe'] == sorted(valid)
    assert result['excluded_features'] == bad
    assert set(result['identity_exclusions']) == set(bad)
    assert cluster['positive_genes'] == sorted(valid[:4])
    test = cluster['tests'][0]
    assert (test['N'],test['K'],test['n'],test['k']) == (valid_count,positive_count,positive_count,positive_count)
    assert test['pvalue'] == pytest.approx(1/comb(valid_count,positive_count))
    assert result['provenance']['native_result'] == before_native
    assert result['provenance']['binding'] == binding.to_dict()
    for i,m in enumerate(cluster['markers']):
        assert (m['feature'],m['score'],m['pvalue'],m['pvalue_adj']) == (names[i],scores[i],.001,.01)
        assert (m['exclusion_reason'] is not None) == (names[i] in bad)
        if names[i] in bad: assert m['identity'] is None
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    unavailable = owner.annotate_markers(binding,store=None,registry=None,evidence=lookup,identity_column='missing')
    assert unavailable['universe'] == []
    assert unavailable['clusters'][0]['status'] == 'insufficient_evidence'
    # Finite statistics on excluded names cannot supply usable gene evidence.
    a.uns['rank_genes_groups']['scores']['group'][len(bad):] = np.nan
    a.write_h5ad(path)
    nonfinite = owner.annotate_markers(binding,store=None,registry=None,evidence=lookup,identity_column='ids')
    assert nonfinite['clusters'][0]['status'] == 'insufficient_evidence'
    assert any('finite score' in reason for reason in nonfinite['clusters'][0]['reasons'])
    a.uns['rank_genes_groups']['scores']['group'][:] = scores
    # Complete ranking remains indispensable even when usable identities exist.
    for field in ('names','scores','pvals','pvals_adj'):
        a.uns['rank_genes_groups'][field] = a.uns['rank_genes_groups'][field][:-1]
    a.write_h5ad(path)
    truncated = owner.annotate_markers(binding,store=None,registry=None,evidence=lookup,identity_column='ids')
    assert truncated['clusters'][0]['status'] == 'insufficient_evidence'
    assert any('full tested' in reason for reason in truncated['clusters'][0]['reasons'])
    assert truncated['clusters'][0]['tests'] == []


def test_fraction_index_preserves_duplicate_native_rows_and_nonfinite_values():
    import numpy as np
    import pandas as pd
    from agent.tools.analysis.marker_candidates import _fractions, _number
    frame = pd.DataFrame({'group': [0.1, np.nan, 0.8, 1.]},
                         index=['repeat', 'unique', 'repeat', 'last'])
    ranked = dict(pts=frame, pts_rest=1-frame)
    indexed = _fractions(ranked, 'group')
    for slot, values in ranked.items():
        for feature in dict.fromkeys(frame.index):
            actual = values.loc[feature, 'group']
            expected = ([_number(v) for v in actual.tolist()] if hasattr(actual, 'tolist')
                        and getattr(actual, 'ndim', 0) else _number(actual))
            assert indexed[slot][feature] == expected
    assert indexed['pts']['repeat'] == [0.1, 0.8]
    assert indexed['pts']['unique'] is None
    assert _fractions(ranked, 'absent') == {}


def test_reference_projection_reuses_only_exact_operation_local_queries(tmp_path, monkeypatch):
    from agent.tools.analysis import marker_candidates as owner
    lookup = reference_fixture(tmp_path, [source_row(marker='Gene', symbol='Gene')])
    calls=[];original=owner._reference
    def observed(*args):
        calls.append(args[1:]);return original(*args)
    monkeypatch.setattr(owner,'_reference',observed)
    reference=owner.MarkerReference(lookup)
    universe=['Gene','Other']
    first=reference.read(universe,'name','Mouse',{})
    second=reference.read(universe,'name','Mouse',{})
    assert first==second and len(calls)==1
    first[0][0]['genes'].clear();first[1]['names'].clear()
    assert reference.read(universe,'name','Mouse',{})==second
    assert len(calls)==1
    universe.clear()
    assert reference.read(['Gene','Other'],'name','Mouse',{})==second
    assert len(calls)==1
    universe=['Gene','Other']
    reference.read(['Gene'],'name','Mouse',{})
    reference.read(['Gene'],'gene_id','Mouse',{})
    reference.read(['Gene'],'gene_id',None,{})
    reference.read(['Gene'],'gene_id',None,{'tissue_type':'Blood'})
    assert len(calls)==5
    # A different operation owns a fresh projection, even for the same resource.
    owner.MarkerReference(lookup).read(universe,'name','Mouse',{})
    assert len(calls)==6
    # Resource mutation within a live scope cannot reuse the prepared projection.
    from pathlib import Path
    path=Path(lookup.path)
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='changed'):
        reference.read(universe,'name','Mouse',{})
