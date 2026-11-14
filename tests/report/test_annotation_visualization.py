"""Read-only projections of source-retained coordinates; no scientific fitting."""
from copy import deepcopy
import hashlib
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd

from agent.report import annotation_visualization as owner


def test_native_transformed_representation_keeps_cell_origin_separate_from_counts():
    from agent.report.visualization import _cell_results
    from agent.schemas import StepOutputRef
    original='a'*64
    steps=[];planned=[]
    methods=['pp.tfidf','tl.lsi','pp.neighbors','tl.umap']
    for i,method in enumerate(methods):
        sid='s'+str(i);parent='s'+str(i-1)
        args={'input_path':'/original.h5ad' if i==0 else StepOutputRef(parent,'matrix_path')}
        source=original if i==0 else str(i-1)*64
        native=dict(contract_version='native-episcanpy-matrix.v1',function_name='episcanpy.'+method,
            matrix_path='/'+sid+'.h5ad',matrix_sha256=str(i)*64,parameters={'kwargs':{}},
            source_identity={'sha256':source},result_slots=['obsm.X_umap'] if method=='tl.umap' else [])
        steps.append(SimpleNamespace(step_id=sid,tool_name='run_episcanpy',result=native,
            resolved_arguments={'input_path':'/original.h5ad' if i==0 else '/'+parent+'.h5ad'}))
        planned.append(SimpleNamespace(step_id=sid,arguments=args,depends_on=() if i==0 else (parent,)))
    run=SimpleNamespace(steps=steps,plan=SimpleNamespace(steps=planned))
    views=_cell_results(run)
    assert views['s3']['cell_source_sha256']==original
    assert views['s3']['source_sha256']=='1'*64  # Transformed LSI counts are not original counts.
    steps[1].result['source_identity']['sha256']='f'*64
    import pytest
    with pytest.raises(ValueError,match='source digest changed'):
        _cell_results(run)


def test_retained_original_umap_is_pinned_and_used_for_each_annotation_level(tmp_path, monkeypatch):
    a=ad.AnnData(np.ones((3,2)),obs=pd.DataFrame(dict(cell_type_final=['A','B','B'],
        cell_type_level_1=['A','B','B']),index=['one','two','three']))
    xy=np.array([[.1,.2],[1.,2.],[3.,4.]])
    a.obsm['X_umap']=xy
    dataset=tmp_path/'annotated.h5ad';a.write_h5ad(dataset)
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    pin=lambda p: dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    data=dict(cells=a.obs_names.tolist(),source_sha256='a'*64,levels_reached=1,
        sources=[pin(manifest),pin(dataset)],group_order=['A','B'],group_counts=[1,2],
        composition=[dict(label='A',n_cells=1),dict(label='B',n_cells=2)],groups_omitted=0,
        genes=[],branches=[],branches_omitted=0)
    monkeypatch.setattr(owner,'presentation_data',lambda *args:deepcopy(data))
    source=dict(producing_step_id='annotation',tool_name='annotate_scATAC_cell_types',
        result_field='manifest_path',artifact_kind='scatac_cell_type_annotation_manifest_json',artifact_path=str(manifest),
        integrity=dict(sha256=pin(manifest)['sha256']))
    snapshot=SimpleNamespace(payload=dict(artifacts=[source],steps=[dict(step_id='annotation',
        tool_name='annotate_scATAC_cell_types')]),cell_results={})
    original=dataset.read_bytes()
    figures=owner.projections(snapshot)
    umaps=[f for f in figures if f.figure_kind=='annotation_umap']
    assert {f.presentation['annotation_level'] for f in umaps}=={'final',1}
    assert all(f.presentation['coordinate_origin']=='retained_original_input' for f in umaps)
    assert all(np.array_equal(f.data.value['coordinates'],xy) for f in umaps)
    assert all(any(s['artifact_path']==str(dataset) and s['integrity']['sha256']==pin(dataset)['sha256']
        for s in f.source_artifacts) for f in umaps)
    assert next(f for f in figures if f.figure_kind=='annotation_composition').presentation['compatible_umap_available']
    assert dataset.read_bytes()==original


def test_marker_projection_uses_exact_unique_columns_on_nonunique_native_axis(tmp_path, monkeypatch):
    import json
    from scipy import sparse

    activity = ad.AnnData(sparse.csr_matrix([[10., 2., 100.], [20., 4., 200.]]),
        obs=pd.DataFrame(index=['one', 'two']),
        var=pd.DataFrame(index=['repeat', 'unique', 'repeat']))
    activity_path = tmp_path/'activity.h5ad'; activity.write_h5ad(activity_path)
    annotated = ad.AnnData(sparse.csr_matrix((2, 1)), obs=pd.DataFrame(
        dict(cell_type_final=['A', 'A'], annotation_branch_id=['branch', 'branch']),
        index=['one', 'two']))
    dataset = tmp_path/'annotated.h5ad'; annotated.write_h5ad(dataset)
    pin = lambda p: dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    hierarchy = dict(source=pin(dataset), input_results=[dict(matrix_path=str(activity_path),
        matrix_sha256=pin(activity_path)['sha256'])], branches=[dict(branch_id='branch',
        parent=None, level=1, n_cells=2, stopping_reason='no_candidate_subdivision',
        annotation_round='round', cluster='group')])
    annotations = {'round': {'clusters': [dict(cluster='group', markers=[
        dict(feature='unique', identity='unique', positive=True),
        dict(feature='repeat', identity='repeat', positive=True)])]}}
    hp = tmp_path/'hierarchy.json'; hp.write_text(json.dumps(hierarchy))
    ap = tmp_path/'annotations.json'; ap.write_text(json.dumps(annotations))
    manifest = tmp_path/'manifest.json'; manifest.write_text('{}')
    pins = [pin(p) for p in [dataset, hp, ap]]
    monkeypatch.setattr(owner.annotation.publication, 'load',
        lambda *args: (dict(files=pins), pin(manifest)['sha256']))
    original = {p: p.read_bytes() for p in [activity_path, dataset, hp, ap]}

    result = owner.presentation_data(manifest, pin(manifest)['sha256'])

    assert result['genes'] == ['unique']  # No arbitrary choice among repeated names.
    assert result['ambiguous_marker_names'] == ['repeat']
    assert owner.presentation_data is owner.annotation.presentation_data
    assert result['per_cell_activity'] == [[2.], [4.]]
    assert result['mean_activity'] == [[3.]]
    assert result['fraction_positive'] == [[1.]]
    assert [r['gene'] for r in result['selected_marker_sources']] == ['unique']
    assert all(p.read_bytes() == content for p, content in original.items())
    activity.obs_names=['two','one']; activity.write_h5ad(activity_path)
    hierarchy['input_results'][0]['matrix_sha256']=pin(activity_path)['sha256']
    hp.write_text(json.dumps(hierarchy)); pins[:]=[pin(p) for p in [dataset,hp,ap]]
    import pytest
    with pytest.raises(owner.annotation.AuthorityError,match='ordered cell axis'):
        owner.presentation_data(manifest,pin(manifest)['sha256'])


def test_hierarchy_layout_preserves_relationships_and_keeps_subtrees_together():
    from matplotlib.figure import Figure
    rows = [dict(branch_id='root-a',parent=None,level=1,n_cells=4),
            dict(branch_id='root-b',parent=None,level=1,n_cells=2),
            dict(branch_id='child-a',parent='root-a',level=2,n_cells=2),
            dict(branch_id='child-b',parent='root-a',level=2,n_cells=2)]
    positions = owner._hierarchy_positions(rows)
    assert set(positions) == {r['branch_id'] for r in rows}
    assert positions['child-a'][1] > positions['root-a'][1] > positions['child-b'][1] > positions['root-b'][1]
    figure = Figure(figsize=(12,8))
    owner.render(figure, SimpleNamespace(kind='annotation_hierarchy', value=dict(branches=rows,branches_omitted=0)))
    ax = figure.axes[0]
    assert len(ax.lines) == 2 and len(ax.texts) == 8
    assert all(len(edge.get_xdata()) == 4 for edge in ax.lines)
    assert all(label.get_bbox_patch() is not None for label in ax.texts)


def test_composition_separates_final_labels_and_current_evidence():
    from matplotlib.figure import Figure
    figure = Figure(figsize=(15,6))
    owner.render(figure, SimpleNamespace(kind='annotation_composition', value=dict(
        groups=['a','b'],counts=[2,1],composition=[dict(label='ancestor',n_cells=3)],
        inherited_label_cells=1,population_summary=[dict(status='assigned',cells=2),dict(status='ambiguous',cells=1)])))
    assert 'may be inherited' in figure.axes[0].get_title()
    assert 'Current-round evidence' == figure.axes[1].get_title()
    assert [p.get_width() for p in figure.axes[1].patches] == [2,1]
    assert '1 cells inherit ancestor labels' in figure.texts[0].get_text()
