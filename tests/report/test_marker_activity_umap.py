"""Bounded native RP presentation: exact cells, unchanged UMAP and honest units."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace as N

import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from agent.report import annotation_visualization as annotation
from agent.report.visualization import build_cell_overlay_visualizations, _png_dimensions


def inputs():
    data=dict(cells=['c','a','b'], genes=[f'G{i}' for i in range(1,9)],
        per_cell_activity=np.arange(24).reshape(3,8).tolist(),
        selected_marker_sources=[dict(group=str(i//2),gene=f'G{i+1}',basis='positive_native_group_marker') for i in range(8)],
        measurement='accepted native Gene Activity; accessibility-derived activity, not measured RNA expression')
    ids=('a','b','c');xy=np.array([[1.,2.],[3.,4.],[5.,6.]])
    source=dict(producing_step_id='annotation',tool_name='annotate_scATAC_cell_types',
        artifact_kind='native_annotation_sidecar',artifact_path='/accepted/rp.npz',integrity=dict(sha256='a'*64))
    return data,ids,xy,[source]


def test_exact_alignment_and_general_marker_rule_preserve_all_cells():
    data,ids,xy,sources=inputs();original=copy.deepcopy(data);coordinates=xy.copy()
    value=annotation.marker_activity_input(data,ids,xy,sources,basis='exact_original_source')
    assert value['genes']==['G1','G3','G5','G7','G2','G4']
    np.testing.assert_array_equal(value['values'],np.asarray(data['per_cell_activity'])[[1,2,0]][:,[0,2,4,6,1,3]])
    np.testing.assert_array_equal(value['coordinates'],coordinates)
    assert value['cell_ids']==list(ids) and value['values'][-1][0]==0
    assert data==original and value['correspondence']['n_cells']==3
    for changed in (['a','b','other'],['a','a','b']):
        with pytest.raises(ValueError):
            annotation.marker_activity_input(dict(data,cells=changed),ids,xy,sources,basis='exact_original_source')
    with pytest.raises(ValueError,match='no representative'):
        annotation.marker_activity_input(dict(data,genes=[],selected_marker_sources=[]),ids,xy,sources,basis='exact_original_source')


def test_each_panel_uses_same_coordinates_and_actual_log_gene_activity_including_zero(monkeypatch):
    data,ids,xy,sources=inputs()
    value=annotation.marker_activity_input(data,ids,xy,sources,basis='exact_original_source')
    projection,=annotation.marker_activity_projection(lambda:value)
    seen=[];scatter=Axes.scatter
    def observed(self,x,y,**kwargs):
        seen.append((np.column_stack((x,y)),kwargs['c']))
        return scatter(self,x,y,**kwargs)
    monkeypatch.setattr(Axes,'scatter',observed)
    annotation.render(Figure(figsize=(12,8)),projection.data)
    assert len(seen)==6
    for j,(coordinates,activity) in enumerate(seen):
        np.testing.assert_array_equal(coordinates,xy)
        np.testing.assert_array_equal(activity,np.log1p(np.asarray(value['values'])[:,j]))
    assert projection.presentation['normalization']=='none added'
    assert 'not measured RNA expression' in projection.presentation['caption']
    assert projection.presentation['dependent_evidence']


def test_marker_bundle_is_an_existing_interaction_publication(tmp_path):
    data,ids,xy,sources=inputs()
    value=annotation.marker_activity_input(data,ids,xy,sources,basis='exact_original_source')
    result=build_cell_overlay_visualizations(lambda:value,tmp_path/'presentation',
        publisher=dict(session_id='session',turn_id='markers'),kind='marker_activity_umap')
    manifest=json.loads(Path(result['manifest_path']).read_text())
    assert manifest['publisher']==dict(kind='interaction',session_id='session',turn_id='markers')
    figure,=manifest['figures']
    assert figure['figure_kind']=='marker_activity_umap'
    assert figure['source_artifacts']==sources
    assert manifest['validation']['scientific_tools_invoked'] is False
    assert _png_dimensions(Path(result['figures'][0]['figure_path']))==(2400,1600)


def test_accepted_distinct_branches_require_common_source_and_exact_cells(monkeypatch):
    from agent.application import result_reading as reader
    from agent.tools.analysis import hierarchical_annotation
    data,ids,xy,sources=inputs();data['source_sha256']='a'*64
    data['sources']=[dict(path='/accepted/annotation.json',sha256='b'*64)]
    identity=dict(source_sha256='a'*64,artifact_sha256='c'*64,
        run_id='coordinates-run',step_id='umap',accepted_step_sha256='d'*64)
    monkeypatch.setattr(reader,'_coordinate_input',lambda *a:(ids,xy,identity,sources[0]))
    step=N(step_id='annotation',tool_name='annotate_scATAC_cell_types',
        result=dict(manifest_path='/accepted/annotation.json',manifest_sha256='b'*64))
    monkeypatch.setattr(reader,'_accepted_step',lambda *a:(step,None,N(run_id='different-run')))
    monkeypatch.setattr(reader,'accepted_step_digest',lambda s:'e'*64)
    monkeypatch.setattr(hierarchical_annotation,'presentation_data',lambda *a:data)
    sessions=N(_application=N(run_store=object()))
    activity=dict(run_id='different-run',step_id='annotation',output_key='manifest_path',accepted_step_sha256='e'*64)
    value=reader.marker_activity_overlay(sessions,{},activity)
    assert value['cell_ids']==list(ids) and value['correspondence']['basis']=='exact_original_source'
    assert value['sources'][-1]['accepted_output']==activity
    data['source_sha256']='f'*64
    with pytest.raises(ValueError,match='no qualified'):
        reader.marker_activity_overlay(sessions,{},activity)
