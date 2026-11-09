"""Saved native QC plots preserve populations and do not run scientific tools."""
import hashlib
import json
from types import SimpleNamespace
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from agent.report import visualization as v


def publication(tmp_path, name, obs, slots, *, uns=None, parameters=None):
    path = tmp_path / (name + '.h5ad')
    ad.AnnData(obs=pd.DataFrame(obs, index=[f'cell-{i}' for i in range(len(next(iter(obs.values()))))]),
        uns=uns or {}).write_h5ad(path)
    return SimpleNamespace(step_id=name, tool_name='run_snapatac2', result=dict(matrix_path=str(path),
        matrix_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), n_cells=len(next(iter(obs.values()))),
        result_slots=slots, function_name='snapatac2.'+name, parameters=parameters or {}))


def test_native_qc_plots_use_saved_values_and_own_profile_populations(tmp_path):
    imported = publication(tmp_path, 'import', dict(n_fragment=[0,10,100,1000,2000,3000]), ['obs.n_fragment'])
    lengths = publication(tmp_path, 'lengths', dict(n_fragment=[0,10,100,1000,2000,3000]),
        ['uns.frag_size_distr'], uns=dict(frag_size_distr=np.array([9,1,4,3])))
    measured = publication(tmp_path, 'tsse', dict(n_fragment=[10,100,1000,2000], tsse=[1,4,np.nan,8]),
        ['obs.tsse','uns.TSS_profile'], uns=dict(TSS_profile=np.array([2,3,7,3,2])))
    selected = publication(tmp_path, 'filter', dict(n_fragment=[2000],tsse=[8]), ['obs'],
        parameters=dict(population=dict(stage='native_cell_filter'), effective_native_parameters=dict(min_counts=1000,min_tsse=4)))
    inputs = v._qc_input([('run',s) for s in (selected,measured,lengths,imported)])
    projection, = v._qc_projection(lambda:inputs)
    assert projection.data.value['depth'] == [10,100,1000,2000]
    assert projection.data.value['tsse'] == [1,4,None,8]
    assert projection.data.value['selected_cells'] == 1
    assert [(p['slot'],p['n_cells']) for p in projection.data.value['profiles']] == [('TSS_profile',4),('frag_size_distr',6)]
    bundle = v.build_cell_overlay_visualizations(lambda:inputs,tmp_path/'presentation',
        publisher=dict(session_id='session',turn_id='question'),kind='qc')
    manifest = json.loads(Path(bundle['manifest_path']).read_text())
    assert manifest['publisher']['kind']=='interaction'
    assert manifest['figures'][0]['presentation']['thresholds']==dict(min_counts=1000,min_tsse=4)
    assert v._png_dimensions(Path(bundle['figures'][0]['figure_path'])) == (2400,2000)
    # Publication reads never overwrite the saved native observations.
    assert hashlib.sha256(Path(measured.result['matrix_path']).read_bytes()).hexdigest()==measured.result['matrix_sha256']


def test_qc_absence_and_changed_source_are_not_fabricated(tmp_path):
    step = publication(tmp_path,'plain',dict(other=[1,2]),['obs.other'])
    with pytest.raises(ValueError,match='no recorded native'):
        v._qc_input([('run',step)])
    step = publication(tmp_path,'depth',dict(n_fragment=[10,20]),['obs.n_fragment'])
    inputs=v._qc_input([('run',step)])
    Path(step.result['matrix_path']).write_bytes(b'changed')
    with pytest.raises(ValueError):v._qc_projection(lambda:inputs)


def test_qc_profile_after_tsse_stays_in_its_recorded_dependency_chain(tmp_path):
    from agent.schemas import StepOutputRef
    imported=publication(tmp_path,'import',dict(n_fragment=[300,400]),['obs.n_fragment'])
    tsse=publication(tmp_path,'tsse',dict(n_fragment=[300,400],tsse=[2,7]),['obs.tsse','uns.TSS_profile'],uns=dict(TSS_profile=np.array([1,2,1])))
    lengths=publication(tmp_path,'lengths',dict(n_fragment=[300,400],tsse=[2,7]),['uns.frag_size_distr'],uns=dict(frag_size_distr=np.array([0,5,2])))
    tsse.resolved_arguments={'input_path':imported.result['matrix_path']}
    lengths.resolved_arguments={'input_path':tsse.result['matrix_path']}
    planned=[SimpleNamespace(step_id='import',arguments={}),
        SimpleNamespace(step_id='tsse',arguments=dict(input_path=StepOutputRef('import','matrix_path'))),
        SimpleNamespace(step_id='lengths',arguments=dict(input_path=StepOutputRef('tsse','matrix_path')))]
    run=SimpleNamespace(run_id='run',steps=[imported,tsse,lengths],plan=SimpleNamespace(steps=planned))
    inputs,=v._run_qc_inputs(run)
    assert inputs['primary']['path']==tsse.result['matrix_path']
    assert {p['slot'] for p in inputs['profiles']}=={'TSS_profile','frag_size_distr'}
    assert {s['producing_step_id'] for s in inputs['sources']}=={'tsse','lengths'}


def test_native_qc_metadata_keeps_run_output_association():
    from agent.application.dialogue_evidence import figure_metadata
    record=dict(figure_kind='qc_metrics',figure_id='qc-one',source_step_ids=['tsse'],
        source_artifacts=[dict(tool_name='run_snapatac2',artifact_kind='native_matrix_h5ad',
            producing_step_id='tsse',run_id='run',role='qc_measurements')],
        presentation=dict(delivery='accepted_native_qc',n_cells=8,selected_cells=None,thresholds={}))
    revision=SimpleNamespace(revision_id='revision',outputs=[SimpleNamespace(name='qc-data',run_id='run',step_id='tsse')])
    metadata=figure_metadata(record,manifest_sha256='a'*64,run_id='run',revision=revision)
    assert metadata['output_names']==['qc-data']
    assert metadata['sources'][0]['run_id']=='run'
