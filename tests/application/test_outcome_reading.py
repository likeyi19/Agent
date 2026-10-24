"""Exact outcome operands and existing export eligibility; no native analysis."""
from dataclasses import asdict, replace
from types import SimpleNamespace
import hashlib
import json

import anndata as ad
import pandas as pd
import numpy as np
import pytest

from agent.application import analysis_delivery, result_reading
from test_analysis_export import accepted_export
from test_dialogue_evidence import app, accepted, files, forbid_work


def comparison_fixture(app, monkeypatch, *, copies=1):
    old = accepted(app, 'inspect_scATAC')[0]
    revision = app.sessions.load('session').revisions[0]
    output = revision.outputs[0]
    captured = dict(dialogue=dict(outputs=[dict(revision_id=old, output_name=output.name,
        accepted_step_sha256=output.accepted_step_sha256, tool='inspect_scATAC')]))
    origin = SimpleNamespace(turn_id='request', snapshot=captured, admitted=dict(
        comparison=dict(left='g0', right='new', detail=dict(section='category_counts',
            field='source:cohort', other_field='result:leiden', offset=0, limit=4))))
    new = accepted(app, 'inspect_scATAC', name='continued', copies=copies)[0]
    saved = app.sessions.load('session')
    turn = saved.turn('continued')
    turn = replace(turn, selections=tuple(replace(s, use=('comparison_result',)) for s in turn.selections))
    current = SimpleNamespace(turn_id='continued', admitted=dict(continuation=dict(origin_turn_id='request')))
    state = SimpleNamespace(revisions=saved.revisions, interactions=(origin, current),
                            turn=lambda turn_id: turn if turn_id == 'continued' else saved.turn(turn_id))
    monkeypatch.setattr(app.sessions, 'load', lambda session_id: state)
    return old, new, origin, current, state


def test_comparison_resolves_original_capture_and_explicit_new_purpose(app, monkeypatch):
    old, new, origin, current, state = comparison_fixture(app, monkeypatch)
    before = files(app)
    left, right, detail = result_reading.comparison_inputs(app.sessions, 'session', current)
    assert left['revision_id'] == old and right['revision_id'] == new
    assert left['output_name'] == right['output_name'] == 'output0'
    # These identical simulated summaries may share a step digest; their exact
    # accepted Revisions still select different original/new scientific results.
    assert left != right
    assert detail == origin.admitted['comparison']['detail']
    assert files(app) == before


def test_comparison_rejects_ambiguous_new_selection(app, monkeypatch):
    _, _, _, current, _ = comparison_fixture(app, monkeypatch, copies=2)
    with pytest.raises(ValueError, match='exactly one'):
        result_reading.comparison_inputs(app.sessions, 'session', current)


@pytest.mark.parametrize('failure', ['capture', 'accepted_anchor'])
def test_comparison_never_substitutes_a_changed_or_unoffered_source(app, monkeypatch, failure):
    _, _, origin, current, state = comparison_fixture(app, monkeypatch)
    if failure == 'capture':
        origin.admitted['comparison']['left'] = 'g99'
    else:
        revision = state.revisions[-1]
        changed = replace(revision.outputs[0], accepted_step_sha256='0'*64)
        state.revisions = state.revisions[:-1] + (replace(revision, outputs=(changed,)),)
    with pytest.raises(ValueError):
        result_reading.comparison_inputs(app.sessions, 'session', current)


def native_result(app, tmp_path, *, name='initial', tool='compute_cell_umap',
                  labels=('A', 'A', 'B', 'B'), ids=('a', 'b', 'c', 'd')):
    path = tmp_path/(name+'.h5ad')
    data = ad.AnnData(obs=pd.DataFrame(dict(leiden=pd.Categorical(labels)), index=ids))
    data.obsm['X_umap'] = np.arange(len(ids)*2,dtype=np.float32).reshape(len(ids),2)
    data.write_h5ad(path)
    revision = accepted(app, tool, name=name, overrides=dict(analysis_path=str(path),n_cells=len(ids),
        analysis_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))[0]
    state = app.sessions.load('session')
    output = next(r for r in state.revisions if r.revision_id == revision).outputs[0]
    return dict(asdict(output), revision_id=revision), path


def source_figure(app,tmp_path,monkeypatch,*,coordinates='new',failure=None):
    from agent.application.local_resources import LocalResourceAdmission
    coordinate,path = native_result(app,tmp_path)
    source = tmp_path/'original.h5ad'
    ids = ('d','c','b','a') if failure=='axis' else ('a','b','c','d')
    values = ['/home/operator/private','B',None,'A'] if failure=='private' else ['A','B',None,'A']
    data = ad.AnnData(obs=pd.DataFrame(dict(original_annotation=pd.Categorical(values),
        measurement=np.arange(4,dtype=float)),index=ids))
    data.write_h5ad(source)
    owner = LocalResourceAdmission(app._workspace,approved_source_roots=(tmp_path,))
    record = owner.register('source-fixture',source,label='Supplied source',
        attribution='Simulated accepted original-source relationship; not biological qualification.',session_id='session')
    binding = owner.compose_group([record.resource_id],attachment_candidates=True)
    saved = app.sessions.load('session')
    turn = saved.turn('initial')
    turn = replace(turn,selections=tuple(replace(s,use=('visualization_coordinates',)) for s in turn.selections))
    origin = SimpleNamespace(turn_id='initial',snapshot=dict(dialogue=dict(outputs=[dict(
        revision_id=coordinate['revision_id'],output_name=coordinate['name'],
        accepted_step_sha256=coordinate['accepted_step_sha256'],tool='compute_cell_umap')])),
        submission=dict(registered_input=binding.attribution(),execution_inputs={}),
        admitted=dict(visualization=dict(coordinates=coordinates,labels='source.0',field='source:original_annotation')))
    monkeypatch.setattr(app.sessions,'load',lambda session_id:SimpleNamespace(
        interactions=(origin,),revisions=saved.revisions,turn=lambda turn_id:turn))
    # Only the accepted original-source identity is simulated. Actual registered
    # identity, pinned source bytes, selected fields and ordered axes are read.
    monkeypatch.setattr(result_reading,'original_source',lambda *a: None if failure=='unqualified' else
        dict(path=str(source),sha256='0'*64 if failure=='wrong_source' else record.source_sha256))
    return origin,coordinate,source,record


@pytest.mark.parametrize('coordinates',['new','g0'])
def test_first_turn_or_captured_coordinates_use_exact_registered_source_field(app,tmp_path,monkeypatch,coordinates):
    origin,coordinate,source,record = source_figure(app,tmp_path,monkeypatch,coordinates=coordinates)
    before = files(app)
    operands = result_reading.visualization_inputs(app.sessions,'session',origin)
    assert operands[0] == coordinate
    assert operands[1] == dict(registered_source=dict(resource_id=record.resource_id,
        record_sha256=record.record_sha256,source_sha256=record.source_sha256))
    projected = result_reading.cell_overlay(app.sessions,*operands)
    assert projected['cell_ids'] == ('a','b','c','d')
    assert projected['labels'].iloc[[0,1,3]].tolist() == ['A','B','A']
    assert projected['labels'].isna().tolist() == [False,False,True,False]
    np.testing.assert_array_equal(projected['coordinates'],np.arange(8,dtype=np.float32).reshape(4,2))
    assert projected['correspondence']['basis'] == 'exact_original_source'
    label_source = projected['sources'][1]
    assert label_source['registered_source']['source_sha256'] == record.source_sha256
    assert not {'accepted_output','producing_step_id','tool_name'} & set(label_source)
    assert label_source['observation_role'].startswith('supplied_source_observation')
    assert files(app) == before


def test_source_indexes_follow_captured_companions_with_a_scalar_selected_h5ad(app,tmp_path,monkeypatch):
    from agent.application.dialogue_execution import _research_intent
    from agent.application.local_resources import LocalResourceAdmission
    from agent.application.session_state import Interaction
    origin,coordinate,source,record = source_figure(app,tmp_path,monkeypatch,coordinates='g0')
    owner = LocalResourceAdmission(app._workspace,approved_source_roots=(tmp_path,))
    primary = tmp_path/'selected-primary.h5ad'
    ad.AnnData(obs=pd.DataFrame(dict(original_annotation=pd.Categorical(['P']*4)),
        index=['a','b','c','d'])).write_h5ad(primary)
    selected = owner.register('selected-primary',primary,label='Selected primary H5AD',
        attribution='Simulated primary source; no population correspondence.',session_id='session')
    scalar = owner.resolve(selected.resource_id,tool_name='inspect_scATAC')
    candidates = owner.compose_group([selected.resource_id,record.resource_id],attachment_candidates=True)
    origin.submission = dict(registered_input=scalar.attribution(),
        execution_inputs=dict(scalar.execution_inputs),candidate_input=candidates.attribution())
    origin.snapshot['dialogue']['results'] = [dict(handle='g0',tool='compute_cell_umap')]
    drawing = origin.admitted['visualization']
    drawing['labels'] = 'source.1'
    def captured():
        return Interaction(origin.turn_id,'Color the accepted coordinates by the exact offered source.',
            coordinate['revision_id'],1,origin.snapshot,admitted=origin.admitted,submission=origin.submission)
    assert _research_intent(captured(),dict(intent='scientific',visualization=drawing))['visualization'] == drawing
    operands = result_reading.visualization_inputs(app.sessions,'session',captured())
    assert operands[0] == coordinate
    assert operands[1]['registered_source']['resource_id'] == record.resource_id
    assert result_reading.cell_overlay(app.sessions,*operands)['labels'].isna().tolist() == [False,False,True,False]
    # source.0 remains the explicitly offered primary, and cannot inherit the
    # companion's exact original-source qualification from its cell count.
    drawing['labels'] = 'source.0'
    operands = result_reading.visualization_inputs(app.sessions,'session',captured())
    assert operands[1]['registered_source']['resource_id'] == selected.resource_id
    with pytest.raises(ValueError,match='qualified scientific population'):
        result_reading.cell_overlay(app.sessions,*operands)


@pytest.mark.parametrize('failure',['wrong_source','unqualified','axis','changed_bytes','record','field','numeric','private'])
def test_registered_figure_source_requires_qualified_bytes_fields_axes_and_safe_labels(app,tmp_path,monkeypatch,failure):
    origin,coordinate,source,record = source_figure(app,tmp_path,monkeypatch,failure=failure)
    coordinates,categories,field = result_reading.visualization_inputs(app.sessions,'session',origin)
    if failure=='changed_bytes': source.write_bytes(source.read_bytes()+b'changed')
    if failure=='record': categories['registered_source']['record_sha256']='0'*64
    if failure=='field': field='source:missing'
    if failure=='numeric': field='source:measurement'
    with pytest.raises(ValueError):
        result_reading.cell_overlay(app.sessions,coordinates,categories,field)


def test_new_coordinate_and_label_purposes_resolve_only_after_acceptance(app,tmp_path,monkeypatch):
    output,_ = native_result(app,tmp_path)
    saved = app.sessions.load('session')
    turn = saved.turn('initial')
    turn = replace(turn,selections=tuple(replace(s,
        use=('visualization_coordinates','visualization_labels')) for s in turn.selections))
    origin = SimpleNamespace(turn_id='initial',admitted=dict(visualization=dict(
        coordinates='new',labels='new',field='result:leiden')))
    monkeypatch.setattr(app.sessions,'load',lambda session_id:SimpleNamespace(
        interactions=(origin,),revisions=saved.revisions,turn=lambda turn_id:turn))
    coordinate,labels,field = result_reading.visualization_inputs(app.sessions,'session',origin)
    assert coordinate == labels == output
    projected = result_reading.cell_overlay(app.sessions,coordinate,labels,field)
    assert projected['correspondence']['basis'] == 'same_accepted_artifact'
    assert projected['labels'].tolist() == ['A','A','B','B']
    turn = replace(turn,status='run_succeeded',revision_id=None,completion_files=(),run_result_sha256=None)
    with pytest.raises(ValueError,match='not been accepted'):
        result_reading.visualization_inputs(app.sessions,'session',origin)


def test_export_projection_uses_existing_inventory_without_preparing_or_qualifying_bytes(app, monkeypatch):
    _, revision_id = accepted_export(app, monkeypatch)
    guarded = forbid_work(app, monkeypatch)
    monkeypatch.setattr('agent.application.matrix_delivery.MatrixDownloadPreparer.prepare',
        lambda *a, **k: pytest.fail('Export description prepared download bytes.'))
    revision = guarded.sessions.load('session').revisions[0]
    output = revision.outputs[0]
    target = dict(revision_id=revision_id, output_name=output.name,
                  accepted_step_sha256=output.accepted_step_sha256, subject=None)
    before = files(app)
    descriptions = analysis_delivery.export_descriptions(guarded.sessions, 'session', (target,))
    assert {d['artifact_type'] for d in descriptions} == {'scientific_analysis', 'scientific_integrated'}
    assert all(d['status'] == 'available_for_preparation' and d['payload_checks'] == 'performed_on_download'
               and d['target'] == 0 for d in descriptions)
    integrated = next(d for d in descriptions if d['original_measurements'])
    compact = next(d for d in descriptions if not d['original_measurements'])
    assert integrated['size_bytes'] is None and compact['size_bytes'] > 0
    assert not integrated['selected_result']['umap_coordinates']
    assert not integrated['selected_result']['leiden_assignments']
    assert integrated['original_feature_axis'] and not compact['original_feature_axis']
    assert str(app.workspace_root) not in json.dumps(descriptions)
    assert all(not {'sha256', 'run_id', 'path'} & set(d) for d in descriptions)
    assert files(app) == before


def test_export_projection_rejects_wrong_accepted_target(app, monkeypatch):
    _, revision_id = accepted_export(app, monkeypatch)
    with pytest.raises(ValueError, match='accepted output'):
        analysis_delivery.export_descriptions(app.sessions, 'session', (
            dict(revision_id=revision_id, output_name='output0', accepted_step_sha256='0'*64),))
