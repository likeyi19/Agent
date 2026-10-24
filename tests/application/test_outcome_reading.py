"""Exact outcome operands and existing export eligibility; no native analysis."""
from dataclasses import asdict, replace
from types import SimpleNamespace
import hashlib
import json

import anndata as ad
import pandas as pd
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


@pytest.mark.parametrize('drawing', [
    dict(coordinates='new', labels='source.0', field='source:cell_type'),
    dict(coordinates='g0', labels='source.0', field='source:cell_type'),
])
def test_precise_unimplemented_source_figure_is_not_rendered(app, monkeypatch, drawing):
    origin = SimpleNamespace(turn_id='request', snapshot={}, admitted=dict(visualization=drawing))
    monkeypatch.setattr(app.sessions, 'load', lambda session_id: SimpleNamespace(interactions=(origin,)))
    monkeypatch.setattr('agent.report.visualization.build_analysis_visualizations',
        lambda *a, **k: pytest.fail('An unsupported source figure entered rendering.'))
    assert result_reading.visualization_inputs(app.sessions, 'session', origin) is None
    result_reading.publish_cell_overlay(app.sessions, 'session', origin, None)


def native_result(app, tmp_path, *, name='initial', tool='compute_cell_umap',
                  labels=('A', 'A', 'B', 'B'), ids=('a', 'b', 'c', 'd'), clustered=True):
    path = tmp_path/(name+'.h5ad')
    ad.AnnData(obs=pd.DataFrame(dict(leiden=pd.Categorical(labels)), index=ids)).write_h5ad(path)
    revision = accepted(app, tool, name=name, overrides=dict(analysis_path=str(path),
        analysis_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), clustering_present=clustered))[0]
    state = app.sessions.load('session')
    output = next(r for r in state.revisions if r.revision_id == revision).outputs[0]
    return dict(asdict(output), revision_id=revision), path


def test_native_figure_recognition_requires_exact_clustered_result_and_pinned_field(app, tmp_path):
    output, path = native_result(app, tmp_path)
    before = files(app)
    assert result_reading.native_visualization(app.sessions, output, output, 'result:leiden')
    assert not result_reading.native_visualization(app.sessions, output, output, 'source:leiden')
    assert not result_reading.native_visualization(app.sessions, output, output, 'result:other')
    wrong = dict(output, accepted_step_sha256='0'*64)
    with pytest.raises(ValueError, match='Changed accepted'):
        result_reading.native_visualization(app.sessions, output, wrong, 'result:leiden')
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError):
        result_reading.native_visualization(app.sessions, output, output, 'result:leiden')
    assert files(app) == before


def test_new_coordinate_and_label_purposes_resolve_without_optional_rerender(app, tmp_path, monkeypatch):
    output, _ = native_result(app, tmp_path)
    saved = app.sessions.load('session')
    turn = saved.turn('initial')
    turn = replace(turn, selections=tuple(replace(s,
        use=('visualization_coordinates', 'visualization_labels')) for s in turn.selections))
    origin = SimpleNamespace(turn_id='initial', admitted=dict(visualization=dict(
        coordinates='new', labels='new', field='result:leiden')))
    monkeypatch.setattr(app.sessions, 'load', lambda session_id: SimpleNamespace(
        interactions=(origin,), turn=lambda turn_id: turn))
    monkeypatch.setattr('agent.report.visualization.build_analysis_visualizations',
        lambda *a, **k: pytest.fail('A new-coordinate request added an overlay render.'))
    coordinate, labels, field = result_reading.visualization_inputs(app.sessions, 'session', origin)
    assert coordinate == labels == dict(output, revision_id=None)
    assert result_reading.native_visualization(app.sessions, coordinate, labels, field)
    result_reading.publish_cell_overlay(app.sessions, 'session', origin, None)


@pytest.mark.parametrize('failure', [None, 'unrelated', 'labels', 'axis', 'uniform'])
def test_native_figure_reuses_only_recorded_identical_leiden_ancestor(app, tmp_path, monkeypatch, failure):
    coordinate, _ = native_result(app, tmp_path, clustered=failure != 'uniform')
    category, _ = native_result(app, tmp_path, name='labels', tool='cluster_cells',
        labels=('A', 'B', 'A', 'B') if failure == 'labels' else ('A', 'A', 'B', 'B'),
        ids=('d', 'c', 'b', 'a') if failure == 'axis' else ('a', 'b', 'c', 'd'))
    step, plan, _ = result_reading._accepted_step(app.run_store, category['run_id'], category['step_id'])
    # Only the recorded dependency is simulated. Accepted locators, pinned H5AD
    # observations and exact typed labels/ordered IDs use production readers.
    monkeypatch.setattr(result_reading, 'analysis_lineage', lambda *a:
        () if failure == 'unrelated' else ((category['run_id'], step, plan),))
    if failure == 'axis':
        with pytest.raises(ValueError, match='identities'):
            result_reading.native_visualization(app.sessions, coordinate, category, 'result:leiden')
    else:
        assert result_reading.native_visualization(app.sessions, coordinate, category, 'result:leiden') is (failure is None)


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
