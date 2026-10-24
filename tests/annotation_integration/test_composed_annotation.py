"""Qualified composition reuses real annotation authority with stub native backend.

Existing tiny annotation fixtures execute/publicize once. Composition performs
no annotation, upstream reconstruction or scientific authority publication.
"""
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from agent.application import OutputSelection, ResearchAgentApplication
from agent.application.analysis_delivery import COMPOSED_CONTRACT, EXPORT_KEY, composed_analysis
from agent.application.matrix_delivery import MatrixDownloadPreparer
from agent.orchestration import AgentRequest, LLMPlanner
from agent.tools.analysis import external_marker_annotation, marker_annotation
from test_annotation import case, Model, wire as canonical_wire
from test_external_entry import external_case, wire as external_wire
from test_application import owner_calls


@pytest.fixture(params=['canonical', 'external'])
def composition_case(request):
    if request.param == 'external':
        args, calls = request.getfixturevalue('external_case')
        original = Path(args['source_path'])
        wire = external_wire()
    else:
        args, calls, _, run = request.getfixturevalue('case')
        original = Path(run.steps[-1].result['matrix_path'])
        wire = canonical_wire()
    return args, calls, original, wire, request.param


def test_accepted_annotation_composes_original_measurements_and_exact_selected_labels(
        composition_case, tmp_path, monkeypatch):
    args, calls, original_path, wire, mode = composition_case
    app = ResearchAgentApplication(tmp_path/'composed-annotation', planner=LLMPlanner(Model(wire)))
    app.sessions.create('annotation')
    request = AgentRequest('accepted-annotation', 'Annotate the explicit synthetic groups.',
        {k:v for k,v in args.items() if k != 'output_dir'})
    state = app.sessions.run('annotation', 'accepted', request,
        (OutputSelection('annotation', 'annotation', 'manifest_path'),), expected_generation=0)
    turn = state.turn('accepted')
    assert turn.status == 'activated', turn
    assert len(calls) == 1
    revision, = state.revisions
    locator = dict(asdict(revision.outputs[0]), revision_id=revision.revision_id)
    run_before = app.run_store.load(turn.run_id)
    native_path = Path(run_before.steps[0].result['annotated_h5ad_path'])
    original, native = ad.read_h5ad(original_path), ad.read_h5ad(native_path)
    protected = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (original_path,native_path)}
    def no_annotation(*args, **kwargs):
        pytest.fail('Composition reentered annotation execution.')
    monkeypatch.setattr(marker_annotation, 'annotate_cell_groups', no_annotation)
    monkeypatch.setattr(external_marker_annotation, 'annotate_cell_groups', no_annotation)
    selection = dict(measurement=dict(accepted_output=locator), observations=[dict(
        accepted_output=locator, field='result:primary_annotation', destination='selected_primary_annotation')],
        representations=[])
    with owner_calls() as upstream:
        artifact = composed_analysis(app.sessions, 'annotation', selection, turn_id='composition-only')
        preparer = MatrixDownloadPreparer()
        with preparer.prepare(artifact) as snapshot:
            raw = b''.join(snapshot.iter_chunks())
            assert snapshot.sha256 == hashlib.sha256(raw).hexdigest()
            assert snapshot.size_bytes == len(raw)
            assert snapshot.source_sha256 == protected[original_path]
        assert preparer.active_count == preparer._reserved_bytes == 0
    assert not upstream and len(calls) == 1
    exported = ad.read_h5ad(io.BytesIO(raw))
    assert type(exported.X) is type(original.X)
    for field in ('data', 'indices', 'indptr'):
        np.testing.assert_array_equal(getattr(exported.X,field),getattr(original.X,field))
    assert exported.obs_names.equals(original.obs_names) and exported.var_names.equals(original.var_names)
    pd.testing.assert_frame_equal(exported.var, original.var)
    pd.testing.assert_series_equal(exported.obs['selected_primary_annotation'],
        native.obs['primary_annotation'], check_names=False)
    assert 'primary_annotation' not in original.obs and 'primary_annotation' not in exported.obs
    metadata = exported.uns[EXPORT_KEY]
    selected = json.loads(metadata['selection_json'])
    field, = selected['observations']
    assert metadata['contract'] == selected['policy'] == COMPOSED_CONTRACT
    assert metadata['recipe_handle'] == artifact.reference.handle
    assert metadata['original_source_sha256'] == protected[original_path]
    assert field['accepted_output'] == locator
    assert field['field'] == 'result:primary_annotation' and field['destination'] == 'selected_primary_annotation'
    assert field['observation_role'] == 'computational_assignment'
    assert field['method']['annotation']['validation_state'] == 'not_assessed'
    if mode == 'external':
        assert field['method']['annotation']['parameters']['count_slot'] == 'X'
        assert (exported.layers['raw_counts'] != original.layers['raw_counts']).nnz == 0
        assert exported.uns['measurement'] == original.uns['measurement']
    else:
        assert exported.uns['matrix_semantics'] == original.uns['matrix_semantics'] == 'fragment_counts'
    assert b'matrix_manifest_path' not in raw and b'source_path' not in raw
    assert app.sessions.load('annotation') == state and app.run_store.load(turn.run_id) == run_before
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path,digest in protected.items())
