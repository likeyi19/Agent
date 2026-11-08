"""Capability, bounded storage and exact resource identity regressions."""
from dataclasses import replace
import hashlib
from types import SimpleNamespace

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from agent.tools.analysis import cell_result, categorical, peak_h5ad
from agent.application.delivery_policy import DeliveryPolicy
from agent.application import matrix_delivery


def result(function='tl.umap', **kwargs):
    return dict(contract_version='native-snapatac2-matrix.v1', function_name='snapatac2.'+function,
        matrix_path='/accepted/result.h5ad', matrix_sha256='a'*64,
        parameters=dict(count_slot='X', kwargs=kwargs), result_slots=['obsm.X_umap'])


def test_actual_written_native_slots_and_custom_keys():
    value = result(key_added='layout')
    assert cell_result.describe('run_snapatac2', value)['coordinates'] is None
    value['result_slots'] = ['obsm.X_layout']
    assert cell_result.describe('run_snapatac2', value)['coordinates'] == 'X_layout'
    value['parameters']['kwargs']['inplace'] = False
    assert cell_result.describe('run_snapatac2', value)['coordinates'] is None
    value = result('tl.leiden', key_added='groups')
    value['result_slots'] = ['obs.groups']
    assert cell_result.describe('run_snapatac2', value)['grouping'] == 'groups'
    value['function_name'] = 'snapatac2.pp.knn'
    assert cell_result.describe('run_snapatac2', value)['grouping'] is None
    assert cell_result.describe('inspect_scATAC', value) is None
    from agent.application.result_reading import _role
    step = SimpleNamespace(tool_name='run_snapatac2', result=value)
    assert _role(step, 'result', 'groups', {}) == 'retained_observation; not a newly produced assignment'


def test_transform_and_layer_do_not_retain_original_count_identity():
    assert cell_result.describe('run_snapatac2', result('pp.binarize'))['preserves_counts'] is False
    value = result()
    value['parameters']['count_slot'] = 'layers/other'
    assert cell_result.describe('run_snapatac2', value)['preserves_counts'] is False


def test_coordinates_read_only_exact_slot_hash_axis_and_chunk(tmp_path):
    path = tmp_path/'cells.h5ad'
    a = ad.AnnData(sparse.eye(3, format='csr'), obs=pd.DataFrame(index=['b','a','c']))
    a.obsm['X_custom'] = np.arange(6, dtype=float).reshape(3, 2)
    a.write_h5ad(path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    ids, xy = categorical.coordinates(path, sha, 'X_custom')
    assert ids == ('b','a','c') and np.array_equal(xy, a.obsm['X_custom'])
    with pytest.raises(ValueError):
        categorical.coordinates(path, sha, 'X_umap')
    with pytest.raises(ValueError):
        categorical.require_same_cells(ids, tuple(reversed(ids)))
    with pytest.raises(ValueError):
        categorical.coordinates(path, '0'*64, 'X_custom')
    with h5py.File(path, 'a') as handle:
        del handle['obsm/X_custom']
        handle['obsm/X_custom'] = h5py.SoftLink('/X')
    with pytest.raises(ValueError):
        categorical.coordinates(path, hashlib.sha256(path.read_bytes()).hexdigest(), 'X_custom')


def test_scientific_storage_inspection_has_no_dataset_scale_admission(tmp_path):
    path = tmp_path/'large-logical.h5ad'
    ad.AnnData(sparse.eye(3, format='csr')).write_h5ad(path)
    with h5py.File(path, 'a') as handle:
        handle.create_dataset('uns/artificial_numeric', shape=(300_000_000,), dtype='float64', chunks=(1024,))
    # The artificial unallocated dataset exceeds the old 2 GiB decoded cap.
    # Inspecting storage does not allocate it; native allocations belong to OS.
    assert peak_h5ad.inspect(path, max_logical_bytes=None) > 2*1024**3


def test_configured_delivery_source_destination_reservation_and_release(tmp_path, monkeypatch):
    monkeypatch.setattr(matrix_delivery, 'validate_matrix_h5ad', lambda *a: None)
    path = tmp_path/'source'; path.write_bytes(b'x'*64)
    artifact = SimpleNamespace(path=path, managed_root=tmp_path, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        size_bytes=64, contract='scatac-cell-by-ccre.v1', manifest={})
    policy = DeliveryPolicy(source_bytes=128, delivered_bytes=128, temporary_bytes=64, logical_bytes=256, chunk_bytes=32)
    preparer = matrix_delivery.MatrixDownloadPreparer(policy)
    with preparer.prepare(artifact) as first:
        assert first.size_bytes == 64
        with pytest.raises(matrix_delivery.MatrixDeliveryError, match='capacity'):
            preparer.prepare(artifact)
    assert preparer.active_count == 0
    with pytest.raises(matrix_delivery.MatrixDeliveryError):
        matrix_delivery.MatrixDownloadPreparer(replace(policy, source_bytes=63)).prepare(artifact)
    with pytest.raises(matrix_delivery.MatrixDeliveryError):
        matrix_delivery.MatrixDownloadPreparer(replace(policy, delivered_bytes=63)).prepare(artifact)


def test_large_hdf5_chunk_rejected_before_decode(tmp_path):
    path = tmp_path/'chunk.h5ad'
    ad.AnnData(sparse.eye(3, format='csr')).write_h5ad(path)
    with h5py.File(path, 'a') as handle:
        handle.create_dataset('uns/large_chunk', shape=(1,), maxshape=(None,), chunks=(1024,), dtype='float64')
    with peak_h5ad.inspection_limits(1024**2, 1024):
        with pytest.raises(ValueError, match='chunk'):
            peak_h5ad.inspect(path)


def test_reference_alias_and_assembly_association_are_exact():
    from agent.application.native_raw_handoff import _context_matches
    from agent.application.resource_continuation import preference_matches
    def row(assembly, identity):
        return dict(species=dict(scientific_name='Homo sapiens', taxonomy_id=9606), assembly=assembly, component_identity_version=2, components=dict(
            reference=dict(sha256=identity), chrom_sizes=dict(sha256=identity+'fai', ordered_sizes=identity+'axis')))
    hg, official, different = row('hg38','194'), row('GRCh38','194'), row('hg38','455')
    context = dict(species='human', assembly='GRCh38', catalog=dict(hg=hg, official=official, different=different))
    assert _context_matches(hg, context)
    assert not _context_matches(different, context)
    assert not _context_matches(hg, dict(context, catalog={}))
    assert not _context_matches(dict(hg, component_identity_version=1), context)
    choices = [dict(label='qualified', aliases=['reviewed-external-name']), dict(label='other', aliases=[])]
    assert preference_matches(dict(literal='reviewed-external-name'), choices, label=lambda row:row['label'],
        identity=lambda row:row['label'], aliases=lambda row:row['aliases']) == choices[:1]


@pytest.mark.parametrize('reason', ['timeout', 'cancel', 'exhaustion'])
def test_native_process_failure_terminates_and_preserves_log(tmp_path, reason):
    import os
    import sys
    from agent.tools._native_process import _run as run
    from agent.tools._cancellation import cancellation_scope, ToolWorkCancelled
    log = tmp_path/'runtime.txt'
    if reason == 'exhaustion':
        # Process failure plumbing, not a hard-memory-limit qualification.
        status = run([sys.executable, '-c', 'raise MemoryError("fixture allocation exhaustion")'],
                     log, environment=dict(os.environ), timeout=10)
        assert status != 0 and b'MemoryError' in log.read_bytes()
    else:
        expected = TimeoutError if reason == 'timeout' else ToolWorkCancelled
        with cancellation_scope(lambda: reason == 'cancel'):
            with pytest.raises(expected):
                run([sys.executable, '-c', 'import time; time.sleep(60)'], log,
                    environment=dict(os.environ), timeout=.05)
    assert log.exists()


def test_unassigned_annotation_palette_retains_missing_category():
    from agent.report.annotation_visualization import _palette
    assert _palette([None, None]) == {None: '#bdbdbd'}


def test_annotation_pending_context_without_files_retains_existing_capture():
    from agent.application.session_state import Interaction
    from agent.application.dialogue_execution import capture_prerequisite, captured_input_group
    pending = dict(origin_turn_id='first', binding_turn_id='first', field='annotation',
        tool='annotate_scATAC_cell_types', resource_context_sha256='a'*64)
    first = Interaction('first', 'Annotate accepted counts', None, 0, {},
        status='clarification', admitted=dict(annotation={}, annotation_preparation='selected-method.v2'), prerequisite=pending)
    reference = capture_prerequisite((first,), None, 0, None, canonical_sources=True)
    assert reference == dict(pending, predecessor_turn_id='first')
    next_turn = Interaction('next', 'The grouping is independent', None, 0, dict(prerequisite=reference))
    assert captured_input_group(next_turn, (first,)) is None


def test_comparison_native_coordinates_require_exact_verified_written_slot(tmp_path):
    value=result();value['source_identity']={'sha256':'b'*64}
    context=SimpleNamespace(validate_anchors=lambda:None,proofs={'native':SimpleNamespace(kind='native_snapatac2_matrix',result_metadata=value)})
    assert cell_result.native_coordinate_source(context,value['matrix_path'],value['matrix_sha256'])==dict(field='X_umap',source_sha256='b'*64)
    assert cell_result.native_coordinate_source(None,value['matrix_path'],value['matrix_sha256']) is None
    assert cell_result.native_coordinate_source(context,value['matrix_path'],'c'*64) is None
    assert cell_result.native_coordinate_source(context,'/wrong/path',value['matrix_sha256']) is None
    value['parameters']['kwargs']['inplace']=False
    with pytest.raises(ValueError):cell_result.native_coordinate_source(context,value['matrix_path'],value['matrix_sha256'])


@pytest.mark.parametrize('selected_first', [False, True])
def test_native_coordinate_lookup_selects_exact_path_among_identical_file_hashes(selected_first):
    selected=result();selected['source_identity']={'sha256':'b'*64}
    other=dict(selected,matrix_path='/other-accepted/result.h5ad',source_identity={'sha256':'c'*64})
    values=[selected,other] if selected_first else [other,selected]
    context=SimpleNamespace(validate_anchors=lambda:None,proofs={str(i):SimpleNamespace(
        kind='native_snapatac2_matrix',result_metadata=value) for i,value in enumerate(values)})
    assert cell_result.native_coordinate_source(context,selected['matrix_path'],selected['matrix_sha256'])==dict(
        field='X_umap',source_sha256='b'*64)
