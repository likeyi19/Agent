"""UA3.6.2c: real conversational science, shared presentation and transport.

The scripted provider supplies typed decisions only. Scanpy and every accepted
scientific/evidence publication use their existing registered implementations.
"""
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import inspect
import io
import json
from pathlib import Path
import sys
import threading

import anndata as ad
from fastapi.testclient import TestClient
import numpy as np
import pytest

from agent.application import InteractiveAgentApplication
from agent.application.matrix_delivery import MatrixPrivacyError, validate_matrix_h5ad
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas import AgentError, ErrorCategory, RunLifecycleStatus
from agent.web.app import create_app
from helpers import wait_turn
from rich_helpers import PROFILES, rich_harness

sys.path.insert(0, str(Path(__file__).parents[1] / 'authority_dag'))
from test_processed_prior_outputs import processed_source, preserved_run_bytes
from test_conversational_processed_continuation import MODIFICATION, PROFILE, interactive


@contextmanager
def processed_delivery_work():
    """Observe actual producer/owner bodies in Web workers as well as main."""
    import scanpy as sc
    from agent.orchestration import LLMPlanner, PlanExecutor, semantic_compiler, verifier
    from agent.tools.analysis import embedding_analysis, neighbors_publication
    from agent.tools.data import fastq_verification_authority
    from agent.tools.models import epizoo
    from agent.application import matrix_delivery, turn_decisions
    from agent.tools.analysis import neighbors_delivery

    functions = {
        'production.neighbors': sc.pp.neighbors,
        'production.leiden': sc.tl.leiden,
        'production.umap': inspect.unwrap(sc.tl.umap),
        'production.epizoo': epizoo.embed_cells,
        'owner.neighbors': neighbors_publication._verify,
        'verification.neighbors': verifier._verify_neighbors,
        'verification.clustering': verifier._verify_clustering,
        'verification.umap': verifier._verify_umap,
        'source.capture': neighbors_publication.capture_inputs,
        'source.check': neighbors_publication.check_sources,
        'source.embeddings': embedding_analysis._load_embeddings,
        'source.cell_ids': embedding_analysis._load_cell_ids,
        'validation.graph_structure': embedding_analysis._validate_neighbors_artifact,
        'artifact.analysis_reads': embedding_analysis._read_analysis,
        'integrity.files': fastq_verification_authority.file_closure,
        'compiler': semantic_compiler.compile_semantic_plan,
        'preflight': PlanExecutor.preflight,
        'interpreter': turn_decisions.interpret,
        'planner': LLMPlanner._plan_detailed,
        'delivery.prepare': matrix_delivery.MatrixDownloadPreparer.prepare,
        'delivery.export': neighbors_delivery.create_neighbors_export,
        'integrity.export_layout': neighbors_delivery._inspect,
        'integrity.open_source': matrix_delivery._open_matrix_source,
    }
    codes = {function.__code__: name for name, function in functions.items()}
    counts, lock = Counter(), threading.Lock()
    previous_main, previous_thread = sys.getprofile(), threading.getprofile()

    def observe(frame, event, arg):
        if event == 'call' and frame.f_code in codes:
            with lock:
                counts[codes[frame.f_code]] += 1

    sys.setprofile(observe)
    threading.setprofile(observe)
    try:
        yield counts
    finally:
        sys.setprofile(previous_main)
        threading.setprofile(previous_thread)


def processed_delivery_harness(tmp_path, *, retain=True):
    """Reusable native-browser fixture, with a genuine accepted graph source."""
    source = processed_source.__wrapped__(Path(tmp_path))
    facade, models = interactive(source, retain=retain)
    return source, facade, models


def revision_path(revision, session='analysis'):
    return f'/api/v1/sessions/{session}/revisions/{revision}'


def no_scientific_or_planning_work(counts):
    forbidden = ('production.', 'owner.', 'source.')
    assert not {name: value for name, value in counts.items()
                if value and (name.startswith(forbidden) or name in ('interpreter', 'planner', 'compiler', 'preflight'))}


def field_facts(evidence):
    return {fact['field']: fact for fact in evidence['facts']}


def safe_json(value, private_root):
    assert str(private_root) not in json.dumps(value, ensure_ascii=False)


def file_bytes(root):
    return {str(path): path.read_bytes() for path in Path(root).rglob('*') if path.is_file()}


def assert_graph_export(response, descriptor, source):
    assert response.status_code == 200, response.text[:200] if response.status_code != 200 else ''
    assert response.headers['content-disposition'] == 'attachment; filename="neighbors.h5ad"'
    assert response.headers['cache-control'] == 'no-store'
    assert response.headers['x-artifact-source-sha256'] == descriptor['sha256']
    assert response.headers['x-artifact-content-sha256'] == hashlib.sha256(response.content).hexdigest()
    assert response.headers['x-artifact-content-sha256'] != descriptor['sha256']
    assert str(source.app.workspace_root).encode() not in response.content
    exported = ad.read_h5ad(io.BytesIO(response.content))
    original = ad.read_h5ad(source.run.steps[0].result['analysis_path'])
    assert tuple(exported.obs_names) == source.ids
    np.testing.assert_array_equal(exported.obsm['X_epizoo'], source.values)
    for key in ('connectivities', 'distances'):
        assert (exported.obsp[key] != original.obsp[key]).nnz == 0


def test_ordinary_processed_result_presentation_downloads_history_and_fresh_service(tmp_path):
    source, service, models = processed_delivery_harness(tmp_path)
    original_revision = source.session.active_revision_id
    original_publication = preserved_run_bytes(source)
    with processed_delivery_work() as production:
        with TestClient(create_app(service)) as client:
            admitted = client.post('/api/v1/sessions/analysis/turns', json=dict(
                turn_id='modify', expected_generation=1, utterance=MODIFICATION))
            assert admitted.status_code == 202, admitted.text
            turn = wait_turn(client, 'modify', session='analysis', timeout=60)
            assert turn['status'] == 'succeeded', turn
    assert production['production.leiden'] == production['production.umap'] == 1
    assert production['production.neighbors'] == production['owner.neighbors'] == 0
    assert production['production.epizoo'] == 0
    assert production['interpreter'] == production['planner'] == 1
    assert not {name: value for name, value in production.items() if name.startswith('source.')}
    assert len(models) == 1 and len(models[0].calls) == 4
    model_calls = len(models[0].calls)
    child_revision = turn['revision_id']
    state = service._application.sessions.load('analysis')
    assert state.generation == 2 and len(state.revisions) == 2
    assert state.revisions[0] == source.session.revisions[0]
    run = service._application.run_store.load(state.turn('modify').run_id)
    assert run.lifecycle_status is RunLifecycleStatus.SUCCEEDED
    assert run.steps[0].resolved_arguments['resolution'] == .5
    assert all(step.verification.artifact_authority is None for step in run.steps)
    before_reads = file_bytes(source.app.workspace_root)
    with processed_delivery_work() as reads:
        with TestClient(create_app(service)) as client:
            path = revision_path(child_revision)
            result = client.get(path)
            assert result.status_code == 200, result.text
            view = result.json()
            assert view['parent_revision_id'] == original_revision and view['is_active']
            assert view['retained_outputs'] == ['neighbors']
            assert [step['tool_name'] for step in view['steps']] == ['cluster_cells', 'compute_cell_umap']
            evidence_response = client.get(path + '/evidence', params={'output_name': 'clusters'})
            assert evidence_response.status_code == 200, evidence_response.text
            evidence = evidence_response.json()
            assert evidence['status'] == 'available'
            facts = field_facts(evidence)
            assert facts['clustering_parameters']['value']['resolution'] == .5
            assert facts['clustering_resolution_origin']['value'] == 'explicit_execution_argument'
            assert facts['explicit_user_parameters']['value']['parameters'] == {'resolution': 'explicit_user_instruction'}
            assert evidence['field_labels']['clustering_parameters']
            artifacts = client.get(path + '/artifacts').json()['artifacts']
            figures = [entry for entry in artifacts if entry['artifact_type'] == 'analysis_figure']
            assert figures, 'The real accepted UMAP should retain its existing reviewed PNG.'
            for figure in figures:
                png = client.get(path + '/artifacts/' + figure['handle'])
                assert png.status_code == 200 and png.content.startswith(b'\x89PNG\r\n\x1a\n')
                assert hashlib.sha256(png.content).hexdigest() == figure['sha256']
            report = next(entry for entry in artifacts if entry['artifact_type'] == 'analysis_report')
            rendered = client.get(path + '/artifacts/' + report['handle'])
            assert rendered.status_code == 200 and str(tmp_path) not in rendered.text
            inventory = client.get(path + '/scientific-artifacts').json()['artifacts']
            assert len(inventory) == 1
            graph = inventory[0]
            assert graph['artifact_type'] == 'scientific_analysis'
            assert graph['filename'] == 'neighbors.h5ad'
            assert graph['label'] == 'Cell neighbors — safe export'
            assert graph['revision_id'] == child_revision and graph['turn_id'] == 'modify'
            assert len(evidence['source']['prior_outputs']) == 1
            assert evidence['source']['prior_outputs'][0]['run_id'] == source.run.run_id
            assert client.get('/api/v1/sessions/analysis/turns/modify/scientific-artifacts').json() == {'artifacts': []}
            downloaded = client.get(path + '/scientific-artifacts/' + graph['handle'])
            assert_graph_export(downloaded, graph, source)
            historical_path = revision_path(original_revision)
            historical = client.get(historical_path).json()
            assert not historical['is_active'] and historical['turn_id'] == 'original'
            old_graph = client.get(historical_path + '/scientific-artifacts').json()['artifacts'][0]
            assert old_graph['handle'] != graph['handle']
            assert_graph_export(client.get(historical_path + '/scientific-artifacts/' + old_graph['handle']), old_graph, source)
            assert client.get(path + '/scientific-artifacts/' + old_graph['handle']).status_code == 404
            refreshed = client.get('/api/v1/sessions/analysis').json()
            assert refreshed['active_revision_id'] == child_revision and refreshed['generation'] == 2
            safe_json([view, evidence, inventory, historical, refreshed], tmp_path)
        assert service._application.sessions.load('analysis') == state
        assert file_bytes(source.app.workspace_root) == before_reads
        assert service._matrix_downloads.active_count == 0
        assert len(models[0].calls) == model_calls

        def forbidden_factory(profile):
            raise AssertionError('Read-only result restoration must not construct a provider.')
        reopened = InteractiveAgentApplication(source.app.workspace_root, model_profiles=(PROFILE,),
            default_profile_id=PROFILE.profile_id,
            planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': forbidden_factory}))
        with TestClient(create_app(reopened)) as client:
            assert client.get('/api/v1/sessions/analysis').json() == refreshed
            assert client.get(path).json() == view
            assert client.get(path + '/evidence', params={'output_name': 'clusters'}).json() == evidence
            assert_graph_export(client.get(historical_path + '/scientific-artifacts/' + old_graph['handle']), old_graph, source)
            switched = client.post('/api/v1/sessions/analysis/activate', json=dict(
                turn_id='back', expected_generation=2, revision_id=original_revision))
            assert switched.status_code == 200
            assert switched.json()['active_revision_id'] == original_revision
            assert client.get(path).json()['is_active'] is False
            assert_graph_export(client.get(path + '/scientific-artifacts/' + graph['handle']), graph, source)
    no_scientific_or_planning_work(reads)
    assert reads['delivery.prepare'] > 0 and reads['integrity.export_layout'] > 0
    assert all(path.read_bytes() == data for path, data in original_publication.items())
    witness = dict(source_revision=original_revision, child_revision=child_revision,
        seed_counts=dict(source.counts), continuation_counts=dict(production), read_counts=dict(reads),
        provider_completions=model_calls, graph_source_sha256=graph['sha256'],
        export_sha256=downloaded.headers['x-artifact-content-sha256'],
        evidence=evidence, revision=view)
    (tmp_path / 'unified-result-delivery-witness.json').write_text(
        json.dumps(witness, indent=2, sort_keys=True, ensure_ascii=False), encoding='utf-8')
    print('unified processed result counts:', json.dumps({
        'continuation': dict(production), 'read_only': dict(reads), 'provider_completions': model_calls}, sort_keys=True))


def test_raw_results_use_same_evidence_transport_and_historical_revision_access(tmp_path):
    fixture = rich_harness(tmp_path, managed_matrix=True, managed_tables=True)
    try:
        old_revision = fixture.initial_revision
        initial_work = fixture.work.snapshot()
        with TestClient(create_app(fixture.service, input_sets=fixture.input_sets)) as client:
            old_path = revision_path(old_revision)
            old_inventory = client.get(old_path + '/scientific-artifacts').json()['artifacts']
            assert {entry['filename'] for entry in old_inventory} == {
                'matrix.h5ad', 'barcodes.tsv.gz', 'lengths.tsv.gz', 'decisions.tsv.gz', 'selected.tsv.gz'}
            old_payloads = {entry['handle']: client.get(old_path + '/scientific-artifacts/' + entry['handle']).content
                            for entry in old_inventory}
            admitted = client.post('/api/v1/sessions/analysis/turns', json=dict(
                turn_id='selection-followup', expected_generation=1, utterance='Set minimum depth to 1.'))
            assert admitted.status_code == 202, admitted.text
            turn = wait_turn(client, 'selection-followup', session='analysis', timeout=60)
            assert turn['status'] == 'succeeded', turn
            child_revision = turn['revision_id']
            path = revision_path(child_revision)
            view = client.get(path).json()
            assert view['parent_revision_id'] == old_revision and view['is_active']
            assert [step['tool_name'] for step in view['steps']] == ['select_scATAC_cells']
            assert set(view['outputs']) == {'selection', 'fragments', 'qc'}
            evidence = client.get(path + '/evidence', params={'output_name': 'selection'}).json()
            assert evidence['status'] == 'available'
            facts = field_facts(evidence)
            assert facts['selection_threshold_origins']['value']['min_qc_fragment_records'] == 'explicit_execution_argument'
            assert 'existing_owner_default' in facts['selection_threshold_origins']['value'].values()
            assert evidence['field_labels']['n_selected']
            child_inventory = client.get(path + '/scientific-artifacts').json()['artifacts']
            assert {entry['filename'] for entry in child_inventory} == {
                'barcodes.tsv.gz', 'lengths.tsv.gz', 'decisions.tsv.gz', 'selected.tsv.gz'}
            assert all(entry['revision_id'] == child_revision for entry in child_inventory)
            after_execution = fixture.work.snapshot()
            assert {key: value - initial_work.get(key, 0) for key, value in after_execution.items()
                    if value != initial_work.get(key, 0)} == {'production.selection': 1, 'owner.selection': 1}
            model_calls = sum(len(model.calls) for model in fixture.models)
            state = fixture.application.sessions.load('analysis')
            before_reads = file_bytes(fixture.application.workspace_root)
            for entry in old_inventory + child_inventory:
                entry_path = revision_path(entry['revision_id'])
                download = client.get(entry_path + '/scientific-artifacts/' + entry['handle'])
                assert download.status_code == 200, download.text
                assert hashlib.sha256(download.content).hexdigest() == entry['sha256']
                assert len(download.content) == entry['size_bytes']
                if entry['handle'] in old_payloads:
                    assert download.content == old_payloads[entry['handle']]
            matrix = next(entry for entry in old_inventory if entry['filename'] == 'matrix.h5ad')
            assert client.get(path + '/scientific-artifacts/' + matrix['handle']).status_code == 404
            refreshed = client.get('/api/v1/sessions/analysis').json()
            assert refreshed['active_revision_id'] == child_revision
            assert client.get(old_path).json()['is_active'] is False
            safe_json([view, evidence, old_inventory, child_inventory, refreshed], tmp_path)
        assert fixture.work.snapshot() == after_execution
        assert sum(len(model.calls) for model in fixture.models) == model_calls
        assert fixture.application.sessions.load('analysis') == state
        assert file_bytes(fixture.application.workspace_root) == before_reads

        def forbidden_factory(profile):
            raise AssertionError('Raw historical presentation must not construct a provider.')
        reopened = InteractiveAgentApplication(fixture.application.workspace_root, model_profiles=PROFILES,
            default_profile_id='alpha', planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': forbidden_factory}))
        with TestClient(create_app(reopened)) as client:
            assert client.get(path).json() == view
            assert client.get(old_path + '/scientific-artifacts').json()['artifacts'] == old_inventory
            assert client.get(old_path + '/scientific-artifacts/' + matrix['handle']).content == old_payloads[matrix['handle']]
        assert fixture.work.snapshot() == after_execution
        assert fixture.application.sessions.load('analysis') == state
        assert reopened._matrix_downloads.active_count == 0
        (tmp_path / 'unified-raw-result-delivery-witness.json').write_text(json.dumps(dict(
            old_revision=old_revision, child_revision=child_revision, evidence=evidence,
            initial_counts=initial_work, final_counts=after_execution, inventory=child_inventory),
            indent=2, sort_keys=True), encoding='utf-8')
    finally:
        fixture.close()


@pytest.mark.parametrize('attack', ['bytes', 'manifest', 'symlink', 'forged', 'wrong_revision', 'wrong_session'])
def test_processed_delivery_rejects_mutation_and_wrong_identity_without_science(tmp_path, attack):
    source, service, models = processed_delivery_harness(tmp_path)
    revision = source.session.active_revision_id
    with TestClient(create_app(service)) as client:
        path = revision_path(revision)
        descriptor = client.get(path + '/scientific-artifacts').json()['artifacts'][0]
        handle = descriptor['handle']
        payload = Path(source.run.steps[0].result['analysis_path'])
        if attack == 'bytes':
            payload.write_bytes(payload.read_bytes() + b'changed accepted bytes')
        elif attack == 'manifest':
            manifest = Path(source.run.steps[0].result['manifest_path'])
            manifest.write_bytes(manifest.read_bytes() + b' ')
        elif attack == 'symlink':
            elsewhere = payload.with_name('private-graph.h5ad')
            payload.rename(elsewhere)
            payload.symlink_to(elsewhere)
        elif attack == 'forged':
            handle = '0' * 64
        elif attack == 'wrong_revision':
            path = revision_path('not-offered')
        else:
            source.app.sessions.create('other')
            path = revision_path(revision, 'other')
        before = source.app.sessions.load('analysis')
        with processed_delivery_work() as reads:
            rejected = client.get(path + '/scientific-artifacts/' + handle)
        assert rejected.status_code == 404, rejected.text
        assert rejected.headers['content-type'].startswith('application/json')
        assert 'content-disposition' not in rejected.headers and str(tmp_path) not in rejected.text
        assert service._matrix_downloads.active_count == 0
        assert source.app.sessions.load('analysis') == before and not models
        no_scientific_or_planning_work(reads)


def test_processed_original_h5ad_is_not_a_matrix_privacy_exception(tmp_path):
    source, service, models = processed_delivery_harness(tmp_path)
    original = Path(source.run.steps[0].result['analysis_path'])
    from agent.tools.analysis.embedding_analysis import PROVENANCE_KEY
    artifact = ad.read_h5ad(original)
    assert str(tmp_path) in json.dumps(artifact.uns[PROVENANCE_KEY], default=str)
    with pytest.raises(MatrixPrivacyError):
        validate_matrix_h5ad(original, 'scatac-cell-by-ccre.v1')
    with processed_delivery_work() as reads:
        entry = service.scientific_artifact_handles('analysis', source.session.active_revision_id)[0]
        assert entry.artifact_type == 'scientific_analysis'
        with service.prepare_scientific_artifact('analysis', source.session.active_revision_id, entry.handle) as exported:
            data = b''.join(exported.iter_chunks())
            assert str(tmp_path).encode() not in data and hashlib.sha256(data).hexdigest() != entry.sha256
    no_scientific_or_planning_work(reads)
    assert not models


def test_historical_dependency_alone_does_not_authorize_child_download_attachment(tmp_path):
    source, service, models = processed_delivery_harness(tmp_path, retain=False)
    parent_revision = source.session.active_revision_id
    turn = service.submit_turn('analysis', 'modify', MODIFICATION, expected_generation=1)
    assert turn.status == 'succeeded'
    child_revision = turn.revision_id
    state = service._application.sessions.load('analysis')
    before = file_bytes(source.app.workspace_root)
    model_calls = len(models[0].calls)
    with processed_delivery_work() as reads:
        with TestClient(create_app(service)) as client:
            parent = revision_path(parent_revision)
            child = revision_path(child_revision)
            graph = client.get(parent + '/scientific-artifacts').json()['artifacts'][0]
            assert client.get(child + '/scientific-artifacts').json() == {'artifacts': []}
            assert client.get('/api/v1/sessions/analysis/turns/modify/scientific-artifacts').json() == {'artifacts': []}
            rejected = client.get(child + '/scientific-artifacts/' + graph['handle'])
            assert rejected.status_code == 404 and 'content-disposition' not in rejected.headers
            assert_graph_export(client.get(parent + '/scientific-artifacts/' + graph['handle']), graph, source)
    assert service._application.sessions.load('analysis') == state
    assert file_bytes(source.app.workspace_root) == before and len(models[0].calls) == model_calls
    no_scientific_or_planning_work(reads)


@pytest.mark.parametrize('attack', ['missing_authority', 'corrupt_authority', 'wrong_output', 'wrong_step', 'failed_run', 'incomplete_run'])
def test_processed_delivery_rejects_unqualified_accepted_metadata_without_reconstruction(tmp_path, monkeypatch, attack):
    """Loader views reject invalid qualification without editing accepted history."""
    source, service, models = processed_delivery_harness(tmp_path)
    revision = source.session.active_revision_id
    descriptor = service.scientific_artifact_handles('analysis', revision)[0]
    before = file_bytes(source.app.workspace_root)
    if attack in {'wrong_output', 'wrong_step'}:
        original_load = service._load
        state = original_load('analysis')
        original_revision = state.revisions[0]
        changed_member = ({'output_key': 'not_selected_field'} if attack == 'wrong_output'
                          else {'step_id': 'not_accepted_step'})
        output = replace(original_revision.outputs[0], **changed_member)
        wrong_revision = replace(original_revision, outputs=(output,))
        origin = state.turn(original_revision.turn_id)
        wrong_turn = replace(origin, selections=(replace(origin.selections[0], **changed_member),))
        changed = replace(state, revisions=(wrong_revision,), turns=(wrong_turn,))
        monkeypatch.setattr(service, '_load', lambda session: changed if session == 'analysis' else original_load(session))
    else:
        original_load = service._application.run_store.load
        step = source.run.steps[0]
        if attack == 'failed_run':
            changed = replace(source.run, lifecycle_status=RunLifecycleStatus.FAILED,
                errors=(AgentError(ErrorCategory.VERIFICATION_ERROR, 'TEST_REJECTED_RUN', 'Rejected fixture run.'),))
        elif attack == 'incomplete_run':
            changed = replace(source.run, lifecycle_status=RunLifecycleStatus.RUNNING, run_verification=None)
        else:
            authority = None
            if attack == 'corrupt_authority':
                authority = step.verification.to_dict()['artifact_authority']
                authority['resources']['result_metadata']['n_cells'] = 10000
            changed_step = replace(step, verification=replace(step.verification, artifact_authority=authority))
            changed = replace(source.run, steps=(changed_step,))
        monkeypatch.setattr(service._application.run_store, 'load', lambda run_id: changed
            if run_id == source.run.run_id else original_load(run_id))
    with processed_delivery_work() as reads:
        with TestClient(create_app(service)) as client:
            response = client.get(revision_path(revision) + '/scientific-artifacts/' + descriptor.handle)
    assert response.status_code == 404, response.text
    assert 'content-disposition' not in response.headers and str(tmp_path) not in response.text
    assert file_bytes(source.app.workspace_root) == before and not models
    assert service._matrix_downloads.active_count == 0
    no_scientific_or_planning_work(reads)
