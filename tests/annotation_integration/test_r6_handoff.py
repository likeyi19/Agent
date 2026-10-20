"""Real Scanpy on simulated counts; annotation component explicitly simulated."""
from dataclasses import asdict
import io
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from fastapi.testclient import TestClient

from agent.application import OutputSelection, ResearchAgentApplication
from agent.orchestration import AgentRequest, LLMPlanner
from agent.tools.analysis import annotation_contract as c, annotation_input, peak_h5ad
from agent.tools.analysis import scatac_annotation as owner, marker_annotation as science
from test_external_entry import external_case, case, wait_turn, wire
from test_conversational_handoff import setup, HandoffModel, send
from test_pca_continuation import Model as PCAModel


def test_native_encodings_and_private_metadata_export(external_case):
    args, calls = external_case
    data = ad.read_h5ad(args['source_path'])
    data.obs['nullable'] = pd.array([1] + [None]*(data.n_obs-1), dtype='Int64')
    for i in range(70):
        data.obs[f'metadata_{i}'] = i
    data.uns['analysis_log'] = '/private/research/pipeline.py'
    data.uns['numeric_cube'] = np.ones((2, 2, 2))
    data.write_h5ad(args['source_path'])
    with h5py.File(args['source_path'], 'a') as f:
        f.attrs['ordinary_producer'] = 'AnnData user workflow'
        f['X/data'].attrs['units'] = 'raw cut-site counts'
    args['source_sha256'] = science.sha256(args['source_path'])
    observation = annotation_input.observe(args['source_path'], args['source_sha256'])
    assert observation['interface_version'] == 2
    result = owner.execute_annotation(args)
    assert len(calls) == 1
    from agent.application.annotation_delivery import create_annotation_export
    manifest = c.load_manifest(result['manifest_path'], result['manifest_sha256'])
    output = io.BytesIO()
    path = Path(result['annotated_h5ad_path'])
    with path.open('rb') as source:
        create_annotation_export(source, output, manifest,
            source_sha256=result['annotated_h5ad_sha256'], source_size_bytes=path.stat().st_size,
            manifest_sha256=result['manifest_sha256'], max_bytes=10*1024**2)
    output.seek(0)
    delivered = ad.read_h5ad(output)
    assert 'analysis_log' not in delivered.uns
    assert ad.read_h5ad(path).uns['analysis_log'] == data.uns['analysis_log']
    assert delivered.obs['nullable'].equals(data.obs['nullable'])
    assert (delivered.X != data.X).nnz == 0
    assert np.array_equal(delivered.uns['numeric_cube'], data.uns['numeric_cube'])


@pytest.mark.parametrize('indirect', ['external_link', 'soft_link', 'virtual'])
def test_indirect_hdf5_storage_remains_rejected(tmp_path, indirect):
    path = tmp_path/'source.h5ad'
    ad.AnnData(sparse.eye(3, format='csr')).write_h5ad(path)
    with h5py.File(path, 'a') as f:
        if indirect == 'external_link':
            f['uns/indirect'] = h5py.ExternalLink('/private/other.h5', '/secret')
        elif indirect == 'soft_link':
            f['uns/indirect'] = h5py.SoftLink('/X')
        else:
            layout = h5py.VirtualLayout(shape=(3,), dtype='float64')
            layout[:] = h5py.VirtualSource('/private/other.h5', 'secret', shape=(3,))
            f.create_virtual_dataset('uns/indirect', layout)
    with pytest.raises(ValueError, match='Indirect|layout'):
        peak_h5ad.inspect(path)


def test_omitted_count_slot_and_no_native_recovery_replay(external_case, monkeypatch):
    args, calls = external_case
    args.pop('count_slot')
    result = owner.execute_annotation(args, 'default')
    assert c.load_manifest(result['manifest_path'])['arguments']['count_slot'] == 'X'
    assert len(calls) == 1
    from agent.tools.analysis import external_marker_annotation
    monkeypatch.setattr(external_marker_annotation, 'annotate_cell_groups',
                        lambda **kw: pytest.fail('Recovery executed native science'))
    assert owner.recover_annotation(args, 'default') == result
    path = Path(result['annotated_h5ad_path'])
    path.write_bytes(path.read_bytes() + b'tampering')
    with pytest.raises(ValueError):
        owner.recover_annotation(args, 'default')


@pytest.mark.parametrize('different_source', [False, True])
def test_pca_clustering_to_annotation_without_reupload(external_case, tmp_path, monkeypatch, different_source):
    args, calls = external_case
    source = Path(args['source_path'])
    data = ad.AnnData(sparse.csr_matrix(np.random.default_rng(4).poisson(.4, (24, 71))),
        obs=pd.DataFrame(index=[f'cell{i}' for i in range(24)]),
        var=pd.DataFrame(index=[f'chr1:{i*2}-{i*2+1}' for i in range(71)]))
    data.write_h5ad(source)
    root = tmp_path/'web'
    upstream = ResearchAgentApplication(root, planner=LLMPlanner(PCAModel()))
    upstream.sessions.create('upload')
    state = upstream.sessions.run('upload', 'clusters',
        AgentRequest('pca-first', 'PCA, clusters and UMAP.', dict(input_path=str(source))),
        (OutputSelection('umap', 'umap', 'analysis_path'),), expected_generation=0)
    assert state.generation == 1
    if different_source:
        changed = data.copy(); changed.X.data[0] += 1
        changed.write_h5ad(source)
    before = {p: p.read_bytes() for p in upstream._workspace.runs.rglob('*') if p.is_file()}
    import scanpy as sc
    from agent.tools.analysis import pca_embedding, neighbors_publication
    def forbidden(*args, **kwargs):
        pytest.fail('Upstream scientific computation repeated')
    for module, name in ((sc.pp, 'pca'), (sc.pp, 'neighbors'), (sc.tl, 'leiden'),
                         (sc.tl, 'umap'), (pca_embedding, '_verify_result'),
                         (neighbors_publication, '_verify')):
        monkeypatch.setattr(module, name, forbidden)

    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        if 'turn_schema_version' in value:
            choices = value['dialogue']['annotation_group_results']
            assert len(choices) == 1
            from agent.application.raw_handoff import AnnotationDeclaration
            if value['dialogue'].get('pending_prerequisite'):
                return json.dumps(dict(turn_schema_version=1, decision=dict(
                    kind='answer_roles', pending='@roles', roles=[], arguments=[],
                    annotation=asdict(AnnotationDeclaration(groups_source='source.1',
                        grouping_provenance='Independent simulated groups supplied for these counts')))))
            declaration = AnnotationDeclaration(source='source.0', groups_result=choices[0]['handle'],
                species='human', assembly='hg38', measurement=c.EXTERNAL_SEMANTICS,
                context='synthetic context')
            return json.dumps(dict(turn_schema_version=1, decision=dict(kind='request_roles',
                target='annotate_scATAC_cell_types', required_roles=[], roles=[], arguments=[],
                annotation=asdict(declaration))))
        if 'semantic_prompt_version' in value:
            plan = wire()
            plan['decision']['steps'][0]['sources'] = [
                row for row in plan['decision']['steps'][0]['sources'] if row['target'] != 'count_slot']
            return json.dumps(plan)
        return super(HandoffModel, self).complete(prompt=prompt, response_schema=response_schema)
    monkeypatch.setattr(HandoffModel, 'complete', complete)
    app, web = setup(args, root, [])
    with TestClient(web) as client:
        upload = client.put('/api/v1/sessions/upload/uploads/matrix?filename=counts.h5ad&input_type=h5ad',
            content=source.read_bytes(), headers={'Content-Type': 'application/octet-stream'})
        assert upload.status_code == 201, upload.text
        # Only original counts are attached; grouping stays inside the application.
        done = send(client, 'annotate', 'Annotate the original cut-site counts with the current Leiden groups.',
                    [upload.json()['resource_id']], generation=1)
        if different_source:
            assert done['status'] == 'clarification', done
            assert 'exact source identity' in done['response']['text']
            assert not calls
            assert app._application.sessions.load('upload').generation == 1
            assert client.get('/api/v1/sessions/upload').status_code == 200
            assert all(p.read_bytes() == value for p, value in before.items())
            content = pd.DataFrame(dict(cell_id=data.obs_names,
                group=['first']*12+['second']*12)).to_csv(sep='\t', index=False).encode()
            groups = client.put('/api/v1/sessions/upload/uploads/groups?filename=groups.tsv&input_type=attachment',
                content=content, headers={'Content-Type':'application/octet-stream'})
            assert groups.status_code == 201, groups.text
            done = send(client, 'correct-groups',
                'Use the attached independent simulated groups for these counts.',
                [groups.json()['resource_id']], generation=1)
            assert done['status'] == 'succeeded', done
            assert len(calls) == 1
            assert all(p.read_bytes() == value for p, value in before.items())
            return
        assert done['status'] == 'succeeded', done
        run = app._application.run_store.load(done['run_id'])
        assert len(calls) == 1
        bound = run.steps[0].resolved_arguments
        assert 'annotation_groups' in bound['groups_path']
        assert 'count_slot' not in bound
        clustered = upstream.run_store.load('pca-first:run').steps[2].result
        expected = ad.read_h5ad(clustered['analysis_path'])
        actual = ad.read_h5ad(run.steps[0].result['annotated_h5ad_path'])
        assert actual.obs_names.equals(data.obs_names)
        assert actual.obs['annotation_group'].astype(str).tolist() == expected.obs['leiden'].astype(str).tolist()
        assert all(p.read_bytes() == value for p, value in before.items())
        assert client.get('/api/v1/sessions/upload').status_code == 200
        from agent.application.annotation_groups import project
        sessions = app._application.sessions
        sessions._interaction_session_id = 'upload'
        interaction = sessions.load('upload').interactions[-1]
        selected = interaction.admitted['annotation']['groups_result']
        changed = data.copy(); changed.X.data[0] += 1
        different = tmp_path/'same-cells-different-measurements.h5ad'; changed.write_h5ad(different)
        with pytest.raises(ValueError, match='exact source identity'):
            project(sessions, interaction, selected,
                dict(source_path=str(different), source_sha256=science.sha256(different)))
        with pytest.raises(ValueError, match='offered'):
            project(sessions, interaction, 'groups.' + '0'*64,
                dict(source_path=str(source), source_sha256=science.sha256(source)))
        # Exact accepted bytes, not equal cell counts, authorize the group file.
        path = Path(clustered['analysis_path']); path.write_bytes(path.read_bytes() + b'tampered')
        with pytest.raises(ValueError, match='identity'):
            project(sessions, interaction, selected,
                dict(source_path=str(source), source_sha256=science.sha256(source)))
