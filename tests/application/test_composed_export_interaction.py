"""Normal export conversation over real storage and simulated accepted science.

The dependency records and native method metadata are deliberate tiny fixtures,
not scientific qualification. No Planner, native method or paid provider runs.
"""
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from agent.application import (
    ApplicationResult, ApplicationStatus, ArtifactReference, InteractiveAgentApplication, OutputSelection,
)
from agent.application.interactive_schemas import ScientificArtifactHandle
from agent.application.session_state import canonical, digest
from agent.orchestration import verification_authority as authority
from agent.providers import PlanningModelFactoryRegistry
from agent.report import evidence as evidence_owner
from agent.schemas import AgentPlan, AgentRequest, PlanStep, StepExecutionResult, StepOutputRef, StepStatus
from agent.schemas.orchestration import _serialize
from agent.schemas.run_state import (
    PersistedRunState, RecoveryPolicySnapshot, RunLifecycleStatus, ToolRecoveryPolicySnapshot,
    fingerprint_plan, fingerprint_recovery_policy,
)
from agent.web.app import create_app
from test_analysis_export import export_fixture, sha
from test_dialogue_evidence import app, forbid_work, passed
from test_interactive_boundary import PROFILES


REQUEST = 'Give me one Scanpy H5AD with the original measurements, later Leiden labels and earlier UMAP.'


def accepted_chain(app, name, stage, *, source=None):
    """Publish scripted accepted metadata with actual recorded local lineage."""
    tool = 'compute_cell_umap' if stage == 'umap' else 'cluster_cells'
    request = AgentRequest(name, 'Simulated accepted scientific chain.', {})
    _, paths = app._prepare_request(request)
    native = export_fixture(paths.scientific, stage=stage, source=source)
    pca = export_fixture(paths.scientific / 'embedding', stage='embedding', source=native.source)
    embedding=pca.native.parent/'embedding.npy'; np.save(embedding,pca.result.obsm['X_pca'])
    cell_ids=pca.native.parent/'cells.txt'; cell_ids.write_text('\n'.join(pca.result.obs_names)+'\n')
    if stage == 'clustering':
        data = ad.read_h5ad(native.native)
        data.obs['leiden'] = pd.Categorical(['0', '1', '2', '2'])
        data.write_h5ad(native.native)
        native.result = data
    defaults = {str:'fixture', int:2, float:.5, bool:True, dict:{}, list:[]}
    def result(tool, **overrides):
        fields = app.registry.get(tool).result_contract.required_fields
        return {k:defaults[t[0]] for k,t in fields.items()} | overrides
    pca_result = result('pca_embed_cells', status='success', input_path=str(native.source),
        analysis_path=str(pca.native), embedding_path=str(embedding), cell_ids_path=str(cell_ids), analysis_sha256=sha(pca.native),
        source_sha256=sha(native.source), n_cells=4, n_features=3, representation='X_pca', embedding_dim=2,
        embedding_dtype='float32', method='PCA', parameters=dict(pca.result.uns['agent_cell_embedding']['parameters']),
        software_versions={'scanpy':'simulated-embedding-version'}, backend='Scanpy', finite=True,
        cell_order_preserved=True)
    graph = result('build_cell_neighbors', status='success', analysis_path=str(pca.native),
        embedding_path=str(embedding), cell_ids_path=str(cell_ids), n_cells=4, embedding_dim=2)
    selected = result(tool, status='success', analysis_path=str(native.native), analysis_sha256=sha(native.native),
        n_cells=4, n_clusters=3 if stage == 'clustering' else 2, clustering_present=True,
        resolution=.75, random_seed=29, algorithm='leiden', cluster_key='leiden', umap_key='X_umap')
    plan = AgentPlan(name+':plan', name, 'scripted-fixture', (
        PlanStep('pca', 'pca_embed_cells', {'input_path':str(native.source),'output_dir':str(pca.native.parent)}),
        PlanStep('neighbors', 'build_cell_neighbors', {'embedding_path':StepOutputRef('pca','embedding_path'),
            'cell_ids_path':StepOutputRef('pca','cell_ids_path'),'output_dir':str(paths.scientific/'graph')},
                 depends_on=('pca',)),
        PlanStep('result', tool, {'analysis_path':StepOutputRef('neighbors','analysis_path'),'output_dir':str(paths.scientific)},
                 depends_on=('neighbors',))))
    resolved = ({'input_path':str(native.source),'output_dir':str(pca.native.parent)},
        {'embedding_path':str(embedding),'cell_ids_path':str(cell_ids),'output_dir':str(paths.scientific/'graph')},
        {'analysis_path':str(pca.native),'output_dir':str(paths.scientific)})
    steps = tuple(StepExecutionResult(p.step_id, p.tool_name, StepStatus.SUCCEEDED, attempt_count=1,
        result=value, resolved_arguments=args, verification=passed('step',p.step_id),
        started_at='2026-10-24T00:00:00+00:00', finished_at='2026-10-24T00:00:00+00:00', duration_seconds=0.)
        for p,value,args in zip(plan.steps, (pca_result,graph,selected), resolved))
    policies = tuple(ToolRecoveryPolicySnapshot(tool, app.registry.get(tool).recovery_policy_version)
                     for tool in sorted(p.tool_name for p in plan.steps))
    policy = RecoveryPolicySnapshot('fixture',1,policies,fingerprint_recovery_policy('fixture',1,policies))
    app.sessions.start_turn('session',name,request,(OutputSelection('selected','result','analysis_path'),),
        expected_generation=app.sessions.load('session').generation)
    app.sessions.link_run('session',name)
    effective,_ = app._prepare_request(request)
    stored = PersistedRunState(3,0,name+':run',effective,RunLifecycleStatus.SUCCEEDED,
        '2026-10-24T00:00:00+00:00','2026-10-24T00:00:00+00:00',plan=plan,
        plan_fingerprint=fingerprint_plan(plan),recovery_policy_snapshot=policy,
        preflight_verification=passed('plan',plan.plan_id),steps=steps,run_verification=passed('run',plan.plan_id))
    app.run_store.create(stored)
    run = stored.to_run_result()
    payload = dict(schema_version=1, artifact_type=evidence_owner.ANALYSIS_EVIDENCE_ARTIFACT_TYPE,status='success',
        run=dict(run_id=run.run_id,request_id=name,plan_id=plan.plan_id,planner_name=plan.planner_name,
            source_run_status='SUCCEEDED',plan_sha256=digest(plan.to_dict()),source_run_result_sha256=digest(run.to_dict())),
        workflow=dict(ordered_steps=[dict(step_id=p.step_id,tool_name=p.tool_name,depends_on=list(p.depends_on))
            for p in plan.steps]),
        steps=[dict(step_id=s.step_id,tool_name=s.tool_name,attempt_count=1,
            facts={k:_serialize(s.result[k]) for k in evidence_owner._TOOL_PROJECTIONS[s.tool_name].fact_fields},
            verification=dict(passed=True,freshly_verified=True,check_names=['fixture']),
            recovery_identity=app.registry.get(s.tool_name).recovery_policy_version) for s in steps],
        artifacts=[],provenance=dict(trace_sha256=digest([]),evidence_projection_version=1,
            registry_tool_identities=[dict(tool_name=p.tool_name,result_contract=app.registry.get(p.tool_name).result_contract.name,
                recovery_identity=app.registry.get(p.tool_name).recovery_policy_version) for p in plan.steps]))
    evidence = paths.evidence / evidence_owner.ANALYSIS_EVIDENCE_FILENAME
    evidence.write_bytes(canonical(payload))
    report = paths.report / 'fixture.json'; report.write_text('{}')
    completed = ApplicationResult(name,run.run_id,ApplicationStatus.SUCCEEDED,run.status,str(paths.root),run,
        ArtifactReference(evidence_owner.ANALYSIS_EVIDENCE_ARTIFACT_TYPE,str(evidence),digest(payload)),
        report=ArtifactReference('fixture',str(report),digest({})))
    app.sessions._record_result('session',name,completed)
    app.sessions.recover('session',name)
    return native


def case(app, monkeypatch, *, failure=None):
    old = accepted_chain(app,'old-map','umap')
    source=old.source
    if failure=='unqualified_measurement_source':
        source=app.workspace_root/'different-original.h5ad'
        independent=ad.read_h5ad(old.source)
        independent.X.data[0]+=10
        independent.write_h5ad(source)
    new = accepted_chain(app,'new-labels','clustering',source=source)
    original_accepted = authority._accepted
    def fixture_anchor(store, run_id, step_id):
        if run_id in ('old-map:run','new-labels:run') and step_id == 'pca':
            return authority._accepted_step(store,run_id,step_id)[0],None,None,'simulated'
        return original_accepted(store,run_id,step_id)
    monkeypatch.setattr(authority,'_accepted',fixture_anchor)
    guarded = forbid_work(app,monkeypatch)
    models=[]; selected={}
    class Model:
        def __init__(self,profile):
            self.model_id=profile.model_id; self.phases=[]; models.append(self)
        def complete(self,*,prompt,response_schema):
            value=json.loads(prompt)
            if 'turn_schema_version' in value:
                self.phases.append('interpretation')
                groups=value['dialogue']['result_groups']
                labels,= [g for g in groups if g['tool']=='cluster_cells']
                layout,= [g for g in groups if g['tool']=='compute_cell_umap']
                assert any(r['field']=='obsm:X_umap' for r in layout['export_fields'])
                assert any(r['field']=='result:leiden' for r in labels['categorical']['fields'])
                selected.update(measurement=layout['handle'],
                    observations=[dict(output=labels['handle'],field='result:leiden',destination='selected_leiden')],
                    representations=[dict(output=layout['handle'],field='obsm:X_umap',destination='X_selected_umap')])
                if failure=='selected_field_unavailable':
                    selected['observations'][0]['field']='result:unavailable_field'
                elif failure=='destination_collision':
                    selected['observations'][0]['destination']='leiden'
                return json.dumps(dict(turn_schema_version=1,decision=dict(kind='answer_scientific',
                    target=dict(output=labels['handle'],subject=None),comparison=None,focus='question',export=selected)))
            if 'dialogue_schema_version' in value:
                self.phases.append('generation')
                recipe=value['evidence']['delivery']['selected_export']
                assert recipe['actual_download_status']=='NOT_RECORDED'
                if failure is None:
                    assert recipe['status']=='available_for_preparation'
                    assert recipe['contents']['observations'][0]['source']['version']==2
                    assert recipe['contents']['representations'][0]['source']['version']==1
                else:
                    assert recipe['status']=='unavailable' and recipe['reason']==failure
                    assert recipe['message'] and str(app.workspace_root) not in recipe['message']
                return json.dumps(dict(support='insufficient_evidence',paragraphs=[dict(kind='limitation',
                    text='No biological identity is established.',citations=[],comparisons=[])]))
            if 'grounding_review_version' in value:
                self.phases.append('review')
                return json.dumps(dict(verdicts={p['id']:'unsupported' for p in value['paragraphs']}))
            pytest.fail('Export entered Planner or another model phase')
    facade=InteractiveAgentApplication(guarded.workspace_root,model_profiles=PROFILES,default_profile_id='alpha',
        registry=guarded.registry,planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':Model}))
    protected={str(p):p.read_bytes() for root in (guarded._workspace.runs,guarded._workspace.run_state)
               for p in root.rglob('*') if p.is_file()}
    return SimpleNamespace(facade=facade,old=old,new=new,models=models,selected=selected,protected=protected,
        before=guarded.sessions.load('session'))


def unchanged(current):
    state=current.facade._application.sessions.load('session')
    assert (state.turns,state.revisions,state.generation,state.active_revision_id)==(
        current.before.turns,current.before.revisions,current.before.generation,current.before.active_revision_id)
    assert all(Path(path).read_bytes()==raw for path,raw in current.protected.items())
    assert current.facade._matrix_downloads.active_count==current.facade._matrix_downloads._reserved_bytes==0


def test_normal_export_answer_keeps_recipe_through_withholding_reopen_and_real_http_download(app,monkeypatch,tmp_path):
    current=case(app,monkeypatch)
    view=current.facade.submit_turn('session','compose',REQUEST,expected_generation=2)
    assert view.status=='answered' and 'could not ground' in view.response.text,view.to_dict()
    assert 'available for preparation' in view.response.text.lower()
    assert view.export_requested is True
    assert current.models[0].phases==['interpretation','generation','review']
    saved=current.facade._application.sessions.load('session').interactions[-1]
    assert saved.presentation is not None and dict(saved.admitted['export'])
    reference,=current.facade.scientific_artifacts_for_turn('session','compose')
    assert reference.artifact_type=='scientific_composition' and reference.revision_id is None
    assert reference.turn_id=='compose' and reference.size_bytes is None and reference.sha256==sha(current.old.source)
    assert str(tmp_path) not in json.dumps(reference.to_dict())
    unchanged(current)
    def no_provider(profile): pytest.fail('Saved export history constructed a provider')
    restored=InteractiveAgentApplication(current.facade._application.workspace_root,model_profiles=PROFILES,
        default_profile_id='alpha',registry=current.facade._application.registry,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':no_provider}))
    assert restored.turn('session','compose')==view
    assert restored.submit_turn('session','compose',REQUEST,expected_generation=2)==view
    assert restored.scientific_artifacts_for_turn('session','compose')==(reference,)
    with TestClient(create_app(restored)) as client:
        endpoint='/api/v1/sessions/session/turns/compose/scientific-artifacts'
        inventory=client.get(endpoint)
        assert inventory.status_code==200 and inventory.json()==dict(artifacts=[reference.to_dict()])
        assert str(tmp_path) not in inventory.text
        response=client.get(endpoint+'/'+reference.handle)
        assert response.status_code==200,response.text
        assert response.headers['content-disposition']=='attachment; filename="composed-analysis.h5ad"'
        assert response.headers['x-artifact-source-sha256']==reference.sha256
        assert response.headers['x-artifact-content-sha256']==hashlib.sha256(response.content).hexdigest()
        assert response.headers['x-artifact-content-sha256']!=reference.sha256
        assert int(response.headers['content-length'])==len(response.content)
        assert client.get(endpoint.replace('/compose/','/old-map/')+'/'+reference.handle).status_code==404
        assert client.get(endpoint.replace('/session/','/another-session/')+'/'+reference.handle).status_code==404
        exported=ad.read_h5ad(io.BytesIO(response.content))
    original=current.old.original
    assert type(exported.X) is type(original.X)
    for key in ('data','indices','indptr'):
        np.testing.assert_array_equal(getattr(exported.X,key),getattr(original.X,key))
    assert exported.obs_names.equals(original.obs_names) and exported.var_names.equals(original.var_names)
    pd.testing.assert_series_equal(exported.obs['leiden'],original.obs['leiden'])
    pd.testing.assert_series_equal(exported.obs['selected_leiden'],current.new.result.obs['leiden'],check_names=False)
    np.testing.assert_array_equal(exported.obsm['X_selected_umap'],current.old.result.obsm['X_umap'])
    metadata=exported.uns['agent_scientific_export']
    selection=json.loads(metadata['selection_json'])
    assert selection['observations'][0]['destination']=='selected_leiden'
    assert selection['representations'][0]['destination']=='X_selected_umap'
    assert selection['observations'][0]['accepted_output']['revision_id']==current.before.revisions[1].revision_id
    assert selection['representations'][0]['accepted_output']['revision_id']==current.before.revisions[0].revision_id
    assert selection['measurement']['source_sha256']==reference.sha256
    assert metadata['recipe_handle']==reference.handle and metadata['original_source_sha256']==reference.sha256
    assert 'does not issue scientific authority' in metadata['authority_scope']
    assert b'/private/' not in response.content and b'must-not-leak' not in response.content
    assert restored.scientific_artifacts_for_turn('session','compose')==(reference,)
    unchanged(current)


@pytest.mark.parametrize('failure,message',[
    ('selected_field_unavailable','observation field is unavailable'),
    ('destination_collision','destination already exists'),
    ('unqualified_measurement_source','one qualified original measurement source'),
])
def test_exact_but_ineligible_export_preserves_answer_and_discloses_safe_owner_reason(app,monkeypatch,tmp_path,failure,message):
    current=case(app,monkeypatch,failure=failure)
    view=current.facade.submit_turn('session','compose',REQUEST,expected_generation=2)
    assert view.response.kind=='answer' and 'recipe is unavailable' in view.response.text
    assert message in view.response.text and str(tmp_path) not in view.response.text
    state=current.facade._application.sessions.load('session')
    assert state.interactions[-1].presentation is not None and state.interactions[-1].admitted['export']
    assert current.facade.scientific_artifacts_for_turn('session','compose')==()
    assert current.models[0].phases==['interpretation','generation','review']
    unchanged(current)


def test_composed_descriptor_is_safe_immutable_and_old_descriptor_keys_stay_exact():
    old=ScientificArtifactHandle('a'*64,'scientific_integrated','b'*64,'accepted-revision','old-turn',None,
        'integrated-analysis.h5ad','Original data with accepted analysis')
    assert set(old.to_dict())=={'handle','artifact_type','sha256','revision_id','turn_id','size_bytes','filename',
        'label','title','purpose','description'}
    assert 'contents' not in old.to_dict() and old.revision_id=='accepted-revision'
    contents=dict(measurement=dict(source_sha256='b'*64),observations=[],representations=[])
    new=ScientificArtifactHandle('c'*64,'scientific_composition','b'*64,None,'compose',None,
        'composed-analysis.h5ad','Original data with selected accepted results',contents=contents)
    with pytest.raises(TypeError): new.contents['measurement']['source_sha256']='d'*64
    for unsafe in (dict(path='/private/lab/input.h5ad'),dict(credentials='must-not-expose')):
        with pytest.raises(ValueError):
            replace(new,contents=unsafe)
    with pytest.raises(ValueError): replace(new,revision_id='fake-scientific-revision')
    with pytest.raises(ValueError): replace(old,revision_id=None)
