"""Bounded real ICTA-4B supplied source grouping, with no initial clustering."""
import json
import os
from pathlib import Path
import shutil
import pytest
from agent.schemas import AgentRequest, AgentPlan, PlanStep
from agent.schemas.orchestration import _serialize
from agent.tools._file_identity import file_sha256
from agent.tools.analysis import hierarchical_annotation as owner


def test_supplied_original_groups_native_multilevel_and_delivery(tmp_path, monkeypatch):
    if os.environ.get('RUN_RAW_NATIVE_QUALIFICATION') != '1' or not os.environ.get('ICTA4B_ACCEPTED_FIXTURE'):
        pytest.skip('Explicit bounded native qualification')
    import anndata as ad
    import pandas as pd
    from agent.application import ResearchAgentApplication, InteractiveAgentApplication, OutputSelection
    from agent.orchestration import FileRunStore
    from agent.orchestration.prior_outputs import bind_output, validate_binding
    from agent.application.matrix_delivery import MatrixDownloadPreparer
    previous = Path(os.environ['ICTA4B_ACCEPTED_FIXTURE'])
    old = FileRunStore(previous/'run_state').load('continued:run')
    args = dict(old.plan.steps[0].arguments)
    existing = os.environ.get('ICTA4B_SUPPLIED_FIXTURE')
    if existing:
        tmp_path = Path(existing)
    source = tmp_path/'uploaded-peaks.h5ad'
    matrix = ad.read_h5ad(args['input_path'])
    matrix.obs['researcher_groups'] = pd.Categorical(['parent A']*120 + ['parent B']*20)
    matrix.obs['alternative_groups'] = pd.Categorical(['other A']*60 + ['other B']*80)
    assert 'grouping' not in matrix.uns
    if not existing:
        matrix.write_h5ad(source)
    sha = file_sha256(source)
    args.pop('grouping_path')
    args.update(input_path=str(source), input_sha256=sha, group_column='researcher_groups')
    app = ResearchAgentApplication(tmp_path/'workspace')
    request = AgentRequest('supplied', 'Annotate using the original researcher_groups column.', {})
    _, paths = app._prepare_request(request)
    args['output_dir'] = str(paths.scientific/'annotation')
    if existing:
        state = app.sessions.load('session')
    else:
        app.sessions.create('session')
        state = app.sessions.run('session', 'supplied', request,
            (OutputSelection('annotation', 'annotation', 'annotated_h5ad_path'),), expected_generation=0,
            prepared_plan=AgentPlan('supplied:plan', 'supplied', 'Explicit source grouping',
                (PlanStep('annotation', owner.TOOL, args),)))
    assert state.revisions, state.to_dict()
    revision = state.revisions[0]
    run = app.run_store.load(revision.run_id)
    assert run.lifecycle_status.value == 'SUCCEEDED', run.to_dict()
    value = _serialize(run.steps[0].result)
    h = json.loads(Path(value['hierarchy_path']).read_text())
    assert h['grouping']['origin'] == 'user_supplied'
    assert h['grouping']['groupby'] == 'researcher_groups'
    assert h['grouping']['source']['sha256'] == sha
    roots = [b for b in h['branches'] if b['level'] == 1]
    assert {b['cluster']: b['n_cells'] for b in roots} == {'parent A':120, 'parent B':20}
    assert all(b['grouping_binding'] is None for b in roots)
    initial = [r for r in h['operations'] if r['branch_id'] == 'initial']
    assert [r['operation_id'] for r in initial] == ['activity','markers']
    projection = initial[1]['native_result']['parameters']['grouping_projection']
    assert projection['origin'] == 'user_supplied'
    assert projection['activity_positions'] == list(range(140))
    assert projection['source']['sha256'] == sha
    assert len(h['branches']) == 4 and value['levels_reached'] == 2
    assert {b['n_cells'] for b in h['branches'] if b['level']==2} == {60}
    assert all(b['grouping_binding'] is not None for b in h['branches'] if b['level']==2)
    for branch in roots:
        steps = [op for op in h['operations'] if op['branch_id']==branch['branch_id']]
        assert [op['native_result']['function_name'] for op in steps[:4]] == [
            'snapatac2.pp.select_features','snapatac2.tl.spectral','snapatac2.pp.knn','snapatac2.tl.leiden']
        assert steps[0]['native_result']['parameters']['cell_subset']['groupby']=='researcher_groups'
    assert all(op['branch_id']!='initial' for op in h['operations'] if op['native_result']['function_name']=='snapatac2.tl.leiden')
    assert len(h['operations']) == 15
    owner.verify_h5ad(value['annotated_h5ad_path'], h)
    binding = bind_output(app.run_store, app.registry, run.run_id, 'annotation', 'dataset')
    validate_binding(binding, store=app.run_store, registry=app.registry, integrity=True)
    from agent.tools import _native_process
    monkeypatch.setattr(_native_process, 'call', lambda *a,**k: pytest.fail('Delivery/recovery repeated science'))
    view = InteractiveAgentApplication(app._workspace.root, model_profiles=(), default_profile_id=None)
    result = view.revision('session', revision.revision_id)
    figures = [a for a in result.artifacts if a.artifact_type=='analysis_figure' and a.figure]
    assert {'annotation_composition','marker_activity_dot','marker_activity_heatmap','annotation_hierarchy'} <= {a.figure['kind'] for a in figures}
    assert all(a.figure['run_id']==run.run_id and tuple(a.figure['step_ids'])==('annotation',)
               and all(source['run_id']==run.run_id and source['step_id']=='annotation'
                       for source in a.figure['sources']) for a in figures)
    from agent.application.annotation_delivery import accepted_annotations
    artifacts = accepted_annotations(app.sessions, 'session', revision, state.turn('supplied'), revision.outputs[0])
    assert {a.reference.filename for a in artifacts} == set(owner.DELIVERY_FILES)
    downloads = {}
    for artifact in artifacts:
        download = MatrixDownloadPreparer().prepare(artifact)
        try:
            import hashlib
            raw = b''.join(download.iter_chunks())
            assert hashlib.sha256(raw).hexdigest()==download.sha256
            downloads[artifact.reference.filename] = dict(sha256=download.sha256,size=len(raw))
        finally:
            download.close()
    for level in (None,1,2):
        assert owner.presentation_data(value['manifest_path'],value['manifest_sha256'],level=level)['cells'] == matrix.obs_names.tolist()
    resumed = app.runtime.resume(run.run_id)
    assert resumed.status.value=='SUCCEEDED'
    assert file_sha256(source)==sha
    (tmp_path/'acceptance.json').write_text(json.dumps(dict(result=value, hierarchy=h, downloads=downloads,
        run_id=run.run_id, revision_id=revision.revision_id, source_sha256=sha,
        figures=[_serialize(a.figure) for a in figures]),indent=2))


def test_accepted_native_grouping_reuse_and_supplied_native_failure(tmp_path):
    if os.environ.get('RUN_RAW_NATIVE_QUALIFICATION') != '1' or not os.environ.get('ICTA4B_SUPPLIED_FIXTURE'):
        pytest.skip('Explicit accepted native fixtures')
    from agent.orchestration import AgentRuntime, FileRunStore
    from agent.orchestration.registry import build_default_tool_registry
    from agent.orchestration.prior_outputs import bind_output
    from agent.schemas import PriorOutputRef
    from agent.schemas.prior_output import PriorOutputBinding
    existing=os.environ.get('ICTA4B_REUSE_FIXTURE')
    if existing:
        tmp_path=Path(existing)
    reg=build_default_tool_registry()
    store=FileRunStore(tmp_path/'run_state')
    old_root=Path(os.environ['ICTA4B_ACCEPTED_FIXTURE'])
    new_root=Path(os.environ['ICTA4B_SUPPLIED_FIXTURE'])
    for directory in (() if existing else (old_root/'run_state',new_root/'workspace/run_state')):
        for path in directory.glob('*.json'):
            shutil.copyfile(path,store.root/path.name)
    runtime=AgentRuntime(registry=reg,run_store=store)
    old=store.load('continued:run')
    h=json.loads(Path(old.steps[0].result['hierarchy_path']).read_text())
    args=dict(old.plan.steps[0].arguments)
    args.update(maximum_levels=1, output_dir=str(tmp_path/'accepted-native'))
    request=AgentRequest('accepted-native',None,{})
    result=(store.load('accepted-native:run').to_run_result() if existing else
        runtime.run_prepared(request,AgentPlan('accepted-native:plan',request.request_id,'Accepted native groups unchanged',
            (PlanStep('annotation',owner.TOOL,args),))))
    assert result.status.value=='SUCCEEDED',result.to_dict()
    reused=json.loads(Path(result.steps[0].result['hierarchy_path']).read_text())
    assert reused['grouping']==h['grouping']
    assert [op['operation_id'] for op in reused['operations']]==['activity','markers']
    assert reused['input_results'][0]==h['input_results'][0]
    assert all(op['native_result']['function_name']!='snapatac2.tl.leiden' for op in reused['operations'])
    # Invalid native parameter remains a real failed child, not a stopping reason.
    supplied=store.load('supplied:run')
    args=dict(supplied.plan.steps[0].arguments)
    args.update(local_kwargs={'pp.select_features':{'n_features':1.5}},
        previous_annotation_path=PriorOutputRef(bind_output(store,reg,supplied.run_id,'annotation','annotation'),'manifest_path'),
        output_dir=str(tmp_path/'native-failure'))
    request=AgentRequest('supplied-failure',None,{})
    failed=(store.load('supplied-failure:run').to_run_result() if existing else
        runtime.run_prepared(request,AgentPlan('supplied-failure:plan',request.request_id,'Native legality retained',
            (PlanStep('annotation',owner.TOOL,args),))))
    assert failed.status.value=='FAILED'
    children=[store.load(e.details['request']['request_id']+':run') for e in failed.trace if e.details.get('event')=='reserved']
    assert len(children)==3 and children[-1].lifecycle_status.value=='FAILED'
    assert len([e for e in failed.trace if e.details.get('event')=='reused'])==2
    assert any(e.code=='NATIVE_FUNCTION_REJECTED' for e in children[-1].errors)
    assert not (tmp_path/'native-failure/manifest.json').exists()
    (tmp_path/'reuse-and-failure.json').write_text(json.dumps(dict(accepted=result.to_dict(),failed=failed.to_dict(),child=children[-1].to_dict()),indent=2))


def test_supplied_projection_rejects_equal_names_without_source_identity(tmp_path):
    if not os.environ.get('ICTA4B_SUPPLIED_FIXTURE'):
        pytest.skip('Explicit accepted small native fixture')
    import anndata as ad
    from agent.application import ResearchAgentApplication
    from agent.orchestration.verification_authority import accepted_authorities
    from agent.schemas.verification_authority import AuthorityError
    from agent.tools.analysis.native_gene_activity import project_grouping
    app=ResearchAgentApplication(Path(os.environ['ICTA4B_SUPPLIED_FIXTURE'])/'workspace')
    run=app.run_store.load('supplied:run')
    h=json.loads(Path(run.steps[0].result['hierarchy_path']).read_text())
    activity=h['input_results'][-1]
    data=ad.read_h5ad(activity['matrix_path'])
    original=ad.read_h5ad(h['source']['path'])
    original.uns['unrelated_copy']='Equal cells and values do not establish original file identity'
    unrelated=tmp_path/'unrelated.h5ad';original.write_h5ad(unrelated)
    args=dict(input_path=activity['matrix_path'],grouping_path=str(unrelated),group_column='researcher_groups',
        function_name='tl.rank_genes_groups',kwargs={'groupby':'researcher_groups'})
    with accepted_authorities(app.run_store,h['activity']['run_id']):
        with pytest.raises(AuthorityError):
            project_grouping(data,args,dict(sha256=activity['matrix_sha256']))
        args['grouping_path']=h['source']['path']
        projected,provenance=project_grouping(data,args,dict(sha256=activity['matrix_sha256']))
        assert projected.obs_names.tolist()==original.obs_names.tolist()
        assert projected.obs.researcher_groups.tolist()==original.obs.researcher_groups.tolist()
        assert provenance['origin']=='user_supplied'


def test_supplied_source_report_and_http_downloads_without_science(tmp_path,monkeypatch):
    if not os.environ.get('ICTA4B_SUPPLIED_FIXTURE'):
        pytest.skip('Explicit accepted small native fixture')
    from agent.application import ResearchAgentApplication, InteractiveAgentApplication
    from agent.orchestration.verification_authority import accepted_authorities
    from agent.report.evidence import build_analysis_evidence
    from agent.report.visualization import build_analysis_visualizations,verify_analysis_visualizations
    from agent.report.analysis_report import build_analysis_report,verify_analysis_report
    from agent.tools import _native_process
    root=Path(os.environ['ICTA4B_SUPPLIED_FIXTURE'])
    app=ResearchAgentApplication(root/'workspace')
    run=app.run_store.load('supplied:run');before=run.to_dict()
    monkeypatch.setattr(_native_process,'call',lambda *a,**k:pytest.fail('Report or HTTP repeated science'))
    with accepted_authorities(app.run_store,run.run_id):
        evidence=build_analysis_evidence(run.to_run_result(),tmp_path/'evidence',registry=app.registry)
        figures=build_analysis_visualizations(run.to_run_result(),evidence,tmp_path/'figures',registry=app.registry)
        assert verify_analysis_visualizations(run.to_run_result(),evidence,figures,registry=app.registry).passed
        report=build_analysis_report(run.to_run_result(),evidence,tmp_path/'report',registry=app.registry,visualization=figures)
        assert verify_analysis_report(run.to_run_result(),evidence,report,registry=app.registry,visualization=figures).passed
    text=(tmp_path/'report/analysis_report/analysis_report.md').read_text()
    assert 'user_supplied' in text and 'researcher_groups' in text and 'not validated' in text
    assert 'scanpy.tl.rank_genes_groups' in text and 'snapatac2.tl.diff_test' in text
    from fastapi.testclient import TestClient
    from agent.web.app import create_app
    import hashlib
    view=InteractiveAgentApplication(root/'workspace',model_profiles=(),default_profile_id=None)
    revision=app.sessions.load('session').revisions[0].revision_id
    result=view.revision('session',revision)
    downloaded=[]
    with TestClient(create_app(view)) as client:
        for artifact in result.artifacts:
            if artifact.artifact_type not in {'analysis_figure','scientific_annotation'}:
                continue
            response=client.get(f'/api/v1/sessions/session/revisions/{revision}/artifacts/{artifact.handle}?download=true')
            assert response.status_code==200,response.text[:200]
            content=view.artifact_content('session',revision,artifact.handle)
            assert hashlib.sha256(response.content).hexdigest()==content.content_sha256
            downloaded.append(artifact.handle)
    assert len(downloaded)>=4
    assert app.run_store.load(run.run_id).to_dict()==before
    (tmp_path/'report-and-http.json').write_text(json.dumps(dict(downloads=downloaded,evidence=evidence,figures=figures,report=report),indent=2))


def test_registered_source_current_publication_reuses_exact_native_children(tmp_path,monkeypatch):
    if not os.environ.get('ICTA4B_SUPPLIED_FIXTURE'):
        pytest.skip('Explicit accepted small native fixture')
    from agent.application import ResearchAgentApplication, OutputSelection
    from agent.application.local_resources import LocalResourceAdmission
    from agent.orchestration.executor import execution_input_guard
    from agent.orchestration.prior_outputs import bind_output
    from agent.schemas import PriorOutputRef
    from agent.tools import _native_process
    root=Path(os.environ['ICTA4B_SUPPLIED_FIXTURE'])
    app=ResearchAgentApplication(tmp_path/'workspace')
    for path in (root/'workspace/run_state').glob('*.json'):
        shutil.copyfile(path,app.run_store.root/path.name)
    old=app.run_store.load('supplied:run')
    args=dict(old.plan.steps[0].arguments)
    resources=LocalResourceAdmission(app._workspace,approved_source_roots=(root,),registry=app.registry)
    record=resources.register('uploaded-groups',args['input_path'],label='Supplied peak counts',attribution='Exact uploaded original obs groups')
    binding=resources.compose_group([record.resource_id],scientific_inputs={'group_column':'researcher_groups'},attachment_candidates=True)
    binding=resources.bind_group(binding,[dict(tool=owner.TOOL,port='source',source='source.0')],verify_source=True)
    submission=dict(execution_inputs=_serialize(binding.execution_inputs),registered_input=binding.execution_attribution())
    request=AgentRequest('registered-supplied','Reuse the supplied original obs grouping.',binding.execution_inputs)
    _,paths=app._prepare_request(request)
    args.update(output_dir=str(paths.scientific/'annotation'),previous_annotation_path=PriorOutputRef(
        bind_output(app.run_store,app.registry,old.run_id,'annotation','annotation'),'manifest_path'))
    app.sessions.create('session')
    monkeypatch.setattr(_native_process,'call',lambda *a,**k:pytest.fail('Exact accepted children should be reused'))
    with execution_input_guard(lambda step,arguments: resources.validate_group_consumer(submission,step,arguments)):
        state=app.sessions.run('session','registered',request,(OutputSelection('annotation','annotation','annotated_h5ad_path'),),
            expected_generation=0,prepared_plan=AgentPlan('registered-supplied:plan',request.request_id,'Exact registered source and current publication',
                (PlanStep('annotation',owner.TOOL,args),)))
    assert state.revisions,state.to_dict()
    run=app.run_store.load(state.revisions[0].run_id)
    assert run.lifecycle_status.value=='SUCCEEDED',run.to_dict()
    assert len([e for e in run.trace if e.details.get('event')=='reused'])==15
    h=json.loads(Path(run.steps[0].result['hierarchy_path']).read_text())
    assert h['grouping']['origin']=='user_supplied' and h['source']['sha256']==record.source_sha256
    (tmp_path/'registered-source.json').write_text(json.dumps(dict(binding=binding.attribution(),run=run.to_dict()),indent=2))
