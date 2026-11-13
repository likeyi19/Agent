"""Opt-in production application qualification; real accepted native science.

Reuses the explicitly selected small ICTA-4B fixture, never historical MAESTRO data.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil

import pytest
from agent.schemas import AgentRequest, AgentPlan, PlanStep, StepOutputRef, PriorOutputRef
from agent.schemas.orchestration import _serialize
from agent.orchestration.prior_outputs import bind_output, validate_binding
from agent.tools.analysis import hierarchical_annotation as owner


@pytest.fixture(scope='module')
def application(tmp_path_factory):
    if os.environ.get('RUN_RAW_NATIVE_QUALIFICATION') != '1' or not os.environ.get('ICTA4B_ACCEPTED_FIXTURE'):
        pytest.skip('Explicit small production accepted fixture required')
    from agent.application import ResearchAgentApplication, OutputSelection
    existing=os.environ.get('ICTA4B_APPLICATION_FIXTURE')
    if existing:
        root=Path(existing)
        app=ResearchAgentApplication(root/'workspace')
        yield app,app.sessions.load('session'),root
        return
    source=Path(os.environ['ICTA4B_ACCEPTED_FIXTURE'])
    root=tmp_path_factory.mktemp('production-annotation')
    app=ResearchAgentApplication(root/'workspace')
    # Copy exact records into isolated application storage. Accepted publications
    # and original records remain at their original pinned locations, untouched.
    for path in (source/'run_state').glob('*.json'):
        shutil.copyfile(path,app.run_store.root/path.name)
    prior=app.run_store.load('continued:run')
    assert prior.lifecycle_status.value=='SUCCEEDED'
    args=dict(prior.plan.steps[0].arguments)
    args.update(maximum_levels=2,previous_annotation_path=PriorOutputRef(
        bind_output(app.run_store,app.registry,prior.run_id,'annotate','annotation'),'manifest_path'))
    request=AgentRequest('production','Annotate cell types in these scATAC cells.',{})
    _,paths=app._prepare_request(request)
    args['output_dir']=str(paths.scientific/'annotation')
    grouping=args['grouping_path']
    plan=AgentPlan('production:plan',request.request_id,'Selected accepted annotation with native global UMAP',(
        PlanStep('spectral','run_snapatac2',dict(input_path=grouping, function_name='tl.spectral',
            kwargs={'n_comps':8,'features':None},output_dir=str(paths.scientific/'spectral'))),
        PlanStep('umap','run_snapatac2',dict(input_path=StepOutputRef('spectral','matrix_path'),function_name='tl.umap',
            output_dir=str(paths.scientific/'umap')),depends_on=('spectral',)),
        PlanStep('annotation',owner.TOOL,args)))
    app.sessions.create('session')
    state=app.sessions.run('session','production',request,
        (OutputSelection('annotation','annotation','annotated_h5ad_path'),OutputSelection('umap','umap','matrix_path')),
        expected_generation=0,prepared_plan=plan)
    assert state.turn('production').status in ('activated','ready','run_succeeded'),state.to_dict()
    yield app,state,root


def test_real_result_figures_downloads_and_level_presentation(application,tmp_path):
    app,state,root=application
    from agent.application import InteractiveAgentApplication
    from agent.application.matrix_delivery import MatrixDownloadPreparer
    view=InteractiveAgentApplication(app._workspace.root, model_profiles=(), default_profile_id=None)
    revision=state.revisions[0]
    result=view.revision('session',revision.revision_id)
    assert result.status=='succeeded'
    stored=app.run_store.load(revision.run_id)
    parent=next(s for s in stored.steps if s.tool_name==owner.TOOL)
    assert parent.verification.artifact_authority['schema_version']==2
    hierarchy=json.loads(Path(parent.result['hierarchy_path']).read_text())
    assert len(hierarchy['operations'])==15
    assert len([e for e in stored.trace if e.details.get('event')=='reused'])==15
    figures=[a for a in result.artifacts if a.artifact_type=='analysis_figure' and a.figure]
    kinds={a.figure['kind'] for a in figures}
    assert {'annotation_composition','marker_activity_dot','marker_activity_heatmap','marker_activity_umap','annotation_umap','annotation_hierarchy'}<=kinds,kinds
    assert {a.figure['annotation_level'] for a in figures if a.figure['kind']=='annotation_umap'}=={'final',1,2}
    before=app.run_store.load(revision.run_id).to_dict()
    final=owner.presentation_data(parent.result['manifest_path'],parent.result['manifest_sha256'])
    level=owner.presentation_data(parent.result['manifest_path'],parent.result['manifest_sha256'],level=1)
    assert final['cells']==level['cells']
    assert final['levels_reached']==level['levels_reached']==2
    assert app.run_store.load(revision.run_id).to_dict()==before
    from agent.application.annotation_delivery import accepted_annotations
    output=next(o for o in revision.outputs if o.name=='annotation')
    artifacts=accepted_annotations(app.sessions,'session',revision,state.turn('production'),output)
    assert {a.reference.filename for a in artifacts}==set(owner.DELIVERY_FILES)
    deliveries={}
    for artifact in artifacts:
        download=MatrixDownloadPreparer().prepare(artifact)
        try:
            raw=b''.join(download.iter_chunks())
            assert hashlib.sha256(raw).hexdigest()==download.sha256
            deliveries[artifact.reference.filename]=dict(sha256=download.sha256,size=len(raw))
            if artifact.reference.filename=='annotated.h5ad':
                path=root/'download.h5ad';path.write_bytes(raw)
                import anndata as ad
                data=ad.read_h5ad(path)
                assert len(data)==140 and 'cell_type_final' in data.obs
        finally:download.close()
    binding=bind_output(app.run_store,app.registry,revision.run_id,'annotation','dataset')
    validate_binding(binding,store=app.run_store,registry=app.registry,integrity=True)
    request=AgentRequest('consumer-'+hashlib.sha256(str(tmp_path).encode()).hexdigest()[:12],'Apply log1p to the selected accepted annotation H5AD.',{})
    _,paths=app._prepare_request(request)
    consumed=app.run_prepared(request,AgentPlan(request.request_id+':plan',request.request_id,'Exact downstream accepted H5AD',(
        PlanStep('read','run_scanpy',dict(input_path=PriorOutputRef(binding,'annotated_h5ad_path'),
            function_name='pp.log1p',output_dir=str(paths.scientific/'consumer'))),)))
    assert consumed.run_result.status.value=='SUCCEEDED',consumed.to_dict()
    (root/'delivery-evidence.json').write_text(json.dumps(dict(deliveries=deliveries,figures=sorted(kinds),
        binding=binding.to_dict(),revision=revision.revision_id),indent=2))


def test_actual_child_parameter_edit_reuses_upstream_and_preserves_parent(application):
    app,state,root=application
    from agent.application import InteractiveAgentApplication
    from agent.application.structured_parameters import result_parameters
    from agent.orchestration import PlanningModelProfile
    from agent.providers import PlanningModelFactoryRegistry
    class ForbiddenModel:
        model_id='no-provider'
        def complete(self, **kwargs):
            raise AssertionError('Result edit must not call a provider')
    profile=PlanningModelProfile('editing','scripted','no-provider')
    interactive=InteractiveAgentApplication(app._workspace.root, model_profiles=(profile,), default_profile_id='editing',
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':lambda _:ForbiddenModel()}))
    revision=state.revisions[0]
    offered=result_parameters(app.sessions,'session',revision.revision_id)
    marker=next(s for s in offered['steps'] if s.get('operation_id')=='markers')
    assert marker['branch_id']=='initial' and marker['run_id'].startswith('annotation-child-')
    previous=app.run_store.load(revision.run_id).to_dict()
    row=dict(step_id=marker['step_id'],tool='run_scanpy',native_function='tl.rank_genes_groups',
        argument='kwargs',value={'tie_correct':True})
    result=interactive.submit_turn('session','edit','',expected_generation=1,
        rerun_revision_id=revision.revision_id,parameter_overrides=[row])
    assert result.status=='succeeded',result.to_dict()
    updated=app.sessions.load('session')
    run=app.run_store.load(updated.revisions[-1].run_id)
    parent=next(s for s in run.steps if s.tool_name==owner.TOOL)
    hierarchy=json.loads(Path(parent.result['hierarchy_path']).read_text())
    markers=next(o for o in hierarchy['operations'] if o['operation_id']=='markers')
    assert markers['native_result']['parameters']['effective_native_parameters']['tie_correct'] is True
    assert markers['binding']['run_id']!=marker['run_id']
    assert len([e for e in run.trace if e.details.get('event')=='reused'])==14
    assert app.run_store.load(revision.run_id).to_dict()==previous


def test_current_source_figures_and_methods_report_without_scientific_rerun(application,tmp_path,monkeypatch):
    app,state,root=application
    from agent.orchestration.verification_authority import accepted_authorities
    from agent.report.evidence import build_analysis_evidence
    from agent.report.visualization import build_analysis_visualizations,verify_analysis_visualizations
    from agent.report.analysis_report import build_analysis_report,verify_analysis_report
    from agent.tools import _native_process
    monkeypatch.setattr(_native_process,'call',lambda *a,**k:pytest.fail('Presentation repeated native science'))
    run=app.run_store.load(state.revisions[0].run_id)
    original=run.to_dict()
    with accepted_authorities(app.run_store,run.run_id):
        evidence=build_analysis_evidence(run.to_run_result(),tmp_path/'evidence',registry=app.registry)
        figures=build_analysis_visualizations(run.to_run_result(),evidence,tmp_path/'figures',registry=app.registry)
        checked=verify_analysis_visualizations(run.to_run_result(),evidence,figures,registry=app.registry)
        assert checked.passed,checked.to_dict()
        report=build_analysis_report(run.to_run_result(),evidence,tmp_path/'report',registry=app.registry,visualization=figures)
        checked=verify_analysis_report(run.to_run_result(),evidence,report,registry=app.registry,visualization=figures)
        assert checked.passed,checked.to_dict()
    text=(tmp_path/'report'/'analysis_report'/'analysis_report.md').read_text()
    for term in ('scanpy.tl.rank_genes_groups','snapatac2.tl.diff_test','producing_run_id','CellMarker'):
        # Producing identities are rendered as human-readable method metadata.
        assert (term in text if term!='producing_run_id' else 'Producing Run:' in text)
    assert app.run_store.load(run.run_id).to_dict()==original
    (root/'report-evidence.json').write_text(json.dumps(dict(evidence=evidence,figures=figures,report=report),indent=2))


def test_fresh_production_report_download_and_http_progress(application,tmp_path,monkeypatch):
    from agent.application import ResearchAgentApplication,InteractiveAgentApplication,OutputSelection
    from agent.tools import _native_process
    from fastapi.testclient import TestClient
    from agent.web.app import create_app
    original,old_state,_=application
    app=ResearchAgentApplication(os.environ.get('ICTA4B_REPORT_FIXTURE',str(tmp_path/'workspace')))
    prior=original.run_store.load(old_state.revisions[0].run_id)
    original_bytes=prior.to_dict()
    if os.environ.get('ICTA4B_REPORT_FIXTURE'):
        state=app.sessions.load('session')
    else:
        for path in original.run_store.root.glob('*.json'):
            shutil.copyfile(path,app.run_store.root/path.name)
        prior=app.run_store.load(old_state.revisions[0].run_id)
        args=dict(next(s for s in prior.plan.steps if s.tool_name==owner.TOOL).arguments)
        args['previous_annotation_path']=PriorOutputRef(bind_output(app.run_store,app.registry,prior.run_id,'annotation','annotation'),'manifest_path')
        request=AgentRequest('report-delivery','Present the selected annotation with its recorded methods.',{})
        _,paths=app._prepare_request(request)
        args['output_dir']=str(paths.scientific/'annotation')
        app.sessions.create('session')
        state=app.sessions.run('session','result',request,(OutputSelection('annotation','annotation','annotated_h5ad_path'),),
            expected_generation=0,prepared_plan=AgentPlan('report-delivery:plan',request.request_id,'Reuse all accepted native children',(
                PlanStep('annotation',owner.TOOL,args),)))
        assert state.revisions,state.to_dict()
    run=app.run_store.load(state.revisions[0].run_id)
    assert len([e for e in run.trace if e.details.get('event')=='reused'])==15
    monkeypatch.setattr(_native_process,'call',lambda *a,**k:pytest.fail('Delivery repeated native work'))
    view=InteractiveAgentApplication(app.workspace_root,model_profiles=(),default_profile_id=None)
    revision=state.revisions[0].revision_id
    result=view.revision('session',revision)
    report=next(a for a in result.artifacts if a.artifact_type=='analysis_report')
    content=view.artifact_content('session',revision,report.handle)
    assert hashlib.sha256(content.data).hexdigest()==content.content_sha256
    text=content.data.decode()
    assert 'scanpy.tl.rank_genes_groups' in text and 'Producing Run:' in text
    assert text.count('Producing Run:')==15
    assert 'snapatac2.tl.diff_test' in text and str(app.workspace_root) not in text
    assert 'Private source/provenance fields omitted' in text
    with TestClient(create_app(view)) as client:
        delivered=client.get(f'/api/v1/sessions/session/revisions/{revision}/artifacts/{report.handle}?download=true')
        assert delivered.status_code==200 and delivered.content==content.data
        assert delivered.headers['X-Artifact-Content-SHA256']==content.content_sha256
        inventory=client.get(f'/api/v1/sessions/session/revisions/{revision}/scientific-artifacts').json()['artifacts']
        assert {a['filename'] for a in inventory}==set(owner.DELIVERY_FILES)
        for artifact in inventory:
            download=client.get(f"/api/v1/sessions/session/revisions/{revision}/scientific-artifacts/{artifact['handle']}")
            assert download.status_code==200,download.text[:300]
            assert hashlib.sha256(download.content).hexdigest()==download.headers['X-Artifact-Content-SHA256']
        status=client.get('/api/v1/sessions/session/turns/result').json()
        progress=next(s['annotation_progress'] for s in status['steps'] if s['tool_name']==owner.TOOL)
        assert progress['completed_children']==15 and not progress['future_total_known']
        assert progress['outcome']=='complete'
        parameters=client.get(f'/api/v1/sessions/session/revisions/{revision}/parameters')
        assert parameters.status_code==200,parameters.text
        assert any(s.get('branch_id')=='initial' and s.get('operation_id')=='markers'
                   for s in parameters.json()['steps'])
    assert app.run_store.load(prior.run_id).to_dict()==original_bytes
    (tmp_path/'api-evidence.json').write_text(json.dumps(dict(progress=progress,
        report_sha256=content.content_sha256,revision=revision),indent=2))
