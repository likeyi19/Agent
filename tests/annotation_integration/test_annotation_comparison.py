"""Real general metric/publication/render/download owners; simulated MAESTRO fixture."""
import json
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import pandas as pd
import pytest

from agent.application import ResearchAgentApplication, OutputSelection
from agent.orchestration import AgentRequest, AgentPlan, PlanStep
from agent.schemas import StepOutputRef
from agent.schemas.orchestration import _serialize
from agent.tools.analysis import cell_label_comparison as owner, scatac_annotation
from test_external_entry import external_case, case


class Planner:
    def __init__(self,args,reference,mapped=False):self.args,self.reference,self.mapped=args,reference,mapped
    def plan(self,request,registry):
        return AgentPlan('compare',request.request_id,'Annotate then compare fixed labels',(
            PlanStep('annotation','annotate_scATAC_cell_types',self.args),
            PlanStep('comparison',owner.TOOL,dict(prediction_path=StepOutputRef('annotation','annotated_h5ad_path'),
                reference_path=str(self.reference),prediction_label_key='primary_annotation',
                reference_label_key='reference',reference_cell_key='cell',shared_library_id='synthetic-library',
                output_dir=str(Path(request.inputs['output_dir'])/'comparison'), **(dict(prediction_mapping={'A':'Common'},reference_mapping={'A':'Common'}) if self.mapped else {})),depends_on=('annotation',))))


@pytest.mark.parametrize("mapped", [False, True])
def test_maestro_comparison_figures_authority_download_and_restart(external_case,tmp_path,monkeypatch,mapped):
    args,calls=external_case
    original=Path(args['source_path']).read_bytes()
    data=ad.read_h5ad(args['source_path']);reference=tmp_path/'reference.csv'
    pd.DataFrame({'cell':data.obs_names[::-1],'reference':['A']*data.n_obs}).to_csv(reference,index=False)
    app=ResearchAgentApplication(tmp_path/'app',planner=Planner(args,reference,mapped));app.sessions.create('research')
    state=app.sessions.run('research','first',AgentRequest('first','Annotate and compare',{}),
        (OutputSelection('comparison','comparison','manifest_path'),),expected_generation=0)
    assert state.turn('first').status=='activated',state.turn('first')
    assert len(calls)==1
    run=app.run_store.load('first:run');result=_serialize(run.steps[-1].result)
    assert result['matched_cells']==data.n_obs and result['unassigned_prediction_cells']>0
    assert run.steps[-1].verification.artifact_authority['artifact_contract']==owner.CONTRACT
    root=app._workspace.run_paths(run.run_id)
    manifest=json.loads(next(root.visualizations.rglob('visualization_manifest.json')).read_text())
    kinds=[f['figure_kind'] for f in manifest['figures']]
    assert kinds==['annotation_composition','marker_activity_dot','marker_activity_distribution','reference_confusion','reference_composition']
    assert manifest['figure_set']['annotation_projection_version']==2
    if mapped:
        assert 'original_metrics' in result and result['mapping']['applied']
        assert all(f['presentation'].get('label_policy')=='explicit_harmonized_labels'
                   for f in manifest['figures'] if f['figure_kind'].startswith('reference_'))
        assert 'Harmonized-label view' in manifest['figures'][-1]['presentation']['caption']
    assert all(len(f['plotting_data_sha256'])==64 for f in manifest['figures'])
    assert 'not measured RNA expression' in manifest['figures'][1]['presentation']['caption']
    from agent.application.native_delivery import accepted_artifacts
    from agent.application.matrix_delivery import MatrixDownloadPreparer
    revision=state.revisions[-1];origin=state.turn('first')
    artifacts=accepted_artifacts(SimpleNamespace(_application=app),'research',revision,origin,revision.outputs[0])
    assert {a.reference.filename for a in artifacts}==set(owner.DELIVERY_FILES)
    for artifact in artifacts:
        with MatrixDownloadPreparer().prepare(artifact) as download:
            payload=b''.join(download.iter_chunks())
        assert payload
        if artifact.role=='aligned-labels.tsv':assert b'primary_annotation' not in payload and b'predicted_label' in payload
    from agent.report.visualization import verify_analysis_visualizations
    figure_manifest=next(root.visualizations.rglob('visualization_manifest.json'))
    evidence=next(root.evidence.rglob('analysis_evidence.json'))
    assert verify_analysis_visualizations(run.to_run_result(),evidence,figure_manifest,registry=app.registry).passed
    png=figure_manifest.parent/manifest['figures'][1]['relative_path'];saved=png.read_bytes();png.write_bytes(saved+b'tampered')
    assert not verify_analysis_visualizations(run.to_run_result(),evidence,figure_manifest,registry=app.registry).passed
    png.write_bytes(saved)
    preserved=app.run_store.load('first:run').to_dict()
    monkeypatch.setattr(scatac_annotation,'execute_annotation',lambda *a,**k:pytest.fail('Native rerun'))
    monkeypatch.setattr(owner,'compare',lambda *a,**k:pytest.fail('Comparison recomputed during reuse'))
    fresh=ResearchAgentApplication(app.workspace_root,planner=Planner(args,reference))
    view=fresh.sessions.evidence('research',revision.revision_id,'comparison')
    assert view.status=='available'
    assert next(f.value for f in view.facts if f.field=='matched_cells')==data.n_obs
    assert fresh.run_store.load('first:run').to_dict()==preserved
    assert Path(args['source_path']).read_bytes()==original
