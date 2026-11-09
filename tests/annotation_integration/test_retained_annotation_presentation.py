"""Fresh simulated MAESTRO output plus real Scanpy science, reports and downloads."""
import hashlib,json,os
from pathlib import Path
from types import SimpleNamespace
import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from test_external_entry import external_case, case
from agent.application import ResearchAgentApplication, OutputSelection
from agent.orchestration import AgentPlan,AgentRequest,PlanStep
from agent.schemas import StepOutputRef


@pytest.fixture
def matrix_source(tmp_path):
    from agent.tools.data import scatac_reference as ref
    fa=tmp_path/'ref.fa';fa.write_text('>chr2\n'+'A'*6001+'\n>chr1\n'+'C'*6001+'\n')
    fai=tmp_path/'ref.fa.fai';fai.write_text('chr2\t6001\t6\t6001\t6002\nchr1\t6001\t6014\t6001\t6002\n')
    bed=tmp_path/'ref.bed';regions=[(c,i*100,i*100+80) for c in ('chr2','chr1') for i in range(4)]
    bed.write_text(''.join(f'{c}\t{a}\t{b}\n' for c,a,b in regions))
    bundle=ref.build_scatac_reference_bundle(species='human',target_assembly='hg38',fasta_path=fa,fai_path=fai,ccre_bed_path=bed)
    pointer=ref.publish_scatac_reference_bundle(bundle,tmp_path/'reference.json')
    path=tmp_path/'source.h5ad'
    data=ad.AnnData(csr_matrix(np.random.default_rng(3).poisson(2,(16,8))),
        obs=pd.DataFrame(index=[f'synthetic-{i}' for i in range(16)]),
        var=pd.DataFrame(index=[f'{c}:{a}-{b}' for c,a,b in regions]))
    data.write_h5ad(path)
    return dict(source_path=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        reference_manifest_path=pointer['manifest_path'],reference_manifest_sha256=pointer['manifest_sha256'],
        species='human',assembly='hg38',matrix_semantics='fragment_counts',output_dir=str(tmp_path/'output'))


def test_native_annotation_figures_factual_report_download_and_restart(external_case,tmp_path,monkeypatch):
    from agent.schemas.orchestration import _serialize
    from agent.application.annotation_delivery import accepted_annotation
    from agent.application.matrix_delivery import MatrixDownloadPreparer
    from agent.tools.analysis import scatac_annotation
    args,calls=external_case;source_bytes=Path(args['source_path']).read_bytes()
    target=os.environ.get('AGENT_RETIREMENT_SMOKE_WORKSPACE')
    workspace=Path(target) if target else tmp_path/'app'
    class Planner:
        def plan(self,request,registry):
            output=Path(request.inputs['output_dir'])
            return AgentPlan('native-presentation',request.request_id,'Synthetic annotation presentation fixture',(
                PlanStep('pca','pca_embed_cells',dict(input_path=args['source_path'],output_dir=str(output/'pca'),n_components=3)),
                PlanStep('neighbors','build_cell_neighbors',dict(embedding_path=StepOutputRef('pca','embedding_path'),
                    cell_ids_path=StepOutputRef('pca','cell_ids_path'),output_dir=str(output/'neighbors'),n_neighbors=4),('pca',)),
                PlanStep('clusters','cluster_cells',dict(analysis_path=StepOutputRef('neighbors','analysis_path'),output_dir=str(output/'clusters')),('neighbors',)),
                PlanStep('umap','compute_cell_umap',dict(analysis_path=StepOutputRef('clusters','analysis_path'),output_dir=str(output/'umap')),('clusters',)),
                PlanStep('annotation','annotate_scATAC_cell_types',dict(args,output_dir=str(output/'annotation')))))
    app=ResearchAgentApplication(workspace,planner=Planner());sid='retirement-annotation-smoke-20261108';app.sessions.create(sid)
    state=app.sessions.run(sid,'retained-native',AgentRequest('retained-native','Inspect the synthetic native annotation presentation',{}),
        (OutputSelection('annotation','annotation','manifest_path'),OutputSelection('umap','umap','analysis_path')),expected_generation=0)
    assert state.turn('retained-native').status=='activated',state.turn('retained-native')
    assert len(calls)==1
    run=app.run_store.load('retained-native:run');result=_serialize(run.steps[-1].result)
    assert 'terminology' not in result
    path=Path(result['manifest_path']);manifest=json.loads(path.read_text())
    assert not any('terminology' in name or 'ontology' in name for name in manifest['sidecars'])
    annotated=ad.read_h5ad(result['annotated_h5ad_path']);native=json.loads((path.parent/'primary/cells.json').read_text())
    expected=[r['primary_annotation'] for r in native]
    assert annotated.obs['primary_annotation'].astype(object).where(annotated.obs['primary_annotation'].notna(),None).tolist()==expected
    assert result['assigned_cells']>0 and result['unassigned_cells']>0 and None in expected
    root=app._workspace.run_paths(run.run_id);figure_manifest=next(root.visualizations.rglob('visualization_manifest.json'))
    figures=json.loads(figure_manifest.read_text())['figures']
    assert {f['figure_kind'] for f in figures}=={'umap_leiden','annotation_composition','annotation_umap','marker_activity_dot','marker_activity_distribution','marker_activity_umap'}
    report=next(root.report.rglob('*.md')).read_text()
    assert 'Native candidate distribution' in report and 'n_features' not in result
    artifact=accepted_annotation(SimpleNamespace(_application=app),sid,state.revisions[-1],state.turn('retained-native'),None,step_id='annotation')
    with MatrixDownloadPreparer().prepare(artifact) as download:payload=b''.join(download.iter_chunks())
    assert payload and Path(args['source_path']).read_bytes()==source_bytes
    preserved=run.to_dict();monkeypatch.setattr(scatac_annotation,'execute_annotation',lambda *a,**k:pytest.fail('Native rerun on read'))
    from agent.application.annotation_delivery import accepted_annotations, EVIDENCE_FILES
    from agent.application.matrix_delivery import MatrixDeliveryError
    import io
    from scipy.sparse import load_npz
    delivered = accepted_annotations(SimpleNamespace(_application=app), sid, state.revisions[-1],
        state.turn('retained-native'), None, step_id='annotation')
    assert {a.role for a in delivered[1:]} == set(EVIDENCE_FILES)
    from agent.application.analysis_delivery import export_descriptions
    output = next(o for o in state.revisions[-1].outputs if o.name == 'annotation')
    descriptions = export_descriptions(app.sessions, sid, [dict(revision_id=state.revisions[-1].revision_id,
        output_name=output.name, accepted_step_sha256=output.accepted_step_sha256)])
    assert {x['handle'] for x in descriptions} == {a.reference.handle for a in delivered}
    assert all(x['actual_download_status'] == 'NOT_RECORDED' for x in descriptions)
    for entry in delivered[1:]:
        with MatrixDownloadPreparer().prepare(entry) as download:
            raw = b''.join(download.iter_chunks())
            assert download.source_sha256 == entry.sha256
        original = entry.path.read_bytes()
        if entry.role.endswith('.json'):
            assert json.loads(raw) == json.loads(original)
        elif entry.role.endswith('.npz'):
            assert raw == original
            assert load_npz(io.BytesIO(raw)).shape[0] == result['n_cells']
        else:
            assert raw.decode().splitlines() == original.decode().splitlines()
    marker = next(a for a in delivered if a.role == 'native-markers.tsv')
    original = marker.path.read_bytes()
    try:
        marker.path.write_bytes(b'changed')
        with pytest.raises(MatrixDeliveryError):
            MatrixDownloadPreparer().prepare(marker)
    finally:
        marker.path.write_bytes(original)
    assert len(calls) == 1
    fresh=ResearchAgentApplication(workspace,planner=Planner())
    assert fresh.sessions.evidence(sid,state.revisions[-1].revision_id,'annotation').status=='available'
    assert fresh.run_store.load(run.run_id).to_dict()==preserved
    if target:
        O=Path(os.environ['AGENT_RETIREMENT_ACCEPTANCE_ROOT'])
        (O/'audit/smoke-result.json').write_text(json.dumps(dict(workspace=str(workspace),session_id=sid,
            revision_id=state.revisions[-1].revision_id,run_id=run.run_id,source_path=args['source_path'],
            source_sha256=hashlib.sha256(source_bytes).hexdigest(),annotation=result,
            scientific_scope='Simulated MAESTRO backend; real native Scanpy PCA/neighbors/Leiden/UMAP on synthetic 16x8 counts. Not original PBMC science.',
            figures=[f['figure_kind'] for f in figures]),indent=2))
