"""Small real Scanpy prerequisite DAG; simulated native annotation is labelled."""
from dataclasses import asdict
import json
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from fastapi.testclient import TestClient
from agent.application.raw_handoff import AnnotationDeclaration
from agent.orchestration import AgentRequest, build_default_tool_registry
from agent.orchestration.semantic_compiler import (
    SemanticPlanCandidate, SemanticPlanStep, SemanticRegisteredChoiceSource,
    SemanticPlanCompileError, build_m92_semantic_compiler_contract, compile_semantic_plan)
from agent.tools.analysis import marker_annotation as science, annotation_contract as contract
from test_external_entry import external_case, case
from test_conversational_handoff import setup, HandoffModel, stage, send


def test_registered_choice_respects_explicit_input_and_allowlist():
    registry = build_default_tool_registry()
    contract = build_m92_semantic_compiler_contract(registry)
    def compile(choice, **inputs):
        return compile_semantic_plan(AgentRequest('choice','Explore cells',dict(input_path='/tmp/cells.h5ad',output_dir='/tmp/out',**inputs)),
            SemanticPlanCandidate((SemanticPlanStep('pca','pca_embed_cells',
                (SemanticRegisteredChoiceSource('preprocessing',choice),)),)), registry, contract, planner_name='test')
    assert compile('tfidf_l2').steps[0].arguments['preprocessing'] == 'tfidf_l2'
    with pytest.raises(SemanticPlanCompileError, match='supplied parameter'):
        compile('tfidf_l2', pca_preprocessing='none')
    with pytest.raises(SemanticPlanCompileError, match='not enabled'):
        compile('invented')


def test_unclustered_h5ad_normal_handoff_composes_prerequisites(external_case, tmp_path, monkeypatch):
    args, calls = external_case
    original = ad.read_h5ad(args['source_path'])
    data = ad.AnnData(sparse.csr_matrix(np.random.default_rng(4).poisson(.4,(24,71))),
        obs=pd.DataFrame(index=[f'cell{i}' for i in range(24)]),
        var=pd.DataFrame(index=[f'chr1:{i*2}-{i*2+1}' for i in range(71)]), uns=dict(original.uns))
    data.write_h5ad(args['source_path'])
    original_complete = HandoffModel.complete
    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        if 'turn_schema_version' in value:
            decision=dict(kind='request_roles',target='annotate_scATAC_cell_types',required_roles=[],roles=[],arguments=[],
                annotation=asdict(AnnotationDeclaration(source='source.0', species='human',assembly='hg38',
                    measurement=contract.EXTERNAL_SEMANTICS,context='synthetic context')))
            return json.dumps(dict(turn_schema_version=1,decision=decision))
        if 'semantic_prompt_version' in value:
            assert value['research_context']['task']['source_observation']['qualified_groupings'] == []
            def step(name,tool,*sources):
                return dict(step_id=name,tool=tool,sources=[dict(target=t,source=s) for t,s in sources],control_dependencies=[])
            def prior(name,port): return dict(kind='step_port',step=name,source_port=port)
            steps=[step('inspection','inspect_scATAC',('dataset',dict(kind='input',input='source_path'))),
                step('pca','pca_embed_cells',('dataset',prior('inspection','dataset')),
                     ('preprocessing',dict(kind='registered_choice',choice='tfidf_l2'))),
                step('neighbors','build_cell_neighbors',('embedding',prior('pca','embedding'))),
                step('clusters','cluster_cells',('analysis',prior('neighbors','neighbors'))),
                step('annotation','annotate_scATAC_cell_types',('clustering',prior('clusters','clustered_analysis')))]
            return json.dumps(dict(schema_version=4,decision=dict(kind='plan',steps=steps)))
        return original_complete(self,prompt=prompt,response_schema=response_schema)
    monkeypatch.setattr(HandoffModel,'complete',complete)
    app, web = setup(args,tmp_path/'web',[])
    with TestClient(web) as client:
        source = stage(client,args)
        done = send(client,'first','Please annotate these cells.',[source])
        assert done['status'] == 'succeeded', done
        run = app._application.run_store.load(done['run_id'])
        assert [s.tool_name for s in run.steps] == ['inspect_scATAC','pca_embed_cells','build_cell_neighbors','cluster_cells','annotate_scATAC_cell_types']
        assert run.steps[1].resolved_arguments['preprocessing'] == 'tfidf_l2'
        assert len(calls) == 1
        annotated = ad.read_h5ad(run.steps[-1].result['annotated_h5ad_path'])
        grouped = ad.read_h5ad(run.steps[-2].result['analysis_path'])
        assert annotated.obs_names.equals(data.obs_names) and annotated.var_names.equals(data.var_names)
        assert (annotated.X != data.X).nnz == 0
        assert annotated.obs['annotation_group'].astype(str).tolist() == grouped.obs['leiden'].astype(str).tolist()
        from agent.tools.analysis.external_marker_annotation import group_rows
        spec = dict(groups=dict(analysis_path=run.steps[-2].result['analysis_path'],
            analysis_sha256=run.steps[-2].result['analysis_sha256'],source_sha256='0'*64))
        with pytest.raises(ValueError,match='exact original'):
            group_rows(data,spec)
