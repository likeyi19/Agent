"""Small real SnapATAC2 prerequisite DAG; simulated native annotation is labelled."""
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
from test_spectral_continuation import native_runtime
from test_conversational_handoff import setup, HandoffModel, stage, send


def test_registered_choice_respects_explicit_input_and_allowlist():
    registry = build_default_tool_registry()
    catalog = build_m92_semantic_compiler_contract(registry)
    def compile(choice, **inputs):
        return compile_semantic_plan(AgentRequest('choice','Explore cells',dict(input_path='/tmp/cells.h5ad',output_dir='/tmp/out',**inputs)),
            SemanticPlanCandidate((SemanticPlanStep('spectral','run_snapatac2',
                (SemanticRegisteredChoiceSource('function_name',choice),)),)), registry, catalog, planner_name='test')
    assert compile('tl.spectral').steps[0].arguments['function_name'] == 'tl.spectral'
    with pytest.raises(SemanticPlanCompileError, match='supplied parameter'):
        compile('tl.spectral', function_name='pp.select_features')
    with pytest.raises(SemanticPlanCompileError, match='not enabled'):
        compile('invented')


def test_unclustered_h5ad_normal_handoff_composes_prerequisites(external_case, tmp_path, monkeypatch, native_runtime):
    args, calls = external_case
    original = ad.read_h5ad(args['source_path'])
    data = ad.AnnData(sparse.csr_matrix(np.random.default_rng(4).poisson(.4,(128,71))),
        obs=pd.DataFrame(index=[f'cell{i}' for i in range(128)]),
        var=pd.DataFrame(index=[f'chr1:{i*2}-{i*2+1}' for i in range(71)]), uns=dict(original.uns))
    data.X[:64, 35:] = 0; data.X[64:, :35] = 0
    data.X.eliminate_zeros()
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
            steps=[step('inspection','inspect_scATAC',('dataset',dict(kind='input',input='source_path')))]
            parent = 'inspection'
            for name, fn in [('features','pp.select_features'), ('spectral','tl.spectral'), ('neighbors','pp.knn'), ('clusters','tl.leiden')]:
                steps.append(step(name,'run_snapatac2',('dataset',prior(parent,'dataset')),
                    ('function_name',dict(kind='registered_choice',choice=fn))))
                parent = name
            steps.append(step('annotation','annotate_scATAC_cell_types',('source',dict(kind='input',input='source_path')),('clustering',prior('clusters','dataset'))))
            return json.dumps(dict(schema_version=4,decision=dict(kind='plan',steps=steps)))
        return original_complete(self,prompt=prompt,response_schema=response_schema)
    monkeypatch.setattr(HandoffModel,'complete',complete)
    app, web = setup(args,tmp_path/'web',[])
    with TestClient(web) as client:
        source = stage(client,args)
        done = send(client,'first','Please annotate these cells.',[source])
        assert done['status'] == 'succeeded', (done,app._application.run_store.load(done['run_id']).errors if done.get('run_id') else None)
        run = app._application.run_store.load(done['run_id'])
        assert [s.tool_name for s in run.steps] == ['inspect_scATAC']+['run_snapatac2']*4+['annotate_scATAC_cell_types']
        assert all('kwargs' not in s.resolved_arguments for s in run.steps[1:-1])
        assert len(calls) == 1
        annotated = ad.read_h5ad(run.steps[-1].result['annotated_h5ad_path'])
        grouped = ad.read_h5ad(run.steps[-2].result['matrix_path'])
        assert annotated.obs_names.equals(data.obs_names) and annotated.var_names.equals(data.var_names)
        assert (annotated.X != data.X).nnz == 0
        assert annotated.obs['annotation_group'].astype(str).tolist() == grouped.obs['leiden'].astype(str).tolist()
