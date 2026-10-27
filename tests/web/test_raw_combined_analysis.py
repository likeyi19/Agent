"""Synthetic reads/references, native scientific owners, scripted language decisions.

This bounded integration test is not biological or live-provider qualification.
"""
from dataclasses import asdict
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent.schemas.orchestration import StepOutputRef
from agent.tools.data import _chromap, scatac_reference
from test_fastq_chat_handoff import chat_web, companion, raw
from test_fastq_execution import FASTQModel, tiny, reads, qualified_runtime
from test_chat_attachments import upload
from helpers import wait_turn

TEXT = ('Process this human 10x ATAC library sample, R1/R2/R3 in lane 001 chunk 001. '
        'R2 is raw barcode sequence requiring whitelist correction; R1 and R3 are genomic mates. '
        'Use minimum fragment records 0 and minimum TSS enrichment 0, build a cCRE matrix, '
        'then TF-IDF with L2 normalization and PCA with 3 components, 4 neighbors, Leiden and UMAP.')


class CombinedModel(FASTQModel):
    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        if 'turn_schema_version' in value:
            args = []
            for tool, argument, literal, prefix in (
                ('select_scATAC_cells', 'min_qc_fragment_records', '0', 'minimum fragment records '),
                ('select_scATAC_cells', 'min_tss_enrichment', '0', 'minimum TSS enrichment '),
                ('pca_embed_cells', 'n_components', '3', 'PCA with '),
                ('build_cell_neighbors', 'n_neighbors', '4', 'components, ')):
                start = TEXT.index(prefix) + len(prefix)
                args.append(dict(tool=tool, argument=argument, literal=literal, start=start, end=start+len(literal)))
            phrase = 'TF-IDF with L2 normalization'; start = TEXT.index(phrase)
            args.append(dict(tool='pca_embed_cells', argument='preprocessing', literal=phrase,
                start=start, end=start+len(phrase), value='tfidf_l2'))
            return json.dumps(dict(turn_schema_version=1, decision=dict(kind='request_roles',
                target='prepare_scATAC_fragments', required_roles=[], roles=[], arguments=args,
                fastq=raw(((0,'R1'),(1,'R2'),(2,'R3')), assay='TENX_ATAC', layout='tenx-atac-r1-r2-r3.v1',
                    namespace='sample', species='human', barcode_policy='raw_sequence_whitelist'))))
        if 'output_selection_schema_version' in value:
            return json.dumps(dict(outputs=[dict(name='matrix',step_id='matrix',output_key='matrix_path'),
                dict(name='umap',step_id='umap',output_key='analysis_path')]))
        result = json.loads(super().complete(prompt=prompt, response_schema=response_schema))
        if result.get('schema_version') == 4:
            def binding(target, kind, source):
                return dict(target=target,source=dict(kind=kind,**{kind:source}))
            result['decision']['steps'].extend([
                dict(step_id='pca',tool='pca_embed_cells',control_dependencies=[],sources=[
                    binding('dataset','step','matrix'),binding('preprocessing','input','pca_preprocessing'),
                    binding('n_components','input','n_components')]),
                dict(step_id='neighbors',tool='build_cell_neighbors',control_dependencies=[],sources=[
                    binding('embedding','step','pca'),binding('n_neighbors','input','n_neighbors')]),
                dict(step_id='clusters',tool='cluster_cells',control_dependencies=[],sources=[binding('analysis','step','neighbors')]),
                dict(step_id='umap',tool='compute_cell_umap',control_dependencies=[],sources=[binding('analysis','step','clusters')]),
            ])
        return json.dumps(result)


def test_native_raw_matrix_and_downstream_share_one_ordinary_web_plan(tiny, reads, monkeypatch):
    binary = os.environ.get('AGENT_CHROMAP_BIN')
    if not binary or not Path(binary).is_file():
        pytest.skip('Qualified native Chromap installation required')
    assert _chromap.sha256(binary) == _chromap.QUALIFIED_EXECUTABLE_SHA256
    # Expand the generated reference before any scientific execution.
    tiny['bed'].write_text(''.join(f'chrTiny\t{i}\t{i+150}\n' for i in range(100,8100,1000)))
    bundle = scatac_reference.build_scatac_reference_bundle(species='human', target_assembly='hg38',
        fasta_path=tiny['fa'],fai_path=tiny['fai'],ccre_bed_path=tiny['bed'])
    tiny['pointer'] = scatac_reference.publish_scatac_reference_bundle(bundle,tiny['root']/'expanded-reference.json')
    barcodes = tuple('ACGTACGTACGT'+''.join('ACGT'[(i >> (2*j)) & 3] for j in range(2))+'AC' for i in range(12))
    records = [(100+1000*j+i,220+1000*j+i,bc,1)
        for i,bc in enumerate(barcodes) for j in range(8) if (i+j)%3 or j==0]
    group, whitelist = reads(records,whitelist=barcodes)
    resource = companion(tiny,whitelist)
    qualified_runtime(tiny,dict(candidate=dict(path=binary)),monkeypatch)
    monkeypatch.setenv('AGENT_QC_ALLOW_SYNTHETIC','1')
    monkeypatch.delenv('AGENT_QC_RESOURCE_CATALOG',raising=False)
    monkeypatch.setenv('AGENT_QC_BEDTOOLS','/usr/bin/bedtools')
    monkeypatch.setenv('AGENT_MATRIX_BEDTOOLS','/usr/bin/bedtools')
    service, uploads, models, app = chat_web(tiny['root'],(resource,),model_type=CombinedModel)
    with TestClient(app) as client:
        client.post('/api/v1/sessions',json=dict(session_id='session'))
        files = {role.value:Path(path) for role,path in group.files}
        ids = [upload(client,role+'.fastq',files[role].read_bytes(),token=role).json()['resource_id']
            for role in ('R1','R2','R3')]
        response = client.post('/api/v1/sessions/session/turns',json=dict(
            turn_id='combined',expected_generation=0,utterance=TEXT,attachment_ids=ids))
        assert response.status_code==202,response.text
        view=wait_turn(client,'combined',timeout=120)
        assert view['revision_id'],view
        run=service._application.run_store.load(view['run_id'])
        assert len(run.steps)==9,run.to_dict()
        assert all(s.verification and s.verification.passed for s in run.steps),run.to_dict()
        pca=next(s for s in run.steps if s.tool_name=='pca_embed_cells')
        matrix=next(s for s in run.steps if s.tool_name=='build_scATAC_cell_by_ccre')
        assert pca.resolved_arguments['input_path']==matrix.result['matrix_path']
        assert pca.resolved_arguments['preprocessing']=='tfidf_l2'
        assert run.plan.steps[5].arguments['input_path']==StepOutputRef('matrix','matrix_path')
        import anndata as ad
        data=ad.read_h5ad(matrix.result['matrix_path'])
        assert data.shape==(12,8) and data.X.nnz>0
        assert pca.result['n_cells']==12 and pca.result['embedding_dim']==3
