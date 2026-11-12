"""Real registered native routes + installed CellMarker, small artificial populations.

Opt in with RUN_RAW_NATIVE_QUALIFICATION=1. Not biological accuracy validation.
"""
import hashlib
import io
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from agent.application import ResearchAgentApplication, OutputSelection
from agent.application.shared_resources import ScientificResourceLibrary
from agent.application.statistical_delivery import create_statistical_export
from agent.application.workspace import ManagedWorkspace
from agent.orchestration import AgentPlan, AgentRequest, PlanStep, StepOutputRef
from agent.orchestration.prior_outputs import bind_output, accepted_step_digest
from agent.schemas.orchestration import _serialize
from agent.tools.analysis.marker_candidates import annotate_markers, candidate_evidence, marker_kwargs
from agent.tools.analysis.native_matrix import SNAPATAC2, SCANPY, EPISCANPY
from agent.tools.data.cellmarker import CellMarkerEvidence


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


@pytest.fixture(scope='module')
def evidence():
    if os.environ.get('RUN_RAW_NATIVE_QUALIFICATION') != '1':
        pytest.skip('Explicit installed native runtime qualification')
    library = ScientificResourceLibrary(Path(os.environ['AGENT_SCIENTIFIC_RESOURCE_ROOT']))
    entries = [r for r in library.entries() if r['kind'] == 'cellmarker'
               and r['resource']['record_url'] == 'https://zenodo.org/records/22808257']
    assert len(entries) == 1
    return CellMarkerEvidence(library, entries[0]['id'])


@pytest.mark.parametrize('route', ['existing', 'episcanpy', 'snapatac2'])
def test_real_activity_markers_and_candidates(tmp_path, evidence, route, monkeypatch):
    names = ['Cd3d', 'Cd3e', 'Trac', 'Cd247', 'Ms4a1', 'Cd79a', 'Cd79b', 'Cd19']
    names += [f'unmatched_gene_{i}' for i in range(92)]
    x = np.ones((24, 100), dtype=np.float32)
    x[:12, :4] = 8
    x[12:, 4:8] = 8
    a = ad.AnnData(sparse.csr_matrix(x),
        obs=pd.DataFrame({'clusters':pd.Categorical(['T_fixture']*12+['B_fixture']*12)},
                         index=[f'cell{i:02}' for i in range(24)]),
        var=pd.DataFrame(index=names))
    a.uns.update(matrix_semantics='gene_activity', species='Mouse')
    source = tmp_path/'original.h5ad'
    a.write_h5ad(source)
    gtf = tmp_path/'genes.gtf'
    gtf.write_text(''.join(f'chr1\tHAVANA\t{kind}\t{201+i*200}\t{300+i*200}\t.\t+\t.\tgene_id "G{i}"; transcript_id "T{i}"; gene_name "{name}";\n'
        for kind in ('gene','transcript') for i, name in enumerate(names)))
    if route == 'snapatac2':
        fragments = tmp_path/'fragments.tsv'
        fragments.write_text(''.join(f'chr1\t{200+j*200+k}\t{220+j*200+k}\t{cell}\t1\n'
            for i,cell in enumerate(a.obs_names) for j in range(100) for k in range(int(x[i,j]))))
        imported = SNAPATAC2.execute(dict(function_name='pp.import_fragments', input_path=str(fragments),
            output_dir=str(tmp_path/'import'), kwargs=dict(chrom_sizes={'chr1':21000}, min_num_fragments=0)))
        imported_before = sha(imported['matrix_path'])
        grouped = ad.read_h5ad(imported['matrix_path'])
        grouped.obs['clusters'] = a.obs['clusters']
        source = tmp_path/'grouped-fragments.h5ad'
        grouped.write_h5ad(source)
        tool, kwargs = 'run_snapatac2', dict(function_name='pp.make_gene_matrix', gene_anno_path=str(gtf), kwargs=dict(upstream=0))
    elif route == 'episcanpy':
        peaks = a.copy()
        peaks.var_names = [f'chr1:{200+i*200}-{220+i*200}' for i in range(100)]
        peaks.uns['matrix_semantics'] = 'peak_counts'
        source = tmp_path/'peaks.h5ad'
        peaks.write_h5ad(source)
        tool, kwargs = 'run_episcanpy', dict(function_name='tl.geneactivity', gene_anno_path=str(gtf), kwargs=dict(upstream=0))
    else:
        tool, kwargs = 'run_scanpy', dict(function_name='pp.calculate_qc_metrics', gene_activity=True,
                                         kwargs=dict(percent_top=None, log1p=False))
    before = sha(source)
    root = tmp_path/'workspace'
    out = root/'runs'/ManagedWorkspace.run_digest('markers:run')/'scientific'
    plan = AgentPlan('markers','markers','Explicit activity and full native Wilcoxon', (
        PlanStep('activity',tool,dict(input_path=str(source),output_dir=str(out/'activity'),**kwargs)),
        PlanStep('markers','run_scanpy',dict(function_name='tl.rank_genes_groups',
            input_path=StepOutputRef('activity','matrix_path'),output_dir=str(out/'markers'),
            kwargs=marker_kwargs('clusters')),('activity',))))
    class Planner:
        def plan(self, request, registry): return plan
    app = ResearchAgentApplication(root, planner=Planner())
    app.sessions.create('session')
    state = app.sessions.run('session','markers',AgentRequest('markers','Explicit native marker comparison',{}),
        (OutputSelection('dataset','markers','matrix_path'),),expected_generation=0)
    run = app.run_store.load('markers:run')
    assert state.turn('markers').status == 'activated', run.errors
    activity, markers = [s.result for s in run.steps]
    data = ad.read_h5ad(markers['matrix_path'])
    np.testing.assert_array_equal(array(data.X),array(ad.read_h5ad(activity['matrix_path']).X))
    assert 'log1p' not in data.uns
    assert markers['source_identity']['accepted_publication']['manifest_sha256'] == activity['manifest_sha256']
    assert data.n_vars == 100
    binding = bind_output(app.run_store,app.registry,run.run_id,'markers','dataset')
    marker_before = sha(markers['matrix_path'])
    # A restarted caller may interpret accepted markers without rerunning science.
    app = ResearchAgentApplication(root)
    for owner in (SCANPY, EPISCANPY, SNAPATAC2):
        monkeypatch.setattr(owner, 'native', lambda *a, **k: pytest.fail('Native work repeated during evidence interpretation'))
    result = annotate_markers(binding, store=app.run_store, registry=app.registry, evidence=evidence, species='Mouse')
    assert len(result['universe']) == 100
    assert len(result['clusters']) == 2
    assert result['provenance']['binding']['accepted_step_sha256'] == accepted_step_digest(run.steps[1])
    assert result['provenance']['native_result']['parameters']['effective_native_parameters']['method'] == 'wilcoxon'
    for cluster in result['clusters']:
        assert len(cluster['markers']) == 100
        assert cluster['comparison'] == 'rest'
        assert cluster['status'] in ('assigned','ambiguous')
        assert len(cluster['positive_genes']) == 4
        assert all(m['score'] > 0 and m['pvalue_adj'] <= .05 for m in cluster['markers'] if m['positive'])
        assert all('pts' in m['detection_fractions'] and 'pts_rest' in m['detection_fractions'] for m in cluster['markers'])
        assert all(t['N'] == 100 and t['n'] == 4 for t in cluster['tests'])
        assert any(t['k'] == 0 and t['pvalue'] == 1 for t in cluster['tests'])
        assert cluster['untestable_candidates']
    categories = set(c for r in result['candidates'] for c in r['evidence']['source_categories'])
    assert categories == {'Method','Single-cell sequencing','Experiment','Review','Company'}
    preferred = result['clusters'][0]['preferred'][0]
    rows = [r for page in candidate_evidence(result,preferred,evidence) for r in page['items']]
    assert rows and all(r['fields']['species'] == 'Mouse' for r in rows)
    assert any(r['fields']['pmid'] for r in rows)
    assert all(r['source_rows'] for r in rows)
    for options in (dict(evidence=None), dict(evidence=evidence, identity_column='missing'),
                    dict(evidence=evidence, species='unavailable'),
                    dict(evidence=evidence, context={'tissue_type':'nonexistent tissue'})):
        absent = annotate_markers(binding,store=app.run_store,registry=app.registry,**options)
        assert all(c['status'] == 'insufficient_evidence' for c in absent['clusters'])
    unresolved = annotate_markers(binding,store=app.run_store,registry=app.registry,evidence=evidence,species='Mouse',marker_alpha=0)
    assert all(c['status'] == 'unresolved' for c in unresolved['clusters'])
    assert sha(source) == before and sha(markers['matrix_path']) == marker_before
    if route == 'snapatac2': assert sha(imported['matrix_path']) == imported_before
    # Existing delivery retains the actual native statistics and activity provenance.
    delivered = io.BytesIO()
    path = Path(markers['matrix_path'])
    with path.open('rb') as stream:
        create_statistical_export(stream,delivered,dict(tool_name='run_scanpy',accepted_step_sha256=accepted_step_digest(run.steps[1])),
            source_sha256=markers['matrix_sha256'],source_size_bytes=path.stat().st_size,manifest_sha256=markers['manifest_sha256'],max_bytes=1024**3)
    exported = ad.read_h5ad(delivered)
    np.testing.assert_array_equal(exported.uns['rank_genes_groups']['scores'],data.uns['rank_genes_groups']['scores'])
    (tmp_path/'candidate-result.json').write_text(json.dumps(_serialize(result),indent=2,allow_nan=False))


@pytest.mark.parametrize('case', ['unmatched', 'no_signal', 'truncated'])
def test_real_negative_evidence_is_not_native_failure(tmp_path, evidence, case):
    names = ['Cd3e','Cd3d','Trac'] + [f'unmatched_gene_{i}' for i in range(27)]
    if case == 'unmatched': names = [f'unmatched_gene_{i}' for i in range(30)]
    x = np.ones((24,30),dtype='float32')
    if case != 'no_signal': x[:12,:3] = 8
    source = tmp_path/'input.h5ad'
    a = ad.AnnData(x,obs=pd.DataFrame({'cluster':pd.Categorical(['one']*12+['two']*12)},index=[f'c{i}' for i in range(24)]),var=pd.DataFrame(index=names))
    a.write_h5ad(source)
    kwargs = marker_kwargs('cluster')
    if case == 'truncated': kwargs['n_genes'] = 3
    class Planner:
        def plan(self,request,registry):
            return AgentPlan('negative',request.request_id,'Explicit native markers',(
                PlanStep('markers','run_scanpy',dict(input_path=str(source),output_dir=str(Path(request.inputs['output_dir'])/'markers'),
                    function_name='tl.rank_genes_groups',gene_activity=True,kwargs=kwargs)),))
    app = ResearchAgentApplication(tmp_path/'app',planner=Planner())
    outcome = app.run(AgentRequest('negative','Explicit native markers',{}))
    assert outcome.status.value == 'SUCCEEDED', outcome.run_result.errors
    binding = bind_output(app.run_store,app.registry,'negative:run','markers','dataset')
    result = annotate_markers(binding,store=app.run_store,registry=app.registry,evidence=evidence,species='Mouse')
    expected = 'unresolved' if case == 'no_signal' else 'insufficient_evidence'
    assert all(c['status'] == expected for c in result['clusters'])
    if case == 'no_signal':
        assert all(t['n'] == 0 and t['pvalue'] == 1 for c in result['clusters'] for t in c['tests'])
    if case == 'truncated':
        assert all(any('full tested' in reason for reason in c['reasons']) for c in result['clusters'])
    (tmp_path/'candidate-result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
