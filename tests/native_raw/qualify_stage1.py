"""Opt-in small genuine 10x PBMC500 qualification; never writes original inputs.

Run under activated agent with --output at a NEW task-owned external directory.
Uses scripted semantic planning, real native science and real Session/RunStore
reuse. No provider/browser or biological accuracy qualification is implied.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from agent.application import OutputSelection, ResearchAgentApplication
from agent.orchestration import AgentPlan, AgentRequest, LLMPlanner, PlanStep, StepOutputRef
from agent.orchestration.prior_outputs import bind_output
from agent.schemas.prior_output import PriorOutputRef
from agent.schemas.orchestration import _serialize


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


class SpectralModel:
    model_id = 'scripted-stage1-native-scatac'
    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        assert value.get('schema_version', value.get('planning_schema_version', 4)) == 4
        steps = []
        for i, fn in enumerate(('pp.select_features', 'tl.spectral', 'pp.knn', 'tl.leiden', 'tl.umap')):
            sources = [dict(target='dataset', source=(dict(kind='input', input='input_path') if i == 0
                else dict(kind='step_port', step='s'+str(i-1), source_port='dataset'))),
                dict(target='function_name', source=dict(kind='registered_choice', choice=fn))]
            steps.append(dict(step_id='s'+str(i), tool='run_snapatac2', sources=sources, control_dependencies=[]))
        return json.dumps(dict(schema_version=4, decision=dict(kind='plan', steps=steps)))


def prepare(root, data):
    original = data/'datasets/original/10x-genomics/pbmc-500-nextgem/cell-atac-2.0.0'
    matrix = original/'atac_pbmc_500_nextgem_filtered_peak_bc_matrix.h5'
    fragments = original/'atac_pbmc_500_nextgem_fragments.tsv.gz'
    with h5py.File(matrix) as h:
        g = h['matrix']; shape = tuple(g['shape'][:])
        x = sparse.csc_matrix((g['data'][:], g['indices'][:], g['indptr'][:]), shape=shape).T.tocsr()
        cells = g['barcodes'].asstr()[:].tolist(); features = g['features/name'].asstr()[:].tolist()
    a = ad.AnnData(x[:128], obs=pd.DataFrame(index=cells[:128]), var=pd.DataFrame(index=features))
    a.uns['matrix_semantics'] = 'peak_counts'
    a.write_h5ad(root/'processed.h5ad')
    with (root/'peaks.bed').open('w') as stream:
        for name in features:
            chrom, region = name.split(':'); start, end = region.split('-')
            stream.write(f'{chrom}\t{start}\t{end}\n')
    vendor = set(cells); selected = set(cells[:128]); background = set()
    # Deterministic 128 supplied called barcodes plus first 128 other observed
    # barcodes. Retain ALL source records for the chosen identities across genome.
    with gzip.open(fragments, 'rt') as stream:
        for line in stream:
            if line.startswith('#'): continue
            barcode = line.split('\t')[3]
            if barcode not in vendor: background.add(barcode)
            if len(background) == 128: break
    selected |= background
    retained = 0
    with gzip.open(fragments, 'rt') as stream, (root/'fragments.tsv').open('w') as out:
        for line in stream:
            if not line.startswith('#') and line.split('\t')[3] in selected:
                out.write(line); retained += 1
    provenance = dict(source_files={str(p):sha(p) for p in (matrix, fragments)},
        selected_called_barcodes=128, selected_other_barcodes=len(background), retained_records=retained,
        limitation='Barcode-enriched small technical sample; thresholds do not qualify full-sample cell calling')
    (root/'input-provenance.json').write_text(json.dumps(provenance, indent=2))
    return provenance


class RawPlanner:
    def __init__(self, root, data): self.root, self.data = root, data
    def plan(self, request, registry):
        steps = []
        functions = ('pp.import_fragments', 'metrics.tsse', 'pp.filter_cells', 'pp.make_peak_matrix',
            'pp.select_features', 'tl.spectral', 'pp.knn', 'tl.leiden', 'tl.umap')
        for i, fn in enumerate(functions):
            args = dict(function_name=fn, output_dir=str(Path(request.inputs['output_dir'])/('s'+str(i))),
                input_path=str(self.root/'fragments.tsv') if i == 0 else StepOutputRef('s'+str(i-1), 'matrix_path'))
            if fn == 'pp.import_fragments':
                args.update(chrom_sizes_path=str(self.data/'resources/references/human/hg38/genome/hg38.fa.fai'),
                    fragment_order='unknown', qc_thresholds='otsu')
            if fn == 'metrics.tsse':
                args['gene_anno_path'] = str(self.data/'resources/references/human/hg38/gencode-49/annotation.gtf')
            if fn == 'pp.filter_cells': args['qc_thresholds'] = 'otsu'
            if fn == 'pp.make_peak_matrix': args['peaks_path'] = str(self.root/'peaks.bed')
            steps.append(PlanStep('s'+str(i), 'run_snapatac2', args, depends_on=() if i == 0 else ('s'+str(i-1),)))
        return AgentPlan(request.request_id, request.request_id, 'Native raw QC to Spectral, Leiden and UMAP', tuple(steps))


def main(root, data):
    root.mkdir(parents=True, exist_ok=False)
    provenance = prepare(root, data)
    outcomes = {}
    for name, planner, inputs, final in (
            ('raw', RawPlanner(root, data), {}, 's8'),
            ('processed', LLMPlanner(SpectralModel()), dict(input_path=str(root/'processed.h5ad')), 's4')):
        app = ResearchAgentApplication(root/name, planner=planner); app.sessions.create(name)
        state = app.sessions.run(name, 'first', AgentRequest(name, 'Native scATAC Spectral clustering and UMAP.', inputs),
            (OutputSelection('dataset', final, 'matrix_path'),), expected_generation=0)
        run = app.run_store.load(name+':run')
        (root/(name+'-run.json')).write_text(json.dumps(run.to_dict(), indent=2))
        assert state.turn('first').status == 'activated', run.errors
        assert all(s.verification.passed and s.attempt_count == 1 for s in run.steps)
        output = ad.read_h5ad(run.steps[-1].result['matrix_path'])
        assert np.isfinite(output.obsm['X_spectral']).all() and output.obsm['X_umap'].shape == (output.n_obs, 2)
        assert 'leiden' in output.obs and output.X.dtype.kind in 'iu'
        # Native feature selection/reduction/clustering preserve matrix and axes.
        matrix_step = run.steps[3] if name == 'raw' else None
        original = ad.read_h5ad(matrix_step.result['matrix_path'] if matrix_step else root/'processed.h5ad')
        assert (original.X != output.X).nnz == 0
        assert original.obs_names.tolist() == output.obs_names.tolist() and original.var_names.tolist() == output.var_names.tolist()
        binding = bind_output(app.run_store, app.registry, name+':run', 's7' if name == 'raw' else 's3', 'dataset')
        snapshot = {p:sha(p) for p in app._workspace.run_paths(name+':run').scientific.rglob('*') if p.is_file()}
        class Continue:
            def plan(self, request, registry):
                return AgentPlan(request.request_id, request.request_id, 'Reuse exact accepted Spectral/Leiden result', (
                    PlanStep('map', 'run_snapatac2', dict(function_name='tl.umap',
                        input_path=PriorOutputRef(binding,'matrix_path'), output_dir=str(Path(request.inputs['output_dir'])/'map'))),))
        restarted = ResearchAgentApplication(root/name, planner=Continue())
        state = restarted.sessions.run(name, 'later', AgentRequest(name+'-later', 'UMAP from the same accepted reduction.', {}),
            (OutputSelection('map', 'map', 'matrix_path'),), expected_generation=1)
        later = restarted.run_store.load(name+'-later:run')
        assert state.turn('later').status == 'activated', later.errors
        assert len(later.steps) == 1 and later.steps[0].attempt_count == 1
        assert all(sha(p) == checksum for p, checksum in snapshot.items())
        repeated = ad.read_h5ad(later.steps[0].result['matrix_path'])
        np.testing.assert_array_equal(output.obsm['X_spectral'], repeated.obsm['X_spectral'])
        assert output.obs.leiden.equals(repeated.obs.leiden)
        outcomes[name] = dict(n_cells=output.n_obs, n_features=output.n_vars,
            spectral_shape=list(output.obsm['X_spectral'].shape), groups=output.obs.leiden.nunique(),
            functions=[s.result['function_name'] for s in run.steps],
            effective_qc=run.steps[2].result['parameters'] if name == 'raw' else None,
            restart_continuation_steps=1, source_counts_and_axes_preserved=True)
    result = dict(provenance=provenance, outcomes=_serialize(outcomes), qualification='genuine native science; scripted planning; bounded technical sample')
    (root/'RESULT.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data',type=Path,default=Path('/home/likeyi/program/agent-data'))
    args=p.parse_args();main(args.output,args.data)
