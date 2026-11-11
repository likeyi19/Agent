"""Opt-in real ASW acceptance, with existing systemd isolation and no providers.

Run under conda agent with PYTHONPATH=src and --output pointing to a NEW external
acceptance directory. Original scientific files and independent evidence are read-only.
"""
import argparse
import json
import os
from pathlib import Path

from test_asw_selection import run_case, run_count_fallback



def verify_reuse(root):
    """New clustering only, reusing exact accepted inputs; check deterministic science."""
    import threading
    import time
    from agent.application import ResearchAgentApplication, OutputSelection
    from agent.orchestration import AgentPlan, AgentRequest, PlanStep
    from agent.orchestration.prior_outputs import bind_output
    from agent.schemas import PriorOutputRef, StepOutputRef
    from agent.schemas.orchestration import _serialize
    app = ResearchAgentApplication(root/'app')
    state = app.sessions.load('asw')
    initial = app.run_store.load(state.revisions[0].run_id)
    old = next(s for s in initial.steps if s.step_id=='clusters')
    planned = next(s for s in initial.plan.steps if s.step_id=='clusters')
    source = planned.arguments['input_path']
    assert isinstance(source, StepOutputRef)
    binding = bind_output(app.run_store, app.registry, initial.run_id, source.step_id, 'dataset')
    request = AgentRequest('asw-exact-reuse', 'Repeat the native bounded selection on the exact accepted input.', {})
    _, paths = app._prepare_request(request)
    args = dict(function_name=old.resolved_arguments['function_name'], parameter_selection='asw',
        input_path=PriorOutputRef(binding, source.output_key), output_dir=str(paths.scientific/'clusters'))
    if 'kwargs' in planned.arguments:
        args['kwargs'] = _serialize(planned.arguments['kwargs'])
    plan = AgentPlan('asw-reuse', request.request_id, 'accepted input reuse', (PlanStep('clusters', old.tool_name, args),))
    samples = []
    stop = threading.Event()
    group = Path('/sys/fs/cgroup/user.slice/user-1001.slice/user@1001.service/app.slice/agent-native-task.service')
    def observe():
        while not stop.wait(.05):
            try:
                samples.append(int((group/'memory.peak').read_text()))
            except FileNotFoundError:
                pass
    monitor = threading.Thread(target=observe, daemon=True);monitor.start();started=time.monotonic()
    try:
        after = app.sessions.run('asw', 'exact-reuse', request, (OutputSelection('analysis','clusters','matrix_path'),),
            expected_generation=state.generation, prepared_plan=plan)
    finally:
        elapsed=time.monotonic()-started;stop.set();monitor.join()
    assert after.turn('exact-reuse').status=='activated'
    result=app.run_store.load(after.revisions[-1].run_id)
    assert len(result.steps)==1
    assert result.steps[0].result['parameters']['parameter_selection']==old.result['parameters']['parameter_selection']
    assert result.steps[0].result['source_identity']==old.result['source_identity']
    reopened=ResearchAgentApplication(root/'app')
    assert reopened.sessions.load('asw').revisions[-1]==after.revisions[-1]
    evidence=dict(run_id=result.run_id, selection=_serialize(result.steps[0].result['parameters']['parameter_selection']),
        exact_candidate_evidence_repeated=True, native_steps=['clusters'], elapsed_seconds=elapsed,
        observed_cgroup_peak_bytes=max(samples) if samples else None, memory_observations=len(samples),
        observation_interval_seconds=.05, source_binding=binding.to_dict())
    (root/'reuse-evidence.json').write_text(json.dumps(evidence,indent=2))
    print(json.dumps(dict(case=root.name,repeat='passed',peak_bytes=evidence['observed_cgroup_peak_bytes'])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-reuse', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Continue assertions/edits after a saved successful initial result; no scientific retry.')
    parser.add_argument('--case', required=True, choices=('snap-spectral', 'snap-epizoo', 'scanpy-pca',
        'scanpy-epizoo', 'epi-leiden', 'epi-louvain', 'epi-kmeans', 'epi-hc', 'epi-epizoo',
        'epi-kmeans-unavailable', 'epi-hc-unavailable'))
    args = parser.parse_args()
    root = args.output / args.case
    if not args.verify_reuse:
        root.mkdir(parents=True, exist_ok=args.resume)
    resources = Path('/home/likeyi/program/agent-data/resources')
    os.environ.update(AGENT_SCIENTIFIC_RESOURCE_ROOT=str(resources/'library'),
        AGENT_NATIVE_MEMORY_BYTES=str(4*1024**3), AGENT_NATIVE_TIMEOUT_SECONDS='600',
        AGENT_SNAPATAC2_PYTHON=str(resources/'runtimes/native-matrix-1/env/bin/python'),
        AGENT_SCANPY_PYTHON=str(resources/'runtimes/episcanpy-0.4.0/env/bin/python'),
        AGENT_EPISCANPY_PYTHON=str(resources/'runtimes/episcanpy-0.4.0/env/bin/python'))
    if args.case.endswith('-unavailable'):
        if args.resume or args.verify_reuse:
            parser.error('Unavailable-ASW acceptance requires a new, complete execution.')
        evidence = run_count_fallback(root, 'tl.kmeans' if 'kmeans' in args.case else 'tl.hc')
        print(json.dumps(dict(case=args.case, status='passed',
            failed_run_id=evidence['failed_run_id'], continued_run_id=evidence['continued_run_id'])))
        return
    if args.verify_reuse:
        verify_reuse(root)
        return
    acceptance = resources.parent/'acceptance'
    counts = acceptance/'scientific-results/stage1-native-scatac-2026-11-10/downloads/processed.h5ad'
    embedding = acceptance/'cross-capability-workflow-completeness/cwc-1/blocker-closeout/workspace/runs/6bee03c04532e4946d01f4c73622c9eb2a0fcea32ea366b31ca5d32ff7e45b0d/scientific/30df8b726fefa6bc3d966fda5cffd01aa241e51759a79e75ab03d36188170204.epizoo_embeddings.h5ad'
    package, function, source, representation, umap = {
        'snap-spectral': ('snapatac2','tl.leiden',counts,'X_spectral',False),
        'snap-epizoo': ('snapatac2','tl.leiden',embedding,'X_epizoo',False),
        'scanpy-pca': ('scanpy','tl.leiden',counts,None,False),
        'scanpy-epizoo': ('scanpy','tl.leiden',embedding,'X_epizoo',False),
        'epi-leiden': ('episcanpy','tl.leiden',counts,None,False),
        'epi-louvain': ('episcanpy','tl.louvain',counts,None,True),
        'epi-kmeans': ('episcanpy','tl.kmeans',counts,None,False),
        'epi-hc': ('episcanpy','tl.hc',counts,None,False),
        'epi-epizoo': ('episcanpy','tl.leiden',embedding,'X_epizoo',False),
    }[args.case]
    source_preparation = None
    if source == counts and representation is None:
        import anndata as ad
        import hashlib
        import numpy as np
        matrix = ad.read_h5ad(counts)
        original = matrix.X
        matrix.X = original.astype('float32')
        assert np.array_equal(matrix.X.data.astype(original.dtype), original.data)
        source = root/'supplied-float-counts.h5ad'
        matrix.write_h5ad(source)
        source_preparation = dict(original=str(counts), original_sha256=hashlib.sha256(counts.read_bytes()).hexdigest(),
            change='Acceptance fixture explicitly supplies lossless float32 counts for native PCA; no production casting.',
            ordered_cells_and_features='unchanged')
    evidence = run_case(root, package, function, source, representation, umap,
                        unavailable=args.case=='snap-epizoo', resume=args.resume)
    evidence['source_preparation'] = source_preparation
    diagnostics = [json.loads(p.read_text()) for p in root.rglob('*execution.json')]
    assert diagnostics and all(r.get('completed') and not r.get('worker_error') for r in diagnostics), diagnostics
    evidence['resource_observations'] = diagnostics
    (root/'evidence.json').write_text(json.dumps(evidence, indent=2))
    print(json.dumps(dict(case=args.case, selected=evidence['selection']['selected_value'],
                         scores=evidence['selection']['candidates'], isolated_calls=len(diagnostics))))


if __name__ == '__main__':
    main()
