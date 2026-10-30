"""Real publication/recovery contracts with a simulated native optimizer."""
from pathlib import Path
import hashlib

import pytest

from agent.application import ResearchAgentApplication, OutputSelection
from agent.orchestration import (AgentRequest, SemanticPlanCandidate, SemanticPlanStep,
    SemanticRequestInputSource, SemanticStepOutputSource, build_semantic_compiler_contract,
    compile_semantic_plan, PlanStep)
from agent.orchestration.semantic_compiler import isolate_output_directories
from agent.tools.analysis import sequence_prediction as shared, sequence_optimization as design
from test_scatac_regulatory_sequences import source
from test_sequence_optimization import native


class Planner:
    def plan(self, request, registry):
        prepare = 'prepare_scatac_regulatory_sequences'
        first = [SemanticRequestInputSource('dataset', 'input_path')]
        for k in ('species','assembly','coordinate_source','coordinate_system','matrix_semantics',
                  'group_key','group_value','top_n','window_size','output_dir'):
            first.append(SemanticRequestInputSource(k, k))
        second = [SemanticStepOutputSource('sequences', 'prepare', 'sequences')]
        for k in ('condition','iterations','edit_start','edit_end','protected_regions','output_dir'):
            second.append(SemanticRequestInputSource(k, k))
        candidate = SemanticPlanCandidate((
            SemanticPlanStep('prepare', prepare, tuple(first)),
            SemanticPlanStep('design', design.TOOL, tuple(second))))
        return compile_semantic_plan(request, candidate, registry,
            build_semantic_compiler_contract(registry), planner_name='simulated-contract')


def test_sequential_publications_exact_consumption_restart_and_tamper(source, native, monkeypatch):
    args, _, _, _ = source
    inputs = {k:v for k,v in args.items() if k != 'output_dir'}
    inputs.update(window_size=2114, top_n=1, condition='B_cell', iterations=1,
        edit_start=900, edit_end=1200, protected_regions='1000:1100')
    root = Path(args['output_dir']).parent/'managed'
    app = ResearchAgentApplication(root, planner=Planner());app.sessions.create('session')
    state = app.sessions.run('session', 'turn', AgentRequest('compound', 'Prepare and optimize', inputs),
        (OutputSelection('designed', 'design', 'manifest_path'),), expected_generation=0)
    assert state.turn('turn').status == 'activated', app.run_store.load('compound:run').errors
    run = app.run_store.load('compound:run');first, second = run.steps
    base = app._workspace.run_paths(run.run_id).scientific
    for step in run.steps:
        destination = Path(step.resolved_arguments['output_dir'])
        assert destination == base/'steps'/hashlib.sha256(step.step_id.encode()).hexdigest()
        assert str(destination) == str(next(s for s in run.plan.steps if s.step_id==step.step_id).arguments['output_dir'])
        assert step.verification.passed
    assert first.resolved_arguments['output_dir'] != second.resolved_arguments['output_dir']
    assert second.resolved_arguments['sequence_path'] == first.result['sequences_path']
    assert second.resolved_arguments['sequence_sha256'] == first.result['sequences_sha256']
    frozen = {p:p.read_bytes() for p in base.rglob('*') if p.is_file()}
    assert len(native)==1
    monkeypatch.setattr(design, 'native_optimize', lambda *a:pytest.fail('Native replay'))
    fresh = ResearchAgentApplication(root, planner=Planner())
    assert fresh.sessions.load('session').to_dict() == state.to_dict()
    assert fresh.resume(run.run_id).status.value == 'SUCCEEDED'
    assert all(p.read_bytes()==value for p,value in frozen.items())
    Path(first.result['sequences_path']).write_text('corrupt')
    with pytest.raises(ValueError):
        shared.verify_result(dict(first.resolved_arguments), dict(first.result))


def test_single_explicit_paths_and_step_identity_are_stable(tmp_path):
    first = PlanStep('a/../b', 'tool', {'output_dir':str(tmp_path)})
    second = PlanStep('b', 'tool', {'output_dir':str(tmp_path)}, depends_on=(first.step_id,))
    # Plan identifiers themselves are not interpreted as path segments.
    isolated = isolate_output_directories((first, second))
    assert len({s.arguments['output_dir'] for s in isolated})==2
    assert all(Path(s.arguments['output_dir']).is_relative_to(tmp_path) for s in isolated)
    assert isolated[1].depends_on == second.depends_on
    assert isolate_output_directories(isolated)==isolated
    assert isolate_output_directories((first,))==(first,)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('boundary', ['published', 'verified'])
def test_interrupted_compound_run_keeps_step_paths_and_recovers_once(source, native, monkeypatch, boundary):
    args, _, _, _ = source
    inputs = {k:v for k,v in args.items() if k != 'output_dir'}
    inputs.update(window_size=2114, top_n=1, condition='B_cell', iterations=1,
        edit_start=900, edit_end=1200, protected_regions='1000:1100')
    root = Path(args['output_dir']).parent/'interrupted'
    app = ResearchAgentApplication(root, planner=Planner())
    class ProcessExit(BaseException): pass
    with monkeypatch.context() as stopped:
        if boundary == 'published':
            from agent.orchestration import executor
            verify = executor.verify_step
            def stop(step, *args, **kwargs):
                if step.step_id == 'prepare': raise ProcessExit()
                return verify(step, *args, **kwargs)
            stopped.setattr(executor, 'verify_step', stop)
        else:
            update = app.run_store.update
            def stop(state, *, expected_revision):
                saved = update(state, expected_revision=expected_revision)
                if any(s.step_id=='prepare' and s.status.value=='SUCCEEDED' for s in saved.steps):
                    raise ProcessExit()
                return saved
            stopped.setattr(app.run_store, 'update', stop)
        with pytest.raises(ProcessExit):
            app.run(AgentRequest('interrupted', 'Prepare and optimize', inputs))
    before = app.run_store.load('interrupted:run')
    assert before.lifecycle_status.value=='RUNNING' and not native
    frozen = {p:p.read_bytes() for p in app._workspace.run_paths(before.run_id).scientific.rglob('*') if p.is_file()}
    from agent.tools.analysis import scatac_regulatory_sequences as preparation
    monkeypatch.setattr(preparation, 'execute', lambda *a, **kw:pytest.fail('Preparation replay'))
    fresh = ResearchAgentApplication(root, planner=Planner())
    resumed = fresh.resume(before.run_id)
    assert resumed.status.value=='SUCCEEDED', resumed
    after = fresh.run_store.load(before.run_id)
    assert after.plan==before.plan and after.plan_fingerprint==before.plan_fingerprint
    assert after.steps[0].resolved_arguments==before.steps[0].resolved_arguments
    assert all(p.read_bytes()==value for p,value in frozen.items())
    assert len(native)==1
