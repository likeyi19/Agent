from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest

from agent.orchestration import (AgentPlan, AgentRequest, AgentRuntime, FileRunStore,
    PlanStep, RunMode, RunStatus, RunLifecycleStatus, StepStatus, ToolRegistry, build_default_tool_registry)
from agent.tools.data import raw_scatac as raw


class FixedPlanner:
    def __init__(self, plan):
        self.value = plan
        self.calls = 0

    def plan(self, request, registry):
        self.calls += 1
        return self.value


def setup(args, *, twice=False):
    default = build_default_tool_registry()
    call = Mock(wraps=default.get('inspect_raw_scATAC').function)
    registry = ToolRegistry(tuple(replace(default.get(n), function=call)
        if n == 'inspect_raw_scATAC' else default.get(n) for n in default.names()))
    steps = (PlanStep('first', 'inspect_raw_scATAC', args),)
    if twice:
        steps += (PlanStep('second', 'inspect_raw_scATAC', args, ('first',)),)
    planner = FixedPlanner(AgentPlan('raw-plan', 'raw-request', 'fixed', steps))
    return registry, planner, call


@pytest.mark.parametrize('kind', ['fastq', 'bam'])
@pytest.mark.parametrize('malformed', [False, True])
def test_runtime_success_persistence_and_terminal_resume(raw_factory, tmp_path, kind, malformed):
    args = raw_factory(kind, malformed=malformed)
    registry, planner, call = setup(args)
    store = FileRunStore(tmp_path / 'store')
    runtime = AgentRuntime(registry=registry, planner=planner, run_store=store)
    result = runtime.run(AgentRequest('raw-request', 'inspect', {}))
    assert result.status is RunStatus.SUCCEEDED, result.errors
    assert result.steps[0].verification.passed
    assert result.steps[0].result['readiness'] == ('INVALID' if malformed else 'READY')
    assert store.load(result.run_id).to_run_result() == result
    # Terminal resume keeps the existing immutable-return contract, even after drift.
    path = next(Path(args['raw_input_paths']).iterdir())
    path.write_bytes(b'changed after terminal success')
    assert runtime.resume(result.run_id) == result
    assert call.call_count == planner.calls == 1
    assert len(list(Path(args['output_dir']).iterdir())) == 1


class ProcessExit(BaseException):
    pass


class InterruptStore:
    def __init__(self, delegate):
        self.delegate = delegate

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def update(self, state, *, expected_revision):
        saved = self.delegate.update(state, expected_revision=expected_revision)
        if (saved.lifecycle_status is RunLifecycleStatus.RUNNING and len(saved.steps) == 2
                and saved.steps[0].status is StepStatus.SUCCEEDED
                and saved.steps[1].status is StepStatus.PENDING):
            raise ProcessExit()
        return saved


@pytest.mark.parametrize('kind', ['fastq', 'bam'])
@pytest.mark.parametrize('drift', [False, True])
def test_nonterminal_resume_revalidates_without_rerunning_completed_tool(raw_factory, tmp_path, kind, drift):
    args = raw_factory(kind)
    registry, planner, call = setup(args, twice=True)
    store = FileRunStore(tmp_path / 'store')
    with pytest.raises(ProcessExit):
        AgentRuntime(registry=registry, planner=planner, run_store=InterruptStore(store)).run(
            AgentRequest('raw-request', 'inspect', {}))
    assert call.call_count == 1
    if drift:
        next(Path(args['raw_input_paths']).iterdir()).write_bytes(b'changed')
    forbidden_planner = Mock()
    forbidden_planner.plan.side_effect = AssertionError('planner called on resume')
    result = AgentRuntime(registry=registry, planner=forbidden_planner, run_store=store).resume('raw-request:run')
    assert result.status is (RunStatus.FAILED if drift else RunStatus.SUCCEEDED), result.errors
    assert call.call_count == (1 if drift else 2)  # Second pending step only.
    assert planner.calls == 1 and forbidden_planner.plan.call_count == 0
    assert len(list(Path(args['output_dir']).iterdir())) == 1


def test_runtime_detects_source_changed_after_tool_before_verification(raw_factory):
    args = raw_factory()
    registry, planner, call = setup(args)
    def mutate(**kwargs):
        result = raw.inspect_raw_scATAC(**kwargs)
        next(Path(args['raw_input_paths']).iterdir()).write_bytes(b'changed')
        return result
    call.side_effect = mutate
    result = AgentRuntime(registry=registry, planner=planner).run(AgentRequest('raw-request', 'inspect', {}))
    assert result.status is RunStatus.FAILED
    assert not result.steps[0].verification.passed
    assert call.call_count == 1


@pytest.mark.parametrize('kind', ['fastq', 'bam'])
def test_no_pysam_plan_only_runtime_and_application_subprocess(tmp_path, kind):
    code = r'''
import importlib.abc, json, sys
from pathlib import Path
class NoPysam(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pysam' or fullname.startswith('pysam.'):
            raise ModuleNotFoundError('pysam intentionally unavailable')
sys.meta_path.insert(0, NoPysam())
from agent.orchestration import AgentRuntime, AgentRequest, LLMPlanner, RunMode, RunStatus, build_default_tool_registry
from agent.application import ResearchAgentApplication
from agent.tools.data import raw_scatac as raw
from agent.tools.data import _raw_fastq, _raw_bam
def forbidden(*args, **kwargs):
    raise AssertionError('raw scientific execution in PLAN_ONLY')
raw.inspect_raw_scATAC = raw._reconstruct = raw._select = forbidden
_raw_fastq.inspect_fastq_inputs = _raw_bam.inspect_bam_inputs = forbidden
import agent.orchestration.registry as registry_module
registry_module.inspect_raw_scATAC = forbidden
# Preserve signature metadata while replacing only execution callable.
from dataclasses import replace
# build_default_tool_registry checks public signatures, so use a signature-preserving guard.
import inspect
from agent.tools import inspect_raw_scATAC as original
forbidden.__signature__ = inspect.signature(original)
registry = build_default_tool_registry()
class Model:
    model_id = 'offline'
    def complete(self, **kwargs):
        return json.dumps({'schema_version':4,'decision':{'kind':'plan','steps':[
            {'step_id':'raw','tool':'inspect_raw_scATAC','sources':[],'control_dependencies':[]}]}})
base = Path(sys.argv[1])
inputs = {'raw_input_paths': '/DOES_NOT_EXIST/input.' + sys.argv[2], 'species':'mouse', 'raw_assay':'TENX_ATAC'}
runtime_request = AgentRequest('runtime', 'inspect raw inputs', {**inputs,'output_dir':str(base/'out')}, RunMode.PLAN_ONLY)
result = AgentRuntime(planner=LLMPlanner(Model()), registry=registry).run(runtime_request)
assert result.status is RunStatus.PLANNED and not result.steps, result.errors
app = ResearchAgentApplication(base/'workspace', planner=LLMPlanner(Model()), registry=registry)
application = app.run(AgentRequest('application', 'inspect raw inputs', inputs, RunMode.PLAN_ONLY))
assert application.status.value == 'PLANNED', application
compiled = application.run_result.plan.steps[0].arguments
assert compiled['raw_input_paths'] == inputs['raw_input_paths']
assert Path(compiled['output_dir']).is_relative_to(base/'workspace')
assert Path(compiled['output_dir']).name == 'scientific'
assert not application.run_result.steps
assert 'pysam' not in sys.modules
assert not list(base.rglob('raw-scatac-intake-*.json'))
assert not (base/'out').exists()
print('PLAN_ONLY without pysam: PASS')
'''
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path), kind], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_fastq_execute_and_verify_without_pysam_subprocess(raw_factory):
    args = raw_factory()
    code = r'''
import importlib.abc, json, sys
class NoPysam(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pysam' or fullname.startswith('pysam.'):
            raise ModuleNotFoundError('blocked optional dependency')
sys.meta_path.insert(0, NoPysam())
from agent.tools import inspect_raw_scATAC
from agent.orchestration.raw_scatac_verifier import verify_raw_scatac_intake
from agent.orchestration import build_default_tool_registry
args = json.loads(sys.argv[1])
result = inspect_raw_scATAC(**args)
verify_raw_scatac_intake(args, result, build_default_tool_registry())
assert result['status'] == 'success' and 'pysam' not in sys.modules
from pathlib import Path
bam = Path(args['raw_input_paths']) / 'missing-backend.bam'
bam.write_bytes(b'content is not opened without backend')
try:
    inspect_raw_scATAC(str(bam), args['output_dir'], raw_assay='SCATAC')
except ValueError as exc:
    assert exc.code == 'RAW_BAM_DEPENDENCY_UNAVAILABLE'
else:
    raise AssertionError('missing BAM backend was ignored')
'''
    result = subprocess.run([sys.executable, '-c', code, json.dumps(args)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
