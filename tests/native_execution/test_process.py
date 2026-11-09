"""Actual kernel isolation, not mocked allocation failures. Opt-in Linux host."""
import json
import os
from pathlib import Path
import sys
import threading
import time
import pytest
from agent.tools import _native_process as owner
from agent.tools._cancellation import cancellation_scope, ToolWorkCancelled

pytestmark = pytest.mark.skipif(os.environ.get('AGENT_TEST_SYSTEMD') != '1',
    reason='Requires explicitly enabled user systemd/cgroup-v2 host qualification')

@pytest.fixture(autouse=True)
def budget(monkeypatch):
    monkeypatch.setenv('AGENT_NATIVE_MEMORY_BYTES', str(512*1024**2))
    monkeypatch.setenv('AGENT_NATIVE_TIMEOUT_SECONDS', '45')


def execute(code, tmp_path):
    return owner.run([sys.executable,'-c',code], tmp_path/'native.log', environment=dict(os.environ))


def diagnostic(tmp_path):
    return json.loads((tmp_path/'native.log.execution.json').read_text())


def test_python_r_membership_and_normal_completion(tmp_path):
    r=Path(os.environ['AGENT_ANNOTATION_RSCRIPT'])
    assert r.is_file()
    output=tmp_path/'r-group'
    code=f'''import os,subprocess
from pathlib import Path
p=Path('/proc/self/cgroup').read_text()
r=subprocess.check_output([{str(r)!r},'--vanilla','-e','cat(readLines("/proc/self/cgroup"), sep="\\n")'],text=True)
assert p.strip()==r.strip(),(p,r)
Path({str(output)!r}).write_text(r)
print('Python/R exact membership',p,flush=True)
'''
    assert execute(code,tmp_path)==0
    d=diagnostic(tmp_path)
    assert d['completed'] and d['membership']['effective']==dict(
        **{'memory.max':str(512*1024**2),'memory.swap.max':'0','memory.oom.group':'1'})
    assert output.exists()


def test_kernel_oom_kills_subtree_parent_survives(tmp_path):
    ready=tmp_path/'ready'
    code=f'''from pathlib import Path
import os,subprocess,sys,time
Path({str(ready)!r}).write_text(str(os.getpid()))
# Two descendants exceed the aggregate cap together, even though each fits.
p=subprocess.Popen([sys.executable,'-c','import time;x=bytearray(300*1024*1024);time.sleep(30)'])
x=bytearray(300*1024*1024)
p.wait()
'''
    with pytest.raises(owner.NativeExecutionError) as exc:execute(code,tmp_path)
    assert exc.value.code=='NATIVE_MEMORY_EXHAUSTED'
    assert ready.exists()  # Workload began after verified boundary, not import OOM.
    assert diagnostic(tmp_path)['systemd']['Result']=='oom-kill'
    with pytest.raises(ProcessLookupError):os.kill(int(ready.read_text()),0)
    assert execute('print("parent still serves new work")',tmp_path)==0


@pytest.mark.parametrize('kind',['cancel','timeout'])
def test_termination_cleans_owned_children(tmp_path,monkeypatch,kind):
    ready=tmp_path/'ready';child=tmp_path/'child'
    code=f'''import subprocess,sys,time,os
from pathlib import Path
p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'],start_new_session=True)
Path({str(child)!r}).write_text(str(p.pid))
Path({str(ready)!r}).write_text(str(os.getpid()))
time.sleep(120)
'''
    if kind=='timeout':monkeypatch.setenv('AGENT_NATIVE_TIMEOUT_SECONDS','12')
    with cancellation_scope(lambda:kind=='cancel' and ready.exists()):
        with pytest.raises(ToolWorkCancelled if kind=='cancel' else owner.NativeExecutionError) as exc:
            execute(code,tmp_path)
    if kind=='timeout':assert exc.value.code=='NATIVE_EXECUTION_TIMEOUT'
    assert ready.exists()
    for path in (ready,child):
        pid=int(path.read_text())
        # Reparented zombies have no memory/work; systemd cgroup must be gone.
        status=Path(f'/proc/{pid}/stat')
        assert not status.exists() or status.read_text().split()[2]=='Z'
    assert not Path(diagnostic(tmp_path)['membership']['cgroup']).exists()


def test_missing_budget_starts_no_process(tmp_path,monkeypatch):
    monkeypatch.delenv('AGENT_NATIVE_MEMORY_BYTES')
    with pytest.raises(owner.NativeExecutionError) as exc:execute('raise AssertionError()',tmp_path)
    assert exc.value.code=='NATIVE_EXECUTION_UNAVAILABLE'
    assert not (tmp_path/'native.log').exists()


def test_pca_publication_preserves_authority_in_kernel_boundary(tmp_path, monkeypatch):
    import anndata as ad
    import numpy as np
    import pandas as pd
    from scipy import sparse
    from agent.orchestration import AgentPlan, AgentRequest, AgentRuntime, FileRunStore, PlanStep
    monkeypatch.setenv('AGENT_NATIVE_MEMORY_BYTES', str(2*1024**3))
    monkeypatch.setenv('AGENT_NATIVE_TIMEOUT_SECONDS', '120')
    source = tmp_path/'counts.h5ad'
    ad.AnnData(sparse.csr_matrix(np.random.default_rng(9).poisson(2, (12, 31))),
        obs=pd.DataFrame(index=[f'cell-{i}' for i in range(12)])).write_h5ad(source)
    class Planner:
        def plan(self, request, registry):
            return AgentPlan('isolated-pca', request.request_id, 'Explicit native isolation test', (
                PlanStep('pca', 'pca_embed_cells', dict(input_path=str(source),
                    output_dir=str(tmp_path/'pca'), preprocessing='tfidf_l2', n_components=3)),))
    run = AgentRuntime(planner=Planner(), run_store=FileRunStore(tmp_path/'state')).run(
        AgentRequest('isolated-pca', 'Kernel qualification', {}))
    assert run.status.value == 'SUCCEEDED', run.errors
    assert run.steps[0].verification.artifact_authority is not None
    record = json.loads((tmp_path/'.pca-execution.json').read_text())
    assert record['completed'] and record['operation'] == 'pca'
    assert record['membership']['effective']['memory.max'] == str(2*1024**3)
    data = ad.read_h5ad(run.steps[0].result['analysis_path'])
    assert data.obsm['X_pca'].shape == (12, 3)


@pytest.mark.parametrize('failure', ['journal', 'launcher'])
def test_supervisor_fault_still_stops_owned_unit(tmp_path, monkeypatch, failure):
    import subprocess
    original = subprocess.run
    def injected(command, **kwargs):
        if failure == 'journal' and command[0] == 'journalctl':
            raise subprocess.TimeoutExpired(command, 10)
        result = original(command, **kwargs)
        if failure == 'launcher' and command[0] == 'systemd-run':
            assert result.returncode == 0
            raise subprocess.TimeoutExpired(command, 15)
        return result
    monkeypatch.setattr(subprocess, 'run', injected)
    if failure == 'launcher':
        with pytest.raises(owner.NativeExecutionError) as exc:
            execute('import time;time.sleep(120)', tmp_path)
        assert exc.value.code == 'NATIVE_EXECUTION_UNAVAILABLE'
    else:
        assert execute('print("completed")', tmp_path) == 0
        assert 'diagnostic_error' in diagnostic(tmp_path)
    state = original(['systemctl', '--user', 'show', 'agent-native-task.service',
                      '--property=LoadState', '--value'], capture_output=True, text=True)
    assert state.returncode == 0 and state.stdout.strip() == 'not-found'
    assert not list((Path('/run/user') / str(os.getuid())).glob('agent-native-*/request.json'))


def test_concurrency_is_shared_across_callers(tmp_path):
    ready=tmp_path/'ready'
    failures=[]
    def first():
        try:execute(f'from pathlib import Path;import time;Path({str(ready)!r}).touch();time.sleep(2)',tmp_path)
        except BaseException as exc:failures.append(exc)
    thread=threading.Thread(target=first);thread.start()
    try:
        deadline=time.monotonic()+30
        while not ready.exists() and thread.is_alive() and time.monotonic()<deadline:time.sleep(.1)
        assert ready.exists(),failures
        with pytest.raises(owner.NativeExecutionError) as exc:execute('raise AssertionError()',tmp_path)
        assert exc.value.code=='NATIVE_EXECUTION_BUSY'
    finally:thread.join(30)
    assert not failures and not thread.is_alive()


def test_kernel_failure_is_terminal_in_existing_session_and_runstore(tmp_path):
    from agent.application import ResearchAgentApplication, OutputSelection
    from agent.orchestration import AgentPlan, AgentRequest, PlanStep, ToolRegistry, ToolSpec, ResultContract
    from agent.orchestration.registry import _classify_tool_exception
    calls=[]
    def artificial():
        calls.append(1)
        execute('x=bytearray(768*1024*1024)',tmp_path)
        return {'value':'must not publish'}
    spec=ToolSpec(name='controlled_artificial',function=artificial,required_arguments={},optional_arguments={},
        result_contract=ResultContract('ArtificialResult',{'value':(str,)}),exception_classifier=_classify_tool_exception)
    registry=ToolRegistry((spec,))
    class Planner:
        def plan(self,request,registry):
            return AgentPlan('test',request.request_id,'Explicit artificial memory stress',
                (PlanStep('stress','controlled_artificial',{}),))
    app=ResearchAgentApplication(tmp_path/'workspace',planner=Planner(),registry=registry)
    app.sessions.create('failure')
    state=app.sessions.run('failure','stress',AgentRequest('controlled-failure','Artificial kernel failure',{}),
        (OutputSelection('output','stress','value'),),expected_generation=0)
    assert state.generation==0 and not state.revisions
    run_id=state.turns[-1].run_id
    run=app.run_store.load(run_id)
    assert run.lifecycle_status.value=='FAILED' and run.steps[0].attempt_count==1
    assert run.steps[0].error.code=='NATIVE_MEMORY_EXHAUSTED'
    assert run.steps[0].result is None and run.steps[0].verification is None
    before=app.run_store.state_path(run_id).read_bytes()
    from agent.orchestration import AgentRuntime
    reloaded=AgentRuntime(run_store=app.run_store,registry=registry)
    assert reloaded.resume(run_id).status.value=='FAILED'
    assert reloaded.resume(run_id).status.value=='FAILED'
    assert calls==[1] and app.run_store.state_path(run_id).read_bytes()==before


@pytest.mark.parametrize('failure', ['unavailable', 'timeout', 'stale'])
def test_manager_preflight_fails_without_launch_or_replacement(tmp_path, monkeypatch, failure):
    import subprocess
    calls = []
    def manager(command, **kwargs):
        calls.append(command)
        assert command == ['systemctl', '--user', 'show', 'agent-native-task.service',
                           '--property=LoadState', '--value']
        if failure == 'timeout':
            raise subprocess.TimeoutExpired(command, 10)
        return subprocess.CompletedProcess(command, 1 if failure == 'unavailable' else 0,
                                           stdout='loaded\n' if failure == 'stale' else '', stderr='')
    monkeypatch.setattr(subprocess, 'run', manager)
    with pytest.raises(owner.NativeExecutionError) as exc:
        execute('raise AssertionError("must not execute")', tmp_path)
    assert exc.value.code == ('NATIVE_EXECUTION_BUSY' if failure == 'stale' else 'NATIVE_EXECUTION_UNAVAILABLE')
    assert len(calls) == 1 and not (tmp_path/'native.log').exists()


def test_private_environment_transfer_permissions_and_cleanup(tmp_path, monkeypatch):
    import stat
    secret = 'artificial-private-token-for-closeout'
    monkeypatch.setenv('AGENT_TEST_PRIVATE_TOKEN', secret)
    ready = tmp_path/'ready'
    release = tmp_path/'release'
    failures = []
    code = f'''import os,time
from pathlib import Path
assert os.environ.get('AGENT_TEST_PRIVATE_TOKEN')
Path({str(ready)!r}).touch()
while not Path({str(release)!r}).exists():time.sleep(.05)
'''
    def run():
        try:
            assert execute(code, tmp_path) == 0
        except BaseException as exc:
            failures.append(exc)
    thread = threading.Thread(target=run)
    thread.start()
    request = None
    try:
        deadline = time.monotonic()+30
        while not ready.exists() and thread.is_alive() and time.monotonic()<deadline:
            time.sleep(.05)
        assert ready.exists(), failures
        requests = list((Path('/run/user')/str(os.getuid())).glob('agent-native-*/request.json'))
        assert len(requests) == 1
        request = requests[0]
        assert stat.S_IMODE(request.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(request.stat().st_mode) == 0o600
        assert request.stat().st_uid == os.getuid()
        membership = json.loads((request.parent/'membership.json').read_text())
        argv = Path(f"/proc/{membership['pid']}/cmdline").read_bytes()
        assert secret.encode() not in argv
    finally:
        release.touch()
        thread.join(30)
    assert not thread.is_alive() and not failures
    assert request is not None and not request.parent.exists()
    assert secret not in (tmp_path/'native.log').read_text()
    assert secret not in (tmp_path/'native.log.execution.json').read_text()
