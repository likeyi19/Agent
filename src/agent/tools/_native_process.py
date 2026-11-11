"""Bounded subprocess waiting for repository-controlled native scientific calls."""
import os
import signal
import subprocess
import time
from ._cancellation import cancellation_checkpoint


def run(command, log_path, *, environment, timeout=7200):
    """No shell or user code; callers own fixed command construction and semantics."""
    return call('command', dict(command=list(map(str, command)), log_path=str(log_path),
        environment=environment, timeout=timeout),
        lambda: _run(command, log_path, environment=environment, timeout=timeout))


def _run(command, log_path, *, environment, timeout):
    start = time.monotonic()
    with log_path.open('wb') as log:
        process = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
            stdout=log, stderr=log, start_new_session=True)
        try:
            while True:
                cancellation_checkpoint()
                if time.monotonic() - start > timeout:
                    raise TimeoutError('Native scientific call exceeded its execution time bound')
                try:
                    return process.wait(timeout=.2)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait()


def r_environment():
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
        R_ENVIRON_USER='/dev/null', R_PROFILE_USER='/dev/null')
    for key in ('R_LIBS_USER', 'R_LIBS', 'R_LIBS_SITE'):
        environment.pop(key, None)
    return environment


# One OS-owned task at a time per service user, including across Agent processes.
# The existing Executor remains responsible for lifecycle, publication and retry.
from contextvars import ContextVar
_CONTROLLED = ContextVar('kernel_controlled_native_worker', default=False)


class NativeExecutionError(RuntimeError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


class NativeWorkerError(Exception):
    def __init__(self, error):
        self.error = error
        super().__init__(error['message'])


def worker_environment():
    """Carry the operation's configured library across the process boundary."""
    from agent.application.shared_resources import current_scientific_library
    environment = dict(os.environ)
    library = current_scientific_library()
    if library is not None:
        environment['AGENT_SCIENTIFIC_RESOURCE_ROOT'] = str(library.root)
    return environment


def call(operation, arguments, local):
    """Run an existing owner in the shared OS boundary; no scientific policy."""
    if _CONTROLLED.get():
        return local()
    import json
    import math
    from pathlib import Path
    import shutil
    import sys
    import tempfile
    from agent.tools.data.authority_context import current, snapshot_worker_context, restore_worker_context
    from agent.orchestration.prior_outputs import merge_context, _STORE
    if sys.platform != 'linux':
        raise NativeExecutionError('NATIVE_EXECUTION_UNAVAILABLE',
            'This execution boundary requires Linux systemd and cgroup v2 memory control.')
    import fcntl
    try:
        memory = int(os.environ['AGENT_NATIVE_MEMORY_BYTES'])
        timeout = float(os.environ.get('AGENT_NATIVE_TIMEOUT_SECONDS', '7200'))
        if memory <= 0 or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError()
    except (KeyError, ValueError):
        raise NativeExecutionError('NATIVE_EXECUTION_UNAVAILABLE',
            'The operator must configure a positive native task memory budget and finite timeout.') from None
    runtime = Path('/run/user') / str(os.getuid())
    if not runtime.is_dir() or runtime.stat().st_uid != os.getuid() or not shutil.which('systemd-run'):
        raise NativeExecutionError('NATIVE_EXECUTION_UNAVAILABLE',
            'Native execution requires a user systemd manager with cgroup v2 memory control.')
    unit = 'agent-native-task.service'
    context = current()
    payload = dict(operation=operation, arguments=arguments, memory_bytes=memory,
        context=snapshot_worker_context(context), environment=worker_environment(),
        store_root=None if _STORE.get() is None else str(_STORE.get().root))
    root_value = arguments.get('values', arguments.get('arguments', {})).get('output_dir')
    root = Path(root_value) if root_value else None
    before = set(root.iterdir()) if root is not None and root.exists() else set()
    diagnostic = (root.parent / ('.' + root.name + '-execution.json') if root is not None else
        Path(arguments['log_path'] + '.execution.json') if operation == 'command' else None)
    record = dict(operation=operation, memory_bytes=memory, swap_bytes=0, timeout_seconds=timeout,
                  unit=unit, mechanism='systemd-user-service/cgroup-v2', started=False)
    with (runtime/'agent-native-task.lock').open('a+b') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise NativeExecutionError('NATIVE_EXECUTION_BUSY', 'A native scientific task already occupies the configured execution slot.') from None
        try:
            status = subprocess.run(['systemctl','--user','show',unit,'--property=LoadState','--value'],
                capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise NativeExecutionError('NATIVE_EXECUTION_UNAVAILABLE',
                'The operating-system execution manager could not be queried.') from exc
        if status.returncode:
            raise NativeExecutionError('NATIVE_EXECUTION_UNAVAILABLE',
                'The user systemd execution manager is unavailable.')
        if status.stdout.strip() != 'not-found':
            raise NativeExecutionError('NATIVE_EXECUTION_BUSY',
                'The native execution unit is still present; the operator must inspect the previous task before another execution.')
        with tempfile.TemporaryDirectory(prefix='agent-native-', dir=runtime) as temp:
            work = Path(temp)
            request = work/'request.json'
            descriptor = os.open(request, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor,'w') as stream:
                json.dump(payload, stream, allow_nan=False)
            command = ['systemd-run','--user','--quiet', '--service-type=exec',
                '--unit='+unit, '--property=MemoryMax='+str(memory), '--property=MemorySwapMax=0',
                '--property=OOMPolicy=kill', '--property=KillMode=control-group', '--property=Restart=no',
                '--property=RuntimeMaxSec='+str(timeout), '--property=TimeoutStopSec=5',
                '--property=RemainAfterExit=yes', sys.executable, '-B', '-I',
                str(Path(__file__).with_name('_native_worker.py')), str(request)]
            # Retain successful membership until the parent checks the response.
            owned = False
            started = time.monotonic()
            try:
                # A timed-out launcher may already have created our unit. The
                # lock and prior absence check establish cleanup ownership.
                owned = True
                launch = subprocess.run(command, capture_output=True, text=True, timeout=15)
                if launch.returncode:
                    record['launch_diagnostic'] = launch.stderr[-4096:]
                    raise NativeExecutionError('NATIVE_EXECUTION_UNAVAILABLE', 'The operating system could not establish the native memory boundary.')
                record['started'] = True
                while True:
                    cancellation_checkpoint()
                    info = subprocess.run(['systemctl','--user','show',unit,
                        '--property=ActiveState,SubState,Result,ExecMainStatus,ExecMainCode,ControlGroup,InvocationID'],
                        capture_output=True, text=True, timeout=10)
                    state = dict(line.split('=',1) for line in info.stdout.splitlines() if '=' in line)
                    record['systemd'] = state
                    if info.returncode or state.get('ActiveState') in ('failed','inactive'):
                        reason = state.get('Result')
                        code = 'NATIVE_MEMORY_EXHAUSTED' if reason == 'oom-kill' else 'NATIVE_EXECUTION_TIMEOUT' if reason == 'timeout' else 'NATIVE_EXECUTION_FAILED'
                        raise NativeExecutionError(code, 'The isolated native task did not complete; see the operator execution diagnostic.')
                    if state.get('SubState') == 'exited':
                        break
                    if time.monotonic()-started > timeout+10:
                        raise NativeExecutionError('NATIVE_EXECUTION_TIMEOUT', 'The native task exceeded its configured execution time.')
                    time.sleep(.1)
                response_path = work/'response.json'
                if not response_path.is_file() or response_path.stat().st_size > 64*1024**2:
                    raise NativeExecutionError('NATIVE_EXECUTION_FAILED', 'The native task returned no bounded execution response.')
                response = json.loads(response_path.read_text())
                if 'error' in response:
                    record['worker_error'] = response['error']
                    error = response['error']
                    if error['type'] == 'MemoryError':
                        raise NativeExecutionError('NATIVE_MEMORY_EXHAUSTED', 'The native worker reported allocation failure within its configured memory boundary.')
                    if error['type'] == 'TimeoutError':
                        raise NativeExecutionError('NATIVE_EXECUTION_TIMEOUT', 'The native subprocess exceeded its configured execution time.')
                    if error['type'] == 'NativeExecutionError':
                        raise NativeExecutionError(error['code'], error['message'])
                    if error['type'] == 'CalledProcessError':
                        raise NativeExecutionError('NATIVE_EXECUTION_FAILED', 'The native subprocess failed; its diagnostic is retained.')
                    if error['type'] == 'AuthorityError':
                        from agent.schemas.verification_authority import AuthorityError
                        raise AuthorityError(error['message'])
                    if error['type'] == 'NativeInvocationError':
                        from agent.tools.analysis.native_matrix import NativeInvocationError
                        raise NativeInvocationError(error['message'])
                    safe = {'ValueError': ValueError, 'FileNotFoundError': FileNotFoundError,
                            'FileExistsError': FileExistsError, 'ImportError': ImportError,
                            'ModuleNotFoundError': ModuleNotFoundError, 'RuntimeError': RuntimeError}
                    if error['type'] in safe:
                        raise safe[error['type']](error['message'])
                    raise NativeWorkerError(error)
                if context is not None:
                    context.validate_anchors()
                    merge_context(context, restore_worker_context(response['context']))
                record['completed'] = True
                return response['result']
            except subprocess.TimeoutExpired as exc:
                record['supervisor_error'] = dict(type=type(exc).__name__, message=str(exc)[:4096])
                raise NativeExecutionError('NATIVE_EXECUTION_UNAVAILABLE',
                    'The operating-system execution supervisor did not respond within its time bound.') from exc
            except BaseException as exc:
                record['supervisor_error'] = dict(type=type(exc).__name__, message=str(exc)[:4096])
                raise
            finally:
                cleanup_error = None
                if owned:
                    try:
                        membership = work/'membership.json'
                        if membership.is_file():
                            record['membership'] = json.loads(membership.read_text())
                        journal = subprocess.run(['journalctl','--user',
                            '_SYSTEMD_INVOCATION_ID='+record.get('systemd',{}).get('InvocationID','missing'),
                            '--no-pager','-n','80','--output=cat'],
                            capture_output=True, text=True, timeout=10)
                        record['log_tail'] = journal.stdout[-16384:]
                    except Exception as exc:
                        record['diagnostic_error'] = str(exc)[:4096]
                    # Diagnostics must never prevent owned subtree termination.
                    try:
                        stopped = subprocess.run(['systemctl','--user','stop',unit], capture_output=True, timeout=15)
                        if stopped.returncode:
                            absent = subprocess.run(['systemctl','--user','show',unit,'--property=LoadState','--value'],
                                capture_output=True, text=True, timeout=10)
                            if absent.returncode or absent.stdout.strip() != 'not-found':
                                raise RuntimeError('systemd could not confirm native task termination')
                    except Exception as exc:
                        cleanup_error = exc
                        record['cleanup_error'] = str(exc)[:4096]
                    try:
                        subprocess.run(['systemctl','--user','reset-failed',unit], capture_output=True, timeout=10)
                    except Exception as exc:
                        record['reset_error'] = str(exc)[:4096]
                    # Remove only unpublished task stages created by this call.
                    if cleanup_error is None and root is not None and root.is_dir():
                        for path in set(root.iterdir())-before:
                            if path.name.startswith(('.matrix-attempt-', '.native-matrix-', '.native-snapatac2-', '.chromvar-', '.chromvar-function-', '.cicero-cds-')) and path.is_dir() and not path.is_symlink():
                                shutil.rmtree(path)
                if diagnostic is not None:
                    diagnostic.parent.mkdir(parents=True, exist_ok=True)
                    diagnostic.write_text(json.dumps(record, indent=2)+'\n')
                if cleanup_error is not None:
                    raise NativeExecutionError('NATIVE_EXECUTION_UNAVAILABLE',
                        'The operator must inspect an unconfirmed native task cleanup before further execution.') from cleanup_error
