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


# OS-owned memory reservations shared by all Agent processes for this user.
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
    cpus = native_cpu_count()
    context = current()
    payload = dict(operation=operation, arguments=arguments, memory_bytes=memory,
        context=snapshot_worker_context(context), environment=worker_environment(),
        store_root=None if _STORE.get() is None else str(_STORE.get().root))
    root_value = arguments.get('values', arguments.get('arguments', {})).get('output_dir')
    root = Path(root_value) if root_value else None
    before = set(root.iterdir()) if root is not None and root.exists() else set()
    diagnostic = (root.parent / ('.' + root.name + '-execution.json') if root is not None else
        Path(arguments['log_path'] + '.execution.json') if operation == 'command' else None)
    queued = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='agent-native-', dir=runtime) as temp:
        with _native_reservation(runtime, memory, cpus, work_dir=temp) as reservation:
            unit = reservation['unit']
            payload['unit'] = unit
            payload['cpu_count'] = cpus
            payload['environment'] = bounded_environment(payload['environment'], cpus)
            if operation == 'command':
                payload['arguments'] = dict(arguments,
                    environment=bounded_environment(arguments['environment'], cpus))
            record = dict(operation=operation, memory_bytes=memory, swap_bytes=0, timeout_seconds=timeout,
                unit=unit, mechanism='systemd-user-service/cgroup-v2', started=False,
                reservation=reservation, admission_seconds=time.monotonic()-queued)
            work = Path(temp)
            request = work/'request.json'
            descriptor = os.open(request, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor,'w') as stream:
                json.dump(payload, stream, allow_nan=False)
            command = ['systemd-run','--user','--quiet', '--service-type=exec',
                '--unit='+unit, '--property=MemoryMax='+str(memory), '--property=MemorySwapMax=0',
                '--property=CPUQuota='+str(100*cpus)+'%', '--property=CPUAccounting=yes',
                '--property=MemoryAccounting=yes',
                '--property=OOMPolicy=kill', '--property=KillMode=control-group', '--property=Restart=no',
                '--property=RuntimeMaxSec='+str(timeout), '--property=TimeoutStopSec=5',
                '--property=RemainAfterExit=yes', sys.executable, '-B', '-I',
                str(Path(__file__).with_name('_native_worker.py')), str(request)]
            # Retain successful membership until the parent checks the response.
            owned = False
            started = time.monotonic()
            try:
                # A timed-out launcher may already have created our unit. The
                # unique reserved unit establishes cleanup ownership.
                owned = True
                launch = subprocess.run(command, capture_output=True, text=True, timeout=15)
                if launch.returncode:
                    record['launch_diagnostic'] = launch.stderr[-4096:]
                    raise NativeExecutionError('NATIVE_EXECUTION_UNAVAILABLE', 'The operating system could not establish the native memory boundary.')
                record['started'] = True
                from datetime import datetime, timezone
                record['started_at'] = datetime.now(timezone.utc).isoformat()
                while True:
                    cancellation_checkpoint()
                    info = subprocess.run(['systemctl','--user','show',unit,
                        '--property=ActiveState,SubState,Result,ExecMainStatus,ExecMainCode,ControlGroup,InvocationID,MemoryPeak,MemoryCurrent,CPUUsageNSec'],
                        capture_output=True, text=True, timeout=10)
                    state = dict(line.split('=',1) for line in info.stdout.splitlines() if '=' in line)
                    record['systemd'] = state
                    for key in ('MemoryCurrent', 'MemoryPeak'):
                        if state.get(key, '').isdigit() and int(state[key]) < 2**63:
                            record['observed_memory_peak_bytes'] = max(record.get('observed_memory_peak_bytes', 0), int(state[key]))
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
                record['worker_resource_usage'] = response.get('resource_usage')
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
                        raise NativeInvocationError(error['message'], code=error['code'])
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
                record['finished_at'] = datetime.now(timezone.utc).isoformat()
                return response['result']
            except subprocess.TimeoutExpired as exc:
                record['supervisor_error'] = dict(type=type(exc).__name__, message=str(exc)[:4096])
                raise NativeExecutionError('NATIVE_EXECUTION_UNAVAILABLE',
                    'The operating-system execution supervisor did not respond within its time bound.') from exc
            except BaseException as exc:
                record['supervisor_error'] = dict(type=type(exc).__name__, message=str(exc)[:4096])
                raise
            finally:
                record['supervised_seconds'] = time.monotonic()-started
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


def native_cpu_count():
    """CPU reservation; kernel quota applies where the CPU controller is delegated."""
    available = _resource_limits()[2]
    try:
        count = int(os.environ.get('AGENT_NATIVE_CPU_COUNT', min(4, available)))
        if not 1 <= count <= available:
            raise ValueError()
        return count
    except ValueError:
        raise NativeExecutionError('NATIVE_EXECUTION_UNAVAILABLE', 'Invalid native CPU reservation.') from None


def bounded_environment(environment, cpus):
    return dict(environment, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                NUMBA_NUM_THREADS=str(cpus), RAYON_NUM_THREADS=str(cpus), POLARS_MAX_THREADS=str(cpus))


def _resource_limits():
    """Current host and ancestor-cgroup capacity; limits are not measured RSS."""
    from pathlib import Path
    info = {k: int(v.split()[0])*1024 for k, v in
            (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())}
    total, available = info['MemTotal'], info['MemAvailable']
    group = Path('/sys/fs/cgroup') / Path('/proc/self/cgroup').read_text().strip().split('::', 1)[1].lstrip('/')
    cpu_capacity = len(os.sched_getaffinity(0))
    for path in (group, *group.parents):
        if not path.is_relative_to('/sys/fs/cgroup'):
            break
        if (path/'memory.max').is_file():
            limit = (path/'memory.max').read_text().strip()
            if limit != 'max':
                total = min(total, int(limit))
                available = min(available, max(0, int(limit)-int((path/'memory.current').read_text())))
        if (path/'cpu.max').is_file():
            quota, period = (path/'cpu.max').read_text().split()
            if quota != 'max':
                cpu_capacity = min(cpu_capacity, max(1, int(quota)//int(period)))
    headroom = max(2*1024**3, total//10)
    try:
        budget = min(int(os.environ.get('AGENT_NATIVE_TOTAL_MEMORY_BYTES', total-headroom)), total-headroom)
    except ValueError:
        raise NativeExecutionError('NATIVE_EXECUTION_UNAVAILABLE', 'Invalid aggregate native memory budget.') from None
    if budget <= 0:
        raise NativeExecutionError('NATIVE_EXECUTION_UNAVAILABLE', 'Insufficient host memory after native headroom.')
    return budget, max(0, available-headroom), cpu_capacity


def resource_capacity():
    """Bound branch workers by reserved memory and CPU, never a fixed branch count."""
    try:
        memory = int(os.environ['AGENT_NATIVE_MEMORY_BYTES'])
        if memory <= 0:
            raise ValueError()
    except (KeyError, ValueError):
        raise NativeExecutionError('NATIVE_EXECUTION_UNAVAILABLE', 'A positive native memory reservation is required.') from None
    budget, available, cpus = _resource_limits()
    return max(1, min(budget//memory, available//memory, cpus//native_cpu_count()))


from contextlib import contextmanager


@contextmanager
def _native_reservation(runtime, memory, cpus, *, work_dir):
    """Extend the existing OS slot with conservative cross-process reservations.

    Each live lease holds an OS lock until its cgroup is stopped. Sum of kernel
    MemoryMax limits never exceeds the common budget. A dead supervisor's lease
    is reclaimed only after its entire native cgroup is confirmed stopped.
    No workload queue, retry, or scientific lifecycle is introduced here.
    """
    import fcntl
    import json
    import uuid
    root = runtime/'agent-native-reservations'
    root.mkdir(mode=0o700, exist_ok=True)
    if root.is_symlink() or root.stat().st_uid != os.getuid():
        raise NativeExecutionError('NATIVE_EXECUTION_UNAVAILABLE', 'Unsafe native reservation directory.')
    lease = None
    unit = 'agent-native-task-' + uuid.uuid4().hex + '.service'
    path = root/(unit + '.json')
    try:
        while lease is None:
            cancellation_checkpoint()
            with (runtime/'agent-native-task.lock').open('a+b') as gate:
                try:
                    fcntl.flock(gate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    time.sleep(.1)
                    continue
                legacy = subprocess.run(['systemctl','--user','show','agent-native-task.service',
                    '--property=LoadState','--value'], capture_output=True, text=True, timeout=10)
                if legacy.returncode:
                    raise NativeExecutionError('NATIVE_EXECUTION_UNAVAILABLE', 'The user systemd execution manager is unavailable.')
                if legacy.stdout.strip() != 'not-found':
                    raise NativeExecutionError('NATIVE_EXECUTION_BUSY', 'Inspect the legacy native execution unit before further work.')
                budget, available, cpu_capacity = _resource_limits()
                if memory > budget or cpus > cpu_capacity:
                    raise NativeExecutionError('NATIVE_EXECUTION_UNAVAILABLE', 'Native reservation exceeds safe host capacity.')
                reserved_memory = reserved_cpu = unused_memory = 0
                for old in sorted(root.glob('*.json')):
                    try:
                        stream = old.open('r+b')
                    except FileNotFoundError:
                        continue
                    with stream:
                        try:
                            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            entry = json.load(stream)
                            budget = min(budget, entry['aggregate_memory_bytes'])
                            reserved_memory += entry['memory_bytes']; reserved_cpu += entry['cpu_count']
                            current = subprocess.run(['systemctl','--user','show',entry['unit'],
                                '--property=MemoryCurrent','--value'], capture_output=True, text=True, timeout=10)
                            used = int(current.stdout.strip()) if current.returncode == 0 and current.stdout.strip().isdigit() else 0
                            if used >= 2**63: used = 0
                            unused_memory += max(0, entry['memory_bytes']-used)
                        else:
                            # Lost supervisor: stop the whole subtree, never resume science.
                            # The filename survives even a crash during the JSON write.
                            import re
                            orphan = old.name.removesuffix('.json')
                            if not re.fullmatch(r'agent-native-task-[0-9a-f]{32}\.service', orphan):
                                raise NativeExecutionError('NATIVE_EXECUTION_UNAVAILABLE', 'Unknown native reservation identity.')
                            subprocess.run(['systemctl','--user','stop',orphan], capture_output=True, timeout=15)
                            check = subprocess.run(['systemctl','--user','show',orphan,
                                '--property=ActiveState','--value'], capture_output=True, text=True, timeout=10)
                            if check.returncode or check.stdout.strip() not in ('inactive','failed',''):
                                raise NativeExecutionError('NATIVE_EXECUTION_UNAVAILABLE', 'Native orphan termination is unconfirmed; reservation retained.')
                            subprocess.run(['systemctl','--user','reset-failed',orphan], capture_output=True, timeout=10)
                            try:
                                stream.seek(0)
                                abandoned = json.load(stream).get('work_dir')
                            except (ValueError, AttributeError):
                                abandoned = None  # Crash during initial lease write, before request creation.
                            if abandoned is not None:
                                from pathlib import Path
                                import shutil
                                work = Path(abandoned)
                                if (work.parent != runtime or not re.fullmatch(r'agent-native-[a-z0-9_]+', work.name)
                                        or work.is_symlink() or work.exists() and work.stat().st_uid != os.getuid()):
                                    raise NativeExecutionError('NATIVE_EXECUTION_UNAVAILABLE', 'Unsafe abandoned native bootstrap directory.')
                                if work.is_dir():
                                    request = work/'request.json'
                                    if request.is_file():
                                        try:
                                            job = json.loads(request.read_text())
                                        except ValueError:
                                            job = {}  # Interrupted request write, before launch.
                                        if job and job.get('unit') != orphan:
                                            raise NativeExecutionError('NATIVE_EXECUTION_UNAVAILABLE', 'Abandoned native bootstrap identity changed.')
                                    shutil.rmtree(work)
                            old.unlink(missing_ok=True)
                if reserved_memory+memory <= budget and reserved_cpu+cpus <= cpu_capacity and unused_memory+memory <= available:
                    lease = path.open('x+b')
                    os.chmod(path, 0o600)
                    fcntl.flock(lease, fcntl.LOCK_EX)
                    reservation = dict(unit=unit, memory_bytes=memory, cpu_count=cpus, work_dir=str(work_dir),
                        aggregate_memory_bytes=budget, reserved_memory_bytes=reserved_memory+memory,
                        reserved_cpu_count=reserved_cpu+cpus, host_headroom_policy='max(2 GiB, 10% effective memory)')
                    lease.write(json.dumps(reservation).encode()); lease.flush(); os.fsync(lease.fileno())
            if lease is None:
                time.sleep(.1)
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise NativeExecutionError('NATIVE_EXECUTION_UNAVAILABLE', 'The native resource manager did not respond.') from exc
    else:
        yield reservation
    finally:
        if lease is not None:
            # call() stops its unit before leaving. On unconfirmed cleanup leave
            # the unlocked record for the next admission's orphan-stop check.
            try:
                check = subprocess.run(['systemctl','--user','show',unit,'--property=ActiveState','--value'],
                                       capture_output=True, text=True, timeout=10)
                if check.returncode == 0 and check.stdout.strip() in ('inactive','failed',''):
                    path.unlink(missing_ok=True)
            finally:
                lease.close()
