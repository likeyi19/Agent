"""Bounded subprocess waiting for repository-controlled native scientific calls."""
import os
import signal
import subprocess
import time
from ._cancellation import cancellation_checkpoint


def run(command, log_path, *, environment, timeout=7200):
    """No shell or user code; callers own fixed command construction and semantics."""
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
