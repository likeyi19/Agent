"""Repository-owned cooperative tool checkpoints; no scientific parameters."""
from contextlib import contextmanager
from contextvars import ContextVar

_check = ContextVar('agent_tool_cancellation', default=None)


class ToolWorkCancelled(Exception):
    pass


@contextmanager
def cancellation_scope(check):
    token = _check.set(check)
    try:
        yield
    finally:
        _check.reset(token)


def cancellation_checkpoint():
    callback = _check.get()
    if callback is not None and callback():
        raise ToolWorkCancelled()
