"""Finite deployment policy for the existing scientific snapshot owner."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import os


@dataclass(frozen=True)
class DeliveryPolicy:
    source_bytes: int = 1024**3
    delivered_bytes: int = 1024**3
    logical_bytes: int = 2 * 1024**3
    temporary_bytes: int = 8 * 1024**3
    chunk_bytes: int = 16 * 1024**2
    transfers: int = 2

    def __post_init__(self):
        if any(type(value) is not int or value < 1 for value in vars(self).values()):
            raise ValueError('Scientific delivery limits must be finite positive integers.')
        if self.chunk_bytes > self.logical_bytes or self.transfers > 32:
            raise ValueError('Invalid scientific delivery chunk/concurrency policy.')

    @classmethod
    def configured(cls, *, source_default=1024**3):
        defaults = cls(source_bytes=source_default, delivered_bytes=source_default)
        return cls(**{name: int(os.environ.get('AGENT_DELIVERY_' + name.upper(), value))
                      for name, value in vars(defaults).items()})


_POLICY = ContextVar('scientific_delivery_policy', default=None)


def current():
    return _POLICY.get() or DeliveryPolicy.configured()


@contextmanager
def scope(policy):
    from agent.tools.analysis.peak_h5ad import inspection_limits
    token = _POLICY.set(policy)
    try:
        with inspection_limits(policy.logical_bytes, policy.chunk_bytes):
            yield
    finally:
        _POLICY.reset(token)
