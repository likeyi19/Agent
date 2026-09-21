"""Frozen bounded planning visibility; private identities never enter provider prompts."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from agent.schemas.prior_output import PriorOutputBinding

_CONTEXT = ContextVar('active_planning_context', default=None)
MAX_CONTEXT_ITEMS = 32


@dataclass(frozen=True)
class ActiveContextItem:
    handle: str
    binding: PriorOutputBinding


@dataclass(frozen=True)
class ActivePlanningContext:
    session_id: str
    base_revision_id: str
    base_generation: int
    items: tuple[ActiveContextItem, ...]

    def __post_init__(self):
        if (any(type(v) is not str or not v for v in (self.session_id, self.base_revision_id))
                or type(self.base_generation) is not int or self.base_generation < 0):
            raise ValueError('Invalid captured context identity.')
        if (type(self.items) is not tuple or len(self.items) > MAX_CONTEXT_ITEMS
                or any(not isinstance(i, ActiveContextItem) or not isinstance(i.binding, PriorOutputBinding)
                       or i.handle != f'ctx.{n}' for n, i in enumerate(self.items))):
            raise ValueError('Invalid bounded active planning context.')

    def binding(self, handle):
        for item in self.items:
            if item.handle == handle:
                return item.binding
        raise ValueError('Unknown or unoffered active context handle.')

    def public_items(self):
        return tuple({'handle': i.handle, 'semantic_type': i.binding.semantic_type,
                      'producer_tool': i.binding.tool_name, 'source_port': i.binding.source_port,
                      'verification_mode': i.binding.verification_mode} for i in self.items)


def current_context():
    return _CONTEXT.get()


@contextmanager
def planning_context(context):
    if not isinstance(context, ActivePlanningContext):
        raise TypeError('Expected a frozen active context.')
    token = _CONTEXT.set(context)
    try:
        yield
    finally:
        _CONTEXT.reset(token)
