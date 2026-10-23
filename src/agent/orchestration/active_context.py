"""Frozen bounded planning visibility; private identities never enter provider prompts."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from agent.schemas.prior_output import PriorOutputBinding
from agent.schemas.orchestration import freeze_json_mapping, _serialize

_CONTEXT = ContextVar('active_planning_context', default=None)
MAX_CONTEXT_ITEMS = 32


@dataclass(frozen=True)
class ActiveContextItem:
    handle: str
    binding: PriorOutputBinding


@dataclass(frozen=True)
class ActivePlanningContext:
    session_id: str
    base_revision_id: str | None
    base_generation: int
    items: tuple[ActiveContextItem, ...]
    results: tuple = ()
    task: object = None

    def __post_init__(self):
        if (type(self.session_id) is not str or not self.session_id
                or self.base_revision_id is not None and (type(self.base_revision_id) is not str or not self.base_revision_id)
                or self.base_revision_id is None and self.items
                or type(self.base_generation) is not int or self.base_generation < 0):
            raise ValueError('Invalid captured context identity.')
        if (type(self.items) is not tuple or len(self.items) > MAX_CONTEXT_ITEMS
                or any(not isinstance(i, ActiveContextItem) or not isinstance(i.binding, PriorOutputBinding)
                       or i.handle != f'ctx.{n}' for n, i in enumerate(self.items))):
            raise ValueError('Invalid bounded active planning context.')
        object.__setattr__(self, 'results', tuple(freeze_json_mapping(row, 'accepted result') for row in self.results))
        object.__setattr__(self, 'task', freeze_json_mapping({} if self.task is None else self.task, 'captured task'))

    def binding(self, handle):
        for item in self.items:
            if item.handle == handle:
                return item.binding
        raise ValueError('Unknown or unoffered active context handle.')

    def public_items(self):
        return tuple({'handle': i.handle, 'semantic_type': i.binding.semantic_type,
                      'producer_tool': i.binding.tool_name, 'source_port': i.binding.source_port,
                      'verification_mode': i.binding.verification_mode} for i in self.items)

    def public_research_context(self):
        selected = set(self.task.get('references', ()))
        selected.update(row['result'] for row in self.task.get('retained_parameters', ()))
        selected.update(row['handle'] for row in self.results if 'current' in row['relations'])
        while True:
            dependencies = {h for row in self.results if row['handle'] in selected for h in row['derived_from']}
            if dependencies <= selected:
                break
            selected.update(dependencies)
        results = [_serialize(row) for row in self.results if row['handle'] in selected]
        for row in results:
            row['reusable_inputs'] = [h for h in row.get('reusable_inputs', ())
                                      if h in {item.handle for item in self.items}]
        task = _serialize(self.task)
        if task.get('delivery'):
            task['delivery_status'] = 'deferred'
        return dict(base=None if self.base_revision_id is None else 'captured_current', results=results, task=task)


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
