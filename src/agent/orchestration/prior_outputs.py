"""Exact historical resolution using an operator-owned RunStore; no discovery."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import hashlib
import json

from agent.schemas.prior_output import PriorOutputBinding, PriorOutputRef
from agent.schemas.verification_authority import AuthorityError
from agent.schemas.run_state import RunLifecycleStatus

_STORE = ContextVar('prior_output_store', default=None)
_LOADING = ContextVar('prior_output_loading', default=())


def accepted_step_digest(step):
    # Match the existing M15.2 locator encoding, including non-ASCII paths/IDs.
    # The authority's own digest retains its original independent encoding.
    return hashlib.sha256(json.dumps(step.to_dict(), sort_keys=True,
        separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


@contextmanager
def prior_output_store(store):
    token = _STORE.set(store)
    try:
        yield
    finally:
        _STORE.reset(token)


def required_store():
    store = _STORE.get()
    if store is None:
        raise AuthorityError('Historical execution requires a trusted configured RunStore.')
    return store


def runtime_store(function):
    @wraps(function)
    def call(runtime, *args, **kwargs):
        with prior_output_store(runtime.run_store):
            return function(runtime, *args, **kwargs)
    return call


def source_load_guard(function):
    @wraps(function)
    def call(store, run_id, *args, **kwargs):
        key = (id(store), run_id)
        stack = _LOADING.get()
        if key in stack or len(stack) >= 64:
            raise AuthorityError('Cyclic or excessive historical dependency closure.')
        token = _LOADING.set(stack + (key,))
        try:
            return function(store, run_id, *args, **kwargs)
        finally:
            _LOADING.reset(token)
    return call


def references(plan):
    if plan is None:
        return ()
    return tuple(a for s in plan.steps for a in s.arguments.values() if isinstance(a, PriorOutputRef))


def source_port(registry, tool_name, name):
    semantic = registry.get(tool_name).semantic_planning
    ports = [] if semantic is None else [p for p in semantic.producer_ports if p.name == name]
    if len(ports) != 1:
        raise AuthorityError('Unknown historical semantic output port.')
    return ports[0]


def bind_output(store, registry, run_id, step_id, port_name):
    from .verification_authority import _accepted
    state = store.load(run_id)
    if state.lifecycle_status is not RunLifecycleStatus.SUCCEEDED:
        raise AuthorityError('Historical source run is not terminal successful.')
    stored, authority, _, _ = _accepted(store, run_id, step_id)
    if authority.record['schema_version'] != 2:
        raise AuthorityError('Historical execution requires reusable schema-2 authority.')
    port = source_port(registry, stored.tool_name, port_name)
    if any(m.field_name not in stored.result for m in port.members):
        raise AuthorityError('Historical output is missing a semantic member.')
    return PriorOutputBinding(run_id, step_id, stored.tool_name, port_name, port.semantic_type,
        accepted_step_digest(stored), authority.record['manifest_sha256'], authority.identity_sha256)


def validate_binding(binding, *, store=None, registry=None, integrity=False):
    from .registry import build_default_tool_registry
    store = required_store() if store is None else store
    registry = build_default_tool_registry() if registry is None else registry
    if bind_output(store, registry, binding.run_id, binding.step_id, binding.source_port) != binding:
        raise AuthorityError('Historical output anchor, publication, or authority changed.')
    if integrity:
        from .verification_authority import _load_context
        _load_context(store, binding.run_id, 'historical_verified_sources.v1').validate_anchors()
    state = store.load(binding.run_id)
    return next(s for s in state.steps if s.step_id == binding.step_id)


def resolve(reference):
    from .registry import build_default_tool_registry
    registry = build_default_tool_registry()
    port = source_port(registry, reference.binding.tool_name, reference.binding.source_port)
    if reference.output_key not in {m.field_name for m in port.members}:
        raise AuthorityError('Historical member is not part of its atomic semantic output.')
    stored = validate_binding(reference.binding, registry=registry)
    return stored.result[reference.output_key]


def channel_for(binding, tool_name, target_port, registry):
    from .semantic_compiler import build_semantic_compiler_contract
    channels = [c for c in build_semantic_compiler_contract(registry).step_output_channels
                if (c.producer_tool_name, c.source_port, c.consumer_tool_name, c.target_port)
                == (binding.tool_name, binding.source_port, tool_name, target_port)]
    if len(channels) != 1:
        raise AuthorityError('Historical output is incompatible with the consumer port.')
    channel = channels[0]
    # Existing special request-role lineage contracts must not be bypassed.
    # M15.3 admits owner-authorized artifact channels without unresolved roles.
    if channel.required_lineage is not None:
        raise AuthorityError('Historical channel requires an unsupported explicit role lineage.')
    return channel


def validate_step_references(step, registry):
    refs = {k: a for k, a in step.arguments.items() if isinstance(a, PriorOutputRef)}
    if not refs:
        return
    semantic = registry.get(step.tool_name).semantic_planning
    covered = set()
    for port in (() if semantic is None else semantic.consumer_ports):
        names = {m.field_name for m in port.members}
        selected = names.intersection(refs)
        if not selected:
            continue
        if selected != names:
            raise AuthorityError('Historical semantic artifact must be selected atomically.')
        bindings = {refs[n].binding for n in names}
        if len(bindings) != 1:
            raise AuthorityError('Mixed historical publication members.')
        binding = next(iter(bindings))
        channel = channel_for(binding, step.tool_name, port.name, registry)
        if any(refs[m.argument_name].output_key != m.output_key for m in channel.members):
            raise AuthorityError('Historical semantic member mapping changed.')
        validate_binding(binding, registry=registry, integrity=True)
        covered.update(names)
    if covered != set(refs):
        raise AuthorityError('Historical reference outside an authorized artifact port.')


def merge_context(target, source):
    for name in ('accepted', 'proofs', 'locations', '_hashes'):
        destination, incoming = getattr(target, name), getattr(source, name)
        if any(k in destination and destination[k] != v for k, v in incoming.items()):
            raise AuthorityError('Conflicting historical scientific authorities.')
        destination.update(incoming)
    previous = target.validate_anchors
    def validate():
        previous()
        source.validate_anchors()
    target.validate_anchors = validate


def import_plan_sources(plan, context, store):
    from .verification_authority import _load_context
    bindings = tuple(dict.fromkeys(r.binding for r in references(plan)))
    imported = set()
    for binding in bindings:
        validate_binding(binding, store=store)
        if binding.run_id not in imported:
            merge_context(context, _load_context(store, binding.run_id, 'historical_verified_sources.v1'))
            imported.add(binding.run_id)
    previous = context.validate_anchors
    def validate():
        previous()
        for binding in bindings:
            validate_binding(binding, store=store)
    context.validate_anchors = validate
    validate()


def binding_for_locator(locator, store, registry):
    from .verification_authority import _accepted
    step, _, _, _ = _accepted(store, locator.run_id, locator.step_id)
    if accepted_step_digest(step) != locator.accepted_step_sha256:
        raise AuthorityError('Active locator accepted-step anchor changed.')
    semantic = registry.get(step.tool_name).semantic_planning
    ports = [] if semantic is None else [p for p in semantic.producer_ports
        if locator.output_key in {m.field_name for m in p.members}]
    if len(ports) != 1:
        raise AuthorityError('Active locator does not identify a unique semantic artifact.')
    return bind_output(store, registry, locator.run_id, locator.step_id, ports[0].name)


def validate_active_outputs(locators, store, registry):
    """Check explicit active publications against owner-described dependency closure.

    No workflow completion, graph diff, science, or session-owned compatibility.
    Multiple different publications of one owner kind require future explicit roles.
    """
    from pathlib import Path
    from agent.tools.data.authority_context import VerificationContext
    from agent.tools.data.scientific_authority import TOOLS
    from .verification_authority import _accepted, _load_context
    context = VerificationContext()
    active = {}
    selected = []
    loaded = set()
    for locator in locators:
        binding = binding_for_locator(locator, store, registry)
        if binding.run_id not in loaded:
            merge_context(context, _load_context(store, binding.run_id, 'historical_verified_sources.v1'))
            loaded.add(binding.run_id)
        _, authority, _, _ = _accepted(store, binding.run_id, binding.step_id)
        kind = TOOLS[binding.tool_name]
        identity = (authority.record['publication_path'], binding.manifest_sha256)
        if kind in active and active[kind] != identity:
            raise AuthorityError('Multiple active publications require explicit distinct scientific roles.')
        active[kind] = identity
        selected.append((kind, *identity))
    visited = set()
    def check(kind, path, sha):
        key = (kind, path, sha)
        if key in visited:
            return
        visited.add(key)
        if kind in active and active[kind] != (path, sha):
            raise AuthorityError('Active downstream output refers to a different active upstream publication.')
        description, _ = context.describe(kind, Path(path), sha)
        for dep in description['upstream'].values():
            check(dep['kind'], dep['manifest_path'], dep['manifest_sha256'])
    for item in selected:
        check(*item)
    context.validate_anchors()
