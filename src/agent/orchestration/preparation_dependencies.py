"""Exact source and data-edge scopes shared by compiler and preparation."""
from collections.abc import Mapping
import hashlib
import json


def accepted_source_identity(binding):
    """Preserve the Session anchor's original UTF-8 canonical byte meaning."""
    value=binding.to_dict() if hasattr(binding,'to_dict') else binding
    raw=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')
    return 'accepted:'+hashlib.sha256(raw).hexdigest()


def operation_sources(step_id, operations, registry=None, *, lineage=False):
    """Project artifact operands along the compiler's validated data edges."""
    from agent.schemas import StepOutputRef, PriorOutputRef
    if registry is None:
        from agent.orchestration.registry import build_default_tool_registry
        registry = build_default_tool_registry()
    by_id = {op.step_id: op for op in operations}
    def visit(value, seen):
        if isinstance(value, PriorOutputRef):
            return {accepted_source_identity(value.binding)}
        if isinstance(value, StepOutputRef):
            return operation(value.step_id, seen, output_key=value.output_key)
        if isinstance(value, Mapping):
            return set().union(*(visit(v, seen) for v in value.values())) if value else set()
        if isinstance(value, (tuple, list)):
            return set().union(*(visit(v, seen) for v in value)) if value else set()
        return {value} if isinstance(value, str) else set()
    def operation(identifier, seen, output_key=None):
        if identifier in seen or identifier not in by_id:
            return set()
        op = by_id[identifier]
        tool = registry.get(op.tool_name)
        specs = {**tool.required_arguments, **tool.optional_arguments}
        fields = None
        if lineage:
            semantic = tool.semantic_planning
            producers = [p for p in semantic.producer_ports
                         if output_key is None or any(m.field_name == output_key for m in p.members)]
            catalog = tool.native_functions
            if catalog is not None and catalog.output_available is not None:
                producers = [p for p in producers if all(catalog.output_available(op.arguments, m.field_name) for m in p.members)]
            ports = {p.lineage_from_port for p in producers if p.lineage_from_port is not None}
            if not ports and output_key is None:
                physical = [p for p in semantic.consumer_ports if p.name in tool.source_input_types
                            and any(op.arguments.get(m.field_name) is not None for m in p.members)]
                ports = {p.name for p in physical if p.required} or {p.name for p in physical}
            # A consumer receives declarations only on its dataset lineage.
            # Ambiguous/missing lineage is not permission to inherit all inputs.
            if len(ports) != 1:
                return set()
            fields = {m.field_name for p in semantic.consumer_ports if p.name in ports for m in p.members}
        return set().union(*(visit(value, seen | {identifier}) for name, value in op.arguments.items()
            if (fields is None or name in fields) and (isinstance(value, (StepOutputRef, PriorOutputRef)) or name in specs and specs[name].planning is not None
                and specs[name].planning.accepted_artifact_kinds)))
    return sorted(operation(step_id, frozenset()))


def operation_dependencies(step_id, operations):
    """Project compiler data references; scheduling edges do not carry a resource."""
    from agent.schemas import StepOutputRef
    def producers(value):
        if isinstance(value, StepOutputRef):
            return {value.step_id}
        if isinstance(value, Mapping):
            return set().union(*(producers(v) for v in value.values())) if value else set()
        if isinstance(value, (tuple, list)):
            return set().union(*(producers(v) for v in value)) if value else set()
        return set()
    by_id = {op.step_id: op for op in operations}
    seen, pending = set(), [step_id]
    while pending:
        current = pending.pop()
        if current in seen or current not in by_id:
            continue
        seen.add(current)
        pending.extend(producers(by_id[current].arguments))
    return sorted(seen)
