"""Shared dispatch for repository-owned scientific publication contracts.

Historical binding and accepted execution state remain orchestration concerns.
Registered owners supply scientific descriptions and compatibility callbacks;
the existing data-layer adapter retains its original contracts and policies.
"""
from pathlib import Path

from agent.schemas.verification_authority import AuthorityError, VerifiedArtifactAuthority


def _raw():
    from .data import scientific_authority
    return scientific_authority


def _registry(registry):
    from agent.orchestration.registry import ToolRegistry, build_default_tool_registry
    if registry is None:
        return build_default_tool_registry()
    if not isinstance(registry, ToolRegistry):
        raise TypeError('Scientific authority requires an immutable ToolRegistry.')
    return registry


def _raw_kinds():
    raw = _raw()
    return {*raw.TOOLS.values(), *raw.MODULES, 'generic_fragments', 'explicit_cells'}


def _registered_hooks(registry):
    registry = _registry(registry)
    kinds = {}
    tools = {}
    raw = _raw()
    raw_kinds = _raw_kinds()
    for name in registry.names():
        hooks = registry.get(name).scientific_authority
        if hooks is None:
            continue
        if name in raw.TOOLS or hooks.kind in raw_kinds:
            raise AuthorityError('Registered owner cannot replace an existing scientific authority contract.')
        if hooks.kind in kinds:
            raise AuthorityError('Scientific authority kind has multiple registered owners.')
        tools[name] = hooks
        kinds[hooks.kind] = hooks
    return tools, kinds


def validate_registry_hooks(registry):
    """Validate explicit owner contracts before a Registry can execute work."""
    _registered_hooks(registry)


def hooks_for_tool(tool_name, registry=None):
    """Return only explicit Registry owner hooks; raw contracts remain separate."""
    return _registered_hooks(registry)[0].get(tool_name)


def supports_tool(tool_name, registry=None):
    return hooks_for_tool(tool_name, registry) is not None or tool_name in _raw().TOOLS


def tool_kind(tool_name, registry=None):
    hooks = hooks_for_tool(tool_name, registry)
    if hooks is not None:
        return hooks.kind
    try:
        return _raw().TOOLS[tool_name]
    except KeyError as exc:
        raise AuthorityError('Unsupported scientific authority contract.') from exc


def qualified_port(tool_name, port_name, registry=None):
    """Require explicit owner qualification without expanding semantic channels."""
    admitted_registry = _registry(registry)
    hooks = hooks_for_tool(tool_name, admitted_registry)
    if hooks is not None:
        return port_name in hooks.producer_ports
    if tool_name not in _raw().TOOLS or not admitted_registry.contains(tool_name):
        return False
    semantic = admitted_registry.get(tool_name).semantic_planning
    return semantic is not None and port_name in {port.name for port in semantic.producer_ports}


def _hooks_for_kind(kind, registry=None):
    hooks = _registered_hooks(registry)[1].get(kind)
    if hooks is None and kind not in _raw_kinds():
        raise AuthorityError('Unsupported scientific authority contract.')
    return hooks


def describe(kind, path, sha, context, *, registry=None):
    hooks = _hooks_for_kind(kind, registry)
    if hooks is None:
        return _raw().describe(kind, path, sha, context)
    return hooks.describe(path, sha, context)


def _publication_record(context, tool_name, arguments, result, execution_identity,
                        result_metadata, *, registry=None):
    """Describe exact accepted provenance; this function does not issue trust."""
    hooks = hooks_for_tool(tool_name, registry)
    if hooks is None:
        tool_kind(tool_name, registry)
        return _raw()._publication_record(context, tool_name, arguments, result,
                                          execution_identity, result_metadata)
    authority = hooks.publication_record(context, tool_name, arguments, result,
                                         execution_identity, result_metadata)
    if (not isinstance(authority, VerifiedArtifactAuthority)
            or authority.record['schema_version'] != 2
            or authority.record['scope'] != 'scientific_correctness.v1'
            or authority.record['execution_identity'] != execution_identity):
        raise AuthorityError('Owner publication lacks current scientific execution authority.')
    return authority


def issue(context, tool_name, arguments, result, execution_identity, *, registry=None):
    hooks = hooks_for_tool(tool_name, registry)
    if hooks is None:
        tool_kind(tool_name, registry)
        return _raw().issue(context, tool_name, arguments, result, execution_identity)
    _, key = context.describe(hooks.kind, Path(result['manifest_path']), result['manifest_sha256'])
    proof = context.proofs.get(key)
    if proof is None or proof.kind != hooks.kind:
        raise AuthorityError('No completed independent scientific proof.')
    return _publication_record(context, tool_name, arguments, result, execution_identity,
                               proof.result_metadata, registry=registry)


def reuse_result(kind, description, proof, kwargs, *, registry=None):
    hooks = _hooks_for_kind(kind, registry)
    if hooks is None:
        return _raw().reuse_result(kind, description, proof, kwargs)
    return hooks.reuse_result(kind, description, proof, kwargs)


def runtime_compatibility(kind, description, kwargs, *, registry=None):
    hooks = _hooks_for_kind(kind, registry)
    if hooks is None:
        return _raw().runtime_compatibility(kind, description, kwargs)
    if hooks.runtime_compatibility is not None:
        return hooks.runtime_compatibility(kind, description, kwargs)
    return None


def proof_metadata(kind, result, context, *, description=None, registry=None):
    """Capture owner metadata only after its independent verifier completes."""
    hooks = _hooks_for_kind(kind, registry)
    if hooks is not None:
        return hooks.proof_metadata(kind, result, context)
    if kind == 'fastq_fragment_production':
        if description is None:
            raise AuthorityError('FASTQ proof metadata requires its exact verified publication locator.')
        _, generic_key = context.describe('generic_fragments', Path(description['manifest_path']),
                                         description['manifest_sha256'])
        if generic_key not in context.proofs:
            raise AuthorityError('Producer verification lacks canonical integrity proof.')
        return dict(contigs=context.proofs[generic_key].result_metadata['contigs'])
    if kind == 'generic_fragments':
        return dict(contigs=result.contigs)
    if kind in ('bam_fragment_production', 'external_fragment_adoption'):
        return dict(contigs=result.fragments.contigs)
    if kind in ('matrix', 'annotation', 'epizoo_adaptation'):
        return result
    return {}


def needs_source_snapshot(kind, description, *, registry=None):
    hooks = _hooks_for_kind(kind, registry)
    if hooks is not None:
        return hooks.verify_sources
    from .data.fragments_authority_contract import CONTRACTS
    from .data.external_matrix_contract import is_external
    return kind in CONTRACTS or (kind == 'matrix' and is_external(description['manifest']))


__all__ = [
    'describe', 'hooks_for_tool', 'issue', 'needs_source_snapshot',
    'proof_metadata', 'qualified_port', 'reuse_result', 'runtime_compatibility',
    'supports_tool', 'tool_kind',
    'validate_registry_hooks',
]
