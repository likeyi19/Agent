"""Load authority only from operator-owned accepted execution persistence.

RunStore is a trusted application dependency, not a user/planner input. Its
existing checksum/revision protections are not authentication against an actor
who can replace the store itself. No manifest/receipt-only adoption API exists.
"""
from agent.schemas import StepStatus
from agent.schemas.verification_authority import (
    AuthorityError, AuthorityValidation, VerifiedArtifactAuthority, digest,
)
from agent.tools.data.fragments_authority_contract import (
    StoredFragmentAuthority, contract_for_tool, validate_publication_integrity,
)
from .durable_tool_recovery import execution_identity
from .registry import build_default_tool_registry

def _accepted(store, run_id, step_id):
    state = store.load(run_id)
    if state.run_id != run_id or state.plan is None:
        raise AuthorityError('Missing accepted execution.')
    steps = [s for s in state.steps if s.step_id == step_id]
    planned = [s for s in state.plan.steps if s.step_id == step_id]
    if len(steps) != 1 or len(planned) != 1:
        raise AuthorityError('Ambiguous accepted step.')
    stored, step = steps[0], planned[0]
    verification = stored.verification
    if (stored.tool_name != step.tool_name
            or stored.status is not StepStatus.SUCCEEDED or stored.result is None or stored.error is not None
            or verification is None or not verification.passed or verification.target_type != 'step'
            or verification.target_id != step_id or verification.artifact_authority is None):
        raise AuthorityError('No recorded scientific authority; legacy qualification is not implicit.')
    spec = build_default_tool_registry().get(step.tool_name)
    identity = execution_identity(run_id, state.plan, step, spec)
    authority = VerifiedArtifactAuthority(verification.artifact_authority)
    if authority.record['schema_version'] == 1:
        contract_for_tool(step.tool_name).require(authority)
    else:
        from agent.tools.data.scientific_authority import TOOLS
        if step.tool_name not in TOOLS:
            raise AuthorityError('Unsupported scientific authority contract.')
    if authority.record['execution_identity'] != identity:
        raise AuthorityError('Authority belongs to a different execution.')
    # Bind handle to the entire accepted step, not only its mutable artifact paths.
    return stored, authority, identity, digest(stored.to_dict())


def load_fragment_authority(store, run_id, step_id, *, source_policy=None):
    """Load newly recorded authority; never create or upgrade legacy provenance."""
    stored, recorded, _, original_anchor = _accepted(store, run_id, step_id)
    if recorded.record['schema_version'] == 2:
        return load_scientific_authority(store, run_id, step_id, source_policy=source_policy or 'historical_verified_sources.v1')
    if source_policy not in (None, 'current_source_freshness.v1'):
        raise AuthorityError('Version 1 authority retains current-source validation semantics.')

    def validate(path, sha, producer_kind):
        stored, authority, identity, anchor = _accepted(store, run_id, step_id)
        if anchor != original_anchor:
            raise AuthorityError('Accepted verification anchor changed.')
        if producer_kind is not None and authority.record['producer_qualification']['kind'] != producer_kind:
            raise AuthorityError('Producer qualification cannot substitute for another producer.')
        if stored.result['manifest_path'] != str(path) or stored.result['manifest_sha256'] != sha:
            raise AuthorityError('Consumer requested a different artifact identity.')
        validate_publication_integrity(authority, stored.resolved_arguments, stored.result, identity)
        if _accepted(store, run_id, step_id)[3] != anchor:
            raise AuthorityError('Accepted verification anchor changed during validation.')
        return AuthorityValidation(authority.identity_sha256)
    handle = StoredFragmentAuthority._from_accepted_store(validate)
    handle.validate(stored.result['manifest_path'], stored.result['manifest_sha256'])
    return handle


def _load_context(store, run_id, source_policy):
    from agent.tools.data.authority_context import VerificationContext, ScientificProof
    from agent.tools.data.scientific_authority import TOOLS, _publication_record
    from agent.schemas.orchestration import freeze_json_mapping
    from pathlib import Path
    if source_policy not in ('historical_verified_sources.v1', 'current_source_freshness.v1'):
        raise AuthorityError('Unsupported source freshness policy.')
    state = store.load(run_id)
    context = VerificationContext()
    selected = []
    for step in state.steps:
        if step.verification is None or step.verification.artifact_authority is None:
            continue
        stored, authority, identity, anchor = _accepted(store, run_id, step.step_id)
        if authority.record['schema_version'] != 2:
            continue  # no implicit upgrade of .1 or legacy verification
        record = authority.record
        context.accepted[record['publication_path'], record['manifest_sha256']] = {
            'record': record, 'identity': authority.identity_sha256}
        selected.append((stored, authority, identity, anchor))
    for stored, authority, identity, anchor in selected:
        record = authority.record
        if set(record['resources']) != {'scientific_proof_sha256', 'manifest_identity', 'result_metadata'}:
            raise AuthorityError('Missing scientific authority resource bindings.')
        described = _publication_record(context, stored.tool_name, stored.resolved_arguments,
            stored.result, identity, record['resources']['result_metadata'])
        if described != authority:
            raise AuthorityError('Persisted authority compatibility, lineage or resource mismatch.')
        if source_policy == 'current_source_freshness.v1':
            expected = authority.to_dict()['historical_sources']
            if context.files(f['path'] for f in expected) != expected:
                raise AuthorityError('Current source bytes differ from historical verified identity.')
        kind = TOOLS[stored.tool_name]
        description, key = context.describe(kind, Path(record['publication_path']), record['manifest_sha256'])
        if _accepted(store, run_id, stored.step_id)[3] != anchor:
            raise AuthorityError('Accepted verification anchor changed.')
        context.proofs[key] = ScientificProof(key, kind, freeze_json_mapping(description, 'proof'),
                                            record['resources']['result_metadata'])
        context.locations[kind, record['publication_path']] = (key, record['manifest_sha256'])
        # Every accepted producer verifier includes canonical v2 integrity. This
        # implication is one-way: a generic proof never grants producer scope.
        from agent.tools.data.fragments_authority_contract import CONTRACTS
        if kind in CONTRACTS:
            generic, generic_key = context.describe('generic_fragments', Path(record['publication_path']), record['manifest_sha256'])
            context.proofs[generic_key] = ScientificProof(generic_key, 'generic_fragments',
                freeze_json_mapping(generic, 'proof'), record['resources']['result_metadata'])
            context.locations['generic_fragments', record['publication_path']] = (generic_key, record['manifest_sha256'])
    anchors = {step.step_id: anchor for step, _, _, anchor in selected}
    def validate_anchors():
        if any(_accepted(store, run_id, step_id)[3] != anchor for step_id, anchor in anchors.items()):
            raise AuthorityError('Accepted execution authority changed during consumption.')
        if source_policy == 'current_source_freshness.v1':
            for _, authority, _, _ in selected:
                expected = authority.to_dict()['historical_sources']
                if context.files(f['path'] for f in expected) != expected:
                    raise AuthorityError('Current source changed during freshness audit.')
    context.validate_anchors = validate_anchors
    validate_anchors()
    return context


def load_scientific_authority(store, run_id, step_id, *, source_policy='historical_verified_sources.v1'):
    stored, authority, _, original = _accepted(store, run_id, step_id)
    if authority.record['schema_version'] != 2:
        raise AuthorityError('Scientific DAG reuse requires explicitly recorded v2 authority.')
    def validate(path, sha, producer_kind):
        if str(path) != authority.record['publication_path'] or sha != authority.record['manifest_sha256']:
            raise AuthorityError('Consumer requested a different artifact identity.')
        if producer_kind is not None and authority.record['producer_qualification'].get('kind') != producer_kind:
            raise AuthorityError('Producer qualification cannot substitute for another producer.')
        _load_context(store, run_id, source_policy).validate_anchors()
        if _accepted(store, run_id, step_id)[3] != original:
            raise AuthorityError('Accepted verification anchor changed.')
        return AuthorityValidation(authority.identity_sha256, VerificationScope.HISTORICAL_INTEGRITY)
    from agent.schemas.verification_authority import VerificationScope
    handle = StoredFragmentAuthority._from_accepted_store(validate)
    handle.validate(stored.result['manifest_path'], stored.result['manifest_sha256'])
    return handle


from contextlib import contextmanager

@contextmanager
def accepted_authorities(store, run_id, *, source_policy='historical_verified_sources.v1'):
    """Explicit scientific operation, never implicitly applied to presentation."""
    from agent.tools.data.authority_context import authority_operation
    context = _load_context(store, run_id, source_policy)
    with authority_operation(context):
        yield context
    context.validate_anchors()
