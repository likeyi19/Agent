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
    contract = contract_for_tool(step.tool_name)
    if (stored.tool_name != step.tool_name
            or stored.status is not StepStatus.SUCCEEDED or stored.result is None or stored.error is not None
            or verification is None or not verification.passed or verification.target_type != 'step'
            or verification.target_id != step_id or verification.artifact_authority is None):
        raise AuthorityError('No recorded scientific authority; legacy qualification is not implicit.')
    spec = build_default_tool_registry().get(step.tool_name)
    identity = execution_identity(run_id, state.plan, step, spec)
    authority = VerifiedArtifactAuthority(verification.artifact_authority)
    contract.require(authority)
    if authority.record['execution_identity'] != identity:
        raise AuthorityError('Authority belongs to a different execution.')
    # Bind handle to the entire accepted step, not only its mutable artifact paths.
    return stored, authority, identity, digest(stored.to_dict())


def load_fragment_authority(store, run_id, step_id):
    """Load newly recorded authority; never create or upgrade legacy provenance."""
    stored, _, _, original_anchor = _accepted(store, run_id, step_id)

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
