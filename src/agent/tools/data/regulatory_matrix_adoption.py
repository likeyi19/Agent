"""M14.1 data-layer adoption through the existing matrix owner and lifecycle.

No registry entry or Planner integration. Schema-2 issuance uses the existing
scientific_authority.issue API inside an explicitly owned VerificationContext.
Parsed authority records never establish accepted execution trust.
"""
from functools import partial
from . import scatac_matrix_adoption as owner, regulatory_matrix_contract as contract

RECOVERY_POLICY = contract.RECOVERY_POLICY
arguments = partial(owner.arguments, contract=contract)
_publication = partial(owner._publication, contract=contract)
_receipt = partial(owner._receipt, contract=contract)
verify_public_result = partial(owner.verify_public_result, contract=contract)
recover_matrix = partial(owner.recover_matrix, contract=contract)
execute_matrix = partial(owner.execute_matrix, contract=contract)


def adopt_cell_by_features(*, source_path, source_sha256, reference_manifest_path,
        reference_manifest_sha256, species, assembly, matrix_semantics, output_dir,
        execution_identity=None):
    args = dict(source_path=source_path, source_sha256=source_sha256,
        reference_manifest_path=reference_manifest_path, reference_manifest_sha256=reference_manifest_sha256,
        species=species, assembly=assembly, matrix_semantics=matrix_semantics, output_dir=output_dir)
    return execute_matrix(args, execution_identity)
