"""Public M14.6 owner adapter; no new counting or publication implementation."""
from . import fragment_feature_matrix as owner

ARGUMENTS = owner.ARGUMENTS
RECOVERY_POLICY = owner.RECOVERY_POLICY
_publication = owner._publication
_receipt = owner._receipt
verify_public_result = owner.verify_public_result
recover = owner.recover_matrix


def execute(arguments, execution_identity=None):
    return owner.build_cell_by_features(**arguments, execution_identity=execution_identity)


def build_scATAC_cell_by_features(*, fragments_manifest_path, fragments_manifest_sha256,
        reference_manifest_path, reference_manifest_sha256, output_dir,
        explicit_cells_manifest_path=None, explicit_cells_manifest_sha256=None,
        selected_cells_manifest_path=None, selected_cells_manifest_sha256=None):
    return execute({k:v for k,v in locals().items() if v is not None})
