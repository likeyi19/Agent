"""Data-layer external adoption with an explicit M14.1 neutral reference.

The registered M11 tool retains its legacy reference scope. Source science,
producer profile, v2 serialization and durable publication are shared unchanged.
"""
from functools import partial
from . import scatac_fragment_import as owner

RECOVERY_POLICY = owner.RECOVERY_POLICY
_publication = owner._publication
_receipt = owner._receipt
verify_public_result = owner.verify_public_result
recover_external_fragments = partial(owner.recover_external_fragments, _neutral_reference=True)
execute_external_fragments = partial(owner.execute_external_fragments, _neutral_reference=True)


def import_external_fragments(*, source_path, source_sha256, source_profile,
        reference_bundle_path, reference_bundle_sha256, namespace, output_dir,
        source_selection='unknown', source_index_path=None, source_index_sha256=None,
        execution_identity=None):
    args = dict(locals()); args.pop('execution_identity')
    return execute_external_fragments(args, execution_identity)
