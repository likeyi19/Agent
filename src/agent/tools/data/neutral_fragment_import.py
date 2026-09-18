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


def import_primary_fragments(*, preparation_path, preparation_sha256, reference_bundle_path,
                            reference_bundle_sha256, namespace, output_dir, execution_identity=None):
    """Preferred neutral admission: mandatory explicit primary-nuclear preparation.

    Historical import_external_fragments remains the explicit unscoped contract;
    its artifacts and receipts are never reinterpreted as primary-scoped.
    """
    from .primary_fragment_preparation import load
    from ._external_fragment_io import resource
    preparation = load(preparation_path, preparation_sha256)
    source, index = preparation['prepared'], preparation['prepared_index']
    args = dict(source_path=source['path'], source_sha256=source['sha256'],
        source_profile='10x-atac-fragments.v1', source_selection='subset_export',
        source_index_path=index['path'], source_index_sha256=index['sha256'],
        reference_bundle_path=reference_bundle_path, reference_bundle_sha256=reference_bundle_sha256,
        namespace=namespace, output_dir=str(output_dir),
        primary_preparation=resource(preparation_path, preparation_sha256))
    return execute_external_fragments(args, execution_identity)
