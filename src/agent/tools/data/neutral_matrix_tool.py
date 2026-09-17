"""Public neutral adoption; reference supplies its explicit species and assembly."""
from dataclasses import asdict
from . import regulatory_matrix_adoption as backend
from .regulatory_feature_reference import load_regulatory_feature_reference

ARGUMENTS = ('source_path','source_sha256','reference_manifest_path',
             'reference_manifest_sha256','matrix_semantics','output_dir')
RECOVERY_POLICY = backend.RECOVERY_POLICY


def expand(args):
    args = dict(args)
    from .scatac_matrix_contract import shape
    shape(args, ARGUMENTS)
    _, reference, _ = load_regulatory_feature_reference(args['reference_manifest_path'],
        expected_sha256=args['reference_manifest_sha256'])
    return dict(args, species=asdict(reference.species), assembly=reference.target_assembly)


def _publication(args, identity):
    return backend._publication(expand(args), identity)


_receipt = backend._receipt


def execute(arguments, execution_identity=None):
    return backend.execute_matrix(expand(arguments), execution_identity)


def recover(arguments, execution_identity):
    return backend.recover_matrix(expand(arguments), execution_identity)


def verify_public_result(args, result):
    return backend.verify_public_result(expand(args), result)


def adopt_scATAC_cell_by_features(source_path, source_sha256, reference_manifest_path,
        reference_manifest_sha256, matrix_semantics, output_dir):
    return execute(locals())
