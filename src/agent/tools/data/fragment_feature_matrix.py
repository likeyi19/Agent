"""M14.6 data-layer matrix owner using the existing publication/recovery envelope."""
from pathlib import Path

from . import scatac_matrix as lifecycle, scatac_matrix_contract as m
from . import fragment_feature_matrix_contract as contract
from ._fragment_feature_binding import bind
from ._cell_by_ccre_io import MatrixLimits
from .scatac_cell_by_ccre import _build_matrix
from .cell_by_ccre_verifier import verify_cell_by_ccre
from .scatac_fragments_v2_verifier import take_snapshots, check_snapshots
from agent.tools._cancellation import cancellation_checkpoint

RECOVERY_POLICY = contract.RECOVERY_POLICY
ARGUMENTS = tuple(f'{prefix}_manifest_{suffix}' for prefix in ('fragments', 'explicit_cells', 'reference')
                  for suffix in ('path', 'sha256')) + ('output_dir',)
SELECTED_ARGUMENTS = tuple(k.replace('explicit_cells', 'selected_cells') for k in ARGUMENTS)


def argument_contract(args):
    from . import selected_feature_matrix_contract as selected
    return selected if set(args) == set(SELECTED_ARGUMENTS) else contract


def cell_prefix(args):
    return 'selected_cells' if set(args) == set(SELECTED_ARGUMENTS) else 'explicit_cells'


def arguments(value):
    args = dict(value)
    if set(args) not in (set(ARGUMENTS), set(SELECTED_ARGUMENTS)): m.fail('MATRIX_CELL_INPUT_REQUIRED')
    for key in args:
        if key.endswith('_sha256'): m.sha(args[key])
        else:
            if isinstance(args[key], Path): args[key] = str(args[key])
            m.absolute_path(args[key])
    return args


def _publication(args, execution_identity):
    args = arguments(args)
    token = lifecycle.digest(dict(arguments=args, policy=RECOVERY_POLICY, execution_identity=execution_identity,
                                  matrix_profile_sha256=argument_contract(args).PROFILE_SHA256))
    return args, Path(args['output_dir']) / ('cell-by-ccre-' + token), token


def _receipt(destination, args, token=None):
    return lifecycle._receipt(destination, args, token, policy=RECOVERY_POLICY, profile=argument_contract(args).PROFILE_SHA256)


def _summary(value, path, sha):
    selected_contract = contract.contract_for(value)
    selected_contract.validate_manifest(value)
    return lifecycle._summary(value, path, sha) | dict(artifact_type=contract.ARTIFACT,
        contract_version=selected_contract.CONTRACT, matrix_profile_id=selected_contract.PROFILE.profile_id,
        matrix_profile_sha256=selected_contract.PROFILE_SHA256)


def verify_public_result(args, result):
    args = arguments(args); path = Path(result['manifest_path']); destination = path.parent.parent
    if (path != path.resolve() or path.name != 'manifest.json' or path.parent.name != 'artifact'
            or destination.parent != Path(args['output_dir'])): m.fail('MATRIX_RESULT_MISMATCH')
    before = take_snapshots([path, destination / 'receipt.json'])
    receipt = _receipt(destination, args)
    if receipt['manifest_sha256'] != result['manifest_sha256']: m.fail('MATRIX_RECOVERY_MISMATCH')
    value = lifecycle._load(path, result['manifest_sha256']); argument_contract(args).validate_manifest(value)
    for key, prefix in (('fragments', 'fragments'), ('cells', cell_prefix(args)), ('reference', 'reference')):
        pointer = value['upstream'][key]
        if (pointer['manifest_path'] != args[prefix + '_manifest_path']
                or pointer['manifest_sha256'] != args[prefix + '_manifest_sha256']): m.fail('MATRIX_LINEAGE_INVALID')
    if dict(result) != _summary(value, path, result['manifest_sha256']): m.fail('MATRIX_RESULT_MISMATCH')
    verify_cell_by_ccre(path, expected_sha256=result['manifest_sha256'], bedtools_path=lifecycle.executable())
    check_snapshots(before)
    return value


def recover_matrix(args, execution_identity):
    args, destination, token = _publication(args, execution_identity)
    if not destination.is_dir() or destination.is_symlink(): m.fail('MATRIX_RECOVERY_UNAVAILABLE')
    receipt = _receipt(destination, args, token); path = destination / 'artifact/manifest.json'
    result = _summary(lifecycle._load(path, receipt['manifest_sha256']), path, receipt['manifest_sha256'])
    verify_public_result(args, result)
    return result


def build_cell_by_features(*, fragments_manifest_path, fragments_manifest_sha256,
        reference_manifest_path, reference_manifest_sha256, output_dir,
        explicit_cells_manifest_path=None, explicit_cells_manifest_sha256=None,
        selected_cells_manifest_path=None, selected_cells_manifest_sha256=None, execution_identity=None):
    """Publish raw canonical-record counts; AGENT_MATRIX_BEDTOOLS is operator configuration.

    Standalone recovery remains deep. Within an owned VerificationContext,
    compatible verification/recovery reuses the matrix owner's proof.
    """
    supplied = dict(locals()); supplied.pop('execution_identity')
    supplied = {k:v for k,v in supplied.items() if v is not None}
    cancellation_checkpoint()
    args, destination, token = _publication(supplied, execution_identity)
    backend = lifecycle.executable()
    pointers = {key: dict(manifest_path=args[prefix + '_manifest_path'],
                         manifest_sha256=args[prefix + '_manifest_sha256'])
                for key, prefix in (('fragments', 'fragments'), ('cells', cell_prefix(args)), ('reference', 'reference'))}
    selected_contract = argument_contract(args)
    def build(output):
        def bind_exact(upstream, limits):
            bound = bind(upstream, limits)
            if bound.contract is not selected_contract: m.fail('MATRIX_LINEAGE_INVALID')
            return bound
        return _build_matrix(pointers, output, backend, MatrixLimits(), contract=selected_contract, binder=bind_exact)
    return lifecycle._execute_publication(args, destination, token, build, _summary,
                                          RECOVERY_POLICY, selected_contract.PROFILE_SHA256)
