"""Read an accepted adaptation for the narrower target-inference consumer.

This does not import or renew adaptation authority. Training-code compatibility
still gates training/recovery through the original owner. Inference consumes an
unchanged accepted publication and the exact native EpiZoo code, not a trainer,
source pretrained checkpoint, SEAM executable, or mapping implementation.
"""
import os
from pathlib import Path

from agent.schemas.verification_authority import AuthorityError
from agent.tools.data.authority_context import current, VerificationContext
from agent.tools.models.epizoo_adaptation import publication, contract
from .run_store import FileRunStore
from .verification_authority import _accepted


def accepted_target(path, sha, run_id, step_id):
    root = os.environ.get('AGENT_EPIZOO_SOURCE_STORE')
    if not root or not Path(root).is_absolute() or Path(root) != Path(root).resolve():
        raise AuthorityError('A trusted operator EpiZoo source RunStore is required.')
    store = FileRunStore(root)
    stored, authority, _, anchor = _accepted(store, run_id, step_id)
    record = authority.to_dict()
    path = Path(path).resolve(strict=True)
    if (stored.tool_name != 'adapt_epizoo_species' or record['schema_version'] != 2
            or record['scope'] != 'scientific_correctness.v1' or record['completion'] != 'succeeded'
            or record['artifact_contract'] != contract.CONTRACT
            or record['artifact_type'] != contract.ARTIFACT
            or record['publication_path'] != str(path) or record['manifest_sha256'] != sha
            or stored.result['manifest_path'] != str(path) or stored.result['manifest_sha256'] != sha):
        raise AuthorityError('Target binding requires the exact accepted adaptation publication.')
    manifest = publication.load_manifest(path, sha)
    from agent.tools.models.epizoo_adaptation.resources import runtime
    # CPU worker counts belong to the captured inference invocation, not to
    # immutable model compatibility or read-only result presentation. Scanpy can
    # change the current worker's thread limits after inference has finished.
    numerical = lambda value: {k: v for k, v in value.items()
                               if k not in ('cpu_threads', 'interop_threads')}
    if numerical(manifest['runtime']) != numerical(runtime()):
        raise AuthorityError('Target inference requires the qualified numerical runtime configuration.')
    if (manifest['profile_sha256'] != record['science_profile']
            or stored.result['checkpoint_sha256'] != manifest['files']['checkpoint.pth']['sha256']):
        raise AuthorityError('Accepted target profile/checkpoint binding differs.')
    expected = {item['path']: item for item in record['files']}
    # Every published sidecar and receipt remains immutable. Require all native
    # EpiZoo Python modules from the accepted execution, including strict loading,
    # model, processing, dataset and inference implementations.
    import epizoo
    native = Path(epizoo.__file__).resolve().parent
    required = {str(path), str(path.parent.parent / 'receipt.json')}
    required.update(str(path.parent / name) for name in manifest['files'])
    required.update(str(p) for p in native.rglob('*.py'))
    recorded_native = {name for name in expected if Path(name).is_relative_to(native)}
    if recorded_native != {str(p) for p in native.rglob('*.py')} or not required <= expected.keys():
        raise AuthorityError('Incomplete accepted target/native implementation binding.')
    context = current() or VerificationContext()
    if context.files(required) != [expected[name] for name in sorted(required)]:
        raise AuthorityError('Accepted target publication or native inference implementation changed.')
    if _accepted(store, run_id, step_id)[3] != anchor:
        raise AuthorityError('Accepted target execution anchor changed during consumption.')
    key = ('target_inference', root, run_id, step_id, anchor)
    if key not in context.source_contexts:
        previous = context.validate_anchors
        def validate():
            previous()
            if _accepted(store, run_id, step_id)[3] != anchor:
                raise AuthorityError('Accepted target execution anchor changed.')
        context.validate_anchors = validate
        context.source_contexts[key] = authority.identity_sha256
    return authority.identity_sha256
