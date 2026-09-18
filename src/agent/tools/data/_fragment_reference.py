"""Closed reference dispatch for external fragments; legacy producers stay fixed."""
import json
from pathlib import Path

from . import scatac_reference as legacy, scatac_fragments_v2 as v2
from . import regulatory_matrix_contract as neutral


def validate(binding):
    keys = ('manifest_path', 'manifest_sha256', 'reference_identity_sha256',
            'species', 'assembly', 'ordered_contig_sha256')
    is_neutral = type(binding.get('species')) is dict
    v2.shape(binding, keys + (('contract_version',) if is_neutral else ()))
    v2.absolute_path(binding['manifest_path'])
    for key in ('manifest_sha256', 'reference_identity_sha256', 'ordered_contig_sha256'):
        v2.sha(binding[key])
    if is_neutral:
        if binding['contract_version'] != neutral.REFERENCE_CONTRACT:
            v2.fail('FRAGMENTS_V2_REFERENCE_MISMATCH')
        neutral.validate_binding(binding['species'], binding['assembly'])
    elif (binding['species'], binding['assembly']) not in (('human', 'hg38'), ('mouse', 'mm10')):
        v2.fail('FRAGMENTS_V2_REFERENCE_MISMATCH')


def load(path, sha, *, neutral_reference=None):
    # Dispatch by a closed manifest discriminator, never by a failed legacy load.
    if neutral_reference is None:
        with Path(path).open('rb') as f:
            raw = f.read(legacy.MAX_MANIFEST_BYTES + 1)
        if len(raw) > legacy.MAX_MANIFEST_BYTES:
            v2.fail('FRAGMENTS_V2_REFERENCE_MISMATCH')
        value = json.loads(raw, object_pairs_hook=v2._pairs)
        neutral_reference = value.get('contract_version') == neutral.REFERENCE_CONTRACT
    loader = neutral.load_reference if neutral_reference else legacy.load_scatac_reference_bundle
    return loader(path, expected_sha256=sha)[1]


def reinspect(bundle):
    if isinstance(bundle, neutral._MatrixReference):
        neutral.reinspect_reference(bundle)
    else:
        legacy.reinspect_scatac_reference_bundle_sources(bundle)
