"""Read-only accepted legacy fragments and bounded file identity helpers."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from . import scatac_fragments_v2 as m
from ._fragments_common import FragmentsError, snapshot
from agent.tools._cancellation import cancellation_checkpoint




def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            cancellation_checkpoint()
            digest.update(chunk)
    return digest.hexdigest()


def take_snapshots(paths):
    result = []
    for item in sorted(set(map(str, paths))):
        path = Path(item)
        if not path.is_absolute() or path != path.resolve():
            m.fail('FRAGMENTS_V2_SOURCE_CHANGED')
        result.append((item, snapshot(path)))
    return tuple(result)


def check_snapshots(before):
    try:
        if take_snapshots(p for p, _ in before) != before:
            m.fail('FRAGMENTS_V2_SOURCE_CHANGED')
    except (OSError, FragmentsError, ValueError):
        m.fail('FRAGMENTS_V2_SOURCE_CHANGED')


@dataclass(frozen=True)
class FragmentVerification:
    """Explicit verification scopes, with an immutable manifest snapshot."""
    manifest_path: str
    manifest_sha256: str
    manifest_bytes: bytes
    contigs: tuple[tuple[str, int], ...]
    snapshots: tuple
    artifact_content: str = 'verified'
    bound_resource_identities: str = 'verified'
    producer_history: str = 'not_verified'
    producer_profile_qualification: str = 'not_established'

    @property
    def manifest(self):
        return json.loads(self.manifest_bytes)

    def check_unchanged(self):
        check_snapshots(self.snapshots)


def verify_fragments_v2(manifest_path, *, expected_sha256, runtime=None):
    from .raw_history import accepted
    from ._fragment_reference import load
    value, _ = accepted('generic_fragments', manifest_path, expected_sha256)
    ref = value['reference']; bundle = load(ref['manifest_path'], ref['manifest_sha256'])
    with open(bundle.genome.fai.path) as stream:
        contigs = tuple((fields[0], int(fields[1])) for fields in (line.split('\t') for line in stream))
    context = __import__('agent.tools.data.authority_context', fromlist=['current']).current()
    description, _ = context.describe('generic_fragments', Path(manifest_path), expected_sha256)
    return FragmentVerification(str(manifest_path), expected_sha256, Path(manifest_path).read_bytes(), contigs,
        take_snapshots(f['path'] for f in description['files']))
