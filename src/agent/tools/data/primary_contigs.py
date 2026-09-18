"""Explicit reference-bound primary nuclear scope; no name-based inference.

Uses the same complete ordered classification model as the legacy QC owner.
Legacy QC serialization/identities remain unchanged; neutral scope is separate.
"""
import json
from pathlib import Path

from . import regulatory_feature_reference as ref, scatac_reference as legacy
from . import scatac_fragments_v2 as v2
from ._fragments_common import canonical, digest
from ._external_fragment_io import resource
from .scatac_fragments_v2_verifier import take_snapshots, check_snapshots

PROFILE = 'primary-nuclear-contigs.v1'
PROFILE_SHA256 = digest(dict(profile=PROFILE, include='explicit canonical primary nuclear chromosomes including primary sex chromosomes',
    exclude='mitochondrial,unplaced,unlocalized,alternate,random,scaffold,other non-primary', inference='none'))
CLASSES = ('primary_nuclear', 'mitochondrial', 'other')
MAX_BYTES = 8 * 1024 * 1024


def fail():
    raise ValueError('PRIMARY_CONTIG_SCOPE_INVALID')


def read(path, sha):
    p = Path(path); before = take_snapshots([p])
    raw = p.read_bytes() if p.stat().st_size <= MAX_BYTES else b''
    if not raw or digest_bytes(raw) != sha: fail()
    value = json.loads(raw, object_pairs_hook=v2._pairs)
    if canonical(value) != raw: fail()
    check_snapshots(before)
    return value


def digest_bytes(raw):
    import hashlib
    return hashlib.sha256(raw).hexdigest()


def verify_scope(path, sha, *, reference_path=None, reference_sha256=None):
    value = read(path, sha)
    v2.shape(value, ('contract_version', 'profile_sha256', 'reference', 'classification_source',
                     'contigs', 'extra_exclusions', 'identity_sha256'))
    if value['contract_version'] != PROFILE or value['profile_sha256'] != PROFILE_SHA256: fail()
    if value['identity_sha256'] != digest({k:v for k,v in value.items() if k != 'identity_sha256'}): fail()
    pointer = value['reference']; v2.resource(pointer)
    if resource(pointer['path'], pointer['sha256']) != pointer: fail()
    _, reference, _ = ref.load_regulatory_feature_reference(pointer['path'], expected_sha256=pointer['sha256'])
    fai = resource(reference.genome.fai.path, reference.genome.fai.sha256)
    dictionary, _, _ = legacy._inspect_fai(Path(fai['path']), Path(reference.genome.fasta.path).stat().st_size)
    rows = value['contigs']
    for row in rows:
        v2.shape(row, ('name', 'length', 'classification'))
        v2.text(row['name']); v2.number(row['length'])
        if row['classification'] not in CLASSES: fail()
    if [(c['name'],c['length']) for c in rows] != list(dictionary.items()): fail()
    primary = {c['name'] for c in rows if c['classification'] == 'primary_nuclear'}
    if not primary: fail()
    # Explicit source-only exclusions permit exclusion, never coordinate aliasing.
    for name, classification in value['extra_exclusions'].items():
        v2.text(name)
        if name in dictionary or classification not in ('mitochondrial', 'other'): fail()
    v2.resource(value['classification_source'])
    if resource(value['classification_source']['path'], value['classification_source']['sha256']) != value['classification_source']: fail()
    if reference_path is not None:
        _, target, _ = ref.load_regulatory_feature_reference(reference_path, expected_sha256=reference_sha256)
        if (target.species != reference.species or target.target_assembly != reference.target_assembly
                or target.genome != reference.genome): fail()
        reference = target
    # Complete vocabulary validation; never subset/reorder/rebuild the reference.
    resource(reference.features.bed.path, reference.features.bed.sha256)
    with Path(reference.features.bed.path).open() as stream:
        for line in stream:
            if line.split('\t',1)[0] not in primary: fail()
    return value


def publish_scope(*, reference_path, reference_sha256, classifications, classification_source,
                  extra_exclusions, output_path):
    """Caller-reviewed complete FAI classifications, with hash-bound provenance.

    QC primary_nuclear_qc classifications can be supplied without changing their
    legacy record; only the neutral projection renames that class.
    """
    rows = [dict(name=c.name, length=c.length,
                 classification='primary_nuclear' if c.classification == 'primary_nuclear_qc' else c.classification)
            for c in classifications]
    value = dict(contract_version=PROFILE, profile_sha256=PROFILE_SHA256,
        reference=resource(reference_path, reference_sha256), classification_source=classification_source,
        contigs=rows, extra_exclusions=extra_exclusions)
    value['identity_sha256'] = digest(value)
    destination=Path(output_path); destination.parent.mkdir(parents=True,exist_ok=True)
    # Validate at a private file before publishing an immutable resource.
    import tempfile, os
    with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
        stage=Path(temp)/'scope.json'; stage.write_bytes(canonical(value))
        sha=digest_bytes(stage.read_bytes());verify_scope(stage,sha)
        with stage.open('rb') as stream: os.fsync(stream.fileno())
        os.link(stage,destination)  # exclusive publication, never overwrite
    return resource(destination,sha)
