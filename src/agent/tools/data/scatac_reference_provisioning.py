"""Explicit H5AD vocabulary -> ordinary reference provisioning, never matrix IO.

The caller supplies source/count/order expectations. Historical origin is not
inferred. An exact derivation receipt is hash-bound in existing cCRE provenance;
the generic reference schema and matrix interface remain unchanged.
"""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from . import scatac_reference as ref
from .scatac_matrix_contract import canonical, absolute_path, sha, integer, fail, ScATACMatrixError
from agent.tools._cancellation import cancellation_checkpoint

PROFILE_ID = 'ordered-h5ad-feature-reference.v1'
MAX_FEATURES = 2_000_000
MAX_RECEIPT_BYTES = 16384


def _hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for data in iter(lambda: f.read(8 * 1024 * 1024), b''):
            cancellation_checkpoint(); h.update(data)
    return h.hexdigest()


def _features(source, contigs):
    import h5py
    seen = set()
    with h5py.File(source, 'r') as f:
        # External/virtual index data are not covered by the source file hash.
        if not isinstance(f.get('var', getlink=True), h5py.HardLink):
            fail('REFERENCE_VOCABULARY_INVALID')
        var = f['var']
        key = var.attrs.get('_index')
        if not isinstance(key, str) or '/' in key or key not in var:
            fail('REFERENCE_VOCABULARY_INVALID')
        if not isinstance(var.get(key, getlink=True), h5py.HardLink):
            fail('REFERENCE_VOCABULARY_INVALID')
        index = var[key]
        if (not isinstance(index, h5py.Dataset) or index.ndim != 1
                or not 0 < len(index) <= MAX_FEATURES or index.is_virtual or index.external
                or h5py.check_string_dtype(index.dtype) is None):
            fail('REFERENCE_VOCABULARY_INVALID')
        for offset in range(0, len(index), 4096):
            cancellation_checkpoint()
            for name in index.asstr()[offset:offset + 4096]:
                if len(name.encode('utf-8')) > ref.MAX_LINE_BYTES:
                    fail('REFERENCE_VOCABULARY_INVALID')
                match = re.fullmatch(r'([^:\s]+):(0|[1-9][0-9]*)-(0|[1-9][0-9]*)', name)
                if match is None:
                    fail('REFERENCE_VOCABULARY_INVALID')
                chrom, s, e = match.groups()
                start, end = int(s), int(e)
                if (name in seen or chrom not in contigs or not 0 <= start < end <= contigs[chrom]
                        or name != f'{chrom}:{start}-{end}'):
                    fail('REFERENCE_VOCABULARY_INVALID')
                seen.add(name)
                yield name, f'{chrom}\t{start}\t{end}\n'.encode('utf-8')


def _derive(source, contigs, output=None):
    ordered = hashlib.sha256(); bed = hashlib.sha256(); count = 0
    for name, line in _features(source, contigs):
        count += 1
        ordered.update((name + '\n').encode('utf-8')); bed.update(line)
        if output is not None:
            output.write(line)
    return count, ordered.hexdigest(), bed.hexdigest()


def _fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def provision_h5ad_reference(*, source_h5ad_path, source_h5ad_sha256,
        expected_feature_count, expected_ordered_feature_sha256, species, assembly,
        fasta_path, fai_path, output_dir, authority):
    """Publish a new resource directory; no overwrite, model data or X access.

    Real and tiny explicit references use the same function. This helper does
    not certify project-canonical identity merely from species or dimensions.
    """
    sha(source_h5ad_sha256); sha(expected_ordered_feature_sha256)
    integer(expected_feature_count, 1, MAX_FEATURES)
    if (species, assembly) not in (('human', 'hg38'), ('mouse', 'mm10')):
        fail('REFERENCE_VOCABULARY_INVALID')
    if type(authority) is not str or not authority.strip() or len(authority) > 512:
        fail('REFERENCE_AUTHORITY_REQUIRED')
    source, fasta, fai = (ref._source_path(p) for p in (source_h5ad_path, fasta_path, fai_path))
    destination = Path(output_dir).absolute()
    absolute_path(str(destination))
    if destination.exists() or destination.is_symlink():
        fail('REFERENCE_OUTPUT_CONFLICT')
    before = [ref._snapshot(p) for p in (source, fasta, fai)]
    if _hash(source) != source_h5ad_sha256:
        fail('REFERENCE_SOURCE_MISMATCH')
    contigs, fai_sha, _ = ref._inspect_fai(fai, before[1][2])
    with tempfile.TemporaryDirectory(prefix='.reference-provision-', dir=destination.parent) as private:
        stage = Path(private) / 'publication'; stage.mkdir()
        with (stage / 'ccre.bed').open('xb') as f:
            count, ordered, bed_sha = _derive(source, contigs, f)
            f.flush(); os.fsync(f.fileno())
        if (count, ordered) != (expected_feature_count, expected_ordered_feature_sha256):
            fail('REFERENCE_EXPECTATION_MISMATCH')
        receipt = dict(artifact_type='agent.ordered-h5ad-reference-derivation', schema_version=1,
            profile_id=PROFILE_ID, authority=authority, species=species, assembly=assembly,
            source_h5ad_path=str(source), source_h5ad_sha256=source_h5ad_sha256,
            feature_count=count, ordered_feature_sha256=ordered, fai_path=str(fai), fai_sha256=fai_sha,
            bed_path=str(destination / 'ccre.bed'), bed_sha256=bed_sha)
        raw = canonical(receipt); receipt_sha = hashlib.sha256(raw).hexdigest()
        if len(raw) > MAX_RECEIPT_BYTES:
            fail('REFERENCE_RECEIPT_INVALID')
        provenance = ref.SourceProvenance(basis='caller_supplied',
            source=str(destination / 'derivation.json'), accession=receipt_sha,
            citation='Project-authoritative ordered H5AD vocabulary; deterministic derivation, not historical BED recovery.')
        bundle = ref.build_scatac_reference_bundle(species=species, target_assembly=assembly,
            fasta_path=fasta, fai_path=fai, ccre_bed_path=stage / 'ccre.bed', ccre_provenance=provenance,
            expected_feature_count=count, expected_ordered_feature_sha256=ordered)
        if bundle.ccre.bed.sha256 != bed_sha:
            fail('REFERENCE_SOURCE_MISMATCH')
        # Paths do not enter the existing portable reference identity.
        bundle = replace(bundle, ccre=replace(bundle.ccre,
            bed=replace(bundle.ccre.bed, path=str(destination / 'ccre.bed'))))
        reference_raw = ref.canonical_reference_bundle_bytes(bundle)
        for name, data in (('derivation.json', raw), ('reference.json', reference_raw)):
            with (stage / name).open('xb') as f:
                f.write(data); f.flush(); os.fsync(f.fileno())
        if before != [ref._snapshot(p) for p in (source, fasta, fai)] or _hash(source) != source_h5ad_sha256:
            fail('REFERENCE_SOURCE_CHANGED')
        cancellation_checkpoint(); _fsync_directory(stage)
        # Trusted caller-owned directory; existing publications are never replaced.
        if destination.exists() or destination.is_symlink():
            fail('REFERENCE_OUTPUT_CONFLICT')
        os.rename(stage, destination); _fsync_directory(destination.parent)
    return dict(manifest_path=str(destination / 'reference.json'),
                manifest_sha256=hashlib.sha256(reference_raw).hexdigest(),
                reference_identity_sha256=bundle.reference_identity_sha256,
                derivation_sha256=receipt_sha)


def verify_h5ad_reference_derivation(manifest_path, *, expected_sha256):
    """Recheck explicit provenance receipt and source-to-BED transformation.

    Generic M11.1 reinspection does not authenticate caller provenance. Invoke
    this provisioning-specific check when claiming verified derivation history.
    """
    _, bundle, _ = ref.load_scatac_reference_bundle(manifest_path, expected_sha256=expected_sha256)
    resources = [Path(manifest_path), Path(bundle.genome.fasta.path), Path(bundle.genome.fai.path),
                 Path(bundle.ccre.bed.path)]
    resource_snapshots = [ref._snapshot(p) for p in resources]
    ref.reinspect_scatac_reference_bundle_sources(bundle)
    provenance = bundle.ccre.bed.provenance
    if provenance.basis != 'caller_supplied' or provenance.source is None:
        fail('REFERENCE_RECEIPT_INVALID')
    receipt_path = ref._source_path(provenance.source)
    initial = ref._snapshot(receipt_path)
    with receipt_path.open('rb') as f:
        raw = f.read(MAX_RECEIPT_BYTES + 1)
    if len(raw) > MAX_RECEIPT_BYTES or hashlib.sha256(raw).hexdigest() != provenance.accession:
        fail('REFERENCE_RECEIPT_INVALID')
    try:
        value = json.loads(raw.decode('utf-8'), object_pairs_hook=ref._duplicate_free,
                           parse_constant=ref._reject_constant)
        keys = ('artifact_type', 'schema_version', 'profile_id', 'authority', 'species', 'assembly',
                'source_h5ad_path', 'source_h5ad_sha256', 'feature_count', 'ordered_feature_sha256',
                'fai_path', 'fai_sha256', 'bed_path', 'bed_sha256')
        if (type(value) is not dict or set(value) != set(keys) or canonical(value) != raw
                or value['artifact_type'] != 'agent.ordered-h5ad-reference-derivation'
                or type(value['schema_version']) is not int or value['schema_version'] != 1
                or value['profile_id'] != PROFILE_ID or type(value['authority']) is not str
                or not value['authority'].strip() or len(value['authority']) > 512):
            fail('REFERENCE_RECEIPT_INVALID')
        integer(value['feature_count'], 1, MAX_FEATURES)
        for key in ('source_h5ad_sha256', 'ordered_feature_sha256', 'fai_sha256', 'bed_sha256'):
            sha(value[key])
        for key in ('source_h5ad_path', 'fai_path', 'bed_path'):
            absolute_path(value[key])
        source = ref._source_path(value['source_h5ad_path']); sha(value['source_h5ad_sha256'])
        before = ref._snapshot(source)
        if _hash(source) != value['source_h5ad_sha256']:
            fail('REFERENCE_SOURCE_MISMATCH')
        fai = Path(bundle.genome.fai.path)
        contigs, fai_sha, _ = ref._inspect_fai(fai, Path(bundle.genome.fasta.path).stat().st_size)
        count, ordered, bed_sha = _derive(source, contigs)
        actual = (bundle.species, bundle.target_assembly, str(fai), fai_sha, bundle.ccre.bed.path,
                  bundle.ccre.bed.sha256, bundle.ccre.feature_count, bundle.ccre.ordered_feature_sha256)
        declared = tuple(value[k] for k in ('species', 'assembly', 'fai_path', 'fai_sha256',
                                            'bed_path', 'bed_sha256', 'feature_count', 'ordered_feature_sha256'))
        if (actual != declared or (count, ordered, bed_sha) !=
                (bundle.ccre.feature_count, bundle.ccre.ordered_feature_sha256, bundle.ccre.bed.sha256)
                or ref._snapshot(source) != before or ref._snapshot(receipt_path) != initial
                or resource_snapshots != [ref._snapshot(p) for p in resources]):
            fail('REFERENCE_SOURCE_MISMATCH')
        return bundle
    except (KeyError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise ScATACMatrixError('REFERENCE_RECEIPT_INVALID') from exc
