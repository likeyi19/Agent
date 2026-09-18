"""Qualified normalized transcript-TSS resources for the shared QC owner.

Raw annotation interpretation/completeness is independently operator-qualified.
This boundary validates canonical TSS/lineage and reference/scope compatibility;
it is deliberately not a universal annotation parser or a second QC engine.
"""
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile

from . import scatac_qc_reference as legacy, regulatory_feature_reference as ref
from . import primary_contigs as scope
from .scatac_qc_profile import PROFILE_SHA256, canonical, fail, tss_windows
from .scatac_fragments_v2_verifier import take_snapshots, check_snapshots

CONTRACT = 'scatac-qc-reference.neutral.v1'
NORMALIZATION = 'qualified-protein-coding-transcript-tss-bed4.v1'
NORMALIZATION_SHA256 = hashlib.sha256(NORMALIZATION.encode()).hexdigest()
MAX_BYTES = scope.MAX_BYTES


@dataclass(frozen=True)
class NeutralQCReference:
    species: ref.SpeciesIdentity
    assembly: str
    parent_manifest: legacy.QCResource
    parent_reference_identity_sha256: str
    primary_scope: legacy.QCResource
    primary_scope_identity_sha256: str
    qualification_evidence: legacy.QCResource
    annotation: legacy.QCAnnotationSource
    tss: legacy.QCResource
    lineage: legacy.QCResource
    contigs: tuple[legacy.QCContig, ...]
    ordered_contig_sha256: str
    tss_rows: int
    ordered_tss_sha256: str
    # Complete immutable closure captured by resource qualification, not discovery.
    dependencies: tuple[legacy.QCResource, ...]
    resource_identity_sha256: str
    scientific_profile_sha256: str = PROFILE_SHA256
    construction_profile: str = legacy.CONSTRUCTION_PROFILE
    contract_version: str = CONTRACT

    def to_dict(self):
        return asdict(self)


def identity(value):
    return hashlib.sha256(CONTRACT.encode() + b'\0' + canonical(
        {k: v for k, v in value.items() if k != 'resource_identity_sha256'})).hexdigest()


def validate(value):
    d = dict(value)
    if set(d) != {f.name for f in fields(NeutralQCReference)}:
        fail('QC_CONTRACT_INVALID')
    if (d['contract_version'] != CONTRACT or d['scientific_profile_sha256'] != PROFILE_SHA256
            or d['construction_profile'] != legacy.CONSTRUCTION_PROFILE
            or len(canonical(d)) > MAX_BYTES):
        fail('QC_CONTRACT_INVALID')
    if identity(d) != d['resource_identity_sha256']:
        fail('QC_IDENTITY_MISMATCH')
    d['species'] = ref.validate_species(d['species'])
    legacy._text(d['assembly'])
    for k in ('resource_identity_sha256', 'parent_reference_identity_sha256',
              'primary_scope_identity_sha256', 'ordered_contig_sha256', 'ordered_tss_sha256'):
        legacy._sha(d[k])
    for k in ('parent_manifest', 'primary_scope', 'qualification_evidence', 'tss', 'lineage'):
        d[k] = legacy._resource(d[k])
    a = legacy._shape(d['annotation'], legacy.QCAnnotationSource)
    if (a['format'] != 'externally_qualified_annotation'
            or a['parser_profile'] != NORMALIZATION or a['parser_identity_sha256'] != NORMALIZATION_SHA256
            or a['qualification'] not in ('synthetic_only', 'source_declared_not_production_qualified')):
        fail('QC_ANNOTATION_PARSER_UNQUALIFIED')
    legacy._text(a['source']); legacy._text(a['release'])
    a['resource'] = legacy._resource(a['resource'], max_bytes=legacy.MAX_ANNOTATION_BYTES)
    d['annotation'] = legacy.QCAnnotationSource(**a)
    if type(d['contigs']) is not list or not 1 <= len(d['contigs']) <= 100_000:
        fail('QC_CONTIG_INVALID')
    cs = tuple(legacy.QCContig(**legacy._shape(c, legacy.QCContig)) for c in d['contigs'])
    for c in cs:
        legacy._text(c.name); legacy._integer(c.length, 1, 2**63-1)
        if c.classification not in ('primary_nuclear_qc', 'mitochondrial', 'other'):
            fail('QC_CONTIG_INVALID')
    if len({c.name for c in cs}) != len(cs) or not any(c.classification == 'primary_nuclear_qc' for c in cs):
        fail('QC_CONTIG_INVALID')
    if hashlib.sha256(''.join(f'{c.name}\t{c.length}\n' for c in cs).encode()).hexdigest() != d['ordered_contig_sha256']:
        fail('QC_CONTIG_INVALID')
    d['contigs'] = cs
    legacy._integer(d['tss_rows'], 1, legacy.MAX_TSS_ROWS)
    if type(d['dependencies']) is not list or not 1 <= len(d['dependencies']) <= 16:
        fail('QC_RESOURCE_INVALID')
    d['dependencies'] = tuple(legacy._resource(r, max_bytes=64 * 1024**3) for r in d['dependencies'])
    return NeutralQCReference(**d)


def resources(bundle):
    return (bundle.parent_manifest, bundle.primary_scope, bundle.qualification_evidence,
            bundle.annotation.resource, bundle.tss, bundle.lineage, *bundle.dependencies)


def verify_integrity(bundle):
    """Recheck qualified bytes only; no transcript extraction or primary inference."""
    from .scatac_fragments_v2_verifier import file_sha256
    before = take_snapshots(r.path for r in resources(bundle))
    for r in resources(bundle):
        if Path(r.path).stat().st_size != r.size_bytes or file_sha256(r.path) != r.sha256:
            fail('QC_RESOURCE_MISMATCH')
    _, parent, _ = ref.load_regulatory_feature_reference(bundle.parent_manifest.path,
        expected_sha256=bundle.parent_manifest.sha256)
    scoped = scope.read(bundle.primary_scope.path, bundle.primary_scope.sha256)
    if (parent.species != bundle.species or parent.target_assembly != bundle.assembly
            or parent.reference_identity_sha256 != bundle.parent_reference_identity_sha256
            or parent.genome.ordered_contig_sha256 != bundle.ordered_contig_sha256
            or scoped['identity_sha256'] != bundle.primary_scope_identity_sha256
            or scoped['contract_version'] != scope.PROFILE
            or scoped['profile_sha256'] != scope.PROFILE_SHA256
            or scope.digest({k: v for k, v in scoped.items() if k != 'identity_sha256'}) != scoped['identity_sha256']):
        fail('QC_PARENT_MISMATCH')
    expected = tuple(legacy.QCContig(c['name'], c['length'],
        'primary_nuclear_qc' if c['classification'] == 'primary_nuclear' else c['classification']) for c in scoped['contigs'])
    if expected != bundle.contigs: fail('QC_PRIMARY_SCOPE_MISMATCH')
    _, scope_parent, _ = ref.load_regulatory_feature_reference(scoped['reference']['path'],
        expected_sha256=scoped['reference']['sha256'])
    if (parent.species != scope_parent.species or parent.target_assembly != scope_parent.target_assembly
            or parent.genome != scope_parent.genome): fail('QC_PARENT_MISMATCH')
    from ._external_fragment_io import resource
    required = [scoped['reference']['path'], scoped['classification_source']['path']]
    for p in (parent, scope_parent):
        required.extend((p.genome.fasta.path, p.genome.fai.path, p.features.bed.path))
    if tuple(legacy.QCResource(**resource(p)) for p in sorted(set(required))) != bundle.dependencies:
        fail('QC_RESOURCE_MISMATCH')
    check_snapshots(before)
    return bundle


def verify_normalized(bundle):
    """Qualification-time exact normalized site/lineage validation, bounded on disk.

    Eligibility/completeness against the raw annotation must be established in
    independent qualification evidence before catalog admission. Hashes alone do
    not establish that biological assertion.
    """
    ranks = {c.name: (i, c) for i, c in enumerate(bundle.contigs)}
    with tempfile.TemporaryDirectory(prefix='qc-normalized-') as temporary:
        with sqlite3.connect(str(Path(temporary) / 'lineage.sqlite')) as db:
            db.execute('PRAGMA cache_size=-8192'); db.execute('PRAGMA temp_store=FILE')
            db.execute('CREATE TABLE tx (rank INTEGER, chrom TEXT, pos INTEGER, strand TEXT, gene TEXT, transcript TEXT UNIQUE)')
            previous = None
            try:
                for n, raw in enumerate(legacy._lines(bundle.lineage.path), 1):
                    if n > legacy.MAX_TRANSCRIPTS: fail('QC_RESOURCE_LIMIT')
                    row = raw.decode('utf-8').removesuffix('\n').split('\t')
                    if len(row) != 5: fail('QC_ROW_INVALID')
                    chrom, p, strand, gene, tx = row
                    for text in row: legacy._text(text)
                    position = legacy._decimal(p)
                    if chrom not in ranks or strand not in ('+', '-'): fail('QC_TSS_INVALID')
                    rank, c = ranks[chrom]
                    if c.classification != 'primary_nuclear_qc': fail('QC_TSS_SCOPE_INVALID')
                    tss_windows(position, c.length)
                    key = (rank, position, strand, gene, tx)
                    if previous is not None and key <= previous: fail('QC_TSS_ORDER_INVALID')
                    previous = key
                    db.execute('INSERT INTO tx VALUES (?,?,?,?,?,?)', (rank, chrom, position, strand, gene, tx))
            except sqlite3.IntegrityError:
                fail('QC_GENE_IDENTITY_INVALID')
            if db.execute('SELECT gene FROM tx GROUP BY gene HAVING count(DISTINCT chrom)>1 OR count(DISTINCT strand)>1 LIMIT 1').fetchone():
                fail('QC_GENE_IDENTITY_INVALID')
            observed = iter(legacy._lines(bundle.tss.path)); ordered = hashlib.sha256(legacy.TSS_DOMAIN); count = 0
            for chrom, p, strand in db.execute('SELECT chrom,pos,strand FROM tx GROUP BY rank,chrom,pos,strand ORDER BY rank,pos,strand'):
                raw = f'{chrom}\t{p}\t{p+1}\t{strand}\n'.encode()
                if next(observed, None) != raw: fail('QC_DERIVATION_MISMATCH')
                count += 1; ordered.update(raw)
            if next(observed, None) is not None or count != bundle.tss_rows or ordered.hexdigest() != bundle.ordered_tss_sha256:
                fail('QC_DERIVATION_MISMATCH')


def qualify_reference(*, reference_path, reference_sha256, primary_scope_path,
        primary_scope_sha256, annotation_path, annotation_source, annotation_release,
        qualification_evidence_path, tss_path, lineage_path, output_path, synthetic=False):
    """Publish validated normalized resources; production still requires catalog review.

    All input resources already exist and remain immutable. This does not create
    raw-annotation provenance or auto-register its own qualification catalog entry.
    """
    if type(synthetic) is not bool: fail('QC_CONTRACT_INVALID')
    _, parent, _ = ref.load_regulatory_feature_reference(reference_path, expected_sha256=reference_sha256)
    scoped = scope.verify_scope(primary_scope_path, primary_scope_sha256,
        reference_path=reference_path, reference_sha256=reference_sha256)
    _, scope_parent, _ = ref.load_regulatory_feature_reference(scoped['reference']['path'],
        expected_sha256=scoped['reference']['sha256'])
    dep_paths = [parent.genome.fasta.path, parent.genome.fai.path, parent.features.bed.path,
                 scoped['reference']['path'], scoped['classification_source']['path'],
                 scope_parent.genome.fasta.path, scope_parent.genome.fai.path, scope_parent.features.bed.path]
    from ._external_fragment_io import resource
    dependencies = tuple(legacy.QCResource(**resource(p)) for p in sorted(set(dep_paths)))
    contigs = tuple(legacy.QCContig(c['name'], c['length'],
        'primary_nuclear_qc' if c['classification'] == 'primary_nuclear' else c['classification']) for c in scoped['contigs'])
    tss_resource = legacy._file(tss_path)
    ordered = hashlib.sha256(legacy.TSS_DOMAIN); n = 0
    for raw in legacy._lines(tss_resource.path):
        ordered.update(raw); n += 1
        if n > legacy.MAX_TSS_ROWS: fail('QC_RESOURCE_LIMIT')
    annotation = legacy.QCAnnotationSource(legacy._file(annotation_path, max_bytes=legacy.MAX_ANNOTATION_BYTES),
        'externally_qualified_annotation', annotation_source, annotation_release,
        'synthetic_only' if synthetic else 'source_declared_not_production_qualified', NORMALIZATION, NORMALIZATION_SHA256)
    bundle = NeutralQCReference(parent.species, parent.target_assembly, legacy._file(reference_path),
        parent.reference_identity_sha256, legacy._file(primary_scope_path), scoped['identity_sha256'],
        legacy._file(qualification_evidence_path), annotation, tss_resource, legacy._file(lineage_path),
        contigs, parent.genome.ordered_contig_sha256, n, ordered.hexdigest(), dependencies, '')
    d = bundle.to_dict(); d['resource_identity_sha256'] = identity(d)
    # JSON conversion makes the serialized shape authoritative, including arrays.
    bundle = validate(json.loads(canonical(d)))
    if bundle.parent_manifest.sha256 != reference_sha256 or bundle.primary_scope.sha256 != primary_scope_sha256:
        fail('QC_SOURCE_CHANGED')
    before = take_snapshots(r.path for r in resources(bundle))
    ref.reinspect_regulatory_feature_reference(parent)
    verify_integrity(bundle); verify_normalized(bundle); check_snapshots(before)
    destination = Path(output_path); legacy.parent._path_text(str(destination))
    raw = canonical(bundle.to_dict())
    fd, temporary = tempfile.mkstemp(prefix='.qc-neutral-', dir=destination.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        check_snapshots(before)
        os.link(temporary, destination); legacy._fsync_dir(destination.parent)
    finally:
        os.unlink(temporary)
    return dict(manifest_path=str(destination), manifest_sha256=hashlib.sha256(raw).hexdigest())


def check_fragment_scope(bundle, manifest):
    """Compare exact owner-bound scope pointers after producer qualification."""
    from .external_fragment_manifest import load_adoption_record
    from .bam_fragment_manifest import load_record
    from .primary_fragment_preparation import load as load_preparation
    from .neutral_bam import load as load_bam_inputs
    for library in manifest['libraries']:
        p = library['provenance']; record = p['producer_record']
        if p['kind'] == 'external_fragment_adoption':
            record = load_adoption_record(record['path'], record['sha256'])
            pointer = record.get('primary_preparation')
            if pointer is None: fail('QC_PRIMARY_SCOPE_REQUIRED')
            pointer = load_preparation(pointer['path'], pointer['sha256'])['scope']
        elif p['kind'] == 'bam_fragment_production':
            record = load_record(record['path'], record['sha256'])
            pointer = record.get('input_binding')
            if pointer is None: fail('QC_PRIMARY_SCOPE_REQUIRED')
            pointer = load_bam_inputs(pointer['path'], pointer['sha256'])['primary_scope']
        else:
            fail('QC_PRIMARY_SCOPE_REQUIRED')
        if pointer != asdict(bundle.primary_scope): fail('QC_PRIMARY_SCOPE_MISMATCH')
