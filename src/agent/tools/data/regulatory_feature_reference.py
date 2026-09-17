"""M14.1 exact species-neutral reference; no raw processing or model claims."""
from dataclasses import asdict, dataclass, replace
import hashlib
import json

from . import scatac_reference as r
from .chromap_reference_index import verify_fasta_fai

ARTIFACT = 'agent.regulatory-feature-reference'
CONTRACT = 'regulatory-feature-reference.v1'
MAX_FEATURES = 2_000_000
CATEGORIES = ('peak_set', 'curated_ccre', 'regulatory_regions')


@dataclass(frozen=True)
class SpeciesIdentity:
    scientific_name: str
    taxonomy_id: int


def validate_species(value):
    d = asdict(value) if type(value) is SpeciesIdentity else value
    d = r._shape(d, SpeciesIdentity)
    r._text(d['scientific_name'])
    if len(d['scientific_name'].split()) < 2:
        r._fail('An explicit scientific species name is required.')
    r._positive(d['taxonomy_id'])
    return SpeciesIdentity(**d)


@dataclass(frozen=True)
class RegulatoryFeatures:
    bed: r.ResourceIdentity
    feature_count: int
    ordered_feature_sha256: str
    category: str
    coordinate_convention: str = r.COORDINATE_CONVENTION
    feature_name_convention: str = r.FEATURE_NAME_CONVENTION
    bounds_checked: bool = True


@dataclass(frozen=True)
class RegulatoryFeatureReference:
    species: SpeciesIdentity
    target_assembly: str
    genome: r.GenomeReference
    features: RegulatoryFeatures
    reference_identity_sha256: str
    assembly_binding: str = 'explicit_declaration'
    species_binding: str = 'caller-declared-scientific-name-and-NCBI-taxonomy-id'
    ordered_identity_algorithm: str = r.ORDERED_IDENTITY_ALGORITHM
    artifact_type: str = ARTIFACT
    schema_version: int = 1
    contract_version: str = CONTRACT

    def to_dict(self):
        return asdict(self)


def identity(bundle):
    value = bundle.to_dict()
    del value['reference_identity_sha256']
    for resource in (value['genome']['fasta'], value['genome']['fai'], value['features']['bed']):
        del resource['path']
        del resource['provenance']
    return hashlib.sha256(b'agent.regulatory-feature-reference.v1\0' + r._json_bytes(value)).hexdigest()


def validate_regulatory_feature_reference(value):
    try:
        d = dict(r._shape(value.to_dict() if type(value) is RegulatoryFeatureReference else value,
                          RegulatoryFeatureReference))
        if (len(r._json_bytes(d)) > r.MAX_MANIFEST_BYTES or d['artifact_type'] != ARTIFACT
                or type(d['schema_version']) is not int or d['schema_version'] != 1
                or d['contract_version'] != CONTRACT or d['assembly_binding'] != 'explicit_declaration'
                or d['species_binding'] != 'caller-declared-scientific-name-and-NCBI-taxonomy-id'
                or d['ordered_identity_algorithm'] != r.ORDERED_IDENTITY_ALGORITHM):
            r._fail('Unsupported regulatory-feature reference contract.')
        d['species'] = validate_species(d['species'])
        r._text(d['target_assembly'])
        g = dict(r._shape(d['genome'], r.GenomeReference))
        g['fasta'], g['fai'] = r._resource(g['fasta']), r._resource(g['fai'])
        r._positive(g['contig_count']); r._sha(g['ordered_contig_sha256'])
        d['genome'] = r.GenomeReference(**g)
        f = dict(r._shape(d['features'], RegulatoryFeatures))
        f['bed'] = r._resource(f['bed'])
        r._positive(f['feature_count']); r._sha(f['ordered_feature_sha256'])
        if (f['feature_count'] > MAX_FEATURES or f['category'] not in CATEGORIES
                or f['coordinate_convention'] != r.COORDINATE_CONVENTION
                or f['feature_name_convention'] != r.FEATURE_NAME_CONVENTION
                or f['bounds_checked'] is not True):
            r._fail('Unsupported feature vocabulary.')
        d['features'] = RegulatoryFeatures(**f)
        r._sha(d['reference_identity_sha256'])
        bundle = RegulatoryFeatureReference(**d)
        if identity(bundle) != bundle.reference_identity_sha256:
            r._fail('Reference identity mismatch.', 'REFERENCE_DIGEST_MISMATCH')
        return bundle
    except r.ScATACReferenceError:
        raise
    except (TypeError, ValueError, KeyError, AttributeError, UnicodeError, RecursionError) as exc:
        raise r.ScATACReferenceError('REFERENCE_CONTRACT_INVALID', 'Invalid neutral reference.') from exc


def build_regulatory_feature_reference(*, species, target_assembly, fasta_path, fai_path,
        feature_bed_path, feature_category, genome_provenance=r.SourceProvenance(),
        feature_provenance=r.SourceProvenance()):
    """Qualify exact FASTA/FAI and ordered BED; declarations do not prove taxonomy."""
    species = validate_species(species)
    r._text(target_assembly)
    if feature_category not in CATEGORIES:
        r._fail('Unsupported feature category.')
    for p in (genome_provenance, feature_provenance):
        if type(p) is not r.SourceProvenance:
            r._fail('Expected SourceProvenance.')
        r._provenance(asdict(p))
    paths = [r._source_path(p) for p in (fasta_path, fai_path, feature_bed_path)]
    before = [r._snapshot(p) for p in paths]
    fasta_sha = r._file_hash(paths[0], fasta=True)
    contigs, fai_sha, contig_sha = r._inspect_fai(paths[1], before[0][2])
    count, bed_sha, feature_sha = r._inspect_bed(paths[2], contigs)
    bundle = RegulatoryFeatureReference(species, target_assembly,
        r.GenomeReference(r.ResourceIdentity(str(paths[0]), fasta_sha, genome_provenance),
                          r.ResourceIdentity(str(paths[1]), fai_sha, genome_provenance), len(contigs), contig_sha),
        RegulatoryFeatures(r.ResourceIdentity(str(paths[2]), bed_sha, feature_provenance),
                           count, feature_sha, feature_category), '')
    bundle = validate_regulatory_feature_reference(replace(bundle, reference_identity_sha256=identity(bundle)))
    # Reuse the stricter streaming FASTA/FAI consistency check, without building an index.
    verify_fasta_fai(bundle)
    if before != [r._snapshot(p) for p in paths]:
        r._fail('Reference changed during qualification.', 'REFERENCE_SOURCE_CHANGED')
    return bundle


def load_regulatory_feature_reference(path, *, expected_sha256=None):
    path = r._source_path(path)
    if expected_sha256 is not None:
        r._sha(expected_sha256)
    with path.open('rb') as f:
        raw = f.read(r.MAX_MANIFEST_BYTES + 1)
    sha = hashlib.sha256(raw).hexdigest()
    if len(raw) > r.MAX_MANIFEST_BYTES or expected_sha256 is not None and sha != expected_sha256:
        r._fail('Reference manifest identity mismatch.', 'REFERENCE_DIGEST_MISMATCH')
    try:
        value = json.loads(raw.decode('utf-8'), object_pairs_hook=r._duplicate_free,
                           parse_constant=r._reject_constant)
        bundle = validate_regulatory_feature_reference(value)
        if r._json_bytes(bundle.to_dict()) != raw:
            r._fail('Noncanonical reference manifest.')
        return path, bundle, sha
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise r.ScATACReferenceError('REFERENCE_CONTRACT_INVALID', 'Invalid neutral reference JSON.') from exc


def reinspect_regulatory_feature_reference(value):
    bundle = validate_regulatory_feature_reference(value)
    current = build_regulatory_feature_reference(species=bundle.species,
        target_assembly=bundle.target_assembly, fasta_path=bundle.genome.fasta.path,
        fai_path=bundle.genome.fai.path, feature_bed_path=bundle.features.bed.path,
        feature_category=bundle.features.category, genome_provenance=bundle.genome.fasta.provenance,
        feature_provenance=bundle.features.bed.provenance)
    if current != bundle:
        r._fail('Reference resources changed.', 'REFERENCE_SOURCE_CHANGED')
    return bundle


def publish_regulatory_feature_reference(value, path):
    bundle = validate_regulatory_feature_reference(value)
    return r._publish_reference_manifest(path, overwrite=False,
        resources=[bundle.genome.fasta, bundle.genome.fai, bundle.features.bed],
        payload=r._json_bytes(bundle.to_dict()), loader=load_regulatory_feature_reference)
