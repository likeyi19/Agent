"""Explicit genome-only deployment approval in the existing local catalog.

Old combined reference approvals are evidence, never implicit genome approvals.
Feature binding still belongs to the unchanged M14.1 scientific owner.
"""
from dataclasses import asdict
from pathlib import Path

from agent.schemas.orchestration import _serialize
from .session_state import digest
from . import reference_supplementation as refs


def identity(requirement, genome):
    genome = _serialize(genome)
    for key in ('fasta', 'fai'):
        genome[key].pop('path')
    return digest(dict(contract='qualified-genome.v1', requirement=requirement, genome=genome))


def validate_record(record):
    from agent.tools.data import scatac_reference as r
    from agent.tools.data.regulatory_feature_reference import validate_species
    q = _serialize(record.reference_qualification)
    if (set(q) != {'contract', 'requirement', 'genome', 'identity_sha256', 'files', 'review', 'evidence'}
            or q['contract'] != 'qualified-genome.v1' or record.source_index is None):
        refs._fail('Invalid genome qualification.')
    req = q['requirement']
    if set(req) != {'species', 'assembly'}:
        refs._fail('Invalid genome declarations.')
    validate_species(req['species']); r._text(req['assembly'])
    g = r._shape(q['genome'], r.GenomeReference)
    for key in ('fasta', 'fai'):
        r._resource(g[key])
    r._positive(g['contig_count']); r._sha(g['ordered_contig_sha256'])
    if (g['fasta']['path'], g['fasta']['sha256']) != (record.source_path, record.source_sha256):
        refs._fail('Genome source binding differs.')
    if (g['fai']['path'], g['fai']['sha256']) != (record.source_index['path'], record.source_index['sha256']):
        refs._fail('Genome index binding differs.')
    if identity(req, g) != q['identity_sha256']:
        refs._fail('Genome identity differs.')
    from .shared_resources import validate_review
    validate_review(q['review'], q['files'])
    if set(q['evidence']) != {'record_sha256', 'manifest_sha256', 'reference_identity_sha256'}:
        refs._fail('Missing original reference qualification evidence.')
    for value in q['evidence'].values(): r._sha(value)
    from agent.tools.data import regulatory_feature_reference, chromap_reference_index
    required = {record.source_path, record.source_index['path'], regulatory_feature_reference.__file__,
                r.__file__, chromap_reference_index.__file__}
    if len(q['files']) != len(required) or {f['path'] for f in q['files']} != required:
        refs._fail('Incomplete genome qualification content closure.')
    for row in q['files']:
        if set(row) != {'path', 'size_bytes', 'sha256'}: refs._fail('Invalid content pin.')
        r._sha(row['sha256']); r._positive(row['size_bytes'])
    pins = {f['path']: f['sha256'] for f in q['files']}
    if pins[record.source_path] != record.source_sha256 or pins[record.source_index['path']] != record.source_index['sha256']:
        refs._fail('Genome qualification pins differ.')


def promote(resources, resource_id, *, operator, provenance, reuse_basis, redistribution=None):
    """Explicit new approval from a verified combined reference; no file copy."""
    record = resources.load(resource_id)
    if record.input_type != 'regulatory_reference':
        refs._fail('Choose an existing owner-qualified feature reference as evidence.')
    reference = refs.check_record(resources, record)
    required = dict(species=asdict(reference.species), assembly=reference.target_assembly)
    genome = asdict(reference.genome)
    files = [f for f in _serialize(record.reference_qualification['files']) if f['path'] != reference.features.bed.path]
    from .shared_resources import review
    q = dict(contract='qualified-genome.v1', requirement=required, genome=genome,
        identity_sha256=identity(required, genome), files=files,
        review=review(operator, provenance, reuse_basis, redistribution, files),
        evidence=dict(record_sha256=record.record_sha256, manifest_sha256=record.source_sha256,
                      reference_identity_sha256=reference.reference_identity_sha256))
    return resources.register('deployment-genome:' + digest(q), reference.genome.fasta.path,
        input_type='genome_reference', source_index_path=reference.genome.fai.path,
        label=(required['species']['scientific_name'] + ' ' + required['assembly'] + ' — genome')[:160],
        attribution='Explicit operator-approved genome reuse; dataset vocabulary and models are not shared.',
        reference_qualification=q)


def check_record(resources, record):
    from agent.tools.data.authority_context import VerificationContext
    validate_record(record)
    if resources.load(record.resource_id) != record:
        refs._fail('Genome registration changed.')
    files = _serialize(record.reference_qualification['files'])
    if VerificationContext().files(row['path'] for row in files) != files:
        refs._fail('Qualified genome bytes or qualification implementation changed.')
    from .shared_resources import check_review
    check_review(_serialize(record.reference_qualification['review']))
    return _serialize(record.reference_qualification['genome'])


def bind(resources, record, source, session_id, origin):
    """Create a private dataset reference using shared bytes, never a shared model."""
    from agent.tools.data import regulatory_feature_reference as owner
    from agent.tools.data import scatac_reference as r, chromap_reference_index
    from agent.tools.data.authority_context import VerificationContext
    genome = check_record(resources, record)
    required = _serialize(origin.admitted['reference_requirement'])
    if {k: required[k] for k in ('species', 'assembly')} != _serialize(record.reference_qualification['requirement']):
        refs._fail('Genome species/assembly conflicts with the task.')
    key = 'genome-feature-binding:' + digest(dict(genome=record.record_sha256,
        source=source.record_sha256, session=session_id, origin=origin.turn_id, requirement=required))
    with resources._lock(resources.registration_id(key)):
        from .local_resources import ResourceAdmissionError
        try:
            existing = resources.load(resources.registration_id(key))
        except ResourceAdmissionError as exc:
            if exc.code != 'LOCAL_RESOURCE_UNAVAILABLE': raise
        else:
            refs.check_record(resources, existing)
            return existing
        # Reuse the managed upload root so the existing source policy applies.
        root = resources._workspace._ensure_directory(resources._workspace.root / 'uploads' / 'qualified_references')
        directory = resources._workspace._ensure_directory(root / digest(key))
        bed = directory / 'features.bed'
        names = refs._feature_names(source.source_path)
        if not bed.exists(): refs._write_bed(bed, names)
        reference = owner.build_regulatory_feature_reference(species=required['species'],
            target_assembly=required['assembly'], fasta_path=genome['fasta']['path'], fai_path=genome['fai']['path'],
            feature_bed_path=bed, feature_category=required['feature_category'],
            genome_provenance=r.SourceProvenance(**genome['fasta']['provenance']),
            feature_provenance=r.SourceProvenance(basis='caller_supplied', source='Exact ordered chr:start-end H5AD feature axis.'))
        if reference.features.ordered_feature_sha256 != r.ordered_identity_sha256(names):
            refs._fail('Prepared feature vocabulary differs from the captured H5AD.')
        manifest = directory / 'reference.json'
        if manifest.exists():
            if owner.load_regulatory_feature_reference(manifest)[1] != reference:
                refs._fail('Previously published feature binding changed.')
        else:
            owner.publish_regulatory_feature_reference(reference, manifest)
        files = VerificationContext().files([genome['fasta']['path'], genome['fai']['path'], bed,
                                             owner.__file__, r.__file__, chromap_reference_index.__file__])
        refs._check_owner_files(reference, files)
        # Catalog registration has its own lock; use a distinct lock above.
    return resources.register(key, manifest, input_type='regulatory_reference', label=record.label,
        attribution='Private task feature binding to an explicitly approved deployment genome.',
        reference_qualification=dict(requirement=required, reference_identity_sha256=reference.reference_identity_sha256,
            ordered_feature_sha256=reference.features.ordered_feature_sha256, files=files,
            session_id=session_id, origin_turn_id=origin.turn_id, source_record_sha256=source.record_sha256,
            scope='session', review=None, genome_resource_id=record.resource_id))
