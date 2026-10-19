"""Private feature binding to a library genome, plus small historical readers.

Experimental feature axes remain task-scoped. Common genome installation belongs
only to ScientificResourceLibrary; there is no promotion writer here.
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
    if (set(q) not in ({'contract', 'requirement', 'genome', 'identity_sha256', 'files', 'review', 'evidence'},
                      {'contract', 'requirement', 'genome', 'identity_sha256', 'files', 'review', 'evidence', 'session_id'})
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
    if 'session_id' in q:
        if not q['session_id'] or q['session_id'] != record.session_id or q['review'] is not None:
            refs._fail('Invalid private genome scope.')
    else:
        if record.session_id is not None: refs._fail('Deployment genome cannot carry a private Session.')
        validate_review(q['review'], q['files'])
    if set(q['evidence']) != {'record_sha256', 'manifest_sha256', 'reference_identity_sha256'}:
        refs._fail('Missing original reference qualification evidence.')
    for value in q['evidence'].values(): r._sha(value)
    from agent.tools.data import regulatory_feature_reference, chromap_reference_index
    required = {record.source_path, record.source_index['path'], regulatory_feature_reference.__file__,
                r.__file__, chromap_reference_index.__file__}
    if {f['path'] for f in q['files']} not in (required, {record.source_path, record.source_index['path']}):
        refs._fail('Incomplete genome qualification content closure.')
    for row in q['files']:
        if set(row) != {'path', 'size_bytes', 'sha256'}: refs._fail('Invalid content pin.')
        r._sha(row['sha256']); r._positive(row['size_bytes'])
    pins = {f['path']: f['sha256'] for f in q['files']}
    if pins[record.source_path] != record.source_sha256 or pins[record.source_index['path']] != record.source_index['sha256']:
        refs._fail('Genome qualification pins differ.')


def persist_private(resources, resource_id):
    """Retain an uploaded, owner-qualified genome independently of task peaks.

    Qualification does not authorize sharing. The original Session remains the
    sole access scope unless the user explicitly installs it for shared future use.
    """
    record = resources.load(resource_id)
    if record.input_type != 'regulatory_reference' or record.reference_qualification['scope'] != 'session':
        refs._fail('A private owner-qualified feature reference is required.')
    reference = refs.check_record(resources, record)
    required = dict(species=asdict(reference.species), assembly=reference.target_assembly)
    genome = asdict(reference.genome)
    from .shared_resources import redistribution_files
    files = [f for f in redistribution_files(_serialize(record.reference_qualification['files'])) if f['path'] != reference.features.bed.path]
    session = record.reference_qualification['session_id']
    q = dict(contract='qualified-genome.v1', requirement=required, genome=genome,
        identity_sha256=identity(required, genome), files=files, review=None, session_id=session,
        evidence=dict(record_sha256=record.record_sha256, manifest_sha256=record.source_sha256,
                      reference_identity_sha256=reference.reference_identity_sha256))
    return resources.register('private-genome:' + digest(q), reference.genome.fasta.path,
        input_type='genome_reference', source_index_path=reference.genome.fai.path,
        label=(required['species']['scientific_name'] + ' ' + required['assembly'] + ' — private genome')[:160],
        attribution='Owner-qualified uploaded genome; private Session reuse only.',
        reference_qualification=q, session_id=session)


def check_record(resources, record):
    if resources.load(record.resource_id) != record:
        refs._fail('Genome registration changed.')
    return check_content(record)


def check_content(record):
    """Read existing qualification pins, without reconstructing genome science."""
    from agent.tools.data.authority_context import VerificationContext
    validate_record(record)
    from .shared_resources import redistribution_files
    files = redistribution_files(_serialize(record.reference_qualification['files']))
    if VerificationContext().files(row['path'] for row in files) != files:
        refs._fail('Qualified genome bytes changed.')
    return _serialize(record.reference_qualification['genome'])


def bind(resources, record, source, session_id, origin):
    """Create a private dataset reference using shared bytes, never a shared model."""
    from agent.tools.data import regulatory_feature_reference as owner
    from agent.tools.data import scatac_reference as r, chromap_reference_index
    from agent.tools.data.authority_context import VerificationContext
    if isinstance(record, dict):
        if resources.library is None: refs._fail('Scientific resource library unavailable.')
        resource = resources.library.check(record)
        genome = resource['genome']
        requirement = {k: resource[k] for k in ('species', 'assembly')}
        record_identity, genome_id, label = record['id'], 'library-' + record['id'], record['label']
    else:
        if record.session_id != session_id:
            refs._fail('Use a library genome or this Session\'s private genome; legacy shared selection is retired.')
        genome = check_record(resources, record)
        requirement = _serialize(record.reference_qualification['requirement'])
        record_identity, genome_id, label = record.record_sha256, record.resource_id, record.label
    required = _serialize(origin.admitted['reference_requirement'])
    if {k: required[k] for k in ('species', 'assembly')} != requirement:
        refs._fail('Genome species/assembly conflicts with the task.')
    key = 'genome-feature-binding:' + digest(dict(genome=record_identity,
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
        files = VerificationContext().files([genome['fasta']['path'], genome['fai']['path'], bed])
        refs._check_owner_files(reference, files)
        # Catalog registration has its own lock; use a distinct lock above.
    return resources.register(key, manifest, input_type='regulatory_reference', label=label,
        attribution='Private task feature binding to an explicitly approved deployment genome.',
        reference_qualification=dict(requirement=required, reference_identity_sha256=reference.reference_identity_sha256,
            ordered_feature_sha256=reference.features.ordered_feature_sha256, files=files,
            session_id=session_id, origin_turn_id=origin.turn_id, source_record_sha256=source.record_sha256,
            scope='session', review=None, genome_resource_id=genome_id))
