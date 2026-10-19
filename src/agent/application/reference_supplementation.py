"""Reference provisioning through existing pending turns and local registrations.

Scientific qualification is exclusively the M14 reference owner. Registrations
record its immutable publication and deployment review; they are not Run authority.
"""
from dataclasses import asdict
from pathlib import Path
import hashlib
import os
import tempfile

from agent.schemas.orchestration import _serialize
from .session_state import digest, SessionConflictError
from .input_sets import ScientificInputSet
from .turn_decisions import IntentError


class ReferenceSupplementError(IntentError):
    def __init__(self, message):
        self.code = 'REFERENCE_SUPPLEMENT_INVALID'
        self.message = message
        super().__init__('invalid_reference')


def _fail(message):
    raise ReferenceSupplementError(message)


def prepare_fai(resources, fasta, directory):
    """Existing htslib preparation, shared by H5AD and explicit BED handoff."""
    path = fasta.source_path
    index = Path(fasta.source_index['path']) if fasta.source_index else directory / 'genome.fai'
    if fasta.source_index:
        _, actual, index_size = resources._source(index)
        if (actual, index_size) != (fasta.source_index['sha256'], fasta.source_index['size_bytes']):
            _fail('Uploaded FAI bytes changed.')
    else:
        import pysam
        # htslib writes an explicit separate index; it never edits the upload.
        if not index.exists():
            descriptor, temporary_name = tempfile.mkstemp(dir=directory, prefix='.genome-', suffix='.fai.part')
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                try:
                    pysam.faidx('-o', str(temporary), path)
                except pysam.utils.SamtoolsError as exc:
                    raise ReferenceSupplementError('FASTA indexing failed. Supply a valid plain FASTA; an existing FAI is optional.') from exc
                os.replace(temporary, index)
            finally:
                temporary.unlink(missing_ok=True)
    return index


def qualify_bed(resources, source, *, parent_path=None, parent_sha256=None, category, directory,
                fasta=None, species=None, assembly=None):
    """Qualify uploaded regions with an existing genome or explicit plain FASTA.

    Uploaded intervals are parsed only by the existing reference owner. No
    sequence, index or BED copy and no deployment registration is needed.
    """
    from agent.tools.data import regulatory_feature_reference as owner, scatac_reference
    from agent.tools.data._fragment_reference import load
    resources._verify_records([source] + ([] if fasta is None else [fasta]))
    parent = load(parent_path, parent_sha256) if parent_path is not None else None
    if parent is None and fasta is None:
        _fail('A qualified genome or an explicit plain FASTA is required.')
    if parent is not None:
        species = parent.species if type(parent.species) is dict else scatac_reference.species_identity(parent.species)
        assembly = parent.target_assembly
        fasta_path, fai_path = parent.genome.fasta.path, parent.genome.fai.path
        provenance = parent.genome.fasta.provenance
    if fasta is not None:
        with open(fasta.source_path, 'rb') as stream:
            if stream.read(2) == b'\x1f\x8b':
                _fail('Supply a plain FASTA; compressed genome upload preparation is unsupported.')
        fasta_path, fai_path = fasta.source_path, prepare_fai(resources, fasta, directory)
        provenance = scatac_reference.SourceProvenance(basis='caller_supplied',
            source='Explicit task FASTA; species and assembly are declared, not independently established.')
    reference = owner.build_regulatory_feature_reference(
        species=species, target_assembly=assembly, fasta_path=fasta_path,
        fai_path=fai_path, feature_bed_path=source.source_path,
        feature_category=category, genome_provenance=provenance,
        feature_provenance=scatac_reference.SourceProvenance(basis='caller_supplied',
            source='Task-selected BED registration ' + source.resource_id + '; ' + source.record_sha256))
    if parent is not None and (reference.genome.fasta.sha256 != parent.genome.fasta.sha256
            or reference.genome.fai.sha256 != parent.genome.fai.sha256
            or reference.genome.ordered_contig_sha256 != parent.genome.ordered_contig_sha256
            or reference.genome.contig_count != parent.genome.contig_count):
        _fail('The selected reference genome changed.')
    path = directory / 'reference.json'
    if path.exists():
        _, previous, sha = owner.load_regulatory_feature_reference(path)
        if previous != reference:
            _fail('Previously qualified custom regions changed.')
        pointer = dict(manifest_path=str(path), manifest_sha256=sha)
    else:
        pointer = owner.publish_regulatory_feature_reference(reference, path)
    resources._verify_records([source] + ([] if fasta is None else [fasta]))
    return {k: pointer[k] for k in ('manifest_path', 'manifest_sha256')}


def requirement(inputs, category):
    from agent.tools.data.regulatory_feature_reference import validate_species, CATEGORIES
    if category not in CATEGORIES or category == 'curated_ccre':
        _fail('Choose peak_set or regulatory_regions; an uploaded matrix cannot establish curated-cCRE status.')
    species = asdict(validate_species(_serialize(inputs.get('species'))))
    assembly = inputs.get('assembly')
    if type(assembly) is not str or not assembly.strip() or len(assembly) > 256:
        _fail('Declare the exact species and assembly before supplying its reference.')
    if not inputs.get('input_path'):
        _fail('Select the original scientific H5AD before requesting a genome reference.')
    return dict(species=species, assembly=assembly, feature_category=category)


def validate_record(record):
    if record.input_type == 'scientific_reference':
        from .shared_resources import validate_record as validate_scientific
        return validate_scientific(record)
    if record.input_type == 'genome_reference':
        from .shared_genome import validate_record as validate_genome
        return validate_genome(record)
    q = _serialize(record.reference_qualification)
    fields = {'requirement', 'reference_identity_sha256', 'ordered_feature_sha256', 'files',
              'session_id', 'origin_turn_id', 'source_record_sha256', 'scope', 'review'}
    if (record.input_type != 'regulatory_reference' or not isinstance(q, dict) or set(q) not in (fields, fields | {'genome_resource_id'})
            or q['scope'] not in ('session', 'deployment') or not isinstance(q['files'], list)
            or not 3 <= len(q['files']) <= 12):
        _fail('Invalid qualified reference registration.')
    if 'genome_resource_id' in q:
        from .local_resources import _identifier
        import re
        if re.fullmatch(r'library-[0-9a-f]{56}', q['genome_resource_id']) is None:
            _identifier(q['genome_resource_id'])
    r = q['requirement']
    if not isinstance(r, dict) or set(r) != {'species', 'assembly', 'feature_category'}:
        _fail('Invalid reference declarations.')
    requirement(dict(**r, input_path='registered'), r['feature_category'])
    from agent.tools.data.scatac_reference import _sha, _text
    for name in ('reference_identity_sha256', 'ordered_feature_sha256', 'source_record_sha256'):
        _sha(q[name])
    for name in ('session_id', 'origin_turn_id'):
        _text(q[name])
    for entry in q['files']:
        if set(entry) != {'path', 'size_bytes', 'sha256'} or not Path(entry['path']).is_absolute():
            _fail('Invalid reference content binding.')
        _sha(entry['sha256'])
        if type(entry['size_bytes']) is not int or entry['size_bytes'] < 1:
            _fail('Invalid reference content size.')
    if q['scope'] == 'session' and q['review'] is not None:
        _fail('Private reference cannot claim deployment review.')
    if q['scope'] == 'deployment':
        from .shared_resources import validate_review
        validate_review(q['review'], q['files'])


def input_set(record):
    if isinstance(record, dict):
        return ScientificInputSet('library-' + record['id'], record['label'], dict(library_genome=record), h5ad_companion=True)
    identity = (record.reference_qualification or {}).get('genome_resource_id', record.resource_id)
    return ScientificInputSet(identity if identity.startswith('library-') else 'reference-' + identity[6:46], record.label,
        dict(reference_manifest_path=record.source_path, reference_manifest_sha256=record.source_sha256),
        h5ad_companion=True)


def _records(resources):
    # Existing registration catalog only; no parallel reference database.
    rows = {}
    for root in resources._approved:
        for record in resources.list_records(source_root=root):
            if record.input_type in ('regulatory_reference', 'genome_reference', 'scientific_reference'):
                rows[record.resource_id] = record
    return tuple(rows.values())


def choices(resources, session_id, required, *, origin_turn_id=None):
    rows = [r for r in resources.library.entries() if r['kind'] == 'genome'
            and {k: r['resource'][k] for k in ('species', 'assembly')} == {k: _serialize(required[k]) for k in ('species', 'assembly')}] if resources.library else []
    for record in _records(resources):
        if record.input_type == 'scientific_reference':
            continue
        q = record.reference_qualification
        if 'genome_resource_id' in q:
            continue  # The genome offers the choice; this is its private resolved binding.
        if record.input_type == 'genome_reference':
            if (record.session_id == session_id
                    and _serialize(q['requirement']) == {k: _serialize(required[k]) for k in ('species', 'assembly')}):
                rows.append(record)
            continue
        if (_serialize(q['requirement']) == _serialize(required) and q['scope'] == 'session'
                and q['session_id'] == session_id and q['origin_turn_id'] == origin_turn_id):
            rows.append(record)
    return tuple(rows)


def check_record(resources, record):
    """Pinned resource integrity; never repeat FASTA sequence qualification."""
    if record.input_type == 'scientific_reference':
        from .shared_resources import check_record as check_scientific
        return check_scientific(resources, record)
    if record.input_type == 'genome_reference':
        from .shared_genome import check_record as check_genome
        return check_genome(resources, record)
    from agent.tools.data.authority_context import VerificationContext
    if resources.load(record.resource_id) != record:
        _fail('The registered reference definition changed.')
    _, sha, size = resources._source(record.source_path)
    if (sha, size) != (record.source_sha256, record.size_bytes):
        _fail('The qualified reference manifest changed.')
    from .shared_resources import redistribution_files
    expected = redistribution_files(_serialize(record.reference_qualification['files']))
    if VerificationContext().files(row['path'] for row in expected) != expected:
        _fail('The qualified reference files changed. Install a new immutable version.')
    from agent.tools.data.regulatory_feature_reference import load_regulatory_feature_reference
    _, reference, _ = load_regulatory_feature_reference(record.source_path, expected_sha256=record.source_sha256)
    _check_owner_files(reference, expected)
    q = record.reference_qualification
    if (dict(species=asdict(reference.species), assembly=reference.target_assembly,
             feature_category=reference.features.category) != _serialize(q['requirement'])
            or reference.reference_identity_sha256 != q['reference_identity_sha256']
            or reference.features.ordered_feature_sha256 != q['ordered_feature_sha256']):
        _fail('Reference qualification identity differs.')
    return reference


def _check_owner_files(reference, files):
    by_path = {row['path']: row['sha256'] for row in files}
    for resource in (reference.genome.fasta, reference.genome.fai, reference.features.bed):
        if by_path.get(resource.path) != resource.sha256:
            _fail('Reference content changed after qualification.')


def pending(service, session_id, expected_generation=None, origin_turn_id=None):
    state = service._load(session_id)
    if not state.interactions:
        _fail('First request the scientific analysis that needs this reference.')
    last = state.interactions[-1]
    ref = last.prerequisite
    if (ref is None or ref['field'] != 'resources' or last.base_generation != state.generation
            or last.base_revision_id != state.active_revision_id
            or expected_generation is not None and state.generation != expected_generation
            or origin_turn_id is not None and ref['origin_turn_id'] != origin_turn_id):
        _fail('The pending reference request changed. Reopen the current scientific request.')
    origin = next(i for i in state.interactions if i.turn_id == ref['origin_turn_id'])
    required = _serialize(origin.admitted.get('reference_requirement'))
    if required is None:
        _fail('The current request is not awaiting a genome reference.')
    return state, origin, required


def public_pending(service, session_id):
    try:
        state, origin, required = pending(service, session_id)
    except ReferenceSupplementError:
        return None
    return dict(expected_generation=state.generation, origin_turn_id=origin.turn_id,
        species=required['species']['scientific_name'], assembly=required['assembly'],
        choices=[dict(handle='resource.' + input_set(r).input_set_id, label=input_set(r).label)
                 for r in choices(service.resources, session_id, required, origin_turn_id=origin.turn_id)])


def _feature_names(path):
    from agent.tools.data.scatac import read_feature_names
    return read_feature_names(path)


def _write_bed(path, names):
    from epizoo.data.ccre import parse_region
    with path.open('x') as stream:
        for name in names:
            chrom, start, end = parse_region(name)
            if name != f'{chrom}:{start}-{end}':
                _fail('Use exact chr:start-end feature identities; coordinate aliases are not inferred.')
            stream.write(f'{chrom}\t{start}\t{end}\n')


def qualify(service, uploads, session_id, *, resource_id, expected_generation, origin_turn_id):
    """Explicit supplementary FASTA, bound to the captured task, never new science intent."""
    from agent.tools.data import regulatory_feature_reference as owner
    from agent.tools.data import scatac_reference, chromap_reference_index
    from agent.tools.data.authority_context import VerificationContext
    state, origin, required = pending(service, session_id, expected_generation, origin_turn_id)
    resources = service.resources
    binding = resources._submission_binding(origin.submission)
    resources.validate_binding(binding)
    source = resources.load(binding.resource_id)
    fasta = uploads._owned(resources.load(resource_id))
    if fasta.input_type != 'fasta':
        _fail('Upload a plain FASTA in the reference-genome role.')
    path, sha, size = resources._source(fasta.source_path)
    if (sha, size) != (fasta.source_sha256, fasta.size_bytes):
        _fail('Uploaded FASTA bytes changed.')
    with open(path, 'rb') as stream:
        if stream.read(2) == b'\x1f\x8b':
            _fail('Compressed FASTA is not supported by this upload role. Supply plain FASTA.')
    key = digest(dict(session=session_id, origin=origin.turn_id, source=source.record_sha256,
                      fasta=fasta.record_sha256, requirement=required))
    registration_key = 'qualified-reference:' + key
    try:
        existing = resources.load(resources.registration_id(registration_key))
    except Exception as exc:
        from .local_resources import ResourceAdmissionError
        if not isinstance(exc, ResourceAdmissionError) or exc.code != 'LOCAL_RESOURCE_UNAVAILABLE':
            raise
    else:
        check_record(resources, existing)
        return dict(choice=input_set(existing).choice(), resource_id=existing.resource_id,
                    qualification='qualified', sharing_scope='private_task')
    root = resources._workspace._ensure_directory(uploads.root / 'qualified_references')
    # One immutable publication per explicit source/task identity. Lock publication
    # to prevent simultaneous retries from regenerating an index or verification.
    lease = uploads._lease('reference-' + key[0] + '.lock')
    try:
        directory = resources._workspace._ensure_directory(root / key)
        index = prepare_fai(resources, fasta, directory)
        bed = directory / 'features.bed'
        names = _feature_names(source.source_path)
        candidate = directory / 'features.bed.part'
        candidate.unlink(missing_ok=True)
        try:
            _write_bed(candidate, names)
            if bed.exists():
                if candidate.read_bytes() != bed.read_bytes():
                    _fail('Previously prepared feature vocabulary changed.')
            else:
                os.replace(candidate, bed)
        finally:
            candidate.unlink(missing_ok=True)
        provenance = scatac_reference.SourceProvenance(basis='caller_supplied',
            source='Explicit browser FASTA registration; publisher and species/assembly declarations are not independently established.')
        reference = owner.build_regulatory_feature_reference(species=required['species'],
            target_assembly=required['assembly'], fasta_path=path, fai_path=index,
            feature_bed_path=bed, feature_category=required['feature_category'], genome_provenance=provenance,
            feature_provenance=scatac_reference.SourceProvenance(basis='caller_supplied', source='Exact ordered chr:start-end H5AD feature axis.'))
        manifest = directory / 'reference.json'
        if manifest.exists():
            _, prior, _ = owner.load_regulatory_feature_reference(manifest)
            if prior != reference:
                _fail('Previously published reference changed.')
        else:
            owner.publish_regulatory_feature_reference(reference, manifest)
        files = VerificationContext().files([path, index, bed])
        _check_owner_files(reference, files)
        # Recheck source/task after the potentially long, streaming qualification.
        if service._load(session_id) != state:
            raise SessionConflictError('Reference qualification raced with another turn.')
        resources.validate_binding(binding)
        record = resources.register(registration_key, manifest, input_type='regulatory_reference',
            label=(required['species']['scientific_name'] + ' ' + required['assembly'] + ' — qualified reference')[:160],
            attribution='M14 reference-owner qualification; private task resource; scientific declarations remain explicit.',
            reference_qualification=dict(requirement=required, reference_identity_sha256=reference.reference_identity_sha256,
                ordered_feature_sha256=reference.features.ordered_feature_sha256, files=files,
                session_id=session_id, origin_turn_id=origin.turn_id, source_record_sha256=source.record_sha256,
                scope='session', review=None))
        return dict(choice=input_set(record).choice(), resource_id=record.resource_id,
                    qualification='qualified', sharing_scope='private_task')
    finally:
        os.close(lease)


def pending_choices(public, interaction, state, resources):
    ref = interaction.snapshot.get('prerequisite')
    if ref is None or ref['field'] != 'resources':
        return
    origin = next(i for i in state.interactions if i.turn_id == ref['origin_turn_id'])
    required = origin.admitted.get('reference_requirement')
    if required is not None:
        public['choices'] = [dict(handle='resource.' + input_set(r).input_set_id, label=input_set(r).label)
            for r in choices(resources, state.session_id, required, origin_turn_id=origin.turn_id)]
        public['reference_requirement'] = _serialize(required)


def validate_completion(record, source, origin, binding):
    from .turn_decisions import IntentError
    q = record.reference_qualification
    if (binding is None or source is None or q is None
            or _serialize(q['requirement']) != _serialize(origin.admitted.get('reference_requirement'))
            or source.resource_id != binding.resource_id or source.record_sha256 != binding.record_sha256
            or q['scope'] == 'session' and (q['origin_turn_id'] != origin.turn_id
                or q['source_record_sha256'] != source.record_sha256)):
        raise IntentError('invalid_prerequisite')


def select(resources, session_id, origin, handle):
    from .turn_decisions import IntentError
    if resources is None:
        raise IntentError('invalid_prerequisite')
    matches = [r for r in choices(resources, session_id, origin.admitted['reference_requirement'],
                origin_turn_id=origin.turn_id) if 'resource.' + input_set(r).input_set_id == handle]
    if len(matches) != 1:
        raise IntentError('invalid_prerequisite')
    record = matches[0]
    binding = resources._submission_binding(origin.submission)
    resources.validate_binding(binding)
    source = resources.load(binding.resource_id)
    if isinstance(record, dict) or record.input_type == 'genome_reference':
        from .shared_genome import bind
        record = bind(resources, record, source, session_id, origin)
    validate_completion(record, source, origin, binding)
    reference = check_record(resources, record)
    from agent.tools.data.scatac_reference import ordered_identity_sha256
    if ordered_identity_sha256(_feature_names(source.source_path)) != reference.features.ordered_feature_sha256:
        raise IntentError('invalid_prerequisite')
    return record, source
