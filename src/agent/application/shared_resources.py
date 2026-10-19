"""Qualified reference variants in LocalResourceAdmission, not a second catalog.

Only existing annotation and QC contracts are projected to existing InputSets.
Promotion is a trusted operator API; uploads and interpreter decisions cannot call
it. Permission evidence records an operator's review, never an inferred license.
"""
from pathlib import Path
from agent.schemas.orchestration import _serialize
from .session_state import canonical, digest
from .reference_supplementation import _fail


def redistribution_files(files):
    """Exclude exact implementation identities, never a filename extension."""
    from agent.tools.data import regulatory_feature_reference, scatac_reference, chromap_reference_index, scatac_qc_reference
    from agent.tools.analysis import annotation_input, annotation_contract, marker_annotation, external_marker_annotation
    implementations = {m.__file__ for m in (regulatory_feature_reference, scatac_reference,
        chromap_reference_index, scatac_qc_reference, annotation_input, annotation_contract,
        marker_annotation, external_marker_annotation)}
    return [row for row in files if row['path'] not in implementations]


def review(operator, provenance, reuse_basis, redistribution, files):
    value = dict(operator=operator, provenance=provenance, reuse_basis=reuse_basis,
                 redistribution=redistribution)
    validate_review(value, files, legacy=False)
    check_review(value)
    return value


def validate_review(value, files, *, legacy=True):
    from agent.tools.data.scatac_reference import _sha
    basic = {'operator', 'provenance', 'reuse_basis'}
    if (not isinstance(value, dict) or set(value) not in (basic, basic | {'redistribution'})
            or any(type(value[k]) is not str or not value[k].strip() or len(value[k]) > 1024 for k in basic)):
        _fail('Deployment reuse requires explicit operator review.')
    if 'redistribution' not in value:
        if legacy: return  # Preserve previously accepted, explicitly reviewed records.
        _fail('New promotion requires pinned redistribution permission evidence.')
    rows = value['redistribution']
    required = {f['sha256'] for f in redistribution_files(files)}
    if (not isinstance(rows, list) or not rows or len(rows) > 32
            or {r.get('sha256') for r in rows if isinstance(r, dict)} != required
            or len(rows) != len(required)):
        _fail('Redistribution review must cover every shared scientific file.')
    for row in rows:
        if set(row) != {'sha256', 'source_url', 'license_url', 'evidence_path', 'evidence_sha256', 'basis'}:
            _fail('Incomplete redistribution evidence.')
        for name in ('sha256', 'evidence_sha256'): _sha(row[name])
        for name in ('source_url', 'license_url'):
            if type(row[name]) is not str or not row[name].startswith('https://') or len(row[name]) > 1024:
                _fail('Public publisher and permission URLs are required.')
        if (not Path(row['evidence_path']).is_absolute() or type(row['basis']) is not str
                or not row['basis'].strip() or len(row['basis']) > 1024):
            _fail('An explicit permitted reuse basis and pinned evidence are required.')


def check_review(value):
    from agent.tools.analysis.marker_annotation import checked_file
    for row in value.get('redistribution', []):
        path = Path(row['evidence_path'])
        if path.stat().st_size > 65536:
            _fail('Permission evidence exceeds its bound.')
        checked_file(path, row['evidence_sha256'])


def _qc_closure(path, sha):
    from agent.tools.data import scatac_qc_reference as owner
    from agent.tools.data._barcode_qc_binding import resource_qualification
    from agent.tools.data._fragment_reference import load
    _, bundle, _ = owner.load_scatac_qc_reference_bundle(path, expected_sha256=sha)
    qualification = resource_qualification(bundle)
    # Existing human/mouse QC scope only; no new annotation parser or format.
    if bundle.contract_version != 'scatac-qc-reference-bundle.v1':
        _fail('This resource handoff supports the existing qualified human/mouse QC bundle.')
    parent = load(bundle.parent_manifest.path, bundle.parent_manifest.sha256)
    if (parent.reference_identity_sha256 != bundle.parent_reference_identity_sha256
            or parent.genome.ordered_contig_sha256 != bundle.ordered_contig_sha256
            or (parent.species, parent.target_assembly) != (bundle.species, bundle.assembly)):
        _fail('QC scope lineage differs from its qualified parent genome.')
    pins = {r.path: r.sha256 for r in (bundle.parent_manifest, bundle.annotation.resource,
        bundle.tss, bundle.lineage, parent.genome.fasta, parent.genome.fai, parent.ccre.bed)}
    pins[str(path)] = sha
    return bundle, pins, qualification


def persist(resources, inputs, *, session_id, origin_turn_id, label, qualification=None, publication_source=None):
    """Publish one private owner-qualified reference in the existing catalog."""
    from agent.tools.analysis import annotation_input, annotation_contract
    from agent.tools.data.authority_context import VerificationContext
    inputs = _serialize(inputs)
    if set(inputs) == {'annotation_spec_path', 'annotation_spec_sha256'}:
        profile = annotation_contract.EXTERNAL_SPEC
        spec = annotation_contract.specification(inputs['annotation_spec_path'], inputs['annotation_spec_sha256'])
        if spec['schema'] != profile: _fail('Unsupported annotation resource version.')
        qualification = qualification or annotation_input.qualify_resources(spec)
        # Grouping is a task declaration, never a reusable reference property.
        spec['grouping_provenance'] = 'Grouping must be supplied by each consuming task'
        # The annotation owner consumes the already-qualified chromosome scope,
        # not a legacy cCRE vocabulary or FASTA. Do not recreate that old gate.
        pins = {}
        if 'contig_scope' in spec:
            scope = spec['contig_scope']
            pins[scope['path']] = scope['sha256']
        pins.update({spec[k]['path']: spec[k]['sha256'] for k in ('gene_resource', 'signature_resource')})
        payload = canonical(spec)
        roots = [root for root in resources._approved if root.is_relative_to(resources._workspace.root)]
        if publication_source is not None:
            if publication_source not in pins: _fail('Publication source is not a bound reference component.')
            roots = [root for root in roots if Path(publication_source).is_relative_to(root)]
            # Roots containing one exact file form a nested chain. Use the
            # narrowest approved location, not a scientific-resource tie policy.
            roots = [max(roots, key=lambda root: len(root.parts))] if roots else []
        if len(roots) != 1:
            _fail('Reference publication requires an explicitly bound workspace attachment root.')
        directory = resources._workspace._ensure_directory(roots[0] / 'qualified_references')
        target = directory / (digest(spec) + '.json')
        # Registration and immutable publication share the existing resource lock.
        with resources._lock(resources.registration_id('reference-publication:' + digest(spec))):
            if target.exists():
                if target.read_bytes() != payload: _fail('Published reference changed.')
            else:
                import os, tempfile
                fd, temporary = tempfile.mkstemp(dir=directory, prefix='.reference-')
                try:
                    with os.fdopen(fd, 'wb') as stream:
                        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
                    os.link(temporary, target)
                    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
                    try: os.fsync(descriptor)
                    finally: os.close(descriptor)
                finally:
                    Path(temporary).unlink(missing_ok=True)
        inputs = dict(annotation_spec_path=str(target), annotation_spec_sha256=digest(spec))
        source = target
        pins[str(target)] = digest(spec)
        # A resource qualification establishes facts about these pinned bytes.
        # Current method compatibility belongs to the consuming owner, not the
        # incidental source-file hash of the implementation that admitted it.
        paths = []
    elif set(inputs) == {'qc_reference_manifest_path', 'qc_reference_manifest_sha256'}:
        source = inputs['qc_reference_manifest_path']
        bundle, pins, qualification = _qc_closure(source, inputs['qc_reference_manifest_sha256'])
        from agent.tools.data import scatac_qc_reference
        # Source reconstruction is a qualification step, not a discovery/read step.
        scatac_qc_reference.reinspect_scatac_qc_reference_bundle_sources(bundle)
        profile = bundle.contract_version
        paths = [scatac_qc_reference.__file__]
    else:
        _fail('Only existing annotation and QC reference contracts are supported.')
    files = VerificationContext().files([*pins, *paths])
    if any(row['sha256'] != pins[row['path']] for row in files if row['path'] in pins):
        _fail('Scientific source bytes differ from the qualified resource pins.')
    q = dict(contract='qualified-scientific-reference.v1', profile=profile, inputs=inputs,
             files=files, qualification=qualification, session_id=session_id,
             origin_turn_id=origin_turn_id, scope='session', review=None)
    return resources.register('scientific-reference:' + digest(q), source,
        input_type='scientific_reference', label=label, attribution='Private qualified scientific reference; no result authority.',
        reference_qualification=q, session_id=session_id)


def validate_record(record):
    from agent.tools.data.scatac_reference import _sha
    q = _serialize(record.reference_qualification)
    if (set(q) != {'contract', 'profile', 'inputs', 'files', 'qualification', 'session_id', 'origin_turn_id', 'scope', 'review'}
            or q['contract'] != 'qualified-scientific-reference.v1'
            or q['scope'] not in ('session', 'deployment')
            or not isinstance(q['qualification'], dict) or not q['qualification']):
        _fail('Invalid scientific resource qualification.')
    key = {'scatac-annotation-peak-resources.v1': 'annotation_spec',
           'scatac-qc-reference-bundle.v1': 'qc_reference_manifest'}.get(q['profile'])
    if (key is None or set(q['inputs']) != {key+'_path', key+'_sha256'}
            or (q['inputs'][key+'_path'], q['inputs'][key+'_sha256']) != (record.source_path, record.source_sha256)
            or not isinstance(q['files'], list) or not 2 <= len(q['files']) <= 32):
        _fail('Unknown or inconsistent scientific reference version.')
    for name in ('session_id', 'origin_turn_id'):
        if type(q[name]) is not str or not q[name] or len(q[name]) > 256: _fail('Missing private origin.')
    paths = set()
    for row in q['files']:
        if (set(row) != {'path', 'sha256', 'size_bytes'} or not Path(row['path']).is_absolute()
                or row['path'] in paths or type(row['size_bytes']) is not int or row['size_bytes'] < 1):
            _fail('Invalid scientific resource pin.')
        paths.add(row['path']); _sha(row['sha256'])
    if not any(r['path'] == record.source_path and r['sha256'] == record.source_sha256 for r in q['files']):
        _fail('Missing reference identity pin.')
    if q['scope'] == 'session':
        if q['review'] is not None or record.session_id != q['session_id']: _fail('Invalid private scope.')
    else:
        if record.session_id is not None: _fail('Invalid deployment scope.')
        validate_review(q['review'], q['files'], legacy=False)


def check_record(resources, record):
    from agent.tools.data.authority_context import VerificationContext
    validate_record(record)
    if resources.load(record.resource_id) != record: _fail('Resource registration changed.')
    q = _serialize(record.reference_qualification)
    from agent.tools.analysis.annotation_contract import EXTERNAL_SPEC
    files = redistribution_files(q['files']) if q['profile'] == EXTERNAL_SPEC else q['files']
    if VerificationContext().files(r['path'] for r in files) != files:
        _fail('Qualified resource bytes changed; admit a new immutable version.')
    if q['review']: check_review(q['review'])
    return q['inputs']


def input_set(record):
    from .input_sets import ScientificInputSet
    return ScientificInputSet('qualified-' + record.resource_id[6:46], record.label,
                              _serialize(record.reference_qualification['inputs']))


def discover(resources, session_id):
    """Scope before projection; listing neither reads data nor repeats science."""
    from .reference_supplementation import _records
    choices = []
    for record in _records(resources):
        if record.input_type != 'scientific_reference': continue
        q = record.reference_qualification
        if q['scope'] != 'deployment' and q['session_id'] != session_id: continue
        choices.append(input_set(record))
    return tuple(choices)


def check_choice(resources, session_id, handle):
    """Verify only the selected reference before new scientific consumption."""
    import re
    if re.fullmatch(r'resource\.qualified-[0-9a-f]{40}', handle) is None: return
    from .reference_supplementation import _records
    for record in _records(resources):
        if record.input_type != 'scientific_reference' or 'resource.' + input_set(record).input_set_id != handle: continue
        q = record.reference_qualification
        if q['scope'] != 'deployment' and q['session_id'] != session_id:
            _fail('Private reference is unavailable in this Session.')
        return check_record(resources, record)
    _fail('Selected shared reference is no longer registered.')


def promote(resources, resource_id, *, operator, provenance, reuse_basis, redistribution=None):
    record = resources.load(resource_id)
    if record.input_type != 'scientific_reference': _fail('Choose a qualified scientific reference.')
    check_record(resources, record)
    q = _serialize(record.reference_qualification)
    q.update(scope='deployment', review=review(operator, provenance, reuse_basis, redistribution, q['files']))
    return resources.register('deployment-reference:' + digest(dict(source=record.record_sha256, review=q['review'])),
        record.source_path, input_type=record.input_type, label=record.label,
        attribution='Explicit operator-reviewed shared scientific reference.', reference_qualification=q)
