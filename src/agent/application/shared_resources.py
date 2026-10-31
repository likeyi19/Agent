"""One persistent library for common references; task uploads stay in Session storage.

Installation is the explicit sharing boundary. Scientific owners qualify once;
lookup projects exact immutable identities without replay or legal-document checks.
The remaining local-record functions read history and retain private task resources.
"""
from pathlib import Path
from agent.schemas.orchestration import _serialize
from .session_state import canonical, digest
from .reference_supplementation import _fail
from contextlib import contextmanager
from contextvars import ContextVar

_SCIENTIFIC_LIBRARY = ContextVar('scientific_resource_library', default=None)


@contextmanager
def scientific_resource_scope(library):
    """Share the deployment library with scientific execution in this operation."""
    token = _SCIENTIFIC_LIBRARY.set(library)
    try:
        yield
    finally:
        _SCIENTIFIC_LIBRARY.reset(token)


def current_scientific_library():
    return _SCIENTIFIC_LIBRARY.get()


def redistribution_files(files):
    """Exclude exact implementation identities, never a filename extension."""
    from agent.tools.data import regulatory_feature_reference, scatac_reference, chromap_reference_index, scatac_qc_reference
    from agent.tools.analysis import annotation_input, annotation_contract, marker_annotation, external_marker_annotation
    implementations = {m.__file__ for m in (regulatory_feature_reference, scatac_reference,
        chromap_reference_index, scatac_qc_reference, annotation_input, annotation_contract,
        marker_annotation, external_marker_annotation)}
    return [row for row in files if row['path'] not in implementations]


def validate_review(value, files, *, legacy=True):
    """Read historical review metadata; new resource installation has no review protocol."""
    if not isinstance(value, dict) or not {'operator', 'provenance', 'reuse_basis'} <= set(value):
        _fail('Invalid historical resource review.')


def _qc_closure(path, sha):
    from agent.tools.data import scatac_qc_reference as owner
    _, bundle, _ = owner.load_scatac_qc_reference_bundle(path, expected_sha256=sha)
    # Existing human/mouse QC scope only; no new annotation parser or format.
    if bundle.contract_version != 'scatac-qc-reference-bundle.v1':
        _fail('This resource handoff supports the existing qualified human/mouse QC bundle.')
    parent = _qc_parent(bundle)
    pins = {r.path: r.sha256 for r in (bundle.parent_manifest, bundle.annotation.resource,
        bundle.tss, bundle.lineage, parent.genome.fasta, parent.genome.fai, parent.ccre.bed)}
    pins[str(path)] = sha
    if parent.annotation is not None:
        pins[parent.annotation.resource.path] = parent.annotation.resource.sha256
    return bundle, pins


def _qc_parent(bundle):
    """Read exact owner lineage without reconstructing or qualifying references."""
    from agent.tools.data._fragment_reference import load
    parent = load(bundle.parent_manifest.path, bundle.parent_manifest.sha256)
    if (parent.reference_identity_sha256 != bundle.parent_reference_identity_sha256
            or parent.genome.ordered_contig_sha256 != bundle.ordered_contig_sha256
            or (parent.species, parent.target_assembly) != (bundle.species, bundle.assembly)):
        _fail('QC scope lineage differs from its qualified parent genome.')
    return parent


def _raw_reference_choices(inputs, identity, label, files):
    """Project a qualified QC closure into existing operation-specific InputSets.

    Only pinned manifests are read here. Selected-file integrity and scientific
    consumption retain their existing owners; a loose QC file is not a catalog.
    """
    from .input_sets import ScientificInputSet
    from agent.tools.data.scatac_qc_reference import load_scatac_qc_reference_bundle
    _, bundle, _ = load_scatac_qc_reference_bundle(inputs['qc_reference_manifest_path'],
        expected_sha256=inputs['qc_reference_manifest_sha256'])
    if bundle.contract_version != 'scatac-qc-reference-bundle.v1':
        return ()
    if not any(row['path'] == bundle.parent_manifest.path
               and row['sha256'] == bundle.parent_manifest.sha256 for row in files):
        _fail('QC parent is outside the qualified resource closure.')
    _qc_parent(bundle)
    values = dict(inputs, reference_bundle_path=bundle.parent_manifest.path,
                  reference_bundle_sha256=bundle.parent_manifest.sha256)
    return tuple(ScientificInputSet(prefix + identity, label + suffix, values,
                 **{role + '_companion': True}) for prefix, suffix, role in (
        ('fref-', ' — fragments reference', 'fragments'),
        ('bref-', ' — BAM reference', 'bam')))


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
        bundle, pins = _qc_closure(source, inputs['qc_reference_manifest_sha256'])
        from agent.tools.data._barcode_qc_binding import resource_qualification
        qualification = resource_qualification(bundle)
        from agent.tools.data import scatac_qc_reference
        # Source reconstruction is a qualification step, not a discovery/read step.
        scatac_qc_reference.reinspect_scatac_qc_reference_bundle_sources(bundle)
        profile = bundle.contract_version
        paths = []
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
    files = redistribution_files(q['files'])
    if VerificationContext().files(r['path'] for r in files) != files:
        _fail('Qualified resource bytes changed; admit a new immutable version.')
    return q['inputs']


def input_set(record):
    from .input_sets import ScientificInputSet
    return ScientificInputSet('qualified-' + record.resource_id[6:46], record.label,
                              _serialize(record.reference_qualification['inputs']))


def discover(resources, session_id):
    """Scope before projection; read pinned manifests, never data or native science."""
    from .reference_supplementation import _records
    choices = list(resources.library.choices()) if resources.library is not None else []
    for record in _records(resources):
        if record.input_type == 'genome_reference':
            if record.session_id == session_id:
                from .input_sets import ScientificInputSet
                choices.append(ScientificInputSet('genome-' + record.resource_id[6:46], record.label, dict(qualified_genome=record.to_dict())))
            continue
        if record.input_type != 'scientific_reference': continue
        q = record.reference_qualification
        if q['scope'] != 'session' or q['session_id'] != session_id: continue
        choices.append(input_set(record))
        if q['profile'] == 'scatac-qc-reference-bundle.v1':
            try:
                choices.extend(_raw_reference_choices(_serialize(q['inputs']), record.resource_id[6:46],
                    record.label, q['files']))
            except (ValueError, OSError):
                pass  # An unavailable parent must not hide unrelated valid resources.
    return tuple(choices)


def check_choice(resources, session_id, handle):
    """Verify only the selected reference before new scientific consumption."""
    import re
    derived = re.fullmatch(r'resource\.(fref-|bref-)([0-9a-f]{56}|[0-9a-f]{40})', handle)
    if derived is not None:
        prefix, identity = derived.groups()
        if resources is None: _fail('Scientific resource library unavailable.')
        if len(identity) == 56:
            if resources.library is None: _fail('Scientific resource library unavailable.')
            entry = resources.library.get(identity)
            if entry['kind'] != 'qc': _fail('Selected reference has no qualified QC parent.')
            resources.library.check(entry)
            choices = _raw_reference_choices(entry['resource'], identity, entry['label'], entry['files'])
        else:
            from .reference_supplementation import _records
            matches = [r for r in _records(resources) if r.input_type == 'scientific_reference'
                       and r.resource_id[6:46] == identity]
            if len(matches) != 1: _fail('Selected private reference is unavailable.')
            record = matches[0]
            q = record.reference_qualification
            if q['scope'] != 'session' or q['session_id'] != session_id:
                _fail('Private reference is unavailable in this Session.')
            values = check_record(resources, record)
            if q['profile'] != 'scatac-qc-reference-bundle.v1':
                _fail('Selected reference has no qualified QC parent.')
            choices = _raw_reference_choices(values, identity, record.label, q['files'])
        return next(s.inputs() for s in choices if s.input_set_id == prefix + identity)
    if handle.startswith('resource.library-'):
        if resources is None or resources.library is None: _fail('Scientific resource library unavailable.')
        entry = resources.library.get(handle.removeprefix('resource.library-'))
        resources.library.check(entry)
        return resources.library.input_set(entry).inputs()
    if re.fullmatch(r'resource\.qualified-[0-9a-f]{40}', handle) is None: return
    from .reference_supplementation import _records
    for record in _records(resources):
        if record.input_type != 'scientific_reference' or 'resource.' + input_set(record).input_set_id != handle: continue
        q = record.reference_qualification
        if q['scope'] == 'deployment':
            _fail('Legacy shared selection is retired; install the reference in ScientificResourceLibrary.')
        if q['session_id'] != session_id:
            _fail('Private reference is unavailable in this Session.')
        return check_record(resources, record)
    _fail('Selected shared reference is no longer registered.')



class ScientificResourceLibrary:
    """Trusted local index. No upload registration, promotion, or result authority.

    The caller explicitly installs a common reference and establishes its usage
    basis at provisioning time. Experimental files have no supported library kind.
    """
    def __init__(self, root):
        self.root = Path(root).expanduser().absolute()
        if self.root != self.root.resolve() or self.root.is_symlink():
            _fail('Resource library must be a canonical directory.')
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'index.json'

    def entries(self):
        import json
        if not self.path.exists(): return ()
        if self.path.is_symlink() or self.path.stat().st_size > 4 * 1024**2:
            _fail('Invalid scientific resource index.')
        def _json_object_pairs(pairs):
            result = {}
            for key, value in pairs:
                if key in result: _fail('Duplicate scientific index key.')
                result[key] = value
            return result
        def _reject_constant(value): _fail('Nonfinite scientific index value.')
        value = json.loads(self.path.read_text(), object_pairs_hook=_json_object_pairs, parse_constant=_reject_constant)
        if (not isinstance(value, dict) or set(value) != {'version', 'resources'}
                or type(value['version']) is not int or value['version'] != 1 or not isinstance(value['resources'], list)
                or len(value['resources']) > 256):
            _fail('Unsupported scientific resource index.')
        identities = set()
        for row in value['resources']:
            if (not isinstance(row, dict) or set(row) != {'id', 'kind', 'label', 'version', 'provenance', 'usage', 'resource', 'files'}
                    or row['kind'] not in ('annotation', 'genome', 'qc', 'epizoo', 'marker_reference', 'liftover_chain', 'epizoo_transfer', 'sequence_predictor', 'sequence_generator', 'motif_collection')
                    or row['id'] != digest({k: v for k, v in row.items() if k != 'id'})[:56]
                    or row['id'] in identities):
                _fail('Scientific resource identity is corrupt.')
            identities.add(row['id'])
        return tuple(value['resources'])

    def get(self, identity):
        matches = [r for r in self.entries() if r['id'] == identity]
        if len(matches) != 1: _fail('Scientific resource is unavailable.')
        return matches[0]

    def sequence_models(self):
        """Installed model metadata, without file locators or execution authority."""
        entries = self.entries()
        return [dict(family=e['resource']['family'], kind=e['kind'],
            conditions=e['resource']['conditions'],
            measurement=e['resource'].get('measurements',
                {'output': 'synthetic regulatory DNA candidates; no activity measurement'}),
            genome_assemblies=[g['resource']['assembly'] for g in entries if g['kind'] == 'genome'])
            for e in entries if e['kind'] in ('sequence_predictor', 'sequence_generator')]

    def check(self, entry):
        from agent.tools.data.authority_context import VerificationContext
        if self.get(entry['id']) != entry: _fail('Scientific resource index changed.')
        if VerificationContext().files(r['path'] for r in entry['files']) != entry['files']:
            _fail('Scientific resource bytes changed; install a new immutable version.')
        return entry['resource']

    def install(self, kind, resource, *, label, version, provenance, usage, own_files=False):
        """Explicit operator/shared-use installation, with one owner qualification.

        Index existing files in place. Exact repeated installation checks integrity
        and returns the same entry without repeating scientific qualification.
        """
        import fcntl, os, tempfile
        from agent.tools.data.authority_context import VerificationContext
        from .input_sets import _label
        resource = _serialize(resource)
        for text in (label, version, provenance, usage): _label(text)
        if own_files:
            resource = self._own_files(kind, resource)
        qualifier = None
        if kind == 'annotation':
            from agent.tools.analysis import annotation_contract as c, annotation_input
            # Retain the owner's scientific specification, not task grouping.
            resource = dict(resource, grouping_provenance='Grouping must be supplied by each consuming task')
            target = self.root / 'annotations' / (digest(resource) + '.json')
            target.parent.mkdir(exist_ok=True)
            self._publish(target, canonical(resource))
            resource = c.specification(target, digest(resource))
            if resource['schema'] != c.EXTERNAL_SPEC: _fail('Direct annotation references required.')
            pins = {r['path']: r['sha256'] for r in (resource['gene_resource'], resource['signature_resource'])}
            if 'contig_scope' in resource:
                pins[resource['contig_scope']['path']] = resource['contig_scope']['sha256']
            pins[str(target)] = digest(resource)
            qualifier = lambda: annotation_input.qualify_resources(resource)
        elif kind == 'motif_collection':
            from agent.tools.analysis import chromvar
            pins = {resource[k]['path']:resource[k]['sha256'] for k in ('pfm','metadata')}
            qualifier = lambda: chromvar.qualify_resource(resource)
        elif kind == 'sequence_generator':
            from agent.tools.analysis import sequence_generation
            pins = {resource[k]['path']:resource[k]['sha256'] for k in ('checkpoint','config','metadata')}
            qualifier = lambda: sequence_generation.qualify_resource(resource)
        elif kind == 'sequence_predictor':
            from agent.tools.analysis import sequence_prediction, sequence_optimization
            qualify_resource = (sequence_optimization.qualify_resource if resource.get('schema') == sequence_optimization.RESOURCE else sequence_prediction.qualify_resource)
            pins = {r['path']: r['sha256'] for r in [*resource['checkpoints'], resource['metadata']]}
            qualifier = lambda: qualify_resource(resource)
        elif kind == 'marker_reference':
            from agent.tools.analysis.annotation_corroboration import reference
            knowledge = resource['knowledge']
            pins = {knowledge[k + '_path']: knowledge[k + '_sha256'] for k in ('snapshot', 'table')}
            qualifier = lambda: reference(resource)
        elif kind == 'genome':
            from agent.tools.data import scatac_reference as r, regulatory_feature_reference as owner
            from agent.tools.data.chromap_reference_index import verify_fasta_fai
            from types import SimpleNamespace
            if set(resource) != {'species', 'assembly', 'genome'}: _fail('Genome species, assembly and exact FASTA/FAI required.')
            owner.validate_species(resource['species']); r._text(resource['assembly'])
            g = dict(resource['genome'])
            for key in ('fasta', 'fai'):
                r._resource(g[key]); g[key] = r.ResourceIdentity(g[key]['path'], g[key]['sha256'], r.SourceProvenance(**g[key]['provenance']))
            genome = r.GenomeReference(**g)
            pins = {v.path: v.sha256 for v in (genome.fasta, genome.fai)}
            qualifier = lambda: verify_fasta_fai(SimpleNamespace(genome=genome))
        elif kind == 'qc':
            if set(resource) != {'qc_reference_manifest_path', 'qc_reference_manifest_sha256'}: _fail('Exact QC reference required.')
            from agent.tools.data import scatac_qc_reference as owner
            # Existing QC owner supplies its content closure and scientific identity.
            bundle, pins = _qc_closure(resource['qc_reference_manifest_path'], resource['qc_reference_manifest_sha256'])
            qualifier = lambda: owner.reinspect_scatac_qc_reference_bundle_sources(bundle)
        elif kind == 'liftover_chain':
            from agent.tools.data.liftover_chain import qualify
            pins={resource[k]['path']:resource[k]['sha256'] for k in ('chain','input_fai','destination_fai')}
            qualifier=lambda: qualify(resource)
        elif kind == 'epizoo_transfer':
            from agent.tools.models.epizoo_adaptation import resources as owner, contract as c
            c.checks.shape(resource,('source_bundle','seam_bundle','source_references','liftover','bedtools'))
            pins={}
            for key in ('source_bundle','seam_bundle'):
                record=resource[key];bundle=c.read_json(record['path'],record['sha256'])
                pins[record['path']]=record['sha256']
                pins.update({r['path']:r['sha256'] for r in bundle['files']+bundle['code']['files']})
            for record in [*resource['source_references'],resource['liftover'],resource['bedtools']]:
                if record is not None: pins[record['path']]=record['sha256']
            def qualifier():
                for key in ('source_bundle','seam_bundle'): owner.load_bundle(**resource[key])
                from agent.tools.data.scatac_reference import load_scatac_reference_bundle,reinspect_scatac_reference_bundle_sources
                for ref in resource['source_references']:
                    reinspect_scatac_reference_bundle_sources(load_scatac_reference_bundle(ref['path'],expected_sha256=ref['sha256'])[1])
        elif kind == 'epizoo':
            from .local_resources import QualifiedEpiZooResource
            from agent.tools.analysis.epizoo_embedding import _pinned_resource_paths, _check_resource_files
            model = QualifiedEpiZooResource(**resource)
            paths = _pinned_resource_paths(model.checkpoint_path, model.species)
            expected = model.inputs()['expected_resource_identity']
            pins = {str(path): expected[key] for key, path in paths.items()}
            def qualifier():
                import torch
                from agent.tools.models import epizoo
                _check_resource_files(paths, expected)
                state = epizoo._unwrap_state_dict(torch.load(model.checkpoint_path,
                    map_location='cpu', weights_only=True, mmap=True))
                epizoo._validate_checkpoint_state_dict(state)
                epizoo._load_resources(epizoo._normalize_species(model.species), paths['frequencies_sha256'].parent)
        else:
            _fail('Only explicit common scientific reference roles can enter the library.')
        files = VerificationContext().files(pins)
        if any(f['sha256'] != pins[f['path']] for f in files): _fail('Resource checksum differs from its declared identity.')
        entry = dict(kind=kind, label=label, version=version, provenance=provenance, usage=usage, resource=resource, files=files)
        entry = dict(id=digest(entry)[:56], **entry)
        with (self.root / '.index.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            rows = self.entries()
            existing = [r for r in rows if r['kind'] == kind and r['resource'] == resource]
            if existing:
                if len(existing) != 1: _fail('Duplicate scientific resources in the library.')
                self.check(existing[0])
                return existing[0]
            qualifier()
            if VerificationContext().files(pins) != files: _fail('Resource changed during scientific validation.')
            content = canonical(dict(version=1, resources=[*rows, entry]))
            fd, name = tempfile.mkstemp(dir=self.root, prefix='.index-')
            try:
                with os.fdopen(fd, 'wb') as stream:
                    stream.write(content); stream.flush(); os.fsync(stream.fileno())
                os.replace(name, self.path)
                fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
                try: os.fsync(fd)
                finally: os.close(fd)
            finally:
                Path(name).unlink(missing_ok=True)
        return entry

    @staticmethod
    def _publish(path, content):
        """Atomically publish an immutable scientific specification."""
        import os, tempfile
        fd, name = tempfile.mkstemp(dir=path.parent, prefix='.reference-')
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(content); stream.flush(); os.fsync(stream.fileno())
            try: os.link(name, path)
            except FileExistsError:
                if path.is_symlink() or path.read_bytes() != content: _fail('Installed reference specification changed.')
            fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try: os.fsync(fd)
            finally: os.close(fd)
        finally:
            Path(name).unlink(missing_ok=True)

    def _copy_file(self, path, sha):
        """Own exact bytes independently of the upload inode and its lifetime."""
        import hashlib, os, shutil, tempfile
        from agent.tools.data.scatac_reference import _sha
        _sha(sha)
        source = Path(path)
        directory = self.root / 'files' / sha
        directory.mkdir(parents=True, exist_ok=True)
        if directory.resolve() != directory: _fail('Library file directory changed.')
        target = directory / source.name
        size = source.stat().st_size
        if target.exists() or target.is_symlink():
            if target.is_symlink() or target.stat().st_size != size:
                _fail('Installed scientific file changed.')
            with target.open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != sha:
                    _fail('Installed scientific file changed.')
            return str(target)
        if shutil.disk_usage(directory).free < size:
            _fail('Insufficient library space for the shared reference.')
        fd, temporary = tempfile.mkstemp(dir=directory, prefix='.copy-')
        try:
            checksum = hashlib.sha256(); copied = 0
            with os.fdopen(fd, 'wb') as writer, source.open('rb') as reader:
                while block := reader.read(1024 * 1024):
                    copied += len(block)
                    if copied > size: _fail('Scientific source changed during copying.')
                    writer.write(block); checksum.update(block)
                writer.flush(); os.fsync(writer.fileno())
            if copied != size or checksum.hexdigest() != sha:
                _fail('Scientific source changed during copying.')
            try: os.link(temporary, target)
            except FileExistsError:
                # Another importer may have published this exact content.
                return self._copy_file(path, sha)
            fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try: os.fsync(fd)
            finally: os.close(fd)
            fd = os.open(directory.parent, os.O_RDONLY | os.O_DIRECTORY)
            try: os.fsync(fd)
            finally: os.close(fd)
            return str(target)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def _own_files(self, kind, resource):
        """Relocate only each supported owner's explicit reference closure."""
        from dataclasses import asdict, replace
        from agent.tools.data import scatac_reference as parent, scatac_qc_reference as qc
        def copy(row):
            return dict(row, path=self._copy_file(row['path'], row['sha256']))
        def manifest(raw):
            import hashlib
            sha = hashlib.sha256(raw).hexdigest()
            path = self.root / 'manifests' / (sha + '.json')
            path.parent.mkdir(exist_ok=True)
            if path.parent.resolve() != path.parent: _fail('Library manifest directory changed.')
            self._publish(path, raw)
            return dict(path=str(path), sha256=sha, size_bytes=len(raw))
        if kind == 'annotation':
            result = dict(resource, gene_resource=copy(resource['gene_resource']),
                          signature_resource=copy(resource['signature_resource']))
            if 'contig_scope' in resource:
                scope = resource['contig_scope']
                relocated = self._own_files('qc', dict(qc_reference_manifest_path=scope['path'],
                    qc_reference_manifest_sha256=scope['sha256']))
                result['contig_scope'] = dict(path=relocated['qc_reference_manifest_path'],
                    sha256=relocated['qc_reference_manifest_sha256'])
            return result
        if kind == 'genome':
            genome = dict(resource['genome'])
            return dict(resource, genome=dict(genome, fasta=copy(genome['fasta']), fai=copy(genome['fai'])))
        if kind == 'liftover_chain':
            return dict(resource, **{key: copy(resource[key])
                for key in ('chain', 'input_fai', 'destination_fai')})
        if kind == 'qc':
            bundle, _ = _qc_closure(resource['qc_reference_manifest_path'], resource['qc_reference_manifest_sha256'])
            _, original, _ = parent.load_scatac_reference_bundle(bundle.parent_manifest.path,
                expected_sha256=bundle.parent_manifest.sha256)
            values = original.to_dict()
            for key in ('fasta', 'fai'): values['genome'][key] = copy(values['genome'][key])
            values['ccre']['bed'] = copy(values['ccre']['bed'])
            if values['annotation'] is not None:
                values['annotation']['resource'] = copy(values['annotation']['resource'])
            # The native parent identity excludes locators; the QC identity binds
            # the newly published parent bytes. Neither original is rewritten.
            relocated = parent.validate_scatac_reference_bundle(values)
            parent_file = manifest(parent.canonical_reference_bundle_bytes(relocated))
            result = replace(bundle, parent_manifest=qc.QCResource(**parent_file),
                annotation=replace(bundle.annotation, resource=qc.QCResource(**copy(asdict(bundle.annotation.resource)))),
                tss=qc.QCResource(**copy(asdict(bundle.tss))), lineage=qc.QCResource(**copy(asdict(bundle.lineage))))
            result = replace(result, resource_identity_sha256=qc._identity(result))
            output = manifest(qc.canonical_qc_reference_bundle_bytes(result))
            return dict(qc_reference_manifest_path=output['path'], qc_reference_manifest_sha256=output['sha256'])
        _fail('Only annotation, genome, QC and chain references support library-owned copying.')

    def install_genome(self, fasta_path, fai_path, *, species, assembly, label, version, provenance, usage):
        """Install external FASTA/FAI directly, without an experimental dataset or Run."""
        from dataclasses import asdict
        from agent.tools.data import scatac_reference as r
        fasta, fai = (r._source_path(p) for p in (fasta_path, fai_path))
        fasta_sha = r._file_hash(fasta, fasta=True)
        contigs, fai_sha, ordered = r._inspect_fai(fai, fasta.stat().st_size)
        source = r.SourceProvenance(basis='caller_supplied', source=provenance)
        genome = r.GenomeReference(r.ResourceIdentity(str(fasta), fasta_sha, source),
            r.ResourceIdentity(str(fai), fai_sha, source), len(contigs), ordered)
        return self.install('genome', dict(species=species, assembly=assembly, genome=asdict(genome)),
            label=label, version=version, provenance=provenance, usage=usage)

    def input_set(self, entry):
        from .input_sets import ScientificInputSet
        kind, resource = entry['kind'], entry['resource']
        if kind == 'annotation':
            inputs = dict(annotation_spec_path=str(self.root / 'annotations' / (digest(resource) + '.json')),
                          annotation_spec_sha256=digest(resource))
        elif kind == 'qc': inputs = resource
        elif kind == 'genome': inputs = dict(library_genome=entry)
        elif kind == 'epizoo_transfer': inputs = dict(library_epizoo_transfer=entry)
        else: _fail('Models use the existing native model selection interface.')
        return ScientificInputSet('library-' + entry['id'], entry['label'], inputs, h5ad_companion=kind in ('genome','epizoo_transfer'))

    def choices(self):
        result = []
        for row in self.entries():
            if row['kind'] in ('epizoo', 'marker_reference', 'liftover_chain', 'sequence_predictor', 'sequence_generator', 'motif_collection'): continue
            result.append(self.input_set(row))
            if row['kind'] == 'qc':
                try:
                    result.extend(_raw_reference_choices(row['resource'], row['id'], row['label'], row['files']))
                except (ValueError, OSError):
                    pass  # Keep the original QC offer; do not offer an invalid parent.
        return tuple(result)

    def models(self):
        from .local_resources import QualifiedEpiZooResource
        return tuple(QualifiedEpiZooResource(**r['resource']) for r in self.entries() if r['kind'] == 'epizoo')


def install_private(resources, session_id, resource_id, *, shared_use):
    """Explicit shared-future-use action over a Session-owned scientific reference.

    Ordinary uploads never call this action. Required reference components become
    library-owned copies; genomes share only FASTA/FAI, never task feature axes.
    """
    if shared_use is not True or resources.library is None:
        _fail('Explicit shared use and a configured scientific library are required.')
    record = resources.load(resource_id)
    if record.input_type not in ('scientific_reference', 'genome_reference', 'regulatory_reference'):
        _fail('Experimental uploads cannot be installed as common references.')
    if (record.session_id or record.reference_qualification.get('session_id')) != session_id:
        _fail('Only this Session\'s private reference may be installed for shared use.')
    from . import reference_supplementation as refs
    value = refs.check_record(resources, record)
    if record.input_type == 'scientific_reference':
        if 'annotation_spec_path' in value:
            from agent.tools.analysis.annotation_contract import specification
            kind, resource = 'annotation', specification(value['annotation_spec_path'], value['annotation_spec_sha256'])
        else: kind, resource = 'qc', value
    elif record.input_type in ('regulatory_reference', 'genome_reference'):
        from dataclasses import asdict
        kind = 'genome'
        if record.input_type == 'regulatory_reference':
            resource = dict(species=asdict(value.species), assembly=value.target_assembly, genome=asdict(value.genome))
        else:
            resource = dict(**_serialize(record.reference_qualification['requirement']), genome=value)
    else:
        _fail('Experimental uploads cannot be installed as common references.')
    return resources.library.install(kind, resource, label=record.label, version=record.source_sha256,
        provenance=record.attribution, usage='Explicit user designation for shared future laboratory computation; reference bytes are not offered for download.',
        own_files=True)
