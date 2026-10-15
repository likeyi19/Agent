"""Task-scoped raw declarations; scientific qualification remains with owners.

This is the raw-library variant of role continuation, not a scientific planner.
Only explicitly captured attachments can become members of one declared library.
"""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import re

from agent.schemas.orchestration import _serialize
from .session_state import digest


@dataclass(frozen=True)
class FastqMember:
    source: str
    role: str
    lane: str
    chunk: str

    def __post_init__(self):
        from agent.tools.data._raw_fastq import ReadRole
        if (type(self.source) is not str or re.fullmatch(r'source\.(?:[0-9]|1[0-5])', self.source) is None
                or self.role not in {r.value for r in ReadRole}
                or any(type(v) is not str or re.fullmatch(r'[0-9]{3}', v) is None for v in (self.lane, self.chunk))):
            raise ValueError('Invalid FASTQ member declaration.')


@dataclass(frozen=True)
class FastqDeclaration:
    assay: str | None = None
    layout: str | None = None
    namespace: str | None = None
    species: str | None = None
    barcode_policy: str | None = None
    resource: str | None = None
    members: tuple[FastqMember, ...] = ()

    def __post_init__(self):
        from agent.tools.data._raw_fastq import FastqLayout
        from agent.tools.data.scatac_library_context import _namespace
        if self.assay not in (None, 'TENX_ATAC') or self.layout not in (None, FastqLayout.A.value, FastqLayout.B.value):
            raise ValueError('Unsupported FASTQ assay/layout declaration.')
        if self.namespace is not None:
            _namespace(self.namespace)
            if len(self.namespace) > 64:
                raise ValueError('Library namespace exceeds the member admission bound.')
        if self.species not in (None, 'human', 'mouse'):
            raise ValueError('Unsupported canonical FASTQ species.')
        if self.barcode_policy not in (None, 'raw_sequence_whitelist'):
            raise ValueError('FASTQ requires declared raw barcode sequences and whitelist correction.')
        if self.resource is not None and (type(self.resource) is not str or re.fullmatch(r'resource\.[a-z][a-z0-9_-]{0,63}', self.resource) is None):
            raise ValueError('Invalid FASTQ resource handle.')
        if (type(self.members) is not tuple or len(self.members) > 16
                or any(type(m) is not FastqMember for m in self.members)
                or len({m.source for m in self.members}) != len(self.members)
                or len({(m.lane, m.chunk, m.role) for m in self.members}) != len(self.members)):
            raise ValueError('Duplicate or invalid FASTQ members.')


@dataclass(frozen=True)
class BamDeclaration:
    source: str | None = None
    assay: str | None = None
    namespace: str | None = None
    species: str | None = None
    assembly: str | None = None
    barcode_policy: str | None = None
    source_profile: str | None = None
    resource: str | None = None

    def __post_init__(self):
        from agent.tools.data.scatac_library_context import _namespace
        if self.source is not None and (type(self.source) is not str or re.fullmatch(r'source\.(?:[0-9]|1[0-5])', self.source) is None):
            raise ValueError('Invalid BAM source handle.')
        if self.namespace is not None:
            _namespace(self.namespace)
        for value, choices in ((self.assay, ('SCATAC', 'TENX_ATAC')),
                (self.species, ('human', 'mouse')), (self.assembly, ('hg38', 'mm10')),
                (self.barcode_policy, ('corrected_cb_already_corrected',)),
                (self.source_profile, ('agent-cb-paired-atac.v1',))):
            if value is not None and value not in choices:
                raise ValueError('Unsupported BAM scientific declaration.')
        if self.species and self.assembly and {'human': 'hg38', 'mouse': 'mm10'}[self.species] != self.assembly:
            raise ValueError('BAM species/assembly declarations conflict.')
        if self.resource is not None and (type(self.resource) is not str or re.fullmatch(r'resource\.[a-z][a-z0-9_-]{0,63}', self.resource) is None):
            raise ValueError('Invalid BAM resource handle.')


def target(field):
    return 'prepare_scATAC_bam_fragments' if field == 'bam' else 'prepare_scATAC_fragments'


def empty(field):
    return BamDeclaration() if field == 'bam' else FastqDeclaration()


def source_handles(declared):
    return ([declared.source] if declared.source else []) if isinstance(declared, BamDeclaration) else [m.source for m in declared.members]


def _field(interaction):
    admitted = interaction.admitted or {}
    return 'bam' if 'bam' in admitted else 'fastq' if 'fastq' in admitted else (interaction.snapshot.get('prerequisite') or {}).get('field', 'fastq')


def validate_arguments(arguments, field='fastq'):
    from .local_resources import _FASTQ_COMPANION_TOOLS, _BAM_COMPANION_TOOLS
    allowed = _BAM_COMPANION_TOOLS if field == 'bam' else _FASTQ_COMPANION_TOOLS
    if any(a.tool not in allowed for a in arguments):
        raise ValueError('FASTQ task arguments must belong to its registered scientific consumers.')


def declaration(value, field='fastq'):
    if field == 'bam':
        if type(value) is not dict or set(value) != set(BamDeclaration.__dataclass_fields__):
            raise ValueError('Invalid BAM declaration shape.')
        return BamDeclaration(**value)
    if type(value) is not dict or set(value) != set(FastqDeclaration.__dataclass_fields__):
        raise ValueError('Invalid FASTQ declaration shape.')
    return FastqDeclaration(**dict(value, members=tuple(FastqMember(**m) for m in value['members'])))


def merge(previous, additions):
    if type(previous) is not type(additions):
        raise ValueError('Raw declaration kind cannot change.')
    values = asdict(previous)
    for key in values.keys() - {'members'}:
        value = getattr(additions, key)
        if value is not None:
            if values[key] is not None and values[key] != value:
                raise ValueError('Captured FASTQ declarations cannot be replaced.')
            values[key] = value
    if isinstance(previous, BamDeclaration):
        return BamDeclaration(**values)
    members = {m.source: m for m in previous.members}
    for member in additions.members:
        if member.source in members and members[member.source] != member:
            raise ValueError('Captured FASTQ membership cannot be replaced.')
        members[member.source] = member
    return FastqDeclaration(**dict(values, members=tuple(members.values())))


def schema(sources, resources, field='fastq'):
    from .turn_decisions import _object
    from agent.tools.data._raw_fastq import FastqLayout, ReadRole
    def optional(choices):
        return {'anyOf': [{'type': 'string', 'enum': list(choices)}, {'type': 'null'}]}
    if field == 'bam':
        return _object(dict(source=optional([s['source'] for s in sources]),
            assay=optional(('SCATAC', 'TENX_ATAC')),
            namespace={'anyOf': [{'type': 'string', 'maxLength': 128}, {'type': 'null'}]},
            species=optional(('human', 'mouse')), assembly=optional(('hg38', 'mm10')),
            barcode_policy=optional(('corrected_cb_already_corrected',)),
            source_profile=optional(('agent-cb-paired-atac.v1',)),
            resource=optional([r['handle'] for r in resources]) if resources else {'type': 'null'}))
    return _object(dict(assay=optional(('TENX_ATAC',)), layout=optional((FastqLayout.A.value, FastqLayout.B.value)),
        namespace={'anyOf': [{'type': 'string', 'maxLength': 128}, {'type': 'null'}]},
        species=optional(('human', 'mouse')), barcode_policy=optional(('raw_sequence_whitelist',)),
        resource=optional([r['handle'] for r in resources]) if resources else {'type': 'null'},
        members={'type': 'array', 'maxItems': 16, 'items': _object(dict(
            source={'type': 'string', 'enum': [s['source'] for s in sources]},
            role={'type': 'string', 'enum': [r.value for r in ReadRole]},
            lane={'type': 'string', 'pattern': '^[0-9]{3}$'}, chunk={'type': 'string', 'pattern': '^[0-9]{3}$'}))}))


def resources(input_sets, field='fastq'):
    if field == 'bam':
        return tuple(s for s in input_sets if s.bam_companion and not
                     {'library_context_path', 'library_context_sha256'}.intersection(s.inputs()))
    return tuple(s for s in input_sets if s.fastq_companion and s.fastq_whitelist is not None)


def offers(input_sets, field='fastq'):
    return [dict(handle='resource.' + s.input_set_id, label=s.label, default=getattr(s, field + '_default'))
            for s in resources(input_sets, field)]


def source_binding(origin, source, interaction, field=None):
    field = field or (_field(source) if source is not None else _field(interaction))
    from .dialogue_execution import _role_binding
    # Reuse the existing exact-prefix/explicit-append rule. The scientific
    # collection is distinct from the immutable captured candidate group.
    if source is not None:
        source = replace(source, admitted=dict(source.admitted,
            registered_input=_serialize(source.admitted[field + '_sources']),
            inputs={}))
    return _role_binding(origin, source, interaction)


def requirements(binding, declared, input_sets):
    field = 'bam' if isinstance(declared, BamDeclaration) else 'fastq'
    from types import SimpleNamespace
    from .local_resources import LocalResourceAdmission, ResourceAdmissionError, _fastq_member
    if field == 'bam':
        missing = [label for key, label in (
            ('source', 'Identify the attached BAM for this task.'),
            ('assay', 'Confirm paired scATAC-seq alignments.'),
            ('namespace', 'Name the single library represented by this BAM.'),
            ('species', 'Specify human or mouse.'),
            ('assembly', 'Specify the source assembly: hg38 or mm10.'),
            ('barcode_policy', 'Confirm from its processing history that CB tags are already corrected cell-barcode identifiers.'),
            ('source_profile', 'Confirm unshifted paired ATAC alignments with duplicate pairs retained.'))
            if getattr(declared, key) is None]
        if declared.source is not None and int(declared.source.split('.')[1]) >= len(binding.members):
            raise ValueError('BAM source is outside captured attachments.')
    else:
        missing = [label for key, label in (
            ('assay', 'Confirm the 10x ATAC assay.'), ('layout', 'Specify the FASTQ read layout before assigning missing reads.'),
            ('namespace', 'Name this single library and confirm that its selected lanes/chunks belong together.'),
            ('species', 'Specify human or mouse for this canonical reference workflow.'),
            ('barcode_policy', 'Confirm raw barcode sequences requiring whitelist correction.')) if getattr(declared, key) is None]
        if not declared.members:
            missing.append('Associate the uploaded reads with their read roles, lanes and chunks.')
        for member in declared.members:
            if int(member.source.split('.')[1]) >= len(binding.members):
                raise ValueError('FASTQ member is outside captured attachments.')
        if declared.layout and declared.namespace and declared.members:
            projected = []
            for m in declared.members:
                member = dict(library_id=declared.namespace, fastq_layout=declared.layout,
                              role=m.role, lane=m.lane, chunk=m.chunk, compression='plain')
                projected.append(SimpleNamespace(fastq_member=_fastq_member(member),
                    source_path='/declared/' + m.source))
            try:
                LocalResourceAdmission._fastq_groups(projected)
            except ResourceAdmissionError as exc:
                if exc.code != 'FASTQ_COLLECTION_INCOMPLETE':
                    raise
                missing.append(exc.message)
    choices = resources(input_sets, field)
    selected = None
    if declared.resource is not None:
        selected = next((s for s in choices if 'resource.' + s.input_set_id == declared.resource), None)
        if selected is None:
            raise ValueError('Raw resource is not offered.')
    else:
        defaults = [s for s in choices if getattr(s, field + '_default')]
        if len(defaults) == 1:
            selected = defaults[0]
    if selected is None:
        missing.append('Choose a qualified BAM reference.' if field == 'bam' and choices else
                       'A qualified BAM reference must be provisioned before preprocessing.' if field == 'bam' else
                       'Choose a configured FASTQ reference and barcode whitelist.' if choices else
                       'A qualified FASTQ reference and barcode whitelist must be provisioned before preprocessing.')
    elif any(k not in selected.inputs() for k in ('reference_bundle_path', 'reference_bundle_sha256')):
        missing.append('The configured resource needs a qualified reference bundle before preprocessing.')
    elif declared.species and selected.inputs().get('species') != declared.species:
        raise ValueError('Declared species conflicts with the chosen reference.')
    return missing, selected


def _captured(interaction, preceding):
    from .dialogue_execution import _role_source
    reference = interaction.snapshot.get('prerequisite')
    if reference is not None and reference['field'] in {'fastq', 'bam'}:
        return _role_source(reference, preceding)
    return interaction, None


def _completion(interaction, origin, source, addition, arguments, rows, registry, field='fastq'):
    from .dialogue_execution import _bind_arguments
    from .input_sets import ScientificInputSet
    binding = source_binding(origin, source, interaction, field)
    if not binding.attachment_candidates or binding.roles or binding.execution_inputs:
        raise ValueError('FASTQ declarations require unbound captured attachments.')
    previous = empty(field) if source is None else declaration(_serialize(source.admitted[field]), field)
    declared = merge(previous, addition)
    values = _bind_arguments(registry, interaction.utterance,
        {} if source is None else _serialize(source.admitted[field + '_arguments']), arguments)
    choices = tuple(ScientificInputSet(**_serialize(r)) for r in rows)
    missing, selected = requirements(binding, declared, choices)
    return binding, declared, values, missing, selected


def admit(sessions, interaction, decision, input_sets):
    """The raw declaration variant of existing role request/answer admission."""
    from .turn_decisions import RequestRoles, IntentError
    from .resource_continuation import catalog
    from .dialogue_execution import _arguments
    from .local_resources import LocalResourceAdmission
    field = 'bam' if decision.bam is not None else (interaction.snapshot.get('prerequisite') or {}).get('field', 'fastq')
    if field not in ('fastq', 'bam') or (decision.fastq is not None and field != 'fastq'):
        raise IntentError('invalid_prerequisite')
    validate_arguments(_arguments(decision), field)
    state = sessions.load(sessions._interaction_session_id)
    initial = isinstance(decision, RequestRoles)
    reference = interaction.snapshot.get('prerequisite')
    if (state.generation != interaction.base_generation or state.active_revision_id != interaction.base_revision_id
            or state.interactions[-1].turn_id != interaction.turn_id):
        raise IntentError('invalid_prerequisite')
    if initial:
        origin, source = interaction, None
        rows = catalog(input_sets)
    else:
        if (decision.pending != '@roles' or reference is None or reference['field'] != field
                or digest(catalog(input_sets)) != reference['resource_context_sha256']):
            raise IntentError('invalid_prerequisite')
        origin, source = _captured(interaction, state.interactions[:-1])
        rows = _serialize(origin.admitted['resource_catalog'])
    binding, declared, values, missing, selected = _completion(interaction, origin, source,
        getattr(decision, field) or empty(field), _arguments(decision), rows, sessions._application.registry, field)
    owner = LocalResourceAdmission(sessions._application._workspace, registry=sessions._application.registry)
    owner.validate_binding(binding)
    # Candidate validation intentionally does not open unused attachments.
    records = [owner.load(binding.members[int(h.split('.')[1])]['resource_id']) for h in source_handles(declared)]
    if any(r.session_id not in (None, state.session_id) for r in records):
        raise IntentError('invalid_prerequisite')
    owner._verify_records(records)
    admitted = dict(kind='clarify' if missing else 'execute', tool=target(field),
        revision_id=interaction.base_revision_id, resource_catalog=rows,
        **{field: asdict(declared), field + '_addition': asdict(getattr(decision, field) or empty(field)),
           field + '_sources': binding.attribution(), field + '_arguments': values},
        inputs=values, registered_input=binding.attribution(), roles=[],
        arguments=[asdict(a) for a in _arguments(decision)])
    if not initial:
        admitted['continuation'] = {k: reference[k] for k in ('origin_turn_id', 'binding_turn_id', 'field')}
    pending = None
    if missing:
        admitted.update(reason='missing_' + field + '_declarations', choices=missing, value_required=True)
        pending = dict(origin_turn_id=origin.turn_id, binding_turn_id=interaction.turn_id,
            field=field, tool=target(field), resource_context_sha256=digest(rows))
    else:
        binding, receipt = prepare(owner, state.session_id, binding, declared, selected, values)
        admitted.update(inputs=_serialize(binding.execution_inputs), registered_input=binding.attribution(),
            **{field + '_context': receipt}, operation='plan',
            request_id='turn-' + digest(dict(session=state.session_id, turn=interaction.turn_id)))
    return admitted, pending


def prepare(owner, session_id, candidates, declared, selected, arguments):
    """Coordinate real intake, independent verification and the context owner.

    No provisional authority is published. The normal planned intake/producer
    still execute and verify under RunStore; this preparation admits a request
    resource against the same exact source bytes and intake content identity.
    """
    from agent.tools.data import raw_scatac, raw_scatac_manifest, scatac_library_context as library
    from agent.orchestration.raw_scatac_verifier import verify_raw_scatac_intake
    field = 'bam' if isinstance(declared, BamDeclaration) else 'fastq'
    identity = digest(dict(sources=candidates.attribution(), declaration=asdict(declared),
                           resource=selected.resource_choice(), source_mode='original'))
    directory = owner._workspace._ensure_directory(owner._workspace.root / (field + '_handoffs'))
    root = owner._workspace._ensure_directory(directory / identity)
    if field == 'bam':
        source = owner.load(candidates.members[int(declared.source.split('.')[1])]['resource_id'])
        originals = [source]
        record = owner.bind_raw_source('bam-handoff:' + identity + ':' + declared.source,
            source, session_id=session_id, input_type='bam')
    else:
        originals, typed = [], []
        for member in declared.members:
            source = owner.load(candidates.members[int(member.source.split('.')[1])]['resource_id'])
            originals.append(source)
            record = owner.bind_raw_source('fastq-handoff:' + identity + ':' + member.source,
                source, session_id=session_id, input_type='fastq',
                declaration=dict(library_id=declared.namespace, fastq_layout=declared.layout,
                                 role=member.role, lane=member.lane, chunk=member.chunk))
            typed.append(record)
        owner._verify_records(originals)
        collection = owner.register_fastq_collection('fastq-handoff:' + identity,
            [r.resource_id for r in typed], label='Declared FASTQ library',
            attribution='Session-scoped FASTQ source handoff ' + identity)
    base = selected.inputs()
    if set(base).intersection(arguments) and any(base[k] != arguments[k] for k in set(base).intersection(arguments)):
        raise ValueError('Explicit scientific arguments conflict with configured resources.')
    values = dict(base, **arguments)
    if field == 'bam':
        declarations = dict(species=declared.species, raw_assay=declared.assay, source_genome_assembly=declared.assembly,
                            source_profile=declared.source_profile)
        if any(k in values and values[k] != v for k, v in declarations.items()):
            raise ValueError('BAM declarations conflict with configured resources.')
        values.update(declarations)
        inspection = dict(raw_input_paths=[dict(path=record.source_path, input_kind='BAM')],
            output_dir=str(root / 'intake'), **{k: v for k, v in declarations.items() if k != 'source_profile'})
    else:
        inspection = dict(raw_input_paths=owner._fastq_sources(typed),
            output_dir=str(root / 'intake'), species=declared.species, raw_assay=declared.assay, fastq_layout=declared.layout)
    result = raw_scatac.inspect_raw_scATAC(**inspection)
    verify_raw_scatac_intake(inspection, result, owner._registry)
    _, manifest, _ = raw_scatac_manifest.load_raw_intake_manifest(result['manifest_path'], expected_sha256=result['manifest_sha256'])
    context = library.build_scatac_library_processing_context(
        intake_manifest_path=result['manifest_path'], expected_intake_sha256=result['manifest_sha256'],
        selection_mode=library.SelectionMode.ALL_GROUPS,
        libraries=(library.LibraryDeclaration(namespace=declared.namespace,
            group_ids=tuple(group.id for group in manifest.groups),
            membership_basis=library.MembershipBasis.SINGLE_GROUP if len(manifest.groups) == 1
                else library.MembershipBasis.CALLER_SHARED_LIBRARY,
            barcode_interpretation=library.BarcodeInterpretation.CORRECTED_IDENTIFIER if field == 'bam' else library.BarcodeInterpretation.RAW_SEQUENCE,
            correction_policy=library.CorrectionPolicy.ALREADY_CORRECTED if field == 'bam' else library.CorrectionPolicy.WHITELIST_REQUIRED,
            whitelist=None if field == 'bam' else library._whitelist(_serialize(selected.fastq_whitelist))),))
    destination = root / 'context.json'
    if destination.exists():
        import hashlib
        context_sha = hashlib.sha256(library.canonical_library_processing_context_bytes(context)).hexdigest()
        library.load_scatac_library_processing_context(destination, expected_sha256=context_sha)
        pointer = dict(manifest_path=str(destination), manifest_sha256=context_sha)
    else:
        pointer = library.publish_scatac_library_processing_context(context, destination)
    owner._verify_records(originals)
    values.update(library_context_path=pointer['manifest_path'], library_context_sha256=pointer['manifest_sha256'])
    binding = (owner.compose_bam(record.resource_id, values, resource_choice=selected.resource_choice()) if field == 'bam'
               else owner.compose_fastq_collection(collection.resource_id, values, resource_choice=selected.resource_choice()))
    return binding, dict(source_mode='original', identity=identity, intake_path=result['manifest_path'], intake_sha256=result['manifest_sha256'],
                         context_path=pointer['manifest_path'], context_sha256=pointer['manifest_sha256'])


def public(interaction, preceding):
    field = _field(interaction)
    origin, source = _captured(interaction, preceding)
    return dict(handle='@roles', field=field, tool=target(field),
        objective=origin.utterance, declarations=_serialize(source.admitted[field]),
        missing=list(source.admitted['choices']))


def validate(interaction, preceding):
    """Replay declarations from durable records; history reads perform no IO."""
    from .session_state import SessionError
    from .dialogue_execution import capture_prerequisite
    from .turn_decisions import _scientific_arguments
    from .local_resources import LocalResourceAdmission
    from .resource_continuation import catalog
    from .input_sets import ScientificInputSet
    from agent.orchestration.registry import build_default_tool_registry
    field = _field(interaction)
    try:
        captured = _serialize(interaction.snapshot)
        expected = capture_prerequisite(preceding, interaction.base_revision_id,
            interaction.base_generation, interaction.submission, captured.get('dialogue', {}).get('requested_predecessor'))
        if captured.get('prerequisite') != expected:
            raise ValueError('FASTQ prerequisite capture changed.')
        admitted = _serialize(interaction.admitted or {})
        if field not in admitted:
            # Received/interpreting or an ordinary clarification preserves the question.
            if interaction.prerequisite is not None and _serialize(interaction.prerequisite) != {
                    k: v for k, v in expected.items() if k != 'predecessor_turn_id'}:
                raise ValueError('FASTQ question lost its captured scope.')
            return
        origin, source = _captured(interaction, preceding)
        rows = admitted['resource_catalog'] if source is None else _serialize(origin.admitted['resource_catalog'])
        if (admitted['resource_catalog'] != rows
                or catalog(tuple(ScientificInputSet(**r) for r in rows)) != rows):
            raise ValueError('FASTQ resource catalog changed.')
        binding, declared, values, missing, selected = _completion(interaction, origin, source,
            declaration(admitted[field + '_addition'], field), _scientific_arguments(admitted['arguments']),
            rows, build_default_tool_registry(), field)
        if (admitted[field] != _serialize(asdict(declared)) or admitted[field + '_sources'] != binding.attribution()
                or admitted[field + '_arguments'] != values or admitted['tool'] != target(field)
                or admitted['revision_id'] != interaction.base_revision_id or admitted['roles']):
            raise ValueError('FASTQ declarations changed.')
        if source is not None and admitted.get('continuation') != {
                k: expected[k] for k in ('origin_turn_id', 'binding_turn_id', 'field')}:
            raise ValueError('FASTQ continuation changed.')
        if missing:
            pending = dict(origin_turn_id=origin.turn_id, binding_turn_id=interaction.turn_id,
                field=field, tool=target(field), resource_context_sha256=digest(rows))
            if (admitted['kind'] != 'clarify' or admitted['choices'] != missing
                    or admitted['inputs'] != values or admitted['registered_input'] != binding.attribution()
                    or _serialize(interaction.prerequisite) != pending or field + '_context' in admitted):
                raise ValueError('FASTQ missing requirements changed.')
        else:
            receipt = admitted[field + '_context']
            if (admitted['kind'] != 'execute' or admitted.get('operation') != 'plan'
                    or (interaction.prerequisite is not None and interaction.prerequisite['field'] != 'parameters') or set(receipt) - {'source_mode'} != {
                        'identity', 'intake_path', 'intake_sha256', 'context_path', 'context_sha256'}):
                raise ValueError('Invalid completed FASTQ binding.')
            from .session_state import sha
            for key in ('identity', 'intake_sha256', 'context_sha256'):
                sha(receipt[key])
            mode = receipt.get('source_mode')
            if mode not in (None, 'original'):
                raise ValueError('Unsupported FASTQ source mode.')
            identity = digest(dict(sources=binding.attribution(), declaration=asdict(declared),
                resource=selected.resource_choice(), **({'source_mode': mode} if mode else {})))
            root = Path(receipt['context_path']).parent
            if (receipt['identity'] != identity or root.name != identity or root.parent.name != field + '_handoffs'
                    or not root.is_absolute() or '..' in root.parts
                    or receipt['context_path'] != str(root / 'context.json')
                    or receipt['intake_path'] != str(root / 'intake' / ('raw-scatac-intake-' + receipt['intake_sha256'] + '.json'))):
                raise ValueError('FASTQ context/source identity changed.')
            bound = LocalResourceAdmission._submission_binding(dict(execution_inputs=admitted['inputs'],
                registered_input=admitted['registered_input']))
            if field == 'bam':
                original = binding.members[int(declared.source.split('.')[1])]
                wanted = selected.inputs() | values | dict(
                    species=declared.species, raw_assay=declared.assay, source_genome_assembly=declared.assembly,
                    source_profile=declared.source_profile,
                    source_path=original['source_path'], source_sha256=original['source_sha256'],
                    raw_input_paths=[dict(path=original['source_path'], input_kind='BAM')],
                    library_context_path=receipt['context_path'], library_context_sha256=receipt['context_sha256'])
                if (mode != 'original' or bound.composition != 'bam-science.v1'
                        or bound.resource_id != LocalResourceAdmission.registration_id('bam-handoff:' + identity + ':' + declared.source)
                        or _serialize(bound.resource_choice) != selected.resource_choice() or admitted['inputs'] != wanted):
                    raise ValueError('Derived BAM binding changed.')
            else:
                if sorted(m['resource_id'] for m in bound.members) != sorted([
                        LocalResourceAdmission.registration_id('fastq-handoff:' + identity + ':' + m.source)
                        for m in declared.members]):
                    raise ValueError('FASTQ derived source identities changed.')
                raw_inputs = admitted['inputs']['raw_input_paths']
                if mode == 'original':
                    from agent.tools.data._raw_fastq import explicit_sources
                    expected_sources = explicit_sources([dict(
                        path=binding.members[int(m.source.split('.')[1])]['source_path'],
                        library_id=declared.namespace, role=m.role, lane=m.lane, chunk=m.chunk)
                        for m in declared.members])
                    if raw_inputs != expected_sources:
                        raise ValueError('FASTQ original-source declarations changed.')
                else:
                    # Read-only interpretation of accepted USI-2B materialized views.
                    from .uploads import H5ADUploadAdmission
                    allowed = []
                    for member in declared.members:
                        technical = dict(library_id=declared.namespace, fastq_layout=declared.layout,
                                         role=member.role, lane=member.lane, chunk=member.chunk)
                        allowed.append({str(root / 'sources' / H5ADUploadAdmission._fastq_name(dict(technical, compression=c)))
                                        for c in ('plain', 'gzip')})
                    if (len(raw_inputs) != len(allowed) or sorted(raw_inputs) != raw_inputs
                            or any(len(set(raw_inputs) & paths) != 1 for paths in allowed)):
                        raise ValueError('Historical FASTQ source locators changed.')
                wanted = selected.inputs() | values | dict(library_context_path=receipt['context_path'],
                    library_context_sha256=receipt['context_sha256'], raw_assay=declared.assay, fastq_layout=declared.layout)
                if (bound.composition != 'fastq-science.v1' or len(bound.members) != len(declared.members)
                        or _serialize(bound.resource_choice) != selected.resource_choice()
                        or {k: v for k, v in admitted['inputs'].items() if k != 'raw_input_paths'} != wanted):
                    raise ValueError('Derived FASTQ binding changed.')
    except (ValueError, TypeError, KeyError, AttributeError, StopIteration) as exc:
        raise SessionError('Invalid task-scoped raw handoff.') from exc
