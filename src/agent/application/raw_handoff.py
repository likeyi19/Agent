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


@dataclass(frozen=True)
class FragmentsDeclaration:
    """External source history and optional user-selected region vocabulary."""
    source: str | None = None
    index_source: str | None = None
    namespace: str | None = None
    species: str | None = None
    assembly: str | None = None
    source_profile: str | None = None
    resource: str | None = None
    feature_source: str | None = None
    feature_category: str | None = None
    genome_source: str | None = None
    genome_index_source: str | None = None

    def __post_init__(self):
        from agent.tools.data.scatac_library_context import _namespace
        from agent.tools.data.external_fragment_manifest import PROFILE_ID
        for value in (self.source, self.index_source, self.feature_source, self.genome_source, self.genome_index_source):
            if value is not None and (type(value) is not str or re.fullmatch(r'source\.(?:[0-9]|1[0-5])', value) is None):
                raise ValueError('Invalid external source handle.')
        handles = source_handles(self)
        if len(handles) != len(set(handles)):
            raise ValueError('Fragment, index and feature roles require distinct files.')
        if self.namespace is not None:
            _namespace(self.namespace)
        if (self.species not in (None, 'human', 'mouse') or self.assembly not in (None, 'hg38', 'mm10')
                or self.source_profile not in (None, PROFILE_ID)
                or self.feature_category not in (None, 'peak_set', 'regulatory_regions')):
            raise ValueError('Unsupported external source declaration.')
        if self.species and self.assembly and {'human': 'hg38', 'mouse': 'mm10'}[self.species] != self.assembly:
            raise ValueError('Fragment species/assembly declarations conflict.')
        if self.resource is not None and (type(self.resource) is not str or re.fullmatch(r'resource\.[a-z][a-z0-9_-]{0,63}', self.resource) is None):
            raise ValueError('Invalid external resource handle.')


def target(field):
    return {'fragments': 'import_scATAC_fragments', 'bam': 'prepare_scATAC_bam_fragments', 'fastq': 'prepare_scATAC_fragments'}[field]


def empty(field):
    return {'fragments': FragmentsDeclaration, 'bam': BamDeclaration, 'fastq': FastqDeclaration}[field]()


def source_handles(declared):
    if isinstance(declared, FragmentsDeclaration):
        return [h for h in (declared.source, declared.index_source, declared.feature_source, declared.genome_source, declared.genome_index_source) if h is not None]
    return ([declared.source] if declared.source else []) if isinstance(declared, BamDeclaration) else [m.source for m in declared.members]


def _field(interaction):
    admitted = interaction.admitted or {}
    return 'fragments' if 'fragments' in admitted else 'bam' if 'bam' in admitted else 'fastq' if 'fastq' in admitted else (interaction.snapshot.get('prerequisite') or {}).get('field', 'fastq')


def validate_arguments(arguments, field='fastq'):
    from .local_resources import _FASTQ_COMPANION_TOOLS, _BAM_COMPANION_TOOLS, _FRAGMENTS_COMPANION_TOOLS
    allowed = {'fragments': _FRAGMENTS_COMPANION_TOOLS, 'bam': _BAM_COMPANION_TOOLS, 'fastq': _FASTQ_COMPANION_TOOLS}[field]
    if any(a.tool not in allowed for a in arguments):
        raise ValueError('FASTQ task arguments must belong to its registered scientific consumers.')


def declaration(value, field='fastq'):
    if field == 'fragments':
        if type(value) is not dict or set(value) != set(FragmentsDeclaration.__dataclass_fields__):
            raise ValueError('Invalid fragment declaration shape.')
        return FragmentsDeclaration(**value)
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
            if key != 'resource' and values[key] is not None and values[key] != value:
                raise ValueError('Captured FASTQ declarations cannot be replaced.')
            values[key] = value
    if isinstance(previous, (BamDeclaration, FragmentsDeclaration)):
        return type(previous)(**values)
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
    if field == 'fragments':
        return _object(dict(source=optional([s['source'] for s in sources]),
            index_source=optional([s['source'] for s in sources]),
            feature_source=optional([s['source'] for s in sources]),
            genome_source=optional([s['source'] for s in sources]),
            genome_index_source=optional([s['source'] for s in sources]),
            namespace={'anyOf': [{'type': 'string', 'maxLength': 128}, {'type': 'null'}]},
            species=optional(('human', 'mouse')), assembly=optional(('hg38', 'mm10')),
            source_profile=optional(('10x-atac-fragments.v1',)),
            feature_category=optional(('peak_set', 'regulatory_regions')),
            resource=optional([r['handle'] for r in resources]) if resources else {'type': 'null'}))
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
    if field == 'fragments':
        return tuple(s for s in input_sets if s.fragments_companion)
    if field == 'bam':
        return tuple(s for s in input_sets if s.bam_companion and not
                     {'library_context_path', 'library_context_sha256'}.intersection(s.inputs()))
    return tuple(s for s in input_sets if s.fastq_companion and s.fastq_whitelist is not None)


def offers(input_sets, field='fastq'):
    return [dict(handle='resource.' + s.input_set_id, label=s.label, default=getattr(s, field + '_default'))
            for s in resources(input_sets, field)]


def resource_evidence(input_sets, field):
    """Read pinned owner manifests, not reference source data or a second catalog.

    These are admission observations, never scientific execution authority.
    Source freshness and full qualification remain with the consuming owners.
    Persisting the owner contract permits history replay without scientific IO.
    """
    from agent.tools.data import scatac_reference as reference
    result = {}
    for choice in resources(input_sets, field):
        values = choice.inputs()
        try:
            if not values.get('reference_bundle_sha256'):
                raise ValueError('A reference content pin is required.')
            if field == 'fragments':
                from agent.tools.data._fragment_reference import load
                bundle = load(values.get('reference_bundle_path'), values['reference_bundle_sha256'])
                result[choice.input_set_id] = getattr(bundle, 'bundle', bundle).to_dict()
            else:
                _, bundle, _ = reference.load_scatac_reference_bundle(
                    values.get('reference_bundle_path'), expected_sha256=values['reference_bundle_sha256'])
                result[choice.input_set_id] = bundle.to_dict()
        except (ValueError, OSError, TypeError):
            result[choice.input_set_id] = None
    return result


def _resolve_resource(declared, choices, evidence, pinned=None):
    """Deterministic choice over owner-qualified contracts; no linguistic rules."""
    from agent.tools.data import scatac_reference as reference
    if set(evidence) != {s.input_set_id for s in choices}:
        raise ValueError('Resource observations do not match the captured catalog.')
    candidates, descriptions = [], {}
    for choice in choices:
        raw = evidence[choice.input_set_id]
        if raw is None:
            descriptions[choice.input_set_id] = 'its pinned reference is unavailable or invalid'
            continue
        if isinstance(declared, FragmentsDeclaration) and raw.get('contract_version') == 'regulatory-feature-reference.v1':
            from agent.tools.data.regulatory_feature_reference import validate_regulatory_feature_reference
            from agent.tools.data.regulatory_matrix_contract import _MatrixReference
            bundle = _MatrixReference(validate_regulatory_feature_reference(raw))
        else:
            bundle = reference.validate_scatac_reference_bundle(raw)
        sample_species = (reference.species_identity(declared.species)
                          if type(bundle.species) is dict and declared.species is not None else declared.species)
        values = choice.inputs()
        if ((values.get('species', bundle.species) != bundle.species)
                or (sample_species is not None and sample_species != bundle.species)
                or (isinstance(declared, (BamDeclaration, FragmentsDeclaration)) and declared.assembly is not None
                    and declared.assembly != bundle.target_assembly)):
            descriptions[choice.input_set_id] = 'its species or assembly conflicts with this task'
            continue
        descriptions[choice.input_set_id] = f'{bundle.species}, {bundle.target_assembly}, {bundle.ccre.feature_count} reference features'
        candidates.append(choice)
    requested = declared.resource or pinned
    if requested is not None:
        selected = next((s for s in choices if 'resource.' + s.input_set_id == requested), None)
        if selected is None:
            return ['The selected resource is unavailable. Supply it or explicitly choose an alternative.'], None
        if selected not in candidates:
            return [f'Cannot use {selected.label}: {descriptions[selected.input_set_id]}. Supply a corrected resource or explicitly choose an alternative.'], None
        return [], selected
    # An available reference cannot establish the sample's species/alignment history.
    if declared.species is None or isinstance(declared, (BamDeclaration, FragmentsDeclaration)) and declared.assembly is None:
        return [], None
    if not candidates:
        needed = ('A qualified fragments reference compatible with the declared species and assembly'
                  if isinstance(declared, FragmentsDeclaration) else 'A qualified BAM reference compatible with the declared species and assembly'
                  if isinstance(declared, (BamDeclaration, FragmentsDeclaration)) else
                  'A qualified FASTQ reference and barcode whitelist compatible with the declared species')
        return [needed + ' must be provisioned before preprocessing.'
                + (' Configured references are unavailable, invalid or incompatible.' if choices else '')], None
    field = 'fragments' if isinstance(declared, FragmentsDeclaration) else 'bam' if isinstance(declared, BamDeclaration) else 'fastq'
    defaults = [s for s in candidates if getattr(s, field + '_default')]
    if len(defaults) == 1:
        return [], defaults[0]
    # Identical mappings/whitelists differ only in display/configuration identity.
    # Stable ID order is an explicit tie policy for these equivalent aliases only.
    def equivalence(choice):
        values = choice.inputs()
        # The same pinned manifest at another locator is the same dependency.
        # All other scientific arguments and provenance remain significant.
        values.pop('reference_bundle_path')
        values.setdefault('species', evidence[choice.input_set_id]['species'])
        return digest(values), digest(_serialize(choice.fastq_whitelist))
    equivalent = {}
    for choice in candidates:
        key = equivalence(choice)
        equivalent[key] = min((choice, equivalent.get(key, choice)), key=lambda s: s.input_set_id)
    candidates = list(equivalent.values())
    if len(candidates) == 1:
        return [], candidates[0]
    return ['Choose a reference: ' + '; '.join(f'{s.label} ({descriptions[s.input_set_id]})' for s in candidates) + '.'], None


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


def requirements(binding, declared, input_sets, *, evidence=None, pinned=None):
    field = 'fragments' if isinstance(declared, FragmentsDeclaration) else 'bam' if isinstance(declared, BamDeclaration) else 'fastq'
    from types import SimpleNamespace
    from .local_resources import LocalResourceAdmission, ResourceAdmissionError, _fastq_member
    if field == 'fragments':
        missing = [label for key, label in (
            ('source', 'Identify the external fragments attachment.'),
            ('namespace', 'Name the single processing library represented by these fragments.'),
            ('species', 'Specify human or mouse.'), ('assembly', 'Specify the source assembly: hg38 or mm10.'),
            ('source_profile', 'Confirm 10x ATAC/ARC fragment semantics: zero-based half-open adjusted intervals, corrected barcode identifiers and read-pair support.'))
            if getattr(declared, key) is None]
        if any(int(h.split('.')[1]) >= len(binding.members) for h in source_handles(declared)):
            raise ValueError('Fragment role is outside captured attachments.')
        if declared.feature_category is not None and declared.feature_source is None:
            missing.append('Attach the custom BED to define the exact matrix regions; its intervals must use the declared assembly and zero-based half-open coordinates.')
        if declared.genome_index_source is not None and declared.genome_source is None:
            missing.append('Identify the plain FASTA paired with the supplied FAI.')
        if declared.genome_source is not None and declared.feature_source is None:
            missing.append('Supply a BED of the declared regions for existing feature-reference qualification.')
        if declared.feature_source is not None and declared.feature_category is None:
            missing.append('Declare whether the supplied zero-based half-open BED defines peaks or regulatory regions in the source assembly.')
    elif field == 'bam':
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
    if evidence is not None:
        questions, selected = _resolve_resource(declared, choices, evidence, pinned)
        if field == 'fragments' and not choices and declared.resource is None and pinned is None:
            # A private BED/genome reference uses the existing reference owner.
            # It cannot silently replace a captured or offered resource choice.
            if declared.feature_category is not None or declared.feature_source is not None or declared.genome_source is not None:
                questions = []
                if declared.genome_source is None:
                    questions.append('Attach the plain genome FASTA for the declared species and assembly to qualify these regions and fragments. An FAI is optional; Agent can prepare it.')
        return missing + questions, selected
    # Bounded reader for accepted USI-2 records, whose default-only policy and
    # exact identities must not be reinterpreted by current resource selection.
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
    if reference is not None and reference['field'] in {'fastq', 'bam', 'fragments'}:
        return _role_source(reference, preceding)
    return interaction, None


def _extend_catalog(previous, current):
    """Supplement a pending task without changing any captured definition."""
    offered = {row['input_set_id']: row for row in current}
    if any(offered.get(row['input_set_id']) != row for row in previous):
        raise ValueError('Captured scientific resource definitions cannot be replaced.')
    return current


def _completion(interaction, origin, source, addition, arguments, rows, registry, field='fastq', evidence=None):
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
    pinned = None if source is None else source.admitted.get('raw_resource_choice')
    missing, selected = requirements(binding, declared, choices, evidence=evidence, pinned=pinned)
    return binding, declared, values, missing, selected


def admit(sessions, interaction, decision, input_sets):
    """The raw declaration variant of existing role request/answer admission."""
    from .turn_decisions import RequestRoles, IntentError
    from .resource_continuation import catalog
    from .dialogue_execution import _arguments
    from .local_resources import LocalResourceAdmission
    field = 'fragments' if decision.fragments is not None else 'bam' if decision.bam is not None else (interaction.snapshot.get('prerequisite') or {}).get('field', 'fastq')
    if field not in ('fastq', 'bam', 'fragments') or (decision.fastq is not None and field != 'fastq'):
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
        if decision.pending != '@roles' or reference is None or reference['field'] != field:
            raise IntentError('invalid_prerequisite')
        origin, source = _captured(interaction, state.interactions[:-1])
        previous_rows = _serialize(source.admitted['resource_catalog'])
        if digest(previous_rows) != reference['resource_context_sha256']:
            raise IntentError('invalid_prerequisite')
        rows = _extend_catalog(previous_rows, catalog(input_sets))
    evidence = resource_evidence(input_sets, field)
    binding, declared, values, missing, selected = _completion(interaction, origin, source,
        getattr(decision, field) or empty(field), _arguments(decision), rows, sessions._application.registry, field, evidence)
    owner = LocalResourceAdmission(sessions._application._workspace, registry=sessions._application.registry)
    owner.validate_binding(binding)
    # Candidate validation intentionally does not open unused attachments.
    records = [owner.load(binding.members[int(h.split('.')[1])]['resource_id']) for h in source_handles(declared)]
    if any(r.session_id not in (None, state.session_id) for r in records):
        raise IntentError('invalid_prerequisite')
    owner._verify_records(records)
    admitted = dict(kind='clarify' if missing else 'execute', tool=target(field),
        revision_id=interaction.base_revision_id, resource_catalog=rows, raw_resource_evidence=evidence,
        raw_resource_choice=(declared.resource or ('resource.' + selected.input_set_id if selected is not None
            else source.admitted.get('raw_resource_choice') if source is not None else None)),
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
        binding, receipt = prepare(owner, state.session_id, binding, declared, selected, values,
            accepted_contexts=tuple(i.admitted[field + '_context'] for i in state.interactions[:-1]
                if i.admitted and field + '_context' in i.admitted))
        admitted.update(inputs=_serialize(binding.execution_inputs), registered_input=binding.attribution(),
            **{field + '_context': receipt}, operation='plan',
            request_id='turn-' + digest(dict(session=state.session_id, turn=interaction.turn_id)))
    return admitted, pending


def _context_identity(candidates, declared, selected, *, mode='original', scope=None):
    declaration = asdict(declared)
    if scope == 'library.v1':
        # Reference selection is independent of library/barcode processing.
        # Keep the genuine whitelist dependency; never reuse across libraries.
        declaration.pop('resource')
        return digest(dict(sources=candidates.attribution(), declaration=declaration,
                           whitelist=_serialize(selected.fastq_whitelist), identity_scope=scope))
    if scope is not None:
        raise ValueError('Unsupported library identity scope.')
    return digest(dict(sources=candidates.attribution(), declaration=declaration,
                       resource=selected.resource_choice(), **({'source_mode': mode} if mode else {})))


def prepare(owner, session_id, candidates, declared, selected, arguments, *, accepted_contexts=()):
    """Coordinate real intake, independent verification and the context owner.

    No provisional authority is published. The normal planned intake/producer
    still execute and verify under RunStore; this preparation admits a request
    resource against the same exact source bytes and intake content identity.
    """
    if isinstance(declared, FragmentsDeclaration):
        return _prepare_fragments(owner, session_id, candidates, declared, selected, arguments)
    from agent.tools.data import raw_scatac, raw_scatac_manifest, scatac_library_context as library
    from agent.orchestration.raw_scatac_verifier import verify_raw_scatac_intake
    field = 'bam' if isinstance(declared, BamDeclaration) else 'fastq'
    identity = _context_identity(candidates, declared, selected, scope='library.v1')
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
    values.setdefault('species', declared.species)
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
    accepted = {_serialize(r)['context_sha256']: _serialize(r) for r in accepted_contexts
                if r['identity'] == identity and r.get('identity_scope') == 'library.v1'}
    if len(accepted) > 1:
        raise ValueError('Conflicting accepted library context identities.')
    if accepted:
        receipt, = accepted.values()
        _, context, _ = library.load_scatac_library_processing_context(
            receipt['context_path'], expected_sha256=receipt['context_sha256'])
        library.validate_library_context_intake_binding(context)
        # Reuse exact admitted content, with fresh source verification. Do not
        # invoke intake production or rebuild the already qualified context.
        _, manifest, _ = raw_scatac_manifest.load_raw_intake_manifest(
            receipt['intake_path'], expected_sha256=receipt['intake_sha256'])
        result = raw_scatac._summary(manifest, Path(receipt['intake_path']), receipt['intake_sha256'])
    else:
        result = raw_scatac.inspect_raw_scATAC(**inspection)
    verify_raw_scatac_intake(inspection, result, owner._registry)
    _, manifest, _ = raw_scatac_manifest.load_raw_intake_manifest(result['manifest_path'], expected_sha256=result['manifest_sha256'])
    if not accepted:
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
    return binding, dict(source_mode='original', identity_scope='library.v1', identity=identity, intake_path=result['manifest_path'], intake_sha256=result['manifest_sha256'],
                         context_path=pointer['manifest_path'], context_sha256=pointer['manifest_sha256'])


def _fragments_identity(candidates, declared, selected):
    return digest(dict(sources=candidates.attribution(), declaration=asdict(declared),
                       resource=None if selected is None else selected.resource_choice()))


def _fragments_values(declared, selected, arguments, reference):
    values = {} if selected is None else selected.inputs()
    if selected is None:
        values.update(reference_bundle_path=reference['manifest_path'], reference_bundle_sha256=reference['manifest_sha256'])
    # Sample declarations belong to the task, not to a reusable reference.
    declarations = dict(namespace=declared.namespace, source_profile=declared.source_profile)
    if any(key in arguments and arguments[key] != value for key, value in declarations.items()):
        raise ValueError('Scientific arguments cannot replace captured fragment declarations.')
    for key, value in (declarations | arguments).items():
        if key in values and values[key] != value:
            raise ValueError('Explicit fragment declarations conflict with configured resources.')
        values[key] = value
    if reference is not None:
        values.update(reference_manifest_path=reference['manifest_path'],
                      reference_manifest_sha256=reference['manifest_sha256'])
    return values


def _prepare_fragments(owner, session_id, candidates, declared, selected, arguments):
    """Existing import binding plus optional owner-qualified private BED resource."""
    from .reference_supplementation import qualify_bed
    identity = _fragments_identity(candidates, declared, selected)
    records = {h: owner.load(candidates.members[int(h.split('.')[1])]['resource_id'])
               for h in source_handles(declared)}
    reference = None
    if declared.feature_source is not None:
        root = owner._workspace._ensure_directory(owner._workspace.root / 'fragments_handoffs')
        directory = owner._workspace._ensure_directory(root / identity)
        values = {} if selected is None else selected.inputs()
        fasta = None
        if declared.genome_source is not None:
            fasta = owner.bind_raw_source('fragments-genome:' + identity, records[declared.genome_source],
                session_id=session_id, input_type='fasta', source_index=records.get(declared.genome_index_source))
        from agent.tools.data.scatac_reference import ScATACReferenceError, species_identity
        from .reference_supplementation import ReferenceSupplementError
        try:
            reference = qualify_bed(owner, records[declared.feature_source],
                parent_path=values.get('reference_bundle_path'), parent_sha256=values.get('reference_bundle_sha256'),
                category=declared.feature_category, directory=directory, fasta=fasta,
                species=species_identity(declared.species), assembly=declared.assembly)
        except ScATACReferenceError as exc:
            raise ReferenceSupplementError('The supplied BED or its reference genome failed owner qualification.') from exc
    record = owner.bind_raw_source('fragments-handoff:' + identity, records[declared.source],
        session_id=session_id, input_type='external_fragments', source_index=records.get(declared.index_source))
    binding = owner.compose_fragments(record.resource_id,
        _fragments_values(declared, selected, arguments, reference),
        resource_choice=None if selected is None else selected.resource_choice())
    owner._verify_records(list(records.values()))
    return binding, dict(identity=identity, feature_reference=reference)


def _validate_fragments_completion(interaction, admitted, candidates, declared, selected, arguments):
    """Pure historical replay; owner resources are checked at new consumption."""
    from .local_resources import LocalResourceAdmission
    from .session_state import sha
    receipt = admitted['fragments_context']
    identity = _fragments_identity(candidates, declared, selected)
    if (set(receipt) != {'identity', 'feature_reference'} or receipt['identity'] != identity
            or admitted['kind'] != 'execute' or admitted.get('operation') != 'plan'
            or (interaction.prerequisite is not None and interaction.prerequisite['field'] != 'parameters')):
        raise ValueError('Invalid completed fragments handoff.')
    pointer = receipt['feature_reference']
    if (pointer is None) != (declared.feature_source is None):
        raise ValueError('Custom feature binding changed.')
    if pointer is not None:
        if set(pointer) != {'manifest_path', 'manifest_sha256'}:
            raise ValueError('Invalid custom feature pointer.')
        sha(pointer['manifest_sha256'])
        path = Path(pointer['manifest_path'])
        if (not path.is_absolute() or '..' in path.parts or path.name != 'reference.json'
                or path.parent.name != identity or path.parent.parent.name != 'fragments_handoffs'):
            raise ValueError('Custom feature identity changed.')
    source = candidates.members[int(declared.source.split('.')[1])]
    wanted = _fragments_values(declared, selected, arguments, pointer) | dict(
        source_path=source['source_path'], source_sha256=source['source_sha256'])
    if declared.index_source is not None:
        index = candidates.members[int(declared.index_source.split('.')[1])]
        wanted.update(source_index_path=index['source_path'], source_index_sha256=index['source_sha256'])
    bound = LocalResourceAdmission._submission_binding(dict(execution_inputs=admitted['inputs'],
        registered_input=admitted['registered_input']))
    if (bound.composition != 'external-fragments-science.v1'
            or bound.resource_id != LocalResourceAdmission.registration_id('fragments-handoff:' + identity)
            or _serialize(bound.resource_choice) != (None if selected is None else selected.resource_choice())
            or admitted['inputs'] != wanted):
        raise ValueError('Derived fragments binding changed.')


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
        rows = admitted['resource_catalog']
        evidence = admitted.get('raw_resource_evidence')
        if (('raw_resource_evidence' in admitted) != ('raw_resource_choice' in admitted)
                or 'raw_resource_evidence' in admitted and type(evidence) is not dict):
            raise ValueError('Invalid resource admission observations.')
        previous_rows = None if source is None else _serialize(source.admitted['resource_catalog'])
        if (catalog(tuple(ScientificInputSet(**r) for r in rows)) != rows
                or previous_rows is not None and (digest(previous_rows) != expected['resource_context_sha256']
                    or (rows != previous_rows if evidence is None else _extend_catalog(previous_rows, rows) != rows))):
            raise ValueError('FASTQ resource catalog changed.')
        binding, declared, values, missing, selected = _completion(interaction, origin, source,
            declaration(admitted[field + '_addition'], field), _scientific_arguments(admitted['arguments']),
            rows, build_default_tool_registry(), field, evidence)
        if evidence is not None and admitted.get('raw_resource_choice') != (declared.resource or
                ('resource.' + selected.input_set_id if selected is not None else
                 source.admitted.get('raw_resource_choice') if source is not None else None)):
            raise ValueError('Captured resource selection changed.')
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
        elif field == 'fragments':
            _validate_fragments_completion(interaction, admitted, binding, declared, selected, values)
        else:
            receipt = admitted[field + '_context']
            if (admitted['kind'] != 'execute' or admitted.get('operation') != 'plan'
                    or (receipt.get('identity_scope') == 'library.v1') != (evidence is not None)
                    or (interaction.prerequisite is not None and interaction.prerequisite['field'] != 'parameters') or set(receipt) - {'source_mode', 'identity_scope'} != {
                        'identity', 'intake_path', 'intake_sha256', 'context_path', 'context_sha256'}):
                raise ValueError('Invalid completed FASTQ binding.')
            from .session_state import sha
            for key in ('identity', 'intake_sha256', 'context_sha256'):
                sha(receipt[key])
            mode = receipt.get('source_mode')
            if mode not in (None, 'original'):
                raise ValueError('Unsupported FASTQ source mode.')
            identity = _context_identity(binding, declared, selected, mode=mode, scope=receipt.get('identity_scope'))
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
                if evidence is not None:
                    wanted.setdefault('species', declared.species)
                if (bound.composition != 'fastq-science.v1' or len(bound.members) != len(declared.members)
                        or _serialize(bound.resource_choice) != selected.resource_choice()
                        or {k: v for k, v in admitted['inputs'].items() if k != 'raw_input_paths'} != wanted):
                    raise ValueError('Derived FASTQ binding changed.')
    except (ValueError, TypeError, KeyError, AttributeError, StopIteration) as exc:
        raise SessionError('Invalid task-scoped raw handoff.') from exc
