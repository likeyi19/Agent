"""Task-scoped source declarations; scientific qualification remains with owners.

These are bounded variants of existing role continuation, not a scientific planner.
Raw file roles use exact captured attachments. Current annotation capture and
selected-method preparation delegate to annotation_preparation; shared declaration
types and narrowly scoped historical readers remain here.
"""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import re

from agent.schemas.orchestration import _serialize
from .session_state import digest
from .epizoo_handoff import EpiZooDeclaration



def _context_text(*values):
    from agent.tools._file_identity import identifier
    for value in values:
        if value is not None:
            identifier(value)
            if len(value) > 128: raise ValueError('Source declaration exceeds its bound')


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
    resource_preference: dict | None = None
    members: tuple[FastqMember, ...] = ()

    def __post_init__(self):
        from .resource_continuation import validate_preference
        validate_preference(self.resource_preference)
        from agent.tools.data._raw_fastq import FastqLayout
        from agent.tools.data.scatac_library_context import _namespace
        if self.assay not in (None, 'TENX_ATAC') or self.layout not in (None, FastqLayout.A.value, FastqLayout.B.value):
            raise ValueError('Unsupported FASTQ assay/layout declaration.')
        if self.namespace is not None:
            _namespace(self.namespace)
            if len(self.namespace) > 64:
                raise ValueError('Library namespace exceeds the member admission bound.')
        _context_text(self.species)
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
    resource_preference: dict | None = None

    def __post_init__(self):
        from .resource_continuation import validate_preference
        validate_preference(self.resource_preference)
        from agent.tools.data.scatac_library_context import _namespace
        if self.source is not None and (type(self.source) is not str or re.fullmatch(r'source\.(?:[0-9]|1[0-5])', self.source) is None):
            raise ValueError('Invalid BAM source handle.')
        if self.namespace is not None:
            _namespace(self.namespace)
        for value, choices in ((self.assay, ('SCATAC', 'TENX_ATAC')),
                (self.barcode_policy, ('corrected_cb_already_corrected',)),
                (self.source_profile, ('agent-cb-paired-atac.v1', 'agent-cb-paired-atac.v2'))):
            if value is not None and value not in choices:
                raise ValueError('Unsupported BAM scientific declaration.')
        _context_text(self.species, self.assembly)
        if (self.species,self.assembly) in (('human','mm10'),('mouse','hg38')):
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
    resource_preference: dict | None = None
    feature_source: str | None = None
    feature_category: str | None = None
    genome_source: str | None = None
    genome_index_source: str | None = None

    def __post_init__(self):
        from .resource_continuation import validate_preference
        validate_preference(self.resource_preference)
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
        if (self.source_profile not in (None, PROFILE_ID)
                or self.feature_category not in (None, 'peak_set', 'regulatory_regions')):
            raise ValueError('Unsupported external source declaration.')
        _context_text(self.species, self.assembly)
        if (self.species,self.assembly) in (('human','mm10'),('mouse','hg38')):
            raise ValueError('Fragment species/assembly declarations conflict.')
        if self.resource is not None and (type(self.resource) is not str or re.fullmatch(r'resource\.[a-z][a-z0-9_-]{0,63}', self.resource) is None):
            raise ValueError('Invalid external resource handle.')


@dataclass(frozen=True)
class AnnotationDeclaration:
    """Source history and biological context for the existing annotation owner."""
    source: str | None = None
    groups_source: str | None = None
    species: str | None = None
    assembly: str | None = None
    measurement: str | None = None
    context: str | None = None
    grouping_provenance: str | None = None
    resource: str | None = None
    groups_result: str | None = None
    grouping_strategy: str | None = None

    def __post_init__(self):
        from agent.tools._file_identity import identifier
        if self.grouping_strategy not in (None, 'reuse_only', 'compute_if_needed'):
            raise ValueError('Invalid grouping strategy.')
        for name in ('source', 'groups_source'):
            value = getattr(self, name)
            if value is not None and (type(value) is not str or re.fullmatch(r'source\.(?:[0-9]|1[0-5])', value) is None):
                raise ValueError('Invalid annotation source handle.')
        if self.source is not None and self.source == self.groups_source:
            raise ValueError('Counts and group TSV require distinct sources.')
        if self.groups_result is not None and (
                type(self.groups_result) is not str or re.fullmatch(r'groups\.[0-9a-f]{64}', self.groups_result) is None
                or self.groups_source is not None):
            raise ValueError('Select an accepted clustering result or an uploaded grouping file.')
        handles = source_handles(self)
        if len(handles) != len(set(handles)):
            raise ValueError('Annotation scientific file roles require distinct sources.')
        for name in ('species', 'assembly', 'measurement', 'context', 'grouping_provenance'):
            value = getattr(self, name)
            if value is not None:
                identifier(value)
                if len(value) > 128:
                    raise ValueError('Annotation declaration exceeds its 128-character bound.')
        if self.resource is not None and (type(self.resource) is not str or re.fullmatch(r'resource\.[a-z][a-z0-9_-]{0,63}', self.resource) is None):
            raise ValueError('Invalid annotation resource handle.')


def target(field, declared=None):
    if field == 'epizoo':
        from .epizoo_handoff import target as epizoo_target
        return epizoo_target(declared or EpiZooDeclaration())
    return {'annotation': 'annotate_scATAC_cell_types', 'fragments': 'run_snapatac2', 'bam': 'run_snapatac2', 'fastq': 'run_chromap'}[field]


def empty(field):
    return {'epizoo': EpiZooDeclaration, 'annotation': AnnotationDeclaration, 'fragments': FragmentsDeclaration, 'bam': BamDeclaration, 'fastq': FastqDeclaration}[field]()


def source_handles(declared):
    if isinstance(declared, EpiZooDeclaration):
        from .epizoo_handoff import sources
        return sources(declared)
    if isinstance(declared, AnnotationDeclaration):
        return [h for h in (declared.source, declared.groups_source) if h is not None]
    if isinstance(declared, FragmentsDeclaration):
        return [h for h in (declared.source, declared.index_source, declared.feature_source, declared.genome_source, declared.genome_index_source) if h is not None]
    return ([declared.source] if declared.source else []) if isinstance(declared, BamDeclaration) else [m.source for m in declared.members]


def _field(interaction):
    admitted = interaction.admitted or {}
    return 'epizoo' if 'epizoo' in admitted else 'annotation' if 'annotation' in admitted else 'fragments' if 'fragments' in admitted else 'bam' if 'bam' in admitted else 'fastq' if 'fastq' in admitted else (interaction.snapshot.get('prerequisite') or {}).get('field', 'fastq')


def validate_downstream_annotation(value):
    """Biological declarations for planned counts; raw intake owns file roles."""
    if (type(value) is not AnnotationDeclaration or source_handles(value)
            or value.groups_result is not None):
        raise ValueError('Raw-to-annotation declarations cannot replace planned count or grouping sources.')


def validate_arguments(arguments, field='fastq'):
    from .epizoo_handoff import TOOLS
    allowed = {'epizoo':TOOLS, 'annotation':('annotate_scATAC_cell_types',),
        'fragments':('run_snapatac2',), 'bam':('run_snapatac2',),
        'fastq':('run_chromap','run_snapatac2')}[field]
    if any(a.tool not in allowed for a in arguments):
        raise ValueError('FASTQ task arguments must belong to its registered scientific consumers.')


def declaration(value, field='fastq'):
    if field == 'epizoo':
        optional={'chain_source','reference_species'}
        if type(value) is not dict or set(value)-optional != set(EpiZooDeclaration.__dataclass_fields__)-optional:
            raise ValueError('Invalid EpiZoo declaration shape.')
        return EpiZooDeclaration(**value)
    if field == 'annotation':
        if type(value) is not dict or set(value) - {'groups_result', 'grouping_strategy'} != set(AnnotationDeclaration.__dataclass_fields__) - {'groups_result', 'grouping_strategy'}:
            raise ValueError('Invalid annotation declaration shape.')
        value = dict(value)
        return AnnotationDeclaration(**value)
    if field == 'fragments':
        if type(value) is not dict or set(value) - {'resource_preference'} != set(FragmentsDeclaration.__dataclass_fields__) - {'resource_preference'}:
            raise ValueError('Invalid fragment declaration shape.')
        return FragmentsDeclaration(**value)
    if field == 'bam':
        if type(value) is not dict or set(value) - {'resource_preference'} != set(BamDeclaration.__dataclass_fields__) - {'resource_preference'}:
            raise ValueError('Invalid BAM declaration shape.')
        return BamDeclaration(**value)
    if type(value) is not dict or set(value) - {'resource_preference'} != set(FastqDeclaration.__dataclass_fields__) - {'resource_preference'}:
        raise ValueError('Invalid FASTQ declaration shape.')
    return FastqDeclaration(**dict(value, members=tuple(FastqMember(**m) for m in value['members'])))


def scientific_metadata_fields(declared):
    """Correctable pending declarations, excluding captured source identities."""
    if type(declared) not in (FastqDeclaration, BamDeclaration, FragmentsDeclaration,
                             AnnotationDeclaration, EpiZooDeclaration):
        raise ValueError('A typed scientific declaration is required.')
    fixed = {'source', 'members', 'namespace', 'operation', 'resource',
             'genome_resource', 'resource_preference', 'uploaded', 'groups_result'}
    return tuple(name for name in declared.__dataclass_fields__
                 if name not in fixed and not name.endswith('_source'))


def merge(previous, additions, *, pending=False, unknown_fields=()):
    """Merge user intent or, by default, strictly reconcile source observations.

    A pending declaration is not accepted scientific authority. Explicit user
    corrections may change its scientific metadata; only unknown_fields retracts
    a previous claim. Nullable additions alone continue to mean omission.
    """
    if type(previous) is not type(additions):
        raise ValueError('Raw declaration kind cannot change.')
    if (type(unknown_fields) is not tuple or len(unknown_fields) > 16
            or len(set(unknown_fields)) != len(unknown_fields)
            or any(type(name) is not str or name not in scientific_metadata_fields(previous)
                   or getattr(additions, name) is not None for name in unknown_fields)
            or unknown_fields and not pending):
        raise ValueError('Unknown declarations must name pending scientific metadata only.')
    values = asdict(previous)
    for name in unknown_fields:
        values[name] = None
    if isinstance(previous, EpiZooDeclaration) and additions.genome_source is not None:
        # An explicit uploaded genome supersedes the resource role only. It
        # never changes the original H5AD or another captured scientific input.
        values['genome_resource'] = additions.genome_resource
    if isinstance(previous, AnnotationDeclaration):
        if pending and additions.groups_source is not None and previous.groups_result is not None:
            # A newly attached grouping can replace an unusable result choice;
            # the captured count source and original registrations stay intact.
            values['groups_result'] = None
            values['grouping_provenance'] = None
    for key in values.keys() - {'members'}:
        value = getattr(additions, key)
        if value is not None:
            correctable = pending and (key in scientific_metadata_fields(previous)
                or isinstance(previous, AnnotationDeclaration) and key == 'groups_result')
            if (key not in ('resource', 'genome_resource', 'resource_preference') and not correctable
                    and values[key] is not None and values[key] != value):
                if pending:
                    from .turn_decisions import IntentError
                    raise IntentError('invalid_prerequisite',
                        diagnostic_reason='captured_' + key + '_changed', operand='declaration')
                raise ValueError('Captured scientific sources or declarations cannot be replaced.')
            values[key] = value
    if isinstance(previous, (BamDeclaration, FragmentsDeclaration, AnnotationDeclaration, EpiZooDeclaration)):
        return type(previous)(**values)
    members = {m.source: m for m in previous.members}
    for member in additions.members:
        if member.source in members and members[member.source] != member:
            raise ValueError('Captured FASTQ membership cannot be replaced.')
        members[member.source] = member
    return FastqDeclaration(**dict(values, members=tuple(members.values())))


def declaration_dict(value):
    """Preserve the exact pre-extension declaration identity when absent."""
    result = asdict(value)
    if result.get('resource_preference') is None:
        result.pop('resource_preference', None)
    if isinstance(value,EpiZooDeclaration):
        for key in ('chain_source','reference_species'):
            if result[key] is None: result.pop(key)
    if isinstance(value, AnnotationDeclaration) and value.groups_result is None:
        result.pop('groups_result')
    if isinstance(value, AnnotationDeclaration) and value.grouping_strategy is None:
        result.pop('grouping_strategy')
    return result


def schema(field='fastq'):
    """Typed declarations; captured identities are resolved by admission."""
    if field == 'epizoo':
        from .epizoo_handoff import schema as epizoo_schema
        return epizoo_schema()
    from .turn_decisions import _object
    from .resource_continuation import preference_schema
    from agent.tools.data._raw_fastq import FastqLayout, ReadRole
    def optional(choices):
        return ({'anyOf': [{'type': 'string', 'enum': list(choices)}, {'type': 'null'}]}
                if choices else {'type': 'null'})
    identity = {'anyOf': [{'type': 'string', 'minLength': 1, 'maxLength': 256}, {'type': 'null'}]}
    if field == 'annotation':
        text = {'anyOf': [{'type': 'string', 'maxLength': 128}, {'type': 'null'}]}
        return _object(dict(source=identity, groups_source=identity, groups_result=identity,
            **{k:text for k in ('species','assembly','context','measurement','grouping_provenance')},
            grouping_strategy=optional(('reuse_only','compute_if_needed')), resource=identity))
    if field == 'fragments':
        return _object(dict(source=identity,
            resource=identity,
            index_source=identity,
            feature_source=identity,
            genome_source=identity,
            genome_index_source=identity,
            namespace={'anyOf': [{'type': 'string', 'maxLength': 128}, {'type': 'null'}]},
            species=identity, assembly=identity,
            source_profile=dict(optional(('10x-atac-fragments.v1',)), description=(
                'Map explicitly supplied 10x ATAC/ARC provenance, already Tn5-adjusted zero-based half-open '
                'intervals, corrected barcodes and read-pair support to this choice. The researcher need '
                'not name source_profile or its internal identifier. Missing provenance stays null.')),
            feature_category=dict(optional(('peak_set', 'regulatory_regions')), description=(
                'Category of the explicitly uploaded feature_source BED only. Leave null when using an installed reference vocabulary, including cCREs. '
                'This field does not select or redescribe the installed reference features.')),
            resource_preference=preference_schema()))
    if field == 'bam':
        return _object(dict(source=identity,
            resource=identity,
            assay=optional(('SCATAC', 'TENX_ATAC')),
            namespace={'anyOf': [{'type': 'string', 'maxLength': 128}, {'type': 'null'}]},
            species=identity, assembly=identity,
            barcode_policy=optional(('corrected_cb_already_corrected',)),
            source_profile=optional(('agent-cb-paired-atac.v2',)),
            resource_preference=preference_schema()))
    return _object(dict(resource=identity, assay=optional(('TENX_ATAC',)), layout=optional((FastqLayout.A.value, FastqLayout.B.value)),
        namespace={'anyOf': [{'type': 'string', 'maxLength': 128}, {'type': 'null'}]},
        species=identity, barcode_policy=optional(('raw_sequence_whitelist',)),
        resource_preference=preference_schema(),
        members={'type': 'array', 'maxItems': 16, 'items': _object(dict(
            source={'type': 'string', 'minLength': 1, 'maxLength': 128},
            role={'type': 'string', 'enum': [r.value for r in ReadRole]},
            lane={'type': 'string', 'pattern': '^[0-9]{3}$'}, chunk={'type': 'string', 'pattern': '^[0-9]{3}$'}))}))


def resources(input_sets, field='fastq'):
    if field == 'epizoo':
        from .epizoo_handoff import resources as epizoo_resources
        return epizoo_resources(input_sets)
    if field == 'annotation':
        return ()  # CellMarker and native references are resolved by annotation_preparation.
    if field == 'fragments':
        return tuple(s for s in input_sets if s.fragments_companion)
    if field == 'bam':
        return tuple(s for s in input_sets if s.bam_companion and not
                     {'library_context_path', 'library_context_sha256'}.intersection(s.inputs()))
    return tuple(s for s in input_sets if s.fastq_companion and s.fastq_whitelist is not None)


def offers(input_sets, field='fastq'):
    if field in ('fastq', 'fragments', 'bam'):
        from .native_raw_handoff import resource_evidence as native_evidence
        return [dict(label=row['label'], species=row['species'], assembly=row['assembly'],
                     inputs=sorted(row['components']))
                for row in native_evidence(input_sets).values() if row is not None]
    result = [dict(handle='resource.' + s.input_set_id, label=s.label, default=getattr(s, field + '_default', False))
              for s in resources(input_sets, field)]
    if field == 'annotation':
        from .interactive import _unsafe_fact_value
        evidence = resource_evidence(input_sets, field)
        for row, choice in zip(result, resources(input_sets, field), strict=True):
            spec = evidence[choice.input_set_id]
            if spec is not None:
                context = {k: spec[k] for k in ('species', 'assembly', 'context') if k in spec}
                if not _unsafe_fact_value(context):
                    row['context'] = context
    return result


def resource_evidence(input_sets, field):
    """Read pinned owner manifests, not reference source data or a second catalog.

    These are admission observations, never scientific execution authority.
    Source freshness and full qualification remain with the consuming owners.
    Persisting the owner contract permits history replay without scientific IO.
    """
    if field == 'epizoo':
        from .epizoo_handoff import evidence
        return evidence(input_sets)
    from agent.tools.data import scatac_reference as reference
    result = {}
    for choice in resources(input_sets, field):
        values = choice.inputs()
        try:
            if field == 'annotation':
                result[choice.input_set_id] = None
                continue
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
        except (ValueError, OSError, TypeError, KeyError):
            result[choice.input_set_id] = None
    return result


def _resolve_resource(declared, choices, evidence, pinned=None):
    """Deterministic choice over owner-qualified contracts; no linguistic rules."""
    from agent.tools.data import scatac_reference as reference
    if isinstance(declared, AnnotationDeclaration):
        raise ValueError('Annotation resources belong to the selected production preparation owner.')
    if set(evidence) != {s.input_set_id for s in choices}:
        raise ValueError('Resource observations do not match the captured catalog.')
    candidates, descriptions, remedies = [], {}, {}
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
            remedy = remedies.get(selected.input_set_id, ' Supply a corrected resource or explicitly choose an alternative.')
            return [f'Cannot use {selected.label}: {descriptions[selected.input_set_id]}.' + remedy], None
        return [], selected
    # An available reference cannot establish the sample's species/alignment history.
    if (declared.species is None or isinstance(declared, (BamDeclaration, FragmentsDeclaration)) and declared.assembly is None):
        return [], None
    if not candidates:
        needed = ('A qualified fragments reference compatible with the declared species and assembly'
                  if isinstance(declared, FragmentsDeclaration) else 'A qualified BAM reference compatible with the declared species and assembly'
                  if isinstance(declared, (BamDeclaration, FragmentsDeclaration)) else
                  'A qualified FASTQ reference and barcode whitelist compatible with the declared species')
        return [needed + ' must be provisioned before preprocessing.'
                + (' Configured references are unavailable, invalid or incompatible.' if choices else '')], None
    field = 'annotation' if isinstance(declared, AnnotationDeclaration) else 'fragments' if isinstance(declared, FragmentsDeclaration) else 'bam' if isinstance(declared, BamDeclaration) else 'fastq'
    # Identical mappings/whitelists differ only in display/configuration identity.
    # Stable ID order is an explicit tie policy for these equivalent aliases only.
    def equivalence(choice):
        values = choice.inputs()
        # The same pinned manifest at another locator is the same dependency.
        # All other scientific arguments and provenance remain significant.
        values.pop('reference_bundle_path')
        values.setdefault('species', evidence[choice.input_set_id]['species'])
        return digest(values), digest(_serialize(choice.fastq_whitelist))
    from .resource_continuation import distinct_choices
    candidates = distinct_choices(candidates, default=lambda s: getattr(s, field + '_default', False),
        identity=equivalence, order=lambda s:s.input_set_id)
    if len(candidates) == 1:
        return [], candidates[0]
    return ['Choose a reference: ' + '; '.join(f'{s.label} ({descriptions[s.input_set_id]})' for s in candidates) + '.'], None


def source_binding(origin, source, interaction, field=None):
    field = field or (_field(source) if source is not None else _field(interaction))
    if (field == 'annotation' and source is not None
            and source.admitted.get('annotation_preparation') in ('selected-method.v2', 'selected-method.v3')):
        from .annotation_preparation import capture_sources
        return capture_sources(origin, source, interaction)
    return attachment_binding(origin, source, interaction, field)


def attachment_binding(origin, source, interaction, field):
    """Existing exact captured-file association, including historical replay."""
    if field in ('annotation', 'epizoo'):
        origin = _candidate_view(origin)
        interaction = _candidate_view(interaction)
    from .dialogue_execution import _role_binding
    # Reuse the existing exact-prefix/explicit-append rule. The scientific
    # collection is distinct from the immutable captured candidate group.
    if source is not None:
        source = replace(source, admitted=dict(source.admitted,
            registered_input=_serialize(source.admitted[field + '_sources']),
            inputs={}))
    return _role_binding(origin, source, interaction)


def requirements(binding, declared, input_sets, *, evidence=None, pinned=None):
    """Read saved pre-component raw declarations; new raw tasks use the native owner."""
    field = 'annotation' if isinstance(declared, AnnotationDeclaration) else 'fragments' if isinstance(declared, FragmentsDeclaration) else 'bam' if isinstance(declared, BamDeclaration) else 'fastq'
    from types import SimpleNamespace
    from .local_resources import LocalResourceAdmission, ResourceAdmissionError, _fastq_member
    if field == 'fragments':
        missing = [label for key, label in (
            ('source', 'Identify the external fragments attachment.'),
            ('namespace', 'Name the single processing library represented by these fragments.'),
            ('species', 'Specify human or mouse.'), ('assembly', 'Specify the source assembly: hg38 or mm10.'),
            ('source_profile', 'Confirm 10x ATAC/ARC fragment semantics: zero-based half-open adjusted intervals, corrected barcode identifiers and read-pair support.'))
            if getattr(declared, key) is None and key not in ('namespace', 'source_profile')]
        if any(int(h.split('.')[1]) >= len(binding.members) for h in source_handles(declared)):
            raise ValueError('Fragment role is outside captured attachments.')
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
            if getattr(declared, key) is None and key not in ('namespace', 'source_profile', 'barcode_policy')]
        if declared.source is not None and int(declared.source.split('.')[1]) >= len(binding.members):
            raise ValueError('BAM source is outside captured attachments.')
    else:
        missing = [label for key, label in (
            ('assay', 'Confirm the 10x ATAC assay.'), ('layout', 'Specify the FASTQ read layout before assigning missing reads.'),
            ('namespace', 'Name this single library and confirm that its selected lanes/chunks belong together.'),
            ('species', 'Specify human or mouse for this canonical reference workflow.'),
            ('barcode_policy', 'Confirm raw barcode sequences requiring whitelist correction.')) if getattr(declared, key) is None]
        if not declared.members:
            missing.append('Associate the uploaded reads with their read roles, lanes and chunks.'
                           if binding.members else
                           'Upload the FASTQ reads and state their read roles, lanes and chunks.')
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
        defaults = [s for s in choices if getattr(s, field + '_default', False)]
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
    admitted = interaction.admitted or {}
    if ('source_view' in interaction.snapshot.get('dialogue', {})
            and 'continuation' not in admitted
            and any(field in admitted for field in ('fastq', 'bam', 'fragments', 'annotation', 'epizoo'))):
        # A new typed request may use offered files while an older task is
        # pending. Only an explicit continuation selects that older objective.
        return interaction, None
    reference = interaction.snapshot.get('prerequisite')
    if reference is not None and reference['field'] in {'fastq', 'bam', 'fragments', 'annotation', 'epizoo'}:
        return _role_source(reference, preceding)
    return interaction, None


def _extend_catalog(previous, current):
    """Supplement a pending task without changing any captured definition."""
    offered = {row['input_set_id']: row for row in current}
    if any(offered.get(row['input_set_id']) != row for row in previous):
        raise ValueError('Captured scientific resource definitions cannot be replaced.')
    return current


def _completion(interaction, origin, source, addition, arguments, rows, registry, field='fastq', evidence=None, source_evidence=None, resource_observation=None, *, unknown_fields=(), declared=None, native=False):
    from .dialogue_execution import _bind_arguments, _with_retained
    from .input_sets import ScientificInputSet
    binding = source_binding(origin, source, interaction, field)
    if not binding.attachment_candidates or binding.roles or binding.execution_inputs:
        raise ValueError('FASTQ declarations require unbound captured attachments.')
    previous = empty(field) if source is None else declaration(_serialize(source.admitted[field]), field)
    # New admission computes this once before observing its exact source. Replay
    # uses the same merge without scientific IO. Source metadata reconciliation
    # remains a separate strict owner check below.
    if declared is None:
        declared = merge(previous, addition, pending=True, unknown_fields=unknown_fields)
    retained = (_with_retained(registry, {}, (origin.admitted or {}).get('retained_parameters', ()))
                if source is None else _serialize(source.admitted[field + '_arguments']))
    # A retracted declaration cannot survive through an older duplicate scalar
    # representation. Explicit current argument evidence is still bound below.
    for name in unknown_fields:
        retained.pop(name, None)
    for name in scientific_metadata_fields(addition):
        if getattr(addition, name) is not None and getattr(previous, name) != getattr(addition, name):
            retained.pop(name, None)
    values = _bind_arguments(registry, interaction.utterance, retained, arguments,
        replace_existing=field in ('annotation', 'epizoo'))
    if (field == 'annotation' and source is not None and addition.grouping_provenance is None
            and 'group_column' in source.admitted[field + '_arguments']
            and values.get('group_column') != source.admitted[field + '_arguments'].get('group_column')):
        declared = replace(declared, grouping_provenance=None)
    preference = getattr(addition, 'resource_preference', None)
    if preference is not None and interaction.utterance[preference['start']:preference['end']] != preference['literal']:
        raise ValueError('Resource preference no longer matches the user request')
    choices = tuple(ScientificInputSet(**_serialize(r)) for r in rows)
    pinned = None if source is None else source.admitted.get('raw_resource_choice')
    if field == 'epizoo':
        from .epizoo_handoff import requirements as epizoo_requirements
        declared, missing, selected = epizoo_requirements(binding, declared, resources(choices, field), evidence, source_evidence, pinned)
    elif field == 'annotation':
        raise ValueError('Annotation uses its current selected production preparation owner.')
    elif native and field in ('fastq', 'fragments', 'bam'):
        from .native_raw_handoff import requirements as native_requirements
        missing, selected = native_requirements(binding, declared, choices, evidence)
    else:
        if getattr(declared, 'resource_preference', None) is not None:
            from .resource_continuation import match_preference
            questions, preferred = match_preference(declared.resource_preference, resources(choices, field))
            if questions:
                return binding, declared, values, questions, None
            declared = replace(declared, resource='resource.' + preferred.input_set_id)
        missing, selected = requirements(binding, declared, choices, evidence=evidence, pinned=pinned)
    return binding, declared, values, missing, selected


def admit(sessions, interaction, decision, input_sets, *, reference_resources=None):
    """The raw declaration variant of existing role request/answer admission."""
    from .turn_decisions import RequestRoles, IntentError
    from .resource_continuation import catalog
    from .local_resources import LocalResourceAdmission
    field = ('epizoo' if decision.epizoo is not None else 'fragments' if decision.fragments is not None
             else 'bam' if decision.bam is not None else 'fastq' if decision.fastq is not None
             else (interaction.snapshot.get('prerequisite') or {}).get('field')
             or ('annotation' if decision.annotation is not None else 'fastq'))
    if field == 'annotation':
        from .annotation_preparation import admit as admit_annotation
        return admit_annotation(sessions, interaction, decision, input_sets)
    if field not in ('fastq', 'bam', 'fragments', 'annotation', 'epizoo') or (decision.fastq is not None and field != 'fastq'):
        raise IntentError('invalid_prerequisite')
    validate_arguments(decision.arguments, field)
    state = sessions.load(sessions._interaction_session_id)
    initial = isinstance(decision, RequestRoles)
    reference = interaction.snapshot.get('prerequisite')
    if (state.generation != interaction.base_generation or state.active_revision_id != interaction.base_revision_id
            or state.interactions[-1].turn_id != interaction.turn_id):
        raise IntentError('invalid_prerequisite')
    if initial:
        from .dialogue_execution import _research_intent
        intent = _research_intent(interaction, decision)
        origin, source = replace(interaction, admitted=intent), None
        rows = catalog(input_sets)
    else:
        if decision.pending != '@roles' or reference is None or reference['field'] != field:
            raise IntentError('invalid_prerequisite')
        origin, source = _captured(interaction, state.interactions[:-1])
        previous_rows = _serialize(source.admitted['resource_catalog'])
        if digest(previous_rows) != reference['resource_context_sha256']:
            raise IntentError('invalid_prerequisite')
        rows = _extend_catalog(previous_rows, catalog(input_sets))
    native = field in ('fastq', 'fragments', 'bam')
    if native:
        from .native_raw_handoff import resource_evidence as native_evidence
        evidence = native_evidence(input_sets, field=field)
    else:
        evidence = resource_evidence(input_sets, field)
    source_evidence = None
    previous = empty(field) if source is None else declaration(_serialize(source.admitted[field]), field)
    addition = getattr(decision, field) or empty(field)
    declared = merge(previous, addition, pending=True,
                     unknown_fields=() if initial else decision.unknown_fields)
    if field == 'epizoo':
        candidate = source_binding(origin, source, interaction, field)
        declared_source = declared.source
        if ((origin.submission or {}).get('candidate_input') is not None
                and declared_source is not None):
            selected = origin.submission['registered_input']
            position = int(declared_source.split('.')[1])
            if (position >= len(candidate.members)
                    or candidate.members[position]['resource_id'] != selected['resource_id']
                    or candidate.members[position]['record_sha256'] != selected['record_sha256']):
                raise IntentError('invalid_prerequisite')
        if declared_source is not None:
            from agent.tools.models.epizoo_inputs import observe_source as observe
            if int(declared_source.split('.')[1]) >= len(candidate.members):
                raise ValueError('EpiZoo source is outside the captured attachments.')
            member = candidate.members[int(declared_source.split('.')[1])]
            owner = LocalResourceAdmission(sessions._application._workspace, registry=sessions._application.registry)
            record = owner.load(member['resource_id'])
            if record.session_id not in (None, state.session_id):
                raise IntentError('invalid_prerequisite')
            owner._verify_records([record])
            source_evidence = observe(record.source_path, record.source_sha256)
            source_evidence['interface_version'] = 2
    binding, declared, values, missing, selected = _completion(interaction, origin, source,
        addition, decision.arguments, rows, sessions._application.registry, field, evidence, source_evidence,
        unknown_fields=() if initial else decision.unknown_fields, declared=declared, native=native)
    owner = LocalResourceAdmission(sessions._application._workspace, registry=sessions._application.registry)
    owner.validate_binding(binding)
    # Candidate validation intentionally does not open unused attachments.
    records = [owner.load(binding.members[int(h.split('.')[1])]['resource_id']) for h in source_handles(declared)]
    if any(r.session_id not in (None, state.session_id) for r in records):
        raise IntentError('invalid_prerequisite')
    owner._verify_records(records)
    if selected is not None and not missing and reference_resources is not None and not native:
        from .shared_resources import check_choice
        check_choice(reference_resources, state.session_id, 'resource.' + selected.input_set_id)
    admitted = dict(kind='clarify' if missing else 'execute', tool=target(field, declared),
        revision_id=interaction.base_revision_id, resource_catalog=rows, raw_resource_evidence=evidence,
        raw_resource_choice=(declared.resource or ('resource.' + selected.input_set_id if selected is not None
            else source.admitted.get('raw_resource_choice') if source is not None else None)),
        **{field: declaration_dict(declared), field + '_addition': declaration_dict(getattr(decision, field) or empty(field)),
           field + '_sources': binding.attribution(), field + '_arguments': values},
        inputs=values, registered_input=binding.attribution(), roles=[],
        arguments=[asdict(a) for a in decision.arguments])
    if native:
        admitted['resource_resolution'] = 'native-inputs.v1'
        previous_annotation = None if source is None else source.admitted.get('downstream_annotation')
        if decision.annotation is not None or previous_annotation is not None:
            addition = decision.annotation or AnnotationDeclaration()
            validate_downstream_annotation(addition)
            combined = merge(AnnotationDeclaration() if previous_annotation is None else
                declaration(_serialize(previous_annotation), 'annotation'), addition, pending=True)
            admitted['downstream_annotation'] = declaration_dict(combined)
            admitted['downstream_annotation_addition'] = declaration_dict(addition)
        # Native selection is per function. A corrected name identifying
        # several component aliases must not retain an older whole-set choice.
        admitted['raw_resource_choice'] = None if selected is None else 'resource.' + selected.input_set_id
    if initial:
        admitted.update(intent)
    if field == 'epizoo':
        admitted['epizoo_source_evidence'] = source_evidence
    if not initial:
        admitted['continuation'] = {k: reference[k] for k in ('origin_turn_id', 'binding_turn_id', 'field')}
        if decision.unknown_fields:
            admitted['unknown_fields'] = list(decision.unknown_fields)
    pending = None
    if missing:
        admitted.update(reason='missing_' + field + '_declarations', choices=missing, value_required=True)
        pending = dict(origin_turn_id=origin.turn_id, binding_turn_id=interaction.turn_id,
            field=field, tool=target(field, declared), resource_context_sha256=digest(rows))
    else:
        if field == 'epizoo':
            from .epizoo_handoff import prepare as prepare_epizoo, EpiZooResourcePrerequisite
            from agent.orchestration.prior_outputs import prior_output_store
            try:
                with prior_output_store(sessions._application.run_store):
                    binding, receipt = prepare_epizoo(owner, state.session_id, binding, declared, selected, values,
                        input_sets=input_sets, resource_owner=reference_resources, origin_turn_id=origin.turn_id)
            except EpiZooResourcePrerequisite as exc:
                source_evidence['transfer_resource_problem']=str(exc)
                admitted.update(kind='clarify',reason='missing_epizoo_declarations',choices=[str(exc)],value_required=True)
                pending=dict(origin_turn_id=origin.turn_id,binding_turn_id=interaction.turn_id,
                    field=field,tool=target(field,declared),resource_context_sha256=digest(rows))
                return admitted,pending
        else:
            binding, receipt = prepare(owner, state.session_id, binding, declared, selected, values,
                resource_evidence=evidence if native else None,
                accepted_contexts=tuple(i.admitted[field + '_context'] for i in state.interactions[:-1]
                    if i.admitted and field + '_context' in i.admitted))
        admitted.update(inputs=_serialize(binding.execution_inputs), registered_input=binding.attribution(),
            **{field + '_context': receipt}, operation='plan',
            request_id='turn-' + digest(dict(session=state.session_id, turn=interaction.turn_id)))
    return admitted, pending


def _context_identity(candidates, declared, selected, *, mode='original', scope=None):
    declaration = declaration_dict(declared)
    if scope == 'library.v1':
        # Reference selection is independent of library/barcode processing.
        # Keep the genuine whitelist dependency; never reuse across libraries.
        declaration.pop('resource')
        return digest(dict(sources=candidates.attribution(), declaration=declaration,
                           whitelist=_serialize(selected.fastq_whitelist), identity_scope=scope))
    if scope is not None:
        raise ValueError('Unsupported library identity scope.')
    return digest(dict(sources=candidates.attribution(), declaration=declaration,
                       resource=None if selected is None else selected.resource_choice(), **({'source_mode': mode} if mode else {})))


def prepare(owner, session_id, candidates, declared, selected, arguments, *, accepted_contexts=(), resource_evidence=None):
    from .native_raw_handoff import prepare as native_prepare
    return native_prepare(owner, session_id, candidates, declared, selected, arguments, resource_evidence=resource_evidence)


def _fragments_identity(candidates, declared, selected):
    return digest(dict(sources=candidates.attribution(), declaration=declaration_dict(declared),
                       resource=None if selected is None else selected.resource_choice()))


def _fragments_values(declared, selected, arguments, reference):
    values = {} if selected is None else selected.inputs()
    # Species already binds the selected reference at resource resolution; it
    # is not a second import argument. Shared BAM/fragment InputSets may carry it.
    values.pop('species', None)
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
    return dict(handle='@roles', field=field, tool=source.admitted['tool'],
        objective=origin.utterance, declarations=_serialize(source.admitted[field]),
        missing=list(source.admitted['choices']),
        resource_options=_serialize(source.admitted.get('resource_options', ())))


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
    if (interaction.admitted or {}).get('annotation_preparation') in ('selected-method.v1', 'selected-method.v2', 'selected-method.v3'):
        from .annotation_preparation import validate as validate_annotation
        return validate_annotation(interaction, preceding)
    try:
        captured = _serialize(interaction.snapshot)
        expected = capture_prerequisite(preceding, interaction.base_revision_id,
            interaction.base_generation, interaction.submission, captured.get('dialogue', {}).get('requested_predecessor'),
            canonical_sources='source_view' in captured.get('dialogue', {}))
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
        if 'downstream_annotation' in admitted:
            addition = declaration(admitted['downstream_annotation_addition'], 'annotation')
            validate_downstream_annotation(addition)
            previous = None if source is None else source.admitted.get('downstream_annotation')
            combined = merge(AnnotationDeclaration() if previous is None else
                declaration(_serialize(previous), 'annotation'), addition, pending=True)
            if admitted['downstream_annotation'] != declaration_dict(combined):
                raise ValueError('Downstream annotation declarations changed.')
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
        registry = build_default_tool_registry()
        binding, declared, values, missing, selected = _completion(interaction, origin, source,
            declaration(admitted[field + '_addition'], field), _scientific_arguments(admitted['arguments']),
            rows, registry, field, evidence, admitted.get(field + '_source_evidence'),
            admitted.get('annotation_resource_observation'), unknown_fields=tuple(admitted.get('unknown_fields', ())),
            native=admitted.get('resource_resolution') == 'native-inputs.v1')
        expected_choice = (None if selected is None else 'resource.' + selected.input_set_id) if admitted.get('resource_resolution') == 'native-inputs.v1' else (declared.resource or
            ('resource.' + selected.input_set_id if selected is not None else
             source.admitted.get('raw_resource_choice') if source is not None else None))
        if evidence is not None and admitted.get('raw_resource_choice') != expected_choice:
            raise ValueError('Captured resource selection changed.')
        if (admitted[field] != _serialize(declaration_dict(declared)) or admitted[field + '_sources'] != binding.attribution()
                or admitted[field + '_arguments'] != values or admitted['tool'] != target(field, declared)
                or admitted['revision_id'] != interaction.base_revision_id or admitted['roles']):
            raise ValueError('FASTQ declarations changed.')
        if source is not None and admitted.get('continuation') != {
                k: expected[k] for k in ('origin_turn_id', 'binding_turn_id', 'field')}:
            raise ValueError('FASTQ continuation changed.')
        if missing or admitted['kind'] == 'clarify':
            pending = dict(origin_turn_id=origin.turn_id, binding_turn_id=interaction.turn_id,
                field=field, tool=target(field, declared), resource_context_sha256=digest(rows))
            # Saved prerequisite prose is immutable presentation, not admission
            # authority. Recompute whether prerequisites are missing and check
            # all typed bindings below, without requiring historical wording to
            # equal the current renderer (or keeping legacy message adapters).
            choices = admitted['choices']
            if (admitted['kind'] != 'clarify' or not isinstance(choices, list) or not choices
                    or any(type(x) is not str or not x.strip() for x in choices)
                    or admitted['inputs'] != values or admitted['registered_input'] != binding.attribution()
                    or _serialize(interaction.prerequisite) != pending or field + '_context' in admitted):
                raise ValueError('FASTQ missing requirements changed.')
        elif field == 'epizoo':
            from .epizoo_handoff import validate_completion
            validate_completion(admitted, binding, declared, selected, values)
        elif field == 'annotation':
            raise ValueError('Annotation uses its current selected production preparation owner.')
        elif (admitted.get(field + '_context') or {}).get('adapter') in ('native-files.v1', 'native-inputs.v1'):
            from .native_raw_handoff import validate as validate_native
            validate_native(admitted, binding, declared, selected, values, field)
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








def _candidate_view(interaction):
    """Exact selected H5AD plus optional companions for typed role admission."""
    from .local_resources import RegisteredInputGroup, available_input_view
    interaction = available_input_view(interaction)
    submission = _serialize(interaction.submission or {})
    if 'candidate_input' not in submission:
        return interaction
    metadata = submission['candidate_input']
    bound = RegisteredInputGroup(metadata['members'], {}, attachment_candidates=True)
    original = submission['registered_input']
    if (bound.attribution() != metadata
            or original.get('tool_name') != 'inspect_scATAC' or original.get('resource_choice') is not None
            or submission['execution_inputs'] != {'input_path': bound.members[0]['source_path']}
            or any(original[k] != bound.members[0][k] for k in ('resource_id', 'record_sha256'))):
        raise ValueError('Candidate view changed the selected source.')
    return replace(interaction, submission=dict(submission, execution_inputs={}, registered_input=metadata))
