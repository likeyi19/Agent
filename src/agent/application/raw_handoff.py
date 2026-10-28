"""Task-scoped source declarations; scientific qualification remains with owners.

These are bounded variants of existing role continuation, not a scientific planner.
Only explicitly captured attachments can supply scientific source roles.
"""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import re

from agent.schemas.orchestration import _serialize
from .session_state import digest
from .epizoo_handoff import EpiZooDeclaration


_ANNOTATION_REFERENCE_MISSING = 'No deployed annotation reference matches the declared biological context. Clarify the species, assembly or tissue applicability when needed; otherwise the operator must provision a compatible gene/signature reference in the scientific resource library.'

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
                (self.source_profile, ('agent-cb-paired-atac.v1', 'agent-cb-paired-atac.v2'))):
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


@dataclass(frozen=True)
class AnnotationUploads:
    """Explicit file roles and declared provenance, never inferred from filenames."""
    gene_source: str | None = None
    signature_source: str | None = None
    gene_provenance: str | None = None
    gene_release: str | None = None
    signature_provenance: str | None = None
    signature_release: str | None = None
    species: str | None = None
    assembly: str | None = None
    context: str | None = None
    gene_namespace: str | None = None

    def __post_init__(self):
        from agent.tools.analysis.marker_annotation import identifier
        for name, value in asdict(self).items():
            if value is None:
                continue
            identifier(value)
            if len(value) > 512 or name.endswith('_source') and re.fullmatch(r'source\.(?:[0-9]|1[0-5])', value) is None:
                raise ValueError('Invalid uploaded annotation resource declaration.')
        if self.gene_source is not None and self.gene_source == self.signature_source:
            raise ValueError('Gene annotations and signatures require distinct files.')


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
    uploaded: AnnotationUploads | None = None
    groups_result: str | None = None
    grouping_strategy: str | None = None

    def __post_init__(self):
        from agent.tools.analysis.marker_annotation import identifier
        if self.grouping_strategy not in (None, 'reuse_only', 'compute_if_needed'):
            raise ValueError('Invalid grouping strategy.')
        if self.uploaded is not None and type(self.uploaded) is not AnnotationUploads:
            raise ValueError('Invalid uploaded resource declarations.')
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
    return {'annotation': 'annotate_scATAC_cell_types', 'fragments': 'import_scATAC_fragments', 'bam': 'prepare_scATAC_bam_fragments', 'fastq': 'prepare_scATAC_fragments'}[field]


def empty(field):
    return {'epizoo': EpiZooDeclaration, 'annotation': AnnotationDeclaration, 'fragments': FragmentsDeclaration, 'bam': BamDeclaration, 'fastq': FastqDeclaration}[field]()


def source_handles(declared):
    if isinstance(declared, EpiZooDeclaration):
        from .epizoo_handoff import sources
        return sources(declared)
    if isinstance(declared, AnnotationDeclaration):
        return [h for h in (declared.source, declared.groups_source,
            *(getattr(declared.uploaded, k) for k in ('gene_source', 'signature_source')
              if declared.uploaded is not None)) if h is not None]
    if isinstance(declared, FragmentsDeclaration):
        return [h for h in (declared.source, declared.index_source, declared.feature_source, declared.genome_source, declared.genome_index_source) if h is not None]
    return ([declared.source] if declared.source else []) if isinstance(declared, BamDeclaration) else [m.source for m in declared.members]


def _field(interaction):
    admitted = interaction.admitted or {}
    return 'epizoo' if 'epizoo' in admitted else 'annotation' if 'annotation' in admitted else 'fragments' if 'fragments' in admitted else 'bam' if 'bam' in admitted else 'fastq' if 'fastq' in admitted else (interaction.snapshot.get('prerequisite') or {}).get('field', 'fastq')


def validate_arguments(arguments, field='fastq'):
    from .local_resources import _FASTQ_COMPANION_TOOLS, _BAM_COMPANION_TOOLS, _FRAGMENTS_COMPANION_TOOLS
    from .epizoo_handoff import TOOLS
    allowed = {'epizoo': TOOLS, 'annotation': ('annotate_scATAC_cell_types',), 'fragments': _FRAGMENTS_COMPANION_TOOLS, 'bam': _BAM_COMPANION_TOOLS, 'fastq': _FASTQ_COMPANION_TOOLS}[field]
    if any(a.tool not in allowed for a in arguments):
        raise ValueError('FASTQ task arguments must belong to its registered scientific consumers.')


def declaration(value, field='fastq'):
    if field == 'epizoo':
        optional={'chain_source','reference_species'}
        if type(value) is not dict or set(value)-optional != set(EpiZooDeclaration.__dataclass_fields__)-optional:
            raise ValueError('Invalid EpiZoo declaration shape.')
        return EpiZooDeclaration(**value)
    if field == 'annotation':
        if type(value) is not dict or set(value) - {'uploaded', 'groups_result', 'grouping_strategy'} != set(AnnotationDeclaration.__dataclass_fields__) - {'uploaded', 'groups_result', 'grouping_strategy'}:
            raise ValueError('Invalid annotation declaration shape.')
        value = dict(value)
        if value.get('uploaded') is not None:
            if type(value['uploaded']) is not dict or set(value['uploaded']) != set(AnnotationUploads.__dataclass_fields__):
                raise ValueError('Invalid uploaded annotation resource shape.')
            value['uploaded'] = AnnotationUploads(**value['uploaded'])
        return AnnotationDeclaration(**value)
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


def scientific_metadata_fields(declared):
    """Correctable pending declarations, excluding captured source identities."""
    if type(declared) not in (FastqDeclaration, BamDeclaration, FragmentsDeclaration,
                             AnnotationDeclaration, EpiZooDeclaration):
        raise ValueError('A typed scientific declaration is required.')
    fixed = {'source', 'members', 'namespace', 'operation', 'resource',
             'genome_resource', 'uploaded', 'groups_result'}
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
        values['uploaded'] = previous.uploaded
        if pending and additions.groups_source is not None and previous.groups_result is not None:
            # A newly attached grouping can replace an unusable result choice;
            # the captured count source and original registrations stay intact.
            values['groups_result'] = None
            values['grouping_provenance'] = None
    for key in values.keys() - {'members'}:
        value = getattr(additions, key)
        if value is not None:
            if key == 'uploaded':
                old = asdict(previous.uploaded or AnnotationUploads())
                for role in ('gene', 'signature'):
                    supplied = getattr(value, role + '_source')
                    if supplied is not None and supplied != old[role + '_source']:
                        old[role + '_provenance'] = None
                        old[role + '_release'] = None
                for name, supplied in asdict(value).items():
                    if supplied is not None:
                        # Explicit uploaded reference corrections retain source
                        # registrations and are recorded as new typed additions.
                        old[name] = supplied
                values[key] = AnnotationUploads(**old)
                continue
            correctable = pending and (key in scientific_metadata_fields(previous)
                or isinstance(previous, AnnotationDeclaration) and key == 'groups_result')
            if (key not in ('resource', 'genome_resource') and not correctable
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
    if isinstance(value,EpiZooDeclaration):
        for key in ('chain_source','reference_species'):
            if result[key] is None: result.pop(key)
    if isinstance(value, AnnotationDeclaration) and value.uploaded is None:
        result.pop('uploaded')
    if isinstance(value, AnnotationDeclaration) and value.groups_result is None:
        result.pop('groups_result')
    if isinstance(value, AnnotationDeclaration) and value.grouping_strategy is None:
        result.pop('grouping_strategy')
    return result


def schema(sources, resources, field='fastq', *, groups=()):
    if field == 'epizoo':
        from .epizoo_handoff import schema as epizoo_schema
        return epizoo_schema(sources, resources)
    from .turn_decisions import _object
    from agent.tools.data._raw_fastq import FastqLayout, ReadRole
    def optional(choices):
        return ({'anyOf': [{'type': 'string', 'enum': list(choices)}, {'type': 'null'}]}
                if choices else {'type': 'null'})
    if field == 'annotation':
        from agent.tools.analysis.annotation_contract import EXTERNAL_SEMANTICS
        text = {'anyOf': [{'type': 'string', 'maxLength': 128}, {'type': 'null'}]}
        uploaded = _object({k: optional([s['source'] for s in sources]) if k.endswith('_source') else
            {'anyOf': [{'type': 'string', 'maxLength': 512}, {'type': 'null'}]}
            for k in AnnotationUploads.__dataclass_fields__})
        return _object(dict(source=optional([s['source'] for s in sources]),
            groups_source=optional([s['source'] for s in sources]),
            groups_result=optional([g['handle'] for g in groups]) if groups else {'type': 'null'},
            **{k: text for k in ('species', 'assembly', 'grouping_provenance')},
            context=dict(text, description='Interpret the user-supported tissue meaning using an exact compatible offered context when available: '
                + repr(sorted({r['context']['context'] for r in resources if r.get('context', {}).get('context')}))
                + '. Do not append demographic or health qualifiers to an equivalent offered tissue identifier. Preserve genuinely different tissue/context as a declaration; never select by availability alone.'),
            grouping_strategy=optional(('reuse_only', 'compute_if_needed')),
            measurement=dict(text, description=(
                f'Use {EXTERNAL_SEMANTICS} for explicitly declared raw Cell Ranger ATAC '
                'cut-site counts. Preserve other declared measurement types as text for '
                'compatibility checking. Use fragment_counts for declared region fragment overlaps, peak_counts for raw peak counts, binary_accessibility for binary region presence; unknown history is null. Integer values, filenames '
                'and resource availability do not establish counting history.')),
            resource=optional([r['handle'] for r in resources]) if resources else {'type': 'null'},
            uploaded={'anyOf': [uploaded, {'type': 'null'}]}))
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
            source_profile=optional(('agent-cb-paired-atac.v2',)),
            resource=optional([r['handle'] for r in resources]) if resources else {'type': 'null'}))
    return _object(dict(assay=optional(('TENX_ATAC',)), layout=optional((FastqLayout.A.value, FastqLayout.B.value)),
        namespace={'anyOf': [{'type': 'string', 'maxLength': 128}, {'type': 'null'}]},
        species=optional(('human', 'mouse')), barcode_policy=optional(('raw_sequence_whitelist',)),
        resource=optional([r['handle'] for r in resources]) if resources else {'type': 'null'},
        members={'type': 'array', 'maxItems': 16 if sources else 0, 'items': _object(dict(
            source={'type': 'string', 'enum': [s['source'] for s in sources]},
            role={'type': 'string', 'enum': [r.value for r in ReadRole]},
            lane={'type': 'string', 'pattern': '^[0-9]{3}$'}, chunk={'type': 'string', 'pattern': '^[0-9]{3}$'}))
            if sources else {'type': 'string'}}))


def resources(input_sets, field='fastq'):
    if field == 'epizoo':
        from .epizoo_handoff import resources as epizoo_resources
        return epizoo_resources(input_sets)
    if field == 'annotation':
        return tuple(s for s in input_sets if 'annotation_spec_path' in s.inputs()
                     or {'qc_reference_manifest_path', 'qc_reference_manifest_sha256'} <= set(s.inputs()))
    if field == 'fragments':
        return tuple(s for s in input_sets if s.fragments_companion)
    if field == 'bam':
        return tuple(s for s in input_sets if s.bam_companion and not
                     {'library_context_path', 'library_context_sha256'}.intersection(s.inputs()))
    return tuple(s for s in input_sets if s.fastq_companion and s.fastq_whitelist is not None)


def offers(input_sets, field='fastq'):
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
                from agent.tools.analysis.annotation_input import resource_base
                result[choice.input_set_id] = resource_base(values)
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
    if set(evidence) != {s.input_set_id for s in choices}:
        raise ValueError('Resource observations do not match the captured catalog.')
    candidates, descriptions, remedies = [], {}, {}
    for choice in choices:
        raw = evidence[choice.input_set_id]
        if raw is None:
            descriptions[choice.input_set_id] = 'its pinned reference is unavailable or invalid'
            continue
        if isinstance(declared, AnnotationDeclaration):
            from agent.tools.analysis.annotation_contract import EXTERNAL_SPEC, EXTERNAL_PRIMARY, direct
            uploaded = declared.uploaded
            scope_only = 'schema' not in raw and 'contig_scope' in raw
            permitted = {'annotation_spec_path', 'annotation_spec_sha256', 'annotation_method', 'count_slot',
                         'qc_reference_manifest_path', 'qc_reference_manifest_sha256'}
            keys = ('species', 'assembly') if uploaded is not None and uploaded.signature_source else ('species', 'assembly', 'context')
            compatible = ((scope_only and uploaded is not None or raw.get('schema') == EXTERNAL_SPEC and direct(raw.get('method')))
                and (scope_only or not set(choice.inputs()) - permitted)
                and direct(choice.inputs().get('annotation_method', EXTERNAL_PRIMARY))
                and all(getattr(declared, key) is None or getattr(declared, key) == raw.get(key)
                        for key in (('species', 'assembly') if scope_only else keys)))
            conflicts = [key for key in (('species', 'assembly') if scope_only else keys)
                         if getattr(declared, key) is not None and getattr(declared, key) != raw.get(key)]
            if (not scope_only and (raw.get('schema') != EXTERNAL_SPEC or not direct(raw.get('method')))
                    or not direct(choice.inputs().get('annotation_method', EXTERNAL_PRIMARY))):
                conflicts.append('method')
            descriptions[choice.input_set_id] = ('compatible annotation reference' if compatible else
                'incompatible annotation ' + (', '.join(conflicts) if conflicts else 'resource contract'))
            if 'measurement' in conflicts:
                remedies[choice.input_set_id] = ''
                descriptions[choice.input_set_id] += ('. Replacing a reference does not change the input measurement. '
                    'Supply genuinely compatible counts or choose a method supporting the original measurement; '
                    'do not relabel or reinterpret the counts')
            if compatible:
                candidates.append(choice)
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
    if (not candidates and isinstance(declared, AnnotationDeclaration) and declared.uploaded is not None
            and declared.uploaded.gene_source is not None and declared.uploaded.signature_source is not None):
        # Explicit complete scientific resources need no unrelated QC InputSet.
        return [], None
    # An available reference cannot establish the sample's species/alignment history.
    if (declared.species is None or isinstance(declared, (BamDeclaration, FragmentsDeclaration)) and declared.assembly is None
            or isinstance(declared, AnnotationDeclaration) and any(getattr(declared, k) is None
                for k in ('assembly', 'measurement', 'context'))):
        return [], None
    if not candidates:
        if isinstance(declared, AnnotationDeclaration):
            details = '; '.join(f'{s.label}: {descriptions[s.input_set_id]}' for s in choices)
            return [_ANNOTATION_REFERENCE_MISSING + (' Available resources: ' + details + '.' if details else '')], None
        needed = ('A qualified annotation gene/marker resource compatible with species, assembly, measurement and tissue context'
                  if isinstance(declared, AnnotationDeclaration) else 'A qualified fragments reference compatible with the declared species and assembly'
                  if isinstance(declared, FragmentsDeclaration) else 'A qualified BAM reference compatible with the declared species and assembly'
                  if isinstance(declared, (BamDeclaration, FragmentsDeclaration)) else
                  'A qualified FASTQ reference and barcode whitelist compatible with the declared species')
        return [needed + ' must be provisioned before preprocessing.'
                + (' Configured references are unavailable, invalid or incompatible.' if choices else '')], None
    field = 'annotation' if isinstance(declared, AnnotationDeclaration) else 'fragments' if isinstance(declared, FragmentsDeclaration) else 'bam' if isinstance(declared, BamDeclaration) else 'fastq'
    defaults = [s for s in candidates if getattr(s, field + '_default', False)]
    if len(defaults) == 1:
        return [], defaults[0]
    # Identical mappings/whitelists differ only in display/configuration identity.
    # Stable ID order is an explicit tie policy for these equivalent aliases only.
    def equivalence(choice):
        values = choice.inputs()
        if isinstance(declared, AnnotationDeclaration) and declared.uploaded is not None:
            # Uploaded roles supersede only those dependencies. Two catalog
            # entries with identical remaining dependencies are exact aliases.
            raw = evidence[choice.input_set_id]
            retained = {k: raw.get(k) for k in ('species', 'assembly', 'contig_scope')}
            for role in ('gene', 'signature'):
                if getattr(declared.uploaded, role + '_source') is None:
                    retained[role + '_resource'] = raw.get(role + '_resource')
            retained['count_slot'] = values.get('count_slot')
            return digest(retained)
        # The same pinned manifest at another locator is the same dependency.
        # All other scientific arguments and provenance remain significant.
        values.pop('annotation_spec_path' if isinstance(declared, AnnotationDeclaration) else 'reference_bundle_path')
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
    field = 'annotation' if isinstance(declared, AnnotationDeclaration) else 'fragments' if isinstance(declared, FragmentsDeclaration) else 'bam' if isinstance(declared, BamDeclaration) else 'fastq'
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


def _completion(interaction, origin, source, addition, arguments, rows, registry, field='fastq', evidence=None, source_evidence=None, resource_observation=None, *, unknown_fields=(), declared=None):
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
    choices = tuple(ScientificInputSet(**_serialize(r)) for r in rows)
    pinned = None if source is None else source.admitted.get('raw_resource_choice')
    if field == 'epizoo':
        from .epizoo_handoff import requirements as epizoo_requirements
        declared, missing, selected = epizoo_requirements(binding, declared, resources(choices, field), evidence, source_evidence, pinned)
    elif field == 'annotation':
        declared, values, missing = _annotation_requirements(binding, declared, values, source_evidence)
        questions, selected = _resolve_resource(declared, resources(choices, field), evidence, pinned)
        if (selected is not None and 'annotation_spec_path' in selected.inputs()
                and 'count_slot' not in values and 'count_slot' in selected.inputs()):
            values['count_slot'] = selected.inputs()['count_slot']
            declared, values, missing = _annotation_requirements(binding, declared, values, source_evidence)
        missing += questions
        if declared.uploaded is not None:
            from agent.tools.analysis.annotation_input import resource_specification
            spec, problems = resource_specification(evidence[selected.input_set_id] if selected else None,
                declared, binding.members)
            missing += problems
            if resource_observation is not None:
                if (set(resource_observation) != {'specification_sha256', 'qualification', 'problem'}
                        or resource_observation['specification_sha256'] != digest(spec)
                        or type(resource_observation['problem']) is not str):
                    raise ValueError('Uploaded reference observation changed its captured specification.')
                qualification = resource_observation['qualification']
                if not resource_observation['problem'] and (type(qualification) is not dict
                        or set(qualification) not in (
                            {'gene_sha256', 'signature_sha256', 'gene_count', 'primary_transcripts', 'candidate_count', 'interpretation'},
                            {'gene_sha256', 'signature_sha256', 'gene_count', 'transcript_count', 'candidate_count', 'interpretation'})
                        or any(qualification[role + '_sha256'] != spec[role + '_resource']['sha256'] for role in ('gene', 'signature'))
                        or any(type(qualification[key]) is not int or qualification[key] <= 0
                               for key in ('gene_count', 'candidate_count', 'primary_transcripts' if 'primary_transcripts' in qualification else 'transcript_count'))
                        or qualification['interpretation'] != 'declared_provenance_and_applicability;not_independent_biological_validation'):
                    raise ValueError('Invalid uploaded reference qualification observation.')
                if resource_observation['problem'] and qualification is not None:
                    raise ValueError('Failed resource qualification cannot carry a success observation.')
                if resource_observation['problem']:
                    missing.append(resource_observation['problem'])
    else:
        missing, selected = requirements(binding, declared, choices, evidence=evidence, pinned=pinned)
    return binding, declared, values, missing, selected


def admit(sessions, interaction, decision, input_sets, *, reference_resources=None):
    """The raw declaration variant of existing role request/answer admission."""
    from .turn_decisions import RequestRoles, IntentError
    from .resource_continuation import catalog
    from .local_resources import LocalResourceAdmission
    field = 'epizoo' if decision.epizoo is not None else 'annotation' if decision.annotation is not None else 'fragments' if decision.fragments is not None else 'bam' if decision.bam is not None else (interaction.snapshot.get('prerequisite') or {}).get('field', 'fastq')
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
    evidence = resource_evidence(input_sets, field)
    source_evidence = None
    previous = empty(field) if source is None else declaration(_serialize(source.admitted[field]), field)
    addition = getattr(decision, field) or empty(field)
    declared = merge(previous, addition, pending=True,
                     unknown_fields=() if initial else decision.unknown_fields)
    if field in ('annotation', 'epizoo'):
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
            if field == 'epizoo':
                from agent.tools.models.epizoo_inputs import observe_source as observe
            else:
                from agent.tools.analysis.annotation_input import observe
            if int(declared_source.split('.')[1]) >= len(candidate.members):
                raise ValueError('Annotation source is outside the captured attachments.')
            member = candidate.members[int(declared_source.split('.')[1])]
            owner = LocalResourceAdmission(sessions._application._workspace, registry=sessions._application.registry)
            record = owner.load(member['resource_id'])
            if record.session_id not in (None, state.session_id):
                raise IntentError('invalid_prerequisite')
            owner._verify_records([record])
            source_evidence = observe(record.source_path, record.source_sha256)
            if field == 'epizoo':
                source_evidence['interface_version'] = 2
            if field == 'annotation':
                from .annotation_groups import choices
                group_choices = choices(sessions, interaction)
                if group_choices:
                    source_evidence['accepted_groups'] = group_choices
    binding, declared, values, missing, selected = _completion(interaction, origin, source,
        addition, decision.arguments, rows, sessions._application.registry, field, evidence, source_evidence,
        unknown_fields=() if initial else decision.unknown_fields, declared=declared)
    owner = LocalResourceAdmission(sessions._application._workspace, registry=sessions._application.registry)
    owner.validate_binding(binding)
    # Candidate validation intentionally does not open unused attachments.
    records = [owner.load(binding.members[int(h.split('.')[1])]['resource_id']) for h in source_handles(declared)]
    if any(r.session_id not in (None, state.session_id) for r in records):
        raise IntentError('invalid_prerequisite')
    owner._verify_records(records)
    if selected is not None and not missing and reference_resources is not None:
        from .shared_resources import check_choice
        check_choice(reference_resources, state.session_id, 'resource.' + selected.input_set_id)
    resource_observation = None
    if field == 'annotation' and declared.uploaded is not None and not missing:
        import csv
        from agent.tools.analysis.annotation_input import resource_specification, qualify_resources
        spec, _ = resource_specification(evidence[selected.input_set_id] if selected else None, declared, binding.members)
        resource_observation = dict(specification_sha256=digest(spec), qualification=None, problem='')
        try:
            resource_observation['qualification'] = qualify_resources(spec)
        except (ValueError, OSError, TypeError, KeyError, csv.Error) as exc:
            from .interactive import _unsafe_fact_value
            detail = str(exc)
            if len(detail) > 512 or _unsafe_fact_value(detail):
                detail = 'The supplied resource cannot be qualified; check its content and declared compatibility.'
            resource_observation['problem'] = 'Uploaded annotation reference: ' + detail
            missing.append(resource_observation['problem'])
    projection = None
    if field == 'annotation' and declared.groups_result is not None and not missing:
        from .annotation_groups import project, GroupAssociationError
        try:
            projection = project(sessions, interaction, declared.groups_result,
                binding.members[int(declared.source.split('.')[1])])
        except GroupAssociationError as exc:
            source_evidence['group_projection_problem'] = str(exc)
            missing.append(str(exc))
    admitted = dict(kind='clarify' if missing else 'execute', tool=target(field, declared),
        revision_id=interaction.base_revision_id, resource_catalog=rows, raw_resource_evidence=evidence,
        raw_resource_choice=(declared.resource or ('resource.' + selected.input_set_id if selected is not None
            else source.admitted.get('raw_resource_choice') if source is not None else None)),
        **{field: declaration_dict(declared), field + '_addition': declaration_dict(getattr(decision, field) or empty(field)),
           field + '_sources': binding.attribution(), field + '_arguments': values},
        inputs=values, registered_input=binding.attribution(), roles=[],
        arguments=[asdict(a) for a in decision.arguments])
    if initial:
        admitted.update(intent)
    if field == 'epizoo':
        admitted['epizoo_source_evidence'] = source_evidence
    if field == 'annotation':
        admitted['annotation_source_evidence'] = source_evidence
        if declared.uploaded is not None:
            admitted['annotation_resource_observation'] = resource_observation
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
        elif field == 'annotation' and declared.groups_result is not None:
            binding, receipt = _prepare_annotation(owner, binding, declared, selected, values,
                                                   projection=projection)
        else:
            binding, receipt = prepare(owner, state.session_id, binding, declared, selected, values,
                accepted_contexts=tuple(i.admitted[field + '_context'] for i in state.interactions[:-1]
                    if i.admitted and field + '_context' in i.admitted))
        admitted.update(inputs=_serialize(binding.execution_inputs), registered_input=binding.attribution(),
            **{field + '_context': receipt}, operation='plan',
            request_id='turn-' + digest(dict(session=state.session_id, turn=interaction.turn_id)))
        if field == 'annotation' and declared.uploaded is not None and reference_resources is not None:
            from .shared_resources import persist
            uploaded_source = declared.uploaded.gene_source or declared.uploaded.signature_source
            persist(reference_resources, dict(annotation_spec_path=receipt['resource']['path'],
                annotation_spec_sha256=receipt['resource']['sha256']), session_id=state.session_id,
                origin_turn_id=origin.turn_id, label='Qualified annotation references',
                qualification=resource_observation['qualification'],
                publication_source=(binding.members[int(uploaded_source.split('.')[1])]['source_path']
                                    if uploaded_source is not None else None))
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


def prepare(owner, session_id, candidates, declared, selected, arguments, *, accepted_contexts=()):
    """Coordinate real intake, independent verification and the context owner.

    No provisional authority is published. The normal planned intake/producer
    still execute and verify under RunStore; this preparation admits a request
    resource against the same exact source bytes and intake content identity.
    """
    if isinstance(declared, AnnotationDeclaration):
        return _prepare_annotation(owner, candidates, declared, selected, arguments)
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
            raise ReferenceSupplementError('The supplied BED or its reference genome failed owner qualification.',
                diagnostic_reason='incompatible_scientific_reference') from exc
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
    return dict(handle='@roles', field=field, tool=source.admitted['tool'],
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
            rows, build_default_tool_registry(), field, evidence, admitted.get(field + '_source_evidence'),
            admitted.get('annotation_resource_observation'), unknown_fields=tuple(admitted.get('unknown_fields', ())))
        if evidence is not None and admitted.get('raw_resource_choice') != (declared.resource or
                ('resource.' + selected.input_set_id if selected is not None else
                 source.admitted.get('raw_resource_choice') if source is not None else None)):
            raise ValueError('Captured resource selection changed.')
        if (admitted[field] != _serialize(declaration_dict(declared)) or admitted[field + '_sources'] != binding.attribution()
                or admitted[field + '_arguments'] != values or admitted['tool'] != target(field, declared)
                or admitted['revision_id'] != interaction.base_revision_id or admitted['roles']):
            raise ValueError('FASTQ declarations changed.')
        if source is not None and admitted.get('continuation') != {
                k: expected[k] for k in ('origin_turn_id', 'binding_turn_id', 'field')}:
            raise ValueError('FASTQ continuation changed.')
        if missing:
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
            _validate_annotation_completion(admitted, binding, declared, selected, values)
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


def _annotation_requirements(binding, declared, arguments, observation):
    """Resolve exact source declarations; never classify groups by their names."""
    from agent.tools.analysis import annotation_contract as c
    values = dict(arguments)
    missing = []
    if any(int(h.split('.')[1]) >= len(binding.members) for h in source_handles(declared)):
        raise ValueError('Annotation source is outside the captured attachments.')
    if declared.source is None:
        if observation is not None:
            raise ValueError('Unexpected annotation source observation.')
        return declared, values, ['Identify the uploaded peak-count H5AD.']
    member = binding.members[int(declared.source.split('.')[1])]
    if (not isinstance(observation, dict) or set(observation) - {
            'accepted_groups', 'interface_version', 'group_projection_problem'} != {
            'source_sha256', 'declarations', 'grouping_columns', 'qualified_groupings'}
            or observation['source_sha256'] != member['source_sha256']):
        raise ValueError('Annotation observation is not bound to the captured source.')
    additions = {k: v for k, v in observation['declarations'].items() if k != 'count_slot'}
    metadata_slot = observation['declarations'].get('count_slot', 'X')
    selected_slot = values.get('count_slot', metadata_slot)
    if observation.get('interface_version', 2) >= 3 and selected_slot != metadata_slot:
        additions.pop('measurement', None)
    declared = merge(declared, AnnotationDeclaration(**additions))
    for key, question in (
            ('species', 'Specify the source species.'), ('assembly', 'Specify the genome assembly.'),
            ('measurement', 'Describe the actual region measurement: raw peak counts, fragment-overlap counts or binary accessibility.'),
            ('context', 'Specify the tissue context for the marker resource.')):
        if getattr(declared, key) is None:
            missing.append(question)
    missing.extend(c.input_problems(declared.species, declared.assembly, declared.measurement))
    if 'count_slot' not in values and 'count_slot' in observation['declarations']:
        values['count_slot'] = observation['declarations']['count_slot']
    # Historical clarification text is immutable. Only old captured observations
    # retain the retired mandatory-slot question; new owner defaults use X.
    if 'count_slot' not in values and 'interface_version' not in observation:
        missing.append('Identify the raw-count slot: X or layers/name.')
    if observation.get('interface_version', 2) not in (2, 3, 4):
        raise ValueError('Unsupported annotation handoff version.')
    if declared.groups_result is not None:
        offered = [r for r in observation.get('accepted_groups', ()) if r['handle'] == declared.groups_result]
        if len(offered) != 1 or 'group_column' in values:
            raise ValueError('Select one offered accepted clustering result without an obs-column override.')
        declared = replace(declared, grouping_provenance='Accepted independent Leiden result ' + declared.groups_result)
        if observation.get('group_projection_problem'):
            missing.append(observation['group_projection_problem'])
    elif declared.groups_source is not None:
        if 'group_column' in values:
            raise ValueError('Select an obs grouping column or an ordered group TSV, not both.')
    elif 'group_column' not in values:
        qualified = observation['qualified_groupings']
        if len(qualified) == 1:
            values['group_column'] = qualified[0]['column']
            if declared.grouping_provenance is None:
                declared = replace(declared, grouping_provenance=qualified[0]['provenance'])
        elif observation.get('interface_version') == 4 and not qualified and declared.grouping_strategy != 'reuse_only':
            # Planning owns computing missing groups. This is a prerequisite
            # declaration, never a claim that clustering has already succeeded.
            declared = replace(declared, grouping_provenance='Exact-source independent clustering selected by the scientific plan')
        else:
            columns = observation['grouping_columns']
            missing.append('Choose the independently derived clustering column in obs'
                + (': ' + ', '.join(columns) if columns else ', or attach an exact ordered cell_id/group TSV')
                + '. Sample, batch and condition columns are not automatically treated as clusters.')
    if 'group_column' in values:
        if values['group_column'] not in observation['grouping_columns']:
            missing.append('The selected obs column is absent or is not a valid complete grouping; it cannot be replaced automatically.')
        for row in observation['qualified_groupings']:
            if row['column'] == values['group_column'] and declared.grouping_provenance is None:
                declared = replace(declared, grouping_provenance=row['provenance'])
    if declared.grouping_provenance is None:
        missing.append('Describe how the chosen groups were derived and whether they were independent of marker/signature inspection.')
    return declared, values, missing


def _annotation_binding(candidates, declared, selected, arguments, pointer, projection=None):
    from .local_resources import RegisteredInputGroup, group_role_inputs
    from agent.orchestration.registry import build_default_tool_registry
    from agent.tools.analysis.annotation_contract import EXTERNAL_PRIMARY, EXTERNAL_SEMANTICS, REGION_PRIMARY
    roles = [dict(tool='annotate_scATAC_cell_types', port='source', source=declared.source)]
    if declared.groups_source is not None:
        roles.append(dict(tool='annotate_scATAC_cell_types', port='groups', source=declared.groups_source))
    members = list(candidates.members)
    if projection is not None:
        if declared.groups_result != projection['selection']['handle']:
            raise ValueError('Grouping projection differs from the explicit selection')
        roles.append(dict(tool='annotate_scATAC_cell_types', port='groups', source=f'source.{len(members)}'))
        members.append(projection['member'])
    values = dict(arguments, annotation_method=EXTERNAL_PRIMARY if declared.measurement == EXTERNAL_SEMANTICS else REGION_PRIMARY,
                  annotation_spec_path=pointer['path'], annotation_spec_sha256=pointer['sha256'])
    bound = RegisteredInputGroup(members, values, roles,
        resource_choice=None if selected is None else selected.resource_choice(), attachment_candidates=True)
    return replace(bound, execution_inputs=group_role_inputs(bound, roles, build_default_tool_registry()))


def _prepare_annotation(owner, candidates, declared, selected, arguments, *, projection=None):
    """Bind a reviewed resource to declared grouping through its existing owner."""
    from agent.tools.analysis import annotation_contract as c
    from agent.tools.analysis.marker_annotation import sha256
    identity = _context_identity(candidates, declared, selected)
    directory = owner._workspace._ensure_directory(owner._workspace.root / 'annotation_handoffs')
    root = owner._workspace._ensure_directory(directory / identity)
    from agent.tools.analysis.annotation_input import resource_base, resource_specification
    specification, problems = resource_specification(resource_base(selected.inputs()) if selected else None, declared, candidates.members)
    if problems:
        raise ValueError('Incomplete uploaded annotation resources.')
    content = c.matrix.canonical(specification)
    path = root / 'resources.json'
    # Ordinary immutable workspace publication, never mutate selected resources.
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError('Captured annotation resource binding changed.')
    else:
        import os, tempfile
        fd, temporary = tempfile.mkstemp(dir=root, prefix='.resources-')
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(content); stream.flush(); os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != content:
                    raise ValueError('Captured annotation resource binding changed.')
            directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            Path(temporary).unlink(missing_ok=True)
    pointer = dict(path=str(path), sha256=sha256(path))
    bound = _annotation_binding(candidates, declared, selected, arguments, pointer, projection)
    owner.validate_binding(bound)
    return bound, dict(identity=identity, specification=specification, resource=pointer,
                       **({'group_projection': projection} if projection is not None else {}))


def _validate_annotation_completion(admitted, candidates, declared, selected, arguments):
    """Replay the captured source/resource binding without scientific IO."""
    from .session_state import sha
    from agent.tools.analysis import annotation_contract as c
    receipt = admitted['annotation_context']
    if (set(receipt) != {'identity', 'specification', 'resource',
                        *(['group_projection'] if declared.groups_result is not None else [])}
            or receipt['identity'] != _context_identity(candidates, declared, selected)
            or admitted['kind'] != 'execute' or admitted.get('operation') != 'plan'):
        raise ValueError('Invalid annotation handoff receipt.')
    from agent.tools.analysis.annotation_input import resource_specification
    spec, problems = resource_specification(admitted['raw_resource_evidence'][selected.input_set_id] if selected else None, declared, candidates.members)
    if problems or declared.uploaded is not None and (not admitted.get('annotation_resource_observation')
            or not admitted['annotation_resource_observation']['qualification']
            or admitted['annotation_resource_observation']['problem']):
        raise ValueError('Missing uploaded resource qualification.')
    pointer = receipt['resource']; path = Path(pointer['path'])
    sha(pointer['sha256'])
    if (receipt['specification'] != spec or set(pointer) != {'path', 'sha256'}
            or not path.is_absolute() or '..' in path.parts or path.name != 'resources.json'
            or path.parent.name != receipt['identity'] or path.parent.parent.name != 'annotation_handoffs'):
        raise ValueError('Annotation resource derivation changed.')
    import hashlib
    if hashlib.sha256(c.matrix.canonical(spec)).hexdigest() != pointer['sha256']:
        raise ValueError('Derived annotation resource digest changed.')
    projection = receipt.get('group_projection')
    if projection is not None and projection['selection'] not in admitted['annotation_source_evidence'].get('accepted_groups', ()):
        raise ValueError('Grouping projection is outside captured accepted results')
    bound = _annotation_binding(candidates, declared, selected, arguments, pointer, projection)
    if _serialize(admitted['inputs']) != _serialize(bound.execution_inputs) or admitted['registered_input'] != bound.attribution():
        raise ValueError('Annotation binding changed.')


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
