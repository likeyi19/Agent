"""Durable, byte-pinned local sources; registration confers no scientific authority.

Sources remain at their operator-approved locations. New consumption verifies the
pinned bytes; reading historical attribution does not require the source to exist.
This boundary shares the application's trusted-local filesystem lifetime contract.
"""
from contextlib import contextmanager
from collections.abc import Mapping
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile

from agent.schemas.orchestration import freeze_json_mapping, _serialize
from .interactive_schemas import ClientError
from .session_state import canonical, digest
from .workspace import ManagedWorkspace, ApplicationWorkspaceError


MAX_RECORD_BYTES = 16_384
MAX_INPUT_BYTES = 65_536
MAX_COLLECTION_MEMBERS = 128
MAX_COLLECTION_BYTES = 32_768
MAX_DISCOVERY_RECORDS = 1024
MAX_DISCOVERY_CHOICES = 128
MAX_GROUP_MEMBERS = 8
_GROUP_COMPOSITION = 'scientific-role-selection.v1'
# Reviewed source ports only; their aliases, roles and types remain Registry facts.
_GROUP_PORTS = (
    ('cluster_cells', 'analysis'),
    ('evaluate_cell_clustering', 'analysis'),
    ('evaluate_cell_clustering', 'ground_truth_dataset'),
    ('transfer_cell_labels', 'reference_dataset'),
    ('transfer_cell_labels', 'query_dataset'),
    ('annotate_scATAC_cell_types', 'source'),
    ('annotate_scATAC_cell_types', 'groups'),
)
_FORMAT = 'agent.local-resource.v1'
_ID = re.compile(r'local-[0-9a-f]{64}')
_SHA = re.compile(r'[0-9a-f]{64}')
_INPUT_TYPES = ('h5ad', 'external_fragments', 'bam', 'fastq', 'groups', 'fasta', 'regulatory_reference', 'genome_reference')
_OPERATION_TYPES = {
    'inspect_scATAC': 'h5ad',
    'adopt_scATAC_cell_by_ccre': 'h5ad',
    'import_scATAC_fragments': 'external_fragments',
    'inspect_raw_scATAC': 'bam',
    'prepare_scATAC_bam_fragments': 'bam',
}
_COLLECTION_OPERATIONS = ('inspect_raw_scATAC', 'prepare_scATAC_fragments')
_H5AD_COMPOSITION = 'h5ad-science.v1'
_FRAGMENTS_COMPOSITION = 'external-fragments-science.v1'
_BAM_COMPOSITION = 'bam-science.v1'
_FASTQ_COMPOSITION = 'fastq-science.v1'
_FASTQ_COMPANION_TOOLS = ('inspect_raw_scATAC', 'prepare_scATAC_fragments',
                        'compute_scATAC_qc', 'select_scATAC_cells', 'build_scATAC_cell_by_ccre')
_FASTQ_SOURCE_ARGUMENTS = {'raw_input_paths', 'source_path', 'source_sha256',
    'source_index_path', 'source_index_sha256', 'output_dir',
    'fragments_manifest_path', 'fragments_manifest_sha256',
    'barcode_qc_manifest_path', 'barcode_qc_manifest_sha256',
    'selected_cells_manifest_path', 'selected_cells_manifest_sha256'}
_FASTQ_SELECTION_ERRORS = ('FASTQ_RESOURCE_AMBIGUOUS',)
_BAM_COMPANION_TOOLS = ('inspect_raw_scATAC', 'prepare_scATAC_bam_fragments',
                       'compute_scATAC_qc', 'select_scATAC_cells', 'build_scATAC_cell_by_ccre')
_BAM_SOURCE_ARGUMENTS = {'raw_input_paths', 'source_path', 'source_sha256',
    'source_index_path', 'source_index_sha256', 'output_dir',
    'fragments_manifest_path', 'fragments_manifest_sha256',
    'barcode_qc_manifest_path', 'barcode_qc_manifest_sha256',
    'selected_cells_manifest_path', 'selected_cells_manifest_sha256'}
_BAM_SELECTION_ERRORS = ('BAM_RESOURCE_AMBIGUOUS',)
_FRAGMENTS_COMPANION_TOOLS = ('import_scATAC_fragments', 'compute_scATAC_qc',
                              'select_scATAC_cells', 'build_scATAC_cell_by_ccre')
_FRAGMENTS_SOURCE_ARGUMENTS = {'source_path', 'source_sha256', 'source_index_path',
    'source_index_sha256', 'output_dir', 'fragments_manifest_path', 'fragments_manifest_sha256',
    'barcode_qc_manifest_path', 'barcode_qc_manifest_sha256',
    'selected_cells_manifest_path', 'selected_cells_manifest_sha256'}
_H5AD_COMPANION_TOOLS = ('epizoo_embed_cells', 'build_cell_neighbors',
                         'cluster_cells', 'compute_cell_umap', 'adopt_scATAC_cell_by_features', 'adapt_epizoo_species')
_H5AD_SOURCE_ARGUMENTS = {'input_path', 'path', 'output_dir', 'embedding_path',
                          'cell_ids_path', 'analysis_path', 'source_path', 'source_sha256',
                          'matrix_manifest_path', 'matrix_manifest_sha256'}
_RESOURCE_SELECTION_ERRORS = ('EPIZOO_RESOURCE_REQUIRED', 'EPIZOO_RESOURCE_AMBIGUOUS')
_MESSAGES = {
    'MULTI_INPUT_ROLE_INVALID': 'The selected inputs need distinct, supported scientific roles.',
    'MULTI_INPUT_SOURCE_MISMATCH': 'The plan does not consume the exact selected inputs in their declared scientific roles.',
    'LOCAL_RESOURCE_UNAVAILABLE': 'The registered local resource is unavailable.',
    'LOCAL_RESOURCE_ACCESS_INVALID': 'The local source is unavailable or outside the approved source policy.',
    'LOCAL_RESOURCE_INTEGRITY_INVALID': 'The local source no longer matches its registered byte identity.',
    'LOCAL_RESOURCE_TYPE_UNSUPPORTED': 'The declared local input type is not supported.',
    'LOCAL_RESOURCE_RECORD_INVALID': 'The local resource record is incomplete, corrupt, or invalid.',
    'LOCAL_RESOURCE_CONFLICT': 'This registration identity already belongs to a different source record.',
    'LOCAL_RESOURCE_BINDING_INVALID': 'The registered input binding or its explicit scientific declarations are invalid.',
    'LOCAL_RESOURCE_CONSUMPTION_MISMATCH': 'The plan does not consume the exact selected scientific resource bindings.',
    'LOCAL_RESOURCE_OPERATION_UNSUPPORTED': 'This operation is not supported by the local resource binding boundary.',
    'LOCAL_RESOURCE_DECLARATION_REQUIRED': 'Required explicit scientific input declarations are missing.',
    'LOCAL_RESOURCE_DISCOVERY_LIMIT': 'The registered resource listing exceeds its supported bound.',
    'FRAGMENTS_SOURCE_REQUIRED': 'Upload and select an external fragments source before using these declarations.',
    'FRAGMENTS_RESOURCE_REQUIRED': 'Select approved fragments reference and source declarations, or ask the operator to configure them.',
    'FRAGMENTS_RESOURCE_AMBIGUOUS': 'More than one fragments default is configured. Select the reference and source declarations explicitly.',
    'FRAGMENTS_REFERENCE_REQUIRED': 'Select an approved reference bundle for the declared species and assembly.',
    'FRAGMENTS_PROFILE_REQUIRED': 'Declare the supported external fragments source profile.',
    'FRAGMENTS_NAMESPACE_REQUIRED': 'Declare the processing namespace for this fragments source.',
    'FRAGMENTS_SOURCE_MISMATCH': 'The plan does not consume the selected registered fragments source and declarations.',
    'BAM_SOURCE_REQUIRED': 'Upload and select a BAM source before using this qualified context.',
    'BAM_RESOURCE_AMBIGUOUS': 'More than one BAM context default is configured. Select the qualified context explicitly before preparing fragments.',
    'BAM_LIBRARY_CONTEXT_REQUIRED': 'Select an approved paired-ATAC library context with established corrected-CB declarations for this exact BAM.',
    'BAM_REFERENCE_REQUIRED': 'Select an approved human/hg38 or mouse/mm10 reference bundle compatible with this BAM.',
    'BAM_PROFILE_REQUIRED': 'Select the qualified paired-ATAC BAM source profile before preparing fragments.',
    'BAM_INTAKE_REQUIRED': 'Provide the verified intake for this exact BAM, or request an inspection as an explicit part of the scientific plan.',
    'BAM_SOURCE_MISMATCH': 'The plan does not consume the selected registered BAM and qualified context.',
    'FASTQ_SOURCE_REQUIRED': 'Upload and complete a FASTQ library before using this qualified context.',
    'FASTQ_COLLECTION_REQUIRED': 'Complete and select the FASTQ library before requesting a scientific operation on its members.',
    'FASTQ_COLLECTION_INVALID': 'The FASTQ library declarations conflict or use an unsupported layout or role.',
    'FASTQ_COLLECTION_INCOMPLETE': 'The FASTQ library is missing required read roles.',
    'FASTQ_RESOURCE_AMBIGUOUS': 'More than one FASTQ context default is configured. Select the qualified context explicitly before preparing fragments.',
    'FASTQ_LIBRARY_CONTEXT_REQUIRED': 'Select an approved library context binding these exact FASTQ sources, barcode policy, namespace and applicable whitelist.',
    'FASTQ_REFERENCE_REQUIRED': 'Select an approved human/hg38 or mouse/mm10 reference bundle compatible with this FASTQ library.',
    'FASTQ_INTAKE_REQUIRED': 'Provide the verified intake for these exact FASTQs, or request inspection explicitly in the scientific plan.',
    'FASTQ_SOURCE_MISMATCH': 'The plan does not consume the selected registered FASTQ library and qualified context.',
    'H5AD_SPECIES_REQUIRED': 'Declare the dataset species as human or mouse using a scientific input selection.',
    'H5AD_SOURCE_MISMATCH': 'The plan does not consume the selected registered H5AD.',
    'EPIZOO_RESOURCE_REQUIRED': 'Select a qualified EpiZoo resource, or ask the operator to configure an applicable qualified default.',
    'EPIZOO_RESOURCE_AMBIGUOUS': 'More than one qualified EpiZoo default applies. Select one explicitly.',
    'EPIZOO_RESOURCE_SELECTION_INVALID': 'The selected EpiZoo resource is unavailable or conflicts with the declared species.',
    'EPIZOO_RESOURCE_IDENTITY_INVALID': 'The selected EpiZoo resource identity is incomplete or invalid.',
    'EPIZOO_RESOURCE_IDENTITY_MISMATCH': 'The EpiZoo resources no longer match the selected qualified identity.',
    'EPIZOO_RESOURCE_CONSUMPTION_MISMATCH': 'The plan does not consume the selected EpiZoo resource and declarations.',
}


class ResourceConfigurationError(ValueError):
    """Invalid operator configuration; never returned as client diagnostics."""


@dataclass(frozen=True)
class QualifiedEpiZooResource:
    """Operator-reviewed exact resources for the existing embedding owner.

    These declarations record qualification and content pins. The scientific
    owner remains responsible for checking the resources actually consumed.
    """

    resource_id: str
    label: str
    species: str
    checkpoint_path: str
    checkpoint_sha256: str
    frequencies_sha256: str
    filter_indices_sha256: str
    qualification: str
    default: bool = False

    def __post_init__(self):
        if (type(self.resource_id) is not str
                or re.fullmatch(r'[a-z][a-z0-9_-]{0,63}', self.resource_id) is None):
            raise ResourceConfigurationError('EpiZoo resource identifiers must be safe lowercase identifiers.')
        for label in (self.label, self.qualification):
            if (type(label) is not str or not label.strip() or len(label) > 512
                    or any(not c.isprintable() for c in label)):
                raise ResourceConfigurationError('Display labels must be nonempty printable text.')
        if self.species not in ('human', 'mouse') or type(self.species) is not str:
            raise ResourceConfigurationError('EpiZoo resources require an explicit supported species.')
        if (type(self.checkpoint_path) is not str or not self.checkpoint_path.strip()
                or len(self.checkpoint_path) > 4096 or '\x00' in self.checkpoint_path):
            raise ResourceConfigurationError('EpiZoo checkpoint paths must be explicitly configured.')
        try:
            from agent.tools.analysis.epizoo_embedding import validate_expected_resource_identity
            validate_expected_resource_identity(self.inputs()['expected_resource_identity'])
        except (ValueError, TypeError) as exc:
            raise ResourceConfigurationError('EpiZoo resource content identities must be exact approved SHA-256 pins.') from exc
        if type(self.default) is not bool:
            raise ResourceConfigurationError('EpiZoo default designation must be boolean.')

    def choice(self):
        return dict(resource_id=self.resource_id, display_label=self.label,
                    species=self.species, is_default=self.default)

    def inputs(self):
        return dict(checkpoint_path=self.checkpoint_path, expected_resource_identity=dict(
            resource_id=self.resource_id, checkpoint_sha256=self.checkpoint_sha256,
            frequencies_sha256=self.frequencies_sha256, filter_indices_sha256=self.filter_indices_sha256))


def qualified_epizoo_resources(resources):
    """Validate the immutable operator resource catalog without loading science."""
    if (type(resources) is not tuple or len(resources) > 64
            or any(not isinstance(resource, QualifiedEpiZooResource) for resource in resources)
            or len({resource.resource_id for resource in resources}) != len(resources)):
        raise ResourceConfigurationError('EpiZoo resources must be a bounded unique configured tuple.')
    return resources


def select_epizoo_resource(inputs, selected_id, resources):
    """Resolve exact qualified selection from typed species, without science."""
    resources = qualified_epizoo_resources(resources)
    target = ('target_model_manifest_path', 'target_model_manifest_sha256')
    if any(k in inputs for k in target):
        if (not all(k in inputs for k in target) or selected_id is not None
                or {'checkpoint_path', 'expected_resource_identity'}.intersection(inputs)):
            raise ResourceAdmissionError('EPIZOO_RESOURCE_SELECTION_INVALID')
        # An exact operator InputSet may bind a target model. Eligibility and
        # current artifact integrity remain the scientific owner's responsibility.
        return inputs, None
    species = inputs.get('species')
    if selected_id is not None:
        selected = next((resource for resource in resources if resource.resource_id == selected_id), None)
        if selected is None:
            raise ResourceAdmissionError('EPIZOO_RESOURCE_SELECTION_INVALID')
        # Retain an explicit qualified choice while species is outstanding.
        # This choice supplies no species declaration or scientific compatibility.
        if species is not None and species != selected.species:
            raise ResourceAdmissionError('EPIZOO_RESOURCE_SELECTION_INVALID')
    else:
        applicable = tuple(resource for resource in resources if resource.default and resource.species == species)
        if len(applicable) != 1:
            # Only actual embedding admission consumes this prerequisite.
            return inputs, ('EPIZOO_RESOURCE_REQUIRED' if not applicable else 'EPIZOO_RESOURCE_AMBIGUOUS')
        selected = applicable[0]
    return {**inputs, **selected.inputs()}, None


class ResourceAdmissionError(ValueError):
    """A sanitized application failure, distinct from scientific qualification."""
    def __init__(self, code):
        self.code = code
        self.message = _MESSAGES[code]
        self.error = ClientError(code, self.message)
        super().__init__(self.message)


def _fail(code):
    return ResourceAdmissionError(code)


def _text(value, maximum, code, *, display=False):
    if (type(value) is not str or not value.strip() or len(value) > maximum
            or any(not char.isprintable() for char in value)):
        raise _fail(code)
    if display and (value.startswith(('/', '~/', 'file://'))
                    or re.match(r'^[A-Za-z]:[\\/]', value)):
        raise _fail(code)
    return value


def _identifier(value):
    if type(value) is not str or _ID.fullmatch(value) is None:
        raise _fail('LOCAL_RESOURCE_RECORD_INVALID')
    return value


def _source_text(value):
    _text(value, 4096, 'LOCAL_RESOURCE_RECORD_INVALID')
    path = Path(value)
    if (not path.is_absolute() or str(path) != value or '..' in path.parts
            or '://' in value):
        raise _fail('LOCAL_RESOURCE_RECORD_INVALID')
    return path


def _input_mapping(values):
    try:
        # Import after module construction: the interactive facade imports this
        # owner but owns the existing safe structured-input boundary.
        from .interactive import _inputs
        result = _inputs(values)
        if len(canonical(result)) > MAX_INPUT_BYTES:
            raise ValueError('Input bound exceeded.')
        return result
    except (ValueError, TypeError, RecursionError, UnicodeError) as exc:
        raise _fail('LOCAL_RESOURCE_BINDING_INVALID') from exc


def _fastq_member(values):
    """Typed read attribution, using only the existing owner's supported enums."""
    from agent.tools.data import _raw_fastq as owner
    try:
        if (not isinstance(values, Mapping)
                or set(values) != {'library_id', 'fastq_layout', 'role', 'lane', 'chunk', 'compression'}
                or type(values['library_id']) is not str
                or any(c in values['library_id'] for c in '/\\:')
                or values['fastq_layout'] not in (owner.FastqLayout.A.value, owner.FastqLayout.B.value)
                or any(type(values[key]) is not str or re.fullmatch(r'[0-9]{3}', values[key]) is None
                       for key in ('lane', 'chunk'))
                or values['compression'] not in ('plain', 'gzip')):
            raise ValueError('Invalid explicit read attribution.')
        _text(values['library_id'], 64, 'FASTQ_COLLECTION_INVALID', display=True)
        role = owner.ReadRole(values['role'])
        if role not in owner.LAYOUT_ROLES[owner.FastqLayout(values['fastq_layout'])]:
            raise ValueError('Role conflicts with the declared layout.')
        return freeze_json_mapping(dict(values), 'fastq_member')
    except (ValueError, TypeError, KeyError) as exc:
        raise _fail('FASTQ_COLLECTION_INVALID') from exc


@dataclass(frozen=True)
class LocalResourceRecord:
    resource_id: str
    input_type: str
    label: str
    source_path: str
    source_sha256: str
    size_bytes: int
    attribution: str
    source_index: object | None = None
    fastq_member: object | None = None
    reference_qualification: object | None = None

    def __post_init__(self):
        _identifier(self.resource_id)
        if self.input_type not in _INPUT_TYPES:
            raise _fail('LOCAL_RESOURCE_TYPE_UNSUPPORTED')
        _text(self.label, 160, 'LOCAL_RESOURCE_RECORD_INVALID', display=True)
        _source_text(self.source_path)
        if type(self.source_sha256) is not str or _SHA.fullmatch(self.source_sha256) is None:
            raise _fail('LOCAL_RESOURCE_RECORD_INVALID')
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise _fail('LOCAL_RESOURCE_RECORD_INVALID')
        _text(self.attribution, 1024, 'LOCAL_RESOURCE_RECORD_INVALID')
        if self.source_index is not None:
            if (self.input_type not in ('external_fragments', 'fasta', 'genome_reference') or not isinstance(self.source_index, Mapping)
                    or set(self.source_index) != {'path', 'sha256', 'size_bytes'}):
                raise _fail('LOCAL_RESOURCE_RECORD_INVALID')
            index = dict(self.source_index)
            _source_text(index['path'])
            if (index['path'] == self.source_path or type(index['sha256']) is not str
                    or _SHA.fullmatch(index['sha256']) is None
                    or type(index['size_bytes']) is not int or index['size_bytes'] < 0):
                raise _fail('LOCAL_RESOURCE_RECORD_INVALID')
            object.__setattr__(self, 'source_index', freeze_json_mapping(index, 'source_index'))
        if self.fastq_member is not None:
            if self.input_type != 'fastq':
                raise _fail('LOCAL_RESOURCE_RECORD_INVALID')
            object.__setattr__(self, 'fastq_member', _fastq_member(self.fastq_member))

        if self.reference_qualification is not None:
            from .reference_supplementation import validate_record
            validate_record(self)
            object.__setattr__(self, 'reference_qualification', freeze_json_mapping(self.reference_qualification, 'reference_qualification'))
        elif self.input_type in ('regulatory_reference', 'genome_reference'):
            raise _fail('LOCAL_RESOURCE_RECORD_INVALID')

    def to_dict(self):
        """Private persistence/binding metadata; never a client representation."""
        # Old unpaired records retain their exact persistence and binding digest.
        return {name: _serialize(getattr(self, name)) for name in self.__dataclass_fields__
                if name not in {'source_index', 'fastq_member', 'reference_qualification'} or getattr(self, name) is not None}

    @property
    def record_sha256(self):
        return digest(self.to_dict())

    def public(self):
        choice = dict(resource_id=self.resource_id, input_type=self.input_type,
                      label=self.label, status='registered')
        if self.source_index is not None:
            choice['has_source_index'] = True
        if self.fastq_member is not None:
            choice['fastq_attribution'] = _serialize(self.fastq_member)
        return choice


@dataclass(frozen=True)
class RegisteredInput:
    resource_id: str
    record_sha256: str
    tool_name: str
    execution_inputs: object
    composition: str | None = None
    resource_selection_error: str | None = None
    resource_choice: object = None

    def __post_init__(self):
        if type(self.tool_name) is not str or self.tool_name not in _OPERATION_TYPES:
            raise _fail('LOCAL_RESOURCE_OPERATION_UNSUPPORTED')
        if (self.composition not in (None, _H5AD_COMPOSITION, _FRAGMENTS_COMPOSITION, _BAM_COMPOSITION)
                or (self.composition == _H5AD_COMPOSITION and self.tool_name != 'inspect_scATAC')
                or (self.composition == _FRAGMENTS_COMPOSITION and self.tool_name != 'import_scATAC_fragments')
                or (self.composition == _BAM_COMPOSITION and self.tool_name != 'inspect_raw_scATAC')
                or (self.resource_selection_error is not None
                    and not (self.composition == _H5AD_COMPOSITION
                             and self.resource_selection_error in _RESOURCE_SELECTION_ERRORS
                             or self.composition == _BAM_COMPOSITION
                             and self.resource_selection_error in _BAM_SELECTION_ERRORS
                             or self.composition == _FRAGMENTS_COMPOSITION
                             and self.resource_selection_error == 'FRAGMENTS_RESOURCE_AMBIGUOUS'))):
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID')
        try:
            _identifier(self.resource_id)
            if type(self.record_sha256) is not str or _SHA.fullmatch(self.record_sha256) is None:
                raise ValueError('Invalid record digest.')
            values = _input_mapping(self.execution_inputs)
            object.__setattr__(self, 'execution_inputs', freeze_json_mapping(values, 'execution_inputs'))
            object.__setattr__(self, 'resource_choice', _resource_choice(self.resource_choice))
        except (ValueError, TypeError) as exc:
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID') from exc

    def attribution(self):
        result = dict(resource_id=self.resource_id, record_sha256=self.record_sha256,
                      tool_name=self.tool_name)
        if self.composition is not None:
            result['composition'] = self.composition
        if self.resource_selection_error is not None:
            result['resource_selection_error'] = self.resource_selection_error
        if self.resource_choice is not None:
            result['resource_choice'] = _serialize(self.resource_choice)
        return result


def _resource_choice(value):
    """Capture operator choice identity; this token grants no scientific authority.

    The trusted composition boundary supplies the definition digest. Scientific
    paths and resource pins remain in the existing typed execution inputs.
    """
    if value is None:
        return None
    if (not isinstance(value, Mapping)
            or set(value) != {'input_set_id', 'definition_sha256'}
            or type(value['input_set_id']) is not str
            or re.fullmatch(r'[a-z][a-z0-9_-]{0,63}', value['input_set_id']) is None
            or type(value['definition_sha256']) is not str
            or _SHA.fullmatch(value['definition_sha256']) is None):
        raise _fail('LOCAL_RESOURCE_BINDING_INVALID')
    return freeze_json_mapping(dict(value), 'resource_choice')


@dataclass(frozen=True)
class RegisteredInputGroup:
    """Ordered independent records captured in the existing turn, not a library.

    Private source locators allow exact immutable replay without reopening files.
    Only detached opaque handles, positions and safe labels reach the Interpreter.
    """

    members: object
    execution_inputs: object
    roles: object = ()
    composition: str = _GROUP_COMPOSITION
    resource_choice: object = None

    def __post_init__(self):
        try:
            if self.composition != _GROUP_COMPOSITION or not 1 <= len(self.members) <= MAX_GROUP_MEMBERS:
                raise ValueError('Invalid source selection.')
            members = []
            for member in self.members:
                if (not isinstance(member, Mapping) or set(member) - {'source_sha256'} != {
                        'resource_id', 'record_sha256', 'source_path', 'input_type', 'label'}):
                    raise ValueError('Invalid selected record.')
                _identifier(member['resource_id'])
                if type(member['record_sha256']) is not str or _SHA.fullmatch(member['record_sha256']) is None:
                    raise ValueError('Invalid record identity.')
                if member['input_type'] not in ('h5ad', 'groups'):
                    raise ValueError('Unsupported selected source kind.')
                if 'source_sha256' in member and (type(member['source_sha256']) is not str
                        or _SHA.fullmatch(member['source_sha256']) is None):
                    raise ValueError('Invalid selected source hash.')
                _source_text(member['source_path'])
                _text(member['label'], 160, 'LOCAL_RESOURCE_RECORD_INVALID', display=True)
                members.append(dict(member))
            if (len({m['resource_id'] for m in members}) != len(members)
                    or len({m['source_path'] for m in members}) != len(members)):
                raise ValueError('Repeated source selection.')
            roles = _group_roles(self.roles, len(members))
            object.__setattr__(self, 'members', tuple(freeze_json_mapping(m, 'member') for m in members))
            object.__setattr__(self, 'roles', tuple(freeze_json_mapping(r, 'role') for r in roles))
            object.__setattr__(self, 'execution_inputs', freeze_json_mapping(
                _input_mapping(self.execution_inputs), 'execution_inputs'))
            object.__setattr__(self, 'resource_choice', _resource_choice(self.resource_choice))
        except (ValueError, TypeError, KeyError) as exc:
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID') from exc

    def attribution(self):
        result = dict(composition=self.composition, members=_serialize(self.members), roles=_serialize(self.roles))
        if self.resource_choice is not None:
            result['resource_choice'] = _serialize(self.resource_choice)
        return result


def _group_roles(values, member_count):
    if not isinstance(values, (tuple, list)) or len(values) > MAX_GROUP_MEMBERS:
        raise ValueError('Invalid roles.')
    roles = []
    for role in values:
        if (not isinstance(role, Mapping) or set(role) != {'tool', 'port', 'source'}
                or (role['tool'], role['port']) not in _GROUP_PORTS
                or role['source'] not in {f'source.{i}' for i in range(member_count)}):
            raise ValueError('Unsupported source role.')
        roles.append(dict(role))
    if (len({(r['tool'], r['port']) for r in roles}) != len(roles)
            or len({r['source'] for r in roles}) != len(roles)):
        raise ValueError('Duplicate or conflicting roles.')
    return sorted(roles, key=lambda r: (r['tool'], r['port']))


def _group_port(registry, tool_name, port_name):
    """One reviewed source port, including atomic path/digest bundles."""
    if (tool_name, port_name) not in _GROUP_PORTS:
        raise _fail('MULTI_INPUT_ROLE_INVALID')
    spec = registry.get(tool_name)
    ports = [p for p in spec.semantic_planning.consumer_ports if p.name == port_name]
    if len(ports) != 1 or len(ports[0].members) not in (1, 2):
        raise _fail('MULTI_INPUT_ROLE_INVALID')
    port = ports[0]
    member = port.members[0]
    sources = [s for s in port.request_sources if tuple(m.name for m in s.members) == tuple(m.name for m in port.members)]
    if len(port.members) == 2 and tuple(m.name for m in port.members) != ('path', 'sha256'):
        raise _fail('MULTI_INPUT_ROLE_INVALID')
    exact = [s for s in sources if s.members[0].input_name == member.field_name]
    choices = exact if exact else sources
    if len(choices) != 1:
        raise _fail('MULTI_INPUT_ROLE_INVALID')
    argument = (dict(spec.required_arguments) | dict(spec.optional_arguments))[member.field_name]
    if argument.planning is None or not argument.planning.accepted_artifact_kinds:
        raise _fail('MULTI_INPUT_ROLE_INVALID')
    return port, choices[0], argument


def group_choices(binding, registry):
    """Safe conditional Interpreter projection from captured ordered membership."""
    if type(binding) is not RegisteredInputGroup:
        raise _fail('LOCAL_RESOURCE_BINDING_INVALID')
    roles = []
    for tool, name in _GROUP_PORTS:
        _, source, argument = _group_port(registry, tool, name)
        role = source.lineage or argument.planning.provenance_role
        roles.append(dict(tool=tool, port=name, role=role.value if role is not None else None))
    return dict(sources=[dict(source=f'source.{i}', position=i + 1,
        label=m['label'], input_type=m['input_type']) for i, m in enumerate(binding.members)], roles=roles)


def _group_declarations(registry):
    """Existing Registry companion fields; selected dataset paths are excluded."""
    aliases = set()
    for tool, port_name in _GROUP_PORTS:
        port, _, _ = _group_port(registry, tool, port_name)
        aliases.update(m.input_name for s in port.request_sources for m in s.members)
        aliases.update(m.field_name for m in port.members)
    declarations = {}
    for tool in sorted({tool for tool, _ in _GROUP_PORTS}):
        spec = registry.get(tool)
        arguments = dict(spec.required_arguments) | dict(spec.optional_arguments)
        for port in spec.semantic_planning.consumer_ports:
            fields = {m.name: m.field_name for m in port.members}
            for source in port.request_sources:
                for member in source.members:
                    if member.input_name not in aliases and fields[member.name] != 'output_dir':
                        declarations.setdefault(member.input_name, []).append(arguments[fields[member.name]])
    return aliases, declarations


def group_role_inputs(binding, roles, registry):
    """Pure exact role replay; content qualification stays with scientific owners."""
    try:
        roles = _group_roles(roles, len(binding.members))
        if roles and {r['source'] for r in roles} != {f'source.{i}' for i in range(len(binding.members))}:
            raise ValueError('Unassigned selected input.')
        aliases, declarations = _group_declarations(registry)
        values = _serialize(binding.execution_inputs)
        existing = {m.input_name for role in binding.roles
                    for m in _group_port(registry, role['tool'], role['port'])[1].members}
        companions = {key: value for key, value in values.items() if key not in existing}
        if set(companions) - set(declarations):
            raise ValueError('Unsupported companion input or substituted dataset path.')
        for key, value in companions.items():
            for argument in declarations[key]:
                argument.validate(key, value)
        for role in roles:
            port, source, argument = _group_port(registry, role['tool'], role['port'])
            spec = registry.get(role['tool'])
            arguments = dict(spec.required_arguments) | dict(spec.optional_arguments)
            fields = {m.name: m.field_name for m in port.members}
            record = binding.members[int(role['source'].split('.')[1])]
            wanted_type = 'groups' if (role['tool'], role['port']) == ('annotate_scATAC_cell_types', 'groups') else 'h5ad'
            if record['input_type'] != wanted_type:
                raise ValueError('File type does not match the explicit scientific role.')
            for member in source.members:
                key = member.input_name
                value = record['source_sha256'] if member.name == 'sha256' else record['source_path']
                if key in companions:
                    raise ValueError('Conflicting consumer source aliases.')
                arguments[fields[member.name]].validate(key, value)
                companions[key] = value
        return companions
    except (ValueError, TypeError, KeyError, LookupError) as exc:
        raise _fail('MULTI_INPUT_ROLE_INVALID') from exc


def validate_group_roles(binding, roles, registry, required_roles, *, complete=False):
    """Validate a captured partial association without making executable inputs.

    Requirements name reviewed Registry consumers, not inferred workflow steps.
    The existing complete group binder remains the sole source-path resolver.
    """
    from .turn_decisions import InputRole, RoleRequirement
    try:
        if type(binding) is not RegisteredInputGroup or type(complete) is not bool:
            raise ValueError('Invalid captured selection.')
        if (not isinstance(required_roles, (tuple, list))
                or not len(binding.members) <= len(required_roles) <= MAX_GROUP_MEMBERS):
            raise ValueError('Invalid required scientific role count.')
        required = []
        for item in required_roles:
            value = dict(tool=item.tool, port=item.port) if isinstance(item, RoleRequirement) else item
            if not isinstance(value, Mapping) or set(value) != {'tool', 'port'}:
                raise ValueError('Invalid scientific role requirement.')
            required.append((value['tool'], value['port']))
        if len(set(required)) != len(required):
            raise ValueError('Duplicate scientific role requirements.')
        aliases = []
        for tool, port in required:
            _, source, _ = _group_port(registry, tool, port)
            aliases.append(source.members[0].input_name)
        if len(set(aliases)) != len(aliases):
            raise ValueError('Conflicting required scientific source aliases.')
        if not isinstance(roles, (tuple, list)):
            raise ValueError('Invalid scientific role associations.')
        values = _group_roles([dict(tool=r.tool, port=r.port, source=r.source)
                              if isinstance(r, InputRole) else r for r in roles], len(binding.members))
        if not {(r['tool'], r['port']) for r in values} <= set(required):
            raise ValueError('Unoffered required scientific role.')
        if complete and (len(values) != len(required)
                or {r['source'] for r in values} != {f'source.{i}' for i in range(len(binding.members))}):
            raise ValueError('Incomplete scientific source associations.')
        return tuple(InputRole(**role) for role in values)
    except (ValueError, TypeError, KeyError, LookupError) as exc:
        raise _fail('MULTI_INPUT_ROLE_INVALID') from exc


def _members(values):
    """Canonical exact references, without biological grouping or path discovery."""
    try:
        if type(values) not in (list, tuple) or not 1 <= len(values) <= MAX_COLLECTION_MEMBERS:
            raise ValueError('Invalid collection size.')
        members = []
        for value in values:
            if not isinstance(value, Mapping) or set(value) != {'resource_id', 'record_sha256'}:
                raise ValueError('Invalid member reference.')
            _identifier(value['resource_id'])
            sha = value['record_sha256']
            if type(sha) is not str or _SHA.fullmatch(sha) is None:
                raise ValueError('Invalid member digest.')
            members.append(dict(resource_id=value['resource_id'], record_sha256=sha))
        if len({member['resource_id'] for member in members}) != len(members):
            raise ValueError('Duplicate or conflicting member identity.')
        return sorted(members, key=lambda member: member['resource_id'])
    except (ValueError, TypeError, KeyError) as exc:
        raise _fail('LOCAL_RESOURCE_BINDING_INVALID') from exc


def _collection_sha256(members):
    return digest(dict(format='agent.local-resource-collection.v1', members=members))


@dataclass(frozen=True)
class LocalResourceCollectionRecord:
    """Durable choice in the same catalog, referencing existing byte records."""
    resource_id: str
    label: str
    members: object
    collection_sha256: str
    attribution: str
    input_type: str = 'fastq'

    def __post_init__(self):
        _identifier(self.resource_id)
        _text(self.label, 160, 'LOCAL_RESOURCE_RECORD_INVALID', display=True)
        _text(self.attribution, 1024, 'LOCAL_RESOURCE_RECORD_INVALID')
        if self.input_type != 'fastq':
            raise _fail('LOCAL_RESOURCE_RECORD_INVALID')
        members = _members(self.members)
        if self.collection_sha256 != _collection_sha256(members):
            raise _fail('LOCAL_RESOURCE_RECORD_INVALID')
        object.__setattr__(self, 'members', tuple(freeze_json_mapping(member, 'member') for member in members))
        if len(canonical(self.to_dict())) > MAX_COLLECTION_BYTES:
            raise _fail('LOCAL_RESOURCE_RECORD_INVALID')

    def to_dict(self):
        return {name: _serialize(getattr(self, name)) for name in self.__dataclass_fields__}

    @property
    def record_sha256(self):
        return digest(self.to_dict())

    def public(self):
        return dict(resource_id=self.resource_id, input_type=self.input_type,
                    label=self.label, status='registered', fastq_collection=True)


@dataclass(frozen=True)
class RegisteredInputCollection:
    """Inline immutable composition of existing records; no second resource store."""
    members: object
    collection_sha256: str
    tool_name: str
    execution_inputs: object
    composition: str | None = None
    resource_selection_error: str | None = None
    resource_choice: object = None

    def __post_init__(self):
        if type(self.tool_name) is not str or self.tool_name not in _COLLECTION_OPERATIONS:
            raise _fail('LOCAL_RESOURCE_OPERATION_UNSUPPORTED')
        if (self.composition not in (None, _FASTQ_COMPOSITION)
                or self.composition == _FASTQ_COMPOSITION and self.tool_name != 'inspect_raw_scATAC'
                or self.resource_selection_error is not None and not (
                    self.composition == _FASTQ_COMPOSITION
                    and self.resource_selection_error in _FASTQ_SELECTION_ERRORS)):
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID')
        members = _members(self.members)
        if self.collection_sha256 != _collection_sha256(members):
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID')
        object.__setattr__(self, 'members', tuple(freeze_json_mapping(member, 'member') for member in members))
        values = _input_mapping(self.execution_inputs)
        object.__setattr__(self, 'execution_inputs', freeze_json_mapping(values, 'execution_inputs'))
        object.__setattr__(self, 'resource_choice', _resource_choice(self.resource_choice))
        if len(canonical(self.attribution())) > MAX_COLLECTION_BYTES:
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID')

    def attribution(self):
        result = dict(members=[_serialize(member) for member in self.members],
                      collection_sha256=self.collection_sha256, tool_name=self.tool_name)
        if self.composition is not None:
            result['composition'] = self.composition
        if self.resource_selection_error is not None:
            result['resource_selection_error'] = self.resource_selection_error
        if self.resource_choice is not None:
            result['resource_choice'] = _serialize(self.resource_choice)
        return result


class LocalResourceAdmission:
    """Small application source store and resolver, with no scientific admission."""
    def __init__(self, workspace, *, approved_source_roots=(), registry=None):
        self._workspace = workspace if isinstance(workspace, ManagedWorkspace) else ManagedWorkspace(workspace)
        self._registry = registry
        try:
            roots = []
            for value in approved_source_roots:
                path = Path(value).expanduser().absolute()
                if path != path.resolve(strict=True) or not path.is_dir():
                    raise ValueError('Invalid source root.')
                roots.append(path)
            self._approved = tuple(sorted(set(roots)))
        except (ValueError, TypeError, OSError, RuntimeError) as exc:
            raise _fail('LOCAL_RESOURCE_ACCESS_INVALID') from exc

    def _directory(self, *, create=False):
        path = self._workspace.root / 'local_resources'
        try:
            if create:
                try:
                    return self._workspace._ensure_directory(path)
                except ApplicationWorkspaceError as exc:
                    if not isinstance(exc.__cause__, FileExistsError):
                        raise
                    # Concurrent first registrations can create the directory
                    # together. Recheck all workspace protections before reuse.
                    return self._workspace._ensure_directory(path)
            self._workspace._assert_contained(path)
            if path.is_symlink() or (path.exists() and (not path.is_dir() or path != path.resolve())):
                raise _fail('LOCAL_RESOURCE_RECORD_INVALID')
            if not path.exists():
                raise _fail('LOCAL_RESOURCE_UNAVAILABLE')
            return path
        except ResourceAdmissionError:
            raise
        except (ValueError, OSError, RuntimeError) as exc:
            raise _fail('LOCAL_RESOURCE_RECORD_INVALID') from exc

    def _path(self, resource_id, *, create=False):
        _identifier(resource_id)
        path = self._directory(create=create) / (resource_id + '.json')
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise _fail('LOCAL_RESOURCE_RECORD_INVALID')
        return path

    @staticmethod
    def fastq_member_declaration(values):
        return _serialize(_fastq_member(values))

    @staticmethod
    def _snapshot(value):
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
                value.st_ctime_ns, value.st_mode)

    def _source(self, source_path, *, approval=False, integrity=False, include_snapshot=False):
        code = 'LOCAL_RESOURCE_INTEGRITY_INVALID' if integrity else 'LOCAL_RESOURCE_ACCESS_INVALID'
        try:
            if not isinstance(source_path, (str, Path)) or '://' in str(source_path):
                raise ValueError('Invalid local source.')
            path = Path(source_path).expanduser().absolute()
            if path != path.resolve(strict=True):
                raise ValueError('Noncanonical source.')
            if approval and not any(path.is_relative_to(root) for root in self._approved):
                raise ValueError('Unapproved source.')
            before = path.stat(follow_symlinks=False)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError('Nonregular source.')
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            sha = hashlib.sha256()
            with os.fdopen(descriptor, 'rb') as stream:
                if self._snapshot(os.fstat(stream.fileno())) != self._snapshot(before):
                    raise _fail('LOCAL_RESOURCE_INTEGRITY_INVALID')
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    sha.update(chunk)
                if self._snapshot(os.fstat(stream.fileno())) != self._snapshot(before):
                    raise _fail('LOCAL_RESOURCE_INTEGRITY_INVALID')
            if (path != path.resolve(strict=True)
                    or self._snapshot(path.stat(follow_symlinks=False)) != self._snapshot(before)):
                raise _fail('LOCAL_RESOURCE_INTEGRITY_INVALID')
            identity = (str(path), sha.hexdigest(), before.st_size)
            return (*identity, self._snapshot(before)) if include_snapshot else identity
        except ResourceAdmissionError:
            raise
        except (ValueError, TypeError, OSError, RuntimeError) as exc:
            raise _fail(code) from exc

    @contextmanager
    def _lock(self, resource_id):
        root = self._directory(create=True)
        descriptor = None
        try:
            descriptor = os.open(root / (resource_id + '.lock'), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError('Invalid resource lock.')
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        except ResourceAdmissionError:
            raise
        except (ValueError, OSError) as exc:
            raise _fail('LOCAL_RESOURCE_RECORD_INVALID') from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)

    @staticmethod
    def registration_id(registration_key):
        """The existing deterministic identity, also usable before source publication."""
        _text(registration_key, 256, 'LOCAL_RESOURCE_RECORD_INVALID')
        return 'local-' + hashlib.sha256(b'agent.local-resource-id.v1\0' + registration_key.encode('utf-8')).hexdigest()

    def register(self, registration_key, source_path, *, input_type='h5ad', label, attribution,
                 expected_source_sha256=None, expected_size_bytes=None, source_index_path=None,
                 expected_source_index_sha256=None, expected_source_index_size_bytes=None,
                 fastq_member=None, reference_qualification=None):
        if input_type not in _INPUT_TYPES:
            raise _fail('LOCAL_RESOURCE_TYPE_UNSUPPORTED')
        resource_id = self.registration_id(registration_key)
        _text(label, 160, 'LOCAL_RESOURCE_RECORD_INVALID', display=True)
        _text(attribution, 1024, 'LOCAL_RESOURCE_RECORD_INVALID')
        expected = expected_source_sha256 is not None or expected_size_bytes is not None
        if expected and (type(expected_source_sha256) is not str or _SHA.fullmatch(expected_source_sha256) is None
                         or type(expected_size_bytes) is not int or expected_size_bytes < 0):
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID')
        source, sha, size, snapshot = self._source(source_path, approval=True, include_snapshot=True)
        if expected and (sha, size) != (expected_source_sha256, expected_size_bytes):
            raise _fail('LOCAL_RESOURCE_INTEGRITY_INVALID')
        index, index_snapshot = None, None
        expected_index = (expected_source_index_sha256 is not None
                          or expected_source_index_size_bytes is not None)
        if source_index_path is not None:
            if input_type not in ('external_fragments', 'fasta', 'genome_reference'):
                raise _fail('LOCAL_RESOURCE_BINDING_INVALID')
            if expected_index and (type(expected_source_index_sha256) is not str
                    or _SHA.fullmatch(expected_source_index_sha256) is None
                    or type(expected_source_index_size_bytes) is not int
                    or expected_source_index_size_bytes < 0):
                raise _fail('LOCAL_RESOURCE_BINDING_INVALID')
            index_path, index_sha, index_size, index_snapshot = self._source(
                source_index_path, approval=True, include_snapshot=True)
            if expected_index and (index_sha, index_size) != (
                    expected_source_index_sha256, expected_source_index_size_bytes):
                raise _fail('LOCAL_RESOURCE_INTEGRITY_INVALID')
            index = dict(path=index_path, sha256=index_sha, size_bytes=index_size)
        elif expected_index:
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID')
        record = LocalResourceRecord(resource_id, input_type, label, source, sha, size, attribution,
                                     index, fastq_member, reference_qualification)
        with self._lock(resource_id):
            path = self._path(resource_id)
            try:
                snapshots = [(source, snapshot)]
                if index is not None:
                    snapshots.append((index['path'], index_snapshot))
                if any(Path(path) != Path(path).resolve(strict=True)
                        or self._snapshot(Path(path).stat(follow_symlinks=False)) != captured
                        for path, captured in snapshots):
                    raise ValueError('Source or index changed before registration publication.')
            except (OSError, ValueError, RuntimeError) as exc:
                raise _fail('LOCAL_RESOURCE_INTEGRITY_INVALID') from exc
            if path.exists():
                if self.load(resource_id) != record:
                    raise _fail('LOCAL_RESOURCE_CONFLICT')
                return record
            self._publish(path, record)
        return record

    def _publish(self, path, record):
        payload = canonical(dict(format=_FORMAT, schema_version=1, record=record.to_dict(), sha256=record.record_sha256))
        maximum = MAX_COLLECTION_BYTES + 1024 if type(record) is LocalResourceCollectionRecord else MAX_RECORD_BYTES
        if len(payload) > maximum:
            raise _fail('LOCAL_RESOURCE_RECORD_INVALID')
        temporary = None
        try:
            descriptor, name = tempfile.mkstemp(dir=path.parent, prefix='.' + path.stem, suffix='.tmp')
            temporary = Path(name)
            with os.fdopen(descriptor, 'wb') as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, path)
            temporary.unlink()
            temporary = None
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except (OSError, ValueError) as exc:
            raise _fail('LOCAL_RESOURCE_RECORD_INVALID') from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def load(self, resource_id):
        path = self._path(resource_id)
        def pairs(items):
            value = {}
            for key, item in items:
                if key in value:
                    raise ValueError('Duplicate JSON key.')
                value[key] = item
            return value
        def constant(value):
            raise ValueError('Nonfinite JSON constant.')
        try:
            with path.open('rb') as stream:
                raw = stream.read(MAX_COLLECTION_BYTES + 1025)
            if len(raw) > MAX_COLLECTION_BYTES + 1024:
                raise ValueError('Resource bound exceeded.')
            value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
            if (type(value) is not dict or set(value) != {'format', 'schema_version', 'record', 'sha256'}
                    or value['format'] != _FORMAT or type(value['schema_version']) is not int
                    or value['schema_version'] != 1 or type(value['record']) is not dict
                    or value['sha256'] != digest(value['record'])):
                raise ValueError('Invalid resource envelope.')
            fields = set(value['record'])
            scalar_fields = set(LocalResourceRecord.__dataclass_fields__)
            if fields == set(LocalResourceCollectionRecord.__dataclass_fields__):
                record = LocalResourceCollectionRecord(**value['record'])
            elif (scalar_fields - {'source_index', 'fastq_member', 'reference_qualification'} <= fields <= scalar_fields
                  and len(raw) <= MAX_RECORD_BYTES):
                record = LocalResourceRecord(**value['record'])
            else:
                raise ValueError('Invalid resource record fields.')
            if record.resource_id != resource_id or record.to_dict() != value['record']:
                raise ValueError('Resource identity changed.')
            return record
        except FileNotFoundError as exc:
            raise _fail('LOCAL_RESOURCE_UNAVAILABLE') from exc
        except (ValueError, TypeError, OSError, KeyError, RecursionError) as exc:
            raise _fail('LOCAL_RESOURCE_RECORD_INVALID') from exc

    def list_records(self, *, source_root, limit=MAX_DISCOVERY_CHOICES, listing='all'):
        """Bounded durable discovery beneath an explicitly selected source root.

        Records establish registration only. Listing neither reads source bytes
        nor confers readability/readiness, and does not discover arbitrary files.
        The caller must project ``record.public()`` before crossing a client boundary.
        """
        if (type(limit) is not int or not 1 <= limit <= MAX_DISCOVERY_CHOICES
                or listing not in ('all', 'choices', 'fastq_members')):
            raise _fail('LOCAL_RESOURCE_DISCOVERY_LIMIT')
        try:
            selected = Path(source_root).absolute()
            if (selected != selected.resolve(strict=True) or not selected.is_dir()
                    or not any(selected.is_relative_to(root) for root in self._approved)):
                raise ValueError('Unapproved discovery root.')
            try:
                directory = self._directory()
            except ResourceAdmissionError as exc:
                if exc.code == 'LOCAL_RESOURCE_UNAVAILABLE':
                    return ()
                raise
            records, inspected = [], 0
            with os.scandir(directory) as entries:
                for entry in entries:
                    if not entry.name.endswith('.json'):
                        continue
                    inspected += 1
                    if inspected > MAX_DISCOVERY_RECORDS:
                        raise _fail('LOCAL_RESOURCE_DISCOVERY_LIMIT')
                    record = self.load(entry.name[:-5])
                    is_member = type(record) is LocalResourceRecord and record.input_type == 'fastq'
                    if listing == 'choices' and is_member or listing == 'fastq_members' and not is_member:
                        continue
                    sources = (self._collection_records(record.members)
                               if type(record) is LocalResourceCollectionRecord else (record,))
                    if all(Path(source.source_path).is_relative_to(selected) for source in sources):
                        records.append(record)
                        if len(records) > limit:
                            raise _fail('LOCAL_RESOURCE_DISCOVERY_LIMIT')
            return tuple(sorted(records, key=lambda record: record.resource_id))
        except ResourceAdmissionError:
            raise
        except (ValueError, TypeError, OSError, RuntimeError) as exc:
            raise _fail('LOCAL_RESOURCE_RECORD_INVALID') from exc

    def _binding_inputs(self, record, tool_name, scientific_inputs):
        if type(tool_name) is not str or tool_name not in _OPERATION_TYPES:
            raise _fail('LOCAL_RESOURCE_OPERATION_UNSUPPORTED')
        if type(record) is not LocalResourceRecord or record.input_type != _OPERATION_TYPES[tool_name]:
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID')
        if tool_name == 'inspect_scATAC':
            source = dict(input_path=record.source_path)
        elif tool_name == 'inspect_raw_scATAC':
            source = dict(raw_input_paths=[record.source_path])
        else:
            source = dict(source_path=record.source_path, source_sha256=record.source_sha256)
            if tool_name == 'import_scATAC_fragments' and record.source_index is not None:
                source.update(source_index_path=record.source_index['path'],
                              source_index_sha256=record.source_index['sha256'])
        return self._declared_inputs(tool_name, source, scientific_inputs)

    def _declared_inputs(self, tool_name, source, scientific_inputs):
        values = _input_mapping(scientific_inputs)
        if self._registry is None:
            from agent.orchestration.registry import build_default_tool_registry
            self._registry = build_default_tool_registry()
        try:
            specification = self._registry.get(tool_name)
            owned = {'path'} if tool_name == 'inspect_scATAC' else set(source)
            required = set(specification.required_arguments) - owned - {'output_dir'}
            allowed = (set(specification.required_arguments) | set(specification.optional_arguments)) - owned - {'output_dir'}
            if set(values) - allowed:
                raise ValueError('Unknown scientific input declaration.')
        except (ValueError, TypeError, LookupError) as exc:
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID') from exc
        if required - set(values):
            raise _fail('LOCAL_RESOURCE_DECLARATION_REQUIRED')
        return values | source

    def _collection_records(self, members):
        records = []
        for member in members:
            record = self.load(member['resource_id'])
            if (type(record) is not LocalResourceRecord or record.input_type != 'fastq'
                    or record.record_sha256 != member['record_sha256']):
                raise _fail('LOCAL_RESOURCE_BINDING_INVALID')
            records.append(record)
        if len({record.source_path for record in records}) != len(records):
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID')
        return records

    @staticmethod
    def _fastq_groups(records):
        """Check explicit role completeness against the existing scientific owner."""
        from agent.tools.data import _raw_fastq as owner
        if (not records or any(record.fastq_member is None for record in records)
                or len({(record.fastq_member['library_id'], record.fastq_member['fastq_layout'])
                        for record in records}) != 1):
            raise _fail('FASTQ_COLLECTION_INVALID')
        groups = {}
        layout = owner.FastqLayout(records[0].fastq_member['fastq_layout'])
        for record in records:
            declared = record.fastq_member
            key = (declared['lane'], declared['chunk'])
            names = groups.setdefault(key, [])
            role = owner.ReadRole(declared['role'])
            locator = owner._name(Path(record.source_path).name)
            if locator is None or (locator.lane, locator.chunk, locator.role) != (*key, role):
                raise _fail('FASTQ_COLLECTION_INVALID')
            if any(name.role is role for name in names):
                raise _fail('FASTQ_COLLECTION_INVALID')
            names.append(owner.FastqName('declared-library', '1', key[0], role, key[1]))
        syntactic_groups = {(str(Path(record.source_path).parent),
                             owner._name(Path(record.source_path).name).sample,
                             owner._name(Path(record.source_path).name).sample_number)
                            for record in records}
        if len(syntactic_groups) != 1:
            raise _fail('FASTQ_COLLECTION_INVALID')
        for (lane, chunk), names in sorted(groups.items()):
            actual, duplicate = owner._layout(names, owner.FastqAssay.TENX_ATAC, layout)
            if actual is layout and not duplicate:
                continue
            missing = owner._missing_roles(names, owner.FastqAssay.TENX_ATAC, layout)
            if missing:
                roles = ', '.join(f'{role.value} ({owner.LAYOUT_ROLES[layout][role].value})'
                                  for role in missing[0][1])
                error = _fail('FASTQ_COLLECTION_INCOMPLETE')
                error.message = f'Lane {lane}, chunk {chunk} is missing required FASTQ roles: {roles}.'
                error.error = ClientError(error.code, error.message)
                error.args = (error.message,)
                raise error
            raise _fail('FASTQ_COLLECTION_INVALID')
        return groups

    def register_fastq_collection(self, registration_key, member_ids, *, label, attribution):
        """Publish a complete typed library choice in the existing local catalog."""
        if (type(member_ids) not in (list, tuple)
                or not 1 <= len(member_ids) <= MAX_COLLECTION_MEMBERS
                or any(type(value) is not str for value in member_ids)
                or len(set(member_ids)) != len(member_ids)):
            raise _fail('FASTQ_COLLECTION_INVALID')
        records = [self.load(value) for value in member_ids]
        if any(type(record) is not LocalResourceRecord or record.input_type != 'fastq'
               for record in records):
            raise _fail('FASTQ_COLLECTION_INVALID')
        self._fastq_groups(records)
        members = _members([dict(resource_id=record.resource_id, record_sha256=record.record_sha256)
                            for record in records])
        record = LocalResourceCollectionRecord(self.registration_id(registration_key), label, members,
            _collection_sha256(members), attribution)
        with self._lock(record.resource_id):
            path = self._path(record.resource_id)
            self._verify_records(records)
            if path.exists():
                if self.load(record.resource_id) != record:
                    raise _fail('LOCAL_RESOURCE_CONFLICT')
                return record
            self._publish(path, record)
        return record

    def _collection_inputs(self, records, tool_name, scientific_inputs):
        if type(tool_name) is not str or tool_name not in _COLLECTION_OPERATIONS:
            raise _fail('LOCAL_RESOURCE_OPERATION_UNSUPPORTED')
        source = (dict(raw_input_paths=sorted(record.source_path for record in records))
                  if tool_name == 'inspect_raw_scATAC' else {})
        return self._declared_inputs(tool_name, source, scientific_inputs)

    @staticmethod
    def _validate_fastq_intake(records, values):
        from agent.tools.data.raw_scatac_manifest import (
            InputKind, SelectionBasis, load_raw_intake_manifest,
        )
        try:
            if values['intake_manifest_sha256'] is None:
                raise ValueError('Registered collection requires an intake digest.')
            _, intake, _ = load_raw_intake_manifest(values['intake_manifest_path'],
                expected_sha256=values['intake_manifest_sha256'])
            expected = {record.source_path: record.size_bytes for record in records}
            if len(intake.files) != len(expected) or {source.path for source in intake.files} != set(expected):
                raise ValueError('Intake membership differs from the registered collection.')
            for source in intake.files:
                if (source.input_kind is not InputKind.FASTQ or source.size_bytes != expected[source.path]
                        or source.selection is not SelectionBasis.EXPLICIT or source.selection_root is not None):
                    raise ValueError('Intake does not explicitly identify registered FASTQs.')
        except (ValueError, TypeError, OSError, KeyError) as exc:
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID') from exc

    @staticmethod
    def _validate_bam_intake(record, values):
        from agent.tools.data.raw_scatac_manifest import (
            InputKind, SelectionBasis, load_raw_intake_manifest,
        )
        try:
            # The loader permits unpinned reads; registered association does not.
            if values['intake_manifest_sha256'] is None:
                raise ValueError('Registered BAM requires an intake digest.')
            _, intake, _ = load_raw_intake_manifest(values['intake_manifest_path'],
                expected_sha256=values['intake_manifest_sha256'])
            if len(intake.files) != 1 or len(intake.groups) != 1:
                raise ValueError('Registered BAM requires a singleton intake.')
            source = intake.files[0]
            # The intake owner validates group membership. Require explicit file
            # selection so producer reinspection cannot rediscover other sources.
            if (source.input_kind is not InputKind.BAM or source.path != record.source_path
                    or source.size_bytes != record.size_bytes
                    or source.selection is not SelectionBasis.EXPLICIT or source.selection_root is not None):
                raise ValueError('Intake does not identify the registered BAM.')
        except (ValueError, TypeError, OSError, KeyError) as exc:
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID') from exc

    def resolve(self, resource_id, *, tool_name, scientific_inputs=None,
                resource_choice=None, verify_source=True):
        record = self.load(resource_id)
        inputs = self._binding_inputs(record, tool_name, scientific_inputs)
        binding = RegisteredInput(record.resource_id, record.record_sha256, tool_name, inputs,
                                  resource_choice=resource_choice)
        self.validate_binding(binding, verify_source=verify_source)
        return binding

    def _h5ad_inputs(self, record, scientific_inputs):
        """Compose request declarations; do not add arguments to inspection."""
        if record.input_type != 'h5ad':
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID')
        values = _input_mapping(scientific_inputs)
        if self._registry is None:
            from agent.orchestration.registry import build_default_tool_registry
            self._registry = build_default_tool_registry()
        declarations = {}
        for name in _H5AD_COMPANION_TOOLS:
            spec = self._registry.get(name)
            arguments = dict(spec.required_arguments) | dict(spec.optional_arguments)
            for key, argument in arguments.items():
                if key not in _H5AD_SOURCE_ARGUMENTS:
                    declarations.setdefault(key, []).append((key, argument))
            if spec.semantic_planning is not None:
                for port in spec.semantic_planning.consumer_ports:
                    fields = {member.name: member.field_name for member in port.members}
                    for source in port.request_sources:
                        for member in source.members:
                            key = fields[member.name]
                            if key not in _H5AD_SOURCE_ARGUMENTS:
                                declarations.setdefault(member.input_name, []).append((key, arguments[key]))
        adoption = self._registry.get('adopt_scATAC_cell_by_features')
        declarations['input_sha256'] = [('source_sha256', adoption.required_arguments['source_sha256'])]
        if set(values) - set(declarations):
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID')
        try:
            for key, value in values.items():
                for field, argument in declarations[key]:
                    if field == 'overwrite' and value is True:
                        raise _fail('LOCAL_RESOURCE_BINDING_INVALID')
                    argument.validate(key, value)
            if 'expected_resource_identity' in values:
                from agent.tools.analysis.epizoo_embedding import validate_expected_resource_identity
                values['expected_resource_identity'] = validate_expected_resource_identity(values['expected_resource_identity'])
                if 'checkpoint_path' not in values:
                    raise _fail('EPIZOO_RESOURCE_IDENTITY_INVALID')
        except ResourceAdmissionError:
            raise
        except (ValueError, TypeError) as exc:
            code = getattr(exc, 'code', 'LOCAL_RESOURCE_BINDING_INVALID')
            raise _fail(code if code in _MESSAGES else 'LOCAL_RESOURCE_BINDING_INVALID') from exc
        if 'input_sha256' in values and values['input_sha256'] != record.source_sha256:
            raise _fail('H5AD_SOURCE_MISMATCH')
        return values | dict(input_path=record.source_path)

    def compose_h5ad(self, resource_id, scientific_inputs=None, *,
                     resource_selection_error=None, resource_choice=None, verify_source=True):
        """One registered H5AD plus explicitly typed focused analysis companions.

        No operation is selected here. Required declarations/resources are checked
        against the actual compiled plan; an inspection needs neither.
        """
        record = self.load(resource_id)
        inputs = self._h5ad_inputs(record, scientific_inputs)
        if resource_selection_error is not None and 'expected_resource_identity' in inputs:
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID')
        binding = RegisteredInput(record.resource_id, record.record_sha256, 'inspect_scATAC', inputs,
                                  _H5AD_COMPOSITION, resource_selection_error, resource_choice)
        self.validate_binding(binding, verify_source=verify_source)
        return binding

    def _fragments_inputs(self, record, scientific_inputs):
        """Validate explicit companions against the existing focused Registry."""
        if record.input_type != 'external_fragments':
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID')
        values = _input_mapping(scientific_inputs)
        if self._registry is None:
            from agent.orchestration.registry import build_default_tool_registry
            self._registry = build_default_tool_registry()
        declarations = {}
        try:
            for name in _FRAGMENTS_COMPANION_TOOLS:
                specification = self._registry.get(name)
                arguments = dict(specification.required_arguments) | dict(specification.optional_arguments)
                for key, argument in arguments.items():
                    if key not in _FRAGMENTS_SOURCE_ARGUMENTS:
                        declarations.setdefault(key, []).append(argument)
                if specification.semantic_planning is not None:
                    for port in specification.semantic_planning.consumer_ports:
                        fields = {member.name: member.field_name for member in port.members}
                        for source in port.request_sources:
                            for member in source.members:
                                key = fields[member.name]
                                if key not in _FRAGMENTS_SOURCE_ARGUMENTS:
                                    declarations.setdefault(member.input_name, []).append(arguments[key])
            if set(values) - set(declarations):
                raise ValueError('Unsupported fragments companion declaration.')
            for key, value in values.items():
                for argument in declarations[key]:
                    argument.validate(key, value)
        except (ValueError, TypeError, LookupError) as exc:
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID') from exc
        source = dict(source_path=record.source_path, source_sha256=record.source_sha256)
        if record.source_index is not None:
            source.update(source_index_path=record.source_index['path'],
                          source_index_sha256=record.source_index['sha256'])
        return values | source

    def compose_fragments(self, resource_id, scientific_inputs=None, *,
                          resource_selection_error=None, resource_choice=None, verify_source=True):
        """One registered source with explicit import/QC/selection/matrix inputs.

        This mapping selects no scientific steps and fills no missing downstream
        requirements. The ordinary Planner and compiler still bind the plan.
        """
        record = self.load(resource_id)
        binding = RegisteredInput(record.resource_id, record.record_sha256, 'import_scATAC_fragments',
            self._fragments_inputs(record, scientific_inputs), _FRAGMENTS_COMPOSITION,
            resource_selection_error, resource_choice)
        self.validate_binding(binding, verify_source=verify_source)
        return binding

    def _validate_fragments_requirements(self, binding, values):
        """Require declared import context at execution and first acceptance only."""
        if binding.resource_selection_error is not None:
            raise _fail(binding.resource_selection_error)
        imported = self._group_registry().get('import_scATAC_fragments')
        required = set(imported.required_arguments) - _FRAGMENTS_SOURCE_ARGUMENTS
        if required - set(values):
            raise _fail('LOCAL_RESOURCE_DECLARATION_REQUIRED')

    def _bam_inputs(self, record, scientific_inputs):
        """Offer exact source identity and supplied declarations, without a workflow."""
        if record.input_type != 'bam':
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID')
        values = _input_mapping(scientific_inputs)
        if self._registry is None:
            from agent.orchestration.registry import build_default_tool_registry
            self._registry = build_default_tool_registry()
        declarations = {}
        try:
            for name in _BAM_COMPANION_TOOLS:
                specification = self._registry.get(name)
                arguments = dict(specification.required_arguments) | dict(specification.optional_arguments)
                for key, argument in arguments.items():
                    if key not in _BAM_SOURCE_ARGUMENTS:
                        declarations.setdefault(key, []).append(argument)
                if specification.semantic_planning is not None:
                    for port in specification.semantic_planning.consumer_ports:
                        fields = {member.name: member.field_name for member in port.members}
                        for source in port.request_sources:
                            for member in source.members:
                                key = fields[member.name]
                                if key not in _BAM_SOURCE_ARGUMENTS:
                                    declarations.setdefault(member.input_name, []).append(arguments[key])
            if set(values) - set(declarations):
                raise ValueError('Unsupported BAM companion declaration.')
            for key, value in values.items():
                for argument in declarations[key]:
                    argument.validate(key, value)
        except (ValueError, TypeError, LookupError) as exc:
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID') from exc
        # Inspection has no required biological declarations. Producer contracts
        # apply only when the ordinary plan actually requests preparation.
        return values | dict(raw_input_paths=[record.source_path],
                             source_path=record.source_path, source_sha256=record.source_sha256)

    def compose_bam(self, resource_id, scientific_inputs=None, *,
                    resource_selection_error=None, resource_choice=None, verify_source=True):
        """One BAM with explicit inspection/producer/downstream request inputs."""
        record = self.load(resource_id)
        binding = RegisteredInput(record.resource_id, record.record_sha256, 'inspect_raw_scATAC',
            self._bam_inputs(record, scientific_inputs), _BAM_COMPOSITION, resource_selection_error,
            resource_choice)
        self.validate_binding(binding, verify_source=verify_source)
        return binding

    def resolve_collection(self, members, *, tool_name, scientific_inputs=None, resource_choice=None):
        members = _members(members)
        records = self._collection_records(members)
        inputs = self._collection_inputs(records, tool_name, scientific_inputs)
        binding = RegisteredInputCollection(members, _collection_sha256(members), tool_name, inputs,
                                           resource_choice=resource_choice)
        self.validate_binding(binding)
        return binding

    def _fastq_inputs(self, records, scientific_inputs):
        """Compose existing contracts around explicit registered read attribution."""
        self._fastq_groups(records)
        values = _input_mapping(scientific_inputs)
        if self._registry is None:
            from agent.orchestration.registry import build_default_tool_registry
            self._registry = build_default_tool_registry()
        declarations = {}
        try:
            for name in _FASTQ_COMPANION_TOOLS:
                specification = self._registry.get(name)
                arguments = dict(specification.required_arguments) | dict(specification.optional_arguments)
                for key, argument in arguments.items():
                    if key not in _FASTQ_SOURCE_ARGUMENTS | {'raw_assay', 'fastq_layout'}:
                        declarations.setdefault(key, []).append(argument)
                if specification.semantic_planning is not None:
                    for port in specification.semantic_planning.consumer_ports:
                        fields = {member.name: member.field_name for member in port.members}
                        for source in port.request_sources:
                            for member in source.members:
                                key = fields[member.name]
                                if key not in _FASTQ_SOURCE_ARGUMENTS | {'raw_assay', 'fastq_layout'}:
                                    declarations.setdefault(member.input_name, []).append(arguments[key])
            if set(values) - set(declarations):
                raise ValueError('Unsupported FASTQ companion declaration.')
            for key, value in values.items():
                for argument in declarations[key]:
                    argument.validate(key, value)
        except (ValueError, TypeError, LookupError) as exc:
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID') from exc
        return values | dict(raw_input_paths=sorted(record.source_path for record in records),
            raw_assay='TENX_ATAC', fastq_layout=records[0].fastq_member['fastq_layout'])

    def compose_fastq_collection(self, resource_id, scientific_inputs=None, *,
                                resource_selection_error=None, resource_choice=None, verify_source=True):
        collection = self.load(resource_id)
        if type(collection) is not LocalResourceCollectionRecord:
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID')
        records = self._collection_records(collection.members)
        binding = RegisteredInputCollection(collection.members, collection.collection_sha256,
            'inspect_raw_scATAC', self._fastq_inputs(records, scientific_inputs),
            _FASTQ_COMPOSITION, resource_selection_error, resource_choice)
        self.validate_binding(binding, verify_source=verify_source)
        return binding

    def _validate_collection(self, binding, *, verify_source):
        records = self._collection_records(binding.members)
        values = _serialize(binding.execution_inputs)
        declarations = {key: value for key, value in values.items()
                        if binding.tool_name != 'inspect_raw_scATAC' or key != 'raw_input_paths'}
        if binding.composition == _FASTQ_COMPOSITION:
            declarations = {key: value for key, value in values.items()
                            if key not in {'raw_input_paths', 'raw_assay', 'fastq_layout'}}
        try:
            expected = (self._fastq_inputs(records, declarations)
                        if binding.composition == _FASTQ_COMPOSITION else
                        self._collection_inputs(records, binding.tool_name, declarations))
        except ResourceAdmissionError as exc:
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID') from exc
        if values != expected:
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID')
        if verify_source:
            if binding.tool_name == 'prepare_scATAC_fragments':
                self._validate_fastq_intake(records, values)
            # Hash and recheck all members after the intake loader finishes.
            self._verify_records(records)
        return values

    def _verify_records(self, records):
        # Recheck the whole collection's snapshots after hashing, so a change to
        # an earlier member while a later member is read cannot yield a binding.
        snapshots = []
        for record in records:
            identities = [(record.source_path, record.source_sha256, record.size_bytes)]
            if record.source_index is not None:
                identities.append((record.source_index['path'], record.source_index['sha256'],
                                   record.source_index['size_bytes']))
            for expected in identities:
                source, sha, size, snapshot = self._source(expected[0], integrity=True, include_snapshot=True)
                if (source, sha, size) != expected:
                    raise _fail('LOCAL_RESOURCE_INTEGRITY_INVALID')
                snapshots.append((Path(source), snapshot))
        try:
            if any(path != path.resolve(strict=True)
                    or self._snapshot(path.stat(follow_symlinks=False)) != snapshot
                    for path, snapshot in snapshots):
                raise ValueError('Collection changed during validation.')
        except (ValueError, OSError, RuntimeError) as exc:
            raise _fail('LOCAL_RESOURCE_INTEGRITY_INVALID') from exc

    def _group_registry(self):
        if self._registry is None:
            from agent.orchestration.registry import build_default_tool_registry
            self._registry = build_default_tool_registry()
        return self._registry

    def compose_group(self, resource_ids, scientific_inputs=None, *, resource_choice=None, verify_source=True):
        """Capture ordered existing records; selection alone proposes no roles."""
        if type(resource_ids) not in (tuple, list) or not 1 <= len(resource_ids) <= MAX_GROUP_MEMBERS:
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID')
        records = [self.load(resource_id) for resource_id in resource_ids]
        if any(type(record) is not LocalResourceRecord or record.input_type not in ('h5ad', 'groups') for record in records):
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID')
        members = [dict(resource_id=r.resource_id, record_sha256=r.record_sha256,
            source_path=r.source_path, source_sha256=r.source_sha256, input_type=r.input_type, label=r.label) for r in records]
        binding = RegisteredInputGroup(members, _input_mapping(scientific_inputs),
                                       resource_choice=resource_choice)
        self.validate_binding(binding, verify_source=verify_source)
        return binding

    def group_choices(self, binding):
        return group_choices(binding, self._group_registry())

    def bind_group(self, binding, roles, *, verify_source=False):
        self.validate_binding(binding, verify_source=verify_source)
        values = group_role_inputs(binding, roles, self._group_registry())
        result = RegisteredInputGroup(binding.members, values, roles, resource_choice=binding.resource_choice)
        self.validate_binding(result, verify_source=False)
        return result

    def validate_binding(self, binding, *, verify_source=True):
        if type(binding) not in (RegisteredInput, RegisteredInputCollection, RegisteredInputGroup) or type(verify_source) is not bool:
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID')
        if type(binding) is RegisteredInputGroup:
            records = [self.load(member['resource_id']) for member in binding.members]
            expected = [dict(resource_id=r.resource_id, record_sha256=r.record_sha256,
                source_path=r.source_path, input_type=r.input_type, label=r.label,
                **({'source_sha256': r.source_sha256} if 'source_sha256' in member else {}))
                for r, member in zip(records, binding.members, strict=True)]
            if expected != _serialize(binding.members):
                raise _fail('LOCAL_RESOURCE_BINDING_INVALID')
            values = group_role_inputs(binding, binding.roles, self._group_registry())
            if values != _serialize(binding.execution_inputs):
                raise _fail('LOCAL_RESOURCE_BINDING_INVALID')
            if verify_source:
                self._verify_records(records)
            return values
        if type(binding) is RegisteredInputCollection:
            return self._validate_collection(binding, verify_source=verify_source)
        record = self.load(binding.resource_id)
        if binding.record_sha256 != record.record_sha256:
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID')
        values = _serialize(binding.execution_inputs)
        if binding.resource_selection_error is not None and 'expected_resource_identity' in values:
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID')
        fields = (('input_path',) if binding.tool_name == 'inspect_scATAC' else
                  ('raw_input_paths',) if binding.tool_name == 'inspect_raw_scATAC' else
                  ('source_path', 'source_sha256'))
        if binding.composition == _BAM_COMPOSITION:
            fields = ('raw_input_paths', 'source_path', 'source_sha256')
        if record.source_index is not None:
            fields += ('source_index_path', 'source_index_sha256')
        declarations = {key: value for key, value in values.items() if key not in fields}
        try:
            expected = (self._h5ad_inputs(record, declarations) if binding.composition == _H5AD_COMPOSITION
                        else self._fragments_inputs(record, declarations)
                            if binding.composition == _FRAGMENTS_COMPOSITION
                        else self._bam_inputs(record, declarations)
                            if binding.composition == _BAM_COMPOSITION
                        else self._binding_inputs(record, binding.tool_name, declarations))
        except ResourceAdmissionError as exc:
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID') from exc
        if values != expected:
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID')
        if verify_source:
            self._verify_records([record])
        if verify_source and binding.tool_name == 'prepare_scATAC_bam_fragments':
            self._validate_bam_intake(record, values)
        return values

    @staticmethod
    def _submission_binding(submission):
        try:
            metadata = submission['registered_input']
            fields = set(metadata) - {'resource_choice'}
            scalar = {'resource_id', 'record_sha256', 'tool_name'}
            if fields == {'composition', 'members', 'roles'}:
                binding_type = RegisteredInputGroup
            elif (fields == scalar or
                    (scalar | {'composition'} <= fields
                     and fields <= scalar | {'composition', 'resource_selection_error'})):
                binding_type = RegisteredInput
            elif (fields == {'members', 'collection_sha256', 'tool_name'} or
                    ({'members', 'collection_sha256', 'tool_name', 'composition'} <= fields
                     and fields <= {'members', 'collection_sha256', 'tool_name',
                                            'composition', 'resource_selection_error'})):
                binding_type = RegisteredInputCollection
            else:
                raise ValueError('Invalid registered input attribution.')
            binding = binding_type(**metadata, execution_inputs=submission['execution_inputs'])
            if binding.attribution() != _serialize(metadata):
                raise ValueError('Noncanonical registered input attribution.')
            return binding
        except (TypeError, ValueError, KeyError) as exc:
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID') from exc

    def validate_submission(self, submission, *, verify_source=True):
        binding = self._submission_binding(submission)
        self.validate_binding(binding, verify_source=verify_source)

    @staticmethod
    def _bam_inspection_arguments(values, arguments):
        actual = _serialize(arguments.get('raw_input_paths'))
        paths = [actual] if type(actual) is str else actual
        if paths != [values['source_path']]:
            raise _fail('BAM_SOURCE_MISMATCH')
        for key in ('species', 'raw_assay', 'source_genome_assembly', 'fastq_layout'):
            if key in values and arguments.get(key) != values[key]:
                raise _fail('BAM_SOURCE_MISMATCH')

    @staticmethod
    def _bam_producer_arguments(binding, values, arguments):
        if binding.resource_selection_error is not None:
            raise _fail(binding.resource_selection_error)
        for keys, code in (
            (('library_context_path', 'library_context_sha256'), 'BAM_LIBRARY_CONTEXT_REQUIRED'),
            (('reference_bundle_path', 'reference_bundle_sha256'), 'BAM_REFERENCE_REQUIRED'),
            (('source_profile',), 'BAM_PROFILE_REQUIRED'),
        ):
            if any(key not in values for key in keys):
                raise _fail(code)
            if any(arguments.get(key) != values[key] for key in keys):
                raise _fail('BAM_SOURCE_MISMATCH')
        if any(arguments.get(key) != values[key] for key in ('source_path', 'source_sha256')):
            raise _fail('BAM_SOURCE_MISMATCH')

    def _validate_bam_plan(self, binding, record, values, plan):
        from agent.schemas.orchestration import StepOutputRef
        by_id = {step.step_id: step for step in plan.steps}
        consumed = False
        for step in plan.steps:
            if step.tool_name == 'inspect_raw_scATAC':
                self._bam_inspection_arguments(values, step.arguments)
                consumed = True
            elif step.tool_name == 'prepare_scATAC_bam_fragments':
                self._bam_producer_arguments(binding, values, step.arguments)
                path = step.arguments.get('intake_manifest_path')
                sha = step.arguments.get('intake_manifest_sha256')
                if isinstance(path, StepOutputRef) or isinstance(sha, StepOutputRef):
                    producer = by_id.get(path.step_id) if isinstance(path, StepOutputRef) else None
                    if (not isinstance(path, StepOutputRef) or not isinstance(sha, StepOutputRef)
                            or path.step_id != sha.step_id or path.output_key != 'manifest_path'
                            or sha.output_key != 'manifest_sha256' or producer is None
                            or producer.tool_name != 'inspect_raw_scATAC'
                            or path.step_id not in step.depends_on):
                        raise _fail('BAM_SOURCE_MISMATCH')
                    self._bam_inspection_arguments(values, producer.arguments)
                else:
                    if any(key not in values for key in ('intake_manifest_path', 'intake_manifest_sha256')):
                        raise _fail('BAM_INTAKE_REQUIRED')
                    if (path, sha) != (values['intake_manifest_path'], values['intake_manifest_sha256']):
                        raise _fail('BAM_SOURCE_MISMATCH')
                    self._validate_bam_intake(record, step.arguments)
                consumed = True
        if not consumed:
            raise _fail('BAM_SOURCE_MISMATCH')

    @staticmethod
    def _fastq_inspection_arguments(values, arguments):
        actual = _serialize(arguments.get('raw_input_paths'))
        if (type(actual) is not list or any(type(path) is not str for path in actual)
                or sorted(actual) != values['raw_input_paths']):
            raise _fail('FASTQ_SOURCE_MISMATCH')
        for key in ('species', 'raw_assay', 'source_genome_assembly', 'fastq_layout'):
            if key in values and arguments.get(key) != values[key]:
                raise _fail('FASTQ_SOURCE_MISMATCH')

    @staticmethod
    def _fastq_producer_arguments(binding, values, arguments):
        if binding.resource_selection_error is not None:
            raise _fail(binding.resource_selection_error)
        for keys, code in (
            (('library_context_path', 'library_context_sha256'), 'FASTQ_LIBRARY_CONTEXT_REQUIRED'),
            (('reference_bundle_path', 'reference_bundle_sha256'), 'FASTQ_REFERENCE_REQUIRED'),
        ):
            if any(key not in values for key in keys):
                raise _fail(code)
            if any(arguments.get(key) != values[key] for key in keys):
                raise _fail('FASTQ_SOURCE_MISMATCH')

    def _validate_fastq_plan(self, binding, values, plan):
        from agent.schemas.orchestration import StepOutputRef
        records = self._collection_records(binding.members)
        by_id = {step.step_id: step for step in plan.steps}
        consumed = False
        for step in plan.steps:
            if step.tool_name == 'inspect_raw_scATAC':
                self._fastq_inspection_arguments(values, step.arguments)
                consumed = True
            elif step.tool_name == 'prepare_scATAC_fragments':
                self._fastq_producer_arguments(binding, values, step.arguments)
                path = step.arguments.get('intake_manifest_path')
                sha = step.arguments.get('intake_manifest_sha256')
                if isinstance(path, StepOutputRef) or isinstance(sha, StepOutputRef):
                    producer = by_id.get(path.step_id) if isinstance(path, StepOutputRef) else None
                    if (not isinstance(path, StepOutputRef) or not isinstance(sha, StepOutputRef)
                            or path.step_id != sha.step_id or path.output_key != 'manifest_path'
                            or sha.output_key != 'manifest_sha256' or producer is None
                            or producer.tool_name != 'inspect_raw_scATAC'
                            or path.step_id not in step.depends_on):
                        raise _fail('FASTQ_SOURCE_MISMATCH')
                    self._fastq_inspection_arguments(values, producer.arguments)
                else:
                    if any(key not in values for key in ('intake_manifest_path', 'intake_manifest_sha256')):
                        raise _fail('FASTQ_INTAKE_REQUIRED')
                    if (path, sha) != (values['intake_manifest_path'], values['intake_manifest_sha256']):
                        raise _fail('FASTQ_SOURCE_MISMATCH')
                    self._validate_fastq_intake(records, step.arguments)
                consumed = True
        if not consumed:
            raise _fail('FASTQ_SOURCE_MISMATCH')

    def _validate_group_consumption(self, binding, steps, *, resolved=False):
        if not binding.roles:
            raise _fail('MULTI_INPUT_ROLE_INVALID')
        registry = self._group_registry()
        authorized = {}
        for role in binding.roles:
            port, _, _ = _group_port(registry, role['tool'], role['port'])
            path = binding.members[int(role['source'].split('.')[1])]['source_path']
            authorized[(role['tool'], role['port'])] = path
            consumers = [s for s in steps if s.tool_name == role['tool']]
            if len(consumers) != 1:
                raise _fail('MULTI_INPUT_SOURCE_MISMATCH')
            arguments = consumers[0].resolved_arguments if resolved else consumers[0].arguments
            if arguments.get(port.members[0].field_name) != path:
                raise _fail('MULTI_INPUT_SOURCE_MISMATCH')
            for member in port.members[1:]:
                if arguments.get(member.field_name) != binding.members[int(role['source'].split('.')[1])]['source_sha256']:
                    raise _fail('MULTI_INPUT_SOURCE_MISMATCH')
        selected_paths = {m['source_path'] for m in binding.members}
        for step in steps:
            spec = registry.get(step.tool_name)
            arguments = step.resolved_arguments if resolved else step.arguments
            for port in spec.semantic_planning.consumer_ports:
                for member in port.members:
                    actual = arguments.get(member.field_name)
                    if type(actual) is str and actual in selected_paths:
                        if authorized.get((step.tool_name, port.name)) != actual:
                            raise _fail('MULTI_INPUT_SOURCE_MISMATCH')
                    elif (not resolved and (step.tool_name, port.name) in _GROUP_PORTS and member == port.members[0]
                            and type(actual) is str):
                        # Unassigned dataset consumers must use explicit semantic
                        # dependencies, never an unregistered literal companion.
                        raise _fail('MULTI_INPUT_SOURCE_MISMATCH')

    def validate_group_consumer(self, submission, tool_name, arguments):
        """Check registration bytes at each actual new source consumption.

        This callback is operation-local execution admission, not scientific
        validation or durable authority. Historical presentation never calls it.
        """
        binding = self._submission_binding(submission)
        if type(binding) is not RegisteredInputGroup:
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID')
        self.validate_binding(binding, verify_source=False)
        records = []
        for role in binding.roles:
            if role['tool'] != tool_name:
                continue
            port, _, _ = _group_port(self._group_registry(), role['tool'], role['port'])
            member = binding.members[int(role['source'].split('.')[1])]
            if arguments.get(port.members[0].field_name) != member['source_path']:
                raise _fail('MULTI_INPUT_SOURCE_MISMATCH')
            if any(arguments.get(m.field_name) != member['source_sha256'] for m in port.members[1:]):
                raise _fail('MULTI_INPUT_SOURCE_MISMATCH')
            records.append(self.load(member['resource_id']))
        if records:
            self._verify_records(records)

    def _validate_resource_choice_consumption(self, binding, steps, *, resolved=False):
        """Check selected request-resource identities through existing Registry ports.

        This association check opens no resource and issues no scientific proof.
        Ports allowing upstream resources retain the compiler's exact dependency
        contract; request-only resource ports must consume their captured members.
        """
        if binding.resource_choice is None:
            return
        from agent.orchestration.registry import ArtifactSemanticKind
        kinds = {ArtifactSemanticKind.SCATAC_REFERENCE_BUNDLE,
                 ArtifactSemanticKind.SCATAC_QC_REFERENCE,
                 ArtifactSemanticKind.SCATAC_LIBRARY_CONTEXT}
        values = _serialize(binding.execution_inputs)
        registry = self._group_registry()
        for step in steps:
            specification = registry.get(step.tool_name)
            if specification.semantic_planning is None:
                continue
            declared = dict(specification.required_arguments) | dict(specification.optional_arguments)
            arguments = step.resolved_arguments if resolved else step.arguments
            for port in specification.semantic_planning.consumer_ports:
                if port.accepted_upstream_types or not any(
                    declared[member.field_name].planning is not None
                    and kinds.intersection(declared[member.field_name].planning.accepted_artifact_kinds)
                    for member in port.members
                ):
                    continue
                sources = [source for source in port.request_sources
                           if all(member.input_name in values for member in source.members)]
                if not sources:
                    continue
                if len(sources) != 1:
                    raise _fail('LOCAL_RESOURCE_CONSUMPTION_MISMATCH')
                fields = {member.name: member.field_name for member in port.members}
                if not port.required and all(arguments.get(field) is None for field in fields.values()):
                    continue
                if any(arguments.get(fields[member.name]) != values[member.input_name]
                       for member in sources[0].members):
                    raise _fail('LOCAL_RESOURCE_CONSUMPTION_MISMATCH')

    @staticmethod
    def _new_target_binding(step, steps, *, resolved=False):
        """Exact paired model output; this never chooses a model by species/order."""
        from agent.schemas import StepOutputRef
        args = step.resolved_arguments if resolved else step.arguments
        path, sha = (args.get('target_model_manifest_' + key) for key in ('path', 'sha256'))
        if resolved:
            from agent.schemas import StepStatus
            if not isinstance(path, str) or not isinstance(sha, str): return False
            matches = [s for s in steps if s.tool_name == 'adapt_epizoo_species'
                       and s.status == StepStatus.SUCCEEDED and s.result and s.verification and s.verification.passed
                       and s.result.get('manifest_path') == path and s.result.get('manifest_sha256') == sha]
            return len(matches) == 1
        return (isinstance(path, StepOutputRef) and isinstance(sha, StepOutputRef)
            and path.step_id == sha.step_id and path.output_key == 'manifest_path'
            and sha.output_key == 'manifest_sha256' and path.step_id in step.depends_on
            and any(s.step_id == path.step_id and s.tool_name == 'adapt_epizoo_species' for s in steps))

    def _validate_neutral_h5ad(self, binding, steps, *, resolved=False):
        if type(binding) is not RegisteredInput or binding.composition != _H5AD_COMPOSITION:
            return
        values = _serialize(binding.execution_inputs)
        for step in steps:
            if step.tool_name != 'adopt_scATAC_cell_by_features':
                continue
            arguments = step.resolved_arguments if resolved else step.arguments
            pairs = {'source_path': 'input_path', 'source_sha256': 'input_sha256',
                     'reference_manifest_path': 'reference_manifest_path',
                     'reference_manifest_sha256': 'reference_manifest_sha256',
                     'matrix_semantics': 'matrix_semantics'}
            if any(k not in values or arguments.get(a) != values[k] for a, k in pairs.items()):
                raise _fail('LOCAL_RESOURCE_CONSUMPTION_MISMATCH')

    def validate_plan(self, submission, plan):
        """Check exact registered-source/resource consumption before execution.

        The existing compiler owns semantic handoffs. This guard only checks
        association with the selected source, never completes a workflow.
        """
        from agent.schemas.orchestration import StepOutputRef
        binding = self._submission_binding(submission)
        values = self.validate_binding(binding)
        self._validate_resource_choice_consumption(binding, plan.steps)
        self._validate_neutral_h5ad(binding, plan.steps)
        if type(binding) is RegisteredInputGroup:
            self._validate_group_consumption(binding, plan.steps)
            return
        if type(binding) is RegisteredInputCollection and binding.composition == _FASTQ_COMPOSITION:
            self._validate_fastq_plan(binding, values, plan)
            return
        if type(binding) is RegisteredInput:
            record = self.load(binding.resource_id)
            if record.input_type == 'bam':
                source = values | dict(source_path=record.source_path, source_sha256=record.source_sha256)
                self._validate_bam_plan(binding, record, source, plan)
                return
        if type(binding) is RegisteredInput and binding.tool_name == 'import_scATAC_fragments':
            self._validate_fragments_requirements(binding, values)
            imports = [step for step in plan.steps if step.tool_name == binding.tool_name]
            if not imports:
                raise _fail('FRAGMENTS_SOURCE_MISMATCH')
            keys = ('source_path', 'source_sha256', 'reference_bundle_path',
                    'reference_bundle_sha256', 'source_profile', 'namespace',
                    'source_index_path', 'source_index_sha256')
            for step in imports:
                if any(step.arguments.get(key) != values.get(key) for key in keys):
                    raise _fail('FRAGMENTS_SOURCE_MISMATCH')
                if 'source_selection' in values and step.arguments.get('source_selection') != values['source_selection']:
                    raise _fail('FRAGMENTS_SOURCE_MISMATCH')
            return
        if type(binding) is not RegisteredInput or binding.tool_name != 'inspect_scATAC':
            return
        by_id = {step.step_id: step for step in plan.steps}
        source_path = values['input_path']
        channels = None
        for step in plan.steps:
            if step.tool_name not in ('inspect_scATAC', 'epizoo_embed_cells'):
                continue
            argument = 'path' if step.tool_name == 'inspect_scATAC' else 'input_path'
            source = step.arguments.get(argument)
            if isinstance(source, StepOutputRef):
                producer = by_id.get(source.step_id)
                if channels is None:
                    from agent.orchestration.semantic_compiler import build_m92_semantic_compiler_contract
                    channels = build_m92_semantic_compiler_contract(self._registry).step_output_channels
                permitted = (producer is not None and source.step_id in step.depends_on
                    and producer.tool_name == 'inspect_scATAC'
                    and producer.arguments.get('path') == source_path
                    and any(channel.producer_tool_name == producer.tool_name
                        and channel.consumer_tool_name == step.tool_name
                        and any(member.output_key == source.output_key and member.argument_name == argument
                                for member in channel.members) for channel in channels))
                if not permitted:
                    raise _fail('H5AD_SOURCE_MISMATCH')
            elif source != source_path:
                raise _fail('H5AD_SOURCE_MISMATCH')
            if step.tool_name == 'epizoo_embed_cells':
                if 'species' not in values:
                    raise _fail('H5AD_SPECIES_REQUIRED')
                new_target = ('target_model_manifest_path' not in values and 'expected_resource_identity' not in values
                              and self._new_target_binding(step, plan.steps))
                if binding.resource_selection_error is not None and not new_target:
                    raise _fail(binding.resource_selection_error)
                if 'expected_resource_identity' not in values and 'target_model_manifest_path' not in values and not new_target:
                    raise _fail('EPIZOO_RESOURCE_REQUIRED')
                keys = (('species', 'assembly', 'matrix_semantics') if new_target else
                        ('species', 'target_model_manifest_path', 'target_model_manifest_sha256',
                         'assembly', 'matrix_semantics', 'target_model_source_run_id',
                         'target_model_source_step_id') if 'target_model_manifest_path' in values
                        else ('species', 'checkpoint_path', 'expected_resource_identity'))
                for key in keys:
                    if step.arguments.get(key) != values.get(key):
                        raise _fail('EPIZOO_RESOURCE_CONSUMPTION_MISMATCH')
                if 'device' in values and step.arguments.get('device') != values['device']:
                    raise _fail('EPIZOO_RESOURCE_CONSUMPTION_MISMATCH')

    def validate_result(self, submission, steps):
        """First acceptance only: bind inspections and FASTQ provenance to sources.

        Scientific owners have already verified the result. This checks application
        identity, including selected producer subsets, without reconstructing science.
        Completed historical turns do not call this guard.
        """
        binding = self._submission_binding(submission)
        self._validate_resource_choice_consumption(binding, steps, resolved=True)
        self._validate_neutral_h5ad(binding, steps, resolved=True)
        if type(binding) is RegisteredInputGroup:
            # This API runs only at first Session acceptance. Recheck the pins
            # after owner verification/presentation too; terminal history reads
            # never call this guard and may outlive source availability.
            self.validate_binding(binding, verify_source=True)
            self._validate_group_consumption(binding, steps, resolved=True)
            return
        if type(binding) is RegisteredInput:
            record = self.load(binding.resource_id)
            if record.input_type == 'bam':
                values = _serialize(binding.execution_inputs) | dict(
                    source_path=record.source_path, source_sha256=record.source_sha256)
                inspections = False
                prepared = False
                consumed = False
                for step in steps:
                    if step.tool_name == 'inspect_raw_scATAC':
                        self._bam_inspection_arguments(values, step.resolved_arguments)
                        inspections = consumed = True
                    elif step.tool_name == 'prepare_scATAC_bam_fragments':
                        self._bam_producer_arguments(binding, values, step.resolved_arguments)
                        self._validate_bam_intake(record, step.resolved_arguments)
                        prepared = consumed = True
                if not consumed:
                    raise _fail('BAM_SOURCE_MISMATCH')
                # Raw-only inspection requires the registered full-byte pin at
                # first acceptance. Accepted production covers that exact source
                # completely, including an explicit upstream inspection; its
                # historical presentation can outlive current source availability.
                self.validate_binding(binding, verify_source=inspections and not prepared)
                return
        inspections = any(step.tool_name in {'inspect_scATAC', 'inspect_raw_scATAC'} for step in steps)
        h5ad_steps = ([step for step in steps if step.tool_name in {'inspect_scATAC', 'epizoo_embed_cells'}]
                     if type(binding) is RegisteredInput and binding.tool_name == 'inspect_scATAC' else [])
        producers = [step for step in steps if step.tool_name == 'prepare_scATAC_fragments']
        imports = ([step for step in steps if step.tool_name == 'import_scATAC_fragments']
                   if type(binding) is RegisteredInput and binding.tool_name == 'import_scATAC_fragments' else [])
        collection = type(binding) is RegisteredInputCollection
        if not inspections and not h5ad_steps and not imports and not (collection and producers):
            return
        if imports:
            values = _serialize(binding.execution_inputs)
            self._validate_fragments_requirements(binding, values)
            keys = ('source_path', 'source_sha256', 'reference_bundle_path',
                    'reference_bundle_sha256', 'source_profile', 'namespace',
                    'source_index_path', 'source_index_sha256')
            for step in imports:
                if any(step.resolved_arguments.get(key) != values.get(key) for key in keys):
                    raise _fail('FRAGMENTS_SOURCE_MISMATCH')
                if ('source_selection' in values
                        and step.resolved_arguments.get('source_selection') != values['source_selection']):
                    raise _fail('FRAGMENTS_SOURCE_MISMATCH')
            # Scientific owners have accepted these exact consumed arguments.
            # Completing presentation after a crash is a historical association
            # read; current source availability must not revoke accepted science.
            # New consumption already verifies both pins before plan execution.
            self.validate_binding(binding, verify_source=False)
            return
        if h5ad_steps:
            values = _serialize(binding.execution_inputs)
            for step in h5ad_steps:
                argument = 'path' if step.tool_name == 'inspect_scATAC' else 'input_path'
                if (step.resolved_arguments.get(argument) != values['input_path']
                        or step.result.get('input_path') != values['input_path']):
                    raise _fail('H5AD_SOURCE_MISMATCH')
                if step.tool_name == 'epizoo_embed_cells':
                    new_target = ('target_model_manifest_path' not in values and 'expected_resource_identity' not in values
                                  and self._new_target_binding(step, steps, resolved=True))
                    if binding.resource_selection_error is not None and not new_target:
                        raise _fail(binding.resource_selection_error)
                    if 'expected_resource_identity' not in values and 'target_model_manifest_path' not in values and not new_target:
                        raise _fail('EPIZOO_RESOURCE_REQUIRED')
                    keys = (('species', 'assembly', 'matrix_semantics') if new_target else
                            ('species', 'target_model_manifest_path', 'target_model_manifest_sha256',
                             'assembly', 'matrix_semantics', 'target_model_source_run_id',
                             'target_model_source_step_id') if 'target_model_manifest_path' in values
                            else ('species', 'checkpoint_path', 'expected_resource_identity'))
                    for key in keys:
                        if step.resolved_arguments.get(key) != values.get(key):
                            raise _fail('EPIZOO_RESOURCE_CONSUMPTION_MISMATCH')
        if collection:
            records = self._collection_records(binding.members)
            paths = sorted(record.source_path for record in records)
            values = _serialize(binding.execution_inputs)
            for step in steps:
                if step.tool_name == 'inspect_raw_scATAC':
                    if binding.composition == _FASTQ_COMPOSITION:
                        self._fastq_inspection_arguments(values, step.resolved_arguments)
                    actual = _serialize(step.resolved_arguments.get('raw_input_paths'))
                    if (type(actual) is not list or any(type(path) is not str for path in actual)
                            or sorted(actual) != paths):
                        raise _fail('LOCAL_RESOURCE_BINDING_INVALID')
        if collection and producers:
            from agent.tools.data.fastq_fragment_manifest import load_manifest
            expected = {record.source_path: dict(path=record.source_path,
                sha256=record.source_sha256, size_bytes=record.size_bytes) for record in records}
            try:
                for step in producers:
                    if binding.composition == _FASTQ_COMPOSITION:
                        self._fastq_producer_arguments(binding, values, step.resolved_arguments)
                    # Check the actual compiled producer inputs too; a Planner may
                    # select a newly inspected manifest through an intra-plan edge.
                    self._validate_fastq_intake(records, step.resolved_arguments)
                    manifest = load_manifest(step.result['manifest_path'],
                        expected_sha256=step.result['manifest_sha256'])
                    for library in manifest['libraries']:
                        provenance = library['provenance']
                        sources = [source['resource'] for source in provenance['sources'] if source['role'] == 'fastq']
                        if provenance['kind'] != 'fastq_fragment_production' or not sources:
                            raise ValueError('Missing qualified FASTQ source provenance.')
                        if any(expected.get(source['path']) != source for source in sources):
                            raise ValueError('Producer consumed different registered bytes.')
            except (ValueError, TypeError, OSError, KeyError) as exc:
                raise _fail('LOCAL_RESOURCE_BINDING_INVALID') from exc
        # An accepted producer has already verified exact source conservation.
        # This opt-in Web composition performs historical association reads;
        # existing raw evidence owners retain their own unfinished-presentation
        # source/mtime checks. Raw-only acceptance and legacy UA2.4 retain pins.
        historical_producer = (collection and producers and binding.composition == _FASTQ_COMPOSITION)
        self.validate_binding(binding, verify_source=not historical_producer)
