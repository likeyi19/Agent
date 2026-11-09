"""Bounded turn interpretation and exact, utterance-grounded scalar admission."""
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .raw_handoff import FastqDeclaration, BamDeclaration, FragmentsDeclaration, AnnotationDeclaration
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import json
import logging
import re
import unicodedata
from agent.schemas.orchestration import freeze_json_mapping, _serialize
from agent.orchestration.planning_diagnostics import log_operation_event

logger = logging.getLogger(__name__)

# Composed scientific requests can need more than eight explicit declarations.
MAX_SCIENTIFIC_ARGUMENTS = 32
PARAMETER_CLARIFICATION_REASONS = (
    'missing_parameter_value', 'ambiguous_parameter', 'invalid_parameter_value',
    'ungrounded_operand', 'conflicting_scientific_parameter')


class IntentError(ValueError):
    def __init__(self, reason, *, diagnostic_reason=None, operand=None, parameters=()):
        self.reason = reason
        self.diagnostic_reason = diagnostic_reason
        self.operand = operand
        self.parameters = parameters
        super().__init__(reason)


class DeclarationGroundingError(IntentError):
    """A structurally valid decision whose exact textual evidence is unresolved."""

    def __init__(self, reason, raw, parameters=()):
        super().__init__(reason, parameters=parameters)
        self.raw = raw


@dataclass(frozen=True)
class IntentDelta:
    parameter: str
    operation: str
    literal: str
    evidence: str


@dataclass(frozen=True)
class ScopedArgument:
    """An operation-qualified scalar declaration at an exact utterance span."""

    tool: str
    argument: str
    literal: str
    start: int
    end: int

    def __post_init__(self):
        if (type(self.tool) is not str or not 0 < len(self.tool) <= 128 or not self.tool.isidentifier()
                or type(self.argument) is not str or not 0 < len(self.argument) <= 128 or not self.argument.isidentifier()
                or type(self.literal) is not str or not 0 < len(self.literal) <= 4096
                or type(self.start) is not int or type(self.end) is not int
                or not 0 <= self.start < self.end <= 4096):
            raise ValueError('Invalid scoped scientific argument declaration.')


@dataclass(frozen=True)
class LeidenResolution(ScopedArgument):
    """The strict historical wire for the original clustering declaration."""

    def __post_init__(self):
        super().__post_init__()
        if self.tool != 'cluster_cells' or self.argument != 'resolution' or len(self.literal) > 80:
            raise ValueError('Invalid explicit Leiden resolution declaration.')


@dataclass(frozen=True)
class ScalarChoiceArgument(ScopedArgument):
    """An LLM-normalized reviewed choice with its verbatim declaration span."""

    value: str

    def __post_init__(self):
        super().__post_init__()
        if (not self.literal.strip() or type(self.value) is not str
                or not self.value.strip() or len(self.value) > 80):
            raise ValueError('Invalid scalar choice declaration.')


@dataclass(frozen=True)
class SourceFieldArgument(ScalarChoiceArgument):
    """An explicit field and source selection for a Registry-owned dataset port."""

    source: str
    mode: str

    def __post_init__(self):
        super().__post_init__()
        if (type(self.source) is not str or re.fullmatch(r'source\.(?:0|[1-9][0-9]*)', self.source) is None
                or self.mode not in ('offered', 'literal')):
            raise ValueError('Invalid source-qualified observation field.')


@dataclass(frozen=True)
class SourceScopedArgument(ScopedArgument):
    """A scalar for the operation consuming one explicit Registry dataset port."""

    port: str
    source: str

    def __post_init__(self):
        super().__post_init__()
        InputRole(self.tool, self.port, self.source)


@dataclass(frozen=True)
class SourceScopedChoiceArgument(ScalarChoiceArgument):
    """The same source scope for a reviewed categorical scalar."""

    port: str
    source: str

    def __post_init__(self):
        super().__post_init__()
        InputRole(self.tool, self.port, self.source)


@dataclass(frozen=True)
class ArgumentMember:
    """One reviewed object member, grounded in the same user utterance."""

    name: str
    kind: str
    literal: str
    start: int
    end: int

    def __post_init__(self):
        ScopedArgument('member', 'value', self.literal, self.start, self.end)
        if (type(self.name) is not str or not 0 < len(self.name) <= 128
                or self.kind not in ('string', 'number', 'json')):
            raise ValueError('Invalid structured argument member.')


@dataclass(frozen=True)
class StructuredArgument(ScopedArgument):
    """An ordinary scientific declaration with individually grounded members."""

    members: tuple[ArgumentMember, ...]

    def __post_init__(self):
        super().__post_init__()
        if (type(self.members) is not tuple or not 1 <= len(self.members) <= 32
                or any(not isinstance(m, ArgumentMember) or not self.start <= m.start < m.end <= self.end
                       for m in self.members)
                or len({m.name for m in self.members}) != len(self.members)):
            raise ValueError('Invalid or conflicting structured argument members.')


@dataclass(frozen=True)
class SourceScopedStructuredArgument(StructuredArgument, SourceScopedArgument):
    """Structured values keep the existing exact dataset source scope."""


@dataclass(frozen=True, kw_only=True)
class NativeFunctionScope:
    native_function: str

    def __post_init__(self):
        super().__post_init__()
        if type(self.native_function) is not str or not 0 < len(self.native_function) <= 80:
            raise ValueError('Invalid native function scope')


@dataclass(frozen=True)
class NativeArgument(NativeFunctionScope, ScopedArgument):
    pass


@dataclass(frozen=True)
class NativeChoiceArgument(NativeFunctionScope, ScalarChoiceArgument):
    pass


@dataclass(frozen=True)
class NativeStructuredArgument(NativeFunctionScope, StructuredArgument):
    pass


@dataclass(frozen=True)
class NativeSourceArgument(NativeFunctionScope, SourceScopedArgument):
    pass


@dataclass(frozen=True)
class NativeSourceChoiceArgument(NativeFunctionScope, SourceScopedChoiceArgument):
    pass


@dataclass(frozen=True)
class NativeSourceStructuredArgument(NativeFunctionScope, SourceScopedStructuredArgument):
    pass


def argument_identity(item):
    return (item.tool, item.argument, getattr(item, 'port', None), getattr(item, 'source', None), getattr(item, 'native_function', None))


def _validate_arguments(arguments, argument=None):
    if (type(arguments) is not tuple or len(arguments) > MAX_SCIENTIFIC_ARGUMENTS
            or any(not isinstance(item, ScopedArgument) for item in arguments)):
        raise ValueError('Invalid bounded scientific arguments.')
    identities = [argument_identity(item) for item in arguments]
    if argument is not None:
        identities.append(argument_identity(argument))
    if len(identities) > MAX_SCIENTIFIC_ARGUMENTS:
        raise ValueError('Scientific argument declaration limit exceeded.')
    if len(identities) != len(set(identities)):
        raise ValueError('Duplicate scoped scientific argument.')


def _normalize_arguments(decision):
    """Read the historical singular declaration; all new writers use arguments."""
    if decision.argument is not None:
        if not isinstance(decision.argument, LeidenResolution):
            raise ValueError('Invalid historical scientific argument.')
        _validate_arguments(decision.arguments, decision.argument)
        object.__setattr__(decision, 'arguments', (ScopedArgument(**asdict(decision.argument)),) + decision.arguments)
        object.__setattr__(decision, 'argument', None)


@dataclass(frozen=True)
class RetainedParameter:
    result: str
    argument: str

    def __post_init__(self):
        if (type(self.result) is not str or re.fullmatch(r'g[0-9]+', self.result) is None
                or type(self.argument) is not str or not self.argument.isidentifier() or len(self.argument) > 128):
            raise ValueError('Invalid accepted parameter reference.')


@dataclass(frozen=True)
class InputRole:
    """One Registry consumer associated with an opaque captured source."""

    tool: str
    port: str
    source: str

    def __post_init__(self):
        if (type(self.tool) is not str or not 0 < len(self.tool) <= 128 or not self.tool.isidentifier()
                or type(self.port) is not str or not 0 < len(self.port) <= 128 or not self.port.isidentifier()
                or type(self.source) is not str or re.fullmatch(r'source\.(?:[0-9]|1[0-5])', self.source) is None):
            raise ValueError('Invalid captured scientific input role.')


def _validate_roles(roles):
    if (type(roles) is not tuple or len(roles) > 8
            or any(not isinstance(item, InputRole) for item in roles)
            or len({(item.tool, item.port, item.source) for item in roles}) != len(roles)):
        raise ValueError('Invalid bounded scientific input roles.')


@dataclass(frozen=True)
class RoleRequirement:
    """One offered Registry consumer awaiting an exact source association."""

    tool: str
    port: str

    def __post_init__(self):
        if (type(self.tool) is not str or not 0 < len(self.tool) <= 128 or not self.tool.isidentifier()
                or type(self.port) is not str or not 0 < len(self.port) <= 128 or not self.port.isidentifier()):
            raise ValueError('Invalid captured scientific role requirement.')


def _validate_role_requirements(requirements):
    if (type(requirements) is not tuple or not 1 <= len(requirements) <= 8
            or any(not isinstance(item, RoleRequirement) for item in requirements)
            or len({(item.tool, item.port) for item in requirements}) != len(requirements)):
        raise ValueError('Invalid bounded scientific role requirements.')


def _validate_visualization(value, *, planned=True):
    _shape(_serialize(value), ('coordinates', 'labels', 'field'))
    coordinates, labels, field = (value[k] for k in ('coordinates', 'labels', 'field'))
    accepted = lambda v: type(v) is str and re.fullmatch(r'g[0-9]+', v) is not None
    source = type(labels) is str and re.fullmatch(r'source\.(?:[0-9]|1[0-5])', labels) is not None
    if (not (accepted(coordinates) or planned and coordinates == 'new')
            or not (accepted(labels) or source or planned and labels == 'new')
            or type(field) is not str or not 0 < len(field) <= 256
            or not field.startswith(('source:', 'result:')) or not field.split(':', 1)[1]
            or source and not field.startswith('source:')):
        raise ValueError('Invalid explicit cell visualization operands.')


def _validate_comparison(value):
    value = _serialize(value)
    _shape(value, ('left', 'right', 'detail'))
    if any(type(value[k]) is not str or re.fullmatch(r'g[0-9]+|new', value[k]) is None
           for k in ('left', 'right')):
        raise ValueError('Invalid exact comparison operands.')
    detail = value['detail']
    if detail is not None:
        validate_scientific_detail(detail, value['right'])
        if (detail['section'] not in ('category_counts', 'category_association')
                or any(type(detail[k]) is not str or not detail[k].startswith(('source:', 'result:'))
                       or not detail[k].split(':', 1)[1] for k in ('field', 'other_field'))):
            raise ValueError('Select both exact comparison fields.')
    if value['left'] == value['right'] and (detail is None or detail['field'] == detail['other_field']):
        raise ValueError('A same-result comparison requires two distinct exact fields.')


def _validate_export(value):
    """One bounded selection of accepted fields, never an export program."""
    value = _serialize(value)
    _shape(value, ('measurement', 'observations', 'representations'))
    accepted = lambda item: type(item) is str and re.fullmatch(r'g[0-9]+', item) is not None
    measurement = value['measurement']
    if not (accepted(measurement) or type(measurement) is str
            and re.fullmatch(r'source\.(?:[0-9]|1[0-5])', measurement) is not None):
        raise ValueError('Select an exact accepted measurement source.')
    total = 0
    for slot, prefixes in (('observations', ('result:', 'source:')), ('representations', ('obsm:',))):
        rows = value[slot]
        if type(rows) is not list or len(rows) > 32:
            raise ValueError('Invalid bounded accepted export fields.')
        total += len(rows)
        destinations = set()
        for row in rows:
            _shape(row, ('output', 'field', 'destination'))
            field, destination = row['field'], row['destination']
            if (not accepted(row['output']) or type(field) is not str or not 0 < len(field) <= 256
                    or not field.startswith(prefixes) or not field.split(':', 1)[1]
                    or type(destination) is not str or not destination.strip() or len(destination) > 256
                    or destination in ('.', '..', '_index', '__categories')
                    or '/' in destination or '\\' in destination
                    or any(ord(c) < 32 or ord(c) == 127 for c in destination)
                    or destination in destinations):
                raise ValueError('Select exact export fields and distinct safe destination names.')
            destinations.add(destination)
    if not 1 <= total <= 32:
        raise ValueError('Select between one and 32 accepted export fields.')


def _validate_input_results(values):
    if (type(values) is not tuple or len(values) > 32
            or any(type(h) is not str or re.fullmatch(r'g[0-9]+', h) is None for h in values)
            or len(set(values)) != len(values)):
        raise ValueError('Invalid accepted scientific input selection.')


def _normalize_research_intent(decision):
    """The same bounded downstream intent survives execution prerequisites."""
    if (type(decision.references) is not tuple or len(decision.references) > 32
            or any(type(h) is not str or re.fullmatch(r'g[0-9]+', h) is None for h in decision.references)
            or len(set(decision.references)) != len(decision.references)
            or type(decision.delivery) is not tuple or len(set(decision.delivery)) != len(decision.delivery)
            or any(k not in ('explanation', 'comparison', 'visualization') for k in decision.delivery)
            or type(decision.unfulfilled) is not tuple or len(decision.unfulfilled) > 8
            or any(type(t) is not str or not t.strip() or len(t) > 512 for t in decision.unfulfilled)
            or len(set(decision.unfulfilled)) != len(decision.unfulfilled)
            or type(decision.retain_parameters) is not tuple or len(decision.retain_parameters) > 8
            or any(not isinstance(p, RetainedParameter) for p in decision.retain_parameters)
            or len(set(decision.retain_parameters)) != len(decision.retain_parameters)):
        raise ValueError('Invalid bounded research intention.')
    _validate_input_results(decision.input_results)
    requested = list(decision.delivery)
    if decision.comparison is not None:
        _validate_comparison(decision.comparison)
        object.__setattr__(decision, 'comparison', freeze_json_mapping(decision.comparison, 'comparison'))
        if 'comparison' not in requested:
            requested.append('comparison')
    if decision.visualization is not None:
        _validate_visualization(decision.visualization)
        object.__setattr__(decision, 'visualization', freeze_json_mapping(decision.visualization, 'visualization'))
        if 'visualization' not in requested:
            requested.append('visualization')
    object.__setattr__(decision, 'delivery', tuple(requested))


@dataclass(frozen=True)
class RequestRoles:
    """Capture an identified scientific objective without an executable plan."""

    target: str
    required_roles: tuple[RoleRequirement, ...]
    roles: tuple[InputRole, ...] = ()
    argument: LeidenResolution | None = None
    arguments: tuple[ScopedArgument, ...] = ()
    context: str = 'none'
    fastq: 'FastqDeclaration | None' = None
    bam: 'BamDeclaration | None' = None
    fragments: 'FragmentsDeclaration | None' = None
    annotation: 'AnnotationDeclaration | None' = None
    epizoo: 'EpiZooDeclaration | None' = None
    input_results: tuple[str, ...] = field(default=(), kw_only=True)
    references: tuple[str, ...] = ()
    delivery: tuple[str, ...] = ()
    unfulfilled: tuple[str, ...] = ()
    visualization: object = None
    comparison: object = None
    retain_parameters: tuple[RetainedParameter, ...] = ()

    def __post_init__(self):
        _normalize_arguments(self)
        _normalize_research_intent(self)
        if self.epizoo is not None:
            from .epizoo_handoff import EpiZooDeclaration, target
            from .raw_handoff import validate_arguments
            if (type(self.epizoo) is not EpiZooDeclaration or self.epizoo.operation is None
                    or any(v is not None for v in (self.fastq, self.bam, self.fragments, self.annotation))
                    or self.target != target(self.epizoo) or self.required_roles or self.roles
                    or self.context != 'none' or self.argument is not None):
                raise ValueError('Invalid EpiZoo task declaration.')
            _validate_arguments(self.arguments)
            validate_arguments(self.arguments, 'epizoo')
            return
        raw_annotation = self.annotation is not None and any(v is not None for v in (self.fastq, self.bam, self.fragments))
        if raw_annotation:
            from .raw_handoff import validate_downstream_annotation
            validate_downstream_annotation(self.annotation)
        if self.annotation is not None and not raw_annotation:
            from .raw_handoff import AnnotationDeclaration, validate_arguments
            if (type(self.annotation) is not AnnotationDeclaration or any(v is not None for v in (self.fastq, self.bam, self.fragments))
                    or self.target != 'annotate_scATAC_cell_types' or self.required_roles or self.roles
                    or self.context not in ('none', 'current') or self.argument is not None):
                raise ValueError('Invalid annotation task declaration.')
            _validate_arguments(self.arguments)
            validate_arguments(self.arguments, 'annotation')
            return
        if self.fragments is not None:
            from .raw_handoff import FragmentsDeclaration, validate_arguments
            if (type(self.fragments) is not FragmentsDeclaration or self.fastq is not None or self.bam is not None
                    or self.target != 'run_snapatac2' or self.required_roles or self.roles
                    or self.context != 'none' or self.argument is not None):
                raise ValueError('Invalid fragments task declaration.')
            _validate_arguments(self.arguments)
            validate_arguments(self.arguments, 'fragments')
            return
        if self.bam is not None:
            from .raw_handoff import BamDeclaration, validate_arguments
            if type(self.bam) is not BamDeclaration or self.fastq is not None or self.roles:
                raise ValueError('Invalid BAM task declaration.')
            validate_arguments(self.arguments, 'bam')
        if self.fastq is not None:
            from .raw_handoff import FastqDeclaration
            if (type(self.fastq) is not FastqDeclaration or self.target != 'run_chromap'
                    or self.required_roles or self.roles or self.context != 'none' or self.argument is not None):
                raise ValueError('Invalid FASTQ task declaration.')
            _validate_arguments(self.arguments)
            from .raw_handoff import validate_arguments
            validate_arguments(self.arguments)
            return
        if self.bam is not None:
            if self.target != 'run_snapatac2' or self.required_roles or self.context != 'none' or self.argument is not None:
                raise ValueError('Invalid BAM task request.')
            _validate_arguments(self.arguments)
            return
        _validate_role_requirements(self.required_roles)
        _validate_roles(self.roles)
        _validate_arguments(self.arguments)
        if (type(self.target) is not str or self.target not in {r.tool for r in self.required_roles}
                or self.context not in ('none', 'current')
                or not {(r.tool, r.port) for r in self.roles} < {
                    (r.tool, r.port) for r in self.required_roles}):
            raise ValueError('Invalid missing scientific role request.')


@dataclass(frozen=True)
class RoleAnswer:
    """A bounded association or exact declaration for one pending objective."""

    pending: str
    roles: tuple[InputRole, ...]
    arguments: tuple[ScopedArgument, ...] = ()
    fastq: 'FastqDeclaration | None' = None
    bam: 'BamDeclaration | None' = None
    fragments: 'FragmentsDeclaration | None' = None
    annotation: 'AnnotationDeclaration | None' = None
    epizoo: 'EpiZooDeclaration | None' = None
    unknown_fields: tuple[str, ...] = ()

    def __post_init__(self):
        if (type(self.pending) is not str or not 0 < len(self.pending) <= 128
                or not (self.roles or self.arguments or self.fastq is not None or self.bam is not None or self.fragments is not None or self.annotation is not None or self.epizoo is not None)):
            raise ValueError('Invalid pending scientific role answer.')
        from .raw_handoff import scientific_metadata_fields
        declarations = [v for v in (self.fastq, self.bam, self.fragments, self.annotation, self.epizoo)
                        if v is not None]
        if (type(self.unknown_fields) is not tuple or len(self.unknown_fields) > 16
                or any(type(name) is not str for name in self.unknown_fields)
                or len(set(self.unknown_fields)) != len(self.unknown_fields)
                or self.unknown_fields and (len(declarations) != 1 or any(
                    name not in scientific_metadata_fields(declarations[0])
                    or getattr(declarations[0], name) is not None for name in self.unknown_fields))):
            raise ValueError('Unknown fields require one pending scientific metadata declaration.')
        if self.epizoo is not None:
            from .epizoo_handoff import EpiZooDeclaration
            from .raw_handoff import validate_arguments
            if (type(self.epizoo) is not EpiZooDeclaration or self.roles
                    or any(v is not None for v in (self.fastq, self.bam, self.fragments, self.annotation))):
                raise ValueError('Invalid EpiZoo task answer.')
            validate_arguments(self.arguments, 'epizoo')
        if self.annotation is not None:
            from .raw_handoff import AnnotationDeclaration, validate_arguments, validate_downstream_annotation
            if type(self.annotation) is not AnnotationDeclaration or self.roles:
                raise ValueError('Invalid annotation task answer.')
            if any(v is not None for v in (self.fastq, self.bam, self.fragments)):
                validate_downstream_annotation(self.annotation)
            else:
                validate_arguments(self.arguments, 'annotation')
        if self.fragments is not None:
            from .raw_handoff import FragmentsDeclaration, validate_arguments
            if type(self.fragments) is not FragmentsDeclaration or self.fastq is not None or self.bam is not None or self.roles:
                raise ValueError('Invalid fragments task answer.')
            validate_arguments(self.arguments, 'fragments')
        if self.bam is not None:
            from .raw_handoff import BamDeclaration, validate_arguments
            if type(self.bam) is not BamDeclaration or self.fastq is not None or self.roles:
                raise ValueError('Invalid BAM task declaration.')
            validate_arguments(self.arguments, 'bam')
        if self.fastq is not None:
            from .raw_handoff import FastqDeclaration
            if type(self.fastq) is not FastqDeclaration or self.roles:
                raise ValueError('Invalid FASTQ member answer.')
            from .raw_handoff import validate_arguments
            validate_arguments(self.arguments)
        _validate_roles(self.roles)
        _validate_arguments(self.arguments)


@dataclass(frozen=True)
class Execute:
    base: str
    operation: str
    target: str
    delta: IntentDelta | None
    argument: LeidenResolution | None = None
    arguments: tuple[ScopedArgument, ...] = ()
    context: str = 'none'
    roles: tuple[InputRole, ...] = ()
    input_results: tuple[str, ...] = field(default=(), kw_only=True)
    references: tuple[str, ...] = ()
    delivery: tuple[str, ...] = ()
    retain_parameters: tuple[RetainedParameter, ...] = ()
    unfulfilled: tuple[str, ...] = ()
    visualization: object = None
    comparison: object = None
    missing_parameters: tuple[str, ...] = ()

    def __post_init__(self):
        _normalize_arguments(self)
        _normalize_research_intent(self)
        if (len(self.missing_parameters)>8 or len(set(self.missing_parameters))!=len(self.missing_parameters)
                or any(not isinstance(k,str) or not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*',k) for k in self.missing_parameters)
                or self.missing_parameters and self.operation!='plan'):
            raise ValueError('Invalid missing scientific parameter identities.')
        if self.context not in ('none', 'current') or (self.operation != 'plan' and self.context != 'none'):
            raise ValueError('Invalid bounded execution context.')
        _validate_arguments(self.arguments)
        _validate_roles(self.roles)
        if self.roles and self.operation != 'plan':
            raise ValueError('Scientific input roles require ordinary planning.')
        if self.operation != 'plan' and (self.input_results or self.references or self.delivery or self.retain_parameters or self.unfulfilled):
            raise ValueError('Invalid bounded research intention.')


@dataclass(frozen=True)
class ExecuteCandidate:
    candidate: str
    evidence: str

    def __post_init__(self):
        if (type(self.candidate) is not str or not 0 < len(self.candidate) <= 128
                or type(self.evidence) is not str or not 0 < len(self.evidence) <= 4096):
            raise ValueError('Invalid explicit candidate selection.')


@dataclass(frozen=True)
class SpeciesAnswer:
    """A declaration answering one offered, exact pending clarification."""

    pending: str
    species: str
    argument: LeidenResolution | None = None
    arguments: tuple[ScopedArgument, ...] = ()

    def __post_init__(self):
        _normalize_arguments(self)
        if (type(self.pending) is not str or not 0 < len(self.pending) <= 128
                or type(self.species) is not str or self.species not in ('human', 'mouse')):
            raise ValueError('Invalid pending species declaration.')
        _validate_arguments(self.arguments)
        if any(isinstance(item, ScalarChoiceArgument) for item in self.arguments):
            raise ValueError('Species answers use the existing species field.')


@dataclass(frozen=True)
class ParameterAnswer:
    """Exact scalar declarations answering one pending scientific request."""

    pending: str
    arguments: tuple[ScopedArgument, ...]

    def __post_init__(self):
        if type(self.pending) is not str or not 0 < len(self.pending) <= 128 or not self.arguments:
            raise ValueError('Invalid pending scientific parameter answer.')
        _validate_arguments(self.arguments)


@dataclass(frozen=True)
class RequestResources:
    target: str
    arguments: tuple[ScopedArgument, ...] = ()
    roles: tuple[InputRole, ...] = ()
    reference_category: str | None = None
    context: str = 'none'
    input_results: tuple[str, ...] = field(default=(), kw_only=True)

    def __post_init__(self):
        if type(self.target) is not str or not self.target:
            raise ValueError('Invalid resource request target.')
        if self.reference_category not in (None, 'peak_set', 'regulatory_regions'):
            raise ValueError('Invalid reference category.')
        if self.context not in ('none', 'current'):
            raise ValueError('Invalid resource request context.')
        _validate_input_results(self.input_results)
        _validate_arguments(self.arguments)


@dataclass(frozen=True)
class ResourceAnswer:
    pending: str
    choice: str
    arguments: tuple[ScopedArgument, ...] = ()

    def __post_init__(self):
        if self.pending != '@resources' or type(self.choice) is not str or not self.choice.startswith('resource.'):
            raise ValueError('Invalid resource choice answer.')
        _validate_arguments(self.arguments)


@dataclass(frozen=True)
class Navigate:
    relation: str


@dataclass(frozen=True)
class Clarify:
    reason: str
    choices: tuple[str, ...] = ()
    value_required: bool = False


ANSWER_INTENTS = ('version', 'matrix', 'selection', 'thresholds', 'changes',
                  'execution', 'reuse', 'comparison', 'provenance', 'verification', 'unsupported', 'scientific', 'guidance')


@dataclass(frozen=True)
class ScientificTarget:
    output: str
    subject: str | None = None

    def __post_init__(self):
        if (type(self.output) is not str or not 0 < len(self.output) <= 128
                or self.subject is not None and (type(self.subject) is not str or not 0 < len(self.subject) <= 128)):
            raise ValueError('Invalid scientific target.')


@dataclass(frozen=True)
class ScientificQuestion:
    target: ScientificTarget
    comparison: ScientificTarget | None = None
    focus: str = 'question'
    detail: dict | None = None
    visualization: object = None
    export: object = None

    def __post_init__(self):
        if (not isinstance(self.target, ScientificTarget)
                or self.comparison is not None and not isinstance(self.comparison, ScientificTarget)
                or self.focus not in ('question', 'continue')):
            raise ValueError('Invalid scientific question.')
        if self.detail is not None:
            validate_scientific_detail(self.detail, self.comparison)
        if self.visualization is not None:
            _validate_visualization(self.visualization, planned=False)
            object.__setattr__(self, 'visualization', freeze_json_mapping(self.visualization, 'visualization'))
        if self.export is not None:
            _validate_export(self.export)
            object.__setattr__(self, 'export', freeze_json_mapping(self.export, 'export'))


def validate_scientific_detail(detail, comparison):
    """Validate the existing two-target selection, not a list of read queries."""
    _shape(detail, ('section', 'field', 'other_field', 'offset', 'limit'))
    from .dialogue_evidence import DetailRequest
    if detail['section'] not in ('categorical_fields', 'category_counts', 'category_association'):
        raise ValueError('Unsupported scientific detail request.')
    single = dict(detail)
    if detail['section'] == 'category_counts' and detail['other_field'] is not None:
        if comparison is None:
            raise ValueError('Independent second counts require a second accepted target.')
        single['other_field'] = None  # Each owner count read still has one field.
    DetailRequest(**single)


@dataclass(frozen=True)
class GuidanceQuestion:
    targets: tuple[ScientificTarget, ...] = ()
    candidate: str | None = None

    def __post_init__(self):
        if (type(self.targets) is not tuple or len(self.targets) > 4
                or any(not isinstance(t, ScientificTarget) for t in self.targets)
                or self.candidate is not None and (type(self.candidate) is not str or not 0 < len(self.candidate) <= 128)
                or self.candidate is not None and self.targets):
            raise ValueError('Invalid bounded guidance question.')


@dataclass(frozen=True)
class Answer:
    intent: str
    relation: str = 'current'
    technical: bool = False
    scientific: ScientificQuestion | None = None
    guidance: GuidanceQuestion | None = None

    def __post_init__(self):
        if self.intent not in ANSWER_INTENTS or self.relation not in RELATIONS or type(self.technical) is not bool:
            raise ValueError('Invalid bounded answer request.')
        if self.intent == 'comparison' and self.relation == 'current':
            raise ValueError('Comparison requires a historical relation.')
        if (self.intent == 'scientific') != isinstance(self.scientific, ScientificQuestion):
            raise ValueError('Scientific answers require a structured question.')
        if (self.intent == 'guidance') != isinstance(self.guidance, GuidanceQuestion):
            raise ValueError('Guidance answers require a structured question.')


TurnDecision = RequestResources | ResourceAnswer | Execute | ExecuteCandidate | SpeciesAnswer | ParameterAnswer | RequestRoles | RoleAnswer | Navigate | Clarify | Answer
RELATIONS = ('current', 'parent', 'previous_active', 'previous')
REASONS = ('missing_parameter_value', 'ambiguous_parameter', 'ambiguous_revision',
           'unsupported_intent', 'unavailable_model_condition', 'invalid_parameter_value', 'ungrounded_operand',
           'unavailable_context', 'invalid_decision', 'planning_failed', 'ambiguous_subject',
           'ambiguous_predecessor', 'incompatible_comparison', 'requires_execution',
           'missing_species', 'ambiguous_species', 'unsupported_species',
           'conflicting_species', 'invalid_reference', 'missing_reference', 'missing_resources', 'invalid_prerequisite', 'conflicting_scientific_parameter')
# These are language aliases, not scientific types/ranges/defaults. Those stay
# exclusively in the registered ArgumentSpecs. Initial interaction scope is QC selection.
ALIASES = {
    'min_tss_enrichment': ('tss', 'tss threshold', 'min tss', 'min tss enrichment', 'minimum tss enrichment'),
    'min_qc_fragment_records': ('min fragments', 'fragment threshold', 'minimum fragments', 'min qc fragment records', 'minimum depth'),
    'min_tss_flank_evidence': ('min tss flank evidence', 'minimum flank evidence', 'flank evidence'),
    'max_qc_fragment_records': ('max qc fragment records', 'maximum depth', 'max fragments'),
    'max_nucleosome_signal': ('max nucleosome signal', 'maximum nucleosome signal', 'nucleosome threshold'),
}


def _shape(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise IntentError('invalid_decision')


def _leiden_argument(value):
    if value is None:
        return None
    _shape(value, ('tool', 'argument', 'literal', 'start', 'end'))
    return LeidenResolution(**value)


def _scientific_arguments(value):
    if type(value) is not list or len(value) > MAX_SCIENTIFIC_ARGUMENTS:
        raise IntentError('invalid_decision')
    result = []
    for item in value:
        cls, keys = _argument_shape(item)
        _shape(item, keys + ('start', 'end'))
        if 'members' in item:
            if type(item['members']) not in (list, tuple):
                raise IntentError('invalid_decision')
            item = dict(item, members=tuple(ArgumentMember(**m)
                                           for m in item['members']))
        result.append(cls(**item))
    return tuple(result)


def _argument_shape(value):
    cls, keys = _ordinary_argument_shape(value)
    if type(value) is dict and 'native_function' in value:
        variants = {ScopedArgument: NativeArgument, ScalarChoiceArgument: NativeChoiceArgument,
            StructuredArgument: NativeStructuredArgument, SourceScopedArgument: NativeSourceArgument,
            SourceScopedChoiceArgument: NativeSourceChoiceArgument,
            SourceScopedStructuredArgument: NativeSourceStructuredArgument}
        if cls not in variants:
            raise IntentError('invalid_decision')
        return variants[cls], keys + ('native_function',)
    return cls, keys


def _ordinary_argument_shape(value):
    choice = type(value) is dict and 'value' in value
    source = type(value) is dict and 'source' in value
    scoped = type(value) is dict and 'port' in value
    if type(value) is dict and 'members' in value:
        keys = ('tool', 'argument', 'literal', 'members')
        return ((SourceScopedStructuredArgument, keys + ('source', 'port')) if scoped
                else (StructuredArgument, keys))
    keys = ('tool', 'argument', 'literal') + (('value',) if choice else ())
    if scoped:
        return (SourceScopedChoiceArgument if choice else SourceScopedArgument), keys + ('source', 'port')
    if source:
        return SourceFieldArgument, keys + ('source', 'mode')
    return (ScalarChoiceArgument if choice else ScopedArgument), keys


def _input_roles(value):
    if type(value) is not list or len(value) > 8:
        raise IntentError('invalid_decision')
    result = []
    for item in value:
        _shape(item, ('tool', 'port', 'source'))
        result.append(InputRole(**item))
    return tuple(result)


def _role_requirements(value):
    if type(value) is not list or not 1 <= len(value) <= 8:
        raise IntentError('invalid_decision')
    result = []
    for item in value:
        _shape(item, ('tool', 'port'))
        result.append(RoleRequirement(**item))
    return tuple(result)


def _numeric_token_span(text, start, end, *, legacy_modifier=False):
    """Lexical numeric identity shared by quote location and scalar admission."""
    if start and (text[start - 1].isalnum() or text[start - 1] in '.+-/_%'):
        return False
    if end < len(text):
        following = text[end]
        modifier = (not legacy_modifier and following == '-'
                    and end + 1 < len(text) and text[end + 1].isalpha())
        sentence_period = (following == '.'
            and (end + 1 == len(text) or not text[end + 1].isdigit()))
        if (following.isalnum() or following in '+-/_%' and not modifier
                or following == '.' and not sentence_period):
            return False
    return True


def _ground_declaration(value, utterance, *, _legacy_numeric_grounding=False, _string_literal=False):
    """Locate provider-selected text; persisted declarations remain strict spans.

    Association belongs to the interpreter. An exact context quote can distinguish
    repeated literals, but neither offsets nor first-match search resolve a new
    ambiguous declaration. This boundary is never used to reload Session records.
    """
    if value is None:
        return None
    if type(value) is dict and value.get('native_function', True) is None:
        value = {k: v for k, v in value.items() if k != 'native_function'}
    cls, keys = _argument_shape(value)
    def members():
        if type(value['members']) not in (list, tuple) or not 1 <= len(value['members']) <= 32:
            raise IntentError('invalid_decision')
        grounded = []
        for member in value['members']:
            _shape(member, ('name', 'kind', 'literal', 'evidence'))
            span = _ground_declaration(dict(tool=value['tool'], argument=value['argument'],
                literal=member['literal'], evidence=member['evidence']), utterance)
            result = dict(name=member['name'], kind=member['kind'],
                **{k: span[k] for k in ('literal', 'start', 'end')})
            grounded.append(result)
        return grounded
    if 'members' in value and 'literal' not in value:
        _shape(value, tuple(k for k in keys if k != 'literal'))
        grounded = members()
        start = min(m['start'] for m in grounded)
        end = max(m['end'] for m in grounded)
        return dict(value, members=grounded, literal=utterance[start:end], start=start, end=end)
    quoted = type(value) is dict and 'evidence' in value
    _shape(value, keys + (('evidence',) if quoted else ('start', 'end')))
    literal = value['literal']
    if type(literal) is not str or not 0 < len(literal) <= 4096:
        raise IntentError('invalid_decision')
    if not quoted:
        # Validate the legacy wire before normalization, including offset types.
        cls(**value)
        if utterance[value['start']:value['end']] == literal:
            return value
    evidence = value['evidence'] if quoted else literal
    if type(evidence) is not str or not 0 < len(evidence) <= 4096:
        raise IntentError('invalid_decision')
    def unique(text, token, *, numeric=False, string=False):
        matches = [m.start() for m in re.finditer('(?=' + re.escape(token) + ')', text)
                   if not numeric or _numeric_token_span(text, m.start(), m.start() + len(token))]
        if string and len(matches) > 1:
            quotes = {'"':'"', "'":"'", '`':'`', '“':'”', '‘':'’', '「':'」', '『':'』'}
            quoted = [i for i in matches if i > 0 and i+len(token) < len(text)
                      and quotes.get(text[i-1]) == text[i+len(token)]]
            if quoted:
                matches = quoted
        if not matches:
            raise IntentError('ungrounded_operand')
        if len(matches) != 1:
            raise IntentError('ambiguous_parameter')
        return matches[0]
    numeric = (not _legacy_numeric_grounding and re.fullmatch(
        r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?:/[+-]?\d+)?', literal) is not None)
    origin = unique(utterance, evidence, numeric=numeric and evidence == literal)
    start = origin + unique(evidence, literal, numeric=numeric, string=_string_literal)
    result = {key: value[key] for key in keys} | dict(start=start, end=start + len(literal))
    if 'members' in result:
        result['members'] = members()
    return result


def parse_decision(raw, *, utterance=None):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result: raise IntentError('invalid_decision')
            result[k] = v
        return result
    try:
        if type(raw) is not str or len(raw.encode()) > 32768: raise IntentError('invalid_decision')
        envelope = json.loads(raw, object_pairs_hook=pairs)
        _shape(envelope, ('turn_schema_version', 'decision'))
        if type(envelope['turn_schema_version']) is not int or envelope['turn_schema_version'] != 1:
            raise IntentError('invalid_decision')
        value = envelope['decision']
        if utterance is not None:
            if type(utterance) is not str or not 0 < len(utterance) <= 4096:
                raise IntentError('invalid_decision')
            failure = None
            failed_parameters = []
            def ground(item):
                nonlocal failure
                try:
                    return _ground_declaration(item, utterance)
                except IntentError as exc:
                    if exc.reason not in {'ungrounded_operand', 'ambiguous_parameter'}:
                        raise
                    failure = exc.reason
                    failed_parameters.append(item['tool'] + '.' + item['argument'])
                    # Structural validation only; these placeholders never leave
                    # this parser or become admitted parameter evidence.
                    placeholder = {k: v for k, v in item.items() if k not in {'evidence', 'start', 'end'}} | dict(start=0, end=1)
                    if placeholder.get('native_function', True) is None:
                        placeholder.pop('native_function')
                    if 'members' in placeholder:
                        placeholder.setdefault('literal', '?')
                        placeholder['members'] = [dict(name=m['name'], kind=m['kind'], literal=m['literal'],
                            start=0, end=1)
                            for m in placeholder['members']]
                    return placeholder
            if 'argument' in value:
                value['argument'] = ground(value['argument'])
            if 'arguments' in value:
                if type(value['arguments']) is not list or len(value['arguments']) > MAX_SCIENTIFIC_ARGUMENTS:
                    raise IntentError('invalid_decision')
                value['arguments'] = [ground(item) for item in value['arguments']]
            for field in ('fastq', 'bam', 'fragments'):
                declared = value.get(field)
                if declared is None:
                    continue
                # Current language output can quote a user preference, never
                # select an ungrounded application handle. Old saved decisions
                # are parsed without an utterance by the historical reader.
                if declared.get('resource') is not None:
                    raise IntentError('ungrounded_operand')
                preference = declared.get('resource_preference')
                if preference is not None:
                    bound = _ground_declaration(dict(tool='run_snapatac2', argument='resource', **preference),
                                                utterance, _string_literal=True)
                    declared['resource_preference'] = {k: bound[k] for k in ('literal', 'start', 'end')}
                declared.setdefault('resource', None)
            parsed = parse_decision(json.dumps(envelope))
            if failure:
                raise DeclarationGroundingError(failure, raw, tuple(dict.fromkeys(failed_parameters)))
            return parsed
        kind = value.get('kind')
        if kind == 'execute_candidate':
            _shape(value, ('kind', 'candidate', 'evidence'))
            return ExecuteCandidate(value['candidate'], value['evidence'])
        if kind == 'answer_prerequisite':
            _shape(value, ('kind', 'pending', 'species') + tuple(
                key for key in ('argument', 'arguments') if key in value))
            return SpeciesAnswer(value['pending'], value['species'],
                                 _leiden_argument(value.get('argument')),
                                 _scientific_arguments(value.get('arguments', [])))
        if kind == 'request_resources':
            _shape(value, ('kind', 'target', 'arguments') + tuple(k for k in ('roles', 'reference_category', 'context', 'input_results') if k in value))
            if 'input_results' in value and type(value['input_results']) is not list:
                raise IntentError('invalid_decision')
            return RequestResources(value['target'], _scientific_arguments(value['arguments']),
                                    _input_roles(value.get('roles', [])), value.get('reference_category'), value.get('context', 'none'),
                                    input_results=tuple(value.get('input_results', ())))
        if kind == 'answer_resources':
            _shape(value, ('kind', 'pending', 'choice', 'arguments'))
            return ResourceAnswer(value['pending'], value['choice'], _scientific_arguments(value['arguments']))
        if kind == 'answer_parameters':
            _shape(value, ('kind', 'pending', 'arguments'))
            return ParameterAnswer(value['pending'], _scientific_arguments(value['arguments']))
        if kind == 'request_roles':
            _shape(value, ('kind', 'target', 'required_roles', 'roles') + tuple(
                key for key in ('argument', 'arguments', 'context', 'fastq', 'bam', 'fragments', 'annotation', 'epizoo',
                                'input_results', 'references', 'delivery', 'unfulfilled', 'visualization', 'comparison', 'retain_parameters') if key in value))
            if any(type(value[k]) is not list for k in ('input_results', 'references', 'delivery', 'unfulfilled', 'retain_parameters') if k in value):
                raise IntentError('invalid_decision')
            from .raw_handoff import declaration
            return RequestRoles(value['target'], (() if (value.get('fastq') is not None or value.get('bam') is not None or value.get('fragments') is not None or value.get('annotation') is not None or value.get('epizoo') is not None) and value['required_roles'] == []
                                                    else _role_requirements(value['required_roles'])),
                                _input_roles(value['roles']), _leiden_argument(value.get('argument')),
                                _scientific_arguments(value.get('arguments', [])), value.get('context', 'none'),
                                declaration(value['fastq']) if 'fastq' in value else None,
                                declaration(value['bam'], 'bam') if 'bam' in value else None,
                                declaration(value['fragments'], 'fragments') if 'fragments' in value else None,
                                declaration(value['annotation'], 'annotation') if value.get('annotation') is not None else None,
                                declaration(value['epizoo'], 'epizoo') if 'epizoo' in value else None,
                                input_results=tuple(value.get('input_results', [])), references=tuple(value.get('references', [])), delivery=tuple(value.get('delivery', [])),
                                unfulfilled=tuple(value.get('unfulfilled', [])), visualization=value.get('visualization'),
                                comparison=value.get('comparison'),
                                retain_parameters=tuple(RetainedParameter(**item) for item in value.get('retain_parameters', [])))
        if kind == 'answer_roles':
            _shape(value, ('kind', 'pending', 'roles') + tuple(
                key for key in ('arguments', 'fastq', 'bam', 'fragments', 'annotation', 'epizoo', 'unknown_fields') if key in value))
            if 'unknown_fields' in value and type(value['unknown_fields']) is not list:
                raise IntentError('invalid_decision')
            from .raw_handoff import declaration
            return RoleAnswer(value['pending'], _input_roles(value['roles']),
                              _scientific_arguments(value.get('arguments', [])),
                              declaration(value['fastq']) if 'fastq' in value else None,
                                declaration(value['bam'], 'bam') if 'bam' in value else None,
                                declaration(value['fragments'], 'fragments') if 'fragments' in value else None,
                                declaration(value['annotation'], 'annotation') if value.get('annotation') is not None else None,
                                declaration(value['epizoo'], 'epizoo') if 'epizoo' in value else None,
                              unknown_fields=tuple(value.get('unknown_fields', ())))
        if kind == 'answer_guidance':
            _shape(value, ('kind', 'targets', 'candidate'))
            if type(value['targets']) is not list: raise IntentError('invalid_decision')
            for target in value['targets']: _shape(target, ('output', 'subject'))
            return Answer('guidance', guidance=GuidanceQuestion(
                tuple(ScientificTarget(**t) for t in value['targets']), value['candidate']))
        # Branch identity determines these internal facts. Accept the complete
        # historical wire only with matching assertions, never partial forms.
        if kind == 'answer_scientific':
            if 'intent' in value:
                if value['intent'] != 'scientific': raise IntentError('invalid_decision')
            else:
                _shape(value, ('kind', 'target', 'comparison', 'focus')
                       + tuple(k for k in ('detail', 'visualization', 'export') if k in value))
            value = dict(value, kind='answer', intent='scientific')
            kind = 'answer'
        elif kind == 'execute_plan':
            if {'kind', 'target'} <= set(value) <= {'kind', 'target', 'argument', 'arguments', 'context', 'roles', 'input_results', 'references', 'delivery', 'retain_parameters', 'unfulfilled', 'visualization', 'comparison', 'missing_parameters'}:
                if type(value['target']) is not str:
                    raise IntentError('invalid_decision')
                if any(type(value[k]) is not list for k in ('input_results', 'references', 'delivery', 'retain_parameters', 'unfulfilled', 'missing_parameters') if k in value):
                    raise IntentError('invalid_decision')
                return Execute('current', 'plan', value['target'], None,
                               _leiden_argument(value.get('argument')),
                               _scientific_arguments(value.get('arguments', [])),
                               value.get('context', 'none'), _input_roles(value.get('roles', [])),
                               tuple(value.get('references', [])), tuple(value.get('delivery', [])),
                               tuple(RetainedParameter(**item) for item in value.get('retain_parameters', [])),
                               tuple(value.get('unfulfilled', [])), value.get('visualization'), value.get('comparison'),
                               tuple(value.get('missing_parameters', [])), input_results=tuple(value.get('input_results', [])))
            else:
                _shape(value, ('kind', 'base', 'operation', 'target', 'delta'))
                if value['operation'] != 'plan' or value['base'] != 'current' or value['delta'] is not None:
                    raise IntentError('invalid_decision')
            value = dict(value, kind='execute', operation='plan', base='current', delta=None)
            kind = 'execute'
        if kind == 'clarify':
            _shape(value, ('kind', 'reason') + (('choices',) if 'choices' in value else ()))
            if value['reason'] not in REASONS: raise IntentError('invalid_decision')
            choices = value.get('choices', [])
            if (not isinstance(choices, list) or len(choices)>8 or len(set(choices))!=len(choices)
                    or any(not isinstance(v,str) or not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*\.[A-Za-z_][A-Za-z_0-9]*',v) for v in choices)
                    or choices and value['reason'] not in PARAMETER_CLARIFICATION_REASONS):
                raise IntentError('invalid_decision')
            return Clarify(value['reason'], tuple(choices), value_required=value['reason'] == 'missing_parameter_value')
        if kind == 'answer':
            if value.get('intent') == 'scientific':
                _shape(value, ('kind', 'intent', 'target', 'comparison', 'focus')
                       + tuple(k for k in ('detail', 'visualization', 'export') if k in value))
                def target(v):
                    _shape(v, ('output', 'subject'))
                    return ScientificTarget(**v)
                return Answer('scientific', scientific=ScientificQuestion(target(value['target']),
                    None if value['comparison'] is None else target(value['comparison']), value['focus'],
                    value.get('detail'), value.get('visualization'), value.get('export')))
            _shape(value, ('kind', 'intent', 'relation', 'technical'))
            return Answer(value['intent'], value['relation'], value['technical'])
        if kind == 'navigate':
            _shape(value, ('kind', 'relation'))
            if value['relation'] not in RELATIONS: raise IntentError('invalid_decision')
            return Navigate(value['relation'])
        _shape(value, ('kind', 'base', 'operation', 'target', 'delta'))
        if (kind != 'execute' or value['base'] not in RELATIONS
                or type(value['target']) is not str or type(value['operation']) is not str
                or value['operation'] != 'plan' and value['target'] not in ('selection', 'matrix')):
            raise IntentError('invalid_decision')
        delta = value['delta']
        if delta is not None:
            _shape(delta, ('parameter', 'operation', 'literal', 'evidence'))
            if any(type(v) is not str for v in delta.values()) or delta['operation'] not in ('set', 'add', 'subtract'):
                raise IntentError('invalid_decision')
            delta = IntentDelta(**delta)
        return Execute(value['base'], value['operation'], value['target'], delta)
    except (ValueError, TypeError, AttributeError, KeyError) as exc:
        if isinstance(exc, IntentError): raise
        raise IntentError('invalid_decision') from exc


def _object(properties):
    return dict(type='object', properties=properties, required=list(properties), additionalProperties=False)


def decision_schema():
    """One stable proposal grammar; Registry/context admission owns identities.

    Definitions describe declaration types, never a cross product of the current
    scientific inventory. Adding tools, kwargs or sources does not change it.
    """
    def text(limit=128):
        return dict(type='string', minLength=1, maxLength=limit)
    def enum(values):
        return dict(type='string', enum=list(values))
    def optional(value):
        return {'anyOf': [value, {'type': 'null'}]}
    def array(value, limit, minimum=0):
        return dict(type='array', items=value, minItems=minimum, maxItems=limit)
    def ref(name):
        return {'$ref': '#/$defs/' + name}
    definitions = {}
    identity = dict(tool=text(), argument=text())
    quote = dict(literal=text(4096), evidence=text(4096))
    definitions['member'] = _object(dict(name=text(), kind=enum(('string', 'number', 'json')), **quote))
    forms = [identity | quote, identity | quote | dict(value=text(80)),
             identity | dict(members=array(ref('member'), 32, 1))]
    variants = []
    for form in forms:
        form = form | dict(native_function=optional(text()))
        variants.extend((_object(form), _object(form | dict(source=text(), port=text()))))
    variants.append(_object(identity | quote | dict(source=text(), mode=enum(('offered', 'literal')), value=text(80))))
    definitions['declaration'] = {'anyOf': variants}
    declarations = array(ref('declaration'), MAX_SCIENTIFIC_ARGUMENTS)
    definitions['role'] = _object(dict(tool=text(), port=text(), source=text()))
    definitions['requirement'] = _object(dict(tool=text(), port=text()))
    definitions['target'] = _object(dict(output=text(), subject=optional(text())))
    definitions['detail'] = _object(dict(section=enum(('categorical_fields', 'category_counts', 'category_association')),
        field=optional(text(256)), other_field=optional(text(256)), offset=dict(type='integer', minimum=0),
        limit=dict(type='integer', minimum=1, maximum=32)))
    definitions['visualization'] = _object(dict(coordinates=text(), labels=text(), field=text(256)))
    definitions['comparison'] = _object(dict(left=text(), right=text(), detail=optional(ref('detail'))))
    definitions['export_field'] = _object(dict(output=text(), field=text(256), destination=text(256)))
    definitions['export'] = _object(dict(measurement=text(), observations=array(ref('export_field'), 32),
                                        representations=array(ref('export_field'), 32)))
    definitions['retained'] = _object(dict(result=text(), argument=text()))
    definitions['delta'] = _object(dict(parameter=text(), operation=enum(('set', 'add', 'subtract')), **quote))
    delivery = dict(input_results=array(text(), 32), references=array(text(), 32),
        delivery=array(enum(('explanation', 'comparison', 'visualization')), 3),
        unfulfilled=array(text(512), 8), visualization=optional(ref('visualization')),
        comparison=optional(ref('comparison')), retain_parameters=array(ref('retained'), 8))
    execution = dict(target=text(), arguments=declarations, roles=array(ref('role'), 8),
                     context=enum(('none', 'current')), **delivery)
    variants = [
        _object(dict(kind=enum(('answer',)), intent=enum(i for i in ANSWER_INTENTS if i not in ('scientific', 'guidance')),
                     relation=enum(RELATIONS), technical=dict(type='boolean'))),
        _object(dict(kind=enum(('clarify',)), reason=enum(PARAMETER_CLARIFICATION_REASONS),
                     choices=array(dict(text(256), pattern=r'^[A-Za-z_][A-Za-z_0-9]*\.[A-Za-z_][A-Za-z_0-9]*$'), 8))),
        _object(dict(kind=enum(('clarify',)), reason=enum(r for r in REASONS if r not in PARAMETER_CLARIFICATION_REASONS),
                     choices=array(text(256), 0))),
        _object(dict(kind=enum(('navigate',)), relation=enum(RELATIONS))),
        _object(dict(kind=enum(('execute',)), base=enum(RELATIONS), operation=text(),
                     target=enum(('selection', 'matrix')), delta=optional(ref('delta')))),
        _object(dict(kind=enum(('answer_scientific',)), target=ref('target'), comparison=optional(ref('target')),
                     focus=enum(('question', 'continue')), detail=optional(ref('detail')),
                     visualization=optional(ref('visualization')), export=optional(ref('export')))),
        _object(dict(kind=enum(('answer_guidance',)), targets=array(ref('target'), 4), candidate=optional(text()))),
        _object(dict(kind=enum(('execute_plan',)), **execution, missing_parameters=array(text(), 8))),
        _object(dict(kind=enum(('request_roles',)), **execution, required_roles=array(ref('requirement'), 8, 1))),
        _object(dict(kind=enum(('answer_roles',)), pending=text(), roles=array(ref('role'), 8), arguments=declarations)),
        _object(dict(kind=enum(('execute_candidate',)), candidate=text(), evidence=text(4096))),
        _object(dict(kind=enum(('request_resources',)), target=text(), arguments=declarations,
                     roles=array(ref('role'), 8), context=enum(('none', 'current')),
                     input_results=array(text(), 32),
                     reference_category=optional(enum(('peak_set', 'regulatory_regions'))))),
        _object(dict(kind=enum(('answer_resources',)), pending=enum(('@resources',)),
                     choice=dict(text(), pattern=r'^resource\.'), arguments=declarations)),
        _object(dict(kind=enum(('answer_prerequisite',)), pending=text(), species=enum(('human', 'mouse')), arguments=declarations)),
        _object(dict(kind=enum(('answer_parameters',)), pending=text(), arguments=array(ref('declaration'), MAX_SCIENTIFIC_ARGUMENTS, 1))),
    ]
    from .raw_handoff import schema as raw_schema, empty, scientific_metadata_fields
    for field in ('fastq', 'bam', 'fragments', 'annotation', 'epizoo'):
        definitions[field] = raw_schema(field)
        roles = array(ref('role'), 0)
        downstream = dict(annotation=optional(ref('annotation'))) if field in ('fastq', 'bam', 'fragments') else {}
        variants.append(_object(dict(kind=enum(('request_roles',)), target=text(), roles=roles,
            required_roles=array(ref('requirement'), 0), arguments=declarations, **{field: ref(field)}, **downstream, **(dict(context=enum(('none', 'current'))) if field == 'annotation' else {}), **delivery)))
        variants.append(_object(dict(kind=enum(('answer_roles',)), pending=text(), roles=roles,
            arguments=declarations, **{field: ref(field)}, **downstream,
            unknown_fields=array(enum(scientific_metadata_fields(empty(field))), 16))))
    return _object(dict(turn_schema_version=dict(type='integer', enum=[1]), decision={'anyOf': variants})) | {'$defs': definitions}


def _interpretation_public(public, utterance):
    """One Registry projection, with exact-name detail expansion and no routing.

    Every capability, function and parameter remains discoverable. Exact API
    mentions select explanatory detail only; they confer no intent or authority.
    """
    dialogue = public.get('dialogue')
    if dialogue is None:
        return public
    outputs = {row['handle']: dict(row) for row in dialogue['outputs']}
    fields = ('available_fields', 'fields_omitted', 'subjects')
    for group in dialogue.get('result_groups', ()):
        canonical = outputs.get(group.get('canonical_output'))
        if canonical is None or not all(k in canonical for k in fields):
            continue
        for handle in group['outputs']:
            member = outputs[handle]
            if member is canonical or any(member.get(k) != canonical[k] for k in fields):
                continue
            for k in fields:
                member.pop(k)
            member['evidence_from'] = canonical['handle']
    capabilities = {name: dict(dialogue.get('capabilities', {}).get(name, {}))
                    for name in dialogue.get('tools', ())}
    mentions = set()
    for tool, item in capabilities.items():
        for function in item.get('functions', {}):
            names = (function, item['package'] + '.' + function, item['package'] + '::' + function)
            if any(re.search(r'(?<![\w.:])' + re.escape(name) + r'(?!\w|\.[A-Za-z_])', utterance) for name in names):
                mentions.add((tool, function))
    pending = dialogue.get('pending_prerequisite') or {}
    contextual = {g['tool'] for g in dialogue.get('result_groups', ()) if g.get('is_active')}
    if pending.get('tool'):
        contextual.add(pending['tool'])
    for tool, item in capabilities.items():
        named_package = bool(item.get('package') and re.search(
            r'(?<![\w.:])' + re.escape(item['package']) + r'(?![\w.:])', utterance, re.I))
        named_tool = bool(re.search(r'(?<![\w.])' + re.escape(tool) + r'(?![\w.])', utterance))
        detailed = named_package or named_tool or tool in contextual
        if not detailed and not any(t == tool for t, _ in mentions):
            item.pop('notes', None)
        item['inputs'] = [{k: v for k, v in role.items() if k != 'tool'}
                          for role in dialogue.get('input_roles', ()) if role['tool'] == tool]
        parameters = {}
        for row in dialogue.get('scientific_parameters', ()):
            if row['tool'] != tool:
                continue
            value = {k: v for k, v in row.items() if k not in ('tool', 'argument')}
            if item.get('functions') and 'members' in value and not detailed:
                needed = {name for fn, names in item['functions'].items() if (tool, fn) in mentions for name in names}
                # Full names/signatures support open task discovery. Only exact
                # API names and captured context expand the explanatory detail.
                value['member_names'] = list(value['members'])
                value['members'] = {k: v for k, v in value['members'].items() if k in needed}
            parameters[row['argument']] = value
        item['parameters'] = parameters
    projected = {k: v for k, v in dialogue.items() if k not in ('tools', 'input_roles', 'scientific_parameters', 'capabilities')}
    return public | {'dialogue': projected | {'outputs': list(outputs.values()), 'capabilities': capabilities}}


def interpret(model, utterance, public, *, diagnostic_context=None):
    groups = public.get('dialogue', {}).get('result_groups', ())
    prompt = json.dumps({'turn_schema_version': 1, 'utterance': utterance, **_interpretation_public(public, utterance),
        'instructions': [
            'dialogue.capabilities is the complete Registry overview, keyed by canonical tool. Each entry gives its purpose, input ports, parameters and, for native packages, exact functions and supported kwargs. Parameter defaults describe native omission behavior, never user declarations. Exact API/package names, pending tasks and active results expand relevant kwarg descriptions; all other names and function signatures remain available for task discovery. The fixed response schema describes proposals only: use exact canonical identities from this context. Existing admission rejects unregistered tools, functions, parameters, sources, ports and associations. The Planner receives complete method details and composes science; you select intent and grounded declarations, never DAG steps. Preserve named implementations, source measurement and native omission defaults. Do not infer preprocessing, biological validity or extra analyses.',
            'For read-only questions about registered tool requirements, installed model applicability or metadata, methodological advice and useful analyses, choose answer_guidance. Tool/resource requirements are context, not accepted empirical evidence; explain limits without inferring biological equivalence. An unsupported model condition can be explained through guidance without executing or substituting a condition. Select up to four relevant exact evidence targets; an empty set is allowed and means no result evidence selected, not scientific absence.',
            'Guidance co-presents evidence; it does not compare measurements or infer common populations/cluster correspondence. Explicit subjects and historical targets obey the same exact reference rules as scientific answers.',
            'For a guidance-option rationale follow-up, targets=[] and candidate is an offered guidance_predecessor reference quoted in the utterance, or @candidate only when a unique candidate exists. For a new objective use candidate=null. No prior generated rationale is evidence.',
            'For explicit execution of an offered execution_candidate use execute_candidate with its exact candidate ID and an exact complete command clause from this utterance as evidence. Never translate an option into execute_plan or construct a workflow here.',
            'Resolve ordinal or named candidate references against the offered stable options and objective. Clarify ambiguous other/that references, stale references, conflicting subject/revision refinements, or missing execution intent. Interest or agreement alone is not execution authorization. Why/evidence questions remain answer_guidance.',
            'Current-turn refinements remain user planning intent, not edits to the original candidate. Keep its captured subjects/revision fixed; an additional comparison may refine the objective but must not silently replace its subject.',
            'First distinguish discussing an existing scientific result from asking about workflow state or requesting new computation.',
            'For what a result shows, means, establishes, why an assignment occurred, or what changed scientifically, choose answer_scientific. Operational selection/matrix/version answers only describe workflow state; they do not explain scientific results.',
            'Questions about the purpose, contents or available variants of downloads for already accepted results use answer_scientific with exact accepted result targets. Existing export descriptors describe available contents and preparation status. Use answer_guidance when the researcher asks about possible new scientific operations, rather than diverting an existing-download question into inspection or readiness advice.',
            *(['For a reusable H5AD composed only from existing accepted results, choose answer_scientific.export. Select measurement as the exact accepted result g whose original measurements are requested, an accepted matrix g, or an explicitly offered source.N. Select observations and representations individually with their exact accepted output g, field and intended destination name. Observation fields use the exact result:/source: names offered in that result group\'s export_fields or categorical.fields; representations select the exact obsm: field in that group\'s export_fields, describing its actual accepted embedding representation or X_umap. Select 1 to 32 fields in total; include only requested fields, never every retained result. Destinations must be distinct within each slot and must not overwrite existing original fields; clarify missing or ambiguous destination choices. No new output, arbitrary obsm field, recency choice or scientific computation is authorized. Exact source lineage and ordered cells remain required at the export owner; an eligible figure or equal cell counts do not qualify original measurements. When sources or fields are unresolved, clarify rather than substituting a result or creating a scientific plan. Inventory describes a preparation recipe; it does not mean downloadable bytes have already been prepared or delivered.'] if groups else []),
            'dialogue.outputs is the accepted-result inventory. is_active identifies current results; semantics reuses registered descriptions/roles. evidence_status=available means a bounded accepted summary is readable. available_fields and subjects describe coverage, not factual values. evidence_from references the exact cooutput carrying identical available_fields, fields_omitted and subjects; all output handles retain their own identities.',
            *(['dialogue.result_groups groups exact cooutputs of the same accepted scientific step. Select a group handle for its whole scientific result, without copying a member or guessing a workflow. Independent groups are distinct results; clarify unresolved ambiguity.'] if groups else []),
            'dialogue.result_referents offers captured @current_result, @most_recently_created and @previous_turn_result scopes. They are different: navigation changes the active result without creating a result, and an immediately preceding answered scientific turn may discuss an existing subject. Select the semantic handle only when its target is unique; otherwise choose the relevant offered group/subject or clarify. No scanning backward or defaulting to another scope.',
            'dialogue.supplied_inputs describes current-turn structured input presence, shallow JSON types and registered consumer expectations only. It exposes no values and establishes neither scientific format, qualification nor execution readiness. Distinguish no supplied inputs from omitted unregistered fields. Input presence can inform intent selection; Planner/compiler/tools still own binding and validation.',
            'Pending scientific declarations are user intent, not accepted evidence. In answer_roles, explicitly stated corrections may update scientific metadata while preserving captured files, source roles, namespace and operation. A null declaration field means no addition. If the user explicitly says a previously recorded scientific claim is unknown or cannot be confirmed, name that offered metadata field in unknown_fields and leave its addition null; do not silently retain it as known. Never retract sources, roles, operation or resource identities through unknown_fields. When the user asks which prerequisites or evidence are needed rather than supplying declarations, choose answer_guidance; the captured pending task can inform conditional advice without scientific execution.',
            'For scientific tasks starting with external fragments use request_roles with target run_snapatac2, fragments and empty required_roles/roles. Bind source and optional feature_source only to explicitly selected attachments; source assembly/species are declarations. Namespace and source_profile may remain null: native import does not require internal profiles or certified barcode-correction history. Index is optional and is not used by native import. Existing fragments_resources supply chromosome sizes, exact regions and available GTF/GFF annotation. Missing biological history stays unknown. For uploaded custom BED preserve its source coordinates and feature_category; it directly supplies the requested regions without a reference-bundle qualification. genome_index_source supplies chromosome sizes; an explicitly supplied genome_source pins the exact FASTA association. Agent resolves the actual native inputs after function selection. resource_preference quotes only an explicitly requested reference name with literal/evidence; otherwise use null, never an internal resource handle. A broad request may compose native import, QC, filtering, a feature matrix and downstream analysis through ordinary planning. Do not ask the researcher to split internal steps or know Registry IDs. Pending fragments use answer_roles and explicit additions. Explicit native-function requests use ordinary execute_plan with the offered native source ports and unchanged native function/kwargs, without task-mode prerequisites.',
            'For EpiZoo target-species inference or native post-training with uploaded H5AD, use request_roles with epizoo and empty roles/required_roles. Set operation initialization with target adapt_epizoo_species for transfer without training, inference with target epizoo_embed_cells, or posttraining_initialized with target posttrain_epizoo_model to train the accepted initialized model; never train merely to obtain embeddings. Bind source from selected_sources. Declare scientific_name, taxonomy_id, assembly and fragment_counts/binary_accessibility only from explicit processing context; unknown fields remain null. For initialization leave strategy null for automatic qualified mapping assessment unless the user explicitly chooses de_novo or mapped_reference. Declare peak_set/regulatory_regions feature_category from the data context. Initialization has zero optimizer updates. For posttraining_initialized reuse the exact initialized model; no new genome or strategy is needed. Inference needs the exact accepted model vocabulary, not a new genome. Initialization may bind genome_source and optional genome_index_source attachments, or a qualified genome_resource. epizoo_resources are scientific models/references, distinct from the conversational LLM. resource and genome_resource express explicit preferences only; otherwise leave null for compatible unique selection. checkpoint_source may identify an uploaded exact qualified checkpoint; weights alone do not establish vocabulary or provenance. Missing mapping/resources must clarify, never invent coordinate correspondence. Use offered scientific arguments for explicit native settings; omit absent settings for owner defaults. Pending epizoo uses answer_roles, @roles and only additions; preserve the original inputs, operation and explicit settings. Never ask the researcher to build manifests, hashes, InputSets or authority records.',
            'When a raw FASTQ/BAM/fragments objective also requests cell-type annotation, retain the raw declaration and include annotation for user-supported species, assembly and tissue context. Match an explicitly stated tissue to the offered annotation resource context. Leave annotation source/groups_source/groups_result/uploaded null: counts and grouping will be selected by the Planner. Do not invent future measurement semantics; the selected native count construction owns them. Pending raw continuations keep their raw declaration and may update these downstream biological declarations.',
            'For region-count H5AD cell-type annotation use request_roles with annotation and empty roles/required_roles. For an uploaded source, bind source and optional exact ordered groups TSV from selected_sources. For accepted scientific inputs, select input_results; context=current permits broad reuse only when no exact scientific input is selected. Keep explanation/figure references in references, leave attachment source null and let the normal Planner select its exact accepted count and clustering ports. Both routes prepare internal specifications only after operation/source selection; never request specification filenames or digests from the researcher. To reuse accepted Leiden groups, select the exact annotation_group_results handle in groups_result; Agent projects the ordered groups after checking the original count-source identity. Do not ask for download/re-upload or grouping provenance for that accepted result. Declare species, assembly, raw measurement history, tissue context and grouping provenance only when supported by the user; missing fields stay null. Agent checks source-bound metadata and resolves compatible annotation_resources. Set grouping_strategy=reuse_only when the user requires existing groups or prohibits recomputing them; otherwise use compute_if_needed or null. If no qualified grouping exists, preserve the annotation goal: the Planner can compute scientifically appropriate grouping prerequisites using existing tools. Do not ask the user to create clusters or specify preprocessing enums. Reuse qualified supplied or accepted groups when available; ask only about genuinely missing biological declarations. Match a supported tissue meaning to an offered exact resource context; do not infer tissue from resource availability. resource is an explicit preference only; otherwise leave null. Use existing scientific arguments for explicit count_slot and group_column; do not invent defaults. Pending annotation requests use answer_roles with annotation and explicit additions or corrections to species, assembly, measurement or tissue context. Use the measurement identifier described in the annotation schema when the declared counting history supports it. Match user-supported tissue meaning to an offered exact resource context, including when a resource becomes available later. Corrections create a new declaration record; they cannot replace captured source files or override conflicting source metadata. A categorical obs column alone does not establish a clustering result. Never invent cell groups, count history or biological context.',
            'For user-supplied MAESTRO references, annotation.uploaded binds gene_source to a native MAESTRO refgenes transcript/exon table and signature_source to a candidate/gene TSV, using only captured source handles. Record explicit reference species, assembly, biological context, gene_namespace, publisher/derivation provenance and release for each supplied component; unknown fields stay null. These are declarations of origin and applicability, never proof of biological validation. Uploaded roles take precedence over catalog components; never fall back from an invalid explicit upload. A marker table alone cannot replace gene annotations or the qualified genome chromosome scope. Available annotation_resources may supply those remaining dependencies, including a genome-scope-only entry. Ask for missing scientific files or declarations, not InputSet IDs, paths, digests or manifests. A pending annotation may receive uploaded resources through answer_roles without losing the source H5AD or original request.',
            'For scientific BAM preprocessing use request_roles with target run_snapatac2, bam and empty required_roles/roles. Preserve the selected file, source assembly, assay and known barcode history. source_profile and namespace can stay null; the native BAM method owns filtering, duplicate flags, endpoints and shifts. It differs from historical Agent policies. Existing bam_resources supply compatible reference files. Ask intelligible questions only about actual missing scientific inputs; do not request FASTQs, sorting, a BAI, an internal profile or retained duplicate pairs merely for native BAM admission. For pending bam use answer_roles with @roles and explicit additions. For raw tasks, resource_preference quotes only an explicitly requested reference name with literal/evidence; omit it otherwise. Agent resolves concrete components after native function selection. Never emit a resource handle. A named native-function request uses execute_plan and native file ports directly, without substituting task defaults.',
            *(['dialogue.selected_sources lists captured original-file candidates, not accepted result inputs. current_turn_sources identifies files selected in this turn; other entries are earlier Session attachments. dialogue.capabilities[tool].inputs offers reviewed Registry consumers. For execute_plan, associate only the original files requested as inputs through roles={tool,port,source} entries using only these offers. Each consumer may have one source; preserve explicit source associations and the declared scientific role. When dialogue.attachment_candidates is true, these are captured candidate attachments: bind only the files explicitly relevant to the requested task; unused files remain staged. Their stored input_type is a representation hint, never proof of scientific eligibility. For scientific FASTQ preprocessing use request_roles with target run_chromap and the bounded fastq variant with empty required_roles and roles. Interpret explicit assay, layout, single-library namespace/grouping, species, raw barcode whitelist policy and member source/read-role/lane/chunk associations. Missing declarations remain null or absent members. Compression is inspected from bytes. FASTQ species belongs only in fastq.species; arguments may contain only the offered FASTQ scientific consumers, never an EpiZoo species argument. Offered fastq_resources handles select reusable configured references and whitelists, never source-bound contexts. Use resource_preference only for a user-quoted reference name; Agent resolves applicable references. Never invent other values or thresholds. Do not invent FASTQ read membership, barcode history, or unsupported consumers. A single missing role may require clarification. A label-column declaration belongs to its offered operation and reference/ground-truth role. First/second may identify these exact selection positions only when the user explicitly associates them; order, filenames and metadata never imply biological roles. When the requested registered operation is unambiguous but source associations are missing, choose request_roles with target, required_roles={tool,port} from these offers, only the explicitly admitted roles, and original explicit scientific arguments. The required consumers must describe the requested objective, include the target tool and have one exact source binding each; the same source may be explicitly assigned to multiple supported consumers. This captures a question, never a scientific plan or invented workflow. If the operation itself is ambiguous, or explicit associations conflict, use clarify/ambiguous_subject without capturing a task. This selection grants no scientific qualification. An unrelated new task receives no prior input roles. Pending parameter answers retain the exact original selection and roles; they cannot switch datasets.']
              if public.get('dialogue', {}).get('selected_sources') else []),
            'Original-file candidates and accepted results are separate source offers. Select a historical file companion through roles only when the requested consumer actually reads it. Multiple unresolved files require clarification; never select by order or transfer old scientific declarations. For operations on accepted current results, use planning_context and omit file roles; do not recompute from an upload just because it is offered. Questions remain read-only. selected_primary_source, when present, identifies the exact explicitly selected H5AD for a new direct annotation/EpiZoo request; companion candidates cannot replace it. A pending task retains its own captured source. When session_source_candidates_overflow is true, only current explicit files remain offered; ask the user to select required historical files in the input controls. No historical subset has been silently selected.',
            'dialogue.embedding_suitability contains read-only scientific-owner observations for exact offered H5AD sources, not execution authority. For ordinary clustering/UMAP prefer available qualified compatible EpiZoo; when no compatible EpiZoo is offered and PCA is eligible, use pca_embed_cells with the same neighbors/Leiden/UMAP tools. Explicit PCA uses PCA; explicit EpiZoo must never switch to PCA after failure. PCA needs no species, assembly or model resource declarations, and defaults to supplied X unchanged. Numeric PCA eligibility does not establish suitable scATAC preprocessing. For count or binary accessibility, the planner can explicitly select preprocessing=tfidf_l2 (smoothed document-frequency TF-IDF with L2 normalization then centered PCA); this is exploratory PCA, not uncentered LSI. Do not transform already processed measurements without an appropriate scientific basis. No cells/features are silently filtered. Omit component count and solver details unless explicitly requested. For a single selected H5AD, execute_plan can use its supplied input_path directly; for grouped candidate inputs bind pca_embed_cells.dataset through the existing roles. Questions on existing results remain read-only and use accepted result context.',
            'Resolve explicit result names or semantic roles against this inventory, preferring current results unless history is requested. Multiple unresolved candidates require ambiguous_subject, never a first match.',
            'A single current result or a unique dialogue.predecessor resolves this result. For short why, value-source or certainty follow-ups, retain that scientific target with @focus (and its subject); do not switch to operational provenance or a different output.',
            'The predecessor contains the exact target, previous target/comparison and prior question, not generated prose. No predecessor means no prior-discussion focus; captured result groups and result_referents remain available. Clarify unresolved short references instead of inventing a topic.',
            'Use unavailable_context only when the requested target is missing or its evidence unavailable. Missing rationale or scientific certainty within an available summary is for the scientific answer stage to explain as insufficient evidence, not a reason to reject the target.',
            'You select intent and exact references, not the scientific conclusion. You need not see factual values to select answer_scientific for an available result. Detailed accepted evidence is loaded after admission.',
            *(['Use answer_scientific.detail section=categorical_fields to discover eligible field names; field=null and offset/limit page field names, independently of category-record pages. A named field can be requested by its exact source:/result: handle even beyond the displayed page: Agent resolves it against the actual authorized source. Copy the exact field name, never guess, rename or supply an HDF5 path. Clarify genuinely ambiguous field/source choices. The predecessor records a prior selected page.'] if any('categorical' in g for g in groups) else []),
            *(['Use section=category_counts for distributions. field belongs to target; to request independent distributions from two accepted results, select comparison and its exact other_field. These are separate cited statistics, requiring no attempted association. With other_field=null only the target distribution is requested. Use section=category_association only for a requested contingency table: other_field belongs to comparison when selected, otherwise target. Counts and association have category-record offset/limit. Missing correspondence limits association, not independent facts. Source observations are not biological ground truth; method questions need no categorical detail.'] if any('categorical' in g for g in groups) else []),
            'Choose execute, navigate, clarify, or answer. Preserve existing operational answer intents.',
            'Operational answer intents: version, matrix, selection, thresholds, changes (originating revision versus parent), execution, reuse, comparison, provenance, verification. Use unsupported only outside the offered operational, scientific-result and guidance capabilities; missing empirical evidence does not make tool-applicability guidance unsupported.',
            'Operational comparison needs parent, previous_active, or previous. Scientific comparison selects exact offered result handles, including explicitly identified historical versions or branches. history_complete=false means older results may be omitted from this bounded inventory; ask for navigation to an unavailable version, never substitute another result. Other answers normally use current.',
            'For questions about scientific results, use kind=answer_scientific with offered output handles. It can return insufficient evidence.',
            'For a new scientific command outside the offered threshold operations use kind=execute_plan, target=one requested registered tool; the Planner composes the entire supported task across multiple operations, including repeated package tools. The Agent derives the plan operation on the current revision without a parameter delta. Never answer a computation request as if it was already computed.',
            *(['For execute_plan, references selects relevant captured result_groups for reading, comparison or presentation; it grants no execution permission. input_results separately selects exact captured result_groups for new scientific computation. Its qualified planning_context ports may supply executable inputs independently of context. context=current supplies broad captured-current reuse only when no exact scientific inputs are selected. A presentation reference alone grants no scientific input selection. Group relations describe the captured base, not a later active pointer. Recorded derived_from describes actual dependencies, not scientific correspondence inferred from equal sizes.'] if groups else []),
            'execute_plan and request_roles retain requested downstream outcomes alongside scientific computation. Use delivery=explanation for requested scientific interpretation. An exact comparison selects comparison.left and comparison.right from captured g handles or new for one explicitly selected planned result. Selecting the same result twice, including new/new, requires two distinct exact fields such as result:leiden and source:cell_type. detail=null requests recorded summary/method comparison of distinct results; category_counts requests independent distributions with exact field and other_field, while category_association requests a source-qualified contingency table. Use existing detail offset/limit bounds. Missing correspondence limits association, not independent facts. Precise comparison and visualization operands already declare those delivery intentions; redundant flags or references are unnecessary. Coarse flags without exact operands do not establish completed outcomes. Original-versus-candidate comparisons that are intrinsic outputs of a scientific tool are its result facts: request explanation for their interpretation, not a separate comparison with missing operands. Native tool figures may use delivery visualization with visualization=null. Do not invent Registry tools or promise fulfillment. A visualization requiring new scientific calculation must still be planned as science. Preserve the same precise operands in request_roles when files or declarations are missing.',
            'For a specific categorical figure, visualization selects coordinates from an explicit captured UMAP g handle or new for an explicitly selected planned UMAP result, labels from an accepted g handle, new for an explicitly selected planned label result, or an offered source.N attachment, and field as an exact result: or source: column. Source attachments require source: fields. The existing output selector identifies each planned-new purpose; never choose a step by name, order or tool guess. Reusing accepted coordinates does not require another UMAP fit. Never choose by latest/last/first, equal cell counts or filenames; clarify ambiguous sources or fields. Without new computation choose answer_scientific with accepted-only visualization operands (no new), preserving the exact requested figure even when its publication is unavailable. Do not fabricate a scientific plan for a presentation-only request. The existence of a different or uniformly colored PNG does not satisfy exact requested coloring.',
            'Preserve the whole scientific objective. Missing essential capabilities, ambiguous associations or incompatible scientific inputs cannot be silently omitted: clarify instead. Missing resolvable files/declarations use existing prerequisites. execute_plan.unfulfilled quotes exact clauses from this utterance for unsupported, independently separable requirements only when the user permits their omission without materially changing the objective. Agent records and discloses them; use [] otherwise. Never use this to skip essential science or resolvable prerequisites. If separability is unclear, clarify. Never add hidden fallback plans or retry a semantic refusal until success.',
            *(['Use retain_parameters for requested earlier settings of the same tool. Carry established scientific declarations without repetition only when rerunning that tool on the same input. Consuming an accepted output is a different input, even if the same tool consumes it: select its offered references and reusable_inputs, not its producer recipe. Never infer scientific declarations for a different or ambiguous input. Select exact offered result/argument pairs; Agent binds their recorded values and provenance. Never copy accepted values into explicit current arguments or manufacture literal evidence from prior defaults. Explicit current changes override only their own parameter; omitted unrelated parameters stay omitted, and unknown historical settings require clarification when essential.'] if groups else []),
            'For selected scientific uploads, structured inputs or accepted-result continuations, dialogue.resource_choices lists existing operator-reviewed InputSets by opaque handle and label. If the original requested task needs a missing or ambiguous reference, profile, namespace, explicit biological context/group specification or resource choice, choose request_resources with its registered target and explicit scientific arguments. Set context=current to preserve explicitly requested accepted-result reuse through resource clarification; no re-upload is needed. This preserves the objective and source for a later answer. Resources are configuration declarations, never proof of scientific compatibility. An offered @resources pending request is answered with answer_resources only when the user selects one of its exact choices for that original objective. Never invent paths, digests, resources, provenance or operations. Unrelated new tasks use ordinary decisions. When dialogue.reference_supplementation is available, a missing genome for declared species/assembly and an uploaded regulatory H5AD can use request_resources with explicit reference_category peak_set or regulatory_regions. This asks for plain FASTA (optional FAI) and preserves the objective. Qualified pending reference choices can answer that original request. Otherwise an operator must provision missing resources. Never fabricate resources.',
            'A supported scientific operation with missing files is not an unsupported capability. Use request_roles with the offered Registry input roles and only exact existing source associations; leave roles empty when no file is available. This captures the original objective for a later upload. Do not construct a workflow or invent source handles. If a required declaration is omitted for a selected input, choose execute_plan and omit that declaration from arguments; the compiler determines its exact prerequisite. Use missing_species only while answering an offered pending species prerequisite. Unsupported capabilities, ambiguous sources, conflicting declarations and incompatible inputs remain distinct.',
            'A package name alone does not request one function. A scientific objective using a named package may compose its registered functions. Explicit native function names fix those identities; mixed ordered function requests remain planned tasks. Use native_function to scope each declaration to its registered function. This scope is not a function_name value. An explicit function selection requires a function_name choice declaration with value equal to the offered canonical function, plus its verbatim literal/evidence. Put native keyword assignments in the kwargs parameter as members; member names are not top-level argument identities. Preserve exact explicit values, never insert unrequested preprocessing. Missing kwargs stay omitted. Ordinary fragments tasks need no sorting declaration: the task handoff supplies unknown input order and native sorting; never fabricate a user sorting flag. If an ordinary fragment-import task explicitly declares barcode grouping, capture the optional fragment_order choice scoped to pp.import_fragments. Never infer it from a filename or ask for it. Never add task fragment_order to explicit native-function requests. Use native import/filter/count defaults unless the user requests alternatives. A low-depth/all-barcode inspection request may bind its requested retention through min_num_fragments; do not lower thresholds simply because an input is small. A supported composition is not unsupported_intent merely because it needs several operations. Exact same-function repeated calls still need unambiguous source scope.',
            'An explicitly named scientific backend, algorithm or implementation constrains the requested operation. Choose an offered registered capability only if it supports that named choice; if unavailable, use clarify/unsupported_intent rather than substitute a similar implementation. For a general operation without a named implementation, select a suitable offered capability. Detailed planning and deterministic validation still enforce compatibility.',
            'Input-role required flags come from Registry ports. Required user files and optional user-provided resources are distinct. Ask for a missing required dataset; request an optional resource file only when the user explicitly selects that source or the scientific owner identifies a missing resource. Do not turn an offered optional port into a mandatory upload. Compatible installed resources remain available through their existing owners; never invent compatibility or replace an explicitly supplied file.',
            'Registered default_when_omitted values apply when the user has no special preference. Do not ask for optional scientific parameters merely because no explicit value was supplied. Preserve explicit requested values; defaults do not supply literal evidence for a user declaration.',
            'execute_plan.arguments contains explicit declarations for canonical tool/argument pairs offered in dialogue.capabilities[tool].parameters. Associate each value with the requested operation; never assign an ambiguous value to an arbitrary step. Copy each literal verbatim. For numeric parameters, literal is only the numeric token, without the parameter name or units: for example literal="2" with evidence="use 2 components". The evidence clause locates the token; do not put the whole clause into literal. In evidence quote an exact uniquely occurring clause from this utterance that associates this parameter with the literal and contains that literal exactly once. Agent computes character offsets. Repeated values need distinct unambiguous evidence; conflicting declarations require clarification unless the user explicitly corrects them. Do not use unrelated mentions as declarations. Different native functions of one package are distinct operations: native_function associates their parameters without requiring source/port. Use source and port only when the user associates a parameter with an operation directly consuming that exact captured original file, especially repeated instances of the same function on different files. Source/port asserts direct original-file consumption, not shared ancestry. In a multi-function workflow, bind its original input through roles and let the Planner compose derived dependencies; do not copy that original source scope onto downstream functions consuming newly produced results. This scopes direct-file values without constructing DAG steps. A value explicitly applying to both files needs one declaration per source; the same evidence clause may support both. Unscoped values never authorize repeated operations. source and port name captured attachments, not accepted result outputs. When consuming an accepted result, do not attach its producer\'s original file as a role or scope scalar choices to that file unless the requested consumer actually reads that file again. Use current accepted context and its exact offered result instead. Do not repeat the same tool/argument/source/port identity.',
            *(['For a json_object parameter, explicit JSON remains one verbatim literal without members. When reviewed members are offered, ordinary name=value or natural-language assignments use one declaration with members, each with its own exact literal/evidence. Do not supply an enclosing literal or invented JSON text; Agent builds the mapping from the grounded members. Select each canonical member name semantically from its description, never by an alias parser. kind=number preserves numeric tokens; kind=string copies the full string excluding quotation delimiters; kind=json copies an explicitly supplied JSON value (including arrays, booleans or null). Preserve explicit types, including invalid native values. Do not fabricate, repair or default omitted members. Do not combine contradictory assignments or explicit JSON and prose values that disagree; clarify ambiguous, conflicting or unrepresentable values. Source/port scope remains the same as any other scientific declaration.']
              if any(item.get('json_object') for item in public.get('dialogue', {}).get('scientific_parameters', ())) else []),
            *(['For an offered exact_string parameter without observation_source, literal is the explicitly supplied value (observation column, processing namespace or DNA sequence as described by the offered parameter), with surrounding quotation marks excluded from its span. Preserve case, Unicode, punctuation and whitespace exactly; do not normalize or invent reference/query/ground-truth relationships. Associate it with its offered scientific operation and provenance_role; ambiguous or conflicting declarations require clarification.']
              if any(item.get('exact_string') for item in public.get('dialogue', {}).get('scientific_parameters', ())) else []),
            'For a parameter with observation_source and an offered H5AD source, declare source, value, mode, literal and evidence. source explicitly selects that tool\'s observation_source dataset port; a matching roles entry is optional and must agree. Other dataset ports still need their own explicit roles. In mode=offered, value is an exact field name from that source\'s source_fields row, while literal/evidence quotes the researcher\'s natural reference to supplied annotations or the field. The researcher need not spell an internal column name. In mode=literal, the researcher explicitly names the column: value must equal the exact verbatim literal, excluding surrounding quotation marks; this may identify a name outside the bounded metadata offer. Never change modes after a rejected binding, omit value, borrow another file\'s field, infer ground truth from field presence, or invent a column from a descriptive phrase. Native owners qualify the selected data and exact axes. With no supported source yet offered, only an explicitly supplied exact column-name literal may be retained in the existing source-free declaration form; otherwise preserve the objective and request its file roles without inventing a name.',
            'For every offered parameter with choices, use the choice declaration form: value is required and must be an exact offered choice, while literal/evidence quotes the actual user declaration. This applies to native function_name and preprocessing choices as well as biological declarations. A literal-only declaration is not a choice, even when native_function supplies the same name. For example, an explicit package.fn request uses argument="function_name", value="fn", native_function="fn", literal="package.fn" and matching evidence, where package/fn are actual registered identities. Never derive a missing choice value from defaults or another declaration. An initial explicit species declaration about the selected cells or dataset belongs in execute_plan.arguments even when the target is a downstream tool. A statement about a model supporting human and mouse is not a declaration of the selected dataset species; omit that argument. Never infer species from filenames, dimensions, features, available resources or checkpoint identity. Unsupported or conflicting explicit dataset declarations require the corresponding clarification; an unresolved declaration in a supported new command uses the compiler-owned handoff above. A user declaration does not establish input compatibility. Pending species answers continue using answer_prerequisite.species, not a choice argument.',
            'A number alone does not request an operation. Explanatory parameter questions remain answers, never execution or edits to a pending request. Clarify ambiguous parameter association or conflicting values; do not invent an amount or replace an invalid value with a default.',
            'Use arguments=[] for omitted scientific parameters. Registered owners and existing compiler/resource contracts decide requirements and defaults; never invent a numeric default, infer scientific input identity, or create arbitrary execution dictionaries. Exact decimal or rational threshold literals stay exact; do not approximate them with floating point values.',
            'Scientific targets: @focus is the captured predecessor target; @previous is its comparison or previous subject. No predecessor means no implicit focus.',
            'Target subject selects only an offered annotation-summary group. Clustering labels and other categorical values are not target subjects: select their exact result and categorical fields through detail. The answer stage reads the requested records with their coverage; never invent a category or treat an unavailable table page as absence.',
            'Subject=null means no new subject is asserted by the model for this turn. '
            'It does not clear an authoritative subject retained through a resolved captured referent; '
            'you need not copy or reconstruct that subject identity. Without a retained subject, null selects the whole result. '
            'Use @focus to retain the predecessor subject, @other only for an unambiguous other subject, '
            'or an offered subject candidate ID/reference quoted in the question. References resolve only within that result; do not invent aliases.',
            'Use focus=continue to preserve the predecessor explanatory question on a subject change; otherwise question. For why/provenance/limitations ask the current question.',
            'Scientific comparison uses two explicit targets; do not infer correspondence between clusters from different results. Ambiguous other/previous requires clarification.',
            'A biggest/most important difference needs an explicit comparison dimension or reviewed criterion. Clarify when it is missing; do not invent a ranking. New markers, significance tests or reannotation require scientific execution, not interpretation.',
            'technical=true only for an explicit request for technical provenance, IDs or hashes.',
            'Select only offered operation, parameter and revision relations. Never emit internal IDs.',
            'For one parameter change, quote the exact complete user command clause as evidence and its numeric literal verbatim.',
            'set requires assignment; add requires increase by; subtract requires decrease by. Never invent an amount.',
            'Stricter without an amount requires missing_parameter_value. Ambiguous thresholds require ambiguous_parameter.',
            'Previous is ambiguous when parent and previous-active differ. Do not choose one silently.',
            'Use relation previous for earlier/previous version or go back one version; parent requires the word parent.',
            'Default parameter changes to selection only. Matrix requires an explicit rebuild request.',
            'Keep selection and rebuild matrix uses delta=null and target=matrix.',
        ] + ([
            'dialogue.planning_context offers opaque, recorded owner-qualified artifacts from the captured current Revision, not present-day integrity or unrestricted memory. Only scientific execution revalidates their authority and bytes.',
            'For execute_plan choose context=current only when the requested new computation explicitly uses or modifies that captured analysis. The Planner selects exact compatible offered artifacts; clarify an unresolved historical subject rather than choosing the first item.',
            'Choose context=none for an independent new task or changed scientific input. Do not transfer prior source-specific declarations or parameters. Questions remain read-only answers, navigation remains navigation, and pending answers continue only their exact captured original request.',
        ] if public.get('dialogue', {}).get('planning_context') else []) + ([
            'dialogue.pending_prerequisite describes one captured scientific request awaiting an explicit human/mouse species declaration. Its objective and selected dataset remain fixed; this context is not scientific compatibility evidence.',
            'Choose answer_prerequisite only when this utterance unambiguously answers that specifically pending species question for the same request and selected dataset. Use its exact offered pending handle and normalize the explicit declaration to the offered human or mouse value. Meaning is yours to interpret; never infer species from filenames, dimensions, features, available checkpoints or unrelated datasets.',
            'Uncertainty or ambiguous species requires clarify/ambiguous_species; another species requires clarify/unsupported_species; a conflicting declaration requires clarify/conflicting_species. Do not guess or emit arbitrary scientific input dictionaries.',
            'If the user changes the analysis objective, choose the ordinary decision for that new request, such as execute_plan. Cancellation or an unrelated message must never answer the old prerequisite; use clarify/unsupported_intent when no other supported decision applies. An answer authorizes continuation only of the captured original objective, never additional operations.',
            'answer_prerequisite.arguments may declare explicit offered scientific parameters in the same reply only for operations already requested by the captured objective. Otherwise use []; original validated scientific parameters remain captured. Do not add operations to answer a species prerequisite.',
        ] if (public.get('dialogue', {}).get('pending_prerequisite') or {}).get('field', 'species') == 'species'
              and public.get('dialogue', {}).get('pending_prerequisite') is not None else []) + ([
            'dialogue.pending_prerequisite describes one captured scientific request awaiting the exact required parameters listed there. Its original objective, inputs and operation scope remain fixed; pending context does not establish scientific compatibility or execution authority.',
            'Choose answer_parameters only when this utterance explicitly supplies or corrects parameters for that same pending request. Use its exact offered pending handle and canonical offered tool/argument pairs with verbatim numeric or exact_string literals and uniquely locating exact evidence quotes. Partial answers are allowed; never invent the remaining values or add operations.',
            'An answer continues only the original explicit execution request through existing scientific validation. Unrelated messages, explanations, uncertainty, cancellation and new objectives must use the ordinary applicable decision, never answer_parameters. Clarify conflicting or ambiguous parameter associations.',
        ] if (public.get('dialogue', {}).get('pending_prerequisite') or {}).get('field') == 'parameters' else []) + ([
            'dialogue.pending_prerequisite describes the exact original scientific objective awaiting the offered required_roles. Captured sources retain their exact order and identity. Explicitly selected newly registered files may be appended only to fill the offered missing roles; previously admitted roles and captured files remain fixed. Only the explicitly described declaration corrections and resource choices are allowed; this context grants no scientific qualification.',

            'For a pending fastq declaration use answer_roles with fastq, empty roles and the exact @roles handle. Preserve the original library and members; supply only explicit additions. Missing values stay null.',
            'Choose answer_roles only when this utterance answers that specific pending question. Use its exact pending handle and offered tool/port/source associations. A partial answer may provide only some missing roles or an exact offered scientific argument; omit unresolved roles instead of assigning by order or guessing. Previously admitted roles cannot be reassigned. Positional references apply only to the captured ordered selection and an explicit role association in this reply.',
            'Contradictory or ambiguous associations require clarify/ambiguous_subject. Unrelated tasks, explanations, uncertainty and cancellation use their ordinary decisions, never answer_roles. An answer continues only the original task; do not add operations or claim resource authority from conversational assertions.',
        ] if (public.get('dialogue', {}).get('pending_prerequisite') or {}).get('field') in {'roles', 'fastq', 'bam', 'fragments', 'annotation', 'epizoo'} else [])},
        sort_keys=True, separators=(',', ':'),
        ensure_ascii=not bool(public.get('dialogue', {}).get('selected_sources')))
    response_schema = decision_schema()
    log_operation_event(logger, 'interpreter.preparation_completed', context=diagnostic_context,
                        prompt=prompt, schema=response_schema)
    response = model.complete(prompt=prompt, response_schema=response_schema)
    decision = parse_decision(response, utterance=utterance)
    log_operation_event(logger, 'interpreter.parse_completed', context=diagnostic_context,
                        kind=type(decision).__name__,
                        reason=decision.reason if isinstance(decision, Clarify) and decision.reason in REASONS else None)
    return decision

def is_exact_string_argument(argument_spec):
    """Reviewed column declarations identified by existing Registry semantics.

    Port uniqueness is checked separately by the shared request-input resolver.
    Scientific parameter status alone never authorizes a free-text operand.
    """
    from agent.orchestration.registry import PlanningProvenanceRole, PlanningSourceEligibility
    planning = argument_spec.planning
    return (str in argument_spec.accepted_types
        and set(argument_spec.accepted_types) <= {str, type(None)} and not argument_spec.choices
        and planning is not None and planning.scientific_parameter
        and not planning.conversational_choice and not planning.accepted_artifact_kinds
        and planning.source_eligibility == PlanningSourceEligibility.REQUEST_INPUT_ONLY
        and (planning.conversational_exact_string or planning.provenance_role in
            (PlanningProvenanceRole.REFERENCE, PlanningProvenanceRole.GROUND_TRUTH)
             or planning.observation_source is not None))


def is_json_object_argument(argument_spec):
    planning = argument_spec.planning
    return (argument_spec.accepted_types == (dict,) and not argument_spec.choices
        and planning is not None and planning.conversational_json_object)


def admit_scientific_argument(declaration, utterance, argument_spec, *, observation_fields=None,
                              _legacy_numeric_boundary=False):
    """Check one semantic declaration's literal identity, then delegate its science."""
    if not isinstance(declaration, ScopedArgument):
        raise IntentError('invalid_decision')
    literal, start, end = declaration.literal, declaration.start, declaration.end
    if (type(utterance) is not str or end > len(utterance)
            or utterance[start:end] != literal):
        raise IntentError('ungrounded_operand')
    if is_json_object_argument(argument_spec):
        # One exact utterance span remains the declaration. JSON is data, never
        # evaluated code; the selected scientific owner reviews native keys.
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError('Duplicate JSON key')
                result[key] = value
            return result
        def invalid_constant(value):
            raise ValueError('Non-finite JSON value')
        try:
            if isinstance(declaration, StructuredArgument):
                allowed = dict(argument_spec.planning.conversational_members)
                if any(m.name not in allowed for m in declaration.members):
                    raise ValueError('Unregistered structured member')
                value = {}
                for member in declaration.members:
                    if utterance[member.start:member.end] != member.literal:
                        raise IntentError('ungrounded_operand')
                    if member.kind == 'json':
                        value[member.name] = json.loads(member.literal, object_pairs_hook=pairs,
                                                       parse_constant=invalid_constant)
                    else:
                        # Reuse scalar lexical admission. These temporary specs
                        # add no scientific choices, defaults or range rules.
                        numeric = member.kind == 'number'
                        types = ((int,) if member.literal.lstrip('+-').isdigit() else (float,)) if numeric else (str,)
                        scalar = replace(argument_spec, accepted_types=types, planning=replace(
                            argument_spec.planning, conversational_json_object=False,
                            conversational_members=(), conversational_exact_string=not numeric))
                        value[member.name] = admit_scientific_argument(ScopedArgument(
                            declaration.tool, declaration.argument, member.literal, member.start, member.end),
                            utterance, scalar)
            else:
                value = json.loads(literal, object_pairs_hook=pairs, parse_constant=invalid_constant)
            if type(value) is not dict:
                raise ValueError('Expected a JSON object')
            freeze_json_mapping(value, 'scientific argument')
            argument_spec.validate(declaration.argument, value)
        except (ValueError, TypeError, RecursionError) as exc:
            raise IntentError('invalid_parameter_value') from exc
        return value
    if isinstance(declaration, ScalarChoiceArgument):
        if (start and utterance[start - 1].isalnum()
                or end < len(utterance) and utterance[end].isalnum()):
            raise IntentError('ungrounded_operand')
        column = argument_spec.planning is not None and argument_spec.planning.observation_source is not None
        if isinstance(declaration, SourceFieldArgument) and not column:
            raise IntentError('invalid_reference', diagnostic_reason='unoffered_source_field',
                              operand='parameter.field')
        if column:
            if isinstance(declaration, SourceFieldArgument) and declaration.mode == 'literal':
                if observation_fields is None or declaration.value != declaration.literal:
                    raise IntentError('invalid_reference', diagnostic_reason='source_field_literal_mismatch',
                                      operand='parameter.field')
                # An explicitly supplied exact name may lie outside the bounded
                # metadata offer. Never manufacture that value from a phrase.
                return admit_scientific_argument(ScopedArgument(declaration.tool, declaration.argument,
                    declaration.literal, declaration.start, declaration.end), utterance, argument_spec)
            if observation_fields is None or declaration.value not in observation_fields:
                raise IntentError('invalid_reference', diagnostic_reason='unoffered_source_field',
                                  operand='parameter.field')
        elif (not argument_spec.planning or not argument_spec.planning.conversational_choice
                or str not in argument_spec.accepted_types or not argument_spec.choices):
            raise IntentError('unsupported_intent')
        try:
            argument_spec.validate(declaration.argument, declaration.value)
        except (ValueError, TypeError) as exc:
            raise IntentError('invalid_parameter_value') from exc
        return declaration.value
    if is_exact_string_argument(argument_spec):
        if not literal.strip():
            raise IntentError('invalid_parameter_value')
        # Quotation delimiters belong to the utterance, not the column value.
        # These are lexical span checks, not natural-language interpretation.
        quotes = {'"': '"', "'": "'", '`': '`', '“': '”', '‘': '’', '「': '」', '『': '』'}
        if len(literal) >= 2 and quotes.get(literal[0]) == literal[-1]:
            raise IntentError('invalid_parameter_value')
        before = utterance[start - 1] if start else ''
        after = utterance[end] if end < len(utterance) else ''
        if (before in quotes or after in quotes.values()) and quotes.get(before) != after:
            raise IntentError('ungrounded_operand')
        if before not in quotes:
            # A directly paired span already identifies the complete quoted
            # value, including inside quoted prose. Otherwise reject an interior
            # fragment of quoted content. Scan families independently so an
            # unrelated unmatched delimiter cannot conceal that content.
            for opener, closer in quotes.items():
                opening = None
                for index, character in enumerate(utterance):
                    if opening is None:
                        if character != opener:
                            continue
                        if opener == "'" and index and utterance[index - 1].isalnum():
                            continue
                        opening = index
                    elif character == closer:
                        if opening < start and end <= index:
                            raise IntentError('ungrounded_operand')
                        opening = None
                if opening is not None:
                    raise IntentError('ungrounded_operand')
        sentence_period = (after == '.'
            and (end + 1 == len(utterance) or utterance[end + 1].isspace()))
        if (before.isalnum() or before in ('.', '_', '+', '-', '/')
                or after.isalnum() or after in ('_', '+', '-', '/')
                or after == '.' and not sentence_period
                or any(c and unicodedata.category(c).startswith('M') for c in (before, after))):
            raise IntentError('ungrounded_operand')
        try:
            argument_spec.validate(declaration.argument, literal)
        except (ValueError, TypeError) as exc:
            raise IntentError('invalid_parameter_value') from exc
        return literal
    if not any(t in argument_spec.accepted_types for t in (int, float)):
        raise IntentError('unsupported_intent')
    # These are numeric-token boundaries, not a classifier for scientific intent
    # or a language parser. The Interpreter owns both assignment and association.
    if not _numeric_token_span(utterance, start, end, legacy_modifier=_legacy_numeric_boundary):
        raise IntentError('ungrounded_operand')
    try:
        if any(c not in '+-./0123456789eE' for c in literal):
            raise InvalidOperation
        types = argument_spec.accepted_types
        if float in types:
            number = Decimal(literal)
            if not number.is_finite():
                raise InvalidOperation
            value = float(number)
            if not Decimal(str(value)).is_finite():
                raise InvalidOperation
        elif int in types and literal.lstrip('+-').isdigit() and literal.lstrip('+-'):
            value = int(literal)
        elif str in types:
            # Preserve the exact lexical representation. The registered owner,
            # including its exact-ratio grammar, owns validity and range checks.
            if '/' in literal:
                Fraction(literal)
            elif not Decimal(literal).is_finite():
                raise InvalidOperation
            value = literal
        else:
            raise InvalidOperation
        argument_spec.validate(declaration.argument, value)
    except (InvalidOperation, ValueError, TypeError, OverflowError, ZeroDivisionError) as exc:
        raise IntentError('invalid_parameter_value') from exc
    return value


def admit_leiden_resolution(declaration, utterance, argument_spec):
    """Keep the historical narrow admission surface on the shared binder."""
    if not isinstance(declaration, LeidenResolution):
        raise IntentError('invalid_decision')
    return admit_scientific_argument(declaration, utterance, argument_spec)


def clauses(utterance):
    if re.search(r"\b(?:not|never|don['’]t|if|unless|maybe|instead|rather|except)\b", utterance, re.I):
        raise IntentError('unsupported_intent')
    return tuple(x.strip().rstrip('.!?').strip() for x in re.split(r'\s+and\s+|;', utterance, flags=re.I) if x.strip())


def relation_from_language(utterance):
    """Admit a small relation vocabulary; do not resolve arbitrary history prose."""
    text = utterance.lower()
    if re.search(r'\bparent(?: version)?\b', text): return 'parent'
    if re.search(r'\b(?:previously active|previous.active)\b|version i was using before', text): return 'previous_active'
    if re.search(r'\b(?:previous|earlier) version\b|\bgo back one version\b', text): return 'previous'
    return 'current'


def resolve_relation(relation, snapshot, utterance):
    language = relation_from_language(utterance)
    if relation != language: raise IntentError('ambiguous_revision')
    mapping = snapshot['relations']
    if relation == 'previous':
        candidates = {mapping.get(k) for k in ('parent', 'previous_active')} - {None}
        if len(candidates) != 1: raise IntentError('ambiguous_revision')
        return candidates.pop()
    if relation not in mapping: raise IntentError('unavailable_context')
    return mapping[relation]


def admit_delta(delta, utterance, offered, argument_specs, focus=None):
    if delta.parameter not in offered: raise IntentError('ambiguous_parameter')
    segments = clauses(utterance)
    if delta.evidence not in segments or segments.count(delta.evidence) != 1:
        raise IntentError('ungrounded_operand')
    number = r'[+-]?(?:0|[1-9][0-9]*)(?:\.[0-9]+|/[1-9][0-9]*)?'
    match = re.fullmatch(r'(set|try|increase|decrease)\s+(.+?)\s*(to|by|=)\s*(' + number + r')', delta.evidence, re.I)
    if match is None: raise IntentError('missing_parameter_value')
    verb, label, connector, literal = match.groups()
    operation = {'set':'set', 'try':'set', 'increase':'add', 'decrease':'subtract'}[verb.lower()]
    if (operation != delta.operation or literal != delta.literal or len(literal) > 80
            or connector.lower() not in (('to','=') if operation == 'set' else ('by',))):
        raise IntentError('ungrounded_operand')
    label = re.sub(r'\s+', ' ', label.lower().replace('_', ' ')).removeprefix('the ')
    if label == 'it':
        if focus != delta.parameter: raise IntentError('ambiguous_parameter')
    elif label not in ALIASES.get(delta.parameter, ()):
        raise IntentError('ambiguous_parameter')
    operand = Fraction(literal)
    if operation != 'set' and operand < 0: raise IntentError('ungrounded_operand')
    spec = argument_specs[delta.parameter]
    current = offered[delta.parameter]
    if operation != 'set' and current is None: raise IntentError('invalid_parameter_value')
    value = operand if operation == 'set' else Fraction(current) + (operand if operation == 'add' else -operand)
    # Numeric representation follows admitted ArgumentSpec types; its owner
    # remains responsible for integer/rational bounds and optional semantics.
    if value.denominator == 1:
        result = value.numerator
    elif str in spec.accepted_types:
        result = f'{value.numerator}/{value.denominator}'
    else:
        raise IntentError('invalid_parameter_value')
    try: spec.validate(delta.parameter, result)
    except ValueError as exc: raise IntentError('invalid_parameter_value') from exc
    return result
