"""Bounded turn interpretation and exact, utterance-grounded scalar admission."""
from dataclasses import dataclass, field, replace
from collections.abc import Mapping
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .raw_handoff import FastqDeclaration, BamDeclaration, FragmentsDeclaration, AnnotationDeclaration
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import json
import logging
import math
import re
import sys
import unicodedata
from agent.schemas.orchestration import freeze_json_mapping, _serialize
from agent.orchestration.planning_diagnostics import log_operation_event

logger = logging.getLogger(__name__)

# Composed scientific requests can need more than eight explicit declarations.
MAX_SCIENTIFIC_ARGUMENTS = 32
PARAMETER_CLARIFICATION_REASONS = (
    'missing_parameter_value', 'ambiguous_parameter', 'invalid_parameter_value',
    'ungrounded_operand', 'conflicting_scientific_parameter')


def protocol_integer(value, *, exact=None):
    """JSON integer semantics for protocol metadata, never scientific operands."""
    if type(value) is int:
        return value
    if type(value) is float:
        if exact is not None:
            # The wire token can contain precision already lost by float (or
            # underflow to zero). Only protocol fields use this Decimal view.
            if type(exact) is Decimal and exact.is_finite() and exact == exact.to_integral_value():
                # Match the existing JSON integer decoder's digit limit before
                # expanding exponent notation into a potentially huge integer.
                digits = sys.get_int_max_str_digits()
                if not exact.is_zero() and digits and exact.adjusted() >= digits:
                    raise ValueError('Protocol integer exceeds the JSON decoder bound.')
                return int(exact)
        elif math.isfinite(value) and value.is_integer():
            return int(value)
    raise ValueError('Expected an exact finite protocol integer.')


def _validate_choice_value(literal, value, limit=80):
    if (not literal.strip() or type(value) is not str or not value.strip()
            or limit is not None and len(value) > limit):
        raise ValueError('Invalid scalar choice declaration.')


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
class ScalarChoiceArgument(ScopedArgument):
    """An LLM-normalized reviewed choice with its verbatim declaration span."""

    value: str

    def __post_init__(self):
        super().__post_init__()
        _validate_choice_value(self.literal, self.value)


@dataclass(frozen=True)
class SourceFieldArgument(ScalarChoiceArgument):
    """An explicit field and source selection for a Registry-owned dataset port."""

    source: str
    mode: str

    def __post_init__(self):
        ScopedArgument.__post_init__(self)
        # Offered field names keep their exact captured domain. Literal fields
        # and unrelated normalized scalar choices retain the ordinary bound.
        _validate_choice_value(self.literal, self.value, None if self.mode == 'offered' else 80)
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
                or self.kind not in ('string', 'number', 'boolean', 'json')):
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


@dataclass(frozen=True, kw_only=True)
class SelectedStepScope:
    """Application-selected member of a captured plan, never a provider guess."""
    step_id: str

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.step_id, str) or not self.step_id:
            raise ValueError('Invalid selected step.')


@dataclass(frozen=True)
class SelectedArgument(SelectedStepScope, ScopedArgument): pass
@dataclass(frozen=True)
class SelectedChoiceArgument(SelectedStepScope, ScalarChoiceArgument): pass
@dataclass(frozen=True)
class SelectedFieldArgument(SelectedStepScope, SourceFieldArgument): pass
@dataclass(frozen=True)
class SelectedStructuredArgument(SelectedStepScope, StructuredArgument): pass
@dataclass(frozen=True)
class SelectedSourceArgument(SelectedStepScope, SourceScopedArgument): pass
@dataclass(frozen=True)
class SelectedSourceChoiceArgument(SelectedStepScope, SourceScopedChoiceArgument): pass
@dataclass(frozen=True)
class SelectedSourceStructuredArgument(SelectedStepScope, SourceScopedStructuredArgument): pass
@dataclass(frozen=True)
class SelectedNativeArgument(SelectedStepScope, NativeArgument): pass
@dataclass(frozen=True)
class SelectedNativeChoiceArgument(SelectedStepScope, NativeChoiceArgument): pass
@dataclass(frozen=True)
class SelectedNativeStructuredArgument(SelectedStepScope, NativeStructuredArgument): pass
@dataclass(frozen=True)
class SelectedNativeSourceArgument(SelectedStepScope, NativeSourceArgument): pass
@dataclass(frozen=True)
class SelectedNativeSourceChoiceArgument(SelectedStepScope, NativeSourceChoiceArgument): pass
@dataclass(frozen=True)
class SelectedNativeSourceStructuredArgument(SelectedStepScope, NativeSourceStructuredArgument): pass


@dataclass(frozen=True)
class ExplicitArgument:
    """Researcher-entered value, with no invented utterance or text spans."""

    tool: str
    argument: str
    value: object
    native_function: str | None = None
    step_id: str | None = None
    port: str | None = None
    source: str | None = None
    resource_id: str | None = None
    origin: str = 'structured'

    def __post_init__(self):
        if (self.origin != 'structured' or type(self.tool) is not str or type(self.argument) is not str
                or not self.tool.isidentifier() or not self.argument.isidentifier()):
            raise ValueError('Invalid explicit parameter identity.')
        for value in (self.native_function, self.step_id):
            if value is not None and (type(value) is not str or not value.strip()):
                raise ValueError('Invalid explicit parameter scope.')
        if self.resource_id is not None and (type(self.resource_id) is not str
                or re.fullmatch(r'local-[0-9a-f]{64}', self.resource_id) is None):
            raise ValueError('Invalid explicit source identity.')
        if self.port is None and (self.source is not None or self.resource_id is not None):
            raise ValueError('Incomplete explicit source scope.')
        if self.port is not None:
            if self.source is not None:
                InputRole(self.tool, self.port, self.source)
            elif self.resource_id is None or type(self.port) is not str or not self.port.isidentifier():
                raise ValueError('Incomplete explicit source scope.')
        # Reuse the ordinary bounded JSON representation; native owners validate science.
        freeze_json_mapping({'value': self.value}, 'explicit parameter')


def argument_identity(item):
    return (item.tool, item.argument, getattr(item, 'port', None), getattr(item, 'source', None), getattr(item, 'native_function', None), getattr(item, 'step_id', None), getattr(item, 'resource_id', None), isinstance(item, ExplicitArgument))


def _validate_arguments(arguments):
    if (type(arguments) is not tuple or len(arguments) > MAX_SCIENTIFIC_ARGUMENTS
            or any(not isinstance(item, (ScopedArgument, ExplicitArgument)) for item in arguments)):
        raise ValueError('Invalid bounded scientific arguments.')
    identities = [argument_identity(item) for item in arguments]
    if len(identities) != len(set(identities)):
        raise ValueError('Duplicate scoped scientific argument.')


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
    if isinstance(value, Mapping) and value.get('kind') == 'marker_activity_umap':
        _shape(_serialize(value), ('kind', 'coordinates', 'activity'))
        if any(type(value[k]) is not str or re.fullmatch(r'g[0-9]+', value[k]) is None
                for k in ('coordinates', 'activity')):
            raise ValueError('Select exact accepted UMAP and native activity sources.')
        return
    if isinstance(value, Mapping) and value.get('kind') == 'qc':
        _shape(_serialize(value), ('kind', 'source'))
        if type(value['source']) is not str or re.fullmatch(r'g[0-9]+', value['source']) is None:
            raise ValueError('Select an exact accepted QC source.')
        return
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
        detail = value['detail'] = validate_scientific_detail(detail, value['right'])
        if (detail['section'] not in ('category_counts', 'category_association')
                or any(type(detail[k]) is not str or not detail[k].startswith(('source:', 'result:'))
                       or not detail[k].split(':', 1)[1] for k in ('field', 'other_field'))):
            raise ValueError('Select both exact comparison fields.')
    if value['left'] == value['right'] and (detail is None or detail['field'] == detail['other_field']):
        raise ValueError('A same-result comparison requires two distinct exact fields.')
    return value


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
            or any(k not in ('explanation', 'comparison', 'visualization', 'files_only') for k in decision.delivery)
            or 'files_only' in decision.delivery and len(decision.delivery) != 1
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
        comparison = _validate_comparison(decision.comparison)
        object.__setattr__(decision, 'comparison', freeze_json_mapping(comparison, 'comparison'))
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
        _normalize_research_intent(self)
        from .raw_handoff import target as declaration_target
        if self.epizoo is not None:
            from .epizoo_handoff import EpiZooDeclaration
            from .raw_handoff import validate_arguments
            if (type(self.epizoo) is not EpiZooDeclaration or self.epizoo.operation is None
                    or any(v is not None for v in (self.fastq, self.bam, self.fragments, self.annotation))
                    or self.target != declaration_target('epizoo', self.epizoo) or self.required_roles or self.roles
                    or self.context != 'none'):
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
                    or self.target != declaration_target('annotation', self.annotation) or self.required_roles or self.roles
                    or self.context not in ('none', 'current')):
                raise ValueError('Invalid annotation task declaration.')
            _validate_arguments(self.arguments)
            validate_arguments(self.arguments, 'annotation')
            return
        if self.fragments is not None:
            from .raw_handoff import FragmentsDeclaration, validate_arguments
            if (type(self.fragments) is not FragmentsDeclaration or self.fastq is not None or self.bam is not None
                    or self.target != declaration_target('fragments', self.fragments) or self.required_roles or self.roles
                    or self.context != 'none'):
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
            if (type(self.fastq) is not FastqDeclaration or self.target != declaration_target('fastq', self.fastq)
                    or self.required_roles or self.roles or self.context != 'none'):
                raise ValueError('Invalid FASTQ task declaration.')
            _validate_arguments(self.arguments)
            from .raw_handoff import validate_arguments
            validate_arguments(self.arguments)
            return
        if self.bam is not None:
            if self.target != declaration_target('bam', self.bam) or self.required_roles or self.context != 'none':
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
    target: str
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
    missing_parameters: tuple = ()

    def __post_init__(self):
        _normalize_research_intent(self)
        if (len(self.missing_parameters)>8
                or any(not isinstance(k, (str, dict)) for k in self.missing_parameters)):
            raise ValueError('Invalid missing scientific parameter identities.')
        if self.context not in ('none', 'current'):
            raise ValueError('Invalid bounded execution context.')
        _validate_arguments(self.arguments)
        _validate_roles(self.roles)


@dataclass(frozen=True)
class ExecuteCandidate:
    candidate: str
    evidence: str
    arguments: tuple = ()

    def __post_init__(self):
        if (type(self.candidate) is not str or not 0 < len(self.candidate) <= 128
                or type(self.evidence) is not str or not 0 < len(self.evidence) <= 4096):
            raise ValueError('Invalid explicit candidate selection.')
        _validate_arguments(self.arguments)


@dataclass(frozen=True)
class SpeciesAnswer:
    """A declaration answering one offered, exact pending clarification."""

    pending: str
    species: str
    arguments: tuple[ScopedArgument, ...] = ()

    def __post_init__(self):
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
        if self.pending != '@resources' or type(self.choice) is not str or not self.choice.startswith(('resource.', 'spp.')):
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


ANSWER_INTENTS = ('scientific', 'guidance')

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
            object.__setattr__(self, 'detail', validate_scientific_detail(self.detail, self.comparison))
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
    if detail['section'] not in DetailRequest.CATEGORICAL_SECTIONS:
        raise ValueError('Unsupported scientific detail request.')
    single = dict(detail)
    if detail['section'] == 'category_counts' and detail['other_field'] is not None:
        if comparison is None:
            raise ValueError('Independent second counts require a second accepted target.')
        single['other_field'] = None  # Each owner count read still has one field.
    normalized = DetailRequest(**single)
    return dict(detail, offset=normalized.offset, limit=normalized.limit)


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
    scientific: ScientificQuestion | None = None
    guidance: GuidanceQuestion | None = None

    def __post_init__(self):
        if self.intent not in ANSWER_INTENTS:
            raise ValueError('Invalid bounded answer request.')
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


def _shape(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise IntentError('invalid_decision')


def _scientific_arguments(value):
    if type(value) is not list or len(value) > MAX_SCIENTIFIC_ARGUMENTS:
        raise IntentError('invalid_decision')
    result = []
    for item in value:
        if item.get('origin') == 'structured':
            result.append(ExplicitArgument(**item))
            continue
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
        cls, keys = variants[cls], keys + ('native_function',)
    if type(value) is dict and 'step_id' in value:
        variants = {ScopedArgument: SelectedArgument, ScalarChoiceArgument: SelectedChoiceArgument,
            SourceFieldArgument: SelectedFieldArgument,
            StructuredArgument: SelectedStructuredArgument, SourceScopedArgument: SelectedSourceArgument,
            SourceScopedChoiceArgument: SelectedSourceChoiceArgument, SourceScopedStructuredArgument: SelectedSourceStructuredArgument,
            NativeArgument: SelectedNativeArgument, NativeChoiceArgument: SelectedNativeChoiceArgument,
            NativeStructuredArgument: SelectedNativeStructuredArgument, NativeSourceArgument: SelectedNativeSourceArgument,
            NativeSourceChoiceArgument: SelectedNativeSourceChoiceArgument, NativeSourceStructuredArgument: SelectedNativeSourceStructuredArgument}
        if cls not in variants:
            raise IntentError('invalid_decision')
        cls, keys = variants[cls], keys + ('step_id',)
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


def _word_continuation(character):
    # Unspaced scripts do not use Latin identifier boundaries. Keep digits,
    # combining marks and identifier letters protected (including non-ASCII).
    return (character.isalnum() and unicodedata.category(character) != 'Lo'
            or unicodedata.category(character).startswith('M'))


def _numeric_token_span(text, start, end):
    """Lexical numeric identity shared by quote location and scalar admission."""
    alnum = _word_continuation
    if start and (alnum(text[start - 1]) or text[start - 1] in '.+-/_%'):
        return False
    if end < len(text):
        following = text[end]
        modifier = (following == '-'
                    and end + 1 < len(text) and text[end + 1].isalpha())
        sentence_period = (following == '.'
            and (end + 1 == len(text) or not text[end + 1].isdigit()))
        if (alnum(following) or following in '+-/_%' and not modifier
                or following == '.' and not sentence_period):
            return False
    return True


def _ground_declaration(value, utterance, *, _string_literal=False,
                        _kind=None, _choices=()):
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
                literal=member['literal'], evidence=member['evidence']), utterance, _kind=member['kind'])
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
    numeric = (re.fullmatch(
        r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?:/[+-]?\d+)?', literal) is not None)
    origin = unique(utterance, evidence, numeric=numeric and evidence == literal)
    number_pattern = r'[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?(?:/[+-]?\d+)?'
    if quoted and (numeric or _kind == 'number'):
        tokens = [m for m in re.finditer(number_pattern, evidence)
                  if _numeric_token_span(evidence, *m.span())]
        if len(tokens) > 1:
            raise IntentError('ambiguous_parameter')
    try:
        offset = unique(evidence, literal, numeric=numeric, string=_string_literal)
    except IntentError as exc:
        if exc.reason != 'ungrounded_operand' or not quoted:
            raise
        # Recover only a single mechanically identifiable value inside the exact
        # association quote. Never search the whole request for a replacement.
        candidates = []
        if numeric or _kind == 'number':
            for match in re.finditer(number_pattern, evidence):
                if _numeric_token_span(evidence, *match.span()):
                    candidates.append((match.start(), match.group()))
        elif _string_literal or _kind == 'string':
            for match in re.finditer(r'"([^"\n]+)"|“([^”\n]+)”|‘([^’\n]+)’|`([^`\n]+)`', evidence):
                group = next(i for i in range(1, 5) if match.group(i) is not None)
                candidates.append((match.start(group), match.group(group)))
        elif _choices:
            for choice in _choices:
                if not isinstance(choice, str):
                    continue
                for match in re.finditer(re.escape(choice), evidence):
                    if (not match.start() or not _word_continuation(evidence[match.start()-1])) and (
                            match.end() == len(evidence) or not _word_continuation(evidence[match.end()])):
                        candidates.append((match.start(), choice))
        elif _kind == 'boolean':
            for token in BOOLEAN_LITERALS:
                for match in re.finditer(re.escape(token), evidence, re.I):
                    if (not match.start() or not _word_continuation(evidence[match.start()-1])) and (
                            match.end() == len(evidence) or not _word_continuation(evidence[match.end()])):
                        candidates.append((match.start(), match.group()))
        elif _kind == 'json':
            decoder = json.JSONDecoder()
            for index, char in enumerate(evidence):
                if char not in '[{tf' or index and _word_continuation(evidence[index-1]):
                    continue
                try:
                    parsed, size = decoder.raw_decode(evidence[index:])
                except ValueError:
                    continue
                if type(parsed) in (bool, list, dict):
                    candidates.append((index, evidence[index:index+size]))
            # Nested values are part of the enclosing value, not alternatives.
            candidates = [(i, v) for i, v in candidates if not any(j < i < j+len(w) for j,w in candidates)]
        if len(candidates) != 1:
            raise IntentError('ambiguous_parameter' if candidates else 'ungrounded_operand') from exc
        offset, literal = candidates[0]
    start = origin + offset
    result = {key: value[key] for key in keys} | dict(start=start, end=start + len(literal))
    result['literal'] = literal
    if 'value' in result and literal in _choices:
        result['value'] = literal
    if 'members' in result:
        result['members'] = members()
    return result


def parse_decision(raw, *, utterance=None, authoritative=()):
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
        precise = json.loads(raw, parse_float=Decimal)
        if protocol_integer(envelope['turn_schema_version'], exact=precise['turn_schema_version']) != 1:
            raise IntentError('invalid_decision')
        value = envelope['decision']
        # Normalize protocol pagination only, before the ordinary typed owners.
        # Scientific kwargs, counts, thresholds and span coordinates retain the
        # existing decoder and validation semantics.
        def pagination(detail, exact_detail):
            if isinstance(detail, dict):
                for key in ('offset', 'limit'):
                    if key in detail:
                        detail[key] = protocol_integer(detail[key], exact=exact_detail[key])
        pagination(value.get('detail'), precise['decision'].get('detail'))
        if isinstance(value.get('comparison'), dict):
            pagination(value['comparison'].get('detail'), precise['decision']['comparison'].get('detail'))
        if utterance is not None:
            if type(utterance) is not str or not 0 < len(utterance) <= 4096:
                raise IntentError('invalid_decision')
            failure = None
            failed_parameters = []
            def ground(item):
                nonlocal failure
                try:
                    if item.get('step_id') is not None:
                        raise IntentError('invalid_decision')  # Only a captured plan selection may add this.
                    from agent.orchestration.registry import build_default_tool_registry
                    from .dialogue_execution import _parameter_input
                    try:
                        _, spec = _parameter_input(build_default_tool_registry(), item['tool'], item['argument'])
                    except IntentError:
                        spec = None  # Structural parsing does not authorize an unknown tool.
                    return _ground_declaration(item, utterance,
                        _string_literal=spec is not None and is_exact_string_argument(spec),
                        _kind=('boolean' if spec.accepted_types == (bool,) else
                               'json' if is_json_object_argument(spec) else
                               'number' if any(t in spec.accepted_types for t in (int,float)) else None) if spec else None,
                        _choices=() if spec is None else spec.choices)
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
            if 'arguments' in value:
                if type(value['arguments']) is not list or len(value['arguments']) > MAX_SCIENTIFIC_ARGUMENTS:
                    raise IntentError('invalid_decision')
                from .structured_parameters import shadowed_arguments
                value['arguments'] = [ground(item) for item in shadowed_arguments(value['arguments'], authoritative)]
            for field in ('fastq', 'bam', 'fragments'):
                declared = value.get(field)
                if declared is None:
                    continue
                preference = declared.get('resource_preference')
                # The interpreter may associate an explicit researcher selection
                # with an offered handle. The existing resource owner checks its
                # captured membership and scientific component identity.
                if declared.get('resource') is not None and preference is None:
                    raise IntentError('ungrounded_operand')
                if preference is not None:
                    bound = _ground_declaration(dict(tool='run_snapatac2', argument='resource', **preference),
                                                utterance, _string_literal=True)
                    declared['resource_preference'] = {k: bound[k] for k in ('literal', 'start', 'end')}
                declared.setdefault('resource', None)
            parsed = parse_decision(json.dumps(envelope))
            if failure:
                raise DeclarationGroundingError(failure, raw, tuple(dict.fromkeys(failed_parameters)))
            return parsed
        if any(item.get('origin') == 'structured' for item in value.get('arguments', ())):
            raise IntentError('invalid_decision')
        kind = value.get('kind')
        if kind == 'execute_candidate':
            _shape(value, ('kind', 'candidate', 'evidence'))
            return ExecuteCandidate(value['candidate'], value['evidence'])
        if kind == 'answer_prerequisite':
            _shape(value, ('kind', 'pending', 'species') + tuple(
                key for key in ('arguments',) if key in value))
            return SpeciesAnswer(value['pending'], value['species'],
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
                key for key in ('arguments', 'context', 'fastq', 'bam', 'fragments', 'annotation', 'epizoo',
                                'input_results', 'references', 'delivery', 'unfulfilled', 'visualization', 'comparison', 'retain_parameters') if key in value))
            if any(type(value[k]) is not list for k in ('input_results', 'references', 'delivery', 'unfulfilled', 'retain_parameters') if k in value):
                raise IntentError('invalid_decision')
            from .raw_handoff import declaration
            return RequestRoles(value['target'], (() if (value.get('fastq') is not None or value.get('bam') is not None or value.get('fragments') is not None or value.get('annotation') is not None or value.get('epizoo') is not None) and value['required_roles'] == []
                                                    else _role_requirements(value['required_roles'])),
                                _input_roles(value['roles']),
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
            if {'kind', 'target'} <= set(value) <= {'kind', 'target', 'arguments', 'context', 'roles', 'input_results', 'references', 'delivery', 'retain_parameters', 'unfulfilled', 'visualization', 'comparison', 'missing_parameters'}:
                if type(value['target']) is not str:
                    raise IntentError('invalid_decision')
                if any(type(value[k]) is not list for k in ('input_results', 'references', 'delivery', 'retain_parameters', 'unfulfilled', 'missing_parameters') if k in value):
                    raise IntentError('invalid_decision')
                return Execute(value['target'],
                               _scientific_arguments(value.get('arguments', [])),
                               value.get('context', 'none'), _input_roles(value.get('roles', [])),
                               tuple(value.get('references', [])), tuple(value.get('delivery', [])),
                               tuple(RetainedParameter(**item) for item in value.get('retain_parameters', [])),
                               tuple(value.get('unfulfilled', [])), value.get('visualization'), value.get('comparison'),
                               tuple(value.get('missing_parameters', [])), input_results=tuple(value.get('input_results', [])))
            raise IntentError('invalid_decision')
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
            raise IntentError('invalid_decision')
        if kind == 'navigate':
            _shape(value, ('kind', 'relation'))
            if value['relation'] not in RELATIONS: raise IntentError('invalid_decision')
            return Navigate(value['relation'])
        raise IntentError('invalid_decision')
    except (ValueError, TypeError, AttributeError, KeyError, InvalidOperation) as exc:
        if isinstance(exc, IntentError): raise
        raise IntentError('invalid_decision') from exc


def _object(properties):
    return dict(type='object', properties=properties, required=list(properties), additionalProperties=False)


def decision_schema(public=None, *, utterance=None):
    """Shared proposal grammar, specialized to current captured offers.

    With no context this remains the historical structural grammar. Live calls
    always supply context; independent admission still owns scope and freshness.
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
    quote = dict(
        literal=dict(text(4096), description='Verbatim substring of the current utterance, even for a normalized API choice.'),
        evidence=dict(text(4096), description='Unique verbatim utterance clause containing literal once.'))
    definitions['member'] = _object(dict(name=text(), kind=enum(('string', 'number', 'boolean', 'json')), **quote))
    forms = [identity | quote, identity | quote | dict(value=text(80)),
             identity | dict(members=array(ref('member'), 32, 1))]
    variants = []
    for form in forms:
        form = form | dict(native_function=dict(optional(text()), description='Registered function scope; function_name uses the same canonical value. Null only without a function.'))
        variants.extend((_object(form), _object(form | dict(source=text(), port=text()))))
    variants.append(_object(identity | quote | dict(source=text(), mode=enum(('offered', 'literal')), value=text(80))))
    definitions['declaration'] = {'anyOf': variants}
    declarations = array(ref('declaration'), MAX_SCIENTIFIC_ARGUMENTS)
    definitions['role'] = _object(dict(tool=text(), port=text(), source=text()))
    definitions['requirement'] = _object(dict(tool=text(), port=text()))
    definitions['target'] = _object(dict(output=text(), subject=optional(text())))
    from .dialogue_evidence import DetailRequest
    definitions['detail'] = DetailRequest.categorical_schema()
    definitions['two_target_detail'] = DetailRequest.categorical_schema(second_target=True)
    definitions['comparison_detail'] = DetailRequest.categorical_schema(comparison=True)
    definitions['visualization'] = {'anyOf': [
        _object(dict(coordinates=text(), labels=text(), field=text(256))),
        _object(dict(kind=enum(('marker_activity_umap',)), coordinates=text(), activity=text())),
        _object(dict(kind=enum(('qc',)), source=text()))]}
    definitions['comparison'] = _object(dict(left=text(), right=text(), detail=optional(ref('comparison_detail'))))
    definitions['export_field'] = _object(dict(output=text(), field=text(256), destination=text(256)))
    # The Export Owner retains the cross-array maximum; two bounded alternatives
    # encode nonemptiness without a count cross product or a new wire format.
    definitions['export'] = {'anyOf': [
        _object(dict(measurement=text(), observations=array(ref('export_field'), 32, 1),
                     representations=array(ref('export_field'), 32))),
        _object(dict(measurement=text(), observations=array(ref('export_field'), 0),
                     representations=array(ref('export_field'), 32, 1)))]}
    definitions['retained'] = _object(dict(result=text(), argument=text()))
    definitions['delivery_intentions'] = {'anyOf': [
        array(enum(('explanation', 'comparison', 'visualization')), 3),
        array(enum(('files_only',)), 1, 1)]}
    definitions['result_selection'] = array(text(), 32)
    definitions['unfulfilled'] = array(text(512), 8)
    definitions['retained_parameters'] = array(ref('retained'), 8)
    delivery = dict(input_results=ref('result_selection'), references=ref('result_selection'),
        delivery=ref('delivery_intentions'),
        unfulfilled=ref('unfulfilled'), visualization=optional(ref('visualization')),
        comparison=optional(ref('comparison')), retain_parameters=ref('retained_parameters'))
    execution = dict(target=text(), arguments=declarations, roles=array(ref('role'), 8),
                     context=enum(('none', 'current')), **delivery)
    variants = [
        _object(dict(kind=enum(('clarify',)), reason=enum(PARAMETER_CLARIFICATION_REASONS),
                     choices=array(dict(text(256), pattern=r'^[A-Za-z_][A-Za-z_0-9]*\.[A-Za-z_][A-Za-z_0-9]*$'), 8))),
        _object(dict(kind=enum(('clarify',)), reason=enum(r for r in REASONS if r not in PARAMETER_CLARIFICATION_REASONS),
                     choices=array(text(256), 0))),
        _object(dict(kind=enum(('navigate',)), relation=enum(RELATIONS))),
        _object(dict(kind=enum(('answer_scientific',)), target=ref('target'), comparison={'type': 'null'},
                     focus=enum(('question', 'continue')), detail=optional(ref('detail')),
                     visualization=optional(ref('visualization')), export=optional(ref('export')))),
        _object(dict(kind=enum(('answer_scientific',)), target=ref('target'), comparison=ref('target'),
                     focus=enum(('question', 'continue')), detail=optional(ref('two_target_detail')),
                     visualization=optional(ref('visualization')), export=optional(ref('export')))),
        _object(dict(kind=enum(('answer_guidance',)), targets=array(ref('target'), 4), candidate={'type': 'null'})),
        _object(dict(kind=enum(('answer_guidance',)), targets=array(ref('target'), 0), candidate=text())),
        _object(dict(kind=enum(('execute_plan',)), **execution, missing_parameters=array({'anyOf': [text(),
            _object(dict(tool=text(), argument=text(), native_function=optional(text()),
                member=optional(text()), source=optional(text()), port=optional(text()), evidence=optional(text(4096))))]}, 8))),
        _object(dict(kind=enum(('request_roles',)), **execution, required_roles=dict(array(ref('requirement'), 8, 1), description='Complete required role set, including already supplied roles; roles contains its supplied strict subset. Prefer execute_plan for an objective with missing inputs so Preparation derives selected-operation requirements.'))),
        _object(dict(kind=enum(('answer_roles',)), pending=text(), roles=array(ref('role'), 8, 1), arguments=declarations)),
        _object(dict(kind=enum(('answer_roles',)), pending=text(), roles=array(ref('role'), 0),
                     arguments=array(ref('declaration'), MAX_SCIENTIFIC_ARGUMENTS, 1))),
        _object(dict(kind=enum(('execute_candidate',)), candidate=text(), evidence=text(4096))),
        _object(dict(kind=enum(('request_resources',)), target=text(), arguments=declarations,
                     roles=array(ref('role'), 8), context=enum(('none', 'current')),
                     input_results=array(text(), 32),
                     reference_category=optional(enum(('peak_set', 'regulatory_regions'))))),
        _object(dict(kind=enum(('answer_resources',)), pending=enum(('@resources',)),
                     choice=dict(text(), pattern=r'^(resource|spp)\.'), arguments=declarations)),
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
    result = _object(dict(turn_schema_version=dict(type='integer', enum=[1]), decision={'anyOf': variants})) | {'$defs': definitions}
    return (_specialized_declaration_targets(result) if public is None
            else _offered_schema(result, public, utterance=utterance))


def _specialized_declaration_targets(schema, tools=None):
    """Use declaration owners before final schema pruning and budget checks."""
    from copy import deepcopy
    from dataclasses import replace
    from .raw_handoff import empty, target
    defs = schema['$defs']
    variants = []
    for variant in schema['properties']['decision']['anyOf']:
        props = variant['properties']
        field = next((f for f in ('fastq', 'bam', 'fragments', 'epizoo', 'annotation')
                      if f in props and '$ref' in props[f]), None)
        if props['kind']['enum'] != ['request_roles'] or field is None:
            variants.append(variant)
            continue
        cases = [empty(field)]
        if field == 'epizoo':
            operations = next(v['enum'] for v in defs[field]['properties']['operation']['anyOf'] if 'enum' in v)
            cases = [replace(empty(field), operation=op) for op in operations]
        for declared in cases:
            owned = target(field, declared)
            if tools is not None and owned not in tools:
                continue
            specialized = deepcopy(variant)
            specialized['properties']['target'] = dict(type='string', enum=[owned])
            if field == 'epizoo':
                specialized['properties'][field] = deepcopy(defs[field])
                specialized['properties'][field]['properties']['operation'] = dict(type='string', enum=[declared.operation])
            variants.append(specialized)
    schema['properties']['decision']['anyOf'] = variants
    return schema


def parameter_choice_offers(public):
    """Only a captured pending question offers parameter clarification identities."""
    pending = public.get('dialogue', {}).get('pending_prerequisite') or {}
    if pending.get('field') != 'parameters':
        return ()
    offered = []
    for row in pending.get('missing', ()):
        if isinstance(row, Mapping):
            tool, argument = row.get('tool_name'), row.get('argument')
            if tool and argument:
                offered.append(tool + '.' + argument)
        elif isinstance(row, str) and ('.' in row or pending.get('tool')):
            offered.append(row if '.' in row else pending['tool'] + '.' + row)
    return tuple(dict.fromkeys(offered))


def _offered_schema(schema, public, *, utterance=None):
    """Specialize shared definitions, not a Registry parameter cross product.

    Empty arrays/nulls express absence; unavailable decisions are removed. Shared
    enum definitions avoid multiplying provider enum budgets across references.
    Association, source integrity, task/version and literal grounding stay with
    admission, including when a provider ignores this schema.
    """
    d = public.get('dialogue', {})
    defs = schema['$defs']
    projection = _interpretation_public(public, utterance or '').get('dialogue', {})
    metadata = projection.get('capabilities', {})
    d = d | {key: projection.get(key, []) for key in d if key.endswith('_resources')}
    def finite(values, *, enum=True):
        if enum and len(values) <= 200:
            return dict(type='string', enum=values)
        # Equal-length groups make $ strict even for a final newline, without
        # lookarounds outside common provider regex grammars. No offer is lost.
        return {'anyOf': [dict(type='string', minLength=n, maxLength=n,
            pattern=r'^(?:' + '|'.join(re.escape(v) for v in values if len(v) == n) + r')$')
            for n in sorted({len(v) for v in values})]}
    def exact(name, values):
        values = list(dict.fromkeys(values))
        if not values:
            return {'type': 'null'}
        defs[name] = finite(values)
        return {'$ref': '#/$defs/' + name}
    def nullable(value):
        return value if value == {'type': 'null'} else {'anyOf': [value, {'type': 'null'}]}
    def items(value, limit):
        return dict(type='array', items=value, maxItems=0 if value == {'type': 'null'} else limit)
    sources = [r['source'] for r in d.get('selected_sources', ())]
    source = exact('offered_source', sources)
    tools = list(d.get('tools', d.get('capabilities', {})))
    tool = exact('offered_tool', tools)
    ports = exact('offered_port', [r['port'] for r in d.get('input_roles', ())])
    parameter_rows = [row | {'argument': name} for cap in metadata.values()
                      for name, row in cap.get('parameters', {}).items()]
    argument = exact('offered_argument', [r['argument'] for r in parameter_rows])
    functions = exact('offered_function', [fn for cap in metadata.values()
                      for fn in cap.get('functions', {})])
    expanded_functions = exact('offered_parameter_function', [fn for cap in metadata.values()
                      for fn in cap.get('function_parameters', {})])
    member = exact('offered_member', [name for row in parameter_rows for name in row.get('members', {})])
    value = exact('offered_value', [v for row in parameter_rows for v in row.get('choices', ())])
    defs['member']['properties']['name'] = member
    outputs = [r['handle'] for r in d.get('outputs', ())]
    groups = [r['handle'] for r in d.get('result_groups', ())]
    targets = outputs + groups + [k for k,v in d.get('result_referents', {}).items() if len(v.get('targets', ())) == 1]
    predecessor = d.get('predecessor') or {}
    if predecessor.get('target'):
        targets.append('@focus')
    if predecessor.get('previous'):
        targets.append('@previous')
    target = exact('offered_target', targets)
    result = exact('offered_result', groups)
    measurement = exact('offered_measurement', groups + sources)
    subject_values = [v for row in d.get('outputs', ())
                      for candidate in row.get('subjects', {}).get('candidates', ())
                      for v in [candidate.get('id'), *candidate.get('references', ())] if isinstance(v, str)]
    if (predecessor.get('target') or {}).get('subject') is not None:
        subject_values.extend(('@focus', '@other'))
    defs['target']['properties'].update(output=target, subject=nullable(exact('offered_subject', subject_values)))
    for name in ('role', 'requirement'):
        defs[name]['properties'].update(tool=tool, port=ports)
    defs['role']['properties']['source'] = source
    # Scoped declarations still admit arbitrary *grounded* values and member
    # names. Exact tool/source selections cannot name absent context entities.
    declaration_forms = []
    for form in defs['declaration']['anyOf']:
        props = form['properties']
        props['tool'] = tool
        props['argument'] = argument
        if 'native_function' in props:
            props['native_function'] = dict(nullable(functions), description=props['native_function']['description'])
        if 'members' in props and member == {'type': 'null'}:
            continue
        if 'value' in props and 'mode' not in props:
            if value == {'type': 'null'}:
                continue
            props['value'] = value
        if 'source' in props:
            if not sources:
                continue
            props['source'] = source
        if 'port' in props:
            if ports == {'type': 'null'}:
                continue
            props['port'] = ports
        if 'mode' in props:
            from copy import deepcopy
            axes = {p.get('observation_axis', 'obs') for p in parameter_rows if p.get('observation_source')}
            offered_fields = [f for row in d.get('source_fields', ()) for axis in axes
                for f in (row.get('fields', ()) if axis == 'obs' else row.get('axes', {}).get(axis, {}).get('fields', ()))]
            if offered_fields:
                offered = deepcopy(form)
                offered['properties']['mode'] = dict(type='string', enum=['offered'])
                offered['properties']['value'] = exact('offered_field', offered_fields)
                declaration_forms.append(offered)
            props['mode'] = dict(type='string', enum=['literal'])
        declaration_forms.append(form)
    defs['declaration']['anyOf'] = declaration_forms
    for field in ('fastq', 'bam', 'fragments', 'annotation', 'epizoo'):
        props = defs[field]['properties']
        for key in props:
            if key == 'source' or key.endswith('_source'):
                props[key] = nullable(source)
        if 'members' in props:
            props['members']['items']['properties']['source'] = source
            if not sources:
                props['members']['maxItems'] = 0
        if 'groups_result' in props:
            props['groups_result'] = nullable(exact('offered_annotation_group',
                [r['handle'] for r in d.get('annotation_group_results', ())]))
        if 'genome_resource' in props:
            props['genome_resource'] = nullable(exact('offered_genome_resource',
                [r['handle'] for r in d.get('epizoo_resources', ()) if 'handle' in r]))
        if 'resource' in props:
            props['resource'] = nullable(exact('offered_' + field + '_resource',
                [r['handle'] for r in d.get(field + '_resources', ()) if 'handle' in r]))
    if len(sources) < 2:
        bam = defs['bam']
        bam['properties']['feature_source'] = {'type': 'null'}
        without_companion = _object({key: value for key, value in bam['properties'].items()
                                     if key != 'feature_source'})
        defs['bam'] = {'anyOf': [without_companion, bam]}
    defs['retained']['properties']['result'] = result
    for form in defs['export']['anyOf']:
        form['properties']['measurement'] = measurement
    defs['export_field']['properties']['output'] = result
    defs['comparison']['properties'].update(left=exact('offered_comparison', groups + ['new']),
                                           right={'$ref': '#/$defs/offered_comparison'})
    # Required accepted operands cannot become null choices when offers vanish.
    # Planned categorical operands retain their supported `new` selections.
    defs['visualization']['anyOf'] = [form for form in defs['visualization']['anyOf']
                                     if groups or 'labels' in form['properties']]
    for form in defs['visualization']['anyOf']:
        props = form['properties']
        if 'labels' in props:
            props['coordinates'] = exact('offered_coordinates', groups + ['new'])
            props['labels'] = exact('offered_labels', groups + sources + ['new'])
        else:
            for key in ('source', 'coordinates', 'activity'):
                if key in props:
                    props[key] = result
    from copy import deepcopy
    defs['accepted_visualization'] = deepcopy(defs['visualization'])
    for form in defs['accepted_visualization']['anyOf']:
        props = form['properties']
        if 'labels' in props:
            props['coordinates'] = result
            props['labels'] = measurement
    pending = d.get('pending_prerequisite') or {}
    candidates = [r['candidate'] for r in d.get('execution_candidates', ())]
    candidate = exact('offered_candidate', candidates)
    guidance = [v for r in d.get('guidance_predecessor', ()) for v in r.get('references', ())]
    if len(d.get('guidance_predecessor', ())) == 1:
        guidance.append('@candidate')
    guidance_candidate = exact('offered_guidance', guidance)
    resource_rows = (pending.get('resource_answer_choices', ()) if pending.get('task_id')
                     else pending.get('choices', ()) if pending.get('field') == 'resources' else ())
    resources = [r['handle'] for r in resource_rows if isinstance(r, Mapping) and 'handle' in r]
    resource = exact('offered_resource_answer', resources)
    parameters = parameter_choice_offers(public)
    parameter = exact('offered_parameter_choice', parameters)
    variants = []
    for variant in schema['properties']['decision']['anyOf']:
        p = variant['properties']
        kind = p['kind']['enum'][0]
        if kind == 'request_resources':
            continue  # Ordinary resource Requirements follow Candidate acquisition.
        if kind == 'clarify' and p['reason']['enum'][0] in PARAMETER_CLARIFICATION_REASONS:
            if not parameters:
                continue
            p['choices'] = dict(items(parameter, 8), minItems=1)
        if kind == 'navigate':
            relations = [r for r in public.get('relations', ()) if r in RELATIONS]
            if not groups and not outputs and not public.get('bases'):
                relations = []
            if not relations:
                continue
            p['relation'] = exact('offered_relation', relations)
        if kind == 'answer_scientific':
            if not targets:
                continue
            p['visualization'] = (nullable({'$ref': '#/$defs/accepted_visualization'})
                                  if groups else {'type': 'null'})
        if kind == 'answer_guidance':
            if p['candidate'] != {'type': 'null'}:
                if not guidance:
                    continue
                p['candidate'] = guidance_candidate
            if not targets:
                p['targets'] = items({'type': 'null'}, 0)
        if kind == 'execute_candidate':
            if not candidates:
                continue
            p['candidate'] = candidate
        if kind.startswith('answer_') and 'pending' in p:
            expected = {'answer_parameters': '@parameters', 'answer_roles': '@roles', 'answer_resources': '@resources'}[kind]
            source_field = pending.get('source_declaration_field', pending.get('field'))
            # Canonical spp resource choices can belong to an intake question.
            if kind == 'answer_resources':
                if not resources:
                    continue
                p['choice'] = resource
            elif kind == 'answer_roles' and source_field in ('fastq','bam','fragments','annotation','epizoo'):
                if source_field not in p or (source_field == 'annotation' and
                        any(field in p for field in ('fastq','bam','fragments'))):
                    continue
            elif pending.get('handle') != expected:
                continue
            p['pending'] = dict(type='string', enum=[expected])
        if 'target' in p and kind in ('execute_plan', 'request_roles', 'request_resources'):
            if not tools:
                continue
            p['target'] = tool
        if kind == 'request_roles' and p.get('required_roles', {}).get('minItems', 0):
            # Ordinary identified objectives acquire a Candidate first. Keep the
            # complete-role wire readable, but do not offer pre-plan role intake.
            continue
        if 'roles' in p and (not sources or ports == {'type': 'null'}):
            if p['roles'].get('minItems', 0):
                continue
            p['roles']['maxItems'] = 0
        if 'arguments' in p and argument == {'type': 'null'}:
            if p['arguments'].get('minItems', 0):
                continue
            p['arguments']['maxItems'] = 0
        for key in ('input_results', 'references'):
            if key in p:
                p[key] = items(result, 32)
        if 'missing_parameters' in p:
            # Current proposals use scoped fields; historical shorthand strings
            # remain accepted by the unchanged reader, not generated here.
            missing = p['missing_parameters']['items']['anyOf'][1]
            missing['properties'].update(tool=tool, argument=argument,
                native_function=nullable(expanded_functions), member=nullable(member),
                source=nullable(source), port=nullable(ports))
            p['missing_parameters'] = items(missing, 8)
            if argument == {'type': 'null'}:
                p['missing_parameters']['maxItems'] = 0
        if 'retain_parameters' in p and not groups:
            p['retain_parameters'] = items({'$ref': '#/$defs/retained'}, 0)
        if 'context' in p and not d.get('planning_context'):
            p['context'] = dict(type='string', enum=['none'])
        variants.append(variant)
    schema['properties']['decision']['anyOf'] = variants
    _specialized_declaration_targets(schema, tools)
    # Remove unused definitions, including raw/answer-only types whose branches
    # are unavailable. Never leave dangling references or empty anyOf/enum.
    used = set()
    def collect(node):
        if isinstance(node, dict):
            if '$ref' in node:
                name = node['$ref'].rsplit('/', 1)[1]
                if name not in used:
                    used.add(name)
                    collect(defs[name])
            for key, value in node.items():
                if key != '$defs':
                    collect(value)
        elif isinstance(node, list):
            for value in node:
                collect(value)
    collect(schema)
    schema['$defs'] = {k:v for k,v in defs.items() if k in used}
    def enum_count(node):
        if isinstance(node, dict):
            return len(node.get('enum', ())) + sum(enum_count(v) for v in node.values())
        if isinstance(node, list):
            return sum(enum_count(v) for v in node)
        return 0
    # Structured Outputs counts enum values globally, including repeated sets.
    # Preserve every identity while reducing only the encoding of large sets.
    for name, definition in sorted(schema['$defs'].items(),
            key=lambda row: len(row[1].get('enum', ())), reverse=True):
        if enum_count(schema) <= 1000:
            break
        if name.startswith('offered_') and 'enum' in definition:
            schema['$defs'][name] = finite(definition['enum'], enum=False)
    return schema


def _intent_instructions():
    """Interpreter intent/grounding contract; scientific recipes belong downstream."""
    return [
        'Classify researcher intent and capture explicit grounded declarations. dialogue.capabilities is the complete Registry-derived inventory: concise purposes, input roles and every native function name. Detailed metadata expands explicit tool/package/function/parameter references or a captured pending question. The Planner composes methods; canonical Preparation resolves requirements; Registry, compiler and native owners decide scientific legality. No package has priority. Preserve named methods and the whole objective; never invent preprocessing, biological identity, defaults or a workflow.',
        'An unclear operation is clarify/ambiguous_subject with choices=[]. A number alone does not request an operation. Do not turn unspecified processing into a parameter question. Parameter clarification choices must be actual current pending offers, never Registry-wide possibilities or prose encoded as identifiers. Unsupported named implementations use unsupported_intent, never a substitute.',
        'dialogue.ordinary_clarification is bounded prior conversation, not a scientific predecessor, evidence or execution authority. Use its researcher objective and actual prior question only when the current utterance continues it. A new unrelated objective starts independently. Prior generated prose supplies no declarations, sources or results. Literal/evidence grounding always quotes the current utterance; exact current sources remain independently captured.',
        'Read-only questions about accepted results, parameters, methods, execution, provenance, verification and existing downloads use answer_scientific with exact offered targets. General methodological advice, requirements, applicability and possible new work use answer_guidance; targets=[] is valid. Missing evidence stays unknown. Questions never authorize computation.',
        'Use execute_plan for a clear supported scientific objective even when a required file, resource, parameter or existing field is absent or awaits researcher selection. Bind only known source roles and explicit declarations; Planner selects operations and Preparation derives their actual missing requirements and exact controls. Missing input is not an unclear objective. Ordinary objectives use execute_plan, including missing external inputs. request_roles is only the offered specialized raw/annotation/model declaration intake, never an ordinary missing-file alternative. Do not use request_resources for an ordinary objective; Preparation obtains applicable reference requirements after planning. Never require optional resource uploads merely because a port is offered.',
        'Bind source.N only from selected_sources to its actual requested tool/port. current_turn_sources marks newly selected files; other sources are available Session candidates, not automatically selected inputs. Labels, order, filenames, counts and input_type hints establish no scientific eligibility or biological role. Explicit positional references may identify the captured selection positions. Unused sources remain unused; new tasks inherit no prior roles.',
        'Raw/composed declarations use their existing request_roles variant with empty required_roles/roles and explicit metadata in the corresponding declaration, not duplicated scientific arguments. Its schema-constrained target identifies the declaration entry owner, not a fixed workflow; preserve the full original processing and exploration objective for Planner composition. Missing fields remain null; unknown_fields retracts only offered declaration metadata explicitly made unknown, never source/role/resource identities. Exact resource handles must be offered; a named reference preference requires verbatim resource_preference literal/evidence. Declared species/assembly is not a named-reference selection. File membership and biological history cannot be inferred.',
        'A package alone is not a single-function request. Capture only explicitly requested methods, never internal workflow decisions. Every explicit native function_name declaration uses value=<registered function> AND native_function=<same registered function>, with verbatim researcher literal/evidence. Scope each native parameter and kwargs member to its explicit function. Example: PCA then neighbors uses separate pp.pca and pp.neighbors scopes, not two unscoped function_name values. The Planner assigns step IDs and dependencies; do not attach original-file scope to downstream consumers of derived outputs.',
        'arguments contains only explicit tool/argument declarations. Copy literal verbatim and quote a uniquely locating exact current-utterance evidence clause containing it once. Numeric literal is the numeric token alone; preserve exact decimal/rational text. Values and members keep their explicit types, including invalid scientific values; never repair them or manufacture defaults. Repeated numeric values require distinct evidence clauses. Conflicting values require clarification unless explicitly corrected.',
        'Use matrix_semantics only for a researcher declaration about an existing input measurement. Keep requested output measurements in the task objective.',
        'For structured parameters, verbatim explicit JSON is one literal; prose assignments use members with canonical name, kind and separate literal/evidence. Do not fabricate JSON or combine conflicting JSON/prose. kind=string excludes quote delimiters and preserves Unicode/case/whitespace; number preserves tokens; boolean quotes explicit true/false, yes/no or equivalent; json quotes an actual JSON value.',
        'For normalized offered choices value is canonical, while literal and evidence remain verbatim researcher text: Chinese “用 Scanpy 构建邻居图” can use literal=构建邻居图, value=pp.neighbors, native_function=pp.neighbors; it cannot use literal=pp.neighbors unless those bytes occur in the utterance. Exact field/representation strings such as X_pca remain exact literals. observation_source mode=offered associates a clearly selected field with its captured source; mode=literal requires the explicitly named column. If the researcher asks to choose an existing field through the interface, omit its declaration and let Preparation inspect the selected consumer source and offer actual fields. Do not clarify before planning or select a field on their behalf. Agent-produced fields are Planner/Preparation dependencies, never researcher declarations. Roles and field source must agree; never borrow another source’s field or change mode after refusal.',
        'Qualitative or ambiguous requested values remain local in execute_plan.missing_parameters, using exact tool, argument, native_function/member and applicable source/port, plus the exact evidence clause. Omitted optional values are not missing requests. A request to keep native defaults ordinarily leaves remaining kwargs omitted; it does not declare the clustering-only parameter_selection selector on unrelated functions. Use that selector only for the offered applicable clustering policy explicitly requested. Do not invent numeric alternatives, optimizer scans or additional operations. Explanatory parameter questions stay read-only.',
        'Pending answers continue the offered question and original objective. Use the exact pending handle and offered source, parameter and resource identities. Partial explicit answers are allowed. A qualified reference selection carries its recorded coordinates for the selected scientific source. Unrelated messages, new objectives, cancellation and uncertainty never answer a pending request. No stale or cross-task identity is valid.',
        'For a pending source question, capture any explicit source identity or scientific correction in the matching answer_roles declaration, including when the message also names a compatible reference component. For example, “这份 BAM 比对到人类 UCSC hg19（GRCh37）。请使用资源库中与 hg19 匹配的人类 GENCODE 19 基因注释。” sets bam.species=human and bam.assembly=hg19 and records GENCODE 19 as the verbatim bam.resource_preference literal using the exact utterance span. Preserve the captured source; use null for fields the current reply leaves unchanged.',
        'Execute an offered execution_candidate only with its exact candidate ID and a complete explicit command clause as evidence. Interest or agreement alone is not authorization. Do not reconstruct its workflow. A rationale question uses answer_guidance with a quoted guidance_predecessor reference (or @candidate if unique); new objectives use candidate=null.',
        'Accepted outputs/groups and result_referents have distinct exact identities. Select a referent only when unique. @focus and @previous require the scientific predecessor, never ordinary clarification. subject=null asserts no new subject and preserves an existing resolved subject; @other requires unambiguous offered context. Categorical labels are detail fields, not annotation subjects. No recency/order/filename guessing.',
        'context=current only for explicit use or modification of captured accepted analysis; context=none for independent input/work. input_results selects offered accepted groups, never source.N. references select exact accepted results. retain_parameters selects requested exact result/argument settings of the same tool/input; never turn old defaults or generated prose into current explicit declarations. Current explicit changes override their own fields only.',
        'Preserve requested explanation/comparison/visualization delivery. Include explanation for ordinary scientific requests, including requests for a result file. Use delivery=["files_only"] only when the researcher explicitly requests files only or literal outputs without explanation; empty delivery is not files-only authorization. unfulfilled may quote only independently separable unsupported clauses whose omission the researcher permits; never drop essential science or resolvable prerequisites. No hidden fallback or retry plan.',
        'Read-only visualization selects exact accepted operands; planned new operands use new only for explicitly requested new computation. Categorical plots select coordinates, labels and exact source:/result: field. marker_activity_umap selects accepted coordinates/activity; qc selects an accepted result with recorded QC. When accepted operands are unavailable, preserve future visualization delivery and the scientific objective with visualization=null. Reusing saved figures is not new scientific computation; missing or unrelated evidence stays unavailable.',
        'Read-only export selects exact measurement (accepted group or offered source), observations/representations with exact offered result fields and requested destination names. Preserve source lineage and distinct destinations; never invent a field, overwrite an original field or imply download transfer has happened. Scientific owners qualify alignment and prepare bytes.',
        'Scientific comparison requires explicit targets and a meaningful requested dimension; do not infer cross-result cell/cluster correspondence or rank importance without a criterion. focus=continue preserves only the scientific predecessor question. Navigation uses offered relations; previous is ambiguous if parent and previous-active differ; parent requires an explicit parent reference.',
    ]


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
    def named(name):
        return bool(re.search(r'(?<![\w.:])' + re.escape(name) + r'(?!\w|\.[A-Za-z_])', utterance, re.I))
    def parameter_named(name):
        # Plain short kwargs (e.g. an English article) are not explicit parameter
        # references. Expand identifier/syntax references, never language aliases.
        return (('_' in name and named(name)) or named('kwargs.' + name)
                or bool(re.search(r'(?<![\w.])' + re.escape(name) +
                    r'\s*(?:=|:|\bis\b|\bto\b)?\s*(?:[-+]?\d|true\b|false\b|[\"\'])', utterance, re.I))
                or bool(re.search(r'\bparameter\s+' + re.escape(name) + r'(?!\w)', utterance, re.I)))
    pending = dialogue.get('pending_prerequisite') or {}
    pending_tools = {pending.get('tool')} | {
        row.get('tool_name') for row in pending.get('missing', ()) if isinstance(row, Mapping)}
    for tool, item in capabilities.items():
        functions = item.get('functions', {})
        package = item.get('package', '')
        named_package = bool(package and re.search(
            r'(?<![\w.:])' + re.escape(package) + r'(?![\w.:])', utterance, re.I))
        detailed = named(tool) or named_package or tool in pending_tools
        explicit_functions = {fn for fn in functions if named(fn) or named(fn.rsplit('.', 1)[-1])
                              or named(package + '.' + fn) or named(package + '::' + fn)}
        pending_functions = {row.get('native_function') for row in pending.get('missing', ())
                             if isinstance(row, Mapping) and row.get('tool_name') == tool}
        predecessor = dialogue.get('predecessor') or {}
        focus_outputs = {target['output'] for key in ('target', 'previous')
                         if (target := predecessor.get(key)) is not None}
        result_functions = {row.get('native_function', row.get('value') if row.get('argument') == 'function_name' else None)
            for group in dialogue.get('result_groups', ()) if group.get('tool') == tool
            and (group.get('is_active') or named(group['handle'])
                 or bool({group['handle'], *group.get('outputs', ())} & focus_outputs))
            for row in group.get('parameters', ())}
        selected = explicit_functions | (set(functions) & (pending_functions | result_functions))
        selected.update(fn for fn, members in functions.items() if any(parameter_named(k) for k in members))
        if detailed and not selected:
            selected = set(functions)
        item.pop('notes', None)
        # Purpose and every exact function remain discoverable, without all kwargs.
        item['description'] = item.get('description', '').split('. ')[0]
        item['inputs'] = [{k: v for k, v in role.items() if k != 'tool'}
                          for role in dialogue.get('input_roles', ()) if role['tool'] == tool]
        if functions:
            item['functions'] = list(functions)
            if selected:
                item['function_parameters'] = {fn: list(functions[fn]) for fn in functions if fn in selected}
        parameters = {}
        for row in dialogue.get('scientific_parameters', ()):
            if row['tool'] != tool:
                continue
            members = row.get('members', {})
            expanded = detailed or selected or named(row['argument']) or any(parameter_named(k) for k in members)
            if not expanded and row['argument'] in ('kwargs', 'function_name'):
                continue
            value = {k: v for k, v in row.items() if k not in ('tool', 'argument')}
            if not expanded:
                value.pop('default_when_omitted', None)
                value['description'] = value.get('description', '').split('. ')[0]
            if functions and 'members' in value:
                needed = {name for fn in selected for name in functions[fn]}
                value['members'] = {k: v for k, v in members.items() if k in needed or parameter_named(k)}
            parameters[row['argument']] = value
        if parameters:
            item['parameters'] = parameters
    projected = {k: v for k, v in dialogue.items() if k not in ('tools', 'input_roles', 'scientific_parameters', 'capabilities')}
    # Libraries remain available to their scientific/guidance owners. Ordinary
    # intent classification sees only explicitly referenced entries or the
    # current declaration question's catalog, not unrelated model conditions.
    def referenced(row, keys):
        return any(named(value) for key in keys for value in (
            row.get(key, ()) if isinstance(row.get(key), (list, tuple)) else (row.get(key),))
            if isinstance(value, str) and value)
    for key in tuple(projected):
        if key.endswith('_resources'):
            field = key[:-len('_resources')]
            projected[key] = [row for row in projected[key]
                if pending.get('field') == field or named(field)
                or referenced(row, ('handle', 'label', 'assembly'))]
    if 'sequence_models' in projected:
        projected['sequence_models'] = [row for row in projected['sequence_models']
            if referenced(row, ('family', 'conditions'))]
    supplied = projected.get('supplied_inputs')
    if supplied is not None:
        projected['supplied_inputs'] = dict(supplied, fields=[dict(row, consumers=[
            {k:v for k,v in consumer.items() if k not in ('description', 'expected_artifact_kinds')}
            for consumer in row.get('consumers', ())]) for row in supplied.get('fields', ())])
    return public | {'dialogue': projected | {'outputs': list(outputs.values()), 'capabilities': capabilities}}


def interpret(model, utterance, public, *, diagnostic_context=None, authoritative=()):
    projected = _interpretation_public(public, utterance)
    prompt = json.dumps({'turn_schema_version': 1, 'utterance': utterance, **projected,
        'instructions': _intent_instructions()}, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    response_schema = decision_schema(public, utterance=utterance)
    log_operation_event(logger, 'interpreter.preparation_completed', context=diagnostic_context,
                        prompt=prompt, schema=response_schema)
    response = model.complete(prompt=prompt, response_schema=response_schema)
    decision = parse_decision(response, utterance=utterance, authoritative=authoritative)
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


BOOLEAN_LITERALS = {'true': True, 'false': False, 'yes': True, 'no': False,
                   '开启': True, '关闭': False, '是': True, '否': False}


def admit_scientific_argument(declaration, utterance, argument_spec, *, observation_fields=None):
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
                        types = ((int,) if member.literal.lstrip('+-').isdigit() else (float,)) if numeric else ((bool,) if member.kind == 'boolean' else (str,))
                        scalar = replace(argument_spec, accepted_types=types, planning=replace(
                            argument_spec.planning, conversational_json_object=False,
                            conversational_members=(), conversational_exact_string=member.kind == 'string'))
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
        if (start and _word_continuation(utterance[start - 1])
                or end < len(utterance) and _word_continuation(utterance[end])):
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
            value = literal if literal in argument_spec.choices else declaration.value
            argument_spec.validate(declaration.argument, value)
        except (ValueError, TypeError) as exc:
            raise IntentError('invalid_parameter_value') from exc
        return value
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
        if (before and _word_continuation(before) or before in ('.', '_', '+', '-', '/')
                or after and _word_continuation(after) or after in ('_', '+', '-', '/')
                or after == '.' and not sentence_period
                or any(c and unicodedata.category(c).startswith('M') for c in (before, after))):
            raise IntentError('ungrounded_operand')
        try:
            argument_spec.validate(declaration.argument, literal)
        except (ValueError, TypeError) as exc:
            raise IntentError('invalid_parameter_value') from exc
        return literal
    if argument_spec.accepted_types == (bool,):
        # Bounded conventional boolean words, independent of scientific tools.
        if (start and _word_continuation(utterance[start-1])
                or end < len(utterance) and _word_continuation(utterance[end])):
            raise IntentError('ungrounded_operand')
        if literal.casefold() not in BOOLEAN_LITERALS:
            raise IntentError('ungrounded_operand')
        return BOOLEAN_LITERALS[literal.casefold()]
    if not any(t in argument_spec.accepted_types for t in (int, float)):
        raise IntentError('unsupported_intent')
    # These are numeric-token boundaries, not a classifier for scientific intent
    # or a language parser. The Interpreter owns both assignment and association.
    if not _numeric_token_span(utterance, start, end):
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
