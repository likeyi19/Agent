"""Bounded turn interpretation and exact, utterance-grounded scalar admission."""
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import json
import re


class IntentError(ValueError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


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
                or type(self.literal) is not str or not 0 < len(self.literal) <= 80
                or type(self.start) is not int or type(self.end) is not int
                or not 0 <= self.start < self.end <= 4096):
            raise ValueError('Invalid scoped scientific argument declaration.')


@dataclass(frozen=True)
class LeidenResolution(ScopedArgument):
    """The strict historical wire for the original clustering declaration."""

    def __post_init__(self):
        super().__post_init__()
        if self.tool != 'cluster_cells' or self.argument != 'resolution':
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


def _validate_arguments(arguments, argument=None):
    if (type(arguments) is not tuple or len(arguments) > 8
            or any(not isinstance(item, ScopedArgument) for item in arguments)):
        raise ValueError('Invalid bounded scientific arguments.')
    identities = [(item.tool, item.argument) for item in arguments]
    if argument is not None:
        identities.append((argument.tool, argument.argument))
    if len(identities) > 8:
        raise ValueError('Scientific argument declaration limit exceeded.')
    if len(identities) != len(set(identities)):
        raise ValueError('Duplicate scoped scientific argument.')


@dataclass(frozen=True)
class Execute:
    base: str
    operation: str
    target: str
    delta: IntentDelta | None
    argument: LeidenResolution | None = None
    arguments: tuple[ScopedArgument, ...] = ()

    def __post_init__(self):
        if self.argument is not None and not isinstance(self.argument, LeidenResolution):
            raise ValueError('Invalid explicit scientific declaration.')
        _validate_arguments(self.arguments, self.argument)


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
        if (type(self.pending) is not str or not 0 < len(self.pending) <= 128
                or type(self.species) is not str or self.species not in ('human', 'mouse')
                or self.argument is not None and not isinstance(self.argument, LeidenResolution)):
            raise ValueError('Invalid pending species declaration.')
        _validate_arguments(self.arguments, self.argument)
        if any(isinstance(item, ScalarChoiceArgument) for item in self.arguments):
            raise ValueError('Species answers use the existing species field.')


@dataclass(frozen=True)
class ParameterAnswer:
    """Numeric declarations answering one exact pending scientific request."""

    pending: str
    arguments: tuple[ScopedArgument, ...]

    def __post_init__(self):
        if type(self.pending) is not str or not 0 < len(self.pending) <= 128 or not self.arguments:
            raise ValueError('Invalid pending scientific parameter answer.')
        _validate_arguments(self.arguments)
        if any(isinstance(item, ScalarChoiceArgument) for item in self.arguments):
            raise ValueError('Parameter answers require numeric declarations.')


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

    def __post_init__(self):
        if (not isinstance(self.target, ScientificTarget)
                or self.comparison is not None and not isinstance(self.comparison, ScientificTarget)
                or self.focus not in ('question', 'continue')):
            raise ValueError('Invalid scientific question.')


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


TurnDecision = Execute | ExecuteCandidate | SpeciesAnswer | ParameterAnswer | Navigate | Clarify | Answer
RELATIONS = ('current', 'parent', 'previous_active', 'previous')
REASONS = ('missing_parameter_value', 'ambiguous_parameter', 'ambiguous_revision',
           'unsupported_intent', 'invalid_parameter_value', 'ungrounded_operand',
           'unavailable_context', 'invalid_decision', 'planning_failed', 'ambiguous_subject',
           'ambiguous_predecessor', 'incompatible_comparison', 'requires_execution',
           'missing_species', 'ambiguous_species', 'unsupported_species',
           'conflicting_species', 'invalid_prerequisite', 'conflicting_scientific_parameter')
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
    if type(value) is not list or len(value) > 8:
        raise IntentError('invalid_decision')
    result = []
    for item in value:
        choice = type(item) is dict and 'value' in item
        _shape(item, ('tool', 'argument', 'literal', 'start', 'end', 'value') if choice
               else ('tool', 'argument', 'literal', 'start', 'end'))
        result.append((ScalarChoiceArgument if choice else ScopedArgument)(**item))
    return tuple(result)


def parse_decision(raw):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result: raise IntentError('invalid_decision')
            result[k] = v
        return result
    try:
        if type(raw) is not str or len(raw.encode()) > 8192: raise IntentError('invalid_decision')
        envelope = json.loads(raw, object_pairs_hook=pairs)
        _shape(envelope, ('turn_schema_version', 'decision'))
        if type(envelope['turn_schema_version']) is not int or envelope['turn_schema_version'] != 1:
            raise IntentError('invalid_decision')
        value = envelope['decision']
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
        if kind == 'answer_parameters':
            _shape(value, ('kind', 'pending', 'arguments'))
            return ParameterAnswer(value['pending'], _scientific_arguments(value['arguments']))
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
                _shape(value, ('kind', 'target', 'comparison', 'focus'))
            value = dict(value, kind='answer', intent='scientific')
            kind = 'answer'
        elif kind == 'execute_plan':
            if {'kind', 'target'} <= set(value) <= {'kind', 'target', 'argument', 'arguments'}:
                if type(value['target']) is not str:
                    raise IntentError('invalid_decision')
                return Execute('current', 'plan', value['target'], None,
                               _leiden_argument(value.get('argument')),
                               _scientific_arguments(value.get('arguments', [])))
            else:
                _shape(value, ('kind', 'base', 'operation', 'target', 'delta'))
                if value['operation'] != 'plan' or value['base'] != 'current' or value['delta'] is not None:
                    raise IntentError('invalid_decision')
            value = dict(value, kind='execute', operation='plan', base='current', delta=None)
            kind = 'execute'
        if kind == 'clarify':
            _shape(value, ('kind', 'reason'))
            if value['reason'] not in REASONS: raise IntentError('invalid_decision')
            return Clarify(value['reason'], value_required=value['reason'] == 'missing_parameter_value')
        if kind == 'answer':
            if value.get('intent') == 'scientific':
                _shape(value, ('kind', 'intent', 'target', 'comparison', 'focus'))
                def target(v):
                    _shape(v, ('output', 'subject'))
                    return ScientificTarget(**v)
                return Answer('scientific', scientific=ScientificQuestion(target(value['target']),
                    None if value['comparison'] is None else target(value['comparison']), value['focus']))
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


def decision_schema(public):
    offered = [o for b in public['bases'].values() for o in b['operations']]
    enum = lambda values: {'type': 'string', 'enum': list(dict.fromkeys(values))}
    argument = {'anyOf': [_object(dict(tool=enum(('cluster_cells',)),
        argument=enum(('resolution',)), literal={'type': 'string', 'minLength': 1, 'maxLength': 80},
        start={'type': 'integer', 'minimum': 0, 'maximum': 4096},
        end={'type': 'integer', 'minimum': 1, 'maximum': 4096})), {'type': 'null'}]}
    parameters = public.get('dialogue', {}).get('scientific_parameters', ())
    identities = tuple(dict.fromkeys((item['tool'], item['argument']) for item in parameters))
    def declaration_schema(item):
        properties = dict(tool=enum((item['tool'],)), argument=enum((item['argument'],)),
            literal={'type': 'string', 'minLength': 1, 'maxLength': 80},
            start={'type': 'integer', 'minimum': 0, 'maximum': 4096},
            end={'type': 'integer', 'minimum': 1, 'maximum': 4096})
        if 'choices' in item:
            properties['value'] = enum(item['choices'])
        return _object(properties)
    declarations = {'type': 'array', 'maxItems': 8, 'items': {'anyOf': [
        declaration_schema(item) for item in parameters]}}
    numeric = [item for item in parameters if 'choices' not in item]
    numeric_declarations = declarations | {'items': {'anyOf': [declaration_schema(item) for item in numeric]}}
    delta = _object(dict(parameter=enum(p for o in offered for p in o['parameters']),
        operation=enum(('set', 'add', 'subtract')), literal={'type':'string'}, evidence={'type':'string'}))
    pending = public.get('dialogue', {}).get('pending_prerequisite')
    species_pending = pending is not None and pending.get('field', 'species') == 'species'
    clarification_reasons = REASONS if species_pending else tuple(r for r in REASONS if r != 'missing_species')
    variants = [_object(dict(kind=enum(('answer',)), intent=enum(i for i in ANSWER_INTENTS if i not in ('scientific', 'guidance')),
        relation=enum(public['relations']), technical={'type':'boolean'})), _object(dict(kind=enum(('clarify',)), reason=enum(clarification_reasons))),
        _object(dict(kind=enum(('navigate',)), relation=enum(public['relations'])))]
    if offered:
        variants.append(_object(dict(kind=enum(('execute',)), base=enum(public['relations']),
            operation=enum(o['handle'] for o in offered), target=enum(('selection','matrix')),
            delta={'anyOf':[delta, {'type':'null'}]})))
    if 'dialogue' in public:
        target = _object(dict(output=enum([o['handle'] for o in public['dialogue']['outputs']]
                             + [g['handle'] for g in public['dialogue'].get('result_groups', ())]
                             + list(public['dialogue'].get('result_referents', {})) + ['@focus', '@previous']),
                             subject={'anyOf':[{'type':'string', 'maxLength':128}, {'type':'null'}]}))
        variants.append(_object(dict(kind=enum(('answer_scientific',)), target=target,
            comparison={'anyOf':[target, {'type':'null'}]}, focus=enum(('question', 'continue')))))
        variants.append(_object(dict(kind=enum(('answer_guidance',)),
            targets={'type':'array', 'maxItems':4, 'items':target},
            candidate={'anyOf':[{'type':'string', 'maxLength':128}, {'type':'null'}]})))
        execution = dict(kind=enum(('execute_plan',)), target=enum(public['dialogue']['tools']), argument=argument)
        if identities:
            execution['arguments'] = declarations
        variants.append(_object(execution))
        candidates = public['dialogue'].get('execution_candidates', ())
        if candidates:
            variants.append(_object(dict(kind=enum(('execute_candidate',)),
                candidate=enum(c['candidate'] for c in candidates), evidence={'type':'string','maxLength':4096})))
        pending = public['dialogue'].get('pending_prerequisite')
        if pending is not None and pending.get('field', 'species') == 'species':
            answer = dict(kind=enum(('answer_prerequisite',)),
                pending=enum((pending['handle'],)), species=enum(pending['choices']), argument=argument)
            if numeric:
                answer['arguments'] = numeric_declarations
            variants.append(_object(answer))
        elif pending is not None and pending.get('field') == 'parameters' and numeric:
            variants.append(_object(dict(kind=enum(('answer_parameters',)),
                pending=enum((pending['handle'],)), arguments=numeric_declarations | {'minItems': 1})))
    return _object(dict(turn_schema_version={'type':'integer','enum':[1]}, decision={'anyOf':variants}))


def interpret(model, utterance, public):
    prompt = json.dumps({'turn_schema_version': 1, 'utterance': utterance, **public,
        'instructions': [
            'For what to analyze next, useful analyses, alternatives or missing evidence for an objective, choose answer_guidance. It is read-only advice, never execution. Select up to four relevant exact evidence targets; an empty set is allowed and means no result evidence selected, not scientific absence.',
            'Guidance co-presents evidence; it does not compare measurements or infer common populations/cluster correspondence. Explicit subjects and historical targets obey the same exact reference rules as scientific answers.',
            'For a guidance-option rationale follow-up, targets=[] and candidate is an offered guidance_predecessor reference quoted in the utterance, or @candidate only when a unique candidate exists. For a new objective use candidate=null. No prior generated rationale is evidence.',
            'For explicit execution of an offered execution_candidate use execute_candidate with its exact candidate ID and an exact complete command clause from this utterance as evidence. Never translate an option into execute_plan or construct a workflow here.',
            'Resolve ordinal or named candidate references against the offered stable options and objective. Clarify ambiguous other/that references, stale references, conflicting subject/revision refinements, or missing execution intent. Interest or agreement alone is not execution authorization. Why/evidence questions remain answer_guidance.',
            'Current-turn refinements remain user planning intent, not edits to the original candidate. Keep its captured subjects/revision fixed; an additional comparison may refine the objective but must not silently replace its subject.',
            'First distinguish discussing an existing scientific result from asking about workflow state or requesting new computation.',
            'For what a result shows, means, establishes, why an assignment occurred, or what changed scientifically, choose answer_scientific. Operational selection/matrix/version answers only describe workflow state; they do not explain scientific results.',
            'dialogue.outputs is the accepted-result inventory. is_active identifies current results; semantics reuses registered descriptions/roles. evidence_status=available means a bounded accepted summary is readable. available_fields and subjects describe coverage, not factual values.',
            'dialogue.result_groups groups exact cooutputs of the same accepted scientific step. Select a group handle for its whole scientific result, without copying a member or guessing a workflow. Independent groups are distinct results; clarify unresolved ambiguity.',
            'dialogue.result_referents offers captured @current_result, @most_recently_created and @previous_turn_result scopes. They are different: navigation changes the active result without creating a result, and an immediately preceding answered scientific turn may discuss an existing subject. Select the semantic handle only when its target is unique; otherwise choose the relevant offered group/subject or clarify. No scanning backward or defaulting to another scope.',
            'dialogue.supplied_inputs describes current-turn structured input presence, shallow JSON types and registered consumer expectations only. It exposes no values and establishes neither scientific format, qualification nor execution readiness. Distinguish no supplied inputs from omitted unregistered fields. Input presence can inform intent selection; Planner/compiler/tools still own binding and validation.',
            'Resolve explicit result names or semantic roles against this inventory, preferring current results unless history is requested. Multiple unresolved candidates require ambiguous_subject, never a first match.',
            'A single current result or a unique dialogue.predecessor resolves this result. For short why, value-source or certainty follow-ups, retain that scientific target with @focus (and its subject); do not switch to operational provenance or a different output.',
            'The predecessor contains the exact target, previous target/comparison and prior question, not generated prose. No predecessor means no prior-discussion focus; captured result groups and result_referents remain available. Clarify unresolved short references instead of inventing a topic.',
            'Use unavailable_context only when the requested target is missing or its evidence unavailable. Missing rationale or scientific certainty within an available summary is for the scientific answer stage to explain as insufficient evidence, not a reason to reject the target.',
            'You select intent and exact references, not the scientific conclusion. You need not see factual values to select answer_scientific for an available result. Detailed accepted evidence is loaded after admission.',
            'Choose execute, navigate, clarify, or answer. Preserve existing operational answer intents.',
            'Answer intents: version, matrix, selection, thresholds, changes (originating revision versus parent), execution, reuse, comparison, provenance, verification, unsupported.',
            'Comparison needs parent, previous_active, or previous. Other answers normally use current.',
            'For questions about scientific results, use kind=answer_scientific with offered output handles. It can return insufficient evidence.',
            'For a new scientific command outside the offered threshold operations use kind=execute_plan, target=the registered tool. The Agent derives the plan operation on the current revision without a parameter delta. Never answer a computation request as if it was already computed.',
            'For an otherwise supported new scientific command with a selected input, an omitted or unresolved required dataset declaration is not unsupported intent and is not a direct missing-prerequisite answer. Choose execute_plan and omit that declaration from arguments. The existing Planner/compiler determines the exact missing prerequisite and the Application creates its durable clarification. Use missing_species only while answering an offered pending species prerequisite; never fabricate a pending request. Do not guess a value or bypass genuinely unsupported capabilities, input identity ambiguity or conflicting explicit declarations.',
            'An explicitly named scientific backend, algorithm or implementation constrains the requested operation. Choose an offered registered capability only if it supports that named choice; if unavailable, use clarify/unsupported_intent rather than substitute a similar implementation. For a general operation without a named implementation, select a suitable offered capability. Detailed planning and deterministic validation still enforce compatibility.',
            'execute_plan.argument is the historical nullable cluster_cells.resolution declaration. execute_plan.arguments contains explicit scalar declarations for canonical tool/argument pairs offered in dialogue.scientific_parameters. Associate each value with the requested operation; never assign an ambiguous value to an arbitrary step. Copy each literal verbatim and give start/end zero-based Unicode character offsets in this utterance. Do not repeat an identity across argument and arguments.',
            'For an offered parameter with choices, value is your semantic normalization to an exact offered choice and literal is the exact dataset declaration in this utterance. An initial explicit species declaration about the selected cells or dataset belongs in execute_plan.arguments even when the target is a downstream tool. A statement about a model supporting human and mouse is not a declaration of the selected dataset species; omit that argument. Never infer species from filenames, dimensions, features, available resources or checkpoint identity. Unsupported or conflicting explicit dataset declarations require the corresponding clarification; an unresolved declaration in a supported new command uses the compiler-owned handoff above. A user declaration does not establish input compatibility. Pending species answers continue using answer_prerequisite.species, not a choice argument.',
            'Use argument=null when resolution is omitted; the scientific owner retains its existing default. A number alone does not request an operation. Explanatory questions about resolution remain answers, never execution or edits to a pending request. Clarify ambiguous argument association or conflicting explicit values; do not guess, invent an amount or replace an invalid value with a default.',
            'Use arguments=[] for omitted scientific parameters. Registered owners and existing compiler/resource contracts decide requirements and defaults; never invent a numeric default, infer scientific input identity, or create arbitrary execution dictionaries. Exact decimal or rational threshold literals stay exact; do not approximate them with floating point values.',
            'Scientific targets: @focus is the captured predecessor target; @previous is its comparison or previous subject. No predecessor means no implicit focus.',
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
            'dialogue.pending_prerequisite describes one captured scientific request awaiting an explicit human/mouse species declaration. Its objective and selected dataset remain fixed; this context is not scientific compatibility evidence.',
            'Choose answer_prerequisite only when this utterance unambiguously answers that specifically pending species question for the same request and selected dataset. Use its exact offered pending handle and normalize the explicit declaration to the offered human or mouse value. Meaning is yours to interpret; never infer species from filenames, dimensions, features, available checkpoints or unrelated datasets.',
            'Uncertainty or ambiguous species requires clarify/ambiguous_species; another species requires clarify/unsupported_species; a conflicting declaration requires clarify/conflicting_species. Do not guess or emit arbitrary scientific input dictionaries.',
            'If the user changes the analysis objective, choose the ordinary decision for that new request, such as execute_plan. Cancellation or an unrelated message must never answer the old prerequisite; use clarify/unsupported_intent when no other supported decision applies. An answer authorizes continuation only of the captured original objective, never additional operations.',
            'answer_prerequisite.argument and arguments may declare explicit offered scientific parameters in the same reply only for operations already requested by the captured objective. Otherwise use null and []; original validated scientific parameters remain captured. Do not add clustering or other operations to answer a species prerequisite.',
        ] if (public.get('dialogue', {}).get('pending_prerequisite') or {}).get('field', 'species') == 'species'
              and public.get('dialogue', {}).get('pending_prerequisite') is not None else []) + ([
            'dialogue.pending_prerequisite describes one captured scientific request awaiting the exact required parameters listed there. Its original objective, inputs and operation scope remain fixed; pending context does not establish scientific compatibility or execution authority.',
            'Choose answer_parameters only when this utterance explicitly supplies or corrects parameters for that same pending request. Use its exact offered pending handle and canonical offered tool/argument pairs with verbatim numeric literals and exact character spans. Partial answers are allowed; never invent the remaining values or add operations.',
            'An answer continues only the original explicit execution request through existing scientific validation. Unrelated messages, explanations, uncertainty, cancellation and new objectives must use the ordinary applicable decision, never answer_parameters. Clarify conflicting or ambiguous parameter associations.',
        ] if (public.get('dialogue', {}).get('pending_prerequisite') or {}).get('field') == 'parameters' else [])},
        sort_keys=True, separators=(',', ':'))
    return parse_decision(model.complete(prompt=prompt, response_schema=decision_schema(public)))


def admit_scientific_argument(declaration, utterance, argument_spec):
    """Check one semantic declaration's literal identity, then delegate its science."""
    if not isinstance(declaration, ScopedArgument):
        raise IntentError('invalid_decision')
    literal, start, end = declaration.literal, declaration.start, declaration.end
    if (type(utterance) is not str or end > len(utterance)
            or utterance[start:end] != literal):
        raise IntentError('ungrounded_operand')
    if isinstance(declaration, ScalarChoiceArgument):
        if (start and utterance[start - 1].isalnum()
                or end < len(utterance) and utterance[end].isalnum()):
            raise IntentError('ungrounded_operand')
        if (not argument_spec.planning or not argument_spec.planning.conversational_choice
                or argument_spec.accepted_types != (str,) or not argument_spec.choices):
            raise IntentError('unsupported_intent')
        try:
            argument_spec.validate(declaration.argument, declaration.value)
        except (ValueError, TypeError) as exc:
            raise IntentError('invalid_parameter_value') from exc
        return declaration.value
    # These are numeric-token boundaries, not a classifier for scientific intent
    # or a language parser. The Interpreter owns both assignment and association.
    if start and (utterance[start - 1].isalnum() or utterance[start - 1] in '.+-/_%'):
        raise IntentError('ungrounded_operand')
    if end < len(utterance):
        following = utterance[end]
        sentence_period = (following == '.'
            and (end + 1 == len(utterance) or not utterance[end + 1].isdigit()))
        if (following.isalnum() or following in '+-/_%'
                or following == '.' and not sentence_period):
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
