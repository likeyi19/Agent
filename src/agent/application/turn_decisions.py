"""Bounded turn interpretation and exact, utterance-grounded numeric admission."""
from dataclasses import asdict, dataclass
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
class Execute:
    base: str
    operation: str
    target: str
    delta: IntentDelta | None


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


TurnDecision = Execute | Navigate | Clarify | Answer
RELATIONS = ('current', 'parent', 'previous_active', 'previous')
REASONS = ('missing_parameter_value', 'ambiguous_parameter', 'ambiguous_revision',
           'unsupported_intent', 'invalid_parameter_value', 'ungrounded_operand',
           'unavailable_context', 'invalid_decision', 'planning_failed', 'ambiguous_subject',
           'ambiguous_predecessor', 'incompatible_comparison', 'requires_execution')
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
            if set(value) != {'kind', 'target'}:
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
    delta = _object(dict(parameter=enum(p for o in offered for p in o['parameters']),
        operation=enum(('set', 'add', 'subtract')), literal={'type':'string'}, evidence={'type':'string'}))
    variants = [_object(dict(kind=enum(('answer',)), intent=enum(i for i in ANSWER_INTENTS if i not in ('scientific', 'guidance')),
        relation=enum(public['relations']), technical={'type':'boolean'})), _object(dict(kind=enum(('clarify',)), reason=enum(REASONS))),
        _object(dict(kind=enum(('navigate',)), relation=enum(public['relations'])))]
    if offered:
        variants.append(_object(dict(kind=enum(('execute',)), base=enum(public['relations']),
            operation=enum(o['handle'] for o in offered), target=enum(('selection','matrix')),
            delta={'anyOf':[delta, {'type':'null'}]})))
    if 'dialogue' in public:
        target = _object(dict(output=enum([o['handle'] for o in public['dialogue']['outputs']] + ['@focus', '@previous']),
                             subject={'anyOf':[{'type':'string', 'maxLength':128}, {'type':'null'}]}))
        variants.append(_object(dict(kind=enum(('answer_scientific',)), target=target,
            comparison={'anyOf':[target, {'type':'null'}]}, focus=enum(('question', 'continue')))))
        variants.append(_object(dict(kind=enum(('answer_guidance',)),
            targets={'type':'array', 'maxItems':4, 'items':target},
            candidate={'anyOf':[{'type':'string', 'maxLength':128}, {'type':'null'}]})))
        variants.append(_object(dict(kind=enum(('execute_plan',)), target=enum(public['dialogue']['tools']))))
    return _object(dict(turn_schema_version={'type':'integer','enum':[1]}, decision={'anyOf':variants}))


def interpret(model, utterance, public):
    prompt = json.dumps({'turn_schema_version': 1, 'utterance': utterance, **public,
        'instructions': [
            'For what to analyze next, useful analyses, alternatives or missing evidence for an objective, choose answer_guidance. It is read-only advice, never execution. Select up to four relevant exact evidence targets; an empty set is allowed and means no result evidence selected, not scientific absence.',
            'Guidance co-presents evidence; it does not compare measurements or infer common populations/cluster correspondence. Explicit subjects and historical targets obey the same exact reference rules as scientific answers.',
            'For a guidance-option rationale follow-up, targets=[] and candidate is an offered guidance_predecessor reference quoted in the utterance, or @candidate only when a unique candidate exists. For a new objective use candidate=null. No prior generated rationale is evidence.',
            'Executing a guidance option is not supported yet. Clarify unsupported_intent for option execution requests; do not translate an option into execute_plan.',
            'First distinguish discussing an existing scientific result from asking about workflow state or requesting new computation.',
            'For what a result shows, means, establishes, why an assignment occurred, or what changed scientifically, choose answer_scientific. Operational selection/matrix/version answers only describe workflow state; they do not explain scientific results.',
            'dialogue.outputs is the accepted-result inventory. is_active identifies current results; semantics reuses registered descriptions/roles. evidence_status=available means a bounded accepted summary is readable. available_fields and subjects describe coverage, not factual values.',
            'Resolve explicit result names or semantic roles against this inventory, preferring current results unless history is requested. Multiple unresolved candidates require ambiguous_subject, never a first match.',
            'A single current result or a unique dialogue.predecessor resolves this result. For short why, value-source or certainty follow-ups, retain that scientific target with @focus (and its subject); do not switch to operational provenance or a different output.',
            'The predecessor contains the exact target, previous target/comparison and prior question, not generated prose. No predecessor means no implicit scientific discussion; clarify unresolved short references instead of inventing a topic.',
            'Use unavailable_context only when the requested target is missing or its evidence unavailable. Missing rationale or scientific certainty within an available summary is for the scientific answer stage to explain as insufficient evidence, not a reason to reject the target.',
            'You select intent and exact references, not the scientific conclusion. You need not see factual values to select answer_scientific for an available result. Detailed accepted evidence is loaded after admission.',
            'Choose execute, navigate, clarify, or answer. Preserve existing operational answer intents.',
            'Answer intents: version, matrix, selection, thresholds, changes (originating revision versus parent), execution, reuse, comparison, provenance, verification, unsupported.',
            'Comparison needs parent, previous_active, or previous. Other answers normally use current.',
            'For questions about scientific results, use kind=answer_scientific with offered output handles. It can return insufficient evidence.',
            'For a new scientific command outside the offered threshold operations use kind=execute_plan, target=the registered tool. The Agent derives the plan operation on the current revision without a parameter delta. Never answer a computation request as if it was already computed.',
            'Scientific targets: @focus is the captured predecessor target; @previous is its comparison or previous subject. No predecessor means no implicit focus.',
            'Subject is null for the whole result, @focus to retain the subject, @other only for an unambiguous other subject, or an offered subject candidate ID/reference quoted in the question. References resolve only within that result; do not invent aliases.',
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
        ]}, sort_keys=True, separators=(',', ':'))
    return parse_decision(model.complete(prompt=prompt, response_schema=decision_schema(public)))


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
