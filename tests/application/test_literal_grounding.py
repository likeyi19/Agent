"""Provider text is located once; durable scientific span validation stays strict."""
import json

import pytest

from agent.application.turn_decisions import (
    IntentError, admit_scientific_argument, parse_decision,
)
from agent.orchestration import build_default_tool_registry


def wire(literal, evidence, *, argument='min_tss_enrichment', tool='select_scATAC_cells', **extra):
    return json.dumps(dict(turn_schema_version=1, decision=dict(kind='execute_plan',
        target=tool, arguments=[dict(tool=tool, argument=argument,
                                    literal=literal, evidence=evidence, **extra)])))


@pytest.mark.parametrize('text,literal,evidence', [
    ('Use TSS 4.5.', '4.5', 'TSS 4.5'),
    ('Use depth 20 and flank evidence 20.', '20', 'flank evidence 20'),
    ('Use TSS -0.5.', '-0.5', 'TSS -0.5'),
    ('Use resolution 1e-3.', '1e-3', 'resolution 1e-3'),
    ('Use column “细胞 类型”.', '细胞 类型', 'column “细胞 类型”'),
    ('Use TSS 4; correction: use TSS 5.', '5', 'correction: use TSS 5'),
])
def test_exact_text_location(text, literal, evidence):
    declaration, = parse_decision(wire(literal, evidence), utterance=text).arguments
    assert text[declaration.start:declaration.end] == literal
    assert declaration.start == text.index(evidence) + evidence.index(literal)


@pytest.mark.parametrize('text,literal,evidence,reason', [
    ('depth 20 and flank 20', '20', '20', 'ambiguous_parameter'),
    ('TSS 4; TSS 4', '4', 'TSS 4', 'ambiguous_parameter'),
    ('depth 20 and flank 20', '20', 'depth 20 and flank 20', 'ambiguous_parameter'),
    ('Use TSS 5', '4', 'Use TSS 5', 'ungrounded_operand'),
    ('Use TSS 5', '5', 'Use TSS  5', 'ungrounded_operand'),
])
def test_no_first_match_or_text_repair(text, literal, evidence, reason):
    with pytest.raises(IntentError, match=reason):
        parse_decision(wire(literal, evidence), utterance=text)


@pytest.mark.parametrize('text,literal', [('TSS 120', '20'), ('TSS -4', '4'),
    ('TSS 1.5', '5'), ('TSS 1e3', '3'), ('TSS 4/3', '3')])
def test_locating_text_does_not_bypass_token_boundary(text, literal):
    declaration, = parse_decision(wire(literal, text), utterance=text).arguments
    spec = build_default_tool_registry().get('select_scATAC_cells').optional_arguments['min_tss_enrichment']
    with pytest.raises(IntentError, match='ungrounded_operand'):
        admit_scientific_argument(declaration, text, spec)


def test_legacy_misindex_is_normalized_only_at_provider_boundary():
    text = 'Use minimum flank evidence 20.'
    value = dict(tool='select_scATAC_cells', argument='min_tss_flank_evidence',
                 literal='20', start=text.index('20') - 1, end=text.index('20') + 1)
    raw = json.dumps(dict(turn_schema_version=1, decision=dict(kind='execute_plan',
        target='select_scATAC_cells', arguments=[value])))
    historical, = parse_decision(raw).arguments
    spec = build_default_tool_registry().get('select_scATAC_cells').optional_arguments['min_tss_flank_evidence']
    with pytest.raises(IntentError, match='ungrounded_operand'):
        admit_scientific_argument(historical, text, spec)
    grounded, = parse_decision(raw, utterance=text).arguments
    assert admit_scientific_argument(grounded, text, spec) == 20
    assert historical.start == grounded.start - 1


def test_normalized_species_choice_and_unicode_exact_string():
    registry = build_default_tool_registry()
    for text, literal, tool, argument, extra, expected in [
        ('These are human cells.', 'human', 'epizoo_embed_cells', 'species', {'value': 'human'}, 'human'),
        ('Use column “细胞 类型”.', '细胞 类型', 'evaluate_cell_clustering', 'label_key', {}, '细胞 类型'),
    ]:
        declaration, = parse_decision(wire(literal, text, tool=tool, argument=argument, **extra), utterance=text).arguments
        spec = registry.get(tool)
        spec = spec.required_arguments.get(argument, spec.optional_arguments.get(argument))
        assert admit_scientific_argument(declaration, text, spec) == expected


def test_mixed_wire_or_forged_fields_are_not_repaired():
    for extra in ({'start': 0, 'end': 1}, {'source': 'trusted'}):
        with pytest.raises(IntentError, match='invalid_decision'):
            parse_decision(wire('4', 'Use TSS 4', **extra), utterance='Use TSS 4')


def test_categorical_parameter_correction_uses_same_exact_evidence_boundary():
    raw = json.loads(wire('human', 'These are human cells.', tool='epizoo_embed_cells',
                          argument='species', value='human'))
    raw['decision'].pop('target')
    raw['decision'].update(kind='answer_parameters', pending='@parameters')
    decision = parse_decision(json.dumps(raw), utterance='These are human cells.')
    specification = build_default_tool_registry().get('epizoo_embed_cells').required_arguments['species']
    assert admit_scientific_argument(decision.arguments[0], 'These are human cells.', specification) == 'human'
