"""Provider text is located once; durable scientific span validation stays strict."""
import json

import pytest

from agent.application.turn_decisions import (
    IntentError, ScopedArgument, admit_scientific_argument, parse_decision,
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
    with pytest.raises(IntentError, match='ungrounded_operand'):
        parse_decision(wire(literal, text), utterance=text)
    start = text.index(literal)
    declaration = ScopedArgument('select_scATAC_cells', 'min_tss_enrichment', literal, start, start + len(literal))
    spec = build_default_tool_registry().get('select_scATAC_cells').optional_arguments['min_tss_enrichment']
    with pytest.raises(IntentError, match='ungrounded_operand'):
        admit_scientific_argument(declaration, text, spec)


@pytest.mark.parametrize('text,literal', [
    ('Extract a 2114-base window.', '2114'),
    ('Use a 200-base window.', '200'),
    ('Use a 5-fold threshold.', '5'),
])
def test_numeric_quantity_modifiers(text, literal):
    declaration, = parse_decision(wire(literal, text), utterance=text).arguments
    spec = build_default_tool_registry().get('select_scATAC_cells').optional_arguments['min_tss_enrichment']
    assert admit_scientific_argument(declaration, text, spec) == float(literal)
    with pytest.raises(IntentError, match='ungrounded_operand'):
        admit_scientific_argument(declaration, text, spec, _legacy_numeric_boundary=True)


@pytest.mark.parametrize('text,literal', [
    ('value 20-30', '20'), ('value 20-30', '30'), ('value 20.5', '20'),
    ('value 20e-3', '20'), ('value 20/30', '20'), ('sample_A20', '20'),
    ('value 20%', '20'), ('value 20-base', '2'), ('value -20-base', '20'),
])
def test_quantity_modifier_does_not_admit_partial_numbers(text, literal):
    with pytest.raises(IntentError, match='ungrounded_operand'):
        parse_decision(wire(literal, text), utterance=text)
    start = text.index(literal)
    declaration = ScopedArgument('select_scATAC_cells', 'min_tss_enrichment', literal, start, start + len(literal))
    spec = build_default_tool_registry().get('select_scATAC_cells').optional_arguments['min_tss_enrichment']
    with pytest.raises(IntentError, match='ungrounded_operand'):
        admit_scientific_argument(declaration, text, spec)


def test_saved_quantity_rejection_retains_its_original_meaning():
    from agent.application.dialogue_execution import _repair_declarations
    text = 'Use a 2114-base window and 2 attempts.'
    tool = 'prepare_scatac_regulatory_sequences'
    raw = wire('2114', '2114-base window', tool=tool, argument='window_size')
    registry = build_default_tool_registry()
    with pytest.raises(IntentError, match='invalid_prerequisite'):
        _repair_declarations(raw, text, registry)  # No rejection under current admission.
    safe, rejected = _repair_declarations(raw, text, registry,
        historical_missing={(tool, 'window_size')})
    assert not safe.arguments and rejected == [(tool, 'window_size')]


@pytest.mark.parametrize('text,literal', [
    ('Use 2 attempts with end 1200.', '2'),
    ('Use -2 and end -1200.', '-2'),
    ('Use 0.2 with 10.2.', '0.2'),
    ('Use 1e-2, not 11e-2.', '1e-2'),
    ('Use 1/2 with 11/2.', '1/2'),
    ('Use 2 attempts and a 2-5 range.', '2'),
])
def test_numeric_quote_locations_use_whole_tokens(text, literal):
    declaration, = parse_decision(wire(literal, text), utterance=text).arguments
    assert declaration.start == text.index(literal)
    assert text[declaration.start:declaration.end] == literal


@pytest.mark.parametrize('text,literal', [
    ('Use 2 here and 2 there, end 1200.', '2'),
    ('Use -2 here and -2 there.', '-2'),
    ('Use 1.2 here and 1.2 there.', '1.2'),
    ('Use 1/2 here and 1/2 there.', '1/2'),
])
def test_independent_numeric_repetition_stays_ambiguous(text, literal):
    with pytest.raises(IntentError, match='ambiguous_parameter'):
        parse_decision(wire(literal, text), utterance=text)


def test_historical_substring_rejection_stays_readable():
    from agent.application.dialogue_execution import _repair_declarations
    text = 'Allow 2 mutation attempts with end 1200.'
    tool = 'optimize_regulatory_sequences'
    raw = wire('2', text, tool=tool, argument='iterations')
    safe, rejected = _repair_declarations(raw, text, build_default_tool_registry(),
        historical_missing={(tool, 'iterations')})
    assert not safe.arguments and rejected == [(tool, 'iterations')]


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
