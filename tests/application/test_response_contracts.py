"""Closed answer contracts and deterministic wording, without scientific fixtures."""
from dataclasses import replace
import json
import pytest
from agent.application import Answer, TurnOutcome
from agent.application.turn_decisions import parse_decision, IntentError, decision_schema
from agent.application.response_facts import TurnResponseFacts, OutputFact, WorkFact, ReuseFact
from agent.application.responses import render, clarification_text, UNSUPPORTED, admit_answer
from agent.application.turn_decisions import Clarify
from agent.application.session_state import Interaction


def envelope(decision):
    return json.dumps(dict(turn_schema_version=1, decision=decision))


@pytest.mark.parametrize('intent', ('version','matrix','selection','thresholds','changes','execution','reuse','provenance','verification','unsupported'))
def test_answer_closed_roundtrip(intent):
    decision = dict(kind='answer',intent=intent,relation='current',technical=False)
    assert parse_decision(envelope(decision)) == Answer(intent)


@pytest.mark.parametrize('change', ({'intent':'biology'}, {'relation':'newest'}, {'technical':'yes'},
                                  {'payload':{'guess':'marker'}}, {'intent':'comparison'}))
def test_invalid_answer_rejected(change):
    with pytest.raises(IntentError):
        parse_decision(envelope(dict(kind='answer',intent='version',relation='current',technical=False) | change))


def test_answer_without_scientific_operations_and_backwards_outcome():
    schema = decision_schema(dict(bases={}, relations=('current',)))
    variants = schema['properties']['decision']['anyOf']
    assert any(v['properties']['kind']['enum'] == ['answer'] for v in variants)
    assert TurnOutcome('navigate','activated').text == ''


def test_render_no_artifacts_no_invented_defaults_and_no_technical_ids():
    facts = TurnResponseFacts('matrix','secret-rid',2,True,'current',(),())
    assert 'no matrix' in render(facts)
    assert 'secret-rid' not in render(facts)
    assert 'secret-rid' in render(facts,technical=True)
    assert 'no active selection thresholds' in render(replace(facts,intent='thresholds'))
    assert render(replace(facts,intent='unsupported')) == UNSUPPORTED
    assert 'label' not in UNSUPPORTED


def test_work_wording_does_not_infer_production_or_reconstruction():
    output = OutputFact('QC','run','step','a'*64,'b'*64,'c'*64,'scientific-authority.v2')
    facts = TurnResponseFacts('reuse','revision',2,True,'current',(output,),(),
        work=(WorkFact('matrix','m',1,1,0),), reuse=(ReuseFact(output,'m',True,True),))
    text = render(facts)
    assert 'tool attempts for matrix' in text and 'No QC production step' in text
    assert 'call totals were not recorded' in text
    assert 'reconstruction was avoided' not in text and 'cached' not in text
    assert facts.owner_reconstruction_calls is None and facts.production_calls is None
    assert 'same-run recovery' in render(replace(facts,work=(WorkFact('matrix','m',1,1,1),)))
    assert render(replace(facts,is_active=False)).startswith('For historical version 2 (not active):')


def test_grounded_relations_technical_optin_and_minimal_clarification():
    interaction = Interaction('turn','How is this version different from the parent?', 'r3',3,
        dict(relations={'current':'r3','parent':'r1','previous_active':'r2'},bases={}))
    assert admit_answer(interaction,Answer('comparison','parent'))['comparison_id'] == 'r1'
    assert admit_answer(replace(interaction,utterance='Was the parent result verified?'),Answer('verification','parent'))['revision_id'] == 'r1'
    with pytest.raises(IntentError):
        admit_answer(replace(interaction,utterance='What changed from the previous version?'),Answer('comparison','parent'))
    with pytest.raises(IntentError): admit_answer(interaction,Answer('comparison','parent',True))
    assert clarification_text(Clarify('missing_parameter_value',('min_tss_enrichment',),True)) == 'What value should I use for minimum tss enrichment?'
    assert 'parent version or the previously active version' in clarification_text(Clarify('ambiguous_revision',('parent','previous_active')))


def test_no_previous_revision_does_not_offer_nonexistent_choices():
    from types import SimpleNamespace
    from agent.application.responses import answer_outcome, UNAVAILABLE
    state = SimpleNamespace(active_revision_id='initial', navigation=(),
        revisions=(SimpleNamespace(revision_id='initial',parent_revision_id=None),))
    sessions = SimpleNamespace(load=lambda _:state)
    outcome = answer_outcome(sessions,'analysis',Answer('comparison','previous'))
    assert outcome.text == UNAVAILABLE and not outcome.clarification.choices
    assert clarification_text(Clarify('ambiguous_revision')) == UNAVAILABLE
