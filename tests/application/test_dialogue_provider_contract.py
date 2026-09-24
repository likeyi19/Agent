"""Strict decision unions and provider-neutral branch normalization."""
import json

import pytest

from agent.application.turn_decisions import decision_schema, parse_decision, IntentError
from test_dialogue_evidence import app, accepted, forbid_work
from test_scientific_dialogue import Model, annotation, ask


@pytest.mark.parametrize('with_operation', [False, True])
def test_strict_union_has_distinct_discriminators(with_operation):
    operations = [dict(handle='op.0', parameters={'min_tss_enrichment':'4'})] if with_operation else []
    schema = decision_schema(dict(bases={'revision':dict(operations=operations)},
        relations=['current'], dialogue=dict(outputs=[dict(handle='r0')], tools=['inspect_scATAC'])))
    variants = schema['properties']['decision']['anyOf']
    tags = [tag for v in variants for tag in v['properties']['kind']['enum']]
    assert len(tags) == len(set(tags))
    assert {'answer_scientific', 'execute_plan'} <= set(tags)
    for variant in variants:
        assert variant['additionalProperties'] is False
        assert set(variant['required']) == set(variant['properties'])
    by_kind = {v['properties']['kind']['enum'][0]: v for v in variants}
    assert set(by_kind['answer_scientific']['properties']) == {'kind', 'target', 'comparison', 'focus'}
    assert set(by_kind['execute_plan']['properties']) == {'kind', 'target'}


@pytest.mark.parametrize('tag,decision', [
    ('answer_scientific', dict(kind='answer',intent='scientific',target=dict(output='r0',subject='7'),comparison=None,focus='question')),
    ('execute_plan', dict(kind='execute',base='current',operation='plan',target='inspect_scATAC',delta=None)),
])
def test_wire_tags_preserve_existing_decisions_and_closed_validation(tag, decision):
    def parse(d):
        return parse_decision(json.dumps(dict(turn_schema_version=1,decision=d)))
    wire = dict(decision,kind=tag)
    assert parse(wire) == parse(decision)
    derived = {'intent'} if tag == 'answer_scientific' else {'operation', 'base', 'delta'}
    minimal = {k:v for k,v in wire.items() if k not in derived}
    assert parse(minimal) == parse(decision)
    with pytest.raises(IntentError): parse(dict(minimal,unexpected=True))
    with pytest.raises(IntentError): parse(dict(wire,unexpected=True))
    with pytest.raises(IntentError):
        parse(dict(wire, **({'intent':'version'} if tag=='answer_scientific' else {'operation':'op.0'})))


@pytest.mark.parametrize('assertions', [
    {'base':'current'}, {'operation':'plan'}, {'delta':None},
    {'base':'current','operation':'plan'}, {'base':'current','delta':None},
    {'operation':'plan','delta':None},
    {'base':'parent','operation':'plan','delta':None},
    {'base':'current','operation':'plan','delta':{}},
])
def test_partial_or_contradictory_execution_assertions_fail_closed(assertions):
    wire = dict(kind='execute_plan',target='inspect_scATAC',**assertions)
    with pytest.raises(IntentError):
        parse_decision(json.dumps(dict(turn_schema_version=1,decision=wire)))


def test_minimal_scientific_wire_preserves_durable_identity_and_followup(app, monkeypatch):
    rid = annotation(app)
    app = forbid_work(app, monkeypatch)
    def wire(output, subject):
        return dict(kind='answer_scientific',target=dict(output=output,subject=subject),
                    comparison=None,focus='question')
    first = ask(app,'first','What does cluster 3 show?',Model(wire('r0','3'),['primary_annotation']))
    follow = ask(app,'follow','Why?',Model(wire('@focus','@focus'),['primary_annotation']))
    assert first.status == follow.status == 'answered'
    state = app.sessions.load('session')
    a, b = state.interactions[-2:]
    assert a.admitted['intent'] == b.admitted['intent'] == 'scientific'
    assert a.admitted['target'] == b.admitted['target']
    assert b.admitted['target']['revision_id'] == rid
    assert b.admitted['target']['subject'] == '3'
    assert b.admitted['predecessor'] == 'first'


@pytest.mark.parametrize('wire,utterance', [
    (dict(kind='execute_plan',target='invented_tool'), 'Run invented_tool.'),
    (dict(kind='execute_plan',target='inspect_scATAC'), 'What does this result show?'),
    (dict(kind='answer_scientific',target=dict(output='r99',subject=None),comparison=None,focus='question'), 'What does this result show?'),
    (dict(kind='answer_scientific',target=dict(output='r0',subject='99'),comparison=None,focus='question'), 'What does cluster 99 show?'),
])
def test_minimal_wire_does_not_bypass_semantic_admission(app, monkeypatch, wire, utterance):
    annotation(app)
    model = Model(wire)
    result = ask(forbid_work(app,monkeypatch),'invalid',utterance,model)
    assert result.status == 'clarification'
    assert len(model.calls) == 1


def test_minimal_execution_cannot_handoff_guidance_candidate(app, monkeypatch):
    from test_scientific_guidance import Model as GuidanceModel, guard
    accepted(app, 'inspect_scATAC')
    app = guard(app, monkeypatch)
    assert ask(app,'guide','What should I analyze next?',GuidanceModel()).guidance
    result = ask(app,'run','Run option 1.',Model(dict(kind='execute_plan',target='inspect_scATAC')))
    assert result.status == 'clarification'
    assert result.clarification.reason == 'unsupported_intent'
