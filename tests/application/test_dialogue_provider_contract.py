"""Regression for Groq strict anyOf discriminator_value_overlap rejection."""
import json

import pytest

from agent.application.turn_decisions import decision_schema, parse_decision, IntentError


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


@pytest.mark.parametrize('tag,decision', [
    ('answer_scientific', dict(kind='answer',intent='scientific',target=dict(output='r0',subject='7'),comparison=None,focus='question')),
    ('execute_plan', dict(kind='execute',base='current',operation='plan',target='inspect_scATAC',delta=None)),
])
def test_wire_tags_preserve_existing_decisions_and_closed_validation(tag, decision):
    def parse(d):
        return parse_decision(json.dumps(dict(turn_schema_version=1,decision=d)))
    wire = dict(decision,kind=tag)
    assert parse(wire) == parse(decision)
    with pytest.raises(IntentError): parse(dict(wire,unexpected=True))
    with pytest.raises(IntentError):
        parse(dict(wire, **({'intent':'version'} if tag=='answer_scientific' else {'operation':'op.0'})))
