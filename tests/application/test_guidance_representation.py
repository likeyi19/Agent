"""Independent reconstruction of Guidance's prompt-only representation.

Accepted metadata fixtures exercise application contracts, not scientific or
biological qualification. The decoder deliberately uses no production renderer
or catalog expansion helper.
"""
from copy import deepcopy
import json

import pytest

from agent.application import scientific_guidance as guidance
from agent.application.session_state import canonical, digest
from agent.orchestration import build_default_tool_registry
from agent.schemas.orchestration import _serialize
from benchmarks.interactive.candidates import CANDIDATES
from benchmarks.interactive.fixtures import scripted_guidance
from benchmarks.interactive.harness import run_attempt
from benchmarks.interactive.scenarios import Scenario, scenario_by_id
from test_dialogue_evidence import app, accepted


def expand_context(rendered):
    """Expand the documented model envelope independently, on JSON-native data."""
    context = deepcopy(rendered)
    readiness = context['readiness']
    if 'global_advisory_facts' not in readiness:
        return context
    context['readiness'] = {tool: dict(readiness['global_advisory_facts'], **row)
                            for tool, row in readiness['by_capability'].items()}
    return context


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def test_model_boundary_retains_authoritative_catalog_context_and_public_readiness(app, monkeypatch):
    from test_scientific_guidance import Model, ask, guard

    accepted(app, 'inspect_scATAC')
    app = guard(app, monkeypatch)
    model = Model(tools=('cluster_cells', 'inspect_scATAC'))
    result = ask(app, model=model)
    assert result.status == 'answered', result.text
    interaction = app.sessions.load('session').interactions[-1]
    public, _ = guidance.context(app.sessions, 'session', interaction)
    source = _serialize(public)
    before = _json(source)
    compact = guidance._model_context(public)
    decoded = expand_context(json.loads(_json(compact)))
    assert canonical(decoded) == canonical(source)
    assert _json(_serialize(public)) == before
    assert _json(guidance._model_context(public)) == _json(compact)
    assert list(decoded['catalog']['tools']) == list(source['catalog']['tools'])
    assert set(decoded['catalog']['tools']) == set(app.registry.names())
    for tool, metadata in source['catalog']['tools'].items():
        assert list(decoded['catalog']['tools'][tool][1]) == list(metadata[1])
    assert interaction.admitted['catalog_sha256'] == digest(public['catalog'])
    assert interaction.admitted['catalog_sha256'] == digest(compact['catalog'])
    for candidate in result.guidance.candidates:
        name = candidate.reference['capability']
        assert _serialize(candidate.readiness) == source['readiness'][name]
        assert candidate.reference['targets'] == interaction.admitted['targets']
    prompt, schema = model.requests[-1]
    payload = json.loads(prompt)
    assert payload['context'] == json.loads(_json(compact))
    assert model.calls[-1]['context'] == source
    assert payload['question'] == interaction.utterance
    assert payload['objective'] == interaction.admitted['focus']
    assert len(payload['instructions']) == 8
    assert list(payload).index('instructions') < list(payload).index('context')
    assert list(compact)[:3] == ['evidence', 'readiness', 'catalog']
    assert compact['catalog'] == source['catalog']
    assert compact['evidence'] == source['evidence']
    assert 'catalog_encoding' not in compact
    global_facts = compact['readiness']['global_advisory_facts']
    assert set(global_facts) == {'capability_registered', 'readiness', 'request_scope', 'accepted_evidence_handles'}
    assert global_facts['request_scope'] == 'no_execution_inputs_bound'
    assert global_facts['readiness'] == 'not_fully_checked'
    for row in compact['readiness']['by_capability'].values():
        assert set(row) == {'required_ports_without_supplied_request_source',
                            'required_scientific_parameters', 'explicit_request_source_choices'}
    # Positive descriptions of the unchanged prose contract, not a token blacklist.
    assert 'paragraph order supplies the structure' in payload['instructions'][1]
    assert 'including conditional passages' in payload['instructions'][2]
    assert 'standalone spelled-out numbers' in payload['instructions'][2]
    assert 'complete target, subject, predicate and value' in payload['instructions'][2]
    enum = schema['properties']['candidates']['items']['properties']['capability']['enum']
    assert enum == payload['offered_capabilities'] == sorted(app.registry.names())


@pytest.mark.parametrize('rows', [
    {},
    {'first': {'readiness': 'not_fully_checked', 'new_fact': {'why': None}}},
    {'first': {'common': True, 'status': 'unknown', 'new_fact': [None, 'literal']},
     'second': {'common': True, 'status': 'qualified', 'new_fact': [None, 'literal'],
                'extra_fact': {'missing_is_distinct': False}},
     'third': {'status': 'unknown', 'new_fact': [None, 'literal']}},
    {'first': {'scalar': True, 'nested': {'value': [True, 1, 1.0]}},
     'second': {'scalar': 1, 'nested': {'value': [1, 1.0, True]}},
     'third': {'scalar': 1.0, 'nested': {'value': [1.0, True, 1]}}},
])
def test_future_readiness_facts_missing_fields_and_differences_are_lossless(rows):
    repeated = {'long_parameter_name': {'$': 3, '$$': 'literal', '@0': 'literal',
                 'long_exact_scientific_description': 'Exact shared semantics remain unchanged.'}}
    original = dict(evidence={'targets': [{'handle': '$', 'subject': '@0'}]},
        catalog={'$': 9, '$$': 8, '@0': 'literal', 'tools': {'first': [repeated] * 8}},
        readiness=rows, limitations=['original limitation'], selected_evidence_count=1,
        captured_output_count=7, future_context={'ordered': ['z', 'a']})
    frozen = deepcopy(original)
    compact = guidance._model_context(original)
    decoded = expand_context(json.loads(_json(compact)))
    assert canonical(decoded) == canonical(original) == canonical(frozen)
    assert list(decoded['readiness']) == sorted(rows)
    assert decoded['future_context']['ordered'] == ['z', 'a']
    assert _json(guidance._model_context(original)) == _json(compact)
    assert compact['catalog'] == original['catalog']
    assert compact['evidence'] == original['evidence']
    assert 'catalog_encoding' not in compact
    assert all(key in {'capability_registered', 'readiness', 'request_scope', 'accepted_evidence_handles'}
               for key in compact['readiness']['global_advisory_facts'])


def test_only_genuinely_global_identical_facts_are_factored():
    rows = {
        'first': dict(capability_registered=True, readiness='not_fully_checked',
                      request_scope='no_execution_inputs_bound', accepted_evidence_handles=['t0'],
                      explicit_request_source_choices=[], future_fact='identical', scalar=True),
        'second': dict(capability_registered=1, readiness='not_fully_checked',
                       request_scope='no_execution_inputs_bound', accepted_evidence_handles=['t1'],
                       explicit_request_source_choices=[], future_fact='identical', scalar=1),
    }
    public = dict(evidence={}, catalog={'tools': {}}, readiness=rows)
    rendered = guidance._model_context(public)
    assert rendered['readiness']['global_advisory_facts'] == dict(
        readiness='not_fully_checked', request_scope='no_execution_inputs_bound')
    assert canonical(expand_context(rendered)) == canonical(public)
    for name, row in rendered['readiness']['by_capability'].items():
        assert row['accepted_evidence_handles'] == rows[name]['accepted_evidence_handles']
        assert row['future_fact'] == 'identical'
        assert row['explicit_request_source_choices'] == []
        assert 'scalar' in row and 'capability_registered' in row


def test_fixed_followup_restricts_offers_without_filtering_catalog(app, monkeypatch):
    from test_scientific_guidance import Model, ask, decision, guard
    from test_scientific_dialogue import annotation

    annotation(app)
    app = guard(app, monkeypatch)
    first = ask(app, utterance='What could I analyze for cluster 3?',
                model=Model(decision('r0', subject='3'), tools=('inspect_scATAC', 'cluster_cells')))
    original = first.guidance.candidates[0]
    name = original.reference['capability']
    model = Model(decision(candidate='first option'), tools=(name,))
    follow = ask(app, 'follow', 'Why would the first option help?', model)
    assert follow.status == 'answered', follow.text
    candidate, = follow.guidance.candidates
    assert candidate.reference == original.reference
    assert candidate.readiness == original.readiness
    prompt, schema = model.requests[-1]
    payload = json.loads(prompt)
    assert payload['offered_capabilities'] == [name]
    assert schema['properties']['candidates']['items']['properties']['capability']['enum'] == [name]
    assert set(expand_context(payload['context'])['catalog']['tools']) == set(app.registry.names())
    assert payload['objective'] == 'What could I analyze for cluster 3?'
    assert payload['context']['evidence']['targets'][0]['subject'] == '3'


@pytest.mark.parametrize('fixture,utterance,subject', [
    ('rollback', 'What analyses could help investigate the accepted inspection summary?', None),
    ('matrix', 'What analyses could help investigate the accepted matrix?', None),
    ('selection', 'What analyses could help investigate the accepted selection?', None),
    ('annotation', scenario_by_id('I13').utterance, '3'),
])
def test_representative_payloads_remain_direct_with_full_semantic_reconstruction(fixture, utterance, subject):
    class Witness:
        def complete(self, *, prompt, response_schema):
            self.prompt, self.schema = prompt, response_schema
            return json.dumps(scripted_guidance(json.loads(prompt)))

    witness = Witness()
    scenario = Scenario('IR2-' + fixture, 'guidance', utterance, fixture=fixture,
        expected={'kind': 'answer', 'referent': '@current_result', 'subject': subject},
        human_review_required=True)
    result = run_attempt(scenario, CANDIDATES[0], model=witness).to_dict()
    assert result['contract_success'] and result['admission_success'], result
    assert all(count == 0 for count in result['safety'].values())
    compact = json.loads(witness.prompt)
    expanded = dict(compact, context=expand_context(compact['context']))
    old_prompt = json.dumps(expanded, ensure_ascii=False)
    schema_bytes = len(json.dumps(witness.schema).encode())
    before = len(old_prompt.encode()) + schema_bytes
    after = len(witness.prompt.encode()) + schema_bytes
    assert after < before, (fixture, before, after)
    assert compact['context']['catalog'] == expanded['context']['catalog']
    assert compact['context']['evidence'] == expanded['context']['evidence']
    assert set(expanded['context']['catalog']['tools']) == set(build_default_tool_registry().names())
    candidate, = result['guidance_candidates']['candidates']
    assert candidate['readiness'] == expanded['context']['readiness']['inspect_scATAC']
    # not_allowed differs from missing advisory request sources.
    for tool, (_, ports, _, _) in expanded['context']['catalog']['tools'].items():
        missing = expanded['context']['readiness'][tool]['required_ports_without_supplied_request_source']
        assert all(name not in missing for name, port in ports.items() if port[1] == 'not_allowed')
