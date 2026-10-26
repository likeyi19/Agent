"""Exact accepted references across answer, categorical and delivery owners.

Fixtures publish simulated accepted records and small H5ADs. No native science
or live provider qualification is implied.
"""
import pytest

from agent.application import result_reading
from agent.application.turn_decisions import decision_schema
from test_categorical_evidence import publication, selected
from test_dialogue_evidence import app, accepted, forbid_work
from test_interaction_context import cooutputs
from test_scientific_dialogue import Model, annotation, ask, question


@pytest.mark.parametrize('selection', ['group', 'first_alias', 'second_alias'])
@pytest.mark.parametrize('names', [('Z summary', 'A metadata'), ('A metadata', 'Z summary')])
def test_historical_result_and_output_aliases_have_one_exact_source(app, monkeypatch, selection, names):
    old, _, _ = cooutputs(app, monkeypatch, names=names, overrides={'n_cells': 17})
    accepted(app, 'inspect_scATAC', name='newer', overrides={'n_cells': 23})
    guarded = forbid_work(app, monkeypatch)
    before = guarded.sessions.load('session')

    def choose(prompt):
        group = next(g for g in prompt['dialogue']['result_groups'] if g['version'] == 1)
        handle = group['handle'] if selection == 'group' else group['outputs'][1 if selection == 'second_alias' else 0]
        return question(handle)

    result = ask(guarded, 'historical', 'Describe the earlier assay result.', Model(choose, ['n_cells']))
    assert result.status == 'answered', result
    claim, = result.scientific.claims
    assert claim.value == 17 and claim.source['revision_id'] == old
    assert claim.source['output_locator']['name'] == 'A metadata'
    after = guarded.sessions.load('session')
    assert (after.revisions, after.turns, after.generation) == (before.revisions, before.turns, before.generation)


@pytest.mark.parametrize('handle', ['r1', 'g1'])
def test_exact_offered_independent_result_does_not_require_internal_names(app, monkeypatch, handle):
    accepted(app, 'inspect_scATAC', copies=2)
    guarded = forbid_work(app, monkeypatch)
    result = ask(guarded, 'selected', 'Explain the second accepted measurement.', Model(question(handle), ['n_cells']))
    assert result.status == 'answered', result
    assert result.scientific.claims[0].source['output_locator']['step_id'] == 'step1'


def test_live_subject_domains_match_the_selected_result_owner(app, monkeypatch):
    annotation(app)
    accepted(app, 'cluster_cells', name='partition')

    def choose(prompt):
        variants = decision_schema(prompt)['properties']['decision']['anyOf']
        scientific = next(v for v in variants if v['properties']['kind']['enum'] == ['answer_scientific'])
        domains = scientific['properties']['target']['anyOf']
        groups = prompt['dialogue']['result_groups']
        for group in groups:
            for handle in (group['handle'], *group['outputs']):
                domain, = [row['properties']['subject'] for row in domains
                           if handle in row['properties']['output']['enum']]
                if group['tool'] == 'annotate_scATAC_cell_types':
                    assert domain['anyOf'][1] == {'type': 'null'}
                    assert '3' in domain['anyOf'][0]['enum']
                    assert 'invented subject' not in domain['anyOf'][0]['enum']
                else:
                    assert domain == {'type': 'null'}
        return dict(kind='clarify', reason='unsupported_intent')

    ask(forbid_work(app, monkeypatch), 'domains', 'Explain the accepted annotations and partition.', Model(choose))


@pytest.mark.parametrize('section', ['category_counts', 'category_association'])
@pytest.mark.parametrize('reverse', [False, True])
def test_distinct_categorical_operands_can_share_one_accepted_artifact(app, monkeypatch, section, reverse):
    rid, _, _ = publication(app)
    guarded = forbid_work(app, monkeypatch)
    before = guarded.sessions.load('session')
    fields = ['result:primary_annotation', 'source:treatment']
    if reverse:
        fields.reverse()
    detail = selected(section, fields[0], fields[1])
    observed = []

    def reply(prompt):
        claims = [c for c in prompt['evidence']['claims'] if c['field'].startswith('detail.category_')]
        observed.extend(claims)
        return dict(support='supported', paragraphs=[dict(kind='finding',
            text='The saved categorical observations provide the descriptive table with its recorded coverage.',
            citations=[c['claim_id'] for c in claims], comparisons=[])])

    result = ask(guarded, 'pair', 'Compare the computed labels and the supplied treatment categories.',
        Model(dict(question('r0', comparison=dict(output='g0', subject=None)), detail=detail), response=reply))
    assert result.status == 'answered', result
    assert observed and {c.source['revision_id'] for c in result.scientific.claims} == {rid}
    if section == 'category_counts':
        assert {c['target'] for c in observed} == {'t0', 't1'}
    else:
        assert observed[0]['value']['n_paired'] == 3
    after = guarded.sessions.load('session')
    assert (after.revisions, after.turns, after.generation) == (before.revisions, before.turns, before.generation)


@pytest.mark.parametrize('failure', ['unknown_output', 'unknown_field', 'self_summary', 'same_field'])
def test_invalid_reference_operands_reject_before_generation_without_science(app, monkeypatch, failure):
    publication(app)
    choice = question('g0')
    if failure == 'unknown_output':
        choice['target']['output'] = 'r999'
    elif failure == 'unknown_field':
        choice['detail'] = selected(field='source:absent')
    else:
        choice['comparison'] = dict(output='r0', subject=None)
        if failure == 'same_field':
            choice['detail'] = selected('category_association', 'result:primary_annotation', 'result:primary_annotation')
    guarded = forbid_work(app, monkeypatch)
    before = guarded.sessions.load('session')
    model = Model(choice)
    result = ask(guarded, failure, 'Describe the selected accepted observations.', model)
    assert result.status != 'answered' and result.scientific is None
    assert len(model.calls) == 1
    after = guarded.sessions.load('session')
    assert (after.revisions, after.turns, after.generation) == (before.revisions, before.turns, before.generation)


def test_answer_and_delivery_resolve_captured_output_aliases_identically(app, monkeypatch):
    cooutputs(app, monkeypatch)
    guarded = forbid_work(app, monkeypatch)
    result = ask(guarded, 'read', 'Explain the saved observation.', Model(question('r0'), ['n_cells']))
    assert result.status == 'answered'
    state = guarded.sessions.load('session')
    interaction = state.interactions[-1]
    for handle in ('r0', 'r1', 'g0'):
        output = result_reading._captured_output(guarded.sessions, state, interaction, handle)
        assert output['name'] == interaction.admitted['target']['output_name']
        assert output['accepted_step_sha256'] == interaction.admitted['target']['accepted_step_sha256']
    with pytest.raises(ValueError, match='captured accepted result'):
        result_reading._captured_output(guarded.sessions, state, interaction, 'g999')
