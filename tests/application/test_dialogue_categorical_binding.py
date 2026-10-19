"""Categorical checks follow displayed prose, without guessing its semantics."""
import pytest

from agent.application import scientific_dialogue as sd


def evidence(values=('independent', 'independent')):
    claims = tuple(sd.ScientificClaim(f'c{i}', f't{i}', None, 'method', value,
        dict(revision_id=f'revision-{i}', evidence_sha256=str(i) * 64,
             pointer='/steps/0/facts/method')) for i, value in enumerate(values))
    public = dict(targets=[dict(handle=c.target, label=f'recorded revision {i}')
                          for i, c in enumerate(claims)],
        claims=[c.to_dict() for c in claims],
        meanings={'scope': 'Method terminology alone does not establish correspondence.'},
        limitations=[], scope='accepted_persisted_summary',
        comparison=dict(mode='direct_accepted_operands', pairs=[dict(
            id='p0', left='c0', right='c1',
            relationship='equal' if values[0] == values[1] else 'different')]))
    return public, claims


def response(*parts):
    return dict(support='supported', paragraphs=[dict(parts=[
        dict(kind='comparison', id='p0'), *parts])])


def text(value):
    return dict(kind='text', text=value)


@pytest.mark.parametrize('separator', [None, 'claim', 'meaning', 'comparison'])
def test_no_categorical_word_is_manufactured_across_parts(separator):
    public, claims = evidence()
    parts = [text('Use the accepted records in')]
    if separator:
        parts.append(dict(kind=separator, id=dict(claim='c0', meaning='scope', comparison='p0')[separator]))
    parts.append(text('dependent explanations with care.'))
    result = sd.render_response(response(*parts), public, claims)
    assert result.claims == claims
    assert result.explanation.startswith('[recorded revision 0: method = "independent"] equal')
    if separator is None:
        assert 'records in dependent explanations' in result.explanation


@pytest.mark.parametrize('parts', [
    [text('The assignment is B cell.')],
    [text('The assignment is B'), text('cell.')],
    [text('The assignment is'), text('B'), text('cell.')],
])
def test_multiword_category_cannot_escape_by_splitting_text(parts):
    public, claims = evidence(('B cell', 'B cell'))
    with pytest.raises(sd._AnswerFailure) as failure:
        sd.render_response(response(*parts), public, claims)
    assert (failure.value.stage, failure.value.reason) == ('factual_binding', 'unbound_categorical_value')


@pytest.mark.parametrize('values', [('leiden', 'leiden'), ('Scanpy', 'other_backend'),
                                   ('assigned', 'unassigned')])
def test_bound_categories_and_reviewed_meanings_preserve_exact_revision_sources(values):
    public, claims = evidence(values)
    # Reviewed meanings may explain method names; text is connective prose.
    public['meanings']['method'] = f'The recorded method term is {values[0]}.'
    result = sd.render_response(response(text('The accepted methods are recorded above.'),
        dict(kind='claim', id='c0'), dict(kind='meaning', id='method')), public, claims)
    assert result.claims == claims  # Repeated identical values do not merge identities.
    assert [c.source['revision_id'] for c in result.claims] == ['revision-0', 'revision-1']
    assert [c.source['pointer'] for c in result.claims] == ['/steps/0/facts/method'] * 2
    assert f'method term is {values[0]}' in result.explanation


@pytest.mark.parametrize('phrase', ['The method is leiden.', 'Leiden is a method name.',
    'The historical result used leiden.', 'The current result used louvain.'])
def test_raw_category_is_not_exempted_by_a_bound_claim_or_another_revision(phrase):
    public, claims = evidence(('leiden', 'louvain'))
    with pytest.raises(sd._AnswerFailure, match='unbound_categorical_value'):
        sd.render_response(response(text(phrase)), public, claims)


@pytest.mark.parametrize('field,value', [('value', 'invented_method'), ('target', 't1'),
                                        ('source', {'revision_id': 'revision-1'}), ('id', 'unoffered')])
def test_category_reference_cannot_substitute_value_or_attribution(field, value):
    public, claims = evidence(('leiden', 'louvain'))
    part = dict(kind='claim', id='c0')
    part[field] = value
    with pytest.raises((sd.IntentError, ValueError)):
        sd.render_response(response(part), public, claims)


@pytest.mark.parametrize('literal', ['99', 'ninety nine', 'zero', '/private/input', 'https://example.org'])
def test_numeric_and_path_restrictions_remain(literal):
    public, claims = evidence()
    with pytest.raises(sd._AnswerFailure, match='unbound_factual_literal'):
        sd.render_response(response(text(literal)), public, claims)


def test_missing_comparison_operand_remains_rejected():
    public, claims = evidence()
    result = response()
    result['paragraphs'][0]['parts'] = [dict(kind='claim', id='c0')]
    with pytest.raises(sd._AnswerFailure, match='missing_operand'):
        sd.render_response(result, public, claims)


def test_raw_method_mention_is_a_contract_violation_not_proof_of_a_false_assertion():
    public, claims = evidence(('leiden', 'leiden'))
    with pytest.raises(sd._AnswerFailure, match='unbound_categorical_value'):
        sd.render_response(response(text('Leiden terminology appears in the accepted records.')), public, claims)


class CapturedResponse:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def complete(self, **kwargs):
        import json
        self.calls += 1
        return json.dumps(self.value)


@pytest.mark.parametrize('values,prose', [
    (('assigned', 'unassigned'), 'The cells were assigned.'),
    (('PCA', 'EpiZoo'), 'PCA and EpiZoo have identical biological coordinates.'),
    (('leiden', 'louvain'), 'The historical result used leiden and had 999 clusters.'),
])
def test_rejected_prose_recovers_only_exact_attributed_facts(values, prose):
    public, claims = evidence(values)
    model = CapturedResponse(response(text(prose)))
    result = sd.generate(model, 'Explain these results.', 'summary', public, claims)
    assert model.calls == 1
    assert result.support == 'insufficient_evidence'
    assert result.claims == claims
    assert prose not in result.explanation and '999' not in result.explanation
    assert 'identical biological coordinates' not in result.explanation
    assert '[recorded revision 0: method = ' in result.explanation
    assert [c.source['revision_id'] for c in result.claims] == ['revision-0', 'revision-1']


@pytest.mark.parametrize('part', [dict(kind='claim', id='forged'),
    dict(kind='claim', id='c0', value='invented'), dict(kind='comparison', id='unoffered')])
def test_recovery_cannot_hide_later_invalid_evidence(part):
    public, claims = evidence(('assigned', 'unassigned'))
    value = response(text('assigned'))
    value['paragraphs'].append(dict(parts=[part]))
    with pytest.raises((sd.IntentError, ValueError)):
        sd.generate(CapturedResponse(value), 'Explain.', 'summary', public, claims)


def test_recovery_preserves_comparison_operand_requirement():
    public, claims = evidence(('PCA', 'EpiZoo'))
    value = dict(support='supported', paragraphs=[dict(parts=[
        dict(kind='claim', id='c0'), text('PCA is better than EpiZoo.')])])
    with pytest.raises(sd._AnswerFailure, match='missing_operand'):
        sd.generate(CapturedResponse(value), 'Compare.', 'summary', public, claims)


def test_unsupported_prose_without_facts_still_fails():
    public, claims = evidence()
    value = dict(support='insufficient_evidence', paragraphs=[dict(parts=[text('99 cells')])])
    with pytest.raises(sd._AnswerFailure, match='unbound_factual_literal'):
        sd.generate(CapturedResponse(value), 'Explain.', 'summary', public, claims)
