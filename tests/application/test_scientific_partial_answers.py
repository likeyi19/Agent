"""R5 unit retention under scripted review; this does not qualify model judgement."""
import json

import pytest

from agent.application import ResearchAgentApplication, SessionConflictError
from agent.application import scientific_dialogue as sd
from agent.application.interactive_schemas import PresentedResponse
from test_dialogue_evidence import app, accepted, forbid_work
from test_scientific_dialogue import ask, question
from test_scientific_explanations import paragraph


class Reviewed:
    def __init__(self, author, verdicts, *, integrity='intact', failure=None, after_review=None):
        self.author, self.verdicts, self.integrity = author, verdicts, integrity
        self.failure, self.after_review, self.calls = failure, after_review, []

    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        self.calls.append(value)
        if 'turn_schema_version' in value:
            return json.dumps(dict(turn_schema_version=1, decision=question()))
        if 'dialogue_schema_version' in value:
            assert value['dialogue_schema_version'] == 4
            return json.dumps(self.author(value['evidence']))
        assert value['grounding_review_version'] == 3
        if self.failure == 'provider':
            raise RuntimeError('PRIVATE_PROVIDER_RESPONSE /private/scientific-path')
        verdicts = {row['id']: dict(verdict=verdict, independent=independent)
                    for row, (verdict, independent) in zip(value['paragraphs'], self.verdicts)}
        result = dict(integrity=self.integrity, verdicts=verdicts)
        if self.failure == 'missing':
            verdicts.pop(next(reversed(verdicts)))
        elif self.failure == 'extra':
            verdicts['answer:99'] = dict(verdict='entailed', independent=True)
        elif self.failure == 'missing_independence':
            next(iter(verdicts.values())).pop('independent')
        elif self.failure == 'non_boolean_independence':
            next(iter(verdicts.values()))['independent'] = 1
        elif self.failure == 'missing_integrity':
            result.pop('integrity')
        if self.after_review:
            self.after_review()
        return json.dumps(result)


def mixed(evidence):
    ids = {claim['field']: claim['claim_id'] for claim in evidence['claims']}
    return dict(support='supported', paragraphs=[
        paragraph('The accepted dataset contains 4 cells.', [ids['n_cells']]),
        paragraph('It retains 7 measured features.', [ids['n_features']]),
        paragraph('Its 12 nonzero entries prove a drug response.', [ids['nnz']]),
        paragraph('These computational results do not independently validate cell types or treatment response.',
                  kind='limitation'),
    ])


def prepare(app, monkeypatch):
    revision, path, _ = accepted(app, 'inspect_scATAC', overrides=dict(n_cells=4, n_features=7, nnz=12))
    guarded = forbid_work(app, monkeypatch)
    protected = {p: p.read_bytes() for p in guarded._workspace.runs.rglob('*') if p.is_file()}
    return guarded, revision, path, protected


def assert_single_review(model):
    assert [next(key for key in ('turn_schema_version', 'dialogue_schema_version', 'grounding_review_version')
                 if key in call) for call in model.calls] == [
                     'turn_schema_version', 'dialogue_schema_version', 'grounding_review_version']


def test_partial_answer_keeps_complete_supported_units_and_exact_citation_union(app, monkeypatch):
    guarded, revision, _, protected = prepare(app, monkeypatch)
    before = guarded.sessions.load('session')
    model = Reviewed(mixed, [('entailed', True), ('entailed', True), ('unsupported', True), ('entailed', True)])
    outcome = ask(guarded, 'mixed', 'What is measured, and does it prove treatment response?', model)
    assert outcome.status == 'answered' and outcome.scientific.support == 'insufficient_evidence'
    assert outcome.scientific.explanation == (
        'The accepted dataset contains 4 cells.\n\nIt retains 7 measured features.\n\n'
        'These computational results do not independently validate cell types or treatment response.')
    assert {claim.field for claim in outcome.scientific.claims} == {'n_cells', 'n_features'}
    assert all(claim.source['revision_id'] == revision for claim in outcome.scientific.claims)
    assert '12 nonzero' not in outcome.text
    assert sd.PARTIAL_NOTICE in outcome.text and sd.PARTIAL_NOTICE in outcome.scientific.limitations
    after = guarded.sessions.load('session')
    assert (after.turns, after.revisions, after.generation) == (before.turns, before.revisions, before.generation)
    assert {p: p.read_bytes() for p in protected} == protected
    assert_single_review(model)


@pytest.mark.parametrize('rejected_first,positive_independent,expected', [
    (False, True, 'answered'),   # Unsupported conclusion follows an independently useful observation.
    (True, True, 'answered'),    # Source order does not decide independence.
    (False, False, 'unavailable'),
    (True, False, 'unavailable'),  # A valid-looking consequence depends on the rejected premise.
])
def test_dependencies_are_reviewed_not_inferred_from_paragraph_order(app, monkeypatch,
        rejected_first, positive_independent, expected):
    guarded, _, _, _ = prepare(app, monkeypatch)
    def author(evidence):
        citation = next(c['claim_id'] for c in evidence['claims'] if c['field'] == 'n_cells')
        rows = [paragraph('The accepted dataset contains 4 cells.', [citation]),
                paragraph('Those cells prove the treatment works.', [citation])]
        return dict(support='supported', paragraphs=list(reversed(rows)) if rejected_first else rows)
    verdicts = [('entailed', positive_independent), ('unsupported', False)]
    if rejected_first:
        verdicts.reverse()
    model = Reviewed(author, verdicts)
    outcome = ask(guarded, 'dependencies', 'Explain the result.', model)
    assert outcome.status == expected
    if expected == 'answered':
        assert outcome.scientific.explanation == 'The accepted dataset contains 4 cells.'
    else:
        assert outcome.scientific is None and '4 cells' not in outcome.text
    assert_single_review(model)


@pytest.mark.parametrize('integrity,reason', [
    ('invalid_attribution', 'A statement uses another version or population.'),
    ('incoherent', 'A required caveat is omitted.'),
    ('incoherent', 'An ambiguous referent depends on rejected text.'),
])
def test_global_integrity_prevents_partial_publication(app, monkeypatch, integrity, reason):
    guarded, _, _, _ = prepare(app, monkeypatch)
    model = Reviewed(mixed, [('entailed', True), ('entailed', True), ('unsupported', True), ('entailed', True)],
                     integrity=integrity)
    outcome = ask(guarded, 'integrity', reason, model)
    assert outcome.status == 'unavailable' and outcome.scientific is None
    assert '4 cells' not in outcome.text
    assert_single_review(model)


def test_invalid_reference_in_rejected_unit_still_prevents_any_review(app, monkeypatch):
    guarded, _, _, _ = prepare(app, monkeypatch)
    def forged(evidence):
        result = mixed(evidence)
        result['paragraphs'][2]['citations'] = ['unoffered']
        return result
    model = Reviewed(forged, [('entailed', True), ('entailed', True), ('unsupported', True), ('entailed', True)])
    outcome = ask(guarded, 'forged', 'Explain.', model)
    assert outcome.status == 'unavailable' and outcome.scientific is None
    assert len(model.calls) == 2


@pytest.mark.parametrize('failure', [
    'provider', 'missing', 'extra', 'missing_independence', 'non_boolean_independence', 'missing_integrity',
])
def test_incomplete_review_cannot_publish_apparently_positive_units(app, monkeypatch, caplog, failure):
    guarded, _, _, _ = prepare(app, monkeypatch)
    model = Reviewed(mixed, [('entailed', True), ('entailed', True), ('unsupported', True), ('entailed', True)],
                     failure=failure)
    outcome = ask(guarded, 'bad-review', 'Explain.', model)
    assert outcome.status == 'unavailable' and outcome.scientific is None
    assert 'PRIVATE_PROVIDER_RESPONSE' not in outcome.text + caplog.text
    assert '/private/scientific-path' not in outcome.text + caplog.text
    assert_single_review(model)


def test_evidence_changed_after_partial_review_withholds_every_unit(app, monkeypatch):
    guarded, _, path, _ = prepare(app, monkeypatch)
    model = Reviewed(mixed, [('entailed', True), ('entailed', True), ('unsupported', True), ('entailed', True)],
                     after_review=lambda: path.write_bytes(path.read_bytes() + b' '))
    outcome = ask(guarded, 'changed', 'Explain.', model)
    assert outcome.status == 'unavailable' and outcome.scientific is None
    assert '4 cells' not in outcome.text
    assert_single_review(model)


def test_no_positive_unit_preserves_controlled_withholding(app, monkeypatch):
    guarded, _, _, _ = prepare(app, monkeypatch)
    model = Reviewed(mixed, [('unsupported', True)] * 4)
    outcome = ask(guarded, 'none', 'Explain.', model)
    assert outcome.status == 'unavailable' and outcome.scientific is None
    assert_single_review(model)


def test_complete_coherent_answer_keeps_dependencies_and_exact_text(app, monkeypatch):
    guarded, _, _, _ = prepare(app, monkeypatch)
    def author(evidence):
        citation = next(c['claim_id'] for c in evidence['claims'] if c['field'] == 'n_cells')
        return dict(support='supported', paragraphs=[
            paragraph('The accepted dataset contains 4 cells.', [citation]),
            paragraph('This count describes the same accepted dataset.', [citation]),
        ])
    model = Reviewed(author, [('entailed', True), ('entailed', False)])
    outcome = ask(guarded, 'valid', 'Explain.', model)
    expected = 'The accepted dataset contains 4 cells.\n\nThis count describes the same accepted dataset.'
    assert outcome.status == 'answered' and outcome.scientific.support == 'supported'
    assert outcome.scientific.explanation == outcome.text == expected
    assert sd.PARTIAL_NOTICE not in outcome.scientific.limitations
    assert_single_review(model)


def test_new_partial_policy_preserves_saved_historical_presentation(app, monkeypatch):
    guarded, _, _, _ = prepare(app, monkeypatch)
    def original(evidence):
        citation = next(c['claim_id'] for c in evidence['claims'] if c['field'] == 'n_cells')
        return dict(support='supported', paragraphs=[
            paragraph('The accepted dataset contains 4 cells.', [citation])])
    old = Reviewed(original, [('entailed', True)])
    first = ask(guarded, 'historical', 'Explain the old accepted state.', old)
    presentation = PresentedResponse('answer', 'answered', first.text).to_dict()
    guarded.sessions.store_presentation('session', 'historical', presentation)
    saved = guarded.sessions.load('session').interactions[-1].presentation
    model = Reviewed(mixed, [('entailed', True), ('entailed', True), ('unsupported', True), ('entailed', True)])
    assert ask(guarded, 'new', 'Explain a supported subset.', model).status == 'answered'
    reopened = ResearchAgentApplication(guarded.workspace_root, registry=guarded.registry)
    assert reopened.sessions.load('session').interactions[-2].presentation == saved
    with pytest.raises(SessionConflictError):
        reopened.sessions.store_presentation('session', 'historical', dict(presentation, text='Rewritten history'))
    assert reopened.sessions.load('session').interactions[-2].presentation == saved
