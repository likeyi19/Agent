"""R5 grounding contract; scripted reviewers establish software behavior only."""
import json
from dataclasses import replace

import pytest

from agent.application import scientific_dialogue as sd
from test_dialogue_evidence import app, accepted, forbid_work
from test_scientific_dialogue import question, ask


def paragraph(text, citations=(), comparisons=(), kind='finding'):
    return dict(text=text, kind=kind, citations=list(citations), comparisons=list(comparisons))


class Model:
    def __init__(self, author, *, verdict='entailed', choice=None, review_callback=None):
        self.author, self.verdict, self.choice = author, verdict, choice or question()
        self.review_callback = review_callback
        self.calls = []

    def complete(self, *, prompt, response_schema):
        p = json.loads(prompt)
        self.calls.append(p)
        if 'turn_schema_version' in p:
            return json.dumps(dict(turn_schema_version=1, decision=self.choice))
        if 'grounding_review_version' in p:
            if self.review_callback:
                self.review_callback()
            return json.dumps(dict(integrity='intact', verdicts={row['id']: dict(verdict=self.verdict, independent=True) for row in p['paragraphs']}))
        return json.dumps(self.author(p['evidence']))


def annotation_answer(e):
    facts = {c['field']: c for c in e['claims']}
    for field in ('groups_omitted', 'assigned_cells', 'unassigned_cells', 'validation_state'):
        assert facts[field]['semantics']['description_source'] == 'reviewed_native_annotation_summary'
    assert facts['groups_omitted']['semantics']['kind'] == 'summary_coverage'
    assert facts['unassigned_cells']['semantics']['kind'] == 'scientific_result'
    return dict(support='supported', paragraphs=[paragraph(
        'MAESTRO computationally assigned 1004 cells; 0 cells were unassigned. '
        'The 2 omitted groups are records excluded from the bounded display, not unidentified groups. '
        'Validation is not_assessed, so the labels are not independently validated biological identities.',
        [facts[f]['claim_id'] for f in ('assigned_cells','unassigned_cells','groups_omitted','validation_state')])])


def test_annotation_semantics_natural_numbers_categories_and_exact_identity(app, monkeypatch):
    rid, _, _ = accepted(app, 'annotate_scATAC_cell_types', overrides=dict(
        n_groups=52, groups_omitted=2, assigned_cells=1004, unassigned_cells=0,
        validation_state='not_assessed', group_summary=[]))
    model = Model(annotation_answer)
    outcome = ask(forbid_work(app, monkeypatch), 'answer', 'Explain annotation.', model)
    assert outcome.status == 'answered'
    assert '1004' in outcome.text and 'not_assessed' in outcome.text
    assert all(c.source['revision_id'] == rid and c.source['evidence_sha256'] for c in outcome.scientific.claims)
    assert len(model.calls) == 3  # interpretation, explanation, distinct semantic review


@pytest.mark.parametrize('text', [
    'The result has 999999 cells.',
    'The parent version has 2 cells.',
    'These are independently validated biological T cells.',
    'This proves a statistically significant causal effect.',
    'The independent populations have identical coordinates.',
    'Omitted groups were unsuccessfully annotated.',
])
def test_negative_semantic_verdict_withholds_entire_unsupported_assertion(app, monkeypatch, text):
    accepted(app, 'inspect_scATAC', overrides={'n_cells':2})
    def author(e):
        c = next(c for c in e['claims'] if c['field'] == 'n_cells')
        return dict(support='supported', paragraphs=[paragraph(text, [c['claim_id']])])
    model = Model(author, verdict='unsupported')
    outcome = ask(forbid_work(app, monkeypatch), 'answer', 'Explain.', model)
    assert outcome.status == 'unavailable' and outcome.scientific is None
    assert text not in outcome.text
    assert model.calls[-1]['paragraphs'][0]['text'] == text


def test_review_failure_and_corruption_do_not_publish(app, monkeypatch):
    _, path, _ = accepted(app, 'inspect_scATAC', overrides={'n_cells':2})
    def author(e):
        c = next(c for c in e['claims'] if c['field'] == 'n_cells')
        return dict(support='supported', paragraphs=[paragraph('There are 2 cells.', [c['claim_id']])])
    model = Model(author, review_callback=lambda: path.write_bytes(path.read_bytes() + b' '))
    result = ask(forbid_work(app, monkeypatch), 'changed', 'Explain.', model)
    assert result.status == 'unavailable'


@pytest.mark.parametrize('failure,stage', [
    ('unsupported', 'semantic_review'), ('provider', 'semantic_review_provider'),
    ('schema', 'semantic_review_schema'),
])
def test_answer_review_failure_preserves_results_and_identity(app, monkeypatch, caplog, failure, stage):
    accepted(app, 'inspect_scATAC')
    app = forbid_work(app, monkeypatch)
    before = app.sessions.load('session')
    files = {p: p.read_bytes() for p in app._workspace.runs.rglob('*') if p.is_file()}
    class Rejected(Model):
        def complete(self, *, prompt, response_schema):
            if 'grounding_review_version' in json.loads(prompt):
                if failure == 'provider': raise RuntimeError('PRIVATE_PROVIDER_DATA')
                if failure == 'schema': return '{"verdicts":{}}'
            return super().complete(prompt=prompt, response_schema=response_schema)
    model = Rejected(lambda e: dict(support='supported', paragraphs=[
        paragraph('The accepted count is recorded.', [e['claims'][0]['claim_id']])]), verdict='unsupported')
    outcome = ask(app, 'review-failure', 'Explain.', model)
    assert outcome.status == 'unavailable' and outcome.scientific is None
    after = app.sessions.load('session')
    assert (after.session_id, after.generation, after.active_revision_id, after.revisions, after.turns) == (
        before.session_id, before.generation, before.active_revision_id, before.revisions, before.turns)
    assert {p: p.read_bytes() for p in app._workspace.runs.rglob('*') if p.is_file()} == files
    assert 'scientific details remain available' in outcome.text
    assert caplog.records[-1].scientific_answer_stage == stage
    assert 'PRIVATE_PROVIDER_DATA' not in caplog.text + outcome.text


@pytest.mark.parametrize('change', [
    lambda p: p.update(citations=['forged']),
    lambda p: p.update(comparisons=['unoffered']),
    lambda p: p.update(citations=[]),
    lambda p: p.update(value=999),
    lambda p: p.update(subject='another group'),
])
def test_structural_forgery_rejected_before_review(app, monkeypatch, change):
    accepted(app, 'inspect_scATAC')
    def author(e):
        p = paragraph('The accepted result contains 2 cells.', [e['claims'][0]['claim_id']])
        change(p)
        return dict(support='supported', paragraphs=[p])
    model = Model(author)
    outcome = ask(forbid_work(app, monkeypatch), 'bad', 'Explain.', model)
    assert outcome.status == 'unavailable'
    assert not any('grounding_review_version' in call for call in model.calls)


def test_comparison_requires_both_exact_operands():
    claims = tuple(sd.ScientificClaim(f'c{i}', f't{i}', None, 'n_cells', 2,
        dict(revision_id=f'r{i}')) for i in range(2))
    public = dict(scope='accepted_persisted_summary', limitations=[],
        comparison=dict(mode='direct_accepted_operands', pairs=[dict(id='p0', left='c0', right='c1', relationship='equal')]))
    value = dict(support='supported', paragraphs=[paragraph('Both have 2 cells.', ['c0'], ['p0'])])
    with pytest.raises(sd._AnswerFailure, match='missing_operand'):
        sd.render_response(value, public, claims)
    value['paragraphs'][0]['citations'].append('c1')
    assert sd.render_response(value, public, claims).claims == claims


def test_guidance_uses_shared_review_and_does_not_publish_rejected_candidates(app, monkeypatch):
    from test_scientific_guidance import Model as GuidanceModel, ask as guide, guard
    accepted(app, 'inspect_scATAC')
    seen = []
    class Rejected(GuidanceModel):
        def complete(self, *, prompt, response_schema):
            p = json.loads(prompt)
            if 'grounding_review_version' in p:
                seen.append(p)
                return json.dumps(dict(integrity='intact', verdicts={row['id']: dict(verdict='unsupported', independent=True) for row in p['paragraphs']}))
            return super().complete(prompt=prompt, response_schema=response_schema)
    result = guide(guard(app, monkeypatch), model=Rejected())
    assert result.status == 'unavailable' and result.guidance is None
    assert set(seen[0]['guidance']) == {'inspect_scATAC'}
    assert seen[0]['guidance']['inspect_scATAC']['readiness']['request_scope'] == 'no_execution_inputs_bound'
    assert app.sessions.load('session').interactions[-1].guidance_candidates is None


@pytest.mark.parametrize('verdict', ['unknown', None, 1])
def test_malformed_semantic_verdict_fails_closed(app, monkeypatch, verdict):
    accepted(app, 'inspect_scATAC')
    model = Model(lambda e: dict(support='supported', paragraphs=[
        paragraph('The accepted count is recorded.', [e['claims'][0]['claim_id']])]), verdict=verdict)
    outcome = ask(forbid_work(app, monkeypatch), 'answer', 'Explain.', model)
    assert outcome.status == 'unavailable' and outcome.scientific is None


def test_interpreter_receives_native_defaults_without_claiming_user_input():
    import inspect
    from agent.application.dialogue_execution import scientific_parameters
    from agent.orchestration import build_default_tool_registry
    registry = build_default_tool_registry()
    rows = scientific_parameters(registry)
    by_id = {(r['tool'], r['argument']): r for r in rows}
    for row in rows:
        tool = registry.get(row['tool'])
        argument = {**tool.required_arguments, **tool.optional_arguments}[row['argument']]
        if argument.planning.default_when_omitted is not inspect.Parameter.empty:
            assert 'default_when_omitted' in row
        assert 'value' not in row and 'literal' not in row
    assert by_id['cluster_cells','resolution']['default_when_omitted'] == 1.0
    assert by_id['select_scATAC_cells','min_qc_fragment_records']['default_when_omitted'] == 1000


@pytest.mark.parametrize('arguments', [{}, {'min_qc_fragment_records': 1700, 'min_tss_flank_evidence': 20}])
def test_threshold_origin_semantics_use_pinned_effective_values(app, monkeypatch, arguments):
    from agent.tools.data.scatac_selection_profile import thresholds
    effective = thresholds(arguments)
    accepted(app, 'select_scATAC_cells', arguments=arguments,
             extra_facts={'effective_thresholds': effective})
    def author(e):
        facts = {c['field']: c for c in e['claims']}
        assert facts['effective_thresholds.min_qc_fragment_records']['value'] == arguments.get('min_qc_fragment_records', 1000)
        assert facts['effective_thresholds.min_tss_enrichment.numerator']['value'] == 4
        assert facts['effective_thresholds.min_tss_enrichment.denominator']['value'] == 1
        assert facts['effective_thresholds.max_qc_fragment_records']['value'] is None
        assert facts['effective_thresholds.min_tss_flank_evidence']['value'] == arguments.get('min_tss_flank_evidence')
        for name in ('min_qc_fragment_records', 'min_tss_enrichment', 'min_tss_flank_evidence', 'max_qc_fragment_records'):
            fact = facts['selection_threshold_origins.' + name]
            assert fact['value'] == ('explicit_execution_argument' if name in arguments else 'existing_owner_default')
            description = fact['semantics']['description']
            assert 'effective_thresholds.' + name in description
            assert 'non-null threshold is enabled' in description
            assert 'null denotes a disabled optional filter' in description
            assert 'not current defaults' in description
        return dict(support='supported', paragraphs=[paragraph('The accepted effective thresholds are recorded.',
            [facts['effective_thresholds.min_qc_fragment_records']['claim_id']])])
    assert ask(forbid_work(app, monkeypatch), 'thresholds', 'Explain the effective thresholds.', Model(author)).status == 'answered'


@pytest.mark.parametrize('mode', ['direct_accepted_operands', 'descriptive_accepted_operands', 'independent_accepted_facts'])
@pytest.mark.parametrize('scope', ['partial', 'independent', 'insufficient'])
def test_comparison_context_does_not_require_comparison_composition(mode, scope):
    claims = tuple(sd.ScientificClaim(f'c{i}', f't{i}', None, 'n_cells', 2, {}) for i in range(2))
    public = dict(scope='accepted_persisted_summary', limitations=[], comparison=dict(mode=mode, pairs=[]))
    rows = [paragraph('Result A records 2 cells.', ['c0'])]
    if scope == 'independent':
        rows.append(paragraph('Result B records 2 cells.', ['c1']))
    elif scope == 'insufficient':
        rows = [paragraph('Cell correspondence is not established by this context.', kind='limitation')]
    result = sd.render_response(dict(support='insufficient_evidence' if scope == 'insufficient' else 'supported',
                                    paragraphs=rows), public, claims)
    assert len(result.claims) == {'partial': 1, 'independent': 2, 'insufficient': 0}[scope]


@pytest.mark.parametrize('attack,expected_stage', [
    ('none', None), ('unsupported', 'semantic_review'), ('provider', 'semantic_review_provider'),
    ('missing', 'semantic_review_schema'), ('extra', 'semantic_review_schema'),
    ('wrong_candidate', 'semantic_review_schema'), ('wrong_paragraph', 'semantic_review_schema'),
    ('list', 'semantic_review_schema'), ('unknown', 'semantic_review_schema'),
    ('duplicate_json_key', 'semantic_review_schema'),
])
def test_guidance_one_review_exact_mapping_and_failure_preservation(app, monkeypatch, caplog, attack, expected_stage):
    from test_scientific_guidance import Model as GuidanceModel, ask as guide, guard
    accepted(app, 'inspect_scATAC')
    app = guard(app, monkeypatch)
    before = app.sessions.load('session')
    files = {p: p.read_bytes() for p in app._workspace.runs.rglob('*') if p.is_file()}
    reviews = []
    class Reviewed(GuidanceModel):
        def complete(self, *, prompt, response_schema):
            p = json.loads(prompt)
            if 'grounding_review_version' not in p:
                return super().complete(prompt=prompt, response_schema=response_schema)
            reviews.append(p)
            assert len(p['paragraphs']) == 6
            assert set(p['guidance']) == {'inspect_scATAC', 'cluster_cells'}
            for row in p['paragraphs']:
                assert row['id'].startswith(row['candidate'] + ':')
                assert row['candidate'] in p['guidance']
            values = {row['id']: dict(verdict='entailed', independent=True)
                      for row in reversed(p['paragraphs'])}
            assert set(response_schema['properties']['verdicts']['required']) == set(values)
            if attack == 'provider': raise RuntimeError('PRIVATE_PROVIDER_DATA')
            if attack == 'unsupported': values['cluster_cells:2']['verdict'] = 'unsupported'
            if attack == 'missing': values.pop('cluster_cells:2')
            if attack == 'extra': values['unoffered:0'] = dict(verdict='entailed', independent=True)
            if attack == 'wrong_candidate': values['other:2'] = values.pop('cluster_cells:2')
            if attack == 'wrong_paragraph': values['cluster_cells:3'] = values.pop('cluster_cells:2')
            if attack == 'unknown': values['cluster_cells:2']['verdict'] = 'probably'
            if attack == 'list': values = list(values.values())
            if attack == 'duplicate_json_key':
                return '{"verdicts":{"cluster_cells:0":"entailed","cluster_cells:0":"unsupported"}}'
            return json.dumps(dict(integrity='intact', verdicts=values))
    model = Reviewed(tools=('inspect_scATAC', 'cluster_cells'))
    result = guide(app, model=model)
    assert len(reviews) == 1
    assert len(model.calls) == 2  # interpreter and generation, plus the one recorded review
    state = app.sessions.load('session')
    assert (state.session_id, state.generation, state.active_revision_id, state.revisions, state.turns) == (
        before.session_id, before.generation, before.active_revision_id, before.revisions, before.turns)
    assert {p: p.read_bytes() for p in app._workspace.runs.rglob('*') if p.is_file()} == files
    assert 'PRIVATE_PROVIDER_DATA' not in caplog.text + result.text
    if expected_stage:
        assert result.status == 'unavailable' and result.guidance is None
        assert state.interactions[-1].guidance_candidates is None
        assert 'accepted result' in result.text
        assert caplog.records[-1].scientific_answer_stage == expected_stage
    else:
        assert result.status == 'answered'
        assert [c.reference['capability'] for c in result.guidance.candidates] == ['cluster_cells', 'inspect_scATAC']
        refs = [dict(c.reference) for c in result.guidance.candidates]
        assert [dict(r) for r in state.interactions[-1].guidance_candidates] == refs


def test_invalid_later_guidance_candidate_prevents_any_review(app, monkeypatch):
    from test_scientific_guidance import Model as GuidanceModel, ask as guide, guard
    accepted(app, 'inspect_scATAC')
    class Invalid(GuidanceModel):
        def complete(self, *, prompt, response_schema):
            assert 'grounding_review_version' not in json.loads(prompt)
            return super().complete(prompt=prompt, response_schema=response_schema)
    result = guide(guard(app, monkeypatch), model=Invalid(tools=('inspect_scATAC', 'cluster_cells'),
        attack=lambda r:r['candidates'][1]['explanation']['paragraphs'][0].update(citations=['forged'])))
    assert result.status == 'unavailable'


def test_guidance_freshness_after_shared_review(app, monkeypatch):
    from test_scientific_guidance import Model as GuidanceModel, ask as guide, guard
    _, path, _ = accepted(app, 'inspect_scATAC')
    class Changed(GuidanceModel):
        def complete(self, *, prompt, response_schema):
            if 'grounding_review_version' in json.loads(prompt):
                path.write_bytes(path.read_bytes() + b' ')
            return super().complete(prompt=prompt, response_schema=response_schema)
    result = guide(guard(app, monkeypatch), model=Changed(tools=('inspect_scATAC', 'cluster_cells')))
    assert result.status == 'unavailable' and result.guidance is None
    assert app.sessions.load('session').interactions[-1].guidance_candidates is None
