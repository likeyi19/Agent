"""Separate accepted facts from qualified measurement pairs; retain claim checks."""
from dataclasses import replace
import json

import pytest

from agent.application import scientific_dialogue as sd
from agent.application.dialogue_evidence import EvidenceFact
from test_dialogue_evidence import app, accepted, forbid_work
from test_scientific_dialogue import Model, ask, question, annotation


def both(value):
    evidence = value['evidence']
    assert evidence['comparison']['mode'] == 'independent_accepted_facts'
    assert evidence['comparison']['pairs'] == []
    return dict(support='supported', paragraphs=[dict(parts=[
        *[dict(kind='claim', id=c['claim_id']) for c in evidence['claims'] if c['field'] == 'n_cells'],
        dict(kind='meaning', id='comparison_scope')])])


@pytest.mark.parametrize('tools', [('inspect_scATAC', 'inspect_scATAC'),
    ('inspect_scATAC', 'build_scATAC_cell_by_features'),
    ('build_scATAC_cell_by_features', 'build_scATAC_cell_by_features')])
def test_independent_facts_exact_attribution_without_measurement_pairs(app, monkeypatch, tools):
    old = accepted(app, tools[0], overrides={'n_cells': 2})[0]
    new = accepted(app, tools[1], name='next', overrides={'n_cells': 4})[0]
    guarded = forbid_work(app, monkeypatch)
    state = guarded.sessions.load('session')
    result = ask(guarded, 'facts', 'Discuss the current result and its parent.',
        Model(question(comparison=dict(output='r1', subject=None)), response=both))
    assert result.status == 'answered'
    assert {(c.source['revision_id'], c.value) for c in result.scientific.claims} == {(old, 2), (new, 4)}
    assert all(c.source['evidence_sha256'] and c.source['pointer'] for c in result.scientific.claims)
    after = guarded.sessions.load('session')
    assert after.revisions == state.revisions and after.generation == state.generation
    assert 'No compatible measurement pairs' in result.text


def test_unoffered_measurement_pair_fails_closed(app, monkeypatch, caplog):
    accepted(app, 'inspect_scATAC')
    accepted(app, 'inspect_scATAC', name='next')
    result = ask(forbid_work(app, monkeypatch), 'invalid', 'Compare with the parent.',
        Model(question(comparison=dict(output='r1', subject=None)), response=dict(
            support='supported', paragraphs=[dict(parts=[dict(kind='comparison', id='p0')])])))
    assert result.status == 'unavailable'
    assert diagnostic(caplog) == ('evidence_binding', 'invalid_reference')


def diagnostic(caplog):
    return [(r.scientific_answer_stage, r.scientific_answer_reason) for r in caplog.records
            if hasattr(r, 'scientific_answer_stage')][-1]


@pytest.mark.parametrize('response,expected', [
    ({'wrong': 'private /path API_SECRET'}, ('response_schema', 'malformed_provider_output')),
    (dict(support='supported', paragraphs=[dict(parts=[dict(kind='claim', id='bogus')])]),
     ('evidence_binding', 'invalid_reference')),
    (dict(support='supported', paragraphs=[dict(parts=[dict(kind='text', text='Count is 99')])]),
     ('factual_binding', 'unbound_factual_literal')),
    (dict(support='supported', paragraphs=[dict(parts=[dict(kind='text', text='fixture')])]),
     ('factual_binding', 'unbound_categorical_value')),
    (dict(support='supported', paragraphs=[dict(parts=[dict(kind='text', text='Recorded.')])]),
     ('evidence_binding', 'missing_evidence')),
])
def test_safe_failure_classification(app, monkeypatch, caplog, response, expected):
    accepted(app, 'cluster_cells')
    result = ask(forbid_work(app, monkeypatch), 'bad', 'Explain the accepted result.', Model(question(), response=response))
    assert result.status == 'unavailable' and result.scientific is None
    assert diagnostic(caplog) == expected
    assert 'API_SECRET' not in caplog.text and '/path' not in caplog.text


@pytest.mark.parametrize('raw', ['not json /private API_SECRET', '{"support":NaN}', '{"support":1,"support":2}'])
def test_malformed_json_diagnostic(app, monkeypatch, caplog, raw):
    accepted(app, 'inspect_scATAC')
    class Broken(Model):
        def complete(self, **kwargs):
            if 'dialogue_schema_version' in json.loads(kwargs['prompt']):
                return raw
            return super().complete(**kwargs)
    result = ask(forbid_work(app, monkeypatch), 'bad', 'Explain.', Broken(question()))
    assert result.status == 'unavailable'
    assert diagnostic(caplog) == ('parsing', 'malformed_provider_output')
    assert 'API_SECRET' not in caplog.text


def test_freshness_failure_has_distinct_stage(app, monkeypatch, caplog):
    _, path, _ = accepted(app, 'inspect_scATAC')
    def mutate(value):
        path.write_bytes(path.read_bytes() + b' ')
        return dict(support='supported', paragraphs=[dict(parts=[dict(kind='claim', id='c0')])])
    result = ask(forbid_work(app, monkeypatch), 'changed', 'Explain.', Model(question(), response=mutate))
    assert result.status == 'unavailable'
    assert diagnostic(caplog) == ('evidence_freshness', 'changed_accepted_evidence')


def test_subject_correspondence_rejection_remains(app, monkeypatch, caplog):
    annotation(app)
    accepted(app, 'annotate_scATAC_cell_types', name='next', overrides=dict(n_groups=1,
        groups_omitted=0, group_summary=[dict(group='3', n_cells=4, primary_annotation='T', status='assigned')]))
    result = ask(forbid_work(app, monkeypatch), 'subjects', 'Compare cluster 3 with cluster 3 in the parent.',
        Model(question('r0', '3', comparison=dict(output='r1', subject='3'))))
    assert result.status == 'unavailable'
    assert diagnostic(caplog) == ('evidence_context', 'incompatible_scientific_context')


def test_clustering_pair_eligibility_needs_qualified_identity_and_method(app):
    targets, views = [], []
    for name in ('original', 'new'):
        rid = accepted(app, 'cluster_cells', name=name, overrides={'algorithm': 'leiden', 'n_cells': 10})[0]
        view = app.sessions.evidence('session', rid, 'output0')
        targets.append(dict(accepted_step_sha256=view.source.output_locator['accepted_step_sha256']))
        views.append(view)
    assert sd._comparison_fields(views, targets) == frozenset()
    qualified = [replace(v, facts=v.facts + (EvidenceFact('clustering_neighbors_identity_sha256',
        'available', 'a'*64, '/plan/steps/0/arguments/analysis_path', artifact_name='accepted_run_result',
        artifact_sha256=v.source.source_run_result_sha256),)) for v in views]
    assert 'n_clusters' in sd._comparison_fields(qualified, targets)
    for field, value in [('clustering_neighbors_identity_sha256', 'b'*64), ('algorithm', 'other'),
                          ('n_cells', 11)]:
        bad = replace(qualified[1], facts=tuple(replace(f, value=value) if f.field == field else f
                                               for f in qualified[1].facts))
        assert sd._comparison_fields([qualified[0], bad], targets) == frozenset()
    unsupported = replace(qualified[1], source=replace(qualified[1].source, result_contract='other'))
    assert sd._comparison_fields([qualified[0], unsupported], targets) == frozenset()


def test_partial_answer_preserves_supported_subset(app, monkeypatch):
    accepted(app, 'inspect_scATAC')
    accepted(app, 'inspect_scATAC', name='next')
    result = ask(forbid_work(app, monkeypatch), 'partial', 'Explain the current result and its parent, and their biological superiority.',
        Model(question(comparison=dict(output='r1', subject=None)), ['n_cells'], support='insufficient_evidence'))
    assert result.status == 'answered' and result.scientific.claims
    assert result.scientific.support == 'insufficient_evidence'


def test_provider_exception_is_sanitized(app, monkeypatch, caplog):
    accepted(app, 'inspect_scATAC')
    class Broken(Model):
        def complete(self, **kwargs):
            if 'dialogue_schema_version' in json.loads(kwargs['prompt']):
                raise RuntimeError('API_SECRET /private/provider-response')
            return super().complete(**kwargs)
    result = ask(forbid_work(app, monkeypatch), 'provider', 'Explain.', Broken(question()))
    assert result.status == 'unavailable'
    assert diagnostic(caplog) == ('provider', 'provider_failure')
    assert 'API_SECRET' not in caplog.text and '/private' not in result.text


def test_comparison_binding_diagnostics(app, monkeypatch, caplog):
    for name in ('original', 'new'):
        accepted(app, 'select_scATAC_cells', name=name,
            arguments=dict(min_qc_fragment_records=1, min_tss_enrichment='4'),
            overrides=dict(qc_identity_sha256='a'*64, n_selected=8, n_rejected=2))
    def different_fields(value):
        claims = value['evidence']['claims']
        ids = [c['claim_id'] for c in claims if (c['target'], c['field']) in
               {('t0', 'n_selected'), ('t1', 'n_rejected')}]
        return dict(support='supported', paragraphs=[dict(parts=[dict(kind='claim', id=i) for i in ids])])
    result = ask(forbid_work(app, monkeypatch), 'pairs', 'Compare with the parent.',
        Model(question(comparison=dict(output='r1', subject=None)), response=different_fields))
    assert result.status == 'unavailable'
    assert diagnostic(caplog) == ('comparison_binding', 'missing_compatible_pair')


def test_cluster_count_must_match_accepted_owner_result(app, monkeypatch):
    from test_clustering_parameter_evidence import repin
    rid, path, payload = accepted(app, 'cluster_cells', overrides={'n_clusters': 4})
    payload['steps'][0]['facts']['n_clusters'] = 99
    repin(app, path, payload)
    view = forbid_work(app, monkeypatch).sessions.evidence('session', rid, 'output0')
    assert view.status == 'unavailable' and not view.facts


def test_same_literal_graph_path_does_not_establish_qualified_lineage(app, monkeypatch):
    rid, _, _ = accepted(app, 'cluster_cells', arguments={'analysis_path': '/recorded/graph.h5ad'})
    view = forbid_work(app, monkeypatch).sessions.evidence('session', rid, 'output0')
    assert view.status == 'available'
    assert not any(f.field == 'clustering_neighbors_identity_sha256' for f in view.facts)


@pytest.mark.parametrize('tool,method', [('pca_embed_cells', 'pca'), ('epizoo_embed_cells', 'epizoo')])
def test_embedding_answer_recovery_keeps_accepted_revision_and_values(app, monkeypatch, tool, method):
    old = accepted(app, tool, overrides={'n_cells': 8})[0]
    new = accepted(app, tool, name='next', overrides={'n_cells': 12})[0]
    def prose(value):
        return dict(support='supported', paragraphs=[dict(parts=[
            *[dict(kind='claim', id=c['claim_id']) for c in value['evidence']['claims'] if c['field']=='n_cells'],
            dict(kind='comparison', id=value['evidence']['comparison']['pairs'][0]['id']),
            dict(kind='text', text=f'These {method} results have 999 identical cells.')])])
    result = ask(forbid_work(app, monkeypatch), 'recovered', 'Discuss the current result and its parent.',
        Model(question(comparison=dict(output='r1', subject=None)), response=prose))
    assert result.status == 'answered' and result.scientific.support == 'insufficient_evidence'
    assert {(c.source['revision_id'], c.value) for c in result.scientific.claims if c.field == 'n_cells'} == {(old, 8), (new, 12)}
    assert '999' not in result.text and 'identical cells' not in result.text
