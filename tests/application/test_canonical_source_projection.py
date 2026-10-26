"""One captured file order owns every current public source projection.

Scripted interpretation only requests prerequisites or clarifies. Distinct fake
owner observations test source association; no algorithm or provider runs.
"""
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from agent.application.epizoo_handoff import EpiZooDeclaration
from agent.application.local_resources import LocalResourceAdmission
from agent.application.raw_handoff import AnnotationDeclaration
from agent.application.session_state import AnalysisSession
from agent.schemas.orchestration import _serialize
import test_offered_source_binding as offered


CURRENT = 'Explain which current input is available for this independent request.'
REPLY = 'Use the newly selected file for the original pending objective.'


class ProjectionModel:
    model_id = offered.PROFILE.model_id

    def __init__(self, calls, *, pending_field='roles', continue_fixed=False):
        self.calls = calls
        self.pending_field = pending_field
        self.continue_fixed = continue_fixed

    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        assert 'turn_schema_version' in value, 'Projection test must not invoke a Planner or answerer.'
        self.calls.append(dict(prompt=value, schema=response_schema))
        if value['utterance'] == offered.OLD:
            if self.pending_field == 'roles':
                decision = dict(kind='request_roles', target='pca_embed_cells',
                    required_roles=[offered.PORT], roles=[],
                    arguments=[offered.number(offered.OLD, '2')])
            else:
                field = self.pending_field
                declaration = (AnnotationDeclaration(source='source.0') if field == 'annotation'
                               else EpiZooDeclaration(operation='inference', source='source.0'))
                decision = dict(kind='request_roles', required_roles=[], roles=[], arguments=[],
                    target='annotate_scATAC_cell_types' if field == 'annotation' else 'epizoo_embed_cells',
                    **{field: asdict(declaration)})
        elif self.continue_fixed:
            decision = dict(kind='answer_roles', pending='@roles', arguments=[],
                            roles=[dict(offered.PORT, source='source.0')])
        else:
            decision = dict(kind='clarify', reason='ambiguous_subject')
        return json.dumps(dict(turn_schema_version=1, decision=decision))


def setup_case(tmp_path, monkeypatch, **options):
    monkeypatch.setattr(offered, 'Model', ProjectionModel)
    app, records, executor, calls = offered.setup_case(tmp_path, **options)
    observations = []

    def observe(path, resources=()):
        observations.append(str(path))
        index = next(i for i, record in enumerate(records) if record.source_path == str(path))
        return dict(pca='eligible' if index == 0 else 'invalid_or_unsupported',
                    epizoo=[], pca_measurements='Simulated observation of source ' + str(index))

    monkeypatch.setattr('agent.tools.analysis.pca_embedding.observe_methods', observe)
    return app, records, executor, calls, observations


def source_enum(schema, field):
    variants = schema['properties']['decision']['anyOf']
    request, = [v for v in variants if v['properties']['kind']['enum'] == ['request_roles']
                and field in v['properties']]
    return request['properties'][field]['properties']['source']['anyOf'][0]['enum']


@pytest.mark.parametrize('pending_field', ['roles', 'annotation', 'epizoo'])
def test_single_primary_after_pending_uses_one_source_order_for_every_offer(
        tmp_path, monkeypatch, pending_field):
    app, records, executor, calls, observations = setup_case(
        tmp_path, monkeypatch, pending_field=pending_field)
    first = offered.submit(app, records[0], 'old', offered.OLD)
    assert first.status == 'clarification' and not executor.plans, first.to_dict()
    observations.clear()
    single = app.resources.compose_h5ad(records[1].resource_id)
    result = app.submit_turn('session', 'current', CURRENT,
                             expected_generation=0, registered_input=single)
    assert result.status == 'clarification' and not executor.plans, result.to_dict()
    public = calls[-1]['prompt']['dialogue']
    assert [row['label'] for row in public['selected_sources']] == [record.label for record in records]
    assert public['current_turn_sources'] == ['source.1']
    assert public['selected_primary_source'] == 'source.1'
    assert public['session_source_candidates'] is True
    assert [row['source'] for row in public['source_fields']] == ['source.0', 'source.1']
    assert observations == [record.source_path for record in records]
    assert [(row['source'], row['pca']) for row in public['embedding_suitability']] == [
        ('source.0', 'eligible'), ('source.1', 'invalid_or_unsupported')]
    assert 'annotation_sources' not in public
    for field in ('annotation', 'epizoo'):
        assert field + '_resources' in public
        assert source_enum(calls[-1]['schema'], field) == ['source.0', 'source.1']
    state = app._application.sessions.load('session')
    current = state.interactions[-1]
    assert current.submission['registered_input']['composition'] == 'h5ad-science.v1'
    assert current.submission['registered_input']['resource_id'] == records[1].resource_id
    captured = current.snapshot['dialogue']['source_view']
    assert [member['resource_id'] for member in captured['registered_input']['members']] == [
        record.resource_id for record in records]
    assert not state.turns and not state.revisions and state.generation == 0


def test_current_single_without_pending_is_primary_before_old_candidates(tmp_path, monkeypatch):
    app, records, executor, calls, observations = setup_case(tmp_path, monkeypatch)
    first = app.submit_turn('session', 'earlier', CURRENT, expected_generation=0,
                           registered_input=app.resources.compose_h5ad(records[0].resource_id),
                           attachment_ids=(records[0].resource_id,))
    assert first.status == 'clarification'
    observations.clear()
    second = app.submit_turn('session', 'current', CURRENT, expected_generation=0,
                            registered_input=app.resources.compose_h5ad(records[1].resource_id),
                            attachment_ids=(records[1].resource_id,))
    assert second.status == 'clarification' and not executor.plans
    public = calls[-1]['prompt']['dialogue']
    assert [row['label'] for row in public['selected_sources']] == [record.label for record in reversed(records)]
    assert public['current_turn_sources'] == ['source.0']
    assert public['selected_primary_source'] == 'source.0'
    assert observations == [record.source_path for record in reversed(records)]
    assert [(row['source'], row['pca']) for row in public['embedding_suitability']] == [
        ('source.0', 'invalid_or_unsupported'), ('source.1', 'eligible')]


def test_new_single_cannot_replace_a_fixed_pending_source(tmp_path, monkeypatch):
    app, records, executor, calls, _ = setup_case(tmp_path, monkeypatch, continue_fixed=True)
    fixed = app.resources.compose_group([records[0].resource_id], dict(n_components=2))
    first = app.submit_turn('session', 'old', offered.OLD, expected_generation=0, registered_input=fixed)
    assert first.status == 'clarification'
    original = app._application.sessions.load('session').interactions[0]
    second = app.submit_turn('session', 'attempt', REPLY, expected_generation=0,
                            registered_input=app.resources.compose_h5ad(records[1].resource_id))
    assert second.status == 'failed' and second.error.code == 'INTERACTIVE_INTENT_BINDING_INVALID'
    public = calls[-1]['prompt']['dialogue']
    assert 'pending_prerequisite' not in public
    assert public['selected_primary_source'] == 'source.0'
    assert [row['label'] for row in public['selected_sources']] == [records[1].label]
    state = app._application.sessions.load('session')
    assert state.interactions[0] == original
    assert not executor.plans and not state.turns and not state.revisions


def test_old_capture_without_source_view_decodes_after_sources_disappear(tmp_path, monkeypatch):
    app, records, executor, _, _ = setup_case(tmp_path, monkeypatch)
    assert offered.submit(app, records[0], 'old', offered.OLD).status == 'clarification'
    state = app._application.sessions.load('session')
    legacy = _serialize(state.to_dict())
    for interaction in legacy['interactions']:
        interaction['snapshot']['dialogue'].pop('source_view', None)
        interaction['snapshot']['dialogue'].pop('source_fields', None)
    for record in records:
        Path(record.source_path).unlink()

    def forbidden(*args, **kwargs):
        pytest.fail('Retained Session decoding attempted source I/O.')

    monkeypatch.setattr('agent.tools.analysis.categorical.fields', forbidden)
    monkeypatch.setattr('agent.tools.analysis.pca_embedding.observe_methods', forbidden)
    monkeypatch.setattr(LocalResourceAdmission, '_verify_records', forbidden)
    loaded = AnalysisSession.from_dict(legacy)
    assert _serialize(loaded.to_dict()) == legacy
    assert loaded.interactions[0].prerequisite['origin_turn_id'] == 'old'
    assert not executor.plans and not loaded.turns and not loaded.revisions
