"""Pinned selection-owner metadata origins; fixtures qualify no new science."""
from dataclasses import replace

import pytest

from agent.application import dialogue_evidence as de
from agent.application.session_state import Interaction, digest
from agent.tools.data.scatac_selection_profile import thresholds
from test_dialogue_evidence import app, accepted, files, forbid_work


def selection(app, arguments=None):
    arguments = dict(min_qc_fragment_records=2, min_tss_enrichment='4.5') if arguments is None else arguments
    return accepted(app, 'select_scATAC_cells', arguments=arguments,
        extra_facts={'effective_thresholds': thresholds(arguments)})


def record_declarations(app, tool, values, *, turn='initial', user_values=None):
    user_values = values if user_values is None else user_values
    utterance, arguments = '', []
    for name, value in user_values.items():
        if value is None:
            continue
        literal = str(value)
        utterance += name + ' = '
        start = len(utterance)
        utterance += literal + '; '
        arguments.append(dict(tool=tool, argument=name, literal=literal,
            start=start, end=start + len(literal)))
    state = app.sessions.load('session')
    item = Interaction(turn, utterance, None, 0, dict(bases={}, relations={}), 'submitted',
        admitted=dict(kind='execute', operation='plan', tool=tool, request_id=turn,
            inputs=user_values, arguments=arguments, arguments_received_inputs_sha256=digest(user_values)))
    app.sessions._store._write(replace(state, interactions=(*state.interactions, item)))


def test_exact_accepted_session_declarations_establish_selection_user_origins(app, monkeypatch):
    arguments = dict(min_qc_fragment_records=2, min_tss_enrichment='4.5')
    rid, _, _ = selection(app, arguments)
    record_declarations(app, 'select_scATAC_cells', arguments)
    guarded = forbid_work(app, monkeypatch)
    before = files(app)
    view = guarded.sessions.evidence('session', rid, 'output0', fields=('explicit_user_parameters',))
    assert view.status == 'available', view
    fact, = view.facts
    assert dict(fact.value['parameters']) == {name: 'explicit_user_instruction' for name in arguments}
    assert fact.value['tool'] == 'select_scATAC_cells' and fact.value['step_id'] == 'step0'
    assert fact.artifact_name == 'accepted_session_interactions'
    assert fact.artifact_sha256 != view.source.source_run_result_sha256
    assert fact.source_pointer == '/interactions'
    assert files(app) == before


def test_unrelated_turn_with_identical_values_never_supplies_user_origin(app, monkeypatch):
    arguments = dict(min_qc_fragment_records=2, min_tss_enrichment='4.5')
    rid, _, _ = selection(app, arguments)
    record_declarations(app, 'select_scATAC_cells', arguments, turn='unrelated')
    view = forbid_work(app, monkeypatch).sessions.evidence('session', rid, 'output0',
        fields=('explicit_user_parameters',))
    assert view.status == 'available'
    assert view.facts[0].status == 'unavailable' and view.facts[0].value is None


def test_session_user_origin_cannot_disagree_with_the_exact_execution(app, monkeypatch):
    arguments = dict(min_qc_fragment_records=2, min_tss_enrichment='4.5')
    rid, _, _ = selection(app, arguments)
    record_declarations(app, 'select_scATAC_cells', arguments,
        user_values=arguments | {'min_qc_fragment_records': 3})
    view = forbid_work(app, monkeypatch).sessions.evidence('session', rid, 'output0')
    assert view.status == 'unavailable' and view.source is None and not view.facts


def test_additional_registered_operation_uses_same_user_origin_projection(app, monkeypatch):
    arguments = {'n_neighbors': 5}
    rid, _, _ = accepted(app, 'build_cell_neighbors', arguments=arguments)
    record_declarations(app, 'build_cell_neighbors', arguments)
    view = forbid_work(app, monkeypatch).sessions.evidence('session', rid, 'output0',
        fields=('explicit_user_parameters',))
    assert view.status == 'available', view
    assert dict(view.facts[0].value['parameters']) == {'n_neighbors': 'explicit_user_instruction'}
    assert view.source.tool_name == 'build_cell_neighbors'


def test_user_origin_answer_uses_immutable_linked_records_and_withholds_step_ids(app, monkeypatch):
    from test_scientific_dialogue import Model, question
    arguments = dict(min_qc_fragment_records=2, min_tss_enrichment='4.5')
    rid, _, _ = selection(app, arguments)
    record_declarations(app, 'select_scATAC_cells', arguments)
    guarded = forbid_work(app, monkeypatch)
    field = 'explicit_user_parameters.parameters.min_qc_fragment_records'
    initial = guarded.sessions.evidence('session', rid, 'output0', fields=('explicit_user_parameters',))
    marker = initial.facts[0].artifact_sha256
    model = Model(question(), (field,))
    response = guarded.sessions.respond('session', 'user-origin', 'Did I explicitly supply the minimum depth?',
        interpreter=model)
    assert response.status == 'answered', response
    claim, = response.scientific.claims
    assert claim.field == field and claim.value == 'explicit_user_instruction'
    assert claim.source['artifact_name'] == 'accepted_session_interactions'
    assert claim.source['artifact_sha256'] == marker
    assert claim.source['pointer'] == '/interactions'
    public = next(call['evidence'] for call in model.calls if 'dialogue_schema_version' in call)
    assert not any(claim['field'] == 'explicit_user_parameters.step_id' for claim in public['claims'])
    final = guarded.sessions.evidence('session', rid, 'output0', fields=('explicit_user_parameters',))
    assert final.facts[0].artifact_sha256 == marker


def test_exact_partial_binding_chain_preserves_user_origins_after_correction(app, monkeypatch):
    tool = 'select_scATAC_cells'
    arguments = dict(min_qc_fragment_records=3, min_tss_enrichment='4.5')
    rid, _, _ = accepted(app, tool, name='final', arguments=arguments,
        extra_facts={'effective_thresholds': thresholds(arguments)})
    pending = dict(origin_turn_id='origin', binding_turn_id='origin', field='parameters',
        tool=tool, missing=['min_qc_fragment_records', 'min_tss_enrichment'], resource_context_sha256=digest([]))
    original = Interaction('origin', 'Select candidate cells.', None, 0, dict(bases={}, relations={}),
        'clarification', admitted=dict(kind='execute', operation='plan', tool=tool, request_id='origin',
            inputs={}), prerequisite=pending)
    text = 'Use minimum depth 2.'
    start = text.index('2')
    dialogue = dict(outputs=[], predecessor=None, requested_predecessor=None, ambiguous=False)
    partial = Interaction('partial', text, None, 0,
        dict(bases={}, relations={}, dialogue=dialogue,
            prerequisite=pending | {'predecessor_turn_id': 'origin'}),
        'clarification', admitted=dict(kind='clarify', reason='missing_parameter_value',
            choices=['min_tss_enrichment'], inputs={'min_qc_fragment_records': 2},
            arguments=[dict(tool=tool, argument='min_qc_fragment_records', literal='2', start=start, end=start+1)],
            parameter_continuation=dict(origin_turn_id='origin', binding_turn_id='origin', field='parameters')),
        prerequisite=pending | {'binding_turn_id': 'partial', 'missing': ['min_tss_enrichment']})
    text = 'Correct minimum depth to 3 and use minimum TSS 4.5.'
    declarations = [dict(tool=tool, argument=name, literal=literal,
        start=text.index(literal), end=text.index(literal)+len(literal))
        for name, literal in (('min_qc_fragment_records', '3'), ('min_tss_enrichment', '4.5'))]
    completed = Interaction('final', text, None, 0,
        dict(bases={}, relations={}, dialogue=dialogue,
            prerequisite=dict(partial.prerequisite) | {'predecessor_turn_id': 'partial'}),
        'submitted', admitted=dict(kind='execute', operation='plan', tool=tool, request_id='final',
            inputs=arguments, arguments=declarations,
            continuation=dict(origin_turn_id='origin', binding_turn_id='partial', field='parameters')))
    state = app.sessions.load('session')
    app.sessions._store._write(replace(state, interactions=(original, partial, completed)))
    view = forbid_work(app, monkeypatch).sessions.evidence('session', rid, 'output0',
        fields=('effective_thresholds', 'explicit_user_parameters'))
    assert view.status == 'available', view
    assert view.facts[0].value['min_qc_fragment_records'] == 3
    assert dict(view.facts[1].value['parameters']) == {name: 'explicit_user_instruction' for name in arguments}


@pytest.mark.parametrize('optional', [{}, {'min_tss_flank_evidence': 7},
    {'min_tss_flank_evidence': None}, {'max_nucleosome_signal': '3/2'}])
def test_selection_origins_use_exact_arguments_and_owner_effective_thresholds(app, monkeypatch, optional):
    arguments = dict(min_qc_fragment_records=2, min_tss_enrichment='4.5', **optional)
    rid, path, _ = selection(app, arguments)
    guarded = forbid_work(app, monkeypatch)
    before = files(app)
    view = guarded.sessions.evidence('session', rid, 'output0',
        fields=('effective_thresholds', 'selection_threshold_origins'))
    assert view.status == 'available', view
    effective, origins = view.facts
    assert effective.value['min_tss_enrichment'] == {'numerator': 9, 'denominator': 2}
    assert effective.value['min_tss_flank_evidence'] == optional.get('min_tss_flank_evidence')
    assert dict(origins.value) == {name: 'explicit_execution_argument' if name in arguments
        else 'existing_owner_default' for name in thresholds(arguments)}
    assert origins.artifact_name == 'accepted_run_result'
    assert origins.artifact_sha256 == view.source.source_run_result_sha256
    assert origins.source_pointer == '/steps/0/resolved_arguments'
    assert view.source.tool_name == 'select_scATAC_cells'
    assert view.source.output_locator['step_id'] == 'step0'
    assert b'selection_threshold_origins' not in path.read_bytes()
    assert files(app) == before


def test_historical_summary_cannot_invent_missing_threshold_provenance(app, monkeypatch):
    rid, _, _ = accepted(app, 'select_scATAC_cells')
    view = forbid_work(app, monkeypatch).sessions.evidence('session', rid, 'output0',
        fields=('selection_threshold_origins',))
    assert view.status == 'available'
    assert view.facts[0].status == 'unavailable'
    assert view.facts[0].reason == 'not_in_accepted_summary'


def test_boolean_or_mismatching_arguments_cannot_supply_selection_origins(app, monkeypatch):
    rid, _, _ = accepted(app, 'select_scATAC_cells',
        arguments={'min_qc_fragment_records': True, 'min_tss_enrichment': '4.5'},
        extra_facts={'effective_thresholds': thresholds(dict(min_qc_fragment_records=1, min_tss_enrichment='4.5'))})
    view = forbid_work(app, monkeypatch).sessions.evidence('session', rid, 'output0')
    assert view.status == 'unavailable' and not view.facts and view.source is None


def test_pinned_threshold_projection_must_match_existing_owner_normalization(app, monkeypatch):
    arguments = dict(min_qc_fragment_records=2, min_tss_enrichment='4.5')
    rid, _, _ = accepted(app, 'select_scATAC_cells', arguments=arguments,
        extra_facts={'effective_thresholds': thresholds(arguments) | {'min_qc_fragment_records': 1000}})
    view = forbid_work(app, monkeypatch).sessions.evidence('session', rid, 'output0')
    assert view.status == 'unavailable' and not view.facts and view.source is None


def test_selection_origin_projection_obeys_existing_size_bound(app, monkeypatch):
    rid, _, _ = selection(app)
    guarded = forbid_work(app, monkeypatch)
    monkeypatch.setattr(de, 'MAX_FACT_BYTES', 8)
    view = guarded.sessions.evidence('session', rid, 'output0', fields=('selection_threshold_origins',))
    assert view.status == 'available'
    assert view.facts[0].status == 'omitted' and view.facts[0].value is None


def test_selection_parameter_question_reads_rational_owner_values_without_science(app, monkeypatch):
    from test_scientific_dialogue import Model, question
    rid, path, _ = selection(app)
    guarded = forbid_work(app, monkeypatch)
    pinned = path.read_bytes()
    fields = ('effective_thresholds.min_qc_fragment_records',
        'effective_thresholds.min_tss_enrichment.numerator',
        'effective_thresholds.min_tss_enrichment.denominator',
        'selection_threshold_origins.min_qc_fragment_records',
        'selection_threshold_origins.min_tss_enrichment',
        'selection_threshold_origins.min_tss_flank_evidence')
    response = guarded.sessions.respond('session', 'parameters', 'Which selection thresholds were used?',
        interpreter=Model(question(), fields))
    assert response.status == 'answered', response
    claims = {claim.field: claim for claim in response.scientific.claims}
    assert claims[fields[0]].value == 2
    assert claims[fields[1]].value == 9
    assert claims[fields[2]].value == 2
    assert claims[fields[3]].value == 'explicit_execution_argument'
    assert claims[fields[4]].value == 'explicit_execution_argument'
    assert claims[fields[5]].value == 'existing_owner_default'
    assert claims[fields[3]].source['revision_id'] == rid
    assert claims[fields[3]].source['artifact_name'] == 'accepted_run_result'
    assert path.read_bytes() == pinned
