"""Retired product entry points cannot return through registration or dialogue."""
import importlib.util
import json
from agent.orchestration import build_default_tool_registry
from agent.application.dialogue_execution import scientific_parameters
from agent.application.turn_decisions import decision_schema


def test_retired_operations_and_modules_are_absent_while_native_science_remains():
    registry=build_default_tool_registry()
    retired={'compare_cell_annotations','evaluate_cell_annotation','corroborate_cell_type_annotation'}
    assert not retired.intersection(registry.names())
    for module in ('cell_label_comparison','annotation_evaluation','cell_type_standardization','annotation_corroboration','marker_validation'):
        assert importlib.util.find_spec('agent.tools.analysis.'+module) is None
    assert {'annotate_scATAC_cell_types','transfer_cell_labels','evaluate_cell_clustering',
        'run_scanpy','run_signac','run_snapatac2','run_episcanpy','inspect_raw_scATAC',
        'run_chromap','build_replicate_pseudobulk','run_replicate_differential_accessibility',
        'build_cell_neighbors','cluster_cells','compute_cell_umap'}<=set(registry.names())
    offers=json.dumps(scientific_parameters(registry))
    assert not any(n in offers for n in (*retired,'prediction_mapping','reference_mapping','string_mapping'))
    assert 'name_evidence' not in json.dumps(decision_schema())
    from agent.orchestration.planning_scope import CAPABILITY_DESCRIPTIONS
    assert 'fixed annotation evaluation' not in CAPABILITY_DESCRIPTIONS['reference_annotation']


def test_annotation_report_preserves_native_names_and_missing_states_without_model_prose():
    from agent.report.analysis_report import render_annotation_facts
    facts=dict(n_cells=5,n_branches=2,levels_reached=1,terminal_populations=2,
        inherited_label_cells=0,branches_omitted=0, population_summary=[
            dict(status='assigned',populations=1,cells=3),
            dict(status='insufficient_evidence',populations=1,cells=2)],branch_summary=[
            dict(branch_id='one',level=1,n_cells=3,label='Original / native+ label',status='assigned',stopping_reason='maximum_levels'),
            dict(branch_id='two',level=1,n_cells=2,label=None,status='insufficient_evidence',stopping_reason='no_supported_subdivision')])
    report=render_annotation_facts(facts)
    assert 'Original / native+ label' in report and '(unassigned)' in report and 'insufficient_evidence' in report
    assert 'Current-round label' in report and 'no_supported_subdivision' in report
    assert 'not independent biological validation' in report



def test_removed_keyed_mapping_wire_is_rejected_by_current_declarations():
    import pytest
    from agent.application.turn_decisions import parse_decision,IntentError
    decision=dict(turn_schema_version=1,decision=dict(kind='execute_plan',target='run_scanpy',arguments=[
        dict(tool='run_scanpy',argument='kwargs',members=[dict(name='resolution',kind='number',
            literal='1',evidence='use 1',name_evidence='resolution')])]))
    with pytest.raises(IntentError):
        parse_decision(json.dumps(decision),utterance='resolution: use 1')
