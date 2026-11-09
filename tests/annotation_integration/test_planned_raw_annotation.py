"""Selected raw count construction can prepare annotation before counts exist."""
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from agent.application import ResearchAgentApplication
from agent.application.annotation_preparation import preparer, AnnotationPrerequisite
from agent.application.raw_handoff import AnnotationDeclaration
from agent.application.resource_continuation import catalog
from agent.schemas import StepOutputRef
from agent.tools.analysis import annotation_input, annotation_contract as c
from test_external_entry import external_case, case
from test_conversational_handoff import choice


@pytest.fixture
def planned(external_case, tmp_path, monkeypatch):
    args, _ = external_case
    app = ResearchAgentApplication(tmp_path/'planned')
    sessions = SimpleNamespace(_application=app)
    admitted = dict(fragments=dict(species='human', assembly='hg38'),
        downstream_annotation=asdict(AnnotationDeclaration(context='synthetic context')),
        resource_catalog=catalog((choice(args),)))
    selected = {
        'import': ('run_snapatac2', dict(function_name='pp.import_fragments', input_path='/original/fragments')),
        'qc': ('run_snapatac2', dict(function_name='metrics.tsse', input_path=StepOutputRef('import','matrix_path'))),
        'filter': ('run_snapatac2', dict(function_name='pp.filter_cells', input_path=StepOutputRef('qc','matrix_path'))),
        'matrix': ('run_snapatac2', dict(function_name='pp.make_peak_matrix', counting_strategy='fragment', input_path=StepOutputRef('filter','matrix_path'))),
        'groups': ('run_snapatac2', dict(function_name='tl.leiden', input_path=StepOutputRef('matrix','matrix_path'))),
    }
    arguments = dict(source_path=StepOutputRef('matrix','matrix_path'), source_sha256=StepOutputRef('matrix','matrix_sha256'),
        group_analysis_path=StepOutputRef('groups','matrix_path'), group_analysis_sha256=StepOutputRef('groups','matrix_sha256'))
    monkeypatch.setattr(annotation_input, 'observe', lambda *a: pytest.fail('Cannot observe a future count file'))
    return preparer(sessions, None, admitted), app.registry.get('annotate_scATAC_cell_types'), arguments, selected


def test_planned_counts_prepare_exact_reference_without_opening_future_data(planned):
    prepare, tool, args, selected = planned
    result = prepare(tool, dict(args), selected)
    spec = c.specification(result['annotation_spec_path'],result['annotation_spec_sha256'])
    assert spec['measurement']=='fragment_counts'
    assert spec['species']=='human' and spec['assembly']=='hg38'
    assert spec['context']=='synthetic context'
    assert result['source_path']==args['source_path']
    assert result['group_analysis_path']==args['group_analysis_path']
    assert result['annotation_method']==c.REGION_PRIMARY


def test_count_preserving_native_chain_keeps_original_count_construction(planned):
    prepare, tool, args, selected = planned
    args['source_path']=StepOutputRef('groups','matrix_path')
    args['source_sha256']=StepOutputRef('groups','matrix_sha256')
    assert prepare(tool,args,selected)['annotation_method']==c.REGION_PRIMARY


@pytest.mark.parametrize('variant', ['paired_default','single','transformed','layer','conflict','missing_groups'])
def test_unknown_or_incompatible_planned_measurements_are_not_relabelled(planned,variant):
    prepare, tool, args, selected = planned
    if variant=='paired_default':selected['matrix'][1].pop('counting_strategy')
    if variant=='single':selected['import'][1]['kwargs']={'is_paired':False}
    if variant=='transformed':selected['matrix'][1]['kwargs']={'use_x':True}
    if variant=='layer':args['count_slot']='layers/unobserved'
    if variant=='conflict':selected['matrix'][1]['kwargs']={'counting_strategy':'insertion'}
    if variant=='missing_groups':
        args.pop('group_analysis_path');args.pop('group_analysis_sha256')
    with pytest.raises((AnnotationPrerequisite, ValueError)):
        prepare(tool,args,selected)


@pytest.mark.parametrize('annotation', [None, AnnotationDeclaration(context='human PBMC')])
def test_raw_interpretation_retains_optional_annotation_context(annotation):
    import json
    from agent.application.raw_handoff import FragmentsDeclaration, declaration_dict
    from agent.application.turn_decisions import parse_decision
    raw=dict(kind='request_roles',target='run_snapatac2',required_roles=[],roles=[],arguments=[],
        fragments=declaration_dict(FragmentsDeclaration(source='source.0',species='human',assembly='hg38')),
        annotation=None if annotation is None else declaration_dict(annotation))
    parsed=parse_decision(json.dumps(dict(turn_schema_version=1,decision=raw)),utterance='Annotate human PBMC fragments on hg38.')
    assert parsed.annotation==annotation
    assert parsed.fragments.source=='source.0'


def test_raw_annotation_cannot_capture_fragment_file_as_a_count_matrix():
    from agent.application.raw_handoff import FragmentsDeclaration
    from agent.application.turn_decisions import RequestRoles
    with pytest.raises(ValueError,match='planned count'):
        RequestRoles('run_snapatac2',(),fragments=FragmentsDeclaration(source='source.0'),
            annotation=AnnotationDeclaration(source='source.0'))
