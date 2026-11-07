"""Bounded external-fragment declarations share raw task admission and replay."""
from dataclasses import asdict, replace
from types import SimpleNamespace
import pytest
from agent.application.raw_handoff import FragmentsDeclaration, FastqDeclaration, BamDeclaration, declaration, merge
from agent.application.native_raw_handoff import requirements
from agent.application.turn_decisions import RequestRoles, RoleAnswer, parse_decision, decision_schema


@pytest.mark.parametrize('changes', [dict(source='/tmp/data'), dict(feature_source='source.16'),
    dict(source='source.0', feature_source='source.0'), dict(source='source.0', index_source='source.0'),
    dict(feature_category='curated_ccre'), dict(source_profile='generic'), dict(namespace='../lib'),
    dict(species='mouse', assembly='hg38'), dict(resource='/reference')])
def test_closed_fragments_declarations(changes):
    with pytest.raises(ValueError): FragmentsDeclaration(**changes)


def test_exact_partial_declarations_and_immutable_roles():
    initial = FragmentsDeclaration(source='source.0', species='human', feature_category='peak_set', resource='resource.old')
    final = merge(initial, FragmentsDeclaration(feature_source='source.1', assembly='hg38', resource='resource.new'))
    assert declaration(asdict(final), 'fragments') == final
    for addition in (FragmentsDeclaration(source='source.2'), FragmentsDeclaration(species='mouse'), FastqDeclaration()):
        with pytest.raises(ValueError): merge(initial, addition)
    with pytest.raises(ValueError): declaration(asdict(final) | dict(qualified=True), 'fragments')
    for conflicting in (dict(fastq=FastqDeclaration()), dict(bam=BamDeclaration())):
        with pytest.raises(ValueError): RoleAnswer('@roles', (), fragments=final, **conflicting)
    with pytest.raises(ValueError): RequestRoles('run_chromap', (), fragments=final)
    with pytest.raises(ValueError): requirements(SimpleNamespace(members=({},)), final, (), evidence={})


def test_source_is_required_but_unused_history_and_references_are_not():
    missing,selected=requirements(SimpleNamespace(members=({},{})),FragmentsDeclaration(),(),{})
    assert selected is None and len(missing)==1 and 'attachment' in missing[0]
    declared=FragmentsDeclaration(source='source.0',feature_category='peak_set')
    assert requirements(SimpleNamespace(members=({},{})),declared,(),{})==([],None)


def test_scientific_arguments_cannot_replace_source_declarations():
    from agent.application.raw_handoff import _fragments_values
    declared = FragmentsDeclaration(namespace='original', source_profile='10x-atac-fragments.v1')
    selected = SimpleNamespace(inputs=lambda: {})
    assert _fragments_values(declared, selected, {'namespace': 'original'}, None)['namespace'] == 'original'
    with pytest.raises(ValueError):
        _fragments_values(declared, selected, {'namespace': 'other'}, None)


def test_ordinary_provenance_can_bind_reviewed_choice_without_internal_name():
    from agent.application.dialogue_execution import _bind_arguments
    from agent.application.turn_decisions import ScalarChoiceArgument
    from agent.orchestration.registry import build_default_tool_registry
    utterance = ('Import this 10x ATAC export with already Tn5-adjusted zero-based half-open '
                 'intervals, corrected barcodes and read-pair support.')
    declaration = ScalarChoiceArgument('import_scATAC_fragments', 'source_profile',
        utterance[7:-1], 7, len(utterance)-1, '10x-atac-fragments.v1')
    from agent.orchestration.raw_history import ReplayRegistry
    with pytest.raises(ValueError): _bind_arguments(build_default_tool_registry(),utterance,{},(declaration,))
    assert _bind_arguments(ReplayRegistry(build_default_tool_registry()),utterance,{},(declaration,),historical=True)=={'source_profile':'10x-atac-fragments.v1'}


@pytest.mark.parametrize('reason', ['invalid_parameter_value', 'ungrounded_operand',
                                    'conflicting_scientific_parameter'])
def test_named_parameter_clarification_matches_offered_wire(reason):
    import json
    result = parse_decision(json.dumps(dict(turn_schema_version=1,
        decision=dict(kind='clarify', reason=reason,
                      choices=['run_snapatac2.function_name']))))
    assert result.reason == reason
    assert result.choices == ('run_snapatac2.function_name',)
