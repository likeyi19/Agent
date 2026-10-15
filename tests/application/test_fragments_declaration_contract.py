"""Bounded external-fragment declarations share raw task admission and replay."""
from dataclasses import asdict, replace
from types import SimpleNamespace
import pytest
from agent.application.raw_handoff import FragmentsDeclaration, FastqDeclaration, BamDeclaration, declaration, merge, requirements
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
    with pytest.raises(ValueError): RequestRoles('prepare_scATAC_fragments', (), fragments=final)
    with pytest.raises(ValueError): requirements(SimpleNamespace(members=({},)), final, (), evidence={})


def test_missing_history_and_roles_remain_questions_without_science():
    missing, selected = requirements(SimpleNamespace(members=({}, {})), FragmentsDeclaration(), (), evidence={})
    assert selected is None
    assert any('corrected' in x for x in missing) and any('assembly' in x for x in missing)
    declared = FragmentsDeclaration(source='source.0', namespace='sample', species='human', assembly='hg38',
        source_profile='10x-atac-fragments.v1', feature_category='peak_set')
    missing, selected = requirements(SimpleNamespace(members=({}, {})), declared, (), evidence={})
    assert any('BED' in x for x in missing) and any('plain genome FASTA' in x for x in missing)
    assert selected is None


def test_scientific_arguments_cannot_replace_source_declarations():
    from agent.application.raw_handoff import _fragments_values
    declared = FragmentsDeclaration(namespace='original', source_profile='10x-atac-fragments.v1')
    selected = SimpleNamespace(inputs=lambda: {})
    assert _fragments_values(declared, selected, {'namespace': 'original'}, None)['namespace'] == 'original'
    with pytest.raises(ValueError):
        _fragments_values(declared, selected, {'namespace': 'other'}, None)
