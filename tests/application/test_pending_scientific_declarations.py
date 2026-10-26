"""Pending user intent can be corrected; scientific source authority stays exact."""
from dataclasses import asdict
import json

import pytest

from agent.application.raw_handoff import (
    AnnotationDeclaration, BamDeclaration, FastqDeclaration, FragmentsDeclaration,
    merge, scientific_metadata_fields,
)
from agent.application.epizoo_handoff import EpiZooDeclaration, requirements
from agent.application.turn_decisions import RoleAnswer, IntentError, parse_decision, decision_schema
from test_dialogue_evidence import app


@pytest.mark.parametrize('initial,addition,field,expected', [
    (EpiZooDeclaration(source='source.0', scientific_name='mouse'),
     EpiZooDeclaration(scientific_name='Mus musculus'), 'scientific_name', 'Mus musculus'),
    (AnnotationDeclaration(source='source.0', context='blood'),
     AnnotationDeclaration(context='human PBMC'), 'context', 'human PBMC'),
    (BamDeclaration(source='source.0', species='human'),
     BamDeclaration(species='mouse'), 'species', 'mouse'),
    (FragmentsDeclaration(source='source.0', assembly='hg38'),
     FragmentsDeclaration(assembly='mm10'), 'assembly', 'mm10'),
    (FastqDeclaration(species='mouse'), FastqDeclaration(species='human'), 'species', 'human'),
])
def test_pending_metadata_corrections_are_not_source_evidence(initial, addition, field, expected):
    with pytest.raises(ValueError, match='cannot be replaced'):
        merge(initial, addition)
    updated = merge(initial, addition, pending=True)
    assert getattr(updated, field) == expected
    if hasattr(initial, 'source'):
        assert updated.source == initial.source


@pytest.mark.parametrize('initial,addition,field', [
    (EpiZooDeclaration(matrix_semantics='fragment_counts'), EpiZooDeclaration(), 'matrix_semantics'),
    (AnnotationDeclaration(measurement='reported counts'), AnnotationDeclaration(), 'measurement'),
    (BamDeclaration(source_profile='agent-cb-paired-atac.v1'), BamDeclaration(), 'source_profile'),
    (FragmentsDeclaration(source_profile='10x-atac-fragments.v1'), FragmentsDeclaration(), 'source_profile'),
    (FastqDeclaration(barcode_policy='raw_sequence_whitelist'), FastqDeclaration(), 'barcode_policy'),
])
def test_explicit_unknown_retracts_history_but_nullable_omission_does_not(initial, addition, field):
    assert merge(initial, addition, pending=True) == initial
    assert getattr(merge(initial, addition, pending=True, unknown_fields=(field,)), field) is None
    with pytest.raises(ValueError):
        merge(initial, addition, unknown_fields=(field,))


@pytest.mark.parametrize('initial,addition', [
    (EpiZooDeclaration(source='source.0'), EpiZooDeclaration(source='source.1')),
    (EpiZooDeclaration(operation='inference'), EpiZooDeclaration(operation='posttraining')),
    (AnnotationDeclaration(source='source.0'), AnnotationDeclaration(source='source.1')),
    (BamDeclaration(namespace='one'), BamDeclaration(namespace='two')),
    (FragmentsDeclaration(index_source='source.1'), FragmentsDeclaration(index_source='source.2')),
])
def test_pending_updates_cannot_replace_source_identity_or_operation(initial, addition):
    with pytest.raises(IntentError) as error:
        merge(initial, addition, pending=True)
    assert error.value.reason == 'invalid_prerequisite'
    assert error.value.operand == 'declaration'


@pytest.mark.parametrize('field', ['source', 'operation', 'namespace', 'resource', 'genome_resource',
                                    'checkpoint_source', 'roles', 'invented'])
def test_unknown_cannot_retract_immutable_handles_or_invent_fields(field):
    with pytest.raises(ValueError):
        RoleAnswer('@roles', (), epizoo=EpiZooDeclaration(), unknown_fields=(field,))


def test_unknown_is_explicit_bounded_typed_intent_and_old_decisions_remain_readable():
    value = dict(kind='answer_roles', pending='@roles', roles=[], arguments=[],
                 epizoo=asdict(EpiZooDeclaration()))
    assert parse_decision(json.dumps(dict(turn_schema_version=1, decision=value))).unknown_fields == ()
    value['unknown_fields'] = ['matrix_semantics']
    decision = parse_decision(json.dumps(dict(turn_schema_version=1, decision=value)))
    assert decision.unknown_fields == ('matrix_semantics',)
    for bad in ('matrix_semantics', ['matrix_semantics', 'matrix_semantics'], ['operation']):
        with pytest.raises((ValueError, IntentError)):
            parse_decision(json.dumps(dict(turn_schema_version=1, decision=value | dict(unknown_fields=bad))))
    with pytest.raises(ValueError):
        RoleAnswer('@roles', (), epizoo=EpiZooDeclaration(matrix_semantics='fragment_counts'),
                   unknown_fields=('matrix_semantics',))


@pytest.mark.parametrize('field', ['fastq', 'bam', 'fragments', 'annotation', 'epizoo'])
def test_unknown_schema_only_offers_the_pending_declarations_metadata(field):
    from agent.application.raw_handoff import empty
    public = dict(bases={}, relations=['current'], dialogue=dict(
        outputs=[], tools=['epizoo_embed_cells'], attachment_candidates=True,
        selected_sources=[dict(source='source.0')],
        pending_prerequisite=dict(field=field)))
    public['dialogue'][field + '_resources'] = []
    schema = decision_schema(public)
    assert '"uniqueItems"' not in json.dumps(schema)
    variants = schema['properties']['decision']['anyOf']
    answer, = [v for v in variants if v['properties']['kind']['enum'] == ['answer_roles']]
    choices = answer['properties']['unknown_fields']['items']['enum']
    assert set(choices) == set(scientific_metadata_fields(empty(field)))
    assert not {'source', 'operation', 'resource'}.intersection(choices)
    if field == 'epizoo':
        assert 'matrix_semantics' in choices


def test_scientific_species_identity_uses_owner_validation_without_alias_inference():
    from types import SimpleNamespace
    binding = SimpleNamespace(members=[dict(source_sha256='a'*64)])
    declared = EpiZooDeclaration(operation='inference', source='source.0',
        scientific_name='mouse', taxonomy_id=10090, assembly='mm10')
    source = dict(interface_version=2, source_sha256='a'*64, feature_membership_sha256='f'*64)
    _, missing, selected = requirements(binding, declared, (), {}, source, None)
    assert selected is None and any('scientific species name' in text for text in missing)
    assert any('processing history' in text for text in missing)
    # Old captured observation questions stay byte-compatible and readable.
    _, legacy, _ = requirements(binding, declared, (), {}, {k:v for k,v in source.items() if k!='interface_version'}, None)
    assert not any('common name' in text for text in legacy)
    corrected = merge(declared, EpiZooDeclaration(scientific_name='Mus musculus'), pending=True)
    _, missing, _ = requirements(binding, corrected, (), {}, source, None)
    assert not any('scientific species name' in text for text in missing)
    assert corrected.matrix_semantics is None


def test_unknown_measurement_is_a_prerequisite_not_a_claim_of_resource_conflict():
    from types import SimpleNamespace
    from agent.application.input_sets import ScientificInputSet
    model = ScientificInputSet('exact', 'Qualified target model', dict(
        target_model_manifest_path='/qualified/model.json', target_model_manifest_sha256='b'*64))
    observed = dict(operation='inference', species=dict(scientific_name='Danio rerio', taxonomy_id=7955),
        assembly='danRer11', matrix_semantics='binary_accessibility', feature_membership_sha256='f'*64)
    declared = EpiZooDeclaration(operation='inference', source='source.0', scientific_name='Danio rerio',
        taxonomy_id=7955, assembly='danRer11', resource='resource.exact')
    source = dict(interface_version=2, source_sha256='a'*64, feature_membership_sha256='f'*64)
    _, missing, selected = requirements(SimpleNamespace(members=[dict(source_sha256='a'*64)]),
        declared, (model,), {'exact': observed}, source, None)
    assert selected is None and any('processing history' in text for text in missing)
    assert not any('incompatible' in text for text in missing)


@pytest.mark.parametrize('advice_status', ['answered', 'unavailable'])
def test_correction_unknown_advice_and_later_continuation_keep_one_objective(tmp_path, monkeypatch, advice_status):
    import anndata as ad
    import numpy as np
    import scipy.sparse as sp
    from agent.application import ResearchAgentApplication, LocalResourceAdmission
    from agent.schemas.orchestration import _serialize
    from test_scientific_guidance import Model, guard, decision as advice

    app = ResearchAgentApplication(tmp_path/'workspace')
    app.sessions.create('session')
    upload = app.workspace_root/'uploads'; upload.mkdir()
    path = upload/'structural.h5ad'
    data = ad.AnnData(sp.csr_matrix(np.eye(3, dtype=np.float32)))
    data.var_names = ['chr1:1-2', 'chr1:3-4', 'chr1:5-6']
    data.write_h5ad(path)
    original = path.read_bytes()
    owner = LocalResourceAdmission(app._workspace, approved_source_roots=(upload,), registry=app.registry)
    record = owner.register('source', path, label='Simulated scATAC input', attribution='Software fixture only.',
                            session_id='session')
    binding = owner.compose_group([record.resource_id], attachment_candidates=True)
    submission = dict(execution_inputs={}, registered_input=binding.attribution())
    app = guard(app, monkeypatch)

    initial = dict(kind='request_roles', target='epizoo_embed_cells', required_roles=[], roles=[], arguments=[],
        epizoo=asdict(EpiZooDeclaration(operation='inference', source='source.0', scientific_name='mouse',
                                      matrix_semantics='fragment_counts')))
    first = app.sessions.respond('session', 'first', 'Please embed these cells with the model.',
                                 interpreter=Model(initial), submission=submission)
    assert first.status == 'clarification'
    if advice_status == 'answered':
        invalid = dict(kind='answer_roles', pending='@roles', roles=[], arguments=[],
                       epizoo=asdict(EpiZooDeclaration(source='source.1')))
        failed = app.sessions.respond('session', 'bad-binding', 'Please keep the same dataset and continue.',
                                      interpreter=Model(invalid))
        assert failed.status == 'failed' and failed.error.code == 'INTERACTIVE_INTENT_BINDING_INVALID'
        kept = app.sessions.load('session').interactions[-1]
        assert kept.admitted is None and kept.prerequisite['origin_turn_id'] == 'first'
        assert kept.prerequisite['binding_turn_id'] == 'first'
        assert len(app.sessions.load('session').interactions) == 2  # No semantic retry.
    corrected = dict(kind='answer_roles', pending='@roles', roles=[], arguments=[],
        epizoo=asdict(EpiZooDeclaration(scientific_name='Mus musculus', taxonomy_id=10090, assembly='mm10')),
        unknown_fields=['matrix_semantics'])
    second = app.sessions.respond('session', 'correct', 'The scientific species is Mus musculus, taxonomy 10090, mm10; counting history is unknown.',
                                  interpreter=Model(corrected))
    assert second.status == 'clarification'
    state = app.sessions.load('session')
    assert state.interactions[-1].admitted['epizoo']['matrix_semantics'] is None
    assert state.interactions[-1].admitted['unknown_fields'] == ('matrix_semantics',)
    prior_identity = _serialize(state.interactions[-1].admitted['registered_input'])
    attack = (lambda response: response['candidates'][0].update(capability='not_registered')) if advice_status == 'unavailable' else None
    guidance_model = Model(advice(), tools=('epizoo_embed_cells',), attack=attack)
    answer = app.sessions.respond('session', 'advice', 'Which evidence would establish the missing measurement history?',
                                  interpreter=guidance_model)
    assert answer.status == advice_status, answer.text
    context = guidance_model.calls[-1]['context']['pending_prerequisite']
    assert context['request']['declarations']['matrix_semantics'] is None
    assert context['request']['declarations']['scientific_name'] == 'Mus musculus'
    assert context['request']['objective'] == 'Please embed these cells with the model.'
    assert context['request']['missing'] and not guidance_model.calls[-1]['context']['evidence']['targets']
    state = app.sessions.load('session')
    assert state.interactions[-1].prerequisite['origin_turn_id'] == 'first'
    restored = ResearchAgentApplication(app.workspace_root, registry=app.registry)
    later = dict(kind='answer_roles', pending='@roles', roles=[], arguments=[],
                 epizoo=asdict(EpiZooDeclaration(matrix_semantics='binary_accessibility')))
    third = restored.sessions.respond('session', 'later', 'The documented preparation establishes binary accessibility.',
                                      interpreter=Model(later))
    assert third.status == 'clarification'  # No qualified model was supplied.
    final = restored.sessions.load('session')
    assert final.interactions[-1].prerequisite['origin_turn_id'] == 'first'
    assert _serialize(final.interactions[-1].admitted['registered_input']) == prior_identity
    assert final.interactions[-1].admitted['epizoo']['matrix_semantics'] == 'binary_accessibility'
    assert 'unknown_fields' not in final.interactions[-1].admitted
    assert not final.revisions and not final.turns and final.generation == 0
    assert not list((app.workspace_root/'run_state').glob('*.json'))
    assert path.read_bytes() == original


@pytest.mark.parametrize('direct', [False, True])
def test_resource_advice_retains_exact_configuration_continuation(app, monkeypatch, direct):
    from agent.application import ResearchAgentApplication
    from agent.application.input_sets import ScientificInputSet
    from agent.application import dialogue_execution
    from agent.application.turns import TurnOutcome
    from test_scientific_guidance import Model, guard, decision as advice

    # A simulated configured parameter supplement exercises ordinary resource
    # continuation. It does not qualify a dataset or execute PCA.
    configured = ScientificInputSet('method', 'Configured PCA settings', {'n_components': 2})
    inputs = dict(input_path='/explicit/unchanged-source.h5ad')
    objective = 'Explore this supplied matrix using the selected configured method settings.'
    request = dict(kind='request_resources', target='pca_embed_cells', arguments=[])
    app = guard(app, monkeypatch)
    first = app.sessions.respond('session', 'first', objective, interpreter=Model(request),
        submission=dict(execution_inputs=inputs), execution_inputs=inputs, input_sets=(configured,))
    assert first.status == 'clarification'
    model = Model(advice(), tools=('pca_embed_cells',), direct=direct)
    answer = app.sessions.respond('session', 'advice', 'What do these settings establish and what remains unchecked?',
        interpreter=model, input_sets=(configured,))
    assert answer.status == 'answered'
    assert model.calls[-1]['context']['pending_prerequisite']['request']['field'] == 'resources'
    seen = []
    def stop_before_planning(sessions, interaction, admitted, *args, **kwargs):
        seen.append(admitted)
        return TurnOutcome('execute', 'failed', text='Software fixture stops before planning or science.')
    monkeypatch.setattr(dialogue_execution, 'execute', stop_before_planning)
    reopened = ResearchAgentApplication(app.workspace_root, registry=app.registry)
    reply = dict(kind='answer_resources', pending='@resources', choice='resource.method', arguments=[])
    reopened.sessions.respond('session', 'reply', 'Use the configured PCA settings.',
        interpreter=Model(reply), input_sets=(configured,))
    assert len(seen) == 1
    assert seen[0]['inputs'] == inputs | dict(n_components=2)
    assert seen[0]['continuation']['origin_turn_id'] == 'first'
    state = reopened.sessions.load('session')
    assert not state.revisions and not state.turns and state.generation == 0
    assert state.interactions[1].prerequisite['origin_turn_id'] == 'first'
