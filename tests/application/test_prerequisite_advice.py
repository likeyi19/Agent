"""Prerequisite explanations use the existing R5 answer/review, with no option authority."""
from dataclasses import asdict, replace
import json

import anndata as ad
import numpy as np
import pytest
from scipy import sparse

from agent.application import LocalResourceAdmission, ResearchAgentApplication
from agent.application.epizoo_handoff import EpiZooDeclaration
from agent.application.raw_handoff import AnnotationDeclaration
from agent.application.session_state import SessionError
from test_dialogue_evidence import app
from test_scientific_guidance import Model, ask, decision, guard


class AdviceModel(Model):
    def complete(self, *, prompt, response_schema):
        public = json.loads(prompt)
        if 'grounding_review_version' in public:
            self.review = public
        return super().complete(prompt=prompt, response_schema=response_schema)


@pytest.mark.parametrize('direct', [False, True])
@pytest.mark.parametrize('paragraphs', [1, 2, 6])
def test_guidance_uses_natural_bounded_paragraphs_without_template(app, monkeypatch, direct, paragraphs):
    app = guard(app, monkeypatch)
    model = AdviceModel(decision(), direct=direct, paragraphs=paragraphs)
    response = ask(app, utterance='What could establish the missing scientific information?', model=model)
    assert response.status == 'answered', response.text
    state = app.sessions.load('session')
    interaction = state.interactions[-1]
    if direct:
        assert response.guidance is None and response.scientific is not None
        assert response.scientific.support == 'insufficient_evidence'
        assert not response.scientific.claims and interaction.guidance_candidates == ()
        assert len(response.scientific.explanation.split('\n\n')) == paragraphs
    else:
        assert len(response.guidance.candidates) == 1
        assert len(response.guidance.candidates[0].explanation.explanation.split('\n\n')) == paragraphs
    assert model.review['retention_policy'] == 'atomic_guidance'
    assert not state.turns and not state.revisions
    assert not list(app.run_store.root.glob('*.json'))


@pytest.mark.parametrize('paragraphs', [0, 7])
def test_advice_preserves_existing_r5_size_bounds(app, monkeypatch, paragraphs):
    response = ask(guard(app, monkeypatch), model=Model(decision(), direct=True, paragraphs=paragraphs))
    assert response.status == 'unavailable'
    assert 'accepted result' not in response.text.lower()
    assert not app.sessions.load('session').revisions


def test_first_result_unknown_advice_is_exactly_presented_without_candidates(tmp_path):
    from agent.application import InteractiveAgentApplication
    from agent.orchestration import PlanningModelProfile
    from agent.providers import PlanningModelFactoryRegistry

    models = []
    def factory(profile):
        model = AdviceModel(decision(), direct=True, paragraphs=2)
        model.model_id = profile.model_id
        models.append(model)
        return model
    profile = PlanningModelProfile('advice', 'scripted', 'simulated/advice')
    service = InteractiveAgentApplication(tmp_path / 'workspace', model_profiles=(profile,),
        default_profile_id=profile.profile_id,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': factory}))
    service.create_session('session')
    text = 'The preparation history is unknown. What records could establish it before an analysis?'
    response = service.submit_turn('session', 'advice', text, expected_generation=0).to_dict()
    assert response['status'] == 'answered', response
    assert response['response']['guidance'] is None
    assert response['response']['scientific']['targets'] == []
    assert response['response']['scientific']['citations'] == []
    snapshot = service._application.sessions.load('session').to_dict()
    assert snapshot['interactions'][0]['utterance'] == text
    assert snapshot['interactions'][0]['guidance_candidates'] == []
    assert not snapshot['turns'] and not snapshot['revisions']
    count = len(models)
    assert service.submit_turn('session', 'advice', text, expected_generation=0).to_dict() == response
    assert len(models) == count and service._application.sessions.load('session').to_dict() == snapshot


@pytest.mark.parametrize('attack', ['unsupported_review', 'invented_claim', 'fake_finding', 'candidate_mix', 'missing_advice'])
def test_advice_rejects_empirical_fabrication_and_retains_whole_review(app, monkeypatch, attack):
    class Attacking(AdviceModel):
        def complete(self, *, prompt, response_schema):
            public = json.loads(prompt)
            result = json.loads(super().complete(prompt=prompt, response_schema=response_schema))
            if 'grounding_review_version' in public and attack == 'unsupported_review':
                first = next(iter(result['verdicts']))
                result['verdicts'][first]['verdict'] = 'unsupported'
            if 'guidance_schema_version' in public:
                if attack == 'invented_claim':
                    result['advice']['paragraphs'][0]['citations'] = ['unoffered']
                elif attack == 'fake_finding':
                    result['advice']['paragraphs'][0]['kind'] = 'finding'
                elif attack == 'candidate_mix':
                    result['candidates'] = [dict(capability='inspect_scATAC', explanation=result['advice'])]
                elif attack == 'missing_advice':
                    del result['advice']
            return json.dumps(result)
    result = ask(guard(app, monkeypatch), model=Attacking(decision(), direct=True))
    assert result.status == 'unavailable' and result.scientific is None and result.guidance is None
    assert 'accepted result' not in result.text.lower()
    interaction = app.sessions.load('session').interactions[-1]
    assert interaction.guidance_candidates is None and interaction.status == 'failed'


@pytest.mark.parametrize('field', ['annotation', 'epizoo'])
@pytest.mark.parametrize('review_fails', [False, True])
def test_raw_owner_requirements_unknown_advice_restart_and_explicit_reply(app, monkeypatch, field, review_fails):
    app = guard(app, monkeypatch)
    root = app.workspace_root / 'uploads'; root.mkdir()
    path = root / 'simulated-counts.h5ad'
    matrix = ad.AnnData(sparse.csr_matrix(np.eye(3, dtype=np.float32)))
    matrix.var_names = ['chr1:1-2', 'chr1:3-4', 'chr1:5-6']
    matrix.write_h5ad(path)
    original_bytes = path.read_bytes()
    owner = LocalResourceAdmission(app._workspace, approved_source_roots=(root,), registry=app.registry)
    source = owner.register('source', path, label='Simulated count source', attribution='Software fixture only.', session_id='session')
    binding = owner.compose_group((source.resource_id,), attachment_candidates=True)
    declared = EpiZooDeclaration(operation='inference', source='source.0') if field == 'epizoo' else AnnotationDeclaration(source='source.0')
    target = 'epizoo_embed_cells' if field == 'epizoo' else 'annotate_scATAC_cell_types'
    initial = dict(kind='request_roles', target=target, required_roles=[], roles=[], arguments=[], **{field:asdict(declared)})
    first = app.sessions.respond('session', 'first', 'Analyze these cells; the preparation history has not been documented.',
        interpreter=Model(initial), submission=dict(execution_inputs={}, registered_input=binding.attribution()))
    assert first.status == 'clarification'
    before = app.sessions.load('session').interactions[-1]

    class OwnerAdvice(AdviceModel):
        def complete(self, *, prompt, response_schema):
            public = json.loads(prompt)
            result = json.loads(super().complete(prompt=prompt, response_schema=response_schema))
            if 'guidance_schema_version' in public:
                request = public['context']['pending_prerequisite']['request']
                assert request['objective'] == before.utterance and request['missing']
                assert request['declarations']['matrix_semantics' if field == 'epizoo' else 'measurement'] is None
                result['advice']['paragraphs'] = [dict(text='Before proceeding, establish the documented requirements: ' +
                    '; '.join(request['missing']), kind='limitation', citations=[], comparisons=[])]
            elif 'grounding_review_version' in public:
                assert public['guidance']['advice']['pending_prerequisite']['request']['field'] == field
                if review_fails:
                    for value in result['verdicts'].values(): value['verdict'] = 'unsupported'
            return json.dumps(result)

    advice = app.sessions.respond('session', 'advice', 'What evidence is still needed, and how could I establish it?',
        interpreter=OwnerAdvice(decision(), direct=True))
    assert advice.status == ('unavailable' if review_fails else 'answered'), advice.text
    kept = app.sessions.load('session').interactions[-1]
    assert kept.prerequisite == before.prerequisite
    assert app.sessions.load('session').interactions[0] == before
    reopened = ResearchAgentApplication(app.workspace_root, registry=app.registry)
    correction = EpiZooDeclaration(matrix_semantics='binary_accessibility') if field == 'epizoo' else AnnotationDeclaration(context='reported study context')
    reply = dict(kind='answer_roles', pending='@roles', roles=[], arguments=[], **{field:asdict(correction)})
    result = reopened.sessions.respond('session', 'reply', 'I can now supply this documented declaration.', interpreter=Model(reply))
    assert result.status == 'clarification'
    state = reopened.sessions.load('session')
    assert state.interactions[-1].prerequisite['origin_turn_id'] == 'first'
    assert state.interactions[-1].admitted[field]['source'] == 'source.0'
    assert state.interactions[-1].admitted[field]['matrix_semantics' if field == 'epizoo' else 'context'] == (
        'binary_accessibility' if field == 'epizoo' else 'reported study context')
    assert not state.turns and not state.revisions and not list(app.run_store.root.glob('*.json'))
    assert path.read_bytes() == original_bytes


def test_advice_does_not_acquire_option_authority_or_allow_missing_completion_record(app, monkeypatch):
    app = guard(app, monkeypatch)
    assert ask(app, model=Model(decision(), direct=True)).status == 'answered'
    state = app.sessions.load('session')
    original = state.interactions[-1]
    with pytest.raises((SessionError, ValueError)):
        replace(state, interactions=(replace(original, guidance_candidates=None),))
    result = ask(app, 'invalid-option', 'Why would the first option help?', Model(decision(candidate='first option')))
    assert result.status == 'clarification' and not app.sessions.load('session').turns


def test_parameter_owner_advice_and_restart_keep_exact_objective_and_source(app, monkeypatch):
    from agent.application import dialogue_execution
    from agent.application.turns import TurnOutcome

    app = guard(app, monkeypatch)
    objective = 'Select cells using a TSS enrichment threshold of -1.'
    start = objective.index('-1')
    declaration = dict(tool='select_scATAC_cells', argument='min_tss_enrichment',
                       literal='-1', start=start, end=start + 2)
    request = dict(kind='execute_plan', target='select_scATAC_cells', arguments=[declaration])
    # This explicit placeholder is not qualified QC. Binding is tested before
    # planning; the eventual continuation is also stopped before consumption.
    inputs = dict(barcode_qc_manifest_path='/explicit/unqualified-qc.json', barcode_qc_manifest_sha256='a' * 64)
    initial = app.sessions.respond('session', 'first', objective, interpreter=Model(request),
        execution_inputs=inputs, submission=dict(execution_inputs=inputs))
    assert initial.status == 'clarification'
    before = app.sessions.load('session').interactions[-1]
    assert before.prerequisite['field'] == 'parameters'
    model = AdviceModel(decision(), direct=True, paragraphs=1)
    advice = ask(app, 'advice', 'What is wrong with that requested threshold and what choice is needed?', model)
    assert advice.status == 'answered'
    pending = model.calls[-1]['context']['pending_prerequisite']['request']
    assert pending['field'] == 'parameters' and pending['objective'] == objective
    assert pending['missing'] == ['min_tss_enrichment']
    assert app.sessions.load('session').interactions[-1].prerequisite == before.prerequisite
    seen = []
    def stop(sessions, interaction, admitted, *args, **kwargs):
        seen.append(admitted)
        return TurnOutcome('execute', 'failed', text='Software fixture stops before planning or science.')
    monkeypatch.setattr(dialogue_execution, 'execute', stop)
    reopened = ResearchAgentApplication(app.workspace_root, registry=app.registry)
    text = 'Use a TSS enrichment threshold of 0.5.'
    position = text.index('0.5')
    reply = dict(kind='answer_parameters', pending='@parameters', arguments=[dict(declaration,
        literal='0.5', start=position, end=position + 3)])
    reopened.sessions.respond('session', 'reply', text, interpreter=Model(reply))
    assert len(seen) == 1
    assert seen[0]['inputs'] == inputs | dict(min_tss_enrichment='0.5')
    assert seen[0]['continuation']['origin_turn_id'] == 'first'
    state = reopened.sessions.load('session')
    assert not state.turns and not state.revisions and not list(app.run_store.root.glob('*.json'))
