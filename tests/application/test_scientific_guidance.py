"""M17.1 scripted guidance through the normal durable interaction interface."""
from dataclasses import replace
import json
import subprocess
import sys

import pytest

from agent.application import ResearchAgentApplication, SessionConflictError
from agent.application import scientific_guidance as guidance
from agent.application import scientific_dialogue as dialogue
from agent.application.turn_decisions import parse_decision, IntentError, decision_schema
from agent.schemas.orchestration import _serialize
from agent.orchestration.executor import PlanExecutor
from test_dialogue_evidence import app, accepted, forbid_work
from test_scientific_dialogue import annotation, Model as ScientificModel, question


def decision(*outputs, subject=None, candidate=None):
    return dict(kind='answer_guidance', targets=[dict(output=o, subject=subject) for o in outputs], candidate=candidate)


class Model:
    def __init__(self, choice=None, tools=('inspect_scATAC',), attack=None, callback=None):
        self.choice = decision('r0') if choice is None else choice
        self.tools, self.attack, self.callback, self.calls = tools, attack, callback, []

    def complete(self, *, prompt, response_schema):
        p = json.loads(prompt)
        self.calls.append(p)
        if 'turn_schema_version' in p:
            return json.dumps(dict(turn_schema_version=1, decision=self.choice))
        assert p['guidance_schema_version'] == 1
        if self.callback: self.callback()
        claims = p['context']['evidence']['claims']
        # One exact scientific fact per supplied result; no manufactured prose facts.
        ids = []
        for target in p['context']['evidence']['targets']:
            rows = [c for c in claims if c['target'] == target['handle']]
            if rows: ids.append(rows[0]['claim_id'])
        explanation = dict(support='supported' if ids else 'insufficient_evidence', paragraphs=[
            dict(parts=[dict(kind='text', text='This could help characterize the stated objective.')]
                       + [dict(kind='claim', id=i) for i in ids]),
            dict(parts=[dict(kind='text', text='Assuming the intended experimental design is appropriate.')]),
            dict(parts=[dict(kind='text', text='This is conditional advice and requires further input.')])])
        result = dict(candidates=[dict(capability=t, explanation=explanation) for t in self.tools])
        if self.attack: self.attack(result)
        return json.dumps(result)


def ask(app, turn='guide', utterance='What should I analyze next?', model=None, **kwargs):
    return app.sessions.respond('session', turn, utterance, interpreter=model or Model(), **kwargs)


def guard(app, monkeypatch):
    app = forbid_work(app, monkeypatch)
    def forbidden(*a, **k): raise AssertionError('Guidance entered planning/preflight')
    monkeypatch.setattr(PlanExecutor, 'preflight', forbidden)
    from agent.orchestration.llm_planner import LLMPlanner
    monkeypatch.setattr(LLMPlanner, 'plan', forbidden)
    return app


def test_general_guidance_exact_references_and_no_science(app, monkeypatch):
    rid, path, _ = accepted(app, 'inspect_scATAC')
    original = path.read_bytes()
    app = guard(app, monkeypatch)
    model = Model(tools=('cluster_cells', 'inspect_scATAC'))
    outcome = ask(app, model=model)
    assert outcome.status == 'answered', outcome.text
    assert len(outcome.guidance.candidates) == 2
    c = outcome.guidance.candidates[0]
    assert c.reference['capability'] == 'cluster_cells'
    assert c.reference['targets'][0]['revision_id'] == rid
    assert c.explanation.claims[0].source['output_locator']['run_id'] == 'initial:run'
    assert c.readiness['capability_registered'] is True
    assert c.readiness['readiness'] == 'not_fully_checked'
    assert c.readiness['request_scope'] == 'no_execution_inputs_bound'
    assert 'analysis' in c.readiness['required_ports_without_supplied_request_source']
    assert 'comparison' not in model.calls[-1]['context']['evidence']
    assert path.read_bytes() == original
    state = app.sessions.load('session')
    assert state.active_revision_id == rid and state.generation == 1
    payload = json.dumps(_serialize(state.interactions[-1].admitted))
    assert 'conditional advice' not in payload and 'rationale' not in payload and 'readiness' not in payload
    assert 'claims' not in payload and 'catalog_sha256' in payload


def test_objective_subject_and_followup_restart_stable_candidate(app, monkeypatch):
    annotation(app)
    app = guard(app, monkeypatch)
    first = ask(app, utterance='Why does cluster 3 differ? What could I analyze?',
        model=Model(decision('r0', subject='3'), tools=('inspect_scATAC','cluster_cells')))
    assert first.status == 'answered', first.text
    ref = first.guidance.candidates[0].reference
    model = Model(decision(candidate='first option'), tools=('cluster_cells',))
    restarted = ResearchAgentApplication(app.workspace_root, registry=app.registry)
    follow = ask(restarted, 'why', 'Why would the first option help?', model)
    assert follow.status == 'answered', follow.text
    assert follow.guidance.candidates[0].reference == ref
    assert model.calls[-1]['objective'] == 'Why does cluster 3 differ? What could I analyze?'
    assert model.calls[-1]['context']['evidence']['targets'][0]['subject'] == '3'
    assert 'conditional advice' not in json.dumps(model.calls)
    # Also load durable reference metadata in an independent process.
    code = 'from agent.application import ResearchAgentApplication; import sys; s=ResearchAgentApplication(sys.argv[1]).sessions.load("session"); print(s.interactions[-1].guidance_candidates[0]["candidate_id"])'
    result = subprocess.run([sys.executable,'-B','-c',code,str(app.workspace_root)],capture_output=True,text=True,check=True)
    assert result.stdout.strip() == ref['candidate_id']


def test_heterogeneous_context_is_not_comparison(app, monkeypatch):
    old, _, _ = accepted(app, 'inspect_scATAC')
    new, _, _ = accepted(app, 'build_scATAC_cell_by_features', name='matrix')
    model = Model(decision('r0','r1'))
    result = ask(guard(app, monkeypatch), utterance='What analyses could help, using current and parent version evidence?', model=model)
    assert result.status == 'answered', result.text
    claims = result.guidance.candidates[0].explanation.claims
    assert {c.source['revision_id'] for c in claims} == {old,new}
    evidence = model.calls[-1]['context']['evidence']
    assert len(evidence['targets']) == 2 and 'comparison' not in evidence
    assert 'historical' in result.text
    assert any('correspondence' in x for x in result.guidance.limitations)


@pytest.mark.parametrize('choice,text', [
    (decision('r99'), 'What next?'),
    (decision('r0',subject='99'), 'What about cluster 99 next?'),
    (decision('r0',subject='3'), 'What about cluster 7 next?'),
    (decision('r0','r0'), 'What next?'),
    (decision(candidate='first option'), 'What next?'),
])
def test_invalid_references_fail_closed(app, monkeypatch, choice, text):
    annotation(app)
    model = Model(choice)
    result = ask(guard(app, monkeypatch), utterance=text, model=model)
    assert result.status == 'clarification'
    assert len(model.calls) == 1


@pytest.mark.parametrize('attack', [
    lambda r:r['candidates'][0].update(capability='discover_motifs'),
    lambda r:r['candidates'][0].update(readiness='runnable_now'),
    lambda r:r['candidates'][0].update(historical_binding_supported=True),
    lambda r:r['candidates'][0].update(compatible=True),
    lambda r:r['candidates'][0]['explanation']['paragraphs'][0]['parts'].append(dict(kind='claim',id='invented')),
    lambda r:r['candidates'][0]['explanation']['paragraphs'][0]['parts'].append(dict(kind='meaning',id='preflight_passed')),
    lambda r:r['candidates'][0]['explanation']['paragraphs'][0]['parts'].append(dict(kind='text',text='There are 999 cells.')),
    lambda r:r['candidates'].append(r['candidates'][0]),
])
def test_invalid_generation_has_no_candidates_or_science(app, monkeypatch, attack):
    accepted(app, 'inspect_scATAC')
    result = ask(guard(app, monkeypatch), model=Model(attack=attack))
    assert result.status == 'unavailable' and result.guidance is None
    i = app.sessions.load('session').interactions[-1]
    assert i.status == 'failed' and i.guidance_candidates is None


def test_empty_evidence_is_unknown_not_impossible(app, monkeypatch):
    accepted(app, 'inspect_scATAC')
    model = Model(decision(), tools=('run_replicate_differential_accessibility',))
    result = ask(guard(app, monkeypatch), model=model)
    assert result.status == 'answered', result.text
    candidate = result.guidance.candidates[0]
    assert not candidate.explanation.claims
    assert candidate.explanation.support == 'insufficient_evidence'
    assert 'group_value' in candidate.readiness['required_ports_without_supplied_request_source']
    assert 'not global impossibility' in result.text
    assert model.calls[-1]['context']['captured_output_count'] == 1


def test_catalog_is_shared_projection_and_no_recommendation_rule(app, monkeypatch):
    accepted(app, 'inspect_scATAC')
    app = guard(app, monkeypatch)
    a = ask(app, model=Model(tools=('inspect_scATAC',)))
    b = ask(app, 'other', model=Model(tools=('evaluate_cell_clustering',)))
    assert a.status == b.status == 'answered'
    assert a.guidance.candidates[0].reference['capability'] != b.guidance.candidates[0].reference['capability']
    from agent.orchestration.semantic_prompt import build_semantic_planning_catalog
    from agent.schemas import AgentRequest
    assert guidance.catalog(app.registry) == build_semantic_planning_catalog(AgentRequest('g','g',{}),app.registry)
    catalog = guidance.catalog(app.registry)
    assert len(catalog['tools']) == 23
    assert 'numerator_condition' in catalog['tools']['run_replicate_differential_accessibility'][1]


def test_retry_preserves_candidate_identity_not_model_order(app, monkeypatch):
    accepted(app, 'inspect_scATAC')
    app = guard(app, monkeypatch)
    a = ask(app, model=Model(tools=('inspect_scATAC','cluster_cells')))
    b = ask(app, model=Model(tools=('cluster_cells','inspect_scATAC')))
    assert a.status == b.status == 'answered'
    assert [c.reference for c in a.guidance.candidates] == [c.reference for c in b.guidance.candidates]
    c = ask(app, model=Model(tools=('cluster_cells',)))
    assert c.status == 'unavailable'
    assert len(app.sessions.load('session').interactions[-1].guidance_candidates) == 2


def test_evidence_mutation_during_answer_rejected(app, monkeypatch):
    _, path, _ = accepted(app,'inspect_scATAC')
    result = ask(guard(app,monkeypatch),model=Model(callback=lambda:path.write_text('{}')))
    assert result.status == 'unavailable'


def test_historical_evidence_needs_explicit_reference(app, monkeypatch):
    accepted(app,'inspect_scATAC')
    accepted(app,'inspect_scATAC',name='second')
    result = ask(guard(app,monkeypatch),model=Model(decision('r1')))
    assert result.status == 'clarification'


def test_navigation_does_not_rebase_candidate(app, monkeypatch):
    original,_,_=accepted(app,'inspect_scATAC')
    app=guard(app,monkeypatch)
    a=ask(app)
    assert a.status=='answered'
    # Metadata-only accepted fixture creates a separate revision; no owner work.
    accepted(app,'inspect_scATAC',name='second')
    m=Model(decision(candidate='first option'))
    result=ask(app,'stale','Why would the first option help?',m,predecessor_turn_id='guide')
    assert result.status=='clarification' and len(m.calls)==1
    assert app.sessions.load('session').active_revision_id!=original


@pytest.mark.parametrize('text,choice',[
    ('Run option 1.',dict(kind='execute_plan',base='current',operation='plan',target='inspect_scATAC',delta=None)),
    ('Option 1 sounds right. Do that.',decision(candidate='option 1')),
])
def test_candidate_execution_is_not_implemented(app,monkeypatch,text,choice):
    accepted(app,'inspect_scATAC')
    app=guard(app,monkeypatch)
    assert ask(app).status=='answered'
    result=ask(app,'run',text,Model(choice))
    assert result.status=='clarification' and result.clarification.reason=='unsupported_intent'


def test_guidance_to_scientific_followup_uses_same_focus(app,monkeypatch):
    annotation(app)
    app=guard(app,monkeypatch)
    assert ask(app,utterance='What could I analyze for cluster 3?',model=Model(decision('r0',subject='3'))).status=='answered'
    result=ask(app,'facts','What does that result show?',ScientificModel(question('@focus'),['primary_annotation']))
    assert result.status=='answered' and result.scientific.claims[0].subject=='3'


def test_bounds_and_corrupt_reference_validation(app,monkeypatch):
    accepted(app,'inspect_scATAC')
    app=guard(app,monkeypatch)
    assert ask(app).status=='answered'
    s=app.sessions.load('session')
    i=s.interactions[-1]
    a=_serialize(i.guidance_candidates)
    a[0]['capability']='fake'
    with pytest.raises(ValueError):
        replace(s,interactions=(replace(i,guidance_candidates=a),))
    monkeypatch.setattr(guidance,'MAX_CONTEXT_BYTES',1)
    assert ask(app,'overflow').status=='unavailable'
    with pytest.raises(IntentError):
        parse_decision(json.dumps(dict(turn_schema_version=1,decision=decision(*(['r0']*5)))))


def test_navigation_during_generation_marks_captured_result_historical(app,monkeypatch):
    old,_,_=accepted(app,'inspect_scATAC')
    new,_,_=accepted(app,'inspect_scATAC',name='second')
    app=guard(app,monkeypatch)
    def navigate():
        app.sessions.switch('session','nav',old,expected_generation=2)
    result=ask(app,model=Model(callback=navigate))
    assert result.status=='answered',result.text
    assert 'historical revision' in result.text
    assert result.guidance.candidates[0].reference['base_revision_id']==new
    assert app.sessions.load('session').active_revision_id==old


def test_missing_and_changed_evidence_fail_before_generation(app,monkeypatch):
    _,path,_=accepted(app,'inspect_scATAC')
    app=guard(app,monkeypatch)
    path.unlink()
    model=Model()
    result=ask(app,model=model)
    assert result.status=='clarification' and len(model.calls)==1


def test_multiple_candidates_require_exact_followup_reference(app,monkeypatch):
    accepted(app,'inspect_scATAC')
    app=guard(app,monkeypatch)
    assert ask(app,model=Model(tools=('inspect_scATAC','cluster_cells'))).status=='answered'
    result=ask(app,'ambiguous','Why would that help?',Model(decision(candidate='@candidate')))
    assert result.status=='clarification'


def test_concurrent_completion_cannot_replace_candidate_identity(app,monkeypatch):
    accepted(app,'inspect_scATAC')
    app=guard(app,monkeypatch)
    other=[]
    def competing():
        other.append(ask(app,model=Model(tools=('cluster_cells',))))
    result=ask(app,model=Model(tools=('inspect_scATAC',),callback=competing))
    assert other[0].status=='answered'
    assert result.status=='unavailable'
    refs=app.sessions.load('session').interactions[-1].guidance_candidates
    assert [r['capability'] for r in refs]==['cluster_cells']


def test_guidance_schema_remains_strict_and_source_scoped(app,monkeypatch):
    accepted(app,'inspect_scATAC')
    app=guard(app,monkeypatch)
    class ValidatingModel(Model):
        def complete(self,*,prompt,response_schema):
            response=super().complete(prompt=prompt,response_schema=response_schema)
            def closed(schema):
                if isinstance(schema, dict):
                    if schema.get('type') == 'object':
                        assert schema['additionalProperties'] is False
                        assert set(schema['required']) == set(schema['properties'])
                    for value in schema.values(): closed(value)
                elif isinstance(schema,list):
                    for value in schema: closed(value)
            closed(response_schema)
            if 'guidance_schema_version' in json.loads(prompt):
                candidate=response_schema['properties']['candidates']['items']['properties']
                assert set(candidate)=={'capability','explanation'}
                assert 'inspect_scATAC' in candidate['capability']['enum']
                assert candidate['explanation']['properties']['paragraphs']['minItems']==3
            return response
    assert ask(app,model=ValidatingModel()).status=='answered'
    state=app.sessions.load('session')
    captured=_serialize(state.interactions[-1].snapshot)
    public=dialogue.public(captured,state,app.registry,sessions=app.sessions,utterance='What next?')
    assert 'guidance_predecessor' in public
    # Registry descriptions and precise input requirements, not a runnable flag.
    c=guidance.catalog(app.registry)['tools']['run_replicate_differential_accessibility']
    assert c[1]['design_type'][0] and c[1]['design_type'][5][2]


def test_plain_m16_interaction_serialization_has_no_guidance_field(app):
    accepted(app,'inspect_scATAC')
    assert ask(app,model=ScientificModel(question())).status=='answered'
    record=app.sessions.load('session').to_dict()['interactions'][0]
    assert 'guidance_candidates' not in record


def test_candidate_references_and_readiness_are_immutable(app):
    accepted(app,'inspect_scATAC')
    result=ask(app)
    candidate=result.guidance.candidates[0]
    with pytest.raises(TypeError): candidate.reference['capability']='fake'
    with pytest.raises(TypeError): candidate.readiness['readiness']='runnable_now'
    value=result.guidance.to_dict()
    value['candidates'][0]['reference']['targets'].clear()
    assert candidate.reference['targets']


def test_unknown_scientific_subject_is_not_inferred_across_outputs(app,monkeypatch):
    accepted(app,'inspect_scATAC')
    app=guard(app,monkeypatch)
    outcome=ask(app,utterance='What could I analyze for cluster 4?',model=Model(decision('r0',subject='4')))
    assert outcome.status=='clarification'


def test_changed_registry_semantics_invalidate_existing_candidate(app,monkeypatch):
    accepted(app,'inspect_scATAC')
    assert ask(app).status=='answered'
    from agent.orchestration import ToolRegistry
    registry=ToolRegistry(tuple(replace(app.registry.get(n),planning=replace(app.registry.get(n).planning,
        description='Changed registered semantics.')) if n=='inspect_scATAC' else app.registry.get(n) for n in app.registry.names()))
    changed=ResearchAgentApplication(app.workspace_root,registry=registry)
    result=ask(guard(changed,monkeypatch),'follow','Why would the first option help?',Model(decision(candidate='first option')))
    assert result.status=='clarification' and result.clarification.reason=='unavailable_context'


def test_corrupt_followup_cannot_change_selected_context(app):
    annotation(app)
    assert ask(app,utterance='What could I analyze for cluster 3?',model=Model(decision('r0',subject='3'))).status=='answered'
    assert ask(app,'why','Why would the first option help?',Model(decision(candidate='first option'))).status=='answered'
    state=app.sessions.load('session')
    follow=state.interactions[-1]
    admitted=_serialize(follow.admitted)
    admitted['targets'][0]['subject']='7'
    admitted['target']['subject']='7'
    with pytest.raises(ValueError):
        replace(state,interactions=state.interactions[:-1]+(replace(follow,admitted=admitted),))


def test_single_target_does_not_satisfy_other_candidate_prerequisites(app):
    accepted(app,'inspect_scATAC')
    result=ask(app,model=Model(tools=('run_replicate_differential_accessibility',)))
    assert result.status=='answered'
    readiness=result.guidance.candidates[0].readiness
    assert readiness['accepted_evidence_handles']==('t0',)
    assert readiness['readiness']=='not_fully_checked'
    assert 'numerator_condition' in readiness['required_scientific_parameters']
    assert 'pseudobulk' in readiness['required_ports_without_supplied_request_source']
    assert not any(k in readiness for k in ('runnable_now','historical_binding_supported','compatible'))


def test_multiple_guidance_sources_do_not_become_implicit_single_subject(app):
    accepted(app,'inspect_scATAC')
    accepted(app,'build_scATAC_cell_by_features',name='second')
    assert ask(app,utterance='What next, given current and parent version evidence?',model=Model(decision('r0','r1'))).status=='answered'
    result=ask(app,'ambiguous','Why?',ScientificModel(question('@focus')))
    assert result.status=='clarification'
