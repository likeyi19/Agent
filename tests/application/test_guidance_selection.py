"""Explicit selection uses stable references and ordinary execution admission."""
import json

import pytest

from agent.application import ResearchAgentApplication
from agent.application.turn_decisions import parse_decision, ExecuteCandidate
from agent.schemas import AgentPlan, PlanStep
from agent.orchestration.active_context import current_context
from test_dialogue_evidence import app, accepted, forbid_work
from test_scientific_guidance import ask, Model as Guidance, decision
from test_scientific_dialogue import Model as Dialogue, question


class Selection(Dialogue):
    def __init__(self, candidate, evidence, **kwargs):
        super().__init__(dict(kind='execute_candidate',candidate=candidate,evidence=evidence), **kwargs)


def offer(app, targets=(), tools=('inspect_scATAC',)):
    result = ask(app, model=Guidance(decision(*targets), tools=tools))
    assert result.status == 'answered', result
    return result.guidance.candidates[0].reference['candidate_id']


def planner_app(app, tmp_path, failure=None):
    import anndata as ad
    from scipy import sparse
    from agent.orchestration.planner import PlannerError
    path = tmp_path / 'input.h5ad'
    ad.AnnData(sparse.csr_matrix([[1,0],[0,2]])).write_h5ad(path)
    calls = []
    class Planner:
        def plan(self, request, registry):
            calls.append((request, current_context()))
            if failure == 'planner': raise PlannerError('Explicit test rejection')
            return AgentPlan('inspect-plan', request.request_id, 'Explicit inspection.',
                (PlanStep('inspect','inspect_scATAC',dict(path=str(path if failure != 'preflight' else tmp_path/'missing.h5ad'))),))
    return ResearchAgentApplication(app.workspace_root, planner=Planner()), path, calls


@pytest.mark.parametrize('utterance,evidence', [
    ('Option 1 sounds right. Do that.','Do that.'),
    ('Run the inspection option.','Run the inspection option.'),
    ('Run option 1, but use X as the comparison.','Run option 1'),
])
def test_selection_restarts_hands_off_and_replays(app, tmp_path, utterance, evidence):
    accepted(app,'inspect_scATAC')
    cid = offer(app)
    original = app.sessions.load('session').interactions[0]
    app, path, calls = planner_app(app,tmp_path)
    model = Selection(cid,evidence)
    result = ask(app,'execute',utterance,model,execution_inputs={'path':str(path)})
    assert result.status == 'activated', result
    assert len(calls) == 1 and calls[0][1].items == ()
    intent = json.loads(calls[0][0].prompt)
    assert intent['objective'] == 'What should I analyze next?'
    assert intent['selected_capability'] == 'inspect_scATAC'
    assert intent['current_user_request'] == utterance
    state = app.sessions.load('session')
    assert state.interactions[0] == original
    assert state.interactions[-1].admitted['selected_candidate']['candidate_id'] == cid
    assert model.calls[0]['dialogue']['execution_candidates'][0]['candidate'] == cid
    assert 'guidance_schema_version' not in json.dumps(model.calls)
    replay = ask(app,'execute',utterance,model,execution_inputs={'path':str(path)})
    assert replay.status == 'activated' and len(calls) == 1


@pytest.mark.parametrize('utterance,evidence', [
    ('That sounds interesting.','That sounds interesting.'),
    ('Do not run option 1.','Do not run option 1.'),
    ('If ready, run option 1.','run option 1.'),
    ('Why would option 1 help?','Run option 1.'),
])
def test_interest_questions_and_invented_command_do_not_execute(app,monkeypatch,utterance,evidence):
    accepted(app,'inspect_scATAC'); cid=offer(app)
    result=ask(forbid_work(app,monkeypatch),'select',utterance,Selection(cid,evidence))
    assert result.status=='clarification'


@pytest.mark.parametrize('bad', ['unknown','stale','weak','evidence','catalog'])
def test_invalid_selection_fails_before_planner(app,monkeypatch,bad):
    _, path, _ = accepted(app,'inspect_scATAC')
    cid=offer(app,targets=('r0',) if bad in ('weak','evidence') else ())
    if bad=='unknown': cid='candidate-from-another-session'
    if bad=='stale': accepted(app,'inspect_scATAC',name='newer')
    if bad=='evidence': path.write_text('{}')
    if bad=='catalog':
        from agent.application import scientific_guidance
        original=scientific_guidance.catalog
        monkeypatch.setattr(scientific_guidance,'catalog',lambda registry: original(registry) | dict(changed=True))
    result=ask(forbid_work(app,monkeypatch),'select','Run option 1.',Selection(cid,'Run option 1.'))
    assert result.status=='clarification',result


@pytest.mark.parametrize('failure',['planner','preflight'])
def test_normal_execution_rejection(app,tmp_path,failure):
    accepted(app,'inspect_scATAC'); cid=offer(app)
    before=app.sessions.load('session')
    app,path,calls=planner_app(app,tmp_path,failure)
    result=ask(app,'select','Run option 1.',Selection(cid,'Run option 1.'),execution_inputs={'path':str(path)})
    assert result.status=='failed',result
    assert len(calls)==1
    state=app.sessions.load('session')
    assert state.active_revision_id==before.active_revision_id and state.generation==before.generation


def test_reference_survives_intervening_discussion(app,tmp_path):
    accepted(app,'inspect_scATAC'); cid=offer(app)
    result=ask(app,'discussion','What does this result show?',Dialogue(question('r0')))
    assert result.status=='answered',result
    app,path,calls=planner_app(app,tmp_path)
    result=ask(app,'select','Run the inspection option.',Selection(cid,'Run the inspection option.'),execution_inputs={'path':str(path)})
    assert result.status=='activated',result
    assert len(calls)==1


def test_strict_minimal_branch():
    payload=dict(turn_schema_version=1,decision=dict(kind='execute_candidate',candidate='id',evidence='Run it.'))
    assert isinstance(parse_decision(json.dumps(payload)),ExecuteCandidate)
    for key,value in [('operation','plan'),('base','current'),('target','inspect_scATAC')]:
        with pytest.raises(Exception): parse_decision(json.dumps(payload | {'decision':payload['decision'] | {key:value}}))


def test_original_ordinals_survive_single_option_rationale(app,tmp_path):
    accepted(app,'inspect_scATAC')
    cid=offer(app,tools=('inspect_scATAC','cluster_cells'))
    refs=app.sessions.load('session').interactions[0].guidance_candidates
    assert refs[0]['capability']=='cluster_cells'  # canonical ordering, not model order
    cid=refs[1]['candidate_id']
    rationale=ask(app,'why','Why would the second option help?',Guidance(decision(candidate='second option')))
    assert rationale.status=='answered',rationale
    app,path,calls=planner_app(app,tmp_path)
    model=Selection(cid,'Run option 2.')
    result=ask(app,'select','Run option 2.',model,execution_inputs={'path':str(path)})
    assert result.status=='activated',result
    options=model.calls[0]['dialogue']['execution_candidates']
    assert [(x['option'],x['candidate']) for x in options]==[(n+1,r['candidate_id']) for n,r in enumerate(refs)]


def test_wrong_session_candidate_is_not_offered(app,tmp_path,monkeypatch):
    accepted(app,'inspect_scATAC'); cid=offer(app)
    from agent.application import OutputSelection
    from agent.schemas import AgentRequest
    app,path,_=planner_app(app,tmp_path)
    app.sessions.create('foreign')
    state=app.sessions.run('foreign','initial',AgentRequest('foreign-request','Inspect supplied input.',{'path':str(path)}),
        (OutputSelection('inspection','inspect','n_cells'),),expected_generation=0)
    assert state.turn('initial').status=='activated'
    offered=app.sessions.respond('foreign','guide','What should I analyze next?',interpreter=Guidance(decision()))
    assert offered.status=='answered'
    assert cid!=offered.guidance.candidates[0].reference['candidate_id']
    result=forbid_work(app,monkeypatch).sessions.respond('foreign','select','Run option 1.',interpreter=Selection(cid,'Run option 1.'))
    assert result.status=='clarification'


def test_explicit_stale_predecessor_cannot_rebase(app,monkeypatch):
    accepted(app,'inspect_scATAC'); cid=offer(app)
    accepted(app,'inspect_scATAC',name='newer')
    model=Selection(cid,'Run option 1.')
    result=ask(forbid_work(app,monkeypatch),'select','Run option 1.',model,predecessor_turn_id='guide')
    assert result.status=='clarification'
    assert result.clarification.reason=='ambiguous_revision'
    assert model.calls[0]['dialogue']['execution_candidates'][0]['current'] is False


def test_missing_subject_fails_closed(app,monkeypatch):
    from agent.application import guidance_selection
    from test_scientific_dialogue import annotation
    annotation(app)
    offered=ask(app,utterance='What could help understand cluster 3?',model=Guidance(decision('r0',subject='3')))
    cid=offered.guidance.candidates[0].reference['candidate_id']
    monkeypatch.setattr(guidance_selection.dialogue,'_subjects',lambda view: ({},True))
    result=ask(forbid_work(app,monkeypatch),'select','Run option 1.',Selection(cid,'Run option 1.'))
    assert result.status=='clarification' and result.clarification.reason=='ambiguous_subject'


@pytest.mark.parametrize('utterance',['Do the other one.','Run that analysis.',
    'Run option 1 for a different subject instead.'])
def test_semantic_ambiguity_and_conflicting_refinement_clarify(app,monkeypatch,utterance):
    accepted(app,'inspect_scATAC'); offer(app,tools=('inspect_scATAC','cluster_cells'))
    model=Dialogue(dict(kind='clarify',reason='ambiguous_subject'))
    result=ask(forbid_work(app,monkeypatch),'select',utterance,model)
    assert result.status=='clarification'
    assert len(model.calls)==1


def test_ordinary_execution_cannot_bypass_selection_after_discussion(app,monkeypatch):
    accepted(app,'inspect_scATAC'); offer(app)
    assert ask(app,'discussion','What does this show?',Dialogue(question('r0'))).status=='answered'
    result=ask(forbid_work(app,monkeypatch),'select','Run option 1.',Dialogue(dict(kind='execute_plan',target='inspect_scATAC')))
    assert result.status=='clarification'


def test_durable_selection_rejects_identity_tampering(app,tmp_path):
    from agent.application.session_state import AnalysisSession
    accepted(app,'inspect_scATAC'); cid=offer(app)
    app,path,_=planner_app(app,tmp_path)
    assert ask(app,'select','Run option 1.',Selection(cid,'Run option 1.'),execution_inputs={'path':str(path)}).status=='activated'
    state=app.sessions.load('session')
    payload=state.to_dict()
    payload['interactions'][-1]['admitted']['selected_candidate']['capability']='cluster_cells'
    with pytest.raises(ValueError): AnalysisSession.from_dict(payload)


def test_capability_removed_after_guidance(app,monkeypatch):
    accepted(app,'inspect_scATAC'); cid=offer(app)
    names=app.registry.names()
    monkeypatch.setattr(type(app.registry),'names',lambda self: tuple(n for n in names if n!='inspect_scATAC'))
    result=ask(forbid_work(app,monkeypatch),'select','Run option 1.',Selection(cid,'Run option 1.'))
    assert result.status=='clarification'


def test_state_change_during_output_selection_prevents_submission(app,tmp_path,monkeypatch):
    accepted(app,'inspect_scATAC'); cid=offer(app)
    app,path,calls=planner_app(app,tmp_path)
    class RacingModel(Selection):
        def complete(self,*,prompt,response_schema):
            if 'output_selection_schema_version' in json.loads(prompt):
                accepted(app,'inspect_scATAC',name='competing')
            return super().complete(prompt=prompt,response_schema=response_schema)
    from agent.orchestration.executor import PlanExecutor
    def forbidden(*args,**kwargs): raise AssertionError('Stale candidate entered execution')
    monkeypatch.setattr(PlanExecutor,'execute',forbidden)
    result=ask(app,'select','Run option 1.',RacingModel(cid,'Run option 1.'),execution_inputs={'path':str(path)})
    assert result.status=='failed',result
    assert len(calls)==1
    assert not any(t.turn_id=='select' for t in app.sessions.load('session').turns)


def test_competing_discussion_heads_cannot_choose_by_completion_order(app,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    accepted(app,'inspect_scATAC'); cid=offer(app)
    started,release=Event(),Event()
    with ThreadPoolExecutor() as pool:
        first=pool.submit(ask,app,'a','Explain this result.',Dialogue(question('r0'),started=started,release=release))
        assert started.wait(10)
        try:
            assert ask(app,'b','Explain this result.',Dialogue(question('r0'))).status=='answered'
        finally:
            release.set()
        assert first.result().status=='answered'
    model=Selection(cid,'Run that analysis.')
    result=ask(forbid_work(app,monkeypatch),'select','Run that analysis.',model)
    assert result.status=='clarification'
    assert model.calls[0]['dialogue']['execution_candidates']==[]
    assert model.calls[0]['dialogue']['ambiguous_predecessor'] is True
