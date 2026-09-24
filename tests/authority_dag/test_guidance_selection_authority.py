"""Candidate handoff reuses actual M15.3 authority and scoped-v4 compilation."""
import json

import pytest

from agent.application import ResearchAgentApplication
from agent.orchestration import LLMPlanner
from agent.schemas.prior_output import PriorOutputRef
from test_prior_outputs import source, application_chain, chain, all_work, ContextModel


class InteractionModel:
    def __init__(self, candidate=None):
        self.candidate = candidate

    def complete(self, *, prompt, response_schema):
        p=json.loads(prompt)
        if 'turn_schema_version' in p:
            if self.candidate:
                decision=dict(kind='execute_candidate',candidate=self.candidate,evidence='Run option 1.')
            else:
                qc=next(o for o in p['dialogue']['outputs'] if o['output_name']=='qc' and 'current' in o['relations'])
                decision=dict(kind='answer_guidance',targets=[dict(output=qc['handle'],subject=None)],candidate=None)
            return json.dumps(dict(turn_schema_version=1,decision=decision))
        if 'output_selection_schema_version' in p:
            return json.dumps(dict(outputs=[dict(name='selection',step_id='new_selection',output_key='manifest_path')]))
        assert 'guidance_schema_version' in p
        return json.dumps(dict(candidates=[dict(capability='select_scATAC_cells',explanation=dict(
            support='insufficient_evidence',paragraphs=[dict(parts=[dict(kind='text',text=t)]) for t in (
                'Selection could help address the stated objective.',
                'The selection criteria require explicit input.',
                'Further validation is required before execution.')]))]))


@pytest.mark.parametrize('failure',[None,'compiler','missing_input'])
def test_guidance_handoff_uses_real_authority_and_normal_compiler(source,failure):
    _,_,old=source
    with all_work() as work:
        offered=old.sessions.respond('analysis','guidance','What could I analyze next using the QC result?',interpreter=InteractionModel())
    assert offered.status=='answered',offered
    assert not work
    cid=offered.guidance.candidates[0].reference['candidate_id']
    planner=ContextModel(handle_override='ctx.999' if failure=='compiler' else None)
    app=ResearchAgentApplication(old.workspace_root,planner=LLMPlanner(planner))
    inputs={'min_qc_fragment_records':1,'min_tss_enrichment':'0'} if failure!='missing_input' else {}
    with all_work() as work:
        outcome=app.sessions.respond('analysis','select','Run option 1.',interpreter=InteractionModel(cid),execution_inputs=inputs)
    state=app.sessions.load('analysis')
    if failure:
        assert outcome.status=='failed',outcome
        assert state.generation==1
        assert not {k:v for k,v in work.items() if k.startswith(('production.','owner.'))}
    else:
        assert outcome.status=='activated',outcome
        run=app.run_store.load(state.turn('select').run_id)
        refs=[v for v in run.plan.steps[0].arguments.values() if isinstance(v,PriorOutputRef)]
        assert len(refs)==2 and refs[0].binding==refs[1].binding
        assert refs[0].binding.run_id=='authority-request:run'
        assert {k:v for k,v in work.items() if k.startswith(('production.','owner.'))}=={'production.selection':1,'owner.selection':1}
        assert {o.name for o in state.revisions[-1].outputs}=={'selection'}
    assert len(planner.calls)>=2
    for payload,_,prompt in planner.calls:
        assert str(app.workspace_root) not in prompt
        assert 'authority-request:run' not in prompt
    assert planner.calls[1][0]['active_outputs'][0]['handle']=='ctx.0'
    intent=json.loads(planner.calls[1][0]['user_request'])
    assert intent['captured_subjects']==[dict(context_handle='ctx.0',output='qc',subject=None)]
