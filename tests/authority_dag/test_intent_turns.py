"""M15.4 real tiny downstream execution and zero-science interaction paths."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import os

import pytest
from agent.application import ResearchAgentApplication
from agent.orchestration import LLMPlanner
from agent.schemas import PriorOutputRef
from test_prior_outputs import source, application_chain, chain, all_work, ContextModel


class Interpreter:
    def __init__(self, decision): self.decision, self.calls = decision, []
    def complete(self, *, prompt, response_schema):
        self.calls.append((json.loads(prompt),response_schema))
        return json.dumps({'turn_schema_version':1,'decision':self.decision})


def change(utterance, *, parameter='min_tss_enrichment', operation='set', literal='7', target='selection', base='current'):
    return Interpreter(dict(kind='execute',base=base,operation='op.0',target=target,
        delta=dict(parameter=parameter,operation=operation,literal=literal,evidence=utterance)))


def app_for(source, matrix=False):
    model=ContextModel(matrix)
    return ResearchAgentApplication(source[2].workspace_root,planner=LLMPlanner(model)), model


def counts(calls):
    return {k:v for k,v in calls.items() if k.startswith(('production.','owner.'))}


def test_set_matrix_navigation_and_historical_branch(source):
    app, planner = app_for(source,True)
    r1=app.sessions.load('analysis').revisions[0]
    utterance='Set the TSS threshold to 7 and rebuild the matrix'
    interpreter=change('Set the TSS threshold to 7',target='matrix')
    with all_work() as work:
        result=app.sessions.respond('analysis','change',utterance,interpreter=interpreter)
    assert result.status=='activated',result
    assert counts(work)=={'production.selection':1,'owner.selection':1,'production.matrix':1,'owner.matrix':1}
    assert work['integrity']>0 and work['presentation']==1
    assert len(interpreter.calls)==1 and len(planner.calls)==2
    state=app.sessions.load('analysis');r2=state.revisions[-1]
    assert {o.name for o in r2.outputs}=={'fragments','qc','selection','matrix'}
    run=app.run_store.load(r2.run_id)
    assert run.steps[0].resolved_arguments['min_tss_enrichment']==7
    old=app.run_store.load(r1.run_id).steps[2]
    assert run.steps[0].resolved_arguments['min_qc_fragment_records']==old.resolved_arguments['min_qc_fragment_records']
    before=run.to_dict()
    nav=Interpreter(dict(kind='navigate',relation='previous'))
    with all_work() as work:
        assert app.sessions.respond('analysis','back','use the previous version',interpreter=nav).status=='activated'
    assert not work and len(planner.calls)==2 and len(nav.calls)==1
    assert app.sessions.load('analysis').active_revision_id==r1.revision_id
    app2, planner2=app_for(source)
    with all_work() as work:
        assert app2.sessions.respond('analysis','branch','Set TSS to 6',interpreter=change('Set TSS to 6',literal='6')).status=='activated'
    assert counts(work)=={'production.selection':1,'owner.selection':1}
    state=app2.sessions.load('analysis');r3=state.revisions[-1]
    assert r3.parent_revision_id==r1.revision_id and r2 in state.revisions
    assert {o.name for o in r3.outputs}=={'fragments','qc','selection'}
    assert app.run_store.load(r2.run_id).to_dict()==before
    branch=app.run_store.load(r3.run_id)
    assert branch.plan.steps[0].arguments['barcode_qc_manifest_path'].binding.run_id==r1.run_id
    for payload,_ in interpreter.calls:
        text=json.dumps(payload)
        assert str(app.workspace_root) not in text and r1.run_id not in text and r1.revision_id not in text


def test_clarification_grounding_and_parameter_contract_before_planning(source):
    app,planner=app_for(source)
    cases=[('Make the TSS threshold stricter',Interpreter(dict(kind='clarify',reason='missing_parameter_value'))),
        ('Set TSS to 7',change('Set TSS to 7',literal='8')),
        ('Set foo_threshold to 7',change('Set foo_threshold to 7',parameter='foo_threshold')),
        ('Make the threshold 7',change('Make the threshold 7')),
        ('Decrease TSS by 1',change('Decrease TSS by 1',operation='subtract',literal='1'))]
    for n,(utterance,interpreter) in enumerate(cases):
        with all_work() as work:
            result=app.sessions.respond('analysis',f'clarify-{n}',utterance,interpreter=interpreter)
        assert result.kind=='clarify' and not work and not planner.calls
        if result.clarification.reason in {'missing_parameter_value','ambiguous_parameter'}:
            assert 'min_tss_enrichment' in result.clarification.choices
            assert 'foo_threshold' not in result.clarification.choices
        assert app.sessions.load('analysis').generation==1
    # An independently sufficient follow-up uses durable active state, without chat replay.
    result=app.sessions.respond('analysis','answer','Set TSS to 7',interpreter=change('Set TSS to 7'))
    assert result.status=='activated'
    assert len(planner.calls)==2


def test_increment_decrement_and_immutable_parameters(source):
    app,planner=app_for(source)
    assert app.sessions.respond('analysis','six','Set TSS to 6',interpreter=change('Set TSS to 6',literal='6')).status=='activated'
    for turn,utterance,op,expected in [('up','Increase the TSS threshold by 1','add',7),
                                      ('down','Decrease it by 1','subtract',6)]:
        with all_work() as work:
            result=app.sessions.respond('analysis',turn,utterance,interpreter=change(utterance,operation=op,literal='1'))
        assert result.status=='activated'
        assert counts(work)=={'production.selection':1,'owner.selection':1}
        state=app.sessions.load('analysis');run=app.run_store.load(state.revisions[-1].run_id)
        assert run.steps[0].resolved_arguments['min_tss_enrichment']==expected
        assert run.steps[0].resolved_arguments['min_qc_fragment_records']==0


def test_navigation_ambiguity_and_fresh_process_metadata(source,tmp_path):
    app,planner=app_for(source)
    r1=app.sessions.load('analysis').active_revision_id
    app.sessions.respond('analysis','two','Set TSS to 6',interpreter=change('Set TSS to 6',literal='6'))
    state=app.sessions.load('analysis');r2=state.active_revision_id
    app.sessions.switch('analysis','explicit-back',r1,expected_generation=2)
    app.sessions.respond('analysis','three','Set TSS to 7',interpreter=change('Set TSS to 7'))
    r3=app.sessions.load('analysis').active_revision_id
    app.sessions.switch('analysis','visit-two',r2,expected_generation=4)
    app.sessions.switch('analysis','visit-three',r3,expected_generation=5)
    # R3's parent is R1; previously active is R2. Provider cannot erase ambiguity.
    for n,relation in enumerate(('previous','parent')):
        with all_work() as work:
            result=app.sessions.respond('analysis',f'ambiguous-{n}','use the previous version',
                interpreter=Interpreter(dict(kind='navigate',relation=relation)))
        assert result.kind=='clarify' and result.clarification.reason=='ambiguous_revision' and not work
    script='''
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'tests/authority_dag'))
from test_intent_turns import Interpreter,all_work
from agent.application import ResearchAgentApplication
app=ResearchAgentApplication(sys.argv[1])
with all_work() as calls:
 result=app.sessions.respond('analysis','natural-before','use the version I was using before',
     interpreter=Interpreter(dict(kind='navigate',relation='previous_active')))
assert result.status=='activated' and not calls
assert app.sessions.load('analysis').active_revision_id==sys.argv[2]
'''
    child=subprocess.run([sys.executable,'-c',script,str(app.workspace_root),r2],capture_output=True,text=True,env=os.environ.copy(),timeout=90)
    assert child.returncode==0,child.stdout+child.stderr
    with all_work() as work:
        assert app.sessions.respond('analysis','parent','use parent version',
            interpreter=Interpreter(dict(kind='navigate',relation='parent'))).status=='activated'
    assert not work and app.sessions.load('analysis').active_revision_id==r1


def test_matrix_only_and_combined_parent_branch(source):
    app,planner=app_for(source,True)
    original=planner.complete
    def matrix_only(*,prompt,response_schema):
        value=json.loads(prompt)
        if 'selection_schema_version' in value: return original(prompt=prompt,response_schema=response_schema)
        planner.calls.append((value,response_schema,prompt))
        items=value['active_outputs']
        fragments=next(i['handle'] for i in items if i['source_port']=='fragments')
        selection=next(i['handle'] for i in items if i['source_port']=='selected_cells')
        return json.dumps({'schema_version':4,'decision':{'kind':'plan','steps':[{
            'step_id':'reconstructed','tool':'build_scATAC_cell_by_ccre','sources':[
                {'target':'fragments','source':{'kind':'context','handle':fragments}},
                {'target':'selected_cells','source':{'kind':'context','handle':selection}}],
            'control_dependencies':[]}]}})
    planner.complete=matrix_only
    with all_work() as work:
        result=app.sessions.respond('analysis','matrix-only','Keep this selection and rebuild the matrix',
            interpreter=Interpreter(dict(kind='execute',base='current',operation='op.0',target='matrix',delta=None)))
    assert result.status=='activated'
    assert counts(work)=={'production.matrix':1,'owner.matrix':1}
    state=app.sessions.load('analysis');r1,r2=state.revisions
    assert next(o for o in r2.outputs if o.name=='selection').run_id==r1.run_id
    app2,_=app_for(source)
    with all_work() as work:
        result=app2.sessions.respond('analysis','combined','Go back to the earlier version and try min_tss = 7',
            interpreter=change('try min_tss = 7',base='previous'))
    assert result.status=='activated' and counts(work)=={'production.selection':1,'owner.selection':1}
    state=app2.sessions.load('analysis')
    assert state.revisions[-1].parent_revision_id==r1.revision_id and r2 in state.revisions


def test_retry_delta_frozen_and_stale_after_navigation(source):
    app,planner=app_for(source)
    app.sessions.respond('analysis','six','Set TSS to 6',interpreter=change('Set TSS to 6',literal='6'))
    state=app.sessions.load('analysis');r1,r2=state.revisions
    original=planner.complete
    offered=[]
    def race(*,prompt,response_schema):
        value=json.loads(prompt)
        if 'active_outputs' in value:
            offered.append(value['active_outputs'])
            if len(offered)==1:
                app.sessions.switch('analysis','racer',r1.revision_id,expected_generation=2)
                planner.handle_override='ctx.999'
            else: planner.handle_override=None
        return original(prompt=prompt,response_schema=response_schema)
    planner.complete=race
    interpreter=change('Increase TSS by 1',operation='add',literal='1')
    with all_work() as work:
        result=app.sessions.respond('analysis','stale','Increase TSS by 1',interpreter=interpreter)
    assert result.status=='stale' and len(interpreter.calls)==1
    assert counts(work)=={'production.selection':1,'owner.selection':1}
    state=app.sessions.load('analysis')
    assert state.active_revision_id==r1.revision_id and state.revisions[-1].parent_revision_id==r2.revision_id
    assert all(o==offered[0] for o in offered) and len(offered)>=2
    run=app.run_store.load(state.revisions[-1].run_id)
    assert run.steps[0].resolved_arguments['min_tss_enrichment']==7


def test_fresh_process_resume_does_not_reinterpret_delta(source,monkeypatch,tmp_path):
    app,planner=app_for(source,True)
    class Exit(BaseException):pass
    # All application instances share FileRunStore persistence, not Python objects.
    from agent.orchestration.run_store import FileRunStore
    update=FileRunStore.update
    def interrupt(store,state,*,expected_revision):
        saved=update(store,state,expected_revision=expected_revision)
        if saved.request.request_id.startswith('turn-') and any(s.step_id=='new_selection' and s.status.value=='SUCCEEDED' for s in saved.steps):
            raise Exit()
        return saved
    monkeypatch.setattr(FileRunStore,'update',interrupt)
    interpreter=change('Set TSS to 7',target='matrix')
    with pytest.raises(Exit):
        app.sessions.respond('analysis','crash','Set TSS to 7 and rebuild the matrix',interpreter=interpreter)
    state=app.sessions.load('analysis');turn=state.turn('crash')
    before=app.run_store.load(turn.run_id).plan
    script='''
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'tests/authority_dag'))
from test_intent_turns import all_work,counts
from agent.application import ResearchAgentApplication
app=ResearchAgentApplication(sys.argv[1])
with all_work() as work:
 result=app.resume(sys.argv[2])
assert result.status.value=='SUCCEEDED',result
assert counts(work)=={'production.matrix':1,'owner.matrix':1},work
assert app.sessions.complete_presentation('analysis','crash').turn('crash').status=='activated'
class Forbidden:
 def complete(self,**kw):raise AssertionError('Interpretation replayed')
assert app.sessions.respond('analysis','crash','Set TSS to 7 and rebuild the matrix',interpreter=Forbidden()).status=='activated'
'''
    child=subprocess.run([sys.executable,'-c',script,str(app.workspace_root),turn.run_id],capture_output=True,text=True,env=os.environ.copy(),timeout=180)
    assert child.returncode==0,child.stdout+child.stderr
    assert app.run_store.load(turn.run_id).plan==before and len(interpreter.calls)==1


def test_failures_and_idempotence_preserve_active_state(source,monkeypatch):
    app,planner=app_for(source)
    baseline=app.sessions.load('analysis')
    # Interpretation and planning failures never mutate active state.
    class Invalid:
        def complete(self,**kwargs): return '{bad json'
    with all_work() as work:
        result=app.sessions.respond('analysis','invalid','Set TSS to 7',interpreter=Invalid())
    assert result.kind=='clarify' and not work and not planner.calls
    from agent.tools.data import _cell_selection_production
    with monkeypatch.context() as patch:
        def fail(*args,**kwargs): raise ValueError('test production failure')
        patch.setattr(_cell_selection_production,'produce',fail)
        result=app.sessions.respond('analysis','failed-production','Set TSS to 7',interpreter=change('Set TSS to 7'))
        assert result.status=='failed'
    state=app.sessions.load('analysis')
    assert state.active_revision_id==baseline.active_revision_id and state.generation==baseline.generation
    class Forbidden:
        def complete(self,**kwargs): raise AssertionError('duplicate interpretation')
    with all_work() as work:
        assert app.sessions.respond('analysis','failed-production','Set TSS to 7',interpreter=Forbidden()).status=='failed'
    assert not work
    # Fail the scoped Planner without admitting an executable plan.
    class BrokenPlan:
        def plan(self,*args,**kwargs): raise ValueError('test planner failure')
    failed=ResearchAgentApplication(app.workspace_root,planner=BrokenPlan())
    result=failed.sessions.respond('analysis','failed-planner','Set TSS to 7',interpreter=change('Set TSS to 7'))
    assert result.status=='failed' and failed.sessions.load('analysis').generation==1


def test_verification_and_presentation_failures_preserve_revision(source,monkeypatch):
    app,planner=app_for(source)
    from agent.orchestration import executor
    verify=executor.verify_step
    def reject(step,*args,**kwargs):
        if step.tool_name=='select_scATAC_cells': raise ValueError('test verification failure')
        return verify(step,*args,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(executor,'verify_step',reject)
        result=app.sessions.respond('analysis','bad-verification','Set TSS to 7',interpreter=change('Set TSS to 7'))
    assert result.status=='failed' and app.sessions.load('analysis').generation==1
    from agent.application import ApplicationStatus,ApplicationError,ApplicationStage
    def failed(self,result,run):
        return self._base_result(result,run,status=ApplicationStatus.FAILED,
            error=ApplicationError('TEST','test presentation failure',ApplicationStage.REPORT))
    with monkeypatch.context() as patch:
        patch.setattr(ResearchAgentApplication,'_complete',failed)
        result=app.sessions.respond('analysis','bad-presentation','Set TSS to 7',interpreter=change('Set TSS to 7'))
    assert result.status=='run_succeeded' and app.sessions.load('analysis').generation==1
    with all_work() as work:
        result=app.sessions.complete_presentation('analysis','bad-presentation')
    assert result.turn('bad-presentation').status=='activated' and not counts(work)
