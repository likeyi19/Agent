"""M15.5 persisted answers over real tiny accepted scientific runs."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import pytest
from agent.application import Answer, ResearchAgentApplication
from test_intent_turns import app_for, change, Interpreter, counts
from test_prior_outputs import source, application_chain, chain, all_work


from contextlib import contextmanager


@contextmanager
def response_work():
    """Compose the existing science profiler with distinct response/metadata counts."""
    from agent.application import responses, response_facts
    from agent.application.session_store import FileSessionStore
    from agent.orchestration.run_store import FileRunStore
    from agent.tools.data.barcode_qc_verifier import verify_barcode_qc
    functions = {
        'response.render': responses.render,
        'response.execution': responses.render_execution,
        'response.navigation': responses.render_navigation,
        'response.clarification': responses.clarification_text,
        'metadata.session_load': FileSessionStore.load,
        'metadata.run_load': FileRunStore.load,
        'integrity.evidence': response_facts._evidence_integrity,
        # All owned_verification decorators share this wrapper code; actual owner
        # bodies remain counted separately by all_work as owner.<kind>.
        'owner.verification_entry': verify_barcode_qc,
    }
    codes = {f.__code__: name for name, f in functions.items()}
    with all_work() as calls:
        previous = sys.getprofile()
        def profile(frame, event, arg):
            previous(frame, event, arg)
            if event == 'call' and frame.f_code in codes: calls[codes[frame.f_code]] += 1
        sys.setprofile(profile)
        try: yield calls
        finally: sys.setprofile(previous)


def ask(app, turn, utterance, intent, relation='current'):
    model = Interpreter(dict(kind='answer', intent=intent, relation=relation, technical=False))
    with response_work() as calls:
        outcome = app.sessions.respond('analysis', turn, utterance, interpreter=model)
    assert not counts(calls) and not calls['integrity'] and not calls['presentation'], calls
    assert calls['response.render'] == 1
    assert calls['metadata.run_load'] > 0 and calls['metadata.session_load'] > 0
    assert len(model.calls) == 1
    assert outcome.status == 'answered', outcome
    assert outcome.text and outcome.facts
    return outcome


def test_mini_conversation_restart_and_selection_only(source):
    app, planner = app_for(source, True)
    first = app.sessions.respond('analysis', 'six', 'Set TSS to 6 and rebuild the matrix',
        interpreter=change('Set TSS to 6', literal='6', target='matrix'))
    assert first.status == 'activated' and first.facts, first
    r1 = app.sessions.load('analysis').active_revision_id
    with response_work() as calls:
        result = app.sessions.respond('analysis', 'seven', 'Set TSS to 7 and rebuild the matrix',
            interpreter=change('Set TSS to 7', target='matrix'))
    assert {k:v for k,v in counts(calls).items() if k != 'owner.verification_entry'} == {'production.selection':1, 'owner.selection':1, 'production.matrix':1, 'owner.matrix':1}
    assert calls['response.execution'] == 1 and calls['response.render'] == 0
    assert result.facts and 'Minimum TSS enrichment is 7' in result.text, result
    assert 'fragments' in result.text and 'QC' in result.text
    print('M155 execution accounting', calls)
    preserved = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in app.workspace_root.rglob('*')
                 if p.is_file() and 'sessions' not in p.relative_to(app.workspace_root).parts}
    previous_calls = len(planner.calls)
    for turn, question, intent in [
        ('version', 'What version are we using?', 'version'),
        ('matrix', 'Which matrix are we using now?', 'matrix'),
        ('selection', 'What selection is active?', 'selection'),
        ('thresholds', 'What TSS threshold are we using?', 'thresholds'),
        ('reran', 'What did you rerun?', 'execution'),
        ('reuse', 'What did you reuse?', 'reuse'),
        ('why', "Why didn't QC rerun?", 'reuse'),
        ('provenance', 'Where did this matrix come from?', 'provenance'),
        ('verification', 'Why could you reuse the QC?', 'verification'),
        ('changes', 'What changed?', 'changes'),
    ]:
        outcome = ask(app, turn, question, intent)
        assert dict(outcome.facts.parameters)['min_tss_enrichment'] == 7
        assert {r.output.role for r in outcome.facts.reuse} == {'QC', 'fragments'}
        assert {w.role for w in outcome.facts.work if w.tool_attempts} == {'cell selection', 'matrix'}
        assert outcome.facts.owner_reconstruction_calls is None
        assert all(o.run_id not in outcome.text for o in outcome.facts.outputs)
        if intent == 'verification': assert outcome.facts.evidence_files_validated > 0
    compare = ask(app, 'compare', 'What changed from the previous version?', 'comparison', 'previous')
    assert ('min_tss_enrichment', 6, 7) in compare.facts.parameter_changes
    assert dict(compare.facts.output_changes) == {'QC':'unchanged', 'fragments':'unchanged', 'cell selection':'changed', 'matrix':'changed'}
    assert len(planner.calls) == previous_calls
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == sha for p, sha in preserved.items())
    # The same accepted answer is reconstructed after restart, not taken from prose.
    script = '''
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'tests/authority_dag'))
from test_intent_turns import all_work,Interpreter
from agent.application import ResearchAgentApplication,Answer
app=ResearchAgentApplication(sys.argv[1])
class Forbidden:
 def complete(self,**kwargs): raise AssertionError('replayed interpretation')
with all_work() as calls:
 for intent in ('version','matrix','thresholds','changes','execution','reuse','provenance','verification'):
  result=app.sessions.answer('analysis',Answer(intent))
  assert result.status=='answered',result
  assert dict(result.facts.parameters)['min_tss_enrichment']==7
 result=app.sessions.respond('analysis','compare','What changed from the previous version?',interpreter=Forbidden())
 assert ('min_tss_enrichment',6,7) in result.facts.parameter_changes
 nav=app.sessions.respond('analysis','back','Use the previous version',interpreter=Interpreter(dict(kind='navigate',relation='previous')))
 assert nav.status=='activated' and 'no scientific computation' in nav.text
 result=app.sessions.answer('analysis',Answer('thresholds'))
 assert dict(result.facts.parameters)['min_tss_enrichment']==6
assert not calls,calls
'''
    child = subprocess.run([sys.executable, '-c', script, str(app.workspace_root)],
        capture_output=True, text=True, env=os.environ.copy(), timeout=90)
    assert child.returncode == 0, child.stdout + child.stderr
    assert app.sessions.load('analysis').active_revision_id == r1
    selection_app, _ = app_for(source)
    with response_work() as selection_calls:
        outcome = selection_app.sessions.respond('analysis', 'selection-only', 'Set TSS to 8',
            interpreter=change('Set TSS to 8', literal='8'))
    assert {k:v for k,v in counts(selection_calls).items() if k != 'owner.verification_entry'} == {'production.selection':1,'owner.selection':1}
    assert selection_calls['response.execution'] == 1
    print('M155 selection-only',dict(selection_calls))
    assert outcome.facts and 'no matrix' in outcome.text, outcome
    matrix = ask(app, 'absent', 'Which matrix is active?', 'matrix')
    assert 'no matrix' in matrix.text and not any(o.role == 'matrix' for o in matrix.facts.outputs)
    # Visit the sibling and return so previous-active differs from the parent.
    state = app.sessions.load('analysis')
    r3 = state.active_revision_id
    sibling = state.revisions[2].revision_id
    app.sessions.switch('analysis', 'visit-sibling', sibling, expected_generation=state.generation)
    app.sessions.switch('analysis', 'return-branch', r3, expected_generation=state.generation + 1)
    with all_work() as calls:
        ambiguous = app.sessions.answer('analysis', Answer('comparison', 'previous'))
        parent = app.sessions.answer('analysis', Answer('comparison', 'parent'))
        previous = app.sessions.answer('analysis', Answer('comparison', 'previous_active'))
    assert not calls
    assert ambiguous.kind == 'clarify' and 'parent' in ambiguous.text
    assert parent.facts.comparison_revision_id != previous.facts.comparison_revision_id


def test_answer_tamper_unsupported_and_no_prose_authority(source, monkeypatch):
    app = source[2]
    baseline = app.sessions.load('analysis')
    with all_work() as calls:
        result = app.sessions.answer('analysis', Answer('unsupported'))
        assert result.status == 'unsupported' and 'assigned label' not in result.text
        result = app.sessions.respond('analysis','unsupported','Why is this cluster T cell?',
            interpreter=Interpreter(dict(kind='answer',intent='unsupported',relation='current',technical=False)))
        assert result.status == 'unsupported'
    assert not calls
    assert app.sessions.load('analysis').generation == baseline.generation
    assert 'text' not in app.sessions.load('analysis').to_dict()['interactions'][-1]
    # An explicit upstream-only output set must not discover historical selection.
    from agent.application import OutputSelection
    from agent.schemas import AgentRequest
    app.sessions.create('upstream-only')
    app.sessions.start_turn('upstream-only','initial',AgentRequest('authority-request','Tiny DAG.',{}),
        (OutputSelection('fragments','fragments','manifest_path'), OutputSelection('qc','qc','manifest_path')),
        expected_generation=0)
    app.sessions.link_run('upstream-only','initial')
    app.sessions.complete_presentation('upstream-only','initial')
    with all_work() as calls:
        absent = app.sessions.answer('upstream-only',Answer('thresholds'))
    assert absent.status == 'answered' and not absent.facts.parameters and 'no active selection thresholds' in absent.text
    assert not calls
    # Required evidence corruption fails closed, without an owner or recovery call.
    turn = baseline.turn('initial')
    evidence_root = app._workspace.run_paths(baseline.revisions[0].run_id).evidence
    item = next(f for f in turn.completion_files if Path(f.path).is_relative_to(evidence_root))
    path = Path(item.path); original = path.read_bytes()
    try:
        path.write_bytes(original + b' ')
        with all_work() as calls:
            assert app.sessions.answer('analysis', Answer('verification')).status == 'unavailable'
        assert not calls
    finally: path.write_bytes(original)
    # Exact result and locator anchors must be checked even on informational paths.
    from dataclasses import replace
    from agent.application.session_state import SessionError
    original_load = app.run_store.load
    for mode in ('missing', 'result'):
        def altered(run_id):
            if mode == 'missing': raise SessionError('missing referenced run')
            run = original_load(run_id)
            step = run.steps[-1]
            return replace(run, steps=run.steps[:-1] + (replace(step, duration_seconds=step.duration_seconds + 1),))
        with monkeypatch.context() as patch:
            patch.setattr(app.run_store, 'load', altered)
            with all_work() as calls:
                assert app.sessions.answer('analysis', Answer('version')).status == 'unavailable'
            assert not calls
    # Session checksum corruption is rejected, with no fallback to previous text.
    session_path = app.sessions._store._path('analysis', '.json')
    original = session_path.read_bytes()
    try:
        session_path.write_bytes(b'{}')
        with all_work() as calls:
            assert app.sessions.answer('analysis', Answer('matrix')).status == 'unavailable'
        assert not calls
    finally: session_path.write_bytes(original)
    # Rehashed metadata still cannot substitute an invalid exact step binding.
    from agent.application.session_state import digest
    record = baseline.to_dict()
    record['revisions'][0]['outputs'][0]['accepted_step_sha256'] = 'a' * 64
    forged = dict(format='agent.analysis-session.v1', record=record, sha256=digest(record))
    try:
        session_path.write_text(json.dumps(forged))
        with all_work() as calls:
            assert app.sessions.answer('analysis',Answer('matrix')).status == 'unavailable'
        assert not calls
    finally: session_path.write_bytes(original)


def test_metadata_only_accounting_and_frozen_answer(source):
    app = source[2]
    scenarios = (
        ('clarification', 'Make TSS stricter', dict(kind='clarify',reason='missing_parameter_value'), 'response.clarification'),
        ('state-answer', 'Which version is active?', dict(kind='answer',intent='version',relation='current',technical=False), 'response.render'),
        ('provenance-answer', 'Where did this matrix come from?', dict(kind='answer',intent='provenance',relation='current',technical=False), 'response.render'),
    )
    for turn, question, decision, renderer in scenarios:
        model = Interpreter(decision)
        with response_work() as calls:
            result = app.sessions.respond('analysis',turn,question,interpreter=model)
        assert not counts(calls) and not calls['integrity'] and not calls['presentation']
        assert calls[renderer] == 1 and len(model.calls) == 1
        if turn == 'clarification':
            assert 'minimum tss enrichment' in result.text and 'Which threshold' not in result.text
        print('M155',turn,dict(calls))
    with response_work() as calls:
        explicit = app.sessions.answer('analysis',Answer('thresholds'))
    assert calls['response.render'] == 1 and not counts(calls) and not calls['integrity']
    assert dict(explicit.facts.parameters)['min_tss_enrichment'] == '0'
    print('M155 structured answer',dict(calls))
    # A repeated answer uses the captured revision, never newly supplied prose.
    class Forbidden:
        def complete(self,**kwargs): raise AssertionError('Interpretation replayed')
    with response_work() as calls:
        repeated = app.sessions.respond('analysis','state-answer','Which version is active?',interpreter=Forbidden())
    assert repeated.facts.revision_id == explicit.facts.revision_id
    assert not counts(calls) and calls['response.render'] == 1


def test_matrix_only_summary_and_navigation_accounting(source):
    app, planner = app_for(source,True)
    original = planner.complete
    def matrix_only(*,prompt,response_schema):
        value = json.loads(prompt)
        if 'selection_schema_version' in value: return original(prompt=prompt,response_schema=response_schema)
        planner.calls.append((value,response_schema,prompt))
        items = value['active_outputs']
        fragments = next(i['handle'] for i in items if i['source_port']=='fragments')
        selection = next(i['handle'] for i in items if i['source_port']=='selected_cells')
        return json.dumps({'schema_version':4,'decision':{'kind':'plan','steps':[{
            'step_id':'rebuilt','tool':'build_scATAC_cell_by_ccre','sources':[
                {'target':'fragments','source':{'kind':'context','handle':fragments}},
                {'target':'selected_cells','source':{'kind':'context','handle':selection}}],
            'control_dependencies':[]}]}})
    planner.complete = matrix_only
    with response_work() as calls:
        outcome = app.sessions.respond('analysis','matrix-only','Keep this selection and rebuild the matrix',
            interpreter=Interpreter(dict(kind='execute',base='current',operation='op.0',target='matrix',delta=None)))
    assert {k:v for k,v in counts(calls).items() if k != 'owner.verification_entry'} == {'production.matrix':1,'owner.matrix':1}
    assert len(planner.calls) == 2 and calls['response.execution'] == 1
    assert outcome.facts and 'tool execution for matrix.' in outcome.text
    assert {r.output.role for r in outcome.facts.reuse} == {'fragments','cell selection'}
    assert {o.role for o in outcome.facts.retained_outputs} == {'fragments','cell selection','QC'}
    print('M155 matrix-only',dict(calls))
    with response_work() as calls:
        comparison = app.sessions.respond('analysis','compare','How is this version different from the parent?',
            interpreter=Interpreter(dict(kind='answer',intent='comparison',relation='parent',technical=False)))
    assert not counts(calls) and not calls['integrity'] and calls['response.render'] == 1
    assert dict(comparison.facts.output_changes)['matrix'] == 'changed'
    print('M155 comparison',dict(calls))
    with response_work() as calls:
        nav = app.sessions.respond('analysis','back','Use the previous version',
            interpreter=Interpreter(dict(kind='navigate',relation='previous')))
    assert not counts(calls) and not calls['integrity'] and not calls['presentation']
    assert calls['response.navigation'] == 1 and 'no scientific computation' in nav.text
    print('M155 navigation',dict(calls))

    class Forbidden:
        def complete(self,**kwargs): raise AssertionError('Answer interpretation replayed')
    with response_work() as calls:
        old = app.sessions.respond('analysis','compare','How is this version different from the parent?',interpreter=Forbidden())
    assert not old.facts.is_active and old.text.startswith('For historical version')
    assert not counts(calls) and calls['response.render'] == 1
