"""M16.2 real interfaces with deterministic providers and accepted metadata fixtures."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import subprocess
import sys
from threading import Event

import pytest

from agent.application import ResearchAgentApplication, SessionConflictError
from agent.application import scientific_dialogue as sd
from agent.application.turn_decisions import parse_decision, IntentError
from test_dialogue_evidence import app, accepted, forbid_work


def question(output='r0', subject=None, comparison=None, focus='question'):
    return dict(kind='answer', intent='scientific', target=dict(output=output, subject=subject),
                comparison=comparison, focus=focus)


class Model:
    def __init__(self, decision, fields=(), *, support='supported', response=None, started=None, release=None):
        self.decision, self.fields, self.support, self.response = decision, fields, support, response
        self.started, self.release, self.calls = started, release, []

    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        self.calls.append(value)
        if 'turn_schema_version' in value:
            choice = self.decision(value) if callable(self.decision) else self.decision
            return json.dumps(dict(turn_schema_version=1, decision=choice))
        if 'output_selection_schema_version' in value:
            return json.dumps(dict(outputs=[dict(name='inspection',step_id='inspect',output_key='n_cells')]))
        assert value['dialogue_schema_version'] == 2
        if self.started: self.started.set()
        if self.release: assert self.release.wait(10)
        if self.response is not None:
            result = self.response(value) if callable(self.response) else self.response
            return json.dumps(result)
        ids = [c['claim_id'] for c in value['evidence']['claims'] if not self.fields or c['field'] in self.fields]
        ids = ids[:6]
        meanings = list(value['evidence']['meanings'])
        parts = [dict(kind='text', text='The accepted evidence records:')]
        parts += [dict(kind='claim', id=i) for i in ids]
        parts += [dict(kind='meaning', id=m) for m in meanings]
        return json.dumps(dict(support=self.support, paragraphs=[dict(parts=parts)]))


def annotation(app, *, omitted=0):
    return accepted(app, 'annotate_scATAC_cell_types', overrides=dict(n_groups=2+omitted,
        groups_omitted=omitted, validation_state='not_assessed', group_summary=[
            dict(group='3', n_cells=8, primary_annotation='CD8 T', status='assigned'),
            dict(group='7', n_cells=5, primary_annotation='B cell', status='assigned')]))[0]


def ask(app, turn, text, model, **kwargs):
    return app.sessions.respond('session', turn, text, interpreter=model, **kwargs)


def test_annotation_multiturn_subject_pair_provenance_and_limitation(app, monkeypatch):
    rid = annotation(app)
    app = forbid_work(app, monkeypatch)
    first = ask(app, 'a', 'What does cluster 3 show?', Model(question(subject='3'), ['primary_annotation']))
    assert first.status == 'answered', first
    assert 'group 3' in first.text and 'CD8 T' in first.text
    why = ask(app, 'b', 'Why?', Model(question('@focus'), ['primary_annotation'], support='insufficient_evidence'))
    assert why.status == 'answered' and why.scientific.support == 'insufficient_evidence'
    assert 'marker rationale' in why.text
    other = ask(app, 'c', 'What about cluster 7?', Model(question('@focus','7',focus='continue'), ['primary_annotation']))
    assert other.status == 'answered' and 'B cell' in other.text and 'group 7' in other.text
    pair = ask(app, 'd', 'How are the two different?', Model(question('@focus', comparison=dict(output='@previous',subject=None)), ['primary_annotation','n_cells']))
    assert pair.status == 'answered', pair
    assert {c.subject for c in pair.scientific.claims} == {'3','7'}
    provenance = ask(app, 'e', 'Where did that value come from?', Model(question('@focus'), ['primary_annotation']))
    assert provenance.scientific.claims[0].source['revision_id'] == rid
    assert provenance.scientific.claims[0].source['pointer'].endswith('/1/primary_annotation')
    certain = ask(app, 'f', 'Does this prove the biological identity?', Model(question('@focus'), ['primary_annotation'], support='insufficient_evidence'))
    assert certain.scientific.support == 'insufficient_evidence'
    state = app.sessions.load('session')
    assert state.generation == 1
    for interaction in state.interactions:
        assert 'claims' not in interaction.admitted and 'text' not in interaction.admitted
        assert 'value' not in interaction.admitted and interaction.admitted['target']['revision_id'] == rid
    assert state.interactions[2].admitted['focus'] == 'Why?'
    assert state.interactions[-1].admitted['predecessor'] == 'e'


@pytest.mark.parametrize('tool,values,fields', [
    ('compute_scATAC_qc', {'n_observed_barcodes':13,'tss_undefined':3}, ['n_observed_barcodes','tss_undefined']),
    ('build_scATAC_cell_by_features', {'n_cells':8,'n_features':15}, ['n_cells','n_features']),
    ('adapt_epizoo_species', {'strategy':'de_novo','purpose':'qualification','completed_step':10}, ['strategy','completed_step']),
])
def test_same_dialogue_across_capabilities(app, monkeypatch, tool, values, fields):
    accepted(app, tool, overrides=values)
    app = forbid_work(app, monkeypatch)
    model = Model(question(), fields)
    result = ask(app, 'a', 'What does this result show?', model)
    assert result.status == 'answered', result
    assert {c.field:c.value for c in result.scientific.claims} == {k:values[k] for k in fields}
    prompt = model.calls[-1]
    assert 'source' not in prompt['evidence']['claims'][0]
    assert str(app.workspace_root) not in json.dumps(prompt)
    follow = ask(app, 'b', 'Why?', Model(question('@focus'), fields, support='insufficient_evidence'))
    assert follow.status == 'answered' and follow.scientific.support == 'insufficient_evidence'


def test_selection_cross_revision_counts_and_thresholds(app, monkeypatch):
    args = dict(min_qc_fragment_records=1, min_tss_enrichment='4')
    first = accepted(app, 'select_scATAC_cells', overrides={'n_selected':12,'qc_identity_sha256':'a'*64},
                     extra_facts={'effective_thresholds':args}, arguments=args)[0]
    args = dict(args, min_tss_enrichment='5')
    second = accepted(app, 'select_scATAC_cells', name='stricter', overrides={'n_selected':8,'qc_identity_sha256':'a'*64},
                      extra_facts={'effective_thresholds':args}, arguments=args)[0]
    app = forbid_work(app, monkeypatch)
    model = Model(question('r0', comparison=dict(output='r1',subject=None)), ['n_selected','effective_thresholds.min_tss_enrichment'])
    response = ask(app, 'compare', 'What changed scientifically from the parent?', model)
    assert response.status == 'answered', response
    assert {c.source['revision_id'] for c in response.scientific.claims} == {first,second}
    assert {c.value for c in response.scientific.claims if c.field == 'n_selected'} == {8,12}
    follow = ask(app, 'why', 'Why did the selected count decrease?',
        Model(question('@focus', comparison=dict(output='@previous',subject=None)), ['n_selected'], support='insufficient_evidence'))
    assert follow.status == 'answered'
    assert 'version 1' in follow.text and 'version 2' in follow.text


def test_ambiguity_truncation_unavailable_and_incompatible(app, monkeypatch):
    rid = annotation(app, omitted=5)
    guarded = forbid_work(app, monkeypatch)
    assert ask(guarded, 'a', 'Explain cluster 3', Model(question(subject='3'), ['primary_annotation'])).status == 'answered'
    ambiguous = ask(guarded, 'other', 'What about the other cluster?', Model(question('@focus','@other')))
    assert ambiguous.clarification.reason == 'ambiguous_subject'
    missing = ask(guarded, 'absent', 'What about cluster 99?', Model(question('@focus','99')))
    assert missing.status == 'clarification' and missing.clarification.reason == 'ambiguous_subject'
    # An unquoted subject substitution is rejected before generation.
    forged = ask(guarded, 'forged', 'Why?', Model(question('@focus','7')))
    assert forged.clarification.reason == 'ambiguous_subject'
    locator = guarded.sessions.load('session').revisions[0].outputs[0]
    path = guarded._workspace.runs / guarded._workspace.run_digest(locator.run_id) / 'evidence' / 'analysis_evidence.json'
    path.write_bytes(path.read_bytes()+b' ')
    corrupt = ask(guarded, 'corrupt', 'Explain the result', Model(question()))
    assert corrupt.status == 'unavailable'


def test_no_implicit_cross_revision_cluster_correspondence(app, monkeypatch):
    first = annotation(app)
    accepted(app, 'annotate_scATAC_cell_types', name='another', overrides=dict(groups_omitted=0,
        group_summary=[dict(group='3',n_cells=8,primary_annotation='other',status='assigned')]))
    guarded = forbid_work(app, monkeypatch)
    result = ask(guarded, 'compare', 'Compare cluster 3 with cluster 3 in the parent.',
        Model(question('r0','3', comparison=dict(output='r1',subject='3'))))
    assert result.status == 'unavailable' and 'correspondence' in result.text


@pytest.mark.parametrize('attack', ['number','number_word','category','new_value','unknown_claim','wrong_subject','legacy_slot'])
def test_provider_cannot_replace_claim_values_or_subjects(app, monkeypatch, attack):
    annotation(app)
    def forged(value):
        c = next(c['claim_id'] for c in value['evidence']['claims'] if c['field']=='primary_annotation')
        part = dict(kind='claim',id=c)
        item = dict(parts=[part])
        if attack == 'number': item['parts'].append(dict(kind='text',text='contains 999 cells'))
        if attack == 'number_word': item['parts'].append(dict(kind='text',text='contains ninety nine cells'))
        if attack == 'category': item['parts'].append(dict(kind='text',text='is CD8 T'))
        if attack == 'new_value': part['value'] = 'invented label'
        if attack == 'unknown_claim': part['id'] = 'c999'
        if attack == 'wrong_subject': part['subject'] = '7'
        if attack == 'legacy_slot': item['parts'] = [dict(kind='text',text='{{'+c+'}}')]
        return dict(support='supported', paragraphs=[item])
    response = ask(forbid_work(app,monkeypatch), 'attack', 'Explain cluster 3', Model(question(subject='3'), response=forged))
    assert response.status == 'unavailable' and response.scientific is None


def test_reordered_completions_require_explicit_predecessor(app, monkeypatch):
    annotation(app)
    app = forbid_work(app,monkeypatch)
    started, release = Event(), Event()
    with ThreadPoolExecutor() as pool:
        a = pool.submit(ask, app, 'a', 'Explain cluster 3', Model(question(subject='3'), ['primary_annotation'], started=started, release=release))
        assert started.wait(10)
        b = ask(app, 'b', 'Explain cluster 7', Model(question(subject='7'), ['primary_annotation']))
        assert b.status == 'answered'
        release.set()
        assert a.result().status == 'answered'
    unclear = ask(app, 'ambiguous', 'Why?', Model(question('@focus')))
    assert unclear.clarification.reason == 'ambiguous_predecessor'
    follow = ask(app, 'continue-a', 'Why?', Model(question('@focus'), ['primary_annotation']), predecessor_turn_id='a')
    assert follow.status == 'answered' and follow.scientific.claims[0].subject == '3'
    state = app.sessions.load('session')
    assert state.interactions[-1].admitted['predecessor'] == 'a'
    with pytest.raises(SessionConflictError):
        ask(app, 'continue-a', 'Why?', Model(question('@focus')), predecessor_turn_id='b')


def test_fresh_process_followup_without_prose(app):
    annotation(app)
    assert ask(app, 'a', 'Explain cluster 3', Model(question(subject='3'), ['primary_annotation'])).status == 'answered'
    code = '''
import json, sys
from agent.application import ResearchAgentApplication
from agent.orchestration.runtime import AgentRuntime
from agent.tools.data.authority_context import VerificationContext
def forbidden(*a, **kw): raise AssertionError('science')
AgentRuntime.run = AgentRuntime.resume = VerificationContext.verify = forbidden
class Model:
 def complete(self, *, prompt, response_schema):
  p=json.loads(prompt)
  if 'turn_schema_version' in p:
   return json.dumps(dict(turn_schema_version=1, decision=dict(kind='answer',intent='scientific',target=dict(output='@focus',subject=None),comparison=None,focus='question')))
  c=next(c['claim_id'] for c in p['evidence']['claims'] if c['field']=='primary_annotation')
  return json.dumps(dict(support='insufficient_evidence',paragraphs=[dict(parts=[dict(kind='text',text='The recorded assignment follows.'),dict(kind='claim',id=c)])]))
app=ResearchAgentApplication(sys.argv[1])
result=app.sessions.respond('session','b','Why?',interpreter=Model())
assert result.status=='answered',result
print(json.dumps(result.scientific.to_dict()))
'''
    run = subprocess.run([sys.executable,'-B','-c',code,str(app.workspace_root)],capture_output=True,text=True,check=True)
    result = json.loads(run.stdout)
    assert result['claims'][0]['subject'] == '3'
    state = app.sessions.load('session')
    assert state.interactions[-1].admitted['predecessor'] == 'a'
    assert all('explanation' not in i.admitted for i in state.interactions)


@pytest.mark.parametrize('minimal', [False, True])
def test_new_science_routes_to_existing_planner_and_executor(app, tmp_path, minimal):
    accepted(app, 'inspect_scATAC')
    import anndata as ad
    from scipy import sparse
    from agent.schemas import AgentPlan, PlanStep
    path = tmp_path / 'input.h5ad'
    ad.AnnData(sparse.csr_matrix([[1,0],[0,2]])).write_h5ad(path)
    calls = []
    class Planner:
        def plan(self, request, registry):
            calls.append(request)
            return AgentPlan('inspect-plan',request.request_id,'Explicit inspection.',
                (PlanStep('inspect','inspect_scATAC',dict(path=request.inputs['path'])),))
    app = ResearchAgentApplication(app.workspace_root, planner=Planner())
    model = Model(dict(kind='execute_plan',target='inspect_scATAC') if minimal else
                  dict(kind='execute',base='current',operation='plan',target='inspect_scATAC',delta=None))
    result = ask(app, 'science', 'Inspect the supplied matrix.', model, execution_inputs={'path':str(path)})
    assert result.kind == 'execute' and result.status == 'activated', result
    assert len(calls) == 1 and calls[0].prompt == 'Inspect the supplied matrix.'
    assert all('dialogue_schema_version' not in p for p in model.calls)
    current = app.sessions.load('session')
    view = app.sessions.evidence('session',current.active_revision_id,'inspection')
    assert view.status == 'available'


def test_question_cannot_be_misrouted_as_new_execution(app, monkeypatch):
    annotation(app)
    app = forbid_work(app,monkeypatch)
    model = Model(dict(kind='execute',base='current',operation='plan',target='run_replicate_differential_accessibility',delta=None))
    result = ask(app, 'q', 'What does this result show?', model)
    assert result.kind == 'clarify'


def test_new_science_cannot_be_misrouted_as_result_interpretation(app, monkeypatch):
    annotation(app)
    model = Model(question())
    result = ask(forbid_work(app,monkeypatch),'science','Compute differential accessibility.',model)
    assert result.clarification.reason == 'requires_execution'
    assert len(model.calls) == 1


def test_historical_target_requires_explicit_reference(app, monkeypatch):
    accepted(app,'inspect_scATAC',overrides={'n_cells':2})
    accepted(app,'inspect_scATAC',name='next',overrides={'n_cells':4})
    app = forbid_work(app,monkeypatch)
    incorrect = ask(app,'bad','What does this result show?',Model(question('r1'),['n_cells']))
    assert incorrect.clarification.reason == 'ambiguous_revision'
    correct = ask(app,'old','Explain the parent result.',Model(question('r1'),['n_cells']))
    assert correct.status == 'answered' and 'historical' in correct.text
    follow = ask(app,'old-follow','Why?',Model(question('@focus'),['n_cells']))
    assert follow.status == 'answered' and follow.scientific.claims[0].value == 2


def test_incompatible_matrix_comparison_and_context_bound(app, monkeypatch):
    accepted(app,'build_scATAC_cell_by_features',overrides={'assembly':'first'})
    accepted(app,'build_scATAC_cell_by_features',name='next',overrides={'assembly':'second'})
    app = forbid_work(app,monkeypatch)
    compared = ask(app,'comparison','Compare with the parent.',Model(question('r0',comparison=dict(output='r1',subject=None))))
    assert compared.status == 'unavailable'
    monkeypatch.setattr(sd,'MAX_CONTEXT_BYTES',10)
    assert ask(app,'bounded','Explain this result',Model(question())).status == 'unavailable'


def test_changed_evidence_is_reloaded_on_followup(app, monkeypatch):
    _, path, _ = accepted(app,'inspect_scATAC')
    app = forbid_work(app,monkeypatch)
    assert ask(app,'a','Explain this result',Model(question(),['n_cells'])).status == 'answered'
    path.write_bytes(path.read_bytes()+b' ')
    model = Model(question('@focus'),['n_cells'])
    assert ask(app,'b','Why?',model).status == 'unavailable'
    assert len(model.calls) == 1


def test_retry_reuses_admitted_references_without_reinterpreting(app, monkeypatch):
    annotation(app)
    app = forbid_work(app,monkeypatch)
    first = ask(app,'a','Explain cluster 3',Model(question(subject='3'),['primary_annotation']))
    model = Model(None,['primary_annotation'])
    repeated = ask(app,'a','Explain cluster 3',model)
    assert repeated.status == 'answered' and repeated.scientific.claims == first.scientific.claims
    assert len(model.calls)==1 and 'dialogue_schema_version' in model.calls[0]


def test_comparison_generation_cannot_drop_an_operand(app, monkeypatch):
    accepted(app,'inspect_scATAC',overrides={'n_cells':2})
    accepted(app,'inspect_scATAC',name='next',overrides={'n_cells':4})
    def omit(value):
        c=next(c['claim_id'] for c in value['evidence']['claims'] if c['field']=='n_cells' and c['target']=='t0')
        return dict(support='supported',paragraphs=[dict(parts=[dict(kind='claim',id=c)])])
    model=Model(question('r0',comparison=dict(output='r1',subject=None)),response=omit)
    assert ask(forbid_work(app,monkeypatch),'bad','Compare with the parent.',model).status=='unavailable'


def test_reviewed_metric_meaning_requires_exact_profile(app, monkeypatch):
    from agent.tools.data.scatac_qc_profile import PROFILE, PROFILE_SHA256
    accepted(app,'compute_scATAC_qc',overrides={'science_profile_sha256':PROFILE_SHA256},
             extra_facts={'tss_method':PROFILE.tss_method})
    model=Model(question(),['tss_undefined'])
    result = ask(forbid_work(app,monkeypatch),'metric','What does TSS enrichment mean?',model)
    assert result.status == 'answered'
    assert 'endpoint incidence per base' in result.text and 'Zero background' in result.text
    from dataclasses import asdict
    assert model.calls[-1]['evidence']['targets'][0]['reviewed_profile']==json.loads(json.dumps(asdict(PROFILE)))


def test_evidence_changed_during_generation_is_rejected(app, monkeypatch):
    _, path, _ = accepted(app,'inspect_scATAC')
    def response(value):
        path.write_bytes(path.read_bytes()+b' ')
        c=next(c['claim_id'] for c in value['evidence']['claims'] if c['field']=='n_cells')
        return dict(support='supported',paragraphs=[dict(parts=[dict(kind='claim',id=c)])])
    result=ask(forbid_work(app,monkeypatch),'a','Explain this result.',Model(question(),response=response))
    assert result.status=='unavailable' and result.scientific is None


def test_navigation_during_generation_marks_captured_result_historical(app, monkeypatch):
    original=accepted(app,'inspect_scATAC',overrides={'n_cells':2})[0]
    newer=accepted(app,'inspect_scATAC',name='next',overrides={'n_cells':4})[0]
    app=forbid_work(app,monkeypatch)
    def response(value):
        app.sessions.switch('session','back',original,expected_generation=2)
        c=next(c['claim_id'] for c in value['evidence']['claims'] if c['field']=='n_cells')
        return dict(support='supported',paragraphs=[dict(parts=[dict(kind='claim',id=c)])])
    result=ask(app,'a','Explain this result.',Model(question(),response=response))
    assert result.status=='answered' and 'historical' in result.text
    assert result.scientific.claims[0].value==4
    assert result.scientific.claims[0].source['revision_id']==newer


def test_dialogue_capture_preserves_m15_three_base_output_bound(app, monkeypatch):
    first=accepted(app,'inspect_scATAC',copies=32)[0]
    second=accepted(app,'inspect_scATAC',name='second',copies=32)[0]
    accepted(app,'inspect_scATAC',name='third',copies=32)
    app.sessions.switch('session','back',second,expected_generation=3)
    model=Model(dict(kind='navigate',relation='parent'))
    result=ask(forbid_work(app,monkeypatch),'nav','Go back to the parent version',model)
    assert result.status=='activated',result
    assert len(model.calls[0]['dialogue']['outputs'])==96
    assert app.sessions.load('session').active_revision_id==first


def test_scientific_wire_is_closed():
    bad = question()
    bad['value'] = 123
    with pytest.raises(IntentError): parse_decision(json.dumps(dict(turn_schema_version=1,decision=bad)))
