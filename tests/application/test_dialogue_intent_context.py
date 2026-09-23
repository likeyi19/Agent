"""Scripted semantic choices test visibility/admission, not live LLM accuracy."""
import json

import pytest

from agent.application import scientific_dialogue as sd
from agent.application import turns
from test_dialogue_evidence import app, accepted
from test_scientific_dialogue import Model, question, ask, annotation


@pytest.mark.parametrize('tool,utterance,field,values,args', [
    ('compute_scATAC_qc','What does the qc result show about this dataset?', 'n_observed_barcodes',{'n_observed_barcodes':13},{}),
    ('select_scATAC_cells','What does the selection result show?', 'n_selected',{'n_selected':8},{'min_qc_fragment_records':1,'min_tss_enrichment':'4'}),
    ('build_scATAC_cell_by_features','What does this matrix result show?', 'n_cells',{'n_cells':8},{}),
    ('adapt_epizoo_species','What does this adaptation result establish?', 'completed_step',{'completed_step':10},{}),
])
def test_generic_available_result_visibility_and_scientific_admission(app, tool, utterance, field, values, args):
    accepted(app,tool,overrides=values,arguments=args)
    def decide(prompt):
        d=prompt['dialogue']; item,=d['outputs']
        assert item['is_active'] and item['evidence_status']=='available'
        assert field in item['available_fields']
        assert 'value' not in item and 'facts' not in item
        spec=app.registry.get(tool)
        assert d['semantics'][tool]['description']==spec.planning.description
        assert d['semantics'][tool]['result_contract']==spec.result_contract.name
        assert 'unavailable_context only' in ' '.join(prompt['instructions'])
        return dict(question(item['handle']),kind='answer_scientific')
    result=ask(app,'explain',utterance,Model(decide,[field]))
    assert result.status=='answered' and result.scientific is not None
    assert result.scientific.claims[0].value==values[field]


def test_why_and_provenance_keep_exact_qc_focus(app):
    rid=accepted(app,'compute_scATAC_qc',overrides={'n_observed_barcodes':13})[0]
    assert ask(app,'first','What does the qc result show?',Model(question(),['n_observed_barcodes'])).scientific
    for turn,text in [('why','Why?'),('source','Where did that value come from?')]:
        def decide(prompt):
            prior=prompt['dialogue']['predecessor']
            assert prior['target']==dict(output='r0',subject=None)
            assert prior['focus'] in ('What does the qc result show?','Why?')
            return dict(question('@focus'),kind='answer_scientific')
        result=ask(app,turn,text,Model(decide,['n_observed_barcodes']))
        assert result.scientific.claims[0].source['revision_id']==rid
    state=app.sessions.load('session')
    assert state.interactions[-1].admitted['predecessor']=='why'


def test_direct_handle_of_durable_focus_is_not_ambiguous(app):
    accepted(app,'compute_scATAC_qc',copies=2,overrides={'n_observed_barcodes':13})
    assert ask(app,'start','What does output0 show?',Model(question('r0'),['n_observed_barcodes'])).scientific
    # A live provider chose the exact focused handle instead of spelling @focus.
    result=ask(app,'why','Why?',Model(question('r0'),['n_observed_barcodes']))
    assert result.scientific is not None
    wrong=ask(app,'wrong','Why?',Model(question('r1'),['n_observed_barcodes']))
    assert wrong.clarification.reason=='ambiguous_subject'


def test_annotation_subject_inventory_and_continuation(app):
    annotation(app)
    def decide(prompt):
        subjects=prompt['dialogue']['outputs'][0]['subjects']
        assert [c['id'] for c in subjects['candidates']]==['3','7'] and subjects['complete']
        assert {c['id'] for c in subjects['candidates']}=={'3','7'}
        return dict(question(subject='3'),kind='answer_scientific')
    assert ask(app,'explain','What does cluster 3 show?',Model(decide,['primary_annotation'])).scientific
    result=ask(app,'other','What about cluster 7?',Model(question('@focus','7',focus='continue'),['primary_annotation']))
    assert result.scientific.claims[0].subject=='7'


def test_ambiguous_selection_is_not_guessed_even_if_provider_picks_first(app):
    accepted(app,'inspect_scATAC',copies=2)
    result=ask(app,'ambiguous','What does the result show?',Model(question()))
    assert result.status=='clarification' and result.clarification.reason=='ambiguous_subject'


def test_missing_evidence_is_visible_without_fabricating_availability(app):
    _,path,_=accepted(app,'inspect_scATAC')
    path.write_text('{}')
    def decide(prompt):
        assert prompt['dialogue']['outputs'][0]['evidence_status']=='unavailable'
        return dict(kind='clarify',reason='unavailable_context')
    assert ask(app,'missing','What does this result show?',Model(decide)).status=='clarification'


def test_operational_version_stays_operational(app):
    accepted(app,'inspect_scATAC')
    result=ask(app,'version','Which revision is currently active?',
        Model(dict(kind='answer',intent='version',relation='current',technical=False)))
    assert result.kind=='answer' and result.scientific is None
    assert app.sessions.load('session').interactions[-1].admitted['intent']=='version'


def test_explicit_threshold_stays_execute(app,monkeypatch):
    accepted(app,'select_scATAC_cells',arguments={'min_qc_fragment_records':1,'min_tss_enrichment':'4'})
    seen=[]
    def execute(sessions,interaction,admitted):
        seen.append(admitted)
        return turns.TurnOutcome('execute','failed')
    monkeypatch.setattr(turns,'execute',execute)
    text='Set the TSS threshold to 5'
    result=ask(app,'execute',text,Model(dict(kind='execute',base='current',operation='op.0',target='selection',
        delta=dict(parameter='min_tss_enrichment',operation='set',literal='5',evidence=text))))
    assert result.kind=='execute' and seen[0]['parameters']['min_tss_enrichment']==5
