"""Provider-independent bounded subject and structured reference contracts."""
import json

import pytest

from agent.application import scientific_dialogue as sd
from agent.application.turn_decisions import IntentError
from agent.report.analysis_report import _FIELD_LABELS
from test_dialogue_evidence import app, accepted, forbid_work
from test_scientific_dialogue import Model, question, annotation, ask


@pytest.mark.parametrize('reference', ['7','cluster 7'])
def test_offered_subject_references_persist_canonical_identity(app, reference):
    annotation(app)
    def decide(prompt):
        candidates=prompt['dialogue']['outputs'][0]['subjects']['candidates']
        assert next(c for c in candidates if c['id']=='7')['references']==['7','cluster 7']
        return question('r0',reference)
    result=ask(app,'canonical','What does cluster 7 show?',Model(decide,['primary_annotation']))
    assert result.status=='answered'
    assert {c.subject for c in result.scientific.claims}=={'7'}
    assert app.sessions.load('session').interactions[-1].admitted['target']['subject']=='7'


def test_resolver_uses_only_offered_candidates_and_rejects_collisions():
    offered=[dict(id='7',references=['7','cluster 7']),dict(id='other',references=['cluster 7'])]
    assert sd.resolve_subject('7',offered)=='7'
    for reference in ('cluster 7','cluster 8','07','7 ','Cluster 7'):
        with pytest.raises(IntentError): sd.resolve_subject(reference,offered)
    # Neither exact IDs nor display forms take precedence over a collision.
    with pytest.raises(IntentError):
        sd.resolve_subject('7',offered+[dict(id='third',references=['7'])])


def test_subject_reference_forms_are_literal_bounded_and_domain_neutral():
    for noun in ('donor','sample','cluster'):
        c=sd.subject_candidates(['7','70'],f'Explain {noun} 7')
        assert sd.resolve_subject(f'{noun} 7',c)=='7'
        assert c[1]['references']==['70']
    c=sd.subject_candidates([str(i) for i in range(100)],'sample 99')
    assert len(c)==32
    with pytest.raises(IntentError):sd.resolve_subject('99',c)


def test_new_claim_parts_bind_without_magic_text_or_duplicate_id_lists(app,monkeypatch):
    annotation(app)
    def response(prompt):
        claim=next(c for c in prompt['evidence']['claims'] if c['field']=='primary_annotation')
        assert claim['subject']=='7'
        return dict(support='supported',paragraphs=[dict(parts=[
            dict(kind='text',text='The accepted assignment is'),dict(kind='claim',id=claim['claim_id'])])])
    result=ask(forbid_work(app,monkeypatch),'parts','Explain cluster 7',Model(question(subject='cluster 7'),response=response))
    assert result.status=='answered' and 'group 7' in result.text and 'B cell' in result.text
    assert result.scientific.claims[0].field=='primary_annotation'


def test_reviewed_semantics_transported_verbatim_missing_units_not_invented(app):
    accepted(app,'compute_scATAC_qc',overrides={'resource_qualification':'operator_qualified'})
    model=Model(question(),['resource_qualification','tss_defined'])
    result=ask(app,'semantics','What does this result establish?',model)
    assert result.status=='answered'
    claims={c['field']:c for c in model.calls[-1]['evidence']['claims']}
    for field in ('resource_qualification','tss_defined','n_qc_fragment_records'):
        assert claims[field]['semantics']['label']==_FIELD_LABELS[field]
        assert claims[field]['semantics']['label_source']=='reviewed_report_field'
        assert 'unit' not in claims[field]['semantics']
    assert claims['artifact_schema_version']['semantics']=={}
    assert 'QC resource qualification' in result.text


def test_generation_union_has_distinct_kinds_and_bounded_reference_enums(app):
    annotation(app)
    class Check(Model):
        def complete(self,*,prompt,response_schema):
            p=json.loads(prompt)
            if 'dialogue_schema_version' in p:
                variants=response_schema['properties']['paragraphs']['items']['properties']['parts']['items']['anyOf']
                assert [v['properties']['kind']['enum'] for v in variants]==[['text'],['claim'],['meaning']]
                assert variants[1]['properties']['id']['enum']==[c['claim_id'] for c in p['evidence']['claims']]
            return super().complete(prompt=prompt,response_schema=response_schema)
    assert ask(app,'schema','Explain cluster 7',Check(question(subject='7'),['primary_annotation'])).status=='answered'


def test_registry_field_description_is_reused_without_rephrasing(app):
    accepted(app,'epizoo_embed_cells',overrides={'species':'human'})
    model=Model(question(),['species'])
    assert ask(app,'registry','Explain this result',model).status=='answered'
    claim=next(c for c in model.calls[-1]['evidence']['claims'] if c['field']=='species')
    assert claim['semantics']['description']==app.registry.get('epizoo_embed_cells').result_contract.planning_fields['species'].description


def test_owner_profile_is_not_exposed_for_a_different_accepted_profile(app):
    from agent.tools.data.scatac_qc_profile import PROFILE
    accepted(app,'compute_scATAC_qc',overrides={'science_profile_sha256':'0'*64},
             extra_facts={'tss_method':PROFILE.tss_method})
    model=Model(question(),['tss_defined'])
    assert ask(app,'different','Explain the result',model).status=='answered'
    assert 'reviewed_profile' not in model.calls[-1]['evidence']['targets'][0]
    assert 'm2' not in model.calls[-1]['evidence']['meanings']
