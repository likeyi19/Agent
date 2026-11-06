"""R5 organization preferences, exact sources and bounded complete language requests.

Accepted metadata and language judgements are scripted; no native science runs.
"""
import json
from types import SimpleNamespace

import pytest

from agent.application import scientific_dialogue as sd
from agent.application.dialogue_evidence import field_semantics
from agent.orchestration.planning_model import PlanningModelError
from test_dialogue_evidence import app, accepted, files, forbid_work
from test_scientific_dialogue import Model, ask, question


def target(app, revision_id):
    revision = next(r for r in app.sessions.load('session').revisions if r.revision_id == revision_id)
    output = revision.outputs[0]
    return dict(revision_id=revision_id, output_name=output.name,
        accepted_step_sha256=output.accepted_step_sha256, subject=None)


def test_related_results_keep_original_and_mapped_facts_with_exact_attribution(app,monkeypatch):
    strict = accepted(app,'compare_cell_annotations',overrides=dict(accuracy=1/3),
        extra_facts=dict(original_metrics=dict(accuracy=1/3),metric_denominators=dict(accuracy_cells=6)))[0]
    mapped = accepted(app,'compare_cell_annotations',name='mapped',overrides=dict(accuracy=5/6),
        extra_facts=dict(original_metrics=dict(accuracy=1/3),metric_denominators=dict(accuracy_cells=6),
            mapping=dict(prediction=dict(rules=[dict(original_label='A',common_label='T')]))))[0]
    guarded=forbid_work(app,monkeypatch);before=files(guarded)
    public,claims=sd.context(guarded.sessions,'session',dict(target=target(app,mapped),comparison=target(app,strict)))
    values={(c.target,c.field):c for c in claims}
    assert values['t0','accuracy'].value==5/6 and values['t1','accuracy'].value==1/3
    assert values['t0','original_metrics'].value['accuracy']==1/3
    assert values['t0','metric_denominators'].value['accuracy_cells']==6
    assert values['t0','mapping'].source['revision_id']==mapped
    assert values['t1','accuracy'].source['revision_id']==strict
    assert public['targets'][0]['is_active'] and not public['targets'][1]['is_active']
    # Equal values and shared definitions never erase a separately accepted source.
    for entry in public['claims']:
        definition=entry['semantics'].get('description_ref')
        if definition:
            assert public['meanings'][definition]==field_semantics(app.registry,'compare_cell_annotations',entry['field'])['description']
    assert files(guarded)==before


def test_distinct_method_results_share_definitions_without_merging_facts(app,monkeypatch):
    one=accepted(app,'pca_embed_cells',overrides=dict(parameters=dict(normalization='none',centered=True),embedding_dim=2))[0]
    two=accepted(app,'pca_embed_cells',name='second',overrides=dict(parameters=dict(normalization='tfidf_l2',centered=True),embedding_dim=3))[0]
    guarded=forbid_work(app,monkeypatch)
    public,claims=sd.context(guarded.sessions,'session',dict(target=target(app,two),comparison=target(app,one)))
    assert {c.value for c in claims if c.field=='embedding_dim'}=={2,3}
    params=[c for c in claims if c.field=='parameters']
    assert len(params)==2 and {c.value['normalization'] for c in params}=={'none','tfidf_l2'}
    assert {c.source['revision_id'] for c in params}=={one,two}
    assert public['comparison']['mode']=='descriptive_accepted_operands'
    entries=[c for c in public['claims'] if c['field']=='parameters']
    assert entries[0]['semantics']==entries[1]['semantics']
    assert len(json.dumps(public).encode())<sd.PREFERRED_CONTEXT_BYTES


def test_each_selected_summary_keeps_essential_fields_beyond_the_old_global_claim_limit(app,monkeypatch):
    extra={f'observed_{i}':i for i in range(75)}
    a=accepted(app,'inspect_scATAC',extra_facts=extra)[0]
    b=accepted(app,'inspect_scATAC',name='second',extra_facts=extra)[0]
    guarded=forbid_work(app,monkeypatch)
    _,claims=sd.context(guarded.sessions,'session',dict(target=target(app,b),comparison=target(app,a)))
    assert len(claims)>64
    for handle in ('t0','t1'):
        assert {c.field:c.value for c in claims if c.target==handle and c.field.startswith('observed_')}==extra
        assert len([c for c in claims if c.target==handle])<=sd.MAX_CLAIMS_PER_TARGET


@pytest.mark.parametrize('size',[200,12000])
def test_valid_contexts_on_both_sides_of_the_old_byte_limit_reach_generation_and_review(app,monkeypatch,size):
    # Whole owner records stay bounded by the existing 16 KiB fact reader.
    extra={f'record_{i}':dict(description='x'*size,denominator=6) for i in range(4)}
    rid=accepted(app,'inspect_scATAC',extra_facts=extra)[0]
    guarded=forbid_work(app,monkeypatch);before=files(guarded)
    public,claims=sd.context(guarded.sessions,'session',dict(target=target(app,rid),comparison=None))
    measured=len(json.dumps(public,ensure_ascii=False).encode())
    assert (measured>32768)==(size==12000)
    stages=[]
    class Observed(Model):
        def complete(self,*,prompt,response_schema):
            payload=json.loads(prompt)
            stages.append(next(k for k in ('turn_schema_version','dialogue_schema_version','grounding_review_version') if k in payload))
            return super().complete(prompt=prompt,response_schema=response_schema)
    model=Observed(question(),['n_cells'])
    result=ask(guarded,'answer','Explain this accepted result.',model)
    assert result.status=='answered' and result.scientific.support=='supported'
    assert len(result.scientific.claims)>0
    assert all(c.source['revision_id']==rid for c in result.scientific.claims)
    assert stages==['turn_schema_version','dialogue_schema_version','grounding_review_version']
    assert len(model.calls)==2 and files(guarded)!=before  # Only the saved Interaction/response is new.
    assert guarded.sessions.load('session').revisions==app.sessions.load('session').revisions


def test_long_owner_limitations_are_not_silently_dropped(app,monkeypatch):
    limitation='A computational result is not independent biological truth. '*20
    rid=accepted(app,'inspect_scATAC',extra_facts=dict(owner_limitations=limitation))[0]
    guarded=forbid_work(app,monkeypatch)
    _,claims=sd.context(guarded.sessions,'session',dict(target=target(app,rid),comparison=None))
    assert next(c for c in claims if c.field=='owner_limitations').value==limitation


def test_whole_request_protection_counts_schema_and_existing_answer_allowance(monkeypatch):
    calls=[];model=SimpleNamespace(complete=lambda **kw:calls.append(kw) or 'response')
    prompt='full instructions and evidence';schema=dict(type='string',enum=['x'*100])
    full=len(prompt.encode())+len(json.dumps(schema,ensure_ascii=False).encode())+sd.MAX_ANSWER_BYTES
    monkeypatch.setattr(sd,'MAX_LANGUAGE_REQUEST_BYTES',full-1)
    with pytest.raises(sd._AnswerFailure) as exc:sd._language_complete(model,prompt,schema,'provider')
    assert exc.value.stage=='request_preparation' and exc.value.reason=='request_resource_limit' and calls==[]
    monkeypatch.setattr(sd,'MAX_LANGUAGE_REQUEST_BYTES',full)
    assert sd._language_complete(model,prompt,schema,'provider')=='response' and len(calls)==1


def test_actual_provider_context_rejection_is_not_called_an_evidence_binding_failure(app,monkeypatch):
    accepted(app,'inspect_scATAC')
    guarded=forbid_work(app,monkeypatch)
    class Rejected(Model):
        def complete(self,*,prompt,response_schema):
            if 'dialogue_schema_version' in json.loads(prompt):
                self.calls.append(json.loads(prompt))
                raise PlanningModelError('private transport text',code='PROVIDER_REQUEST_TOO_LARGE')
            return super().complete(prompt=prompt,response_schema=response_schema)
    model=Rejected(question())
    result=ask(guarded,'answer','Explain.',model)
    assert result.status=='unavailable' and 'provider rejected' in result.text and 'too large' in result.text
    assert 'binding' not in result.text and 'private transport' not in result.text and len(model.calls)==2
