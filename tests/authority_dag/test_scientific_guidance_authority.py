"""Read-only guidance over real accepted owner publications, both producer routes."""
import json

from test_prior_outputs import source, application_chain, chain, all_work
from test_dialogue_evidence_authority import snapshot


class Guidance:
    def complete(self, *, prompt, response_schema):
        p = json.loads(prompt)
        if 'grounding_review_version' in p:
            return json.dumps(dict(integrity='intact', verdicts={row['id']: dict(verdict='entailed', independent=True) for row in p['paragraphs']}))
        if 'turn_schema_version' in p:
            targets = [dict(output=o['handle'],subject=None) for o in p['dialogue']['outputs']
                       if 'current' in o['relations'] and o['output_name'] in ('qc','selection','matrix')]
            assert 2 <= len(targets) <= 3
            return json.dumps(dict(turn_schema_version=1,decision=dict(kind='answer_guidance',targets=targets,candidate=None)))
        e=p['context']['evidence']
        assert 'comparison' not in e
        claims = [c for c in e['claims'] if c['field'].startswith('detail.length_histogram.')]
        assert claims, e
        return json.dumps(dict(candidates=[dict(capability='inspect_scATAC', explanation=dict(
            support='supported',paragraphs=[dict(text='The accepted histogram describes observed fragment lengths.', kind='finding', citations=[claims[0]['claim_id']], comparisons=[]),
                dict(text='Assuming the declared objective remains applicable.', kind='conditional', citations=[], comparisons=[]),
                dict(text='Further inputs may be needed; this is conditional advice.', kind='conditional', citations=[], comparisons=[])]))]))


def test_real_guidance_reuses_owner_evidence_and_detail_without_work(source):
    _,_,app=source
    before=snapshot(app)
    with all_work() as calls:
        result=app.sessions.respond('analysis','guidance','What should I analyze next?',interpreter=Guidance())
    assert result.status=='answered',result.text
    assert not calls,calls
    claim=result.guidance.candidates[0].explanation.claims[0]
    assert claim.source['artifact_sha256']
    assert claim.source['source_run_result_sha256']
    after=snapshot(app)
    assert {k:v for k,v in after.items() if not k.startswith('sessions/')} == {
        k:v for k,v in before.items() if not k.startswith('sessions/')}
