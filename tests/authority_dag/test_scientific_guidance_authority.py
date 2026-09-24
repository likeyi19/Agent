"""Read-only guidance over real accepted owner publications, both producer routes."""
import json

from test_prior_outputs import source, application_chain, chain, all_work
from test_dialogue_evidence_authority import snapshot


class Guidance:
    def complete(self, *, prompt, response_schema):
        p = json.loads(prompt)
        if 'turn_schema_version' in p:
            targets = [dict(output=o['handle'],subject=None) for o in p['dialogue']['outputs']
                       if 'current' in o['relations'] and o['output_name'] in ('qc','selection','matrix')]
            assert 2 <= len(targets) <= 3
            return json.dumps(dict(turn_schema_version=1,decision=dict(kind='answer_guidance',targets=targets,candidate=None)))
        e=p['context']['evidence']
        assert 'comparison' not in e
        claims = [c for c in e['claims'] if c['field'].startswith('detail.length_histogram.')]
        assert claims, e
        parts=[dict(kind='claim',id=claims[0]['claim_id'])]
        return json.dumps(dict(candidates=[dict(capability='inspect_scATAC', explanation=dict(
            support='supported',paragraphs=[dict(parts=parts),
                dict(parts=[dict(kind='text',text='Assuming the declared objective remains applicable.')]),
                dict(parts=[dict(kind='text',text='Further inputs may be needed; this is conditional advice.')])]))]))


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
