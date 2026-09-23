"""Interleave M16 answers and unchanged M15 science/navigation on tiny real owners."""
import json

from test_prior_outputs import source, application_chain, chain, all_work
from test_intent_turns import app_for, change, Interpreter


class Dialogue:
    def __init__(self, comparison=False): self.comparison = comparison
    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        if 'turn_schema_version' in value:
            outputs = value['dialogue']['outputs']
            def target(relation):
                matching = [o for o in outputs if o['output_name']=='selection' and relation in o['relations']]
                assert len(matching)==1
                return dict(output=matching[0]['handle'],subject=None)
            return json.dumps(dict(turn_schema_version=1,decision=dict(kind='answer',intent='scientific',
                target=target('current'), comparison=target('parent') if self.comparison else None,focus='question')))
        ids = [c['claim_id'] for c in value['evidence']['claims'] if c['field']=='n_selected']
        return json.dumps(dict(support='supported',paragraphs=[dict(
            parts=[dict(kind='text',text='The accepted selection records:')]+[dict(kind='claim',id=c) for c in ids])]))


def test_dialogue_execution_navigation_and_reuse_stay_separate(source):
    app, planner = app_for(source)
    original = app.sessions.load('analysis').active_revision_id
    with all_work() as calls:
        first = app.sessions.respond('analysis','explain','Explain selection results.',interpreter=Dialogue())
    assert first.status=='answered' and not calls, (first,calls)
    executed = app.sessions.respond('analysis','change','Set TSS to 7',interpreter=change('Set TSS to 7',target='selection'))
    assert executed.status=='activated',executed
    newer = app.sessions.load('analysis').active_revision_id
    assert newer != original
    with all_work() as calls:
        comparison = app.sessions.respond('analysis','compare','Compare selection with the parent.',interpreter=Dialogue(True))
        navigation = app.sessions.respond('analysis','rollback','Go back to the parent version',
            interpreter=Interpreter(dict(kind='navigate',relation='parent')))
        old = app.sessions.respond('analysis','old','Explain selection results.',interpreter=Dialogue())
    assert not calls, calls
    assert comparison.status == old.status == 'answered', (comparison,old)
    assert navigation.status=='activated'
    assert {c.source['revision_id'] for c in comparison.scientific.claims} == {original,newer}
    assert {c.source['revision_id'] for c in old.scientific.claims} == {original}
    assert old.scientific.claims[0].value == first.scientific.claims[0].value
