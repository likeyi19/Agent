"""Existing Web choice and member editor transport the conditional native input."""
import pytest
from fastapi.testclient import TestClient
from agent.schemas import VerificationResult, VerificationCheck
from agent.web.app import create_app
from helpers import wait_turn
from test_preparation_owner_contracts import Model, setup


def test_barcode_choice_and_selected_member_editor_use_existing_http_controls(tmp_path):
    model=Model('run_snapatac2','pp.make_fragment_file',roles=False)
    app,executor=setup(tmp_path,model)
    path=tmp_path/'source.bam';path.write_bytes(b'Unopened Web contract fixture')
    first=app.submit_turn('session','task','Convert this BAM to fragments.',expected_generation=0,
        execution_inputs=dict(input_path=str(path)))
    assert first.status=='clarification'
    before=app._application.sessions.load('session').interactions[0]
    checked=[]
    def preflight(plan):
        checked.append(plan)
        return VerificationResult(False,'plan',plan.plan_id,(VerificationCheck('withheld',False,'Contract gate'),))
    executor.preflight=preflight
    model.complete=lambda **kwargs:pytest.fail('Exact Web Answer called a model')
    choice=next(c for c in first.clarification_choices if c.get('choice_identity')=='native-parameter.barcode_tag')
    with TestClient(create_app(app)) as client:
        selected=client.post('/api/v1/sessions/session/turns',json=dict(turn_id='mechanism',utterance='',expected_generation=0,
            clarification_choice=dict(question_turn_id='task',handle=choice['handle'])))
        assert selected.status_code==202,selected.text
        second=wait_turn(client,'mechanism')
        editor,=second['clarification_choices']
        assert editor['action']=='parameter' and editor['member']=='barcode_tag'
        assert editor['native_function']=='pp.make_fragment_file' and editor['step_id']=='method'
        metadata=client.get('/api/v1/parameters',params=dict(tool=editor['tool'],function=editor['native_function']))
        assert metadata.status_code==200,metadata.text
        fields=[p for p in metadata.json()['tools'][0]['kwargs'] if p['argument']==editor['member']]
        assert len(fields)==1 and 'str' in fields[0]['types']
        submitted=client.post('/api/v1/sessions/session/turns',json=dict(turn_id='value',utterance='',expected_generation=0,
            clarification_choice=dict(question_turn_id='mechanism',handle=editor['handle']),
            parameter_overrides=[dict(tool=editor['tool'],native_function=editor['native_function'],
                step_id=editor['step_id'],argument='kwargs',value={'barcode_tag':'RG'})]))
        assert submitted.status_code==202,submitted.text
        final=wait_turn(client,'value')
        assert final['error']['code']=='PLAN_PREFLIGHT_FAILED',final
    assert dict(checked[-1].steps[0].arguments['kwargs'])=={'barcode_tag':'RG'}
    assert not executor.plans
    assert app._application.sessions.load('session').interactions[0]==before
