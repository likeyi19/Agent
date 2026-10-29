"""Strict Qwen transport, private reasoning exclusion and normal factory selection."""
import copy
import json
import sys
import traceback
from types import MappingProxyType,SimpleNamespace

import pytest

from agent.providers import AlibabaPlanningModel,AlibabaPlanningError
from agent.providers import alibaba_planning as owner
from agent.orchestration import PlanningModel

SCHEMA={'type':'object','properties':{'ok':{'type':'boolean'}},'required':['ok'],'additionalProperties':False}
CONTENT=' {"ok":true} '


def response(**changes):
    value=dict(model='qwen3.8-max',choices=[dict(finish_reason='stop',message=dict(
        role='assistant',content=CONTENT,reasoning_content='PRIVATE_REASONING_SENTINEL'))])
    value.update(changes);return value


class Client:
    def __init__(self,value=None,error=None):
        self.value=response() if value is None else value;self.error=error;self.calls=[]
        self.chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))
    def create(self,**kwargs):
        self.calls.append(kwargs)
        if self.error:raise self.error
        return self.value


def test_strict_schema_exact_content_no_reasoning_and_no_fallback(caplog):
    client=Client();model=AlibabaPlanningModel(model='qwen3.8-max',timeout=120,_client=client)
    assert isinstance(model,PlanningModel) and model.model_id=='alibaba:qwen3.8-max'
    schema=MappingProxyType(copy.deepcopy(SCHEMA))
    assert model.complete(prompt='Intended context',response_schema=schema)==CONTENT
    assert client.calls==[dict(model='qwen3.8-max',messages=[dict(role='user',content='Intended context')],
        response_format=dict(type='json_schema',json_schema=dict(name='agent_plan',strict=True,schema=SCHEMA)),
        stream=False,timeout=120.,max_completion_tokens=4096,extra_body=dict(enable_thinking=False))]
    assert 'PRIVATE_REASONING_SENTINEL' not in caplog.text
    client.calls[0]['response_format']['json_schema']['schema']['required'].append('x')
    assert schema==SCHEMA


@pytest.mark.parametrize('change',[
    {'finish_reason':'length'},{'finish_reason':'tool_calls'},
    {'message':dict(role='assistant',content=None,reasoning_content='PRIVATE_REASONING_SENTINEL')},
    {'message':dict(role='user',content=CONTENT)},
    {'message':dict(role='assistant',content=CONTENT,tool_calls=[{}])},
    {'message':dict(role='assistant',content=CONTENT,function_call={"name":"tool"})},
    {'message':dict(role='assistant',content='',reasoning_content=CONTENT)},
])
def test_incomplete_or_nonassistant_output_is_never_planning_content(change,caplog):
    value=response();value['choices'][0].update(change);client=Client(value)
    with pytest.raises(AlibabaPlanningError) as err:
        AlibabaPlanningModel(model='qwen3.8-max',_client=client).complete(prompt='test',response_schema=SCHEMA)
    assert err.value.code=='PROVIDER_COMPLETION_INCOMPLETE' and len(client.calls)==1
    assert 'PRIVATE_REASONING_SENTINEL' not in caplog.text+str(err.value)


@pytest.mark.parametrize('value',[response(model='another-model'),response(choices=[]),response(error={"message":"private"})])
def test_failed_or_other_model_response_is_rejected(value):
    with pytest.raises(AlibabaPlanningError):
        AlibabaPlanningModel(model='qwen3.8-max',_client=Client(value)).complete(prompt='test',response_schema=SCHEMA)


def test_refusal_and_sdk_error_are_sanitized_without_retry(caplog):
    value=response();value['choices'][0]['message']['refusal']='private refusal'
    with pytest.raises(AlibabaPlanningError) as error:
        AlibabaPlanningModel(model='qwen3.8-max',_client=Client(value)).complete(prompt='test',response_schema=SCHEMA)
    assert error.value.code=='PROVIDER_REFUSED' and 'private refusal' not in str(error.value)
    class Failure(Exception):status_code=429;headers={'Retry-After':'3'}
    client=Client(error=Failure('credential=SECRET_PRIVATE reasoning_content=PRIVATE_REASONING_SENTINEL'))
    with pytest.raises(AlibabaPlanningError) as error:
        AlibabaPlanningModel(model='qwen3.8-max',_client=client).complete(prompt='test',response_schema=SCHEMA)
    assert error.value.code=='PROVIDER_RATE_LIMITED' and error.value.retry_after_seconds==3
    rendered=''.join(traceback.format_exception(error.type,error.value,error.tb))+caplog.text
    assert 'SECRET_PRIVATE' not in rendered and 'PRIVATE_REASONING_SENTINEL' not in rendered and len(client.calls)==1


def test_private_server_environment_constructs_sdk_once(monkeypatch):
    calls=[]
    def create(**kwargs):calls.append(kwargs);return Client()
    monkeypatch.setitem(sys.modules,'openai',SimpleNamespace(OpenAI=create))
    monkeypatch.setenv('DASHSCOPE_API_KEY','test-private-key')
    monkeypatch.setenv('DASHSCOPE_BASE_URL','https://workspace.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1')
    monkeypatch.setenv('DASHSCOPE_MODEL','ignored-model')
    model=AlibabaPlanningModel(model='qwen3.8-max')
    assert model.model=='qwen3.8-max'
    assert calls==[dict(api_key='test-private-key',base_url='https://workspace.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1',max_retries=0)]


@pytest.mark.parametrize('url',['','http://host/v1','https://user:secret@host/v1','https://host/v1?key=secret'])
def test_missing_or_invalid_endpoint_fails_before_network(url,monkeypatch):
    monkeypatch.setenv('DASHSCOPE_API_KEY','private');monkeypatch.setenv('DASHSCOPE_BASE_URL',url)
    monkeypatch.setitem(sys.modules,'openai',SimpleNamespace(OpenAI=lambda **k:pytest.fail('constructed SDK')))
    with pytest.raises(AlibabaPlanningError) as error:AlibabaPlanningModel(model='qwen3.8-max')
    assert error.value.code=='PLANNING_PROVIDER_CONFIGURATION_FAILED' and 'secret' not in str(error.value)


def test_missing_key_is_an_explicit_prerequisite(monkeypatch):
    monkeypatch.delenv('DASHSCOPE_API_KEY',raising=False)
    monkeypatch.setenv('DASHSCOPE_BASE_URL','https://workspace.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1')
    with pytest.raises(AlibabaPlanningError,match='DASHSCOPE_API_KEY'):AlibabaPlanningModel(model='qwen3.8-max')
