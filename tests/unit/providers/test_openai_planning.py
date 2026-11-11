"""Network-free tests for the optional OpenAI planning adapter."""

from __future__ import annotations

import importlib
import json
import math
from types import SimpleNamespace

import pytest

from agent.orchestration import (
    AgentRequest,
    AgentRuntime,
    LLMPlanner,
    PlanningModel,
    PlanningWireMode,
    build_default_tool_registry,
    build_semantic_planning_prompt,
    build_semantic_wire_v4_schema,
)
from agent.orchestration.llm_planner import _response_schema
from agent.providers import (
    OpenAIPlanningDependencyError,
    OpenAIPlanningError,
    OpenAIPlanningModel,
)


_OUTPUT = (
    '{"schema_version":3,"status":"unsupported","steps":[],'
    '"reason":"safe"}'
)
_V4_OUTPUT = json.dumps(
    {
        "schema_version": 4,
        "decision": {
            "kind": "plan",
            "steps": [
                {
                    "step_id": "inspect",
                    "tool": "inspect_scATAC",
                    "sources": [],
                    "control_dependencies": [],
                }
            ],
        },
    }
)


def _response(
    *,
    status: str = "completed",
    output_text: object = _OUTPUT,
    output: object = (),
    error: object = None,
):
    return SimpleNamespace(
        status=status,
        output_text=output_text,
        output=output,
        error=error,
    )


class FakeResponses:
    def __init__(self, result=None, *, error: Exception | None = None) -> None:
        self.result = _response() if result is None else result
        self.error = error
        self.calls: list[dict[str, object]] = []

    @property
    def with_raw_response(self):
        def create(**kwargs):
            result = self.create(**kwargs)
            return SimpleNamespace(parse=lambda: result, status_code=200, request_id='req_test')
        return SimpleNamespace(create=create)

    def create(self, **kwargs):
        schema = kwargs.get("response_format", {}).get("schema") or kwargs.get("text", {}).get("format", {}).get("schema", {})
        if "selection_schema_version" in schema.get("properties", {}):
            self.scope_calls = getattr(self, "scope_calls", []) + [kwargs]
            return _response(output_text=json.dumps({"selection_schema_version": 1, "decision": {
                "kind": "select", "capability_ids": list(json.loads(kwargs["input"])["capabilities"])}}))
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


class FakeClient:
    def __init__(self, result=None, *, error: Exception | None = None) -> None:
        self.responses = FakeResponses(result, error=error)


def _adapter(
    result=None,
    *,
    error: Exception | None = None,
    model: str = "gpt-test",
    timeout: float = 12.5,
) -> tuple[OpenAIPlanningModel, FakeClient]:
    client = FakeClient(result, error=error)
    return (
        OpenAIPlanningModel(
            model=model,
            timeout=timeout,
            _client=client,
        ),
        client,
    )


def _schema(input_path: str = "/synthetic/input.h5ad"):
    return _response_schema(
        build_default_tool_registry(),
        AgentRequest(
            "provider-schema",
            "Inspect the supplied dataset.",
            {"input_path": input_path},
        ),
    )


def test_adapter_satisfies_planning_model_protocol() -> None:
    adapter, _ = _adapter()

    assert isinstance(adapter, PlanningModel)


def test_configured_model_identity_and_timeout_are_stable() -> None:
    adapter, _ = _adapter(model=" gpt-planner-2026 ", timeout=17)

    assert adapter.model == "gpt-planner-2026"
    assert adapter.model_id == "openai:gpt-planner-2026"
    assert adapter.timeout == 17.0


def test_responses_api_request_maps_prompt_and_schema_exactly() -> None:
    adapter, client = _adapter()
    prompt = "exact planning prompt"
    response_schema = _schema()
    plain_schema = json.loads(json.dumps(response_schema))

    returned = adapter.complete(
        prompt=prompt,
        response_schema=response_schema,
    )

    assert returned == _OUTPUT
    assert len(client.responses.calls) == 1
    request = client.responses.calls[0]
    assert request["model"] == "gpt-test"
    assert request["input"] == prompt
    assert request["text"] == {
        "format": {
            "type": "json_schema",
            "name": "agent_plan",
            "strict": True,
            "schema": plain_schema,
        }
    }
    assert plain_schema["properties"]["schema_version"]["enum"] == [3]
    assert "anyOf" in plain_schema["properties"]["steps"]["items"]
    assert request["store"] is False
    assert request["background"] is False
    assert request["timeout"] == 12.5
    assert "tools" not in request
    assert "previous_response_id" not in request
    assert "conversation" not in request


def test_adapter_adds_no_dataset_or_file_values() -> None:
    adapter, client = _adapter()
    prompt = "prompt-without-local-data"
    private_path = "/private/secret-dataset.h5ad"

    adapter.complete(prompt=prompt, response_schema=_schema(private_path))

    serialized = json.dumps(client.responses.calls[0], sort_keys=True)
    assert private_path not in serialized
    assert client.responses.calls[0]["input"] == prompt


def test_success_returns_exact_output_text_without_normalization() -> None:
    exact = "  {\n  \"status\": \"plan\"\n}\n"
    adapter, _ = _adapter(_response(output_text=exact))

    returned = adapter.complete(prompt="prompt", response_schema=_schema())

    assert returned == exact


def test_semantic_v4_round_trip_uses_generic_responses_transport() -> None:
    secret = "/private/OPENAI-V4-SECRET.h5ad"
    request = AgentRequest(
        "openai-v4", "Inspect the supplied dataset.", {"input_path": secret}
    )
    registry = build_default_tool_registry()
    adapter, client = _adapter(_response(output_text=_V4_OUTPUT))

    plan = LLMPlanner(adapter, wire_mode=PlanningWireMode.V4).plan(
        request, registry
    )

    assert not getattr(client.responses, 'scope_calls', ())
    transmitted = client.responses.calls[0]
    assert transmitted["input"] == build_semantic_planning_prompt(
        request, registry
    )
    assert transmitted["text"]["format"]["schema"] == json.loads(
        json.dumps(build_semantic_wire_v4_schema(registry, request))
    )
    serialized = json.dumps(transmitted, sort_keys=True)
    assert secret not in serialized
    assert "tools" not in transmitted
    schema = transmitted["text"]["format"]["schema"]
    assert [
        step["properties"]["tool"]["enum"][0]
        for step in schema["$defs"]["step"]["anyOf"]
    ] == sorted(name for name in registry.names() if name != "invoke_native_function")
    from tests.application.test_composed_scientific_declarations import expand
    expanded = expand(schema)
    def input_domains(node):
        if isinstance(node, dict):
            if 'input' in node.get('properties', {}):
                yield node['properties']['input']['enum']
            for value in node.values(): yield from input_domains(value)
        elif isinstance(node, list):
            for value in node: yield from input_domains(value)
    assert list(input_domains(expanded))
    assert all(domain == ['input_path'] for domain in input_domains(expanded))
    assert plan.steps[0].arguments == {"path": secret}


@pytest.mark.parametrize("missing", [None, "", "   ", 3])
def test_missing_output_fails_cleanly(missing) -> None:
    adapter, _ = _adapter(_response(output_text=missing))

    with pytest.raises(OpenAIPlanningError, match="did not contain output text"):
        adapter.complete(prompt="prompt", response_schema=_schema())


@pytest.mark.parametrize("status", ["incomplete", "failed", "cancelled", None])
def test_noncompleted_response_fails_cleanly(status) -> None:
    adapter, _ = _adapter(_response(status=status))

    with pytest.raises(OpenAIPlanningError, match="was not completed") as raised:
        adapter.complete(prompt="prompt", response_schema=_schema())
    assert raised.value.code == "PROVIDER_COMPLETION_INCOMPLETE"


def test_provider_reported_error_fails_without_exposing_error_object() -> None:
    adapter, _ = _adapter(
        _response(error={"authorization": "Bearer provider-secret"})
    )

    with pytest.raises(OpenAIPlanningError) as raised:
        adapter.complete(prompt="prompt", response_schema=_schema())

    assert "provider-secret" not in str(raised.value)
    assert "authorization" not in str(raised.value).casefold()


def test_refusal_fails_without_exposing_refusal_text() -> None:
    refusal_text = "refusal containing provider-secret"
    refusal = SimpleNamespace(type="refusal", refusal=refusal_text)
    message = SimpleNamespace(type="message", content=[refusal])
    adapter, _ = _adapter(_response(output=[message], output_text=""))

    with pytest.raises(OpenAIPlanningError, match="was refused") as raised:
        adapter.complete(prompt="prompt", response_schema=_schema())

    assert refusal_text not in str(raised.value)
    assert raised.value.code == "PROVIDER_REFUSED"


def test_sdk_exception_is_converted_to_sanitized_adapter_error() -> None:
    secret = "Authorization: Bearer api-secret; raw HTTP response"
    adapter, client = _adapter(error=RuntimeError(secret))

    with pytest.raises(OpenAIPlanningError) as raised:
        adapter.complete(prompt="prompt", response_schema=_schema())

    assert len(client.responses.calls) == 1
    assert str(raised.value) == "OpenAI planning request failed."
    assert "api-secret" not in str(raised.value)
    assert "HTTP" not in str(raised.value)


def test_http_413_is_terminal_request_too_large_without_raw_body() -> None:
    error = RuntimeError("TPM limit and private provider response")
    error.status_code = 413
    adapter, _ = _adapter(error=error)

    with pytest.raises(OpenAIPlanningError) as raised:
        adapter.complete(prompt="prompt", response_schema=_schema())

    assert raised.value.code == "PROVIDER_REQUEST_TOO_LARGE"
    assert raised.value.retry_after_seconds is None
    assert str(raised.value) == (
        "The planning request exceeds the provider's accepted size."
    )
    assert "TPM" not in str(raised.value)
    assert "private" not in str(raised.value)


def test_numeric_retry_after_is_normalized_without_persisting_headers() -> None:
    error = RuntimeError("raw provider body")
    error.status_code = 429
    error.response = SimpleNamespace(
        headers={"Retry-After": "2.5", "Authorization": "Bearer secret"}
    )
    adapter, _ = _adapter(error=error)

    with pytest.raises(OpenAIPlanningError) as raised:
        adapter.complete(prompt="prompt", response_schema=_schema())

    assert raised.value.code == "PROVIDER_RATE_LIMITED"
    assert raised.value.retry_after_seconds == 2.5
    assert "Authorization" not in str(raised.value)
    assert "secret" not in str(raised.value)


def test_credentials_are_not_copied_into_request_or_result(monkeypatch) -> None:
    secret = "sk-test-do-not-copy"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    adapter, client = _adapter()

    result = adapter.complete(prompt="prompt", response_schema=_schema())

    assert secret not in result
    assert secret not in json.dumps(client.responses.calls)


@pytest.mark.parametrize(
    "timeout",
    [0, -1, True, None, "60", math.inf, -math.inf, math.nan],
)
def test_invalid_timeout_is_rejected(timeout) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        OpenAIPlanningModel(model="gpt-test", timeout=timeout, _client=FakeClient())


@pytest.mark.parametrize("model", ["", "   ", None, 3])
def test_invalid_model_is_rejected(model) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        OpenAIPlanningModel(model=model, _client=FakeClient())


def test_invalid_client_boundary_is_rejected() -> None:
    with pytest.raises(TypeError, match=r"responses\.with_raw_response\.create"):
        OpenAIPlanningModel(model="gpt-test", _client=object())


def test_invalid_prompt_or_schema_is_rejected_before_request() -> None:
    adapter, client = _adapter()

    with pytest.raises(ValueError, match="prompt"):
        adapter.complete(prompt="", response_schema=_schema())
    with pytest.raises(TypeError, match="response_schema"):
        adapter.complete(prompt="prompt", response_schema=[])  # type: ignore[arg-type]

    assert client.responses.calls == []


def test_optional_sdk_absence_has_clear_lazy_error(monkeypatch) -> None:
    monkeypatch.setitem(__import__("sys").modules, "openai", None)

    with pytest.raises(
        OpenAIPlanningDependencyError, match="optional OpenAI SDK"
    ) as raised:
        OpenAIPlanningModel(model="gpt-test")
    assert raised.value.code == "PLANNING_PROVIDER_DEPENDENCY_MISSING"


def test_provider_neutral_orchestration_import_does_not_import_openai(
    monkeypatch,
) -> None:
    import sys

    monkeypatch.delitem(sys.modules, "openai", raising=False)
    orchestration = importlib.import_module("agent.orchestration")
    importlib.reload(orchestration)

    assert "openai" not in sys.modules
    assert AgentRuntime().planner is None


def test_default_client_disables_sdk_retries(monkeypatch) -> None:
    import sys
    from types import ModuleType

    calls: list[dict[str, object]] = []
    fake = ModuleType("openai")
    fake.OpenAI = lambda **kwargs: calls.append(kwargs) or FakeClient()
    monkeypatch.setitem(sys.modules, "openai", fake)

    OpenAIPlanningModel(model="gpt-test")

    assert calls == [{"max_retries": 0}]


@pytest.mark.parametrize('status,code',[('incomplete','PROVIDER_COMPLETION_INCOMPLETE'),('completed','PROVIDER_OUTPUT_MISSING')])
def test_incomplete_and_missing_outputs_have_distinct_safe_diagnostics(caplog,status,code):
    response=_response(output_text='')
    response.status=status
    response.output=[]
    response.incomplete_details=SimpleNamespace(reason='max_output_tokens')
    response.usage=SimpleNamespace(input_tokens=11,output_tokens=0,total_tokens=11,private='/private/secret')
    adapter,client=_adapter(response)
    with pytest.raises(OpenAIPlanningError) as caught:adapter.complete(prompt='PRIVATE PROMPT',response_schema=_schema())
    assert caught.value.code==code and len(client.responses.calls)==1
    log=caplog.text
    assert '"completion_status":"'+status+'"' in log
    assert '"incomplete_reason":"max_output_tokens"' in log and '"output_item_count":0' in log
    assert 'PRIVATE PROMPT' not in log and '/private/secret' not in log


@pytest.mark.parametrize('usage_state', ['absent', 'null', 'zero', 'measured'])
@pytest.mark.parametrize('status', ['completed', 'incomplete'])
def test_official_sdk_wire_presence_transport_and_text(caplog, usage_state, status):
    openai = pytest.importorskip('openai')
    httpx = pytest.importorskip('httpx')
    exact = '  {"ok":true}\n'
    body = dict(id='resp_private', object='response', status=status,
        model='gpt-test', error=None, max_output_tokens=16384,
        reasoning={'effort': 'medium'},
        incomplete_details={'reason': 'max_output_tokens'} if status == 'incomplete' else None,
        output=[] if status == 'incomplete' else [
            {'type': 'reasoning', 'id': 'private-reasoning', 'summary': []},
            {'type': 'message', 'id': 'private-message', 'role': 'assistant', 'status': 'completed',
             'content': [{'type': 'output_text', 'text': exact, 'annotations': []}]}])
    if usage_state == 'null':
        body['usage'] = None
    elif usage_state in ('zero', 'measured'):
        n = 0 if usage_state == 'zero' else 17
        body['usage'] = dict(input_tokens=n, output_tokens=n, total_tokens=2*n,
            input_tokens_details={'cached_tokens': n}, output_tokens_details={'reasoning_tokens': n})
    requests = []
    def handle(request):
        requests.append(json.loads(request.content))
        assert request.url == 'https://api.openai.com/v1/responses'
        return httpx.Response(200, json=body, headers={'x-request-id': 'req_wiretest'})
    with openai.OpenAI(api_key='test-secret', base_url='https://api.openai.com/v1', max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
        adapter = OpenAIPlanningModel(model='gpt-test', _client=client)
        schema = {'type': 'object', 'properties': {}, 'additionalProperties': False}
        if status == 'completed':
            assert adapter.complete(prompt='PRIVATE PROMPT', response_schema=schema) == exact
        else:
            with pytest.raises(OpenAIPlanningError) as caught:
                adapter.complete(prompt='PRIVATE PROMPT', response_schema=schema)
            assert caught.value.code == 'PROVIDER_COMPLETION_INCOMPLETE'
    assert requests == [dict(model='gpt-test', input='PRIVATE PROMPT', store=False,
        background=False, text={'format': {'type': 'json_schema', 'name': 'agent_plan', 'strict': True, 'schema': schema}})]
    row = json.loads(caplog.records[-1].message.split('Agent operation diagnostic ')[1])
    assert row['http_status'] == 200 and row['provider_request_id'] == 'req_wiretest'
    assert row['usage_state'] == (usage_state if usage_state in ('absent', 'null') else 'present')
    if usage_state in ('absent', 'null'):
        assert 'input_tokens' not in row and 'output_tokens' not in row and 'reasoning_tokens' not in row
    else:
        assert row['input_tokens'] == row['output_tokens'] == row['reasoning_tokens'] == n
    assert row['max_output_tokens'] == 16384 and row['reasoning_effort'] == 'medium'
    assert row['output_types'] == ({} if status == 'incomplete' else {'reasoning': 1, 'message': 1})
    assert row['content_types'] == ({} if status == 'incomplete' else {'output_text': 1})
    for secret in ('PRIVATE PROMPT', exact, 'test-secret', 'resp_private', 'private-message'):
        assert secret not in caplog.text


def test_official_sdk_schema_http_error_is_separate_from_incomplete(caplog):
    openai = pytest.importorskip('openai')
    httpx = pytest.importorskip('httpx')
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(400, headers={'x-request-id': 'req_schema'}, json={'error': {
            'type': 'invalid_request_error', 'code': 'invalid_json_schema',
            'message': 'PRIVATE SCHEMA ECHO', 'param': 'text.format.schema'}})
    with openai.OpenAI(api_key='test-secret', max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
        adapter = OpenAIPlanningModel(model='gpt-test', _client=client)
        with pytest.raises(OpenAIPlanningError) as caught:
            adapter.complete(prompt='private', response_schema={})
    assert len(calls) == 1
    assert caught.value.code != 'PROVIDER_COMPLETION_INCOMPLETE'
    row = json.loads(caplog.records[-1].message.split('Agent operation diagnostic ')[1])
    assert row['http_status'] == 400 and row['provider_request_id'] == 'req_schema'
    assert row['event'] == 'openai.request_failed'
    assert 'completion_status' not in row and 'PRIVATE SCHEMA ECHO' not in caplog.text
