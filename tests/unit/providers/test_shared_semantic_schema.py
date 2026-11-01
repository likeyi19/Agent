"""All existing adapters carry the same compact wire-v4 schema unchanged."""
import json

import pytest

from agent.orchestration import AgentRequest, build_default_tool_registry
from agent.orchestration.semantic_wire_v4 import build_semantic_wire_v4_schema


@pytest.mark.parametrize('provider', ('openai', 'groq', 'gemini', 'openrouter', 'alibaba'))
def test_compact_planning_schema_is_transmitted_without_provider_rewriting(provider):
    if provider == 'openai':
        from test_openai_planning import _adapter
        model, client = _adapter()
    elif provider == 'groq':
        from test_groq_planning import _adapter
        model, client = _adapter()
    elif provider == 'gemini':
        from test_gemini_planning import _adapter
        model, client = _adapter()
    elif provider == 'openrouter':
        from test_openrouter_planning import adapter
        model, client = adapter()
    else:
        from test_alibaba_planning import Client
        from agent.providers import AlibabaPlanningModel
        client = Client()
        model = AlibabaPlanningModel(model='qwen3.8-max', _client=client)
    schema = build_semantic_wire_v4_schema(build_default_tool_registry(), AgentRequest(
        'provider-contract', 'Inspect.', {'input_path': '/PRIVATE.h5ad', 'output_dir': '/PRIVATE'}))
    before = json.loads(json.dumps(schema))
    model.complete(prompt='exact prompt', response_schema=schema)
    if provider in ('openai', 'groq'):
        calls = client.responses.calls
        transmitted = calls[0]['text']['format']['schema']
    elif provider == 'gemini':
        calls = client.interactions.calls
        transmitted = calls[0]['response_format']['schema']
    else:
        calls = client.calls if provider == 'alibaba' else client.chat.completions.calls
        transmitted = calls[0]['response_format']['json_schema']['schema']
    assert len(calls) == 1
    assert transmitted == before == json.loads(json.dumps(schema))
    assert len(transmitted['$defs']) > 5
    assert 'PRIVATE' not in json.dumps(calls)
