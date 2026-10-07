"""Benchmark-only exact free candidates using the existing strict Chat adapter.

OpenRouter documents ``provider.require_parameters`` to prevent schema options
being silently ignored: https://openrouter.ai/docs/guides/features/structured-outputs
No prompt changes, response healing, model fallbacks or hosting-provider pinning.
"""
from __future__ import annotations

from collections.abc import Mapping
import os
import re
from types import SimpleNamespace

from agent.orchestration.planning_model import PlanningModelError, classify_provider_exception
from .candidates import Availability, Candidate, build_candidate
from .diagnostics import _redactor, sanitize_provider_error


OPENROUTER_CANDIDATES = (
    Candidate('openrouter-nemotron-3-super', 'openrouter',
              'nvidia/nemotron-3-super-120b-a12b:free', 'openai-chat-completions',
              'OPENROUTER_API_KEY', 'q1d-openrouter-nemotron-3-super',
              'https://openrouter.ai/api/v1'),
    Candidate('openrouter-nex-n2.5-pro', 'openrouter',
              'nex-agi/nex-n2.5-pro:free', 'openai-chat-completions',
              'OPENROUTER_API_KEY', 'q1d-openrouter-nex-n2-5-pro',
              'https://openrouter.ai/api/v1'),
    Candidate('openrouter-minimax-m3', 'openrouter',
              'minimax/minimax-m3:free', 'openai-chat-completions',
              'OPENROUTER_API_KEY', 'q1d-openrouter-minimax-m3',
              'https://openrouter.ai/api/v1'),
)


Q1E_CANDIDATES = (
    Candidate('openrouter-nex-n2.5-mini', 'openrouter',
              'nex-agi/nex-n2.5-mini:free', 'openai-chat-completions',
              'OPENROUTER_API_KEY', 'q1e-openrouter-nex-n2-5-mini',
              'https://openrouter.ai/api/v1'),
    Candidate('openrouter-apodex-1.1-mini', 'openrouter',
              'apodex/apodex-1.1-mini:free', 'openai-chat-completions',
              'OPENROUTER_API_KEY', 'q1e-openrouter-apodex-1-1-mini',
              'https://openrouter.ai/api/v1'),
)


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _candidates(candidates):
    candidates = tuple(candidates)
    if (not candidates or any(c not in OPENROUTER_CANDIDATES + Q1E_CANDIDATES for c in candidates)
            or len({c.candidate_id for c in candidates}) != len(candidates)):
        raise ValueError('Use distinct exact admitted OpenRouter candidates.')
    return candidates


def discover_models(client, *, candidates=OPENROUTER_CANDIDATES,
                    secrets=(), sensitive_values=()):
    """One unfiltered catalog request; exact IDs only, never aliases or variants."""
    candidates = _candidates(candidates)
    redact = _redactor(secrets, sensitive_values)
    report = dict(catalog_requests=1, candidates=[])
    try:
        response = client.models.list(timeout=60.0)
        cards = _field(response, 'data')
        if (not isinstance(cards, (list, tuple)) or len(cards) > 4096
                or _field(response, 'has_more', False)
                or any(type(_field(card, 'id')) is not str for card in cards)):
            raise ValueError('Invalid or incomplete bounded model catalog.')
        for candidate in candidates:
            if sum(_field(card, 'id') == candidate.model_id for card in cards) > 1:
                raise ValueError('Duplicate exact model catalog identity.')
    except Exception as exc:
        diagnostic = sanitize_provider_error(exc, secrets=secrets, sensitive_values=sensitive_values)
        code, _ = classify_provider_exception(exc)
        for candidate in candidates:
            availability = Availability('operational_failure', code, candidate.model_id,
                                        'models.list', http_status=diagnostic['http_status'])
            report['candidates'].append(dict(candidate=candidate.manifest(),
                discovery_result='DISCOVERY_OPERATIONAL_FAILURE', availability=availability.to_dict(),
                supported_parameters=None, pricing={}, diagnosis=diagnostic))
        return report
    for candidate in candidates:
        matches = [card for card in cards if _field(card, 'id') == candidate.model_id]
        card = matches[0] if matches else None  # uniqueness checked above
        available = card is not None
        availability = Availability('available' if available else 'unavailable',
            'CANDIDATE_MODEL_AVAILABLE' if available else 'CANDIDATE_MODEL_UNAVAILABLE',
            candidate.model_id, 'models.list', candidate.model_id if available else None)
        parameters = _field(card, 'supported_parameters')
        safe_parameters = (list(parameters) if isinstance(parameters, (list, tuple))
            and len(parameters) <= 64 and all(type(p) is str and re.fullmatch(r'[a-z_]{1,64}', p)
                                            for p in parameters) else None)
        if safe_parameters is not None:
            safe_parameters = [p for p in safe_parameters if redact(p, 64) == p]
        prices = {}
        for key in ('prompt', 'completion', 'request'):
            value = _field(_field(card, 'pricing'), key)
            if (type(value) is str and len(value) <= 32
                    and re.fullmatch(r'\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', value)
                    and redact(value, 32) == value):
                prices[key] = value
        report['candidates'].append(dict(candidate=candidate.manifest(),
            discovery_result='EXACT_MODEL_AVAILABLE' if available else 'EXACT_MODEL_UNAVAILABLE',
            availability=availability.to_dict(), supported_parameters=safe_parameters,
            pricing=prices, diagnosis=None))
    return report


def build_openrouter_candidate(candidate, *, availability, client=None, environment=None,
                              secrets=(), sensitive_values=()):
    """Inject only routing strictness and bounded diagnostics into generic transport."""
    _candidates((candidate,))
    if (not isinstance(availability, Availability)
            or availability.requested_model_id != candidate.model_id
            or (availability.status == 'available' and availability.provider_model_id != candidate.model_id)):
        raise ValueError('Exact catalog evidence must match the requested candidate.')
    env = os.environ if environment is None else environment
    runtime = build_candidate(candidate, client=client, environment=env)
    raw_client = runtime._client
    original = raw_client.chat.completions.create
    known_secrets = tuple(secrets) + (env.get('OPENROUTER_API_KEY', ''),)
    requests = 0
    runtime.transport_diagnostic = None
    runtime.routing_observation = None

    def create(**kwargs):
        nonlocal requests
        if requests:
            raise PlanningModelError('Qualification completion limit exceeded.',
                                     code='PROVIDER_COMPLETION_LIMIT_EXCEEDED')
        requests += 1
        # The shared adapter owns prompt/schema. The transport owns this API-only option.
        kwargs['extra_body'] = {
            'provider': {'require_parameters': True, 'allow_fallbacks': False,
                         'max_price': {'prompt': 0, 'completion': 0, 'request': 0}},
            'plugins': [{'id': name, 'enabled': False} for name in
                        ('response-healing', 'context-compression', 'web')],
        }
        kwargs['extra_headers'] = {'X-OpenRouter-Metadata': 'enabled'}
        try:
            response = original(**kwargs)
        except Exception as exc:
            runtime.transport_diagnostic = sanitize_provider_error(
                exc, secrets=known_secrets, sensitive_values=sensitive_values)
            raise
        metadata = _field(response, 'openrouter_metadata')
        if isinstance(metadata, Mapping):
            # Record only booleans, bounded counts and documented enum values.
            pipeline = _field(metadata, 'pipeline', [])
            modified = bool(pipeline) if isinstance(pipeline, (list, tuple)) else True
            requested = _field(metadata, 'requested')
            attempt = _field(metadata, 'attempt')
            runtime.routing_observation = dict(
                requested_model_matches=requested == candidate.model_id if requested is not None else None,
                pipeline_modified=modified,
                upstream_attempt=attempt if type(attempt) is int and 0 <= attempt <= 32 else None)
        return response

    def close():
        if client is None:
            raw_client.close()

    facade = SimpleNamespace(models=raw_client.models,
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=close)
    runtime._client = facade
    runtime._model._client = facade
    runtime.availability = availability
    return runtime


def classify_readiness(row, discovery, diagnosis):
    """Keep semantic/admission outcomes distinct from transport admission readiness."""
    if discovery['discovery_result'] == 'EXACT_MODEL_UNAVAILABLE':
        return dict(classification='EXACT_MODEL_UNAVAILABLE', failure_category=None)
    diagnostic = diagnosis or discovery.get('diagnosis') or {}
    status = diagnostic.get('http_status')
    configuration_code = (diagnostic.get('provider_error') or {}).get('code')
    external_configuration = status in (401, 403) or configuration_code in {
        'PLANNING_PROVIDER_CONFIGURATION_FAILED', 'PLANNING_PROVIDER_DEPENDENCY_MISSING',
        'CANDIDATE_ENDPOINT_MISSING'} or row.get('configuration_error_code') in {
        'PLANNING_PROVIDER_CONFIGURATION_FAILED', 'PLANNING_PROVIDER_DEPENDENCY_MISSING',
        'CANDIDATE_ENDPOINT_MISSING'}
    if discovery['discovery_result'] == 'DISCOVERY_OPERATIONAL_FAILURE':
        return dict(classification='EXTERNAL_CONFIGURATION_REQUIRED' if external_configuration
                    else 'OPERATIONALLY_BLOCKED', failure_category='operational')
    attempt = row.get('attempt')
    if attempt and attempt['contract_success'] is True:
        return dict(classification='READY_FOR_Q2', failure_category=attempt['failure_category'])
    if attempt and attempt['contract_success'] is False:
        return dict(classification='CONTRACT_INCOMPATIBLE', failure_category='contract')
    error = diagnostic.get('provider_error', {})
    message = (error.get('message') or '').lower()
    explicit_schema_rejection = (
        status in (400, 422) and any(term in message for term in (
            'json_schema', 'json schema', 'response_format', 'structured output', 'text.format'))
        or status == 404 and 'no endpoints found that support' in message
            and 'parameters' in message)
    if explicit_schema_rejection:
        return dict(classification='CONTRACT_INCOMPATIBLE', failure_category='contract')
    return dict(classification='EXTERNAL_CONFIGURATION_REQUIRED' if external_configuration
                else 'OPERATIONALLY_BLOCKED', failure_category='operational')
