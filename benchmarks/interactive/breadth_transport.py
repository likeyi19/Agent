"""Bounded breadth transport over the frozen, already admitted model cohort.

Only SDK requests are observed here. Agent owns prompts, schemas, parsing and
production recovery. OpenRouter's accepted one-call admission wrapper is fresh
for each completion, allowing the existing Layer-2 recovery path to remain
visible within an explicit attempt budget.
"""
from __future__ import annotations

import os
import re
import time
from types import SimpleNamespace

from agent.orchestration.planning_model import (
    PlanningModelError, normalized_retry_after_seconds,
)

from .candidates import Availability, CANDIDATES, build_candidate
from .diagnostics import _redactor, sanitize_provider_error
from .openrouter import OPENROUTER_CANDIDATES, build_openrouter_candidate


COHORT = (CANDIDATES[0], CANDIDATES[1], OPENROUTER_CANDIDATES[0])
_ORGANIZATION_TOKEN = re.compile(r'\borg_[A-Za-z0-9_-]+\b', re.IGNORECASE)


class PayloadValidationFailure(BaseException):
    """Frozen request validation failed before any provider invocation."""


def _reporting_error(value):
    """Redact operator tokens in persisted diagnostics/prose after model scoring."""
    if isinstance(value, str):
        return _ORGANIZATION_TOKEN.sub('[REDACTED ORGANIZATION]', value)
    if isinstance(value, dict):
        return {_reporting_error(key): _reporting_error(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_reporting_error(item) for item in value]
    return value


class BreadthTransport:
    """One independent model instance with cached exact discovery evidence."""

    def __init__(self, candidate, *, availability, client=None, environment=None,
                 max_calls=5, before_complete=None):
        if candidate not in COHORT:
            raise ValueError('Use an exact frozen breadth candidate.')
        if type(max_calls) is not int or not 1 <= max_calls <= 5:
            raise ValueError('The attempt completion budget must be between one and five.')
        if before_complete is not None and not callable(before_complete):
            raise TypeError('The pre-completion validator must be callable or None.')
        if (not isinstance(availability, Availability)
                or availability.status != 'available'
                or availability.requested_model_id != candidate.model_id
                or availability.provider_model_id != candidate.model_id):
            raise PlanningModelError('Exact breadth candidate availability was not accepted.',
                                     code='CANDIDATE_MODEL_NOT_VALIDATED')
        self._environment = dict(os.environ if environment is None else environment)
        self._secrets = tuple(value for key, value in self._environment.items()
                              if key.endswith('_API_KEY') and value)
        self._sensitive = tuple(value for key, value in self._environment.items()
            if value and any(label in key.upper() for label in (
                'WORKSPACE_ID', 'WORK_SPACE_ID', 'PROJECT_ID', 'ACCOUNT_ID', 'CONSUMER')))
        self._redact = _redactor(self._secrets, self._sensitive)
        self._runtime = build_candidate(candidate, client=client, environment=self._environment)
        self.candidate = self._runtime.candidate
        self._definition = candidate
        self._client = self._runtime._client
        self._owns_client = client is None
        self._closed = False
        self.availability = availability
        self.max_calls = max_calls
        self._before_complete = before_complete
        self.availability_requests = 0
        self.completion_calls = 0
        self.diagnostics = []
        self._runtime.availability = availability
        if candidate.adapter_family == 'groq-responses':
            self._create = self._client.responses.create
            self._observed = SimpleNamespace(models=self._client.models,
                responses=SimpleNamespace(create=self._observe))
            self._runtime._model._client = self._observed
        else:
            self._create = self._client.chat.completions.create
            self._observed = SimpleNamespace(models=self._client.models,
                chat=SimpleNamespace(completions=SimpleNamespace(create=self._observe)))

    @property
    def model_id(self):
        return self.candidate.provider_id + ':' + self.candidate.model_id

    def profile(self):
        return self.candidate.profile()

    def manifest(self):
        return self.candidate.manifest()

    def validate_availability(self):
        return self.availability

    def _observe(self, **kwargs):
        if self.completion_calls >= self.max_calls:
            raise PlanningModelError('Breadth attempt completion budget exceeded.',
                                     code='PROVIDER_COMPLETION_LIMIT_EXCEEDED')
        self.completion_calls += 1
        row = dict(request_index=self.completion_calls, provider_outcome='returned',
                   http_status=None, diagnostic=None, retry_after_seconds=None,
                   routing_observation=None, request_started_monotonic=time.monotonic(),
                   request_finished_monotonic=None)
        self.diagnostics.append(row)
        try:
            return self._create(**kwargs)
        except Exception as exc:
            diagnostic = _reporting_error(sanitize_provider_error(
                exc, secrets=self._secrets, sensitive_values=self._sensitive))
            retry_after = normalized_retry_after_seconds(exc)
            if retry_after is not None and self._redact(str(retry_after), 128) != str(retry_after):
                retry_after = None
            row.update(provider_outcome='failed', http_status=diagnostic['http_status'],
                       diagnostic=diagnostic, retry_after_seconds=retry_after)
            raise
        finally:
            row['request_finished_monotonic'] = time.monotonic()

    def complete(self, *, prompt, response_schema):
        if self._closed:
            raise PlanningModelError('Breadth transport is closed.',
                                     code='PLANNING_PROVIDER_CONFIGURATION_FAILED')
        if self.completion_calls >= self.max_calls:
            raise PlanningModelError('Breadth attempt completion budget exceeded.',
                                     code='PROVIDER_COMPLETION_LIMIT_EXCEEDED')
        if self._before_complete is not None:
            try:
                self._before_complete(prompt=prompt, response_schema=response_schema)
            except Exception:
                # A mismatch is a benchmark hard stop, never a recoverable SDK error.
                raise PayloadValidationFailure('Frozen request validation failed before provider invocation.') from None
        if self.candidate.adapter_family == 'groq-responses':
            return self._runtime.complete(prompt=prompt, response_schema=response_schema)
        runtime = build_openrouter_candidate(self._definition,
            availability=self.availability, client=self._observed,
            environment=self._environment, secrets=self._secrets, sensitive_values=self._sensitive)
        previous_requests = len(self.diagnostics)
        try:
            response = runtime.complete(prompt=prompt, response_schema=response_schema)
        finally:
            if len(self.diagnostics) > previous_requests:
                self.diagnostics[-1]['routing_observation'] = runtime.routing_observation
        routing = runtime.routing_observation
        if routing and (routing['pipeline_modified'] or routing['requested_model_matches'] is False
                        or (routing['upstream_attempt'] or 0) > 1):
            raise PlanningModelError('Observed provider routing violated the frozen transport.',
                                     code='PLANNING_PROVIDER_CONFIGURATION_FAILED')
        return response

    def close(self):
        if not self._closed:
            self._closed = True
            if self._owns_client:
                self._runtime.close()


def build_transport(candidate, *, availability, client=None, environment=None, max_calls=5,
                    before_complete=None):
    """Construct without requests; discovery belongs to the breadth runner."""
    return BreadthTransport(candidate, availability=availability, client=client,
                            environment=environment, max_calls=max_calls,
                            before_complete=before_complete)
