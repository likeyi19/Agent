"""Reusable bounded paid transport and cost reservations; F1 execution retired.

The retained default budget profile describes historical F1 constraints. It is
not the current Agent's planning or answer call budget. No experiment is run here.
"""
from decimal import Decimal
import json
from . import run_breadth as breadth
from .candidates import Candidate, build_candidate
from .diagnostics import sanitize_provider_error
from .openrouter import build_openrouter_candidate

BASELINE = '615697a0ecbea1d0b47ff45a292588167370c60b'
SELECTED = ('I01', 'I10', 'I08', 'I14')
SCHEDULE = ('admission', *SELECTED)
CANDIDATE = Candidate('openrouter-gpt-5.6-sol', 'openrouter', 'openai/gpt-5.6-sol',
    'openai-chat-completions', 'OPENROUTER_API_KEY', 'f1-openrouter-gpt-5-6-sol',
    'https://openrouter.ai/api/v1')
PROMPT_PRICE = Decimal('0.000002')
COMPLETION_PRICE = Decimal('0.00001')
OUTPUT_LIMIT = 4096
INPUT_BOUND = 30000
SPENDING_CEILING = Decimal('1.00')
CALL_CEILING = 9


class FrontierHardStop(BaseException):
    """A safety/configuration stop must bypass Agent's production recovery."""

def validate_discovery(discovery):
    if (discovery.get('discovery_result') != 'EXACT_MODEL_AVAILABLE'
            or discovery.get('candidate') != CANDIDATE.manifest()
            or discovery.get('matched_exact_model_id') != CANDIDATE.model_id
            or discovery.get('exact_match_count') != 1):
        raise ValueError('The exact frontier candidate was not discovered.')
    parameters = discovery.get('supported_parameters') or ()
    if not {'response_format', 'structured_outputs', 'max_completion_tokens'} <= set(parameters):
        raise ValueError('Required strict-output and bounded-output metadata is missing.')
    prices = discovery.get('pricing', {})
    if (Decimal(prices.get('prompt', '-1')) != PROMPT_PRICE
            or Decimal(prices.get('completion', '-1')) != COMPLETION_PRICE
            or Decimal(prices.get('request', '0')) != 0):
        raise ValueError('Live pricing differs from the admitted F1 prices.')

def input_token_bound(request):
    # UTF-8 bytes bound byte-tokenizer input conservatively. Include the separately
    # submitted JSON schema and 2,048 tokens for API/message/schema framing.
    content = json.dumps({'messages': request['messages'],
                          'response_format': request['response_format']},
                         ensure_ascii=False, separators=(',', ':'))
    return len(content.encode()) + 2048

class SpendingGuard:
    """Reserve worst-case cost before dispatch, never reclaim it for extra work."""

    def __init__(self, *, ceiling=SPENDING_CEILING):
        self.ceiling = Decimal(ceiling)
        if not self.ceiling.is_finite() or not 0 < self.ceiling <= SPENDING_CEILING:
            raise ValueError('F1 spending ceiling must be positive and at most one dollar.')
        self.reserved = Decimal(0)
        self.requests = []
        self.started = []

    def begin(self, case):
        if len(self.started) >= len(SCHEDULE) or case != SCHEDULE[len(self.started)]:
            raise FrontierHardStop('F1 scheduling exceeds the exact admitted run matrix.')
        self.started.append(case)

    def reserve(self, *, request, case):
        if not self.started or self.started[-1] != case:
            raise FrontierHardStop('Unadmitted F1 attempt.')
        count = sum(r['case'] == case for r in self.requests)
        if len(self.requests) >= CALL_CEILING or count >= (5 if case == 'I01' else 1):
            raise FrontierHardStop('F1 provider-call ceiling exceeded.')
        fmt = request.get('response_format', {})
        provider = request.get('extra_body', {}).get('provider', {})
        if (request.get('model') != CANDIDATE.model_id
                or fmt.get('type') != 'json_schema'
                or fmt.get('json_schema', {}).get('strict') is not True
                or request.get('max_completion_tokens') != OUTPUT_LIMIT
                or provider.get('allow_fallbacks') is not False
                or provider.get('require_parameters') is not True
                or provider.get('max_price') != {'prompt': 2, 'completion': 10, 'request': 0}):
            raise FrontierHardStop('Exact routing, strict schema or paid limits changed.')
        bound = input_token_bound(request)
        maximum = PROMPT_PRICE * bound + COMPLETION_PRICE * OUTPUT_LIMIT
        if bound > INPUT_BOUND or maximum > Decimal('0.101') or self.reserved + maximum > self.ceiling:
            raise FrontierHardStop('F1 per-request or aggregate spending reservation exceeded.')
        self.reserved += maximum
        self.requests.append(dict(case=case, request_index=len(self.requests) + 1,
            input_token_upper_bound=bound, completion_token_upper_bound=OUTPUT_LIMIT,
            reserved_cost_usd=str(maximum), total_reserved_cost_usd=str(self.reserved)))

    def manifest(self):
        return dict(spending_ceiling_usd=str(self.ceiling), reserved_upper_bound_usd=str(self.reserved),
            requests=self.requests, prompt_price_usd_per_token=str(PROMPT_PRICE),
            completion_price_usd_per_token=str(COMPLETION_PRICE),
            per_request_price_exposed=False, positive_per_request_endpoint_price_forbidden=True,
            max_completion_tokens=OUTPUT_LIMIT, input_token_upper_bound_limit=INPUT_BOUND,
            sdk_retries=0, reasoning_configuration='provider_defaults_unspecified',
            balance_read=False, remaining_balance_estimate=None)

class FrontierTransport:
    """Fresh existing OpenRouter admission wrapper per allowed production call."""

    def __init__(self, *, availability, ledger, case, before_complete, environment, client=None):
        self.candidate = CANDIDATE
        self.owner = build_candidate(CANDIDATE, client=client, environment=environment)
        self.availability = availability
        self.ledger, self.case, self.validator = ledger, case, before_complete
        self.environment, self.diagnostics = environment, []
        self.private = breadth._private_values(environment)

    def complete(self, *, prompt, response_schema):
        try:
            self.validator(prompt=prompt, response_schema=response_schema)
        except Exception:
            raise FrontierHardStop('Frozen request validation failed before provider dispatch.') from None
        row = None
        def reserve(*, request):
            nonlocal row
            self.ledger.reserve(request=request, case=self.case)
            row = dict(request_index=len(self.diagnostics) + 1,
                provider_outcome='returned', diagnostic=None, usage=None,
                response_model=None, routing_observation=None)
            self.diagnostics.append(row)
        runtime = build_openrouter_candidate(CANDIDATE, availability=self.availability,
            admitted_candidates=(CANDIDATE,), client=self.owner._client,
            environment=self.environment, secrets=self.private,
            paid_policy={'provider_max_price': {'prompt': 2, 'completion': 10, 'request': 0},
                         'max_completion_tokens': OUTPUT_LIMIT}, before_dispatch=reserve)
        try:
            return runtime.complete(prompt=prompt, response_schema=response_schema)
        except Exception as exc:
            if row is not None:
                row.update(provider_outcome='failed', diagnostic=runtime.transport_diagnostic
                    or sanitize_provider_error(exc, secrets=self.private))
            raise
        finally:
            if row is not None:
                row.update(usage=runtime.usage_observation,
                    response_model=runtime.response_model_observation,
                    routing_observation=runtime.routing_observation)
                model = runtime.response_model_observation
                routing = runtime.routing_observation
                if model is not None and model.get('matches_requested') is not True:
                    raise FrontierHardStop('Provider returned a different model identity.')
                if routing and (routing['requested_model_matches'] is False or routing['pipeline_modified']
                                or (routing['upstream_attempt'] or 0) > 1):
                    raise FrontierHardStop('Observed routing violated the admitted exact-model policy.')
                usage = runtime.usage_observation or {}
                reservation = self.ledger.requests[-1]
                if (usage.get('prompt_tokens', 0) > reservation['input_token_upper_bound']
                        or usage.get('completion_tokens', 0) > OUTPUT_LIMIT
                        or ('cost' in usage and Decimal(str(usage['cost']))
                            > Decimal(reservation['reserved_cost_usd']))):
                    raise FrontierHardStop('Provider usage or cost exceeded its reserved request bound.')

    def close(self):
        self.owner.close()

def _usage(rows):
    observations = [d['usage'] for r in rows for d in r.get('diagnostics', ()) if d.get('usage')]
    totals = {}
    for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
        values = [v[key] for v in observations if key in v]
        totals[key] = sum(values) if len(values) == len(observations) and observations else None
    costs = [Decimal(str(v['cost'])) for v in observations if 'cost' in v]
    totals['reported_cost_usd'] = str(sum(costs, Decimal(0))) if observations and len(costs) == len(observations) else None
    totals['requests_with_usage'] = len(observations)
    totals['billing_inferred'] = False
    return totals

def run(*args, **kwargs):
    raise RuntimeError("Historical qualification runner retired by R5; preserved reports remain readable. "
                       "Use the current interactive harness for new checks.")

def main(*args, **kwargs):
    raise RuntimeError("Historical qualification runner retired by R5; preserved reports remain readable. "
                       "Use the current interactive harness for new checks.")

if __name__ == '__main__':
    main()
