"""F1's explicit admission gate and four-case screen; no breadth or repeats.

Agent owns every prompt, schema, parser, admission and planning recovery. This
runner adds only an exact candidate, paid transport limits and a spending ledger.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

from agent.orchestration.planning_scope import PlanningScope, fingerprint
from benchmarks.planner.benchmark import guarded_registry

from . import harness, run_breadth as breadth
from .candidates import Availability, Candidate, build_candidate
from .diagnostics import sanitize_provider_error
from .openrouter import OpenRouterPolicyViolation, build_openrouter_candidate
from .run_stability import _frozen_context, _interaction
from .scenarios import canonical_scenarios, scenario_by_id


BASELINE = '615697a0ecbea1d0b47ff45a292588167370c60b'
SELECTED = ('I01', 'I10', 'I08', 'I14')
SCHEDULE = ('admission', *SELECTED)
CANDIDATE = Candidate('openrouter-gpt-5.6-sol', 'openrouter', 'openai/gpt-5.6-sol',
    'openai-chat-completions', 'OPENROUTER_API_KEY', 'f1-openrouter-gpt-5-6-sol',
    'https://openrouter.ai/api/v1')
HISTORY = Path('evals/interactive_model_qualification_q2_1b_2026-10-07/merged_results.json')
HISTORICAL_AUDIT = Path('evals/interactive_model_qualification_q2_2_2026-10-07/frozen_input_audit.json')
PROMPT_PRICE = Decimal('0.000002')
COMPLETION_PRICE = Decimal('0.00001')
OUTPUT_LIMIT = 4096
INPUT_BOUND = 30000
SPENDING_CEILING = Decimal('1.00')
CALL_CEILING = 9


class FrontierHardStop(BaseException):
    """A safety/configuration stop must bypass Agent's production recovery."""


def validate_matrix(schedule=SCHEDULE):
    if tuple(schedule) != SCHEDULE:
        raise ValueError('F1 permits exactly one admission and I01/I10/I08/I14 once, in that order.')
    return dict(admission_smokes=1, semantic_attempts=4, expected_completions=7,
                conservative_completion_ceiling=CALL_CEILING, schedule=list(SCHEDULE))


def baseline(*, require_clean=False):
    checkout = breadth.repository_baseline(require_clean=require_clean, experiment_baseline=BASELINE)
    refs = {name: breadth._git('rev-parse', name) for name in ('HEAD', 'main', 'origin/main')}
    branch = breadth._git('branch', '--show-current')
    changes = breadth._git('diff', '--name-only').splitlines()
    if branch != 'main' or set(refs.values()) != {checkout['infrastructure_commit']}:
        raise ValueError('Accepted main and aligned current infrastructure references are required.')
    return dict(branch=branch, refs=refs, **checkout, tracked_changes=changes,
                semantic_contract_files_unchanged=True,
                infrastructure_policy='Benchmark transport/runners/docs/tests; semantic owners remain frozen.')


def smoke_scenario():
    return replace(scenario_by_id('I01'), surface='interpreter',
                   expected={'kind': 'execute', 'tool': 'inspect_scATAC'})


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


def frozen_validation():
    checkout = baseline()
    assert fingerprint([s.to_dict() for s in canonical_scenarios()]) == breadth.SCENARIOS_FINGERPRINT
    saved = json.loads(HISTORY.read_text())
    prior = json.loads(HISTORICAL_AUDIT.read_text())
    historical_contexts = prior['plan']['frozen_contexts']
    registry, _ = guarded_registry()
    contexts, payloads = {}, []
    for sid in SELECTED:
        scenario = scenario_by_id(sid)
        context = _frozen_context(scenario)
        if context != historical_contexts[sid]:
            raise ValueError('Reconstructed historical context changed.')
        rows = [r for r in saved['results'] if r['scenario_id'] == sid
                and r['candidate']['candidate_id'] == breadth.COHORT[0].candidate_id]
        if len(rows) != 1:
            raise ValueError('Historical comparator identity is ambiguous.')
        attempt = rows[0]['attempt']
        for key, wanted in (('scenario_fingerprint', context['scenario_fingerprint']),
                            ('context_fingerprint', context['public_context_fingerprint']),
                            ('request_fingerprint', context['request_fingerprint']),
                            ('registry_fingerprint', context['registry_fingerprint'])):
            if attempt[key] != wanted:
                raise ValueError('Frozen historical semantic input changed.')
        probes = breadth._probe_contracts(scenario)
        for call in attempt['calls']:
            if call['recovery_kind'] != 'initial':
                continue
            if sid == 'I01' and call['surface'] == 'stage_b':
                selected = next(c['capability_ids'] for c in attempt['calls'] if c['surface'] == 'stage_a')
                pair = breadth._layer2_contract(scenario, registry,
                    PlanningScope(registry, tuple(selected)).visible_tool_names)
            else:
                pair = probes[call['surface']]
            if breadth._pair(*pair) != (call['prompt_fingerprint'], call['schema_fingerprint']):
                raise ValueError('Frozen historical prompt/schema changed.')
            payloads.append(dict(scenario_id=sid, surface=call['surface'],
                prompt_fingerprint=fingerprint(pair[0]), schema_fingerprint=fingerprint(pair[1]),
                prompt_utf8_bytes=len(pair[0].encode()),
                schema_utf8_bytes=len(json.dumps(pair[1], ensure_ascii=False, separators=(',', ':')).encode())))
        contexts[sid] = context
    smoke = smoke_scenario()
    smoke_pair = breadth._probe_contracts(smoke)['interpreter']
    smoke_manifest = dict(scenario_fingerprint=smoke.fingerprint(),
        prompt_fingerprint=fingerprint(smoke_pair[0]), schema_fingerprint=fingerprint(smoke_pair[1]),
        prompt_utf8_bytes=len(smoke_pair[0].encode()),
        schema_utf8_bytes=len(json.dumps(smoke_pair[1], ensure_ascii=False, separators=(',', ':')).encode()))
    sizes = [dict(prompt_utf8_bytes=smoke_manifest['prompt_utf8_bytes'],
                  schema_utf8_bytes=smoke_manifest['schema_utf8_bytes']), *payloads]
    estimate = sum((PROMPT_PRICE * (p['prompt_utf8_bytes'] + p['schema_utf8_bytes'] + 2304)
                    + COMPLETION_PRICE * OUTPUT_LIMIT for p in sizes), Decimal(0))
    return dict(baseline=checkout, candidate=CANDIDATE.manifest(), matrix=validate_matrix(),
        scenarios_fingerprint=breadth.SCENARIOS_FINGERPRINT, frozen_contexts=contexts,
        historical_initial_payload_checks=payloads, smoke=smoke_manifest,
        historical_ledger_sha256=hashlib.sha256(HISTORY.read_bytes()).hexdigest(),
        historical_audit_sha256=hashlib.sha256(HISTORICAL_AUDIT.read_bytes()).hexdigest(),
        execution_source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__), Path('benchmarks/interactive/openrouter.py'))},
        zero_science_guards_validated=True, provider_calls=0,
        cost_estimate=dict(expected_seven_call_output_ceiling_estimate_usd=str(estimate),
            hard_nine_call_upper_bound_usd=str(CALL_CEILING * (PROMPT_PRICE * INPUT_BOUND
                + COMPLETION_PRICE * OUTPUT_LIMIT)), spending_ceiling_usd=str(SPENDING_CEILING),
            estimates_use_conservative_utf8_byte_input_bound=True))


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


def run(*, output, live=False):
    output = Path(output)
    if live:
        baseline(require_clean=True)
    discovery = json.loads((output / 'discovery.json').read_text())
    validate_discovery(discovery)
    validation = frozen_validation()
    private = breadth._private_values(os.environ)
    def save(name, value):
        (output / name).write_text(json.dumps(breadth._safe_json(value, private),
            indent=2, allow_nan=False, ensure_ascii=False) + '\n')
    if not live:
        if (output / 'offline_validation.json').exists():
            raise ValueError('Do not overwrite previous F1 dry validation.')
        save('offline_validation.json', validation)
        print(json.dumps({'phase': 'F1_DRY_VALIDATED', 'matrix': validation['matrix'],
                          'cost_estimate': validation['cost_estimate']}, indent=2))
        return validation
    if (output / 'results.json').exists():
        raise ValueError('F1 live runs cannot repeat or resume existing evidence.')
    recorded = json.loads((output / 'offline_validation.json').read_text())
    if recorded != validation:
        raise ValueError('Offline validation changed before paid dispatch.')
    if not os.environ.get('OPENROUTER_API_KEY', '').strip():
        raise ValueError('Activate the existing agent Conda environment before F1.')
    availability = Availability('available', 'CANDIDATE_MODEL_AVAILABLE', CANDIDATE.model_id,
                                'models.list', CANDIDATE.model_id)
    ledger = SpendingGuard()
    report = dict(phase='F1', candidate=CANDIDATE.manifest(), matrix=validate_matrix(),
        discovery=discovery, offline_validation_fingerprint=fingerprint(validation),
        credential_present_in_agent_environment=True, results=[], human_review_queue=[],
        hard_stop=None, admission_gate_passed=False, semantic_attempts=0)
    for case in SCHEDULE:
        if case != 'admission' and not report['admission_gate_passed']:
            break
        ledger.begin(case)
        scenario = smoke_scenario() if case == 'admission' else scenario_by_id(case)
        checks = []
        runtime = None
        row = dict(case=case, scenario_id=scenario.scenario_id, attempt_id='f1-admission-1' if case == 'admission' else 'f1-1',
                   attempt=None, diagnostics=[], payload_checks=checks, status=None)
        original_context = harness.build_context
        def context_build(*args, **kwargs):
            context = original_context(*args, **kwargs)
            if case != 'admission':
                expected = validation['frozen_contexts'][case]
                if (fingerprint(context.public) != expected['public_context_fingerprint']
                        or fingerprint(_interaction(context)) != expected['private_captured_context_fingerprint']):
                    raise FrontierHardStop('Captured live fixture context changed.')
            return context
        try:
            runtime = FrontierTransport(availability=availability, ledger=ledger, case=case,
                before_complete=breadth._payload_guard(scenario, checks), environment=os.environ)
            row['diagnostics'] = runtime.diagnostics
            with patch.object(harness, 'build_context', context_build):
                attempt = harness.run_attempt(scenario, CANDIDATE, model=runtime, attempt_id=row['attempt_id']).to_dict()
            row['attempt'] = attempt
            if any(attempt['safety'].values()):
                raise harness.ZeroScienceViolation('F1 entered forbidden science.')
            row['status'] = breadth.status_for_attempt(attempt)
            if case == 'admission':
                report['admission_gate_passed'] = all(attempt.get(k) is True for k in
                    ('contract_success', 'semantic_success', 'admission_success')) and row['status'] == 'PASS'
                if not report['admission_gate_passed']:
                    report['hard_stop'] = dict(case=case, reason='ADMISSION_SMOKE_FAILED', category=attempt['failure_category'])
            else:
                report['semantic_attempts'] += 1
                if row['status'] == 'HUMAN_REVIEW_PENDING':
                    report['human_review_queue'].append(dict(scenario_id=case, status=row['status'],
                        explanation=attempt.get('explanation'), support=attempt.get('support'),
                        human_semantic_review=None, fixture_provenance=attempt['fixture_provenance']))
            # No benchmark retry follows an operational result. Stop independent
            # work only for configuration/balance failures or repeated failures.
            # I01's existing production recovery remains inside run_attempt.
            failed = [d for d in runtime.diagnostics if d['provider_outcome'] == 'failed']
            stop_status = next((d['diagnostic'].get('http_status') for d in failed
                if d.get('diagnostic') and d['diagnostic'].get('http_status') in (401, 402, 403)), None)
            previous_failed = sum(d['provider_outcome'] == 'failed' for r in report['results']
                                  for d in r.get('diagnostics', ()))
            if stop_status is not None or (failed and previous_failed + len(failed) >= 2):
                report['hard_stop'] = dict(case=case, reason='CONFIGURATION_BALANCE_OR_REPEATED_PROVIDER_FAILURE',
                                          category='operational', http_status=stop_status)
        except (FrontierHardStop, OpenRouterPolicyViolation) as exc:
            row['status'] = 'HARD_STOP'
            report['hard_stop'] = dict(case=case, reason=str(exc), category='configuration_or_exact_model_or_cost_guard')
        except harness.ZeroScienceViolation:
            row['status'] = 'SAFETY_FAIL'
            report['hard_stop'] = dict(case=case, reason='ZERO_SCIENCE_GUARD_FIRED', category='scientific_safety')
        except Exception as exc:
            row['status'] = 'HARD_STOP'
            report['hard_stop'] = dict(case=case, reason='LOCAL_CONFIGURATION_OR_RUNNER_FAILURE',
                category='configuration', diagnosis=sanitize_provider_error(exc, secrets=private))
        finally:
            if runtime is not None:
                runtime.close()
            report['results'].append(row)
            report['spending_guard'] = ledger.manifest()
            report['actual_usage'] = _usage(report['results'])
            report['counts'] = dict(statuses=dict(Counter(r.get('status') for r in report['results'])),
                provider_calls=len(ledger.requests), semantic_attempts=report['semantic_attempts'])
            save('results.json', report)
            save('token_cost_accounting.json', dict(policy=ledger.manifest(), actual_usage=report['actual_usage']))
            save('human_review_queue.json', report['human_review_queue'])
        print(json.dumps(dict(case=case, status=row['status'], provider_calls=len(row['diagnostics']),
                             aggregate_calls=len(ledger.requests))), flush=True)
        if report['hard_stop']:
            break
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    gate = parser.add_mutually_exclusive_group(required=True)
    gate.add_argument('--dry-run', action='store_true')
    gate.add_argument('--live', action='store_true')
    args = parser.parse_args()
    run(output=args.output, live=args.live)


if __name__ == '__main__':
    main()
