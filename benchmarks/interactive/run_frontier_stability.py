"""F2: exactly eight independent repeats of F1's admitted paid candidate.

All scientific, language, parsing and recovery contracts remain Agent-owned.
The runner preserves F1 bytes and adds only repeat scheduling and a shared
conservative spending ledger. New scientific prose always awaits human review.
"""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import time
from unittest.mock import patch

from agent.orchestration.planning_scope import PlanningScope, fingerprint
from agent.orchestration.semantic_wire_v4 import parse_semantic_wire_v4
from benchmarks.planner.benchmark import guarded_registry

from . import harness, run_breadth as breadth, run_frontier as frontier
from .candidates import Availability
from .diagnostics import sanitize_provider_error
from .openrouter import OpenRouterPolicyViolation
from .run_stability import _Observed, _interaction
from .scenarios import scenario_by_id


CANDIDATE = frontier.CANDIDATE
SELECTED = frontier.SELECTED
SCHEDULE = tuple((sid, number) for sid in SELECTED for number in (2, 3))
COMPLETION_CEILING = 16
F1_DIRECTORY = Path('evals/frontier_comparator_f1_gpt56sol_2026-10-07')
F1_SOURCES = (Path('benchmarks/interactive/run_frontier.py'),
              Path('benchmarks/interactive/openrouter.py'),
              Path('tests/benchmarks/test_interactive_frontier_transport.py'))
FrontierHardStop = frontier.FrontierHardStop


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_matrix(schedule=SCHEDULE):
    if tuple(schedule) != SCHEDULE:
        raise ValueError('F2 permits exactly I01/I10/I08/I14, attempts 2 and 3, in that order.')
    return dict(candidate=CANDIDATE.manifest(), selected_scenarios=list(SELECTED),
        admission_smokes=0, historical_attempts_per_case=1, new_attempts_per_case=2,
        total_new_attempts=8, expected_direct_completions=12,
        conservative_completion_ceiling=COMPLETION_CEILING,
        planned_attempts=[dict(scenario_id=sid, attempt_number=n, attempt_id=f'f2-{n}',
            candidate_id=CANDIDATE.candidate_id) for sid, n in SCHEDULE],
        other_candidate_calls=0, full_breadth=False)


class SpendingGuard(frontier.SpendingGuard):
    """Separate attempt reservations with F1's exact transport configuration."""

    def begin(self, case, attempt_number):
        if (len(self.started) >= len(SCHEDULE)
                or (case, attempt_number) != SCHEDULE[len(self.started)]):
            raise FrontierHardStop('F2 scheduling exceeds the exact eight-attempt matrix.')
        self.started.append((case, attempt_number))

    def reserve(self, *, request, case):
        if not self.started or self.started[-1][0] != case:
            raise FrontierHardStop('Unadmitted F2 attempt.')
        number = self.started[-1][1]
        count = sum(r['case'] == case and r['attempt_number'] == number for r in self.requests)
        if len(self.requests) >= COMPLETION_CEILING or count >= (5 if case == 'I01' else 1):
            raise FrontierHardStop('F2 provider-call ceiling exceeded.')
        fmt = request.get('response_format', {})
        extra = request.get('extra_body', {})
        provider = extra.get('provider', {})
        if (request.get('model') != CANDIDATE.model_id
                or fmt.get('type') != 'json_schema'
                or fmt.get('json_schema', {}).get('strict') is not True
                or request.get('max_completion_tokens') != frontier.OUTPUT_LIMIT
                or provider.get('allow_fallbacks') is not False
                or provider.get('require_parameters') is not True
                or provider.get('max_price') != {'prompt': 2, 'completion': 10, 'request': 0}
                or any(key in request or key in extra for key in
                    ('reasoning', 'reasoning_effort', 'temperature', 'top_p'))):
            raise FrontierHardStop('F1 exact routing, strict schema or sampling defaults changed.')
        bound = frontier.input_token_bound(request)
        maximum = frontier.PROMPT_PRICE * bound + frontier.COMPLETION_PRICE * frontier.OUTPUT_LIMIT
        if (bound > frontier.INPUT_BOUND or maximum > Decimal('0.101')
                or self.reserved + maximum > self.ceiling):
            raise FrontierHardStop('F2 per-request or aggregate spending reservation exceeded.')
        self.reserved += maximum
        self.requests.append(dict(case=case, attempt_number=number,
            request_index=len(self.requests) + 1, input_token_upper_bound=bound,
            completion_token_upper_bound=frontier.OUTPUT_LIMIT, reserved_cost_usd=str(maximum),
            total_reserved_cost_usd=str(self.reserved)))

    def manifest(self):
        return dict(super().manifest(), phase='F2', completion_ceiling=COMPLETION_CEILING,
            started_attempts=[dict(case=sid, attempt_number=n) for sid, n in self.started],
            reservation_reclaimed=False, hard_aggregate_ceiling_may_stop_production_recovery=True)


def validate_f1(original=F1_DIRECTORY):
    """Check the entire retained evidence set without writing any F1 artifact."""
    directory = Path(original)
    manifest_path = directory / 'evidence_manifest.json'
    hashes = json.loads(manifest_path.read_text()).get('sha256')
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError('F1 needs its complete retained evidence manifest.')
    required = {'results.json', 'offline_validation.json', 'discovery.json',
                'frozen_input_audit.json', 'closeout.json', 'independent_audit.json'}
    if not required <= set(hashes):
        raise ValueError('F1 evidence manifest is incomplete.')
    actual_files = {p.name for p in directory.iterdir() if p.is_file() and p.name != '.gitignore'}
    if actual_files != set(hashes) | {'evidence_manifest.json'}:
        raise ValueError('F1 retained evidence file set changed.')
    for name, digest in hashes.items():
        if Path(name).name != name or name in ('.', '..') or _sha(directory / name) != digest:
            raise ValueError('F1 retained evidence bytes changed.')
    saved = json.loads((directory / 'results.json').read_text())
    offline = json.loads((directory / 'offline_validation.json').read_text())
    frontier.validate_discovery(json.loads((directory / 'discovery.json').read_text()))
    if (saved.get('phase') != 'F1' or saved.get('candidate') != CANDIDATE.manifest()
            or saved.get('admission_gate_passed') is not True or saved.get('hard_stop') is not None
            or saved.get('semantic_attempts') != 4
            or saved.get('counts', {}).get('provider_calls') != 7
            or [r.get('case') for r in saved.get('results', ())] != list(frontier.SCHEDULE)):
        raise ValueError('F2 requires the successful exact-candidate F1 gate and four-case screen.')
    for row in saved['results']:
        expected_status = 'HUMAN_REVIEW_PENDING' if row['case'] == 'I08' else 'PASS'
        attempt = row.get('attempt') or {}
        if (row.get('status') != expected_status or not attempt.get('contract_success')
                or attempt.get('admission_success') is not True or any(attempt.get('safety', {}).values())
                or attempt.get('candidate') != harness.candidate_identity(CANDIDATE)):
            raise ValueError('The accepted F1 attempt or zero-science identity changed.')
    if sum(len(r.get('diagnostics', ())) for r in saved['results']) != 7:
        raise ValueError('F1 completion accounting changed.')
    sources = offline.get('execution_source_sha256', {})
    expected_sources = {'benchmarks/interactive/run_frontier.py', 'benchmarks/interactive/openrouter.py'}
    source_names = {}
    if isinstance(sources, dict):
        for name in sources:
            if isinstance(name, str):
                for expected in expected_sources:
                    if name in (expected, str(Path(expected).resolve())):
                        source_names[name] = expected
    if (not isinstance(sources, dict) or len(sources) != len(expected_sources)
            or set(source_names) != set(sources) or set(source_names.values()) != expected_sources
            or any(not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest)
                   for digest in sources.values())
            or saved.get('offline_validation_fingerprint') != fingerprint(offline)):
        raise ValueError('Retained F1 execution metadata or its bound offline validation changed.')
    # Historical source hashes identify F1's original execution, not today's
    # infrastructure. Current semantic builders reproduce the retained inputs;
    # current execution source bytes are guarded independently for each repeat.
    identity = dict(directory=str(directory.resolve()), manifest_sha256=_sha(manifest_path),
                    sha256=hashes, results_sha256=_sha(directory / 'results.json'),
                    execution_source_sha256=dict(sources))
    return saved, identity


def validate_review(path, saved):
    path = Path(path)
    review = json.loads(path.read_text())
    first = next(r for r in saved['results'] if r['case'] == 'I08')
    if (review.get('scenario_id') != 'I08' or review.get('candidate_id') != CANDIDATE.candidate_id
            or review.get('external_human_semantic_review') != 'PASS'
            or review.get('historical_status') != 'HUMAN_REVIEW_PENDING'
            or ('retained_answer' in review and review['retained_answer'] != first['attempt']['explanation'])):
        raise ValueError('F1 I08 requires its separate matching external human PASS disposition.')
    return dict(path=str(path.resolve()), sha256=_sha(path), human_semantic_review='PASS',
                historical_status='HUMAN_REVIEW_PENDING', rewrites_historical_evidence=False)


def _source_identity():
    return {str(path): _sha(path) for path in (*F1_SOURCES,
        Path('benchmarks/interactive/run_breadth.py'),
        Path('benchmarks/interactive/run_stability.py'), Path(__file__))}


def stability_validate(original=F1_DIRECTORY, *, human_review_path, discovery=None):
    saved, identity = validate_f1(original)
    original_offline = json.loads((Path(original) / 'offline_validation.json').read_text())
    frozen = frontier.frozen_validation()
    review = validate_review(human_review_path, saved)
    if frozen['candidate'] != CANDIDATE.manifest():
        raise ValueError('Frozen frontier candidate changed.')
    for key in ('frozen_contexts', 'historical_ledger_sha256', 'historical_audit_sha256'):
        if frozen.get(key) != original_offline.get(key):
            raise ValueError('F1 historical or reconstructed semantic context changed.')
    if discovery is not None:
        frontier.validate_discovery(discovery)
    registry, _ = guarded_registry()
    payloads = []
    historical = []
    for sid in SELECTED:
        scenario = scenario_by_id(sid)
        row = next(r for r in saved['results'] if r['case'] == sid)
        attempt = row['attempt']
        expected = frozen['frozen_contexts'][sid]
        for key, expected_key in (('scenario_fingerprint', 'scenario_fingerprint'),
                ('context_fingerprint', 'public_context_fingerprint'),
                ('request_fingerprint', 'request_fingerprint'), ('registry_fingerprint', 'registry_fingerprint')):
            if attempt.get(key) != expected[expected_key]:
                raise ValueError('F1 exact scenario/request/context/Registry identity changed.')
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
                raise ValueError('F1 prompt/schema no longer reproduces from unchanged production builders.')
            payloads.append(dict(scenario_id=sid, surface=call['surface'],
                prompt_fingerprint=call['prompt_fingerprint'], schema_fingerprint=call['schema_fingerprint'],
                prompt_utf8_bytes=len(pair[0].encode()),
                schema_utf8_bytes=len(json.dumps(pair[1], ensure_ascii=False, separators=(',', ':')).encode())))
        historical.append(dict(scenario_id=sid, attempt_number=1, attempt_id=row['attempt_id'],
            row_fingerprint=fingerprint(row), historical_status=row['status'],
            derived_status='PASS' if sid == 'I08' else row['status'],
            external_human_semantic_review='PASS' if sid == 'I08' else None))
    # Use the unchanged historical wider I01 scope for a conservative estimate;
    # actual F2 scope is selected by Agent and validated against its builders.
    estimate_payloads = frozen['historical_initial_payload_checks']
    estimate = 2 * sum((frontier.PROMPT_PRICE * (p['prompt_utf8_bytes'] + p['schema_utf8_bytes'] + 2304)
        + frontier.COMPLETION_PRICE * frontier.OUTPUT_LIMIT for p in estimate_payloads), Decimal(0))
    unrestricted = COMPLETION_CEILING * (frontier.PROMPT_PRICE * frontier.INPUT_BOUND
        + frontier.COMPLETION_PRICE * frontier.OUTPUT_LIMIT)
    return dict(phase='F2', baseline=frontier.BASELINE, git_baseline=frontier.baseline(),
        candidate=CANDIDATE.manifest(), matrix=validate_matrix(), f1_identity=identity,
        f1_review=review, historical_attempts=historical, frozen_contexts=frozen['frozen_contexts'],
        frozen_initial_payload_checks=payloads, historical_semantic_validation=frozen,
        execution_source_sha256=_source_identity(), discovery_fingerprint=fingerprint(discovery),
        zero_science_guards_validated=True, provider_calls=0,
        stage_b_comparison='Provider-selected scope; exact unchanged production builder for each accepted scope. No historical scope is forced.',
        cost_estimate=dict(expected_twelve_call_output_ceiling_estimate_usd=str(estimate),
            unrestricted_sixteen_call_upper_bound_usd=str(unrestricted),
            enforced_spending_ceiling_usd=str(frontier.SPENDING_CEILING),
            aggregate_ceiling_may_stop_before_all_recovery_calls=True,
            estimates_use_conservative_utf8_byte_input_bound=True))


def _recovery_calls(attempt):
    return sum(c.get('recovery_kind') != 'initial' for c in (attempt or {}).get('calls', ()))


def _classification(entries):
    if len(entries) != 3 or any(e['final_status'] in ('HUMAN_REVIEW_PENDING', 'HARD_STOP', 'SAFETY_FAIL') for e in entries):
        return 'INDETERMINATE'
    failures = sum(e['final_status'] != 'PASS' for e in entries)
    return ('STABLE_IN_SCREEN' if failures == 0 else
            'MOSTLY_STABLE_WITH_ISOLATED_FAILURE' if failures == 1 else 'UNSTABLE')


def stability_matrix(saved, report):
    result = {}
    for sid in SELECTED:
        old = next(r for r in saved['results'] if r['case'] == sid)
        entries = [dict(attempt_number=1, attempt_id='f1-1', historical=True,
            historical_status=old['status'], final_status='PASS' if sid == 'I08' else old['status'],
            deterministic_contract_accepted=(old['attempt']['contract_success'] is True
                and old['attempt']['admission_success'] is True),
            external_human_semantic_review='PASS' if sid == 'I08' else None,
            direct_or_recovered='recovered' if _recovery_calls(old['attempt']) else 'direct')]
        entries.extend(dict(attempt_number=r['attempt_number'], attempt_id=r['attempt_id'],
            historical=False, final_status=r['status'], direct_or_recovered=r['direct_or_recovered'],
            deterministic_contract_accepted=((r.get('attempt') or {}).get('contract_success') is True
                and (r.get('attempt') or {}).get('admission_success') is True),
            exact_target=(r.get('attempt') or {}).get('admitted'),
            clarification_reason=(r.get('attempt') or {}).get('clarification_reason'))
            for r in report['results'] if r['scenario_id'] == sid)
        contracts = [dict(final_status='PASS' if e['deterministic_contract_accepted'] else
            'HARD_STOP' if e['final_status'] in ('HARD_STOP', 'SAFETY_FAIL') else 'CONTRACT_FAIL')
            for e in entries]
        result[sid] = dict(attempts=entries, classification=_classification(entries),
            deterministic_contract_classification=_classification(contracts))
    return result


def run(*, output, original=F1_DIRECTORY, live=False, human_review_path=None,
        environment=None, client_factory=None, spacing=0, clock=time.monotonic, sleeper=time.sleep):
    output = Path(output)
    if live and client_factory is None:
        frontier.baseline(require_clean=True)
    env = os.environ if environment is None else environment
    private = breadth._private_values(env)
    if type(spacing) not in (int, float) or not 0 <= spacing <= 60:
        raise ValueError('Independent attempt spacing must be bounded to 0–60 seconds.')
    review_path = output / 'f1_i08_human_review_disposition.json' if human_review_path is None else Path(human_review_path)
    discovery = json.loads((output / 'discovery.json').read_text())
    frontier.validate_discovery(discovery)
    plan = stability_validate(original, human_review_path=review_path, discovery=discovery)
    saved, identity = validate_f1(original)

    def save(name, value):
        path = output / name
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(breadth._safe_json(value, private), indent=2,
            allow_nan=False, ensure_ascii=False) + '\n')
        temporary.replace(path)

    if not live:
        if (output / 'offline_validation.json').exists():
            raise ValueError('Do not overwrite F2 dry-run evidence.')
        save('offline_validation.json', plan)
        save('repeat_plan.json', plan)
        print(json.dumps(dict(phase='F2_DRY_VALIDATED', matrix=plan['matrix'],
                            cost_estimate=plan['cost_estimate']), indent=2), flush=True)
        return breadth._safe_json(plan, private)
    if (output / 'results.json').exists():
        raise ValueError('F2 cannot repeat or resume retained attempts.')
    if json.loads((output / 'offline_validation.json').read_text()) != plan:
        raise ValueError('F2 validation changed before paid dispatch.')
    if not env.get('OPENROUTER_API_KEY', '').strip():
        raise ValueError('Use the existing configured agent Conda environment for F2.')
    availability = Availability('available', 'CANDIDATE_MODEL_AVAILABLE', CANDIDATE.model_id,
                                'models.list', CANDIDATE.model_id)
    ledger = SpendingGuard()
    report = dict(phase='F2', candidate=CANDIDATE.manifest(), plan=plan, results=[],
        f1_identity=identity, human_review_queue=[], stability_matrix={}, hard_stop=None,
        waits=[], min_attempt_spacing_seconds=spacing, complete=False,
        credential_present_in_agent_environment=True)
    deadline = 0.0
    original_build = harness.build_context
    original_planner_observations = harness._planner_observations
    for sid, number in SCHEDULE:
        wait = max(0.0, deadline - clock())
        if wait:
            report['waits'].append(dict(scenario_id=sid, attempt_number=number, seconds=wait))
            sleeper(wait)
        ledger.begin(sid, number)
        scenario = scenario_by_id(sid)
        checks, contexts, observations, structures = [], [], [], []
        row = dict(case=sid, scenario_id=sid, attempt_number=number, attempt_id=f'f2-{number}',
            candidate=CANDIDATE.manifest(), attempt=None, diagnostics=[], payload_checks=checks,
            frozen_context_checks=contexts, observations=observations, stage_b_structures=structures,
            status=None, direct_or_recovered='direct', recovery_calls=0)
        expected = plan['frozen_contexts'][sid]
        payload_guard = breadth._payload_guard(scenario, checks)

        def context_build(*args, **kwargs):
            context = original_build(*args, **kwargs)
            actual = dict(public_context_fingerprint=fingerprint(context.public),
                private_captured_context_fingerprint=fingerprint(_interaction(context)))
            if any(actual[key] != expected[key] for key in actual):
                raise FrontierHardStop('F2 captured context changed before provider transmission.')
            contexts.append(actual)
            return context

        def before_complete(*, prompt, response_schema):
            if runtime is not None and any((d.get('diagnostic') or {}).get('http_status')
                    in (401, 402, 403) for d in runtime.diagnostics):
                raise FrontierHardStop('F2 configuration or balance failure forbids further provider dispatch.')
            if not contexts or _source_identity() != plan['execution_source_sha256']:
                raise FrontierHardStop('F2 source/context identity changed before transmission.')
            frontier.baseline(require_clean=client_factory is None)
            _, current = validate_f1(original)
            if current != identity or validate_review(review_path, saved) != plan['f1_review']:
                raise FrontierHardStop('Retained F1 evidence or external review changed.')
            payload_guard(prompt=prompt, response_schema=response_schema)

        def planner_observations(recorder, request, registry, diagnostics=()):
            original_planner_observations(recorder, request, registry, diagnostics)
            for call, response in zip(recorder.calls, recorder.responses):
                if call['surface'] != 'stage_b' or call.get('decision_kind') != 'plan':
                    continue
                selected = next(c['capability_ids'] for c in reversed(recorder.calls[:recorder.calls.index(call)])
                                if c['surface'] == 'stage_a' and c.get('capability_ids'))
                candidate = parse_semantic_wire_v4(response, request, registry,
                    visible_tool_names=PlanningScope(registry, tuple(selected)).visible_tool_names)
                structures.append(dict(recovery_kind=call['recovery_kind'], steps=[dict(
                    step_id=s.step_id, tool=s.tool_name, control_dependencies=list(s.control_dependencies),
                    sources=[dict(target_port=source.target_port,
                        kind=type(source).__name__, **{key: getattr(source, key) for key in
                            ('input_name', 'step_id', 'output_name', 'handle') if hasattr(source, key)})
                        for source in s.sources]) for s in candidate.steps]))

        runtime = None
        try:
            client = None if client_factory is None else client_factory(
                candidate=CANDIDATE, case=sid, attempt_number=number)
            runtime = frontier.FrontierTransport(availability=availability, ledger=ledger, case=sid,
                before_complete=before_complete, environment=env, client=client)
            row['diagnostics'] = runtime.diagnostics
            model = _Observed(runtime, observations)
            with patch.object(harness, 'build_context', context_build), patch.object(
                    harness, '_planner_observations', planner_observations):
                attempt = harness.run_attempt(scenario, CANDIDATE, model=model,
                                              attempt_id=row['attempt_id']).to_dict()
            row['attempt'] = attempt
            if any(attempt['safety'].values()):
                raise harness.ZeroScienceViolation('F2 entered forbidden science.')
            row['status'] = breadth.status_for_attempt(attempt)
            row['recovery_calls'] = _recovery_calls(attempt)
            row['direct_or_recovered'] = 'recovered' if row['recovery_calls'] else 'direct'
            if row['status'] == 'HUMAN_REVIEW_PENDING':
                report['human_review_queue'].append(dict(scenario_id=sid, attempt_number=number,
                    attempt_id=row['attempt_id'], candidate_id=CANDIDATE.candidate_id,
                    status='HUMAN_REVIEW_PENDING', explanation=attempt.get('explanation'),
                    support=attempt.get('support'), admitted=attempt.get('admitted'),
                    fixture_provenance=attempt['fixture_provenance'], human_semantic_review=None))
            failed = [d for d in runtime.diagnostics if d['provider_outcome'] == 'failed']
            stop_status = next((d['diagnostic'].get('http_status') for d in failed
                if d.get('diagnostic') and d['diagnostic'].get('http_status') in (401, 402, 403)), None)
            previous = sum(d['provider_outcome'] == 'failed' for r in report['results']
                           for d in r.get('diagnostics', ()))
            if stop_status is not None or (failed and previous + len(failed) >= 2):
                report['hard_stop'] = dict(scenario_id=sid, attempt_number=number,
                    reason='CONFIGURATION_BALANCE_OR_REPEATED_PROVIDER_FAILURE',
                    category='operational', http_status=stop_status)
        except (FrontierHardStop, OpenRouterPolicyViolation) as exc:
            row['status'] = 'HARD_STOP'
            report['hard_stop'] = dict(scenario_id=sid, attempt_number=number, reason=str(exc),
                                     category='configuration_or_exact_model_or_cost_guard')
        except harness.ZeroScienceViolation:
            row['status'] = 'SAFETY_FAIL'
            report['hard_stop'] = dict(scenario_id=sid, attempt_number=number,
                                     reason='ZERO_SCIENCE_GUARD_FIRED', category='scientific_safety')
        except Exception as exc:
            row['status'] = 'HARD_STOP'
            report['hard_stop'] = dict(scenario_id=sid, attempt_number=number,
                reason='LOCAL_CONFIGURATION_OR_RUNNER_FAILURE', category='configuration',
                diagnosis=sanitize_provider_error(exc, secrets=private))
        finally:
            if runtime is not None:
                runtime.close()
            report['results'].append(row)
            report['spending_guard'] = ledger.manifest()
            report['actual_usage'] = frontier._usage(report['results'])
            cost = report['actual_usage']['reported_cost_usd']
            report['actual_usage']['average_cost_per_scenario_attempt_usd'] = (
                None if cost is None else str(Decimal(cost) / len(report['results'])))
            report['counts'] = dict(statuses=dict(Counter(r['status'] for r in report['results'])),
                scenario_attempts=len(report['results']), completion_requests=len(ledger.requests),
                production_recovery_calls=sum(r['recovery_calls'] for r in report['results']),
                operational_events=sum(d['provider_outcome'] == 'failed' for r in report['results']
                    for d in r.get('diagnostics', ())))
            report['stability_matrix'] = stability_matrix(saved, report)
            save('results.json', report)
            save('token_cost_accounting.json', dict(policy=ledger.manifest(), actual_usage=report['actual_usage']))
            save('human_review_queue.json', report['human_review_queue'])
            save('stability_matrix.json', report['stability_matrix'])
            save('frozen_input_audit.json', dict(f1_identity=identity, frozen_contexts=plan['frozen_contexts'],
                frozen_initial_payload_checks=plan['frozen_initial_payload_checks'],
                actual_attempts=[dict(scenario_id=r['scenario_id'], attempt_number=r['attempt_number'],
                    payload_checks=r['payload_checks'], frozen_context_checks=r['frozen_context_checks'])
                    for r in report['results']], stage_b_comparison=plan['stage_b_comparison']))
        print(json.dumps(dict(scenario_id=sid, attempt_number=number, status=row['status'],
            completion_requests=len(row['diagnostics']), aggregate_calls=len(ledger.requests))), flush=True)
        if report['hard_stop']:
            break
        deadline = clock() + spacing
    report['complete'] = len(report['results']) == 8 and report['hard_stop'] is None
    _, preserved = validate_f1(original)
    if preserved != identity:
        report['complete'] = False
        report['hard_stop'] = dict(reason='F1_EVIDENCE_CHANGED_AT_CLOSEOUT', category='integrity')
    report['f1_evidence_preserved'] = preserved == identity
    save('results.json', report)
    return breadth._safe_json(report, private)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--original', type=Path, default=F1_DIRECTORY)
    parser.add_argument('--human-review', type=Path)
    parser.add_argument('--spacing', type=float, default=0)
    gate = parser.add_mutually_exclusive_group(required=True)
    gate.add_argument('--dry-run', action='store_true')
    gate.add_argument('--live', action='store_true')
    args = parser.parse_args()
    run(output=args.output, original=args.original, live=args.live,
        human_review_path=args.human_review, spacing=args.spacing)


if __name__ == '__main__':
    main()
