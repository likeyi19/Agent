"""One independent breadth attempt per frozen scenario/candidate; no tuning."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from statistics import median
import subprocess
import time
from tempfile import TemporaryDirectory
from urllib.parse import quote, quote_plus

from benchmarks.planner.benchmark import guarded_registry
from agent.orchestration.planning_model import PlanningModelError
from agent.orchestration.planning_scope import fingerprint
from agent.orchestration.planning_scope import PlanningScope
from agent.orchestration.semantic_prompt import build_semantic_planning_prompt
from agent.orchestration.semantic_wire_v4 import build_semantic_wire_v4_schema
from agent.schemas import AgentRequest
from .breadth_transport import COHORT, PayloadValidationFailure, _reporting_error, build_transport
from .candidates import Availability, build_candidate
from .diagnostics import sanitize_provider_error
from .fixtures import build_context
from .harness import candidate_identity, run_attempt, zero_science
from .openrouter import discover_models
from .run_smoke import comparison_manifest
from .scenarios import canonical_scenarios


BASELINE = '48d83625e3da45aad5b7687d14ecf2118b29f036'
SCENARIOS_FINGERPRINT = '588501b58a95714ef1a7cf8dcdc8a3ef483aa0f4401494941bd9ef233a33ccf3'
FAILURES = {'operational': 'OPERATIONAL_FAIL', 'contract': 'CONTRACT_FAIL',
            'semantic': 'SEMANTIC_FAIL', 'deterministic_admission': 'ADMISSION_FAIL'}
INFRASTRUCTURE_PATHS = frozenset({
    'benchmarks/interactive/openrouter.py',
    'benchmarks/interactive/run_breadth.py',
    'benchmarks/interactive/run_stability.py',
    'benchmarks/interactive/run_frontier.py',
    'benchmarks/interactive/run_frontier_stability.py',
    'benchmarks/interactive/README.md',
})


def _git(*arguments):
    return subprocess.check_output(['git', *arguments], text=True).strip()


def _infrastructure_path(name):
    """Transport, runners, documentation and offline tests do not own semantics."""
    path = Path(name)
    return (name in INFRASTRUCTURE_PATHS
        or (name.startswith('tests/benchmarks/') and path.suffix == '.py'))


def repository_baseline(*, require_clean=False, experiment_baseline=BASELINE):
    """Freeze semantic owners while allowing reviewed infrastructure descendants."""
    if not isinstance(experiment_baseline, str) or not re.fullmatch(r'[0-9a-f]{40}', experiment_baseline):
        raise ValueError('Qualification requires an exact experiment commit identity.')
    head = _git('rev-parse', 'HEAD')
    try:
        if _git('merge-base', experiment_baseline, head) != experiment_baseline:
            raise ValueError('Qualification baseline must be an ancestor of the checkout.')
    except subprocess.CalledProcessError as exc:
        raise ValueError('Qualification baseline is unavailable or unrelated.') from exc
    original = _git('ls-tree', '-r', '--name-only', experiment_baseline, '--', 'benchmarks/interactive').splitlines()
    frozen = ['src/agent', 'benchmarks/planner', *[name for name in original
        if name.endswith('.py') and not _infrastructure_path(name)]]
    if (_git('diff', '--name-only', experiment_baseline, head, '--', *frozen)
            or _git('diff', '--name-only', experiment_baseline, '--', *frozen)):
        raise ValueError('Frozen Agent/planner/qualification contracts changed.')
    tracked = _git('diff', '--name-only')
    index = _git('diff', '--cached', '--name-only')
    if any(not _infrastructure_path(name) for name in (*tracked.splitlines(), *index.splitlines())):
        raise ValueError('Pending qualification changes must be infrastructure, documentation or offline tests.')
    if require_clean and (tracked or index):
        raise ValueError('Live qualification requires a clean tracked tree and index.')
    return dict(experiment_baseline=experiment_baseline, infrastructure_commit=head,
                tracked_clean=not tracked, index_clean=not index)


def validate_call_identity(row, scenario, candidate):
    """A completed cell needs observed calls with the same exact attempt identity."""
    attempt = row['attempt']
    calls = attempt.get('calls', ())
    count = row['provider_completions']
    if (row['status'] != status_for_attempt(attempt)
            or type(count) is not int or count <= 0 or not calls
            or count > (5 if scenario.surface == 'layer2' else 1)
            or len(row['diagnostics']) != count or count > len(calls)
            or any(call.get('candidate') != candidate_identity(candidate)
                or call.get('scenario_id') != scenario.scenario_id
                or call.get('attempt_id') != row['attempt_id'] for call in calls)):
        raise ValueError('Historical call identity or completion accounting is inconsistent.')


def _private_values(environment):
    return tuple(value for name, value in environment.items() if value and
        (name.endswith('_API_KEY') or any(label in name.upper() for label in
         ('WORKSPACE_ID', 'WORK_SPACE_ID', 'PROJECT_ID', 'ACCOUNT_ID', 'CONSUMER'))))


def _safe_json(value, private):
    # Model prose is retained only after private values are removed. Scoring precedes redaction.
    variants = {encoded for item in private for encoded in
                (item, quote(item, safe=''), quote_plus(item), json.dumps(item)[1:-1])}
    patterns = [re.compile(re.escape(item), re.IGNORECASE) for item in sorted(variants, key=len, reverse=True)]
    def clean(item):
        if isinstance(item, str):
            for pattern in patterns:
                item = pattern.sub('[REDACTED]', item)
            return item
        if isinstance(item, dict): return {clean(key): clean(value) for key, value in item.items()}
        if isinstance(item, (tuple, list)): return [clean(value) for value in item]
        return item
    return _reporting_error(clean(value))


def _matrix(scenarios, candidates):
    canonical = {s.scenario_id: s.fingerprint() for s in canonical_scenarios()}
    if (not scenarios or len({s.scenario_id for s in scenarios}) != len(scenarios)
            or any(canonical.get(s.scenario_id) != s.fingerprint() for s in scenarios)):
        raise ValueError('Use distinct unchanged canonical scenarios.')
    if (not candidates or len({c.candidate_id for c in candidates}) != len(candidates)
            or any(c not in COHORT for c in candidates)):
        raise ValueError('Use only distinct frozen cohort candidates.')
    return [(s, c) for s in scenarios for c in candidates]


def dry_validate(*, scenarios=None, candidates=COHORT, environment=None, require_credentials=False):
    scenarios = tuple(canonical_scenarios() if scenarios is None else scenarios)
    candidates = tuple(candidates)
    matrix = _matrix(scenarios, candidates)
    assert fingerprint([s.to_dict() for s in canonical_scenarios()]) == SCENARIOS_FINGERPRINT
    checkout = repository_baseline(require_clean=require_credentials)
    env = os.environ if environment is None else environment
    presence = {c.credential_env: bool(env.get(c.credential_env)) for c in candidates}
    if require_credentials and not all(presence.values()):
        raise ValueError('Frozen cohort credentials must be present before live execution.')
    contexts = {}
    for scenario in scenarios:
        registry, guard = guarded_registry()
        with TemporaryDirectory(prefix='agent-breadth-dry-') as scratch:
            with zero_science(guard, read_only=True) as counters:
                context = build_context(Path(scratch), scenario.fixture, registry,
                    utterance=scenario.utterance,
                    execution_inputs=None if scenario.request is None else scenario.request.inputs,
                    include_interaction=scenario.surface != 'layer2')
                contexts[scenario.scenario_id] = fingerprint(context.public)
        assert not any(counters.values())
    return dict(baseline=BASELINE, infrastructure_commit=checkout['infrastructure_commit'],
        scenarios_fingerprint=SCENARIOS_FINGERPRINT,
        scenarios=[s.to_dict() | {'fingerprint': s.fingerprint(), 'context_fingerprint': contexts[s.scenario_id]} for s in scenarios],
        candidates=[c.manifest() for c in candidates], credentials_present=presence,
        attempt_order=[[s.scenario_id, c.candidate_id] for s, c in matrix],
        expected_attempts=len(matrix),
        expected_direct_completions=sum(3 if s.surface == 'layer2' else 1 for s, c in matrix),
        conservative_completion_ceiling=sum(5 if s.surface == 'layer2' else 1 for s, c in matrix),
        zero_science_guards_validated=True, provider_calls=0)


def status_for_attempt(attempt):
    category = attempt['failure_category']
    if category in FAILURES: return FAILURES[category]
    if attempt['contract_success'] is not True: return 'CONTRACT_FAIL'
    if attempt['human_review_required']: return 'HUMAN_REVIEW_PENDING'
    if attempt['semantic_success'] is True: return 'PASS'
    if attempt['semantic_success'] is False: return 'SEMANTIC_FAIL'
    raise ValueError('Attempt lacks a conclusive frozen-contract observation.')


def _discover(candidates, environment):
    evidence, requests = {}, 0
    private = _private_values(environment)
    for provider in dict.fromkeys(c.provider_id for c in candidates):
        group = tuple(c for c in candidates if c.provider_id == provider)
        runtime = None
        try:
            runtime = build_candidate(group[0], environment=environment)
            _, guard = guarded_registry()
            with zero_science(guard, read_only=True):
                if provider == 'openrouter':
                    report = discover_models(runtime._client, candidates=group, secrets=private)
                    requests += report['catalog_requests']
                    for row in report['candidates']:
                        evidence[row['candidate']['candidate_id']] = row
                else:
                    requests += 1
                    catalog = runtime._client.models.list(timeout=60.0)
                    cards = catalog.data
                    if not isinstance(cards, (list, tuple)) or len(cards) > 4096 or getattr(catalog, 'has_more', False):
                        raise ValueError('Invalid or incomplete bounded catalog.')
                    for candidate in group:
                        matches = [card for card in cards if getattr(card, 'id', None) == candidate.model_id]
                        if len(matches) > 1: raise ValueError('Duplicate exact catalog identity.')
                        available = len(matches) == 1
                        evidence[candidate.candidate_id] = dict(
                            availability=Availability('available' if available else 'unavailable',
                                'CANDIDATE_MODEL_AVAILABLE' if available else 'CANDIDATE_MODEL_UNAVAILABLE',
                                candidate.model_id, 'models.list', candidate.model_id if available else None).to_dict(),
                            diagnosis=None)
        except Exception as exc:
            diagnostic = _reporting_error(sanitize_provider_error(exc, secrets=private))
            for candidate in group:
                evidence[candidate.candidate_id] = dict(availability=Availability(
                    'operational_failure', 'PLANNING_PROVIDER_ERROR', candidate.model_id,
                    'models.list', http_status=diagnostic['http_status']).to_dict(), diagnosis=diagnostic)
        finally:
            if runtime is not None: runtime.close()
    return _reporting_error(dict(catalog_requests=requests, candidates=evidence))


def _limit_evidence(diagnostics, previous_429):
    # Later production recovery obtaining a response supersedes a transient failure.
    if not diagnostics or diagnostics[-1]['provider_outcome'] == 'returned':
        return None, 0, None
    row = diagnostics[-1]
    diagnostic = row.get('diagnostic') or {}
    if _provider_wide(row):
        return 'explicit_account_limit_or_block', previous_429, row
    if row.get('http_status', diagnostic.get('http_status')) != 429:
        return None, 0, None
    count = previous_429 + 1
    if _delay_seconds(row) > 60:
        return 'reset_outside_bounded_window', count, row
    if count >= 2:
        return 'consecutive_unrecovered_model_rate_limits', count, row
    return None, count, None


def _provider_wide(row):
    error = (row.get('diagnostic') or {}).get('provider_error') or {}
    codes = {str(error.get(k) or '').lower() for k in ('code', 'reason', 'type')}
    return bool(codes.intersection({'account_blocked', 'account_disabled', 'account_suspended',
        'account_quota_exceeded', 'organization_quota_exhausted', 'billing_hard_limit_reached',
        'insufficient_quota', 'invalid_api_key'}))


def _duration(value):
    if not isinstance(value, str): return 0.0
    if re.fullmatch(r'\d+(?:\.\d+)?', value): return float(value)
    if not re.fullmatch(r'(?:\d+(?:\.\d+)?(?:ms|s|m|h|d))+', value): return 0.0
    scales = {'ms': .001, 's': 1, 'm': 60, 'h': 3600, 'd': 86400}
    return sum(float(number) * scales[unit] for number, unit in
               re.findall(r'(\d+(?:\.\d+)?)(ms|s|m|h|d)', value))


def _delay_seconds(row):
    delay = row.get('retry_after_seconds') or 0.0
    headers = (row.get('diagnostic') or {}).get('rate_limit') or {}
    for name, value in headers.items():
        if 'remaining' in name and value == '0':
            delay = max(delay, _duration(headers.get(name.replace('remaining', 'reset'))))
    return delay


def _wait_until(row):
    if not row or row['provider_outcome'] == 'returned': return 0.0
    return (row.get('request_finished_monotonic') or time.monotonic()) + _delay_seconds(row)


def run_breadth(*, output, scenarios=None, candidates=COHORT, discovery=None, factory=None,
                environment=None, attempt_id='breadth-1'):
    repository_baseline(require_clean=factory is None)
    scenarios = tuple(canonical_scenarios() if scenarios is None else scenarios)
    candidates = tuple(candidates)
    matrix = _matrix(scenarios, candidates)
    env = os.environ if environment is None else environment
    private = _private_values(env)
    preflight = dry_validate(scenarios=scenarios, candidates=candidates, environment=env,
                            require_credentials=factory is None and discovery is None)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / '.gitignore').write_text('*\n')
    def save(name, value):
        safe = _safe_json(value, private)
        encoded = json.dumps(safe, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
        target = output / name
        temporary = target.with_suffix(target.suffix + '.tmp')
        temporary.write_text(encoded)
        temporary.replace(target)
    save('preflight.json', preflight)
    discovery = _discover(candidates, env) if discovery is None else discovery
    if 'candidates' not in discovery: discovery = dict(catalog_requests=0, candidates=discovery)
    save('discovery.json', discovery)
    manifest = comparison_manifest()
    manifest.update(completion_limit_per_candidate=None,
                    completion_limit_per_isolated_attempt=1,
                    completion_limit_per_layer2_attempt=5,
                    attempt_order='scenario_major_frozen_cohort', repetitions=1)
    report = dict(schema_version=1, phase='Q2.1', created_at_utc=datetime.now(timezone.utc).isoformat(),
        baseline=BASELINE, comparison_manifest=manifest, preflight=preflight,
        discovery=discovery, attempt_order=preflight['attempt_order'], results=[], candidate_counts={},
        blocked_providers={}, blocked_candidates={}, stability='not_yet_measured', q2_2_run=False)
    factory = factory or (lambda candidate, availability, max_calls: build_transport(
        candidate, availability=availability, environment=env, max_calls=max_calls))
    rate_counts = {}
    for index, (scenario, candidate) in enumerate(matrix, 1):
        evidence = discovery['candidates'][candidate.candidate_id]
        availability = Availability(**evidence['availability'])
        row = dict(scenario_id=scenario.scenario_id, candidate=candidate.manifest(),
            attempt_id=attempt_id, attempt_index=index, surface=scenario.surface,
            status=None, reason=None, attempted=False, provider_completions=0,
            recovery_calls=0, attempt=None, diagnostics=[], stability='not_yet_measured')
        if candidate.provider_id in report['blocked_providers']:
            row.update(status='NOT_RUN_PROVIDER_LIMIT', reason=report['blocked_providers'][candidate.provider_id])
        elif candidate.candidate_id in report['blocked_candidates']:
            row.update(status='NOT_RUN_PROVIDER_LIMIT', reason=report['blocked_candidates'][candidate.candidate_id])
        elif availability.status != 'available':
            row.update(status='OPERATIONAL_FAIL', reason=dict(kind='model_availability', evidence=evidence))
        else:
            runtime = None
            try:
                try:
                    runtime = factory(candidate, availability, 5 if scenario.surface == 'layer2' else 1)
                except PlanningModelError as exc:
                    row.update(status='OPERATIONAL_FAIL', reason=dict(kind='transport_configuration',
                        diagnostic=sanitize_provider_error(exc, secrets=private)))
                    report['results'].append(row)
                    save('results.json', report)
                    continue
                attempt = run_attempt(scenario, candidate, model=runtime, attempt_id=attempt_id).to_dict()
                assert not any(attempt['safety'].values())
                assert attempt['context_fingerprint'] == next(s['context_fingerprint'] for s in preflight['scenarios'] if s['scenario_id'] == scenario.scenario_id)
                row.update(attempted=True, attempt=attempt, status=status_for_attempt(attempt),
                    reason=attempt.get('error_code') or attempt['failure_category'],
                    harness_completion_invocations=len(attempt['calls']),
                    provider_completions=runtime.completion_calls,
                    recovery_calls=sum(c['recovery_kind'] != 'initial'
                                       for c in attempt['calls'][:runtime.completion_calls]),
                    diagnostics=list(getattr(runtime, 'diagnostics', ())))
                assert row['provider_completions'] == len(row['diagnostics']) <= len(attempt['calls'])
            finally:
                if runtime is not None: runtime.close()
            reason, rate_counts[candidate.candidate_id], trigger = _limit_evidence(
                row['diagnostics'], rate_counts.get(candidate.candidate_id, 0))
            if reason:
                limits = report['blocked_providers'] if _provider_wide(trigger) else report['blocked_candidates']
                key = candidate.provider_id if _provider_wide(trigger) else candidate.candidate_id
                limits[key] = dict(
                    reason=reason, triggering_scenario=scenario.scenario_id,
                    triggering_candidate=candidate.candidate_id, evidence=trigger)
        report['results'].append(row)
        save('results.json', report)
        print(json.dumps(dict(index=index, scenario=scenario.scenario_id, candidate=candidate.candidate_id,
            status=row['status'], completions=row['provider_completions'], recovery=row['recovery_calls']), sort_keys=True), flush=True)
    for candidate in candidates:
        rows = [r for r in report['results'] if r['candidate']['candidate_id'] == candidate.candidate_id]
        latencies = [sum(c['latency_seconds'] or 0 for c in r['attempt']['calls']) for r in rows if r['attempt']]
        report['candidate_counts'][candidate.candidate_id] = dict(
            attempts=sum(r['attempted'] for r in rows), completions=sum(r['provider_completions'] for r in rows),
            recovery_calls=sum(r['recovery_calls'] for r in rows), statuses=dict(Counter(r['status'] for r in rows)),
            provider_failures=sum(c['provider_outcome'] == 'failed' for r in rows for c in r['diagnostics']),
            median_attempt_seconds=median(latencies) if latencies else None,
            total_provider_seconds=sum(latencies))
    report['human_review_queue'] = [dict(scenario_id=r['scenario_id'], candidate=r['candidate'],
        status=r['status'], explanation=r['attempt'].get('explanation'),
        support=r['attempt'].get('support'), guidance_candidates=r['attempt'].get('guidance_candidates'),
        fixture_provenance=r['attempt']['fixture_provenance']) for r in report['results']
        if r['attempt'] and r['attempt']['human_review_required'] and r['status'] == 'HUMAN_REVIEW_PENDING']
    report['safety'] = dict(scientific_calls=0, executor_entries=0, reconstruction_entries=0, scientific_step_results=0)
    report['complete'] = len(report['results']) == len(matrix)
    save('results.json', report)
    save('human_review.json', report['human_review_queue'])
    return _safe_json(report, private)


def _original(original):
    if isinstance(original, dict):
        return json.loads(json.dumps(original)), None
    path = Path(original)
    if path.is_dir(): path = path / 'results.json'
    raw = path.read_bytes()
    return json.loads(raw), dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest())


def _phase(payload):
    return next((phase for key, phase in (
        ('turn_schema_version', 'interpreter'), ('selection_schema_version', 'stage_a'),
        ('dialogue_schema_version', 'answer'), ('guidance_schema_version', 'guidance'))
        if key in payload), 'stage_b')


class _ProbeStop(BaseException):
    pass


def _probe_contracts(scenario):
    """Capture frozen production payloads offline; never supply witnesses to live models."""
    captured = {}
    class Probe:
        def complete(self, *, prompt, response_schema):
            phase = _phase(json.loads(prompt))
            captured[phase] = (prompt, response_schema)
            if scenario.surface == 'layer2' and phase == 'interpreter':
                return json.dumps({'turn_schema_version': 1,
                    'decision': {'kind': 'execute_plan', 'target': 'inspect_scATAC'}})
            if scenario.surface == 'layer2' and phase == 'stage_a':
                return json.dumps({'selection_schema_version': 1,
                    'decision': {'kind': 'select', 'capability_ids': ['processed_inspection']}})
            raise _ProbeStop()
    try:
        run_attempt(scenario, COHORT[0], model=Probe(), attempt_id='offline-contract-probe')
    except _ProbeStop:
        pass
    assert captured
    return captured


def _layer2_contract(scenario, registry, names):
    request = scenario.request
    # Application adds this key. Only its shallow type is provider-visible.
    inputs = dict(request.inputs, output_dir='/synthetic/private-planning')
    effective = AgentRequest(request.request_id, request.prompt, inputs, request.mode)
    return (build_semantic_planning_prompt(effective, registry, visible_tool_names=names),
            build_semantic_wire_v4_schema(registry, effective, visible_tool_names=names))


def _pair(prompt, schema):
    return fingerprint(prompt), fingerprint(schema)


def resume_validate(original, *, environment=None, require_credentials=False):
    repository_baseline(require_clean=require_credentials)
    saved, identity = _original(original)
    preflight = dry_validate(environment=environment)
    env = os.environ if environment is None else environment
    if require_credentials and not env.get('GROQ_API_KEY', '').strip():
        raise ValueError('Groq credential must be present before continuation.')
    if saved.get('baseline') != BASELINE or saved.get('phase') != 'Q2.1':
        raise ValueError('Continuation requires the frozen original Q2.1 ledger.')
    for name in ('baseline', 'scenarios_fingerprint', 'scenarios', 'candidates', 'attempt_order'):
        if saved['preflight'][name] != preflight[name]:
            raise ValueError('Original frozen scenario/context/candidate inventory differs.')
    current_manifest = comparison_manifest()
    for group in ('src/agent', 'benchmarks/planner'):
        old = saved.get('comparison_manifest', {}).get('source_fingerprints', {}).get(group)
        if old is not None and current_manifest.get('source_fingerprints', {}).get(group) != old:
            raise ValueError('Frozen production or planner source fingerprint changed.')
    matrix = _matrix(canonical_scenarios(), COHORT)
    if len(saved['results']) != 42: raise ValueError('Original ledger must contain 42 cells.')
    expected_done = {('I01', COHORT[0].candidate_id)} | {
        (s.scenario_id, COHORT[2].candidate_id) for s in canonical_scenarios()}
    done, missing = [], []
    for (scenario, candidate), row in zip(matrix, saved['results']):
        key = (scenario.scenario_id, candidate.candidate_id)
        if row['scenario_id'] != key[0] or row['candidate'] != candidate.manifest() or row['surface'] != scenario.surface:
            raise ValueError('Original cell identities/order do not match the frozen matrix.')
        if key in expected_done:
            attempt = row['attempt']
            if not row['attempted'] or not attempt or row['provider_completions'] <= 0:
                raise ValueError('An existing completed cell lacks accepted attempt identity.')
            validate_call_identity(row, scenario, candidate)
            if (attempt['scenario_fingerprint'] != scenario.fingerprint()
                    or attempt['scenario_id'] != scenario.scenario_id
                    or attempt['candidate'] != candidate_identity(candidate)
                    or attempt['attempt_id'] != row['attempt_id']
                    or any(c['candidate'] != candidate_identity(candidate) for c in attempt['calls'])
                    or attempt['context_fingerprint'] != next(s['context_fingerprint']
                        for s in preflight['scenarios'] if s['scenario_id'] == scenario.scenario_id)
                    or any(attempt['safety'].values())):
                raise ValueError('Original completed cell fingerprint or safety mismatch.')
            done.append(list(key))
        else:
            if (row['attempt'] is not None or row['attempted'] or row['provider_completions']
                    or row['diagnostics'] or row['status'] != 'NOT_RUN_PROVIDER_LIMIT'):
                raise ValueError('Missing cell has prior observation or lacks suppression evidence.')
            missing.append(list(key))
    assert len(done) == 15 and len(missing) == 27
    registry, _ = guarded_registry()
    probes = {s.scenario_id: _probe_contracts(s) for s in canonical_scenarios()}
    references = []
    for row in saved['results']:
        if row['attempt'] is None: continue
        scenario = next(s for s in canonical_scenarios() if s.scenario_id == row['scenario_id'])
        for call in row['attempt']['calls']:
            if call['recovery_kind'] != 'initial': continue
            if scenario.surface == 'layer2' and call['surface'] == 'stage_b':
                selected = next(c['capability_ids'] for c in row['attempt']['calls'] if c['surface'] == 'stage_a')
                expected = _layer2_contract(scenario, registry, PlanningScope(registry, tuple(selected)).visible_tool_names)
            else:
                expected = probes[scenario.scenario_id][call['surface']]
            if _pair(*expected) != (call['prompt_fingerprint'], call['schema_fingerprint']):
                raise ValueError('Frozen initial prompt/schema differs from the original observation.')
            references.append(dict(scenario_id=scenario.scenario_id, surface=call['surface'],
                prompt_fingerprint=call['prompt_fingerprint'], schema_fingerprint=call['schema_fingerprint']))
    return dict(phase='Q2.1b', baseline=BASELINE, infrastructure_commit=preflight['infrastructure_commit'],
        original_identity=identity,
        original_ledger_fingerprint=fingerprint(saved), total_cells=42, existing_attempts=15,
        missing_attempts=27, planned_cells=missing, preserved_cells=done,
        planned_nemotron_calls=0, planned_gpt_oss_i01_calls=0,
        expected_direct_completions=29, conservative_completion_ceiling=31,
        credentials_present={'GROQ_API_KEY': bool(env.get('GROQ_API_KEY'))},
        preflight=preflight, frozen_initial_payload_checks=references,
        zero_science_guards_validated=True, provider_calls=0, catalog_requests=0)


def _payload_guard(scenario, checks):
    probes = _probe_contracts(scenario)
    registry, _ = guarded_registry()
    def validate(*, prompt, response_schema):
        payload = json.loads(prompt)
        phase = _phase(payload)
        base = dict(payload)
        recovery = [key for key in ('repair', 'failover') if key in base]
        for key in recovery: base.pop(key)
        base_prompt = prompt
        if scenario.surface == 'layer2' and phase == 'stage_b':
            alternatives = response_schema['$defs']['step']['anyOf']
            enums = [alternative['properties']['tool']['enum'] for alternative in alternatives]
            if not all(len(enum) == 1 for enum in enums):
                raise ValueError('Scoped schema must identify each offered tool exactly.')
            names = tuple(sorted(enum[0] for enum in enums))
            expected = _layer2_contract(scenario, registry, names)
        else:
            expected = probes[phase]
        if recovery:
            base_prompt = json.dumps(base, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ':'))
            expected = (json.dumps(json.loads(expected[0]), ensure_ascii=False,
                        allow_nan=False, sort_keys=True, separators=(',', ':')), expected[1])
        if _pair(base_prompt, response_schema) != _pair(*expected):
            raise ValueError('Unexpected frozen prompt or schema fingerprint.')
        checks.append(dict(surface=phase, prompt_fingerprint=fingerprint(prompt),
            base_prompt_fingerprint=fingerprint(base_prompt), schema_fingerprint=fingerprint(response_schema),
            recovery_envelope=recovery, validated_before_sdk=True))
    return validate


def resume_breadth(*, original, output, factory=None, environment=None,
                  min_model_spacing_seconds=None):
    saved, identity = _original(original)
    env = os.environ if environment is None else environment
    private = _private_values(env)
    spacing = (60 if factory is None else 0) if min_model_spacing_seconds is None else min_model_spacing_seconds
    if type(spacing) not in (int, float) or not 0 <= spacing <= 60:
        raise ValueError('Independent-attempt spacing must be bounded to 0–60 seconds.')
    plan = resume_validate(original, environment=env, require_credentials=factory is None)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / '.gitignore').write_text('*\n')
    def save(name, value):
        path = output / name
        temp = path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(_safe_json(value, private), indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        temp.replace(path)
    save('resume_plan.json', plan)
    merged = json.loads(json.dumps(saved))
    merged['phase'] = 'Q2.1+Q2.1b'
    merged['historical_halts'] = {'Q2.1': saved['blocked_providers']}
    report = dict(phase='Q2.1b', baseline=BASELINE, original_identity=identity, resume_plan=plan,
        results=[], payload_checks={}, blocked_candidates={}, blocked_providers={},
        waits=[], catalog_requests=0, min_model_spacing_seconds=spacing,
        comparison_manifest=comparison_manifest(), stability='not_yet_measured', q2_2_run=False)
    sources = {f"{r['scenario_id']}:{r['candidate']['candidate_id']}": 'Q2.1'
        for r in saved['results'] if r['attempt'] is not None}
    rates, deadlines = {}, {}
    rows = {(r['scenario_id'], r['candidate']['candidate_id']): i for i, r in enumerate(saved['results'])}
    for continuation_index, (scenario_id, candidate_id) in enumerate(plan['planned_cells'], 1):
        scenario = next(s for s in canonical_scenarios() if s.scenario_id == scenario_id)
        candidate = next(c for c in COHORT if c.candidate_id == candidate_id)
        key = (scenario_id, candidate_id)
        index = rows[key]
        halt = report['blocked_providers'].get(candidate.provider_id) or report['blocked_candidates'].get(candidate_id)
        if halt:
            row = dict(saved['results'][index], reason=halt, attempt_id='completion-1')
        else:
            wait = max(0.0, deadlines.get(candidate_id, 0) - time.monotonic())
            if wait:
                assert wait <= 60
                report['waits'].append(dict(candidate_id=candidate_id, before_scenario=scenario_id, seconds=wait))
                print(json.dumps(dict(wait_seconds=round(wait, 3), candidate=candidate_id, before_scenario=scenario_id)), flush=True)
                time.sleep(wait)
            checks = []
            guard = _payload_guard(scenario, checks)
            def make(c, availability, max_calls):
                if factory is not None:
                    return factory(c, availability, max_calls, before_complete=guard)
                return build_transport(c, availability=availability, environment=env,
                                       max_calls=max_calls, before_complete=guard)
            cell = run_breadth(output=output / 'attempts' / f'{scenario_id}-{candidate_id}',
                scenarios=(scenario,), candidates=(candidate,), discovery=saved['discovery'],
                factory=make, environment=env, attempt_id='completion-1')
            row = cell['results'][0]
            row['attempt_index'] = saved['results'][index]['attempt_index']
            report['payload_checks'][f'{scenario_id}:{candidate_id}'] = checks
            reason, rates[candidate_id], trigger = _limit_evidence(row['diagnostics'], rates.get(candidate_id, 0))
            if reason:
                scope = report['blocked_providers'] if _provider_wide(trigger) else report['blocked_candidates']
                scope[candidate.provider_id if _provider_wide(trigger) else candidate_id] = dict(
                    reason=reason, phase='Q2.1b', triggering_scenario=scenario_id,
                    triggering_candidate=candidate_id, evidence=trigger)
            elif row['diagnostics']:
                last = row['diagnostics'][-1]
                deadlines[candidate_id] = max(_wait_until(last),
                    (last.get('request_finished_monotonic') or time.monotonic()) + spacing)
        report['results'].append(row)
        merged['results'][index] = row
        sources[f'{scenario_id}:{candidate_id}'] = 'Q2.1b'
        report['completed_planned_cells'] = continuation_index
        merged['sources'] = sources
        merged['continuation_phase'] = 'Q2.1b'
        save('continuation.json', report)
        save('merged_results.json', merged)
    for candidate in COHORT:
        rows_for_candidate = [r for r in report['results'] if r['candidate']['candidate_id'] == candidate.candidate_id]
        report.setdefault('candidate_counts', {})[candidate.candidate_id] = dict(
            attempts=sum(r['attempted'] for r in rows_for_candidate),
            completions=sum(r['provider_completions'] for r in rows_for_candidate),
            recovery_calls=sum(r['recovery_calls'] for r in rows_for_candidate),
            provider_failures=sum(d['provider_outcome'] == 'failed' for r in rows_for_candidate for d in r['diagnostics']),
            statuses=dict(Counter(r['status'] for r in rows_for_candidate)))
    merged['blocked_candidates'] = report['blocked_candidates']
    merged['blocked_providers'] = report['blocked_providers']
    merged['candidate_counts'] = {}
    for candidate in COHORT:
        all_rows = [r for r in merged['results'] if r['candidate']['candidate_id'] == candidate.candidate_id]
        merged['candidate_counts'][candidate.candidate_id] = dict(
            attempts=sum(r['attempted'] for r in all_rows), completions=sum(r['provider_completions'] for r in all_rows),
            recovery_calls=sum(r['recovery_calls'] for r in all_rows),
            provider_failures=sum(d['provider_outcome'] == 'failed' for r in all_rows for d in r['diagnostics']),
            statuses=dict(Counter(r['status'] for r in all_rows)))
    merged['human_review_queue'] = [dict(scenario_id=r['scenario_id'], candidate=r['candidate'],
        status=r['status'], explanation=r['attempt'].get('explanation'), support=r['attempt'].get('support'),
        guidance_candidates=r['attempt'].get('guidance_candidates'), fixture_provenance=r['attempt']['fixture_provenance'])
        for r in merged['results'] if r['attempt'] and r['attempt']['human_review_required']
        and r['status'] == 'HUMAN_REVIEW_PENDING']
    merged['comparative_breadth_complete'] = all(r['attempt'] is not None for r in merged['results'])
    for old, new in zip(saved['results'], merged['results']):
        if old['attempt'] is not None: assert old == new
    report['complete'] = len(report['results']) == 27
    report['safety'] = merged['safety']
    save('continuation.json', report)
    save('merged_results.json', merged)
    save('human_review.json', merged['human_review_queue'])
    return _safe_json(dict(continuation=report, merged=merged), private)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', type=Path, help='Preserved original Q2.1 results or directory.')
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps(resume_validate(args.resume, require_credentials=True) if args.resume
                         else dry_validate(require_credentials=True), indent=2))
    elif args.live and args.output is not None:
        if args.resume: resume_breadth(original=args.resume, output=args.output)
        else: run_breadth(output=args.output)
    else:
        parser.error('Use --dry-run or explicit --live with a new --output directory.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
