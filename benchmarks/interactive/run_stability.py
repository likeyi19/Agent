"""Q2.2: eight frozen GPT-OSS repeats using the existing breadth machinery."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import time
from unittest.mock import patch

from agent.application import turn_decisions as decisions
from agent.orchestration.planning_scope import PlanningScope, capability_index, fingerprint
from agent.schemas.orchestration import _serialize
from benchmarks.planner.benchmark import guarded_registry

from . import harness
from . import run_breadth as breadth
from .breadth_transport import COHORT, PayloadValidationFailure, _reporting_error, build_transport
from .fixtures import build_context
from .scenarios import scenario_by_id

SELECTED_SCENARIOS = ('I01', 'I10', 'I08', 'I14')
CANDIDATE = COHORT[0]
COMPLETION_CEILING = 16
HUMAN_REVIEW_PATH = Path('evals/interactive_model_qualification_q2_1c_2026-10-07/human_review_disposition.json')
PRESERVED_BREADTH_FILES = ('benchmarks/interactive/run_breadth.py',
    'benchmarks/interactive/breadth_transport.py', 'tests/benchmarks/test_interactive_breadth.py')


def _baseline():
    checkout = breadth.repository_baseline(require_clean=True)
    refs = {name: subprocess.check_output(['git', 'rev-parse', name], text=True).strip()
            for name in ('HEAD', 'main', 'origin/main')}
    branch = subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip()
    tracked = subprocess.check_output(['git', 'diff', '--name-only'], text=True).strip()
    index = subprocess.check_output(['git', 'diff', '--cached', '--name-only'], text=True).strip()
    if branch != 'main' or set(refs.values()) != {checkout['infrastructure_commit']} or tracked or index:
        raise ValueError('Frozen main baseline and clean tracked tree/index are required.')
    return dict(branch=branch, refs=refs, **checkout)


def _source_identity():
    manifest = breadth.comparison_manifest()
    return dict(groups=manifest.get('source_fingerprints', {}), preserved_breadth={
        name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in PRESERVED_BREADTH_FILES})


def _interaction(context):
    return {f.name: _serialize(getattr(context.interaction, f.name)) for f in fields(context.interaction)}


def _frozen_context(scenario):
    registry, guard = guarded_registry()
    with TemporaryDirectory(prefix='agent-stability-context-') as root:
        with harness.zero_science(guard, read_only=True) as safety:
            context = build_context(Path(root), scenario.fixture, registry, utterance=scenario.utterance,
                execution_inputs=None if scenario.request is None else scenario.request.inputs,
                include_interaction=scenario.surface != 'layer2')
            result = dict(scenario_id=scenario.scenario_id, scenario_fingerprint=scenario.fingerprint(),
                utterance=scenario.utterance, public_context_fingerprint=fingerprint(context.public),
                private_captured_context_fingerprint=fingerprint(_interaction(context)),
                request_fingerprint=None if scenario.request is None else fingerprint(scenario.request.to_dict()),
                registry_fingerprint=PlanningScope(registry, tuple(capability_index(registry))).scope_fingerprint,
                private_context_provenance='Reconstructed exact committed fixture; historical private snapshots were not persisted.')
        assert not any(safety.values())
    return result


def _review(path):
    raw = Path(path).read_bytes()
    review = json.loads(raw)
    if (review.get('scenario_id') != 'I08' or review.get('candidate_id') != CANDIDATE.candidate_id
            or review.get('external_human_semantic_review') != 'PASS'
            or review.get('historical_status') != 'HUMAN_REVIEW_PENDING'):
        raise ValueError('I08 needs the separate accepted operator review disposition.')
    return dict(path=str(Path(path).resolve()), sha256=hashlib.sha256(raw).hexdigest(),
                human_semantic_review='PASS', historical_status='HUMAN_REVIEW_PENDING')


def stability_validate(original, *, environment=None, require_credentials=False,
                       human_review_path=HUMAN_REVIEW_PATH):
    saved, identity = breadth._original(original)
    env = os.environ if environment is None else environment
    checkout = breadth.repository_baseline(require_clean=require_credentials)
    if (saved.get('baseline') != breadth.BASELINE or len(saved.get('results', ())) != 42
            or not saved.get('comparative_breadth_complete')):
        raise ValueError('Q2.2 requires the preserved, actually completed 42-cell breadth ledger.')
    if require_credentials and not env.get('GROQ_API_KEY', '').strip():
        raise ValueError('Groq credential must be present before live stability repeats.')
    manifest = breadth.comparison_manifest()
    for group in ('src/agent', 'benchmarks/planner'):
        old = saved.get('comparison_manifest', {}).get('source_fingerprints', {}).get(group)
        if old is not None and manifest.get('source_fingerprints', {}).get(group) != old:
            raise ValueError('Frozen production/planner sources changed.')
    review = _review(human_review_path)
    registry, _ = guarded_registry()
    frozen, references, historical = {}, [], []
    for scenario_id in SELECTED_SCENARIOS:
        scenario = scenario_by_id(scenario_id)
        rows = [r for r in saved['results'] if r['scenario_id'] == scenario_id
                and r['candidate']['candidate_id'] == CANDIDATE.candidate_id]
        if len(rows) != 1:
            raise ValueError('Exactly one historical attempt is required for each selected case.')
        row = rows[0]
        attempt = row['attempt']
        if not row['attempted'] or not attempt:
            raise ValueError('Selected historical cell has no completed attempt.')
        breadth.validate_call_identity(row, scenario, CANDIDATE)
        expected = _frozen_context(scenario)
        if (row['candidate'] != CANDIDATE.manifest() or not row['attempted'] or not attempt
                or attempt['scenario_id'] != scenario_id or attempt['attempt_id'] != row['attempt_id']
                or attempt['candidate'] != harness.candidate_identity(CANDIDATE)
                or attempt['scenario_fingerprint'] != expected['scenario_fingerprint']
                or attempt['context_fingerprint'] != expected['public_context_fingerprint']
                or attempt['request_fingerprint'] != expected['request_fingerprint']
                or attempt['registry_fingerprint'] != expected['registry_fingerprint']
                or any(attempt['safety'].values())):
            raise ValueError('Historical selected-case identity/context/safety changed.')
        if scenario_id == 'I08':
            disposition = json.loads(Path(human_review_path).read_text())
            if (row['status'] != 'HUMAN_REVIEW_PENDING' or not attempt['contract_success']
                    or not attempt['admission_success'] or not attempt.get('explanation')
                    or ('retained_answer' in disposition
                        and disposition['retained_answer'] != attempt['explanation'])):
                raise ValueError('Operator review does not match the admitted historical I08 Answer.')
        assert _frozen_context(scenario) == expected
        probes = breadth._probe_contracts(scenario)
        for call in attempt['calls']:
            if call['recovery_kind'] != 'initial':
                continue
            if scenario.surface == 'layer2' and call['surface'] == 'stage_b':
                selected = next(c['capability_ids'] for c in attempt['calls'] if c['surface'] == 'stage_a')
                pair = breadth._layer2_contract(scenario, registry, PlanningScope(registry, tuple(selected)).visible_tool_names)
            else:
                pair = probes[call['surface']]
            if breadth._pair(*pair) != (call['prompt_fingerprint'], call['schema_fingerprint']):
                raise ValueError('Historical frozen prompt/schema does not reproduce.')
            references.append(dict(scenario_id=scenario_id, surface=call['surface'],
                prompt_fingerprint=call['prompt_fingerprint'], schema_fingerprint=call['schema_fingerprint']))
        frozen[scenario_id] = expected
        historical.append(dict(scenario_id=scenario_id, attempt_number=1, status=row['status'],
            source=identity, row_fingerprint=fingerprint(row), historical_attempt_id=row['attempt_id']))
    schedule = [dict(scenario_id=s, attempt_id=f'stability-{n}', attempt_number=n,
                     candidate_id=CANDIDATE.candidate_id) for s in SELECTED_SCENARIOS for n in (2, 3)]
    assert len(schedule) == 8
    return dict(phase='Q2.2', baseline=breadth.BASELINE,
        infrastructure_commit=checkout['infrastructure_commit'], candidate=CANDIDATE.manifest(),
        selected_scenarios=list(SELECTED_SCENARIOS), historical_attempts=historical,
        historical_attempt_count_per_case=1, planned_attempts=schedule, total_new_attempts=8,
        expected_direct_completions=12, conservative_completion_ceiling=COMPLETION_CEILING,
        planned_qwen_calls=0, planned_nemotron_calls=0, catalog_requests=0,
        credentials_present={'GROQ_API_KEY': bool(env.get('GROQ_API_KEY'))},
        frozen_contexts=frozen, frozen_initial_payload_checks=references, human_review=review,
        zero_science_guards_validated=True, source_identity=_source_identity(),
        git_baseline=_baseline() if require_credentials else None,
        stage_b_comparison='I01 scope is provider-selected. Each Stage-B payload must equal unchanged production builders for that exact accepted scope; scope changes are recorded, never forced to historical scope.')


class _Observed:
    """Retain only bounded typed interpreter choices; return every response unchanged."""
    def __init__(self, inner, observations):
        self.inner, self.observations = inner, observations

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def complete(self, *, prompt, response_schema):
        response = self.inner.complete(prompt=prompt, response_schema=response_schema)
        if breadth._phase(json.loads(prompt)) == 'interpreter':
            observation = dict(surface='interpreter', typed_valid=False)
            try:
                decision = decisions.parse_decision(response)
                observation.update(typed_valid=True, kind=harness._decision_kind(decision))
                if isinstance(decision, decisions.Answer):
                    observation['intent'] = decision.intent
                    if decision.scientific is not None:
                        q = decision.scientific
                        observation.update(target=dict(output=q.target.output, subject=q.target.subject),
                            comparison=None if q.comparison is None else dict(
                                output=q.comparison.output, subject=q.comparison.subject), focus=q.focus)
                elif isinstance(decision, decisions.Execute):
                    observation.update(operation=decision.operation, target=decision.target)
                elif isinstance(decision, decisions.Clarify):
                    observation['reason'] = decision.reason
            except decisions.IntentError:
                pass
            self.observations.append(observation)
        return response


def _directness(row):
    return 'recovered' if row.get('recovery_calls', 0) else 'direct'


def _matrix(saved, report):
    result = {}
    for scenario_id in SELECTED_SCENARIOS:
        historical = next(r for r in saved['results'] if r['scenario_id'] == scenario_id
            and r['candidate']['candidate_id'] == CANDIDATE.candidate_id)
        first = dict(attempt_number=1, historical=True, source=report['plan']['historical_attempts'][
            SELECTED_SCENARIOS.index(scenario_id)], final_status=historical['status'],
            direct_or_recovered=_directness(historical))
        if scenario_id == 'I08':
            first['external_human_semantic_review'] = 'PASS'
        entries = [first]
        entries.extend(dict(attempt_number=r['attempt_number'], historical=False,
            attempt_id=r['attempt_id'], final_status=r['status'], direct_or_recovered=_directness(r),
            record_path=f"attempts/{scenario_id}-{r['attempt_id']}/results.json")
            for r in report['results'] if r['scenario_id'] == scenario_id)
        result[scenario_id] = entries
    return result


def run_stability(*, original, output, factory=None, environment=None, clock=time.monotonic,
                  sleeper=time.sleep, spacing=60, human_review_path=HUMAN_REVIEW_PATH):
    if type(spacing) not in (int, float) or not 0 <= spacing <= 60:
        raise ValueError('Independent attempt spacing must be bounded to 0–60 seconds.')
    saved, identity = breadth._original(original)
    env = os.environ if environment is None else environment
    private = breadth._private_values(env)
    plan = stability_validate(original, environment=env, require_credentials=factory is None,
                              human_review_path=human_review_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / '.gitignore').write_text('*\n')

    def save(name, value):
        path = output / name
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(_reporting_error(breadth._safe_json(value, private)),
            indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        temporary.replace(path)

    report = dict(phase='Q2.2', baseline=breadth.BASELINE, plan=plan, original_identity=identity,
        results=[], payload_checks={}, frozen_context_checks={}, observations={}, waits=[],
        blocked_models={}, blocked_providers={}, catalog_requests=0, min_model_spacing_seconds=spacing,
        human_review_queue=[], stability_matrix={}, counts={}, complete=False,
        safety=dict(scientific_calls=0, executor_entries=0, reconstruction_entries=0, scientific_step_results=0))
    save('prelive_plan.json', plan)
    reserved, rate_count, deadline = 0, 0, 0.0
    original_build = harness.build_context
    original_safe_json = breadth._safe_json
    for item in plan['planned_attempts']:
        scenario = scenario_by_id(item['scenario_id'])
        attempt_id = item['attempt_id']
        key = f'{scenario.scenario_id}:{attempt_id}'
        halt = report['blocked_models'].get(CANDIDATE.candidate_id) or report['blocked_providers'].get(CANDIDATE.provider_id)
        if halt:
            row = dict(scenario_id=scenario.scenario_id, attempt_id=attempt_id,
                attempt_number=item['attempt_number'], candidate=CANDIDATE.manifest(),
                status='NOT_RUN_PROVIDER_LIMIT', reason=halt, attempted=False, attempt=None,
                provider_completions=0, recovery_calls=0, diagnostics=[])
        else:
            wait = max(0.0, deadline - clock())
            if wait:
                assert wait <= 60
                report['waits'].append(dict(before_scenario=scenario.scenario_id,
                    before_attempt_id=attempt_id, seconds=wait))
                print(json.dumps(dict(wait_seconds=round(wait, 3), before=key)), flush=True)
                sleeper(wait)
            checks, contexts, observations = [], [], []
            expected = plan['frozen_contexts'][scenario.scenario_id]
            payload_guard = breadth._payload_guard(scenario, checks)

            def context_build(*args, **kwargs):
                context = original_build(*args, **kwargs)
                check = dict(public_context_fingerprint=fingerprint(context.public),
                    private_captured_context_fingerprint=fingerprint(_interaction(context)))
                if any(check[k] != expected[k] for k in check):
                    raise PayloadValidationFailure('Frozen captured context differs before model transmission.')
                contexts.append(check)
                return context

            def before_complete(*, prompt, response_schema):
                nonlocal reserved
                if not contexts or reserved >= COMPLETION_CEILING:
                    raise PayloadValidationFailure('Context/budget precondition failed before model transmission.')
                if _source_identity() != plan['source_identity']:
                    raise PayloadValidationFailure('Frozen source identity changed before model transmission.')
                if factory is None:
                    _baseline()
                payload_guard(prompt=prompt, response_schema=response_schema)
                reserved += 1

            def make(candidate, availability, max_calls, *, before_complete=None):
                assert candidate == CANDIDATE
                runtime = (factory(candidate, availability, max_calls, before_complete=before_complete)
                    if factory is not None else build_transport(candidate, availability=availability,
                        environment=env, max_calls=max_calls, before_complete=before_complete))
                return _Observed(runtime, observations)

            def make_guarded(candidate, availability, max_calls):
                return make(candidate, availability, max_calls, before_complete=before_complete)

            try:
                with patch.object(harness, 'build_context', context_build), patch.object(
                        breadth, '_safe_json', lambda value, secrets: _reporting_error(original_safe_json(value, secrets))):
                    cell = breadth.run_breadth(output=output / 'attempts' / f'{scenario.scenario_id}-{attempt_id}',
                        scenarios=(scenario,), candidates=(CANDIDATE,), discovery=saved['discovery'],
                        factory=make_guarded, environment=env, attempt_id=attempt_id)
            except BaseException:
                report['payload_checks'][key] = checks
                report['frozen_context_checks'][key] = contexts
                report['observations'][key] = observations
                report['hard_stop_before_additional_calls'] = True
                save('results.json', report)
                raise
            row = cell['results'][0]
            row['attempt_number'] = item['attempt_number']
            report['payload_checks'][key] = checks
            report['frozen_context_checks'][key] = contexts
            report['observations'][key] = observations
            if row['attempt'] is not None and any(row['attempt']['safety'].values()):
                raise harness.ZeroScienceViolation('Stability attempted scientific execution.')
            reason, rate_count, trigger = breadth._limit_evidence(row['diagnostics'], rate_count)
            if reason:
                scope = report['blocked_providers'] if breadth._provider_wide(trigger) else report['blocked_models']
                scope[CANDIDATE.provider_id if breadth._provider_wide(trigger) else CANDIDATE.candidate_id] = dict(
                    reason=reason, triggering_scenario=scenario.scenario_id, triggering_attempt=attempt_id,
                    evidence=trigger)
            last = row['diagnostics'][-1] if row['diagnostics'] else None
            deadline = clock() + max(spacing, 0 if last is None or last['provider_outcome'] == 'returned'
                                     else breadth._delay_seconds(last))
        report['results'].append(row)
        if (row['attempt'] is not None and scenario.human_review_required
                and row['status'] == 'HUMAN_REVIEW_PENDING'):
            report['human_review_queue'].append(dict(scenario_id=scenario.scenario_id,
                attempt_id=attempt_id, attempt_number=item['attempt_number'], status=row['status'],
                explanation=row['attempt'].get('explanation'), support=row['attempt'].get('support'),
                human_semantic_review=None, fixture_provenance=row['attempt']['fixture_provenance']))
        report['stability_matrix'] = _matrix(saved, report)
        report['counts'] = dict(attempts=sum(r['attempted'] for r in report['results']),
            completions=sum(r['provider_completions'] for r in report['results']),
            recovery_calls=sum(r['recovery_calls'] for r in report['results']),
            operational_events=sum(d['provider_outcome'] == 'failed' for r in report['results'] for d in r['diagnostics']),
            statuses=dict(Counter(r['status'] for r in report['results'])), reserved_pre_sdk_checks=reserved)
        save('results.json', report)
        save('frozen_input_audit.json', dict(plan=plan, contexts=report['frozen_context_checks'],
            payloads=report['payload_checks']))
        save('stability_matrix.json', report['stability_matrix'])
        save('human_review.json', report['human_review_queue'])
    report['complete'] = len(report['results']) == 8
    report['all_eight_attempted'] = report['counts']['attempts'] == 8
    assert report['counts']['completions'] <= COMPLETION_CEILING
    save('results.json', report)
    return _reporting_error(breadth._safe_json(report, private))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--human-review', type=Path, default=HUMAN_REVIEW_PATH,
                        help='Separate operator disposition for historical I08; never changes its result.')
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps(stability_validate(args.original, require_credentials=True,
                                           human_review_path=args.human_review), indent=2))
    elif args.live and args.output is not None:
        run_stability(original=args.original, output=args.output, human_review_path=args.human_review)
    else:
        parser.error('Use --dry-run or explicit --live with a new --output directory.')


if __name__ == '__main__':
    main()
