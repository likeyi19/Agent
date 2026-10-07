"""Explicit one-catalog, one-Interpreter-per-exact-free-candidate admission smoke."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from benchmarks.planner.benchmark import guarded_registry
from agent.orchestration.planning_model import PlanningModelError
from .candidates import Availability, build_candidate
from .diagnostics import sanitize_provider_error
from .harness import zero_science
from .openrouter import (
    OPENROUTER_CANDIDATES, _candidates, build_openrouter_candidate,
    classify_readiness, discover_models,
)
from .run_smoke import smoke as shared_smoke


def smoke(*, output, candidates=OPENROUTER_CANDIDATES, client=None, environment=None):
    candidates = _candidates(candidates)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / '.gitignore').write_text('*\n')
    env = os.environ if environment is None else environment
    secrets = tuple(value for key, value in env.items() if key.endswith('_API_KEY') and value)
    sensitive = tuple(value for key, value in env.items() if value and any(label in key.upper()
        for label in ('WORKSPACE_ID', 'WORK_SPACE_ID', 'PROJECT_ID', 'ACCOUNT_ID', 'CONSUMER')))
    owner = None
    runtimes = {}
    try:
        discovery = None
        if client is None:
            try:
                owner = build_candidate(candidates[0], environment=env)
                client = owner._client
            except PlanningModelError as exc:
                diagnosis = sanitize_provider_error(exc, secrets=secrets, sensitive_values=sensitive)
                discovery = dict(catalog_requests=0, candidates=[dict(
                    candidate=candidate.manifest(), discovery_result='DISCOVERY_OPERATIONAL_FAILURE',
                    availability=Availability('operational_failure', exc.code, candidate.model_id,
                                              'models.list').to_dict(),
                    supported_parameters=None, pricing={}, diagnosis=diagnosis) for candidate in candidates])
        _, guard = guarded_registry()
        with zero_science(guard, read_only=True) as counters:
            if discovery is None:
                discovery = discover_models(client, candidates=candidates,
                                            secrets=secrets, sensitive_values=sensitive)
        discovery['safety'] = counters
        (output / 'discovery.json').write_text(json.dumps(discovery, indent=2, allow_nan=False) + '\n')
        by_id = {row['candidate']['candidate_id']: row for row in discovery['candidates']}

        def factory(candidate):
            runtime = build_openrouter_candidate(candidate,
                availability=Availability(**by_id[candidate.candidate_id]['availability']),
                client=client, environment=env, secrets=secrets, sensitive_values=sensitive)
            runtimes[candidate.candidate_id] = runtime
            return runtime

        results = shared_smoke(output=output / 'interpreter', candidates=candidates, factory=factory)
        rows = []
        for row in results['candidates']:
            candidate_id = row['candidate']['candidate_id']
            evidence = by_id[candidate_id]
            runtime = runtimes.get(candidate_id)
            diagnostic = runtime.transport_diagnostic if runtime is not None else None
            readiness = classify_readiness(row, evidence, diagnostic)
            routing = runtime.routing_observation if runtime is not None else None
            if routing and (routing['pipeline_modified'] or routing['requested_model_matches'] is False
                            or (routing['upstream_attempt'] or 0) > 1):
                readiness = dict(classification='EXTERNAL_CONFIGURATION_REQUIRED', failure_category='operational')
            rows.append(dict(candidate=row['candidate'], discovery=evidence,
                shared_catalog_evidence=True, discovery_requests=0,
                generation_requests=row['completion_calls'], retries=0,
                diagnostic=diagnostic, routing_observation=routing,
                configuration_error_code=row.get('configuration_error_code'),
                harness_status=row['status'], attempt=row['attempt'],
                **readiness))
            assert row['completion_calls'] <= 1 and row['availability_requests'] == 0
        report = dict(schema_version=1, purpose='Q1d_exact_free_model_admission_only',
            endpoint_family='openrouter_public_openai_compatible',
            credential_present=bool(env.get('OPENROUTER_API_KEY')),
            transport='existing_generic_strict_chat_completions',
            provider_preferences={'require_parameters': True, 'allow_fallbacks': False,
                                  'max_price': {'prompt': 0, 'completion': 0, 'request': 0}},
            disabled_plugins=['response-healing', 'context-compression', 'web'],
            shared_catalog_requests=discovery['catalog_requests'],
            total_generation_requests=sum(row['generation_requests'] for row in rows),
            retries=0, separate_minimal_probe=False,
            strict_interpreter_is_compatibility_probe=True, candidates=rows,
            comparison_manifest=results['comparison_manifest'],
            scientific_calls=0, execution_entry_attempts=0, scientific_step_results=0,
            reconstruction_entries=0, q2_run=False, qualification_claim=False)
        (output / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        for row in rows:
            print(json.dumps(dict(candidate=row['candidate']['candidate_id'],
                discovery=row['discovery']['discovery_result'], classification=row['classification'],
                generation_requests=row['generation_requests'], diagnostic=row['diagnostic']),
                sort_keys=True), flush=True)
        return report
    finally:
        if owner is not None:
            owner.close()
        elif callable(getattr(client, 'close', None)):
            client.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.live:
        parser.error('OpenRouter admission smoke requires explicit --live authorization.')
    smoke(output=args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
