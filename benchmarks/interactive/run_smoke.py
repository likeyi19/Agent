"""Explicitly authorized, one-call-per-candidate transport/contract smoke.

Model discovery precedes generation. A parsed decision is readiness evidence,
not model qualification. This runner has no repeat, fallback or tuning option.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

from agent.orchestration.planning_model import PlanningModelError
from agent.orchestration.planning_scope import fingerprint

from .candidates import CANDIDATES, build_candidate
from .harness import run_attempt
from .scenarios import canonical_scenarios


def comparison_manifest():
    root = Path(__file__).resolve().parents[2]
    groups = ('src/agent', 'benchmarks/planner', 'benchmarks/interactive')
    return dict(
        repository_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        source_fingerprints={group: fingerprint({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root / group).rglob('*.py'))}) for group in groups},
        scenarios_fingerprint=fingerprint([s.to_dict() for s in canonical_scenarios()]),
        dependencies={name: importlib.metadata.version(name) for name in ('openai', 'google-genai')},
        sdk_retries=0, sampling_controls='production_defaults_unspecified',
        semantics='unchanged_agent_contracts', qualification_claim=False,
        completion_limit_per_candidate=1, repetitions=1,
    )


def smoke(*, output, candidates=CANDIDATES, factory=build_candidate):
    """One fresh interpreter attempt for each available exact candidate."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    case = canonical_scenarios()[0]
    scenario = replace(case, surface='interpreter', expected={'kind': 'execute', 'tool': 'inspect_scATAC'})
    report = dict(schema_version=1, purpose='transport_and_contract_readiness_only',
                  comparison_manifest=comparison_manifest(), candidates=[])
    path = output / 'results.json'
    def save():
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    save()
    for candidate in candidates:
        runtime = None
        row = dict(candidate=candidate.manifest(), status=None, availability=None,
                   availability_requests=0, completion_calls=0, attempt=None,
                   scientific_calls=0, execution_entry_attempts=0, scientific_step_results=0)
        try:
            runtime = factory(candidate)
            row['candidate'] = runtime.manifest()
            availability = runtime.validate_availability()
            row['availability'] = availability.to_dict()
            if availability.status != 'available':
                row['status'] = ('provider_model_unavailable' if availability.status == 'unavailable'
                                 else 'operational_provider_failure')
            else:
                attempt = run_attempt(scenario, runtime.candidate, model=runtime,
                                      attempt_id='smoke-1').to_dict()
                row['attempt'] = attempt
                row.update({k: attempt['safety'][k] for k in (
                    'scientific_calls', 'execution_entry_attempts', 'scientific_step_results')})
                row['status'] = ('structured_smoke_succeeded' if attempt['contract_success'] is True
                    else 'operational_provider_failure' if attempt['failure_category'] == 'operational'
                    else 'incompatible_transport_schema_behavior')
        except PlanningModelError as exc:
            codes = {'CANDIDATE_ENDPOINT_MISSING', 'PLANNING_PROVIDER_CONFIGURATION_FAILED',
                     'PLANNING_PROVIDER_DEPENDENCY_MISSING'}
            row.update(status='operational_provider_failure',
                       configuration_error_code=exc.code if exc.code in codes else 'PLANNING_PROVIDER_ERROR')
        finally:
            if runtime is not None:
                row['availability_requests'] = runtime.availability_requests
                row['completion_calls'] = runtime.completion_calls
                if runtime.completion_calls > 1:
                    raise AssertionError('Smoke exceeded its candidate completion bound')
                runtime.close()
        report['candidates'].append(row)
        save()
        print(json.dumps(dict(candidate=candidate.candidate_id, status=row['status'],
            availability_code=None if row['availability'] is None else row['availability']['code'],
            completion_calls=row['completion_calls'], availability_requests=row['availability_requests'],
            contract_success=None if row['attempt'] is None else row['attempt']['contract_success']), sort_keys=True), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Authorize exact five-candidate availability and smoke calls.')
    parser.add_argument('--output', type=Path, required=True, help='New local directory; existing results are never overwritten.')
    args = parser.parse_args(argv)
    if not args.live:
        parser.error('Live smoke requires explicit --live authorization.')
    smoke(output=args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
