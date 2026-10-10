"""Bounded live A3/A4 mouse-continuation requalification for UA3.5.4c.

Reuses the frozen transport, recording, policy and scientific execution boundary.
Fresh catalog discovery precedes the two ordinary interpreted submissions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from agent.application.local_resources import select_epizoo_resource
from agent.orchestration.registry import build_default_tool_registry
from agent.providers import PlanningModelFactoryRegistry
from agent.web.config import load_web_configuration

from . import run_ua354a as base, run_ua354b as previous
from .diagnostics import sanitize_provider_error
from .ua354a_fixtures import BoundedScienceExecutor


SCHEDULE = ('A3', 'A4')


class Runner(previous.Runner):
    def __init__(self, output):
        super().__init__(output)
        self.report.pop('human_answer_fixture', None)
        self.report.update(kind='UA3.5.4c', schedule=list(SCHEDULE),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            ua354b_runner_sha256=hashlib.sha256(Path(previous.__file__).read_bytes()).hexdigest(),
            reused_matrix='UA3.5.4b A3/A4 only',
            evaluation_schedule_override=list(SCHEDULE),
            scientific_fixture_scope='Retained registered H5AD only; no QC/neighbors construction.')
        self.save()

    def run(self, args):
        config = load_web_configuration(args.operator_config)
        registry = build_default_tool_registry()
        executor = BoundedScienceExecutor(registry)
        workspace = Path(args.workspace).absolute()
        workspace.mkdir(parents=True, exist_ok=False)
        app = previous.configured_application(workspace / 'application',
            model_profiles=(base.CANDIDATE.profile(),), default_profile_id=base.CANDIDATE.profile_id,
            planning_model_factory_registry=PlanningModelFactoryRegistry({'openrouter': self.factory}),
            registry=registry, executor=executor,
            approved_source_roots=(Path(args.source_h5ad).absolute().parent,),
            epizoo_resources=config.epizoo_resources)
        source = app.resources.register('ua354a-h5ad', args.source_h5ad,
            label='Registered single-cell H5AD', attribution='Retained UA3.2.2 bounded acceptance source')
        inputs, error = select_epizoo_resource(dict(device='cuda'), None, config.epizoo_resources)
        bound = app.resources.compose_h5ad(source.resource_id, inputs, resource_selection_error=error)
        self.artifact('fixtures.json', dict(source_registration=source.to_dict(),
            resource_choices=[resource.choice() for resource in config.epizoo_resources]))
        if args.fixtures_json:
            self.report['prior_fixture_artifact'] = dict(path=str(Path(args.fixtures_json).absolute()),
                sha256=hashlib.sha256(Path(args.fixtures_json).read_bytes()).hexdigest(),
                recorded_metadata_only=True, scientific_reconstruction=False)
            self.save()
        app.create_session('ap')
        # Only the evaluation module's admitted case matrix changes. Its guard,
        # recording and ordinary Application calls retain their existing bounds.
        with patch.object(base, 'SCHEDULE', SCHEDULE):
            self.turn(app, 'A3', 'ap', 'Analyze this dataset with the available EpiZoo model.',
                      registered_input=bound)
            self.dependent(app, 'A4', 'ap', 'The cells came from mice.', 'species')
        self.artifact('execution-boundary.json', executor.observations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('output', 'workspace', 'source-h5ad', 'operator-config'):
        parser.add_argument('--' + argument, required=True)
    parser.add_argument('--fixtures-json', help='Record unchanged prior fixture metadata bytes only.')
    args = parser.parse_args()
    runner = Runner(args.output)
    try:
        runner.discovery()
        runner.run(args)
    except BaseException as exc:
        runner.report['hard_stop'] = dict(type=type(exc).__name__,
            diagnostic=sanitize_provider_error(exc, secrets=runner.private))
    finally:
        runner.save()
    print(json.dumps(dict(output=str(runner.output),
        model_interface_calls=len(runner.report['calls']),
        provider_completions=sum((call.get('provider') or {}).get('completion_calls', 0)
                                 for call in runner.report['calls']),
        hard_stop=runner.report['hard_stop']), ensure_ascii=False), flush=True)
    return 2 if runner.report['hard_stop'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
