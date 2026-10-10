"""Bounded UA3.5.4a live qualification through the ordinary interactive facade.

No scripted semantic decisions, production patches, semantic repair or heavy
science. Raw semantic artifacts are private evaluation records, never authority.
Run with the agent Conda environment and PYTHONPATH=src:.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import time

from agent.application import InteractiveAgentApplication
from agent.application.local_resources import select_epizoo_resource
from agent.orchestration.planning_model import PlanningModelError
from agent.orchestration.registry import build_default_tool_registry
from agent.providers.model_registry import PlanningModelFactoryRegistry
from agent.web.config import load_web_configuration
from benchmarks.interactive.candidates import Availability, build_candidate
from benchmarks.interactive.diagnostics import sanitize_provider_error
from benchmarks.interactive.openrouter import build_openrouter_candidate, discover_models
from benchmarks.interactive.run_breadth import _private_values, _safe_json
from benchmarks.interactive.run_frontier import (
    CANDIDATE, COMPLETION_PRICE, OUTPUT_LIMIT, PROMPT_PRICE, input_token_bound,
)

POLICY = {'provider_max_price': {'prompt': 2, 'completion': 10, 'request': 0},
          'max_completion_tokens': OUTPUT_LIMIT}
SCHEDULE = ('A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3',
            'C1', 'C2', 'C3', 'C4', 'N1', 'N2', 'N3')


class Stop(BaseException):
    """Hard evaluation bounds bypass production recovery."""


class SpendingGuard:
    """Existing F1 reservation method with this explicitly bounded matrix."""

    def __init__(self):
        self.reserved = Decimal(0)
        self.rows = []

    def reserve(self, *, request, case):
        provider = request.get('extra_body', {}).get('provider', {})
        fmt = request.get('response_format', {})
        if (case not in SCHEDULE or len(self.rows) >= 48
                or sum(r['case'] == case for r in self.rows) >= 6
                or request.get('model') != CANDIDATE.model_id
                or fmt.get('type') != 'json_schema'
                or fmt.get('json_schema', {}).get('strict') is not True
                or request.get('max_completion_tokens') != OUTPUT_LIMIT
                or provider.get('allow_fallbacks') is not False
                or provider.get('require_parameters') is not True
                or provider.get('max_price') != POLICY['provider_max_price']):
            raise Stop('Exact routing, schema, output bound or call limit changed.')
        bound = input_token_bound(request)
        amount = PROMPT_PRICE * bound + COMPLETION_PRICE * OUTPUT_LIMIT
        if bound > 150000 or self.reserved + amount > Decimal('5.00'):
            raise Stop('Input or aggregate five-dollar reservation exceeded.')
        self.reserved += amount
        self.rows.append(dict(case=case, input_token_upper_bound=bound,
            completion_token_upper_bound=OUTPUT_LIMIT, reserved_cost_usd=str(amount),
            total_reserved_cost_usd=str(self.reserved)))


class Runner:
    def __init__(self, output):
        self.output = Path(output).absolute()
        self.output.mkdir(parents=True, exist_ok=False)
        self.private = _private_values(os.environ)
        self.guard = SpendingGuard()
        self.case = None
        self.availability = None
        self.dispatched_schemas = set()
        self.report = dict(kind='UA3.5.4a', candidate=CANDIDATE.manifest(),
            schedule=list(SCHEDULE), results=[], calls=[], discovery=None,
            hard_stop=None, live=True, scripted_semantic_calls=0,
            planning_recovery_policy='ordinary_default_None',
            evaluation_max_dispatches_per_turn_schema=1, spending_ceiling_usd='5.00',
            completion_ceiling=48, per_turn_completion_ceiling=6,
            input_byte_token_upper_bound=150000, sdk_retries=0,
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        self.save()

    def artifact(self, name, value):
        safe = _safe_json(value, self.private)
        (self.output / name).write_text(
            json.dumps(safe, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

    def save(self):
        self.report['reservations'] = self.guard.rows
        self.report['reserved_cost_usd'] = str(self.guard.reserved)
        self.artifact('results.json', self.report)

    def discovery(self):
        owner = build_candidate(CANDIDATE, environment=os.environ)
        try:
            found = discover_models(owner._client, candidates=(CANDIDATE,),
                admitted_candidates=(CANDIDATE,), secrets=self.private)
        finally:
            owner.close()
        self.report['discovery'] = found
        self.save()
        row, = found['candidates']
        if row['discovery_result'] != 'EXACT_MODEL_AVAILABLE':
            raise Stop('Exact development-primary model discovery failed.')
        if not {'response_format', 'structured_outputs', 'max_completion_tokens'} <= set(row['supported_parameters'] or ()):
            raise Stop('Required strict-output or bounded-output metadata is absent.')
        prices = row['pricing']
        if (Decimal(prices.get('prompt', '-1')) != PROMPT_PRICE
                or Decimal(prices.get('completion', '-1')) != COMPLETION_PRICE
                or Decimal(prices.get('request', '0')) != 0):
            raise Stop('Public pricing differs from the accepted paid policy.')
        self.availability = Availability(**row['availability'])

    def reserve(self, request):
        self.guard.reserve(request=request, case=self.case)
        self.save()

    def factory(self, profile):
        if profile != CANDIDATE.profile():
            raise Stop('Unexpected profile requested.')
        runner = self

        class Capture:
            model_id = CANDIDATE.model_id

            def complete(self, *, prompt, response_schema):
                call_id = len(runner.report['calls']) + 1
                row = dict(call_id=call_id, case=runner.case,
                    artifact=f'call-{call_id:02d}.json', provider=None)
                runner.report['calls'].append(row)
                record = dict(prompt=prompt, response_schema=response_schema,
                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    schema_sha256=hashlib.sha256(json.dumps(response_schema,
                        sort_keys=True, separators=(',', ':')).encode()).hexdigest())
                runner.artifact(row['artifact'], record)
                runner.save()
                schema_key = (runner.case, record['schema_sha256'])
                if schema_key in runner.dispatched_schemas:
                    row['guard_blocked'] = 'One dispatch per turn/schema; no semantic repair or transport retry.'
                    row['provider'] = dict(completion_calls=0)
                    runner.save()
                    raise PlanningModelError('Evaluation repeat bound reached.',
                        code='PROVIDER_COMPLETION_LIMIT_EXCEEDED')
                runner.dispatched_schemas.add(schema_key)
                runtime = build_openrouter_candidate(CANDIDATE,
                    availability=runner.availability, admitted_candidates=(CANDIDATE,),
                    paid_policy=POLICY, environment=os.environ, secrets=runner.private,
                    before_dispatch=lambda *, request: runner.reserve(request))
                start = time.monotonic()
                try:
                    response = runtime.complete(prompt=prompt, response_schema=response_schema)
                    record['response'] = response
                    try:
                        record['response_json'] = json.loads(response)
                    except (ValueError, TypeError):
                        record['response_json'] = None
                    return response
                except Exception as exc:
                    record['exception'] = sanitize_provider_error(exc, secrets=runner.private)
                    raise
                finally:
                    row['elapsed_seconds'] = round(time.monotonic() - start, 3)
                    row['provider'] = dict(completion_calls=runtime.provider_completion_calls,
                        usage=runtime.usage_observation,
                        response_model=runtime.response_model_observation,
                        routing=runtime.routing_observation,
                        diagnosis=runtime.transport_diagnostic)
                    record['provider'] = row['provider']
                    runner.artifact(row['artifact'], record)
                    runtime.close()
                    runner.save()
                    usage = runtime.usage_observation or {}
                    if runtime.provider_completion_calls:
                        reservation = runner.guard.rows[-1]
                        if (usage.get('prompt_tokens', 0) > reservation['input_token_upper_bound']
                                or usage.get('completion_tokens', 0) > OUTPUT_LIMIT
                                or ('cost' in usage and Decimal(str(usage['cost'])) > Decimal(reservation['reserved_cost_usd']))):
                            raise Stop('Reported usage exceeded its reservation.')

            def close(self):
                pass

        return Capture()

    def turn(self, app, case, sid, utterance, **kwargs):
        if case != SCHEDULE[len(self.report['results'])]:
            raise Stop('Evaluation matrix order changed.')
        self.case = case
        state = app._application.sessions.load(sid)
        row = dict(case=case, session_id=sid, utterance=utterance,
                   before_generation=state.generation, call_start=len(self.report['calls']))
        self.report['results'].append(row)
        try:
            view = app.submit_turn(sid, case.lower(), utterance,
                expected_generation=state.generation, **kwargs)
            row['view'] = view.to_dict()
            if view.run_id:
                self.artifact(f'{case}-run.json', app._application.run_store.load(view.run_id).to_dict())
            if view.revision_id:
                revision = next(r for r in app._application.sessions.load(sid).revisions
                    if r.revision_id == view.revision_id)
                for output in revision.outputs:
                    self.artifact(f'{case}-evidence-{output.name}.json',
                        app.evidence(sid, view.revision_id, output.name).to_dict())
        except Exception as exc:
            row['exception'] = dict(type=type(exc).__name__, code=getattr(exc, 'code', None),
                diagnostic=sanitize_provider_error(exc, secrets=self.private))
        finally:
            after = app._application.sessions.load(sid)
            row['call_end'] = len(self.report['calls'])
            self.artifact(f'{case}-session.json', after.to_dict())
            self.save()
        print(json.dumps(dict(case=case, calls=row['call_end']-row['call_start'],
            status=row.get('view', {}).get('status'), exception=row.get('exception', {}).get('code'))), flush=True)
        return row

    def dependent(self, app, case, sid, utterance, field):
        state = app._application.sessions.load(sid)
        pending = state.interactions[-1].prerequisite if state.interactions else None
        if pending is not None and pending.get('field') == field:
            return self.turn(app, case, sid, utterance)
        if case != SCHEDULE[len(self.report['results'])]:
            raise Stop('Evaluation matrix order changed.')
        self.report['results'].append(dict(case=case, session_id=sid, utterance=utterance,
            blocked='No live-created matching pending prerequisite; no fabricated state or repair.'))
        self.save()

    def run(self, args):
        from .ua354a_fixtures import (
            BoundedScienceExecutor, build_neighbors_fixture,
            build_selection_fixture, synthetic_qc_environment,
        )
        config = load_web_configuration(args.operator_config)
        registry = build_default_tool_registry()
        executor = BoundedScienceExecutor(registry)
        workspace = Path(args.workspace).absolute()
        workspace.mkdir(parents=True, exist_ok=False)
        if args.fixtures_json:
            fixtures = json.loads(Path(args.fixtures_json).read_text())
            neighbors, selection = fixtures['neighbors'], fixtures['selection']
        else:
            neighbors = build_neighbors_fixture(workspace / 'fixtures' / 'neighbors')
            selection = build_selection_fixture(workspace / 'fixtures' / 'selection')
        self.artifact('fixtures.json', dict(neighbors=neighbors, selection=selection,
            source_h5ad=str(Path(args.source_h5ad).absolute()),
            resource_choices=[r.choice() for r in config.epizoo_resources]))
        app = InteractiveAgentApplication(workspace / 'application',
            model_profiles=(CANDIDATE.profile(),), default_profile_id=CANDIDATE.profile_id,
            planning_model_factory_registry=PlanningModelFactoryRegistry({'openrouter': self.factory}),
            registry=registry, executor=executor,
            approved_source_roots=(Path(args.source_h5ad).absolute().parent,),
            epizoo_resources=config.epizoo_resources)
        source = app.resources.register('ua354a-h5ad', args.source_h5ad,
            label='Registered single-cell H5AD', attribution='Retained UA3.2.2 bounded acceptance source')
        inputs, error = select_epizoo_resource(dict(device='cuda'), None, config.epizoo_resources)
        bound = app.resources.compose_h5ad(source.resource_id, inputs, resource_selection_error=error)
        for sid in ('a', 'ap', 'b1', 'b2', 'b3', 'c', 'n1', 'n2', 'n3'):
            app.create_session(sid)
        self.turn(app, 'A1', 'a', 'Use this dataset to run the supported EpiZoo analysis.', registered_input=bound)
        self.dependent(app, 'A2', 'a', 'These cells are from human', 'species')
        self.turn(app, 'A3', 'ap', 'Analyze this dataset with the available EpiZoo model.', registered_input=bound)
        self.dependent(app, 'A4', 'ap', 'The cells came from mice.', 'species')
        for case, sid, utterance in (
            ('B1', 'b1', 'Cluster these cells using Leiden with resolution 0.7.'),
            ('B2', 'b2', 'Cluster these cells using Leiden.'),
            ('B3', 'b3', 'Group these cells with Leiden, setting its granularity to 0.7.'),
        ):
            self.turn(app, case, sid, utterance, execution_inputs=neighbors['execution_inputs'])
        with synthetic_qc_environment():
            self.turn(app, 'C1', 'c', 'Select cells from these barcode QC results.',
                execution_inputs=selection['execution_inputs'])
            self.dependent(app, 'C2', 'c', 'Keep barcodes with at least 1 QC fragment record.', 'parameters')
            self.dependent(app, 'C3', 'c', 'Actually, require at least 2 QC fragment records instead.', 'parameters')
            self.dependent(app, 'C4', 'c', 'Use a minimum TSS enrichment of 0.5.', 'parameters')
        self.turn(app, 'N1', 'n1', 'Cluster these cells using Leiden with resolution -0.7.',
            execution_inputs=neighbors['execution_inputs'])
        self.turn(app, 'N2', 'n2', 'Run two independent Leiden clusterings on these cells; use resolution 0.7.',
            execution_inputs=neighbors['execution_inputs'])
        self.turn(app, 'N3', 'n3', 'Use EpiAgent to embed the cells in this dataset.', registered_input=bound)
        self.artifact('execution-boundary.json', executor.observations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--source-h5ad', required=True)
    parser.add_argument('--operator-config', required=True)
    parser.add_argument('--fixtures-json', help='Reuse previously generated real owner fixtures.')
    parser.add_argument('--discovery-json', help='Reuse exact catalog evidence from this evaluation, without another lookup.')
    args = parser.parse_args()
    runner = Runner(args.output)
    try:
        if args.discovery_json:
            prior = json.loads(Path(args.discovery_json).read_text())['discovery']
            row, = prior['candidates']
            if (row['candidate'] != CANDIDATE.manifest()
                    or row['discovery_result'] != 'EXACT_MODEL_AVAILABLE'
                    or Decimal(row['pricing']['prompt']) != PROMPT_PRICE
                    or Decimal(row['pricing']['completion']) != COMPLETION_PRICE
                    or Decimal(row['pricing'].get('request', '0')) != 0
                    or not {'response_format', 'structured_outputs', 'max_completion_tokens'} <= set(row['supported_parameters'] or ())):
                raise Stop('Reused exact discovery identity/policy changed.')
            runner.report['discovery'] = prior
            runner.report['discovery_reused_from'] = str(Path(args.discovery_json).absolute())
            runner.availability = Availability(**row['availability'])
            runner.save()
        else:
            runner.discovery()
        runner.run(args)
    except BaseException as exc:
        runner.report['hard_stop'] = dict(type=type(exc).__name__,
            diagnostic=sanitize_provider_error(exc, secrets=runner.private))
    finally:
        runner.save()
    print(json.dumps(dict(output=str(runner.output), calls=len(runner.report['calls']),
        hard_stop=runner.report['hard_stop']), ensure_ascii=False), flush=True)
    return 2 if runner.report['hard_stop'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
