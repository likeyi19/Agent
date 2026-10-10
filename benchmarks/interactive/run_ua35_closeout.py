"""Bounded live initial-declaration qualification, with every tool withheld.

This runner reuses the frozen UA3.5.4a transport, paid guards and raw response
recording. It exercises only ordinary interpretation, binding and admission;
UA3.5.5 remains the scientific and native-browser execution witness.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from agent.application import InteractiveAgentApplication
from agent.application.local_resources import select_epizoo_resource
from agent.orchestration.executor import ExecutionOutcome, PlanExecutor
from agent.orchestration.registry import build_default_tool_registry
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas import AgentError, ErrorCategory, ExecutionTraceEvent, TraceEventType
from agent.schemas.orchestration import _serialize
from agent.web.config import load_web_configuration

from . import run_ua354a as base
from .diagnostics import sanitize_provider_error
from .candidates import Availability
from .run_ua354b import RECOVERY_POLICY


SCHEDULE = ('A1', 'A2', 'A3', 'A4', 'B1', 'B2')
EXPLICIT = ('Analyze these human scATAC-seq cells with EpiZoo, construct their '
            'neighbor graph, perform Leiden clustering, and compute UMAP.')
OMITTED = ('Analyze these scATAC-seq cells with EpiZoo, construct their neighbor '
           'graph, perform Leiden clustering, and compute UMAP.')
MODEL_MENTION = 'EpiZoo supports human and mouse. Analyze these scATAC-seq cells with EpiZoo.'
TOOLS = ('epizoo_embed_cells', 'build_cell_neighbors', 'cluster_cells', 'compute_cell_umap')
CASE_LABELS = {
    'A1': 'initial_explicit_human_without_companion',
    'A2': 'species_omitted',
    'A3': 'live_pending_human_answer',
    'A4': 'model_supported_species_is_not_dataset_declaration',
    'B1': 'matching_typed_human_companion',
    'B2': 'conflicting_typed_mouse_companion',
}


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def exact_human_declaration(declaration, utterance):
    """Check accepted current-utterance evidence without prescribing its span."""
    start, end = declaration.get('start'), declaration.get('end')
    literal = declaration.get('literal')
    return (declaration.get('tool') == 'epizoo_embed_cells'
        and declaration.get('argument') == 'species'
        and declaration.get('value') == 'human'
        and type(literal) is str and bool(literal.strip())
        and type(start) is int and type(end) is int
        and 0 <= start < end <= len(utterance)
        and utterance[start:end] == literal)


class AdmissionOnlyExecutor(PlanExecutor):
    """Record ordinary preflight and withhold every plan before all tools.

    In particular, an unexpected model-selected inspection or clustering plan
    cannot escape this evaluation boundary. No outcome is scientific success.
    """

    def __init__(self, registry, runner):
        super().__init__(registry)
        self.runner = runner
        self.observations = []

    def execute(self, plan, **kwargs):
        preflight = self.preflight(plan)
        row = _serialize(dict(case=self.runner.case, plan=plan.to_dict(),
            preflight=preflight.to_dict(), delegated=False,
            tool_names=tuple(step.tool_name for step in plan.steps),
            scientific_execution_count=0,
            evaluation_boundary='after_compilation_admission_and_preflight_before_all_tools'))
        self.observations.append(row)
        self.runner.artifact('execution-boundary.json', self.observations)
        error = AgentError(category=ErrorCategory.ENVIRONMENT_ERROR,
            code='EVALUATION_EXECUTION_WITHHELD',
            message='UA3.5 final qualification withheld this plan before every scientific tool.',
            details=dict(evaluation_only=True, preflight_passed=preflight.passed,
                scientific_execution_count=0, withheld_tools=row['tool_names']))
        event = ExecutionTraceEvent(sequence=0, event_type=TraceEventType.PLAN_VALIDATION,
            timestamp=datetime.now(timezone.utc).isoformat(), message=error.message,
            details=dict(evaluation_only=True, preflight_passed=preflight.passed,
                scientific_execution_count=0, error_code=error.code))
        return ExecutionOutcome(step_results=(), errors=(error,), trace=(event,))


class Runner(base.Runner):
    def __init__(self, output):
        super().__init__(output)
        self.output.chmod(0o700)
        self.constructions = []
        self.report.update(kind='UA3.5-final-initial-declaration-admission',
            schedule=list(SCHEDULE), case_labels=CASE_LABELS,
            planning_recovery_policy=RECOVERY_POLICY.to_dict(),
            discovery_policy='fresh_exact_catalog_once',
            scientific_execution_count=0, foundation_model_loads=0,
            scientific_executor='all_plans_withheld_after_ordinary_preflight',
            scripted_semantic_calls=0, model_constructions=self.constructions,
            script_sha256=file_sha(__file__), base_runner_sha256=file_sha(base.__file__),
            ua354b_runner_sha256=file_sha(Path(__file__).with_name('run_ua354b.py')),
            scope=('Initial declarations and ordinary admission only. No input-axis '
                   'compatibility, scientific execution, Revision or browser acceptance.'))
        self.save()

    def save(self):
        prior = self.report.get('prior_paid_run') or {}
        calls = self.report['calls']
        completions = sum((call.get('provider') or {}).get('completion_calls', 0) for call in calls)
        cost = sum((Decimal(str(((call.get('provider') or {}).get('usage') or {}).get('cost', 0)))
                    for call in calls), Decimal(0))
        self.report['aggregate_paid_usage'] = dict(current_interface_calls=len(calls),
            current_provider_completions=completions, current_provider_reported_cost_usd=str(cost),
            aggregate_provider_completions=prior.get('provider_completions', 0) + completions,
            aggregate_provider_reported_cost_usd=str(Decimal(prior.get('provider_reported_cost_usd', '0')) + cost),
            aggregate_dispatch_reservations=len(self.guard.rows),
            aggregate_reserved_cost_usd=str(self.guard.reserved))
        super().save()

    def seed_prior_budget(self, path):
        """Count the preserved initial dispatches against unchanged total bounds."""
        path = Path(path).absolute()
        prior = json.loads(path.read_text())
        calls, rows = prior['calls'], prior['reservations']
        completions = sum((call.get('provider') or {}).get('completion_calls', 0) for call in calls)
        reserved = sum((Decimal(row['reserved_cost_usd']) for row in rows), Decimal(0))
        cost = sum((Decimal(str(((call.get('provider') or {}).get('usage') or {}).get('cost', 0)))
                    for call in calls), Decimal(0))
        if (self.report['calls'] or self.guard.rows or prior['candidate'] != base.CANDIDATE.manifest()
                or prior['spending_ceiling_usd'] != '5.00' or prior['completion_ceiling'] != 48
                or prior['per_turn_completion_ceiling'] != 6
                or prior['input_byte_token_upper_bound'] != 150000 or prior['sdk_retries'] != 0
                or len(rows) != completions or len(calls) != completions
                or not 0 < completions < 48 or not Decimal(0) < reserved < Decimal('5.00')
                or reserved != Decimal(prior['reserved_cost_usd']) or cost > reserved
                or any(row['case'] not in SCHEDULE or row['input_token_upper_bound'] > 150000
                       or row['completion_token_upper_bound'] != base.OUTPUT_LIMIT for row in rows)
                or any(sum(row['case'] == case for row in rows) > 6 for case in SCHEDULE)
                or any(call.get('guard_blocked') or (call.get('provider') or {}).get('completion_calls') != 1
                       for call in calls)):
            raise base.Stop('Preserved initial budget identity or aggregate bounds changed.')
        # Prefixing historical case IDs distinguishes the original immutable
        # turns from this post-contract requalification's fresh Session turns.
        self.guard.rows = [dict(row, original_case=row['case'], case='initial:' + row['case'],
                                evaluation_id='initial') for row in rows]
        self.guard.reserved = reserved
        self.report.update(qualification_stage='post_contract_requalification',
            completion_ceiling_scope='aggregate_preserved_initial_and_current_dispatches',
            prior_paid_run=dict(path=str(path), sha256=file_sha(path),
                interface_calls=len(calls), provider_completions=completions,
                conservative_reservation_usd=str(reserved), provider_reported_cost_usd=str(cost),
                guard_blocks=sum(bool(call.get('guard_blocked')) for call in calls),
                runner_sha256=prior['script_sha256'],
                remaining_completion_reservations=48 - len(rows),
                remaining_reserved_budget_usd=str(Decimal('5.00') - reserved),
                original_raw_results_preserved=True,
                fresh_turn_case_identity='historical reservation case IDs prefixed initial:'))
        self.save()

    def reuse_discovery(self, path):
        """Reuse this evaluation's exact catalog evidence without another lookup."""
        path = Path(path).absolute()
        value = json.loads(path.read_text())
        found = value['discovery'] if 'discovery' in value else value
        row, = found['candidates']
        if (row['candidate'] != base.CANDIDATE.manifest()
                or row['discovery_result'] != 'EXACT_MODEL_AVAILABLE'
                or Decimal(row['pricing']['prompt']) != base.PROMPT_PRICE
                or Decimal(row['pricing']['completion']) != base.COMPLETION_PRICE
                or Decimal(row['pricing'].get('request', '0')) != 0
                or not {'response_format', 'structured_outputs', 'max_completion_tokens'}
                    <= set(row['supported_parameters'] or ())):
            raise base.Stop('Reused exact discovery identity or policy changed.')
        self.report.update(discovery=found, discovery_policy='same_evaluation_exact_catalog_reused',
            discovery_reused_from=str(path), discovery_reused_source_sha256=file_sha(path))
        self.availability = Availability(**row['availability'])
        self.save()

    def factory(self, profile):
        model = super().factory(profile)
        self.constructions.append(profile.profile_id)
        self.save()
        return model

    def interpreter_choice(self, row):
        for call in self.report['calls'][row['call_start']:row['call_end']]:
            artifact = json.loads((self.output / call['artifact']).read_text())
            try:
                prompt = json.loads(artifact['prompt'])
            except (ValueError, TypeError):
                continue
            if 'turn_schema_version' in prompt:
                return (artifact.get('response_json') or {}).get('decision')
        return None

    def disposition(self, app, row, executor, source, human_resource, *, expected):
        row['case_label'] = CASE_LABELS[row['case']]
        if row.get('blocked'):
            row['disposition'] = 'BLOCKED'
            self.save()
            return
        state = app._application.sessions.load(row['session_id'])
        interaction = next((item for item in state.interactions
                            if item.turn_id == row['case'].lower()), None)
        choice = self.interpreter_choice(row)
        checks = dict(raw_interpreter_choice_retained=choice is not None,
            no_revision=not state.revisions and state.generation == 0,
            no_scientific_steps=not row.get('view', {}).get('steps'))
        observations = [item for item in executor.observations if item['case'] == row['case']]
        if expected == 'pending':
            admitted = {} if interaction is None else _serialize(interaction.admitted) or {}
            pending = None if interaction is None else _serialize(interaction.prerequisite)
            run_id = row.get('view', {}).get('run_id')
            run = app._application.run_store.load(run_id) if run_id else None
            checks.update(no_dataset_species=admitted.get('inputs', {}).get('species') is None,
                durable_compiler_prerequisite=pending is not None and pending.get('field') == 'species'
                    and pending.get('origin_turn_id') == row['case'].lower(),
                missing_species_clarification=row.get('view', {}).get('status') == 'clarification'
                    and row.get('view', {}).get('response', {}).get('clarification', {}).get('reason') == 'missing_species',
                compiler_missing_species=run is not None and any(
                    error.code == 'MISSING_REQUIRED_SOURCE' and error.details.get('target_port') == 'species'
                    for error in run.errors),
                executor_not_entered=not observations)
        elif expected == 'conflict':
            response = row.get('view', {}).get('response') or {}
            error = row.get('view', {}).get('error') or {}
            reason = (response.get('clarification') or {}).get('reason')
            checks.update(conflict_reported=reason in ('conflicting_species', 'conflicting_scientific_parameter')
                    or error.get('code') in ('EPIZOO_RESOURCE_SELECTION_INVALID', 'CONFLICTING_SCIENTIFIC_PARAMETER'),
                executor_not_entered=not observations,
                no_executable_turn=all(turn.request_id is None and turn.run_id is None
                    and turn.status == 'clarification' and not turn.selections for turn in state.turns),
                no_run=row.get('view', {}).get('run_id') is None)
        else:
            admitted = {} if interaction is None else _serialize(interaction.admitted) or {}
            inputs = admitted.get('inputs', {})
            declared = [] if choice is None else [item for item in choice.get('arguments', [])
                if item.get('tool') == 'epizoo_embed_cells' and item.get('argument') == 'species']
            checks.update(no_species_clarification=row.get('view', {}).get('status') != 'clarification',
                bound_human=inputs.get('species') == 'human',
                exact_registered_source=inputs.get('input_path') == source.source_path
                    and admitted.get('registered_input', {}).get('record_sha256') == source.record_sha256,
                exact_human_resource=all(inputs.get(key) == value for key, value in human_resource.inputs().items()),
                executor_preflight_passed=len(observations) == 1 and observations[0]['preflight']['passed'],
                deliberately_withheld=(row.get('view', {}).get('error') or {}).get('code') == 'EVALUATION_EXECUTION_WITHHELD',
                ordinary_turn_link=len(state.turns) == 1,
                expected_operations=len(observations) == 1 and tuple(
                    name for name in observations[0]['tool_names'] if name != 'inspect_scATAC') == TOOLS)
            if expected == 'initial':
                checks.update(live_exact_species_declaration=len(declared) == 1
                    and exact_human_declaration(declared[0], row['utterance']),
                    no_typed_species_companion='species' not in _serialize(interaction.submission['execution_inputs'])
                    if interaction is not None else False)
            elif expected == 'continuation':
                checks.update(exact_original_task_continuation=admitted.get('continuation') ==
                    dict(origin_turn_id='a2', species='human'),
                    live_pending_answer=choice is not None and choice.get('kind') == 'answer_prerequisite'
                        and choice.get('pending') == '@species' and choice.get('species') == 'human')
            elif expected == 'companion':
                checks['matching_typed_companion_retained'] = interaction is not None and \
                    _serialize(interaction.submission['execution_inputs']).get('species') == 'human'
        row.update(expected_boundary=expected, raw_interpreter_choice=choice,
            checks=checks, disposition='PASS' if all(checks.values()) else 'FAIL')
        self.save()
        print(json.dumps(dict(case=row['case'], disposition=row['disposition'],
            failed_checks=[name for name, passed in checks.items() if not passed])), flush=True)

    def run(self, args):
        config = load_web_configuration(args.operator_config)
        registry = build_default_tool_registry()
        executor = AdmissionOnlyExecutor(registry, self)
        workspace = Path(args.workspace).absolute()
        workspace.mkdir(parents=True, exist_ok=False, mode=0o700)
        source_path = Path(args.source_h5ad).absolute()
        if file_sha(source_path) != args.source_sha256:
            raise base.Stop('The retained genuine human input bytes differ from their accepted identity.')

        def application():
            return InteractiveAgentApplication(workspace / 'application',
                model_profiles=(base.CANDIDATE.profile(),), default_profile_id=base.CANDIDATE.profile_id,
                planning_model_factory_registry=PlanningModelFactoryRegistry({'openrouter': self.factory}),
                registry=registry, executor=executor, planning_recovery_policy=RECOVERY_POLICY,
                approved_source_roots=(source_path.parent,), epizoo_resources=config.epizoo_resources)

        app = application()
        source = app.resources.register('ua35-human-input', source_path,
            label='Retained canonical human H5AD',
            attribution='Exact M12 C0 32-cell input reused after genuine UA3.5.5 owner execution.')
        human_resources = [resource for resource in config.epizoo_resources
                           if resource.species == 'human' and resource.default]
        if len(human_resources) != 1:
            raise base.Stop('The accepted human catalog no longer has one default.')
        human_resource = human_resources[0]

        def binding(species=None):
            if species is None:
                # Match the ordinary Web branch with no companion/resource
                # choice; missing scientific prerequisites are diagnosed later.
                return app.resources.compose_h5ad(source.resource_id, {})
            inputs = dict(species=species)
            resolved, error = select_epizoo_resource(inputs, None, config.epizoo_resources)
            return app.resources.compose_h5ad(source.resource_id, resolved, resource_selection_error=error)

        unqualified, human, mouse = binding(), binding('human'), binding('mouse')
        self.artifact('input-resource-context.json', dict(source_registration=source.to_dict(),
            source_sha256=args.source_sha256, source_bytes=source_path.stat().st_size,
            resource_choices=[resource.choice() for resource in config.epizoo_resources],
            initial_companion=None, matching_companion=dict(species='human'),
            conflicting_companion=dict(species='mouse'), planning_profile=base.CANDIDATE.manifest(),
            operator_configuration_edited=False, original_checkpoint_loaded=False))
        for sid in ('explicit', 'omitted', 'model-mention', 'matching', 'conflicting'):
            app.create_session(sid)
        with patch.object(base, 'SCHEDULE', SCHEDULE):
            row = self.turn(app, 'A1', 'explicit', EXPLICIT, registered_input=unqualified)
            self.disposition(app, row, executor, source, human_resource, expected='initial')
            row = self.turn(app, 'A2', 'omitted', OMITTED, registered_input=unqualified)
            self.disposition(app, row, executor, source, human_resource, expected='pending')
            # Reconstruction reads the durable pending state; it supplies no
            # semantic answer, execution plan or fabricated prerequisite.
            before = len(self.report['calls'])
            app = application()
            app.reopen_session('omitted')
            self.report['pending_restoration_dispatches'] = len(self.report['calls']) - before
            row = self.dependent(app, 'A3', 'omitted', 'These are human cells.', 'species')
            if row is None:
                row = self.report['results'][-1]
            self.disposition(app, row, executor, source, human_resource, expected='continuation')
            row = self.turn(app, 'A4', 'model-mention', MODEL_MENTION, registered_input=unqualified)
            self.disposition(app, row, executor, source, human_resource, expected='pending')
            row = self.turn(app, 'B1', 'matching', EXPLICIT, registered_input=human)
            self.disposition(app, row, executor, source, human_resource, expected='companion')
            row = self.turn(app, 'B2', 'conflicting', EXPLICIT, registered_input=mouse)
            self.disposition(app, row, executor, source, human_resource, expected='conflict')
        self.artifact('execution-boundary.json', executor.observations)
        self.report['disposition_counts'] = {name: sum(row.get('disposition') == name
            for row in self.report['results']) for name in ('PASS', 'FAIL', 'BLOCKED')}
        self.report['scientific_execution_count'] = 0
        self.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Explicitly dispatch the bounded real-provider matrix.')
    for argument in ('output', 'workspace', 'source-h5ad', 'operator-config'):
        parser.add_argument('--' + argument, required=True)
    parser.add_argument('--source-sha256', default='0c601bf017053f82fa34273156f647148c3ccddecf6646fcd9fcb4cc4aabe264')
    parser.add_argument('--prior-budget-json', help='Preserved initial results; count original dispatch reservations cumulatively.')
    parser.add_argument('--discovery-json', help='Reuse this evaluation\'s exact catalog evidence without another lookup.')
    args = parser.parse_args()
    if not args.live:
        parser.error('Paid provider qualification requires --live after deterministic acceptance.')
    if args.prior_budget_json and not args.discovery_json:
        parser.error('Post-contract requalification requires the initial evaluation\'s --discovery-json.')
    runner = Runner(args.output)
    try:
        if args.prior_budget_json:
            runner.seed_prior_budget(args.prior_budget_json)
        if args.discovery_json:
            runner.reuse_discovery(args.discovery_json)
        else:
            runner.discovery()
        runner.run(args)
    except BaseException as exc:
        runner.report['hard_stop'] = dict(type=type(exc).__name__,
            diagnostic=sanitize_provider_error(exc, secrets=runner.private))
    finally:
        runner.save()
    result = dict(output=str(runner.output), calls=len(runner.report['calls']),
        provider_completions=sum((call.get('provider') or {}).get('completion_calls', 0)
                                 for call in runner.report['calls']),
        scientific_execution_count=0, disposition_counts=runner.report.get('disposition_counts'),
        hard_stop=runner.report['hard_stop'])
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if runner.report['hard_stop']:
        return 2
    return 0 if all(row.get('disposition') == 'PASS' for row in runner.report['results']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
