"""Small qualification instrument; never a Planner or scientific executor.

Raw model responses are transient. Reports contain typed observations, safe
identities and fingerprints, never prompts, credentials or exception prose.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from unittest.mock import patch

from agent.application import ResearchAgentApplication
from agent.application import scientific_dialogue as dialogue
from agent.application import scientific_guidance as guidance
from agent.application import turn_decisions as decisions
from agent.application import turns, dialogue_execution
from agent.orchestration import AgentRuntime, LLMPlanner, PlanningModelError, RunMode
from agent.orchestration.executor import PlanExecutor
from agent.orchestration.planner import PlannerError
from agent.orchestration.llm_planner import _PROVIDER_ERROR_CODES
from agent.orchestration.planning_model import classify_provider_exception
from agent.orchestration.planning_scope import (
    PlanningScope, capability_index, fingerprint, parse_selection, selection_request,
)
from agent.orchestration.semantic_wire_v4 import parse_semantic_wire_v4
from agent.orchestration.semantic_compiler import build_semantic_compiler_contract
from agent.schemas import AgentRequest
from agent.schemas.orchestration import _serialize
from benchmarks.planner.benchmark import guarded_registry

from .fixtures import build_context


class ZeroScienceViolation(BaseException):
    """An execution guard fired; cannot be converted into a normal outcome."""


@dataclass(frozen=True)
class AttemptResult:
    data: dict

    def to_dict(self):
        return json.loads(json.dumps(_serialize(self.data), ensure_ascii=False, allow_nan=False))


def candidate_identity(candidate):
    """Whitelist non-secret candidate fields, independently of adapter labels."""
    from .candidates import Candidate
    if not isinstance(candidate, Candidate):
        raise TypeError('Qualification requires a validated Candidate descriptor')
    manifest = candidate.manifest()
    return {key: manifest.get(key) for key in (
        'candidate_id', 'provider_id', 'model_id', 'adapter_family', 'profile_id',
        'timeout_seconds', 'endpoint_identity',
    )}


def _provider_code(exc):
    return exc.code if isinstance(exc, PlanningModelError) and exc.code in _PROVIDER_ERROR_CODES else 'PLANNING_PROVIDER_ERROR'


class RecordingModel:
    def __init__(self, model, candidate, scenario, attempt_id, registry_fingerprint):
        self.inner = model
        self.identity = candidate_identity(candidate)
        self.model_id = self.identity['provider_id'] + ':' + self.identity['model_id']
        self.scenario = scenario
        self.attempt_id = attempt_id
        self.registry_fingerprint = registry_fingerprint
        self.calls = []
        self.responses = []  # In-memory diagnostics only; deliberately not serialized.
        self.compiled_plan = None

    def complete(self, *, prompt, response_schema):
        payload = json.loads(prompt)
        phase = next((name for key, name in (
            ('turn_schema_version', 'interpreter'),
            ('selection_schema_version', 'stage_a'),
            ('dialogue_schema_version', 'answer'),
            ('guidance_schema_version', 'guidance'),
            ('output_selection_schema_version', 'output_selection'),
        ) if key in payload), 'stage_b')
        row = dict(
            scenario_id=self.scenario.scenario_id, attempt_id=self.attempt_id,
            surface=phase, candidate=self.identity,
            prompt_fingerprint=fingerprint(prompt), schema_fingerprint=fingerprint(response_schema),
            registry_fingerprint=self.registry_fingerprint,
            request_fingerprint=None if self.scenario.request is None else fingerprint(self.scenario.request.to_dict()),
            provider_outcome='returned', error_code=None, latency_seconds=None,
            http_status=None,
            contract_success=None, decision_kind=None, semantic_success=None,
            compiler_success=None, preflight_success=None,
            recovery_kind='repair' if 'repair' in payload else 'failover' if 'failover' in payload else 'initial',
            measurement='production_recovery' if self.scenario.surface == 'layer2' else 'direct',
        )
        self.calls.append(row)
        self.responses.append(None)
        started = time.monotonic()
        try:
            response = self.inner.complete(prompt=prompt, response_schema=response_schema)
            self.responses[-1] = response
            row['response_fingerprint'] = fingerprint(response) if isinstance(response, str) else None
            row['response_bytes'] = len(response.encode()) if isinstance(response, str) else None
            return response
        except Exception as exc:
            row['provider_outcome'] = 'failed'
            cause = exc.__cause__ if isinstance(exc, PlanningModelError) else exc
            status = getattr(cause, 'status_code', None)
            if type(status) is int: row['http_status'] = status
            row['error_code'] = (_provider_code(exc) if isinstance(exc, PlanningModelError)
                                 else classify_provider_exception(exc)[0])
            raise PlanningModelError('Sanitized qualification provider failure.', code=row['error_code'],
                retry_after_seconds=getattr(exc, 'retry_after_seconds', None) if isinstance(exc, PlanningModelError) else None) from None
        finally:
            row['latency_seconds'] = time.monotonic() - started


@contextmanager
def zero_science(registry_guard, *, read_only=False):
    """Independent callable, execution-entry and reconstruction guards."""
    counters = dict(scientific_calls=0, execution_entry_attempts=0,
                    scientific_step_results=0, forbidden_read_only_entries=0)

    def forbidden(*args, **kwargs):
        counters['execution_entry_attempts'] += 1
        raise ZeroScienceViolation('Scientific execution prohibited')

    def readonly_forbidden(*args, **kwargs):
        counters['forbidden_read_only_entries'] += 1
        raise ZeroScienceViolation('Read-only qualification entered production/reconstruction')

    with ExitStack() as stack:
        for owner, name in ((AgentRuntime, '_run_execute'), (AgentRuntime, '_run_durable_execute'),
                            (PlanExecutor, 'execute'), (turns, 'execute')):
            stack.enter_context(patch.object(owner, name, forbidden))
        if read_only:
            from agent.application import service
            from agent.orchestration import verifier
            from agent.report import evidence, analysis_report
            from agent.tools.data.authority_context import VerificationContext
            owners = (
                (AgentRuntime, ('run', 'resume')),
                (LLMPlanner, ('plan', 'plan_with_recovery', 'plan_with_diagnostics', '_plan_detailed', '_select_scope')),
                (PlanExecutor, ('preflight',)),
                (ResearchAgentApplication, ('run', 'resume', '_complete')),
                (VerificationContext, ('verify',)),
                (verifier, ('verify_step', 'verify_run')),
                (evidence, ('verify_step', 'verify_run', 'build_analysis_evidence', 'verify_analysis_evidence')),
                (service, ('build_analysis_evidence', 'verify_analysis_evidence', 'build_analysis_report')),
                (analysis_report, ('build_analysis_report', 'verify_analysis_report')),
            )
            for owner, names in owners:
                for name in names:
                    stack.enter_context(patch.object(owner, name, readonly_forbidden))
        try:
            yield counters
        finally:
            counters['scientific_calls'] = registry_guard.count
            if any(counters.values()):
                raise ZeroScienceViolation('Qualification safety assertion failed')


def _decision_kind(decision):
    return {decisions.Execute: 'execute', decisions.ExecuteCandidate: 'execute_candidate',
            decisions.Answer: 'answer', decisions.Clarify: 'clarify',
            decisions.Navigate: 'navigate'}[type(decision)]


def _admit(context, decision, inputs):
    if isinstance(decision, decisions.Clarify):
        return None
    if isinstance(decision, decisions.Execute) and decision.operation == 'plan':
        return dialogue_execution.admit(context.sessions, context.interaction, decision, inputs)
    if isinstance(decision, decisions.Answer):
        if decision.intent == 'scientific':
            return dialogue.admit(context.sessions, context.interaction, decision.scientific)
        if decision.intent == 'guidance':
            return guidance.admit(context.sessions, context.interaction, decision.guidance)
        if decision.intent == 'unsupported':
            return dict(kind='answer', intent='unsupported')
        from agent.application.responses import admit_answer
        return admit_answer(context.interaction, decision)
    if isinstance(decision, decisions.ExecuteCandidate):
        from agent.application.guidance_selection import admit
        return admit(context.sessions, context.interaction, decision, inputs)
    return turns.admit(context.sessions, context.interaction, decision)


def _safe_admitted(admitted):
    return None if admitted is None else {key: _serialize(admitted[key]) for key in (
        'kind', 'intent', 'operation', 'tool', 'target', 'comparison', 'targets', 'revision_id',
    ) if key in admitted}


def _semantic(scenario, result, *, context=None):
    """Scenario-specific accepted properties; never lexical or scientific inference."""
    expected = scenario.expected
    checks = []
    if 'kind' in expected:
        wanted = expected['kind']
        checks.append(result['decision_kind'] == wanted)
    if 'reason' in expected:
        checks.append(result.get('clarification_reason') == expected['reason'])
    if 'tool' in expected:
        checks.append(result.get('selected_tool') == expected['tool'] if scenario.surface in ('interpreter', 'layer2')
                      else expected['tool'] in result.get('plan_tools', ()))
    if 'capabilities' in expected:
        checks.append(set(result.get('capability_ids', ())) == set(expected['capabilities']))
    if 'tools' in expected:
        checks.append(set(result.get('plan_tools', ())) == set(expected['tools']))
    if 'support' in expected:
        checks.append(result.get('support') == expected['support'])
    if 'request_sources' in expected:
        checks.append(all(result.get('request_sources', {}).get(port) == source
                          for port, source in expected['request_sources'].items()))
    if 'target_turn' in expected and context is not None:
        admitted = result.get('admitted') or {}
        target = admitted.get('target') or {}
        state = context.sessions.load(context.session_id)
        matches = [r for r in state.revisions if r.turn_id == expected['target_turn']]
        checks.append(len(matches) == 1 and target.get('revision_id') == matches[0].revision_id)
    if 'subject' in expected:
        target = (result.get('admitted') or {}).get('target') or {}
        checks.append(target.get('subject') == expected['subject'])
    if 'output_name' in expected:
        target = (result.get('admitted') or {}).get('target') or {}
        checks.append(target.get('output_name') == expected['output_name'])
    return all(checks) if checks else None


def _planner_observations(recorder, request, registry, diagnostics=()):
    """Parse transient returned candidates independently of presentation failures."""
    scope = None
    planning_index = 0
    details = [d.to_details() if hasattr(d, 'to_details') else dict(d) for d in diagnostics]
    for row, response in zip(recorder.calls, recorder.responses):
        if row['surface'] not in ('stage_a', 'stage_b'):
            continue
        planning_index += 1
        observed = [d for d in details if d.get('provider_call_index') == planning_index]
        if observed:
            row['recovery_kind'] = observed[0]['attempt_kind']
            # Stable diagnostics contain identifiers, never provider exception prose.
            row['diagnostics'] = observed
        if row['provider_outcome'] == 'failed':
            continue
        try:
            if row['surface'] == 'stage_a':
                scope = parse_selection(response, registry)
                row.update(contract_success=True, decision_kind='select',
                           capability_ids=list(scope.capability_ids))
            else:
                visible = scope.visible_tool_names if scope is not None else (
                    PlanningScope(registry, recorder.scenario.fixed_capability_ids).visible_tool_names)
                candidate = parse_semantic_wire_v4(response, request, registry, visible_tool_names=visible)
                bindings = {}
                rules = build_semantic_compiler_contract(registry).request_bindings
                for step in candidate.steps:
                    for source in step.sources:
                        if not hasattr(source, 'input_name'): continue
                        for rule in rules:
                            if (rule.tool_name == step.tool_name and rule.target_port == source.target_port
                                    and rule.selector == source.input_name):
                                bindings[step.tool_name + '.' + rule.argument_name] = rule.input_name
                    # Recover omitted unambiguous bindings from actual compiler output.
                    if recorder.compiled_plan is not None:
                        compiled = [s for s in recorder.compiled_plan.steps if s.step_id == step.step_id]
                        if len(compiled) == 1:
                            for argument, value in compiled[0].arguments.items():
                                key = step.tool_name + '.' + argument
                                origins = {r.input_name for r in rules if r.tool_name == step.tool_name
                                    and r.argument_name == argument and r.input_name in request.inputs
                                    and request.inputs[r.input_name] == value}
                                if key not in bindings and len(origins) == 1: bindings[key] = origins.pop()
                row.update(contract_success=True, decision_kind='plan',
                           plan_tools=[step.tool_name for step in candidate.steps],
                           request_sources=bindings)
        except PlannerError as exc:
            row.update(contract_success=exc.code == 'UNSUPPORTED_REQUEST',
                       decision_kind='unsupported' if exc.code == 'UNSUPPORTED_REQUEST' else None,
                       parse_error_code=exc.code)
        failures = [d for d in observed if d.get('outcome') in ('failed', 'rejected')
                    and d.get('stage') != 'recovery']
        if failures:
            failure = failures[-1]
            row['failure_code'] = failure['code']
            row['failure_stage'] = failure['stage']
            if row['contract_success'] and row['decision_kind'] == 'plan':
                row['compiler_success'] = failure['stage'] == 'preflight'
                if failure['stage'] == 'preflight': row['preflight_success'] = False
        elif any(d.get('candidate_preflight_passed') is True for d in observed):
            row.update(compiler_success=True, preflight_success=True)


def _classify(result, recorder):
    if recorder.calls and recorder.calls[-1]['provider_outcome'] == 'failed':
        return 'operational'
    if result['contract_success'] is False: return 'contract'
    if any(result[key] is False for key in ('admission_success', 'compiler_success', 'preflight_success')):
        return 'deterministic_admission'
    if result['semantic_success'] is False: return 'semantic'
    return None


def run_attempt(scenario, candidate, *, model, attempt_id='attempt-1', registry=None):
    """Run one frozen scenario, with no harness retry or implicit provider call."""
    guarded, guard = guarded_registry()
    if registry is not None:
        # Caller metadata may differ; scientific callables are always replaced.
        from agent.orchestration import ToolRegistry
        guarded = ToolRegistry(tuple(replace(registry.get(n), function=guard.function(n)) for n in registry.names()))
    registry_id = PlanningScope(guarded, tuple(capability_index(guarded))).scope_fingerprint
    recorder = RecordingModel(model, candidate, scenario, attempt_id, registry_id)
    result = dict(schema_version=1, scenario_id=scenario.scenario_id, attempt_id=attempt_id,
        surface=scenario.surface, candidate=candidate_identity(candidate), registry_fingerprint=registry_id,
        scenario_fingerprint=scenario.fingerprint(), human_review_required=scenario.human_review_required,
        request_fingerprint=None if scenario.request is None else fingerprint(scenario.request.to_dict()),
        contract_success=None, semantic_success=None, admission_success=None,
        compiler_success=None, preflight_success=None, decision_kind=None, genuine_clarify=False,
        failure_category=None, error_code=None, calls=recorder.calls, initial_outcome=None, final_outcome=None)
    request = scenario.request
    diagnostics = []
    readonly = scenario.surface in ('interpreter', 'answer', 'guidance')
    with TemporaryDirectory(prefix='agent-interactive-qualification-') as scratch:
        with zero_science(guard, read_only=readonly) as counters:
            context = build_context(Path(scratch), scenario.fixture, guarded, utterance=scenario.utterance,
                                    execution_inputs=None if request is None else request.inputs,
                                    include_interaction=scenario.surface != 'layer2')
            result['context_fingerprint'] = fingerprint(context.public)
            result['fixture_provenance'] = context.provenance
            try:
                if scenario.surface == 'interpreter':
                    decision = decisions.interpret(recorder, scenario.utterance, context.public)
                    result.update(contract_success=True, decision_kind=_decision_kind(decision),
                                  genuine_clarify=isinstance(decision, decisions.Clarify))
                    if isinstance(decision, decisions.Clarify):
                        result['clarification_reason'] = decision.reason
                    if isinstance(decision, decisions.Execute): result['selected_tool'] = decision.target
                    try:
                        admitted = _admit(context, decision, {} if request is None else request.inputs)
                        result.update(admission_success=True, admitted=_safe_admitted(admitted))
                    except decisions.IntentError as exc:
                        result.update(admission_success=False, error_code=exc.reason)
                elif scenario.surface == 'stage_a':
                    prompt, schema = selection_request(request, guarded)
                    scope = parse_selection(recorder.complete(prompt=prompt, response_schema=schema), guarded)
                    result.update(contract_success=True, decision_kind='select', capability_ids=list(scope.capability_ids))
                elif scenario.surface == 'stage_b':
                    scope = PlanningScope(guarded, scenario.fixed_capability_ids)
                    planner = LLMPlanner(recorder, profile=candidate.profile())
                    attempt = planner._plan_detailed(request, guarded, scope=scope)
                    diagnostics = attempt.diagnostics
                    recorder.compiled_plan = attempt.plan
                    result.update(contract_success=True, compiler_success=True, decision_kind='plan',
                                  plan_tools=[s.tool_name for s in attempt.plan.steps])
                    if scenario.preflight:
                        verification = PlanExecutor(guarded).preflight(attempt.plan)
                        result['preflight_success'] = verification.passed
                        if verification.error: result['error_code'] = verification.error.code
                elif scenario.surface == 'answer':
                    question = decisions.ScientificQuestion(decisions.ScientificTarget(
                        scenario.expected.get('referent', '@current_result'), scenario.expected.get('subject')))
                    admitted = dialogue.admit(context.sessions, context.interaction, question)
                    result.update(admission_success=True, admitted=_safe_admitted(admitted))
                    public, claims = dialogue.context(context.sessions, context.session_id, admitted)
                    result['evidence_context_fingerprint'] = fingerprint(_serialize(public))
                    response = dialogue.generate(recorder, scenario.utterance, admitted['focus'], public, claims)
                    result.update(contract_success=True, decision_kind='answer', support=response.support,
                                  explanation=response.explanation)
                elif scenario.surface == 'guidance':
                    question = decisions.GuidanceQuestion(targets=(decisions.ScientificTarget(
                        scenario.expected.get('referent', '@current_result'), scenario.expected.get('subject')),))
                    admitted = guidance.admit(context.sessions, context.interaction, question)
                    interaction = replace(context.interaction, admitted=admitted, status='admitted')
                    context.sessions._store._update(context.session_id, lambda state: replace(state,
                        interactions=tuple(interaction if i.turn_id == interaction.turn_id else i for i in state.interactions)))
                    result.update(admission_success=True, admitted=_safe_admitted(admitted))
                    public, _ = guidance.context(context.sessions, context.session_id, interaction)
                    result['evidence_context_fingerprint'] = fingerprint(_serialize(public))
                    outcome = guidance.answer(context.sessions, context.session_id, interaction, recorder)
                    result.update(contract_success=outcome.guidance is not None, decision_kind='answer',
                        explanation=outcome.text, guidance_candidates=None if outcome.guidance is None else outcome.guidance.to_dict())
                elif scenario.surface == 'layer2':
                    result['requested_mode'] = 'PLAN_ONLY'
                    planned = []
                    def handoff(sessions, interaction, admitted, interpreter):
                        nonlocal request
                        result.update(admission_success=True, selected_tool=admitted['tool'], admitted=_safe_admitted(admitted))
                        def accept(effective, plan):
                            if admitted['tool'] not in {s.tool_name for s in plan.steps}:
                                result.update(admission_success=False, error_code='INTERPRETER_TARGET_MISSING')
                                raise decisions.IntentError('planning_failed')
                        planner = LLMPlanner(recorder, profile=candidate.profile())
                        app = ResearchAgentApplication(Path(scratch) / 'planning',
                            planner=turns._AdmittedPlanner(planner, accept), registry=guarded)
                        original = AgentRequest(admitted['request_id'], interaction.utterance, admitted['inputs'], RunMode.PLAN_ONLY)
                        request, _ = app._prepare_request(original)
                        run = app.runtime.run(request)
                        planned.append(run)
                        return turns.TurnOutcome('execute', 'planned' if run.status.value == 'PLANNED' else 'failed')
                    with patch.object(dialogue_execution, 'execute', handoff):
                        outcome = context.sessions.respond(context.session_id, context.interaction.turn_id,
                            scenario.utterance, interpreter=recorder,
                            execution_inputs={} if request is None else request.inputs)
                    # Parse the actual interpreter completion, not its fallback presentation.
                    result['presentation_kind'] = outcome.kind
                    decision = decisions.parse_decision(recorder.responses[0]) if recorder.responses[0] is not None else None
                    result.update(contract_success=decision is not None,
                        decision_kind=None if decision is None else _decision_kind(decision),
                        interpreter_decision_kind=None if decision is None else _decision_kind(decision),
                        genuine_clarify=isinstance(decision, decisions.Clarify))
                    if isinstance(decision, decisions.Clarify): result['clarification_reason'] = decision.reason
                    if planned:
                        run = planned[0]
                        recorder.compiled_plan = run.plan
                        counters['scientific_step_results'] += len(run.steps)
                        if run.steps: raise ZeroScienceViolation('PLAN_ONLY produced scientific step results')
                        result.update(planning_only=run.planning_only, run_status=run.status.value,
                            compiler_success=True if run.plan is not None else None,
                            preflight_success=None if run.verification is None else run.verification.passed,
                            plan_tools=[] if run.plan is None else [s.tool_name for s in run.plan.steps])
                        diagnostics = [event.details for event in run.trace if 'diagnostic_schema_version' in event.details]
                        if run.errors and result['error_code'] is None: result['error_code'] = run.errors[0].code
                    elif decision is not None and not result['genuine_clarify']:
                        result.update(admission_success=False, error_code='INTERACTION_NOT_ADMITTED')
                else:
                    raise ValueError('Unknown qualification surface')
            except PlanningModelError as exc:
                result['error_code'] = _provider_code(exc)
            except PlannerError as exc:
                diagnostics = exc.diagnostics
                result['error_code'] = exc.code
                if exc.code == 'UNSUPPORTED_REQUEST':
                    result.update(contract_success=True, decision_kind='unsupported')
                elif recorder.calls and recorder.calls[-1]['provider_outcome'] == 'returned':
                    # Exact parser observation below distinguishes compiler rejection.
                    result['contract_success'] = False
            except decisions.IntentError as exc:
                result['error_code'] = exc.reason
                if result['contract_success'] is True or result['admission_success'] is True:
                    result['admission_success'] = False
                else: result['contract_success'] = False
            except (ValueError, TypeError, KeyError) as exc:
                result.update(contract_success=False, error_code='INVALID_STRUCTURED_RESPONSE')
            _planner_observations(recorder, request, guarded, diagnostics)
            planner_calls = [row for row in recorder.calls if row['surface'] in ('stage_a', 'stage_b')]
            if planner_calls:
                last = planner_calls[-1]
                result['contract_success'] = last['contract_success']
                if last.get('decision_kind') is not None:
                    result['decision_kind'] = last['decision_kind']
                result['request_sources'] = last.get('request_sources', {})
                if last['compiler_success'] is not None:
                    result['compiler_success'] = last['compiler_success']
                if last['preflight_success'] is not None:
                    result['preflight_success'] = last['preflight_success']
                if scenario.surface == 'stage_b':
                    last.update(compiler_success=result['compiler_success'], preflight_success=result['preflight_success'])
            elif recorder.calls:
                recorder.calls[-1].update(contract_success=result['contract_success'], decision_kind=result['decision_kind'])
            if recorder.calls and recorder.calls[-1]['provider_outcome'] == 'failed':
                result['contract_success'] = None
                recorder.calls[-1]['contract_success'] = None
            result['semantic_properties_pass'] = _semantic(scenario, result, context=context) if result['contract_success'] else None
            result['semantic_success'] = (None if scenario.human_review_required
                                           else result['semantic_properties_pass'])
            result['semantic_adjudication'] = ('human_review_pending' if scenario.human_review_required
                                                else 'bounded_scenario_properties')
            if planner_calls and result['compiler_success'] is False and result['error_code'] == 'UNSUPPORTED_REQUEST':
                # A typed refusal never reached compiler; it is not a compiler failure.
                result['compiler_success'] = None
            result['failure_category'] = _classify(result, recorder)
            for row in recorder.calls:
                if row['surface'] == 'interpreter' and row['provider_outcome'] == 'returned':
                    raw = recorder.responses[recorder.calls.index(row)]
                    try:
                        typed = decisions.parse_decision(raw)
                        row.update(contract_success=True, decision_kind=_decision_kind(typed))
                    except decisions.IntentError:
                        row['contract_success'] = False
                row['semantic_success'] = (False if row['decision_kind'] == 'unsupported'
                    and scenario.expected.get('kind') != 'unsupported' else row['semantic_success'])
                row['failure_category'] = ('operational' if row['provider_outcome'] == 'failed'
                    else 'contract' if row['contract_success'] is False
                    else 'deterministic_admission' if row['compiler_success'] is False or row['preflight_success'] is False
                    else 'semantic' if row['semantic_success'] is False else None)
            if len(recorder.calls) == 1:
                recorder.calls[0]['semantic_success'] = result['semantic_success']
                recorder.calls[0]['failure_category'] = result['failure_category']
            result['initial_outcome'] = dict(planner_calls[1] if len(planner_calls) > 1 and planner_calls[0]['surface'] == 'stage_a'
                                             else recorder.calls[0]) if recorder.calls else None
            result['final_outcome'] = dict(decision_kind=result['decision_kind'], contract_success=result['contract_success'],
                semantic_success=result['semantic_success'], compiler_success=result['compiler_success'],
                preflight_success=result['preflight_success'], failure_category=result['failure_category'],
                status=result.get('run_status'))
        result['safety'] = counters
    return AttemptResult(result)
