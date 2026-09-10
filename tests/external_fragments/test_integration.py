"""Scripted planning, real adoption, durable recovery, cancellation and reporting."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from agent.application import ResearchAgentApplication
from agent.orchestration import (AgentRequest, AgentRuntime, AgentPlan, PlanStep, LLMPlanner,
    PlanningWireMode, PlannerError, RunMode, FileRunStore, StepStatus, ToolRegistry, build_default_tool_registry)
from agent.tools.data import scatac_fragment_import as public, external_fragments as production
from agent.tools.data import external_fragment_manifest as m
from agent.report import build_analysis_evidence, verify_analysis_evidence, AnalysisEvidenceError


class Model:
    model_id = 'scripted-external'
    def __init__(self, payload): self.payload = payload
    def complete(self, *, prompt, response_schema):
        self.prompt = prompt; self.schema = response_schema
        return json.dumps(self.payload)


def wire(index=False):
    ports = [('source', 'source_path'), ('reference', 'reference_bundle_path'),
             ('source_profile', 'source_profile'), ('namespace', 'namespace')]
    if index: ports.append(('source_index', 'source_index_path'))
    return {'schema_version': 4, 'decision': {'kind': 'plan', 'steps': [
        {'step_id': 'adopt', 'tool': 'import_scATAC_fragments',
         'sources': [{'target': port, 'source': {'kind': 'input', 'input': name}} for port, name in ports],
         'control_dependencies': []}]}}


def request(args, mode=RunMode.EXECUTE):
    return AgentRequest('external', 'Adopt these declared external ATAC fragments without changing biological records.',
                        {k: v for k, v in args.items() if k != 'output_dir'}, mode)


class FixedPlanner:
    def __init__(self, args, downstream=False):
        steps = [PlanStep('adopt', 'import_scATAC_fragments', args)]
        if downstream:
            steps.append(PlanStep('after', 'inspect_scATAC', {'path': '/never-open'}, ('adopt',)))
        self.plan_value = AgentPlan('external-plan', 'external', 'Adopt external fragments.', tuple(steps))
    def plan(self, request, registry): return self.plan_value


class ProcessExit(BaseException): pass


class StopStore:
    def __init__(self, store, before=True, cancel=None):
        self.store = store; self.before = before; self.cancel = cancel
    def __getattr__(self, name): return getattr(self.store, name)
    def update(self, state, *, expected_revision):
        completed = any(s.tool_name == 'import_scATAC_fragments' and s.status is StepStatus.SUCCEEDED for s in state.steps)
        if completed and self.before: raise ProcessExit()
        saved = self.store.update(state, expected_revision=expected_revision)
        if completed:
            if self.cancel: self.cancel(saved.run_id)
            else: raise ProcessExit()
        return saved


@pytest.mark.parametrize('index', [False, True])
def test_application_v4_plan_only_zero_scientific_io(tmp_path, monkeypatch, index):
    args = dict(source_path='/PRIVATE/external.dat', source_sha256='1'*64, source_profile=m.PROFILE_ID,
        namespace='explicit', reference_bundle_path='/PRIVATE/reference', reference_bundle_sha256='2'*64)
    if index: args.update(source_index_path='/PRIVATE/index', source_index_sha256='3'*64)
    def forbidden(*a, **k): pytest.fail('Scientific IO during PLAN_ONLY')
    from agent.tools.data import _external_fragment_io, scatac_reference
    for module, name in ((production, 'prepare_in_stage'), (public, 'verification_runtime'),
        (_external_fragment_io, 'resource'), (scatac_reference, 'load_scatac_reference_bundle')):
        monkeypatch.setattr(module, name, forbidden)
    registry = build_default_tool_registry()
    registry = ToolRegistry(tuple(replace(registry.get(n), function=forbidden) for n in registry.names()))
    model = Model(wire(index))
    app = ResearchAgentApplication(tmp_path / 'workspace', planner=LLMPlanner(model), registry=registry)
    result = app.run(request(args, RunMode.PLAN_ONLY))
    assert result.status.value == 'PLANNED', result
    assert not result.run_result.steps and result.evidence is result.report is result.visualization is None
    step = result.run_result.plan.steps[0]
    for key, value in args.items(): assert step.arguments[key] == value
    assert '/PRIVATE' not in model.prompt and '/PRIVATE' not in str(model.schema)
    assert not list(app._workspace.run_paths(result.run_id).scientific.iterdir())


def test_explicit_v3_and_optional_parameter_preservation(source_factory):
    args = source_factory(encoding='bgzf', indexed=True, selection='subset_export')
    req = AgentRequest('external', 'Adopt external fragments.', args, RunMode.PLAN_ONLY)
    payload = {'schema_version': 3, 'status': 'plan', 'reason': None, 'steps': [
        {'step_id': 'adopt', 'tool_name': 'import_scATAC_fragments', 'arguments': {
            n: {'binding_type': 'input', 'input_name': n} for n in args},
         'depends_on': [], 'description': 'Adopt external fragments.'}]}
    plan = LLMPlanner(Model(payload), wire_mode=PlanningWireMode.V3).plan(req, build_default_tool_registry())
    assert dict(plan.steps[0].arguments) == args
    plan4 = LLMPlanner(Model(wire(True))).plan(req, build_default_tool_registry())
    assert dict(plan4.steps[0].arguments) == args
    assert len(plan4.steps) == 1


@pytest.mark.parametrize('missing', ['source_sha256', 'reference_bundle_sha256', 'namespace', 'source_profile'])
def test_missing_explicit_choices_fail_closed(source_factory, missing):
    args = source_factory(); del args[missing]
    with pytest.raises(PlannerError):
        LLMPlanner(Model(wire())).plan(request(args, RunMode.PLAN_ONLY), build_default_tool_registry())


def test_registry_contract_and_no_hidden_result_io(source_factory, monkeypatch):
    args = source_factory(); result = public.import_scATAC_fragments(**args)
    registry = build_default_tool_registry(); spec = registry.get('import_scATAC_fragments')
    assert len(registry.names()) == 17
    assert spec.recovery_policy_version == public.RECOVERY_POLICY
    assert not spec.retryable_error_codes
    assert spec.semantic_planning.producer_ports[0].semantic_type == 'scatac_fragments.v2'
    monkeypatch.setattr(public, 'verify_public_result', lambda *a: pytest.fail('Scientific IO in result validation'))
    registry.validate_result(spec.name, result)
    for changed in (result | {'total_support': True}, result | {'n_libraries': 2}, result | {'arbitrary': 1}):
        with pytest.raises(ValueError): registry.validate_result(spec.name, changed)


@pytest.mark.parametrize('mutation', [None, 'source', 'bgzf', 'tabix', 'manifest'])
def test_application_execute_report_resume_and_corruption(source_factory, tmp_path, monkeypatch, mutation):
    args = source_factory(strand=True, encoding='gzip')
    app = ResearchAgentApplication(tmp_path / 'workspace', planner=LLMPlanner(Model(wire())))
    result = app.run(request(args))
    assert result.status.value == 'SUCCEEDED', result
    assert result.evidence and result.report and result.visualization is None
    text = Path(result.report.path).read_text()
    assert 'External fragment adoption succeeded' in text and 'not independently reconstructed' in text
    assert 'not called or QC-passed cells' in text and 'No additional Tn5 shift' in text
    assert 'Canonical FASTQ fragments' not in text and 'aCgT-1' not in text and 'opaque sample' not in text
    evidence = json.loads(Path(result.evidence.path).read_bytes())
    facts = evidence['steps'][0]['facts']
    assert facts['route'] == 'external_fragment_adoption' and facts['conservation_verification'] == 'verified'
    assert facts['source_record_count'] == facts['n_fragment_records'] == 2
    assert facts['source_selection'] == 'unknown' and facts['source_encoding'] == 'gzip'
    assert len(evidence['artifacts']) == 6
    if mutation:
        step = result.run_result.steps[0]; path = Path(step.result['manifest_path'])
        value = json.loads(path.read_bytes())
        target = Path(args['source_path']) if mutation == 'source' else path if mutation == 'manifest' else path.parent / value['libraries'][0][mutation]['path']
        target.write_bytes(target.read_bytes() + b'changed')
    monkeypatch.setattr(production, 'prepare_in_stage', lambda *a: pytest.fail('Adoption repeated on resume'))
    resumed = ResearchAgentApplication(tmp_path / 'workspace').resume(result.run_id)
    if mutation:
        assert resumed.status.value == 'FAILED' and resumed.error.code == 'APP_EVIDENCE_FAILED'
    else:
        assert resumed == result and Path(resumed.report.path).read_text() == text


@pytest.mark.parametrize('before', [False, True])
@pytest.mark.parametrize('mutation', [None, 'source', 'missing_receipt', 'corrupt_receipt', 'wrong_receipt'])
def test_exact_durable_recovery(source_factory, tmp_path, monkeypatch, before, mutation):
    args = source_factory(); store = FileRunStore(tmp_path / 'store')
    with pytest.raises(ProcessExit):
        AgentRuntime(planner=FixedPlanner(args), run_store=StopStore(store, before)).run(AgentRequest('external', 'Adopt.', {}))
    state = store.load('external:run')
    assert state.steps[0].status is (StepStatus.RUNNING if before else StepStatus.SUCCEEDED)
    receipt = next(Path(args['output_dir']).glob('external-fragments-*/receipt.json'))
    if mutation == 'source': Path(args['source_path']).write_bytes(b'changed')
    elif mutation == 'missing_receipt': receipt.unlink()
    elif mutation == 'corrupt_receipt': receipt.write_bytes(b'corrupt')
    elif mutation == 'wrong_receipt':
        value = json.loads(receipt.read_bytes()); value['arguments_sha256'] = '0'*64; receipt.write_text(json.dumps(value))
    monkeypatch.setattr(production, 'prepare_in_stage', lambda *a: pytest.fail('Adoption repeated during recovery'))
    result = AgentRuntime(run_store=store).resume('external:run')
    assert result.status.value == ('FAILED' if mutation else 'SUCCEEDED'), result.errors


def test_application_publication_before_checkpoint(source_factory, tmp_path, monkeypatch):
    args = source_factory(); root = tmp_path / 'workspace'
    app = ResearchAgentApplication(root, planner=LLMPlanner(Model(wire())))
    app.runtime._run_store = StopStore(app.run_store)
    with pytest.raises(ProcessExit): app.run(request(args))
    monkeypatch.setattr(production, 'prepare_in_stage', lambda *a: pytest.fail('Adoption repeated'))
    recovered = ResearchAgentApplication(root).resume('external:run')
    assert recovered.status.value == 'SUCCEEDED' and recovered.report and recovered.evidence


def test_cancellation_before_and_after_adoption(source_factory, tmp_path):
    args = source_factory(); store = FileRunStore(tmp_path / 'store')
    registry = build_default_tool_registry()
    def forbidden(*a, **k): pytest.fail('Downstream step ran after cancellation')
    registry = ToolRegistry(tuple(replace(registry.get(n), function=forbidden) if n == 'inspect_scATAC' else registry.get(n) for n in registry.names()))
    runtime = AgentRuntime(registry=registry, planner=FixedPlanner(args, True), run_store=store)
    runtime._run_store = StopStore(store, before=False, cancel=lambda run_id: runtime.cancel(run_id))
    result = runtime.run(AgentRequest('external', 'Adopt.', {}))
    assert result.status.value == 'CANCELLED' and result.steps[0].status is StepStatus.SUCCEEDED


def test_no_overwrite_or_automatic_rerun(source_factory):
    args = source_factory(); result = public.import_scATAC_fragments(**args)
    before = Path(result['manifest_path']).read_bytes()
    with pytest.raises(m.ExternalFragmentsError, match='CONFLICT'): public.import_scATAC_fragments(**args)
    assert Path(result['manifest_path']).read_bytes() == before


def test_v4_complete_catalog_size_and_explicit_null_index(source_factory):
    from agent.orchestration.semantic_prompt import build_semantic_planning_prompt
    from agent.orchestration.semantic_wire_v4 import build_semantic_wire_v4_schema
    registry = build_default_tool_registry()
    query = AgentRequest('size-regression', 'Inspect this scATAC-seq dataset and summarize its stored matrix metadata.',
                         {'input_path': '/synthetic/inspect.h5ad'})
    prompt = build_semantic_planning_prompt(query, registry)
    schema = build_semantic_wire_v4_schema(registry, query)
    size = len(prompt.encode()) + len(json.dumps(schema, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode())
    assert size <= 30_500  # M11.4c: complete 17-tool catalog is 30,480 bytes; budget unchanged.
    assert 'import_scATAC_fragments' in prompt and 'Historical processing remains declared' in prompt
    args = source_factory() | {'source_index_path': None, 'source_index_sha256': None}
    query = AgentRequest('external', 'Adopt external fragments.', args, RunMode.PLAN_ONLY)
    plan = LLMPlanner(Model(wire(True))).plan(query, registry)
    assert 'source_index_path' in plan.steps[0].arguments
    assert plan.steps[0].arguments['source_index_path'] is None
    assert plan.steps[0].arguments['source_index_sha256'] is None



def test_cancel_before_execution_has_no_adoption_io(source_factory, tmp_path, monkeypatch):
    args = source_factory()
    class CancelOnCreate(FileRunStore):
        def create(self, state):
            saved = super().create(state); self.request_cancellation(state.run_id); return saved
    monkeypatch.setattr(production, 'prepare_in_stage', lambda *a: pytest.fail('Adoption ran after cancellation'))
    result = AgentRuntime(planner=FixedPlanner(args), run_store=CancelOnCreate(tmp_path / 'store')).run(
        AgentRequest('external', 'Adopt.', {}))
    assert result.status.value == 'CANCELLED'
    assert not Path(args['output_dir']).exists()


def test_recovery_checkpoint_failure_can_be_resumed(source_factory, tmp_path, monkeypatch):
    from agent.orchestration.run_store import RunStoreError
    args = source_factory(); store = FileRunStore(tmp_path / 'store')
    with pytest.raises(ProcessExit):
        AgentRuntime(planner=FixedPlanner(args), run_store=StopStore(store)).run(AgentRequest('external', 'Adopt.', {}))
    class FailedCheckpoint:
        def __getattr__(self, name): return getattr(store, name)
        def update(self, *a, **k): raise RunStoreError('Injected checkpoint failure.')
    monkeypatch.setattr(production, 'prepare_in_stage', lambda *a: pytest.fail('Adoption repeated'))
    with pytest.raises(RunStoreError): AgentRuntime(run_store=FailedCheckpoint()).resume('external:run')
    assert store.load('external:run').steps[0].status is StepStatus.RUNNING
    assert AgentRuntime(run_store=store).resume('external:run').status.value == 'SUCCEEDED'


def test_unpublished_running_step_is_not_reexecuted(source_factory, tmp_path, monkeypatch):
    args = source_factory(); store = FileRunStore(tmp_path / 'store')
    def crash(*a, **k): raise ProcessExit()
    monkeypatch.setattr(production, 'prepare_in_stage', crash)
    with pytest.raises(ProcessExit):
        AgentRuntime(planner=FixedPlanner(args), run_store=store).run(AgentRequest('external', 'Adopt.', {}))
    monkeypatch.setattr(production, 'prepare_in_stage', lambda *a: pytest.fail('Unpublished step automatically rerun'))
    result = AgentRuntime(run_store=store).resume('external:run')
    assert result.status.value == 'FAILED'
    assert any(e.code == 'EXTERNAL_FRAGMENTS_RECOVERY_UNAVAILABLE' for e in result.errors)


@pytest.mark.parametrize('field', ['policy', 'result_contract', 'facts'])
def test_evidence_authority_is_explicit_and_closed(source_factory, tmp_path, field):
    args = source_factory(); registry = build_default_tool_registry()
    run = AgentRuntime(planner=FixedPlanner(args), registry=registry).run(AgentRequest('external', 'Adopt.', {}))
    if field == 'facts':
        result = build_analysis_evidence(run, tmp_path / 'evidence', registry=registry)
        path = Path(result['evidence_path']); payload = json.loads(path.read_bytes())
        payload['steps'][0]['facts']['total_support'] += 1; path.write_text(json.dumps(payload))
        assert not verify_analysis_evidence(run, path, registry=registry).passed
    else:
        spec = registry.get('import_scATAC_fragments')
        if field == 'policy': changed = replace(spec, recovery_policy_version='unreviewed')
        else: changed = replace(spec, result_contract=replace(spec.result_contract,
            required_fields={**spec.result_contract.required_fields, 'unreviewed': (str,)}))
        other = ToolRegistry(tuple(changed if n == spec.name else registry.get(n) for n in registry.names()))
        with pytest.raises(AnalysisEvidenceError):
            build_analysis_evidence(run, tmp_path / 'evidence', registry=other)
