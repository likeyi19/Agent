"""Scripted planning, real adoption, durable recovery, cancellation and reporting."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from agent.application import ResearchAgentApplication
from agent.orchestration import (AgentRequest, AgentRuntime, AgentPlan, PlanStep, LLMPlanner,
    PlanningWireMode, PlannerError, RunMode, FileRunStore, StepStatus, ToolRegistry, build_default_tool_registry)
from agent.tools.data import scatac_bam_fragments as public, bam_fragments as production
from agent.tools.data import bam_fragment_manifest as m
from agent.report import build_analysis_evidence, verify_analysis_evidence, AnalysisEvidenceError


class Model:
    model_id = 'scripted-external'
    def __init__(self, payload): self.payload = payload
    def complete(self, *, prompt, response_schema):
        self.prompt = prompt; self.schema = response_schema
        return json.dumps(self.payload)


def wire():
    ports = [('source','source_path'),('intake','intake_manifest_path'),
             ('library_context','library_context_path'),('reference','reference_bundle_path'),
             ('source_profile','source_profile')]
    return {'schema_version':4,'decision':{'kind':'plan','steps':[
        {'step_id':'adopt','tool':'prepare_scATAC_bam_fragments',
         'sources':[{'target':p,'source':{'kind':'input','input':n}} for p,n in ports],
         'control_dependencies':[]}]}}


def request(args, mode=RunMode.EXECUTE):
    return AgentRequest('external', 'Adopt these declared external ATAC fragments without changing biological records.',
                        {k: v for k, v in args.items() if k != 'output_dir'}, mode)


class FixedPlanner:
    def __init__(self, args, downstream=False):
        steps = [PlanStep('adopt', 'prepare_scATAC_bam_fragments', args)]
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
        completed = any(s.tool_name == 'prepare_scATAC_bam_fragments' and s.status is StepStatus.SUCCEEDED for s in state.steps)
        if completed and self.before: raise ProcessExit()
        saved = self.store.update(state, expected_revision=expected_revision)
        if completed:
            if self.cancel: self.cancel(saved.run_id)
            else: raise ProcessExit()
        return saved


def test_application_v4_plan_only_zero_scientific_io(tmp_path,monkeypatch):
    args={k:('/PRIVATE/'+k if k.endswith('_path') else '1'*64) for k in m.ARGUMENTS if k not in ('output_dir','source_profile')}
    args['source_profile']=m.PROFILE_ID
    def forbidden(*a,**k):pytest.fail('Scientific IO during PLAN_ONLY')
    from agent.tools.data import _bam_fragment_io as io, scatac_reference
    for module,name in ((production,'prepare_in_stage'),(public,'verification_runtime'),
                        (io,'project'),(io,'resource'),(io,'bind'),(scatac_reference,'load_scatac_reference_bundle')):
        monkeypatch.setattr(module,name,forbidden)
    registry=build_default_tool_registry()
    registry=ToolRegistry(tuple(replace(registry.get(n),function=forbidden) for n in registry.names()))
    model=Model(wire());app=ResearchAgentApplication(tmp_path/'workspace',planner=LLMPlanner(model),registry=registry)
    result=app.run(request(args,RunMode.PLAN_ONLY))
    assert result.status.value=='PLANNED',result
    assert not result.run_result.steps and result.evidence is result.report is result.visualization is None
    for key,value in args.items():assert result.run_result.plan.steps[0].arguments[key]==value
    assert '/PRIVATE' not in model.prompt and '/PRIVATE' not in str(model.schema)
    assert not list(app._workspace.run_paths(result.run_id).scientific.iterdir())


def test_v3_and_v4_grouped_bindings(bam_factory):
    args=bam_factory();req=AgentRequest('bam','Prepare declared BAM fragments.',args,RunMode.PLAN_ONLY)
    payload={'schema_version':3,'status':'plan','reason':None,'steps':[{'step_id':'adopt',
        'tool_name':'prepare_scATAC_bam_fragments','arguments':{n:{'binding_type':'input','input_name':n} for n in args},
        'depends_on':[],'description':'Prepare BAM fragments.'}]}
    assert dict(LLMPlanner(Model(payload),wire_mode=PlanningWireMode.V3).plan(req,build_default_tool_registry()).steps[0].arguments)==args
    assert dict(LLMPlanner(Model(wire())).plan(req,build_default_tool_registry()).steps[0].arguments)==args


def test_reviewed_intake_upstream_pair_expands_without_compiler_special_case(bam_factory):
    from agent.orchestration import StepOutputRef
    args=bam_factory();args['raw_input_paths']=[args['source_path']]
    del args['intake_manifest_path']; del args['intake_manifest_sha256']
    payload=wire();steps=payload['decision']['steps']
    for source in steps[0]['sources']:
        if source['target']=='intake':source['source']={'kind':'step','step':'intake'}
    steps.insert(0,{'step_id':'intake','tool':'inspect_raw_scATAC',
        'sources':[{'target':'raw_input','source':{'kind':'input','input':'raw_input_paths'}}],
        'control_dependencies':[]})
    plan=LLMPlanner(Model(payload)).plan(AgentRequest('bam','Inspect the selected BAM and prepare declared fragments.',args,RunMode.PLAN_ONLY),build_default_tool_registry())
    step=plan.steps[1]
    assert step.arguments['intake_manifest_path']==StepOutputRef('intake','manifest_path')
    assert step.arguments['intake_manifest_sha256']==StepOutputRef('intake','manifest_sha256')
    assert step.depends_on==('intake',)


def test_application_upstream_intake_executes_with_existing_context(bam_factory,tmp_path):
    args=bam_factory();args['raw_input_paths']=[args['source_path']]
    args.update(species='human',raw_assay='SCATAC',source_genome_assembly='hg38')
    del args['intake_manifest_path'];del args['intake_manifest_sha256']
    payload=wire();steps=payload['decision']['steps']
    for source in steps[0]['sources']:
        if source['target']=='intake':source['source']={'kind':'step','step':'intake'}
    steps.insert(0,{'step_id':'intake','tool':'inspect_raw_scATAC',
        'sources':[{'target':'raw_input','source':{'kind':'input','input':'raw_input_paths'}}],
        'control_dependencies':[]})
    app=ResearchAgentApplication(tmp_path/'workspace',planner=LLMPlanner(Model(payload)))
    result=app.run(request(args))
    assert result.status.value=='SUCCEEDED',result
    assert result.evidence and result.report
    assert [s.tool_name for s in result.run_result.steps]==['inspect_raw_scATAC','prepare_scATAC_bam_fragments']


@pytest.mark.parametrize('missing',['source_sha256','intake_manifest_sha256','library_context_sha256','reference_bundle_sha256','source_profile'])
def test_missing_explicit_binding(bam_factory,missing):
    args=bam_factory();del args[missing]
    with pytest.raises(PlannerError):LLMPlanner(Model(wire())).plan(request(args,RunMode.PLAN_ONLY),build_default_tool_registry())


def test_registry_contract_and_no_hidden_result_io(bam_factory, monkeypatch):
    args = bam_factory(); result = public.prepare_scATAC_bam_fragments(**args)
    registry = build_default_tool_registry(); spec = registry.get('prepare_scATAC_bam_fragments')
    assert len(registry.names()) == 18
    assert spec.recovery_policy_version == public.RECOVERY_POLICY
    assert not spec.retryable_error_codes
    assert spec.semantic_planning.producer_ports[0].semantic_type == 'scatac_fragments.v2'
    monkeypatch.setattr(public, 'verify_public_result', lambda *a: pytest.fail('Scientific IO in result validation'))
    registry.validate_result(spec.name, result)
    for changed in (result | {'total_support': True}, result | {'n_libraries': 2}, result | {'arbitrary': 1}):
        with pytest.raises(ValueError): registry.validate_result(spec.name, changed)


@pytest.mark.parametrize('mutation', [None, 'source', 'bgzf', 'tabix', 'manifest'])
def test_application_execute_report_resume_and_corruption(bam_factory, tmp_path, monkeypatch, mutation):
    args = bam_factory()
    app = ResearchAgentApplication(tmp_path / 'workspace', planner=LLMPlanner(Model(wire())))
    result = app.run(request(args))
    assert result.status.value == 'SUCCEEDED', result
    assert result.evidence and result.report and result.visualization is None
    text = Path(result.report.path).read_text()
    assert 'BAM fragment preparation succeeded' in text and 'not independently established' in text
    assert 'not called or QC-passed cells' in text and 'not Cell Ranger reproduction' in text
    assert 'aCgT-1' not in text
    evidence=json.loads(Path(result.evidence.path).read_bytes());facts=evidence['steps'][0]['facts']
    assert facts['route']=='bam_fragment_production' and facts['bam_transformation_verification']=='independently_recomputed'
    assert facts['eligible_pairs']==facts['total_support']==1
    assert len(evidence['artifacts'])==8
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
@pytest.mark.parametrize('mutation', [None, 'source', 'missing_receipt', 'corrupt_receipt', 'wrong_receipt', 'wrong_policy'])
def test_exact_durable_recovery(bam_factory, tmp_path, monkeypatch, before, mutation):
    args = bam_factory(); store = FileRunStore(tmp_path / 'store')
    with pytest.raises(ProcessExit):
        AgentRuntime(planner=FixedPlanner(args), run_store=StopStore(store, before)).run(AgentRequest('external', 'Adopt.', {}))
    state = store.load('external:run')
    assert state.steps[0].status is (StepStatus.RUNNING if before else StepStatus.SUCCEEDED)
    receipt = next(Path(args['output_dir']).glob('bam-fragments-*/receipt.json'))
    if mutation == 'source': Path(args['source_path']).write_bytes(b'changed')
    elif mutation == 'missing_receipt': receipt.unlink()
    elif mutation == 'corrupt_receipt': receipt.write_bytes(b'corrupt')
    elif mutation == 'wrong_receipt':
        value = json.loads(receipt.read_bytes()); value['arguments_sha256'] = '0'*64; receipt.write_text(json.dumps(value))
    elif mutation == 'wrong_policy':
        value = json.loads(receipt.read_bytes()); value['policy'] = 'unreviewed-policy'; receipt.write_text(json.dumps(value))
    monkeypatch.setattr(production, 'prepare_in_stage', lambda *a: pytest.fail('Adoption repeated during recovery'))
    result = AgentRuntime(run_store=store).resume('external:run')
    assert result.status.value == ('FAILED' if mutation else 'SUCCEEDED'), result.errors


def test_application_publication_before_checkpoint(bam_factory, tmp_path, monkeypatch):
    args = bam_factory(); root = tmp_path / 'workspace'
    app = ResearchAgentApplication(root, planner=LLMPlanner(Model(wire())))
    app.runtime._run_store = StopStore(app.run_store)
    with pytest.raises(ProcessExit): app.run(request(args))
    monkeypatch.setattr(production, 'prepare_in_stage', lambda *a: pytest.fail('Adoption repeated'))
    recovered = ResearchAgentApplication(root).resume('external:run')
    assert recovered.status.value == 'SUCCEEDED' and recovered.report and recovered.evidence


def test_cancellation_before_and_after_adoption(bam_factory, tmp_path):
    args = bam_factory(); store = FileRunStore(tmp_path / 'store')
    registry = build_default_tool_registry()
    def forbidden(*a, **k): pytest.fail('Downstream step ran after cancellation')
    registry = ToolRegistry(tuple(replace(registry.get(n), function=forbidden) if n == 'inspect_scATAC' else registry.get(n) for n in registry.names()))
    runtime = AgentRuntime(registry=registry, planner=FixedPlanner(args, True), run_store=store)
    runtime._run_store = StopStore(store, before=False, cancel=lambda run_id: runtime.cancel(run_id))
    result = runtime.run(AgentRequest('external', 'Adopt.', {}))
    assert result.status.value == 'CANCELLED' and result.steps[0].status is StepStatus.SUCCEEDED


def test_no_overwrite_or_automatic_rerun(bam_factory):
    args = bam_factory(); result = public.prepare_scATAC_bam_fragments(**args)
    before = Path(result['manifest_path']).read_bytes()
    with pytest.raises(m.BamFragmentsError, match='CONFLICT'): public.prepare_scATAC_bam_fragments(**args)
    assert Path(result['manifest_path']).read_bytes() == before


def test_cancel_before_execution_has_no_adoption_io(bam_factory, tmp_path, monkeypatch):
    args = bam_factory()
    class CancelOnCreate(FileRunStore):
        def create(self, state):
            saved = super().create(state); self.request_cancellation(state.run_id); return saved
    monkeypatch.setattr(production, 'prepare_in_stage', lambda *a: pytest.fail('Adoption ran after cancellation'))
    result = AgentRuntime(planner=FixedPlanner(args), run_store=CancelOnCreate(tmp_path / 'store')).run(
        AgentRequest('external', 'Adopt.', {}))
    assert result.status.value == 'CANCELLED'
    assert not Path(args['output_dir']).exists()


def test_recovery_checkpoint_failure_can_be_resumed(bam_factory, tmp_path, monkeypatch):
    from agent.orchestration.run_store import RunStoreError
    args = bam_factory(); store = FileRunStore(tmp_path / 'store')
    with pytest.raises(ProcessExit):
        AgentRuntime(planner=FixedPlanner(args), run_store=StopStore(store)).run(AgentRequest('external', 'Adopt.', {}))
    class FailedCheckpoint:
        def __getattr__(self, name): return getattr(store, name)
        def update(self, *a, **k): raise RunStoreError('Injected checkpoint failure.')
    monkeypatch.setattr(production, 'prepare_in_stage', lambda *a: pytest.fail('Adoption repeated'))
    with pytest.raises(RunStoreError): AgentRuntime(run_store=FailedCheckpoint()).resume('external:run')
    assert store.load('external:run').steps[0].status is StepStatus.RUNNING
    assert AgentRuntime(run_store=store).resume('external:run').status.value == 'SUCCEEDED'


def test_unpublished_running_step_is_not_reexecuted(bam_factory, tmp_path, monkeypatch):
    args = bam_factory(); store = FileRunStore(tmp_path / 'store')
    def crash(*a, **k): raise ProcessExit()
    monkeypatch.setattr(production, 'prepare_in_stage', crash)
    with pytest.raises(ProcessExit):
        AgentRuntime(planner=FixedPlanner(args), run_store=store).run(AgentRequest('external', 'Adopt.', {}))
    monkeypatch.setattr(production, 'prepare_in_stage', lambda *a: pytest.fail('Unpublished step automatically rerun'))
    result = AgentRuntime(run_store=store).resume('external:run')
    assert result.status.value == 'FAILED'
    assert any(e.code == 'BAM_FRAGMENTS_RECOVERY_UNAVAILABLE' for e in result.errors)


@pytest.mark.parametrize('field', ['policy', 'result_contract', 'facts'])
def test_evidence_authority_is_explicit_and_closed(bam_factory, tmp_path, field):
    args = bam_factory(); registry = build_default_tool_registry()
    run = AgentRuntime(planner=FixedPlanner(args), registry=registry).run(AgentRequest('external', 'Adopt.', {}))
    if field == 'facts':
        result = build_analysis_evidence(run, tmp_path / 'evidence', registry=registry)
        path = Path(result['evidence_path']); payload = json.loads(path.read_bytes())
        payload['steps'][0]['facts']['total_support'] += 1; path.write_text(json.dumps(payload))
        assert not verify_analysis_evidence(run, path, registry=registry).passed
    else:
        spec = registry.get('prepare_scATAC_bam_fragments')
        if field == 'policy': changed = replace(spec, recovery_policy_version='unreviewed')
        else: changed = replace(spec, result_contract=replace(spec.result_contract,
            required_fields={**spec.result_contract.required_fields, 'unreviewed': (str,)}))
        other = ToolRegistry(tuple(changed if n == spec.name else registry.get(n) for n in registry.names()))
        with pytest.raises(AnalysisEvidenceError):
            build_analysis_evidence(run, tmp_path / 'evidence', registry=other)


@pytest.mark.parametrize('before',[True,False])
def test_application_cancellation_boundaries(bam_factory,tmp_path,monkeypatch,before):
    args=bam_factory();app=ResearchAgentApplication(tmp_path/'workspace',planner=FixedPlanner(args,True))
    store=app.run_store
    if before:
        class CancelCreate:
            def __getattr__(self,name):return getattr(store,name)
            def create(self,state):
                saved=store.create(state);store.request_cancellation(state.run_id);return saved
        app.runtime._run_store=CancelCreate()
        monkeypatch.setattr(production,'prepare_in_stage',lambda *a:pytest.fail('BAM ran after cancellation'))
    else:
        app.runtime._run_store=StopStore(store,before=False,cancel=lambda run_id:app.runtime.cancel(run_id))
    result=app.run(request(args))
    assert result.status.value=='CANCELLED',result
    if before:assert not Path(args['output_dir']).exists()
    else:
        assert result.run_result.steps[0].status is StepStatus.SUCCEEDED
        assert Path(result.run_result.steps[0].result['manifest_path']).is_file()
