"""Scripted v4/v3 selection; no scientific IO or runtime configuration needed."""
import json
from dataclasses import replace
import pytest
from agent.orchestration import (AgentRequest, AgentRuntime, LLMPlanner, RunMode, RunStatus,
    PlanningWireMode, PlannerError, StepOutputRef, build_default_tool_registry)
from agent.orchestration.registry import ArtifactSemanticKind, ToolRegistry
from test_fragments_orchestration import Model


def request(upstream=False):
    inputs={'library_context_path':'/PRIVATE/context.json','library_context_sha256':'1'*64,
        'reference_bundle_path':'/PRIVATE/reference.json','reference_bundle_sha256':'2'*64,
        'output_dir':'/PRIVATE/output'}
    if upstream:inputs.update(raw_input_paths='/PRIVATE/raw.fastq',species='human',raw_assay='TENX_ATAC')
    else:inputs.update(intake_manifest_path='/PRIVATE/intake.json',intake_manifest_sha256='3'*64)
    return AgentRequest('fragments-request','Prepare these verified raw scATAC FASTQs into canonical fragments.',inputs,RunMode.PLAN_ONLY)


def wire(upstream=False):
    sources=[{'target':port,'source':{'kind':'input','input':name}} for port,name in (
        ('library_context','library_context_path'),('reference','reference_bundle_path'))]
    sources.append({'target':'intake','source':{'kind':'step','step':'inspect'}} if upstream else
        {'target':'intake','source':{'kind':'input','input':'intake_manifest_path'}})
    steps=([{'step_id':'inspect','tool':'inspect_raw_scATAC','sources':[],'control_dependencies':[]}] if upstream else [])
    steps.append({'step_id':'fragments','tool':'prepare_scATAC_fragments','sources':sources,'control_dependencies':[]})
    return {'schema_version':4,'decision':{'kind':'plan','steps':steps}}


@pytest.mark.parametrize('upstream',[False,True])
def test_v4_plan_only_without_runtime_or_scientific_io(monkeypatch,upstream):
    from agent.tools.data import scatac_fragments as public, fastq_fragments as private
    from agent.tools.data import _fragments_binding as binding, _fragments_fastq as scan
    from agent.tools.data import scatac_fragments_verifier as verify
    def forbidden(*a,**k):pytest.fail('Scientific IO during PLAN_ONLY')
    monkeypatch.delenv('AGENT_CHROMAP_BIN',raising=False);monkeypatch.delenv('AGENT_CHROMAP_INDEX_ROOT',raising=False)
    for module,name in ((public,'resolve_execution'),(public,'verification_runtime'),(private,'prepare_fastq_fragments'),
        (binding,'preflight'),(scan,'scan_group'),(verify,'verify_fragments')):monkeypatch.setattr(module,name,forbidden)
    registry=build_default_tool_registry()
    registry=ToolRegistry(tuple(replace(registry.get(n),function=forbidden) for n in registry.names()))
    model=Model(wire(upstream));req=request(upstream)
    result=AgentRuntime(registry=registry,planner=LLMPlanner(model)).run(req)
    assert result.status is RunStatus.PLANNED and not result.steps,result.errors
    step=result.plan.steps[-1]
    assert step.arguments['library_context_path']==req.inputs['library_context_path']
    assert step.arguments['library_context_sha256']==req.inputs['library_context_sha256']
    assert step.arguments['reference_bundle_sha256']==req.inputs['reference_bundle_sha256']
    assert step.arguments['output_dir']==req.inputs['output_dir']
    assert step.depends_on == (('inspect',) if upstream else ())
    if upstream:
        assert step.arguments['intake_manifest_path']==StepOutputRef('inspect','manifest_path')
        assert step.arguments['intake_manifest_sha256']==StepOutputRef('inspect','manifest_sha256')
    else:assert step.arguments['intake_manifest_sha256']==req.inputs['intake_manifest_sha256']
    assert '/PRIVATE' not in model.prompt and '/PRIVATE' not in str(model.schema)
    for token in ('Chromap','bgzip','tabix','AGENT_CHROMAP','support_bits'):
        assert token not in model.prompt
    assert 'scatac_fragments.v1' in model.prompt
    assert 'output_dir' not in json.dumps(model.payload)


def test_registry_ports_and_executable_allowlist():
    registry=build_default_tool_registry();spec=registry.get('prepare_scATAC_fragments')
    assert set(spec.required_arguments)=={'intake_manifest_path','intake_manifest_sha256','library_context_path',
        'library_context_sha256','reference_bundle_path','reference_bundle_sha256','output_dir'}
    assert not spec.optional_arguments and not spec.retryable_error_codes
    assert spec.recovery_policy_version=='prepare-scatac-fragments-fastq-v1'
    ports={p.name:p for p in spec.semantic_planning.consumer_ports}
    assert set(ports)=={'intake','library_context','reference','output_dir'}
    assert ports['intake'].accepted_upstream_types==('raw_scatac_intake_manifest.v1',)
    assert not ports['library_context'].accepted_upstream_types and not ports['reference'].accepted_upstream_types
    assert spec.semantic_planning.producer_ports[0].semantic_type=='scatac_fragments.v1'
    assert len(spec.semantic_planning.producer_ports)==1
    assert ArtifactSemanticKind.SCATAC_FRAGMENTS not in (ArtifactSemanticKind.RAW_SCATAC,
        ArtifactSemanticKind.RAW_SCATAC_SEQUENCING,ArtifactSemanticKind.RAW_SCATAC_INTAKE_MANIFEST)


@pytest.mark.parametrize('target',['chromap','index','threads','whitelist','dedup','tn5'])
def test_unoffered_choices_rejected(target):
    payload=wire();payload['decision']['steps'][0]['sources'].append(
        {'target':target,'source':{'kind':'input','input':'reference_bundle_path'}})
    with pytest.raises(PlannerError):LLMPlanner(Model(payload)).plan(request(),build_default_tool_registry())


def test_direct_route_never_auto_inserts_inspection():
    plan=LLMPlanner(Model(wire())).plan(request(),build_default_tool_registry())
    assert [s.tool_name for s in plan.steps]==['prepare_scATAC_fragments']


def test_missing_context_hash_fails_closed():
    req=request();inputs=dict(req.inputs);del inputs['library_context_sha256']
    with pytest.raises(PlannerError):LLMPlanner(Model(wire())).plan(replace(req,inputs=inputs),build_default_tool_registry())


def test_explicit_v3_registry_projection():
    req=request()
    payload={'schema_version':3,'status':'plan','reason':None,'steps':[{'step_id':'fragments',
        'tool_name':'prepare_scATAC_fragments','arguments':{n:{'binding_type':'input','input_name':n} for n in req.inputs},
        'depends_on':[],'description':'Prepare canonical fragments.'}]}
    plan=LLMPlanner(Model(payload),wire_mode=PlanningWireMode.V3).plan(req,build_default_tool_registry())
    assert dict(plan.steps[0].arguments)==dict(req.inputs)


@pytest.mark.parametrize('port',['library_context','reference'])
def test_request_only_artifacts_cannot_come_from_intake_producer(port):
    payload=wire(True)
    sources=payload['decision']['steps'][-1]['sources']
    sources[:]=[s for s in sources if s['target']!=port]
    sources.append({'target':port,'source':{'kind':'step','step':'inspect'}})
    with pytest.raises(PlannerError):LLMPlanner(Model(payload)).plan(request(True),build_default_tool_registry())
