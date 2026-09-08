import json
from dataclasses import replace

import pytest

from agent.orchestration import (
    AgentRequest, LLMPlanner, PlanningWireMode, PlannerError, RunMode,
    build_default_tool_registry, build_semantic_planning_catalog,
    SemanticPlanCandidate, SemanticPlanStep, SemanticRequestInputSource,
    SemanticStepOutputSource, SemanticPlanCompileError, compile_semantic_plan,
    build_m92_semantic_compiler_contract,
)
from agent.orchestration.registry import ArtifactSemanticKind, PlanningToolRole, ToolRegistry
from agent.orchestration.error_policy import build_recovery_policy_snapshot


def wire(sources=None):
    return {'schema_version': 4, 'decision': {'kind': 'plan', 'steps': [{
        'step_id': 'raw', 'tool': 'inspect_raw_scATAC',
        'sources': [] if sources is None else sources, 'control_dependencies': []}]}}


class Model:
    model_id = 'scripted-raw-intake'

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def complete(self, *, prompt, response_schema):
        self.calls += 1
        self.prompt, self.schema = prompt, response_schema
        return json.dumps(self.payload)


def request():
    return AgentRequest('raw-request',
        'Inspect these raw mouse scATAC FASTQs and tell me whether they are ready for preprocessing.',
        {'raw_input_paths': ['/PRIVATE/sample_S1_R1_001.fastq', '/PRIVATE/sample_S1_R2_001.fastq'],
         'output_dir': '/PRIVATE/managed', 'species': 'mouse', 'raw_assay': 'TENX_ATAC',
         'fastq_layout': 'tenx-atac-r1-i2-r2.v1'}, RunMode.PLAN_ONLY)


def compile_candidate(req, candidate, registry):
    return compile_semantic_plan(req, candidate, registry, build_m92_semantic_compiler_contract(registry))


def test_registry_and_semantic_artifact_boundaries():
    registry = build_default_tool_registry()
    assert len(registry.names()) == 12
    assert [n for n in registry.names() if 'raw' in n] == ['inspect_raw_scATAC']
    assert ArtifactSemanticKind.RAW_SCATAC.value == 'raw_scatac'
    assert len({ArtifactSemanticKind.RAW_SCATAC, ArtifactSemanticKind.RAW_SCATAC_SEQUENCING,
                ArtifactSemanticKind.RAW_SCATAC_INTAKE_MANIFEST}) == 3
    old = registry.get('inspect_scATAC')
    assert old.semantic_planning.producer_ports[0].semantic_type == 'raw_scatac_dataset.v1'
    spec = registry.get('inspect_raw_scATAC')
    assert spec.planning.role is PlanningToolRole.INSPECTION
    assert set(spec.required_arguments) == {'raw_input_paths', 'output_dir'}
    assert set(spec.optional_arguments) == {'species', 'raw_assay', 'source_genome_assembly', 'fastq_layout'}
    ports = spec.semantic_planning
    assert {p.name for p in ports.consumer_ports} == {'raw_input', 'output_dir', *spec.optional_arguments}
    raw_port = next(p for p in ports.consumer_ports if p.name == 'raw_input')
    assert [s.selector for s in raw_port.request_sources] == ['raw_input_paths']
    assert not raw_port.accepted_upstream_types
    assert len(ports.producer_ports) == 1
    producer = ports.producer_ports[0]
    assert producer.name == 'intake_manifest'
    assert producer.semantic_type == 'raw_scatac_intake_manifest.v1'
    assert {m.field_name for m in producer.members} == {'manifest_path', 'manifest_sha256'}
    assert set(spec.result_contract.planning_fields) == {'manifest_path', 'manifest_sha256'}


def test_wire_v4_selections_compile_structured_values_without_model_literals():
    req, registry = request(), build_default_tool_registry()
    sources = [{'target': target, 'source': {'kind': 'input', 'input': name}}
               for target, name in [('raw_input', 'raw_input_paths'), ('species', 'species'),
                   ('raw_assay', 'raw_assay'), ('fastq_layout', 'fastq_layout')]]
    model = Model(wire(sources))
    plan = LLMPlanner(model).plan(req, registry)
    assert model.calls == 1
    assert dict(plan.steps[0].arguments) == dict(req.inputs)
    assert plan.steps[0].depends_on == ()
    assert '/PRIVATE' not in model.prompt and '/PRIVATE' not in str(model.schema)
    catalog = str(build_semantic_planning_catalog(req, registry))
    assert 'raw_input_paths' in catalog and 'array' in catalog
    assert 'inspect_raw_scATAC' in catalog
    for forbidden in ('inspect_fastq_inputs', 'inspect_bam_inputs', '/PRIVATE'):
        assert forbidden not in catalog
    assert 'input_kind' not in str(model.payload)
    assert 'target_genome_assembly' not in str(model.payload)
    assert 'output_dir' not in str(model.payload)  # Unique managed source bound mechanically.


def test_single_tool_unique_request_bindings_need_no_redundant_sources():
    req = request()
    plan = LLMPlanner(Model(wire())).plan(req, build_default_tool_registry())
    assert dict(plan.steps[0].arguments) == dict(req.inputs)


@pytest.mark.parametrize('target', ['input_kind', 'barcode_tag', 'target_genome_assembly'])
def test_unoffered_target_is_rejected(target):
    model = Model(wire([{'target': target, 'source': {'kind': 'input', 'input': 'raw_input_paths'}}]))
    with pytest.raises(PlannerError):
        LLMPlanner(model).plan(request(), build_default_tool_registry())


def test_h5ad_request_name_is_not_authorized_for_raw_input():
    req = AgentRequest('r', 'inspect', {'input_path': '/x.h5ad', 'output_dir': '/out'})
    candidate = SemanticPlanCandidate((SemanticPlanStep('raw', 'inspect_raw_scATAC',
        (SemanticRequestInputSource('raw_input', 'input_path'),)),))
    with pytest.raises(SemanticPlanCompileError):
        compile_candidate(req, candidate, build_default_tool_registry())


@pytest.mark.parametrize('direction', ['raw-to-epizoo', 'h5ad-to-raw'])
def test_incompatible_semantic_channels_fail_closed(direction):
    registry = build_default_tool_registry()
    req = AgentRequest('r', 'inspect', {'input_path': '/x.h5ad', 'raw_input_paths': '/x.bam',
        'output_dir': '/out', 'species': 'human'})
    steps = (SemanticPlanStep('raw', 'inspect_raw_scATAC'),
        SemanticPlanStep('embed', 'epizoo_embed_cells',
            (SemanticStepOutputSource('dataset', 'raw', 'intake_manifest'),))) if direction == 'raw-to-epizoo' else (
        SemanticPlanStep('processed', 'inspect_scATAC'),
        SemanticPlanStep('raw', 'inspect_raw_scATAC',
            (SemanticStepOutputSource('raw_input', 'processed', 'dataset'),)))
    with pytest.raises(SemanticPlanCompileError):
        compile_candidate(req, SemanticPlanCandidate(steps), registry)


def test_registry_addition_does_not_change_historical_recovery_identity():
    current = build_default_tool_registry()
    old = ToolRegistry(tuple(current.get(n) for n in current.names() if n != 'inspect_raw_scATAC'))
    req = AgentRequest('r', 'inspect', {'input_path': '/x.h5ad'})
    candidate = SemanticPlanCandidate((SemanticPlanStep('inspect', 'inspect_scATAC'),))
    a, b = (compile_candidate(req, candidate, r) for r in (old, current))
    assert a == b
    assert build_recovery_policy_snapshot(a, old, max_attempts_per_step=2) == build_recovery_policy_snapshot(
        b, current, max_attempts_per_step=2)


def test_explicit_v3_high_level_request_binding():
    req = request()
    payload = {'schema_version': 3, 'status': 'plan', 'reason': None, 'steps': [{
        'step_id': 'raw', 'tool_name': 'inspect_raw_scATAC', 'arguments': {
            'source_genome_assembly': None,
            **{name: {'binding_type': 'input', 'input_name': name} for name in req.inputs}},
        'depends_on': [], 'description': 'Inspect raw sequencing inputs.'}]}
    # V3 has its existing keyed binding shape; no compatibility parser changes.
    model = Model(payload)
    plan = LLMPlanner(model, wire_mode=PlanningWireMode.V3).plan(req, build_default_tool_registry())
    assert dict(plan.steps[0].arguments) == dict(req.inputs)


def test_v3_can_reject_input_free_requests_with_extended_registry():
    model = Model({'schema_version': 3, 'status': 'unsupported', 'steps': [], 'reason': 'No scientific inputs.'})
    with pytest.raises(PlannerError) as exc:
        LLMPlanner(model, wire_mode=PlanningWireMode.V3).plan(AgentRequest('r', 'explain', {}), build_default_tool_registry())
    assert exc.value.code == 'UNSUPPORTED_REQUEST' and model.calls == 1
