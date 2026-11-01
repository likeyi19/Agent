"""Lossless complete-contract checks for shared wire-v4 objects."""
from copy import deepcopy
from contextlib import nullcontext
import json

import pytest

from agent.orchestration import AgentRequest, ToolRegistry, build_default_tool_registry
from agent.orchestration import semantic_wire_v4 as wire
from agent.orchestration.active_context import ActiveContextItem, ActivePlanningContext, planning_context
from agent.schemas.prior_output import PriorOutputBinding
from schema_expansion import expand_schema
from test_semantic_wire_v4 import _request


@pytest.fixture(scope='module')
def registry():
    return build_default_tool_registry()


def _pair(monkeypatch, registry, request):
    compact = wire.build_semantic_wire_v4_schema(registry, request)
    with monkeypatch.context() as patch:
        patch.setattr(wire, '_share_closed_objects', lambda schema: schema)
        original = wire.build_semantic_wire_v4_schema(registry, request)
    return original, compact


def _assert_equivalent(original, compact):
    snapshot = deepcopy(original)
    assert expand_schema(original) == expand_schema(compact)
    assert wire._share_closed_objects(original) == compact
    assert original == snapshot
    assert wire._share_closed_objects(compact) == compact
    assert len(json.dumps(compact)) <= len(json.dumps(original))


@pytest.mark.parametrize('case', (
    'inspect_canonical', 'downstream_canonical', 'label_transfer_canonical',
    'label_transfer_optional', 'differential_accessibility_paired_covariates',
))
def test_complete_fixture_grammars_are_identical(monkeypatch, registry, case):
    original, compact = _pair(monkeypatch, registry, _request(case))
    _assert_equivalent(original, compact)
    assert len(compact['$defs']['step']['anyOf']) == len(registry.names()) == 36


@pytest.mark.parametrize('mode', ('no_inputs', 'all_inputs', 'one_missing_member', 'native_inputs'))
def test_all_registered_domains_and_resource_correlations(monkeypatch, registry, mode):
    # Full availability exercises every registered source, including grouped
    # raw inputs, optional resources, EpiZoo species, annotation and native args.
    names = {member.input_name for name in registry.names()
             for port in registry.get(name).semantic_planning.consumer_ports
             for source in port.request_sources for member in source.members}
    if mode == 'no_inputs':
        names = set()
    elif mode == 'one_missing_member':
        names = {name for name in names if not name.endswith('sha256')}
    elif mode == 'native_inputs':
        names = {'input_path', 'function_name', 'kwargs', 'matrix_semantics', 'count_slot', 'output_dir'}
    request = AgentRequest(mode, 'Plan.', dict.fromkeys(sorted(names), 'PRIVATE'))
    _assert_equivalent(*_pair(monkeypatch, registry, request))


def test_historical_handles_and_typed_ports_stay_exact(monkeypatch, registry):
    types = ('chromvar_scores_h5ad.v1', 'scatac_matrix_h5ad.v1', 'scatac_barcode_qc.v1')
    items = tuple(ActiveContextItem(f'ctx.{i}', PriorOutputBinding(
        'PRIVATE:run', 'PRIVATE-step', tool, port, semantic, '1'*64, '2'*64, '3'*64))
        for i, (tool, port, semantic) in enumerate((
            ('compute_motif_accessibility', 'scores', types[0]),
            ('build_scATAC_matrix', 'dataset', types[1]),
            ('compute_scATAC_qc', 'barcode_qc', types[2]))))
    with planning_context(ActivePlanningContext('private-session', 'private-revision', 3, items)):
        original, compact = _pair(monkeypatch, registry, _request('downstream_canonical'))
    _assert_equivalent(original, compact)
    assert 'PRIVATE' not in json.dumps(compact)
    assert all(f'ctx.{i}' in json.dumps(compact) for i in range(3))


def test_sharing_never_merges_nearly_equal_domains_or_collides_with_existing_defs():
    def obj(values):
        return {'type': 'object', 'properties': {'choice': {'enum': values}},
                'required': ('choice',), 'additionalProperties': False}
    a, b = obj(('a', 'b')), obj(('a', 'c'))
    original = {'anyOf': (a, a, b, b), '$defs': {'s0': {'enum': ('keep',)}}}
    compact = wire._share_closed_objects(original)
    _assert_equivalent(original, compact)
    assert compact['$defs']['s0'] == original['$defs']['s0']
    assert compact['anyOf'][0] != compact['anyOf'][2]


def test_registration_order_does_not_affect_sharing(registry):
    reverse = ToolRegistry(tuple(registry.get(name) for name in reversed(registry.names())))
    request = _request('differential_accessibility_paired_covariates')
    assert wire.build_semantic_wire_v4_schema(registry, request) == wire.build_semantic_wire_v4_schema(reverse, request)


def test_no_single_use_definitions_after_parent_factoring(monkeypatch, registry):
    original, compact = _pair(monkeypatch, registry, _request('differential_accessibility_paired_covariates'))
    text = json.dumps(compact)
    for name in compact['$defs'].keys() - original['$defs'].keys():
        assert text.count('"#/$defs/' + name + '"') >= 2


@pytest.mark.parametrize('case', (
    'inspect_canonical', 'downstream_canonical', 'label_transfer_canonical',
    'label_transfer_optional', 'differential_accessibility_paired_covariates',
))
def test_equivalent_wire_requests_compile_to_identical_plans(monkeypatch, registry, case):
    from agent.orchestration import LLMPlanner
    from agent.schemas.run_state import fingerprint_plan
    from test_semantic_wire_v4 import (
        _nested_payload, _wire_step, _downstream_steps, _transfer_steps, _paired_da_steps,
    )
    steps = {
        'inspect_canonical': lambda: (_wire_step('inspect', 'inspect_scATAC'),),
        'downstream_canonical': _downstream_steps,
        'label_transfer_canonical': _transfer_steps,
        'label_transfer_optional': lambda: _transfer_steps(optional=True),
        'differential_accessibility_paired_covariates': _paired_da_steps,
    }[case]()
    response = json.dumps(_nested_payload(steps))

    class Model:
        model_id = 'schema-equivalence'
        schemas = []

        def complete(self, *, prompt, response_schema):
            self.schemas.append(response_schema)
            return response

    model = Model()
    request = _request(case)
    compact_plan = LLMPlanner(model).plan(request, registry)
    with monkeypatch.context() as patch:
        patch.setattr(wire, '_share_closed_objects', lambda schema: schema)
        original_plan = LLMPlanner(model).plan(request, registry)
    assert expand_schema(model.schemas[0]) == expand_schema(model.schemas[1])
    assert compact_plan == original_plan
    assert fingerprint_plan(compact_plan) == fingerprint_plan(original_plan)


@pytest.mark.parametrize('case', ('duplicate', 'unused', 'ambiguous', 'fabricated', 'incompatible'))
def test_compiler_rejections_are_unchanged(monkeypatch, registry, case):
    from agent.orchestration import LLMPlanner, PlannerError
    inputs = {'input_path': '/input.h5ad', 'output_dir': '/out'}
    source = {'target': 'dataset', 'source': {'kind': 'input', 'input': 'input_path'}}
    steps = [{'step_id': 'inspect', 'tool': 'inspect_scATAC',
              'sources': [source], 'control_dependencies': []}]
    context = nullcontext()
    if case == 'duplicate':
        steps[0]['sources'].append(deepcopy(source))
    elif case == 'unused':
        inputs['function_name'] = 'computeVariability'
        steps[0]['tool'] = 'run_chromvar'
        source['source'] = {'kind': 'context', 'handle': 'ctx.0'}
        binding = PriorOutputBinding(
            'prior:run', 'motifs', 'compute_motif_accessibility', 'scores',
            'chromvar_scores_h5ad.v1', '1'*64, '2'*64, '3'*64)
        context = planning_context(ActivePlanningContext(
            'session', 'revision', 1, (ActiveContextItem('ctx.0', binding),)))
    elif case == 'ambiguous':
        inputs = {'reference_input_path': '/ref.h5ad', 'query_input_path': '/query.h5ad'}
        steps[0]['sources'] = []
    elif case == 'fabricated':
        source['source']['input'] = 'fabricated'
    else:
        steps.append({'step_id': 'neighbors', 'tool': 'build_cell_neighbors',
                      'sources': [{'target': 'embedding', 'source': {'kind': 'step', 'step': 'inspect'}}],
                      'control_dependencies': []})

    class Model:
        model_id = 'schema-rejection-equivalence'

        def complete(self, **kwargs):
            return json.dumps({'schema_version': 4, 'decision': {'kind': 'plan', 'steps': steps}})

    request = AgentRequest(case, 'Inspect.', inputs)
    with context:
        with pytest.raises(PlannerError) as compact:
            LLMPlanner(Model()).plan(request, registry)
        with monkeypatch.context() as patch:
            patch.setattr(wire, '_share_closed_objects', lambda schema: schema)
            with pytest.raises(PlannerError) as original:
                LLMPlanner(Model()).plan(request, registry)
    assert compact.value.code == original.value.code
    assert str(compact.value) == str(original.value)
    if case == 'duplicate':
        assert compact.value.code == 'DUPLICATE_TARGET_SOURCE'
    elif case == 'unused':
        assert compact.value.code == 'UNUSED_REQUEST_INPUT'
