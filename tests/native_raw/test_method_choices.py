"""Planner method choices use existing scalar ports; no live provider or PBMC data."""
from pathlib import Path
import pytest
from agent.orchestration import AgentRequest, build_default_tool_registry
from agent.orchestration.native_parameters import input_name
from agent.orchestration.semantic_compiler import (
    SemanticPlanCandidate, SemanticPlanStep, SemanticRegisteredChoiceSource as Choice,
    SemanticRequestInputSource as Input, SemanticPlanCompileError,
    SemanticStepOutputSource as Output,
    build_semantic_compiler_contract, compile_semantic_plan,
)
from agent.tools.analysis.native_matrix import SNAPATAC2
from agent.schemas.orchestration import _serialize
from test_task_defaults import sample, native, call

TOOL = 'run_snapatac2'
FUNCTION = 'pp.make_peak_matrix'


def compile_method(*sources, **inputs):
    registry = build_default_tool_registry()
    request = AgentRequest('method-choice', 'Cluster these fragments; cell-type annotation will follow.',
        dict(input_path='/captured/fragments.h5ad', output_dir='/managed/result', **inputs))
    candidate = SemanticPlanCandidate((SemanticPlanStep('matrix', TOOL,
        (Choice('function_name', FUNCTION), *sources)),))
    return compile_semantic_plan(request, candidate, registry, build_semantic_compiler_contract(registry))


def test_planner_can_select_native_counting_without_user_declaration():
    step, = compile_method(Choice('counting_strategy', 'fragment')).steps
    assert step.arguments['counting_strategy'] == 'fragment'
    assert 'kwargs' not in step.arguments
    assert SNAPATAC2.native_arguments(dict(step.arguments)) == {'counting_strategy': 'fragment'}


def test_no_method_choice_keeps_native_default_omitted():
    step, = compile_method().steps
    assert 'counting_strategy' not in step.arguments
    assert SNAPATAC2.native_arguments(dict(step.arguments)) == {}
    # Direct native calls have no task-derived representation choice.
    assert SNAPATAC2.native_arguments(dict(function_name=FUNCTION, input_path='/captured/native')) == {}


def test_explicit_scalar_choice_cannot_be_overridden():
    key = input_name(TOOL, FUNCTION, 'counting_strategy')
    with pytest.raises(SemanticPlanCompileError, match='supplied parameter'):
        compile_method(Choice('counting_strategy', 'fragment'), **{key: 'insertion'})
    step, = compile_method(Input('counting_strategy', key), **{key: 'insertion'}).steps
    assert SNAPATAC2.native_arguments(dict(step.arguments)) == {'counting_strategy': 'insertion'}


def test_existing_native_kwargs_remain_exact_and_conflicts_fail():
    args = dict(function_name=FUNCTION, input_path='/captured/native', kwargs={'counting_strategy': 'insertion', 'chunk_size': 200})
    assert SNAPATAC2.native_arguments(args) == args['kwargs']
    assert SNAPATAC2.native_arguments(dict(args, counting_strategy='insertion')) == args['kwargs']
    with pytest.raises(ValueError, match='conflicting counting_strategy'):
        SNAPATAC2.native_arguments(dict(args, counting_strategy='fragment'))
    assert args['kwargs'] == {'counting_strategy': 'insertion', 'chunk_size': 200}


def test_counting_choice_is_function_scoped_and_not_arbitrary_kwargs():
    with pytest.raises(ValueError, match='make_peak_matrix'):
        SNAPATAC2.native_arguments(dict(function_name='pp.filter_cells', input_path='/captured/native', counting_strategy='fragment'))
    with pytest.raises(SemanticPlanCompileError):
        compile_method(Choice('counting_strategy', 'invented'))
    with pytest.raises(SemanticPlanCompileError):
        compile_method(Choice('kwargs', 'invented'))


def test_compiled_native_kwargs_preserve_user_method():
    key = input_name(TOOL, FUNCTION, 'kwargs')
    step, = compile_method(**{key: {'counting_strategy': 'insertion'}}).steps
    assert 'counting_strategy' not in step.arguments
    assert SNAPATAC2.native_arguments(_serialize(step.arguments)) == {'counting_strategy': 'insertion'}
    # A contradictory choice cannot replace or silently rewrite the user kwargs.
    with pytest.raises(SemanticPlanCompileError, match='supplied native kwargs'):
        compile_method(Choice('counting_strategy', 'fragment'), **{key: {'counting_strategy': 'insertion'}})


@pytest.mark.parametrize('reverse', [False, True])
def test_explicit_native_function_keeps_omitted_defaults(reverse):
    registry = build_default_tool_registry()
    contract = build_semantic_compiler_contract(registry)
    request = AgentRequest('explicit-native', 'Call pp.make_peak_matrix with native defaults.',
        dict(input_path='/captured/native', output_dir='/managed/result', function_name=FUNCTION))
    def compile_sources(sources):
        return compile_semantic_plan(request, SemanticPlanCandidate((
            SemanticPlanStep('matrix', TOOL, tuple(reversed(sources)) if reverse else sources),
        )), registry, contract)
    sources = (Input('function_name', 'function_name'),)
    step, = compile_sources(sources).steps
    assert SNAPATAC2.native_arguments(dict(step.arguments)) == {}
    with pytest.raises(SemanticPlanCompileError, match='retains omitted native defaults'):
        compile_sources((*sources, Choice('counting_strategy', 'fragment')))


def test_user_counting_declaration_uses_existing_function_scope():
    from agent.application.turn_decisions import _ground_declaration, _scientific_arguments
    from agent.application.dialogue_execution import _bind_arguments
    text = 'Use fragment-overlap counts for the matrix.'
    raw = dict(tool=TOOL, argument='counting_strategy', native_function=FUNCTION,
               value='fragment', literal='fragment-overlap counts', evidence='fragment-overlap counts')
    declarations = _scientific_arguments([_ground_declaration(raw, text)])
    assert _bind_arguments(build_default_tool_registry(), text, {}, declarations) == {
        input_name(TOOL, FUNCTION, 'counting_strategy'): 'fragment'}


def test_raw_source_binding_admits_compiled_method_choice(sample):
    import hashlib
    from agent.application.local_resources import LocalResourceAdmission
    owner = LocalResourceAdmission(sample.root/'resources', approved_source_roots=(sample.root,))
    source = owner.register('source', sample.source, input_type='attachment', label='Synthetic fragments', attribution='test')
    values = {input_name(TOOL, 'pp.import_fragments', 'fragment_order'): 'unknown'}
    for function, port, path in [('pp.import_fragments', 'chrom_sizes', sample.sizes), (FUNCTION, 'peaks', sample.peaks)]:
        values[input_name(TOOL, function, port+'_path')] = str(path)
        values[input_name(TOOL, function, port+'_sha256')] = hashlib.sha256(path.read_bytes()).hexdigest()
    binding = owner.compose_native(source.resource_id, values)
    request = AgentRequest('composed', 'QC and cluster; annotation will follow.',
                           dict(binding.execution_inputs, output_dir=str(sample.root/'compiled')))
    registry = build_default_tool_registry()
    plan = compile_semantic_plan(request, SemanticPlanCandidate((
        SemanticPlanStep('import', TOOL, (Choice('function_name', 'pp.import_fragments'), Input('dataset', 'input_path'),
                                         Input('input_sha256', 'input_sha256'))),
        SemanticPlanStep('matrix', TOOL, (Choice('function_name', FUNCTION), Output('dataset', 'import', 'dataset'),
                                         Choice('counting_strategy', 'fragment'))),
    )), registry, build_semantic_compiler_contract(registry))
    owner._validate_native_consumption(binding, plan.steps)
    assert 'counting_strategy' not in plan.steps[0].arguments
    assert plan.steps[1].arguments['counting_strategy'] == 'fragment'
    assert not any('counting_strategy' in key for key in request.inputs)
    assert plan.steps[1].arguments['peaks_path'] == str(sample.peaks)


def test_real_selected_native_counts_and_publication_read_without_replay(native, sample, monkeypatch):
    import anndata as ad
    from agent.tools.analysis import annotation_contract
    imported = call(sample, 'source', 'pp.import_fragments', chrom_sizes_path=str(sample.sizes), fragment_order='unknown')
    selected = call(sample, 'selected-method', FUNCTION, imported['matrix_path'],
                    peaks_path=str(sample.peaks), counting_strategy='fragment')
    explicit = call(sample, 'explicit-method', FUNCTION, imported['matrix_path'],
                    peaks_path=str(sample.peaks), kwargs={'counting_strategy': 'fragment'})
    a, b = (ad.read_h5ad(r['matrix_path']) for r in (selected, explicit))
    assert (a.X != b.X).nnz == 0 and a.obs_names.equals(b.obs_names) and a.var_names.equals(b.var_names)
    assert selected['measurement'] == 'fragment_counts'
    assert annotation_contract.input_problems('human', 'hg38', selected['measurement']) == []
    assert selected['parameters']['kwargs'] == {}
    assert selected['parameters']['counting_strategy'] == 'fragment'
    assert selected['parameters']['effective_native_parameters']['counting_strategy'] == 'fragment'
    before = Path(selected['matrix_path']).read_bytes()
    monkeypatch.setattr(SNAPATAC2, 'native', lambda *a: pytest.fail('Recovery repeated native science'))
    SNAPATAC2.verify_publication(selected['manifest_path'], expected_sha256=selected['manifest_sha256'])
    assert Path(selected['matrix_path']).read_bytes() == before
