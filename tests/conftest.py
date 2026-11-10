"""Generic intent tests use a nonexecuting fixture catalog independent of science.

Those tests exercise source/parameter binding and withhold execution. Native
scATAC tests use the actual production catalog; this fixture does not restore
retired PCA execution or change any production module on disk.
"""
import sys
from pathlib import Path
import pytest

@pytest.fixture
def isolated_binding_catalog(monkeypatch):
    from agent.orchestration import registry as r
    from agent.application import local_resources
    from agent.tools import _native_process
    monkeypatch.setattr(_native_process,'call',lambda operation,arguments,local:local())
    registry = r.build_default_tool_registry()
    native = registry.get('run_snapatac2')
    dataset = r.SemanticConsumerPortSpec('dataset',(r._semantic_member('dataset','input_path'),),True,
        request_sources=(r._semantic_request('input_path',r._semantic_request_member('dataset','input_path')),),
        accepted_upstream_types=('raw_scatac_dataset.v1','scatac_matrix_h5ad.v1','native_matrix_h5ad.v1'))
    def forbidden(input_path, output_dir, n_components=None):
        pytest.fail('Generic binding fixture reached scientific execution')
    spec = r.ToolSpec(name='mock_embed_cells',function=forbidden,exception_classifier=r._classify_analysis_exception,
        required_arguments={key:native.required_arguments[key] for key in ('input_path','output_dir')},
        optional_arguments={'n_components':r._planning_argument((int,type(None)),
            'Explicit numeric value for binding tests only.',scientific_parameter=True)},
        result_contract=r.ResultContract('FixtureEmbeddingResult',
            {key:(str,) for key in ('embedding_path','cell_ids_path','analysis_path')},
            planning_fields={k:r._planning_result('Simulated fixture output',bindable=True) for k in ('embedding_path','cell_ids_path','analysis_path')}),
        recovery_policy_version='fixture-embedding-v1',
        planning=r._tool_planning(r.PlanningToolRole.OPERATION,'Nonexecuting test embedding fixture',capability_ids=('embedding_analysis',)),
        semantic_planning=r.SemanticToolSpec(consumer_ports=(dataset,r._semantic_argument_port('output_dir',required=True),
            r._semantic_argument_port('n_components',required=False)),
            producer_ports=(r.SemanticProducerPortSpec('embedding','cell_embedding_bundle.v1',
                (r._semantic_member('embedding','embedding_path'),r._semantic_member('cell_ids','cell_ids_path')),
                lineage_from_port='dataset'),)))
    extended = r.ToolRegistry(tuple(registry.get(n) for n in registry.names())+(spec,))
    # These owners intentionally use default catalog factories for saved intent
    # replay. Keep every view inside this test on the same explicit fixture.
    production_factory = r.build_default_tool_registry
    active = True
    def factory():
        # Imports made while the fixture is active can retain this callable.
        # Such aliases must return to the production catalog after teardown.
        return extended if active else production_factory()
    for module in list(sys.modules.values()):
        in_tests = str(getattr(module,'__file__','')).startswith(str(Path(__file__).parent) + '/')
        if module is None or not (getattr(module,'__name__','').startswith('agent.') or in_tests):
            continue
        if callable(getattr(module,'build_default_tool_registry',None)):
            monkeypatch.setattr(module,'build_default_tool_registry',factory)
    monkeypatch.setattr(local_resources,'_GROUP_PORTS',local_resources._GROUP_PORTS+(('mock_embed_cells','dataset'),))
    monkeypatch.setattr(local_resources,'_H5AD_COMPANION_TOOLS',local_resources._H5AD_COMPANION_TOOLS+('mock_embed_cells',))
    try:
        yield
    finally:
        active = False
