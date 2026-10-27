"""Qualified H5AD publications use one file interface; consumers own suitability."""
import pytest
from agent.orchestration import build_default_tool_registry, build_semantic_compiler_contract


@pytest.mark.parametrize('producer,key', [
    ('build_scATAC_cell_by_ccre', 'matrix_path'),
    ('build_scATAC_cell_by_features', 'matrix_path'),
    ('adopt_scATAC_cell_by_ccre', 'matrix_path'),
    ('annotate_scATAC_cell_types', 'annotated_h5ad_path'),
])
def test_published_matrix_can_feed_pca_without_intermediate_inspection(producer, key):
    contract = build_semantic_compiler_contract(build_default_tool_registry())
    channels = [c for c in contract.step_output_channels
        if c.producer_tool_name == producer and c.consumer_tool_name == 'pca_embed_cells']
    assert len(channels) == 1
    channel, = channels
    assert channel.source_port == channel.target_port == 'dataset'
    assert [(m.output_key, m.argument_name) for m in channel.members] == [(key, 'input_path')]


@pytest.mark.parametrize('producer,key', [
    ('build_scATAC_cell_by_ccre', 'matrix_path'),
    ('build_scATAC_cell_by_features', 'matrix_path'),
    ('adopt_scATAC_cell_by_ccre', 'matrix_path'),
    ('annotate_scATAC_cell_types', 'annotated_h5ad_path'),
])
def test_published_matrix_can_feed_read_only_inspection(producer, key):
    contract = build_semantic_compiler_contract(build_default_tool_registry())
    channels = [c for c in contract.step_output_channels
        if c.producer_tool_name == producer and c.consumer_tool_name == 'inspect_scATAC']
    assert len(channels) == 1
    channel, = channels
    assert channel.source_port == channel.target_port == 'dataset'
    assert [(m.output_key, m.argument_name) for m in channel.members] == [(key, 'path')]
