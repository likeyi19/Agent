"""Column-name capture for generic label tables and native annotation groups."""
from dataclasses import replace
import hashlib
import io

import anndata as ad
import pandas as pd
import pytest

from agent.application.local_resources import RegisteredInputGroup, source_field_choices, source_field_inputs
from agent.application.dialogue_execution import _bind_arguments
from agent.application.turn_decisions import SourceFieldArgument, IntentError
from agent.orchestration import build_default_tool_registry
from agent.tools.analysis.cell_label_comparison import table_fields


def member(path,kind):
    return dict(resource_id="local-"+hashlib.sha256(str(path).encode()).hexdigest(), record_sha256='b'*64,source_path=str(path),input_type=kind,label='Development source',
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_table_header_names_bind_exact_source_and_replay_without_files(tmp_path):
    a=tmp_path/'prediction.h5ad';ad.AnnData(obs=pd.DataFrame({'predicted':['A','B']},index=['one','two'])).write_h5ad(a)
    b=tmp_path/'reference.attachment';b.write_text('cell_id\treference_type\nlater\tPRIVATE_ROW_VALUE\n')
    group=RegisteredInputGroup([member(a,'h5ad'),member(b,'attachment')],{},attachment_candidates=True)
    choices=source_field_choices(group)
    assert choices[1]['fields']==['cell_id','reference_type'] and choices[1]['complete']
    assert 'PRIVATE_ROW_VALUE' not in str(choices)
    registry=build_default_tool_registry()
    roles=[dict(tool='compare_cell_annotations',port='prediction_dataset',source='source.0'),dict(tool='compare_cell_annotations',port='reference_dataset',source='source.1')]
    fields=source_field_inputs(registry,group,roles,choices)
    utterance='Use the reference annotation labels.';literal='reference annotation'
    d=SourceFieldArgument('compare_cell_annotations','reference_label_key',literal,utterance.index(literal),utterance.index(literal)+len(literal),'reference_type','source.1','offered')
    inputs=_bind_arguments(registry,utterance,{},(d,),field_choices=fields)
    assert inputs['reference_label_key']=='reference_type'
    a.unlink();b.unlink()
    assert source_field_inputs(registry,group,roles,choices)==fields
    with pytest.raises(IntentError):
        _bind_arguments(registry,utterance,{},(replace(d,value='unoffered'),),field_choices=fields)


def test_annotation_column_uses_atomic_path_checksum_source_port(tmp_path):
    p=tmp_path/'counts.h5ad';ad.AnnData(obs=pd.DataFrame({'clusters':['a','b']},index=['one','two'])).write_h5ad(p)
    group=RegisteredInputGroup([member(p,'h5ad')],{})
    registry=build_default_tool_registry();choices=source_field_choices(group)
    fields=source_field_inputs(registry,group,[dict(tool='annotate_scATAC_cell_types',port='source',source='source.0')],choices)
    utterance='Reuse the independent clusters.';literal='independent clusters'
    d=SourceFieldArgument('annotate_scATAC_cell_types','group_column',literal,utterance.index(literal),utterance.index(literal)+len(literal),'clusters','source.0','offered')
    assert _bind_arguments(registry,utterance,{},(d,),field_choices=fields)['group_column']=='clusters'


@pytest.mark.parametrize('raw',[b'a\ta\n',b'a,b,c',b'\x00,other\n',b'x'*16385+b',b\n',b'only-one\n'])
def test_table_header_discovery_is_bounded_and_names_only(raw):
    with pytest.raises(ValueError):table_fields(io.BytesIO(raw))
