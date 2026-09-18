from pathlib import Path
import pytest
from agent.tools.data import explicit_cells as c
from agent.tools.data.authority_context import VerificationContext, authority_operation


@pytest.mark.parametrize('rows', [[], [('lib','z'),('lib','A')], [('a','same'),('b','same')]])
def test_exact_order_and_empty(tmp_path, rows):
    ptr = c.publish_explicit_cells(cells=rows, declaration='Explicit caller choice', output_dir=tmp_path/'cells')
    value = c.verify_explicit_cells(ptr['manifest_path'], expected_sha256=ptr['manifest_sha256'])
    actual = list(c.iter_rows(tmp_path/'cells/cells.tsv.gz'))
    assert [(r[1],r[2]) for r in actual] == rows
    assert value['selected_count'] == len(rows)
    assert not {'qc', 'thresholds', 'cell_call_method', 'cell_call_state'} & value.keys()
    with pytest.raises(ValueError): c.publish_explicit_cells(cells=rows, declaration='x', output_dir=tmp_path/'cells')


@pytest.mark.parametrize('rows', [[('lib','A'),('lib','A')], [('','A')], [('lib','')],
    [('lib','a b')], [('lib','a\tb')], [('lib','é')], [('bad namespace','A')], [('lib','A'*257)],
    [(None,'A')], [('lib',None)], ['AB'], [('lib',)]])
def test_invalid_or_duplicate(tmp_path, rows):
    with pytest.raises(ValueError):
        c.publish_explicit_cells(cells=rows, declaration='Explicit choice', output_dir=tmp_path/'cells')
    assert not (tmp_path/'cells').exists()


def test_mutation_invalidates_cell_proof(tmp_path):
    context = VerificationContext()
    with authority_operation(context):
        ptr = c.publish_explicit_cells(cells=[('lib','A')], declaration='Choice', output_dir=tmp_path/'cells')
        c.verify_explicit_cells(ptr['manifest_path'], expected_sha256=ptr['manifest_sha256'])
        table = Path(ptr['manifest_path']).parent/'cells.tsv.gz'
        table.write_bytes(table.read_bytes() + b'changed')
        with pytest.raises(ValueError): c.verify_explicit_cells(ptr['manifest_path'], expected_sha256=ptr['manifest_sha256'])
