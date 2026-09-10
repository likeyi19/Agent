import pytest
from agent.tools._cancellation import cancellation_scope, ToolWorkCancelled
from agent.tools.data import _cell_selection_production as production, _cell_selection_contract as m
from agent.tools.data.scatac_selection_profile import thresholds


def test_cancellation_during_large_table_stream(tmp_path,monkeypatch):
    seen=[]
    def rows(*a):
        for i in range(10000):
            seen.append(i)
            yield 'ns',f'b{i:05}',(2,2,1,1,0,2,0,0)
    monkeypatch.setattr(m,'qc_rows',rows)
    with cancellation_scope(lambda:len(seen)>=1024),pytest.raises(ToolWorkCancelled):
        production.produce('/unused',{},thresholds(dict(min_qc_fragment_records=0,min_tss_enrichment=0)),tmp_path)
    assert len(seen)==1024


def test_production_calculation_reads_only_qc_table(selection_case,tmp_path,monkeypatch):
    from agent.tools.data import _barcode_qc_contract as qc, _barcode_qc_binding as binding
    args,_=selection_case
    q=qc.load_manifest(args['barcode_qc_manifest_path'],args['barcode_qc_manifest_sha256']).to_dict()
    monkeypatch.setattr(binding,'bind',lambda *a:pytest.fail('Selection calculation reread fragments/resources'))
    result=production.produce(args['barcode_qc_manifest_path'],q,thresholds(args),tmp_path)
    assert result['row_count']==q['row_count']
