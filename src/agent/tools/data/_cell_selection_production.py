"""Streaming explicit decisions over freshly verified QC rows only."""
import gzip
import hashlib
from . import _cell_selection_contract as m, _barcode_qc_contract as qc
from .scatac_selection_profile import decide, encode_cell_id, REASONS
from agent.tools._cancellation import cancellation_checkpoint


def produce(qc_path, value, policy, stage):
    reasons={r:0 for r in REASONS};selected=0;n=0;order=hashlib.sha256(m.ORDER_DOMAIN)
    with (stage/'decisions.tsv.gz').open('xb') as dr, (stage/'selected.tsv.gz').open('xb') as sr:
        with gzip.GzipFile(filename='',mode='wb',fileobj=dr,mtime=0,compresslevel=6) as d, gzip.GzipFile(filename='',mode='wb',fileobj=sr,mtime=0,compresslevel=6) as s:
            d.write(m.DECISION_HEADER);s.write(m.SELECTED_HEADER)
            for ns,bc,counts in m.qc_rows(qc_path,value):
                n+=1;failed=decide(counts,policy);rendered=encode_cell_id(ns,bc);chosen='false' if failed else 'true'
                d.write(('\t'.join((ns,bc,rendered,'not_assessed',chosen,chosen,','.join(failed) or 'NONE'))+'\n').encode('ascii'))
                for reason in failed: reasons[reason]+=1
                if not failed:
                    s.write(f'{selected}\t{ns}\t{bc}\t{rendered}\n'.encode('ascii'))
                    order.update(qc.identity(ns,bc));selected+=1
                if n%1024==0: cancellation_checkpoint()
    return dict(row_count=n,selected_count=selected,rejected_count=n-selected,
        cell_call_not_assessed_count=n,reason_counts=reasons,
        readiness='selected_candidates_available' if selected else 'no_selected_cells',
        ordered_selected_sha256=order.hexdigest(),decisions=qc.resource(stage/'decisions.tsv.gz'),selected=qc.resource(stage/'selected.tsv.gz'))
