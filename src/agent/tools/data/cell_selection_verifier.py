"""Independent threshold parsing, decisions, identity rendering and reconstruction."""
import base64
from fractions import Fraction
import hashlib
from pathlib import Path
import re

from . import _cell_selection_contract as m, _barcode_qc_contract as qc
from .barcode_qc_verifier import verify_barcode_qc
from .scatac_selection_profile import fail, REASONS
from .scatac_fragments_v2_verifier import take_snapshots, check_snapshots
from agent.tools._cancellation import cancellation_checkpoint


def canonical_thresholds(args):
    result={}
    for key in ('min_qc_fragment_records','min_tss_enrichment','min_tss_flank_evidence','max_qc_fragment_records','max_nucleosome_signal'):
        raw=args.get(key)
        if raw is None:
            if key in ('min_qc_fragment_records','min_tss_enrichment'): fail('SELECTION_THRESHOLD_REQUIRED')
            result[key]=None;continue
        if key not in ('min_tss_enrichment','max_nucleosome_signal'):
            if type(raw) is not int or raw<0 or raw>10**18: fail('SELECTION_THRESHOLD_INVALID')
            result[key]=raw;continue
        if type(raw) is int:
            numerator,denominator=raw,1
        elif type(raw) is str and len(raw)<=80 and re.fullmatch(r'(?:0|[1-9][0-9]{0,18})(?:\.[0-9]{1,18}|/[1-9][0-9]{0,18})?',raw):
            if '/' in raw: numerator,denominator=map(int,raw.split('/'))
            elif '.' in raw:
                whole,decimal=raw.split('.');denominator=10**len(decimal);numerator=int(whole)*denominator+int(decimal)
            else: numerator,denominator=int(raw),1
        else: fail('SELECTION_THRESHOLD_INVALID')
        f=Fraction(numerator,denominator)
        if f<0 or max(f.numerator,f.denominator)>10**18: fail('SELECTION_THRESHOLD_INVALID')
        result[key]=dict(numerator=f.numerator,denominator=f.denominator)
    return result


def failures(counts, t):
    depth=counts[1];background=counts[3]+counts[4]
    ratio=lambda v:Fraction(v['numerator'],v['denominator'])
    flags={
        REASONS[0]:depth<t['min_qc_fragment_records'],
        REASONS[1]:background==0,
        REASONS[2]:background>0 and Fraction(200*counts[2],101*background)<ratio(t['min_tss_enrichment']),
        REASONS[3]:t['min_tss_flank_evidence'] is not None and background<t['min_tss_flank_evidence'],
        REASONS[4]:t['max_qc_fragment_records'] is not None and depth>t['max_qc_fragment_records'],
        REASONS[5]:t['max_nucleosome_signal'] is not None and counts[5]==0,
        REASONS[6]:t['max_nucleosome_signal'] is not None and counts[5]>0 and Fraction(counts[6],counts[5])>ratio(t['max_nucleosome_signal'])}
    return tuple(k for k in REASONS if flags[k])


def verify_cell_selection(path, *, expected_sha256):
    path=Path(path);before=take_snapshots([path]);manifest=m.load_manifest(path,expected_sha256);value=manifest.to_dict();args=value['arguments']
    qp=Path(args['barcode_qc_manifest_path']);q=verify_barcode_qc(qp,expected_sha256=args['barcode_qc_manifest_sha256']).to_dict()
    before=tuple(sorted(before+take_snapshots([qp,qp.parent/q['table']['path'],qp.parent/q['histogram']['path'],path.parent/'decisions.tsv.gz',path.parent/'selected.tsv.gz'])))
    if value['qc_identity_sha256']!=q['identity_sha256'] or value['qc_lineage']!=m.lineage(q): fail('SELECTION_QC_LINEAGE_MISMATCH')
    t=canonical_thresholds(args)
    if t!=value['thresholds']: fail('SELECTION_THRESHOLD_MISMATCH')
    for key in ('decisions','selected'):
        if qc.resource(path.parent/value[key]['path'])!=value[key]: fail('SELECTION_SIDECAR_MISMATCH')
    decisions=qc.gzip_lines(path.parent/'decisions.tsv.gz',qc.MAX_BARCODES+1)
    selected_rows=qc.gzip_lines(path.parent/'selected.tsv.gz',qc.MAX_BARCODES+1)
    if next(decisions,None)!=m.DECISION_HEADER or next(selected_rows,None)!=m.SELECTED_HEADER: fail('SELECTION_ROW_INVALID')
    n=0;selected=0;reason_counts={r:0 for r in REASONS};order=hashlib.sha256(m.ORDER_DOMAIN)
    for ns,bc,counts in m.qc_rows(qp,q):
        n+=1;failed=failures(counts,t)
        rendered='scatac-cell.v1.'+'.'.join(base64.b64encode(x.encode('ascii')).decode('ascii').replace('+','-').replace('/','_').rstrip('=') for x in (ns,bc))
        state='false' if failed else 'true'
        expected=f'{ns}\t{bc}\t{rendered}\tnot_assessed\t{state}\t{state}\t'+(','.join(failed) if failed else 'NONE')+'\n'
        if next(decisions,None)!=expected.encode('ascii'): fail('SELECTION_DECISION_MISMATCH')
        for reason in failed: reason_counts[reason]+=1
        if not failed:
            if next(selected_rows,None)!=f'{selected}\t{ns}\t{bc}\t{rendered}\n'.encode('ascii'): fail('SELECTION_SELECTED_MISMATCH')
            order.update((ns+'\t'+bc+'\n').encode('ascii'));selected+=1
        if n%1024==0: cancellation_checkpoint()
    if next(decisions,None) is not None or next(selected_rows,None) is not None: fail('SELECTION_ROW_MISMATCH')
    if (n!=value['row_count'] or selected!=value['selected_count'] or reason_counts!=value['reason_counts']
        or order.hexdigest()!=value['ordered_selected_sha256']): fail('SELECTION_SUMMARY_MISMATCH')
    check_snapshots(before);cancellation_checkpoint()
    return manifest
