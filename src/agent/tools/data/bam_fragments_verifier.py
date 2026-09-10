"""Independent BAM scientific reconstruction; never imports production helpers."""
from dataclasses import dataclass
import hashlib
from itertools import groupby, zip_longest
from pathlib import Path
import re
import tempfile

from . import bam_fragment_manifest as m, _bam_fragment_io as io, _external_fragment_io as physical
from . import scatac_fragments_v2 as v2
from ._fragments_common import digest
from .scatac_fragments_v2_verifier import verify_fragments_v2, FragmentVerification
from ._fragment_io import _bgzf_lines


@dataclass(frozen=True)
class BamProductionVerification:
    fragments: FragmentVerification
    source_content: str = 'verified'
    transformation: str = 'independently_recomputed'
    producer_history: str = 'declared_not_independently_established'


def _reference(header, names, bound):
    entries = header.get('SQ')
    if not isinstance(entries,list) or not 1 <= len(entries) <= 4096 or len(entries) != len(names): m.fail('BAM_FRAGMENTS_HEADER_INVALID')
    lengths = dict(bound['contigs']); seen = set(); checks=[]
    for i,entry in enumerate(entries):
        name=entry.get('SN'); length=entry.get('LN')
        if (name in seen or name not in lengths or type(length) is not int or lengths[name]!=length
                or names[i] != (name,length)): m.fail('BAM_FRAGMENTS_REFERENCE_MISMATCH')
        seen.add(name)
        if 'M5' in entry: checks.append(name)
    md5=io.sequence_md5(bound['bundle'],checks)
    for entry in entries:
        if 'M5' in entry and (not isinstance(entry['M5'],str) or entry['M5'].lower()!=md5[entry['SN']]): m.fail('BAM_FRAGMENTS_REFERENCE_MISMATCH')
    groups={}; libraries=set()
    for rg in header.get('RG',[]):
        name=rg.get('ID')
        if not name or name in groups or len(groups)>=256: m.fail('BAM_FRAGMENTS_HEADER_INVALID')
        groups[name]=rg
        if rg.get('LB'): libraries.add(rg['LB'])
    if len(libraries)>1 or (libraries and libraries!={bound['library'].source_library_id}): m.fail('BAM_FRAGMENTS_LIBRARY_MISMATCH')
    return groups


def _read(record, references, groups):
    flag=record['flag']; tags={}
    if type(flag) is not int or flag<0 or flag>4095: m.fail('BAM_FRAGMENTS_FLAGS_INVALID')
    for key,value,kind in record['tags']:
        printable = all(32<=ord(ch)<=126 for ch in value)
        if key in tags or kind!='Z' or not value or len(value)>65536 or not printable or (key!='RG' and ' ' in value): m.fail('BAM_FRAGMENTS_TAG_INVALID')
        tags[key]=value
    if 'CB' in tags and len(tags['CB'])>256: m.fail('BAM_FRAGMENTS_TAG_INVALID')
    if 'RG' in tags and tags['RG'] not in groups: m.fail('BAM_FRAGMENTS_LIBRARY_MISMATCH')
    if 'SA' in tags:
        values=tags['SA'].split(';')
        if values.pop()!='' or not values: m.fail('BAM_FRAGMENTS_TAG_INVALID')
        for value in values:
            columns=value.split(',')
            if (len(columns)!=6 or not columns[0] or columns[2] not in ('+','-')
                    or re.fullmatch('[1-9][0-9]*',columns[1]) is None
                    or re.fullmatch(r'(?:[1-9][0-9]*[MIDNSHP=X])+',columns[3]) is None
                    or any(re.fullmatch('[0-9]+',columns[j]) is None for j in (4,5))): m.fail('BAM_FRAGMENTS_TAG_INVALID')
    for rid,pos in ((record['rid'],record['start']),(record['mrid'],record['mpos'])):
        if rid < -1 or rid >= len(references) or pos < -1: m.fail('BAM_FRAGMENTS_COORDINATE_INVALID')
        if (rid==-1 and pos!=-1) or (rid!=-1 and not 0<=pos<references[rid][1]): m.fail('BAM_FRAGMENTS_COORDINATE_INVALID')
    cigar=record['cigar'] or []; span=qlen=0; unsupported=False
    for i,(operation,size) in enumerate(cigar):
        if operation not in range(9) or size<1: m.fail('BAM_FRAGMENTS_CIGAR_INVALID')
        if operation in (0,2,3,7,8): span+=size
        if operation in (0,1,4,7,8): qlen+=size
        if operation==5 and i!=0 and i!=len(cigar)-1: m.fail('BAM_FRAGMENTS_CIGAR_INVALID')
        if operation==4:
            left_ok=all(op==5 for op,_ in cigar[:i]); right_ok=all(op==5 for op,_ in cigar[i+1:])
            if not (left_ok or right_ok): m.fail('BAM_FRAGMENTS_CIGAR_INVALID')
        unsupported |= operation in (3,6)
    if cigar and record['seq_len'] is not None and record['seq_len']!=qlen: m.fail('BAM_FRAGMENTS_CIGAR_INVALID')
    end=record['start']+span
    if not flag&4:
        if (record['rid']==-1 or span==0 or not cigar or end>references[record['rid']][1]
                or record['end']!=end): m.fail('BAM_FRAGMENTS_COORDINATE_INVALID')
    if cigar:
        endpoint=cigar[-1][0] if flag&16 else cigar[0][0]
        unsupported |= endpoint not in (0,7,8)
    return record,tags,end,unsupported


def _template(primaries, has_supplement, references):
    if 64 not in primaries or 128 not in primaries: m.fail('BAM_FRAGMENTS_TEMPLATE_INVALID')
    x,y=primaries[64],primaries[128]; a,ta,ae,ua=x; b,tb,be,ub=y
    if ta.get('RG')!=tb.get('RG'): m.fail('BAM_FRAGMENTS_LIBRARY_MISMATCH')
    if 'CB' in ta and 'CB' in tb and ta['CB']!=tb['CB']: m.fail('BAM_FRAGMENTS_BARCODE_MISMATCH')
    for current,other in ((a,b),(b,a)):
        if ((current['flag']>>3)&1)!=((other['flag']>>2)&1): m.fail('BAM_FRAGMENTS_MATE_MISMATCH')
        if other['flag']&4 == 0:
            if ((current['flag']>>5)&1)!=((other['flag']>>4)&1): m.fail('BAM_FRAGMENTS_MATE_MISMATCH')
        if other['flag']&4 == 0 or current['mrid'] >= 0:
            if (current['mrid'],current['mpos'])!=(other['rid'],other['start']): m.fail('BAM_FRAGMENTS_MATE_MISMATCH')
    unmapped=bool((a['flag']|b['flag'])&12)
    if not unmapped and a['rid']==b['rid'] and (a['tlen']!=0 or b['tlen']!=0):
        length=max(ae,be)-min(a['start'],b['start'])
        if a['tlen']+b['tlen']!=0 or abs(b['tlen'])!=length: m.fail('BAM_FRAGMENTS_TLEN_MISMATCH')
    orientations=(bool(a['flag']&16),bool(b['flag']&16))
    forward,reverse=(y,x) if orientations[0] else (x,y)
    f,_,fe,_=forward; r,_,re,_=reverse
    geometry=orientations[0]==orientations[1] or f['start']>r['start'] or fe>re
    start,stop=f['start']+4,re-5
    reasons=(unmapped,a['rid']!=b['rid'],not(a['flag']&2 and b['flag']&2),
        bool((a['flag']|b['flag'])&512),has_supplement or 'SA' in ta or 'SA' in tb,
        ua or ub,geometry,any(z['mapq'] not in range(30,255) for z in (a,b)),
        'CB' not in ta or 'CB' not in tb,
        a['rid']<0 or not 0<=start<stop<=references[a['rid']][1])
    for reason,excluded in zip(m.EXCLUSIONS,reasons):
        if excluded: return None,reason
    return (references[a['rid']][0],start,stop,ta['CB']),None


def reconstruct(bound, directory, runtime):
    raw=directory/'verify.reads'; ordered=directory/'verify.names'; fragments=directory/'verify.fragments'
    header,refs,stream,n=io.project(bound['source']['path'],raw)
    groups=_reference(header,refs,bound); io.sort_templates(raw,ordered,directory,runtime)
    counts=dict(header_sha256=digest(header),sq_sha256=digest(refs),stream_sha256=stream,n_records=n,
        n_templates=0,n_primary_pairs=0,n_secondary=0,n_supplementary=0,eligible_pairs=0,
        exclusions={k:0 for k in m.EXCLUSIONS})
    ranks={name:i for i,(name,_) in enumerate(bound['contigs'])}
    with fragments.open('xb') as out:
        for _, records in groupby(io.records(ordered),lambda x:x[0]):
            pair={}; supplement=False
            for _,record in records:
                validated=_read(record,refs,groups); flag=record['flag']
                if flag&256: counts['n_secondary']+=1
                if flag&2048: counts['n_supplementary']+=1; supplement=True
                if flag&256 or flag&2048: continue
                label=flag&192
                if not flag&1 or label not in (64,128) or label in pair: m.fail('BAM_FRAGMENTS_TEMPLATE_INVALID')
                pair[label]=validated
            fragment,reason=_template(pair,supplement,refs)
            counts['n_templates']+=1; counts['n_primary_pairs']+=1
            if reason: counts['exclusions'][reason]+=1
            else:
                name,left,right,barcode=fragment; counts['eligible_pairs']+=1
                out.write(f'{ranks[name]}\t{left}\t{right}\t{barcode}\t\t1\n'.encode())
    sorted_fragments=directory/'verify.sorted'; physical.sort_ranked(fragments,sorted_fragments,directory,runtime)
    expected=directory/'verify.expected'; identifiers=directory/'verify.barcodes'
    h=hashlib.sha256(); total=maximum=count=0
    with sorted_fragments.open('rb') as source,expected.open('xb') as output,identifiers.open('xb') as bc:
        previous=None; support=0
        def emit(key,support):
            rank,start,stop,barcode=key
            data=f'{bound["contigs"][rank][0]}\t{start}\t{stop}\t{barcode}\t{support}\n'.encode()
            output.write(data); h.update(data); bc.write(barcode.encode()+b'\n')
        for line in source:
            fields=line.decode().split('\t'); key=(int(fields[0]),int(fields[1]),int(fields[2]),fields[3])
            if previous is not None and key!=previous:
                emit(previous,support); count+=1; total+=support; maximum=max(maximum,support); support=0
            previous=key; support+=1
            if support>2**64-1: m.fail('BAM_FRAGMENTS_SUPPORT_OVERFLOW')
        if previous is not None:
            emit(previous,support); count+=1; total+=support; maximum=max(maximum,support)
    if total<1 or total>2**128-1: m.fail('BAM_FRAGMENTS_CONSERVATION_MISMATCH')
    summary=dict(canonical_record_stream_sha256=h.hexdigest(),n_fragment_records=count,sum_support=total,
        n_distinct_barcodes=io.distinct_count(identifiers,directory,runtime),max_support=maximum)
    return expected,counts,summary


def verify_bam_fragments(manifest_path, *, expected_sha256, runtime, temporary_root=None):
    # Qualify executable bytes before any reconstruction subprocess uses them.
    physical.verify_packaging(runtime)
    try:
        initial=io.take_snapshots([manifest_path]); manifest=v2.load_fragments_manifest_v2(manifest_path,expected_sha256=expected_sha256)
        if len(manifest['libraries'])!=1: m.fail('BAM_FRAGMENTS_RECORD_MISMATCH')
        root=Path(manifest_path).parent; entry=manifest['libraries'][0]; p=entry['provenance']
        if (p['kind']!='bam_fragment_production' or p['profile']['id']!=m.PROFILE_ID
                or p['profile']['resource']['path']!=str(root/'profile.json')
                or p['producer_record'] is None or p['producer_record']['path']!=str(root/'production.json')): m.fail('BAM_FRAGMENTS_PROFILE_UNSUPPORTED')
        snapshots=io.take_snapshots([root/'profile.json',root/'production.json'])
        if io.resource(root/'profile.json')!=p['profile']['resource'] or p['profile']['resource']['sha256']!=m.sha_bytes(m.PROFILE_BYTES): m.fail('BAM_FRAGMENTS_PROFILE_UNSUPPORTED')
        record=m.load_record(root/'production.json',p['producer_record']['sha256'])
        # Binding freshly reinspects M10 BAM observations, so qualify the decoder
        # before binding as well as before the complete reconstruction pass.
        if record['runtime']!=io.runtime_identity(): m.fail('BAM_FRAGMENTS_RUNTIME_MISMATCH')
        bound=io.bind(record['arguments'])
        if any(record[k]!=bound[k] for k in ('source','intake','context','reference','namespace','group_id','context_identity_sha256')): m.fail('BAM_FRAGMENTS_RECORD_MISMATCH')
        if (record['reference']!=manifest['reference'] or entry['namespace']!=record['namespace']
                or entry['strand']!={'mode':'absent','definition':None}
                or p!=m.provenance(record,p['profile']['resource'],p['producer_record'])): m.fail('BAM_FRAGMENTS_RECORD_MISMATCH')
        with tempfile.TemporaryDirectory(prefix='agent-bam-verify-',dir=temporary_root) as temp:
            expected,counts,summary=reconstruct(bound,Path(temp),runtime)
            if counts!=record['qualification'] or summary!=record['canonical'] or any(entry[k]!=summary[k] for k in v2.SUMMARY_KEYS): m.fail('BAM_FRAGMENTS_CONSERVATION_MISMATCH')
            with expected.open('rb') as source:
                if any(a!=b for a,b in zip_longest(source,_bgzf_lines(root/entry['bgzf']['path']))): m.fail('BAM_FRAGMENTS_TRANSFORMATION_MISMATCH')
        for key in ('bgzf','tabix'):
            if record['outputs'][key]!={f:entry[key][f] for f in ('sha256','size_bytes')}: m.fail('BAM_FRAGMENTS_RECORD_MISMATCH')
        # Producer-specific checks precede the generic artifact verification.
        generic=verify_fragments_v2(manifest_path,expected_sha256=expected_sha256,runtime=runtime)
        if io.resource(record['source']['path'])!=record['source']: m.fail('BAM_FRAGMENTS_SOURCE_CHANGED')
        generic.check_unchanged(); io.check_snapshots(bound['snapshots']); io.check_snapshots(snapshots); io.check_snapshots(initial)
        return BamProductionVerification(generic)
    except m.BamFragmentsError: raise
    except (ValueError,OSError,KeyError,TypeError,IndexError,UnicodeError): m.fail('BAM_FRAGMENTS_VERIFICATION_FAILED')
