"""Production implementation of the frozen corrected-CB paired ATAC policy."""
import hashlib
from itertools import groupby
from pathlib import Path
import re

from . import bam_fragment_manifest as m, _bam_fragment_io as io
from . import _external_fragment_io as physical, scatac_fragments_v2 as v2
from ._fragments_common import canonical, digest, MAX_SUPPORT, MAX_TOTAL, PACKAGING_POLICY, run_stage


def header_check(header, dictionary, binding):
    sq = header.get('SQ', [])
    if not sq or len(sq) > 4096 or [(s.get('SN'),s.get('LN')) for s in sq] != dictionary:
        m.fail('BAM_FRAGMENTS_HEADER_INVALID')
    contigs = dict(binding['contigs'])
    if len({n for n,_ in dictionary}) != len(dictionary): m.fail('BAM_FRAGMENTS_HEADER_INVALID')
    for s in sq:
        if s['SN'] not in contigs or type(s['LN']) is not int or s['LN'] != contigs[s['SN']]: m.fail('BAM_FRAGMENTS_REFERENCE_MISMATCH')
    md5 = io.sequence_md5(binding['bundle'], [s['SN'] for s in sq if 'M5' in s])
    if any(s['M5'].lower() != md5[s['SN']] for s in sq if 'M5' in s): m.fail('BAM_FRAGMENTS_REFERENCE_MISMATCH')
    rgs = header.get('RG', [])
    if len(rgs) > 256 or len({r.get('ID') for r in rgs}) != len(rgs): m.fail('BAM_FRAGMENTS_HEADER_INVALID')
    lbs = {r.get('LB') for r in rgs if r.get('LB')}
    if len(lbs) > 1 or (lbs and binding['library'].source_library_id not in lbs): m.fail('BAM_FRAGMENTS_LIBRARY_MISMATCH')
    return {r['ID'] for r in rgs}


def alignment(r, dictionary, rgs):
    """Validate all projected records, even those that will not contribute."""
    tags = {}
    for name, value, kind in r['tags']:
        lower = 32 if name == 'RG' else 33
        if name in tags or kind != 'Z' or not value or len(value) > 65536 or any(ord(c)<lower or ord(c)>126 for c in value):
            m.fail('BAM_FRAGMENTS_TAG_INVALID')
        tags[name] = value
    if 'CB' in tags and re.fullmatch('[!-~]{1,256}', tags['CB']) is None: m.fail('BAM_FRAGMENTS_TAG_INVALID')
    if 'RG' in tags and tags['RG'] not in rgs: m.fail('BAM_FRAGMENTS_LIBRARY_MISMATCH')
    if 'SA' in tags:
        entries = tags['SA'].split(';')
        if entries[-1] != '' or any(re.fullmatch(r'[^,;]+,[1-9][0-9]*,[+-],(?:[1-9][0-9]*[MIDNSHP=X])+,[0-9]+,[0-9]+', x) is None for x in entries[:-1]):
            m.fail('BAM_FRAGMENTS_TAG_INVALID')
    f = r['flag']
    if not 0 <= f <= 4095: m.fail('BAM_FRAGMENTS_FLAGS_INVALID')
    for k in ('rid','mrid'):
        if not -1 <= r[k] < len(dictionary): m.fail('BAM_FRAGMENTS_REFERENCE_MISMATCH')
    for k, refkey in (('start','rid'),('mpos','mrid')):
        if r[k] < -1 or (r[refkey] == -1 and r[k] != -1) or (r[refkey] >= 0 and not 0 <= r[k] < dictionary[r[refkey]][1]):
            m.fail('BAM_FRAGMENTS_COORDINATE_INVALID')
    cigar = r['cigar'] or []
    if any(op not in range(9) or n <= 0 for op,n in cigar): m.fail('BAM_FRAGMENTS_CIGAR_INVALID')
    ops = [op for op,n in cigar]
    # H only at ends; S only at ends inside an optional H.
    for i, op in enumerate(ops):
        if op == 5 and i not in (0,len(ops)-1): m.fail('BAM_FRAGMENTS_CIGAR_INVALID')
        if op == 4 and not (all(x == 5 for x in ops[:i]) or all(x == 5 for x in ops[i+1:])): m.fail('BAM_FRAGMENTS_CIGAR_INVALID')
    query = sum(n for op,n in cigar if op in (0,1,4,7,8))
    span = sum(n for op,n in cigar if op in (0,2,3,7,8))
    if cigar and r['seq_len'] is not None and r['seq_len'] != query: m.fail('BAM_FRAGMENTS_CIGAR_INVALID')
    end = r['start'] + span
    if not f & 4 and (r['rid'] < 0 or not cigar or span <= 0 or r['end'] != end or end > dictionary[r['rid']][1]):
        m.fail('BAM_FRAGMENTS_COORDINATE_INVALID')
    unsupported = bool(cigar) and (3 in ops or 6 in ops or ops[-1 if f & 16 else 0] not in (0,7,8))
    return r | {'tag_values':tags, 'computed_end':end, 'unsupported':unsupported}


def pair(first, second, split, dictionary):
    a,b = first,second
    if a['tag_values'].get('RG') != b['tag_values'].get('RG'): m.fail('BAM_FRAGMENTS_LIBRARY_MISMATCH')
    cb1,cb2 = a['tag_values'].get('CB'),b['tag_values'].get('CB')
    if cb1 is not None and cb2 is not None and cb1 != cb2: m.fail('BAM_FRAGMENTS_BARCODE_MISMATCH')
    for r, mate in ((a,b),(b,a)):
        if bool(r['flag']&8) != bool(mate['flag']&4): m.fail('BAM_FRAGMENTS_MATE_MISMATCH')
        if not mate['flag']&4 and bool(r['flag']&32) != bool(mate['flag']&16): m.fail('BAM_FRAGMENTS_MATE_MISMATCH')
        if (not mate['flag']&4 or r['mrid'] != -1) and (r['mrid'],r['mpos']) != (mate['rid'],mate['start']): m.fail('BAM_FRAGMENTS_MATE_MISMATCH')
    both_mapped = not (a['flag']&12 or b['flag']&12)
    same = a['rid'] == b['rid']
    if both_mapped and same:
        span = max(a['computed_end'],b['computed_end']) - min(a['start'],b['start'])
        if (a['tlen'] or b['tlen']) and (a['tlen'] != -b['tlen'] or abs(a['tlen']) != span): m.fail('BAM_FRAGMENTS_TLEN_MISMATCH')
    if not both_mapped: return None,'unmapped'
    if not same: return None,'discordant'
    if not a['flag']&2 or not b['flag']&2: return None,'improper'
    if a['flag']&512 or b['flag']&512: return None,'qc_failed'
    if split or 'SA' in a['tag_values'] or 'SA' in b['tag_values']: return None,'split'
    if a['unsupported'] or b['unsupported']: return None,'unsupported_cigar'
    if bool(a['flag']&16) == bool(b['flag']&16): return None,'geometry'
    forward,reverse = (b,a) if a['flag']&16 else (a,b)
    if forward['start'] > reverse['start'] or forward['computed_end'] > reverse['computed_end']: return None,'geometry'
    if any(r['mapq'] < 30 or r['mapq'] == 255 for r in (a,b)): return None,'mapq'
    if cb1 is None or cb2 is None: return None,'missing_cb'
    left,right = forward['start']+4,reverse['computed_end']-5
    if not 0 <= left < right <= dictionary[a['rid']][1]: return None,'short_shifted'
    return (dictionary[a['rid']][0], left, right, cb1),None


def generate(binding, directory, runtime):
    projected = directory/'reads.tmp'; ordered = directory/'names.tmp'; ranked = directory/'ranked.tmp'
    header, dictionary, stream, n = io.project(binding['source']['path'], projected)
    rgs = header_check(header,dictionary,binding)
    io.sort_templates(projected,ordered,directory,runtime)
    q = dict(header_sha256=digest(header),sq_sha256=digest(dictionary),stream_sha256=stream,
        n_records=n,n_templates=0,n_primary_pairs=0,n_secondary=0,n_supplementary=0,
        eligible_pairs=0,exclusions=dict.fromkeys(m.EXCLUSIONS,0))
    ranks = {name:i for i,(name,_) in enumerate(binding['contigs'])}
    with ranked.open('xb') as target:
        for _, group in groupby(io.records(ordered), key=lambda x:x[0]):
            primary = {}; split = False
            for _, raw in group:
                r = alignment(raw,dictionary,rgs); f = r['flag']
                q['n_secondary'] += bool(f&256); q['n_supplementary'] += bool(f&2048)
                split |= bool(f&2048)
                if f&2304: continue
                if not f&1 or bool(f&64) == bool(f&128): m.fail('BAM_FRAGMENTS_TEMPLATE_INVALID')
                end = 1 if f&64 else 2
                if end in primary: m.fail('BAM_FRAGMENTS_TEMPLATE_INVALID')
                primary[end] = r
            if set(primary) != {1,2}: m.fail('BAM_FRAGMENTS_TEMPLATE_INVALID')
            q['n_templates'] += 1; q['n_primary_pairs'] += 1
            fragment,reason = pair(primary[1],primary[2],split,dictionary)
            if reason: q['exclusions'][reason] += 1
            else:
                name,start,end,barcode = fragment; q['eligible_pairs'] += 1
                target.write(f'{ranks[name]}\t{start}\t{end}\t{barcode}\t\t1\n'.encode())
    if not q['eligible_pairs']: m.fail('BAM_FRAGMENTS_EMPTY')
    return ranked,q


def checked_support(value, increment):
    result = value + increment
    if type(value) is not int or type(increment) is not int or value < 0 or increment < 1 or result > MAX_SUPPORT:
        m.fail('BAM_FRAGMENTS_SUPPORT_OVERFLOW')
    return result


def aggregate(ranked, contigs, directory, runtime):
    ordered = directory/'fragments.sorted'; plain = directory/'canonical.tmp'; barcodes = directory/'barcodes.tmp'
    physical.sort_ranked(ranked,ordered,directory,runtime)
    count=total=maximum=0; h=hashlib.sha256()
    with ordered.open('rb') as source, plain.open('xb') as output, barcodes.open('xb') as identifiers:
        rows = (line.decode().rstrip('\n').split('\t') for line in source)
        for key, group in groupby(rows, key=lambda f:tuple(f[:4])):
            support = 0
            for fields in group: support = checked_support(support,int(fields[5]))
            rank,left,right,barcode = key
            raw = f'{contigs[int(rank)][0]}\t{left}\t{right}\t{barcode}\t{support}\n'.encode()
            output.write(raw); h.update(raw); identifiers.write(barcode.encode()+b'\n')
            count += 1; total += support; maximum=max(maximum,support)
            if total > MAX_TOTAL or count > MAX_TOTAL: m.fail('BAM_FRAGMENTS_SUPPORT_OVERFLOW')
    return plain,dict(canonical_record_stream_sha256=h.hexdigest(),n_fragment_records=count,sum_support=total,
        n_distinct_barcodes=io.distinct_count(barcodes,directory,runtime),max_support=maximum)


def prepare_in_stage(arguments, stage, runtime):
    physical.verify_packaging(runtime); backend = io.runtime_identity(); binding = io.bind(arguments)
    directory = stage/'libraries'/binding['namespace']; directory.mkdir(parents=True)
    ranked,qualification = generate(binding,directory,runtime)
    plain,summary = aggregate(ranked,binding['contigs'],directory,runtime)
    bgzf=directory/'fragments.tsv.gz'
    with bgzf.open('xb') as out:
        run_stage([runtime.bgzip,*PACKAGING_POLICY['bgzip'],plain],cwd=directory,code='BAM_FRAGMENTS_PACKAGING_FAILED',output=out)
    run_stage([runtime.tabix,'-p','bed',bgzf],cwd=directory,code='BAM_FRAGMENTS_PACKAGING_FAILED')
    outputs={k:io.resource(p) for k,p in (('bgzf',bgzf),('tabix',Path(str(bgzf)+'.tbi')))}
    record=m.validate_record(dict(artifact_type='agent.bam-fragment-production',schema_version=1,
        contract_version=m.PRODUCTION_CONTRACT,profile_sha256=m.sha_bytes(m.PROFILE_BYTES),arguments=arguments,
        **{k:binding[k] for k in ('source','intake','context','reference','namespace','group_id','context_identity_sha256')},
        source_history=m.HISTORY,runtime=backend,qualification=qualification,canonical=summary,
        outputs={k:{f:v[f] for f in ('sha256','size_bytes')} for k,v in outputs.items()}))
    for p in directory.iterdir():
        if p not in (bgzf,Path(str(bgzf)+'.tbi')): p.unlink()
    profile=stage/'profile.json'; profile.write_bytes(m.PROFILE_BYTES)
    producer=stage/'production.json'; producer.write_bytes(canonical(record))
    entry=dict(namespace=binding['namespace'],**summary,strand={'mode':'absent','definition':None},
        provenance=m.provenance(record,io.resource(profile),io.resource(producer)))
    entry.update({k:v|{'path':str(Path(v['path']).relative_to(stage))} for k,v in outputs.items()})
    value=dict(artifact_type=v2.ARTIFACT_TYPE,schema_version=2,contract_version=v2.CONTRACT_VERSION,
        reference=binding['reference'],semantics=v2.SEMANTICS,libraries=[entry])
    value['fragments_identity_sha256']=v2.fragments_identity(value)
    (stage/'manifest.json').write_bytes(v2.canonical_fragments_manifest_v2_bytes(value))
    if io.resource(arguments['source_path']) != binding['source']: m.fail('BAM_FRAGMENTS_SOURCE_CHANGED')
    io.check_snapshots(binding['snapshots'])
    return value,binding['snapshots']
