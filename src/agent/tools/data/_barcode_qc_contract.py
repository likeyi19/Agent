"""Closed, IO-free barcode QC contracts and exact canonical row syntax."""
from dataclasses import dataclass
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path

from . import scatac_fragments_v2 as v2
from .scatac_qc_profile import canonical, fail, unsigned, PROFILE_SHA256, PROFILE

ARTIFACT = 'agent.scatac-barcode-qc'
CONTRACT = 'scatac-barcode-qc.v1'
POLICY = 'compute-scatac-qc-v1'
MAX_MANIFEST = 1024*1024
MAX_ROW = 4096
MAX_BARCODES = 10_000_000
MAX_RECORDS = 1_000_000_000
MAX_SIDECAR = 16*1024**3
ORDER_DOMAIN = b'agent.ordered-observed-barcodes.v1\0'
COUNTS = ('n_fragment_records','n_qc_fragment_records','tss_center_count',
    'tss_left_flank_count','tss_right_flank_count','n_nucleosome_free',
    'n_mononucleosomal','n_longer')
COLUMNS = ('namespace','barcode_identifier',*COUNTS,'tss_numerator','tss_denominator',
    'tss_validity_reason','nucleosome_numerator','nucleosome_denominator','nucleosome_validity_reason')
HEADER = ('\t'.join(COLUMNS)+'\n').encode()
ARGUMENTS = ('fragments_manifest_path','fragments_manifest_sha256',
    'qc_reference_manifest_path','qc_reference_manifest_sha256','output_dir')
MANIFEST_FIELDS = ('artifact_type','schema_version','contract_version','policy','arguments',
    'reference_identity_sha256','qc_resource_identity_sha256','science_profile_sha256',
    'tss_method','producer_authority','resource_qualification','backend_identity',
    'ordered_barcode_sha256','row_count','table','histogram','summary','identity_sha256')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def validate_arguments(value):
    args = dict(value)
    v2.shape(args, ARGUMENTS)
    for key in ARGUMENTS:
        if key.endswith('_sha256'): v2.sha(args[key])
        else:
            if isinstance(args[key],Path): args[key]=str(args[key])
            v2.absolute_path(args[key])
    return args


def ratio_fields(numerator, denominator, reason):
    if not denominator: return ('NA','NA',reason)
    f=Fraction(numerator,denominator)
    return str(f.numerator),str(f.denominator),'DEFINED'


def row_bytes(namespace,barcode,counts):
    identity(namespace,barcode)
    if len(counts)!=8: fail('QC_ROW_INVALID')
    values=tuple(unsigned(x) for x in counts)
    total,qc,c,l,r,free,mono,longer=values
    if total<1 or qc>total or free+mono+longer!=qc: fail('QC_ROW_INVALID')
    fields=(namespace,barcode,*map(str,values),
        *ratio_fields(200*c,101*(l+r),'ZERO_TSS_BACKGROUND'),
        *ratio_fields(mono,free,'ZERO_NUCLEOSOME_FREE'))
    raw=('\t'.join(fields)+'\n').encode('ascii')
    if len(raw)>MAX_ROW: fail('QC_ROW_LIMIT')
    return raw


def identity(namespace,barcode):
    v2.token(namespace)
    if type(barcode) is not str or not 1<=len(barcode)<=256 or any(not 33<=ord(c)<=126 for c in barcode): fail('QC_IDENTITY_INVALID')
    return (namespace+'\t'+barcode+'\n').encode('ascii')


def resource(path):
    p=Path(path); size=p.stat().st_size
    if not p.is_file() or not 0<size<=MAX_SIDECAR: fail('QC_RESOURCE_LIMIT')
    h=hashlib.sha256()
    from agent.tools._cancellation import cancellation_checkpoint
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block);cancellation_checkpoint()
    return dict(path=p.name,sha256=h.hexdigest(),size_bytes=size)


def manifest_identity(value):
    return hashlib.sha256(b'agent.barcode-qc-identity.v1\0'+canonical(
        {k:v for k,v in value.items() if k!='identity_sha256'})).hexdigest()


def _fraction(value):
    if value is None: return
    v2.shape(value,('numerator','denominator'))
    n,d=value['numerator'],value['denominator']
    # Numerators can be count*200; no floating point is authoritative.
    if type(n) is not int or type(d) is not int or n<0 or d<1 or max(n,d)>2**160: fail('QC_SUMMARY_INVALID')
    f=Fraction(n,d)
    if (f.numerator,f.denominator)!=(n,d): fail('QC_SUMMARY_INVALID')


def validate_summary(s,n):
    v2.shape(s,(*COUNTS,'depth_min','depth_max','depth_mean','tss_defined','tss_undefined',
        'tss_min','tss_max','nucleosome_defined','nucleosome_undefined','nucleosome_min','nucleosome_max',
        'histogram_records','histogram_overflow'))
    for k,v in s.items():
        if k in ('depth_mean','tss_min','tss_max','nucleosome_min','nucleosome_max'): _fraction(v)
        else: unsigned(v)
    if (not n<=s['n_fragment_records']<=MAX_RECORDS or s['n_qc_fragment_records']>s['n_fragment_records']
        or sum(s[k] for k in COUNTS[5:])!=s['n_qc_fragment_records']
        or not 1<=s['depth_min']<=s['depth_max']<=s['n_fragment_records']
        or s['histogram_records']!=s['n_qc_fragment_records']
        or s['histogram_overflow']>s['histogram_records']): fail('QC_SUMMARY_INVALID')
    if s['depth_mean'] is None or s['depth_mean']!=fraction_json(Fraction(s['n_fragment_records'],n)):
        fail('QC_SUMMARY_INVALID')
    for prefix in ('tss','nucleosome'):
        if s[prefix+'_defined']+s[prefix+'_undefined']!=n: fail('QC_SUMMARY_INVALID')
        for suffix in ('min','max'):
            if (s[prefix+'_'+suffix] is None)!=(s[prefix+'_defined']==0): fail('QC_SUMMARY_INVALID')
        if s[prefix+'_defined']:
            lo,hi=s[prefix+'_min'],s[prefix+'_max']
            if lo['numerator']*hi['denominator']>hi['numerator']*lo['denominator']: fail('QC_SUMMARY_INVALID')


def validate_manifest(value):
    try:
        v2.shape(value,MANIFEST_FIELDS)
        if len(canonical(value))>MAX_MANIFEST: fail('QC_MANIFEST_LIMIT')
        if (value['artifact_type']!=ARTIFACT or type(value['schema_version']) is not int
            or value['schema_version']!=1 or value['contract_version']!=CONTRACT or value['policy']!=POLICY
            or value['science_profile_sha256']!=PROFILE_SHA256 or value['tss_method']!=PROFILE.tss_method): fail('QC_MANIFEST_INVALID')
        validate_arguments(value['arguments'])
        for k in ('reference_identity_sha256','qc_resource_identity_sha256','ordered_barcode_sha256','identity_sha256'): v2.sha(value[k])
        n=value['row_count']
        if type(n) is not int or not 1<=n<=MAX_BARCODES: fail('QC_RESOURCE_LIMIT')
        for k,name in (('table','barcodes.tsv.gz'),('histogram','lengths.tsv.gz')):
            r=value[k];v2.shape(r,('path','sha256','size_bytes'));v2.sha(r['sha256'])
            if r['path']!=name or type(r['size_bytes']) is not int or not 0<r['size_bytes']<=MAX_SIDECAR: fail('QC_RESOURCE_INVALID')
        v2.shape(value['producer_authority'],('kind','profile_ids','verification_basis','history'))
        a=value['producer_authority']
        if a['kind'] not in ('fastq_fragment_production','bam_fragment_production','external_fragment_adoption'): fail('QC_PRODUCER_UNQUALIFIED')
        if type(a['profile_ids']) is not list or not 1<=len(a['profile_ids'])<=3: fail('QC_PRODUCER_UNQUALIFIED')
        for p in a['profile_ids']: v2.token(p)
        for k in ('verification_basis','history'): v2.token(a[k])
        v2.shape(value['resource_qualification'],('mode','catalog_sha256','basis'))
        qualification=value['resource_qualification'];v2.token(qualification['basis'])
        if qualification['mode']=='synthetic_only':
            if qualification['catalog_sha256'] is not None: fail('QC_RESOURCE_UNQUALIFIED')
        elif qualification['mode']=='operator_qualified': v2.sha(qualification['catalog_sha256'])
        else: fail('QC_RESOURCE_UNQUALIFIED')
        v2.shape(value['backend_identity'],('profile','executable_sha256','runtime_sha256','compression'))
        backend=value['backend_identity'];v2.sha(backend['executable_sha256']);v2.sha(backend['runtime_sha256'])
        if backend['profile']!='bedtools-qc-point-incidence.v1' or backend['compression']!='gzip-mtime0-level6.v1': fail('QC_BACKEND_INVALID')
        validate_summary(value['summary'],n)
        if value['identity_sha256']!=manifest_identity(value): fail('QC_IDENTITY_MISMATCH')
        return value
    except (KeyError,TypeError,OverflowError): fail('QC_MANIFEST_INVALID')


@dataclass(frozen=True)
class BarcodeQCManifest:
    canonical_bytes: bytes

    def __post_init__(self):
        if type(self.canonical_bytes) is not bytes or len(self.canonical_bytes)>MAX_MANIFEST: fail('QC_MANIFEST_INVALID')
        d=json.loads(self.canonical_bytes,object_pairs_hook=v2._pairs,parse_constant=lambda _:fail('QC_MANIFEST_INVALID'))
        if canonical(validate_manifest(d))!=self.canonical_bytes: fail('QC_MANIFEST_INVALID')

    def to_dict(self): return json.loads(self.canonical_bytes)


def load_manifest(path,expected_sha256):
    v2.absolute_path(str(path));v2.sha(expected_sha256)
    with open(path,'rb') as f: raw=f.read(MAX_MANIFEST+1)
    if len(raw)>MAX_MANIFEST or hashlib.sha256(raw).hexdigest()!=expected_sha256: fail('QC_MANIFEST_MISMATCH')
    return BarcodeQCManifest(raw)


def write_gzip(path,lines):
    with open(path,'xb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0,compresslevel=6) as f:
            for line in lines: f.write(line)


def gzip_lines(path,max_rows):
    with gzip.open(path,'rb') as f:
        for i in range(max_rows+1):
            line=f.readline(MAX_ROW+1)
            if not line: return
            if i==max_rows or len(line)>MAX_ROW or not line.endswith(b'\n'): fail('QC_ROW_INVALID')
            yield line


def fraction_json(value):
    return None if value is None else dict(numerator=value.numerator,denominator=value.denominator)
