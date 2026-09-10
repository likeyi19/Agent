import gzip
import hashlib
import json
from fractions import Fraction
from pathlib import Path
import pytest
from agent.tools.data import scatac_barcode_qc as public,_barcode_qc_contract as m,_barcode_qc_binding as binding
from agent.tools.data.barcode_qc_verifier import verify_barcode_qc


@pytest.fixture
def published(qc_case):
    return public.compute_scATAC_qc(**qc_case[0])


@pytest.mark.parametrize('mutation',['window_contig','window_start','window_end','window_label',
    'nonoverlap','endpoint_format','endpoint_id','extra_column','missing_newline'])
def test_malformed_backend_incidence_never_publishes(qc_case,monkeypatch,mutation):
    from agent.tools.data import _barcode_qc_production as production
    args,*_=qc_case;original=production._intersection
    def malformed(bound,scratch,a,b,genome):
        path=original(bound,scratch,a,b,genome)
        lines=path.read_text().splitlines(keepends=True)
        fields=lines[0].rstrip('\n').split('\t')
        if mutation=='window_contig':fields[4]='NOT_A_REFERENCE_CONTIG'
        elif mutation=='window_start':fields[5]=str(int(fields[5])+1)
        elif mutation=='window_end':fields[6]=str(int(fields[6])+1)
        elif mutation=='window_label':fields[7]=fields[7][:2]+'999999999'
        elif mutation=='nonoverlap':
            for line in b.read_text().splitlines():
                window=line.split('\t')
                if window[3][0]==fields[7][0] and not (window[0]==fields[0] and int(window[1])<=int(fields[1])<int(window[2])):
                    fields[4:]=window;break
            else:pytest.fail('Fixture requires a valid nonoverlapping window')
        elif mutation=='endpoint_format':fields[1]='0'+fields[1]
        elif mutation=='endpoint_id':fields[3]='not-an-integer'
        elif mutation=='extra_column':fields.append('unexpected')
        lines[0]='\t'.join(fields)+'\n'
        if mutation=='missing_newline':lines[-1]=lines[-1].removesuffix('\n')
        path.write_text(''.join(lines));return path
    monkeypatch.setattr(production,'_intersection',malformed)
    with pytest.raises(ValueError,match='QC_INTERSECTION_INVALID'):public.compute_scATAC_qc(**args)
    assert not list(Path(args['output_dir']).glob('barcode-qc-*'))
    assert not list(Path(args['output_dir']).glob('.qc-attempt-*'))


@pytest.mark.parametrize('field,value',[('depth_mean',None),('tss_min',{'numerator':100,'denominator':1})])
def test_invalid_summary_shape_rejected_without_source_io(published,field,value):
    manifest=json.loads(Path(published['manifest_path']).read_bytes())
    manifest['summary'][field]=value;manifest['identity_sha256']=m.manifest_identity(manifest)
    with pytest.raises(ValueError,match='QC_SUMMARY_INVALID'):m.validate_manifest(manifest)


@pytest.mark.parametrize('mutation',['total','qc','center','flank','enrichment','validity','bins','nucleosome_ratio',
    'drop','add','order','namespace','barcode','summary','histogram'])
def test_self_consistent_forgeries(published,mutation):
    path=Path(published['manifest_path']);value=json.loads(path.read_bytes());root=path.parent
    lines=gzip.decompress((root/'barcodes.tsv.gz').read_bytes()).decode().splitlines()
    rows=[line.split('\t') for line in lines[1:]]
    if mutation in ('total','qc','center','flank','enrichment','bins','nucleosome_ratio'):
        col={'total':2,'qc':3,'center':4,'flank':5,'enrichment':10,'bins':7,'nucleosome_ratio':13}[mutation]
        rows[0][col]=str(int(rows[0][col])+1)
        if mutation=='qc':
            rows[0][2]=str(int(rows[0][2])+1);rows[0][7]=str(int(rows[0][7])+1)
        if mutation=='bins':
            rows[0][7]=str(int(rows[0][7])-2);rows[0][8]=str(int(rows[0][8])+1)
        if mutation not in ('enrichment','nucleosome_ratio'):
            rows[0]=m.row_bytes(*rows[0][:2],tuple(map(int,rows[0][2:10]))).decode().rstrip('\n').split('\t')
    elif mutation=='validity':rows[0][10:13]=['NA','NA','ZERO_TSS_BACKGROUND']
    elif mutation=='drop':rows.pop()
    elif mutation=='add':rows.append(['library_0','zz',*rows[0][2:]])
    elif mutation=='order':rows.reverse()
    elif mutation=='namespace':rows[0][0]='another'
    elif mutation=='barcode':rows[0][1]='Another-1'
    elif mutation=='histogram':
        h=gzip.decompress((root/'lengths.tsv.gz').read_bytes()).replace(b'1\t1\n',b'1\t2\n',1)
        (root/'lengths.tsv.gz').write_bytes(gzip.compress(h,mtime=0));value['histogram']=m.resource(root/'lengths.tsv.gz')
    raw=m.HEADER+''.join('\t'.join(r)+'\n' for r in rows).encode()
    (root/'barcodes.tsv.gz').write_bytes(gzip.compress(raw,mtime=0));value['table']=m.resource(root/'barcodes.tsv.gz')
    # Rebuild the claimed universe and summaries from the forged table as well
    # as all hashes. Detection cannot rely only on stale declared aggregates.
    value['row_count']=len(rows)
    counts=[list(map(int,r[2:10])) for r in rows]
    summary=dict(zip(m.COUNTS,map(sum,zip(*counts))))
    summary.update(depth_min=min(c[0] for c in counts),depth_max=max(c[0] for c in counts),
        depth_mean=m.fraction_json(Fraction(summary['n_fragment_records'],len(rows))),
        histogram_records=summary['n_qc_fragment_records'],histogram_overflow=0)
    for prefix,column in (('tss',10),('nucleosome',13)):
        values=[Fraction(int(r[column]),int(r[column+1])) for r in rows if r[column]!='NA']
        summary.update({prefix+'_defined':len(values),prefix+'_undefined':len(rows)-len(values),
            prefix+'_min':m.fraction_json(min(values)) if values else None,
            prefix+'_max':m.fraction_json(max(values)) if values else None})
    if mutation=='summary':summary['depth_max']+=1
    value['summary']=summary
    order=hashlib.sha256(m.ORDER_DOMAIN)
    for r in rows:order.update(m.identity(r[0],r[1]))
    value['ordered_barcode_sha256']=order.hexdigest()
    value['identity_sha256']=m.manifest_identity(value);path.write_bytes(m.canonical(m.validate_manifest(value)))
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    m.load_manifest(path,sha)
    assert m.resource(root/'barcodes.tsv.gz')==value['table']
    assert m.resource(root/'lengths.tsv.gz')==value['histogram']
    with pytest.raises(ValueError):verify_barcode_qc(path,expected_sha256=sha)


@pytest.mark.parametrize('which',['fragments','reference','annotation','table','backend'])
def test_resource_mutation(qc_case,published,which,monkeypatch):
    args,_,_,qc,_=qc_case;path=Path(published['manifest_path'])
    if which=='backend':monkeypatch.setenv('AGENT_QC_BEDTOOLS','/bin/false')
    else:
        target={'fragments':args['fragments_manifest_path'],'reference':args['qc_reference_manifest_path'],
            'annotation':qc.annotation.resource.path,'table':str(path.parent/'barcodes.tsv.gz')}[which]
        with open(target,'ab') as f:f.write(b'changed')
    with pytest.raises((ValueError,OSError)):public.verify_public_result(args,published)


def test_reference_mismatch(fixture_factory):
    a,*_=fixture_factory();b,*_=fixture_factory()
    a.update(qc_reference_manifest_path=b['qc_reference_manifest_path'],qc_reference_manifest_sha256=b['qc_reference_manifest_sha256'])
    with pytest.raises(ValueError,match='REFERENCE_MISMATCH'):public.compute_scATAC_qc(**a)


def test_fabricated_producer_is_not_generically_qualified(qc_case):
    args,*_=qc_case
    from agent.tools.data import scatac_fragments_v2 as v2
    from agent.tools.data.scatac_fragment_reader import open_verified_fragments
    path=Path(args['fragments_manifest_path']);value=json.loads(path.read_bytes())
    value['libraries'][0]['provenance']['profile']['id']='fabricated-profile.v1'
    value['fragments_identity_sha256']=v2.fragments_identity(value)
    path.write_bytes(v2.canonical_fragments_manifest_v2_bytes(value));args['fragments_manifest_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    # Generic verification explicitly accepts opaque bound profile identity.
    view=open_verified_fragments(path,expected_sha256=args['fragments_manifest_sha256'],runtime=binding.packaging_runtime())
    assert view.verification.producer_profile_qualification=='not_established'
    with pytest.raises(ValueError):public.compute_scATAC_qc(**args)
    assert not Path(args['output_dir']).exists()
