"""Pure fixtures: no installed Chromap, sort, bgzip, tabix, or network needed."""
from dataclasses import asdict, replace
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from agent.tools.data import _chromap as c
from agent.tools.data import _fragments_common as common
from agent.tools.data import _fragments_binding as binding
from agent.tools.data import _fragments_canonical as production
from agent.tools.data import fastq_fragments as api
from agent.tools.data import scatac_fragments_verifier as verifier
from agent.tools.data import scatac_fragments_manifest as manifest
from agent.tools.data._fragments_fastq import scan_group
from fragments_helpers import BC, backend, inputs_for, bgzf_bytes


@pytest.fixture
def bound(tiny, reads):
    group, white = reads([(100,200,BC,300)])
    inputs = inputs_for(tiny, [group], white)
    return inputs, group, white


@pytest.fixture
def fake_runtime(monkeypatch, tiny):
    """Mock only external process boundaries; retain real parsers/bindings/verifier."""
    exe = tiny['root']/'fake-executable'; exe.write_text('fixture')
    runtime = common.FragmentsRuntime(*([str(exe)]*4))
    monkeypatch.setattr(c, 'identify_backend', lambda *a, **k: backend())
    monkeypatch.setattr(api, 'verify_packaging', lambda _: None)
    monkeypatch.setattr(verifier, 'verify_packaging', lambda _: None)
    control = {'rows': f'chrTiny\t104\t195\t{BC}\t300\n'.encode(), 'interrupt': None,
               'fail': None, 'after': None, 'calls': []}
    def stage(argv, *, cwd, code, output=None):
        if '--preset' in argv: kind = 'chromap'
        elif '--parallel=1' in argv: kind = 'sort'
        elif '-c' in argv: kind = 'bgzip'
        else: kind = 'tabix'
        control['calls'].append(kind)
        if control['interrupt'] == kind: raise KeyboardInterrupt()
        if control['fail'] == kind: common.fail(code)
        if kind == 'chromap':
            Path(argv[argv.index('--output')+1]).write_bytes(control['rows'])
        elif kind == 'sort':
            lines = Path(argv[-1]).read_bytes().splitlines(True)
            lines.sort(key=lambda line: (tuple(map(int,line.split(b'\t')[:3])), line.split(b'\t')[3]))
            Path(argv[argv.index('-o')+1]).write_bytes(b''.join(lines))
        elif kind == 'bgzip': output.write(bgzf_bytes(Path(argv[-1]).read_bytes()))
        else: Path(str(argv[-1])+'.tbi').write_bytes(b'fake TBI')
        if control['after']: control['after'](kind)
    monkeypatch.setattr(api, 'run_stage', stage)
    monkeypatch.setattr(production, 'run_stage', stage)
    def query(runtime, path, region):
        rows = gzip.decompress(Path(path).read_bytes()).splitlines(True)
        if ':' in region:
            chrom, interval = region.split(':'); left,right = map(int,interval.split('-')); left-=1
        else: chrom=region; left=0; right=2**29
        return hashlib.sha256(b''.join(row for row in rows if row.split(b'\t')[0].decode()==chrom
            and int(row.split(b'\t')[1])<right and int(row.split(b'\t')[2])>left)).hexdigest()
    monkeypatch.setattr(verifier, '_query', query)
    def listing(argv, **kwargs):
        assert argv[1]=='-l'
        return SimpleNamespace(returncode=0,stdout=b'chrTiny\n')
    monkeypatch.setattr(verifier.subprocess, 'run', listing)
    return runtime, control


@pytest.fixture
def published(tiny, bound, fake_runtime):
    inputs, _, _ = bound; runtime, control = fake_runtime
    result = api.prepare_fastq_fragments(inputs=inputs, runtime=runtime, output_dir=tiny['root']/'published')
    return result, runtime, control


def test_pure_complete_boundary(published):
    result, runtime, control = published
    value = verifier.verify_fragments(result['manifest_path'], runtime=runtime,
                                     expected_sha256=result['manifest_sha256'])
    assert result['total_support']==300 and value['libraries'][0]['max_support']==300
    assert control['calls']==['chromap','sort','bgzip','tabix']


@pytest.mark.parametrize('mode', ['plain','gzip','crlf','suffix'])
def test_full_scan_supported_forms(reads, mode):
    group, _ = reads([(100,200,BC,300)])
    for role, name in group.files:
        p=Path(name); payload=p.read_bytes()
        if mode=='crlf': payload=payload.replace(b'\n',b'\r\n')
        if mode=='suffix':
            lines=payload.splitlines(True)
            for i in range(0,len(lines),4): lines[i]=lines[i].rstrip()+ (b'/2\n' if role is c.ReadRole.R3 else b'/1\n')
            payload=b''.join(lines)
        p.write_bytes(gzip.compress(payload) if mode=='gzip' else payload)
    result=scan_group(group,16)
    assert result['record_count']==300 and len(result['decoded_role_sha256'])==3


@pytest.mark.parametrize('case,code', [
    ('name','FRAGMENTS_READ_NAME_MISMATCH'), ('order','FRAGMENTS_READ_NAME_MISMATCH'),
    ('count','FRAGMENTS_RECORD_COUNT_MISMATCH'), ('truncated','FRAGMENTS_FASTQ_MALFORMED'),
    ('plus','FRAGMENTS_FASTQ_MALFORMED'), ('header','FRAGMENTS_FASTQ_MALFORMED'),
    ('quality','FRAGMENTS_FASTQ_MALFORMED'), ('sequence','FRAGMENTS_FASTQ_MALFORMED'),
    ('length','FRAGMENTS_BARCODE_LENGTH_UNSUPPORTED'), ('gzip_crc','FRAGMENTS_FASTQ_MALFORMED'),
    ('gzip_truncated','FRAGMENTS_FASTQ_MALFORMED')])
def test_full_scan_failures_beyond_m10_prefix(reads, case, code):
    group,_=reads([(100,200,BC,300)])
    p=Path(dict(group.files)[c.ReadRole.R2]); lines=p.read_bytes().splitlines(True)
    i=280*4
    if case=='name': lines[i]=b'@wrong\n'
    elif case=='order': lines[i],lines[i+4]=lines[i+4],lines[i]
    elif case=='count': del lines[-4:]
    elif case=='truncated': del lines[-1:]
    elif case=='plus': lines[i+2]=b'+wrong\n'
    elif case=='header': lines[i]=b'bad\n'
    elif case=='quality': lines[i+3]=b' \n'
    elif case=='sequence': lines[i+1]=b'X'*16+b'\n'
    elif case=='length': lines[i+1]=b'A'*33+b'\n';lines[i+3]=b'I'*33+b'\n'
    payload=b''.join(lines)
    if case.startswith('gzip'):
        payload=gzip.compress(payload)
        payload=payload[:-4] if case=='gzip_truncated' else payload[:-8]+bytes([payload[-8]^1])+payload[-7:]
    p.write_bytes(payload)
    with pytest.raises(common.FragmentsError,match=code):scan_group(group,16)


@pytest.mark.parametrize('field,code', [('intake_sha256','FRAGMENTS_INTAKE_MISMATCH'),
    ('context_sha256','FRAGMENTS_CONTEXT_MISMATCH'),('reference_sha256','FRAGMENTS_REFERENCE_MISMATCH'),
    ('index_sha256','FRAGMENTS_REFERENCE_MISMATCH')])
def test_binding_digest_mismatch_before_scan(bound, field, code):
    inputs,_,_=bound
    with pytest.raises(common.FragmentsError,match=code):
        binding.preflight(replace(inputs,**{field:'0'*64}),backend())


@pytest.mark.parametrize('target,code', [('whitelist','FRAGMENTS_WHITELIST_MISMATCH'),
    ('index','CHROMAP_INDEX_INVALID'),('reference','FRAGMENTS_REFERENCE_MISMATCH'),
    ('raw','FRAGMENTS_SOURCE_CHANGED_BEFORE_EXECUTION')])
def test_fresh_resource_reinspection(tiny,bound,target,code):
    inputs,group,white=bound
    path={'whitelist':white.resource.path,'index':str(tiny['root']/'index/reference.chromap'),
          'reference':str(tiny['bed']),'raw':group.files[0][1]}[target]
    with open(path,'ab') as out:out.write(b'changed')
    with pytest.raises(common.FragmentsError,match=code):binding.preflight(inputs,backend())


@pytest.mark.parametrize('line,code', [
    (b'chrOther\t1\t2\t'+BC.encode()+b'\t1\n','FRAGMENTS_COORDINATES_INVALID'),
    (b'chrTiny\t-1\t2\t'+BC.encode()+b'\t1\n','FRAGMENTS_COORDINATES_INVALID'),
    (b'chrTiny\t2\t2\t'+BC.encode()+b'\t1\n','FRAGMENTS_COORDINATES_INVALID'),
    (b'chrTiny\t1\t12001\t'+BC.encode()+b'\t1\n','FRAGMENTS_COORDINATES_INVALID'),
    (b'chrTiny\t1\t2\tAAAA\t1\n','FRAGMENTS_OUTPUT_MALFORMED'),
    (b'chrTiny\t1\t2\t'+BC.encode()+b'\t0\n','FRAGMENTS_SUPPORT_INVALID'),
    (b'chrTiny\t1\t2\t'+BC.encode()+b'\t18446744073709551616\n','FRAGMENTS_SUPPORT_INVALID'),
    (b'chrTiny\t1\t2\t'+BC.encode()+b'\t1.0\n','FRAGMENTS_SUPPORT_INVALID'),
    (b'chrTiny\t1\t2\t'+BC.encode()+b'\t1\textra\n','FRAGMENTS_OUTPUT_MALFORMED'),
    (b'chrTiny\t1\t2\t'+BC.encode()+b'\t1','FRAGMENTS_OUTPUT_MALFORMED')])
def test_raw_parser_rejects(line,code):
    with pytest.raises(common.FragmentsError,match=code):
        production.raw_record(line,{'chrTiny':(0,12000)},{BC},16)


def test_max_support_exact():
    line=f'chrTiny\t1\t2\t{BC}\t{2**64-1}\n'.encode()
    assert production.raw_record(line,{'chrTiny':(0,12000)},{BC},16)[-1]==2**64-1


@pytest.mark.parametrize('stage', ['chromap','sort','bgzip','tabix'])
@pytest.mark.parametrize('mode', ['interrupt','fail'])
def test_process_failure_never_publishes(tiny,bound,fake_runtime,stage,mode):
    inputs,_,_=bound; runtime,control=fake_runtime; control[mode]=stage
    with pytest.raises(KeyboardInterrupt if mode=='interrupt' else common.FragmentsError):
        api.prepare_fastq_fragments(inputs=inputs,runtime=runtime,output_dir=tiny['root']/'published')
    assert not (tiny['root']/'published').exists()
    assert not list(tiny['root'].glob('.published-attempt-*'))


@pytest.mark.parametrize('rows,code', [(b'','FRAGMENTS_EMPTY'),(b'bad\n','FRAGMENTS_OUTPUT_MALFORMED'),
    ((f'chrTiny\t104\t195\t{BC}\t1\n'*2).encode(),'FRAGMENTS_CANONICALIZATION_FAILED')])
def test_invalid_backend_output_never_publishes(tiny,bound,fake_runtime,rows,code):
    inputs,_,_=bound;runtime,control=fake_runtime;control['rows']=rows
    with pytest.raises(common.FragmentsError,match=code):
        api.prepare_fastq_fragments(inputs=inputs,runtime=runtime,output_dir=tiny['root']/'published')
    assert not (tiny['root']/'published').exists()


def test_source_change_after_scan(tiny,bound,fake_runtime,monkeypatch):
    inputs,group,_=bound;runtime,_=fake_runtime; real=api.scan_group
    def changed(*args):
        result=real(*args)
        with open(group.files[0][1],'ab') as out:out.write(b'changed')
        return result
    monkeypatch.setattr(api,'scan_group',changed)
    with pytest.raises(common.FragmentsError,match='FRAGMENTS_SOURCE_CHANGED_DURING_EXECUTION'):
        api.prepare_fastq_fragments(inputs=inputs,runtime=runtime,output_dir=tiny['root']/'published')


@pytest.mark.parametrize('where', ['chromap','tabix'])
def test_source_change_during_stages(tiny,bound,fake_runtime,where):
    inputs,group,_=bound;runtime,control=fake_runtime
    def after(kind):
        if kind==where:
            with open(group.files[0][1],'ab') as out:out.write(b'changed')
    control['after']=after
    with pytest.raises(common.FragmentsError):
        api.prepare_fastq_fragments(inputs=inputs,runtime=runtime,output_dir=tiny['root']/'published')
    assert not (tiny['root']/'published').exists()


@pytest.mark.parametrize('kind', ['bgzf','tabix'])
def test_published_corruption(published,kind):
    result,runtime,_=published;value=manifest.load_fragments_manifest(result['manifest_path'])
    path=Path(result['manifest_path']).parent/value['libraries'][0][kind]['path']
    path.write_bytes(path.read_bytes()[:-3]+b'bad')
    with pytest.raises(common.FragmentsError):verifier.verify_fragments(result['manifest_path'],runtime=runtime)


def test_changed_whitelist_after_publication(published,bound):
    result,runtime,_=published;_,_,white=bound
    Path(white.resource.path).write_text('A'*16+'\n')
    with pytest.raises(common.FragmentsError,match='FRAGMENTS_WHITELIST_MISMATCH'):
        verifier.verify_fragments(result['manifest_path'],runtime=runtime)


@pytest.mark.parametrize('mutation', ['unknown','bool','sum','duplicate_library','duplicate_key','nonfinite','bad_sha'])
def test_strict_manifest(published,mutation):
    result,_,_=published;path=Path(result['manifest_path']); value=json.loads(path.read_bytes())
    if mutation=='unknown':value['extra']=1
    elif mutation=='bool':value['schema_version']=True
    elif mutation=='sum':value['libraries'][0]['sum_support']=2**128
    elif mutation=='duplicate_library':value['libraries']*=2
    elif mutation=='bad_sha':value['libraries'][0]['bgzf']['sha256']='x'*64
    payload=json.dumps(value).encode()
    if mutation=='duplicate_key':payload=payload.replace(b'"schema_version": 1',b'"schema_version": 1, "schema_version": 1')
    elif mutation=='nonfinite':payload=payload.replace(b'"schema_version": 1',b'"schema_version": NaN')
    path.write_bytes(payload)
    with pytest.raises(common.FragmentsError):manifest.load_fragments_manifest(path)


def test_lightweight_loader_and_portability(published,monkeypatch):
    result,_,_=published
    monkeypatch.setattr(binding,'fresh_intake',lambda _:pytest.fail('Unexpected raw IO'))
    value=manifest.load_fragments_manifest(result['manifest_path'])
    other=deepcopy(value)
    other['inputs']['reference_path']='/relocated/reference.json'
    other['libraries'][0]['bgzf']['sha256']='1'*64
    assert manifest.portable_identity(other)==value['fragments_identity_sha256']
    other['libraries'][0]['namespace']='changed'
    assert manifest.portable_identity(other)!=value['fragments_identity_sha256']


def test_independent_verifier_recomputes_counts(published,monkeypatch):
    result,runtime,_=published;path=Path(result['manifest_path'])
    value=manifest.load_fragments_manifest(path);value['libraries'][0]['sum_support']+=1
    value['fragments_identity_sha256']=manifest.portable_identity(value)
    path.write_bytes(manifest.canonical_fragments_manifest_bytes(value))
    monkeypatch.setattr(production,'raw_record',lambda *a:pytest.fail('Production parser called'))
    with pytest.raises(common.FragmentsError,match='FRAGMENTS_VERIFICATION_MISMATCH'):
        verifier.verify_fragments(path,runtime=runtime)


@pytest.mark.parametrize('case', ['gzip','truncated','crc','unordered','duplicate','unknown','bad_support'])
def test_independent_stream_rejects(tmp_path,case):
    raw=f'chrTiny\t104\t195\t{BC}\t300\n'.encode()
    if case=='unordered':raw+=f'chrTiny\t1\t2\t{BC}\t1\n'.encode()
    if case=='duplicate':raw*=2
    if case=='unknown':raw=raw.replace(b'chrTiny',b'chrOther')
    if case=='bad_support':raw=raw.replace(b'300',b'0')
    packed=bgzf_bytes(raw)
    if case=='gzip':packed=gzip.compress(raw)
    if case=='truncated':packed=packed[:-1]
    if case=='crc':packed=packed[:25]+bytes([packed[25]^1])+packed[26:]
    path=tmp_path/'fragments.gz';path.write_bytes(packed)
    entry={'barcode_length':16,'canonical_record_stream_sha256':hashlib.sha256(raw).hexdigest(),
           'n_fragment_records':1,'sum_support':300,'max_support':300,'n_distinct_barcodes':1}
    with pytest.raises(common.FragmentsError):verifier.verify_stream(path,entry,[('chrTiny',12000)],{BC})


def test_fai_rank_numeric_and_barcode_sort(tiny,fake_runtime):
    runtime,_=fake_runtime;other='A'*16
    raw=tiny['root']/'raw.bed'
    raw.write_text(f'chr1\t1\t2\t{BC}\t1\nchr2\t10\t20\t{BC}\t1\n'
        f'chr2\t2\t3\t{BC}\t1\nchr2\t2\t3\t{other}\t1\n')
    path,_=production.canonicalize(raw,directory=tiny['root'],contigs=[('chr2',100),('chr1',100)],
                                   whitelist={BC,other},barcode_length=16,runtime=runtime)
    rows=path.read_text().splitlines()
    assert rows[0]==f'chr2\t2\t3\t{other}\t1' and rows[-1].startswith('chr1\t')


@pytest.mark.parametrize('length',[31,32,33])
def test_executable_barcode_bound_before_full_scan(tiny,reads,length):
    token=('ACGT'*9)[:length]
    group,white=reads([(100,200,token,1)],whitelist=(token,))
    inputs=inputs_for(tiny,[group],white)
    if length==33:
        with pytest.raises(common.FragmentsError,match='FRAGMENTS_BARCODE_LENGTH_UNSUPPORTED'):
            binding.preflight(inputs,backend())
    else:assert binding.preflight(inputs,backend())['libraries'][0][0].whitelist.barcode_length==length


def test_unrepresentable_path_preflight(tiny,reads):
    group,white=reads([(100,200,BC,1)],name='lane,unsafe')
    inputs=inputs_for(tiny,[group],white)
    with pytest.raises(common.FragmentsError,match='FRAGMENTS_INPUT_BINDING_UNREPRESENTABLE'):
        binding.preflight(inputs,backend())


def test_missing_whitelist_closed_context(tiny,bound):
    inputs,_,_=bound;path=Path(inputs.context_path);value=json.loads(path.read_bytes())
    value['libraries'][0]['whitelist']=None
    path.write_text(json.dumps(value))
    inputs=replace(inputs,context_sha256=c.sha256(path))
    with pytest.raises(common.FragmentsError,match='FRAGMENTS_CONTEXT_MISMATCH'):
        binding.preflight(inputs,backend())


def test_other_intake_context_binding(tiny,bound):
    from agent.tools.data import scatac_library_context as lc
    inputs,_,_=bound;path=Path(inputs.context_path)
    _,context,_=lc.load_scatac_library_processing_context(path)
    context=replace(context,intake=replace(context.intake,manifest_sha256='0'*64))
    context=replace(context,context_identity_sha256=lc._context_digest(context))
    path.write_bytes(lc.canonical_library_processing_context_bytes(context))
    with pytest.raises(common.FragmentsError,match='FRAGMENTS_CONTEXT_MISMATCH'):
        binding.preflight(replace(inputs,context_sha256=c.sha256(path)),backend())


def test_species_gate_with_consistent_other_reference(tiny,bound):
    from agent.tools.data import scatac_reference as ref, chromap_reference_index as ci
    inputs,_,_=bound
    bundle=ref.build_scatac_reference_bundle(species='mouse',target_assembly='mm10',
        fasta_path=tiny['fa'],fai_path=tiny['fai'],ccre_bed_path=tiny['bed'])
    pointer=ref.publish_scatac_reference_bundle(bundle,tiny['root']/'mouse.json')
    old=ci.load_chromap_reference_index(inputs.index_path)
    changed=replace(old,**ci._binding(bundle,pointer['manifest_sha256'],backend()))
    changed=replace(changed,index_identity_sha256=ci._identity(changed))
    Path(inputs.index_path).write_bytes(ci.canonical_chromap_reference_index_bytes(changed))
    inputs=replace(inputs,reference_path=pointer['manifest_path'],reference_sha256=pointer['manifest_sha256'],
                   index_sha256=c.sha256(inputs.index_path))
    with pytest.raises(common.FragmentsError,match='FRAGMENTS_SPECIES_MISMATCH'):
        binding.preflight(inputs,backend())


def test_tbi_limit_no_csi_fallback(bound,monkeypatch):
    from agent.tools.data import chromap_reference_index as ci
    inputs,_,_=bound
    monkeypatch.setattr(ci,'verify_fasta_fai',lambda _: (('chrTiny',2**29+1),))
    with pytest.raises(common.FragmentsError,match='FRAGMENTS_INDEX_POLICY_UNSUPPORTED'):
        binding.preflight(inputs,backend())


@pytest.mark.parametrize('point',['scan','verification','manifest'])
def test_nonprocess_interruption(tiny,bound,fake_runtime,monkeypatch,point):
    inputs,_,_=bound;runtime,_=fake_runtime
    def interrupt(*a,**k):raise KeyboardInterrupt()
    target={'scan':'scan_group','verification':'verify_fragments','manifest':'canonical_fragments_manifest_bytes'}[point]
    monkeypatch.setattr(api,target,interrupt)
    with pytest.raises(KeyboardInterrupt):
        api.prepare_fastq_fragments(inputs=inputs,runtime=runtime,output_dir=tiny['root']/'published')
    assert not (tiny['root']/'published').exists()
    assert not list(tiny['root'].glob('.published-attempt-*'))


def test_missing_chromap_output(tiny,bound,fake_runtime,monkeypatch):
    inputs,_,_=bound;runtime,_=fake_runtime
    monkeypatch.setattr(api,'run_stage',lambda *a,**k:None)
    with pytest.raises(common.FragmentsError,match='CHROMAP_OUTPUT_INCOMPLETE'):
        api.prepare_fastq_fragments(inputs=inputs,runtime=runtime,output_dir=tiny['root']/'published')


def test_concurrent_publication_one_winner(tiny,bound,fake_runtime):
    from concurrent.futures import ThreadPoolExecutor
    inputs,_,_=bound;runtime,control=fake_runtime
    def attempt(_):
        try:return api.prepare_fastq_fragments(inputs=inputs,runtime=runtime,output_dir=tiny['root']/'published')['status']
        except common.FragmentsError as e:return e.code
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(attempt,range(2)))
    assert sorted(results)==['FRAGMENTS_ARTIFACT_CONFLICT','success']
    assert control['calls'].count('chromap')==1


def test_fake_tbi_queries_cannot_pass_with_wrong_rows(published,monkeypatch):
    result,runtime,_=published
    monkeypatch.setattr(verifier,'_query',lambda *args:'0'*64)
    with pytest.raises(common.FragmentsError,match='FRAGMENTS_INDEX_MISMATCH'):
        verifier.verify_fragments(result['manifest_path'],runtime=runtime)


def test_invalid_executable_never_reaches_preflight(tiny,bound,monkeypatch):
    inputs,_,_=bound;exe=tiny['root']/'wrong';exe.write_text('wrong');exe.chmod(0o700)
    monkeypatch.setattr(api,'preflight',lambda *a,**k:pytest.fail('Executed after invalid binary'))
    with pytest.raises(c.ChromapError,match='Executable hash differs'):
        api.prepare_fastq_fragments(inputs=inputs,runtime=common.FragmentsRuntime(str(exe)),
                                   output_dir=tiny['root']/'published')


def test_many_bgzf_blocks_and_large_support(tmp_path):
    raw=b''.join(f'chrTiny\t{i}\t{i+1}\t{BC}\t{2**64-1}\n'.encode() for i in range(5000))
    path=tmp_path/'wide.gz';path.write_bytes(bgzf_bytes(raw))
    entry={'barcode_length':16,'canonical_record_stream_sha256':hashlib.sha256(raw).hexdigest(),
        'n_fragment_records':5000,'sum_support':5000*(2**64-1),'max_support':2**64-1,'n_distinct_barcodes':1}
    per,_,_=verifier.verify_stream(path,entry,[('chrTiny',12000)],{BC})
    assert per['chrTiny'].hexdigest()==hashlib.sha256(raw).hexdigest()


def test_partial_manifest_write_never_publishes(tiny,bound,fake_runtime,monkeypatch):
    inputs,_,_=bound;runtime,_=fake_runtime;original=Path.write_bytes
    def interrupted(path,data):
        if path.name=='manifest.json' and '-attempt-' in str(path):
            original(path,data[:len(data)//2]);raise KeyboardInterrupt()
        return original(path,data)
    monkeypatch.setattr(Path,'write_bytes',interrupted)
    with pytest.raises(KeyboardInterrupt):
        api.prepare_fastq_fragments(inputs=inputs,runtime=runtime,output_dir=tiny['root']/'published')
    assert not (tiny['root']/'published').exists()
    assert not list(tiny['root'].glob('.published-attempt-*'))


def test_unexpected_backend_artifact_rejected(tiny,bound,fake_runtime):
    inputs,_,_=bound;runtime,control=fake_runtime
    def after(kind):
        if kind=='chromap':
            stage=next(tiny['root'].glob('.published-attempt-*'))
            (stage/'libraries'/'library_0'/'unreviewed').write_text('unexpected')
    control['after']=after
    with pytest.raises(common.FragmentsError,match='CHROMAP_OUTPUT_INCOMPLETE'):
        api.prepare_fastq_fragments(inputs=inputs,runtime=runtime,output_dir=tiny['root']/'published')


def test_missing_published_component_has_stable_error(published):
    result,runtime,_=published;value=manifest.load_fragments_manifest(result['manifest_path'])
    (Path(result['manifest_path']).parent/value['libraries'][0]['tabix']['path']).unlink()
    with pytest.raises(common.FragmentsError,match='FRAGMENTS_VERIFICATION_MISMATCH'):
        verifier.verify_fragments(result['manifest_path'],runtime=runtime)


def test_validator_rejects_unhashable_nested_shape(published):
    result,_,_=published;value=manifest.load_fragments_manifest(result['manifest_path'])
    value['libraries'][0]['group_ids']=[{}]
    with pytest.raises(common.FragmentsError):manifest.validate_fragments_manifest(value)


@pytest.mark.parametrize('path',['relative.json','/tmp/../reference.json','/tmp/ref\n.json'])
def test_input_locations_rejected_before_source_io(bound,monkeypatch,path):
    inputs,_,_=bound
    monkeypatch.setattr(binding.m,'load_raw_intake_manifest',lambda *a,**k:pytest.fail('Source opened'))
    with pytest.raises(common.FragmentsError,match='FRAGMENTS_INPUT_BINDING_UNREPRESENTABLE'):
        binding.preflight(replace(inputs,reference_path=path),backend())


@pytest.mark.parametrize('namespace',['manifest.json','libraries','process.log'])
def test_valid_namespaces_do_not_collide(tiny,bound,fake_runtime,namespace):
    from agent.tools.data import scatac_library_context as lc
    inputs,_,_=bound;runtime,_=fake_runtime;path=Path(inputs.context_path)
    _,context,_=lc.load_scatac_library_processing_context(path)
    context=replace(context,libraries=(replace(context.libraries[0],namespace=namespace),))
    context=replace(context,context_identity_sha256=lc._context_digest(context))
    path.write_bytes(lc.canonical_library_processing_context_bytes(context))
    inputs=replace(inputs,context_sha256=c.sha256(path))
    result=api.prepare_fastq_fragments(inputs=inputs,runtime=runtime,output_dir=tiny['root']/'published')
    value=verifier.verify_fragments(result['manifest_path'],runtime=runtime)
    assert value['libraries'][0]['namespace']==namespace
