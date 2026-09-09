"""Pure Agent-owned contract tests; no compiled backend required."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from agent.tools.data import _chromap as c
from agent.tools.data import chromap_reference_index as idx
from agent.tools.data import scatac_reference as ref
from agent.tools.data.scatac_library_context import inspect_barcode_whitelist

@pytest.fixture
def backend():
    return c.BackendIdentity(c.QUALIFIED_EXECUTABLE_SHA256,'x86_64','Linux','little',64,c.policy_sha256())

@pytest.fixture
def fake_build(tiny,backend,monkeypatch):
    exe=tiny['root']/'fake-executable';exe.write_bytes(b'unit-test-placeholder')
    calls=[]
    monkeypatch.setattr(c,'identify_backend',lambda *a,**k:backend)
    def run(argv,**kwargs):
        calls.append(argv)
        Path(argv[argv.index('--output')+1]).write_bytes(b'tiny-native-index')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(idx.subprocess,'run',run)
    args=dict(reference_manifest_path=tiny['pointer']['manifest_path'],
        expected_reference_sha256=tiny['pointer']['manifest_sha256'],
        executable=exe,backend=backend,output_dir=tiny['root']/'index')
    return args,calls


def test_patch_identity_repository_artifact():
    patch=Path(__file__).resolve().parents[2]/'third_party/patches/chromap/support-v1.patch'
    assert c.sha256(patch)==c.PATCH_SHA256
    text=patch.read_text()
    assert 'uint64_t num_dups_' in text
    assert 'AGENT_DUPLICATE_SUPPORT_OVERFLOW' in text
    assert 'src/mapping_generator' not in text


def test_patch_and_source_mismatch(tmp_path):
    p=tmp_path/'wrong';p.write_text('foreign source')
    with pytest.raises(c.ChromapError,match='identity'):
        c.verify_patch_source(p,p)

@pytest.mark.parametrize('field,value',[
    ('upstream_commit','0'*40),('patch_sha256','0'*64),('upstream_version','0.3.3'),
    ('backend_policy','stock'),('policy_sha256','0'*64),('source_tar_sha256','0'*64),
    ('machine','aarch64'),('system','Darwin'),('byteorder','big'),('pointer_bits',True),
    ('executable_sha256','bad')])
def test_backend_fail_closed(backend,field,value):
    with pytest.raises(c.ChromapError): c.validate_backend(replace(backend,**{field:value}))


def test_backend_executable_required(tmp_path):
    with pytest.raises(c.ChromapError): c.identify_backend(tmp_path/'absent',expected_sha256=c.QUALIFIED_EXECUTABLE_SHA256)


def test_argv_layout_order_and_policy(tiny,reads):
    a,w=reads([(100,200,'ACGT'*4,1)],name='a')
    b,_=reads([(100,200,'ACGT'*4,1)],name='b',layout=c.FastqLayout.B)
    native=tiny['root']/'native';native.write_bytes(b'index')
    args=dict(fasta=tiny['fa'],index=native,output=tiny['root']/'out',whitelist=w)
    argv=c.mapping_argv('/explicit/chromap',groups=[b,a],**args)
    assert argv==c.mapping_argv('/explicit/chromap',groups=[a,b],**args)
    assert argv[1:1+len(c.FIXED_FLAGS)]==c.FIXED_FLAGS
    for flag,meaning in [('-1',c.ReadMeaning.GENOMIC_1),('-2',c.ReadMeaning.GENOMIC_2),('-b',c.ReadMeaning.BARCODE)]:
        expected=[]
        for g in sorted([a,b],key=lambda g:g.group_id):
            role=next(r for r,m in c.LAYOUT_ROLES[g.layout].items() if m is meaning)
            expected.append(dict(g.files)[role])
        assert argv[argv.index(flag)+1]==','.join(expected)
    assert '--read-format' not in argv and '--summary' not in argv
    assert not any('I1_001' in s for s in argv)

@pytest.mark.parametrize('case',['missing_white','long','wrong_white','missing_role','duplicate_group','foreign_role','foreign_layout','conflict'])
def test_mapping_preflight(tiny,reads,case):
    g,w=reads([(100,200,'ACGT'*4,1)])
    native=tiny['root']/'native';native.write_bytes(b'index')
    out=tiny['root']/'out';groups=[g]
    if case=='missing_white':w=None
    elif case=='long':
        p=tiny['root']/'long';p.write_text('A'*33+'\n');w=inspect_barcode_whitelist(p)
    elif case=='wrong_white':Path(w.resource.path).write_text('T'*16+'\n')
    elif case=='missing_role':groups=[replace(g,files=g.files[:-2])]
    elif case=='duplicate_group':groups=[g,g]
    elif case=='foreign_role':groups=[replace(g,files=g.files+((c.ReadRole.I2,str(native)),))]
    elif case=='foreign_layout':groups=[replace(g,layout='A')]
    elif case=='conflict':out.write_text('preserve')
    with pytest.raises(c.ChromapError):
        c.mapping_argv('/exe',fasta=tiny['fa'],index=native,output=out,groups=groups,whitelist=w)


def test_fasta_fai_exact_and_no_ccre_input(tiny):
    tiny['bed'].unlink()
    assert idx.verify_fasta_fai(tiny['bundle'])==(('chrTiny',12000),)

@pytest.mark.parametrize('change',['length','name','offset','width','sequence','empty','duplicate','order'])
def test_fasta_fai_rejects(tiny,change):
    fa=tiny['fa'];fai=tiny['fai'];b=tiny['bundle']
    if change=='length':fai.write_text(fai.read_text().replace('12000\t9','11999\t9'))
    elif change=='name':fai.write_text(fai.read_text().replace('chrTiny','chrOther'))
    elif change=='offset':fai.write_text(fai.read_text().replace('\t9\t','\t10\t'))
    elif change=='width':fai.write_text(fai.read_text().replace('12001','12002'))
    elif change=='sequence':fa.write_text(fa.read_text().replace('A','T',1))
    elif change=='empty':fa.write_text('>chrTiny\n')
    elif change=='duplicate':fa.write_text(fa.read_text()+fa.read_text())
    elif change=='order':fa.write_text('>extra\nACGT\n'+fa.read_text())
    with pytest.raises(c.ChromapError):idx.verify_fasta_fai(b)


def test_index_roundtrip_reuse_and_concurrent_publication(tiny,fake_build):
    args,calls=fake_build
    with ThreadPoolExecutor(max_workers=2) as pool:
        values=list(pool.map(lambda _:idx.prepare_chromap_reference_index(**args),range(2)))
    assert values[0]==values[1] and len(calls)==1
    value=values[0];manifest=args['output_dir']/'manifest.json'
    assert idx.load_chromap_reference_index(manifest)==value
    assert json.loads(manifest.read_bytes())==asdict(value)
    assert set(p.name for p in args['output_dir'].iterdir())=={'reference.chromap','manifest.json','build.log'}
    assert 'ccre' not in str(calls[0])
    assert calls[0][-4:]==('--kmer','17','--window','7')
    assert not list(tiny['root'].glob('.index-*/'))

@pytest.mark.parametrize('case',['index','manifest','backend','reference','incomplete'])
def test_index_reuse_conflicts_preserve_bytes(tiny,fake_build,case):
    args,calls=fake_build;value=idx.prepare_chromap_reference_index(**args)
    manifest=args['output_dir']/'manifest.json';native=args['output_dir']/'reference.chromap'
    if case=='index':native.write_bytes(b'changed')
    elif case=='manifest':manifest.write_bytes(b'{}')
    elif case=='backend':args['backend']=replace(args['backend'],executable_sha256='b'*64)
    elif case=='reference':Path(args['reference_manifest_path']).write_bytes(b'{}')
    elif case=='incomplete':manifest.unlink()
    original=native.read_bytes()
    with pytest.raises(ValueError):idx.prepare_chromap_reference_index(**args)
    assert native.read_bytes()==original and len(calls)==1

@pytest.mark.parametrize('field,value',[('kmer',19),('window',8),('schema_version',True),
    ('index_size_bytes',0),('index_file','../escape'),('contract_version','other'),
    ('index_identity_sha256','a'*64),('reference_identity_sha256','b'*64)])
def test_index_contract_closed(fake_build,field,value):
    args,_=fake_build;v=idx.prepare_chromap_reference_index(**args)
    with pytest.raises(c.ChromapError):idx.validate_chromap_reference_index(replace(v,**{field:value}))
    d=asdict(v);d['extra']=None
    with pytest.raises(c.ChromapError):idx.validate_chromap_reference_index(d)


def test_failed_build_and_source_mutation_cleanup(tiny,fake_build,monkeypatch):
    args,_=fake_build
    def run(argv,**kw):
        Path(argv[argv.index('--output')+1]).write_bytes(b'partial')
        tiny['fa'].write_text(tiny['fa'].read_text().replace('A','C',1))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(idx.subprocess,'run',run)
    with pytest.raises(c.ChromapError):idx.prepare_chromap_reference_index(**args)
    assert not args['output_dir'].exists() and not list(tiny['root'].glob('.index-*/'))


def test_nonzero_build_cleanup(tiny,fake_build,monkeypatch):
    args,_=fake_build
    monkeypatch.setattr(idx.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=1))
    with pytest.raises(c.ChromapError,match='build failed'):idx.prepare_chromap_reference_index(**args)
    assert not args['output_dir'].exists()


def test_index_json_strict(fake_build):
    args,_=fake_build;idx.prepare_chromap_reference_index(**args)
    p=args['output_dir']/'manifest.json';raw=p.read_bytes()
    for payload in (b'{"x":1,"x":2}',b'{"x":NaN}',b' '*65537,b'[]'):
        p.write_bytes(payload)
        with pytest.raises(c.ChromapError):idx.load_chromap_reference_index(p)
    p.write_bytes(raw)
    with pytest.raises(c.ChromapError):idx.load_chromap_reference_index(p,expected_sha256='0'*64)


def test_build_record_pin():
    p=Path(__file__).resolve().parents[2]/'third_party/patches/chromap/qualification-build.json'
    assert c.sha256(p)==c.BUILD_RECORD_SHA256
    data=json.loads(p.read_text())
    assert data['candidate']['sha256']==c.QUALIFIED_EXECUTABLE_SHA256
    assert data['patch_sha256']==c.PATCH_SHA256
    assert data['git_archive_tar_sha256']==c.SOURCE_TAR_SHA256


def test_unqualified_stock_identity_rejected():
    with pytest.raises(c.ChromapError) as error:
        c.identify_backend('/irrelevant',expected_sha256='623a8cbc08ff541d102cb5fc314546e4971ee1225f5c157749dd0cc7649d9b31')
    assert error.value.code=='CHROMAP_EXECUTABLE_UNQUALIFIED'


def test_linked_runtime_drift(monkeypatch):
    monkeypatch.setattr(c.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout=b''))
    with pytest.raises(c.ChromapError) as error:c.verify_linked_runtime('/exe')
    assert error.value.code=='CHROMAP_RUNTIME_MISMATCH'


def test_m10_context_library_bindings(tiny,reads):
    from agent.tools.data import _raw_fastq as fastq
    from agent.tools.data import raw_scatac_manifest as m
    from agent.tools.data import scatac_library_context as lib
    a,w=reads([(100,200,'ACGT'*4,2)],name='context_A')
    b,_=reads([(100,200,'ACGT'*4,3)],name='context_B')
    paths=[path for g in (a,b) for _,path in g.files]
    intake=fastq.inspect_fastq_inputs(paths,assay=fastq.FastqAssay.TENX_ATAC,
        species='human',declared_layout=fastq.FastqLayout.A)
    ptr=m.publish_raw_intake_manifest(intake,tiny['root']/'intake.json')
    declarations=tuple(lib.LibraryDeclaration(namespace=f'library_{i}',group_ids=(g.id,),
        membership_basis=lib.MembershipBasis.SINGLE_GROUP,
        barcode_interpretation=lib.BarcodeInterpretation.RAW_SEQUENCE,
        correction_policy=lib.CorrectionPolicy.WHITELIST_REQUIRED,whitelist=w)
        for i,g in enumerate(intake.groups))
    context=lib.build_scatac_library_processing_context(intake_manifest_path=ptr['manifest_path'],
        expected_intake_sha256=ptr['manifest_sha256'],libraries=declarations,
        selection_mode=lib.SelectionMode.ALL_GROUPS)
    assert lib.validate_library_context_intake_binding(context)==context
    assert len(context.libraries)==2
    assert {g.group_id for l in context.libraries for g in l.groups}=={g.id for g in intake.groups}
    assert len({(l.namespace,'ACGT'*4) for l in context.libraries})==2


def test_other_valid_reference_binding_cannot_replace_index(tiny,fake_build):
    args,calls=fake_build;original=idx.prepare_chromap_reference_index(**args)
    tiny['bed'].write_text('chrTiny\t200\t300\n')
    other=ref.build_scatac_reference_bundle(species='human',target_assembly='hg38',
        fasta_path=tiny['fa'],fai_path=tiny['fai'],ccre_bed_path=tiny['bed'])
    pointer=ref.publish_scatac_reference_bundle(other,tiny['root']/'other-reference.json')
    args.update(reference_manifest_path=pointer['manifest_path'],expected_reference_sha256=pointer['manifest_sha256'])
    with pytest.raises(c.ChromapError) as error:idx.prepare_chromap_reference_index(**args)
    assert error.value.code=='CHROMAP_INDEX_BINDING_MISMATCH'
    assert len(calls)==1
    assert idx.load_chromap_reference_index(args['output_dir']/'manifest.json')==original


def test_patcher_rejects_foreign_source_before_applying(tmp_path,monkeypatch):
    p=tmp_path/'foreign';p.write_text('foreign')
    monkeypatch.setattr(c.subprocess,'run',lambda *a,**k:pytest.fail('must not apply'))
    with pytest.raises(c.ChromapError):c.prepare_support_source(p,p,tmp_path/'out')
    assert not (tmp_path/'out').exists()
