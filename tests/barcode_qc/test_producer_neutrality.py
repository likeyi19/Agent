"""Accepted synthetic producer artifacts, no Chromap alignment/BAM production.

FASTQ's external aligner boundary is fixture-controlled. BAM's independently
reconstructed fixture retains real source/header/support authority. All three
producer verifiers and real packaging/BEDTools execute without mocking QC.
"""
import gzip
import json
from pathlib import Path
import runpy
import subprocess

from agent.tools.data import _chromap as c, fastq_fragments as fastq
from agent.tools.data import scatac_qc_reference as qr, scatac_barcode_qc as qc
from agent.tools.data import _bam_fragment_io as io, bam_fragment_manifest as bm
from agent.tools.data import bam_fragments_verifier as bv, scatac_fragments_v2 as v2
from agent.tools.data import raw_scatac_manifest as raw, scatac_library_context as lc
from agent.tools.data.raw_scatac import inspect_raw_scATAC
from agent.tools.data._fragments_common import FragmentsRuntime, canonical
from agent.tools.data.scatac_fragment_import import import_scATAC_fragments
from agent.tools.data.external_fragment_manifest import PROFILE_ID
from agent.tools.data.scatac_fragments_v2_verifier import FragmentVerificationRuntime


def fastq_fixture(tmp_path, monkeypatch, namespaces=1):
    fixture_dir=Path(__file__).parents[1]/'chromap'
    monkeypatch.syspath_prepend(str(fixture_dir))
    from fragments_helpers import BC, inputs_for, backend
    fixtures=runpy.run_path(str(fixture_dir/'conftest.py'))
    tiny=fixtures['tiny'].__wrapped__(tmp_path)
    reads=fixtures['reads'].__wrapped__(tiny)
    groups=[]
    for i in range(namespaces):
        group,white=reads([(996,1110,BC,2),(2996,3106,BC,3)],name=f'reads-{i}')
        groups.append(group)
    inputs=inputs_for(tiny,groups,white)
    exe=tmp_path/'fixture-chromap';exe.write_text('fixture only')
    monkeypatch.setattr(c,'identify_backend',lambda *a,**k:backend())
    original=fastq.run_stage
    def stage(argv,**kwargs):
        if '--preset' in argv:
            Path(argv[argv.index('--output')+1]).write_text(f'chrTiny\t1000\t1105\t{BC}\t2\nchrTiny\t3000\t3101\t{BC}\t3\n')
        else:original(argv,**kwargs)
    monkeypatch.setattr(fastq,'run_stage',stage)
    result=fastq.prepare_fastq_fragments(inputs=inputs,runtime=FragmentsRuntime(str(exe)),output_dir=tmp_path/'fastq')
    annotation=tmp_path/'annotation.tsv'
    annotation.write_text('chrTiny\t3000\t3500\t+\tg1\tt1\tprotein_coding\n'
        'chrTiny\t2500\t3001\t-\tg2\tt2\tprotein_coding\nchrTiny\t3100\t3500\t+\tg3\tt3\tprotein_coding\n')
    qr.build_scatac_qc_reference_bundle(parent_manifest_path=tiny['pointer']['manifest_path'],
        parent_manifest_sha256=tiny['pointer']['manifest_sha256'],annotation_path=annotation,
        annotation_source='synthetic',annotation_release='1',
        classifications=(qr.QCContig('chrTiny',12000,'primary_nuclear_qc'),),
        classification_source='explicit-test',output_dir=tmp_path/'qc-reference')
    return tiny,result,BC


def bam_fixture(tiny,barcode):
    import pysam
    root=tiny['root'];bam=root/'input.bam'
    with pysam.AlignmentFile(str(bam),'wb',header={'HD':{'VN':'1.6'},'SQ':[{'SN':'chrTiny','LN':12000}]}) as out:
        for i,(left,right) in enumerate(((996,1110),(2996,3106))):
            for flag,start,mate,tlen in ((99,left,right-50,right-left),(147,right-50,left,left-right)):
                r=pysam.AlignedSegment(out.header);r.query_name=f'template-{i}';r.flag=flag
                r.reference_id=r.next_reference_id=0;r.reference_start=start;r.next_reference_start=mate
                r.template_length=tlen;r.cigarstring='50M';r.mapping_quality=30;r.query_sequence='A'*50
                r.set_tag('CB',barcode);out.write(r)
    intake=inspect_raw_scATAC(str(bam),str(root/'bam-intake'),species='human',raw_assay='SCATAC',source_genome_assembly='hg38')
    _,manifest,_=raw.load_raw_intake_manifest(intake['manifest_path'])
    context=lc.build_scatac_library_processing_context(intake_manifest_path=intake['manifest_path'],
        expected_intake_sha256=intake['manifest_sha256'],selection_mode=lc.SelectionMode.ALL_GROUPS,
        libraries=(lc.LibraryDeclaration(namespace='library_0',group_ids=(manifest.groups[0].id,),
            membership_basis=lc.MembershipBasis.SINGLE_GROUP,barcode_interpretation=lc.BarcodeInterpretation.CORRECTED_IDENTIFIER,
            correction_policy=lc.CorrectionPolicy.ALREADY_CORRECTED),))
    cp=lc.publish_scatac_library_processing_context(context,root/'bam-context.json')
    args=dict(intake_manifest_path=intake['manifest_path'],intake_manifest_sha256=intake['manifest_sha256'],
        library_context_path=cp['manifest_path'],library_context_sha256=cp['manifest_sha256'],
        reference_bundle_path=tiny['pointer']['manifest_path'],reference_bundle_sha256=tiny['pointer']['manifest_sha256'],
        source_path=str(bam),source_sha256=c.sha256(bam),source_profile=bm.PROFILE_ID,output_dir=str(root/'bam-output'))
    bound=io.bind(args);stage=root/'bam-fixture';directory=stage/'libraries/library_0';directory.mkdir(parents=True)
    plain,qualification,summary=bv.reconstruct(bound,directory,FragmentVerificationRuntime())
    bgzf=directory/'fragments.tsv.gz'
    with bgzf.open('wb') as target:subprocess.run(['/usr/bin/bgzip','-c',str(plain)],stdout=target,check=True)
    subprocess.run(['/usr/bin/tabix','-p','bed',str(bgzf)],check=True)
    outputs={k:io.resource(p) for k,p in (('bgzf',bgzf),('tabix',Path(str(bgzf)+'.tbi')))}
    record=bm.validate_record(dict(artifact_type='agent.bam-fragment-production',schema_version=1,
        contract_version=bm.PRODUCTION_CONTRACT,profile_sha256=bm.sha_bytes(bm.PROFILE_BYTES),arguments=args,
        **{k:bound[k] for k in ('source','intake','context','reference','namespace','group_id','context_identity_sha256')},
        source_history=bm.HISTORY,runtime=io.runtime_identity(),qualification=qualification,canonical=summary,
        outputs={k:{f:v[f] for f in ('sha256','size_bytes')} for k,v in outputs.items()}))
    profile=stage/'profile.json';profile.write_bytes(bm.PROFILE_BYTES)
    producer=stage/'production.json';producer.write_bytes(canonical(record))
    entry=dict(namespace='library_0',**summary,strand={'mode':'absent','definition':None},
        provenance=bm.provenance(record,io.resource(profile),io.resource(producer)))
    entry.update({k:v|{'path':str(Path(v['path']).relative_to(stage))} for k,v in outputs.items()})
    value=dict(artifact_type=v2.ARTIFACT_TYPE,schema_version=2,contract_version=v2.CONTRACT_VERSION,
        reference=bound['reference'],semantics=v2.SEMANTICS,libraries=[entry])
    value['fragments_identity_sha256']=v2.fragments_identity(value)
    path=stage/'manifest.json';path.write_bytes(v2.canonical_fragments_manifest_v2_bytes(value))
    return dict(manifest_path=str(path),manifest_sha256=c.sha256(path))


def compute(root,result,index):
    reference=root/'qc-reference/manifest.json'
    result=qc.compute_scATAC_qc(fragments_manifest_path=result['manifest_path'],fragments_manifest_sha256=result['manifest_sha256'],
        qc_reference_manifest_path=str(reference),qc_reference_manifest_sha256=c.sha256(reference),output_dir=str(root/f'qc-{index}'))
    path=Path(result['manifest_path']);manifest=json.loads(path.read_bytes())
    return manifest,(path.parent/'barcodes.tsv.gz').read_bytes(),(path.parent/'lengths.tsv.gz').read_bytes()


def test_three_producers_same_nonzero_tss_metrics(tmp_path,monkeypatch):
    tiny,fq,barcode=fastq_fixture(tmp_path,monkeypatch)
    bam=bam_fixture(tiny,barcode)
    source=tmp_path/'external.tsv';source.write_text(f'chrTiny\t1000\t1105\t{barcode}\t200\nchrTiny\t3000\t3101\t{barcode}\t900\n')
    external=import_scATAC_fragments(source_path=str(source),source_sha256=c.sha256(source),source_profile=PROFILE_ID,
        reference_bundle_path=tiny['pointer']['manifest_path'],reference_bundle_sha256=tiny['pointer']['manifest_sha256'],
        namespace='library_0',output_dir=str(tmp_path/'external'))
    results=[compute(tmp_path,r,i) for i,r in enumerate((fq,bam,external))]
    assert len({r[1] for r in results})==len({r[2] for r in results})==1
    assert all(r[0]['summary']==results[0][0]['summary'] for r in results)
    summary=results[0][0]['summary']
    assert summary['n_fragment_records']==2 and summary['tss_defined']==1
    assert summary['tss_center_count']>0 and summary['tss_left_flank_count']>0
    assert [r[0]['producer_authority']['kind'] for r in results]==[
        'fastq_fragment_production','bam_fragment_production','external_fragment_adoption']
    assert [json.loads(Path(r['manifest_path']).read_bytes())['libraries'][0]['sum_support'] for r in (fq,bam,external)]==[5,2,1100]


def test_same_barcode_in_multiple_namespaces_is_never_collapsed(tmp_path,monkeypatch):
    _,fq,barcode=fastq_fixture(tmp_path,monkeypatch,namespaces=2)
    manifest,table,_=compute(tmp_path,fq,0)
    rows=[line.split('\t') for line in gzip.decompress(table).decode().splitlines()[1:]]
    assert [(r[0],r[1]) for r in rows]==[('library_0',barcode),('library_1',barcode)]
    assert manifest['row_count']==2 and manifest['summary']['n_fragment_records']==4
    assert rows[0][2:]==rows[1][2:]
