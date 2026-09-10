"""Real qualified three-producer convergence; no biological input or network."""
from pathlib import Path
import pytest
from fragments_helpers import BC, inputs_for
from agent.tools.data import _chromap as c
from agent.tools.data._fragments_common import FragmentsRuntime
from agent.tools.data.fastq_fragments import prepare_fastq_fragments
from agent.tools.data.scatac_fragment_reader import open_verified_fragments, FragmentRecord
from agent.tools.data.scatac_fragments_v2_verifier import FragmentVerificationRuntime


def test_three_real_routes_share_v2_boundary(tiny, reads, executables):
    import pysam
    from agent.tools.data import raw_scatac_manifest as raw, scatac_library_context as lc
    from agent.tools.data.raw_scatac import inspect_raw_scATAC
    from agent.tools.data.scatac_bam_fragments import prepare_scATAC_bam_fragments
    from agent.tools.data.scatac_fragment_import import import_scATAC_fragments
    from agent.tools.data.bam_fragment_manifest import PROFILE_ID as BAM_PROFILE
    from agent.tools.data.external_fragment_manifest import PROFILE_ID as EXTERNAL_PROFILE

    group, white = reads([(100,200,BC,1)])
    runtime = FragmentsRuntime(executables['candidate']['path'])
    inputs = inputs_for(tiny, [group], white, executable=runtime.chromap)
    fastq = prepare_fastq_fragments(inputs=inputs, output_dir=tiny['root']/'fastq-out', runtime=runtime)

    bam = tiny['root']/'source.bam'
    with pysam.AlignmentFile(str(bam),'wb',header={'HD':{'VN':'1.6'},'SQ':[{'SN':'chrTiny','LN':12000}]}) as out:
        for flag,start,mate,tlen in ((99,100,150,100),(147,150,100,-100)):
            read=pysam.AlignedSegment(out.header);read.query_name='one';read.flag=flag
            read.reference_id=read.next_reference_id=0;read.reference_start=start
            read.next_reference_start=mate;read.template_length=tlen;read.cigarstring='50M'
            read.mapping_quality=30;read.query_sequence='A'*50;read.set_tag('CB',BC);out.write(read)
    intake=inspect_raw_scATAC(str(bam),str(tiny['root']/'bam-intake'),species='human',raw_assay='SCATAC',source_genome_assembly='hg38')
    _, manifest, _=raw.load_raw_intake_manifest(intake['manifest_path'])
    context=lc.build_scatac_library_processing_context(intake_manifest_path=intake['manifest_path'],
        expected_intake_sha256=intake['manifest_sha256'],selection_mode=lc.SelectionMode.ALL_GROUPS,
        libraries=(lc.LibraryDeclaration(namespace='library_0',group_ids=(manifest.groups[0].id,),
            membership_basis=lc.MembershipBasis.SINGLE_GROUP,
            barcode_interpretation=lc.BarcodeInterpretation.CORRECTED_IDENTIFIER,
            correction_policy=lc.CorrectionPolicy.ALREADY_CORRECTED),))
    context=lc.publish_scatac_library_processing_context(context,tiny['root']/'bam-context.json')
    reference={'reference_bundle_path':tiny['pointer']['manifest_path'],
               'reference_bundle_sha256':tiny['pointer']['manifest_sha256']}
    bam_result=prepare_scATAC_bam_fragments(intake['manifest_path'],intake['manifest_sha256'],
        context['manifest_path'],context['manifest_sha256'],**reference,
        source_path=str(bam),source_sha256=c.sha256(bam),source_profile=BAM_PROFILE,
        output_dir=str(tiny['root']/'bam-out'))

    external=tiny['root']/'external.tsv';external.write_text(f'chrTiny\t104\t195\t{BC}\t1\n')
    adopted=import_scATAC_fragments(source_path=str(external),source_sha256=c.sha256(external),
        source_profile=EXTERNAL_PROFILE,**reference,namespace='library_0',output_dir=str(tiny['root']/'external-out'))
    views=[open_verified_fragments(result['manifest_path'],expected_sha256=result['manifest_sha256'],
        runtime=FragmentVerificationRuntime()) for result in (fastq,bam_result,adopted)]
    expected=[FragmentRecord('library_0','chrTiny',104,195,BC,1,None)]
    assert all(v.contract_version=='scatac-fragments.v2' for v in views)
    assert all(list(v.iter_fragments('library_0'))==expected for v in views)
    assert all(v.manifest['reference']==views[0].manifest['reference'] for v in views)
    assert [v.libraries[0].provenance['kind'] for v in views]==[
        'fastq_fragment_production','bam_fragment_production','external_fragment_adoption']
    assert len({v.libraries[0].support_meaning['definition'] for v in views})==3
    assert all(v.verification.artifact_content=='verified' for v in views)
