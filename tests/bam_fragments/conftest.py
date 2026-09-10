"""Tiny real BAMs and packaging, no network/GPU/aligner/biological inputs."""
from pathlib import Path
import hashlib
import pytest

from agent.tools.data import scatac_reference as ref, scatac_library_context as lc
from agent.tools.data import raw_scatac_manifest as raw, bam_fragment_manifest as m
from agent.tools.data.raw_scatac import inspect_raw_scATAC


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pair(name='template', **kwargs):
    a=dict(name=name,flag=99,rid=0,pos=10,cigar='20M',mrid=0,mpos=60,tlen=70,mapq=30,cb='aCgT-1')
    b=dict(name=name,flag=147,rid=0,pos=60,cigar='20M',mrid=0,mpos=10,tlen=-70,mapq=30,cb='aCgT-1')
    a.update(kwargs.get('a',{})); b.update(kwargs.get('b',{})); return [a,b]


@pytest.fixture
def bam_factory(tmp_path):
    ps=pytest.importorskip('pysam'); counter=0
    def make(records=None, *, species='human', namespace='explicit_library', sq=None, rg=None, corrected=True,
             reference_names=('chr2','chr1')):
        nonlocal counter
        root=tmp_path/f'case-{counter}';counter+=1;root.mkdir()
        first,second=reference_names
        fasta=root/'ref.fa';fasta.write_text('>'+first+'\n'+'A'*1000+'\n>'+second+'\n'+'C'*1000+'\n')
        ps.faidx(str(fasta));bed=root/'ccre.bed';bed.write_text(first+'\t0\t10\n'+second+'\t0\t10\n')
        assembly='hg38' if species=='human' else 'mm10'
        bundle=ref.build_scatac_reference_bundle(species=species,target_assembly=assembly,fasta_path=fasta,
            fai_path=Path(str(fasta)+'.fai'),ccre_bed_path=bed)
        reference=ref.publish_scatac_reference_bundle(bundle,root/'reference.json')
        header={'HD':{'VN':'1.6','SO':'unsorted'},'SQ':sq or [{'SN':first,'LN':1000},{'SN':second,'LN':1000}]}
        if rg is not None: header['RG']=rg
        path=root/'input.bam'
        with ps.AlignmentFile(str(path),'wb',header=header) as output:
            for spec in pair() if records is None else records:
                r=ps.AlignedSegment(output.header);r.query_name=spec.get('name','template');r.flag=spec.get('flag',99)
                r.reference_id=spec.get('rid',0);r.reference_start=spec.get('pos',10)
                r.cigarstring=spec.get('cigar','20M');r.mapping_quality=spec.get('mapq',30)
                r.next_reference_id=spec.get('mrid',0);r.next_reference_start=spec.get('mpos',60);r.template_length=spec.get('tlen',0)
                length=sum(n for op,n in r.cigartuples or [] if op in (0,1,4,7,8))
                r.query_sequence='A'*length if length else None
                if spec.get('cb','aCgT-1') is not None: r.set_tag('CB',spec.get('cb','aCgT-1'))
                if 'rg' in spec:r.set_tag('RG',spec['rg'])
                for k,v,t in spec.get('tags',[]):r.set_tag(k,v,value_type=t,replace=False)
                output.write(r)
        intake=inspect_raw_scATAC(str(path),str(root/'intake'),species=species,raw_assay='SCATAC',source_genome_assembly=assembly)
        _,manifest,_=raw.load_raw_intake_manifest(intake['manifest_path'])
        context=lc.build_scatac_library_processing_context(intake_manifest_path=intake['manifest_path'],
            expected_intake_sha256=intake['manifest_sha256'],selection_mode=lc.SelectionMode.ALL_GROUPS,libraries=(lc.LibraryDeclaration(namespace=namespace,
            group_ids=(manifest.groups[0].id,),membership_basis=lc.MembershipBasis.SINGLE_GROUP,
            barcode_interpretation=lc.BarcodeInterpretation.CORRECTED_IDENTIFIER if corrected else lc.BarcodeInterpretation.RAW_SEQUENCE,
            correction_policy=lc.CorrectionPolicy.ALREADY_CORRECTED if corrected else lc.CorrectionPolicy.UNQUALIFIED_RAW_BAM),))
        pointer=lc.publish_scatac_library_processing_context(context,root/'context.json')
        return dict(intake_manifest_path=intake['manifest_path'],intake_manifest_sha256=intake['manifest_sha256'],
            library_context_path=pointer['manifest_path'],library_context_sha256=pointer['manifest_sha256'],
            reference_bundle_path=reference['manifest_path'],reference_bundle_sha256=reference['manifest_sha256'],
            source_path=str(path),source_sha256=sha(path),source_profile=m.PROFILE_ID,output_dir=str(root/'out'))
    return make
