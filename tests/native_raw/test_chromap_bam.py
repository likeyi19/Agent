"""Small native FASTQ and BAM executions; fixtures are explicitly simulated."""
import os
from pathlib import Path
import random
import pytest
import pysam
import anndata as ad

from agent.tools.data import chromap_native
from agent.tools.analysis.native_matrix import SNAPATAC2

pytestmark=pytest.mark.skipif(os.environ.get('RUN_RAW_NATIVE_QUALIFICATION')!='1',reason='Explicit native runtime qualification')


def test_native_chromap_lanes_correction_shift_and_support(tmp_path):
    assert os.environ.get('AGENT_CHROMAP_BIN')
    rng=random.Random(11);dna=''.join(rng.choice('ACGT') for _ in range(10000))
    reference=tmp_path/'reference.fa';reference.write_text('>chr1\n'+dna+'\n')
    index=chromap_native.execute(dict(reference_path=str(reference),function_name='index',output_dir=str(tmp_path/'index')))
    barcode='ACGTACGTACGTACGT';white=tmp_path/'whitelist';white.write_text(barcode+'\n')
    paths={key:[] for key in ('read1_path','read2_path','barcode_path')}
    for lane in range(2):
        for key,seq in [('read1_path',dna[1000:1050]),('read2_path',dna[1100:1150].translate(str.maketrans('ACGT','TGCA'))[::-1]),('barcode_path',barcode)]:
            p=tmp_path/(key+str(lane));paths[key].append(str(p))
            rows=[]
            for i in range(150):
                actual='T'+seq[1:] if key=='barcode_path' and lane==1 and i==0 else seq
                rows.append(f'@lane{lane}_{i}\n{actual}\n+\n'+('I'*len(actual))+'\n')
            p.write_text(''.join(rows))
    result=chromap_native.execute(dict(reference_path=str(reference),index_path=index['index_path'],whitelist_path=str(white),**paths,
        function_name='map',kwargs={'preset':'atac'},output_dir=str(tmp_path/'map')))
    fields=Path(result['fragments_path']).read_text().strip().split('\t')
    assert fields[:4]==['chr1','1004','1145',barcode]
    # The actual installed support-preserving native build is qualified here.
    assert result['software_versions']['support_representation']=='qualified uint64 patch'
    assert int(fields[4])==300
    sizes=tmp_path/'sizes';sizes.write_text('chr1\t10000\n')
    imported=SNAPATAC2.execute(dict(input_path=result['fragments_path'],function_name='pp.import_fragments',chrom_sizes_path=str(sizes),
        kwargs=dict(min_num_fragments=1,sorted_by_barcode=False),output_dir=str(tmp_path/'import')))
    assert ad.read_h5ad(imported['matrix_path']).obs.n_fragment.tolist()==[1]


def test_native_bam_filters_duplicate_flags_and_mapq_and_preserves_barcodes(tmp_path):
    bam=tmp_path/'paired.bam'
    with pysam.AlignmentFile(bam,'wb',header={'HD':{'VN':'1.6'},'SQ':[{'SN':'chr1','LN':1000}]}) as output:
        for q,left,right,cb,dup,mapq in [('a',100,180,'A-1',False,60),('b',100,180,'A-1',True,60),('c',110,180,'B-1',False,20),('d',150,170,'B-1',False,60)]:
            for first in (True,False):
                record=pysam.AlignedSegment();record.query_name=q;record.query_sequence='A'*50
                record.flag=(99 if first else 147)|(1024 if dup else 0);record.reference_id=0
                record.reference_start=left if first else right;record.mapping_quality=mapq;record.cigarstring='50M'
                record.next_reference_id=0;record.next_reference_start=right if first else left
                record.template_length=(right+50-left)*(1 if first else -1);record.set_tag('CB',cb);output.write(record)
    before=bam.read_bytes()
    result=SNAPATAC2.execute(dict(input_path=str(bam),function_name='pp.make_fragment_file',kwargs=dict(barcode_tag='CB',chunk_size=1000),output_dir=str(tmp_path/'out')))
    assert Path(result['fragments_path']).read_text().splitlines()==['chr1\t104\t225\tA-1\t1','chr1\t154\t215\tB-1\t1']
    assert result['matrix_path'] is None and result['n_cells'] is None
    assert bam.read_bytes()==before
