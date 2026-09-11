import json
from pathlib import Path

import anndata as ad

from barcode_qc.test_producer_neutrality import fastq_fixture, bam_fixture, compute
from agent.tools.data.scatac_fragment_import import import_scATAC_fragments
from agent.tools.data.external_fragment_manifest import PROFILE_ID
from agent.tools.data.scatac_cell_selection import select_scATAC_cells
from agent.tools.data.scatac_cell_by_ccre import build_cell_by_ccre
from agent.tools.data._matrix_bedtools import file_sha


def matrix(root, tiny, fragments, i, empty=False):
    compute(root,fragments,i)
    (qp,)=(root/f'qc-{i}').glob('*/manifest.json')
    selected=select_scATAC_cells(barcode_qc_manifest_path=str(qp),barcode_qc_manifest_sha256=file_sha(qp),
        output_dir=str(root/f'selection-{i}'),min_qc_fragment_records=999 if empty else 0,min_tss_enrichment='0')
    return build_cell_by_ccre(fragments_manifest_path=fragments['manifest_path'],fragments_manifest_sha256=fragments['manifest_sha256'],
        selection_manifest_path=selected['manifest_path'],selection_manifest_sha256=selected['manifest_sha256'],
        reference_manifest_path=tiny['pointer']['manifest_path'],reference_manifest_sha256=tiny['pointer']['manifest_sha256'],
        output_dir=str(root/f'matrix-{i}'),bedtools_path='/usr/bin/bedtools')


def test_three_producer_complete_chains(tmp_path,monkeypatch):
    features=[('chrTiny',3000,3100),('chrTiny',2999,3102),('chrTiny',3001,3002),('chrTiny',5,6)]
    zero='TTTTTTTTTTTTTTTT'
    tiny,fq,bc=fastq_fixture(tmp_path,monkeypatch,ccre_rows=features,zero_barcode=zero)
    bam=bam_fixture(tiny,bc,zero_barcode=zero)
    source=tmp_path/'external.tsv';source.write_text(f'chrTiny\t1000\t1105\t{bc}\t200\nchrTiny\t1100\t1205\t{zero}\t7\nchrTiny\t3000\t3101\t{bc}\t900\n')
    external=import_scATAC_fragments(source_path=str(source),source_sha256=file_sha(source),source_profile=PROFILE_ID,
        reference_bundle_path=tiny['pointer']['manifest_path'],reference_bundle_sha256=tiny['pointer']['manifest_sha256'],
        namespace='library_0',output_dir=str(tmp_path/'external'))
    results=[matrix(tmp_path,tiny,r,i) for i,r in enumerate((fq,bam,external))]
    manifests=[json.loads(Path(r['manifest_path']).read_bytes()) for r in results]
    assert len({r['logical_matrix_sha256'] for r in results})==1
    assert len({r['ordered_selected_sha256'] for r in manifests})==1
    assert len({r['ordered_feature_sha256'] for r in manifests})==1
    assert all(r['diagnostic']==results[0]['diagnostic'] for r in results)
    assert results[0]['diagnostic']['n_selected_fragment_records_total']==3
    assert results[0]['diagnostic']['n_selected_fragment_records_overlapping_any_ccre']==1
    for result in results:
        x=ad.read_h5ad(Path(result['manifest_path']).parent/'matrix.h5ad')
        assert x.X.toarray().tolist()==[[1,1,1,0],[0,0,0,0]] and result['zero_row_count']==1
    assert [json.loads(Path(r['manifest_path']).read_bytes())['libraries'][0]['sum_support'] for r in (fq,bam,external)]==[6,3,1107]
    # Same three authorities also converge with no selected cells.
    empty=[matrix(tmp_path,tiny,r,i+3,empty=True) for i,r in enumerate((fq,bam,external))]
    assert len({r['logical_matrix_sha256'] for r in empty})==1


def test_namespace_collision(tmp_path,monkeypatch):
    tiny,fq,bc=fastq_fixture(tmp_path,monkeypatch,namespaces=2,ccre_rows=[('chrTiny',3000,3100)])
    result=matrix(tmp_path,tiny,fq,0)
    x=ad.read_h5ad(Path(result['manifest_path']).parent/'matrix.h5ad')
    assert x.obs['namespace'].tolist()==['library_0','library_1']
    assert x.obs['barcode_identifier'].tolist()==[bc,bc]
    assert x.X.toarray().tolist()==[[1],[1]]


def test_strand_labels_do_not_weight_records(fixture_factory):
    base=[('chr2',1000,1001,'A',1),('chr2',3000,3001,'A',2)]
    results=[build_cell_by_ccre(**fixture_factory(rows=[(*r,strand) for r in base])) for strand in ('+','-','.')]
    assert len({r['logical_matrix_sha256'] for r in results})==1


def test_authoritative_order_is_not_rendered_lexical_order(fixture_factory):
    rows=[('chr2',1000,1001,'aL',1),('chr2',1000,1001,'aM',1),('chr2',3000,3001,'aL',1)]
    result=build_cell_by_ccre(**fixture_factory(rows=rows))
    x=ad.read_h5ad(Path(result['manifest_path']).parent/'matrix.h5ad')
    assert x.obs['barcode_identifier'].tolist()==['aL','aM']
    assert x.obs_names.tolist()!=sorted(x.obs_names)
    assert x.X.toarray().tolist()==[[1,1,2,1,0],[0,0,1,0,0]]
