"""Tiny independently qualified external inputs; real pinned BEDTools mechanics."""
import hashlib
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]))
import pytest
from agent.tools.data import scatac_reference as ref,scatac_qc_reference as qr
from agent.tools.data import external_fragment_manifest as em
from agent.tools.data.scatac_fragment_import import import_scATAC_fragments


@pytest.fixture(autouse=True)
def runtime(monkeypatch):
    monkeypatch.setenv('AGENT_QC_BEDTOOLS','/usr/bin/bedtools')
    monkeypatch.setenv('AGENT_QC_ALLOW_SYNTHETIC','1')
    monkeypatch.delenv('AGENT_QC_RESOURCE_CATALOG',raising=False)


@pytest.fixture
def fixture_factory(tmp_path):
    counter=0
    def make(rows=None, namespace="library_0", features=None, species="human", empty=False):
        nonlocal counter
        root=tmp_path/f'case-{counter}';counter+=1;root.mkdir()
        fa=root/'ref.fa';fai=root/'ref.fa.fai';offset=0;entries=[];sequence=''
        for name in ('chr2','chr1','organelle'):
            sequence+='>'+name+'\n'+'A'*6001+'\n'
            offset+=len(name)+2;entries.append(f'{name}\t6001\t{offset}\t6001\t6002\n');offset+=6002
        fa.write_text(sequence);fai.write_text(''.join(entries));bed=root/'ccre.bed';bed.write_text(''.join('\t'.join(map(str,r))+'\n' for r in (features or [('chr2',3000,3010),('chr2',2950,3051),('chr2',0,6000),('chr2',2999,3001),('chr1',1,2)])))
        reference=ref.build_scatac_reference_bundle(species=species,target_assembly='hg38' if species=='human' else 'mm10',fasta_path=fa,fai_path=fai,ccre_bed_path=bed)
        pointer=ref.publish_scatac_reference_bundle(reference,root/'reference.json')
        tss=root/'transcripts.tsv';tss.write_text('chr2\t3000\t3500\t+\tg1\tt1\tprotein_coding\n'
            'chr2\t2500\t3001\t-\tg2\tt2\tprotein_coding\n'
            'chr2\t3001\t3500\t+\tg3\tt3\tprotein_coding\n'
            'chr1\t3000\t3500\t+\tg4\tt4\tprotein_coding\n')
        qc=qr.build_scatac_qc_reference_bundle(parent_manifest_path=pointer['manifest_path'],parent_manifest_sha256=pointer['manifest_sha256'],
            annotation_path=tss,annotation_source='synthetic',annotation_release='1',
            classifications=tuple(qr.QCContig(n,6001,'mitochondrial' if n=='organelle' else 'primary_nuclear_qc') for n in ('chr2','chr1','organelle')),
            classification_source='explicit-test',output_dir=root/'qc-reference')
        if rows is None:
            rows=[('chr2',1000,1001,'A',99),('chr2',2950,3000,'A',2),('chr2',3000,3001,'A',200),
                  ('chr2',1000,1001,'B',3),('chr2',3000,3011,'B',7),
                  ('chr1',1000,1001,'Z',400)]
        source=root/'source.bed';source.write_text(''.join('\t'.join(map(str,r))+'\n' for r in rows))
        import_args=dict(source_path=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_profile=em.PROFILE_ID,
            reference_bundle_path=pointer['manifest_path'],reference_bundle_sha256=pointer['manifest_sha256'],namespace=namespace,
            source_selection='unknown',output_dir=str(root/'adoption'))
        fragments=import_scATAC_fragments(**import_args)
        manifest=root/'qc-reference/manifest.json'
        args=dict(fragments_manifest_path=fragments['manifest_path'],fragments_manifest_sha256=fragments['manifest_sha256'],
            qc_reference_manifest_path=str(manifest),qc_reference_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),output_dir=str(root/'qc-output'))
        from agent.tools.data.scatac_barcode_qc import compute_scATAC_qc
        from agent.tools.data.scatac_cell_selection import select_scATAC_cells
        q = compute_scATAC_qc(**args)
        selected = select_scATAC_cells(barcode_qc_manifest_path=q['manifest_path'],
            barcode_qc_manifest_sha256=q['manifest_sha256'], output_dir=str(root/'selection'),
            min_qc_fragment_records=100000 if empty else 0, min_tss_enrichment='0')
        return dict(fragments_manifest_path=fragments['manifest_path'],fragments_manifest_sha256=fragments['manifest_sha256'],
            selection_manifest_path=selected['manifest_path'],selection_manifest_sha256=selected['manifest_sha256'],
            reference_manifest_path=pointer['manifest_path'],reference_manifest_sha256=pointer['manifest_sha256'],
            output_dir=str(root/'matrix'),bedtools_path='/usr/bin/bedtools')
    return make
