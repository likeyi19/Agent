import hashlib
import gzip
import subprocess
from pathlib import Path

import pytest

from agent.tools.data import regulatory_feature_reference as reference
from agent.tools.data import explicit_cells
from agent.tools.data.neutral_fragment_import import import_external_fragments


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def factory(tmp_path, monkeypatch):
    monkeypatch.setenv('AGENT_MATRIX_BEDTOOLS', '/usr/bin/bedtools')
    counter = 0
    def make(cells=(('lib', 'B'), ('lib', 'A'), ('lib', 'Z')), support=1000, strand=False,
             extra_rows=(), encoding='plain', indexed=False):
        nonlocal counter
        root = tmp_path / str(counter); counter += 1; root.mkdir()
        fa = root / 'genome.fa'; fa.write_text('>chr1\n' + 'A'*100 + '\n')
        fai = root / 'genome.fa.fai'; fai.write_text('chr1\t100\t6\t100\t101\n')
        bed = root / 'features.bed'; bed.write_text('chr1\t30\t40\nchr1\t0\t20\nchr1\t10\t25\nchr1\t90\t100\n')
        resources = dict(species={'scientific_name': 'Danio rerio', 'taxonomy_id': 7955},
            target_assembly='synthetic-M14.6', fasta_path=fa, fai_path=fai,
            feature_bed_path=bed, feature_category='peak_set')
        ref = reference.publish_regulatory_feature_reference(reference.build_regulatory_feature_reference(**resources), root / 'reference.json')
        source = root / 'source.tsv'
        rows = [(0, 10, 'A'), (10, 15, 'A'), (20, 30, 'A'), (30, 40, 'B'),
                (10, 20, 'B'), (50, 60, 'B'), (60, 70, 'Z'), (0, 99, 'unselected')]
        rows.extend(extra_rows)
        source.write_text(''.join(f'chr1\t{a}\t{b}\t{bc}\t{support}' + ('\t+' if strand else '') + '\n' for a, b, bc in rows))
        if encoding == 'gzip': source.write_bytes(gzip.compress(source.read_bytes(), mtime=0))
        if encoding == 'bgzf':
            raw = b''.join(sorted(source.read_bytes().splitlines(keepends=True), key=lambda r: int(r.split(b'\t')[1])))
            source.write_bytes(subprocess.run(['/usr/bin/bgzip','-c'], input=raw, capture_output=True, check=True).stdout)
        fragment_args = dict(source_path=str(source), source_sha256=sha(source), source_profile='10x-atac-fragments.v1',
            reference_bundle_path=ref['manifest_path'], reference_bundle_sha256=ref['manifest_sha256'],
            namespace='lib', output_dir=str(root / 'fragments'))
        if indexed:
            subprocess.run(['/usr/bin/tabix','-p','bed',str(source)], check=True, capture_output=True)
            index = Path(str(source)+'.tbi')
            fragment_args.update(source_index_path=str(index), source_index_sha256=sha(index))
        fragments = import_external_fragments(**fragment_args)
        ordered = explicit_cells.publish_explicit_cells(cells=cells, declaration='Caller-specified technical fixture; no QC assertion.', output_dir=root / 'cells')
        args = dict(fragments_manifest_path=fragments['manifest_path'], fragments_manifest_sha256=fragments['manifest_sha256'],
            explicit_cells_manifest_path=ordered['manifest_path'], explicit_cells_manifest_sha256=ordered['manifest_sha256'],
            reference_manifest_path=ref['manifest_path'], reference_manifest_sha256=ref['manifest_sha256'], output_dir=str(root / 'output'))
        return args, resources, fragment_args
    return make
