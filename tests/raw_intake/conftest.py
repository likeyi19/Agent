"""Temporary synthetic sequencing fixtures for public M10.4 boundaries."""
from pathlib import Path

import pytest


@pytest.fixture
def raw_factory(tmp_path):
    def make(kind='fastq', *, species='human', assembly=None, malformed=False, root='raw', tags=True):
        folder = tmp_path / root
        folder.mkdir(exist_ok=True)
        if kind == 'fastq':
            for role in ('R1', 'R2', 'R3'):
                (folder / f'sample_S1_L001_{role}_001.fastq').write_bytes(
                    b'broken\n' if malformed else b'@read\nACGT\n+\nIIII\n')
        else:
            pysam = pytest.importorskip('pysam')
            path = folder / 'input.bam'
            if malformed:
                path.write_bytes(b'not BAM')
            else:
                header = {'HD': {'VN': '1.6', 'SO': 'coordinate'}, 'SQ': [{
                    'SN': 'chr1', 'LN': 10000, 'SP': species,
                    'AS': assembly or ('mm10' if species == 'mouse' else 'hg38')}]}
                with pysam.AlignmentFile(str(path), 'wb', header=header) as out:
                    read = pysam.AlignedSegment(out.header)
                    read.query_name = 'PRIVATE_READ'
                    read.query_sequence = 'ACGT'
                    read.query_qualities = pysam.qualitystring_to_array('IIII')
                    read.reference_id = 0
                    read.reference_start = 1
                    read.cigarstring = '4M'
                    if tags:
                        read.set_tag('CB', 'PRIVATE_BARCODE-1', value_type='Z')
                    out.write(read)
        return {'raw_input_paths': str(folder), 'output_dir': str(tmp_path / ('out-' + root)),
                'species': species, 'raw_assay': 'TENX_ATAC'}
    return make
