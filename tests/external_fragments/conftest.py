"""Tiny external fragments and real local packaging; never align or use BAM tools."""
import gzip
import hashlib
from pathlib import Path
import subprocess

import pytest

from agent.tools.data import scatac_reference as ref, external_fragment_manifest as m
from agent.tools.data import _external_fragment_io as io, scatac_fragments_v2_verifier as v2verify


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def packaging_only(monkeypatch):
    # Commands are real; qualification has its own explicit test.
    monkeypatch.setattr(io, 'verify_packaging', lambda _: None)
    monkeypatch.setattr(v2verify, 'verify_packaging', lambda _: None)
    from agent.tools.data import _chromap
    monkeypatch.setattr(_chromap, 'identify_backend', lambda *a, **k: pytest.fail('External adoption probed an aligner'))
    monkeypatch.delenv('AGENT_CHROMAP_BIN', raising=False)
    monkeypatch.delenv('AGENT_CHROMAP_INDEX_ROOT', raising=False)


@pytest.fixture
def source_factory(tmp_path):
    counter = 0
    def make(*, encoding='plain', strand=False, rows=None, headers=b'# opaque sample provenance\n',
             species='human', indexed=False, selection='unknown'):
        nonlocal counter
        root = tmp_path / f'case-{counter}'; counter += 1; root.mkdir()
        fasta = root / 'reference.fa'; fasta.write_text('>chr2\n' + 'A'*1000 + '\n>chr1\n' + 'C'*1000 + '\n')
        fai = root / 'reference.fa.fai'; fai.write_text('chr2\t1000\t6\t1000\t1001\nchr1\t1000\t1013\t1000\t1001\n')
        bed = root / 'ccre.bed'; bed.write_text('chr2\t0\t10\nchr1\t0\t10\n')
        bundle = ref.build_scatac_reference_bundle(species=species, target_assembly='hg38' if species == 'human' else 'mm10',
            fasta_path=fasta, fai_path=fai, ccre_bed_path=bed)
        reference = ref.publish_scatac_reference_bundle(bundle, root / 'reference.json')
        if rows is None:
            rows = [b'chr2\t4\t95\taCgT-1\t301' + (b'\t+' if strand else b'') + b'\n',
                    b'chr1\t0\t1000\tBC-2\t256' + (b'\t-' if strand else b'') + b'\n']
        raw = headers + b''.join(rows)
        path = root / 'input.misleading.bam'
        if encoding == 'plain': path.write_bytes(raw)
        elif encoding == 'gzip': path.write_bytes(gzip.compress(raw, mtime=0))
        else:
            path.write_bytes(subprocess.run(['/usr/bin/bgzip', '-c'], input=raw, capture_output=True, check=True).stdout)
        args = dict(source_path=str(path), source_sha256=sha(path), source_profile=m.PROFILE_ID,
            reference_bundle_path=reference['manifest_path'], reference_bundle_sha256=reference['manifest_sha256'],
            namespace='explicit_library', output_dir=str(root / 'output'), source_selection=selection)
        if indexed:
            subprocess.run(['/usr/bin/tabix', '-p', 'bed', str(path)], capture_output=True, check=True)
            index = root / 'explicit-source-index'; Path(str(path) + '.tbi').rename(index)
            args.update(source_index_path=str(index), source_index_sha256=sha(index))
        return args
    return make
