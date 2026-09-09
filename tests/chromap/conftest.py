"""Generated synthetic resources. No biological reference or network required."""
from pathlib import Path
import hashlib
import os
import random
import subprocess
import pytest
from agent.tools.data import _chromap as c
from agent.tools.data import scatac_reference as ref
from agent.tools.data.scatac_library_context import inspect_barcode_whitelist

BC = 'ACGTACGTACGTACGT'

def reverse_complement(seq):
    return seq.translate(str.maketrans('ACGTN', 'TGCAN'))[::-1]

@pytest.fixture
def tiny(tmp_path):
    rng = random.Random(112)
    seq = ''.join(rng.choices('ACGT', k=12000))
    fa = tmp_path / 'ref.fa'; fa.write_text('>chrTiny\n' + seq + '\n')
    fai = tmp_path / 'ref.fa.fai'; fai.write_text(f'chrTiny\t{len(seq)}\t9\t{len(seq)}\t{len(seq)+1}\n')
    bed = tmp_path / 'ccre.bed'; bed.write_text('chrTiny\t100\t200\n')
    bundle = ref.build_scatac_reference_bundle(species='human', target_assembly='hg38',
        fasta_path=fa, fai_path=fai, ccre_bed_path=bed)
    pointer = ref.publish_scatac_reference_bundle(bundle, tmp_path / 'reference.json')
    return dict(root=tmp_path, seq=seq, fa=fa, fai=fai, bed=bed, bundle=bundle, pointer=pointer)

@pytest.fixture
def reads(tiny):
    def make(records, *, name='reads', layout=c.FastqLayout.A, whitelist=(BC,)):
        """records: (start, end, raw barcode, multiplicity[, swap_mates])."""
        p = tiny['root'] / name; p.mkdir()
        roles = {meaning: role for role, meaning in c.LAYOUT_ROLES[layout].items()}
        paths = {role: p / f'Synthetic_S1_L001_{role.value}_001.fastq' for role in roles.values()}
        output = {role: [] for role in paths}
        serial = 0
        for rec in records:
            start, end, bc, count = rec[:4]
            length = min(50, end-start)
            seqs = {c.ReadMeaning.GENOMIC_1: tiny['seq'][start:start+length],
                c.ReadMeaning.GENOMIC_2: reverse_complement(tiny['seq'][end-length:end]),
                c.ReadMeaning.BARCODE: bc, c.ReadMeaning.SAMPLE_INDEX: 'GATTACAA'}
            if len(rec) > 4 and rec[4]:
                seqs[c.ReadMeaning.GENOMIC_1], seqs[c.ReadMeaning.GENOMIC_2] = seqs[c.ReadMeaning.GENOMIC_2], seqs[c.ReadMeaning.GENOMIC_1]
            for _ in range(count):
                for meaning, seq in seqs.items():
                    output[roles[meaning]].append(f'@pair{serial}\n{seq}\n+\n'+ 'I'*len(seq)+'\n')
                serial += 1
        for role, lines in output.items():
            paths[role].write_text(''.join(lines))
        white = p / 'whitelist.txt'; white.write_text('\n'.join(whitelist)+'\n')
        gid = hashlib.sha256(str(p).encode()).hexdigest()
        return c.QualificationGroup(gid, layout, tuple((role, str(path)) for role,path in paths.items())), inspect_barcode_whitelist(white)
    return make

@pytest.fixture(scope='session')
def executables():
    if os.environ.get('RUN_CHROMAP_QUALIFICATION') != '1':
        pytest.skip('Explicit RUN_CHROMAP_QUALIFICATION=1 required')
    import json
    record_path = os.environ.get('AGENT_CHROMAP_QUALIFICATION_RECORD')
    if not record_path:
        pytest.fail('Explicit build identity record required')
    record = json.loads(Path(record_path).read_text())
    if record['upstream_commit'] != c.UPSTREAM_COMMIT or record['patch_sha256'] != c.PATCH_SHA256:
        pytest.fail('Build record has wrong source/patch identity')
    for mode in ('stock', 'candidate'):
        p = Path(record[mode]['path'])
        assert c.sha256(p) == record[mode]['sha256']
        assert subprocess.check_output([str(p),'--version'],stderr=subprocess.STDOUT).decode().strip() == c.UPSTREAM_VERSION
    return record

@pytest.fixture
def run_backend(tiny, executables):
    indexes = {}
    def run(mode, group, white, *, name, threads=1, groups=None, index=None, mutate=None):
        exe = executables[mode]['path']
        if index is None:
            if mode not in indexes:
                idx = tiny['root'] / (mode+'.index')
                r = subprocess.run(c.index_argv(exe,tiny['fa'],idx),capture_output=True)
                assert r.returncode == 0, r.stderr.decode()
                indexes[mode] = idx
            index = indexes[mode]
        output = tiny['root'] / (name+'.bed')
        argv = list(c.mapping_argv(exe,fasta=tiny['fa'],index=index,output=output,
            groups=groups or [group],whitelist=white))
        argv[argv.index('--num-threads')+1] = str(threads)  # test-only audit probe
        if mutate:
            mutate(argv)
        r = subprocess.run(argv,capture_output=True)
        (tiny['root']/(name+'.log')).write_bytes(r.stdout+r.stderr)
        raw = output.read_text() if output.exists() else ''
        rows = [tuple(line.split('\t')) for line in raw.splitlines()]
        return r, sorted(rows), raw
    return run
