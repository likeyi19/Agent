import gzip
import hashlib
import json
from pathlib import Path

import pytest

from agent.tools.data import external_fragment_manifest as m, scatac_fragments_v2 as v2
from agent.tools.data import external_fragments as production
from agent.tools.data import scatac_fragment_import as public
from agent.tools.data.scatac_fragment_reader import open_verified_fragments
from agent.tools.data._fragments_common import MAX_SUPPORT, verify_packaging
from agent.tools.data.scatac_fragments_v2_verifier import FragmentVerificationRuntime


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.mark.parametrize('encoding', ['plain', 'gzip', 'bgzf'])
@pytest.mark.parametrize('strand', [False, True])
@pytest.mark.parametrize('species', ['human', 'mouse'])
def test_adoption_encodings_layouts_and_references(source_factory, encoding, strand, species):
    args = source_factory(encoding=encoding, strand=strand, species=species)
    result = public.import_scATAC_fragments(**args)
    checked = public.verify_public_result(args, result)
    view = open_verified_fragments(result['manifest_path'], expected_sha256=result['manifest_sha256'], runtime=FragmentVerificationRuntime())
    rows = list(view.iter_fragments(args['namespace']))
    assert [(r.start, r.end, r.barcode_identifier, r.support) for r in rows] == [(4, 95, 'aCgT-1', 301), (0, 1000, 'BC-2', 256)]
    assert [r.strand for r in rows] == (['+', '-'] if strand else [None, None])
    assert result['source_encoding'] == encoding and result['species'] == species
    assert result['total_support'] == 557 and result['n_libraries'] == 1
    assert checked.source_content == checked.canonicalization_conservation == 'verified'
    assert checked.producer_history == 'declared_not_reconstructed'
    p = view.libraries[0].provenance
    assert p['support'] == m.SUPPORT and p['kind'] == 'external_fragment_adoption'
    assert p['source_selection'] == 'unspecified'
    record = m.load_adoption_record(p['producer_record']['path'], p['producer_record']['sha256'])
    assert record['source']['header_records'] == 1 and record['source_index'] is None
    assert 'opaque sample' not in json.dumps(record)


def test_real_packaging_qualification():
    verify_packaging(FragmentVerificationRuntime())


def test_unsorted_strands_exact_support_and_determinism(source_factory):
    raw = [f'chr2\t4\t95\taCgT-1\t{n}\t{s}\n'.encode() for n, s in ((MAX_SUPPORT, '.'), (256, '-'), (255, '+'))]
    raw.insert(0, b'chr1\t1\t2\tBC-2\t1\t+\n')
    args = source_factory(rows=raw)
    result = public.import_scATAC_fragments(**args)
    view = open_verified_fragments(result['manifest_path'], expected_sha256=result['manifest_sha256'], runtime=FragmentVerificationRuntime())
    rows = list(view.iter_fragments(args['namespace']))
    assert [r.support for r in rows] == [255, 256, MAX_SUPPORT, 1]
    assert [r.strand for r in rows] == ['+', '-', '.', '+']
    assert result['total_support'] == MAX_SUPPORT + 512
    other = public.import_scATAC_fragments(**(args | {'output_dir': str(Path(args['output_dir']).with_name('repeat'))}))
    second = v2.load_fragments_manifest_v2(other['manifest_path'], expected_sha256=other['manifest_sha256'])
    first = view.manifest
    assert first['fragments_identity_sha256'] == second['fragments_identity_sha256']
    for kind in ('bgzf', 'tabix'):
        assert first['libraries'][0][kind] == second['libraries'][0][kind]


@pytest.mark.parametrize('selection', ['full_export', 'subset_export', 'unknown'])
def test_explicit_coverage(source_factory, selection):
    args = source_factory(selection=selection); result = public.import_scATAC_fragments(**args)
    value = v2.load_fragments_manifest_v2(result['manifest_path'], expected_sha256=result['manifest_sha256'])
    assert value['libraries'][0]['provenance']['source_selection'] == m.SELECTION[selection]


@pytest.mark.parametrize('indexed', [False, True])
def test_source_index_optional_and_explicit(source_factory, indexed):
    args = source_factory(encoding='bgzf', indexed=indexed)
    Path(args['source_path'] + '.tbi').write_bytes(b'not the selected index')
    result = public.import_scATAC_fragments(**args)
    public.verify_public_result(args, result)


@pytest.mark.parametrize('change', ['support_balanced', 'strand', 'coordinate'])
def test_conservation_is_stronger_than_aggregates_and_output_hashes(source_factory, monkeypatch, change):
    args = source_factory(strand=True); original = production.canonicalize
    def internally_consistent_fault(*a, **k):
        path, summary = original(*a, **k)
        rows = [line.decode().rstrip('\n').split('\t') for line in path.read_bytes().splitlines(True)]
        if change == 'support_balanced':
            rows[0][4] = '300'; rows[1][4] = '257'
        elif change == 'strand': rows[0][5] = '-'
        else: rows[0][1] = '5'
        path.write_bytes(b''.join(('\t'.join(row) + '\n').encode() for row in rows))
        summary['canonical_record_stream_sha256'] = sha(path)
        summary['max_support'] = max(int(row[4]) for row in rows)
        return path, summary
    monkeypatch.setattr(production, 'canonicalize', internally_consistent_fault)
    with pytest.raises(m.ExternalFragmentsError, match='CONSERVATION_MISMATCH'):
        public.import_scATAC_fragments(**args)


@pytest.mark.parametrize('change', ['drop', 'duplicate'])
def test_independent_source_cardinality_reconstruction(source_factory, monkeypatch, change):
    args = source_factory(); original = production.scan_source
    def false_source_facts(*a, **k):
        facts = original(*a, **k); ranked = Path(a[3]); rows = ranked.read_bytes().splitlines(True)
        if change == 'drop': rows.pop()
        else:
            # Invent another unique coordinate to evade output duplicate-key checks.
            fields = rows[-1].decode().rstrip('\n').split('\t'); fields[1] = '1'
            rows.append(('\t'.join(fields) + '\n').encode())
        ranked.write_bytes(b''.join(rows))
        facts['n_records'] = len(rows)
        facts['sum_support'] = sum(int(row.split(b'\t')[-1]) for row in rows)
        return facts
    monkeypatch.setattr(production, 'scan_source', false_source_facts)
    with pytest.raises(m.ExternalFragmentsError, match='SOURCE_MISMATCH'):
        public.import_scATAC_fragments(**args)


@pytest.mark.parametrize('mutation', ['extra', 'profile', 'namespace', 'selection', 'source_shape'])
def test_closed_adoption_record(source_factory, mutation):
    args = source_factory(); result = public.import_scATAC_fragments(**args)
    path = Path(result['manifest_path']).parent / 'adoption.json'
    record = json.loads(path.read_bytes())
    if mutation == 'extra': record['arbitrary_provenance'] = {}
    elif mutation == 'profile': record['profile_sha256'] = '0'*64
    elif mutation == 'namespace': record['namespace'] = ''
    elif mutation == 'selection': record['source_selection'] = 'infer'
    else: record['source']['extra'] = 'bad'
    with pytest.raises(ValueError): m.validate_adoption_record(record)



@pytest.mark.parametrize('mutation', ['corrupt', 'wrong', 'plain', 'gzip', 'hash'])
def test_bad_source_indexes(source_factory, mutation):
    args = source_factory(encoding='bgzf', indexed=True)
    if mutation in ('plain', 'gzip'):
        raw = gzip.decompress(Path(args['source_path']).read_bytes())
        Path(args['source_path']).write_bytes(raw if mutation == 'plain' else gzip.compress(raw))
        args['source_sha256'] = sha(args['source_path'])
    elif mutation == 'corrupt':
        Path(args['source_index_path']).write_bytes(b'bad'); args['source_index_sha256'] = sha(args['source_index_path'])
    elif mutation == 'hash': args['source_index_sha256'] = '0'*64
    else:
        other = source_factory(encoding='bgzf', indexed=True, rows=[b'chr1\t10\t20\tB\t1\n'])
        args['source_index_path'] = other['source_index_path']; args['source_index_sha256'] = other['source_index_sha256']
    with pytest.raises(ValueError): public.import_scATAC_fragments(**args)


@pytest.mark.parametrize('row', [
    b'chr2\t0\t1\tB\t0\n', b'chr2\t0\t1\tB\t-1\n', b'chr2\t0\t1\tB\t1.5\n',
    f'chr2\t0\t1\tB\t{MAX_SUPPORT+1}\n'.encode(), b'chr2\t00\t1\tB\t1\n',
    b'chr2\t0\t1\tB\t1\t?\n', b'chr2\t0\t1\t\t1\n', b'chr2\t0\t1\tB B\t1\n',
    b'chrX\t0\t1\tB\t1\n', b'chr2\t-1\t1\tB\t1\n', b'chr2\t0\t1001\tB\t1\n',
    b'chr2\t1\t1\tB\t1\n', b'\n', b'chr2\t0\t1\tB\t1', b'chr2\t0\t1\tB\t1\r\n',
])
def test_invalid_data(source_factory, row):
    with pytest.raises(ValueError): public.import_scATAC_fragments(**source_factory(rows=[row]))


@pytest.mark.parametrize('rows', [
    [b'chr2\t0\t1\tB\t1\n', b'chr2\t0\t2\tB\t1\t+\n'],
    [b'chr2\t0\t1\tB\t1\n', b'# too late\n'],
    [b'chr2\t0\t1\tB\t1\n', b'chr2\t0\t1\tB\t5\n'],
    [b'chr2\t0\t1\tB\t1\t+\n'] * 2,
    [],
])
def test_layout_header_duplicate_and_empty_failures(source_factory, rows):
    with pytest.raises(ValueError): public.import_scATAC_fragments(**source_factory(rows=rows))


@pytest.mark.parametrize('headers', [b'#' + b'a'*m.MAX_LINE + b'\n', b'#\n'*(m.MAX_HEADERS+1),
    (b'#' + b'a'*2046 + b'\n') * 513])
def test_header_limits(source_factory, headers):
    with pytest.raises(ValueError): public.import_scATAC_fragments(**source_factory(headers=headers))


@pytest.mark.parametrize('encoding', ['gzip', 'bgzf'])
@pytest.mark.parametrize('mutation', ['truncate', 'crc'])
def test_corrupt_compressed_sources(source_factory, encoding, mutation):
    args = source_factory(encoding=encoding); path = Path(args['source_path']); raw = path.read_bytes()
    if mutation == 'truncate': raw = raw[:-6]
    else: raw = raw[:20] + bytes([raw[20] ^ 1]) + raw[21:]
    path.write_bytes(raw); args['source_sha256'] = sha(path)
    with pytest.raises(ValueError): public.import_scATAC_fragments(**args)


@pytest.mark.parametrize('change', ['hash', 'profile', 'namespace', 'reference', 'index_pair'])
def test_invalid_explicit_bindings(source_factory, change):
    args = source_factory()
    if change == 'hash': args['source_sha256'] = '0'*64
    elif change == 'profile': args['source_profile'] = 'infer-cellranger'
    elif change == 'namespace': args['namespace'] = '../escape'
    elif change == 'reference': args['reference_bundle_sha256'] = '0'*64
    else: args['source_index_path'] = args['source_path']
    with pytest.raises(ValueError): public.import_scATAC_fragments(**args)


def test_checked_aggregate_overflow(source_factory, monkeypatch):
    args = source_factory()
    monkeypatch.setattr(production, 'MAX_TOTAL', 400)
    with pytest.raises(m.ExternalFragmentsError, match='OVERFLOW'): public.import_scATAC_fragments(**args)


def test_source_mutation_during_scan(source_factory, monkeypatch):
    args = source_factory(); original = production.scan_source
    def changing(*a, **k):
        value = original(*a, **k)
        Path(args['source_path']).write_bytes(Path(args['source_path']).read_bytes() + b'# changed\n')
        return value
    monkeypatch.setattr(production, 'scan_source', changing)
    with pytest.raises(ValueError): public.import_scATAC_fragments(**args)


@pytest.mark.parametrize('change', ['source', 'bgzf', 'tabix', 'manifest', 'profile', 'record'])
def test_fresh_corruption(source_factory, change):
    args = source_factory(); result = public.import_scATAC_fragments(**args)
    manifest_path = Path(result['manifest_path']); value = json.loads(manifest_path.read_bytes()); e = value['libraries'][0]
    path = {'source': Path(args['source_path']), 'manifest': manifest_path,
        'profile': manifest_path.parent / 'profile.json', 'record': manifest_path.parent / 'adoption.json'}.get(change)
    if path is None: path = manifest_path.parent / e[change]['path']
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError): public.verify_public_result(args, result)


@pytest.mark.parametrize('change', ['drop', 'duplicate', 'support', 'strand', 'coordinate'])
def test_independent_verifier_rejects_production_faults(source_factory, monkeypatch, change):
    args = source_factory(strand=True); original = production.canonicalize
    def defective(*a, **k):
        path, summary = original(*a, **k); rows = path.read_bytes().splitlines(True)
        if change == 'drop': rows.pop()
        elif change == 'duplicate': rows.append(rows[-1])
        else:
            fields = rows[0].decode().rstrip('\n').split('\t')
            fields[{'support': 4, 'strand': 5, 'coordinate': 1}[change]] = {'support': '302', 'strand': '-', 'coordinate': '5'}[change]
            rows[0] = ('\t'.join(fields) + '\n').encode()
        path.write_bytes(b''.join(rows))
        summary['canonical_record_stream_sha256'] = sha(path)
        return path, summary
    monkeypatch.setattr(production, 'canonicalize', defective)
    with pytest.raises(ValueError): public.import_scATAC_fragments(**args)
    assert not list(Path(args['output_dir']).glob('external-fragments-*'))


def test_verifier_does_not_use_production_parser(source_factory, monkeypatch):
    args = source_factory(); result = public.import_scATAC_fragments(**args)
    def forbidden(*a, **k): pytest.fail('Production parser used as verifier')
    for name in ('scan_source', 'parse_record', 'canonicalize'):
        monkeypatch.setattr(production, name, forbidden)
    public.verify_public_result(args, result)
