"""External source validation and non-biological canonicalization only."""
import hashlib
from pathlib import Path
import re

from . import external_fragment_manifest as m, _external_fragment_io as io
from . import scatac_fragments_v2 as v2
from ._fragments_common import MAX_SUPPORT, MAX_TOTAL, PACKAGING_POLICY, canonical, run_stage


def parse_record(line, dictionary):
    try:
        fields = line[:-1].decode('utf-8').split('\t')
        if len(fields) not in (5, 6):
            m.fail('EXTERNAL_FRAGMENTS_RECORD_INVALID')
        name, left, right, barcode, support = fields[:5]
        strand = fields[5] if len(fields) == 6 else ''
        if (name not in dictionary or any(re.fullmatch('0|[1-9][0-9]{0,18}', n) is None for n in (left, right))
                or re.fullmatch('[!-~]{1,256}', barcode) is None
                or re.fullmatch('[1-9][0-9]{0,19}', support) is None
                or (len(fields) == 6 and strand not in ('+', '-', '.'))):
            m.fail('EXTERNAL_FRAGMENTS_RECORD_INVALID')
        left, right, support = int(left), int(right), int(support)
        if not 0 <= left < right <= dictionary[name][1] or not 1 <= support <= MAX_SUPPORT:
            m.fail('EXTERNAL_FRAGMENTS_RECORD_INVALID')
        return dictionary[name][0], left, right, barcode, strand, support, len(fields)
    except (UnicodeError, ValueError) as exc:
        if isinstance(exc, m.ExternalFragmentsError):
            raise
        m.fail('EXTERNAL_FRAGMENTS_RECORD_INVALID')


def scan_source(path, expected_sha256, contigs, ranked):
    source_resource = io.resource(path, expected_sha256)
    kind = io.encoding(path); dictionary = {n: (i, length) for i, (n, length) in enumerate(contigs)}
    decoded = hashlib.sha256(); data = hashlib.sha256(); headers = hashlib.sha256()
    n = total = header_bytes = header_records = 0; columns = None
    with Path(ranked).open('xb') as output:
        for line in io.lines(path, kind):
            if (len(line) > m.MAX_LINE or not line.endswith(b'\n') or b'\r' in line or b'\0' in line):
                m.fail('EXTERNAL_FRAGMENTS_RECORD_INVALID')
            decoded.update(line)
            if line.startswith(b'#'):
                if columns is not None:
                    m.fail('EXTERNAL_FRAGMENTS_HEADER_INVALID')
                try:
                    line.decode('utf-8')
                except UnicodeError:
                    m.fail('EXTERNAL_FRAGMENTS_HEADER_INVALID')
                headers.update(line); header_bytes += len(line); header_records += 1
                if header_bytes > m.MAX_HEADER_BYTES or header_records > m.MAX_HEADERS:
                    m.fail('EXTERNAL_FRAGMENTS_HEADER_INVALID')
                continue
            rank, start, end, barcode, strand, support, width = parse_record(line, dictionary)
            if columns is not None and columns != width:
                m.fail('EXTERNAL_FRAGMENTS_LAYOUT_MIXED')
            columns = width; n += 1; total += support; data.update(line)
            if n > MAX_TOTAL or total > MAX_TOTAL:
                m.fail('EXTERNAL_FRAGMENTS_SUPPORT_OVERFLOW')
            output.write(f'{rank}\t{start}\t{end}\t{barcode}\t{strand}\t{support}\n'.encode())
    if not n:
        m.fail('EXTERNAL_FRAGMENTS_EMPTY')
    return dict(resource=source_resource, encoding=kind, decoded_sha256=decoded.hexdigest(),
        data_sha256=data.hexdigest(), header_sha256=headers.hexdigest(), header_bytes=header_bytes,
        header_records=header_records, columns=columns, n_records=n, sum_support=total)


def canonicalize(ranked, contigs, directory, runtime, columns):
    sorted_path = directory / 'sorted.tmp'; path = directory / 'canonical.tmp'
    io.sort_ranked(ranked, sorted_path, directory, runtime)
    count = total = maximum = 0; barcodes = set(); previous = None; digest = hashlib.sha256()
    with sorted_path.open('rb') as source, path.open('xb') as target:
        for line in iter(lambda: source.readline(m.MAX_LINE + 1), b''):
            rank, start, end, barcode, strand, support = line.decode().rstrip('\n').split('\t')
            rank, start, end, support = map(int, (rank, start, end, support))
            key = rank, start, end, barcode, strand
            if previous is not None and key <= previous:
                m.fail('EXTERNAL_FRAGMENTS_DUPLICATE_KEY')
            previous = key
            data = f'{contigs[rank][0]}\t{start}\t{end}\t{barcode}\t{support}'
            data += ('\t' + strand if columns == 6 else '') + '\n'
            raw = data.encode(); target.write(raw); digest.update(raw)
            count += 1; total += support; maximum = max(maximum, support); barcodes.add(barcode)
            if count > MAX_TOTAL or total > MAX_TOTAL:
                m.fail('EXTERNAL_FRAGMENTS_SUPPORT_OVERFLOW')
    return path, dict(canonical_record_stream_sha256=digest.hexdigest(), n_fragment_records=count,
                     sum_support=total, n_distinct_barcodes=len(barcodes), max_support=maximum)


def prepare_in_stage(arguments, stage, runtime):
    """Build a private bundle. Caller owns final publication and receipt."""
    io.verify_packaging(runtime)
    reference, contigs, reference_paths = io.reference(arguments['reference_bundle_path'], arguments['reference_bundle_sha256'])
    paths = reference_paths + [arguments['source_path']]
    if arguments['source_index_path'] is not None:
        paths.append(arguments['source_index_path'])
    before = io.take_snapshots(paths)
    directory = stage / 'libraries' / arguments['namespace']; directory.mkdir(parents=True)
    ranked = directory / 'ranked.tmp'
    source = scan_source(arguments['source_path'], arguments['source_sha256'], contigs, ranked)
    index = None if arguments['source_index_path'] is None else io.resource(arguments['source_index_path'], arguments['source_index_sha256'])
    io.check_source_index(arguments['source_path'], arguments['source_index_path'], source['encoding'], contigs, stage, runtime)
    plain, summary = canonicalize(ranked, contigs, directory, runtime, source['columns'])
    record = m.validate_adoption_record(dict(artifact_type='agent.external-fragment-adoption', schema_version=1,
        contract_version=m.ADOPTION_CONTRACT, profile_sha256=m.sha_bytes(m.PROFILE_BYTES), source=source,
        source_index=index, reference=reference, namespace=arguments['namespace'],
        source_selection=arguments['source_selection'], transformation_policy=m.TRANSFORMATION, canonical=summary))
    bgzf = directory / 'fragments.tsv.gz'
    with bgzf.open('xb') as output:
        run_stage([runtime.bgzip, *PACKAGING_POLICY['bgzip'], plain], cwd=directory,
                  code='EXTERNAL_FRAGMENTS_PACKAGING_FAILED', output=output)
    run_stage([runtime.tabix, '-p', 'bed', bgzf], cwd=directory, code='EXTERNAL_FRAGMENTS_INDEX_MISMATCH')
    for temporary in (ranked, plain, directory / 'sorted.tmp', directory / 'process.log'):
        temporary.unlink(missing_ok=True)
    profile_path = stage / 'profile.json'; profile_path.write_bytes(m.PROFILE_BYTES)
    record_path = stage / 'adoption.json'; record_path.write_bytes(canonical(record))
    entry = dict(namespace=arguments['namespace'], **summary,
        strand={'mode': 'present' if source['columns'] == 6 else 'absent',
                'definition': m.STRAND_DEFINITION if source['columns'] == 6 else None},
        provenance=m.provenance(record, io.resource(profile_path), io.resource(record_path)))
    for key, path in (('bgzf', bgzf), ('tabix', Path(str(bgzf) + '.tbi'))):
        entry[key] = io.resource(path) | {'path': str(path.relative_to(stage))}
    value = dict(artifact_type=v2.ARTIFACT_TYPE, schema_version=2, contract_version=v2.CONTRACT_VERSION,
                 reference=reference, semantics=v2.SEMANTICS, libraries=[entry])
    value['fragments_identity_sha256'] = v2.fragments_identity(value)
    (stage / 'manifest.json').write_bytes(v2.canonical_fragments_manifest_v2_bytes(value))
    io.check_snapshots(before)
    return value, before
