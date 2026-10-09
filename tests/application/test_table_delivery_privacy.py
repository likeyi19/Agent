"""Closed original QC/selection tables retain scientific IDs and gzip bytes."""

import gzip
import io
from types import MappingProxyType

import pytest

from agent.application.table_delivery import TablePrivacyError, validate_scientific_table
from agent.tools.data import _barcode_qc_contract as qc
from agent.tools.data import _cell_selection_contract as selection
from agent.tools.data.scatac_selection_profile import encode_cell_id


ROLES = ((qc.CONTRACT, 'table'), (qc.CONTRACT, 'histogram'),
         (selection.CONTRACT, 'decisions'), (selection.CONTRACT, 'selected'))


def compressed(data, **kwargs):
    stream = io.BytesIO()
    with gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0,
                       compresslevel=6, **kwargs) as output:
        output.write(data)
    return stream.getvalue()


def content(role, namespace='lab', barcode='ACGT-1'):
    rendered = encode_cell_id(namespace, barcode)
    if role == 'table':
        return qc.HEADER + qc.row_bytes(namespace, barcode, (10, 8, 3, 1, 1, 5, 2, 1))
    if role == 'histogram':
        return b''.join(f'{index}\t0\n'.encode() for index in range(1, 1002))
    if role == 'decisions':
        return selection.DECISION_HEADER + f'{namespace}\t{barcode}\t{rendered}\tnot_assessed\ttrue\ttrue\tNONE\n'.encode()
    return selection.SELECTED_HEADER + f'0\t{namespace}\t{barcode}\t{rendered}\n'.encode()


def manifest(contract, role, count=1):
    return {'contract_version': contract,
            'selected_count' if role == 'selected' else 'row_count': count}


@pytest.mark.parametrize('contract,role', ROLES)
def test_reviewed_original_table_layouts_preserve_compressed_bytes(contract, role):
    original = compressed(content(role))
    snapshot = io.BytesIO(original)
    validate_scientific_table(snapshot, contract, role, manifest(contract, role))
    assert not snapshot.closed
    assert snapshot.getvalue() == original


@pytest.mark.parametrize('role', ['table', 'decisions', 'selected'])
@pytest.mark.parametrize('barcode', ['patient-A/replicate-1', 'sample#1.ATCG-1', 'A:B|C=1', 'batch\\ATCG'])
def test_opaque_scientific_identifiers_are_preserved(role, barcode):
    contract = qc.CONTRACT if role == 'table' else selection.CONTRACT
    original = compressed(content(role, namespace='lib.1-ABC_2', barcode=barcode))
    snapshot = io.BytesIO(original)
    validate_scientific_table(snapshot, contract, role, manifest(contract, role))
    assert snapshot.getvalue() == original


@pytest.mark.parametrize('role', ['table', 'decisions', 'selected'])
@pytest.mark.parametrize('barcode', ['/home/operator/private.tsv', '~/private.tsv',
                                    'C:\\Users\\operator\\private.tsv',
                                    '\\\\server\\private.tsv', 'file:///home/operator/private.tsv',
                                    'source=/srv/operator/private.tsv', "'/srv/private.tsv'"])
def test_unmistakable_private_paths_in_identifiers_are_ineligible(role, barcode):
    contract = qc.CONTRACT if role == 'table' else selection.CONTRACT
    with pytest.raises(TablePrivacyError):
        validate_scientific_table(io.BytesIO(compressed(content(role, barcode=barcode))),
                                  contract, role, manifest(contract, role))


def test_owner_gzip_writer_has_no_filename_comment_or_extra_metadata(tmp_path):
    path = tmp_path / 'barcodes.tsv.gz'
    qc.write_gzip(path, [content('table')])
    original = path.read_bytes()
    with path.open('rb') as stream:
        validate_scientific_table(stream, qc.CONTRACT, 'table', manifest(qc.CONTRACT, 'table'))
        assert not stream.closed
    assert path.read_bytes() == original


@pytest.mark.parametrize('flag,metadata', [(8, b'/home/operator/private.tsv\0'),
                                         (16, b'/home/operator/private\0'),
                                         (4, b'\x04\x00data'), (2, b'\x00\x00')])
def test_optional_gzip_metadata_is_rejected(flag, metadata):
    original = compressed(content('table'))
    modified = original[:3] + bytes([flag]) + original[4:10] + metadata + original[10:]
    with pytest.raises(TablePrivacyError):
        validate_scientific_table(io.BytesIO(modified), qc.CONTRACT, 'table')


@pytest.mark.parametrize('offset,value', [(4, 1), (8, 2), (9, 3)])
def test_unreviewed_gzip_header_is_rejected(offset, value):
    modified = bytearray(compressed(content('table')))
    modified[offset] = value
    with pytest.raises(TablePrivacyError):
        validate_scientific_table(io.BytesIO(modified), qc.CONTRACT, 'table')


@pytest.mark.parametrize('tamper', ['crc', 'size', 'truncate', 'second_member', 'metadata', 'padding'])
def test_corrupt_or_additional_gzip_framing_is_rejected(tamper):
    original = compressed(content('table'))
    if tamper in {'crc', 'size'}:
        modified = bytearray(original)
        modified[-8 if tamper == 'crc' else -4] ^= 1
    elif tamper == 'truncate':
        modified = original[:-1]
    elif tamper == 'second_member':
        modified = original + compressed(b'')
    else:
        modified = original + (b'/home/operator/private' if tamper == 'metadata' else b'\0')
    with pytest.raises(TablePrivacyError):
        validate_scientific_table(io.BytesIO(modified), qc.CONTRACT, 'table')


@pytest.mark.parametrize('contract,role', ROLES)
def test_table_roles_are_closed_and_contract_specific(contract, role):
    with pytest.raises(TablePrivacyError):
        validate_scientific_table(io.BytesIO(compressed(content(role))), contract, 'source_path')
    other = selection.CONTRACT if contract == qc.CONTRACT else qc.CONTRACT
    with pytest.raises(TablePrivacyError):
        validate_scientific_table(io.BytesIO(compressed(content(role))), other, role)


@pytest.mark.parametrize('role', ['table', 'decisions', 'selected'])
@pytest.mark.parametrize('tamper', ['header', 'extra_field', 'private_literal', 'missing_lf', 'non_ascii'])
def test_unreviewed_payload_fields_are_rejected(role, tamper):
    contract = qc.CONTRACT if role == 'table' else selection.CONTRACT
    raw = content(role)
    header, row = raw.split(b'\n', 1)
    if tamper == 'header':
        raw = header + b'\tsource_path\n' + row
    elif tamper == 'extra_field':
        raw = header + b'\n' + row[:-1] + b'\t/home/operator/private\n'
    elif tamper == 'private_literal':
        fields = row[:-1].split(b'\t')
        fields[2 if role == 'table' else 0 if role == 'selected' else 3] = b'/home/operator/private'
        raw = header + b'\n' + b'\t'.join(fields) + b'\n'
    elif tamper == 'missing_lf':
        raw = raw[:-1]
    else:
        raw = raw.replace(b'ACGT-1', b'\xff', 1)
    with pytest.raises(TablePrivacyError):
        validate_scientific_table(io.BytesIO(compressed(raw)), contract, role)


def test_empty_selected_file_is_the_original_valid_output():
    original = compressed(selection.SELECTED_HEADER)
    snapshot = io.BytesIO(original)
    validate_scientific_table(snapshot, selection.CONTRACT, 'selected',
                              manifest(selection.CONTRACT, 'selected', 0))
    assert snapshot.getvalue() == original


@pytest.mark.parametrize('role', ['table', 'decisions', 'selected'])
def test_accepted_row_count_is_exact(role):
    contract = qc.CONTRACT if role == 'table' else selection.CONTRACT
    for count in (0, 2):
        with pytest.raises(TablePrivacyError):
            validate_scientific_table(io.BytesIO(compressed(content(role))), contract, role,
                                      manifest(contract, role, count))


def test_headerless_histogram_has_exact_structural_bins():
    raw = content('histogram')
    for altered in (b'length\tcount\n' + raw, raw.replace(b'2\t0\n', b'1\t0\n', 1),
                    raw[:-len(b'1001\t0\n')], raw + b'1002\t0\n'):
        with pytest.raises(TablePrivacyError):
            validate_scientific_table(io.BytesIO(compressed(altered)), qc.CONTRACT, 'histogram')


def test_frozen_manifest_and_conflicting_contract():
    snapshot = io.BytesIO(compressed(content('table')))
    accepted = MappingProxyType(manifest(qc.CONTRACT, 'table'))
    validate_scientific_table(snapshot, qc.CONTRACT, 'table', accepted)
    with pytest.raises(TablePrivacyError):
        validate_scientific_table(snapshot, qc.CONTRACT, 'table',
                                  {'contract_version': selection.CONTRACT, 'row_count': 1})


def test_oversized_compressed_line_is_rejected_with_bounded_reads():
    class TrackedStream(io.BytesIO):
        sizes = []

        def read(self, size=-1):
            self.sizes.append(size)
            assert 0 < size <= 64 * 1024
            return super().read(size)

    snapshot = TrackedStream(compressed(qc.HEADER + b'x' * (10 * 1024 * 1024) + b'\n'))
    with pytest.raises(TablePrivacyError):
        validate_scientific_table(snapshot, qc.CONTRACT, 'table', manifest(qc.CONTRACT, 'table'))
    assert snapshot.sizes


def test_excess_rows_fail_before_unbounded_expansion():
    raw = content('table')
    row = raw.split(b'\n', 1)[1]
    snapshot = io.BytesIO(compressed(raw + row * 100_000))
    with pytest.raises(TablePrivacyError):
        validate_scientific_table(snapshot, qc.CONTRACT, 'table', manifest(qc.CONTRACT, 'table'))


def test_large_chunk_boundaries_and_trailer_are_consumed():
    rows = [qc.row_bytes('lab', f'ACGT-{index}', (10, 8, 3, 1, 1, 5, 2, 1))
            for index in range(6000)]
    original = compressed(qc.HEADER + b''.join(rows))
    snapshot = io.BytesIO(original)
    validate_scientific_table(snapshot, qc.CONTRACT, 'table', manifest(qc.CONTRACT, 'table', len(rows)))
    assert snapshot.getvalue() == original


def test_eligibility_does_not_recalculate_counts_or_thresholds(monkeypatch):
    import agent.tools.data._barcode_qc_production as qc_production
    import agent.tools.data._cell_selection_production as selection_production
    import agent.tools.data.barcode_qc_verifier as qc_verifier
    import agent.tools.data.cell_selection_verifier as selection_verifier
    import agent.tools.data.scatac_selection_profile as selection_profile

    def prohibited(*args, **kwargs):
        raise AssertionError('delivery must not invoke scientific production or verification')

    for module, name in ((qc_production, 'produce'), (selection_production, 'produce'),
                         (qc_verifier, 'verify_barcode_qc'), (selection_verifier, 'verify_cell_selection'),
                         (selection_profile, 'decide'), (selection_profile, 'thresholds')):
        monkeypatch.setattr(module, name, prohibited)
    for contract, role in ROLES:
        validate_scientific_table(io.BytesIO(compressed(content(role))), contract, role,
                                  manifest(contract, role))
