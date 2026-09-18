"""Exact caller-ordered namespace/barcode declarations, without QC claims."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile

from . import scatac_matrix_contract as m, scatac_fragments_v2 as v2
from ._cell_selection_contract import SELECTED_HEADER
from ._barcode_qc_contract import gzip_lines, MAX_BARCODES
from .scatac_selection_profile import encode_cell_id, decode_cell_id
from .scatac_fragments_v2_verifier import file_sha256, take_snapshots, check_snapshots
from .scatac_qc_reference import _fsync_dir
from .authority_context import owned_verification, publication_moved
from agent.tools._cancellation import cancellation_checkpoint

ARTIFACT = 'agent.scatac-explicit-cells'
CONTRACT = 'scatac-explicit-cells.v1'
PROFILE = 'caller-declared-ordered-barcodes.v1'
PROFILE_SHA256 = hashlib.sha256(PROFILE.encode()).hexdigest()


def identity(value):
    return hashlib.sha256(CONTRACT.encode() + b'\0' + m.canonical(
        {k: v for k, v in value.items() if k != 'identity_sha256'})).hexdigest()


def validate(value):
    m.shape(value, ('artifact_type', 'schema_version', 'contract_version', 'declaration',
                    'selected_count', 'ordered_selected_sha256', 'table', 'identity_sha256'))
    if (value['artifact_type'] != ARTIFACT or type(value['schema_version']) is not int
            or value['schema_version'] != 1 or value['contract_version'] != CONTRACT):
        m.fail('EXPLICIT_CELLS_CONTRACT_INVALID')
    v2.text(value['declaration'])
    m.integer(value['selected_count'], 0, MAX_BARCODES)
    m.sha(value['ordered_selected_sha256']); m.sha(value['identity_sha256'])
    m.shape(value['table'], ('path', 'sha256', 'size_bytes'))
    if value['table']['path'] != 'cells.tsv.gz': m.fail('EXPLICIT_CELLS_CONTRACT_INVALID')
    m.sha(value['table']['sha256']); m.integer(value['table']['size_bytes'], 1)
    if identity(value) != value['identity_sha256']: m.fail('EXPLICIT_CELLS_IDENTITY_INVALID')
    return value


def load_manifest(path, expected_sha256):
    m.absolute_path(str(path)); m.sha(expected_sha256)
    with Path(path).open('rb') as f: raw = f.read(m.MAX_MANIFEST_BYTES + 1)
    if len(raw) > m.MAX_MANIFEST_BYTES or hashlib.sha256(raw).hexdigest() != expected_sha256:
        m.fail('EXPLICIT_CELLS_MANIFEST_MISMATCH')
    value = validate(json.loads(raw, object_pairs_hook=v2._pairs, parse_constant=lambda _: m.fail()))
    if raw != m.canonical(value): m.fail('EXPLICIT_CELLS_CONTRACT_INVALID')
    return value


def iter_rows(path):
    lines = gzip_lines(path, MAX_BARCODES + 1)
    if next(lines, None) != SELECTED_HEADER: m.fail('EXPLICIT_CELLS_AXIS_INVALID')
    for index, line in enumerate(lines):
        fields = line.decode('ascii').rstrip('\n').split('\t')
        if len(fields) != 4 or fields[0] != str(index): m.fail('EXPLICIT_CELLS_AXIS_INVALID')
        _, namespace, barcode, rendered = fields
        if decode_cell_id(rendered) != (namespace, barcode): m.fail('EXPLICIT_CELLS_AXIS_INVALID')
        yield index, namespace, barcode, rendered


@owned_verification('explicit_cells')
def verify_explicit_cells(manifest_path, *, expected_sha256):
    root = Path(manifest_path).parent; table = root / 'cells.tsv.gz'
    before = take_snapshots([manifest_path, table])
    value = load_manifest(manifest_path, expected_sha256)
    if table.stat().st_size != value['table']['size_bytes'] or file_sha256(table) != value['table']['sha256']:
        m.fail('EXPLICIT_CELLS_TABLE_MISMATCH')
    digest = hashlib.sha256(b'agent.ordered-selected-cells.v1\0'); count = 0
    with tempfile.TemporaryDirectory(prefix='agent-explicit-cells-') as scratch:
        with sqlite3.connect(str(Path(scratch) / 'unique.sqlite')) as db:
            db.execute('CREATE TABLE cells(ns TEXT, bc TEXT, PRIMARY KEY(ns,bc)) WITHOUT ROWID')
            try:
                for _, ns, bc, _ in iter_rows(table):
                    db.execute('INSERT INTO cells VALUES(?,?)', (ns, bc))
                    digest.update((ns + '\t' + bc + '\n').encode('ascii')); count += 1
                    if count % 4096 == 0: cancellation_checkpoint()
            except sqlite3.IntegrityError:
                m.fail('EXPLICIT_CELLS_DUPLICATE')
    if count != value['selected_count'] or digest.hexdigest() != value['ordered_selected_sha256']:
        m.fail('EXPLICIT_CELLS_AXIS_INVALID')
    check_snapshots(before)
    return value


def publish_explicit_cells(*, cells, declaration, output_dir):
    """Publish exact iterable of (namespace, barcode) tuples in caller order.

    Empty sets are valid. Duplicate/invalid identifiers fail; no discovery,
    sorting, thresholds or biological interpretation occurs.
    """
    v2.text(declaration)
    destination = Path(output_dir); m.absolute_path(str(destination))
    if destination != destination.resolve() or destination.exists(): m.fail('MATRIX_OUTPUT_CONFLICT')
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.parent / ('.' + destination.name + '.cells-lock')
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600); os.close(fd)
    try:
        with tempfile.TemporaryDirectory(prefix='.explicit-cells-', dir=destination.parent) as scratch:
            stage = Path(scratch) / 'artifact'; stage.mkdir(); table = stage / 'cells.tsv.gz'
            digest = hashlib.sha256(b'agent.ordered-selected-cells.v1\0'); count = 0
            with table.open('xb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as f:
                f.write(SELECTED_HEADER)
                for pair in cells:
                    if not isinstance(pair, (tuple, list)) or len(pair) != 2: m.fail('EXPLICIT_CELLS_AXIS_INVALID')
                    ns, bc = pair; rendered = encode_cell_id(ns, bc)
                    if count >= MAX_BARCODES: m.fail('MATRIX_RESOURCE_LIMIT')
                    f.write(f'{count}\t{ns}\t{bc}\t{rendered}\n'.encode('ascii'))
                    digest.update((ns + '\t' + bc + '\n').encode('ascii')); count += 1
                    if count % 4096 == 0: cancellation_checkpoint()
            value = dict(artifact_type=ARTIFACT, schema_version=1, contract_version=CONTRACT,
                         declaration=declaration, selected_count=count, ordered_selected_sha256=digest.hexdigest(),
                         table=dict(path=table.name, sha256=file_sha256(table), size_bytes=table.stat().st_size))
            value['identity_sha256'] = identity(value)
            manifest = stage / 'manifest.json'; manifest.write_bytes(m.canonical(validate(value)))
            sha = file_sha256(manifest)
            verify_explicit_cells(manifest, expected_sha256=sha)
            for path in (table, manifest):
                with path.open('rb') as f: os.fsync(f.fileno())
            _fsync_dir(stage); cancellation_checkpoint()
            if destination.exists() or destination.is_symlink(): m.fail('MATRIX_OUTPUT_CONFLICT')
            os.rename(stage, destination); _fsync_dir(destination.parent)
            publication_moved(stage, destination)
            return dict(manifest_path=str(destination / 'manifest.json'), manifest_sha256=sha)
    finally:
        lock.unlink()
