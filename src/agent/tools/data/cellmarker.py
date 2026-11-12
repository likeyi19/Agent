"""CellMarker static evidence: faithful SQLite storage and exact, read-only lookup.

No annotation, alias inference, evidence weights or public scientific tool. Source
positions count CSV data records from one (the header is excluded).
"""
from collections import defaultdict
from contextlib import closing
from copy import deepcopy
import csv
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import zipfile

from agent.tools._file_identity import file_closure, snapshots, unchanged

FORMAT = 'cellmarker-evidence.sqlite.v1'
RESOURCE = 'cellmarker-static-resource.v1'
COLUMNS = tuple('species tissue_class tissue_type uberon_id disease_category disease '
    'cell_name_class cell_name cellontology_id marker symbol gene_id gene_type gene_name '
    'uniprot_id technology_seq marker_source pmid title journal year series_id method_details'.split())
ARCHIVES = tuple(f'{name}_cell_marker.zip' for name in ('all', 'human', 'mouse', 'method', 'single'))
MEMBER = 'all_cell_marker.txt'
FIELDS = ', '.join('"' + name + '"' for name in COLUMNS)


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _read(path):
    # mode=ro protects the main database while permitting connection-local TEMP
    # query inputs. No persistent query state, SQL variable-count limit or WAL.
    connection = sqlite3.connect(Path(path).as_uri() + '?mode=ro', uri=True)
    connection.execute('PRAGMA trusted_schema=OFF')
    return connection


def _rows(archive, checksum):
    with zipfile.ZipFile(archive) as z:
        with z.open(MEMBER) as stream:
            def lines():
                for raw in stream:
                    checksum.update(raw)
                    yield raw.decode('utf-8', errors='strict')
            reader = csv.reader(lines(), delimiter='\t', strict=True)
            if tuple(next(reader, ())) != COLUMNS:
                raise ValueError('CellMarker All header differs from the 23 source columns.')
            for position, row in enumerate(reader, 1):
                if len(row) != len(COLUMNS):
                    raise ValueError(f'CellMarker source record {position} has an invalid field count.')
                yield position, *row


def _metadata(connection):
    value = json.loads(connection.execute('SELECT value FROM metadata WHERE key = ?', ('source',)).fetchone()[0])
    if (value.get('format') != FORMAT or value.get('columns') != list(COLUMNS)
            or value.get('member') != MEMBER or connection.execute('PRAGMA user_version').fetchone()[0] != 1):
        raise ValueError('Unsupported CellMarker index format.')
    return value


def verify_index(archive, index, *, expected_sha256):
    """Compare every indexed field/position with All; never change either file."""
    before = snapshots([archive, index])
    source = file_closure([archive])[0]
    if source['sha256'] != expected_sha256:
        raise ValueError('CellMarker All archive checksum differs from its declared identity.')
    with closing(_read(index)) as connection:
        metadata = _metadata(connection)
        expected_source = dict(name=Path(archive).name, sha256=source['sha256'], size_bytes=source['size_bytes'])
        if metadata['archive'] != expected_source:
            raise ValueError('CellMarker index belongs to a different source archive.')
        columns = connection.execute('PRAGMA table_info(evidence)').fetchall()
        if [(r[1], r[2], r[3], r[5]) for r in columns] != [
                ('source_row', 'INTEGER', 0, 1), *[(name, 'TEXT', 1, 0) for name in COLUMNS]]:
            raise ValueError('CellMarker evidence schema changed.')
        for field in ('marker', 'symbol', 'gene_id'):
            indexed = connection.execute(f'PRAGMA index_info(evidence_{field})').fetchall()
            if [r[2] for r in indexed] != ['species', field]:
                raise ValueError('CellMarker lookup index changed.')
        if connection.execute('PRAGMA integrity_check').fetchall() != [('ok',)]:
            raise ValueError('CellMarker SQLite integrity check failed.')
        cursor = connection.execute(f'SELECT source_row, {FIELDS} FROM evidence ORDER BY source_row')
        checksum, count, species = hashlib.sha256(), 0, set()
        for row in _rows(archive, checksum):
            if cursor.fetchone() != row:
                raise ValueError(f'CellMarker source correspondence differs at record {row[0]}.')
            count += 1
            species.add(row[1])
        if (cursor.fetchone() is not None or count != metadata['rows']
                or checksum.hexdigest() != metadata['member_sha256']
                or sorted(species) != metadata['species']):
            raise ValueError('CellMarker index is incomplete or has additional records.')
    unchanged(before)
    return metadata


def build_index(archive, destination, *, expected_sha256):
    """Stream All to a temporary database; verify before exclusive publication.

    An existing file is verified and reused, never replaced, even if invalid.
    Reproducible bytes exclude paths, timestamps and runtime-specific metadata.
    """
    archive, destination = Path(archive).resolve(), Path(destination).absolute()
    before = snapshots([archive])
    source = file_closure([archive])[0]
    if source['sha256'] != expected_sha256:
        raise ValueError('CellMarker All archive checksum differs from its declared identity.')
    if destination.exists() or destination.is_symlink():
        verify_index(archive, destination, expected_sha256=expected_sha256)
        return file_closure([destination])[0]
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=destination.parent, prefix='.cellmarker-', suffix='.sqlite')
    os.close(fd)
    temporary = Path(name)
    try:
        with closing(sqlite3.connect(temporary)) as connection:
            connection.execute('PRAGMA user_version=1')
            connection.execute('CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
            definition = ', '.join('"' + field + '" TEXT NOT NULL' for field in COLUMNS)
            connection.execute('CREATE TABLE evidence (source_row INTEGER PRIMARY KEY, ' + definition + ')')
            insert = 'INSERT INTO evidence VALUES (' + ','.join('?' for _ in range(24)) + ')'
            checksum, count, species, batch = hashlib.sha256(), 0, set(), []
            for row in _rows(archive, checksum):
                count += 1
                species.add(row[1])
                batch.append(row)
                if len(batch) == 2000:
                    connection.executemany(insert, batch)
                    batch.clear()
            connection.executemany(insert, batch)
            for field in ('marker', 'symbol', 'gene_id'):
                connection.execute(f'CREATE INDEX evidence_{field} ON evidence (species, {field})')
            metadata = dict(format=FORMAT, columns=list(COLUMNS), member=MEMBER,
                archive=dict(name=archive.name, sha256=source['sha256'], size_bytes=source['size_bytes']),
                member_sha256=checksum.hexdigest(), rows=count, species=sorted(species))
            connection.execute('INSERT INTO metadata VALUES (?, ?)', ('source', _json(metadata)))
            connection.commit()
        verify_index(archive, temporary, expected_sha256=expected_sha256)
        unchanged(before)
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            verify_index(archive, destination, expected_sha256=expected_sha256)
        fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return file_closure([destination])[0]
    finally:
        temporary.unlink(missing_ok=True)


def resource_files(resource):
    """The existing library owns file hashing and immutable registration."""
    if (set(resource) != {'schema', 'record_url', 'archives', 'index', 'index_format'}
            or resource['schema'] != RESOURCE or resource['index_format'] != FORMAT
            or not isinstance(resource['record_url'], str) or not resource['record_url']
            or set(resource['archives']) != set(ARCHIVES)):
        raise ValueError('CellMarker resource requires the five original archives and one versioned index.')
    files = [*resource['archives'].values(), resource['index']]
    if any(set(row) != {'path', 'sha256', 'size_bytes'} for row in files):
        raise ValueError('CellMarker files require exact path, SHA-256 and size identities.')
    if len({row['path'] for row in files}) != 6:
        raise ValueError('CellMarker source archives and index must be distinct files.')
    return {row['path']: row['sha256'] for row in files}


def qualify_resource(resource):
    resource_files(resource)
    archive = resource['archives'][ARCHIVES[0]]
    result = verify_index(archive['path'], resource['index']['path'], expected_sha256=archive['sha256'])
    if result['archive']['name'] != ARCHIVES[0]:
        raise ValueError('CellMarker operational index must use the All archive.')
    # Sizes in the declared resource agree with the library's pinned files too.
    if any(Path(row['path']).stat().st_size != row['size_bytes']
           for row in [*resource['archives'].values(), resource['index']]):
        raise ValueError('CellMarker resource file size differs from its declared identity.')
    return result


class CellMarkerEvidence:
    """One resolved library resource for an operation's read-only batch lookups.

    Resolution hashes its file closure once. Subsequent queries/pages use the
    existing exact-file snapshot checks and verify the catalog entry is unchanged.
    Create a new instance for a new operation; nothing is cached across restarts.
    """
    def __init__(self, library, resource_id):
        self.library = library
        self.entry = library.get(resource_id)
        if self.entry['kind'] != 'cellmarker':
            raise ValueError('Selected scientific resource is not CellMarker evidence.')
        self._before = snapshots(row['path'] for row in self.entry['files'])
        self.resource = library.check(self.entry)
        resource_files(self.resource)
        self.path = self.resource['index']['path']
        with closing(_read(self.path)) as connection:
            self.metadata = _metadata(connection)
        self.identity = dict(resource_id=resource_id, version=self.entry['version'],
            record_url=self.resource['record_url'], index_format=FORMAT,
            index_sha256=self.resource['index']['sha256'],
            archive=self.metadata['archive'], member=MEMBER,
            member_sha256=self.metadata['member_sha256'])
        self._check()

    def _check(self):
        if self.library.get(self.entry['id']) != self.entry:
            raise ValueError('CellMarker scientific resource registration changed.')
        unchanged(self._before)

    def candidates(self, *, species=None, filters=None):
        """Exact scoped cell names, including candidates with no queried genes.

        Species remains part of the identity; no cross-species label merging.
        This read-only inventory does not rank or qualify any association.
        """
        filters = dict(filters or {})
        if (any(not isinstance(v, str) for v in filters.values())
                or not set(filters) <= set(COLUMNS) - {'species'}
                or species is not None and not isinstance(species, str)):
            raise ValueError('CellMarker queries use exact strings and existing source columns.')
        if species is not None:
            filters['species'] = species
        where = ' AND '.join(f'"{key}" = ?' for key in sorted(filters)) or '1'
        self._check()
        with closing(_read(self.path)) as connection:
            rows = connection.execute('SELECT DISTINCT species, cell_name FROM evidence WHERE '
                + where + ' ORDER BY species, cell_name', [filters[k] for k in sorted(filters)]).fetchall()
        self._check()
        return [dict(species=s, cell_name=c) for s, c in rows]

    def query(self, names=(), *, gene_ids=(), species=None, filters=None, page_size=128):
        """Yield deterministic technical pages; consuming all pages returns all hits.

        Names match literal marker and symbol, IDs match literal gene_id strings.
        No namespace or numeric conversion is inferred. Empty input strings have
        no evidence; None for a filter means omitted only by omitting that key.
        Filters accept exact source-column strings. Duplicate input names retain
        their input positions without duplicating source evidence in the result.
        """
        names = [names] if isinstance(names, str) else list(names)
        gene_ids = [gene_ids] if isinstance(gene_ids, str) else list(gene_ids)
        filters = dict(filters or {})
        if (any(not isinstance(v, str) for v in [*names, *gene_ids, *filters.values()])
                or not set(filters) <= set(COLUMNS) - {'species'}
                or species is not None and not isinstance(species, str)):
            raise ValueError('CellMarker queries use exact strings and existing source columns.')
        if type(page_size) is not int or page_size < 1:
            raise ValueError('Technical page size must be a positive integer.')
        request = dict(names=names, gene_ids=gene_ids, species=species, filters=filters)
        request = dict(id=hashlib.sha256(_json(request).encode()).hexdigest(), **request)
        inputs = {}
        for kind, values in (('name', names), ('gene_id', gene_ids)):
            mapping = defaultdict(list)
            for position, value in enumerate(values):
                if value != '':
                    mapping[value].append(dict(kind=kind, position=position, value=value))
            inputs[kind] = mapping
        selected_species = [species] if species is not None else self.metadata['species']
        # Per-query TEMP inputs remove SQLite's host-parameter ceiling. The main
        # database is opened read-only; concurrent query iterators are independent.
        self._check()
        with closing(_read(self.path)) as connection:
            connection.execute('CREATE TEMP TABLE requested (kind TEXT, value TEXT)')
            connection.executemany('INSERT INTO requested VALUES (?, ?)',
                ((kind, value) for kind, mapping in inputs.items() for value in mapping))
            branches, parameters = [], []
            for field, kind in (('marker', 'name'), ('symbol', 'name'), ('gene_id', 'gene_id')):
                conditions = [f'e.species IN ({",".join("?" for _ in selected_species)})', 'q.kind = ?']
                values = [*selected_species, kind]
                for column, value in sorted(filters.items()):
                    conditions.append(f'e."{column}" = ?')
                    values.append(value)
                branches.append(f'SELECT e.source_row FROM requested q CROSS JOIN evidence e '
                    f'INDEXED BY evidence_{field} WHERE e.{field} = q.value AND ' + ' AND '.join(conditions))
                parameters.extend(values)
            sql = (f'SELECT {FIELDS}, group_concat(source_row), count(*) FROM evidence '
                'WHERE source_row IN (' + ' UNION '.join(branches) + ') '
                f'GROUP BY {FIELDS} ORDER BY min(source_row)')
            cursor = connection.execute(sql, parameters)
            pending, page = cursor.fetchone(), 0
            while True:
                items = []
                while pending is not None and len(items) < page_size:
                    fields = dict(zip(COLUMNS, pending[:23]))
                    matches = [dict(query, field=field)
                        for field, kind in (('marker', 'name'), ('symbol', 'name'), ('gene_id', 'gene_id'))
                        for query in inputs[kind].get(fields[field], ())]
                    items.append(dict(fields=fields, matches=matches,
                        source_rows=sorted(map(int, pending[23].split(','))), multiplicity=pending[24]))
                    pending = cursor.fetchone()
                self._check()
                yield dict(query=deepcopy(request), resource=deepcopy(self.identity),
                           page=page, complete=pending is None, items=items)
                if pending is None:
                    break
                page += 1
