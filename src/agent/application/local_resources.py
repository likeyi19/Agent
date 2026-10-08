"""Durable, byte-pinned local sources; registration confers no scientific authority.

Sources remain at their operator-approved locations. New consumption verifies the
pinned bytes; reading historical attribution does not require the source to exist.
This boundary shares the application's trusted-local filesystem lifetime contract.
"""
from contextlib import contextmanager
from collections.abc import Mapping
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile

from agent.schemas.orchestration import freeze_json_mapping, _serialize
from .interactive_schemas import ClientError
from .session_state import canonical, digest
from .workspace import ManagedWorkspace, ApplicationWorkspaceError


MAX_RECORD_BYTES = 16_384
MAX_INPUT_BYTES = 65_536
MAX_COLLECTION_MEMBERS = 128
MAX_COLLECTION_BYTES = 32_768
_FORMAT = 'agent.local-resource.v1'
_ID = re.compile(r'local-[0-9a-f]{64}')
_SHA = re.compile(r'[0-9a-f]{64}')
_INPUT_TYPES = ('h5ad', 'external_fragments', 'bam', 'fastq')
_OPERATION_TYPES = {
    'inspect_scATAC': 'h5ad',
    'adopt_scATAC_cell_by_ccre': 'h5ad',
    'import_scATAC_fragments': 'external_fragments',
    'inspect_raw_scATAC': 'bam',
    'prepare_scATAC_bam_fragments': 'bam',
}
_COLLECTION_OPERATIONS = ('inspect_raw_scATAC', 'prepare_scATAC_fragments')
_MESSAGES = {
    'LOCAL_RESOURCE_UNAVAILABLE': 'The registered local resource is unavailable.',
    'LOCAL_RESOURCE_ACCESS_INVALID': 'The local source is unavailable or outside the approved source policy.',
    'LOCAL_RESOURCE_INTEGRITY_INVALID': 'The local source no longer matches its registered byte identity.',
    'LOCAL_RESOURCE_TYPE_UNSUPPORTED': 'The declared local input type is not supported.',
    'LOCAL_RESOURCE_RECORD_INVALID': 'The local resource record is incomplete, corrupt, or invalid.',
    'LOCAL_RESOURCE_CONFLICT': 'This registration identity already belongs to a different source record.',
    'LOCAL_RESOURCE_BINDING_INVALID': 'The registered input binding or its explicit scientific declarations are invalid.',
    'LOCAL_RESOURCE_OPERATION_UNSUPPORTED': 'This operation is not supported by the local resource binding boundary.',
    'LOCAL_RESOURCE_DECLARATION_REQUIRED': 'Required explicit scientific input declarations are missing.',
}


class ResourceAdmissionError(ValueError):
    """A sanitized application failure, distinct from scientific qualification."""
    def __init__(self, code):
        self.code = code
        self.message = _MESSAGES[code]
        self.error = ClientError(code, self.message)
        super().__init__(self.message)


def _fail(code):
    return ResourceAdmissionError(code)


def _text(value, maximum, code, *, display=False):
    if (type(value) is not str or not value.strip() or len(value) > maximum
            or any(not char.isprintable() for char in value)):
        raise _fail(code)
    if display and (value.startswith(('/', '~/', 'file://'))
                    or re.match(r'^[A-Za-z]:[\\/]', value)):
        raise _fail(code)
    return value


def _identifier(value):
    if type(value) is not str or _ID.fullmatch(value) is None:
        raise _fail('LOCAL_RESOURCE_RECORD_INVALID')
    return value


def _source_text(value):
    _text(value, 4096, 'LOCAL_RESOURCE_RECORD_INVALID')
    path = Path(value)
    if (not path.is_absolute() or str(path) != value or '..' in path.parts
            or '://' in value):
        raise _fail('LOCAL_RESOURCE_RECORD_INVALID')
    return path


def _input_mapping(values):
    try:
        # Import after module construction: the interactive facade imports this
        # owner but owns the existing safe structured-input boundary.
        from .interactive import _inputs
        result = _inputs(values)
        if len(canonical(result)) > MAX_INPUT_BYTES:
            raise ValueError('Input bound exceeded.')
        return result
    except (ValueError, TypeError, RecursionError, UnicodeError) as exc:
        raise _fail('LOCAL_RESOURCE_BINDING_INVALID') from exc


@dataclass(frozen=True)
class LocalResourceRecord:
    resource_id: str
    input_type: str
    label: str
    source_path: str
    source_sha256: str
    size_bytes: int
    attribution: str

    def __post_init__(self):
        _identifier(self.resource_id)
        if self.input_type not in _INPUT_TYPES:
            raise _fail('LOCAL_RESOURCE_TYPE_UNSUPPORTED')
        _text(self.label, 160, 'LOCAL_RESOURCE_RECORD_INVALID', display=True)
        _source_text(self.source_path)
        if type(self.source_sha256) is not str or _SHA.fullmatch(self.source_sha256) is None:
            raise _fail('LOCAL_RESOURCE_RECORD_INVALID')
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise _fail('LOCAL_RESOURCE_RECORD_INVALID')
        _text(self.attribution, 1024, 'LOCAL_RESOURCE_RECORD_INVALID')

    def to_dict(self):
        """Private persistence/binding metadata; never a client representation."""
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @property
    def record_sha256(self):
        return digest(self.to_dict())

    def public(self):
        return dict(resource_id=self.resource_id, input_type=self.input_type,
                    label=self.label, status='registered')


@dataclass(frozen=True)
class RegisteredInput:
    resource_id: str
    record_sha256: str
    tool_name: str
    execution_inputs: object

    def __post_init__(self):
        if type(self.tool_name) is not str or self.tool_name not in _OPERATION_TYPES:
            raise _fail('LOCAL_RESOURCE_OPERATION_UNSUPPORTED')
        try:
            _identifier(self.resource_id)
            if type(self.record_sha256) is not str or _SHA.fullmatch(self.record_sha256) is None:
                raise ValueError('Invalid record digest.')
            values = _input_mapping(self.execution_inputs)
            object.__setattr__(self, 'execution_inputs', freeze_json_mapping(values, 'execution_inputs'))
        except (ValueError, TypeError) as exc:
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID') from exc

    def attribution(self):
        return dict(resource_id=self.resource_id, record_sha256=self.record_sha256,
                    tool_name=self.tool_name)


def _members(values):
    """Canonical exact references, without biological grouping or path discovery."""
    try:
        if type(values) not in (list, tuple) or not 1 <= len(values) <= MAX_COLLECTION_MEMBERS:
            raise ValueError('Invalid collection size.')
        members = []
        for value in values:
            if not isinstance(value, Mapping) or set(value) != {'resource_id', 'record_sha256'}:
                raise ValueError('Invalid member reference.')
            _identifier(value['resource_id'])
            sha = value['record_sha256']
            if type(sha) is not str or _SHA.fullmatch(sha) is None:
                raise ValueError('Invalid member digest.')
            members.append(dict(resource_id=value['resource_id'], record_sha256=sha))
        if len({member['resource_id'] for member in members}) != len(members):
            raise ValueError('Duplicate or conflicting member identity.')
        return sorted(members, key=lambda member: member['resource_id'])
    except (ValueError, TypeError, KeyError) as exc:
        raise _fail('LOCAL_RESOURCE_BINDING_INVALID') from exc


def _collection_sha256(members):
    return digest(dict(format='agent.local-resource-collection.v1', members=members))


@dataclass(frozen=True)
class RegisteredInputCollection:
    """Inline immutable composition of existing records; no second resource store."""
    members: object
    collection_sha256: str
    tool_name: str
    execution_inputs: object

    def __post_init__(self):
        if type(self.tool_name) is not str or self.tool_name not in _COLLECTION_OPERATIONS:
            raise _fail('LOCAL_RESOURCE_OPERATION_UNSUPPORTED')
        members = _members(self.members)
        if self.collection_sha256 != _collection_sha256(members):
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID')
        object.__setattr__(self, 'members', tuple(freeze_json_mapping(member, 'member') for member in members))
        values = _input_mapping(self.execution_inputs)
        object.__setattr__(self, 'execution_inputs', freeze_json_mapping(values, 'execution_inputs'))
        if len(canonical(self.attribution())) > MAX_COLLECTION_BYTES:
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID')

    def attribution(self):
        return dict(members=[_serialize(member) for member in self.members],
                    collection_sha256=self.collection_sha256, tool_name=self.tool_name)


class LocalResourceAdmission:
    """Small application source store and resolver, with no scientific admission."""
    def __init__(self, workspace, *, approved_source_roots=(), registry=None):
        self._workspace = workspace if isinstance(workspace, ManagedWorkspace) else ManagedWorkspace(workspace)
        self._registry = registry
        try:
            roots = []
            for value in approved_source_roots:
                path = Path(value).expanduser().absolute()
                if path != path.resolve(strict=True) or not path.is_dir():
                    raise ValueError('Invalid source root.')
                roots.append(path)
            self._approved = tuple(sorted(set(roots)))
        except (ValueError, TypeError, OSError, RuntimeError) as exc:
            raise _fail('LOCAL_RESOURCE_ACCESS_INVALID') from exc

    def _directory(self, *, create=False):
        path = self._workspace.root / 'local_resources'
        try:
            if create:
                try:
                    return self._workspace._ensure_directory(path)
                except ApplicationWorkspaceError as exc:
                    if not isinstance(exc.__cause__, FileExistsError):
                        raise
                    # Concurrent first registrations can create the directory
                    # together. Recheck all workspace protections before reuse.
                    return self._workspace._ensure_directory(path)
            self._workspace._assert_contained(path)
            if path.is_symlink() or (path.exists() and (not path.is_dir() or path != path.resolve())):
                raise _fail('LOCAL_RESOURCE_RECORD_INVALID')
            if not path.exists():
                raise _fail('LOCAL_RESOURCE_UNAVAILABLE')
            return path
        except ResourceAdmissionError:
            raise
        except (ValueError, OSError, RuntimeError) as exc:
            raise _fail('LOCAL_RESOURCE_RECORD_INVALID') from exc

    def _path(self, resource_id, *, create=False):
        _identifier(resource_id)
        path = self._directory(create=create) / (resource_id + '.json')
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise _fail('LOCAL_RESOURCE_RECORD_INVALID')
        return path

    @staticmethod
    def _snapshot(value):
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
                value.st_ctime_ns, value.st_mode)

    def _source(self, source_path, *, approval=False, integrity=False, include_snapshot=False):
        code = 'LOCAL_RESOURCE_INTEGRITY_INVALID' if integrity else 'LOCAL_RESOURCE_ACCESS_INVALID'
        try:
            if not isinstance(source_path, (str, Path)) or '://' in str(source_path):
                raise ValueError('Invalid local source.')
            path = Path(source_path).expanduser().absolute()
            if path != path.resolve(strict=True):
                raise ValueError('Noncanonical source.')
            if approval and not any(path.is_relative_to(root) for root in self._approved):
                raise ValueError('Unapproved source.')
            before = path.stat(follow_symlinks=False)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError('Nonregular source.')
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            sha = hashlib.sha256()
            with os.fdopen(descriptor, 'rb') as stream:
                if self._snapshot(os.fstat(stream.fileno())) != self._snapshot(before):
                    raise _fail('LOCAL_RESOURCE_INTEGRITY_INVALID')
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    sha.update(chunk)
                if self._snapshot(os.fstat(stream.fileno())) != self._snapshot(before):
                    raise _fail('LOCAL_RESOURCE_INTEGRITY_INVALID')
            if (path != path.resolve(strict=True)
                    or self._snapshot(path.stat(follow_symlinks=False)) != self._snapshot(before)):
                raise _fail('LOCAL_RESOURCE_INTEGRITY_INVALID')
            identity = (str(path), sha.hexdigest(), before.st_size)
            return (*identity, self._snapshot(before)) if include_snapshot else identity
        except ResourceAdmissionError:
            raise
        except (ValueError, TypeError, OSError, RuntimeError) as exc:
            raise _fail(code) from exc

    @contextmanager
    def _lock(self, resource_id):
        root = self._directory(create=True)
        descriptor = None
        try:
            descriptor = os.open(root / (resource_id + '.lock'), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError('Invalid resource lock.')
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        except ResourceAdmissionError:
            raise
        except (ValueError, OSError) as exc:
            raise _fail('LOCAL_RESOURCE_RECORD_INVALID') from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def register(self, registration_key, source_path, *, input_type='h5ad', label, attribution):
        if input_type not in _INPUT_TYPES:
            raise _fail('LOCAL_RESOURCE_TYPE_UNSUPPORTED')
        _text(registration_key, 256, 'LOCAL_RESOURCE_RECORD_INVALID')
        _text(label, 160, 'LOCAL_RESOURCE_RECORD_INVALID', display=True)
        _text(attribution, 1024, 'LOCAL_RESOURCE_RECORD_INVALID')
        resource_id = 'local-' + hashlib.sha256(b'agent.local-resource-id.v1\0' + registration_key.encode('utf-8')).hexdigest()
        source, sha, size, snapshot = self._source(source_path, approval=True, include_snapshot=True)
        record = LocalResourceRecord(resource_id, input_type, label, source, sha, size, attribution)
        with self._lock(resource_id):
            path = self._path(resource_id)
            try:
                if (Path(source) != Path(source).resolve(strict=True)
                        or self._snapshot(Path(source).stat(follow_symlinks=False)) != snapshot):
                    raise ValueError('Source changed before registration publication.')
            except (OSError, ValueError, RuntimeError) as exc:
                raise _fail('LOCAL_RESOURCE_INTEGRITY_INVALID') from exc
            if path.exists():
                if self.load(resource_id) != record:
                    raise _fail('LOCAL_RESOURCE_CONFLICT')
                return record
            self._publish(path, record)
        return record

    def _publish(self, path, record):
        payload = canonical(dict(format=_FORMAT, schema_version=1, record=record.to_dict(), sha256=record.record_sha256))
        if len(payload) > MAX_RECORD_BYTES:
            raise _fail('LOCAL_RESOURCE_RECORD_INVALID')
        temporary = None
        try:
            descriptor, name = tempfile.mkstemp(dir=path.parent, prefix='.' + path.stem, suffix='.tmp')
            temporary = Path(name)
            with os.fdopen(descriptor, 'wb') as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, path)
            temporary.unlink()
            temporary = None
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except (OSError, ValueError) as exc:
            raise _fail('LOCAL_RESOURCE_RECORD_INVALID') from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def load(self, resource_id):
        path = self._path(resource_id)
        def pairs(items):
            value = {}
            for key, item in items:
                if key in value:
                    raise ValueError('Duplicate JSON key.')
                value[key] = item
            return value
        def constant(value):
            raise ValueError('Nonfinite JSON constant.')
        try:
            with path.open('rb') as stream:
                raw = stream.read(MAX_RECORD_BYTES + 1)
            if len(raw) > MAX_RECORD_BYTES:
                raise ValueError('Resource bound exceeded.')
            value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
            if (type(value) is not dict or set(value) != {'format', 'schema_version', 'record', 'sha256'}
                    or value['format'] != _FORMAT or type(value['schema_version']) is not int
                    or value['schema_version'] != 1 or type(value['record']) is not dict
                    or set(value['record']) != set(LocalResourceRecord.__dataclass_fields__)
                    or value['sha256'] != digest(value['record'])):
                raise ValueError('Invalid resource envelope.')
            record = LocalResourceRecord(**value['record'])
            if record.resource_id != resource_id:
                raise ValueError('Resource identity changed.')
            return record
        except FileNotFoundError as exc:
            raise _fail('LOCAL_RESOURCE_UNAVAILABLE') from exc
        except (ValueError, TypeError, OSError, KeyError, RecursionError) as exc:
            raise _fail('LOCAL_RESOURCE_RECORD_INVALID') from exc

    def _binding_inputs(self, record, tool_name, scientific_inputs):
        if type(tool_name) is not str or tool_name not in _OPERATION_TYPES:
            raise _fail('LOCAL_RESOURCE_OPERATION_UNSUPPORTED')
        if record.input_type != _OPERATION_TYPES[tool_name]:
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID')
        if tool_name == 'inspect_scATAC':
            source = dict(input_path=record.source_path)
        elif tool_name == 'inspect_raw_scATAC':
            source = dict(raw_input_paths=[record.source_path])
        else:
            source = dict(source_path=record.source_path, source_sha256=record.source_sha256)
        return self._declared_inputs(tool_name, source, scientific_inputs)

    def _declared_inputs(self, tool_name, source, scientific_inputs):
        values = _input_mapping(scientific_inputs)
        if self._registry is None:
            from agent.orchestration.registry import build_default_tool_registry
            self._registry = build_default_tool_registry()
        try:
            specification = self._registry.get(tool_name)
            owned = {'path'} if tool_name == 'inspect_scATAC' else set(source)
            required = set(specification.required_arguments) - owned - {'output_dir'}
            allowed = (set(specification.required_arguments) | set(specification.optional_arguments)) - owned - {'output_dir'}
            if set(values) - allowed:
                raise ValueError('Unknown scientific input declaration.')
        except (ValueError, TypeError, LookupError) as exc:
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID') from exc
        if required - set(values):
            raise _fail('LOCAL_RESOURCE_DECLARATION_REQUIRED')
        return values | source

    def _collection_records(self, members):
        records = []
        for member in members:
            record = self.load(member['resource_id'])
            if record.input_type != 'fastq' or record.record_sha256 != member['record_sha256']:
                raise _fail('LOCAL_RESOURCE_BINDING_INVALID')
            records.append(record)
        if len({record.source_path for record in records}) != len(records):
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID')
        return records

    def _collection_inputs(self, records, tool_name, scientific_inputs):
        if type(tool_name) is not str or tool_name not in _COLLECTION_OPERATIONS:
            raise _fail('LOCAL_RESOURCE_OPERATION_UNSUPPORTED')
        source = (dict(raw_input_paths=sorted(record.source_path for record in records))
                  if tool_name == 'inspect_raw_scATAC' else {})
        return self._declared_inputs(tool_name, source, scientific_inputs)

    @staticmethod
    def _validate_fastq_intake(records, values):
        from agent.tools.data.raw_scatac_manifest import (
            InputKind, SelectionBasis, load_raw_intake_manifest,
        )
        try:
            if values['intake_manifest_sha256'] is None:
                raise ValueError('Registered collection requires an intake digest.')
            _, intake, _ = load_raw_intake_manifest(values['intake_manifest_path'],
                expected_sha256=values['intake_manifest_sha256'])
            expected = {record.source_path: record.size_bytes for record in records}
            if len(intake.files) != len(expected) or {source.path for source in intake.files} != set(expected):
                raise ValueError('Intake membership differs from the registered collection.')
            for source in intake.files:
                if (source.input_kind is not InputKind.FASTQ or source.size_bytes != expected[source.path]
                        or source.selection is not SelectionBasis.EXPLICIT or source.selection_root is not None):
                    raise ValueError('Intake does not explicitly identify registered FASTQs.')
        except (ValueError, TypeError, OSError, KeyError) as exc:
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID') from exc

    @staticmethod
    def _validate_bam_intake(record, values):
        from agent.tools.data.raw_scatac_manifest import (
            InputKind, SelectionBasis, load_raw_intake_manifest,
        )
        try:
            # The loader permits unpinned reads; registered association does not.
            if values['intake_manifest_sha256'] is None:
                raise ValueError('Registered BAM requires an intake digest.')
            _, intake, _ = load_raw_intake_manifest(values['intake_manifest_path'],
                expected_sha256=values['intake_manifest_sha256'])
            if len(intake.files) != 1 or len(intake.groups) != 1:
                raise ValueError('Registered BAM requires a singleton intake.')
            source = intake.files[0]
            # The intake owner validates group membership. Require explicit file
            # selection so producer reinspection cannot rediscover other sources.
            if (source.input_kind is not InputKind.BAM or source.path != record.source_path
                    or source.size_bytes != record.size_bytes
                    or source.selection is not SelectionBasis.EXPLICIT or source.selection_root is not None):
                raise ValueError('Intake does not identify the registered BAM.')
        except (ValueError, TypeError, OSError, KeyError) as exc:
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID') from exc

    def resolve(self, resource_id, *, tool_name, scientific_inputs=None):
        record = self.load(resource_id)
        inputs = self._binding_inputs(record, tool_name, scientific_inputs)
        binding = RegisteredInput(record.resource_id, record.record_sha256, tool_name, inputs)
        self.validate_binding(binding)
        return binding

    def resolve_collection(self, members, *, tool_name, scientific_inputs=None):
        members = _members(members)
        records = self._collection_records(members)
        inputs = self._collection_inputs(records, tool_name, scientific_inputs)
        binding = RegisteredInputCollection(members, _collection_sha256(members), tool_name, inputs)
        self.validate_binding(binding)
        return binding

    def _validate_collection(self, binding, *, verify_source):
        records = self._collection_records(binding.members)
        values = _serialize(binding.execution_inputs)
        declarations = {key: value for key, value in values.items()
                        if binding.tool_name != 'inspect_raw_scATAC' or key != 'raw_input_paths'}
        try:
            expected = self._collection_inputs(records, binding.tool_name, declarations)
        except ResourceAdmissionError as exc:
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID') from exc
        if values != expected:
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID')
        if verify_source:
            if binding.tool_name == 'prepare_scATAC_fragments':
                self._validate_fastq_intake(records, values)
            # Hash and recheck all members after the intake loader finishes.
            self._verify_records(records)
        return values

    def _verify_records(self, records):
        # Recheck the whole collection's snapshots after hashing, so a change to
        # an earlier member while a later member is read cannot yield a binding.
        snapshots = []
        for record in records:
            source, sha, size, snapshot = self._source(record.source_path, integrity=True, include_snapshot=True)
            if (source, sha, size) != (record.source_path, record.source_sha256, record.size_bytes):
                raise _fail('LOCAL_RESOURCE_INTEGRITY_INVALID')
            snapshots.append((Path(source), snapshot))
        try:
            if any(path != path.resolve(strict=True)
                    or self._snapshot(path.stat(follow_symlinks=False)) != snapshot
                    for path, snapshot in snapshots):
                raise ValueError('Collection changed during validation.')
        except (ValueError, OSError, RuntimeError) as exc:
            raise _fail('LOCAL_RESOURCE_INTEGRITY_INVALID') from exc

    def validate_binding(self, binding, *, verify_source=True):
        if type(binding) not in (RegisteredInput, RegisteredInputCollection) or type(verify_source) is not bool:
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID')
        if type(binding) is RegisteredInputCollection:
            return self._validate_collection(binding, verify_source=verify_source)
        record = self.load(binding.resource_id)
        if binding.record_sha256 != record.record_sha256:
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID')
        values = _serialize(binding.execution_inputs)
        fields = (('input_path',) if binding.tool_name == 'inspect_scATAC' else
                  ('raw_input_paths',) if binding.tool_name == 'inspect_raw_scATAC' else
                  ('source_path', 'source_sha256'))
        declarations = {key: value for key, value in values.items() if key not in fields}
        try:
            expected = self._binding_inputs(record, binding.tool_name, declarations)
        except ResourceAdmissionError as exc:
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID') from exc
        if values != expected:
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID')
        if verify_source and self._source(record.source_path, integrity=True) != (record.source_path, record.source_sha256, record.size_bytes):
            raise _fail('LOCAL_RESOURCE_INTEGRITY_INVALID')
        if verify_source and binding.tool_name == 'prepare_scATAC_bam_fragments':
            self._validate_bam_intake(record, values)
        return values

    @staticmethod
    def _submission_binding(submission):
        try:
            metadata = submission['registered_input']
            if set(metadata) == {'resource_id', 'record_sha256', 'tool_name'}:
                binding_type = RegisteredInput
            elif set(metadata) == {'members', 'collection_sha256', 'tool_name'}:
                binding_type = RegisteredInputCollection
            else:
                raise ValueError('Invalid registered input attribution.')
            binding = binding_type(**metadata, execution_inputs=submission['execution_inputs'])
            if binding.attribution() != _serialize(metadata):
                raise ValueError('Noncanonical registered input attribution.')
            return binding
        except (TypeError, ValueError, KeyError) as exc:
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID') from exc

    def validate_submission(self, submission, *, verify_source=True):
        binding = self._submission_binding(submission)
        self.validate_binding(binding, verify_source=verify_source)

    def validate_result(self, submission, steps):
        """First acceptance only: bind inspections and FASTQ provenance to sources.

        Scientific owners have already verified the result. This checks application
        identity, including selected producer subsets, without reconstructing science.
        Completed historical turns do not call this guard.
        """
        binding = self._submission_binding(submission)
        inspections = any(step.tool_name in {'inspect_scATAC', 'inspect_raw_scATAC'} for step in steps)
        producers = [step for step in steps if step.tool_name == 'prepare_scATAC_fragments']
        collection = type(binding) is RegisteredInputCollection
        if not inspections and not (collection and producers):
            return
        if collection:
            records = self._collection_records(binding.members)
            paths = sorted(record.source_path for record in records)
            for step in steps:
                if step.tool_name == 'inspect_raw_scATAC':
                    actual = _serialize(step.resolved_arguments.get('raw_input_paths'))
                    if (type(actual) is not list or any(type(path) is not str for path in actual)
                            or sorted(actual) != paths):
                        raise _fail('LOCAL_RESOURCE_BINDING_INVALID')
        if collection and producers:
            from agent.tools.data.fastq_fragment_manifest import load_manifest
            expected = {record.source_path: dict(path=record.source_path,
                sha256=record.source_sha256, size_bytes=record.size_bytes) for record in records}
            try:
                for step in producers:
                    # Check the actual compiled producer inputs too; a Planner may
                    # select a newly inspected manifest through an intra-plan edge.
                    self._validate_fastq_intake(records, step.resolved_arguments)
                    manifest = load_manifest(step.result['manifest_path'],
                        expected_sha256=step.result['manifest_sha256'])
                    for library in manifest['libraries']:
                        provenance = library['provenance']
                        sources = [source['resource'] for source in provenance['sources'] if source['role'] == 'fastq']
                        if provenance['kind'] != 'fastq_fragment_production' or not sources:
                            raise ValueError('Missing qualified FASTQ source provenance.')
                        if any(expected.get(source['path']) != source for source in sources):
                            raise ValueError('Producer consumed different registered bytes.')
            except (ValueError, TypeError, OSError, KeyError) as exc:
                raise _fail('LOCAL_RESOURCE_BINDING_INVALID') from exc
        # All association reads precede the final byte/snapshot check.
        self.validate_binding(binding)
