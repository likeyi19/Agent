"""Durable, byte-pinned local sources; registration confers no scientific authority.

Sources remain at their operator-approved locations. New consumption verifies the
pinned bytes; reading historical attribution does not require the source to exist.
This boundary shares the application's trusted-local filesystem lifetime contract.
"""
from contextlib import contextmanager
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
from .workspace import ManagedWorkspace


MAX_RECORD_BYTES = 16_384
MAX_INPUT_BYTES = 65_536
_FORMAT = 'agent.local-resource.v1'
_ID = re.compile(r'local-[0-9a-f]{64}')
_SHA = re.compile(r'[0-9a-f]{64}')
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
        if self.input_type != 'h5ad':
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
        if type(self.tool_name) is not str or self.tool_name not in {'inspect_scATAC', 'adopt_scATAC_cell_by_ccre'}:
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
        if input_type != 'h5ad':
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
        source = {'inspect_scATAC': dict(input_path=record.source_path),
                  'adopt_scATAC_cell_by_ccre': dict(source_path=record.source_path, source_sha256=record.source_sha256)}
        if type(tool_name) is not str or tool_name not in source:
            raise _fail('LOCAL_RESOURCE_OPERATION_UNSUPPORTED')
        values = _input_mapping(scientific_inputs)
        if self._registry is None:
            from agent.orchestration.registry import build_default_tool_registry
            self._registry = build_default_tool_registry()
        try:
            specification = self._registry.get(tool_name)
            owned = {'path'} if tool_name == 'inspect_scATAC' else set(source[tool_name])
            required = set(specification.required_arguments) - owned - {'output_dir'}
            allowed = (set(specification.required_arguments) | set(specification.optional_arguments)) - owned - {'output_dir'}
            if set(values) - allowed:
                raise ValueError('Unknown scientific input declaration.')
        except (ValueError, TypeError, LookupError) as exc:
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID') from exc
        if required - set(values):
            raise _fail('LOCAL_RESOURCE_DECLARATION_REQUIRED')
        return values | source[tool_name]

    def resolve(self, resource_id, *, tool_name, scientific_inputs=None):
        record = self.load(resource_id)
        inputs = self._binding_inputs(record, tool_name, scientific_inputs)
        binding = RegisteredInput(record.resource_id, record.record_sha256, tool_name, inputs)
        self.validate_binding(binding)
        return binding

    def validate_binding(self, binding, *, verify_source=True):
        if type(binding) is not RegisteredInput or type(verify_source) is not bool:
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID')
        record = self.load(binding.resource_id)
        if binding.record_sha256 != record.record_sha256:
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID')
        values = _serialize(binding.execution_inputs)
        fields = ('input_path',) if binding.tool_name == 'inspect_scATAC' else ('source_path', 'source_sha256')
        declarations = {key: value for key, value in values.items() if key not in fields}
        try:
            expected = self._binding_inputs(record, binding.tool_name, declarations)
        except ResourceAdmissionError as exc:
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID') from exc
        if values != expected:
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID')
        if verify_source and self._source(record.source_path, integrity=True) != (record.source_path, record.source_sha256, record.size_bytes):
            raise _fail('LOCAL_RESOURCE_INTEGRITY_INVALID')
        return values

    def validate_submission(self, submission, *, verify_source=True):
        try:
            metadata = submission['registered_input']
            if set(metadata) != {'resource_id', 'record_sha256', 'tool_name'}:
                raise ValueError('Invalid registered input attribution.')
            binding = RegisteredInput(**metadata, execution_inputs=submission['execution_inputs'])
        except (TypeError, ValueError, KeyError) as exc:
            raise _fail('LOCAL_RESOURCE_BINDING_INVALID') from exc
        self.validate_binding(binding, verify_source=verify_source)
