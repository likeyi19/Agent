"""Operation-local scientific proofs and accepted artifact authorities.

Only orchestration opens this scope. Standalone and recovery verifiers have no
scope and retain deep behavior. Proofs are created only after the real verifier
returns and its complete non-source closure remains stable.
"""
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

from agent.schemas.verification_authority import AuthorityError, digest
from agent.schemas.orchestration import freeze_json_mapping, _serialize
from ._fragments_common import snapshot

_CURRENT = ContextVar('scientific_authority_operation', default=None)


def current():
    return _CURRENT.get()


@contextmanager
def authority_operation(context):
    token = _CURRENT.set(context)
    try:
        yield context
    finally:
        _CURRENT.reset(token)


@dataclass(frozen=True)
class ScientificProof:
    identity: str
    kind: str
    description: object
    result_metadata: object


class VerificationContext:
    def __init__(self):
        self.proofs = {}
        self.locations = {}
        self.accepted = {}
        self._hashes = {}
        self.validate_anchors = lambda: None
        self.execution = None
        self.defer_hash_cancellation = False

    def register_execution(self, kind, output_dir, identity):
        self.execution = (kind, Path(output_dir), identity)

    def execution_for(self, kind, path):
        # Canonical integrity is implied by a qualified producer, but never
        # identifies that producer or borrows a downstream execution identity.
        if kind == 'generic_fragments':
            return 'canonical_integrity_dependency'
        for (name, _), accepted in self.accepted.items():
            if name == str(path):
                return accepted['record']['execution_identity']
        if self.execution is not None:
            owner_kind, root, identity = self.execution
            if kind == owner_kind and Path(path).is_relative_to(root):
                return identity
        return 'independently_verified_dependency'

    def files(self, paths):
        from .fastq_verification_authority import file_closure
        entries = []
        for name in sorted(set(map(str, paths))):
            key = (name, snapshot(name))
            if key not in self._hashes:
                # Authority bookkeeping must not add cancellation boundaries to
                # the existing producer/publication lifecycle. Owned scientific
                # work retains its original cooperative checkpoints.
                from agent.tools._cancellation import cancellation_scope
                boundary = cancellation_scope(None) if self.defer_hash_cancellation else nullcontext()
                with boundary:
                    entry = file_closure([name])[0]
                if key != (name, snapshot(name)):
                    raise AuthorityError('File changed during authority validation.')
                self._hashes[key] = freeze_json_mapping(entry, 'file_binding')
            entries.append(_serialize(self._hashes[key]))
        return entries

    def describe(self, kind, path, sha):
        from .scientific_authority import describe
        description = describe(kind, path, sha, self)
        accepted = self.accepted.get((str(path), description['manifest_sha256']))
        if accepted is not None:
            expected = _serialize(accepted['record']['files'])
            if self.files(f['path'] for f in expected) != expected:
                raise AuthorityError('Accepted publication/receipt integrity changed.')
        key = digest(description['proof_identity'])
        location = (kind, str(path))
        if location in self.locations and self.locations[location] != (key, description['manifest_sha256']):
            raise AuthorityError('Previously verified artifact or dependency changed.')
        return description, key

    def verify(self, kind, function, args, kwargs):
        from .scientific_authority import reuse_result, runtime_compatibility
        self.validate_anchors()
        path = Path(args[0] if args else kwargs.get('path', kwargs.get('manifest_path')))
        sha = kwargs.get('expected_sha256')
        description, key = self.describe(kind, path, sha)
        runtime_compatibility(kind, description, kwargs)
        custom_limits = False
        if kind == 'matrix':
            from ._cell_by_ccre_io import MatrixLimits
            custom_limits = kwargs.get('limits', MatrixLimits()) != MatrixLimits()
        if key in self.proofs and not custom_limits:
            self.locations[kind, str(path)] = (key, description['manifest_sha256'])
            self.validate_anchors()
            return reuse_result(kind, description, self.proofs[key], kwargs)
        # Source freshness is owned here, not by later downstream consumption.
        from .fragments_authority_contract import CONTRACTS
        sources = description['historical_sources'] if kind in CONTRACTS else []
        paths = [f['path'] for f in description['files'] + sources]
        before = {p: snapshot(p) for p in paths}
        result = function(*args, **kwargs)
        after, after_key = self.describe(kind, path, sha)
        if key != after_key or before != {p: snapshot(p) for p in paths}:
            raise AuthorityError('Scientific verification inputs changed before completion.')
        self.validate_anchors()
        if kind == 'fastq_fragment_production':
            _, generic_key = self.describe('generic_fragments', path, sha)
            if generic_key not in self.proofs:
                raise AuthorityError('Producer verification lacks canonical integrity proof.')
            metadata = dict(contigs=self.proofs[generic_key].result_metadata['contigs'])
        elif kind == 'generic_fragments':
            metadata = dict(contigs=result.contigs)
        elif kind in ('bam_fragment_production', 'external_fragment_adoption'):
            metadata = dict(contigs=result.fragments.contigs)
        elif kind == 'matrix':
            metadata = result
        else:
            metadata = {}
        proof = ScientificProof(key, kind, freeze_json_mapping(after, 'scientific_proof'),
                                freeze_json_mapping(metadata, 'proof_metadata'))
        self.proofs[key] = proof
        self.locations[kind, str(path)] = (key, description['manifest_sha256'])
        return result

    def moved(self, source, destination):
        source, destination = Path(source), Path(destination)
        for (kind, name), (key, _) in tuple(self.locations.items()):
            path = Path(name)
            if path.is_relative_to(source):
                new = destination / path.relative_to(source)
                description, current_key = self.describe(kind, new, None)
                if current_key != key:
                    raise AuthorityError('Publication move changed independently verified content.')
                self.locations[kind, str(new)] = (key, description['manifest_sha256'])
                del self.locations[kind, name]


def owned_verification(kind):
    def decorate(function):
        @wraps(function)
        def verify(*args, **kwargs):
            context = current()
            if context is None:
                return function(*args, **kwargs)
            return context.verify(kind, function, args, kwargs)
        return verify
    return decorate


def publication_moved(source, destination):
    if current() is not None:
        from agent.tools._cancellation import cancellation_scope
        # Publication has committed; preserve terminal-success cancellation ordering.
        with cancellation_scope(None):
            current().moved(source, destination)


def execution_authorities(function):
    @wraps(function)
    def execute(*args, **kwargs):
        if kwargs.get('durable_run_id') is None or kwargs.get('completed_steps'):
            return function(*args, **kwargs)
        context = VerificationContext()
        if context is not None:
            context.defer_hash_cancellation = True
        with authority_operation(context):
            return function(*args, **kwargs)
    return execute
