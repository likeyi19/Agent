"""Generic owner admission and dispatch, separate from scientific acceptance."""
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.orchestration.registry import (
    ErrorClassification, ResultContract, ScientificAuthorityHooks,
    SemanticPortMember, SemanticProducerPortSpec, SemanticToolSpec,
    ToolRegistry, ToolSpec, build_default_tool_registry,
)
from agent.schemas import ErrorCategory
from agent.schemas.verification_authority import AuthorityError, VerifiedArtifactAuthority
from agent.tools import scientific_authority as shared
from agent.tools.data import scientific_authority as raw


IDENTITY = 'a' * 64


def authority(execution_identity=IDENTITY):
    return VerifiedArtifactAuthority(dict(
        schema_version=2, artifact_type='reviewed-test-artifact', artifact_contract='test.v1',
        publication_path='/reviewed/manifest.json', manifest_sha256='b' * 64,
        files=[dict(path='/reviewed/manifest.json', sha256='b' * 64, size_bytes=1)],
        execution_identity=execution_identity, arguments_sha256='c' * 64,
        receipt_sha256='d' * 64, upstream={'reviewed_input': 'e' * 64},
        resources={'reviewed_proof': 'f' * 64}, science_profile='reviewed-profile.v1',
        producer_qualification={}, verifier={'id': 'reviewed-owner', 'compatibility_version': '1'},
        scope='scientific_correctness.v1', completion='succeeded',
        integrity_scope='artifact_integrity_lineage_historical_sources.v1',
        source_policy='historical_verified_sources.v1', historical_sources=[]))


@pytest.fixture
def owner():
    calls = []
    def describe(*args):
        calls.append(('describe', args))
        return {'owner': 'future-science'}
    def publication_record(*args):
        calls.append(('publication', args))
        return authority(args[4])
    def reuse_result(*args):
        calls.append(('reuse', args))
        return {'reviewed': True}
    def proof_metadata(*args):
        calls.append(('metadata', args))
        return {'owner': 'future-science-metadata'}
    def runtime(*args):
        calls.append(('runtime', args))
    hooks = ScientificAuthorityHooks('reviewed_future_kind', ('qualified',), describe,
        publication_record, reuse_result, proof_metadata, runtime)
    return hooks, calls


def spec(hooks, name='reviewed_future_tool'):
    return ToolSpec(name, lambda: {}, {}, {},
        ResultContract('reviewed-result', {'qualified_path': (str,), 'other_path': (str,)}),
        lambda exc: ErrorClassification(ErrorCategory.TOOL_EXECUTION_ERROR, 'TEST'),
        semantic_planning=SemanticToolSpec(producer_ports=(
            SemanticProducerPortSpec('qualified', 'reviewed_future.v1',
                                    (SemanticPortMember('artifact', 'qualified_path'),)),
            SemanticProducerPortSpec('unqualified', 'other_future.v1',
                                    (SemanticPortMember('artifact', 'other_path'),)),
        )), scientific_authority=hooks)


def test_hooks_are_immutable_and_ports_are_explicit(owner):
    hooks, _ = owner
    registry = ToolRegistry((spec(hooks),))
    with pytest.raises(FrozenInstanceError):
        hooks.kind = 'different'
    assert shared.supports_tool('reviewed_future_tool', registry)
    assert shared.tool_kind('reviewed_future_tool', registry) == hooks.kind
    assert shared.hooks_for_tool('reviewed_future_tool', registry) is hooks
    assert shared.qualified_port('reviewed_future_tool', 'qualified', registry)
    assert not shared.qualified_port('reviewed_future_tool', 'unqualified', registry)
    assert not shared.qualified_port('reviewed_future_tool', 'missing', registry)
    assert not shared.supports_tool('unregistered_tool', registry)
    with pytest.raises(AuthorityError, match='Unsupported'):
        shared.tool_kind('unregistered_tool', registry)


@pytest.mark.parametrize('change', [
    {'kind': ''}, {'kind': 1}, {'producer_ports': []}, {'producer_ports': ()},
    {'producer_ports': ('qualified', 'qualified')}, {'producer_ports': ('',)},
    {'describe': None}, {'publication_record': 'callable'}, {'reuse_result': None},
    {'proof_metadata': None}, {'runtime_compatibility': 1}, {'verify_sources': 1},
])
def test_hooks_reject_invalid_owner_contracts(owner, change):
    hooks, _ = owner
    with pytest.raises((TypeError, ValueError)):
        replace(hooks, **change)


def test_tools_require_declared_semantic_qualification(owner):
    hooks, _ = owner
    with pytest.raises(ValueError, match='declared producer ports'):
        spec(replace(hooks, producer_ports=('undeclared',)))
    with pytest.raises(ValueError, match='declared producer ports'):
        replace(spec(hooks), semantic_planning=None)
    with pytest.raises(TypeError, match='Invalid scientific authority'):
        replace(spec(hooks), scientific_authority={})


def test_future_owner_callbacks_dispatch_without_tool_name_branches(owner):
    hooks, calls = owner
    registry = ToolRegistry((spec(hooks),))
    context = object()
    path, sha = Path('/reviewed/manifest.json'), 'b' * 64
    description, proof, kwargs = {'manifest': {}}, object(), {'limit': 2}
    assert shared.describe(hooks.kind, path, sha, context, registry=registry) == {'owner': 'future-science'}
    assert shared.proof_metadata(hooks.kind, {'result': True}, context, registry=registry) == {
        'owner': 'future-science-metadata'}
    assert shared.reuse_result(hooks.kind, description, proof, kwargs, registry=registry) == {'reviewed': True}
    shared.runtime_compatibility(hooks.kind, description, kwargs, registry=registry)
    record = shared._publication_record(context, 'reviewed_future_tool', {}, {}, IDENTITY,
                                        {}, registry=registry)
    assert record == authority()
    assert calls == [
        ('describe', (path, sha, context)),
        ('metadata', (hooks.kind, {'result': True}, context)),
        ('reuse', (hooks.kind, description, proof, kwargs)),
        ('runtime', (hooks.kind, description, kwargs)),
        ('publication', (context, 'reviewed_future_tool', {}, {}, IDENTITY, {})),
    ]
    assert shared.needs_source_snapshot(hooks.kind, description, registry=registry)
    changed = ToolRegistry((spec(replace(hooks, verify_sources=False, runtime_compatibility=None)),))
    assert not shared.needs_source_snapshot(hooks.kind, description, registry=changed)
    assert shared.runtime_compatibility(hooks.kind, description, kwargs, registry=changed) is None


def test_distinct_future_owners_share_dispatch_without_collapsing_identity(owner):
    first, calls = owner
    second = replace(first, kind='another_reviewed_kind', describe=lambda *args: {'owner': 'second'})
    registry = ToolRegistry((spec(first), spec(second, 'another_reviewed_tool')))
    assert shared.tool_kind('reviewed_future_tool', registry) == first.kind
    assert shared.tool_kind('another_reviewed_tool', registry) == second.kind
    assert shared.describe(first.kind, '/first', None, None, registry=registry) == {'owner': 'future-science'}
    assert shared.describe(second.kind, '/second', None, None, registry=registry) == {'owner': 'second'}
    assert calls == [('describe', ('/first', None, None))]


def test_authority_issuance_requires_exact_completed_owner_proof(owner):
    hooks, calls = owner
    registry = ToolRegistry((spec(hooks),))
    result = {'manifest_path': '/reviewed/manifest.json', 'manifest_sha256': 'b' * 64}
    proof = SimpleNamespace(kind=hooks.kind, result_metadata={'scientific': 'checked'})
    locators = []
    def describe(*args):
        locators.append(args)
        return {}, 'proof-identity'
    context = SimpleNamespace(describe=describe, proofs={})
    with pytest.raises(AuthorityError, match='completed independent'):
        shared.issue(context, 'reviewed_future_tool', {}, result, IDENTITY, registry=registry)
    context.proofs['proof-identity'] = SimpleNamespace(kind='different_owner', result_metadata={})
    with pytest.raises(AuthorityError, match='completed independent'):
        shared.issue(context, 'reviewed_future_tool', {}, result, IDENTITY, registry=registry)
    context.proofs['proof-identity'] = proof
    assert shared.issue(context, 'reviewed_future_tool', {}, result, IDENTITY, registry=registry) == authority()
    assert locators == [(hooks.kind, Path(result['manifest_path']), result['manifest_sha256'])] * 3
    assert calls == [('publication', (context, 'reviewed_future_tool', {}, result, IDENTITY, proof.result_metadata))]


def test_owner_cannot_return_unbound_or_incomplete_authority(owner):
    hooks, _ = owner
    for callback in (lambda *args: {}, lambda *args: authority('0' * 64)):
        registry = ToolRegistry((spec(replace(hooks, publication_record=callback)),))
        with pytest.raises(AuthorityError, match='scientific execution authority'):
            shared._publication_record(object(), 'reviewed_future_tool', {}, {}, IDENTITY,
                                       {}, registry=registry)


def test_duplicate_and_shadowing_owner_kinds_fail_closed(owner):
    hooks, _ = owner
    with pytest.raises(AuthorityError, match='multiple registered owners'):
        ToolRegistry((spec(hooks), spec(hooks, 'second_future_tool')))
    for kind in ('matrix', 'generic_fragments', 'explicit_cells', 'qc'):
        with pytest.raises(AuthorityError, match='replace an existing'):
            ToolRegistry((spec(replace(hooks, kind=kind)),))
    with pytest.raises(AuthorityError, match='replace an existing'):
        ToolRegistry((spec(hooks, 'compute_scATAC_qc'),))


@pytest.mark.parametrize('operation', ['describe', 'reuse_result', 'runtime_compatibility',
                                       'proof_metadata', 'needs_source_snapshot'])
def test_unregistered_artifact_kind_never_gets_generic_trust(operation):
    registry = ToolRegistry(())
    arguments = {'describe': (Path('/unused'), None, None),
        'reuse_result': ({}, object(), {}), 'runtime_compatibility': ({}, {}),
        'proof_metadata': ({}, None), 'needs_source_snapshot': ({},)}
    with pytest.raises(AuthorityError, match='Unsupported'):
        getattr(shared, operation)('unqualified_kind', *arguments[operation], registry=registry)


def test_raw_owner_fallback_preserves_dispatch_and_qualification(monkeypatch):
    registry = ToolRegistry(())
    calls = []
    for name in ('describe', 'reuse_result', 'runtime_compatibility', '_publication_record', 'issue'):
        def observe(*args, name=name):
            calls.append((name, args))
            return name
        monkeypatch.setattr(raw, name, observe)
    assert shared.describe('qc', '/q', 'b' * 64, 'context', registry=registry) == 'describe'
    assert shared.reuse_result('qc', {}, 'proof', {}, registry=registry) == 'reuse_result'
    assert shared.runtime_compatibility('qc', {}, {}, registry=registry) == 'runtime_compatibility'
    assert shared._publication_record('context', 'compute_scATAC_qc', {}, {}, IDENTITY, {},
                                      registry=registry) == '_publication_record'
    assert shared.issue('context', 'compute_scATAC_qc', {}, {}, IDENTITY, registry=registry) == 'issue'
    assert [name for name, args in calls] == [
        'describe', 'reuse_result', 'runtime_compatibility', '_publication_record', 'issue']
    assert shared.tool_kind('compute_scATAC_qc', registry) == 'qc'
    assert shared.supports_tool('build_scATAC_cell_by_ccre', registry)
    assert not shared.supports_tool('cluster_cells', registry)
    assert not shared.supports_tool('compute_cell_umap', registry)
    actual = build_default_tool_registry()
    assert shared.qualified_port('compute_scATAC_qc', 'barcode_qc', actual)
    assert not shared.qualified_port('compute_scATAC_qc', 'missing', actual)


def test_raw_proof_metadata_and_source_snapshot_policies_are_preserved():
    registry = ToolRegistry(())
    contigs = (('chr1', 100),)
    generic = SimpleNamespace(contigs=contigs)
    producer = SimpleNamespace(fragments=generic)
    for kind, result in (('generic_fragments', generic),
                         ('bam_fragment_production', producer),
                         ('external_fragment_adoption', producer)):
        assert shared.proof_metadata(kind, result, None, registry=registry) == {'contigs': contigs}
    for kind in ('matrix', 'annotation', 'epizoo_adaptation'):
        result = {'scientific': 'metadata'}
        assert shared.proof_metadata(kind, result, None, registry=registry) is result
    for kind in ('qc', 'selection', 'explicit_cells'):
        assert shared.proof_metadata(kind, {}, None, registry=registry) == {}
    assert shared.needs_source_snapshot('bam_fragment_production', {}, registry=registry)
    assert not shared.needs_source_snapshot('generic_fragments', {}, registry=registry)
    assert not shared.needs_source_snapshot('qc', {}, registry=registry)
    assert shared.needs_source_snapshot('matrix', {'manifest': {
        'contract_version': 'scatac-cell-by-ccre.external.v1'}}, registry=registry)
    assert not shared.needs_source_snapshot('matrix', {'manifest': {
        'contract_version': 'scatac-cell-by-ccre.v1'}}, registry=registry)
    description = {'manifest_path': '/exact/manifest.json', 'manifest_sha256': 'b' * 64}
    locators = []
    def describe(*args):
        locators.append(args)
        return {}, 'exact-generic-key'
    context = SimpleNamespace(describe=describe, proofs={})
    with pytest.raises(AuthorityError, match='exact verified publication locator'):
        shared.proof_metadata('fastq_fragment_production', {}, context, registry=registry)
    with pytest.raises(AuthorityError, match='canonical integrity proof'):
        shared.proof_metadata('fastq_fragment_production', {}, context, description=description, registry=registry)
    context.proofs['exact-generic-key'] = SimpleNamespace(result_metadata={'contigs': contigs})
    assert shared.proof_metadata('fastq_fragment_production', {}, context,
                                description=description, registry=registry) == {'contigs': contigs}
    assert locators == [('generic_fragments', Path('/exact/manifest.json'), 'b' * 64)] * 2
