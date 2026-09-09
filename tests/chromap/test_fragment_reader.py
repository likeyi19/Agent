"""Compatibility through the real v1 domain/verifier with external stages faked."""
import hashlib
import json
from pathlib import Path

import pytest

from agent.orchestration import build_default_tool_registry
from agent.tools.data import scatac_fragment_reader as reader
from agent.tools.data import scatac_fragments_manifest as manifest
from agent.tools.data import scatac_fragments_verifier as verifier
from agent.tools.data import scatac_fragments as public
from agent.tools.data.scatac_fragments_v2_verifier import FragmentVerificationRuntime
from agent.report.fragments import SUPPORT_DEFINITION
from fragments_helpers import BC
from test_fragments_contracts import bound, fake_runtime, published


def test_existing_v1_stream_and_exact_metadata_remain_v1(published):
    result, runtime, control = published
    path = Path(result['manifest_path']); original = path.read_bytes()
    legacy = manifest.load_fragments_manifest(path, expected_sha256=result['manifest_sha256'])
    neutral_runtime = FragmentVerificationRuntime(bgzip=runtime.bgzip, tabix=runtime.tabix, sort=runtime.sort)
    view = reader.open_verified_fragments(path, expected_sha256=result['manifest_sha256'], runtime=neutral_runtime)
    assert view.contract_version == 'scatac-fragments.v1'
    assert view.manifest == legacy
    assert view.verification.producer_profile_qualification == 'legacy_qualified_chromap_policy'
    assert view.verification.bound_resource_identities == 'legacy_v1_defined_checks'
    assert view.verification.producer_history == 'not_verified'
    rows = list(view.iter_fragments('library_0'))
    assert rows == [reader.FragmentRecord('library_0', 'chrTiny', 104, 195, BC, 300, None)]
    assert rows[0].cell_identity == ('library_0', BC)
    assert view.libraries[0].support_meaning['definition'] == legacy['semantics']['support']
    assert view.libraries[0].provenance['mapping_policy'] == legacy['mapping_policy']
    assert path.read_bytes() == original and hashlib.sha256(original).hexdigest() == result['manifest_sha256']
    assert verifier.verify_fragments(path, runtime=runtime, expected_sha256=result['manifest_sha256']) == legacy
    assert control['calls'].count('chromap') == 1


def test_reader_reuses_exact_v1_verifier_and_never_v2(published, monkeypatch):
    from agent.tools.data import scatac_fragments_v2_verifier as v2
    result, runtime, _ = published; original = verifier.verify_fragments; calls = []
    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(verifier, 'verify_fragments', counted)
    monkeypatch.setattr(v2, 'verify_fragments_v2', lambda *a, **k: pytest.fail('V1 migrated to V2'))
    reader.open_verified_fragments(result['manifest_path'], expected_sha256=result['manifest_sha256'], runtime=runtime)
    assert calls == [True]


@pytest.mark.parametrize('encoding', ['utf-8', 'utf-16', 'utf-8-sig'])
def test_v1_noncanonical_manifest_bytes_are_not_rewritten(published, encoding):
    result, runtime, _ = published; path = Path(result['manifest_path'])
    legacy = json.loads(path.read_bytes())
    payload = json.dumps(legacy, indent=2).encode(encoding)
    path.write_bytes(payload); digest = hashlib.sha256(payload).hexdigest()
    assert verifier.verify_fragments(path, expected_sha256=digest, runtime=runtime) == legacy
    view = reader.open_verified_fragments(path, expected_sha256=digest, runtime=runtime)
    assert view.verification.manifest_bytes == payload
    assert view.verification.manifest_sha256 == digest and path.read_bytes() == payload


def test_v1_whitelist_drift_still_fails(published, bound):
    result, runtime, _ = published; _, _, white = bound
    Path(white.resource.path).write_text('T' * 16 + '\n')
    with pytest.raises(ValueError):
        reader.open_verified_fragments(result['manifest_path'], expected_sha256=result['manifest_sha256'], runtime=runtime)


def test_v1_tool_result_and_projection_authorities_remain_fixed(published):
    result, _, _ = published
    registry = build_default_tool_registry()
    spec = registry.get('prepare_scATAC_fragments')
    registry.validate_result(spec.name, result)
    assert len(registry.names()) == 13
    assert spec.recovery_policy_version == public.RECOVERY_POLICY == 'prepare-scatac-fragments-fastq-v1'
    assert spec.semantic_planning.producer_ports[0].semantic_type == 'scatac_fragments.v1'
    assert not spec.optional_arguments
    assert SUPPORT_DEFINITION == (
        'Exact number of Chromap-generated paired mappings assigned to the accepted '
        'cell-level duplicate group under chromap-atac-agent-support-v1, including '
        'the retained representative.')


def test_future_fastq_v2_can_bind_complete_legacy_provenance(published, bound, monkeypatch):
    """Representability fixture only: no production adapter or migration API."""
    from agent.tools.data import scatac_fragments_v2 as m
    from agent.tools.data import scatac_fragments_v2_verifier as v
    from agent.tools.data._fragments_common import canonical

    result, runtime, control = published
    inputs, group, _ = bound
    old_path = Path(result['manifest_path']); original = old_path.read_bytes()
    legacy = manifest.load_fragments_manifest(old_path, expected_sha256=result['manifest_sha256'])

    def resource(path):
        path = Path(path)
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    size_bytes=path.stat().st_size)

    # The accepted closed v1 record is a concrete witness that the opaque bound
    # producer-record slot can retain every field, including decoded-role scans,
    # qualified backend, index, whitelist, namespaces and packaging identities.
    record = old_path.with_name('synthetic-producer-record.json'); record.write_bytes(original)
    profile = old_path.with_name('synthetic-profile.json')
    profile.write_bytes(canonical(legacy['backend']))
    sources = [{'role': 'fastq', 'resource': resource(path)} for _, path in group.files]
    sources += [{'role': role, 'resource': resource(path)} for role, path in (
        ('intake_manifest', inputs.intake_path), ('library_context', inputs.context_path))]
    sources.sort(key=lambda s: (s['role'], s['resource']['sha256'], s['resource']['path']))
    provenance = dict(kind='fastq_fragment_production', contract_version='fragment-producer-provenance.v1',
        profile={'id': legacy['backend']['backend_policy'], 'resource': resource(profile)},
        sources=sources, producer_record=resource(record),
        support={'unit': 'paired_mappings', 'definition': legacy['semantics']['support']},
        processing={key: {'status': 'declared', 'description': legacy['semantics']['tn5'] if key == 'tn5'
            else 'Exact policy retained in the bound producer record and its context.'} for key in m.PROCESSING_KEYS},
        source_selection='all_source_records', verification_requirement='producer-specific-verifier-required.v1')
    entry = legacy['libraries'][0]
    value = dict(artifact_type=m.ARTIFACT_TYPE, schema_version=2, contract_version=m.CONTRACT_VERSION,
        reference=dict(manifest_path=inputs.reference_path, manifest_sha256=inputs.reference_sha256,
            reference_identity_sha256=legacy['lineage']['reference_identity'],
            species=legacy['lineage']['species'], assembly=legacy['lineage']['assembly'],
            ordered_contig_sha256=legacy['lineage']['ordered_contig_sha256']), semantics=m.SEMANTICS,
        libraries=[{key: entry[key] for key in ('namespace', 'bgzf', 'tabix', *m.SUMMARY_KEYS)} |
                   {'strand': {'mode': 'absent', 'definition': None}, 'provenance': provenance}])
    value['fragments_identity_sha256'] = m.fragments_identity(value)
    payload = m.canonical_fragments_manifest_v2_bytes(value)
    path = old_path.with_name('synthetic-v2.json'); path.write_bytes(payload)
    monkeypatch.setattr(v, 'verify_packaging', lambda _: None)
    monkeypatch.setattr(v, '_query', verifier._query)
    view = reader.open_verified_fragments(path, expected_sha256=hashlib.sha256(payload).hexdigest(), runtime=runtime)
    retained = view.libraries[0].provenance['producer_record']
    assert manifest.load_fragments_manifest(retained['path'], expected_sha256=retained['sha256']) == legacy
    assert Path(retained['path']).read_bytes() == original
    assert view.libraries[0].provenance == provenance
    assert view.libraries[0].support_meaning['definition'] == legacy['semantics']['support']
    assert list(view.iter_fragments('library_0')) == [reader.FragmentRecord('library_0', 'chrTiny', 104, 195, BC, 300, None)]
    assert view.verification.producer_profile_qualification == 'not_established'
    assert view.verification.producer_history == 'not_verified'
    assert old_path.read_bytes() == original and control['calls'].count('chromap') == 1
    record.write_bytes(original + b' ')
    with pytest.raises(m.FragmentsV2Error, match='RESOURCE_MISMATCH'):
        v.verify_fragments_v2(path, expected_sha256=hashlib.sha256(payload).hexdigest(), runtime=runtime)
