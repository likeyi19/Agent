"""Current FASTQ v2 boundary and explicit retirement of transitional v1."""
import hashlib
import json
from pathlib import Path
import pytest
from agent.orchestration import build_default_tool_registry
from agent.tools.data import scatac_fragment_reader as reader, scatac_fragments as public
from agent.tools.data import scatac_fragments_v2 as v2, fastq_fragment_manifest as producer
from agent.tools.data.fastq_fragments_verifier import verify_fragments
from agent.tools.data._fragments_common import FragmentsError
from fragments_helpers import BC
from test_fragments_contracts import bound, fake_runtime, published


def test_fastq_v2_common_records_and_full_provenance(published):
    result, runtime, control = published
    path = Path(result['manifest_path']); original = path.read_bytes()
    view = reader.open_verified_fragments(path, expected_sha256=result['manifest_sha256'], runtime=runtime)
    assert result['artifact_schema_version'] == 2 and view.contract_version == 'scatac-fragments.v2'
    assert list(view.iter_fragments('library_0')) == [reader.FragmentRecord('library_0','chrTiny',104,195,BC,300,None)]
    p = view.libraries[0].provenance
    assert p['kind'] == 'fastq_fragment_production'
    record = producer.load_record(p['producer_record']['path'], expected_sha256=p['producer_record']['sha256'])
    assert record['contract_version'] == 'fastq-fragment-production.v1'
    assert record['libraries'][0]['scans'] and record['libraries'][0]['whitelist_sha256']
    assert record['lineage']['index_identity'] and record['backend']['executable_sha256']
    assert record['mapping_policy'] and record['packaging_policy']
    assert view.libraries[0].support_meaning == {'unit':'paired_mappings','definition':record['semantics']['support']}
    assert view.verification.producer_profile_qualification == 'not_established'
    assert verify_fragments(path, runtime=runtime, expected_sha256=result['manifest_sha256']) == view.manifest
    assert original == path.read_bytes() and control['calls'].count('chromap') == 1


def test_current_tool_contract_and_policy(published):
    result, _, _ = published
    registry = build_default_tool_registry(); spec = registry.get('prepare_scATAC_fragments')
    registry.validate_result(spec.name, result)
    assert len(registry.names()) == 16
    assert spec.recovery_policy_version == public.RECOVERY_POLICY == 'prepare-scatac-fragments-fastq-v2'
    assert spec.semantic_planning.producer_ports[0].semantic_type == 'scatac_fragments.v2'
    with pytest.raises(ValueError):
        registry.validate_result(spec.name, result | {'artifact_schema_version':1,'contract_version':'scatac-fragments.v1'})


@pytest.mark.parametrize('interface', ['reader','fastq','generic'])
def test_retired_v1_rejected_without_rewrite(tmp_path, monkeypatch, interface):
    from agent.tools.data import scatac_fragments_v2_verifier as generic
    path = tmp_path/'historical.json'
    payload = b'{"artifact_type":"agent.scatac-fragments","schema_version":1,"contract_version":"scatac-fragments.v1"}'
    path.write_bytes(payload); sha = hashlib.sha256(payload).hexdigest()
    def forbidden(*args, **kwargs): pytest.fail('Retired artifact triggered execution')
    monkeypatch.setattr(public, 'resolve_execution', forbidden)
    if interface == 'reader': call = reader.open_verified_fragments
    elif interface == 'fastq': call = verify_fragments
    else: call = generic.verify_fragments_v2
    with pytest.raises(ValueError, match='CONTRACT_RETIRED'):
        call(path, expected_sha256=sha, runtime=None)
    assert path.read_bytes() == payload and list(tmp_path.iterdir()) == [path]


def test_old_receipt_policy_is_not_reinterpreted(tmp_path):
    from agent.tools.data._fragments_common import canonical, digest
    args = {'output_dir':str(tmp_path)}
    destination, token, args_sha = public._publication(args, 'execution')
    destination.mkdir()
    payload = canonical(dict(artifact_type='agent.fragments-execution-receipt',schema_version=1,
        policy='prepare-scatac-fragments-fastq-v1',execution_identity=token,
        arguments_sha256=args_sha,manifest_sha256='0'*64))
    receipt = destination/'receipt.json'; receipt.write_bytes(payload)
    with pytest.raises(FragmentsError, match='RECOVERY_MISMATCH'):
        public.recover_fragments(args, 'execution')
    assert receipt.read_bytes() == payload


def test_reader_does_not_qualify_unknown_producer_science(published):
    result,runtime,_ = published
    view = reader.open_verified_fragments(result['manifest_path'], expected_sha256=result['manifest_sha256'], runtime=runtime)
    assert view.verification.artifact_content == 'verified'
    assert view.verification.bound_resource_identities == 'verified'
    assert view.verification.producer_profile_qualification == 'not_established'
