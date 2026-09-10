"""Producer record cross-bindings remain stronger than generic v2 integrity."""
from pathlib import Path
import pytest
from agent.tools.data import fastq_fragment_manifest as m, scatac_fragments_v2 as v2
from agent.tools.data.fastq_fragments_verifier import verify_fragments
from agent.tools.data.scatac_fragments_v2_verifier import verify_fragments_v2
from agent.tools.data._chromap import sha256
from test_fragments_contracts import bound, fake_runtime, published


def test_v2_source_budget_fails_before_alignment(tiny, bound, fake_runtime, monkeypatch):
    from agent.tools.data.fastq_fragments import prepare_fastq_fragments
    inputs, group, _ = bound; runtime, control = fake_runtime
    monkeypatch.setattr(v2, 'MAX_SOURCES', len(group.files) + 1)
    with pytest.raises(ValueError, match='FRAGMENTS_V2_CONTRACT_INVALID'):
        prepare_fastq_fragments(inputs=inputs, output_dir=tiny['root']/'over-budget', runtime=runtime)
    assert control['calls'] == []
    assert not (tiny['root']/'over-budget').exists()


@pytest.mark.parametrize('field',['decoded_role','normalized_ids','record_count','whitelist','role_binding','index_identity'])
def test_self_consistent_v2_cannot_forge_fastq_provenance(published, field):
    result,runtime,_=published;path=Path(result['manifest_path']);root=path.parent
    record=m.load_record(root/'production.json');entry=record['libraries'][0]
    scan=next(iter(entry['scans'].values()))
    if field=='decoded_role':scan['decoded_role_sha256'][0]='0'*64
    elif field=='normalized_ids':scan['normalized_id_sha256']='0'*64
    elif field=='record_count':scan['record_count']+=1
    elif field=='whitelist':entry['whitelist_sha256']='0'*64
    elif field=='role_binding':entry['role_binding_sha256']='0'*64
    elif field=='index_identity':record['lineage']['index_identity']='0'*64
    (root/'production.json').write_bytes(m.canonical_record_bytes(record))
    value=m.build_manifest(record,root);path.write_bytes(v2.canonical_fragments_manifest_v2_bytes(value))
    # Resource bytes and stream integrity are valid, but producer claims are false.
    verify_fragments_v2(path,expected_sha256=sha256(path),runtime=runtime)
    with pytest.raises(ValueError,match='MISMATCH'):
        verify_fragments(path,expected_sha256=sha256(path),runtime=runtime)


@pytest.mark.parametrize('field',['backend','mapping_policy','semantics','packaging_policy','profile_sha256'])
def test_frozen_fastq_policies_cannot_be_changed(published, field):
    result,_,_=published;record=m.load_record(Path(result['manifest_path']).parent/'production.json')
    if field=='profile_sha256':record[field]='0'*64
    else:record[field]['unreviewed']='changed'
    with pytest.raises(ValueError):m.validate_record(record)


def test_record_does_not_accept_obsolete_fragment_artifact(published):
    result,_,_=published;record=m.load_record(Path(result['manifest_path']).parent/'production.json')
    record['artifact_type']='agent.scatac-fragments';record['contract_version']='scatac-fragments.v1'
    with pytest.raises(ValueError):m.validate_record(record)
