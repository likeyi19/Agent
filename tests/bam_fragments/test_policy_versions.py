"""Real-world containment/TLEN conventions, exact v1 readers and v2 publication."""
import hashlib
import json
from pathlib import Path

import pytest
from .conftest import pair, sha
from agent.tools.data import bam_fragment_manifest as m, bam_fragments as producer
from agent.tools.data import bam_fragments_verifier as verifier, scatac_bam_fragments as owner
from agent.tools.data.scatac_fragments_v2_verifier import FragmentVerificationRuntime
from .test_science import output


def contained(tlen=39):
    # Same geometry as genuine Cell Ranger ATAC PBMC500 pairs. Three-prime
    # overhangs do not change the inward-facing five-prime endpoints.
    return pair(a=dict(pos=20,cigar='50M',mpos=10,tlen=tlen),
                b=dict(pos=10,cigar='49M',mpos=20,tlen=-tlen))


@pytest.mark.parametrize('tlen',[39,60,0])
def test_containment_counts_from_cigar_not_tlen(bam_factory,tlen):
    _,rows,record=output(bam_factory(contained(tlen)))
    assert rows==['chr2\t24\t54\taCgT-1\t1']
    assert record['contract_version']=='bam-fragment-production.v2'
    assert record['qualification']['eligible_pairs']==1


def test_legacy_profile_identities_remain_exact():
    assert hashlib.sha256(m.profile_bytes(m.LEGACY_PROFILE_ID)).hexdigest()=='79647fb6e37b89df21405765d058c20cdbe4a2e9b43df310da0c38fe9292a8e0'
    assert hashlib.sha256(m.profile_bytes(m.LEGACY_NEUTRAL_PROFILE_ID)).hexdigest()=='ed57f1a5a5ad7494cdc56b7804546bbcc2c6634e53c1635726d15029120e70bf'


def test_legacy_publication_is_readable_but_cannot_execute(bam_factory,tmp_path):
    args=bam_factory();args['source_profile']=m.LEGACY_PROFILE_ID
    # Construct a historical-format fixture using the shared strict v1 policy.
    stage=tmp_path/'legacy';stage.mkdir();runtime=FragmentVerificationRuntime()
    value,_=producer.prepare_in_stage(args,stage,runtime)
    manifest=stage/'manifest.json'
    verifier.verify_bam_fragments(manifest,expected_sha256=sha(manifest),runtime=runtime)
    with pytest.raises(m.BamFragmentsError,match='PROFILE_READ_ONLY'):
        owner.execute_bam_fragments(args)


def test_v1_cannot_acquire_new_containment_authority(bam_factory,tmp_path):
    args=bam_factory(contained());args['source_profile']=m.LEGACY_PROFILE_ID
    stage=tmp_path/'legacy';stage.mkdir()
    with pytest.raises(m.BamFragmentsError,match='TLEN_MISMATCH'):
        producer.prepare_in_stage(args,stage,FragmentVerificationRuntime())


def test_forged_v1_record_is_still_reconstructed_with_old_rules(bam_factory,tmp_path):
    args=bam_factory(contained())
    stage=tmp_path/'forged';stage.mkdir();runtime=FragmentVerificationRuntime()
    value,_=producer.prepare_in_stage(args,stage,runtime)
    # Relabelling a valid v2 result as v1 must not launder the new policy.
    record=json.loads((stage/'production.json').read_text())
    record['arguments']['source_profile']=m.LEGACY_PROFILE_ID
    record['contract_version']=m.LEGACY_PRODUCTION_CONTRACT
    record['profile_sha256']=hashlib.sha256(m.profile_bytes(m.LEGACY_PROFILE_ID)).hexdigest()
    (stage/'profile.json').write_bytes(m.profile_bytes(m.LEGACY_PROFILE_ID))
    from agent.tools.data._fragments_common import canonical
    from agent.tools.data._bam_fragment_io import resource
    (stage/'production.json').write_bytes(canonical(record))
    value['libraries'][0]['provenance']=m.provenance(record,resource(stage/'profile.json'),resource(stage/'production.json'))
    from agent.tools.data import scatac_fragments_v2 as v2
    value['fragments_identity_sha256']=v2.fragments_identity(value)
    (stage/'manifest.json').write_bytes(v2.canonical_fragments_manifest_v2_bytes(value))
    with pytest.raises(m.BamFragmentsError,match='TLEN_MISMATCH'):
        verifier.verify_bam_fragments(stage/'manifest.json',expected_sha256=sha(stage/'manifest.json'),runtime=runtime)
