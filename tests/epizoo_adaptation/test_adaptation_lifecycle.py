"""Publication/authority tests with an explicitly mocked scientific owner."""
from pathlib import Path
import pytest
from agent.tools.models.epizoo_adaptation import publication as p,contract as c
from agent.tools.data.authority_context import VerificationContext,authority_operation,owned_verification
from agent.tools.data.scientific_authority import issue
from agent.schemas.verification_authority import AuthorityError,VerifiedArtifactAuthority
from agent.tools._cancellation import cancellation_scope,ToolWorkCancelled


@pytest.fixture
def lifecycle(tmp_path,matrix_factory,profile,monkeypatch):
    binding=matrix_factory();resource=tmp_path/'resource';resource.write_text('test-only mocked resource')
    bundles=[]
    for name in ('source','seam'):
        path=tmp_path/(name+'.json')
        record=c.write_json(path,dict(files=[c.file_record(resource)],code=dict(files=[])))
        bundles.append(dict(path=record['path'],sha256=record['sha256']))
    args=dict(matrices=[binding],source_bundle=bundles[0],seam_bundle=bundles[1],strategy='de_novo',mapping=None,
              profile=profile,device='cuda:0',output_dir=str(tmp_path/'adaptation'))
    calls=dict(build=0,verify=0)
    monkeypatch.setattr(p.resources,'runtime',lambda:dict(test_runtime=True))
    def science(path,*,expected_sha256):
        calls['verify']+=1
        value=p.load_manifest(path,expected_sha256)
        return p._summary(value,Path(path),expected_sha256)
    monkeypatch.setattr(p,'verify_target',owned_verification('epizoo_adaptation')(science))
    def build(args,root):
        calls['build']+=1;root.mkdir();(root/'training').mkdir()
        names=('checkpoint.pth','sequence_embeddings.npy','sequences.txt','document_frequency.npy','sentence_tokens.npy',
               'sentence_indptr.npy','cells.jsonl','features.txt','preprocessing.json','execution.json','training/training_log.csv')
        files={}
        for name in names:
            path=root/name;path.write_text('mock scientific payload')
            files[name]={k:v for k,v in c.file_record(path).items() if k!='path'}
        value=dict(artifact_type=c.ARTIFACT,contract_version=c.CONTRACT,schema_version=1,profile_sha256=c.PROFILE_SHA256,
            arguments=args,reference={},source_identity='a'*64,seam_identity='b'*64,
            preprocessing=dict(n_cells=2,n_features=3),execution=dict(completed_step=2),files=files,runtime=p.resources.runtime())
        record=c.write_json(root/'manifest.json',value)
        p.verify_target(root/'manifest.json',expected_sha256=record['sha256'])
        return dict(manifest_path=record['path'],manifest_sha256=record['sha256'])
    monkeypatch.setattr(p,'_build',build)
    return args,calls,resource


def test_owner_once_issuance_and_reuse(lifecycle):
    args,calls,_=lifecycle;context=VerificationContext()
    with authority_operation(context):
        output=p.adapt_epizoo_species(**args,execution_identity='a'*64)
        result=output['result'];record=VerifiedArtifactAuthority(output['authority'])
        p.verify_public_result(args,result)
        assert calls==dict(build=1,verify=1)
        assert record.record['schema_version']==2 and record.record['artifact_contract']==c.CONTRACT
        assert record.record['verifier']['id']=='epizoo.species-adaptation-structural'
        with pytest.raises(AuthorityError):issue(VerificationContext(),'adapt_epizoo_species',args,result,'a'*64)
        with pytest.raises(AuthorityError):issue(context,'adapt_epizoo_species',args,result,'b'*64)
        Path(result['checkpoint_path']).write_text('mutated checkpoint')
        with pytest.raises(AuthorityError):p.verify_public_result(args,result)
        assert calls==dict(build=1,verify=1)


def test_resource_mutation_invalidates_reuse(lifecycle):
    args,calls,resource=lifecycle
    with authority_operation(VerificationContext()):
        result=p.adapt_epizoo_species(**args,execution_identity='a'*64)['result']
        resource.write_text('changed')
        with pytest.raises(AuthorityError):p.verify_public_result(args,result)
        assert calls['verify']==1


def test_completed_recovery_never_retrains(lifecycle):
    args,calls,_=lifecycle
    result=p.adapt_epizoo_species(**args,execution_identity='a'*64)['result']
    assert p.recover_adaptation(args,'a'*64)==result
    assert calls==dict(build=1,verify=2)
    with pytest.raises((ValueError,FileNotFoundError)):p.recover_adaptation(args,'b'*64)
    with pytest.raises(ValueError):p.adapt_epizoo_species(**args,execution_identity='a'*64)


def test_interruption_is_not_resume(lifecycle,monkeypatch):
    args,calls,_=lifecycle
    def interrupted(args,root):root.mkdir();raise ToolWorkCancelled()
    monkeypatch.setattr(p,'_build',interrupted)
    with pytest.raises(ToolWorkCancelled):p.adapt_epizoo_species(**args,execution_identity='a'*64)
    assert not list(Path(args['output_dir']).glob('epizoo-adaptation-*'))
    assert not list(Path(args['output_dir']).glob('.matrix-attempt-*'))
    with pytest.raises(FileNotFoundError):p.recover_adaptation(args,'a'*64)


def test_cancel_before_publication(lifecycle):
    args,calls,_=lifecycle
    with cancellation_scope(lambda:True),pytest.raises(ToolWorkCancelled):
        p.adapt_epizoo_species(**args,execution_identity='a'*64)
    assert calls['build']==0
