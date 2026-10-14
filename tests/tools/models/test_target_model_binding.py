"""Narrow accepted-model consumption; no training authority renewal."""
from pathlib import Path
from types import SimpleNamespace, ModuleType
import hashlib
import sys
import pytest
from agent.orchestration import target_model_binding as binding
from agent.tools.data.authority_context import VerificationContext, authority_operation
from agent.tools.models.epizoo_adaptation import resources
from agent.schemas.verification_authority import AuthorityError


@pytest.fixture
def accepted(tmp_path, monkeypatch):
    root=tmp_path/'publication'; root.mkdir(); artifact=root/'artifact';artifact.mkdir()
    native=tmp_path/'epizoo';native.mkdir(); (native/'__init__.py').write_text('')
    module=ModuleType('epizoo'); module.__file__=str(native/'__init__.py')
    monkeypatch.setitem(sys.modules,'epizoo',module)
    model=artifact/'manifest.json';model.write_text('{}')
    (artifact/'checkpoint.pth').write_bytes(b'simulated checkpoint identity, not a scientific model')
    (root/'receipt.json').write_text('{}')
    paths=[model,artifact/'checkpoint.pth',root/'receipt.json',native/'__init__.py']
    files=VerificationContext().files(paths)
    sha=hashlib.sha256(model.read_bytes()).hexdigest()
    record=dict(schema_version=2,scope='scientific_correctness.v1',completion='succeeded',
        artifact_contract=binding.contract.CONTRACT,artifact_type=binding.contract.ARTIFACT,
        publication_path=str(model),manifest_sha256=sha,science_profile='profile',files=files)
    manifest=dict(profile_sha256='profile',runtime={},files={'checkpoint.pth':{'sha256':'checkpoint'}})
    stored=SimpleNamespace(tool_name='adapt_epizoo_species',result=dict(manifest_path=str(model),
        manifest_sha256=sha,checkpoint_sha256='checkpoint'))
    authority=SimpleNamespace(to_dict=lambda:record,identity_sha256='accepted-authority')
    monkeypatch.setenv('AGENT_EPIZOO_SOURCE_STORE',str(tmp_path/'store'))
    monkeypatch.setattr(binding,'FileRunStore',lambda root:None)
    monkeypatch.setattr(binding,'_accepted',lambda *a:(stored,authority,'execution','anchor'))
    monkeypatch.setattr(binding.publication,'load_manifest',lambda *a:manifest)
    monkeypatch.setattr(resources,'runtime',lambda:{})
    return model,sha,stored,record,native,manifest


def test_consumes_accepted_publication_without_training_reconstruction(accepted,monkeypatch):
    path,sha,*_=accepted
    monkeypatch.setattr(binding.publication,'verify_target',lambda *a,**k:pytest.fail('reconstructed adaptation'))
    with authority_operation(VerificationContext()) as context:
        assert binding.accepted_target(path,sha,'run','step')=='accepted-authority'
        assert binding.accepted_target(path,sha,'run','step')=='accepted-authority'
        assert len(context.source_contexts)==1
        context.validate_anchors()


@pytest.mark.parametrize('mutation',['checkpoint','native','receipt','missing_native','extra_native',
                                      'wrong_model','wrong_tool','wrong_runtime','untrusted_store'])
def test_rejects_invalid_consumption(accepted,monkeypatch,mutation):
    path,sha,stored,record,native,manifest=accepted
    if mutation=='checkpoint':(path.parent/'checkpoint.pth').write_bytes(b'tampered')
    if mutation=='native':(native/'__init__.py').write_text('changed')
    if mutation=='receipt':(path.parent.parent/'receipt.json').write_text('changed')
    if mutation=='missing_native':(native/'__init__.py').unlink()
    if mutation=='extra_native':(native/'new.py').write_text('new')
    if mutation=='wrong_model':record['manifest_sha256']='wrong'
    if mutation=='wrong_tool':stored.tool_name='inspect_scATAC'
    if mutation=='wrong_runtime':manifest['runtime']={'changed':True}
    if mutation=='untrusted_store':monkeypatch.delenv('AGENT_EPIZOO_SOURCE_STORE')
    with pytest.raises((AuthorityError,FileNotFoundError)):
        binding.accepted_target(path,sha,'run','step')


def test_anchor_change_during_consumption(accepted,monkeypatch):
    path,sha,*_=accepted
    original=binding._accepted;calls=[]
    def changed(*a):
        calls.append(1);s,r,e,k=original(*a)
        return s,r,e,k if len(calls)==1 else 'changed'
    monkeypatch.setattr(binding,'_accepted',changed)
    with pytest.raises(AuthorityError,match='anchor changed'):
        binding.accepted_target(path,sha,'run','step')


def test_readonly_consumption_does_not_require_training_thread_limits(accepted,monkeypatch):
    path,sha,_,_,_,manifest=accepted
    manifest['runtime']={'cpu_threads':4,'interop_threads':24}
    monkeypatch.setattr(resources,'runtime',lambda:{'cpu_threads':32,'interop_threads':24})
    assert binding.accepted_target(path,sha,'run','step')=='accepted-authority'
