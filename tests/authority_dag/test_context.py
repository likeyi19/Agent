"""Failure, publication and execution isolation of operation-local proofs."""
from pathlib import Path
import pytest
from agent.schemas.verification_authority import AuthorityError
from agent.tools._cancellation import cancellation_checkpoint, cancellation_scope, ToolWorkCancelled
from agent.tools.data.authority_context import VerificationContext, authority_operation, publication_moved


def test_publication_preserves_committed_success_cancellation_order(monkeypatch):
    context=VerificationContext(); checked=[]
    def moved(source,destination):
        cancellation_checkpoint()
        checked.append((source,destination))
    monkeypatch.setattr(context,'moved',moved)
    with authority_operation(context), cancellation_scope(lambda:True):
        publication_moved('/private','/published')
        with pytest.raises(ToolWorkCancelled): cancellation_checkpoint()
    assert checked==[('/private','/published')]


@pytest.mark.parametrize('error',[ValueError('invalid science'),ToolWorkCancelled()])
def test_failed_or_interrupted_verifier_cannot_issue_proof(tmp_path,monkeypatch,error):
    from agent.tools.data import scientific_authority as adapter
    context=VerificationContext(); path=tmp_path/'manifest.json';path.write_text('{}')
    description=dict(files=[],historical_sources=[])
    monkeypatch.setattr(context,'describe',lambda *a:(description,'identity'))
    monkeypatch.setattr(adapter,'runtime_compatibility',lambda *a:None)
    def failed(*a,**kw): raise error
    with pytest.raises(type(error)):
        context.verify('qc',failed,(path,),{})
    assert context.proofs==context.locations==context.accepted=={}


def test_same_path_replacement_invalidates_file_identity(tmp_path):
    context=VerificationContext(); path=tmp_path/'payload';path.write_bytes(b'abcd')
    first=context.files([path])
    replacement=tmp_path/'replacement';replacement.write_bytes(b'abce');replacement.replace(path)
    assert context.files([path])!=first


def test_execution_namespaces_do_not_share_new_scientific_proofs(tmp_path):
    context=VerificationContext()
    context.register_execution('qc',tmp_path/'one','1'*64)
    assert context.execution_for('qc',tmp_path/'one/manifest.json')=='1'*64
    context.accepted[str(tmp_path/'one/manifest.json'),'sha']={'record':{'execution_identity':'1'*64}}
    context.register_execution('qc',tmp_path/'two','2'*64)
    assert context.execution_for('qc',tmp_path/'one/manifest.json')=='1'*64
    assert context.execution_for('qc',tmp_path/'two/manifest.json')=='2'*64
    assert context.execution_for('selection',tmp_path/'one/unaccepted.json')=='independently_verified_dependency'


def test_shared_output_root_cannot_assign_upstream_a_downstream_execution(tmp_path):
    context=VerificationContext(); path=tmp_path/'fragments/manifest.json'
    canonical=context.execution_for('generic_fragments',path)
    producer=context.execution_for('bam_fragment_production',path)
    for kind in ('qc','selection','matrix'):
        context.register_execution(kind,tmp_path,kind)
        assert context.execution_for('generic_fragments',path)==canonical
        assert context.execution_for('bam_fragment_production',path)==producer


@pytest.mark.parametrize('defer',[False,True])
def test_hash_bookkeeping_preserves_cancellation_scope(tmp_path,defer):
    context=VerificationContext();context.defer_hash_cancellation=defer
    path=tmp_path/'payload';path.write_bytes(b'content')
    with cancellation_scope(lambda:True):
        if defer:
            assert context.files([path])[0]['size_bytes']==7
        else:
            with pytest.raises(ToolWorkCancelled):context.files([path])
            assert context._hashes=={}
        with pytest.raises(ToolWorkCancelled):cancellation_checkpoint()
