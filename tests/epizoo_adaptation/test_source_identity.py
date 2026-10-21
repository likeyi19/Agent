"""Source provenance must survive vendoring without trusting an enclosing Git repo."""
from copy import deepcopy
from pathlib import Path

import epizoo
import pytest

from agent.tools.models.epizoo_adaptation import contract as c, resources


@pytest.fixture
def snapshot(tmp_path):
    root=tmp_path/'snapshot'
    package=root/'epizoo';package.mkdir(parents=True)
    (package/'__init__.py').write_text('"""Explicit software provenance fixture."""\n')
    config=root/'config/SEAM_config';config.mkdir(parents=True)
    (config/'config.json').write_text('{}')
    files=[]
    for path in sorted((package/'__init__.py',config/'config.json')):
        record=c.file_record(path)
        files.append(dict(record,path=str(path.relative_to(root))))
    value=dict(schema='agent-vendored-backend.v1',package='epizoo',
        upstream_repository='https://github.com/likeyi19/EpiZoo',
        revision='029cd631d0f9806a646c4a3b42ce10b958f2b67f',
        original_source=dict(path='/historical/independent/EpiZoo',git_status_porcelain=''),
        source_files=deepcopy(files),files=files,source_identity_sha256=c.digest(files),
        snapshot_identity_sha256=c.digest(files),changes=[])
    def save():
        (root/'SOURCE.json').write_bytes(c.canonical(value))
    save()
    return package,value,save


def test_bundle_records_upstream_and_exact_manifest_without_git(snapshot,monkeypatch):
    root,value,_=snapshot
    monkeypatch.setattr(epizoo,'__file__',str(root/'__init__.py'))
    monkeypatch.setattr(resources,'runtime',lambda:{})
    def forbidden(*args,**kwargs):pytest.fail('Vendored provenance consulted enclosing Agent Git history')
    monkeypatch.setattr(resources.subprocess,'check_output',forbidden)
    result=resources.code_bundle()
    assert result['revision']==value['revision']
    assert c.file_record(root.parent/'SOURCE.json') in result['files']


def test_full_manifest_metadata_changes_bundle_identity(snapshot,monkeypatch):
    root,value,save=snapshot
    monkeypatch.setattr(epizoo,'__file__',str(root/'__init__.py'))
    monkeypatch.setattr(resources,'runtime',lambda:{})
    before=resources.code_bundle()
    value['original_source']['git_status_porcelain']=' M explicitly-recorded-working-source'
    save()
    assert resources.code_bundle()!=before


@pytest.mark.parametrize('change',['schema','revision','identity','malformed_files','traversal','duplicate','source_identity'])
def test_malformed_or_rehashed_source_manifest_fails(snapshot,change):
    root,value,save=snapshot
    if change=='schema':value['schema']='unknown'
    elif change=='revision':value['revision']='Agent HEAD is not an EpiZoo revision'
    elif change=='identity':value['snapshot_identity_sha256']='0'*64
    elif change=='malformed_files':value['files']={'untrusted':'not a record list'}
    elif change=='traversal':value['files'][0]['path']='../outside.py'
    elif change=='duplicate':value['files'].append(deepcopy(value['files'][0]))
    else:
        value['source_files'][0]['sha256']='0'*64
    if change in ('malformed_files','traversal','duplicate'):
        value['snapshot_identity_sha256']=c.digest(value['files'])
    save()
    with pytest.raises(ValueError):resources._source_revision(root)


@pytest.mark.parametrize('path',['epizoo/__init__.py','config/SEAM_config/config.json'])
def test_source_and_configuration_corruption_fail(snapshot,path):
    root,_,_=snapshot
    (root.parent/path).write_bytes(b'changed source bytes')
    with pytest.raises(ValueError,match='source bytes changed'):resources._source_revision(root)


@pytest.mark.parametrize('path',['epizoo/injected.py','config/SEAM_config/injected.json'])
def test_unlisted_runtime_source_or_configuration_fails(snapshot,path):
    root,_,_=snapshot
    (root.parent/path).write_bytes(b'not in the qualified closure')
    with pytest.raises(ValueError,match='closure changed'):resources._source_revision(root)


def test_source_symlink_cannot_escape_even_to_equal_bytes(snapshot,tmp_path):
    root,_,_=snapshot
    path=root/'__init__.py'
    outside=tmp_path/'outside.py';outside.write_bytes(path.read_bytes())
    path.unlink();path.symlink_to(outside)
    with pytest.raises(ValueError,match='symlinks'):resources._source_revision(root)


def test_missing_manifest_cannot_claim_enclosing_agent_revision(snapshot,monkeypatch):
    root,_,_=snapshot
    (root.parent/'SOURCE.json').unlink()
    monkeypatch.setattr(resources.subprocess,'check_output',lambda *args,**kwargs:str(root.parent.parent)+'\n')
    with pytest.raises(ValueError,match='enclosing repository'):resources._source_revision(root)


def test_independent_repository_retains_explicit_git_provenance(tmp_path,monkeypatch):
    root=tmp_path/'independent/epizoo';root.mkdir(parents=True)
    revision='1'*40
    def git(argv,**kwargs):
        return str(root.parent)+'\n' if argv[-1]=='--show-toplevel' else revision+'\n'
    monkeypatch.setattr(resources.subprocess,'check_output',git)
    assert resources._source_revision(root)==(revision,None)
