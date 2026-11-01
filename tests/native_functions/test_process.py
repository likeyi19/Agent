"""Cancellation, timeout and bounded shared delivery are execution policies."""
import os
import sys
import time
from pathlib import Path
import pytest
from agent.tools import _native_process as owner
from agent.tools._cancellation import cancellation_scope,ToolWorkCancelled


def test_cancellation_ends_native_process_group_without_completion(tmp_path):
    pid=tmp_path/'pid';ready=tmp_path/'ready'
    # This is test-owned code, never a public native execution interface.
    code="import os,time;from pathlib import Path;Path(%r).write_text(str(os.getpid()));Path(%r).touch();time.sleep(30)"%(str(pid),str(ready))
    with cancellation_scope(lambda:ready.exists()):
        with pytest.raises(ToolWorkCancelled):owner.run([sys.executable,'-c',code],tmp_path/'log',environment=dict(os.environ))
    with pytest.raises(ProcessLookupError):os.kill(int(pid.read_text()),0)


def test_timeout_does_not_retry(tmp_path):
    start=time.monotonic()
    with pytest.raises(TimeoutError):owner.run([sys.executable,'-c','import time;time.sleep(30)'],tmp_path/'log',environment=dict(os.environ),timeout=.1)
    assert time.monotonic()-start<5


@pytest.mark.parametrize('label', [b'id', b'='])
def test_native_tsv_projection_can_grow_by_more_than_fixed_h5ad_overhead(tmp_path,label):
    import hashlib
    from types import SimpleNamespace
    from agent.application.matrix_delivery import MatrixDownloadPreparer
    from agent.tools.analysis import chromvar_native
    raw=b'source_index\tsource_id\tvalue\n'+(b'0\t'+label+b'\t1\n')*100000
    p=tmp_path/'result.tsv';p.write_bytes(raw);sha=hashlib.sha256(raw).hexdigest()
    artifact=SimpleNamespace(contract=chromvar_native.CONTRACT,manifest={'tool_name':chromvar_native.TOOL},role='result.tsv',
        size_bytes=len(raw),sha256=sha,path=p,managed_root=tmp_path)
    preparer=MatrixDownloadPreparer()
    output=preparer.prepare(artifact)
    try:
        data=b''.join(output.iter_chunks())
        assert len(data)-len(raw)>65536
        assert output.size_bytes==len(data) and output.sha256==hashlib.sha256(data).hexdigest()
        assert output.source_sha256==sha
        assert preparer._reserved_bytes==max(len(raw),len(data))
        assert preparer.active_count==1
    finally:output.close()
    assert preparer._reserved_bytes==0 and preparer.active_count==0


@pytest.mark.parametrize('capacity', ['exact', 'growth_denied', 'output_limit'])
def test_projection_admits_only_actual_growth_and_releases_on_failure(tmp_path,monkeypatch,capacity):
    import hashlib
    from types import SimpleNamespace
    from agent.application import matrix_delivery as delivery
    from agent.application.native_delivery import prepare
    from agent.tools.analysis import chromvar_native
    import io
    raw=b'source_id\n'+b'=\n'*1000
    expected=prepare(io.BytesIO(raw),'result.tsv')
    p=tmp_path/'result.tsv';p.write_bytes(raw)
    artifact=SimpleNamespace(contract=chromvar_native.CONTRACT,manifest={'tool_name':chromvar_native.TOOL},role='result.tsv',
        size_bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),path=p,managed_root=tmp_path)
    # The old worst-case multiplier would unnecessarily reject the exact case.
    free=(len(raw) if capacity=='growth_denied' else len(expected))+delivery._DISK_MARGIN_BYTES
    monkeypatch.setattr(delivery.shutil,'disk_usage',lambda _:SimpleNamespace(free=free))
    if capacity=='output_limit':monkeypatch.setattr(delivery,'MAX_MATRIX_BYTES',len(expected)-1)
    preparer=delivery.MatrixDownloadPreparer()
    if capacity=='exact':
        result=preparer.prepare(artifact)
        try:
            assert b''.join(result.iter_chunks())==expected
            assert preparer._reserved_bytes==len(expected) and preparer.active_count==1
        finally:result.close()
    else:
        with pytest.raises(delivery.MatrixDeliveryError) as error:preparer.prepare(artifact)
        assert error.value.code==('SCIENTIFIC_DOWNLOAD_BUSY' if capacity=='growth_denied' else 'SCIENTIFIC_DOWNLOAD_LIMIT')
    assert preparer._reserved_bytes==0 and preparer.active_count==0
    assert p.read_bytes()==raw
