"""Software contracts with a simulated native process; genuine runs are external acceptance."""
import json
from pathlib import Path
import sys
import anndata as ad
import numpy as np
import pandas as pd
import pytest
from agent.application.native_functions import bind, execute, history
from agent.tools.analysis import native_callable as owner
from agent.tools.analysis import native_callable_worker as worker
from agent.web.config import WebConfiguration, build_interactive_application


@pytest.fixture
def app(tmp_path, monkeypatch):
    from agent.tools import _native_process
    from agent.providers import PlanningModelFactoryRegistry
    monkeypatch.setattr(PlanningModelFactoryRegistry, 'create', lambda *a, **k: pytest.fail('Provider called'))
    monkeypatch.setattr(_native_process, 'call', lambda operation, arguments, local: local())
    monkeypatch.setattr(owner, 'call', lambda operation, arguments, local: local())
    monkeypatch.setattr(owner, 'executable', lambda p: Path(sys.executable))
    original = Path.cwd()
    def run(command, log_path, **kwargs):
        request = json.loads(Path(command[-1]).read_text())
        log_path.write_text('simulated native process')
        try:
            worker.invoke(request)
            return 0
        finally:
            import os
            os.chdir(original)
    monkeypatch.setattr(owner, 'run', run)
    application = build_interactive_application(WebConfiguration(tmp_path, (), None, upload_root=tmp_path/'uploads'))
    application._application.sessions.create('native')
    return application


def install(monkeypatch, fn):
    monkeypatch.setattr(worker, 'functions', lambda package: {'function': fn})
    monkeypatch.setattr(owner, 'discover', lambda package, function: dict(functions=[worker.metadata(function, fn)]))


def call(app, name, inputs=None, parameters=None):
    payload = dict(call_id=name, expected_generation=app._application.sessions.load('native').generation,
        package='scanpy', function='function', inputs=inputs or {}, parameters=parameters or {}, outputs={})
    execute(app, 'native', payload, bind(app, 'native', payload))
    return history(app, 'native')['calls'][-1]


def test_omission_types_none_and_no_provider(app, monkeypatch):
    calls = []
    def fn(integer=17, floating=3.5, boolean=True, text='native', vector=None, nullable='omitted'):
        calls.append((integer, floating, boolean, text, vector, nullable))
        return {'scalar': integer, 'values': (floating, boolean, text, vector, nullable)}
    install(monkeypatch, fn)
    params = dict(integer=3, floating=2.25, boolean=False, text='literal', vector=[1,2], nullable=None)
    row = call(app, 'typed', parameters=params)
    assert row['status'] == 'succeeded', row
    assert calls == [(3, 2.25, False, 'literal', [1,2], None)]
    first = app._application.run_store.load(app._application.sessions.load('native').turn('typed').run_id)
    assert first.request.prompt is None
    assert dict(first.steps[0].resolved_arguments['parameters'])['nullable'] is None
    assert first.steps[0].attempt_count == 1
    call(app, 'omitted')
    assert calls[-1] == (17,3.5,True,'native',None,'omitted')
    assert len(calls) == 2


def test_mutation_axis_change_and_exact_reuse(app, monkeypatch):
    path = app.resources._workspace.root/'uploads'/'input.h5ad'
    ad.AnnData(np.arange(12,dtype=float).reshape(4,3)).write_h5ad(path)
    original = path.read_bytes()
    record = app.resources.register('input', path, input_type='attachment', label='input.h5ad', attribution='test', session_id='native')
    def subset(data): data._inplace_subset_obs([True,False,True,False])
    install(monkeypatch, subset)
    row = call(app, 'subset', inputs={'data':dict(adapter='anndata',resource_id=record.resource_id)})
    assert row['status']=='succeeded', row
    assert row['preview']['return_value']['value'] is None
    assert row['preview']['modified_inputs']['data']['shape']==[2,3]
    assert path.read_bytes()==original
    def consume(data): return data.X.copy()
    install(monkeypatch,consume)
    second=call(app,'reuse',inputs={'data':dict(adapter='accepted',revision_id=row['revision_id'],output='modified:data')})
    assert second['status']=='succeeded'
    assert second['preview']['modified_inputs']=={}
    assert second['preview']['return_value']['shape']==[2,3]
    state=app._application.sessions.load('native')
    assert state.revisions[-1].parent_revision_id==row['revision_id']
    assert path.read_bytes()==original
    # Accepted downloads use the ordinary snapshot owner, and reopening runs no function.
    artifacts=app.scientific_artifact_handles('native',row['revision_id'])
    file=next(v for v in artifacts if v.filename.endswith('.h5ad'))
    snapshot=app.prepare_scientific_artifact('native',row['revision_id'],file.handle)
    try: assert b''.join(snapshot.iter_chunks())[:8]==b'\x89HDF\r\n\x1a\n'
    finally: snapshot.close()


def test_returns_are_complete_previews_are_bounded(tmp_path):
    value={'series':pd.Series([1,2],name=None), 'named_frame':pd.DataFrame({'x':[1,2]}).rename_axis(index='row',columns='column'), 'array':np.arange(1000,dtype=np.int32), 'tuple':(None,True,'text'), 'frame':pd.DataFrame({'x':np.array([1,2],dtype=np.int64)})}
    nonfinite = worker.preserve(float('inf'), tmp_path, 'nonfinite')
    assert worker.preview(nonfinite, tmp_path)['preview'] == 'inf'
    assert worker.restore(nonfinite, tmp_path) == float('inf')
    saved=worker.preserve(value,tmp_path,'result')
    restored=worker.restore(saved,tmp_path)
    np.testing.assert_array_equal(restored['array'],value['array'])
    assert type(restored['tuple']) is tuple
    pd.testing.assert_frame_equal(restored['frame'],value['frame'])
    pd.testing.assert_frame_equal(restored['named_frame'],value['named_frame'])
    pd.testing.assert_series_equal(restored['series'],value['series'])
    preview=worker.preview(saved,tmp_path)
    assert len(preview['items'][2]['value']['preview'])<300


def test_no_data_and_unsupported_output_after_success(tmp_path,monkeypatch):
    def no_data(): return None
    install(monkeypatch,no_data)
    request=dict(package='scanpy',function='function',inputs={},parameters={},outputs={},stage=str(tmp_path))
    import os
    before=Path.cwd()
    try: worker.invoke(request)
    finally: os.chdir(before)
    result=json.loads((tmp_path/'result.json').read_text())
    assert result==dict(return_value=dict(type='builtins.NoneType',encoding='json',value=None),modified_inputs={},produced_files=[])
    other=tmp_path/'unsupported';other.mkdir()
    def unsupported():
        Path('retained.txt').write_text('native output before unsupported return')
        return object()
    install(monkeypatch, unsupported)
    try:
        with pytest.raises(worker.RepresentationError): worker.invoke(dict(request,stage=str(other)))
    finally: os.chdir(before)
    assert json.loads((other/'execution.json').read_text())['native_execution']=='succeeded'
    assert not (other/'result.json').exists()
    partial = json.loads((other/'unpublished.json').read_text())
    assert (other/'objects'/partial['produced_files'][0]['file']).read_text() == 'native output before unsupported return'


def test_files_and_multiple_outputs_are_distinct(tmp_path,monkeypatch):
    def generate():
        Path('actual.txt').write_text('actual native bytes')
        return [1, np.arange(3)]
    install(monkeypatch,generate)
    import os
    before=Path.cwd()
    try: worker.invoke(dict(package='scanpy',function='function',inputs={},parameters={},outputs={},stage=str(tmp_path)))
    finally: os.chdir(before)
    result=json.loads((tmp_path/'result.json').read_text())
    assert result['return_value']['encoding']=='list'
    assert result['modified_inputs']=={}
    assert len(result['produced_files'])==1
    assert (tmp_path/'objects'/result['produced_files'][0]['file']).read_text()=='actual native bytes'


def test_structured_requests_never_enter_planner(app):
    from agent.schemas import AgentRequest
    with pytest.raises(ValueError,match='cannot enter planning'):
        app._application.runtime.run(AgentRequest('direct',None,{}))


def test_restarted_native_http_identity_is_exact(app, monkeypatch):
    from agent.web.app import _LocalWorkers
    install(monkeypatch, lambda value=1: value)
    call(app, 'saved', parameters={'value': 1})
    workers = _LocalWorkers(app, 1)
    payload = dict(call_id='saved', expected_generation=0, package='scanpy', function='function', inputs={}, parameters={'value': 1}, outputs={})
    try:
        workers.submit_native('native', payload)
        assert not workers.receipts
        with pytest.raises(ValueError, match='different parameters'):
            workers.submit_native('native', dict(payload, parameters={'value': 2}))
        assert len(app._application.sessions.load('native').turns) == 1
    finally:
        workers.close()


def test_long_table_cells_are_bounded_only_in_preview(tmp_path):
    value = pd.DataFrame({'text': ['x' * 100000]})
    saved = worker.preserve(value, tmp_path, 'long-table')
    assert len(worker.preview(saved, tmp_path)['preview']) < 1000
    pd.testing.assert_frame_equal(worker.restore(saved, tmp_path), value)
