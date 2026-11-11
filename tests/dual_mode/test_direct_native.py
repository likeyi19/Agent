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


def test_figure_axes_nested_identity_and_actual_rendering(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots()
    try:
        axes.plot([1, 3, 2], label='Native trace')
        axes.set_title('Actual native figure')
        reference = tmp_path / 'reference.png'
        figure.savefig(reference)
        value = {'figure': figure, 'nested': [axes, (figure, 7)]}
        saved = worker.preserve(value, tmp_path, 'result')
        members = saved['items']
        descriptors = [members[0]['value'], members[1]['value']['items'][0],
            members[1]['value']['items'][1]['items'][0]]
        assert [d['type'] for d in descriptors] == [
            'matplotlib.figure.Figure', type(axes).__module__ + '.' + type(axes).__qualname__, 'matplotlib.figure.Figure']
        assert len({d['file'] for d in descriptors}) == 1
        assert all(d['restorable'] is False for d in descriptors)
        assert (tmp_path / descriptors[0]['file']).read_bytes() == reference.read_bytes()
        assert '<svg' in (tmp_path / descriptors[0]['images']['svg']).read_text()
        assert len(list(tmp_path.glob('result*.png'))) == 1
        assert len(list(tmp_path.glob('result*.svg'))) == 1
        assert worker.preview(saved, tmp_path)['items'][0]['value']['file'] == descriptors[0]['file']
        with pytest.raises(worker.RepresentationError, match='cannot restore'):
            worker.restore(saved, tmp_path)
    finally:
        plt.close(figure)


def test_none_plot_captures_only_new_open_figures(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    import os
    existing = plt.figure()
    def plot():
        figure, ax = plt.subplots()
        ax.plot([2, 5, 1])
        # Native None stays None; no forced show/save/return argument.
    install(monkeypatch, plot)
    before = Path.cwd()
    try:
        worker.invoke(dict(package='scanpy', function='function', inputs={}, parameters={}, outputs={}, stage=str(tmp_path)))
    finally:
        os.chdir(before)
        for figure in worker.open_figures():
            if figure is not existing: plt.close(figure)
        plt.close(existing)
    data = json.loads((tmp_path / 'result.json').read_text())
    assert data['return_value'] == dict(type='builtins.NoneType', encoding='json', value=None)
    assert data['modified_inputs'] == {}
    assert len(data['produced_files']) == 2
    assert {Path(f['file']).suffix for f in data['produced_files']} == {'.png', '.svg'}
    assert all(f['representation'] == 'rendered_figure' for f in data['produced_files'])


def test_closed_unreturned_figure_is_not_fabricated(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    import os
    def plot():
        figure, ax = plt.subplots()
        ax.plot([1, 3])
        plt.close(figure)
    install(monkeypatch, plot)
    before = Path.cwd()
    try:
        worker.invoke(dict(package='scanpy', function='function', inputs={}, parameters={}, outputs={}, stage=str(tmp_path)))
    finally:
        os.chdir(before)
    data = json.loads((tmp_path/'result.json').read_text())
    assert data['return_value']['value'] is None and data['produced_files'] == []


def test_rendered_objects_not_offered_for_reuse_but_data_members_are(app, monkeypatch):
    import matplotlib.pyplot as plt
    figure, ax = plt.subplots()
    ax.plot([1, 2])
    try:
        install(monkeypatch, lambda: [figure, np.arange(3)])
        row = call(app, 'visual')
        assert row['status'] == 'succeeded', row
        assert [v['key'] for v in row['outputs']] == ['return/1']
        install(monkeypatch, lambda data: data)
        payload = dict(call_id='invalid-reuse', expected_generation=1, package='scanpy', function='function',
            inputs={'data': dict(adapter='accepted', revision_id=row['revision_id'], output='return/0')})
        with pytest.raises(ValueError, match='cannot restore'):
            bind(app, 'native', payload)
        assert len(app._application.sessions.load('native').turns) == 1
        # Images and ZIP go through the ordinary byte-checked delivery owner.
        import zipfile
        import io
        downloaded = {}
        for handle in app.scientific_artifact_handles('native', row['revision_id']):
            snapshot = app.prepare_scientific_artifact('native', row['revision_id'], handle.handle)
            try: downloaded[handle.filename] = b''.join(snapshot.iter_chunks())
            finally: snapshot.close()
        png = next(k for k in downloaded if k.endswith('.png'))
        assert downloaded[png].startswith(b'\x89PNG\r\n\x1a\n')
        with zipfile.ZipFile(io.BytesIO(downloaded['outputs.zip'])) as archive:
            assert archive.read('objects/' + png) == downloaded[png]
        install(monkeypatch, lambda: pytest.fail('Reopening replayed science'))
        assert history(app, 'native')['calls'][-1] == row
    finally:
        plt.close(figure)


def test_rds_is_not_bound_to_a_python_runtime(app, monkeypatch):
    import agent.application.native_functions as native
    install(monkeypatch, lambda data: data)
    monkeypatch.setattr(native, '_accepted', lambda *a: (None, {}))
    monkeypatch.setattr(native, '_descriptors', lambda manifest: {'return': dict(type='R.GRanges', encoding='r', restorable=True)})
    payload = dict(call_id='r-to-python', expected_generation=0, package='scanpy', function='function',
        inputs={'data': dict(adapter='accepted', revision_id='selected', output='return')})
    with pytest.raises(ValueError, match='requires an installed R runtime'):
        bind(app, 'native', payload)
    assert not app._application.sessions.load('native').turns


def test_mixed_scientific_members_keep_values_and_native_types(tmp_path):
    from scipy import sparse
    value = [np.int32(4), sparse.csc_matrix([[0., 3.], [1., 0.]]),
        ad.AnnData(np.array([[1., 2.]])),
        {'x': (False, None, pd.Series([3, 4], name='series'))}]
    saved = worker.preserve(value, tmp_path, 'mixed')
    restored = worker.restore(saved, tmp_path)
    assert type(restored[0]) is np.int32
    assert sparse.isspmatrix_csc(restored[1])
    np.testing.assert_array_equal(restored[1].toarray(), value[1].toarray())
    np.testing.assert_array_equal(restored[2].X, value[2].X)
    pd.testing.assert_series_equal(restored[3]['x'][2], value[3]['x'][2])
    preview = worker.preview(saved, tmp_path)
    assert preview['items'][1]['shape'] == [2, 2]
    assert '3.0' in preview['items'][1]['preview']


def test_polars_ipc_retains_native_values(tmp_path):
    pl = pytest.importorskip('polars')
    value = pl.DataFrame({'a': [1, 2], 'text': ['native', None]})
    saved = worker.preserve(value, tmp_path, 'polars')
    assert worker.restore(saved, tmp_path).equals(value)
    assert worker.preview(saved, tmp_path)['preview'] == value.to_dicts()


def test_fortran_array_preview_keeps_native_order_and_full_data(tmp_path):
    value = np.asfortranarray(np.arange(120).reshape(12, 10))
    saved = worker.preserve(value, tmp_path, 'fortran')
    assert worker.preview(saved, tmp_path)['preview'] == repr(list(range(24)))
    restored = worker.restore(saved, tmp_path)
    assert restored.flags.f_contiguous
    np.testing.assert_array_equal(restored, value)
