"""Opt-in genuine installed-runtime output qualification; evidence stays outside Git.

PYTHONPATH=src conda run --no-capture-output -n agent python tests/dual_mode/qualify_outputs.py /absolute/evidence
Uses the existing systemd/cgroup boundary. No scientific packages are installed.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import zipfile

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from agent.application.native_functions import bind, execute, history, _accepted
from agent.providers import PlanningModelFactoryRegistry
from agent.tools.analysis import native_callable as owner
from agent.tools.analysis import native_callable_worker as worker
from agent.web.config import WebConfiguration, build_interactive_application

root = Path(sys.argv[1]).resolve()
root.mkdir(parents=True, exist_ok=True)
continuing = '--continue' in sys.argv[2:]
# Remove credentials before discovery, execution and any worker inheritance.
removed = [k for k in os.environ if any(s in k for s in ('API_KEY', 'ACCESS_TOKEN', 'AUTH_TOKEN'))]
for key in removed:
    os.environ.pop(key)
os.environ.setdefault('AGENT_NATIVE_MEMORY_BYTES', str(4 * 1024**3))
os.environ.setdefault('AGENT_NATIVE_TIMEOUT_SECONDS', '180')
os.environ['MPLBACKEND'] = 'Agg'
library = Path(os.environ.get('AGENT_SCIENTIFIC_RESOURCE_ROOT', '/home/likeyi/program/agent-data/resources/library'))
config = WebConfiguration(root / 'workspace', (), None, upload_root=root / 'workspace/uploads', scientific_resource_root=library)
app = build_interactive_application(config)
if not continuing: app._application.sessions.create('outputs')
provider_calls = []
def forbid_provider(*args, **kwargs):
    provider_calls.append(True)
    raise AssertionError('Native Functions called a provider')
PlanningModelFactoryRegistry.create = forbid_provider
records = json.loads((root/'native-calls.json').read_text()) if continuing else []
downloads = root / 'downloads'; downloads.mkdir(exist_ok=continuing)


def write(name, value):
    (root / name).write_text(json.dumps(value, indent=2) + '\n')


def call(label, package, function, inputs=None, parameters=None, outputs=None, expected='succeeded'):
    payload = dict(call_id=label, expected_generation=app._application.sessions.load('outputs').generation,
        package=package, function=function, inputs=inputs or {}, parameters=parameters or {}, outputs=outputs or {})
    saved_call = next((r for r in records if r['label'] == label), None)
    if saved_call is None:
        execute(app, 'outputs', payload, bind(app, 'outputs', payload))
        row = history(app, 'outputs')['calls'][-1]
        records.append(dict(label=label, request=payload, result=row))
        write('native-calls.json', records)
    else:
        assert dict(payload, expected_generation=saved_call['request']['expected_generation']) == saved_call['request']
        row = next(r for r in history(app, 'outputs')['calls'] if r['call_id'] == label)
        assert row == saved_call['result']
    assert row['status'] == expected, row
    run = app._application.run_store.load(app._application.sessions.load('outputs').turn(label).run_id)
    assert run.steps[0].attempt_count == 1
    if expected != 'succeeded':
        return row, None
    manifest = _accepted(app, 'outputs', row['revision_id'])[1]
    data = json.loads((Path(manifest['arguments']['output_dir']) / 'result.json').read_text())
    saved = {}
    destination = downloads / label; destination.mkdir(exist_ok=continuing)
    for handle in app.scientific_artifact_handles('outputs', row['revision_id']):
        snapshot = app.prepare_scientific_artifact('outputs', row['revision_id'], handle.handle)
        try:
            raw = b''.join(snapshot.iter_chunks())
            assert hashlib.sha256(raw).hexdigest() == handle.sha256
            assert len(raw) == handle.size_bytes
            saved[handle.filename] = raw
            (destination / handle.filename).write_bytes(raw)
        finally:
            snapshot.close()
    with zipfile.ZipFile(io.BytesIO(saved['outputs.zip'])) as archive:
        assert json.loads(archive.read('result.json')) == data
        for name, raw in saved.items():
            if name != 'outputs.zip': assert archive.read('objects/' + name) == raw
    write('download-verification.json', {p.relative_to(downloads).as_posix(): dict(
        sha256=hashlib.sha256(p.read_bytes()).hexdigest(), size_bytes=p.stat().st_size)
        for p in downloads.rglob('*') if p.is_file()})
    return row, data


source = config.upload_root / 'small.h5ad'
rng = np.random.default_rng(42)
data = ad.AnnData(sparse.csr_matrix(rng.poisson(2, (12, 4)).astype(float)),
    obs=pd.DataFrame({'group': pd.Categorical(['A']*6 + ['B']*6), 'depth': np.arange(12, dtype=float)},
        index=['cell'+str(i) for i in range(12)]), var=pd.DataFrame(index=['f'+str(i) for i in range(4)]))
data.obsm['provided'] = rng.normal(size=(12, 2))
if not continuing: data.write_h5ad(source)
source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
resource = (app.resources.load(records[0]['request']['inputs']['adata']['resource_id']) if continuing else
    app.resources.register('small', source, input_type='attachment', label='small.h5ad', attribution='Synthetic output qualification', session_id='outputs'))
input_data = {'adata': dict(adapter='anndata', resource_id=resource.resource_id)}
figure, f = call('figure', 'scanpy', 'pl.embedding', input_data, dict(basis='provided', return_fig=True, show=False))
assert f['return_value']['type'] == 'matplotlib.figure.Figure'
axes, a = call('axes', 'scanpy', 'pl.embedding', input_data, dict(basis='provided', show=False))
assert a['return_value']['encoding'] == 'matplotlib' and a['return_value']['type'].split('.')[-1] in ('Axes', 'AxesSubplot')
none, n = call('none-plot', 'scanpy', 'pl.embedding', input_data, dict(basis='provided', show=True))
assert n['return_value']['value'] is None and len(n['produced_files']) == 2
nested, d = call('nested-axes', 'scanpy', 'pl.dotplot', input_data, dict(var_names=['f0','f1'], groupby='group', show=False))
assert d['return_value']['encoding'] == 'dict'
assert len({v['value']['file'] for v in d['return_value']['items']}) == 1
assert len([p for p in (downloads/'nested-axes').iterdir() if p.suffix == '.png']) == 1
listed, li = call('axes-list', 'scanpy', 'pl.embedding', input_data, dict(basis='provided', color=['depth','f0'], show=False))
assert li['return_value']['encoding'] == 'list'
qc, q = call('dataframes', 'scanpy', 'pp.calculate_qc_metrics', input_data, dict(inplace=False, percent_top=None))
assert [m['encoding'] for m in q['return_value']['items']] == ['dataframe', 'dataframe']
frame, fr = call('table-preview', 'scanpy', 'get.obs_df', input_data, dict(keys=['depth','f0']))
assert fr['return_value']['encoding'] == 'dataframe' and fr['return_value']['shape'] == [12, 2]
assert len(json.loads(fr['return_value']['preview'])['data']) == 8
normalized, nm = call('sparse-dict', 'scanpy', 'pp.normalize_total', input_data, dict(inplace=False, target_sum=100))
assert nm['modified_inputs'] == {} and nm['return_value']['encoding'] == 'dict'
mutation, m = call('mutation', 'scanpy', 'pp.log1p', {'X': input_data['adata']})
assert m['return_value']['value'] is None and m['modified_inputs']['X']['encoding'] == 'anndata'
reused, r = call('reuse-anndata', 'scanpy', 'pp.log1p', {'X': dict(adapter='accepted', revision_id=mutation['revision_id'], output='modified:X')}, dict(copy=True))
assert r['return_value']['encoding'] == 'anndata' and r['modified_inputs'] == {}
array, ar = call('numpy-return', 'scanpy', 'pp.log1p', {'X': dict(adapter='array', value=[[1.,3.],[5.,7.]])}, dict(copy=True))
assert ar['return_value']['encoding'] == 'numpy' and ar['modified_inputs'] == {}
call('native-failure', 'scanpy', 'pl.embedding', input_data, dict(basis='absent', show=False), expected='failed')
unsupported, _ = call('unsupported-dotplot', 'scanpy', 'pl.dotplot', input_data,
    dict(var_names=['f0'], groupby='group', return_fig=True), expected='failed')
assert unsupported['native_error']['native_execution'] == 'succeeded'
assert 'DotPlot' in unsupported['native_error']['message']
granges, gr = call('granges', 'Signac', 'StringToGRanges', parameters={'regions': ['chr1-1-10','chr2-20-40']})
assert gr['return_value']['classes'] == ['GRanges'] and gr['return_value']['restorable']
strings, st = call('reuse-granges', 'Signac', 'GRangesToString',
    {'grange': dict(adapter='accepted', revision_id=granges['revision_id'], output='return')})
assert strings['preview']['return_value']['preview'] == ['chr1-1-10','chr2-20-40']
call('reuse-r-vector', 'Signac', 'StringToGRanges',
    {'regions': dict(adapter='accepted', revision_id=strings['revision_id'], output='return')})
files, fi = call('native-h5ad-file', 'scanpy', 'write', input_data, outputs={'filename': 'actual.h5ad'})
assert fi['return_value']['value'] is None and len(fi['produced_files']) == 1
bed = config.upload_root / 'tiny.bed'
if not continuing or not bed.exists():
    bed.write_text('chr1\t10\t20\ta\t0\t+\nchr1\t10\t20\tb\t0\t+\nchr1\t30\t40\tc\t0\t-\n')
bed_resource = app.resources.register('tiny-bed', bed, input_type='attachment', label='tiny.bed', attribution='Synthetic file qualification', session_id='outputs')
cli, cl = call('cli-stdout', 'MACS3', 'filterdup', {'ifile': dict(adapter='file', resource_id=bed_resource.resource_id)},
    dict(format='BED', gsize='1000', keepduplicates='1', verbose=0, threads=1))
assert any(f['file'] == 'stdout.txt' for f in cl['produced_files'])
assert b'chr1' in (downloads/'cli-stdout/stdout.txt').read_bytes()

# Worker-only contracts exercise base/stats/graphics types without exposing
# additional packages in the product or accepting researcher-supplied R code.
r_cases = json.loads((root/'r-worker-cases.json').read_text()) if continuing and (root/'r-worker-cases.json').exists() else []
def r_worker(label, package, function, parameters=None, inputs=None, expected=0):
    stage = root / label
    if continuing and (stage/('result.json' if expected == 0 else 'error.json')).is_file():
        return json.loads((stage/('result.json' if expected == 0 else 'error.json')).read_text())
    stage.mkdir(); (stage/'work').mkdir(); (stage/'objects').mkdir()
    request = dict(package=package, function=function, stage=str(stage), inputs=inputs or {}, parameters=parameters or {}, outputs={})
    (stage/'request.json').write_text(json.dumps(request))
    code = owner.run([str(owner.executable('Signac')), '--vanilla', str(Path(owner.__file__).parent/'r/native_callable.R'), str(stage/'request.json')], stage/'native.log', environment=owner.r_environment())
    assert code == expected, (stage/'native.log').read_text()
    r_cases.append(dict(label=label, request=request, exit_code=code,
        result=json.loads((stage/('result.json' if expected == 0 else 'error.json')).read_text())))
    write('r-worker-cases.json', r_cases)
    return r_cases[-1]['result']

density = r_worker('r-s3-density', 'stats', 'density', dict(x=[1,2,4,5,7], n=32))
assert density['return_value']['classes'] == ['density']
approx = r_worker('r-s3-reuse', 'stats', 'approx', dict(n=6),
    {'x': dict(adapter='r_accepted', path=str(root/'r-s3-density/objects/return.rds'))})
assert approx['return_value']['restorable']
graphics = r_worker('r-graphics', 'graphics', 'plot', dict(x=[1,4,2]))
pdf = root/'r-graphics/objects'/graphics['produced_files'][0]['file']
assert pdf.read_bytes().startswith(b'%PDF') and b'%%EOF' in pdf.read_bytes()
for label, package, function, params in [('r-environment','base','new.env',{}), ('r-pointer','methods','new',{'Class':'externalptr'})]:
    failure = r_worker(label, package, function, params, expected=1)
    assert failure['native_execution'] == 'succeeded' and failure['result_representation'] == 'failed'
    assert 'runtime resource' in failure['message']

# Polars is installed in the existing SnapATAC2 runtime, not the Agent runtime.
polars_stage = root/'polars'; polars_stage.mkdir(exist_ok=continuing)
polars_script = polars_stage/'check.py'
polars_script.write_text('''import importlib.util,json,sys
from pathlib import Path
import polars as pl
spec=importlib.util.spec_from_file_location('native_outputs',sys.argv[1])
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
root=Path(sys.argv[2]);value=pl.DataFrame({'a':[1,2],'text':['native',None]})
saved=worker.preserve(value,root,'polars')
assert worker.restore(saved,root).equals(value)
assert worker.preview(saved,root)['preview']==value.to_dicts()
(root/'checked.json').write_text(json.dumps(worker.preview(saved,root)))
''')
if not continuing or not (polars_stage/'checked.json').exists():
    assert owner.run([str(owner.executable('snapatac2')), '-I', str(polars_script), worker.__file__, str(polars_stage)],
        polars_stage/'native.log', environment=owner.r_environment()) == 0

before = history(app, 'outputs')
state = app._application.sessions.load('outputs')
pins = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in config.workspace_root.rglob('*') if p.is_file()}
reopened = build_interactive_application(config)
assert history(reopened, 'outputs') == before
assert reopened._application.sessions.load('outputs') == state
assert pins == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in config.workspace_root.rglob('*') if p.is_file()}
assert hashlib.sha256(source.read_bytes()).hexdigest() == source_sha
assert not provider_calls
write('summary.json', dict(baseline='73a826bf8ef72eea52046d26d4cbb015ecf67b28', native_calls=len(records),
    r_worker_cases=len(r_cases), provider_calls=len(provider_calls), credential_variables_removed=len(removed),
    source_sha256=source_sha, reopened_without_changes=True, native_history=before,
    python=sys.version, runtimes={p:str(owner.executable(p)) for p in ('scanpy','Signac','MACS3')}))
print('Genuine native output qualification passed:', len(records), 'product calls;', len(r_cases), 'R worker calls; zero providers', flush=True)
