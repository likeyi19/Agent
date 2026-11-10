"""Native workflow composition and explicit/default parameter boundaries."""
import json
from pathlib import Path
import subprocess
import os

import pytest

from agent.orchestration import AgentRequest, build_default_tool_registry
from agent.orchestration.semantic_compiler import build_semantic_compiler_contract
from agent.orchestration.semantic_wire_v4 import parse_semantic_wire_v4
from agent.orchestration.semantic_compiler import compile_semantic_plan
from agent.tools.analysis.native_matrix import SNAPATAC2


def test_ordinary_spectral_native_plan_composes_without_dedicated_pca(tmp_path):
    from qualify_stage1 import SpectralModel
    registry=build_default_tool_registry()
    request=AgentRequest('spectral','Cluster processed peak counts.',dict(input_path=str(tmp_path/'counts.h5ad'),output_dir=str(tmp_path/'out')))
    wire=SpectralModel().complete(prompt='{}',response_schema={})
    candidate=parse_semantic_wire_v4(wire,request,registry)
    plan=compile_semantic_plan(request,candidate,registry,build_semantic_compiler_contract(registry))
    assert [s.arguments['function_name'] for s in plan.steps]==['pp.select_features','tl.spectral','pp.knn','tl.leiden','tl.umap']
    assert all('kwargs' not in s.arguments and s.tool_name=='run_snapatac2' for s in plan.steps)
    assert plan.steps[-1].arguments['input_path'].step_id=='s3'


def test_uncalled_retention_and_explicit_native_defaults_remain_distinct():
    base=dict(function_name='pp.import_fragments',input_path='/fragments',output_dir='/output',kwargs={'chrom_sizes':{'chr1':100}})
    assert 'min_num_fragments' not in SNAPATAC2.native_arguments(base)
    assert SNAPATAC2.native_arguments(dict(base,qc_thresholds='otsu'))['min_num_fragments']==0
    assert SNAPATAC2.native_arguments(dict(base,qc_thresholds='otsu',kwargs=dict(base['kwargs'],min_num_fragments=50)))['min_num_fragments']==50
    assert SNAPATAC2.native_arguments(dict(input_path='/input',function_name='pp.filter_cells',qc_thresholds='otsu',kwargs=dict(min_counts=200,min_tsse=None)))==dict(min_counts=200,min_tsse=None)
    assert 'pp.call_cells' in SNAPATAC2.FUNCTIONS and 'pp.add_tile_matrix' in SNAPATAC2.FUNCTIONS
    assert SNAPATAC2.native_arguments(dict(input_path='/input',function_name='pp.call_cells',kwargs=dict(use_rep='n_fragment',inplace=False)))==dict(use_rep='n_fragment',inplace=False)
    with pytest.raises(ValueError): SNAPATAC2.native_arguments(dict(input_path='/input',function_name='pp.call_cells',qc_thresholds='otsu'))


@pytest.mark.parametrize('function', ['pp.make_peak_matrix', 'pp.add_tile_matrix'])
def test_matrix_counting_choice_has_one_owner_and_preserves_explicit_values(function):
    args=dict(input_path='/input',function_name=function,counting_strategy='fragment')
    assert SNAPATAC2.native_arguments(args)=={'counting_strategy':'fragment'}
    assert SNAPATAC2.native_arguments(dict(args,kwargs={'counting_strategy':'fragment','inplace':False}))=={
        'counting_strategy':'fragment','inplace':False}
    with pytest.raises(ValueError,match='conflicting'):
        SNAPATAC2.native_arguments(dict(args,kwargs={'counting_strategy':'insertion'}))
    assert 'counting_strategy' not in SNAPATAC2.native_arguments(dict(input_path='/input',function_name=function))


@pytest.mark.skipif(os.environ.get('RUN_RAW_NATIVE_QUALIFICATION')!='1',reason='Requires isolated native Otsu runtime')
def test_otsu_uses_full_independent_populations_and_explicit_minima():
    # The fitting sets must be independent. A joint >=100 depth prefilter would
    # remove a high-TSSe low-depth barcode and change the TSS threshold.
    script=Path(__file__).parents[2]/'src/agent/tools/analysis/snapatac_raw_worker.py'
    # Test the actual worker helper in its installed scientific environment.
    code='''import importlib.util,numpy as np
from skimage.filters import threshold_otsu
spec=importlib.util.spec_from_file_location('raw_worker', %r);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
obs={'n_fragment':np.array([10,100,101,150,4000,8000]),'tsse':np.array([30.,.5,1.,1.1,5.,10.])}
kw,detail=m.otsu_thresholds(obs,{})
assert kw['min_counts']==10**float(threshold_otsu(np.log10(np.array([100,101,150,4000,8000])),nbins=5000))
assert kw['min_tsse']==float(threshold_otsu(np.array([30.,1.,1.1,5.,10.]),nbins=5000))
assert detail['input_barcodes']==6 and detail['thresholds']['min_tsse']['fitting_barcodes']==5
kw,detail=m.otsu_thresholds(obs,{'min_counts':None,'max_tsse':20})
assert 'min_counts' not in kw and detail['thresholds']['min_counts']['method']=='explicit'
kw,detail=m.otsu_thresholds(obs,{'min_counts':5,'min_tsse':None});assert kw=={}
try:m.otsu_thresholds({'n_fragment':[1,2],'tsse':[0.,.5]}, {})
except ValueError:pass
else:raise AssertionError('Empty fitting distribution must request explicit thresholds')
''' % str(script.resolve())
    subprocess.run([os.environ['AGENT_SNAPATAC2_PYTHON'],'-I','-c',code],check=True)
