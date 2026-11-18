"""Scientific-owner promises and genuine source conditions; no native execution."""
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import pytest
from agent.schemas import StepOutputRef
from agent.orchestration import build_default_tool_registry
from agent.orchestration.scientific_inputs import OperationPreparationContext,ScientificInputRequired,ScientificDiagnostic
from agent.tools.analysis import native_matrix


def context(selected=(),tool='run_scanpy'):
    registry=build_default_tool_registry()
    return OperationPreparationContext(registry.get(tool),'consumer',registry,dict(selected),{}, {},[],None,None,{})


def source(tmp_path):
    path=tmp_path/'source.h5ad'
    data=ad.AnnData(np.eye(4),obs=pd.DataFrame(index=['a','b','c','d']));data.write_h5ad(path)
    return path


@pytest.mark.parametrize('selection', ['asw', 'native_defaults', 'unknown'])
@pytest.mark.parametrize('package,function', [('SCANPY','pp.neighbors'), ('SNAPATAC2','pp.import_fragments'), ('SIGNAC','RunPCA')])
def test_clustering_selection_is_rejected_before_source_observation(package,function,selection,monkeypatch):
    owner=getattr(native_matrix,package)
    ctx=context(tool='run_'+owner.package.lower())
    def no_observation(*args,**kwargs):
        raise AssertionError('Inapplicable selection must be diagnosed before observation')
    monkeypatch.setattr(type(ctx),'existing',no_observation)
    args=dict(function_name=function,parameter_selection=selection,input_path='/unobserved.h5ad')
    with pytest.raises(ScientificDiagnostic) as refusal:owner.prepare_operation(args,ctx)
    assert refusal.value.code=='NATIVE_PARAMETER_SELECTION_NOT_APPLICABLE'
    assert args['parameter_selection']==selection


def test_selection_applicability_is_shared_with_native_owner_and_preserves_eligibility():
    from agent.tools.analysis.native_clustering import ELIGIBLE
    for package,function in ELIGIBLE:
        owner=getattr(native_matrix,package.upper())
        for value in ('asw','native_defaults'):
            args=dict(function_name=function,parameter_selection=value)
            assert owner.parameter_selection_applicable(args)
            owner.native_arguments(args)
    args=dict(function_name='pp.neighbors',parameter_selection='native_defaults')
    with pytest.raises(ValueError,match='six adapted'):native_matrix.SCANPY.native_arguments(args)
    assert native_matrix.SCANPY.parameter_selection_applicable(dict(function_name='pp.neighbors'))


@pytest.mark.parametrize('selection',['asw','native_defaults'])
def test_inapplicable_clustering_selection_is_canonical_invalid_without_reservation(tmp_path,selection):
    import json
    from agent.application import preparation_state as records
    from test_preparation_owner_contracts import Model,setup
    class Selected(Model):
        def complete(self,*,prompt,response_schema):
            result=json.loads(super().complete(prompt=prompt,response_schema=response_schema))
            value=json.loads(prompt)
            if 'turn_schema_version' in value:
                result['decision']['arguments']=[dict(tool='run_scanpy',argument='parameter_selection',
                    literal=selection,evidence='parameter_selection='+selection,value=selection,native_function='pp.neighbors')]
            return json.dumps(result)
    model=Selected('run_scanpy','pp.neighbors',roles=False);app,executor=setup(tmp_path,model)
    result=app.submit_turn('session','task','Use pp.neighbors parameter_selection='+selection,
        expected_generation=0,execution_inputs=dict(input_path=str(source(tmp_path))))
    state=app._application.sessions.load('session')
    assert result.error.code=='NATIVE_PARAMETER_SELECTION_NOT_APPLICABLE',result.to_dict()
    assert records.observation(state.interactions[-1])['status']=='INVALID'
    assert not executor.plans and not state.turns and not state.revisions


def test_native_dpt_graph_and_root_are_preparable_without_invented_defaults(tmp_path):
    path=source(tmp_path);ctx=context()
    args=dict(input_path=str(path),function_name='tl.dpt')
    with pytest.raises(ScientificDiagnostic,match='graph') as absent:native_matrix.SCANPY.prepare_operation(args,ctx)
    assert absent.value.code=='NATIVE_GRAPH_REQUIRED'
    selected={'pca':('run_scanpy',dict(input_path=str(path),function_name='pp.pca')),
        'graph':('run_scanpy',dict(input_path=StepOutputRef('pca','matrix_path'),function_name='pp.neighbors',use_rep='X_pca'))}
    ctx=context(selected.items());args['input_path']=StepOutputRef('graph','matrix_path')
    with pytest.raises(ScientificInputRequired) as root:native_matrix.SCANPY.prepare_operation(args,ctx)
    req,=root.value.requirements
    assert req.argument=='root_cell' and req.kind=='scalar' and not req.choices
    assert native_matrix.SCANPY.prepare_operation(dict(args,root_cell='a'),ctx)['root_cell']=='a'
    assert 'root_cell' not in args
    promised=ctx.planned(args['input_path'])
    assert 'scanpy:neighbors' in promised['graphs'] and 'X_pca' in promised['representations']
    assert not promised.get('observed')


def test_planned_native_representation_consumed_automatically_by_cicero(tmp_path):
    path=source(tmp_path)
    selected={'pca':('run_scanpy',dict(input_path=str(path),function_name='pp.pca'))}
    ctx=context(selected.items(),tool='run_cicero_cds')
    args=dict(input_path=StepOutputRef('pca','matrix_path'),function_name='make_cicero_cds')
    assert ctx.tool.preparation_owner(args,ctx)['reduced_coordinates_key']=='X_pca'
    selected['pca'][1]['kwargs']={'inplace':False}
    assert 'produced_representation' not in ctx.planned(args['input_path'])


def test_mapping_reference_is_registered_producer_property(tmp_path):
    path=source(tmp_path)
    selected={'map':('run_chromap',dict(function_name='map',reference_path='/ref.fa',reference_sha256='a'*64)),
        'import':('run_snapatac2',dict(input_path=StepOutputRef('map','fragments_path'),function_name='pp.import_fragments'))}
    ctx=context(selected.items())
    promised=ctx.planned(StepOutputRef('import','matrix_path'))
    assert promised['source_reference']==dict(step_id='map',path='/ref.fa',sha256='a'*64)
    assert ctx.existing(StepOutputRef('import','matrix_path')) is None


def test_explicit_root_is_only_adapted_on_task_local_matrix(tmp_path):
    path=source(tmp_path);stage=tmp_path/'stage';stage.mkdir()
    matrix,_=native_matrix.prepare_input(dict(input_path=str(path),function_name='tl.dpt',root_cell='c'),stage)
    assert matrix.uns['iroot']==2 and 'iroot' not in ad.read_h5ad(path).uns
    with pytest.raises(ValueError,match='exact input observation'):
        native_matrix.prepare_input(dict(input_path=str(path),function_name='tl.dpt',root_cell='absent'),stage)


def test_diffmap_retains_graph_independently_of_counts_and_inspection_is_transparent(tmp_path):
    path=source(tmp_path)
    data=ad.read_h5ad(path);data.obsm['X_pca']=np.eye(4)
    data.obsp['connectivities']=np.eye(4);data.obsp['distances']=np.eye(4)
    data.uns.update(iroot=999,neighbors=dict(connectivities_key='connectivities',distances_key='distances'));data.write_h5ad(path)
    selected={'inspect':('inspect_scATAC',dict(path=str(path))),
        'graph':('run_scanpy',dict(input_path=StepOutputRef('inspect','input_path'),function_name='pp.neighbors',use_rep='X_pca')),
        'diffmap':('run_scanpy',dict(input_path=StepOutputRef('graph','matrix_path'),function_name='tl.diffmap'))}
    ctx=context(selected.items())
    assert native_matrix.SCANPY.prepare_operation(dict(input_path=StepOutputRef('inspect','input_path'),function_name='tl.leiden'),ctx)
    assert native_matrix.SCANPY.prepare_operation(dict(input_path=StepOutputRef('inspect','input_path'),function_name='pp.neighbors',use_rep='X_pca'),ctx)
    with pytest.raises(ScientificInputRequired):
        native_matrix.SCANPY.prepare_operation(dict(input_path=StepOutputRef('diffmap','matrix_path'),function_name='tl.dpt'),ctx)
    assert 'scanpy:neighbors' in ctx.planned(StepOutputRef('diffmap','matrix_path'))['graphs']


@pytest.mark.parametrize('adaptation',[dict(input_orientation='features_by_cells'),dict(cell_subset=dict(positions=[2,3]))])
def test_root_is_not_reused_across_changed_row_identity(tmp_path,adaptation):
    path=source(tmp_path);data=ad.read_h5ad(path)
    data.uns.update(iroot=0,neighbors=dict(connectivities_key='connectivities',distances_key='distances'))
    data.obsp['connectivities']=np.eye(4);data.obsp['distances']=np.eye(4);data.write_h5ad(path)
    args=dict(input_path=str(path),function_name='tl.dpt',**adaptation)
    assert native_matrix.SCANPY.input_properties(args,context())['root'] is False
    stage=tmp_path/'stage';stage.mkdir()
    with pytest.raises(ValueError,match='explicit root_cell'):
        native_matrix.prepare_input(args,stage)
    with pytest.raises(ScientificDiagnostic):
        native_matrix.SCANPY.prepare_operation(dict(args,root_cell='a'),context())
    assert native_matrix.SCANPY.input_properties(args,context())['graphs']==()


@pytest.mark.parametrize('kwargs,expected',[({},'X_spectral_mnn'),({'key_added':'corrected'},'corrected'),({'inplace':False},None)])
def test_mnc_and_tsne_exact_produced_representations(kwargs,expected,tmp_path):
    path=source(tmp_path);data=ad.read_h5ad(path);data.obsm['X_spectral']=np.eye(4);data.write_h5ad(path)
    selected={'correct':('run_snapatac2',dict(input_path=str(path),function_name='pp.mnc_correct',kwargs=kwargs))}
    ctx=context(selected.items())
    assert ctx.planned(StepOutputRef('correct','matrix_path')).get('produced_representation')==expected
    selected['tsne']=('run_scanpy',dict(input_path=str(path),function_name='tl.tsne'))
    ctx=context(selected.items())
    assert ctx.planned(StepOutputRef('tsne','matrix_path'))['produced_representation']=='X_tsne'


def test_root_parameter_has_only_dpt_alias_and_precise_owner_diagnosis(tmp_path):
    from agent.orchestration.native_parameters import input_name
    ctx=context();tool=ctx.tool
    sources=[s.selector for p in tool.semantic_planning.consumer_ports for s in p.request_sources]
    assert input_name('run_scanpy','tl.dpt','root_cell') in sources
    assert input_name('run_scanpy','pp.pca','root_cell') not in sources
    with pytest.raises(ScientificDiagnostic) as invalid:
        native_matrix.SCANPY.prepare_operation(dict(input_path=str(source(tmp_path)),function_name='pp.pca',root_cell='a'),ctx)
    assert invalid.value.code=='NATIVE_ROOT_NOT_APPLICABLE'


def test_symbolic_row_adaptation_invalidates_root_and_graph(tmp_path):
    path=source(tmp_path);data=ad.read_h5ad(path);data.uns['iroot']=0;data.write_h5ad(path)
    selected={'graph':('run_scanpy',dict(input_path=str(path),function_name='pp.neighbors'))}
    ctx=context(selected.items());args=dict(input_path=StepOutputRef('graph','matrix_path'),function_name='tl.dpt',cell_subset=dict(positions=[2,3]))
    props=native_matrix.SCANPY.input_properties(args,ctx)
    assert props['root'] is False and props['graphs']==()
    with pytest.raises(ScientificDiagnostic,match='graph'):native_matrix.SCANPY.prepare_operation(args,ctx)


def test_required_native_field_uses_owner_metadata_or_planned_groups(tmp_path,monkeypatch):
    path=source(tmp_path);data=ad.read_h5ad(path);data.obs['supplied']=pd.Categorical(['a','a','b','b']);data.write_h5ad(path)
    import sys
    monkeypatch.setenv('AGENT_SCANPY_PYTHON',sys.executable)
    monkeypatch.setattr(native_matrix,'_installed_parameter_metadata',lambda *a:{'groupby':{'required':True,'types':['str']}})
    ctx=context()
    with pytest.raises(ScientificInputRequired) as missing:
        native_matrix.SCANPY.prepare_operation(dict(input_path=str(path),function_name='tl.rank_genes_groups'),ctx)
    req,=missing.value.requirements
    assert req.member=='groupby' and req.choices==('supplied',)
    selected={'graph':('run_scanpy',dict(input_path=str(path),function_name='pp.neighbors')),
        'groups':('run_scanpy',dict(input_path=StepOutputRef('graph','matrix_path'),function_name='tl.leiden',kwargs={'key_added':'my_groups'}))}
    ctx=context(selected.items())
    bound=native_matrix.SCANPY.prepare_operation(dict(input_path=StepOutputRef('groups','matrix_path'),function_name='tl.rank_genes_groups'),ctx)
    assert bound['kwargs']['groupby']=='my_groups'


def test_root_observation_reads_only_bounded_axis_metadata(tmp_path,monkeypatch):
    import h5py
    from agent.tools.analysis import peak_h5ad
    path=source(tmp_path)
    # An oversized unallocated X must never be traversed to name a root.
    with h5py.File(path,'r+') as handle:
        del handle['X'];handle.create_dataset('X',shape=(1000000,1000000),dtype='f8',chunks=(1,1))
    assert native_matrix.selected_root_axis(path,{})==['a','b','c','d']
    with peak_h5ad.inspection_limits(8,8):
        with pytest.raises(ValueError,match='limit|budget'):
            native_matrix.selected_root_axis(path,{})
