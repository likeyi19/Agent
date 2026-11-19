"""Reviewed package functions on task-local AnnData/matrix objects.

Packages own their numerical methods. This owner shares only bounded input
adaptation, invocation, exact axes and the existing native publication lifecycle.
"""
import json
import csv
import os
from functools import lru_cache
from pathlib import Path
import shutil
import tempfile

import anndata as ad
import h5py
import numpy as np
from anndata.io import read_elem
from scipy import sparse
from scipy.io import mmwrite, mmread

from agent.schemas.orchestration import _serialize
from agent.schemas.verification_authority import AuthorityError, digest
from agent.tools._native_process import run, r_environment, NativeExecutionError
from agent.tools.data.authority_context import current
from agent.tools._file_identity import file_closure
from . import peak_h5ad
from .native_publication import NativePublication
from .sequence_prediction import arguments as path_arguments, library, read, write
from .sequence_reference import metadata_declarations


def native_kwargs(functions, args):
    """Shared data-only JSON/function boundary; scientific object checks stay with owners."""
    function, kwargs = args.get('function_name'), args.get('kwargs', {})
    if function not in functions or type(kwargs) is not dict or set(kwargs) - set(functions[function]):
        raise ValueError('Unsupported native function or argument; no method is substituted')
    def ordinary(value):
        if value is None or type(value) in (bool, int, float, str):
            return True
        if type(value) is list:
            return all(ordinary(v) for v in value)
        if type(value) is dict:
            return all(type(k) is str and ordinary(v) for k, v in value.items())
        return False
    try:
        if not ordinary(kwargs):
            raise ValueError('Native kwargs require data-only JSON values')
        json.dumps(kwargs, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError('Native kwargs must be representable as data-only strict JSON') from exc
    return kwargs


FACT_FIELDS = ('function_name', 'n_cells', 'n_features', 'measurement', 'parameters',
    'source_identity', 'software_versions', 'native_output_type', 'result_slots', 'limitations')
RESULT_FIELDS = dict(status=(str,), contract_version=(str,), manifest_path=(str,), manifest_sha256=(str,),
    matrix_path=(str,), matrix_sha256=(str,), function_name=(str,), n_cells=(int,), n_features=(int,),
    measurement=(str,), parameters=(dict,), source_identity=(dict,), software_versions=(dict,),
    native_output_type=(str,), result_slots=(list,), limitations=(str,))
DELIVERY_FILES = {
    'result.h5ad': ('Native result with exact ordered cells and features', 'application/octet-stream'),
    'metadata.json': ('Native arguments, source measurement and output interpretation', 'application/json'),
    'native-receipt.json': ('Installed native function completion and versions', 'application/json'),
}
LIMITATIONS = ('Independent native invocation, not a completed analysis workflow or biological validation. '
    'Source annotations are supplied metadata. Matrix transformations, feature masks and embeddings have distinct meanings. '
    'Native feature names such as genes do not establish biological feature identity. No inferred filtering, genome conversion or upstream rerun. Omitted native arguments retain installed defaults; '
    'stochastic results are qualified only for the recorded runtime and seed. Native nonfinite values are retained, not repaired or claimed usable.')


class NativeInvocationError(ValueError):
    """The native package failed on the unchanged invocation."""
    def __init__(self, message, *, code='NATIVE_FUNCTION_REJECTED'):
        if code not in {'NATIVE_FUNCTION_REJECTED', 'NATIVE_CLUSTER_COUNT_REQUIRED','NATIVE_ASW_UNAVAILABLE'}:
            raise ValueError('Unknown native invocation error code')
        self.code = code
        super().__init__(message)


def arguments(values):
    return path_arguments(_serialize(values))


def declarations(handle, args):
    slot = args.get('count_slot', 'X')
    if slot != 'X' and not slot.startswith('layers/'):
        slot = 'layers/' + slot
    effective = metadata_declarations(handle, slot, args, read_elem(handle['var']))
    # A declaration is not reference qualification or permission to relabel X.
    uns = handle.get('uns', {})
    recorded = {}
    for key in ('species', 'assembly', 'coordinate_system', 'matrix_semantics'):
        if key in uns and isinstance(value := read_elem(uns[key]), str):
            recorded[key] = value
    effective['_source_declarations'] = recorded
    measurement=peak_h5ad.matrix_measurement(handle,slot.replace('/','.'))
    if measurement is not None:
        effective['matrix_semantics']=measurement
    effective['count_slot'] = slot
    return effective


def prepare_input(args, stage, *, source=None):
    """AnnData owns encoding; only task-local objects can be mutated by packages."""
    peak_h5ad.inspect(args['input_path'], require_matrix=False, max_logical_bytes=None)
    with h5py.File(args['input_path'], 'r') as handle:
        effective = declarations(handle, args)
    original = ad.read_h5ad(args['input_path'])
    orientation = args.get('input_orientation', 'cells_by_features')
    if orientation == 'features_by_cells':
        original = original.T.copy()
    elif orientation != 'cells_by_features':
        raise ValueError('Unknown explicit AnnData input orientation')
    slot = effective['count_slot']
    x = original.X if slot == 'X' else original.layers[slot.removeprefix('layers/')]
    # The bounded HDF5/AnnData readers admit ordinary serialized annotations,
    # not arbitrary Python objects. Retain them exactly as native AnnData calls
    # do; a retained reduction is not asserted to have been recomputed from X.
    matrix = original.copy()
    matrix.X = x.copy() if x is not None else None
    effective['matrix_semantics'] = effective.get('matrix_semantics') or 'unspecified_retained_measurement'
    matrix.uns['matrix_semantics'] = effective['matrix_semantics']
    if args.get('group_column') is not None and args.get('grouping_path') is None:
        raise ValueError('An explicit projected group column requires its grouping source')
    if args.get('grouping_path') is not None:
        from .native_gene_activity import project_grouping
        matrix, effective['_grouping_projection'] = project_grouping(matrix, args, source)
    if args.get('cell_subset') is not None:
        matrix, membership = subset_input(matrix, args, source or file_closure([args['input_path']])[0])
        effective['_cell_subset'] = membership
    if args.get('function_name')=='tl.dpt' and args.get('root_cell') is None:
        root=matrix.uns.get('iroot')
        if (args.get('cell_subset') is not None or orientation=='features_by_cells'
                or not isinstance(root,(int,np.integer)) or isinstance(root,(bool,np.bool_)) or not 0<=root<matrix.n_obs):
            raise ValueError('DPT requires an explicit root_cell after row adaptation or a valid source uns.iroot.')
    if args.get('root_cell') is not None:
        if args.get('function_name')!='tl.dpt' or not matrix.obs_names.is_unique or args['root_cell'] not in matrix.obs_names:
            raise ValueError('DPT root_cell must name one exact input observation.')
        matrix.uns['iroot']=int(matrix.obs_names.get_loc(args['root_cell']))
    matrix.write_h5ad(stage / 'input.h5ad')
    return matrix, effective


def subset_input(matrix, args, source):
    """Derive selection provenance; AnnData owns the actual row indexing.

    Annotation-owned group selection supplies only groupby/groups on its bound
    source. Explicit positions may select part of a group or repeat observations.
    Hierarchical branch completeness/distinctness is not this adapter's policy.
    """
    from .native_gene_activity import axis_identity
    selection = args['cell_subset']
    if type(selection) is not dict or set(selection) - {'source_sha256', 'positions', 'cell_ids', 'population', 'groupby', 'groups'}:
        raise ValueError('Cell subset requires source positions or an actual source grouping')
    if 'source_sha256' in selection and selection['source_sha256'] != source['sha256']:
        raise AuthorityError('Cell subset source bytes changed')
    group_positions = None
    if 'groupby' in selection or 'groups' in selection:
        if type(selection.get('groups')) is not list or selection.get('groupby') not in matrix.obs:
            raise ValueError('Select an actual obs grouping and group values')
        group_positions = np.flatnonzero(matrix.obs[selection['groupby']].isin(selection['groups'])).tolist()
    positions = selection.get('positions', group_positions)
    if type(positions) is not list or any(type(i) is not int or not 0 <= i < matrix.n_obs for i in positions):
        raise AuthorityError('Cell subset positions do not identify source observations')
    identities = matrix.obs_names[positions].tolist()
    if 'cell_ids' in selection and selection['cell_ids'] != identities:
        raise AuthorityError('Cell subset identities do not match exact source positions')
    membership = dict(selection, source_sha256=source['sha256'], positions=positions, cell_ids=identities,
        source_axes=axis_identity(args['input_path']),
        position_axis='var' if args.get('input_orientation') == 'features_by_cells' else 'obs',
        provenance='exact source positions',
        inherited_slot_meaning='AnnData-sliced source slots; global graphs and coordinates are not newly computed local results')
    if group_positions is not None:
        if not set(positions) <= set(group_positions):
            raise AuthorityError('Cell subset differs from the selected source grouping membership')
        membership['provenance'] = 'exact source obs membership; group labels retain their source meaning'
        membership['complete_group_membership'] = set(positions) == set(group_positions)
    return matrix[positions, :].copy(), membership


@lru_cache(maxsize=128)
def _installed_parameter_metadata(executable, package, function):
    import subprocess
    script = Path(__file__).with_name('native_matrix_worker.py')
    try:
        result = subprocess.run([executable, '-B', str(script), '--parameters', package, function],
            capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except (OSError, subprocess.CalledProcessError, ValueError):
        return {}


def selected_root_axis(path, arguments):
    """Read only the explicitly selected root axis and optional subset field."""
    from .categorical import _series
    from .peak_h5ad import metadata_item
    with h5py.File(path,'r') as handle:
        axis='var' if arguments.get('input_orientation')=='features_by_cells' else 'obs'
        group=metadata_item(handle,axis)
        index_path=axis+'/'+group.attrs['_index']
        selection=arguments.get('cell_subset')
        paths=[index_path]+([axis+'/'+selection['groupby']] if selection is not None and selection.get('groupby') else [])
        peak_h5ad.inspect(handle,require_matrix=False,metadata_paths=paths)
        ids=list(map(str,_series(metadata_item(handle,index_path))))
        selection=arguments.get('cell_subset')
        if selection is not None:
            positions=selection.get('positions')
            if positions is None and selection.get('groupby'):
                values=_series(metadata_item(handle,axis+'/'+selection['groupby']))
                positions=np.flatnonzero(values.isin(selection['groups'])).tolist()
            if positions is not None:
                ids=[ids[i] for i in positions]
        return ids


class MatrixPackage:
    """One package interface, several compatible public functions, one lifecycle."""
    FACT_FIELDS, RESULT_FIELDS, DELIVERY_FILES = FACT_FIELDS, RESULT_FIELDS, DELIVERY_FILES
    arguments = staticmethod(arguments)

    def input_properties(self, arguments, context, *, seen=()):
        props=dict(self._input_properties(arguments,context,seen=seen))
        if arguments.get('cell_subset') is not None or arguments.get('input_orientation')=='features_by_cells':
            props.update(root=False,graphs=())
            if arguments.get('input_orientation')=='features_by_cells':props['representations']=()
        return props

    def _input_properties(self, arguments, context, *, seen=()):
        """Shallow existing metadata and symbolic promises remain distinguishable."""
        from agent.schemas import StepOutputRef
        source=arguments.get('input_path')
        observed=context.existing(source)
        if observed is None and isinstance(source,StepOutputRef):
            return context.planned(source,seen=seen)
        if observed is None:
            return {}
        from agent.tools._file_identity import checked_file
        checked_file(observed.path,observed.sha256)
        with h5py.File(observed.path,'r') as handle:
            from .peak_h5ad import metadata_item
            representations=metadata_item(handle,'obsm')
            reps=tuple(representations) if isinstance(representations,h5py.Group) else ()
            graphs=[]
            uns=metadata_item(handle,'uns')
            if isinstance(uns,h5py.Group):
                for key in uns:
                    item=metadata_item(handle,'uns/'+key)
                    if isinstance(item,h5py.Group) and all(n in item for n in ('connectivities_key','distances_key')):
                        selected_paths=['uns/'+key+'/'+n for n in ('connectivities_key','distances_key')]
                        peak_h5ad.inspect(handle,require_matrix=False,metadata_paths=selected_paths,max_logical_bytes=1024**2)
                        names=[read_elem(metadata_item(handle,name)) for name in selected_paths]
                        if all(isinstance(n,str) and metadata_item(handle,'obsp/'+n) is not None for n in names):
                            # Presence is a native schema check, never package
                            # provenance or scientific graph qualification.
                            graphs.append('scanpy:'+key)
            root=metadata_item(handle,'uns/iroot')
            valid_root=isinstance(root,h5py.Dataset) and root.shape==() and root.dtype.kind in 'iu' and 0<=int(root[()])<len(handle['obs'][handle['obs'].attrs['_index']])
            if metadata_item(handle,'obsp/distances') is not None:
                graphs.append('snapatac2:distances')
        if arguments.get('cell_subset') is not None or arguments.get('input_orientation')=='features_by_cells':
            valid_root=False
            graphs=[]
        contents=peak_h5ad.scientific_properties(observed.path, ('x','fragment_storage','paired_fragments','native_values','storage_kind'))
        checked_file(observed.path,observed.sha256)
        return dict(contents,representations=reps,graphs=tuple(graphs),root=valid_root,observed=True)

    def scientific_contract(self, arguments):
        from dataclasses import replace
        from functools import partial
        from agent.orchestration.scientific_inputs import ScientificContract, ScientificDiagnostic
        from . import snapatac_raw
        from .cell_result import count_measurement, grouping_key, matrix_transform, representation_key
        from .native_clustering import selection_contract
        function=arguments.get('function_name');kwargs=arguments.get('kwargs',{})
        if function not in self.FUNCTIONS:
            return ScientificContract()
        resolver=partial(self._inherited_output_properties,arguments)
        if self.package=='snapatac2' and function in snapatac_raw.FUNCTIONS:
            return replace(snapatac_raw.scientific_contract(arguments),resolve_output=resolver)
        matrix_functions={'pp.pca','tl.pca','pp.log1p','pp.scale','pp.normalize_total',
            'pp.calculate_qc_metrics','pp.highly_variable_genes','pp.select_features',
            'pp.filter_features','pp.binarize','pp.tfidf','tl.spectral','tl.diff_test',
            'RunTFIDF','FindTopFeatures','RunSVD','BinarizeCounts','corSparse','Jaccard'}
        dataset={'format':'h5ad'};parameters={};effects={};output={'format':'h5ad'}
        native_rep=kwargs.get('use_rep',arguments.get('use_rep'))
        snap_obsm=self.package=='snapatac2' and function in ('pp.knn','tl.umap','tl.dbscan','pp.mnc_correct')
        if function in matrix_functions:
            target=self._matrix_slot(arguments.get('count_slot','X'))
            if kwargs.get('layer'):target='layers.'+kwargs['layer']
            elif kwargs.get('obsm'):target='obsm.'+kwargs['obsm']
            elif kwargs.get('use_raw'):target='raw.X'
            dataset[target]=True
        if self.package in ('scanpy','episcanpy') and function in ('pp.neighbors','tl.tsne') and native_rep is None:
            slot=self._matrix_slot(arguments.get('count_slot','X'))
            dataset[('scanpy_auto_rep' if kwargs.get('n_pcs')!=0 else 'scanpy_x_rep') if slot=='x' else slot]=True
        if self.package in ('scanpy','episcanpy') and native_rep=='X':
            slot=self._matrix_slot(arguments.get('count_slot','X'))
            dataset['scanpy_x_rep' if slot=='x' else slot]=True
        if arguments.get('count_slot','X')!='X':dataset[self._matrix_slot(arguments['count_slot'])]=True
        if function=='tl.spectral':
            features=kwargs.get('features','selected')
            parameters['features']={'default':'selected','string_axis':'var','null':'all_features'}
            if isinstance(features,str):dataset['var.'+features]='boolean'
        representation=kwargs.get('use_rep',arguments.get('use_rep'))
        if self.package=='episcanpy' and function in ('tl.kmeans','tl.hc'):representation='X_pca'
        if self.package=='snapatac2' and function in ('pp.knn','tl.umap','tl.dbscan') and representation is None:
            representation='X_spectral'
            parameters['use_rep']={'default':representation,'string_axis':'obsm'}
        if isinstance(representation,str) and (representation!='X' or snap_obsm):dataset['obsm.'+representation]=True
        if self.package in ('scanpy','episcanpy') and function in ('tl.leiden','tl.louvain','tl.umap','tl.dpt','tl.diffmap','tl.paga'):
            if kwargs.get('obsp') is not None and kwargs.get('neighbors_key') is not None:
                raise ScientificDiagnostic('NATIVE_GRAPH_SELECTOR_CONFLICT','Select either obsp or neighbors_key, not both.')
            if kwargs.get('obsp'):
                dataset['obsp.'+kwargs['obsp']]=True
            else:
                role='distances' if function=='tl.paga' and kwargs.get('model','v1.2')=='v1.2' else 'connectivities'
                dataset['graph.scanpy_'+role+'.'+(kwargs.get('neighbors_key') or 'neighbors')]=True
            if 'obsp' in self.FUNCTIONS[function]:parameters['adjacency']={'alternatives':('obsp[kwargs.obsp]','graph[kwargs.neighbors_key]')}
        if self.package=='snapatac2' and function=='tl.leiden':dataset['graph.snapatac2_distances']=True
        changed_cells=arguments.get('cell_subset') is not None or arguments.get('input_orientation')=='features_by_cells'
        if changed_cells:
            invalidated=tuple(name for name in dataset if name.startswith(('graph.','obsp.')) or
                arguments.get('input_orientation')=='features_by_cells' and name.startswith('obsm.'))
            if invalidated:parameters['invalidated_input_properties']={'dataset':invalidated}
        if function=='tl.dpt':parameters['root']={'alternatives':('root_cell','uns.iroot'),'diffmap':'native_if_missing'}
        if self.package=='episcanpy' and function=='pp.lsi':
            layer=kwargs.get('tf_idf_layer','tf-idf')
            parameters['tf_idf_layer']={'default':'tf-idf','string_axis':'layers'}
            if not isinstance(layer,str) or not layer:
                raise ScientificDiagnostic('NATIVE_TFIDF_LAYER_REQUIRED','Supply the exact nonempty TF-IDF layer key.')
            dataset['layers.'+layer]=True
        if self.package=='episcanpy' and function=='tl.geneactivity':
            dataset[self._matrix_slot(arguments.get('count_slot','X'))]=True
            parameters['resources']=('gene_anno',)
            effects['feature_axis']='replace'
            output.update(x=True,measurement='episcanpy_gene_domain_sum')
        if self.package=='episcanpy' and function=='tl.rank_features':
            parameters['method_attribution']='wrapper_may_replace_requested_statistic; publication_reports_native_result'
            if kwargs.get('method','')!='':dataset['uns.omic']=True
        requires={'dataset':dataset}
        if self.package=='Signac' and function in ('corSparse','Jaccard'):
            requires['second']={'format':'h5ad', self._matrix_slot(arguments.get('second_count_slot','X')):True}
            if function=='Jaccard':parameters['required_ports']=('second',)
            output['uns.Signac_pairwise' if function=='Jaccard' or arguments.get('second_path') is not None else 'obsp.Signac_corSparse']=True
        if self.package=='Signac' and function=='FindTopFeatures':
            output.update(dict.fromkeys(('var.Signac_count','var.Signac_percentile','var.Signac_native_rank'),True))
        if self.package=='Signac' and function=='RunSVD':output.update({'varm.LSI':True,'uns.Signac_svd':True})
        inplace=kwargs.get('inplace',True) is not False
        if inplace:
            if rep:=representation_key(self.package,function,kwargs):output['obsm.'+rep]=True
            if group:=grouping_key(self.package,function,kwargs):output['obs.'+group]=True
            if function=='pp.select_features':output['var.selected']='boolean'
            if function=='pp.tfidf':output['layers.'+kwargs.get('layer_tfidf_key','tf-idf')]=True
            if function=='pp.neighbors':
                key=kwargs.get('key_added')
                for role in ('connectivities','distances'):
                    slot=((key+'_') if key else '')+role
                    if key!='neighbors':
                        output['graph.scanpy_'+role+'.'+(key or 'neighbors')]=True
                        output['graph_slot.scanpy_'+role+'.'+(key or 'neighbors')]=slot
                    output['obsp.'+slot]=True
                if key is None:output['graph.snapatac2_distances']=False
            if function=='pp.knn':output.update({'obsp.distances':True,'graph.snapatac2_distances':True,
                                                'graph.scanpy_distances.neighbors':False})
        else:effects['independent_return']=True
        if arguments.get('cell_subset') is not None:effects['cell_axis']='subset'
        if arguments.get('input_orientation')=='features_by_cells':effects['axes']='transpose'
        if arguments.get('count_slot','X')!='X':effects['x_from']=self._matrix_slot(arguments['count_slot'])
        target, measurement=matrix_transform(self.package,function,kwargs)
        if function=='pp.tfidf':
            effects['measurement_change']={'layers.'+kwargs.get('layer_tfidf_key','tf-idf'):'episcanpy_tfidf'}
        if target is not None:
            effects['measurement_change']={target:measurement}
            if function=='pp.normalize_total' and isinstance(kwargs.get('layers'),(list,tuple)):
                effects['measurement_change'].update({'layers.'+name:measurement for name in kwargs['layers']})
            if function=='pp.normalize_total' and kwargs.get('layers')=='all':
                effects['all_layer_measurements']=measurement
            if target=='X':output.update(x=True,measurement=measurement)
        for slot, measurement in effects.get('measurement_change', {}).items():
            if not slot.startswith('obsm.'):
                output['measurement.'+slot]=measurement
                if (counts:=count_measurement(measurement)) is not None:
                    output['count_values.'+slot]=counts
        return ScientificContract(inputs=requires,parameters=parameters,outputs={'dataset':output},effects=effects,
                                  selection=selection_contract(self.package,function),resolve_output=resolver)

    @staticmethod
    def _matrix_slot(slot):
        return 'x' if slot=='X' else 'layers.'+slot.removeprefix('layers/')

    def _inherited_output_properties(self, arguments, port, context, *, seen=(), names=()):
        if port not in ('dataset','fragments'):
            return {}
        function=arguments.get('function_name');kwargs=arguments.get('kwargs',{})
        from .cell_result import grouping_key, matrix_transform, planned_count_preservation, representation_key
        from .snapatac_raw import FUNCTIONS as RAW_FUNCTIONS
        source=arguments.get('input_path')
        requested=tuple(name for name in names if '.' in name or name in ('scanpy_auto_rep','scanpy_x_rep'))
        if self.package=='snapatac2' and function in RAW_FUNCTIONS:
            inherited=(self.input_properties(arguments,context,seen=seen)
                if function not in ('pp.make_fragment_file','pp.import_fragments') else
                context.planned(source,seen=seen))
            props=dict(inherited) if function not in ('pp.make_fragment_file','pp.import_fragments') else {}
            if inherited.get('source_reference'):props['source_reference']=inherited['source_reference']
            if function not in ('pp.make_fragment_file','pp.import_fragments'):
                if function in ('pp.add_tile_matrix','pp.make_peak_matrix','pp.make_gene_matrix'):
                    requested=tuple(key for key in requested if not key.startswith(('var.','layers.','measurement.','count_values.')))
                    if not kwargs.get('inplace',function=='pp.add_tile_matrix'):
                        requested=tuple(key for key in requested if key.startswith('obs.'))
                if kwargs.get('inplace',True) and function in ('pp.filter_cells','pp.call_cells'):
                    requested=tuple(key for key in requested if not key.startswith(('graph.','graph_slot.')))
                    props={key:value for key,value in props.items() if not key.startswith(('graph.','graph_slot.'))}
                if requested:props.update(context.properties(source,requested,seen=seen))
            if function in ('pp.add_tile_matrix','pp.make_peak_matrix','pp.make_gene_matrix'):
                from .snapatac_raw_worker import peak_measurement
                storage='X_region_aggregation' if kwargs.get('use_x') else inherited.get('storage_kind')
                props.pop('measurement',None)
                if storage is not None:
                    strategy=kwargs.get('counting_strategy',arguments.get('counting_strategy','paired-insertion'))
                    measurement=peak_measurement(storage,strategy)
                    if function=='pp.make_gene_matrix':
                        measurement='snapatac2_gene_region_sum' if kwargs.get('use_x') else 'snapatac2_gene_'+measurement
                    from .cell_result import count_measurement
                    props['measurement']=props['measurement.X']=measurement
                    if (counts:=count_measurement(measurement)) is not None:props['count_values.X']=counts
                if not kwargs.get('inplace',function=='pp.add_tile_matrix'):
                    props={name:value for name,value in props.items()
                           if name!='storage_kind' and not name.startswith(('obsm.','obsp.','graph.','uns.','var.','varm.','layers.',
                                                                           'count_values.layers.','measurement.layers.','regions.'))}
            props.pop('observed',None)
            return props
        inherited=self.input_properties(arguments,context,seen=seen)
        # Slot retention and count preservation are distinct native properties.
        # Reviewed matrix functions retain cell slots unless the input adapter
        # changes the observation axis. Raw ingestion is handled above.
        retains_slots=(self.package in ('scanpy','episcanpy','snapatac2')
            and function in self.FUNCTIONS and arguments.get('cell_subset') is None
            and arguments.get('input_orientation','cells_by_features')=='cells_by_features')
        props=dict(inherited) if retains_slots else {}
        # read_native presents an explicitly selected layer as native X. This
        # is a source selection, not an inferred or reconstructed measurement.
        if function=='tl.geneactivity':
            props={name:value for name,value in props.items() if not name.startswith(('layers.','var.','varm.'))}
            requested=tuple(name for name in requested if not name.startswith(('layers.','var.','varm.','count_values.layers.','measurement.layers.')))
        if retains_slots and requested:props.update(context.properties(source,requested,seen=seen))
        if function=='pp.knn' and kwargs.get('inplace',True):
            selectors=tuple(name for name in requested if name.startswith('graph.scanpy_'))
            slots=tuple('graph_slot.'+name.partition('.')[2] for name in selectors)
            resolved=context.properties(source,slots,seen=seen)
            for name, slot in zip(selectors,slots):
                if resolved.get(slot)=='distances':props[name]=False
        if function=='pp.normalize_total' and kwargs.get('layers')=='all':
            _, measurement=matrix_transform(self.package,function,kwargs)
            if measurement is not None:
                for name in requested:
                    if name.startswith('measurement.layers.'):props[name]=measurement
                    if name.startswith('count_values.layers.'):props[name]=False
        if arguments.get('count_slot','X')!='X':
            props.pop('measurement',None)
            slot=self._matrix_slot(arguments['count_slot'])
            props.pop('x',None)
            props.pop('count_values.X',None)
            props.pop('measurement.X',None)
            selected=context.properties(source,(slot,'measurement.'+slot,'count_values.'+slot),seen=seen)
            if slot in selected:
                props['x']=selected[slot]
                if selected[slot]:props.update(scanpy_auto_rep=True,scanpy_x_rep=True)
            if 'measurement.'+slot in selected:
                props['measurement']=props['measurement.X']=selected['measurement.'+slot]
            if 'count_values.'+slot in selected:props['count_values.X']=selected['count_values.'+slot]
        props.pop('observed',None)
        props.pop('produced_representation',None)
        if inherited.get('source_reference'):
            props['source_reference']=inherited['source_reference']
        props['preserves_counts']=planned_count_preservation(context.tool.name,arguments)
        inplace=kwargs.get('inplace',True) is not False
        reps=set(props.get('representations',()))
        if inplace:
            rep = representation_key(self.package,function,kwargs)
            if rep:
                reps.add(rep)
                props['produced_representation']=rep
            group=grouping_key(self.package,function,kwargs)
            if group:props.update(grouping_key=group,grouping_step=context.step_id)
            graphs=set(props.get('graphs',()))
            if function=='pp.neighbors' and self.package in ('scanpy','episcanpy'):
                graphs.add('scanpy:'+kwargs.get('key_added','neighbors'))
            if function=='pp.knn' and self.package=='snapatac2':graphs.add('snapatac2:distances')
            props['graphs']=tuple(sorted(graphs))
        props['representations']=tuple(sorted(reps))
        return props

    def prepare_operation(self, arguments, context):
        """Preparable presence checks; native numerical/graph validation still runs."""
        from agent.orchestration.scientific_inputs import ScientificDiagnostic, ScientificInputRequired, parameter_requirements
        function=arguments.get('function_name');kwargs=arguments.get('kwargs',{})
        if function is not None and not self.parameter_selection_applicable(arguments):
            raise ScientificDiagnostic('NATIVE_PARAMETER_SELECTION_NOT_APPLICABLE',
                'Parameter selection applies only to the six adapted native clustering functions.')
        if arguments.get('root_cell') is not None and function!='tl.dpt':
            raise ScientificDiagnostic('NATIVE_ROOT_NOT_APPLICABLE','root_cell applies only to DPT.')
        if function is None or 'input_path' not in arguments:
            return arguments
        if self.package=='Signac':
            return arguments
        from .snapatac_raw import FUNCTIONS as RAW_FUNCTIONS
        if self.package=='snapatac2' and function in RAW_FUNCTIONS:
            from .snapatac_raw import prepare_operation
            return prepare_operation(arguments,context)
        if arguments.get('require_asw') and arguments.get('parameter_selection')=='asw':
            policy=self.scientific_contract(arguments).selection.get('asw',{})
            if policy.get('parameter')=='resolution' and 'resolution' not in kwargs:
                value=arguments.get('input_path')
                source=context.existing(value)
                if source is not None:
                    from .native_clustering import matching_graphs
                    if self.package=='snapatac2':
                        slot='distances'
                    else:
                        key='graph_slot.scanpy_connectivities.'+(kwargs.get('neighbors_key') or 'neighbors')
                        slot=kwargs.get('obsp') or context.properties(value,(key,)).get(key)
                    records=matching_graphs(self.package,slot,source.graph_inputs)
                    if len(records)!=1 or records[0].get('representation_qualified') is False:
                        raise ScientificDiagnostic('ASW_GRAPH_PREREQUISITE',
                            'Requested ASW needs exact graph-to-representation provenance; select a verified native neighbor result.')
        metadata=self.parameter_metadata(function)
        missing=[name for name in self.FUNCTIONS.get(function,()) if metadata.get(name,{}).get('required') and name not in kwargs]
        # The authorized ASW adapter resolves these native count operands after
        # selected-method execution; an explicit native-default request cannot.
        if function in ('tl.kmeans','tl.hc') and arguments.get('parameter_selection')!='native_defaults':
            missing=[name for name in missing if name!='num_clusters']
        values=dict(arguments);bound_kwargs=dict(kwargs);needs=[]
        for name in missing:
            known=context.planned_parameter_value(arguments,metadata[name])
            if known is not None:
                bound_kwargs[name]=known
                continue
            from agent.orchestration.scientific_inputs import ScientificInputRequirement
            observed=context.existing(arguments.get('input_path'))
            choices=observed.fields('obs') if observed is not None and name in ('groupby','groups','batch') else ()
            needs.append(ScientificInputRequirement('kwargs','kwargs','scalar',member=name,
                choices=tuple(choices),dependencies=('source:bytes',),
                applicability=dict(expected_types=metadata.get(name,{}).get('types',()),native_required=True),
                messages=('Supply the native required '+name+' for '+function+'.',)))
        if bound_kwargs!=kwargs:values['kwargs']=bound_kwargs
        if needs:raise ScientificInputRequired(*needs,bindings=values)
        arguments=values
        if function=='tl.dpt':
            if arguments.get('root_cell'):
                observed=context.existing(arguments.get('input_path'))
                if observed is not None:
                    ids=selected_root_axis(observed.path,arguments)
                    if ids.count(arguments['root_cell'])!=1:
                        raise ScientificDiagnostic('NATIVE_ROOT_INVALID','DPT root_cell must identify one observation in the selected input population.')
            elif not self.input_properties(arguments,context).get('root'):
                raise ScientificInputRequired(*parameter_requirements(context.tool,('root_cell',)),bindings=arguments)
        return arguments

    def __init__(self, package, version, runtime, functions, native_arguments, guidance):
        self.package, self.version, self.runtime = package, version, runtime
        if package == 'Signac':
            self.invoke = self.invoke_signac
        if package == 'episcanpy':
            self.invoke = self.invoke_episcanpy
        if package == 'snapatac2':
            from . import snapatac_raw as raw
            self.invoke = self.invoke_snapatac2
            functions = {**functions, **raw.FUNCTIONS}
            native_arguments = tuple(native_arguments) + tuple((key, 'Native ' + key + '; omitted retains its installed default.')
                for key in sorted(set().union(*map(set, raw.FUNCTIONS.values())) - {k for k, _ in native_arguments}))
            self.RESULT_FIELDS = dict(RESULT_FIELDS, matrix_path=(str, type(None)), matrix_sha256=(str, type(None)),
                n_cells=(int, type(None)), n_features=(int, type(None)),
                fragments_path=(str, type(None)), fragments_sha256=(str, type(None)))
            self.DELIVERY_FILES = {**DELIVERY_FILES, **raw.RAW_FILES}
        self.FUNCTIONS, self.guidance = functions, guidance
        members = set().union(*map(set, functions.values()))
        self.NATIVE_ARGUMENTS = tuple((key, text) for key, text in native_arguments if key in members)
        self.TOOL = 'run_' + package.lower()
        self.KIND = 'native_' + package.lower() + '_matrix'
        self.CONTRACT = self.KIND.replace('_', '-') + '.v1'
        self.POLICY = self.CONTRACT.replace('.v1', '-v1')
        # The profile identifies the adapter, not its growing function catalog.
        # Exact function, kwargs and installed implementation remain pinned in
        # each publication. Compatible additions need no historical migration.
        self.PROFILE = digest(dict(contract=self.CONTRACT, version=version,
            adaptation='ordinary AnnData; native returns and exact subset positions.v4'))
        self.VERIFIER = dict(id='agent.' + self.KIND, compatibility_version='1')
        if package == 'Signac':
            self.PROFILE = digest(dict(contract=self.CONTRACT, version=version, adaptation='ordered independent pairwise native matrices.v5'))
        if package == 'snapatac2':
            self.PROFILE = digest(dict(contract=self.CONTRACT, version=version,
                adaptation='native raw and matrix functions; lossless KD-tree f32 execution view; retained representation.v6'))
        self.publication = NativePublication(tool=self.TOOL, kind=self.KIND, contract=self.CONTRACT,
            profile=self.PROFILE, verifier=self.VERIFIER, files=raw.files if package == 'snapatac2' else tuple(DELIVERY_FILES),
            arguments=arguments, outputs=raw.outputs if package == 'snapatac2' else {'matrix': 'result.h5ad'})
        for name in ('load', 'result', 'verify_publication', 'verify_result', 'recover'):
            setattr(self, name, getattr(self.publication, name))

    def output_available(self, args, output_key):
        """Static publication availability from the same native output owner."""
        if self.package != 'snapatac2':
            return True
        from .snapatac_raw import outputs
        produced = outputs({'arguments': args})
        for name in ('matrix', 'fragments'):
            if output_key in (name + '_path', name + '_sha256'):
                return produced[name] is not None
        return True

    def parameter_selection_applicable(self, args):
        from .native_clustering import ELIGIBLE
        selection = args.get('parameter_selection')
        return selection is None or ((self.package, args.get('function_name')) in ELIGIBLE
            and selection in ('asw', 'native_defaults'))

    def native_arguments(self, args):
        function = args.get('function_name')
        kwargs = native_kwargs(self.FUNCTIONS, args)
        if args.get('gene_activity') is not None and type(args['gene_activity']) is not bool:
            raise ValueError('gene_activity is an explicit input declaration')
        if self.package == 'snapatac2':
            from .snapatac_raw import FUNCTIONS
            if function in FUNCTIONS and any(k in args for k in ('gene_activity', 'cell_subset', 'input_orientation')):
                raise ValueError('AnnData input adaptations apply to native matrix consumers, not raw functions')
        if not self.parameter_selection_applicable(args):
            raise ValueError('Parameter selection applies only to the six adapted native clustering functions')
        if args.get('use_rep') is not None:
            if 'use_rep' not in self.FUNCTIONS[function]:
                raise ValueError('Representation selection requires a native use_rep parameter')
            kwargs = dict(kwargs)
            kwargs.setdefault('use_rep', args['use_rep'])
        if self.package == 'snapatac2':
            from . import snapatac_raw as raw
            qc=raw.active_qc(args)
            if qc is not None:
                if qc not in ('otsu','native_defaults') or function not in ('pp.import_fragments', 'pp.filter_cells'):
                    raise ValueError('Otsu threshold selection applies only to uncalled import or native filtering')
                if qc=='otsu' and function == 'pp.import_fragments':
                    kwargs = dict(kwargs)
                    kwargs.setdefault('min_num_fragments', 0)
            if args.get('counting_strategy') is not None:
                if function not in ('pp.make_peak_matrix', 'pp.add_tile_matrix', 'pp.make_gene_matrix'):
                    raise ValueError('counting_strategy applies only to pp.make_peak_matrix, pp.add_tile_matrix or pp.make_gene_matrix')
                if 'counting_strategy' in kwargs and kwargs['counting_strategy'] != args['counting_strategy']:
                    raise ValueError('Explicit native kwargs and scalar have conflicting counting_strategy values')
                kwargs = dict(kwargs, counting_strategy=args['counting_strategy'])
            if args.get('fragment_order') is not None:
                if function != 'pp.import_fragments' or args['fragment_order'] not in ('unknown', 'barcode'):
                    raise ValueError('fragment_order is source knowledge for native fragment import only')
                kwargs = dict(kwargs)
                kwargs.setdefault('sorted_by_barcode', args['fragment_order'] == 'barcode')
            if function in raw.FUNCTIONS:
                raw.input_paths(args)
            elif any(args.get(path) is not None or args.get(sha) is not None for path, sha, _ in raw.PORTS.values()):
                raise ValueError('Raw reference files do not apply to this native matrix function')
        if self.package == 'episcanpy':
            if function == 'tl.geneactivity':
                if not args.get('gene_anno_path'):
                    raise ValueError('episcanpy.tl.geneactivity requires the gene_anno GTF input file')
            elif args.get('gene_anno_path') is not None or args.get('gene_anno_sha256') is not None:
                raise ValueError('Gene annotation applies only to episcanpy.tl.geneactivity')
        if (args.get('second_path') or args.get('second_count_slot') or args.get('second_sha256')) and (self.package != 'Signac' or function not in ('corSparse', 'Jaccard')):
            raise ValueError('A second matrix applies only to Signac corSparse/Jaccard')
        if function == 'Jaccard' and not args.get('second_path'):
            raise ValueError('Signac Jaccard requires two actual matrix sources')
        # A supplied spectral mask is scientific input, not ordinary kwargs.
        # This interface binds the exact existing var field (or all features).
        if args['function_name'] == 'tl.spectral' and isinstance(kwargs.get('features'), (list, dict)):
            raise ValueError('Bind a spectral feature mask through the input AnnData var field')
        if args['function_name'] == 'tl.umap' and any(isinstance(kwargs.get(k), (list, dict)) for k in ('init', 'init_pos')):
            raise ValueError('UMAP initialization arrays are scientific objects, not ordinary kwargs')
        for key in ('layer_tfidf_key', 'tf_idf_layer', 'features', 'use_rep', 'key_added'):
            value = kwargs.get(key)
            if isinstance(value, str) and ('/' in value or '\\' in value or value in ('.', '..')):
                raise ValueError('Native object keys must be local field names')
        return kwargs

    def parameter_metadata(self, function):
        """Inspect the same installed runtime used by native dispatch; run no science."""
        if function not in self.FUNCTIONS or self.package == 'Signac':
            return {}
        override = os.environ.get('AGENT_' + self.package.upper() + '_PYTHON')
        try:
            executable = Path(override) if override else library().root.parent / 'runtimes' / self.runtime / 'env/bin/python'
        except NativeExecutionError:
            executable=None
        metadata=(_installed_parameter_metadata(str(executable), self.package, function)
                  if executable is not None and executable.is_file() else {})
        if self.package=='snapatac2':
            from .snapatac_raw import parameter_metadata
            metadata=parameter_metadata(function,metadata)
        # Required grouping operands may use a registered producer's guaranteed
        # grouping key. This same rule owns preparation and final scope checks.
        return {name:({**item,'value_from_planned_input':dict(port='dataset',property='grouping_key')}
            if name in ('groupby','groups') and item.get('required') else item) for name,item in metadata.items()}

    def native(self, args, stage, matrix):
        is_r = self.package == 'Signac'
        override = os.environ.get('AGENT_' + self.package.upper() + ('_RSCRIPT' if is_r else '_PYTHON'))
        executable = Path(override) if override else library().root.parent / 'runtimes' / self.runtime / 'env/bin' / ('Rscript' if is_r else 'python')
        if is_r:
            from .native_runtimes import rscript
            executable = rscript(self.package)
        from . import snapatac_raw as raw
        raw_call = self.package == 'snapatac2' and args['function_name'] in raw.FUNCTIONS
        script = Path(__file__).parent / ('snapatac_raw_worker.py' if raw_call else 'r/signac_functions_v1.R' if is_r else 'native_matrix_worker.py')
        if not executable.is_file():
            raise NativeExecutionError('NATIVE_EXECUTION_UNAVAILABLE',
                f'Provision the isolated {self.package} {self.version} scientific runtime')
        implementation_paths = [executable.resolve(), script]
        if raw_call:
            implementation_paths.append(script.with_name('native_matrix_worker.py'))
        if not is_r:
            implementation_paths.append(executable.parent.parent / 'pyvenv.cfg')
            if (args.get('grouping_path') is not None or
                    (self.package, args['function_name']) in {('snapatac2', 'pp.make_gene_matrix'), ('episcanpy', 'tl.geneactivity')}):
                implementation_paths.append(script.with_name('native_gene_activity.py'))
            if not raw_call:
                implementation_paths.extend(script.with_name(name) for name in ('native_clustering.py', 'cell_result.py'))
        implementation = file_closure(implementation_paths)
        request = dict(stage=str(stage), package=self.package, version=self.version,
            function_name=args['function_name'], kwargs=self.native_arguments(args), second_matrix=bool(args.get('second_path')))
        if self.package == 'episcanpy' and args['function_name'] == 'tl.geneactivity':
            request['gene_anno_path'] = args['gene_anno_path']
            request['activity_inputs'] = {k: args[k] for k in ('species', 'assembly', 'count_slot', 'matrix_semantics') if k in args}
        if not raw_call and not is_r:
            source = file_closure([args['input_path']])[0]
            request.update(source_sha256=source['sha256'], parameter_selection=args.get('parameter_selection'),
                           require_asw=args.get('require_asw',False),parameter_origin=args.get('parameter_origin'))
            context = current()
            if context is not None:
                context.validate_anchors()
                from .cell_result import NATIVE, describe
                lineage = {}
                for proof in context.proofs.values():
                    result = proof.result_metadata
                    for tool in NATIVE:
                        if proof.kind != 'native_' + tool.removeprefix('run_') + '_matrix':
                            continue
                        descriptor = describe(tool, result)
                        lineage[result.get('matrix_path'), result.get('matrix_sha256')] = (proof, descriptor)
                        if result.get('matrix_path') == str(args['input_path']) and result.get('matrix_sha256') == source['sha256']:
                            request['graph_inputs'] = _serialize(result['parameters'].get('graph_inputs', []))
                # Only this source's accepted ancestry can identify a custom
                # UMAP slot. The most recent writer owns an overwritten field.
                visualizations, written, visited = set(), set(), set()
                location = (str(args['input_path']), source['sha256'])
                while location in lineage and location not in visited:
                    visited.add(location)
                    proof, descriptor = lineage[location]
                    result = proof.result_metadata
                    coordinate = descriptor['coordinates'] if descriptor else None
                    if coordinate and coordinate not in written:
                        visualizations.add(coordinate)
                    written.update(slot[5:] for slot in result['result_slots'] if slot.startswith('obsm.'))
                    original = proof.description['manifest']['arguments']['input_path']
                    location = (original, result['source_identity']['sha256'])
                request['visualization_representations'] = sorted(visualizations)
            if args.get('cell_subset') is not None or args.get('input_orientation') == 'features_by_cells':
                request['graph_inputs'] = []
        if raw_call:
            from .snapatac_raw import active_qc
            request['raw_inputs'] = {key: args[key] for key in ('input_path', 'matrix_semantics', 'species', 'assembly',
                *(record[0] for record in raw.PORTS.values())) if key in args}
            if qc:=active_qc(args):request['raw_inputs']['qc_thresholds']=qc
        request_sha = write(stage / 'request.json', request)
        if is_r:
            mmwrite(stage / 'input.mtx', sparse.csc_matrix(matrix.X.T))
            matrix.obs.to_csv(stage / 'cells.tsv', sep='\t', columns=[], index_label='cell_id', quoting=csv.QUOTE_NONE)
            matrix.var.to_csv(stage / 'features.tsv', sep='\t', columns=[], index_label='feature_id', quoting=csv.QUOTE_NONE)
        code = run([str(executable), '--vanilla' if is_r else '-I', str(script), str(stage / 'request.json')],
            stage / 'native.log', environment=r_environment())
        if code:
            shutil.copyfile(stage / 'native.log', stage.parent / 'native-failure.log')
            error = read(stage / 'native-error.json') if (stage / 'native-error.json').is_file() else {}
            message = error.get('message', '')[:2000]
            if not message or peak_h5ad._PATH.search(message) or any(ord(c) < 32 for c in message):
                message = 'Native function failed; consult the retained private diagnostic'
            if (error.get('type') == 'RequiredClusterCount' and self.package == 'episcanpy'
                    and args['function_name'] in ('tl.kmeans', 'tl.hc')
                    and args.get('parameter_selection') == 'asw' and 'num_clusters' not in request['kwargs']
                    and error.get('request_sha256') == request_sha):
                # Diagnostics survive staging cleanup, but confer no result authority.
                write(stage.parent / 'native-error.json', dict(error, source_sha256=source['sha256']))
                raise NativeInvocationError(message, code='NATIVE_CLUSTER_COUNT_REQUIRED')
            if (error.get('type')=='ExplicitASWUnavailable' and args.get('require_asw')
                    and args.get('parameter_selection')=='asw' and error.get('request_sha256')==request_sha):
                write(stage.parent / 'native-error.json', dict(error, source_sha256=source['sha256']))
                raise NativeInvocationError(message,code='NATIVE_ASW_UNAVAILABLE')
            raise NativeInvocationError(message)
        # Preserve warnings/progress privately; logs confer no publication authority.
        shutil.copyfile(stage / 'native.log', stage.parent / 'native.log')
        receipt = read(stage / 'native-receipt.json')
        if (receipt['completion'] != 'succeeded' or receipt['request_sha256'] != request_sha
                or receipt['function_name'] != args['function_name'] or receipt['package'] != self.package
                or receipt['versions'][self.package] != self.version):
            raise AuthorityError('Native completion differs from the exact requested invocation')
        if file_closure(implementation_paths) != implementation:
            raise AuthorityError('Native implementation changed during invocation')
        native_implementation = receipt.get('implementation', [])
        if native_implementation and file_closure(r['path'] for r in native_implementation) != sorted(native_implementation, key=lambda r:r['path']):
            raise AuthorityError('Native package implementation changed during invocation')
        receipt['implementation'] = sorted(implementation + native_implementation, key=lambda r:r['path'])
        if is_r:
            if args.get('second_path'):
                values = mmread(stage/'output.mtx').toarray()
                import pandas as pd
                other_cells = pd.read_csv(stage/'second-cells.tsv', sep='\t', dtype=str, keep_default_na=False).cell_id.tolist()
                if values.shape != (matrix.n_obs, len(other_cells)):
                    raise AuthorityError('Native pairwise axes differ from the exact two inputs')
                matrix.uns['Signac_pairwise'] = dict(values=values, row_cells=matrix.obs_names.to_numpy(dtype=str),
                    column_cells=np.array(other_cells, dtype=str), function_name=args['function_name'])
            elif args['function_name'] == 'corSparse':
                matrix.obsp['Signac_corSparse'] = mmread(stage / 'output.mtx').tocsr()
            elif args['function_name'] == 'FindTopFeatures':
                import pandas as pd
                table = pd.read_csv(stage / 'feature-statistics.tsv', sep='\t', index_col=0,
                    dtype={'feature_id': str}, keep_default_na=False, quoting=csv.QUOTE_NONE)
                if not table.index.is_unique or set(table.index) != set(matrix.var_names):
                    raise AuthorityError('Native feature statistics changed feature identities')
                table['native_rank'] = np.arange(len(table))
                table = table.loc[matrix.var_names]
                for key in table:
                    matrix.var['Signac_' + key] = table[key].to_numpy()
            elif args['function_name'] == 'RunSVD':
                import pandas as pd
                for filename, names, container, key in (
                    ('embeddings.tsv', matrix.obs_names, matrix.obsm, 'X_lsi'),
                    ('loadings.tsv', matrix.var_names, matrix.varm, 'LSI')):
                    table = pd.read_csv(stage / filename, sep='\t', index_col=0,
                        dtype={'id': str}, keep_default_na=False, quoting=csv.QUOTE_NONE)
                    if table.index.tolist() != names.tolist():
                        raise AuthorityError('Native reduction changed ordered axis identities')
                    container[key] = table.to_numpy(dtype=float)
                matrix.uns['Signac_svd'] = read(stage / 'svd.json')
            else:
                matrix.X = mmread(stage / 'output.mtx').T.tocsr()
            matrix.write_h5ad(stage / 'result.h5ad')
        return receipt

    def execute(self, values, execution_identity=None):
        from agent.tools._native_process import call
        return call('matrix', dict(package=self.package, values=values, execution_identity=execution_identity),
                    lambda: self._execute(values, execution_identity))

    def _execute(self, values, execution_identity=None):
        args = arguments(values)
        if self.package == 'snapatac2':
            from . import snapatac_raw as raw
            if args['function_name'] in raw.FUNCTIONS:
                return raw.execute(self, args, execution_identity)
        kwargs = self.native_arguments(args)
        root = Path(args['output_dir']); root.mkdir(parents=True, exist_ok=True)
        if any(root.iterdir()):
            raise ValueError('Output directory is not empty; retry requires a new authorized execution')
        source = file_closure([args['input_path']])[0]
        if args.get('input_sha256') and source['sha256'] != args['input_sha256']:
            raise AuthorityError('Selected input bytes changed')
        gene_activity = self.package == 'episcanpy' and args['function_name'] == 'tl.geneactivity'
        annotation = file_closure([args['gene_anno_path']])[0] if gene_activity else None
        if annotation and args.get('gene_anno_sha256') and annotation['sha256'] != args['gene_anno_sha256']:
            raise AuthorityError('Selected gene annotation bytes changed')
        second_source = file_closure([args['second_path']])[0] if args.get('second_path') else None
        if second_source and args.get('second_sha256') and second_source['sha256'] != args['second_sha256']:
            raise AuthorityError('Selected second matrix bytes changed')
        with tempfile.TemporaryDirectory(prefix='.native-matrix-', dir=root) as temp:
            stage = Path(temp)
            matrix, effective = prepare_input(args, stage, source=source)
            cells, features = matrix.obs_names.tolist(), matrix.var_names.tolist()
            if self.package == 'Signac':
                if any(any(c in s for c in '\t\r\n') for s in cells + features):
                    raise ValueError('R matrix exchange labels cannot contain TSV delimiters')
                values = matrix.X.data if sparse.issparse(matrix.X) else np.asarray(matrix.X)
                peak_h5ad.native_double_values(values)
            if second_source:
                other_stage = stage/'second'; other_stage.mkdir()
                other, other_effective = prepare_input(dict(input_path=args['second_path'],
                    **({'count_slot': args['second_count_slot']} if args.get('second_count_slot') is not None else {})), other_stage)
                if features != other.var_names.tolist():
                    raise ValueError('Two-matrix comparison requires identical ordered feature identities; no implicit alignment')
                peak_h5ad.native_double_values(other.X.data if sparse.issparse(other.X) else np.asarray(other.X))
                mmwrite(stage/'second.mtx', sparse.csc_matrix(other.X.T))
                other.obs.to_csv(stage/'second-cells.tsv', sep='\t', columns=[], index_label='cell_id', quoting=csv.QUOTE_NONE)
                effective['second_path'] = args['second_path']
            receipt = self.native(effective, stage, matrix)
            peak_h5ad.inspect(stage / 'result.h5ad', require_matrix=False)
            output = ad.read_h5ad(stage / 'result.h5ad')
            positions = receipt.get('source_axis_positions')
            ci, fi = list(range(len(cells))), list(range(len(features)))
            if positions is not None:
                if self.package != 'scanpy' or args['function_name'] != 'pp.highly_variable_genes' or not kwargs.get('subset'):
                    raise AuthorityError('Unexpected native axis change')
                ci, fi = positions['cells'], positions['features']
                if ci != list(range(len(cells))) or any(type(i) is not int or not 0 <= i < len(features) for i in fi) or fi != sorted(set(fi)):
                    raise AuthorityError('Invalid native subset source positions')
            if output.obs_names.tolist() != [cells[i] for i in ci] or (not gene_activity and output.var_names.tolist() != [features[i] for i in fi]):
                raise AuthorityError('Native result changed the exact ordered input axes')
            if not gene_activity and args['function_name'] not in ('pp.binarize', 'pp.log1p', 'pp.normalize_total', 'RunTFIDF', 'BinarizeCounts', 'pp.scale') and matrix.X is not None:
                original = ad.read_h5ad(stage / 'input.h5ad')[ci, fi]
                before, after = (sparse.csr_matrix(x, copy=True) for x in (original.X, output.X))
                for retained in (before, after):
                    retained.sum_duplicates()
                    retained.eliminate_zeros()
                    retained.sort_indices()
                if (before.shape != after.shape
                        or not np.array_equal(before.indptr, after.indptr)
                        or not np.array_equal(before.indices, after.indices)
                        or not np.array_equal(before.data, after.data, equal_nan=True)):
                    raise AuthorityError('Native annotation/embedding function changed source X')
            for slot in receipt['result_slots']:
                group, _, key = slot.partition('.')
                value = output.X if group == 'X' else getattr(output, group)[key]
                if group in ('X', 'layers', 'obsm', 'varm', 'obsp'):
                    values = value.data if sparse.issparse(value) else np.asarray(value)
                    if values.dtype.kind not in 'biufc':
                        raise AuthorityError('Native numeric result changed representation')
            measurement = receipt.get('measurement') or effective['matrix_semantics']
            output.uns['matrix_semantics'] = measurement
            for slot, semantics in receipt.get('layer_semantics', {}).items():
                output.uns.setdefault('layer_semantics', {})[slot] = semantics
            summary = dict(status='success', contract_version=self.CONTRACT,
                function_name=self.package + ('::' if self.package == 'Signac' else '.') + args['function_name'],
                n_cells=output.n_obs, n_features=output.n_vars, measurement=measurement,
                parameters=dict(kwargs=kwargs, omitted_optional_arguments='native defaults', count_slot=effective['count_slot']),
                source_identity=dict(sha256=source['sha256'], size_bytes=source['size_bytes'], measurement=effective['matrix_semantics'],
                    declarations=effective['_source_declarations'],
                    supplied_declarations={k: args[k] for k in ('species', 'assembly', 'coordinate_system', 'matrix_semantics') if k in args}),
                software_versions=receipt['versions'], native_output_type=receipt['native_output_type'],
                result_slots=receipt['result_slots'], limitations=LIMITATIONS)
            if 'effective_parameters' in receipt:
                summary['parameters']['effective_native_parameters'] = receipt['effective_parameters']
                summary['parameters']['result_shapes'] = receipt.get('result_shapes', {})
            if 'input_adaptation' in receipt:
                summary['parameters']['input_adaptation'] = receipt['input_adaptation']
            for key in ('graph_inputs', 'parameter_selection','native_method_attribution'):
                if key in receipt:
                    summary['parameters'][key] = receipt[key]
            if receipt.get('native_method_attribution',{}).get('conflict'):
                summary['limitations'] += ' The installed epiScanpy wrapper replaced the requested ranking method; see native_method_attribution.'
            if '_cell_subset' in effective:
                summary['parameters']['cell_subset'] = effective['_cell_subset']
            if '_grouping_projection' in effective:
                summary['parameters']['grouping_projection'] = effective['_grouping_projection']
            if args.get('input_orientation') == 'features_by_cells':
                summary['parameters']['input_orientation'] = 'explicit native AnnData transpose: features_by_cells to cells_by_features'
            selection = receipt.get('parameter_selection')
            if selection and selection['origin'] == 'automatic':
                summary['parameters']['kwargs'] = dict(kwargs, **{selection['parameter']: selection['selected_value']})
                summary['limitations'] = LIMITATIONS + ' Bounded ASW selection is an internal separation criterion, not biological validation.'
            if second_source:
                summary['source_identity']['second'] = dict(sha256=second_source['sha256'], size_bytes=second_source['size_bytes'],
                    count_slot=other_effective['count_slot'], measurement=other_effective['matrix_semantics'], n_cells=other.n_obs)
            output.uns['agent_native_invocation'] = dict(function=summary['function_name'], source_sha256=source['sha256'],
                source_measurement=effective['matrix_semantics'], parameters_json=json.dumps(summary['parameters']),
                annotation_policy='Retained source annotations and reductions are not recomputed; native-written slots replace existing values.')
            if positions is not None:
                output.uns['agent_native_invocation']['source_axis_positions'] = positions
            output.write_h5ad(root / 'result.h5ad')
            if gene_activity:
                from .native_gene_activity import publish_description
                summary['source_identity']['references'] = [dict(role='gene_anno', sha256=annotation['sha256'])]
                publish_description(root / 'result.h5ad', args, receipt, summary, source)
            else:
                from .native_gene_activity import reuse_description
                reuse_description(root / 'result.h5ad', args, summary, source)
            write(root / 'native-receipt.json', receipt)
            write(root / 'metadata.json', dict(summary=summary, source=source, receipt=receipt,
                input_orientation='cells by features; Signac receives features by cells'))
            expected = {r['path']: r for r in [source, *([annotation] if annotation else []), *([second_source] if second_source else []), *receipt['implementation']]}
            if '_grouping_projection' in effective:
                pin = effective['_grouping_projection']['source']
                expected[pin['path']] = pin
            sources = file_closure(expected)
            if sources != sorted(expected.values(), key=lambda r: r['path']):
                raise AuthorityError('Source or runtime changed during invocation')
        value = dict(contract=self.CONTRACT, profile=self.PROFILE, arguments=args, execution_identity=execution_identity,
            result=summary, files=file_closure(root / name for name in DELIVERY_FILES), sources=sources, native_completion='succeeded')
        path = root / 'manifest.json'; sha = write(path, value)
        if current() is not None:
            current().register_execution(self.KIND, root, execution_identity)
        return self.verify_publication(path, expected_sha256=sha)

    def invoke(self, input_path, output_dir, function_name, kwargs=None, input_sha256=None, count_slot=None, matrix_semantics=None, use_rep=None, parameter_selection=None, root_cell=None,
               gene_activity=None, cell_subset=None, input_orientation=None, grouping_path=None, group_column=None, require_asw=None,
               parameter_origin=None):
        return self.execute({k: v for k, v in locals().items() if k != "self" and v is not None})

    def invoke_signac(self, input_path, output_dir, function_name, kwargs=None, input_sha256=None, count_slot=None, matrix_semantics=None, second_path=None, second_sha256=None, second_count_slot=None):
        return self.execute({k: v for k, v in locals().items() if k != 'self' and v is not None})

    def invoke_episcanpy(self, input_path, output_dir, function_name, kwargs=None, input_sha256=None, count_slot=None,
                         matrix_semantics=None, use_rep=None, parameter_selection=None, root_cell=None, gene_anno_path=None,
                         gene_anno_sha256=None, species=None, assembly=None, gene_activity=None, cell_subset=None, input_orientation=None, require_asw=None,
                         parameter_origin=None):
        return self.execute({k: v for k, v in locals().items() if k != 'self' and v is not None})

    def invoke_snapatac2(self, input_path, output_dir, function_name, kwargs=None, input_sha256=None, count_slot=None, use_rep=None,
                        matrix_semantics=None, chrom_sizes_path=None, chrom_sizes_sha256=None, gene_anno_path=None,
                        gene_anno_sha256=None, peaks_path=None, peaks_sha256=None, whitelist_path=None,
                        whitelist_sha256=None, species=None, assembly=None, fragment_order=None, counting_strategy=None,
                        qc_thresholds=None, parameter_selection=None, gene_activity=None, cell_subset=None, input_orientation=None, require_asw=None,
                        qc_population=None,qc_selection=None,parameter_origin=None):
        return self.execute({k: v for k, v in locals().items() if k != 'self' and v is not None})


EPISCANPY = MatrixPackage('episcanpy', '0.4.0', 'episcanpy-0.4.0', {
    'tl.geneactivity': ('key_added', 'upstream', 'feature_type', 'annotation', 'layer_name', 'raw', 'copy'),
    'tl.pca': ('n_comps', 'zero_center', 'svd_solver', 'random_state', 'return_info', 'use_highly_variable', 'dtype', 'copy', 'chunked', 'chunk_size'),
    'pp.neighbors': ('n_neighbors', 'n_pcs', 'use_rep', 'knn', 'random_state', 'method', 'metric', 'metric_kwds', 'copy'),
    'tl.dpt': ('n_dcs', 'n_branchings', 'min_group_size', 'allow_kendall_tau_shift', 'neighbors_key', 'copy'),
    'tl.tsne': ('n_pcs', 'use_rep', 'perplexity', 'early_exaggeration', 'learning_rate', 'random_state', 'use_fast_tsne', 'n_jobs', 'copy'),
    'tl.umap': ('min_dist', 'spread', 'n_components', 'maxiter', 'alpha', 'gamma', 'negative_sample_rate', 'init_pos', 'random_state', 'a', 'b', 'copy'),
    'pp.binarize': ('copy',),
    'pp.tfidf': ('norm', 'layer_tfidf_key'),
    'pp.log1p': ('base', 'copy', 'chunked', 'chunk_size'),
    'pp.normalize_total': ('target_sum', 'exclude_highly_expressed', 'max_fraction', 'key_added', 'layers', 'layer_norm', 'inplace'),
    'pp.lsi': ('n_components', 'tf_idf_layer', 'algorithm', 'n_iter', 'random_state'),
    'tl.leiden': ('resolution', 'restrict_to', 'random_state', 'key_added', 'directed', 'use_weights', 'n_iterations', 'neighbors_key', 'obsp', 'copy'),
    'tl.louvain': ('resolution', 'random_state', 'restrict_to', 'key_added', 'flavor', 'directed', 'use_weights', 'neighbors_key', 'obsp', 'copy'),
    'tl.kmeans': ('num_clusters',),
    'tl.hc': ('num_clusters',),
    'tl.rank_features': ('groupby', 'omic', 'use_raw', 'groups', 'reference', 'n_features', 'rankby_abs', 'key_added', 'copy', 'method', 'corr_method'),
}, (
    ('upstream', 'geneactivity upstream window in bp; native default 5000.'),
    ('feature_type', 'geneactivity exact GTF feature type; native default gene.'),
    ('annotation', 'geneactivity exact GTF source category; native default HAVANA, explicit null accepts all sources.'),
    ('layer_name', 'geneactivity native argument; installed 0.4.0 ignores this and returns new X.'),
    ('raw', 'geneactivity selects raw.X; native indexing still uses current var positions. Differing axes are recorded, not repaired or rejected by Agent.'),
    ('n_comps', 'Native n_comps for the selected function; omitted retains its installed default.'),
    ('zero_center', 'pca native centering; omitted retains its installed default.'),
    ('svd_solver', 'pca installed native solver.'),
    ('random_state', 'Native random_state for the selected function; omitted retains its installed default.'),
    ('return_info', 'pca native return_info flag; AnnData results retain native slots.'),
    ('use_highly_variable', 'pca uses the existing var highly_variable flag when requested.'),
    ('dtype', 'pca native result dtype.'),
    ('copy', 'Native copy flag; returned AnnData is retained, source bytes stay immutable.'),
    ('chunked', 'pca/log1p native chunked processing flag.'),
    ('chunk_size', 'pca/log1p native processing chunk size.'),
    ('n_neighbors', 'neighbors native neighborhood size; omitted retains its installed default.'),
    ('n_pcs', 'neighbors/tsne component count; native representation selection applies.'),
    ('use_rep', 'Exact native representation selector; omission retains installed selection/default.'),
    ('knn', 'neighbors hard nearest-neighbor threshold versus full kernel.'),
    ('method', 'Native method for the selected function; omitted retains its installed default.'),
    ('metric', 'neighbors native metric name; no executable callable.'),
    ('metric_kwds', 'neighbors JSON metric options; no executable callbacks.'),
    ('n_dcs', 'dpt diffusion component count; existing uns.iroot identifies the supplied root cell.'),
    ('n_branchings', 'dpt native branching count.'),
    ('min_group_size', 'dpt minimum group fraction or size.'),
    ('allow_kendall_tau_shift', 'dpt native branching behavior.'),
    ('neighbors_key', 'dpt/leiden/louvain existing uns neighborhood key; omitted uses native selection.'),
    ('perplexity', 'tsne native perplexity.'),
    ('early_exaggeration', 'tsne native early exaggeration.'),
    ('learning_rate', 'tsne native learning rate; omitted retains its installed default.'),
    ('use_fast_tsne', 'tsne native implementation choice; no adapter substitution.'),
    ('n_jobs', 'tsne native worker count.'),
    ('min_dist', 'umap native min_dist; omitted uses installed default.'),
    ('spread', 'umap native spread; omitted uses installed default.'),
    ('n_components', 'Native n_components for the selected function; omitted retains its installed default.'),
    ('maxiter', 'umap native maxiter; omitted uses installed default.'),
    ('alpha', 'Native alpha; omitted retains the installed function default.'),
    ('gamma', 'umap native gamma; omitted uses installed default.'),
    ('negative_sample_rate', 'umap native negative_sample_rate; omitted uses installed default.'),
    ('init_pos', 'umap native init_pos; omitted uses installed default.'),
    ('a', 'umap native a; omitted uses installed default.'),
    ('b', 'umap native b; omitted uses installed default.'),
    ('norm', 'tfidf row normalization; installed default l1, not the docstring l2.'),
    ('layer_tfidf_key', 'tfidf destination layer, default tf-idf; X stays unchanged.'),
    ('base', 'log1p logarithm base; omission uses natural log.'),
    ('target_sum', 'normalize_total target library size; omitted uses the native median.'),
    ('exclude_highly_expressed', 'normalize_total excludes high-expression features from normalization-factor calculation.'),
    ('max_fraction', 'normalize_total native high-expression fraction threshold.'),
    ('key_added', 'Native key_added for the selected function; omitted retains its installed default.'),
    ('layers', 'normalize_total exact layer names or all; no external objects.'),
    ('layer_norm', 'normalize_total layer scaling: null, after, or X.'),
    ('inplace', 'normalize_total: false leaves AnnData unchanged; epiScanpy discards the Scanpy return.'),
    ('tf_idf_layer', 'lsi existing TF-IDF layer; default tf-idf. No adapter-generated TF-IDF.'),
    ('algorithm', 'lsi native TruncatedSVD solver, default arpack.'),
    ('n_iter', 'Native n_iter; omitted retains the installed function default.'),
    ('resolution', 'leiden/louvain native clustering resolution.'),
    ('restrict_to', 'Native restrict_to; omitted retains the installed function default.'),
    ('directed', 'Native directed; omitted retains the installed function default.'),
    ('use_weights', 'Native use_weights; omitted retains the installed function default.'),
    ('n_iterations', 'leiden native iterations, -1 until convergence; cap 1000 positive iterations.'),
    ('obsp', 'Native obsp; omitted retains the installed function default.'),
    ('flavor', 'Native flavor; omitted retains the installed function default.'),
    ('num_clusters', 'Native num_clusters; omitted retains the installed function default.'),
    ('groupby', 'Exact obs grouping field; native statistics retain supplied feature semantics.'),
    ('omic', 'Native omic; omitted retains the installed function default.'),
    ('use_raw', 'Native use_raw; omitted retains the installed function default.'),
    ('groups', 'Native groups; omitted retains the installed function default.'),
    ('reference', 'Native reference; omitted retains the installed function default.'),
    ('n_features', 'rank_features number of ranked features per group; no matrix subsetting.'),
    ('rankby_abs', 'Native rankby_abs; omitted retains the installed function default.'),
    ('corr_method', 'Native corr_method; omitted retains the installed function default.'),
), 'Native preprocessing, PCA/TF-IDF/LSI, neighbors, Leiden/Louvain, UMAP/t-SNE, DPT, kmeans/hierarchical clustering and feature ranking. kmeans/hc consume X_pca; graph methods need compatible Scanpy neighbors. DPT also needs diffusion maps and supplied uns.iroot. tfidf adds a layer; lsi consumes it. normalize_total(inplace=False) is a native no-op returning None. rank_features retains native method overrides/errors. No Agent-added preprocessing.')


SNAPATAC2 = MatrixPackage('snapatac2', '2.9.0', 'native-matrix-1', {
    'tl.umap': ('n_comps', 'use_dims', 'use_rep', 'key_added', 'random_state', 'inplace', 'n_neighbors', 'min_dist', 'spread', 'metric', 'n_epochs', 'learning_rate', 'init', 'negative_sample_rate', 'repulsion_strength', 'local_connectivity', 'set_op_mix_ratio', 'a', 'b', 'verbose', 'n_jobs'),
    'pp.knn': ('n_neighbors', 'use_dims', 'use_rep', 'method', 'inplace', 'random_state'),
    'tl.leiden': ('resolution', 'objective_function', 'min_cluster_size', 'n_iterations', 'random_state', 'key_added', 'weighted', 'inplace'),
    'pp.select_features': ('n_features', 'filter_lower_quantile', 'filter_upper_quantile', 'max_iter', 'inplace', 'n_jobs', 'verbose'),
    'tl.spectral': ('n_comps', 'features', 'random_state', 'sample_size', 'sample_method', 'distance_metric', 'weighted_by_sd', 'chunk_size', 'inplace', 'num_threads'),
    'tl.dbscan': ('eps', 'min_samples', 'leaf_size', 'n_jobs', 'use_rep', 'key_added'),
    'pp.mnc_correct': ('batch', 'n_neighbors', 'n_clusters', 'n_iter', 'use_rep', 'use_dims', 'groupby', 'key_added', 'inplace', 'n_jobs'),
    'tl.marker_regions': ('groupby', 'pvalue'),
    'tl.diff_test': ('cell_group1', 'cell_group2', 'features', 'covariates', 'direction', 'min_log_fc', 'min_pct'),
}, (
    ('n_comps', 'Native n_comps for the selected function; omitted retains its installed default.'),
    ('use_dims', 'umap/knn/mnc_correct leading component count or exact integer indices.'),
    ('use_rep', 'Exact native representation selector; omission retains installed selection/default.'),
    ('key_added', 'Native key_added for the selected function; omitted retains its installed default.'),
    ('random_state', 'Native random_state for the selected function; omitted retains its installed default.'),
    ('inplace', 'Native mutation flag; false preserves independent returns in uns.agent_native_return.'),
    ('n_neighbors', 'umap/knn/mnc_correct native neighbor count; omitted retains installed default.'),
    ('min_dist', 'umap.UMAP native min_dist; omitted uses installed default.'),
    ('spread', 'umap.UMAP native spread; omitted uses installed default.'),
    ('metric', 'umap.UMAP native metric; omitted uses installed default.'),
    ('n_epochs', 'umap.UMAP native n_epochs; omitted uses installed default.'),
    ('learning_rate', 'umap.UMAP native learning_rate; omitted uses installed default.'),
    ('init', 'umap.UMAP native init; omitted uses installed default.'),
    ('negative_sample_rate', 'umap.UMAP native negative_sample_rate; omitted uses installed default.'),
    ('repulsion_strength', 'umap.UMAP native repulsion_strength; omitted uses installed default.'),
    ('local_connectivity', 'umap.UMAP native local_connectivity; omitted uses installed default.'),
    ('set_op_mix_ratio', 'umap.UMAP native set_op_mix_ratio; omitted uses installed default.'),
    ('a', 'umap.UMAP native a; omitted uses installed default.'),
    ('b', 'umap.UMAP native b; omitted uses installed default.'),
    ('verbose', 'Native progress verbosity for the selected function.'),
    ('n_jobs', 'Native parallel jobs for the selected function.'),
    ('method', 'Native method for the selected function; omitted retains its installed default.'),
    ('resolution', 'leiden native clustering resolution.'),
    ('objective_function', 'leiden modularity or CPM objective.'),
    ('min_cluster_size', 'leiden minimum cluster size; smaller groups become -1.'),
    ('n_iterations', 'leiden native iterations, -1 until convergence; cap 1000 positive iterations.'),
    ('weighted', 'leiden uses exp(-distance) weights when true.'),
    ('n_features', 'select_features maximum selected features; no matrix subsetting.'),
    ('filter_lower_quantile', 'select_features lower count quantile exclusion.'),
    ('filter_upper_quantile', 'select_features upper count quantile exclusion.'),
    ('max_iter', 'select_features iterations; default 1.'),
    ('features', 'Native existing mask field for spectral; explicit feature IDs/indices for diff_test.'),
    ('sample_size', 'spectral native Nystrom sample size or fraction.'),
    ('sample_method', 'spectral native sampling method.'),
    ('distance_metric', 'spectral native similarity choice.'),
    ('weighted_by_sd', 'spectral native eigenvector scaling.'),
    ('chunk_size', 'spectral native processing chunk size.'),
    ('num_threads', 'spectral native thread count; installed default 32.'),
    ('n_clusters', 'Native n_clusters; omitted retains the installed function default.'),
    ('eps', 'Native eps; omitted retains the installed function default.'),
    ('min_samples', 'Native min_samples; omitted retains the installed function default.'),
    ('leaf_size', 'Native leaf_size; omitted retains the installed function default.'),
    ('batch', 'Exact obs batch field for mnc_correct; no invented batch labels.'),
    ('n_iter', 'Native n_iter; omitted retains the installed function default.'),
    ('groupby', 'Exact obs grouping field; native statistics retain supplied feature semantics.'),
    ('pvalue', 'Native pvalue; omitted retains the installed function default.'),
    ('cell_group1', 'Explicit ordered source cell names or indices for diff_test.'),
    ('cell_group2', 'Explicit ordered source cell names or indices for diff_test.'),
    ('covariates', 'Native covariates; omitted retains the installed function default.'),
    ('direction', 'Native direction; omitted retains the installed function default.'),
    ('min_log_fc', 'Native min_log_fc; omitted retains the installed function default.'),
    ('min_pct', 'Native min_pct; omitted retains the installed function default.'),
), 'Native BAM/fragments ingestion, QC, filtering and region matrices, alongside AnnData feature selection, spectral embeddings, graphs and clustering. Each call uses its selected native function and actual count unit. Non-inplace returns remain independent in uns.agent_native_return; no automatic embedding or biological validation.')

SIGNAC = MatrixPackage('Signac', '1.16.0', 'signac-1', {
    'RunSVD': ('assay', 'n', 'scale.embeddings', 'reduction.key', 'scale.max', 'verbose', 'irlba.work', 'tol'),
    'RunTFIDF': ('method', 'scale.factor', 'verbose'),
    'BinarizeCounts': ('verbose',),
    'FindTopFeatures': ('min.cutoff', 'verbose'),
    'Jaccard': (),
    'corSparse': ('cov',),
}, (
    ('assay', 'RunSVD assay name recorded by native DimReduc; no assay construction.'),
    ('n', 'RunSVD requested singular components; native clamps to cells minus one; cap 512.'),
    ('scale.embeddings', 'RunSVD native embedding scaling, default true.'),
    ('reduction.key', 'RunSVD native component key, default LSI_.'),
    ('scale.max', 'RunSVD optional symmetric clipping of scaled embeddings.'),
    ('verbose', 'RunTFIDF progress; BinarizeCounts/FindTopFeatures matrix methods accept but ignore it.'),
    ('irlba.work', 'RunSVD native IRLBA work space, default n*3.'),
    ('tol', 'RunSVD native IRLBA convergence tolerance.'),
    ('method', 'Native method for the selected function; omitted retains its installed default.'),
    ('scale.factor', 'RunTFIDF native scaling, default 10000; method-specific effect.'),
    ('min.cutoff', 'FindTopFeatures matrix method accepts but ignores this assay-only selection argument; returns counts and percentile ranking, no filtering.'),
    ('cov', 'corSparse: return covariance instead of correlation; optional second matrix supplies native Y.'),
), 'Native matrix transformations/reductions preserve source axes. corSparse uses features by cells, with optional exact second matrix Y. Jaccard requires two cells-by-features matrices; no automatic binarization. Both require identical ordered features; rectangular cross-cell values and both cell axes are stored in uns.Signac_pairwise. Single-source corSparse remains obsp. No fake Seurat object, hidden TF-IDF or alignment. FindTopFeatures matrix method ignores min.cutoff.')

SCANPY = MatrixPackage('scanpy', '1.9.8', 'episcanpy-0.4.0', {
    'tl.diffmap': ('n_comps', 'neighbors_key', 'random_state', 'copy'),
    'pp.scale': ('zero_center', 'max_value', 'copy', 'layer', 'obsm'),
    'pp.calculate_qc_metrics': ('expr_type', 'var_type', 'qc_vars', 'percent_top', 'layer', 'use_raw', 'inplace', 'log1p', 'parallel'),
    'pp.normalize_total': ('target_sum', 'exclude_highly_expressed', 'max_fraction', 'key_added', 'layer', 'inplace', 'copy'),
    'pp.log1p': ('base', 'copy', 'chunked', 'chunk_size', 'layer', 'obsm'),
    'pp.highly_variable_genes': ('layer', 'n_top_genes', 'min_disp', 'max_disp', 'min_mean', 'max_mean', 'span', 'n_bins', 'flavor', 'subset', 'inplace', 'batch_key', 'check_values'),
    'pp.pca': ('n_comps', 'zero_center', 'svd_solver', 'random_state', 'return_info', 'use_highly_variable', 'dtype', 'copy', 'chunked', 'chunk_size'),
    'pp.neighbors': ('n_neighbors', 'n_pcs', 'use_rep', 'knn', 'random_state', 'method', 'metric', 'metric_kwds', 'key_added', 'copy'),
    'tl.leiden': ('resolution', 'restrict_to', 'random_state', 'key_added', 'directed', 'use_weights', 'n_iterations', 'neighbors_key', 'obsp', 'copy'),
    'tl.umap': ('min_dist', 'spread', 'n_components', 'maxiter', 'alpha', 'gamma', 'negative_sample_rate', 'init_pos', 'random_state', 'a', 'b', 'copy', 'method', 'neighbors_key'),
    'tl.tsne': ('n_pcs', 'use_rep', 'perplexity', 'early_exaggeration', 'learning_rate', 'random_state', 'use_fast_tsne', 'n_jobs', 'copy', 'metric'),
    'tl.paga': ('groups', 'use_rna_velocity', 'model', 'neighbors_key', 'copy'),
    'tl.rank_genes_groups': ('groupby', 'use_raw', 'groups', 'reference', 'n_genes', 'rankby_abs', 'pts', 'key_added', 'copy', 'method', 'corr_method', 'tie_correct', 'layer'),
    'tl.score_genes': ('gene_list', 'ctrl_size', 'gene_pool', 'n_bins', 'score_name', 'random_state', 'copy', 'use_raw'),
    'tl.dpt': ('n_dcs', 'n_branchings', 'min_group_size', 'allow_kendall_tau_shift', 'neighbors_key', 'copy'),
}, (
    ('n_comps', 'Native n_comps for the selected function; omitted retains its installed default.'),
    ('neighbors_key', 'Existing uns neighborhood key for the selected native function.'),
    ('random_state', 'Native random_state for the selected function; omitted retains its installed default.'),
    ('copy', 'Native copy flag; returned AnnData is retained, source bytes stay immutable.'),
    ('zero_center', 'pca/scale native centering; centered scale materializes sparse values.'),
    ('max_value', 'scale native clipping threshold; omission does not clip.'),
    ('layer', 'Native layer for the selected function; omitted retains its installed default.'),
    ('obsm', 'Native obsm for the selected function; omitted retains its installed default.'),
    ('expr_type', 'Native expr_type; omitted retains the installed function default.'),
    ('var_type', 'Native QC feature noun only, not biological qualification.'),
    ('qc_vars', 'Existing boolean var annotation keys for native QC summaries.'),
    ('percent_top', 'Native QC rank positions; omission retains installed defaults.'),
    ('use_raw', 'Native use_raw; omitted retains the installed function default.'),
    ('inplace', 'Native mutation flag; false preserves independent returns in uns.agent_native_return.'),
    ('log1p', 'Native log1p; omitted retains the installed function default.'),
    ('parallel', 'Native parallel; omitted retains the installed function default.'),
    ('target_sum', 'normalize_total target library size; omitted uses the native median.'),
    ('exclude_highly_expressed', 'normalize_total excludes high-expression features from normalization-factor calculation.'),
    ('max_fraction', 'normalize_total native high-expression fraction threshold.'),
    ('key_added', 'Native key_added for the selected function; omitted retains its installed default.'),
    ('layers', 'normalize_total exact layer names or all; no external objects.'),
    ('layer_norm', 'normalize_total layer scaling: null, after, or X.'),
    ('base', 'log1p logarithm base; omission uses natural log.'),
    ('chunked', 'pca/log1p native chunked processing flag.'),
    ('chunk_size', 'pca/log1p native processing chunk size.'),
    ('n_top_genes', 'Native n_top_genes; omitted retains the installed function default.'),
    ('min_disp', 'Native min_disp; omitted retains the installed function default.'),
    ('max_disp', 'Native max_disp; omitted retains the installed function default.'),
    ('min_mean', 'Native min_mean; omitted retains the installed function default.'),
    ('max_mean', 'Native max_mean; omitted retains the installed function default.'),
    ('span', 'Native span; omitted retains the installed function default.'),
    ('n_bins', 'Native n_bins; omitted retains the installed function default.'),
    ('flavor', 'Native flavor; omitted retains the installed function default.'),
    ('subset', 'Native highly_variable_genes subsetting; resulting feature positions and axes are retained.'),
    ('batch_key', 'Native batch_key; omitted retains the installed function default.'),
    ('check_values', 'Native check_values; omitted retains the installed function default.'),
    ('svd_solver', 'pca installed native solver.'),
    ('return_info', 'pca native return_info flag; AnnData results retain native slots.'),
    ('use_highly_variable', 'pca uses the existing var highly_variable flag when requested.'),
    ('dtype', 'pca native result dtype.'),
    ('n_neighbors', 'neighbors native neighborhood size; omitted retains its installed default.'),
    ('n_pcs', 'neighbors/tsne component count; native representation selection applies.'),
    ('use_rep', 'Exact native representation selector; omission retains installed selection/default.'),
    ('knn', 'neighbors hard nearest-neighbor threshold versus full kernel.'),
    ('method', 'Native method for the selected function; omitted retains its installed default.'),
    ('metric', 'neighbors/tsne native metric name; no executable callable.'),
    ('metric_kwds', 'neighbors JSON metric options; no executable callbacks.'),
    ('resolution', 'leiden native clustering resolution.'),
    ('restrict_to', 'Native restrict_to; omitted retains the installed function default.'),
    ('directed', 'Native directed; omitted retains the installed function default.'),
    ('use_weights', 'Native use_weights; omitted retains the installed function default.'),
    ('n_iterations', 'leiden native iterations, -1 until convergence; cap 1000 positive iterations.'),
    ('obsp', 'Native obsp; omitted retains the installed function default.'),
    ('min_dist', 'umap native min_dist; omitted uses installed default.'),
    ('spread', 'umap native spread; omitted uses installed default.'),
    ('n_components', 'Native n_components for the selected function; omitted retains its installed default.'),
    ('maxiter', 'umap native maxiter; omitted uses installed default.'),
    ('alpha', 'Native alpha; omitted retains the installed function default.'),
    ('gamma', 'umap native gamma; omitted uses installed default.'),
    ('negative_sample_rate', 'umap native negative_sample_rate; omitted uses installed default.'),
    ('init_pos', 'umap native init_pos; omitted uses installed default.'),
    ('a', 'umap native a; omitted uses installed default.'),
    ('b', 'umap native b; omitted uses installed default.'),
    ('perplexity', 'tsne native perplexity.'),
    ('early_exaggeration', 'tsne native early exaggeration.'),
    ('learning_rate', 'tsne native learning rate; omitted retains its installed default.'),
    ('use_fast_tsne', 'tsne native implementation choice; no adapter substitution.'),
    ('n_jobs', 'tsne native worker count.'),
    ('groups', 'Native groups; omitted retains the installed function default.'),
    ('use_rna_velocity', 'Native use_rna_velocity; omitted retains the installed function default.'),
    ('model', 'Native model; omitted retains the installed function default.'),
    ('groupby', 'Exact obs grouping field; native statistics retain supplied feature semantics.'),
    ('reference', 'Native reference; omitted retains the installed function default.'),
    ('n_genes', 'Native n_genes; omitted retains the installed function default.'),
    ('rankby_abs', 'Native rankby_abs; omitted retains the installed function default.'),
    ('pts', 'Native pts; omitted retains the installed function default.'),
    ('corr_method', 'Native corr_method; omitted retains the installed function default.'),
    ('tie_correct', 'Native tie_correct; omitted retains the installed function default.'),
    ('gene_list', 'Exact supplied feature IDs for score_genes; the native name does not establish gene annotation.'),
    ('ctrl_size', 'Native ctrl_size; omitted retains the installed function default.'),
    ('gene_pool', 'Exact background feature IDs for score_genes.'),
    ('score_name', 'Native score_name; omitted retains the installed function default.'),
    ('n_dcs', 'dpt diffusion component count; existing uns.iroot identifies the supplied root cell.'),
    ('n_branchings', 'dpt native branching count.'),
    ('min_group_size', 'dpt minimum group fraction or size.'),
    ('allow_kendall_tau_shift', 'dpt native branching behavior.'),
), 'Native AnnData QC, normalization/log/scale, variable features, PCA, neighbors, Leiden/UMAP, diffusion maps/t-SNE, PAGA, feature ranking/scoring and DPT. Non-inplace returns are separate in uns.agent_native_return; subset records exact source positions. Names such as genes retain supplied feature meaning. Native defaults and internal representation selection apply; no Agent-added preprocessing.')

PACKAGES = (EPISCANPY, SNAPATAC2, SIGNAC, SCANPY)
