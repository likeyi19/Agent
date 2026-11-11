"""Reviewed chromVAR functions on ordinary counts or explicitly identified scores."""
import math
from pathlib import Path
import shutil
import tempfile
import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem, sparse_dataset
from scipy.io import mmwrite
from agent.schemas.orchestration import _serialize
from agent.schemas.verification_authority import AuthorityError, digest
from agent.tools.data.authority_context import current
from agent.tools._file_identity import file_closure
from agent.tools._native_process import run, r_environment
from .sequence_prediction import arguments as _arguments, read, write
from .sequence_reference import metadata_declarations, check_declarations
from . import peak_h5ad
from .native_publication import NativePublication

TOOL = 'run_chromvar'
KIND = 'native_chromvar_statistics'
CONTRACT = 'chromvar-native-statistics.v1'
POLICY = 'chromvar-native-statistics-v1'
HISTORICAL_PROFILE = digest(dict(contract=CONTRACT, functions=['computeExpectations','getFragmentsPerPeak','getFragmentsPerSample','computeVariability'],
    adaptation='ordered counts or explicitly recorded chromVAR deviations and z scores; no filtering'))
PREVIOUS_PROFILE = digest(dict(contract=CONTRACT, adaptation='ordered numeric matrix; native numerical validation; recorded deviations and z scores.v2'))
PROFILE = digest(dict(contract=CONTRACT, adaptation='ordered numeric counts, scores or actual supplied bias; native background indices.v3'))
VERIFIER = dict(id='agent.chromvar-function-completion', compatibility_version='1')
FUNCTIONS = {
    'getBackgroundPeaks': ('niterations', 'w', 'bs'),
    'getTotalFragments': (),
    'getSampleCorrelation': ('threshold',),
    'getSampleDistance': ('threshold', 'initial_dims'),
    'differentialDeviations': ('alternative', 'parametric'),
    'differentialVariability': ('parametric',),
    'computeExpectations': ('norm',),
    'getFragmentsPerPeak': (),
    'getFragmentsPerSample': (),
    'computeVariability': ('bootstrap_error','bootstrap_samples','bootstrap_quantiles','na.rm'),
}
DEVIATION_FUNCTIONS = ('computeVariability', 'getSampleCorrelation', 'getSampleDistance', 'differentialDeviations', 'differentialVariability')
NATIVE_ARGUMENTS = (
    ('niterations', 'getBackgroundPeaks native number of background samples per peak.'),
    ('w', 'getBackgroundPeaks native sampling distribution width.'),
    ('bs', 'getBackgroundPeaks native bin count per dimension.'),
    ('threshold', 'Native variability threshold for sample correlation/distance.'),
    ('initial_dims', 'Native distance PCA dimensions; no supplied executable distance function.'),
    ('alternative', 'differentialDeviations native alternative hypothesis.'),
    ('parametric', 'Native deviation/variability test choice; not replicate-level accessibility inference.'),
    ('norm','computeExpectations: native per-cell normalization; omitted uses native FALSE.'),
    ('bootstrap_error','computeVariability: native bootstrap confidence bounds; omitted uses TRUE.'),
    ('bootstrap_samples','computeVariability: native bootstrap resamples; omitted uses 1000.'),
    ('bootstrap_quantiles','computeVariability: native two quantiles as a JSON number array.'),
    ('na.rm','computeVariability: native missing-value handling; omitted uses TRUE.'),
)
LIMITATIONS = ('Native functions are independent calls, with no unrequested filtering, motif scanning or workflow completion. Background indices are native 1-based references to features.tsv, with replacement, not new genomic peaks or biological qualification. '
    'Count totals retain the supplied measurement and are not independent molecules. Expectations retain the installed native norm/group behavior. '
    'computeVariability uses chromVAR z scores across supplied cells; its native chi-square/BH statistics are not replicate-level condition tests, TF binding or biological validation. '
    'Undefined native numbers remain explicit. The requested invocation assigns supplied X and z_scores to native deviations and z assays; their upstream scientific origin is not independently verified. '
    'Omitted kwargs use native defaults. Bootstrap results without an explicit seed do not promise reproducibility.')
FACT_FIELDS = ('function_name','n_cells','n_features','result_rows','result_columns','result_axis','native_output_type',
    'measurement','parameters','source_identity','software_versions','undefined_values','limitations')
RESULT_FIELDS = dict(status=(str,),contract_version=(str,),manifest_path=(str,),manifest_sha256=(str,),
    table_path=(str,),table_sha256=(str,),function_name=(str,),n_cells=(int,),n_features=(int,),result_rows=(int,),
    result_columns=(list,),result_axis=(str,),native_output_type=(str,),measurement=(str,),parameters=(dict,),
    source_identity=(dict,),software_versions=(dict,),undefined_values=(int,),limitations=(str,))
DELIVERY_FILES = {
    'result.tsv': ('Native chromVAR result with ordered source identities','text/tab-separated-values; charset=utf-8'),
    'cells.tsv': ('Exact input cell order','text/tab-separated-values; charset=utf-8'),
    'features.tsv': ('Exact input feature or motif order','text/tab-separated-values; charset=utf-8'),
    'metadata.json': ('Native function, parameters and source semantics','application/json'),
    'native-receipt.json': ('Native chromVAR function completion and versions','application/json'),
}
FILES = tuple(DELIVERY_FILES)
MEASUREMENTS = ('peak_counts','fragment_counts','insertion_counts','binary_accessibility')


class NativeInvocationError(ValueError):
    """The installed function rejected the unchanged requested parameters."""


def arguments(values):
    return _arguments(_serialize(values))


def native_arguments(args):
    function = args.get('function_name')
    if function not in FUNCTIONS:
        raise ValueError('Unsupported chromVAR function; no function is substituted')
    kwargs = args.get('kwargs', {})
    if type(kwargs) is not dict or set(kwargs) - set(FUNCTIONS[function]):
        raise ValueError('Unreviewed native arguments for the selected function; input objects come from bound sources')
    for key, value in kwargs.items():
        if key == 'bootstrap_quantiles' and type(value) is list:
            if len(value) > 128 or any(type(v) not in (int,float) or not math.isfinite(v) for v in value):
                raise ValueError('Quantiles require a bounded native numeric vector')
        elif not (value is None or type(value) in (bool,int) or type(value) is float and math.isfinite(value)
                  or type(value) is str and len(value)<=128 and not any(ord(c)<32 for c in value)):
            raise ValueError('Native arguments must be bounded JSON data')
    seed = args.get('seed')
    if seed is not None and (type(seed) is not int or not 0<=seed<=2147483647):
        raise ValueError('Explicit R seed must be an integer from 0 to 2147483647')
    if function not in ('computeVariability', 'getBackgroundPeaks') and seed is not None:
        raise ValueError('Seed applies only to computeVariability or getBackgroundPeaks')
    if function not in ('computeExpectations', 'differentialDeviations', 'differentialVariability') and args.get('group_key') is not None:
        raise ValueError('group_key applies only to expectations or differential deviations/variability')
    if function != 'getBackgroundPeaks' and args.get('bias_key') is not None:
        raise ValueError('bias_key applies only to getBackgroundPeaks')
    return kwargs


def declarations(handle,args):
    slot = args.get('count_slot','X')
    if slot != 'X' and not slot.startswith('layers/'):
        slot = 'layers/'+slot
    effective = metadata_declarations(handle,slot,args,read_elem(handle['var']))
    effective['count_slot'] = slot
    return effective


def prepare_input(args,stage):
    peak_h5ad.inspect(args['input_path'],require_matrix=False)
    with h5py.File(args['input_path'],'r') as handle:
        obs,var = read_elem(handle['obs']),read_elem(handle['var'])
        cells,features = tuple(obs.index),tuple(var.index)
        if any(any(c in s for c in '\t\r\n') for s in cells+features):
            raise ValueError('R exchange axis labels cannot contain TSV delimiters')
        cell_table = pd.DataFrame(dict(source_index=np.arange(len(cells)),cell_id=cells))
        feature_table = pd.DataFrame(dict(source_index=np.arange(len(features)),feature_id=features))
        if args['function_name'] in DEVIATION_FUNCTIONS:
            if args.get('count_slot') is not None or args.get('matrix_semantics') is not None:
                raise ValueError('This function consumes supplied deviations and z_scores, not a declared count slot')
            if 'X' not in handle or 'layers/z_scores' not in handle:
                raise ValueError('This native deviation function requires supplied X deviations and the z_scores layer')
            for slot,target in [('X','deviations.bin'),('layers/z_scores','z.bin')]:
                matrix = read_elem(handle[slot])
                if not isinstance(matrix,np.ndarray) or matrix.shape!=(len(cells),len(features)) or matrix.dtype.kind not in 'fi':
                    raise ValueError('Recorded chromVAR scores require bounded dense cell-by-motif arrays')
                np.asarray(matrix,dtype='<f8',order='C').tofile(stage/target)
            if 'name' in var:
                names=var['name'].tolist()
                if any(not isinstance(s,str) or any(c in s for c in '\t\r\n') for s in names):
                    raise ValueError('Native motif names must be bounded TSV strings')
                feature_table['name']=names
            effective=dict(args,measurement=('supplied_chromVAR_z_assay'
                if args['function_name'] in ('computeVariability', 'differentialVariability')
                else 'supplied_chromVAR_deviations_with_z_assay'))
        else:
            effective=declarations(handle,args)
            # Count-only functions do not require genome/coordinate declarations.
            check_declarations(args['input_path'],effective)
            element=handle[effective['count_slot']]
            matrix=sparse_dataset(element) if isinstance(element,h5py.Group) else element
            if matrix.shape!=(len(cells),len(features)):
                raise ValueError('Counts and ordered axes differ')
            x=peak_h5ad.read_native_matrix(matrix)
            mmwrite(stage/'counts.mtx',x.T.tocsc())
            effective['measurement']=effective.get('matrix_semantics', 'unspecified_measurement')
            if args['function_name'] == 'getBackgroundPeaks':
                key = args.get('bias_key', 'bias')
                if key not in var:
                    raise ValueError('getBackgroundPeaks requires actual per-peak bias in var; select bias_key')
                bias = var[key].to_numpy()
                if bias.dtype.kind not in 'fiu' or bias.shape != (len(features),):
                    raise ValueError('The supplied bias column must be numeric in original feature order')
                peak_h5ad.native_double_values(bias)
                feature_table['bias'] = bias
                effective['measurement'] = 'native_1_based_background_feature_indices'

        if args.get('group_key') is not None:
            values=obs[args['group_key']].tolist()
            if any(not isinstance(s,str) or not s or any(c in s for c in '\t\r\n') for s in values):
                raise ValueError('Native grouping requires supplied nonmissing text labels in exact cell order')
            cell_table['group']=values
        cell_table.to_csv(stage/'cells.tsv',sep='\t',index=False)
        feature_table.to_csv(stage/'features.tsv',sep='\t',index=False)
    return effective,len(cells),len(features)


def native(args,stage):
    from .native_runtimes import rscript as runtime_rscript
    rscript=str(runtime_rscript('chromVAR'))
    if not Path(rscript).is_file():
        raise ValueError('Provision the optional installed chromVAR R runtime')
    script=Path(__file__).parent/'r/chromvar_functions_v1.R'
    implementation=file_closure([rscript,script])
    request=dict(stage=str(stage),function_name=args['function_name'],kwargs=native_arguments(args),seed=args.get('seed'),group_key=args.get('group_key'))
    request_sha=write(stage/'request.json',request)
    code=run([rscript,'--vanilla',str(script),str(stage/'request.json')],stage/'native.log',environment=r_environment())
    if code:
        shutil.copyfile(stage/'native.log',stage.parent/'native-failure.log')
        message=read(stage/'native-error.json').get('message','Native function rejected its inputs')[:2000] if (stage/'native-error.json').is_file() else 'Native function failed; consult the retained private runtime diagnostic'
        if peak_h5ad._PATH.search(message) or any(ord(c)<32 for c in message):
            message='Native function rejected its inputs; consult the retained private runtime diagnostic'
        raise NativeInvocationError(message)
    receipt=read(stage/'native-receipt.json')
    if receipt['completion']!='succeeded' or receipt['request_sha256']!=request_sha or receipt['function_name']!=args['function_name']:
        raise AuthorityError('Native completion differs from the exact requested invocation')
    if file_closure([rscript,script])!=implementation:
        raise AuthorityError('Native implementation changed during invocation')
    receipt['implementation']=implementation
    return receipt


def execute(values, execution_identity=None):
    from agent.tools._native_process import call
    return call('chromvar_native', dict(values=values, execution_identity=execution_identity),
                lambda: _execute(values, execution_identity))


def _execute(values, execution_identity=None):
    args=arguments(values);kwargs=native_arguments(args)
    root=Path(args['output_dir']);root.mkdir(parents=True,exist_ok=True)
    if any(root.iterdir()):
        raise ValueError('Output directory is not empty; native retry requires a new execution')
    source=file_closure([args['input_path']])[0]
    if args.get('input_sha256') and source['sha256']!=args['input_sha256']:
        raise AuthorityError('Selected input bytes changed')
    with tempfile.TemporaryDirectory(prefix='.chromvar-function-',dir=root) as temp:
        stage=Path(temp);effective,n,m=prepare_input(args,stage)
        receipt=native(effective,stage)
        table=pd.read_csv(stage/'result.tsv',sep='\t',keep_default_na=False,dtype={'source_id':str})
        axis = receipt['result_axis']
        if receipt['input_dim'] != [m,n] or receipt['rows'] != len(table) or receipt['columns'] != list(table):
            raise AuthorityError('Native result shape differs from its completion receipt')
        if axis == 'scalar':
            if len(table) != 1 or list(table) != ['value']:
                raise AuthorityError('Invalid native scalar representation')
        else:
            axis_file = 'cells' if axis in ('cells', 'cells_by_cells') else 'features'
            axes = pd.read_csv(stage/(axis_file+'.tsv'), sep='\t', keep_default_na=False, dtype={'cell_id':str,'feature_id':str})
            if (table['source_id'].tolist() != axes['cell_id' if axis_file=='cells' else 'feature_id'].tolist()
                    or table['source_index'].tolist() != list(range(len(axes)))):
                raise AuthorityError('Native result and exact ordered input axes differ')
            if axis == 'cells_by_cells' and list(table)[2:] != ['cell_' + str(i) for i in range(n)]:
                raise AuthorityError('Native matrix columns differ from ordered cells.tsv')
        for name in ('result.tsv','cells.tsv','features.tsv'):
            shutil.copyfile(stage/name,root/name)
        summary=dict(status='success',contract_version=CONTRACT,function_name='chromVAR::'+args['function_name'],
            n_cells=n,n_features=m,result_rows=len(table),result_columns=list(table),result_axis=axis,
            native_output_type=receipt['native_output_type'],measurement=effective['measurement'],
            parameters=dict(kwargs=kwargs,omitted_optional_arguments='native defaults',seed=args.get('seed'),
                group_key=args.get('group_key'),count_slot=effective.get('count_slot'),
                **({'bias_key': args.get('bias_key', 'bias')} if args['function_name'] == 'getBackgroundPeaks' else {})),
            source_identity=dict(sha256=source['sha256'],size_bytes=source['size_bytes']),
            software_versions=receipt['versions'],undefined_values=receipt['undefined_values'],limitations=LIMITATIONS)
        write(root/'native-receipt.json',receipt)
        write(root/'metadata.json',dict(summary=summary,source=source,receipt=receipt,
            input_orientation='H5AD cells by features; native R features by cells',result_axis=axis))
        expected={r['path']:r for r in [source,*receipt['implementation']]}
        sources=file_closure(expected)
        if sources!=sorted(expected.values(),key=lambda r:r['path']):
            raise AuthorityError('Source or runtime changed during invocation')
    value=dict(contract=CONTRACT,profile=PROFILE,arguments=args,execution_identity=execution_identity,
        result=summary,files=file_closure(root/name for name in FILES),sources=sources,native_completion='succeeded')
    path=root/'manifest.json';sha=write(path,value)
    if current() is not None:
        current().register_execution(KIND,root,execution_identity)
    return verify_publication(path,expected_sha256=sha)


def run_chromvar(input_path,output_dir,function_name,kwargs=None,input_sha256=None,count_slot=None,matrix_semantics=None,group_key=None,seed=None,bias_key=None):
    return execute({k:v for k,v in locals().items() if v is not None})


publication=NativePublication(tool=TOOL,kind=KIND,contract=CONTRACT,profile=PROFILE,verifier=VERIFIER,
    files=FILES,arguments=arguments,outputs={'table':'result.tsv'},historical_profiles=(HISTORICAL_PROFILE, PREVIOUS_PROFILE))
load=publication.load
result=publication.result
verify_publication=publication.verify_publication
verify_result=publication.verify_result
recover=publication.recover
