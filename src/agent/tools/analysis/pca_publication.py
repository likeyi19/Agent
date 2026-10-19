"""PCA publication through the existing operation proof and authority mechanisms."""
from pathlib import Path
from agent.schemas.verification_authority import AuthorityError, VerifiedArtifactAuthority, digest
from agent.schemas.orchestration import _serialize
from agent.tools.data.authority_context import current, owned_verification
from agent.tools.data.fastq_verification_authority import file_closure
from .neighbors_publication import read_json, _write_json, _shape, _file, _sha

KIND = 'cell_pca'
CONTRACT = 'cell-pca-publication.v1'
PROFILE = digest(dict(method='centered PCA', solver='arpack', preprocessing='supplied X unchanged',
                      verification='centered-projection-leading-eigenspace.v1'))
VERIFIER = dict(id='agent.cell-pca-source-fit', compatibility_version='1')


def arguments(values):
    args = dict(values)
    if set(args)-{'input_path','output_dir','n_components','random_seed','overwrite'}:
        raise AuthorityError('Unexpected PCA argument.')
    for key in ('input_path','output_dir'):args[key]=str(Path(args[key]).resolve())
    for key,value in (('n_components',None),('random_seed',0),('overwrite',False)):args.setdefault(key,value)
    return args


def load(path,sha=None):
    value,actual=read_json(Path(path),sha)
    _shape(value,('contract','profile','execution_identity','arguments','result','files','sources'))
    if value['contract']!=CONTRACT or value['profile']!=PROFILE or value['arguments']!=arguments(value['arguments']):
        raise AuthorityError('Invalid PCA publication contract.')
    result=value['result'];root=Path(value['arguments']['output_dir'])
    if value['execution_identity'] is not None:_sha(value['execution_identity'])
    if (type(value['files']) is not list or type(value['sources']) is not list
            or len(value['files']) != 2 or len(value['sources']) != 1):
        raise AuthorityError('Invalid PCA publication file closure.')
    for entry in value['files']+value['sources']:_file(entry)
    if (sorted(f['path'] for f in value['files']) != sorted([result['analysis_path'],result['cell_ids_path']])
            or value['sources'][0]['path'] != value['arguments']['input_path']
            or result['input_path'] != value['arguments']['input_path']
            or value['sources'][0]['sha256'] != result['source_sha256']
            or next(f['sha256'] for f in value['files'] if f['path']==result['analysis_path']) != result['analysis_sha256']):
        raise AuthorityError('PCA file identities differ from their declared role.')
    if (Path(path).parent!=root or Path(result['analysis_path']).parent!=root
            or result['embedding_path']!=result['analysis_path'] or Path(result['cell_ids_path']).parent!=root):
        raise AuthorityError('PCA publication membership differs.')
    return value,actual


def _result(value,path,sha):
    return dict(value['result'],manifest_path=str(path),manifest_sha256=sha)


def publish(args,result,path):
    args=arguments(args);ctx=current();identity=None
    if ctx is not None and ctx.execution is not None and ctx.execution[:2]==(KIND,Path(args['output_dir'])):
        identity=ctx.execution[2]
    value=dict(contract=CONTRACT,profile=PROFILE,execution_identity=identity,arguments=args,result=result,
        files=file_closure([result['analysis_path'],result['cell_ids_path']]),sources=file_closure([args['input_path']]))
    sha=_write_json(path,value,overwrite=args['overwrite'])
    return verify_publication(path,expected_sha256=sha)


@owned_verification(KIND)
def verify_publication(path,*,expected_sha256=None):
    value,sha=load(path,expected_sha256)
    for field in ('files','sources'):
        if file_closure(f['path'] for f in value[field])!=value[field]:
            raise AuthorityError('PCA source or publication bytes changed.')
    from .pca_embedding import _verify_result
    _verify_result(value['arguments'],value['result'])
    return _result(value,path,sha)


def verify_result(args,result):
    value,sha=load(result['manifest_path'],result['manifest_sha256'])
    if value['arguments']!=arguments(args) or dict(result)!=verify_publication(result['manifest_path'],expected_sha256=sha):
        raise AuthorityError('PCA result differs from exact execution publication.')
    return value


def describe(path,sha,context):
    value,actual=load(path,sha)
    files=context.files([path,*(f['path'] for f in value['files'])])
    if [f for f in files if f['path']!=str(path)]!=value['files']:
        raise AuthorityError('PCA publication integrity changed.')
    identity=dict(kind=KIND,manifest=value,files=files,upstream={},historical_sources=value['sources'],
        execution_identity=context.execution_for(KIND,path),profile=PROFILE,verifier=VERIFIER,
        qualification={},policy='historical_verified_sources.v1')
    return dict(kind=KIND,manifest_path=str(path),manifest_sha256=actual,manifest=value,
        proof_identity=identity,files=files,historical_sources=value['sources'],upstream={},
        profile=PROFILE,verifier=VERIFIER,qualification={})


def reuse_result(kind,description,proof,kwargs):
    return _result(description['manifest'],description['manifest_path'],description['manifest_sha256'])


def proof_metadata(kind,result,context):return dict(result)


def publication_record(context,tool_name,args,result,execution_identity,result_metadata):
    value,sha=load(result['manifest_path'],result['manifest_sha256'])
    if (tool_name!='pca_embed_cells' or value['execution_identity']!=execution_identity
            or value['arguments']!=arguments(args) or _serialize(result_metadata)!=dict(result)
            or _result(value,result['manifest_path'],sha)!=dict(result)):
        raise AuthorityError('PCA publication lacks exact producing execution identity.')
    description,key=context.describe(KIND,Path(result['manifest_path']),sha)
    return VerifiedArtifactAuthority(dict(schema_version=2,artifact_type='agent.cell-pca',artifact_contract=CONTRACT,
        publication_path=result['manifest_path'],manifest_sha256=sha,files=description['files'],
        execution_identity=execution_identity,arguments_sha256=digest(arguments(args)),receipt_sha256=sha,
        upstream={'producer_inputs':digest(description['proof_identity'])},
        resources=dict(scientific_proof_sha256=key,manifest_identity=value,result_metadata=dict(result)),
        science_profile=PROFILE,producer_qualification={},verifier=VERIFIER,scope='scientific_correctness.v1',
        completion='succeeded',integrity_scope='artifact_integrity_lineage_historical_sources.v1',
        source_policy='historical_verified_sources.v1',historical_sources=description['historical_sources']))
