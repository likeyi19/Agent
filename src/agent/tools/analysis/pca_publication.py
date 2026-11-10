"""Read-only accepted PCA publications; no execution, replay or new publication."""
from pathlib import Path
from agent.schemas.verification_authority import AuthorityError, VerifiedArtifactAuthority, digest
from agent.schemas.orchestration import _serialize
from .neighbors_publication import read_json, _shape, _file, _sha

KIND = 'cell_pca'
CONTRACT = 'cell-pca-publication.v1'
PROFILE = digest(dict(method='centered PCA', solver='arpack', preprocessing='supplied X unchanged',
                      verification='centered-projection-leading-eigenspace.v1'))
TFIDF_CONTRACT = 'cell-pca-publication.v2'
TFIDF_PROFILE = digest(dict(method='centered PCA', solver='arpack',
    preprocessing='sklearn.TfidfTransformer;norm=l2;smooth_idf=True;sublinear_tf=False',
    verification='centered-projection-leading-eigenspace.v1'))

def _identity(args):
    return (TFIDF_CONTRACT, TFIDF_PROFILE) if args.get('preprocessing') == 'tfidf_l2' else (CONTRACT, PROFILE)


VERIFIER = dict(id='agent.cell-pca-source-fit', compatibility_version='1')


def arguments(values):
    args = dict(values)
    if set(args)-{'input_path','output_dir','n_components','random_seed','overwrite','preprocessing'}:
        raise AuthorityError('Unexpected PCA argument.')
    for key in ('input_path','output_dir'):args[key]=str(Path(args[key]).resolve())
    for key,value in (('n_components',None),('random_seed',0),('overwrite',False)):args.setdefault(key,value)
    if args.get('preprocessing', 'none') not in ('none', 'tfidf_l2'):
        raise AuthorityError('Invalid PCA preprocessing.')
    if args.get('preprocessing') == 'none': args.pop('preprocessing')
    return args


def load(path,sha=None):
    value,actual=read_json(Path(path),sha)
    _shape(value,('contract','profile','execution_identity','arguments','result','files','sources'))
    if (value['contract'],value['profile'])!=_identity(value['arguments']) or value['arguments']!=arguments(value['arguments']):
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


def verify_publication(path, *, expected_sha256=None):
    from agent.tools.data.raw_history import accepted
    value, sha = load(path, expected_sha256)
    accepted(KIND, path, sha)
    return _result(value, path, sha)


def describe(path,sha,context):
    value,actual=load(path,sha)
    files=context.files([path,*(f['path'] for f in value['files'])])
    if [f for f in files if f['path']!=str(path)]!=value['files']:
        raise AuthorityError('PCA publication integrity changed.')
    identity=dict(kind=KIND,manifest=value,files=files,upstream={},historical_sources=value['sources'],
        execution_identity=context.execution_for(KIND,path),profile=value['profile'],verifier=VERIFIER,
        qualification={},policy='historical_verified_sources.v1')
    return dict(kind=KIND,manifest_path=str(path),manifest_sha256=actual,manifest=value,
        proof_identity=identity,files=files,historical_sources=value['sources'],upstream={},
        profile=value['profile'],verifier=VERIFIER,qualification={})


def reuse_result(kind,description,proof,kwargs):
    return _result(description['manifest'],description['manifest_path'],description['manifest_sha256'])


def proof_metadata(kind,result,context):return dict(result)


def publication_record(context, tool_name, args, result, execution_identity, result_metadata):
    from agent.tools.data.raw_history import accepted
    value, record = accepted(KIND, result['manifest_path'], result['manifest_sha256'], context=context)
    if (tool_name != 'pca_embed_cells' or record['execution_identity'] != execution_identity
            or record['arguments_sha256'] != digest(arguments(args))
            or record['resources']['result_metadata'] != _serialize(result_metadata)
            or _result(value, result['manifest_path'], result['manifest_sha256']) != dict(result)):
        raise AuthorityError('Historical PCA differs from its accepted execution.')
    return VerifiedArtifactAuthority(record)


def hooks():
    from agent.orchestration.registry import ScientificAuthorityHooks
    return ScientificAuthorityHooks(kind=KIND, producer_ports=('embedding',), describe=describe,
        publication_record=publication_record, reuse_result=reuse_result, proof_metadata=proof_metadata)
