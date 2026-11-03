"""Existing native completion/publication lifecycle shared by compatible owners.

No scientific replay or new authority: exact manifests, completion and file
closures are admitted by the existing VerificationContext and RunStore hooks.
"""
from pathlib import Path
from agent.schemas.orchestration import _serialize
from agent.schemas.verification_authority import AuthorityError, VerifiedArtifactAuthority, digest
from agent.tools.data.authority_context import owned_verification
from agent.tools.data.fastq_verification_authority import file_closure
from .sequence_prediction import read


class NativePublication:
    def __init__(self, *, tool, kind, contract, profile, verifier, files, arguments, outputs,
                 historical_profiles=()):
        self.tool, self.kind, self.contract = tool, kind, contract
        self.profile, self.verifier, self.files = profile, verifier, files
        self.arguments, self.outputs = arguments, outputs
        self.historical_profiles = frozenset(historical_profiles)
        self.verify_publication = owned_verification(kind)(self.verify_publication)

    def load(self,path,sha=None):
        value = read(path,sha);root = Path(path).parent
        files = self.files(value) if callable(self.files) else self.files
        if (value['contract']!=self.contract or value['profile'] not in {self.profile, *self.historical_profiles} or root!=Path(value['arguments']['output_dir'])
                or Path(path).name!='manifest.json' or value['native_completion']!='succeeded'
                or {r['path'] for r in value['files']}!={str(root/name) for name in files} or len(value['files'])!=len(files)):
            raise AuthorityError('Invalid native scientific publication identity or membership')
        return value,file_closure([path])[0]['sha256']


    def result(self,value,path,sha):
        result = dict(value['result'], manifest_path=str(path), manifest_sha256=sha)
        outputs = self.outputs(value) if callable(self.outputs) else self.outputs
        for prefix, name in outputs.items():
            if name is None:
                result.update({prefix+'_path': None, prefix+'_sha256': None})
                continue
            pin = next(r for r in value['files'] if Path(r['path']).name == name)
            result.update({prefix+'_path': pin['path'], prefix+'_sha256': pin['sha256']})
        return result


    def verify_publication(self,path,*,expected_sha256):
        value,sha = self.load(path,expected_sha256)
        if file_closure(r['path'] for r in value['files']+value['sources'])!=sorted(value['files']+value['sources'],key=lambda r:r['path']):
            raise AuthorityError('Native scientific publication or source integrity changed')
        return self.result(value,path,sha)


    def verify_result(self,args,output):
        value,sha = self.load(output['manifest_path'],output['manifest_sha256'])
        if value['arguments']!=self.arguments(args) or _serialize(output)!=self.verify_publication(output['manifest_path'],expected_sha256=sha):
            raise AuthorityError('Native scientific result differs from its producing request')


    def recover(self,args,execution_identity):
        args = self.arguments(args);path = Path(args['output_dir'])/'manifest.json';value,sha = self.load(path)
        if value['arguments']!=args or value['execution_identity']!=execution_identity:
            raise AuthorityError('Recovery cannot authorize a native retry')
        return self.verify_publication(path,expected_sha256=sha)


    def describe(self,path,sha,context):
        value,actual = self.load(path,sha);files = context.files([path,*(r['path'] for r in value['files'])])
        if [r for r in files if r['path']!=str(path)]!=value['files']:
            raise AuthorityError('Native scientific artifact integrity changed')
        identity = dict(kind=self.kind,manifest=value,files=files,historical_sources=value['sources'],upstream={},
            execution_identity=context.execution_for(self.kind,path),profile=value['profile'],verifier=self.verifier,qualification={})
        return dict(kind=self.kind,manifest_path=str(path),manifest_sha256=actual,manifest=value,proof_identity=identity,
            files=files,historical_sources=value['sources'],upstream={},profile=value['profile'],verifier=self.verifier,qualification={})


    def reuse_result(self,kind,description,proof,kwargs):
        return self.result(description['manifest'],description['manifest_path'],description['manifest_sha256'])


    def proof_metadata(self,kind,output,context):
        return dict(output)


    def publication_record(self,context,tool_name,args,output,execution_identity,result_metadata):
        value,sha = self.load(output['manifest_path'],output['manifest_sha256'])
        if (value['profile']!=self.profile or tool_name!=self.tool or value['execution_identity']!=execution_identity or value['arguments']!=self.arguments(args)
                or _serialize(result_metadata)!=_serialize(output) or self.result(value,output['manifest_path'],sha)!=_serialize(output)):
            raise AuthorityError('Native scientific publication lacks its exact producing execution')
        description,key = context.describe(self.kind,Path(output['manifest_path']),sha)
        return VerifiedArtifactAuthority(dict(schema_version=2,artifact_type='agent.'+self.kind,artifact_contract=self.contract,
            publication_path=output['manifest_path'],manifest_sha256=sha,files=description['files'],execution_identity=execution_identity,
            arguments_sha256=digest(self.arguments(args)),receipt_sha256=sha,upstream={'producer_inputs':digest(description['proof_identity'])},
            resources=dict(scientific_proof_sha256=key,manifest_identity=value,result_metadata=_serialize(output)),science_profile=self.profile,
            producer_qualification={},verifier=self.verifier,scope='scientific_correctness.v1',completion='succeeded',
            integrity_scope='artifact_integrity_lineage_historical_sources.v1',source_policy='historical_verified_sources.v1',historical_sources=description['historical_sources']))
