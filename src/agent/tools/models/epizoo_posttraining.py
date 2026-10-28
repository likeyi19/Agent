"""Post-train an exact initialized EpiZooX through the existing model owner."""
from pathlib import Path
from copy import deepcopy
from .epizoo_adaptation import contract as c, publication as owner
from . import species_adaptation as adaptation
from .native_posttraining import lifecycle_profile, admit_resources
from agent.tools.data import scatac_matrix_contract as matrix
from agent.tools.data.neutral_matrix_binding import matrix_semantics, reference_binding

ARGUMENTS = ('matrix_manifest_path','matrix_manifest_sha256','target_model_manifest_path',
             'target_model_manifest_sha256','output_dir')
OPTIONAL_ARGUMENTS = (*adaptation.OPTIONAL_ARGUMENTS,'target_model_source_run_id','target_model_source_step_id')
RESULT_FIELDS = tuple(k for k in adaptation.RESULT_FIELDS if k!='adaptation_spec_sha256') + ('parent_model_manifest_sha256',)
RECOVERY_POLICY = c.POLICY


def expand(arguments):
    args=dict(arguments)
    if set(args)-set(ARGUMENTS)-set(OPTIONAL_ARGUMENTS) or not set(ARGUMENTS)<=set(args):
        raise ValueError('Invalid post-training arguments.')
    for k in ARGUMENTS:
        if k.endswith('_sha256'): c.checks.sha(args[k])
        else: c.checks.absolute_path(str(args[k]))
    parent=owner.load_manifest(args['target_model_manifest_path'],args['target_model_manifest_sha256'])
    if parent['contract_version']!=c.LIFECYCLE_CONTRACT or parent['arguments']['profile']['purpose']!='initialization':
        raise ValueError('Post-training requires an initialized EpiZooX checkpoint.')
    value=matrix.load_manifest_bytes(Path(args['matrix_manifest_path']).read_bytes())
    if c.file_record(args['matrix_manifest_path'])['sha256']!=args['matrix_manifest_sha256']:
        raise ValueError('Training matrix identity differs.')
    if (value['species']!=parent['reference']['species'] or value['assembly']!=parent['reference']['target_assembly']
            or value['ordered_feature_sha256']!=parent['preprocessing']['ordered_feature_sha256']
            or reference_binding(value)['identity_sha256']!=parent['reference']['reference_identity_sha256']
            or matrix_semantics(value)!=parent['arguments']['profile']['matrix_semantics']):
        raise ValueError('Training matrix does not match the exact initialized model vocabulary/reference/measurement.')
    expanded=deepcopy(parent['arguments'])
    expanded.update(matrices=[dict(manifest_path=str(args['matrix_manifest_path']),manifest_sha256=args['matrix_manifest_sha256'])],
        profile=lifecycle_profile(matrix_semantics=matrix_semantics(value),operation='posttraining',
            **{k:args.get(k) for k in adaptation.OPTIONAL_ARGUMENTS}),
        parent=dict(manifest_path=str(args['target_model_manifest_path']),manifest_sha256=args['target_model_manifest_sha256'],
            run_id=args.get('target_model_source_run_id'),step_id=args.get('target_model_source_step_id')),
        output_dir=str(args['output_dir']))
    return owner.arguments(expanded)


def _summary(args, base):
    return dict(adaptation._summary(args['matrix_manifest_path'], base),
                parent_model_manifest_sha256=args['target_model_manifest_sha256'])


def _publication(args, identity): return owner._publication(expand(args),identity)
_receipt=owner._receipt


def execute(arguments, execution_identity=None):
    if execution_identity is None: raise ValueError('Post-training requires durable execution.')
    expanded=expand(arguments)
    parent=owner._parent(expanded)
    admit_resources(expanded['profile'],parent['preprocessing']['n_features'],expanded['output_dir'])
    output=owner.adapt_epizoo_species(**expanded,execution_identity=execution_identity)
    return _summary(arguments,output['result'])


def recover(arguments,execution_identity):
    return _summary(arguments,owner.recover_adaptation(expand(arguments),execution_identity))


def verify_public_result(args,result):
    value=owner.verify_public_result(expand(args),{k:result[k] for k in adaptation.BASE_FIELDS})
    if dict(result)!=_summary(args,{k:result[k] for k in adaptation.BASE_FIELDS}):
        raise ValueError('Post-training result differs.')
    return value


def posttrain_epizoo_model(matrix_manifest_path,matrix_manifest_sha256,target_model_manifest_path,
        target_model_manifest_sha256,output_dir,max_steps=None,learning_rate=None,training_batch_size=None,
        target_model_source_run_id=None,target_model_source_step_id=None):
    return execute({k:v for k,v in locals().items() if v is not None})
