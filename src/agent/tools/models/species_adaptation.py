"""Public integration of the frozen M14.2 owner; no scientific implementation."""
from pathlib import Path
from .epizoo_adaptation import contract as c, publication as backend, resources
from agent.tools.data import scatac_matrix_contract as matrix

ARGUMENTS = ('matrix_manifest_path','matrix_manifest_sha256','adaptation_spec_path',
             'adaptation_spec_sha256','strategy','output_dir')
RECOVERY_POLICY = c.POLICY
SPECIFICATION = 'epizoo-adaptation-inputs.v1'
BASE_FIELDS = ('status','artifact_type','contract_version','manifest_path','manifest_sha256',
               'checkpoint_path','checkpoint_sha256','completed_step','purpose','n_cells','n_features')
FACT_FIELDS = ('species','assembly','reference_identity_sha256','ordered_feature_sha256',
    'source_matrix_identity_sha256','source_model_identity_sha256','seam_bundle_identity_sha256',
    'strategy','profile_sha256','optimizer_steps','amp_skipped_steps','mapping_identity_sha256',
    'adaptation_spec_sha256')
RESULT_FIELDS = (*BASE_FIELDS, *FACT_FIELDS)


def expand(args):
    args = dict(args)
    matrix.shape(args, ARGUMENTS)
    for key in ARGUMENTS:
        if key.endswith('_sha256'): matrix.sha(args[key])
        elif key.endswith('_path') or key == 'output_dir': matrix.absolute_path(str(args[key]))
    if args['strategy'] not in ('de_novo','mapped_reference'):
        raise ValueError('Explicit adaptation strategy required; no fallback.')
    spec = c.read_json(args['adaptation_spec_path'], args['adaptation_spec_sha256'])
    matrix.shape(spec, ('contract_version','reference_identity_sha256','species','assembly',
                        'source_bundle','seam_bundle','mapping','execution_profile'))
    if spec['contract_version'] != SPECIFICATION: raise ValueError('Unsupported adaptation specification.')
    from agent.tools.data.regulatory_matrix_contract import validate_binding
    validate_binding(spec['species'],spec['assembly'])
    matrix.sha(spec['reference_identity_sha256'])
    if spec['execution_profile'] not in ('qualification.v1','production-candidate.v1'):
        raise ValueError('Explicit qualified execution profile required.')
    current_code = resources.code_bundle()
    for key in ('source_bundle','seam_bundle'):
        matrix.shape(spec[key],('path','sha256'))
        matrix.absolute_path(spec[key]['path']); matrix.sha(spec[key]['sha256'])
        bundle = c.read_json(spec[key]['path'],spec[key]['sha256'])
        if bundle.get('code') != current_code:
            raise ValueError('Qualified EpiZoo revision/code/runtime differs from current backend.')
    if args['strategy']=='mapped_reference':
        if not isinstance(spec['mapping'],dict):
            raise ValueError('mapped_reference requires explicit chain and retained source reference resources.')
        mapping=spec['mapping']
        matrix.shape(mapping,('source_species','source_reference','target_reference_identity','chain',
                             'liftover','bedtools','min_overlap','direction','tool_versions'))
        if (mapping['direction']!='target-to-source' or mapping['source_species'] not in ('human','mouse') or
                mapping['target_reference_identity']!=spec['reference_identity_sha256']):
            raise ValueError('Mapping direction/source/target reference mismatch.')
        for key in ('chain','liftover','bedtools','source_reference'): c.check_file(mapping[key])
    qualification = spec['execution_profile'] == 'qualification.v1'
    profile = c.execution_profile(purpose='qualification' if qualification else 'production',
        seed=0,batch_size=1 if qualification else 4,sequence_batch_size=4 if qualification else 128,
        max_steps=10 if qualification else 500000,save_steps=10 if qualification else 4000,
        log_steps=1 if qualification else 500)
    expanded = backend.arguments(dict(matrices=[dict(manifest_path=str(args['matrix_manifest_path']),
        manifest_sha256=args['matrix_manifest_sha256'])],source_bundle=spec['source_bundle'],
        seam_bundle=spec['seam_bundle'],strategy=args['strategy'],mapping=spec['mapping'],
        profile=profile,device='cuda:0',output_dir=str(args['output_dir'])))
    # Metadata-only checks precede all model loading/SEAM/training. The owner
    # subsequently verifies conservation and complete scientific compatibility.
    value = matrix.load_manifest_bytes(Path(args['matrix_manifest_path']).read_bytes())
    from agent.tools.data import regulatory_matrix_contract as neutral
    if (c.file_record(args['matrix_manifest_path'])['sha256'] != args['matrix_manifest_sha256'] or
        value['contract_version'] != neutral.CONTRACT or value['matrix_semantics'] != 'fragment_counts' or
        value['species'] != spec['species'] or value['assembly'] != spec['assembly'] or
        value['reference']['identity_sha256'] != spec['reference_identity_sha256']):
        raise ValueError('Adaptation matrix/reference/species/assembly/semantics mismatch.')
    return expanded


def _publication(args, identity):
    return backend._publication(expand(args), identity)


_receipt = backend._receipt


def _summary(args, base):
    value = backend.load_manifest(base['manifest_path'],base['manifest_sha256'])
    source = matrix.load_manifest_bytes(Path(args['matrix_manifest_path']).read_bytes())
    return dict(base,species=source['species'],assembly=source['assembly'],
        reference_identity_sha256=source['reference']['identity_sha256'],
        ordered_feature_sha256=source['ordered_feature_sha256'],source_matrix_identity_sha256=source['identity_sha256'],
        source_model_identity_sha256=value['source_identity'],seam_bundle_identity_sha256=value['seam_identity'],
        strategy=value['arguments']['strategy'],profile_sha256=value['profile_sha256'],
        optimizer_steps=value['execution']['optimizer_steps'],amp_skipped_steps=value['execution']['amp_skipped_steps'],
        mapping_identity_sha256=c.digest(value['arguments']['mapping']) if value['arguments']['mapping'] else None,
        adaptation_spec_sha256=args['adaptation_spec_sha256'])


def execute(arguments, execution_identity=None):
    if execution_identity is None: raise ValueError('Adaptation requires durable Agent execution.')
    expanded = expand(arguments)
    output = backend.adapt_epizoo_species(**expanded, execution_identity=execution_identity)
    return _summary(arguments,output['result'])


def verify_public_result(args, result):
    expanded = expand(args)
    base = {key:result[key] for key in BASE_FIELDS}
    value = backend.verify_public_result(expanded,base)
    if dict(result) != _summary(args,base): raise ValueError('Adaptation public result mismatch.')
    return value


def recover(arguments, execution_identity):
    result = backend.recover_adaptation(expand(arguments),execution_identity)
    return _summary(arguments,result)


def adapt_epizoo_species(matrix_manifest_path, matrix_manifest_sha256, adaptation_spec_path,
        adaptation_spec_sha256, strategy, output_dir):
    return execute(locals())
