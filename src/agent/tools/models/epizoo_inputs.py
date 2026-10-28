"""Shallow, pinned EpiZoo input observations for conversational admission.

These observations do not grant execution authority. The existing model,
reference, matrix and adaptation owners still qualify each actual consumption.
"""
from pathlib import Path
from agent.tools.data import scatac_matrix_contract as matrix
from .epizoo_adaptation import contract as c, publication


def observe_source(path, sha):
    from agent.tools.data.scatac import read_feature_names
    if Path(path).stat().st_size > 1024 ** 3 or c.file_record(path)['sha256'] != sha:
        raise ValueError('H5AD identity or supported admission size differs.')
    names = read_feature_names(path)
    return dict(source_sha256=sha, feature_membership_sha256=c.digest(sorted(names)))


def observe_resource(inputs):
    if 'library_epizoo_transfer' in inputs:
        from .epizoo_adaptation.resources import code_bundle
        resource=inputs['library_epizoo_transfer']['resource']
        code=code_bundle()
        for key in ('source_bundle','seam_bundle'):
            if c.read_json(**dict(path=resource[key]['path'],sha=resource[key]['sha256']))['code']!=code:
                raise ValueError('Transfer resources belong to a different backend/owner version.')
        return dict(operation='initialization',universal_target=True)
    if 'target_model_manifest_path' in inputs:
        value = publication.load_manifest(inputs['target_model_manifest_path'], inputs['target_model_manifest_sha256'])
        path = Path(inputs['target_model_manifest_path']).parent / 'features.txt'
        if c.file_record(path)['sha256'] != value['files']['features.txt']['sha256']:
            raise ValueError('Target vocabulary changed.')
        return dict(operation='inference', model_state=value['arguments']['profile']['purpose'], species=value['reference']['species'],
            assembly=value['reference']['target_assembly'],
            matrix_semantics=value['arguments']['profile']['matrix_semantics'],
            feature_membership_sha256=c.digest(sorted(path.read_text().splitlines())),
            checkpoint_sha256=value['files']['checkpoint.pth']['sha256'])
    spec = c.read_json(inputs['adaptation_spec_path'], inputs['adaptation_spec_sha256'])
    matrix.shape(spec, ('contract_version', 'reference_identity_sha256', 'species', 'assembly',
                        'source_bundle', 'seam_bundle', 'mapping', 'execution_profile'))
    if spec['contract_version'] != 'epizoo-adaptation-inputs.v2' or spec['execution_profile'] != 'native.v2':
        raise ValueError('Conversational native training requires native.v2 resources; historical profiles remain explicit.')
    from agent.tools.data.regulatory_matrix_contract import validate_binding
    validate_binding(spec['species'], spec['assembly'])
    source = c.read_json(spec['source_bundle']['path'], spec['source_bundle']['sha256'])
    from .epizoo_adaptation.resources import code_bundle
    seam = c.read_json(spec['seam_bundle']['path'], spec['seam_bundle']['sha256'])
    if source['code'] != code_bundle() or seam['code'] != source['code']:
        raise ValueError('Qualified native resource code/runtime differs; admit a new owner-qualified version.')
    return dict(operation='posttraining', species=spec['species'], assembly=spec['assembly'],
        reference_identity_sha256=spec['reference_identity_sha256'], mapping=spec['mapping'],
        checkpoint_sha256=source['checkpoint']['sha256'])


def qualify_resource(inputs):
    """Reuse owner qualification, with no training or new model parsing."""
    if 'library_epizoo_transfer' in inputs:
        from .epizoo_adaptation.resources import load_bundle
        resource=inputs['library_epizoo_transfer']['resource']
        for key in ('source_bundle','seam_bundle'): load_bundle(**resource[key])
    elif 'target_model_manifest_path' in inputs:
        from .epizoo_target_inference import bind_model
        bind_model(inputs['target_model_manifest_path'], inputs['target_model_manifest_sha256'],
            inputs.get('target_model_source_run_id'), inputs.get('target_model_source_step_id'))
    else:
        from .epizoo_adaptation.resources import load_bundle
        spec = c.read_json(inputs['adaptation_spec_path'], inputs['adaptation_spec_sha256'])
        for key in ('source_bundle', 'seam_bundle'):
            load_bundle(spec[key]['path'], spec[key]['sha256'])
