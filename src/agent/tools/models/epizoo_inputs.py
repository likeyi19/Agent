"""Shallow, pinned EpiZoo input observations for conversational admission.

These observations do not grant execution authority. The existing model,
reference, matrix and adaptation owners still qualify each actual consumption.
"""
from pathlib import Path
from agent.tools.data import scatac_matrix_contract as matrix
from .epizoo_adaptation import contract as c, publication


def observe_source(path, sha):
    import h5py
    from anndata.io import read_elem
    from agent.tools.analysis.peak_h5ad import inspect, MAX_BYTES
    if Path(path).stat().st_size > MAX_BYTES or c.file_record(path)['sha256'] != sha:
        raise ValueError('H5AD identity or supported admission size differs.')
    inspect(path)
    with h5py.File(path, 'r') as handle:
        names = read_elem(handle['var']).index.tolist()
    if not names or len(set(names)) != len(names):
        raise ValueError('EpiZoo requires a nonempty unique feature vocabulary.')
    return dict(source_sha256=sha, feature_membership_sha256=c.digest(sorted(names)))


def observe_resource(inputs):
    if 'target_model_manifest_path' in inputs:
        value = publication.load_manifest(inputs['target_model_manifest_path'], inputs['target_model_manifest_sha256'])
        path = Path(inputs['target_model_manifest_path']).parent / 'features.txt'
        if c.file_record(path)['sha256'] != value['files']['features.txt']['sha256']:
            raise ValueError('Target vocabulary changed.')
        return dict(operation='inference', species=value['reference']['species'],
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
    if 'target_model_manifest_path' in inputs:
        from .epizoo_target_inference import bind_model
        bind_model(inputs['target_model_manifest_path'], inputs['target_model_manifest_sha256'],
            inputs.get('target_model_source_run_id'), inputs.get('target_model_source_step_id'))
    else:
        from .epizoo_adaptation.resources import load_bundle
        spec = c.read_json(inputs['adaptation_spec_path'], inputs['adaptation_spec_sha256'])
        for key in ('source_bundle', 'seam_bundle'):
            load_bundle(spec[key]['path'], spec[key]['sha256'])
