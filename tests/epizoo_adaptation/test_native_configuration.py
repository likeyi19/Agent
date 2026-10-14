"""Native configuration overrides remain exact and historical profiles closed."""
from copy import deepcopy
from dataclasses import asdict
import pytest

from agent.tools.models import native_posttraining as native
from agent.tools.models.epizoo_adaptation import contract as c


def test_native_defaults_and_explicit_overrides():
    from epizoo.train.posttrain import EpiZooXPostTrainConfig
    defaults = asdict(EpiZooXPostTrainConfig())
    default_steps = defaults['max_steps']
    for key in ('device', 'output_dir'): defaults.pop(key)
    profile = native.resolve(matrix_semantics='binary_accessibility', max_steps=default_steps)
    assert profile['trainer'] == defaults
    short = native.resolve(matrix_semantics='binary_accessibility', max_steps=10,
                           learning_rate=1e-5, training_batch_size=1)
    assert short['trainer'] == dict(defaults, max_steps=10, lr=1e-5)
    assert c.validate_execution(short) == short
    assert c.profile_sha256(short) == native.METHOD_SHA256
    assert short['batch_size'] == 1
    assert c.PROFILE_SHA256 == c.digest(c.PROFILE)
    assert c.BINARY_PROFILE_SHA256 == c.digest(c.BINARY_PROFILE)


@pytest.mark.parametrize('change', [dict(max_steps=0), dict(max_steps=True),
    dict(max_steps=1.5), dict(learning_rate=0), dict(learning_rate=float('inf')),
    dict(learning_rate=True), dict(training_batch_size=0), dict(training_batch_size=5)])
def test_invalid_settings_fail_without_fallback(change):
    with pytest.raises(ValueError):
        native.resolve(**(dict(matrix_semantics='binary_accessibility', max_steps=10) | change))


@pytest.mark.parametrize('field', ['lr', 'max_steps', 'keep_last', 'warmup_steps'])
def test_effective_configuration_tampering_rejected(field):
    profile = native.resolve(matrix_semantics='fragment_counts', max_steps=10)
    profile['trainer'][field] += 1
    with pytest.raises(ValueError): c.validate_execution(profile)


def test_typed_conversational_values_bind_registered_training_inputs():
    from agent.application.dialogue_execution import _bind_arguments
    from agent.application.turn_decisions import ScopedArgument
    from agent.orchestration.registry import build_default_tool_registry
    utterance = 'Train for 2000 steps at learning rate 1e-5 with batch size 1.'
    values = [('max_steps', '2000', utterance.index('2000')),
              ('learning_rate', '1e-5', utterance.index('1e-5')),
              ('training_batch_size', '1', utterance.rindex('1'))]
    arguments = tuple(ScopedArgument('adapt_epizoo_species', key, literal, start, start+len(literal))
                      for key, literal, start in values)
    assert _bind_arguments(build_default_tool_registry(), utterance, {}, arguments) == dict(
        max_steps=2000, learning_rate=1e-5, training_batch_size=1)


def test_native_sequence_read_does_not_write_shared_reference_directory(tmp_path):
    from types import SimpleNamespace as N
    shared = tmp_path/'shared'; shared.mkdir()
    scratch = tmp_path/'scratch'; scratch.mkdir()
    fasta = shared/'genome.fa'; fasta.write_text('>chr1\nACGTACGT\n')
    index = shared/'separately-qualified.fai'; index.write_text('chr1\t8\t6\t8\t9\n')
    reference = N(genome=N(fasta=N(path=str(fasta)), fai=N(path=str(index))))
    before = {p.name:p.read_bytes() for p in shared.iterdir()}
    assert native.extract_sequences(reference, ['chr1:1-5','chr1:5-8'], scratch) == ['CGTA','CGT']
    assert {p.name:p.read_bytes() for p in shared.iterdir()} == before
    assert not list(scratch.iterdir())


@pytest.mark.parametrize('overrides', [{},{'max_steps':2000},{'learning_rate':1e-5},
                                      {'training_batch_size':1},{'max_steps':20,'learning_rate':2e-5}])
def test_omitted_parameters_inherit_owner_defaults(overrides):
    from epizoo.train.posttrain import EpiZooXPostTrainConfig
    defaults=asdict(EpiZooXPostTrainConfig()); defaults.pop('device');defaults.pop('output_dir')
    p=native.resolve(matrix_semantics='binary_accessibility',**overrides)
    expected=dict(defaults)
    if 'max_steps' in overrides:expected['max_steps']=overrides['max_steps']
    if 'learning_rate' in overrides:expected['lr']=overrides['learning_rate']
    assert p['trainer']==expected and p['batch_size']==overrides.get('training_batch_size',4)
    assert p['requested_overrides']=={k:overrides.get(k) for k in ('max_steps','learning_rate','training_batch_size')}
    assert c.validate_execution(p)==p
    assert defaults['max_steps']==500000
    explicit=native.resolve(matrix_semantics='binary_accessibility',max_steps=500000)
    assert c.digest(p)!=c.digest(explicit) if not overrides else True


@pytest.mark.parametrize('capacity', ['enough','cuda_missing','gpu_low','disk_low'])
def test_default_schedule_resource_admission_without_training(tmp_path,monkeypatch,capacity):
    import torch,shutil
    from types import SimpleNamespace
    p=native.resolve(matrix_semantics='binary_accessibility')
    estimate=native.resource_requirements(p,17922)
    assert estimate['minimum_checkpoint_disk_bytes']>estimate['model_state_bytes']*6
    assert estimate['runtime_seconds'] is None and estimate['activation_memory_bytes'] is None
    monkeypatch.setattr(torch.cuda,'is_available',lambda:capacity!='cuda_missing')
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda device:(0 if capacity=='gpu_low' else 24*1024**3,24*1024**3))
    monkeypatch.setattr(shutil,'disk_usage',lambda p:SimpleNamespace(free=0 if capacity=='disk_low' else 100*1024**3))
    if capacity=='enough': assert native.admit_resources(p,17922,tmp_path)==estimate
    else:
        with pytest.raises(ValueError): native.admit_resources(p,17922,tmp_path)
    assert p['trainer']['max_steps']==500000


def test_running_training_notice_explains_defaults_and_lower_bounds(tmp_path):
    spec=c.write_json(tmp_path/'spec.json',dict(contract_version='epizoo-adaptation-inputs.v2'))
    matrix=c.write_json(tmp_path/'matrix.json',dict(shape=[16,17922]))
    args=dict(adaptation_spec_path=spec['path'],adaptation_spec_sha256=spec['sha256'],
              matrix_manifest_path=matrix['path'],matrix_manifest_sha256=matrix['sha256'])
    notice=native.training_notice(args)
    assert '500,000 attempts' in notice and '5e-05' in notice and 'batch 4' in notice
    assert 'lower bound' in notice and 'Runtime is not calibrated' in notice
    assert str(tmp_path) not in notice
    assert '10 attempts' in native.training_notice(dict(args,max_steps=10))
    assert native.training_notice({}) is None
