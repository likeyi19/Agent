from copy import deepcopy
import random
import numpy as np
import pytest
import torch
from agent.tools.models.epizoo_adaptation import contract as c,backend,resources
from agent.tools.models.epizoo_adaptation.publication import arguments
from agent.tools._cancellation import cancellation_scope,ToolWorkCancelled


@pytest.mark.parametrize('key,value',[('max_length',4096),('cca_alpha',0),('random_sample',False),('dtype','float16'),('frozen',[]),('num_workers',1)])
def test_no_silent_science_overrides(profile,key,value):
    profile[key]=value
    with pytest.raises(ValueError):c.validate_execution(profile)


@pytest.mark.parametrize('key,value',[('mode','sr'),('lr',1e-3),('warmup_steps',0),('freeze_seq_emb',False),('sr_weight',0),('cca_weight',0),('use_amp',False)])
def test_fixed_trainer(profile,key,value):
    profile['trainer'][key]=value
    with pytest.raises(ValueError):c.validate_execution(profile)


@pytest.mark.parametrize('key,value',[('seed',-1),('seed',True),('max_steps',11),('batch_size',5),('sequence_batch_size',129),('purpose','unknown')])
def test_execution_bounds(key,value):
    args=dict(purpose='qualification',seed=0,batch_size=1,sequence_batch_size=1,max_steps=2,save_steps=2,log_steps=1)
    args[key]=value
    with pytest.raises(ValueError):c.execution_profile(**args)


def test_seed_all_rngs_and_restore():
    before=(random.getstate(),np.random.get_state(),torch.get_rng_state())
    with backend.seeded(7):a=(random.random(),np.random.rand(),torch.rand(2))
    with backend.seeded(7):b=(random.random(),np.random.rand(),torch.rand(2))
    assert a[:2]==b[:2] and torch.equal(a[2],b[2])
    assert random.getstate()==before[0]
    np.testing.assert_array_equal(np.random.get_state()[1],before[1][1])
    assert torch.equal(torch.get_rng_state(),before[2])


def test_cancellation_before_loader_work():
    with cancellation_scope(lambda:True),pytest.raises(ToolWorkCancelled):list(backend.CheckedLoader([1]))


def test_file_identity_mutation(tmp_path):
    p=tmp_path/'resource';p.write_bytes(b'original');record=c.file_record(p)
    p.write_bytes(b'mutated!')
    with pytest.raises(ValueError):c.check_file(record)


def test_missing_seam_fails(tmp_path):
    with pytest.raises(FileNotFoundError):resources.qualify_seam_bundle(checkpoint_path=tmp_path/'absent',
        config_dir=tmp_path/'missing',source_url=resources.SEAM_URL,output_path=tmp_path/'bundle.json')


def test_public_registry_integration_retains_backend_profile():
    from agent.orchestration.registry import build_default_tool_registry
    spec=build_default_tool_registry().get('adapt_epizoo_species')
    assert spec.recovery_policy_version==c.POLICY
    assert 'profile' not in spec.required_arguments
