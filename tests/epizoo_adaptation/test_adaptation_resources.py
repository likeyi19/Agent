"""Resource contract tests; filesystem checks mocked, no substitute execution weights."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import runpy
import pytest
import torch
from agent.tools.models.epizoo_adaptation import resources as r
from agent.tools.models.epizoo_adaptation.contract import digest


@pytest.fixture
def source_contract(monkeypatch,tmp_path):
    from epizoo.models.epizoo import EpiZoo,EpiZooConfig
    from agent.tools.models.epizoo import MODEL_CONFIG
    cfg=EpiZooConfig(**dict(MODEL_CONFIG))
    with torch.device('meta'): model=EpiZoo(cfg)
    buffers=['cca_loss_fn.pos_weight','signal_loss_fn.pos_weight']
    checkpoint=dict(path=str(tmp_path/'weights'),sha256=r.SOURCE_SHA,size_bytes=5231645507)
    value=dict(contract_version='epizoo-source-bundle.v1',checkpoint=checkpoint,architecture=asdict(cfg),
        retained={},files=[checkpoint],missing_buffer_exception=buffers,
        tensors={k:dict(shape=list(v.shape),dtype='torch.float32' if k in buffers else 'torch.float16') for k,v in model.state_dict().items()},
        code=dict(test_code=True))
    for species,sha in r.FILTER_SHA.items():
        record=dict(path=str(tmp_path/species),sha256=sha,size_bytes=1)
        value['files'].append(record)
        value['retained'][species]=dict(count=MODEL_CONFIG[species+'_vocab_size'],filter=record,
            ordered_sha256=r.RETAINED_ORDER_SHA[species])
    monkeypatch.setattr(r,'check_file',lambda record:None)
    monkeypatch.setattr(r,'code_bundle',lambda:dict(test_code=True))
    return value


@pytest.mark.parametrize('bad',['checkpoint','architecture','filter','count','order','schema','buffer','code'])
def test_rehashed_resource_contract_fails(source_contract,bad):
    value=source_contract
    if bad=='checkpoint':value['checkpoint']['sha256']='0'*64
    elif bad=='architecture':value['architecture']['num_layers']=1
    elif bad=='filter':value['retained']['human']['filter']['sha256']='0'*64
    elif bad=='count':value['retained']['human']['count']=1
    elif bad=='order':value['retained']['human']['ordered_sha256']='0'*64
    elif bad=='schema':value['tensors']['ccre_emb.weight']['shape']=[1,512]
    elif bad=='buffer':value['missing_buffer_exception'].append('rank_emb.weight')
    else:value['code']={'test_code':False}
    value['identity_sha256']=digest(value)
    with pytest.raises(ValueError):r.verify_bundle(value)


def test_model_key_shape_contract(source_contract):
    source_contract['identity_sha256']=digest(source_contract)
    assert r.verify_bundle(source_contract)==source_contract


def _qualification_arguments():
    path=Path(__file__).with_name('qualify_real_backend.py')
    return runpy.run_path(str(path),run_name='qualification_location_test')['_parse_arguments']


def test_real_weight_qualification_locations_are_separate(tmp_path,monkeypatch):
    checkpoint=tmp_path/'models'/'pretrained_EpiZoo.pth'
    auxiliary=tmp_path/'references'/'vocabulary'
    checkout=tmp_path/'EpiZoo'
    monkeypatch.setenv('AGENT_EPIZOO_CHECKPOINT',str(checkpoint))
    monkeypatch.setenv('AGENT_EPIZOO_RESOURCES_DIR',str(auxiliary))
    args=_qualification_arguments()(['--output',str(tmp_path/'output'),'--epizoo',str(checkout)])
    assert args.checkpoint==checkpoint
    assert args.resources_dir==auxiliary
    assert args.epizoo==checkout
    assert not args.output.exists()


def test_real_weight_qualification_explicit_locations_override_environment(tmp_path,monkeypatch):
    monkeypatch.setenv('AGENT_EPIZOO_CHECKPOINT',str(tmp_path/'unused'/'checkpoint.pth'))
    monkeypatch.setenv('AGENT_EPIZOO_RESOURCES_DIR',str(tmp_path/'unused-resources'))
    models=tmp_path/'explicit-models';auxiliary=tmp_path/'explicit-auxiliary'
    args=_qualification_arguments()(['--output',str(tmp_path/'output'),'--epizoo',str(tmp_path/'checkout'),
                                    '--models',str(models),'--resources-dir',str(auxiliary)])
    assert args.checkpoint==models/'pretrained_EpiZoo.pth'
    assert args.resources_dir==auxiliary
    assert not args.output.exists()


def test_real_weight_qualification_has_no_old_path_default(tmp_path,monkeypatch):
    monkeypatch.delenv('AGENT_EPIZOO_CHECKPOINT',raising=False)
    monkeypatch.delenv('AGENT_EPIZOO_RESOURCES_DIR',raising=False)
    with pytest.raises(ValueError,match='AGENT_EPIZOO_CHECKPOINT'):
        _qualification_arguments()(['--output',str(tmp_path/'output'),'--epizoo',str(tmp_path/'checkout')])
    assert not (tmp_path/'output').exists()
