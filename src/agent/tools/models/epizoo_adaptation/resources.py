"""Explicit resource qualification. No weights or scientific implementations here."""
from dataclasses import asdict
import importlib.metadata
from pathlib import Path
import subprocess

from .contract import file_record,check_file,digest,read_json,write_json

SOURCE_SHA = '6b2d13fdbd54a9b0d56efa5afa81bc4832b813f4eac8662e4d93cc08c9d9b39a'
FILTER_SHA = dict(human='994d9c3e87208074e695c4c418b28d9587dd8991ad033cf33e62f96ceebc7875',
                  mouse='96a80287ae085d7e9e10d05f0dd7d5b266ad86d08b91b01ccc07af6b9b01e393')
RETAINED_ORDER_SHA = dict(human='3bd740e6613a13c6dfd41d986e9f82da67478ccdaefea15f90b503cda73268b8',
    mouse='0fce8d7b1c84b1cd0764c3ce6f3979ff06bf9b4baa0c7c86a11f225fd9c0a2e9')
SEAM_URL = 'https://drive.google.com/file/d/1VlPnwrvMxkvKkqEOvM_pciCwJH5Jgb8Y/view?usp=drive_link'
SEAM_SHA = 'd40b5626622a302f693f2e36ad4f4942e83cf6daf602e1add5216f72b75e2530'
SEAM_CONFIG_SHA = {
    'config.json':'ba9bdafaff0cc3e30556927474d4a179519a9864012bed2628e9f1bc23c84bfd',
    'tokenizer.json':'5d178e8ce2ba55df97fff197f4b30f40133b95d7096be398c2df6b526c5d8cd3',
    'tokenizer_config.json':'f9d18c81f4dd9dd7db02e9f27cc1203228147d890bfce9167c3af6465ff5b769',
    'configuration_bert.py':'95fc868641b87bbcd7a32d2cd7b9f4769c27592e129daf167d14b5b8c74ec4c5',
    'bert_layers.py':'87f201b5b1fea8472e12547588b4347639249afccdd3a3550f0980e19676c39d',
    'bert_padding.py':'44d1c68afb1f585fdc66c150d4c60f1ed44a89c006abc57d50531d71940d7421',
    'flash_attn_triton.py':'a5e139c7a756c723da24d831a190c4e8047b512777b074bc55ec13e0c7ba308f'}


def runtime():
    import torch
    import platform
    packages=('torch','numpy','pandas','scipy','anndata','transformers','tokenizers','flash-attn','pyfaidx','triton')
    return dict(packages={p:importlib.metadata.version(p) for p in packages},cuda=torch.version.cuda,
        python=platform.python_version(),cpu_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
        cudnn=torch.backends.cudnn.version(),tf32_matmul=False,tf32_cudnn=False,
        deterministic_algorithms=False,cudnn_benchmark=False)


def code_bundle():
    import epizoo
    root=Path(epizoo.__file__).resolve().parent
    revision=subprocess.check_output(['git','-C',str(root.parent),'rev-parse','HEAD'],text=True).strip()
    files=[file_record(p) for p in sorted(root.rglob('*.py'))]
    # Agent adapters are part of the compatibility closure as well.
    files += [file_record(p) for p in sorted(Path(__file__).parent.glob('*.py'))]
    return dict(epizoo_root=str(root),revision=revision,files=files,runtime=runtime())


def verify_bundle(value):
    if value['contract_version'] not in ('epizoo-source-bundle.v1','epizoo-seam-bundle.v1'):
        raise ValueError('Unsupported model resource contract.')
    if value['identity_sha256']!=digest({k:v for k,v in value.items() if k!='identity_sha256'}):
        raise ValueError('Bundle identity mismatch.')
    common={'contract_version','identity_sha256','files','code','checkpoint','tensors'}
    fields=(common|{'architecture','retained','missing_buffer_exception'} if value['contract_version']=='epizoo-source-bundle.v1'
            else common|{'source_url','config_dir','trust_remote_code'})
    if set(value)!=fields or value['checkpoint'] not in value['files']:
        raise ValueError('Incomplete resource bundle.')
    if value['contract_version']=='epizoo-source-bundle.v1':
        import torch
        from epizoo.models.epizoo import EpiZoo,EpiZooConfig
        from agent.tools.models.epizoo import MODEL_CONFIG
        if value['checkpoint']['sha256']!=SOURCE_SHA or value['architecture']!=asdict(EpiZooConfig(**dict(MODEL_CONFIG))):
            raise ValueError('Unqualified source checkpoint/architecture.')
        if set(value['retained'])!=set(FILTER_SHA): raise ValueError('Missing retained source species.')
        for species,sha in FILTER_SHA.items():
            if value['retained'][species]['filter']['sha256']!=sha or value['retained'][species]['filter'] not in value['files']:
                raise ValueError('Unqualified retained source vocabulary.')
            if value['retained'][species]['count']!=MODEL_CONFIG[species+'_vocab_size'] or value['retained'][species]['ordered_sha256']!=RETAINED_ORDER_SHA[species]:
                raise ValueError('Retained checkpoint row identity differs.')
        buffers=['cca_loss_fn.pos_weight','signal_loss_fn.pos_weight']
        if value['missing_buffer_exception']!=buffers: raise ValueError('Unexpected source buffer exception.')
        with torch.device('meta'): expected=EpiZoo(EpiZooConfig(**value['architecture'])).state_dict()
        schema={k:dict(shape=list(v.shape),dtype='torch.float32' if k in buffers else 'torch.float16') for k,v in expected.items()}
        if value['tensors']!=schema: raise ValueError('Source tensor schema differs from architecture.')
    else:
        if value['checkpoint']['sha256']!=SEAM_SHA: raise ValueError('Unqualified SEAM weights.')
        if value['trust_remote_code'] is not True or not isinstance(value['source_url'],str) or not value['source_url']:
            raise ValueError('Explicit SEAM provenance/custom-code binding required.')
        current_files={str(p.resolve()) for p in Path(value['config_dir']).iterdir() if p.is_file()}
        if current_files!={f['path'] for f in value['files'] if f!=value['checkpoint']}:
            raise ValueError('SEAM config/tokenizer/custom-code closure differs.')
        for name,sha in SEAM_CONFIG_SHA.items():
            if file_record(Path(value['config_dir'])/name)['sha256']!=sha:
                raise ValueError('Unqualified SEAM configuration/tokenizer/custom code.')
        import torch
        # DNABERT custom ALiBi construction does not support torch's meta device.
        # Its exact pinned checkpoint was strictly qualified against the CPU
        # model at provisioning; mmap inspects that immutable tensor schema here.
        check_file(value['checkpoint'])
        expected=torch.load(value['checkpoint']['path'],map_location='cpu',weights_only=True,mmap=True)
        if value['tensors']!={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in expected.items()}:
            raise ValueError('SEAM tensor schema differs from its pinned configuration.')
    for record in value['files']: check_file(record)
    if value['code']!=code_bundle(): raise ValueError('Backend code/runtime changed.')
    return value


def load_bundle(path,sha256):
    return verify_bundle(read_json(path,sha256))


def _finish(value,path):
    value['identity_sha256']=digest(value)
    verify_bundle(value)
    write_json(path,value)
    return value


def qualify_source_bundle(*,checkpoint_path,resources_dir,output_path):
    import torch
    from epizoo.qualification import qualify_source
    from epizoo.models.epizoo import EpiZooConfig
    from agent.tools.models.epizoo import MODEL_CONFIG,_load_resources,HUMAN_CONFIG,MOUSE_CONFIG
    checkpoint=file_record(checkpoint_path)
    if checkpoint['sha256']!=SOURCE_SHA: raise ValueError('Unqualified pretrained checkpoint.')
    files=[checkpoint]; retained={}
    for config in (HUMAN_CONFIG,MOUSE_CONFIG):
        info=_load_resources(config,resources_dir)
        if info.filter_sha256!=FILTER_SHA[config.name]: raise ValueError('Wrong/reordered retained checkpoint vocabulary.')
        records=[file_record(info.filter_path),file_record(info.frequency_path)]
        files.extend(records)
        retained[config.name]=dict(count=len(info.retained_names),filter=records[0],frequency=records[1],
            ordered_sha256=__import__('hashlib').sha256(('\n'.join(info.retained_names)+'\n').encode()).hexdigest())
    cfg=EpiZooConfig(**dict(MODEL_CONFIG))
    state=torch.load(checkpoint['path'],map_location='cpu',weights_only=True,mmap=True)
    state=qualify_source(state,cfg)
    schema={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in state.items()}
    return _finish(dict(contract_version='epizoo-source-bundle.v1',checkpoint=checkpoint,
        architecture=asdict(cfg),retained=retained,tensors=schema,
        missing_buffer_exception=['cca_loss_fn.pos_weight','signal_loss_fn.pos_weight'],files=files,code=code_bundle()),output_path)


def qualify_seam_bundle(*,checkpoint_path,config_dir,source_url,output_path):
    import json
    import torch
    from epizoo.models.seam import SEAM,SEAMConfig
    from epizoo.data.datasets import SEAMDataset,collate_fn_seam
    config_dir=Path(config_dir).resolve(strict=True)
    if file_record(checkpoint_path)['sha256']!=SEAM_SHA:
        raise ValueError('Initial SEAM profile requires the supplied real checkpoint.')
    config=json.loads((config_dir/'config.json').read_text())
    if any('--' in v or '/' in v for v in config['auto_map'].values()):
        raise ValueError('Remote custom-code references are prohibited.')
    for name in ('config.json','tokenizer.json','tokenizer_config.json','configuration_bert.py','bert_layers.py','bert_padding.py','flash_attn_triton.py'):
        if not (config_dir/name).is_file(): raise ValueError(f'Missing SEAM resource: {name}')
    files=[file_record(checkpoint_path)]+[file_record(p) for p in sorted(config_dir.iterdir()) if p.is_file()]
    for name,sha in SEAM_CONFIG_SHA.items():
        if file_record(config_dir/name)['sha256']!=sha:
            raise ValueError('Unqualified SEAM configuration/tokenizer/custom code.')
    model=SEAM(SEAMConfig(str(config_dir),trust_remote_code=True))
    state=torch.load(checkpoint_path,map_location='cpu',weights_only=True,mmap=True)
    expected=model.state_dict()
    if set(state)!=set(expected): raise ValueError('SEAM checkpoint keys incompatible; no key remapping.')
    for key,value in state.items():
        if value.shape!=expected[key].shape or value.dtype!=expected[key].dtype or not torch.isfinite(value).all():
            raise ValueError(f'Invalid SEAM tensor: {key}')
    model.load_state_dict(state,strict=True)
    model.requires_grad_(False).eval()
    dataset=SEAMDataset(['ACGT'*32],str(config_dir),max_length=512,return_index=True)
    batch=collate_fn_seam([dataset[0]])
    with torch.no_grad(): result=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'])
    if result.shape!=(1,512) or not torch.isfinite(result).all(): raise ValueError('SEAM forward incompatible.')
    return _finish(dict(contract_version='epizoo-seam-bundle.v1',source_url=source_url,
        checkpoint=files[0],config_dir=str(config_dir),trust_remote_code=True,files=files,
        tensors={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in state.items()},code=code_bundle()),output_path)
