"""Fixed isolated entry point; all diffusion mathematics and decoding are native."""
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys


def main(request_path):
    import numpy as np
    import torch
    from dnadiffusion.models.pretrained_unet import PretrainedUNet
    from dnadiffusion.models.diffusion import Diffusion
    from dnadiffusion.utils.sample_util import create_sample

    path = Path(request_path)
    request = json.loads(path.read_text())
    settings, sampling = request['settings'], request['sampling']
    torch.set_num_threads(settings['threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    pretrained, loading = PretrainedUNet.from_pretrained(request['model_directory'],
        local_files_only=True, output_loading_info=True)
    if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise ValueError('Pretrained checkpoint did not load exactly')
    model = Diffusion(pretrained.model, **request['schedule']).to(settings['device']).eval()
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('Qualified sampling requires float32 pretrained parameters')
    random.seed(sampling['seed']); np.random.seed(sampling['seed']); torch.manual_seed(sampling['seed'])
    mapping = {r['index']:r['label'] for r in request['condition_mapping']}
    label = mapping[sampling['condition_index']]
    count, batch = sampling['number_of_sequences'], settings['batch_size']
    full, remainder = divmod(count, batch)
    # Each native call receives an exactly divisible count. No oversampling,
    # discarding, duplication, or replacement of the native sampler is needed.
    calls = ([(full * batch, batch)] if full else []) + ([(remainder,remainder)] if remainder else [])
    sequences, batches = [], []
    original = Path.cwd()
    try:
        for i, (n, bs) in enumerate(calls):
            directory = path.parent / f'batch-{i}'
            (directory/'data/outputs').mkdir(parents=True)
            os.chdir(directory)
            create_sample(model,cell_type=sampling['condition_index'],sample_bs=bs,
                conditional_numeric_to_tag=mapping,number_of_samples=n,
                cond_weight_to_metric=sampling['guidance_scale'],save_dataframe=True,sequence_length=200)
            produced = (directory/'data/outputs'/f'{label}.txt').read_text().splitlines()
            if len(produced) != n: raise ValueError('Native sampling returned an incorrect count')
            sequences.extend(produced); batches.append(dict(number_of_sequences=n,batch_size=bs))
    finally:
        os.chdir(original)
    receipt = dict(completion='succeeded',settings=settings,sampling=sampling,schedule=request['schedule'],
        native_calls=batches,loading=loading,python=platform.python_version(),
        packages={name:importlib.metadata.version(name) for name in
                  ('dnadiffusion','torch','transformers','safetensors','numpy')},
        device=str(model.device),device_name=torch.cuda.get_device_name(model.device) if model.device.type=='cuda' else platform.processor(),
        precision='float32; TF32 disabled; no autocast',
        native_api='PretrainedUNet.from_pretrained -> Diffusion -> create_sample')
    path.with_suffix('.result.json').write_text(json.dumps(dict(sequences=sequences,receipt=receipt),allow_nan=False))


if __name__ == '__main__':
    main(sys.argv[1])
