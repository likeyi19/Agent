"""Fixed isolated entry point for unmodified CREsted DeepPBMC native ISE.

No CREsted code or weights are redistributed here. This adapter calls the
installed academic research package; runtime/device settings are operator-owned.
"""
import json
import os
from pathlib import Path
import sys


def main():
    request = Path(sys.argv[1])
    config = json.loads(request.read_text())
    settings = config['settings']
    os.environ['KERAS_BACKEND'] = 'torch'
    import numpy as np
    import torch
    import keras
    import crested
    torch.set_num_threads(settings['threads'])
    keras.utils.set_random_seed(settings['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = settings['device']
    if device != 'cpu' and not torch.cuda.is_available():
        raise RuntimeError('Configured CUDA device is unavailable')
    model = crested.utils.load_model(config['checkpoint'])
    model.to(device)
    if model.input_shape != (None, 2114, 4) or model.output_shape != (None, 7):
        raise ValueError('DeepPBMC model shape differs')
    effective = config['optimization']
    trajectories, candidates = [], []
    # Native ISE holds all substitutions in memory. Process independent input
    # records separately without altering its search or mutation batch size.
    with keras.device(device):
        for sequence in config['sequences']:
            intermediate, edited = crested.tl.design.in_silico_evolution(
                n_mutations=effective['iterations'], target=effective['target_index'],
                model=model, starting_sequences=[sequence], return_intermediate=True,
                no_mutation_flanks=tuple(effective['flanks']),
                protected_positions=[tuple(r) for r in effective['protected_ranges']],
                class_penalty_weights=np.asarray(effective['penalty_weights']))
            trajectories.extend(intermediate)
            candidates.extend(edited)
    def plain(value):
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, np.generic): return value.item()
        raise TypeError(type(value).__name__)
    result = dict(candidates=candidates, trajectories=trajectories,
        receipt=dict(completion='succeeded', settings=settings, optimization=effective,
            crested=crested.__version__, keras=keras.__version__, torch=torch.__version__,
            device=str(next(model.parameters()).device), backend=keras.backend.backend(),
            precision='float32; TF32 disabled; no autocast', native_prediction_batch_size=32,
            method='crested.tl.design.in_silico_evolution'))
    request.with_suffix('.result.json').write_text(json.dumps(result, default=plain, allow_nan=False))


if __name__ == '__main__':
    main()
