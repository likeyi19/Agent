"""Resolve an explicit schedule using the pinned native trainer's defaults."""
from copy import deepcopy
from dataclasses import asdict
from functools import lru_cache
import json
import math
from agent.schemas.verification_authority import digest

METHOD = dict(id='epizoo-native-posttraining.v2', defaults='pinned-EpiZooXPostTrainConfig',
    overrides=['max_steps', 'learning_rate', 'training_batch_size'],
    admission='explicit-step-budget;cuda:0;qualified-loader-batch-at-most-4')
METHOD_SHA256 = digest(METHOD)
# METHOD is the frozen native-v2 scientific identity. Its original admission
# label is historical; default omission is now allowed. Resolved trainer values
# and requested_overrides remain part of the exact execution/publication hash.


class TrainingResourceError(ValueError):
    code = 'EPIZOO_TRAINING_CAPACITY_UNAVAILABLE'


def resolve(*, matrix_semantics, max_steps=None, learning_rate=None, training_batch_size=None):
    from epizoo.train.posttrain import EpiZooXPostTrainConfig
    from .epizoo_adaptation import contract as c
    if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
        raise ValueError('Training steps must be a positive integer; omit to use the pinned native default.')
    if learning_rate is not None and (type(learning_rate) not in (int, float)
            or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError('Training learning rate must be finite and positive.')
    if training_batch_size is not None and (type(training_batch_size) is not int or not 1 <= training_batch_size <= 4):
        raise ValueError('The qualified 24 GiB loader supports training batch sizes 1 through 4.')
    config = asdict(EpiZooXPostTrainConfig())
    for key in ('device', 'output_dir'): config.pop(key)
    if max_steps is not None: config['max_steps'] = max_steps
    if learning_rate is not None: config['lr'] = learning_rate
    value = deepcopy(c.scientific_profile(matrix_semantics))
    value.update(id=METHOD['id'], purpose='native', trainer=config,
        batch_size=4 if training_batch_size is None else training_batch_size, sequence_batch_size=4,
        requested_overrides=dict(max_steps=max_steps, learning_rate=learning_rate, training_batch_size=training_batch_size))
    return value


def validate(value):
    if value != resolve(matrix_semantics=value['matrix_semantics'], **value['requested_overrides']):
        raise ValueError('Native effective configuration differs from its explicit overrides.')
    return value


def resource_requirements(profile, n_features):
    """Necessary storage/state bounds, not an activation-memory or time promise."""
    weights, trainable = _state_sizes(n_features,json.dumps(profile['architecture'],sort_keys=True))
    cfg = profile['trainer']
    copies = min(cfg['max_steps']//cfg['save_steps'], cfg['keep_last']) + 2
    return dict(model_state_bytes=weights, minimum_gpu_bytes=weights+3*trainable,
                minimum_checkpoint_disk_bytes=copies*(weights+1024**2),
                runtime_seconds=None, activation_memory_bytes=None)


@lru_cache(maxsize=8)
def _state_sizes(n_features, architecture):
    import torch
    from epizoo.models.epizoo_x import EpiZooX, EpiZooXConfig
    with torch.device('meta'):
        model = EpiZooX(EpiZooXConfig(vocab_size=n_features+4, **json.loads(architecture)))
    parameters = dict(model.named_parameters())
    # Native freeze_seq_emb and fp32 parameters; gradients and AdamW moments
    # exist for all other trainable parameters. Activations are additional.
    weights = sum(v.numel()*4 for v in model.state_dict().values())
    trainable = sum(v.numel()*4 for k,v in parameters.items() if k != 'seq_emb.weight')
    return weights,trainable


def training_notice(arguments):
    """Display only resolved native requests; never infer post-training intent."""
    from .epizoo_adaptation import contract as c
    try:
        spec = c.read_json(arguments['adaptation_spec_path'],arguments['adaptation_spec_sha256'])
        if spec['contract_version'] != 'epizoo-adaptation-inputs.v2': return None
        profile = resolve(matrix_semantics='fragment_counts', **{k:arguments.get(k) for k in
                          ('max_steps','learning_rate','training_batch_size')})
        matrix = c.read_json(arguments['matrix_manifest_path'],arguments['matrix_manifest_sha256'])
        estimate = resource_requirements(profile,matrix['shape'][1])
        cfg=profile['trainer']
        return (f"Native post-training: {cfg['max_steps']:,} attempts, learning rate {cfg['lr']:g}, "
            f"batch {profile['batch_size']}. Omitted settings use native/qualified defaults. "
            f"Model/optimizer GPU-state lower bound {estimate['minimum_gpu_bytes']/1024**3:.2f} GiB; "
            f"checkpoint storage lower bound {estimate['minimum_checkpoint_disk_bytes']/1024**3:.2f} GiB. "
            'Additional activation and preprocessing memory/storage are required. Runtime is not calibrated. '
            'CUDA and free capacity are checked before dispatch; no automatic schedule reduction. '
            'The schedule is not a biological convergence guarantee.')
    except (KeyError,ValueError,TypeError,OSError):
        return None


def admit_resources(profile, n_features, output_dir):
    """Use local CUDA/disk capacity; never change scientific settings to fit."""
    import shutil
    import torch
    from pathlib import Path
    estimate = resource_requirements(profile,n_features)
    if not torch.cuda.is_available():
        raise TrainingResourceError('Post-training requires cuda:0. Restore CUDA availability before retrying.')
    free,_ = torch.cuda.mem_get_info('cuda:0')
    parent = Path(output_dir)
    while not parent.exists(): parent = parent.parent
    if free < estimate['minimum_gpu_bytes']:
        raise TrainingResourceError('Insufficient free GPU memory even for model and optimizer state. Release GPU resources before retrying; settings were not changed.')
    if shutil.disk_usage(parent).free < estimate['minimum_checkpoint_disk_bytes']+64*1024**2:
        raise TrainingResourceError('Insufficient disk for the resolved checkpoint retention schedule. Free disk or choose another managed workspace before retrying.')
    return estimate


def extract_sequences(reference, features, scratch_parent):
    """Give native pyfaidx a disposable index location beside a genome link.

    Native EpiZoo chooses a sibling FAI and can rebuild it. The exact admitted
    FASTA stays shared and untouched; only the small qualified index is copied.
    No sequence, coordinate, alias or ordering algorithm is implemented here.
    """
    from pathlib import Path
    import tempfile
    import shutil
    from epizoo.data.ccre import extract_dna_sequences
    with tempfile.TemporaryDirectory(prefix='.native-fasta-', dir=scratch_parent) as directory:
        fasta = Path(directory) / 'genome.fa'
        fasta.symlink_to(reference.genome.fasta.path)
        shutil.copyfile(reference.genome.fai.path, str(fasta) + '.fai')
        return extract_dna_sequences(str(fasta), features, fix_chrom_name=False,
                                     on_error='raise', show_progress=False)
