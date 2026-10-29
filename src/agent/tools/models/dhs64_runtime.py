"""Fixed native DHS launcher, executed only in the isolated TensorFlow runtime.

No Agent imports or model implementation. The parent owns input qualification,
publication and cancellation; this process loads released native models once.
"""
import importlib.metadata
import json
from pathlib import Path
import platform
import sys


def main(request_path):
    request = json.loads(Path(request_path).read_text())
    sys.path.insert(0, request['source_root'])
    import numpy as np
    import tensorflow as tf
    from enhancerdesign import model, sequence

    settings = request['settings']
    tf.config.threading.set_intra_op_parallelism_threads(settings['intra_threads'])
    tf.config.threading.set_inter_op_parallelism_threads(settings['inter_threads'])
    tf.keras.utils.set_random_seed(settings['seed'])
    # CPU is an explicit deployment choice, not a failure fallback.
    if settings['device'] == 'cpu':
        tf.config.set_visible_devices([], 'GPU')
    elif not tf.config.list_physical_devices('GPU'):
        raise ValueError('Configured TensorFlow GPU is unavailable')
    x = sequence.one_hot_encode(request['sequences'], max_seq_len=500,
                                padding='left').astype('float32')
    predictions = []
    for checkpoint in request['checkpoints']:
        m = model.load_model(checkpoint)
        predictions.append(np.stack(m.predict(x, batch_size=settings['batch_size'], verbose=0)))
        tf.keras.backend.clear_session()
    np.savez(Path(request_path).with_suffix('.npz'), folds=np.stack(predictions))
    receipt = dict(python=platform.python_version(), packages={name: importlib.metadata.version(name)
        for name in ('tensorflow-cpu', 'keras', 'numpy', 'pandas', 'scipy', 'biopython', 'editdistance')},
        settings=settings, devices=[d.name for d in tf.config.get_visible_devices()],
        completion='succeeded', shape=list(np.stack(predictions).shape))
    Path(request_path).with_suffix('.receipt.json').write_text(json.dumps(receipt))


if __name__ == '__main__':
    main(sys.argv[1])
