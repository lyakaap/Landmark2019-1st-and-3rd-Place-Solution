from pathlib import Path
from easydict import EasyDict as edict

import numpy as np
import h5py


def save_index_dataset(fn_out, dataset):
    Path(fn_out).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(fn_out, 'w') as f:
        f.create_dataset('feats_index', data=dataset.feats_index.astype(np.float32))
        f.create_dataset('feats_test', data=dataset.feats_test.astype(np.float32))
        f.create_dataset('ids_index',
                         data=dataset.ids_index.astype(dtype='S16'))
        f.create_dataset('ids_test',
                         data=dataset.ids_test.astype(dtype='S16'))


def l2norm_numpy(x):
    return x / np.linalg.norm(x, ord=2, axis=1, keepdims=True)


def load_index_dataset(fn, normalize=False):
    dataset = edict({
        'ids_index': None,
        'ids_test': None,
        'feats_index': None,
        'feats_test': None,
    })
    with h5py.File(fn, 'r') as f:
        for key in dataset.keys():
            data = f[key][()]
            if data.dtype == 'S16':
                data = data.astype(str)

            dataset[key] = data

    if normalize:
        dataset.feats_test = l2norm_numpy(dataset.feats_test)
        dataset.feats_index = l2norm_numpy(dataset.feats_index)

    return dataset
