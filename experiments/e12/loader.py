import h5py
import numpy as np
from easydict import EasyDict as edict


def l2norm_numpy(x):
    return x / np.linalg.norm(x, ord=2, axis=1, keepdims=True)


def load_train_dataset_singlefile(fn):
    dataset = edict({
        'ids_train': None,
        'ids_test': None,
        'feats_train': None,
        'feats_test': None,
    })
    with h5py.File(fn, 'r') as f:
        for key in dataset.keys():
            data = f[key][()]
            if data.dtype == 'S16':
                data = data.astype(str)
            dataset[key] = data

    dataset.feats_test = l2norm_numpy(dataset.feats_test)
    dataset.feats_train = l2norm_numpy(dataset.feats_train)

    return dataset


def load_train_dataset(blocksize=64):
    fmt = ('yokoo/landmark/experiments/v7/'
           'feats_train_{blockidx}_ms_M_ep4_batch_size-32_'
           'epochs-5_pooling-G,G,G,G.h5')

    dataset = edict({
        'ids_train': [],
        'ids_test': [],
        'feats_train': [],
        'feats_test': [],
    })
    for blockidx in range(blocksize):
        fn = fmt.format(blockidx=blockidx)
        with h5py.File(fn, 'r') as f:
            for key in dataset.keys():
                data = f[key][()]
                if data.dtype == 'S16':
                    data = data.astype(str)
                dataset[key].append(data)
    for key in dataset.keys():
        dataset[key] = np.concatenate(dataset[key], axis=0)

    dataset.feats_test = l2norm_numpy(dataset.feats_test)
    dataset.feats_train = l2norm_numpy(dataset.feats_train)

    return dataset
