import warnings
import multiprocessing
import os
from pathlib import Path

import tqdm
import h5py
from easydict import EasyDict as edict
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from skimage.transform import AffineTransform
from skimage.measure import ransac as _ransac
import faiss

from src import utils


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
LOGFORMAT = '%(asctime)s %(levelname)s %(message)s'


def l2norm_numpy(x):
    return x / np.linalg.norm(x, ord=2, axis=1, keepdims=True)


def main():
    # train19_train19, used for generating cleaned subsets
    # output: 'input/train19_train19_search_top100_RANSAC.csv'
    euclidean_search_train()
    ransac_delf_train()

    # test19-train19, used for recognition
    # output: 'input/test19_train19_search_top100_RANSAC.csv'
    fn_npy = f'{ROOT}input/test19_train19_faiss_search_top100.npy'
    euclidean_search_test(fn_npy)
    for blockid in range(1, 64+1):
        ransac_delf_test(fn_npy, blockid)

    fn_out = f'{ROOT}input/test19_train19_search_top100_RANSAC.csv'
    with open(fn_out, 'w') as fw:
        for blockid in range(1, 64+1):
            fn_in = f'{ROOT}input/test19_train19_ransac{blockid}.csv'
            with open(fn_in, 'r') as f:
                for line in f:
                    fw.write(line)


def load_train_dataset(blocksize=64):
    # For train19-train19 RANSAC
    fmt = ROOT + 'exp/v7/feats_train_{blockidx}_ms_M_ep4_batch_size-32_epochs-5_pooling-G,G,G,G.h5'  # noqa

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


def load_train_ensemble_dataset():
    test_dirs = [
        ROOT + 'exp/v19c/feats_test19_ms_L2_ep4_scaleup_ep3_freqthresh-2_loss-cosface_pooling-G,G,G,G_verifythresh-30/',  # noqa
        ROOT + 'exp/v20c/feats_test19_ms_L2_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30/',  # noqa
        ROOT + 'exp/v21c/feats_test19_ms_L2_ep6_scaleup_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30/',  # noqa
        ROOT + 'exp/v22c/feats_test19_ms_L2_ep4_scaleup_ep3_base_margin-0.4_freqthresh-2_verifythresh-30/',  # noqa
        ROOT + 'exp/v23c/feats_test19_ms_L2_ep6_scaleup_ep5_augmentation-middle_epochs-7_freqthresh-3_verifythresh-30/',  # noqa
        ROOT + 'exp/v24c/feats_test19_ms_L2_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-cosface_verifythresh-30/',  # noqa
    ]
    train_dirs = [
        ROOT + 'exp/v19c/feats_train_ms_L2_ep4_scaleup_ep3_freqthresh-2_loss-cosface_pooling-G,G,G,G_verifythresh-30/',  # noqa
        ROOT + 'exp/v20c/feats_train_ms_L2_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30/',  # noqa
        ROOT + 'exp/v21c/feats_train_ms_L2_ep6_scaleup_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30/',  # noqa
        ROOT + 'exp/v22c/feats_train_ms_L2_ep4_scaleup_ep3_base_margin-0.4_freqthresh-2_verifythresh-30/',  # noqa
        ROOT + 'exp/v23c/feats_train_ms_L2_ep6_scaleup_ep5_augmentation-middle_epochs-7_freqthresh-3_verifythresh-30/',  # noqa
        ROOT + 'exp/v24c/feats_train_ms_L2_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-cosface_verifythresh-30/',  # noqa
    ]
    weights = [
        0.5,
        1.0,
        1.0,
        0.5,
        1.0,
        1.0,
    ]

    ids_test, feats_test = utils.prepare_ids_and_feats(
        test_dirs, weights, normalize=True)
    ids_train, feats_train = utils.prepare_ids_and_feats(
        train_dirs, weights, normalize=True)

    return edict(
        ids_test=ids_test,
        ids_train=ids_train,
        feats_test=feats_test,
        feats_train=feats_train)


def euclidean_search_train():
    for block_id in range(1, 32+1):
        faiss_search_results_train_train(block_id=block_id, n_splits=32)


def euclidean_search_test(fn_npy):
    ds = load_train_ensemble_dataset()

    cpu_index = faiss.IndexFlatL2(ds[f'feats_train'].shape[1])
    cpu_index.add(ds[f'feats_train'])

    _, all_ranks = cpu_index.search(x=ds[f'feats_test'], k=100)
    Path(fn_npy).parent.mkdir(parents=True, exist_ok=True)
    np.save(fn_npy, all_ranks)


def ransac_delf_train():
    for block_id in range(1, 32+1):
        matching_delf_train_train(block_id)


def ransac_delf_test(fn_npy, blockid):
    x = np.load(fn_npy)

    ds = load_train_ensemble_dataset()
    ids = ds.ids_test.tolist()

    id2idxs = {id_: idx
               for idx, id_ in enumerate(ds.ids_test.tolist())}
    index_idx2ids = {idx: id_
                     for idx, id_ in enumerate(ds.ids_train.tolist())}

    query_list = []
    for idx, id_ in enumerate(ids):
        if idx % 64 == blockid - 1:
            query_list.append(id_)

    map_args = []
    fn_out = f'{ROOT}input/test19_train19_ransac{blockid}.csv'
    p = multiprocessing.Pool(multiprocessing.cpu_count())

    with tqdm.tqdm(total=len(query_list) * 100) as pbar,\
            open(fn_out, 'w') as f_ransac:
        for id_ in query_list:
            test_idx = id2idxs[id_]

            for rank in range(100):
                index_idx = x[test_idx, rank]
                index_id = index_idx2ids[index_idx]
                try:
                    lhs = load_desc_loc(
                        f'{ROOT}input/delf/test/{id_}.npz')
                    rhs = load_desc_loc(
                        f'{ROOT}input/delf/train/{index_id}.npz')
                    map_args.append(edict(
                        lhs=lhs, rhs=rhs, imid=index_id, query=id_))
                except Exception:
                    pass

                if len(map_args) > 1000:
                    ransac_scores = calc_ransac_scores_exp12(map_args, p)
                    pbar.update(1000)
                    for query_id, imid, score in ransac_scores:
                        f_ransac.write(f'{query_id}\t{imid}\t{score}\n')
                        f_ransac.flush()
                    map_args = []

        if len(map_args) > 0:
            ransac_scores = calc_ransac_scores_exp12(map_args, p)
            pbar.update(len(map_args))
            for query_id, imid, score in ransac_scores:
                f_ransac.write(f'{query_id}\t{imid}\t{score}\n')
                f_ransac.flush()

            map_args = []


def load_train19_landmark_dict():
    train19_csv = ROOT + 'input/train.csv'
    df_trn = pd.read_csv(train19_csv, usecols=['id', 'landmark_id'])
    return {r['id']: r['landmark_id'] for _, r in df_trn.iterrows()}


def faiss_search_results_train_train(block_id=1, n_splits=1):
    dataset = load_train_dataset()
    print('Loading train19 landmark dict')
    landmark_dict = load_train19_landmark_dict()

    size_train = dataset.feats_train.shape[0]
    part_size = int(size_train / n_splits)
    idx_train_start = (block_id - 1) * part_size
    idx_train_end = (block_id) * part_size
    if block_id == n_splits:
        idx_train_end = size_train

    cpu_index = faiss.IndexFlatL2(dataset.feats_train.shape[1])
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    gpu_index.add(dataset.feats_train)
    dists, topk_idx = gpu_index.search(
        x=dataset.feats_train[idx_train_start:idx_train_end],
        k=1000)

    df = pd.DataFrame(
        dataset.ids_train[idx_train_start:idx_train_end],
        columns=['id'])
    df['images'] = np.apply_along_axis(
        ' '.join, axis=1, arr=dataset.ids_train[topk_idx])

    print('generate sub')
    rows = []
    for imidx, (_, r) in tqdm.tqdm(enumerate(df.iterrows()),
                                   total=len(df)):
        landmark_id = landmark_dict[r['id']]
        same_landmark_images = []
        for rank, imid in enumerate(r.images.split(' ')):
            if landmark_id == landmark_dict[imid]:
                same_landmark_images.append(
                    f'{rank}:{dists[imidx, rank]:.8f}:{imid}')
                if len(same_landmark_images) >= 100:
                    break

        rows.append({
            'id': r['id'],
            'landmark_id': landmark_id,
            'matched': ' '.join(same_landmark_images),
        })

    fn = (ROOT + 'input/' +
          f'train19_train19_faiss_search_same_landmarks_blk{block_id}.csv.gz')
    Path(fn).parent.mkdir(parents=True, exist_ok=True)

    print('to_csv')
    df = pd.DataFrame(rows).to_csv(fn, index=False,
                                   compression='gzip')


def matching_delf_train_train(block_id):
    fn = (ROOT + 'input/' +
          f'train19_train19_faiss_search_same_landmarks_blk{block_id}.csv.gz')
    df = pd.read_csv(fn)

    fn_out = (ROOT + 'input/' +
              f'train19_train19_verified_blk{block_id}.csv')

    p = multiprocessing.Pool(multiprocessing.cpu_count())
    with open(fn_out, 'w') as f:
        map_args = []
        for _, r in tqdm.tqdm(df.iterrows(), total=len(df)):
            id_ = r['id']
            matched = r['matched'].split(' ')
            if len(matched) <= 1:
                continue

            matched = matched[1:]
            for match in matched:
                rank, l2_dist, imid = match.split(':')
                rank = int(rank)
                l2_dist = float(l2_dist)

                map_args.append(
                    edict(lhs=load_desc_loc(
                          ROOT + f'input/delf/train/{id_}.npz'),
                          rhs=load_desc_loc(
                          ROOT + f'input/delf/train/{imid}.npz'),
                          imid=imid,
                          query=id_))

            if len(map_args) > 1000:
                ransac_scores = calc_ransac_scores_exp12(map_args, p)
                for query_id, imid, score in ransac_scores:
                    f.write(f'{query_id}\t{imid}:{score}\n')
                    f.flush()
                    map_args = []

        if len(map_args) > 1000:
            ransac_scores = calc_ransac_scores_exp12(map_args, p)
            for query_id, imid, score in ransac_scores:
                f.write(f'{query_id}\t{imid}:{score}\n')
                f.flush()
    p.close()


def load_desc_loc(fn):
    data = np.load(fn)
    return edict(desc=data['desc'], loc=data['loc'])


def calc_ransac_scores_exp12(map_args, p):
    ransac_scores = p.map(__calc_ransac_score, map_args)
    return ransac_scores


def __calc_ransac_score(q):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        score = 0
        try:
            score = get_inliers(q.lhs.loc, q.lhs.desc, q.rhs.loc, q.rhs.desc)
        except Exception:
            pass
    return (q.query, q.imid, score)


def get_inliers(loc1, desc1, loc2, desc2):
    n_feat1, n_feat2 = loc1.shape[0], loc2.shape[0]

    # from scipy.spatial import cKDTree
    KD_THRESH = 0.8
    d1_tree = cKDTree(desc1)
    distances, indices = d1_tree.query(desc2, distance_upper_bound=KD_THRESH)

    loc2_to_use = np.array([
        loc2[i, ] for i in range(n_feat2) if indices[i] != n_feat1])
    loc1_to_use = np.array([
        loc1[indices[i], ] for i in range(n_feat2) if indices[i] != n_feat1])

    np.random.seed(114514)

    # from skimage.measure import ransac as _ransac
    # from skimage.transform import AffineTransform
    model_robust, inliers = _ransac(
        (loc1_to_use, loc2_to_use),
        AffineTransform,
        min_samples=3,
        residual_threshold=20,
        max_trials=1000)

    return sum(inliers)


if __name__ == '__main__':
    main()
