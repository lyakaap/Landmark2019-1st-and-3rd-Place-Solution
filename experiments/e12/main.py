import warnings
import multiprocessing
from logging import getLogger, Formatter, StreamHandler, INFO
# import json
from collections import Counter
from pathlib import Path

from scipy.spatial import cKDTree
from skimage.transform import AffineTransform
from skimage.measure import ransac as _ransac
import numpy as np
import pandas as pd
from easydict import EasyDict as edict
import click
import tqdm
import h5py
import faiss

import loader
from util import timer
# from ransac import run_ransac


LOGFORMAT = '%(asctime)s %(levelname)s %(message)s'
logger = getLogger('landmark')
logger.setLevel(INFO)


@click.group()
def cli():
    handler = StreamHandler()
    handler.setLevel(INFO)
    handler.setFormatter(Formatter(LOGFORMAT))

    logger.setLevel(INFO)
    logger.addHandler(handler)


@cli.command()
@click.option('-b', '--block-id', type=int, default=1)
def prep_faiss_search_results(block_id):
    __prep_faiss_search_results(block_id=block_id)


@cli.command()
@click.option('-b', '--block-id', type=int, default=1)
def prep_delf_results(block_id):
    __prep_delf1905_results(block_id)


@cli.command()
def prep_test19_filelist():
    rows = []
    with open('data/cvpr19_landmark/test/test.csv', 'r') as f:
        f.readline()
        for line in f:
            id_ = line.strip()
            rows.append('{}/{}/{}/{}/{}.jpg'.format(
                'data/cvpr19_landmark/test',
                id_[0], id_[1], id_[2], id_))

    for blockid in range(16):
        with open(f'data/working/test19.lst_blk{blockid+1}', 'w') as f:
            for i, r in enumerate(rows):
                if i % 16 == blockid:
                    f.write(r + '\n')


@cli.command()
def prep_index19_filelist():
    rows = []
    with open('data/cvpr19_landmark/index/index.csv', 'r') as f:
        f.readline()
        for line in f:
            id_ = line.strip()
            rows.append('{}/{}/{}/{}/{}.jpg'.format(
                'data/cvpr19_landmark/index',
                id_[0], id_[1], id_[2], id_))

    for blockid in range(32):
        with open(f'data/working/index19.lst_blk{blockid+1}', 'w') as f:
            for i, r in enumerate(rows):
                if i % 32 == blockid:
                    f.write(r + '\n')


@cli.command()
def proc_cleaned_train19_dataset():
    df_ic = []
    for blockid in range(1, 33):
        df_part = pd.read_csv(
            f'data/working/exp12/train19_train19_verified_blk{blockid}.csv',
            names=['id', 'result'],
            delimiter="\t")
        df_part.loc[:, 'cnt'] = df_part.result.apply(
            lambda x: int(x.split(':')[1]))
        df_part = df_part[df_part.cnt >= 50]
        df_ic.append(df_part)
    df_ic = pd.concat(df_ic, sort=False)
    df_ic = df_ic.groupby('id', as_index=False).count()

    df = pd.read_csv('data/cvpr19_landmark/train/train.csv',
                     usecols=['id', 'landmark_id'])
    df_ic = df_ic.merge(df, how='left', on='id')

    rows = []
    for landmark_id, df_part in tqdm.tqdm(df_ic.groupby('landmark_id')):
        if len(df_part) < 2:
            continue

        rows.append(dict(
            landmark_id=landmark_id,
            images=' '.join(df_part['id'].tolist())))
    pd.DataFrame(rows)[['landmark_id', 'images']].to_csv(
        'data/working/exp12/train19_cleaned_verifythresh50_freqthresh1.csv',
        index=False)

    df_ic = df_ic[df_ic.cnt >= 3]
    rows = []
    for landmark_id, df_part in tqdm.tqdm(df_ic.groupby('landmark_id')):
        if len(df_part) < 2:
            continue

        rows.append(dict(
            landmark_id=landmark_id,
            images=' '.join(df_part['id'].tolist())))
    pd.DataFrame(rows)[['landmark_id', 'images']].to_csv(
        'data/working/exp12/train19_cleaned_verifythresh50_freqthresh3.csv',
        index=False)

    df_ic = df_ic[df_ic.cnt >= 5]

    rows = []
    for landmark_id, df_part in tqdm.tqdm(df_ic.groupby('landmark_id')):
        if len(df_part) < 2:
            continue

        rows.append(dict(
            landmark_id=landmark_id,
            images=' '.join(df_part['id'].tolist())))
    pd.DataFrame(rows)[['landmark_id', 'images']].to_csv(
        'data/working/exp12/train19_cleaned_verifythresh50_freqthresh5.csv',
        index=False)


@cli.command()
def proc_cleaned_train19_dataset_verifythresh40():
    df_ic = []
    for blockid in range(1, 33):
        df_part = pd.read_csv(
            f'data/working/exp12/train19_train19_verified_blk{blockid}.csv',
            names=['id', 'result'],
            delimiter="\t")
        df_part.loc[:, 'cnt'] = df_part.result.apply(
            lambda x: int(x.split(':')[1]))
        df_part = df_part[df_part.cnt >= 40]
        df_ic.append(df_part)
    df_ic = pd.concat(df_ic, sort=False)
    df_ic = df_ic.groupby('id', as_index=False).count()

    df = pd.read_csv('data/cvpr19_landmark/train/train.csv',
                     usecols=['id', 'landmark_id'])
    df_ic = df_ic.merge(df, how='left', on='id')

    rows = []
    for landmark_id, df_part in tqdm.tqdm(df_ic.groupby('landmark_id')):
        if len(df_part) < 2:
            continue

        rows.append(dict(
            landmark_id=landmark_id,
            images=' '.join(df_part['id'].tolist())))
    pd.DataFrame(rows)[['landmark_id', 'images']].to_csv(
        'data/working/exp12/train19_cleaned_verifythresh40_freqthresh1.csv',
        index=False)

    df_ic = df_ic[df_ic.cnt >= 3]
    rows = []
    for landmark_id, df_part in tqdm.tqdm(df_ic.groupby('landmark_id')):
        if len(df_part) < 2:
            continue

        rows.append(dict(
            landmark_id=landmark_id,
            images=' '.join(df_part['id'].tolist())))
    pd.DataFrame(rows)[['landmark_id', 'images']].to_csv(
        'data/working/exp12/train19_cleaned_verifythresh40_freqthresh3.csv',
        index=False)

    df_ic = df_ic[df_ic.cnt >= 5]

    rows = []
    for landmark_id, df_part in tqdm.tqdm(df_ic.groupby('landmark_id')):
        if len(df_part) < 2:
            continue

        rows.append(dict(
            landmark_id=landmark_id,
            images=' '.join(df_part['id'].tolist())))
    pd.DataFrame(rows)[['landmark_id', 'images']].to_csv(
        'data/working/exp12/train19_cleaned_verifythresh40_freqthresh5.csv',
        index=False)


@cli.command()
def proc_cleaned_train19_dataset_verifythresh40_r2():
    df_ic = []
    for blockid in range(1, 33):
        df_part = pd.read_csv(
            f'data/working/exp12/train19_train19_verified_blk{blockid}.csv',
            names=['id', 'result'],
            delimiter="\t")
        df_part.loc[:, 'cnt'] = df_part.result.apply(
            lambda x: int(x.split(':')[1]))
        df_part = df_part[df_part.cnt >= 40]
        df_ic.append(df_part)
    df_ic = pd.concat(df_ic, sort=False)
    df_ic = df_ic.groupby('id', as_index=False).count()

    df = pd.read_csv('data/cvpr19_landmark/train/train.csv',
                     usecols=['id', 'landmark_id'])
    df_ic = df_ic.merge(df, how='left', on='id')

    df_ic = df_ic[df_ic.cnt >= 2]
    rows = []
    for landmark_id, df_part in tqdm.tqdm(df_ic.groupby('landmark_id')):
        if len(df_part) < 2:
            continue

        rows.append(dict(
            landmark_id=landmark_id,
            images=' '.join(df_part['id'].tolist())))
    pd.DataFrame(rows)[['landmark_id', 'images']].to_csv(
        'data/working/exp12/train19_cleaned_verifythresh40_freqthresh2.csv',
        index=False)


@cli.command()
def proc_cleaned_train19_dataset_verifythresh30():
    df_ic = []
    for blockid in range(1, 33):
        df_part = pd.read_csv(
            f'data/working/exp12/train19_train19_verified_blk{blockid}.csv',
            names=['id', 'result'],
            delimiter="\t")
        df_part.loc[:, 'cnt'] = df_part.result.apply(
            lambda x: int(x.split(':')[1]))
        df_part = df_part[df_part.cnt >= 30]
        df_ic.append(df_part)
    df_ic = pd.concat(df_ic, sort=False)
    df_ic = df_ic.groupby('id', as_index=False).count()

    df = pd.read_csv('data/cvpr19_landmark/train/train.csv',
                     usecols=['id', 'landmark_id'])
    df_ic = df_ic.merge(df, how='left', on='id')

    rows = []
    for landmark_id, df_part in tqdm.tqdm(df_ic.groupby('landmark_id')):
        if len(df_part) < 2:
            continue

        rows.append(dict(
            landmark_id=landmark_id,
            images=' '.join(df_part['id'].tolist())))
    pd.DataFrame(rows)[['landmark_id', 'images']].to_csv(
        'data/working/exp12/train19_cleaned_verifythresh30_freqthresh1.csv',
        index=False)

    df_ic = df_ic[df_ic.cnt >= 2]
    rows = []
    for landmark_id, df_part in tqdm.tqdm(df_ic.groupby('landmark_id')):
        if len(df_part) < 2:
            continue

        rows.append(dict(
            landmark_id=landmark_id,
            images=' '.join(df_part['id'].tolist())))
    pd.DataFrame(rows)[['landmark_id', 'images']].to_csv(
        'data/working/exp12/train19_cleaned_verifythresh30_freqthresh2.csv',
        index=False)

    df_ic = df_ic[df_ic.cnt >= 3]

    rows = []
    for landmark_id, df_part in tqdm.tqdm(df_ic.groupby('landmark_id')):
        if len(df_part) < 2:
            continue

        rows.append(dict(
            landmark_id=landmark_id,
            images=' '.join(df_part['id'].tolist())))
    pd.DataFrame(rows)[['landmark_id', 'images']].to_csv(
        'data/working/exp12/train19_cleaned_verifythresh30_freqthresh3.csv',
        index=False)


@cli.command()
def proc_cleaned_train19_dataset_verifythresh20():
    df_ic = []
    for blockid in range(1, 33):
        df_part = pd.read_csv(
            f'data/working/exp12/train19_train19_verified_blk{blockid}.csv',
            names=['id', 'result'],
            delimiter="\t")
        df_part.loc[:, 'cnt'] = df_part.result.apply(
            lambda x: int(x.split(':')[1]))
        df_part = df_part[df_part.cnt >= 20]
        df_ic.append(df_part)
    df_ic = pd.concat(df_ic, sort=False)
    df_ic = df_ic.groupby('id', as_index=False).count()

    df = pd.read_csv('data/cvpr19_landmark/train/train.csv',
                     usecols=['id', 'landmark_id'])
    df_ic = df_ic.merge(df, how='left', on='id')

    rows = []
    for landmark_id, df_part in tqdm.tqdm(df_ic.groupby('landmark_id')):
        if len(df_part) < 2:
            continue

        rows.append(dict(
            landmark_id=landmark_id,
            images=' '.join(df_part['id'].tolist())))
    pd.DataFrame(rows)[['landmark_id', 'images']].to_csv(
        'data/working/exp12/train19_cleaned_verifythresh20_freqthresh1.csv',
        index=False)

    df_ic = df_ic[df_ic.cnt >= 2]
    rows = []
    for landmark_id, df_part in tqdm.tqdm(df_ic.groupby('landmark_id')):
        if len(df_part) < 2:
            continue

        rows.append(dict(
            landmark_id=landmark_id,
            images=' '.join(df_part['id'].tolist())))
    pd.DataFrame(rows)[['landmark_id', 'images']].to_csv(
        'data/working/exp12/train19_cleaned_verifythresh20_freqthresh2.csv',
        index=False)

    df_ic = df_ic[df_ic.cnt >= 3]

    rows = []
    for landmark_id, df_part in tqdm.tqdm(df_ic.groupby('landmark_id')):
        if len(df_part) < 2:
            continue

        rows.append(dict(
            landmark_id=landmark_id,
            images=' '.join(df_part['id'].tolist())))
    pd.DataFrame(rows)[['landmark_id', 'images']].to_csv(
        'data/working/exp12/train19_cleaned_verifythresh20_freqthresh3.csv',
        index=False)


@cli.command()
def prep_test_filepath_list():
    __prep_test_filepath_list()


@cli.command()
def scoring_with_top100_arcfacefish_v4():
    __scoring_with_top100_arcfacefish_v4()


@cli.command()
def scoring_with_top100_arcfacefish_v3():
    __scoring_with_top100_arcfacefish_v3()


@cli.command()
def scoring_with_top100_arcfacefish_v2():
    __scoring_with_top100_arcfacefish_v2()


@cli.command()
def scoring_with_top100_arcfacefish_v1():
    __scoring_with_top100_arcfacefish_v1()


def l2norm_numpy(x):
    return x / np.linalg.norm(x, ord=2, axis=1, keepdims=True)


def load_train19_landmark_dict():
    train19_csv = 'data/cvpr19_landmark/train/train.csv'
    df_trn = pd.read_csv(train19_csv, usecols=['id', 'landmark_id'])
    return {r['id']: r['landmark_id'] for _, r in df_trn.iterrows()}


def __prep_faiss_search_results(block_id=1):
    # 1 ~ 32

    dataset = loader.load_train_dataset()
    with timer('Loading train19 landmark dict'):
        landmark_dict = load_train19_landmark_dict()

    size_train = dataset.feats_train.shape[0]
    part_size = int(size_train / 32)
    idx_train_start = (block_id - 1) * part_size
    idx_train_end = (block_id) * part_size
    if block_id == 32:
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
    for imidx, (_, r) in tqdm.tqdm(enumerate(df.iterrows()), total=len(df)):
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

    fn = ('data/working/exp12/'
          f'train19_train19_faiss_search_same_landmarks_blk{block_id}.csv.gz')
    Path(fn).parent.mkdir(parents=True, exist_ok=True)

    print('to_csv')
    df = pd.DataFrame(rows).to_csv(fn, index=False,
                                   compression='gzip')


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


def __prep_delf1905_resume(block_id):
    fn = ('data/working/exp12/'
          f'train19_train19_faiss_search_same_landmarks_blk{block_id}.csv.gz')
    df = pd.read_csv(fn)

    last_image_id = ''
    fn_out = f'data/working/exp12/train19_train19_verified_blk{block_id}.csv'

    with open(fn_out, 'r') as f:
        for line in f:
            last_image_id = line.split('\t')[0]

    is_new = False

    p = multiprocessing.Pool(multiprocessing.cpu_count())
    with open(fn_out, 'a') as f:
        map_args = []
        for _, r in tqdm.tqdm(df.iterrows(), total=len(df)):
            id_ = r['id']
            if not is_new:
                if last_image_id == id_:
                    is_new = True
                continue

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
                            f'data/input/delf_190505/train19/{id_}.npz'),
                          rhs=load_desc_loc(
                            f'data/input/delf_190505/train19/{imid}.npz'),
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


def __prep_delf1905_results(block_id):
    fn_out = f'data/working/exp12/train19_train19_verified_blk{block_id}.csv'
    if Path(fn_out).exists():
        __prep_delf1905_resume(block_id)
    else:
        __prep_delf1905_newfile(block_id)


def __prep_delf1905_newfile(block_id):
    fn = ('data/working/exp12/'
          f'train19_train19_faiss_search_same_landmarks_blk{block_id}.csv.gz')
    df = pd.read_csv(fn)

    fn_out = f'data/working/exp12/train19_train19_verified_blk{block_id}.csv'
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
                            f'data/input/delf_190505/train19/{id_}.npz'),
                          rhs=load_desc_loc(
                            f'data/input/delf_190505/train19/{imid}.npz'),
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


def __prep_test_filepath_list():
    with open('data/working/test_exists.lst', 'r') as f,\
            open('data/working/exp12/test_filepath.lst', 'w') as fw:
        for line in f:
            id_ = line.strip()
            fw.write(f'data/input/test_r1024/{id_}.jpg\n')


def __search(dataset, fn_out, dba_niters=2, qe_topk=10):
    Path(fn_out).parent.mkdir(parents=True, exist_ok=True)

    feats_test_and_train = np.concatenate([
        dataset.feats_test,
        dataset.feats_train], axis=0)

    for i in range(dba_niters):
        print(f'[iter-{i}] start DBA...')

        with timer('Prep faiss index'):
            cpu_train = faiss.IndexFlatL2(feats_test_and_train.shape[1])
            gpu_train = faiss.index_cpu_to_all_gpus(cpu_train)
            gpu_train.add(feats_test_and_train)

        with timer('Search'):
            dists, topk_idx = gpu_train.search(x=feats_test_and_train,
                                               k=qe_topk)

        with timer('Weighting'):
            weights = np.logspace(
                0, -1.5, qe_topk).reshape(1, qe_topk, 1).astype(np.float32)
            feats_test_and_train = (
                feats_test_and_train[topk_idx] * weights).sum(axis=1)

    with timer('l2norm'):
        feats_test_and_train = l2norm_numpy(
            feats_test_and_train.astype(np.float32))

    with timer('Save search results'):
        feats_test, feats_train = np.split(feats_test_and_train,
                                           [len(dataset.feats_test)], axis=0)

        with h5py.File(fn_out, 'a') as f:
            f.create_dataset('feats_train', data=feats_train)
            f.create_dataset('feats_test', data=feats_test)
            f.create_dataset('ids_train',
                             data=dataset.ids_train.astype(dtype='S16'))
            f.create_dataset('ids_test',
                             data=dataset.ids_test.astype(dtype='S16'))


def __scoring_with_top100_arcfacefish_v1():
    dataset = loader.load_train_dataset()

    with timer('Loading train19 landmark dict'):
        landmark_dict = load_train19_landmark_dict()

    fn_sub = 'data/working/exp12/v7_fish_dba2qe10_train19_v1.csv.gz'

    cpu_index = faiss.IndexFlatL2(dataset.feats_train.shape[1])
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    gpu_index.add(dataset.feats_train)
    dists, topk_idx = gpu_index.search(x=dataset.feats_test, k=100)

    df = pd.DataFrame(dataset.ids_test, columns=['id'])
    df['images'] = np.apply_along_axis(
        ' '.join, axis=1, arr=dataset.ids_train[topk_idx])

    print('generate sub')
    rows = []
    max_value = sum([np.exp(np.sqrt(i)) for i in range(30)])
    for _, r in tqdm.tqdm(df.iterrows(), total=len(df)):
        image_ids = [name.split('/')[-1] for name in r.images.split(' ')]
        counter = Counter()
        for i, image_id in enumerate(image_ids[:30]):
            landmark_id = landmark_dict[image_id]
            # counter[landmark_id] += (2.0 - x) * 0.001
            counter[landmark_id] += np.exp(-np.sqrt(i + 1))
        landmark_id, score = counter.most_common(1)[0]
        score = score / max_value
        rows.append({
            'id': r['id'],
            'landmarks': f'{landmark_id} {score:.9f}',
        })

    print('to_csv')
    df = pd.DataFrame(rows)
    df_sub = pd.read_csv('data/recognition_sample_submission.csv')
    df_sub = df_sub[['id']].merge(df, how='left', on='id')
    df_sub[['id', 'landmarks']].to_csv(fn_sub,
                                       index=False,
                                       compression='gzip')


def __scoring_with_top100_arcfacefish_v2():
    dataset = loader.load_train_dataset()

    fn_out = 'data/working/exp12/v7_fish_dba2qe10.h5'
    if not Path(fn_out).exists():
        __search(dataset, fn_out, dba_niters=2, qe_topk=10)
    dataset = loader.load_train_dataset_singlefile(fn_out)

    with timer('Loading train19 landmark dict'):
        landmark_dict = load_train19_landmark_dict()

    fn_sub = 'data/working/exp12/v7_fish_dba2qe10_train19_v2.csv.gz'

    cpu_index = faiss.IndexFlatL2(dataset.feats_train.shape[1])
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    gpu_index.add(dataset.feats_train)
    dists, topk_idx = gpu_index.search(x=dataset.feats_test, k=100)

    df = pd.DataFrame(dataset.ids_test, columns=['id'])
    df['images'] = np.apply_along_axis(
        ' '.join, axis=1, arr=dataset.ids_train[topk_idx])

    print('generate sub')
    rows = []
    max_value = sum([np.exp(np.sqrt(i)) for i in range(30)])
    for _, r in tqdm.tqdm(df.iterrows(), total=len(df)):
        image_ids = [name.split('/')[-1] for name in r.images.split(' ')]
        counter = Counter()
        for i, image_id in enumerate(image_ids[:30]):
            landmark_id = landmark_dict[image_id]
            counter[landmark_id] += np.exp(-np.sqrt(i + 1))
        landmark_id, score = counter.most_common(1)[0]
        score = score / max_value
        rows.append({
            'id': r['id'],
            'landmarks': f'{landmark_id} {score:.9f}',
        })

    print('to_csv')
    df = pd.DataFrame(rows)
    df_sub = pd.read_csv('data/recognition_sample_submission.csv')
    df_sub = df_sub[['id']].merge(df, how='left', on='id')
    df_sub[['id', 'landmarks']].to_csv(fn_sub,
                                       index=False,
                                       compression='gzip')


def __scoring_with_top100_arcfacefish_v3():
    dataset = loader.load_train_dataset()

    with timer('Loading train19 landmark dict'):
        landmark_dict = load_train19_landmark_dict()

    fn_sub = 'data/working/exp12/v7_fish_nodba_top20_train19_v3.csv.gz'

    cpu_index = faiss.IndexFlatL2(dataset.feats_train.shape[1])
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    gpu_index.add(dataset.feats_train)
    dists, topk_idx = gpu_index.search(x=dataset.feats_test, k=100)

    df = pd.DataFrame(dataset.ids_test, columns=['id'])
    df['images'] = np.apply_along_axis(
        ' '.join, axis=1, arr=dataset.ids_train[topk_idx])

    print('generate sub')
    rows = []
    max_value = sum([np.exp(np.sqrt(i)) for i in range(20)])
    for _, r in tqdm.tqdm(df.iterrows(), total=len(df)):
        image_ids = [name.split('/')[-1] for name in r.images.split(' ')]
        counter = Counter()
        for i, image_id in enumerate(image_ids[:20]):
            landmark_id = landmark_dict[image_id]
            counter[landmark_id] += np.exp(-np.sqrt(i + 1))
        landmark_id, score = counter.most_common(1)[0]
        score = score / max_value
        rows.append({
            'id': r['id'],
            'landmarks': f'{landmark_id} {score:.9f}',
        })

    print('to_csv')
    df = pd.DataFrame(rows)
    df_sub = pd.read_csv('data/recognition_sample_submission.csv')
    df_sub = df_sub[['id']].merge(df, how='left', on='id')
    df_sub[['id', 'landmarks']].to_csv(fn_sub,
                                       index=False,
                                       compression='gzip')


def __scoring_with_top100_arcfacefish_v4():
    dataset = loader.load_train_dataset()

    fn_out = 'data/working/exp12/v7_fish_dba2qe10.h5'
    if not Path(fn_out).exists():
        __search(dataset, fn_out, dba_niters=2, qe_topk=10)
    dataset = loader.load_train_dataset_singlefile(fn_out)

    with timer('Loading train19 landmark dict'):
        landmark_dict = load_train19_landmark_dict()

    fn_sub = 'data/working/exp12/v7_fish_nodba_top40_train19_v4.csv.gz'

    cpu_index = faiss.IndexFlatL2(dataset.feats_train.shape[1])
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    gpu_index.add(dataset.feats_train)
    dists, topk_idx = gpu_index.search(x=dataset.feats_test, k=100)

    df = pd.DataFrame(dataset.ids_test, columns=['id'])
    df['images'] = np.apply_along_axis(
        ' '.join, axis=1, arr=dataset.ids_train[topk_idx])

    print('generate sub')
    rows = []
    max_value = sum([np.exp(np.sqrt(i)) for i in range(40)])
    for _, r in tqdm.tqdm(df.iterrows(), total=len(df)):
        image_ids = [name.split('/')[-1] for name in r.images.split(' ')]
        counter = Counter()
        for i, image_id in enumerate(image_ids[:40]):
            landmark_id = landmark_dict[image_id]
            counter[landmark_id] += np.exp(-np.sqrt(i + 1))
        landmark_id, score = counter.most_common(1)[0]
        score = score / max_value
        rows.append({
            'id': r['id'],
            'landmarks': f'{landmark_id} {score:.9f}',
        })

    print('to_csv')
    df = pd.DataFrame(rows)
    df_sub = pd.read_csv('data/recognition_sample_submission.csv')
    df_sub = df_sub[['id']].merge(df, how='left', on='id')
    df_sub[['id', 'landmarks']].to_csv(fn_sub,
                                       index=False,
                                       compression='gzip')


if __name__ == '__main__':
    cli()
