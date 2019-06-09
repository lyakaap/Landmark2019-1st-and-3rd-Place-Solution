from logging import getLogger, Formatter, StreamHandler, INFO
import subprocess
import time
import gzip

import tqdm
import click
import pandas as pd
import numpy as np

from util import timer
import ensloader
import loader
import euclidsearch


logger = getLogger('landmark')


@click.group()
def cli():
    LOGFORMAT = '%(asctime)s %(levelname)s %(message)s'
    handler = StreamHandler()
    handler.setLevel(INFO)
    handler.setFormatter(Formatter(LOGFORMAT))
    logger.setLevel(INFO)
    logger.addHandler(handler)


@cli.command()
def stg2ens_0602w1_delfdba_top1000():
    fn_dba = 'data/working/exp45/stg2e45_ens0602w1_delfdbaqe_norerank_topk1000.h5'
    fn_npy = 'data/working/exp45/stg2e45_ens0602w1_delfdbaqe_norerank_topk1000.npy'

    with timer('Load ensemble descriptors'):
        ds = ensloader.load_desc('modelset_v0602',
                                 'modelset_v0602_weight_v1',
                                 mode='retrieval')
        ds_trn = ensloader.load_desc('modelset_v0602',
                                     'modelset_v0602_weight_v1',
                                     mode='retrieval_dbatrain')

    with timer('DELF-DBAQE'):
        # alpha DBA(index+test)
        ds = delfdbaqe(ds, ds_trn, qe_topk=5, thresh=90)
        loader.save_index_dataset(fn_dba, ds)

    with timer('Generate submission file'):
        euclidsearch.gpux4_euclidsearch_from_dataset(ds, fn_npy, topk=1000)


@cli.command()
def stg2ens_0602w1_delfdba_top300():
    fn_dba = 'data/working/exp45/stg2e45_ens0602w1_delfdbaqe_norerank_topk300.h5'
    fn_npy = 'data/working/exp45/stg2e45_ens0602w1_delfdbaqe_norerank_topk300.npy'

    with timer('Load ensemble descriptors'):
        ds = ensloader.load_desc('modelset_v0602',
                                 'modelset_v0602_weight_v1',
                                 mode='retrieval')
        ds_trn = ensloader.load_desc('modelset_v0602',
                                     'modelset_v0602_weight_v1',
                                     mode='retrieval_dbatrain')

    with timer('DELF-DBAQE'):
        # alpha DBA(index+test)
        ds = delfdbaqe(ds, ds_trn, qe_topk=5, thresh=90)
        loader.save_index_dataset(fn_dba, ds)

    with timer('Generate submission file'):
        euclidsearch.gpux4_euclidsearch_from_dataset(ds, fn_npy, topk=300)


@cli.command()
def stg2ens_0602w1_delfdba70():
    """
    3349/3349
    225861/225861
    """
    fn_dba = 'data/working/exp45/stg2e45_ens0602w1_delfdbaqe70_norerank.h5'
    fn_npy = 'data/working/exp45/stg2e45_ens0602w1_delfdbaqe70_norerank.npy'

    with timer('Load ensemble descriptors'):
        ds = ensloader.load_desc('modelset_v0602',
                                 'modelset_v0602_weight_v1',
                                 mode='retrieval')
        ds_trn = ensloader.load_desc('modelset_v0602',
                                     'modelset_v0602_weight_v1',
                                     mode='retrieval_dbatrain')

    with timer('DELF-DBAQE'):
        # alpha DBA(index+test)
        ds = delfdbaqe(ds, ds_trn, qe_topk=5, thresh=70)
        loader.save_index_dataset(fn_dba, ds)

    with timer('Generate submission file'):
        euclidsearch.gpux4_euclidsearch_from_dataset(ds, fn_npy)


@cli.command()
def stg2ens_0602w1_delfdba120():
    """
    """
    fn_dba = 'data/working/exp45/stg2e45_ens0602w1_delfdbaqe120_norerank.h5'
    fn_npy = 'data/working/exp45/stg2e45_ens0602w1_delfdbaqe120_norerank.npy'

    with timer('Load ensemble descriptors'):
        ds = ensloader.load_desc('modelset_v0602',
                                 'modelset_v0602_weight_v1',
                                 mode='retrieval')
        ds_trn = ensloader.load_desc('modelset_v0602',
                                     'modelset_v0602_weight_v1',
                                     mode='retrieval_dbatrain')

    with timer('DELF-DBAQE'):
        # alpha DBA(index+test)
        ds = delfdbaqe(ds, ds_trn, qe_topk=5, thresh=120)
        loader.save_index_dataset(fn_dba, ds)

    with timer('Generate submission file'):
        euclidsearch.gpux4_euclidsearch_from_dataset(ds, fn_npy)


@cli.command()
def stg2ens_0602w1_delfdba():
    fn_dba = 'data/working/exp45/stg2e45_ens0602w1_delfdbaqe_norerank.h5'
    fn_npy = 'data/working/exp45/stg2e45_ens0602w1_delfdbaqe_norerank.npy'

    with timer('Load ensemble descriptors'):
        ds = ensloader.load_desc('modelset_v0602',
                                 'modelset_v0602_weight_v1',
                                 mode='retrieval')
        ds_trn = ensloader.load_desc('modelset_v0602',
                                     'modelset_v0602_weight_v1',
                                     mode='retrieval_dbatrain')

    with timer('DELF-DBAQE'):
        # alpha DBA(index+test)
        ds = delfdbaqe(ds, ds_trn, qe_topk=5, thresh=90)
        loader.save_index_dataset(fn_dba, ds)

    with timer('Generate submission file'):
        euclidsearch.gpux4_euclidsearch_from_dataset(ds, fn_npy)


def delfdbaqe(ds, ds_trn, qe_topk=5, thresh=90):
    print(ds.ids_test.shape, ds.ids_index.shape)
    test_id2idx_map = {id_: idx for idx, id_ in enumerate(ds.ids_test)}
    index_id2idx_map = {id_: idx for idx, id_ in enumerate(ds.ids_index)}
    train_id2idx_map = {id_: idx for idx, id_ in enumerate(ds_trn.ids_train)}

    df1 = pd.read_csv('data/working/test19_train19_search_top100_RANSAC.csv')
    df1 = df1[df1.inliers_count >= thresh]
    df2 = pd.read_csv('data/working/test19_index19_search_top100_RANSAC.csv')
    df2 = df2[df2.inliers_count >= thresh]

    df = pd.concat([df1, df2], sort=False)
    df = df.sort_values(by='inliers_count', ascending=False)
    df = df.groupby('test_id').agg({
        'index_id': lambda x: ' '.join(x[:qe_topk])})

    # cnt = 0
    for test_id, r in tqdm.tqdm(df.iterrows(), total=len(df)):
        expand_descs = []
        expand_weights = []

        # print('test_id', test_id)
        # test_id2idx_map
        desc_testidx = test_id2idx_map[test_id]
        desc = ds.feats_test[desc_testidx]
        expand_descs.append(desc)
        expand_weights.append(1.0)

        for imid in r['index_id'].split(' '):
            if imid in index_id2idx_map:
                # print('index_id, index', imid)
                # index_id2idx_map
                desc_indexidx = index_id2idx_map[imid]
                desc = ds.feats_index[desc_indexidx]
                expand_descs.append(desc)

                w = expand_descs[0].dot(desc)
                expand_weights.append(w)

            else:
                print('index_id, train', imid)
                # train_id2idx_map
                desc_trainidx = train_id2idx_map[imid]
                desc = ds_trn.feats_train[desc_trainidx]
                expand_descs.append(desc)

                w = expand_descs[0].dot(desc)
                expand_weights.append(w)

        # Update test descriptor
        # print(expand_weights)
        desc = np.array(expand_weights).dot(
            np.array(expand_descs)) / np.array(expand_weights).sum()
        ds.feats_test[desc_testidx] = desc

        # cnt += 1
        # if cnt > 20:
        #     break

    # cnt = 0
    df = pd.read_csv('data/working/index19_index19_search_top100_RANSAC.csv')
    df = df[df.inliers_count >= thresh]
    df = df[df.test_id != df.index_id]
    df = df.sort_values(by='inliers_count', ascending=False)
    df = df.groupby('test_id').agg({
        'index_id': lambda x: ' '.join(x[:qe_topk])})

    for index_id, r in tqdm.tqdm(df.iterrows(), total=len(df)):
        # print(index_id, r, index_id2idx_map[index_id])
        # break
        expand_descs = []
        expand_weights = []

        # print('index_id', index_id)
        # test_id2idx_map
        desc_indexidx = index_id2idx_map[index_id]
        desc = ds.feats_index[desc_indexidx]
        expand_descs.append(desc)
        expand_weights.append(1.0)

        for imid in r['index_id'].split(' '):
            # print('index_id, index', imid)
            # index_id2idx_map
            desc_indexidx2 = index_id2idx_map[imid]
            desc2 = ds.feats_index[desc_indexidx2]
            expand_descs.append(desc2)

            w = expand_descs[0].dot(desc2)
            expand_weights.append(w)

        # Update test descriptor
        # print(expand_weights)
        desc = np.array(expand_weights).dot(
            np.array(expand_descs)) / np.array(expand_weights).sum()
        ds.feats_index[desc_indexidx] = desc

        # cnt += 1
        # if cnt > 20:
        #     break

    return ds


@cli.command()
def debug():
    pass


if __name__ == '__main__':
    cli()
