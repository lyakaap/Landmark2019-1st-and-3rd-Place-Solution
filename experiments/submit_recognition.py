from collections import Counter
from logging import getLogger
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import tqdm

from src import utils

LOGFORMAT = '%(asctime)s %(levelname)s %(message)s'
logger = getLogger('landmark')


def supervised_dba(ids_train, feats_train, train19_csv,
                   n_iter=1,
                   qe_topk=3,
                   weighting_scheme='alpha',
                   alpha=3,
                   t=0.8,
                   verifythresh=40,
                   freqthresh=5,
                   ):
    clean_csv_path = f'/fs2/groups2/gca50080/sp4/working/exp12/train19_cleaned_verifythresh{verifythresh}_freqthresh{freqthresh}.csv'
    clean_df = pd.read_csv(clean_csv_path)
    verified_ids = np.concatenate(clean_df['images'].str.split(' '))
    is_verified = np.isin(ids_train, verified_ids)
    feats_clean_train = feats_train[is_verified]

    lid_info = train19_csv.sort_index()[['landmark_id']].values.squeeze()[is_verified]

    for i in range(n_iter):
        logger.info(f'[iter-{i}] start DBA...')
        cpu_index = faiss.IndexFlatIP(feats_clean_train.shape[1])
        cpu_index.add(feats_clean_train)
        sims, topk_idx = cpu_index.search(x=feats_clean_train, k=qe_topk)

        if weighting_scheme == 'alpha':  # α-{QE/DBA}
            weights = np.expand_dims(sims ** alpha, axis=-1).astype(np.float32)
        elif weighting_scheme == 'trunc_poly':
            weights = np.expand_dims(np.where(sims > t, 1.0, (sims / t) ** alpha), axis=-1).astype(np.float32)
        else:
            weights = np.logspace(0, -1.5, qe_topk).reshape(1, qe_topk, 1).astype(np.float32)

        topk_lids = lid_info[topk_idx]
        mask = topk_lids == topk_lids[:, [0]]
        weights = weights * np.expand_dims(mask, axis=-1)

        feats_clean_train = (feats_clean_train[topk_idx] * weights).sum(axis=1)
        feats_clean_train = utils.l2norm_numpy(feats_clean_train.astype(np.float32))

    feats_train[is_verified] = feats_clean_train

    return feats_train


def get_df_and_dists(train19_csv,
                     topk=100,
                     n_iter=1,
                     qe_topk=3,
                     weighting_scheme='alpha',
                     alpha=3,
                     t=0.8,
                     verifythresh=40,
                     freqthresh=5,
                     ):
    ROOT = '/fs2/groups2/gca50080/yokoo/landmark/'
    test_dirs = [
        ROOT + 'experiments/v19c/feats_test19_ms_L2_ep4_scaleup_ep3_freqthresh-2_loss-cosface_pooling-G,G,G,G_verifythresh-30/',
        ROOT + 'experiments/v20c/feats_test19_ms_L2_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30/',
        ROOT + 'experiments/v21c/feats_test19_ms_L2_ep6_scaleup_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30/',
        ROOT + 'experiments/v22c/feats_test19_ms_L2_ep4_scaleup_ep3_base_margin-0.4_freqthresh-2_verifythresh-30/',
        ROOT + 'experiments/v23c/feats_test19_ms_L2_ep6_scaleup_ep5_augmentation-middle_epochs-7_freqthresh-3_verifythresh-30/',
        ROOT + 'experiments/v24c/feats_test19_ms_L2_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-cosface_verifythresh-30/',
    ]
    train_dirs = [
        ROOT + 'experiments/v19c/feats_train_ms_L2_ep4_scaleup_ep3_freqthresh-2_loss-cosface_pooling-G,G,G,G_verifythresh-30/',
        ROOT + 'experiments/v20c/feats_train_ms_L2_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30/',
        ROOT + 'experiments/v21c/feats_train_ms_L2_ep6_scaleup_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30/',
        ROOT + 'experiments/v22c/feats_train_ms_L2_ep4_scaleup_ep3_base_margin-0.4_freqthresh-2_verifythresh-30/',
        ROOT + 'experiments/v23c/feats_train_ms_L2_ep6_scaleup_ep5_augmentation-middle_epochs-7_freqthresh-3_verifythresh-30/',
        ROOT + 'experiments/v24c/feats_train_ms_L2_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-cosface_verifythresh-30/',
    ]
    weights = [
        0.5,
        1.0,
        1.0,
        0.5,
        1.0,
        1.0,
    ]

    logger.info('load ids and features.')
    ids_test, feats_test = utils.prepare_ids_and_feats(test_dirs, weights, normalize=True)
    ids_train, feats_train = utils.prepare_ids_and_feats(train_dirs, weights, normalize=True)
    logger.info('done.')

    if n_iter > 0:
        feats_train = supervised_dba(ids_train=ids_train,
                                     feats_train=feats_train,
                                     train19_csv=train19_csv,
                                     n_iter=n_iter,
                                     qe_topk=qe_topk,
                                     weighting_scheme=weighting_scheme,
                                     alpha=alpha,
                                     t=t,
                                     verifythresh=verifythresh,
                                     freqthresh=freqthresh
                                     )

    logger.info('build index...')
    cpu_index = faiss.IndexFlatL2(feats_train.shape[1])
    cpu_index.add(feats_train)
    dists, topk_idx = cpu_index.search(x=feats_test, k=topk)
    logger.info('query search done.')

    df = pd.DataFrame(ids_test, columns=['id'])
    df['images'] = np.apply_along_axis(' '.join, axis=1, arr=ids_train[topk_idx])

    np.save('/fs2/groups2/gca50080/yokoo/landmark/output/dists_top5__ens3_DBAx1.npy', dists)
    np.save('/fs2/groups2/gca50080/yokoo/landmark/output/topk_idx_top5___ens3_DBAx1.npy', topk_idx)

    return df, dists


def ban_distractor_lids(sub_filename, output_filename, ban_thresh=250):
    subm = pd.read_csv(sub_filename)

    subm['landmark_id'], subm['score'] = list(zip(*subm['landmarks'].apply(lambda x: str(x).split(' '))))
    subm['score'] = subm['score'].astype(np.float32)
    subm = subm.sort_values('score', ascending=False)

    pred_lid_count = subm.groupby('landmark_id')['score'].count().sort_values(ascending=False)
    pred_lid_count.index = pred_lid_count.index.astype(int)
    ban_list = pred_lid_count[pred_lid_count > ban_thresh]

    for key, val in pred_lid_count[pred_lid_count > ban_thresh].items():
        subm['landmarks'][subm['landmarks'].str.contains(f'{key} ')] = f'{key} -{val}'

    subm[['id', 'landmarks']].to_csv(output_filename, index=False, compression='gzip')


def ban_final():
    import argparse
    import faiss
    import numpy as np
    import pandas as pd
    import os
    import subprocess
    import tqdm
    from collections import Counter
    from src import utils

    topk = 100
    ROOT = '/opt/landmark/'

    test_dirs = [
        ROOT + 'experiments/v19c/feats_test19_ms_L2_ep4_scaleup_ep3_freqthresh-2_loss-cosface_pooling-G,G,G,G_verifythresh-30/',
        ROOT + 'experiments/v20c/feats_test19_ms_L2_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30/',
        ROOT + 'experiments/v21c/feats_test19_ms_L2_ep6_scaleup_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30/',
        ROOT + 'experiments/v22c/feats_test19_ms_L2_ep4_scaleup_ep3_base_margin-0.4_freqthresh-2_verifythresh-30/',
        ROOT + 'experiments/v23c/feats_test19_ms_L2_ep6_scaleup_ep5_augmentation-middle_epochs-7_freqthresh-3_verifythresh-30/',
        ROOT + 'experiments/v24c/feats_test19_ms_L2_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-cosface_verifythresh-30/',
    ]

    weights = [
        0.5,
        1.0,
        1.0,
        0.5,
        1.0,
        1.0,
    ]  # intuition

    ids_test, feats_test = utils.prepare_ids_and_feats(test_dirs, weights, normalize=True)

    # train19_csv = pd.read_pickle('../input/train.pkl')[['id', 'landmark_id']].set_index('id').sort_index()
    # landmark_dict = train19_csv.to_dict()['landmark_id']

    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    # co.float16 = False

    vres = []
    for _ in range(4):
        res = faiss.StandardGpuResources()
        vres.append(res)

    subm = pd.read_csv('../output/stage2_submit_banthresh30_ens3_top3_DBAx1_v44r7.csv.gz')
    subm['landmark_id'], subm['score'] = list(zip(*subm['landmarks'].apply(lambda x: str(x).split(' '))))
    subm['score'] = subm['score'].astype(np.float32)
    subm = subm.sort_values('score', ascending=False).set_index('id')

    ban_thresh = 30
    freq = subm['landmark_id'].value_counts()
    ban_lids = freq[freq > ban_thresh].index

    is_ban = np.isin(ids_test, subm[subm['landmark_id'].isin(ban_lids)].index)
    ban_ids_test = ids_test[is_ban]
    not_ban_ids_test = ids_test[~is_ban]
    ban_feats_test = feats_test[is_ban]
    not_ban_feats_test = feats_test[~is_ban]

    print('build index...')
    cpu_index = faiss.IndexFlatL2(not_ban_feats_test.shape[1])
    gpu_index = faiss.index_cpu_to_gpu_multiple_py(vres, cpu_index, co)
    gpu_index.add(not_ban_feats_test)
    dists, topk_idx = gpu_index.search(x=ban_feats_test, k=100)
    print('query search done.')

    subm = pd.read_csv('../output/stage2_submit_banthresh30_ens3_top3_DBAx1_v44r7.csv.gz')
    subm['landmark_id'], subm['score'] = list(zip(*subm['landmarks'].apply(lambda x: str(x).split(' '))))
    subm['score'] = subm['score'].astype(np.float32)
    subm = subm.sort_values('score', ascending=False).set_index('id')

    new_ban_ids = np.unique(not_ban_ids_test[topk_idx[dists < 0.5]])
    subm.loc[new_ban_ids, 'landmarks'] = subm.loc[new_ban_ids, 'landmark_id'] + ' 0'
    # subm.loc[new_ban_ids, 'landmarks'] = subm.loc[new_ban_ids, 'landmark_id'] + ' ' + (subm.loc[new_ban_ids, 'score'] * 0.001).map(str)

    output_filename = '../output/l2dist_0.5.csv.gz'
    subm.reset_index()[['id', 'landmarks']].to_csv(output_filename, index=False, compression='gzip')


def main(skip_nn_seach=False):
    train19_csv = pd.read_csv('/fs2/groups2/gca50080/sp4/cvpr19_landmark/train/train.csv',
                              usecols=['id', 'landmark_id'], index_col=0).sort_index()
    landmark_dict = train19_csv.to_dict()['landmark_id']

    topk = 3

    fn_sub = f'/fs2/groups2/gca50080/yokoo/landmark/output/stage2_submit_ens3_top{topk}_DBAx1.csv.gz'
    Path(fn_sub).parent.mkdir(parents=True, exist_ok=True)

    if skip_nn_seach:  # 計算済みの中間ファイルを使う
        dists = np.load('/fs2/groups2/gca50080/yokoo/landmark/output/dists_top5__ens3_DBAx1.npy')
        topk_idx = np.load('/fs2/groups2/gca50080/yokoo/landmark/output/topk_idx_top5__ens3_DBAx1.npy')
        df_test = pd.read_csv('/fs2/groups2/gca50080/sp4/cvpr19_landmark/test/test.csv', usecols=[0])
        ids_test = df_test['id'].sort_values().values
        ids_train = train19_csv.index.values
        df = pd.DataFrame(ids_test, columns=['id'])
        df['images'] = np.apply_along_axis(' '.join, axis=1, arr=ids_train[topk_idx])
    else:
        df, dists = get_df_and_dists(train19_csv=train19_csv,
                                     topk=topk,
                                     n_iter=1,
                                     qe_topk=3,
                                     weighting_scheme='alpha',
                                     alpha=3,
                                     verifythresh=40,
                                     freqthresh=5,
                                     )

    logger.info('generate sub')
    rows = []
    for imidx, (_, r) in tqdm.tqdm(enumerate(df.iterrows()), total=len(df)):
        image_ids = [name.split('/')[-1] for name in r.images.split(' ')]
        counter = Counter()
        for i, image_id in enumerate(image_ids[:topk]):
            landmark_id = landmark_dict[image_id]

            counter[landmark_id] += 1.0 - dists[imidx, i]

        landmark_id, score = counter.most_common(1)[0]
        rows.append({
            'id': r['id'],
            'landmarks': f'{landmark_id} {score:.9f}',
        })

    logger.info('to_csv')
    pd.DataFrame(rows).to_csv(fn_sub, index=False, compression='gzip')

    ban_thresh = 250
    ban_distractor_lids(sub_filename=fn_sub,
                        output_filename=f'/fs2/groups2/gca50080/yokoo/landmark/output/stage2_submit_banthresh{ban_thresh}_ens3_top{topk}_DBAx1.csv.gz',
                        ban_thresh=ban_thresh,
                        )
    # !kaggle c submit -c landmark-recognition-2019 -m '' -f /fs2/groups2/gca50080/yokoo/landmark/output/


"""does not work
    # for k-reciprocal NN
    feats_concat = np.concatenate([feats_test, feats_train], axis=0)
    cpu_index = faiss.IndexFlatL2(feats_concat.shape[1])
    cpu_index.add(feats_concat)
    krnn_dists, krnn_topk_idx = cpu_index.search(x=feats_concat, k=topk)
    krnn_dists = np.load('/fs2/groups2/gca50080/yokoo/landmark/output/krnn_dists_top5__uni_DBAx1.npy')
    krnn_topk_idx = np.load('/fs2/groups2/gca50080/yokoo/landmark/output/krnn_topk_idx_top5__uni_DBAx1.npy')
    fn_sub = f'/fs2/groups2/gca50080/yokoo/landmark/output/stage2_submit_uni_krnn_top{topk}_DBAx{n_iter}.csv.gz'
            score = 1.0 - dists[imidx, i]
            idx_on_krnn_mat = train19_csv.index.get_loc(image_id) + len(ids_test)
            if np.isin(imidx, krnn_topk_idx[idx_on_krnn_mat]):
                score = score * 1.5
            counter[landmark_id] += score
    
    lid_freq = train19_csv['landmark_id'].value_counts()
    lid_freq = lid_freq.where(lid_freq < topk, topk)
    max_dist = dists.max()
    counter[landmark_id] += (max_dist - dists[imidx, i]) / lid_freq.loc[landmark_id]
    
    ROOT = '/fs2/groups2/gca50080/yokoo/landmark/'
    nlc_results_train = pd.read_csv(ROOT + 'output/nlc_results_train.csv', index_col=0)
    df_recog19 = pd.read_pickle(ROOT + 'input/train.pkl').set_index('id')
    df_recog19 = df_recog19.merge(nlc_results_train, left_index=True, right_index=True)
    df_recog19 = df_recog19[['landmark_id', 'conf', 'pred_label', 'pred_landmark']]
    landmark_ratio = df_recog19.groupby('landmark_id')['pred_landmark'].agg(['count', 'mean'])
    fn_sub = f'/fs2/groups2/gca50080/yokoo/landmark/output/stage2_submit_uni_nl_online_top{topk}_DBAx{n_iter}.csv.gz'
    max_dist = dists.max()
"""