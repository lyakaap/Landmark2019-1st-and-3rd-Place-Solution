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

    np.save('data/working/exp44/dists_top5__ens3_DBAx1.npy', dists)
    np.save('data/working/exp44/topk_idx_top5__ens3_DBAx1.npy', topk_idx)

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


def main(skip_nn_seach=False):
    train19_csv = pd.read_csv('/fs2/groups2/gca50080/sp4/cvpr19_landmark/train/train.csv',
                              usecols=['id', 'landmark_id'], index_col=0).sort_index()
    landmark_dict = train19_csv.to_dict()['landmark_id']

    topk = 3

    fn_sub = f'data/working/exp44/stage2_submit_ens3_top{topk}_nodba.csv.gz'
    Path(fn_sub).parent.mkdir(parents=True, exist_ok=True)

    if skip_nn_seach:  # 計算済みの中間ファイルを使う
        dists = np.load('data/working/exp44/dists_top5__ens3_DBAx1.npy')
        topk_idx = np.load('data/working/exp44/topk_idx_top5__ens3_DBAx1.npy')
        df_test = pd.read_csv('/fs2/groups2/gca50080/sp4/cvpr19_landmark/test/test.csv', usecols=[0])
        ids_test = df_test['id'].sort_values().values
        ids_train = train19_csv.index.values
        df = pd.DataFrame(ids_test, columns=['id'])
        df['images'] = np.apply_along_axis(' '.join, axis=1, arr=ids_train[topk_idx])
    else:
        df, dists = get_df_and_dists(train19_csv=train19_csv,
                                     topk=topk,
                                     n_iter=0,
                                     qe_topk=3,
                                     weighting_scheme='alpha',
                                     alpha=3,
                                     verifythresh=40,
                                     freqthresh=5,
                                     )

    ransac_match_dict = {}
    with open('data/working/test19_train19_search_top100_RANSAC.csv', 'r') as f:
        f.readline()
        for line in f:
            elems = line.split(',')
            ransac_match_dict[elems[0] + elems[1]] = int(elems[2])

    logger.info('generate sub')
    rows = []
    for imidx, (_, r) in tqdm.tqdm(enumerate(df.iterrows()), total=len(df)):
        image_ids = [name.split('/')[-1] for name in r.images.split(' ')]
        counter = Counter()
        for i, image_id in enumerate(image_ids[:topk]):
            landmark_id = landmark_dict[image_id]

            counter[landmark_id] += 1.0 - dists[imidx, i]
            if (r['id'] + image_id) in ransac_match_dict:
                cnt = ransac_match_dict[r['id'] + image_id]
                counter[landmark_id] += (min(70, cnt) / 70.0)

        landmark_id, score = counter.most_common(1)[0]
        rows.append({
            'id': r['id'],
            'landmarks': f'{landmark_id} {score:.9f}',
        })

    logger.info('to_csv')
    pd.DataFrame(rows).to_csv(fn_sub, index=False, compression='gzip')

    ban_thresh = 30
    ban_distractor_lids(sub_filename=fn_sub,
                        output_filename=f'data/working/exp44/stage2_submit_banthresh{ban_thresh}_ens3_top{topk}_nodba.csv.gz',
                        ban_thresh=ban_thresh,
                        )
    # !kaggle c submit -c landmark-recognition-2019 -m '' -f /fs2/groups2/gca50080/yokoo/landmark/output/


if __name__ == '__main__':
    main(skip_nn_seach=True)
