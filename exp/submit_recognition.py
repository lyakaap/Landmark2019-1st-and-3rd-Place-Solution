import os
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
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'


def get_df_and_dists(topk=100):
    test_dirs = [
        ROOT + 'exp/v19c/feats_test19_ms_L2_ep4_scaleup_ep3_freqthresh-2_loss-cosface_pooling-G,G,G,G_verifythresh-30/',
        ROOT + 'exp/v20c/feats_test19_ms_L2_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30/',
        ROOT + 'exp/v21c/feats_test19_ms_L2_ep6_scaleup_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30/',
        ROOT + 'exp/v22c/feats_test19_ms_L2_ep4_scaleup_ep3_base_margin-0.4_freqthresh-2_verifythresh-30/',
        ROOT + 'exp/v23c/feats_test19_ms_L2_ep6_scaleup_ep5_augmentation-middle_epochs-7_freqthresh-3_verifythresh-30/',
        ROOT + 'exp/v24c/feats_test19_ms_L2_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-cosface_verifythresh-30/',
    ]
    train_dirs = [
        ROOT + 'exp/v19c/feats_train_ms_L2_ep4_scaleup_ep3_freqthresh-2_loss-cosface_pooling-G,G,G,G_verifythresh-30/',
        ROOT + 'exp/v20c/feats_train_ms_L2_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30/',
        ROOT + 'exp/v21c/feats_train_ms_L2_ep6_scaleup_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30/',
        ROOT + 'exp/v22c/feats_train_ms_L2_ep4_scaleup_ep3_base_margin-0.4_freqthresh-2_verifythresh-30/',
        ROOT + 'exp/v23c/feats_train_ms_L2_ep6_scaleup_ep5_augmentation-middle_epochs-7_freqthresh-3_verifythresh-30/',
        ROOT + 'exp/v24c/feats_train_ms_L2_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-cosface_verifythresh-30/',
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

    logger.info('build index...')
    cpu_index = faiss.IndexFlatL2(feats_train.shape[1])
    cpu_index.add(feats_train)
    dists, topk_idx = cpu_index.search(x=feats_test, k=topk)
    logger.info('query search done.')

    df = pd.DataFrame(ids_test, columns=['id'])
    df['images'] = np.apply_along_axis(' '.join, axis=1, arr=ids_train[topk_idx])

    return df, dists


def ban_distractor_lids(sub_filename, output_filename, ban_thresh=250):
    subm = pd.read_csv(sub_filename)

    subm['landmark_id'], subm['score'] = list(zip(*subm['landmarks'].apply(lambda x: str(x).split(' '))))
    subm['score'] = subm['score'].astype(np.float32)
    subm = subm.sort_values('score', ascending=False)

    pred_lid_count = subm.groupby('landmark_id')['score'].count().sort_values(ascending=False)
    pred_lid_count.index = pred_lid_count.index.astype(int)

    for key, val in pred_lid_count[pred_lid_count > ban_thresh].items():
        subm['landmarks'][subm['landmarks'].str.contains(f'{key} ')] = f'{key} -{val}'

    subm[['id', 'landmarks']].to_csv(output_filename, index=False, compression='gzip')


def main():
    train19_csv = pd.read_csv(ROOT + 'input/train/train.csv',
                              usecols=['id', 'landmark_id'], index_col=0).sort_index()
    landmark_dict = train19_csv.to_dict()['landmark_id']

    topk = 3

    fn_sub = ROOT + f'output/submit_recognition.csv.gz'
    Path(fn_sub).parent.mkdir(parents=True, exist_ok=True)

    df, dists = get_df_and_dists(topk=topk)

    ransac_match_dict = {}
    with open(ROOT + 'input/test19_train19_search_top100_RANSAC.csv', 'r') as f:
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
                        output_filename=fn_sub,
                        ban_thresh=ban_thresh,
                        )


if __name__ == '__main__':
    main()
