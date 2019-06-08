import argparse
import faiss
import numpy as np
import pandas as pd
import tqdm
from collections import Counter
import os
import subprocess
from src import utils


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
        print(f'[iter-{i}] start DBA...')
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


topk = 100
ROOT = '/fs2/groups2/gca50080/yokoo/landmark/'

index_dirs = [
    ROOT + 'experiments/v19c/feats_index19_ms_L2_ep4_scaleup_ep3_freqthresh-2_loss-cosface_pooling-G,G,G,G_verifythresh-30/',  # 0.29039
    ROOT + 'experiments/v20c/feats_index19_ms_L2_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30/',  # 0.29601
    ROOT + 'experiments/v21c/feats_index19_ms_L2_ep6_scaleup_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30/',  # 0.28569
    ROOT + 'experiments/v22c/feats_index19_ms_L2_ep4_scaleup_ep3_base_margin-0.4_freqthresh-2_verifythresh-30/',  # 0.28660
    ROOT + 'experiments/v23c/feats_index19_ms_L2_ep6_scaleup_ep5_augmentation-middle_epochs-7_freqthresh-3_verifythresh-30/',  # 0.29168
    ROOT + 'experiments/v24c/feats_index19_ms_L2_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-cosface_verifythresh-30/',  # 0.29422
]
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
]  # intuition

# （アンサンブル）concatしたあとはnormalizeせずそのまま
ids_index, feats_index = utils.prepare_ids_and_feats(index_dirs, weights, normalize=True)
ids_test, feats_test = utils.prepare_ids_and_feats(test_dirs, weights, normalize=True)
ids_train, feats_train = utils.prepare_ids_and_feats(train_dirs, weights, normalize=True)

train19_csv = pd.read_pickle('/fs2/groups2/gca50080/yokoo/landmark/input/train.pkl')[['id', 'landmark_id']].set_index('id').sort_index()
landmark_dict = train19_csv.to_dict()['landmark_id']

feats_train = supervised_dba(ids_train=ids_train,
                             feats_train=feats_train,
                             train19_csv=train19_csv,
                             n_iter=1,
                             qe_topk=3,
                             weighting_scheme='alpha',
                             alpha=3,
                             t=0.8,
                             verifythresh=40,
                             freqthresh=5,
                             )

# test-train
print('build index...')
cpu_index = faiss.IndexFlatL2(feats_train.shape[1])
cpu_index.add(feats_train)
dists, topk_idx = cpu_index.search(x=feats_test, k=3)
print('query search done.')

df = pd.DataFrame(ids_test, columns=['id'])
df['images'] = np.apply_along_axis(' '.join, axis=1, arr=ids_train[topk_idx])

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

test_train = pd.DataFrame(rows)
test_train['landmark_id'], test_train['score'] = list(zip(*test_train['landmarks'].apply(lambda x: str(x).split(' '))))
test_train['score'] = test_train['score'].astype(np.float32)
test_train = test_train.sort_values('score', ascending=False).set_index('id')
test_train.to_csv('/fs2/groups2/gca50080/yokoo/landmark/output/test_train.csv')

# index-train
print('build index...')
cpu_index = faiss.IndexFlatL2(feats_train.shape[1])
cpu_index.add(feats_train)
dists, topk_idx = cpu_index.search(x=feats_index, k=3)
print('query search done.')

df = pd.DataFrame(ids_index, columns=['id'])
df['images'] = np.apply_along_axis(' '.join, axis=1, arr=ids_train[topk_idx])

print('generate sub')
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

index_train = pd.DataFrame(rows)
index_train['landmark_id'], index_train['score'] = list(
    zip(*index_train['landmarks'].apply(lambda x: str(x).split(' '))))
index_train['score'] = index_train['score'].astype(np.float32)
index_train = index_train.sort_values('score', ascending=False).set_index('id')

index_train.to_csv('/fs2/groups2/gca50080/yokoo/landmark/output/index_train.csv')


print('build index...')
cpu_index = faiss.IndexFlatL2(feats_index.shape[1])
cpu_index.add(feats_index)
dists, topk_idx = cpu_index.search(x=feats_test, k=300)
print('query search done.')

np.save('/fs2/groups2/gca50080/yokoo/landmark/output/testindex_dists_top300_ens3.npy', dists)
np.save('/fs2/groups2/gca50080/yokoo/landmark/output/testindex_topk_idx_top300_ens3.npy', topk_idx)


df = pd.DataFrame(ids_test, columns=['id'])
df['images'] = np.apply_along_axis(' '.join, axis=1, arr=ids_index[topk_idx])


# subm = pd.read_csv('/fs2/groups2/gca50080/yokoo/landmark/output/stg2e45_ens0602w1_delfdbaqe_norerank.csv.gz')
subm = pd.read_csv('/fs2/groups2/gca50080/sp4/working/exp45/stg2e45_ens0602w1_delfdbaqe_norerank_topk1000.csv.gz')
test_train = pd.read_csv('/fs2/groups2/gca50080/yokoo/landmark/output/test_train.csv', index_col=0)
index_train = pd.read_csv('/fs2/groups2/gca50080/yokoo/landmark/output/index_train.csv', index_col=0)

subm['test_pred_lid'] = test_train.loc[subm['id'], 'landmark_id'].values
subm['index_id_list'] = subm['images'].str.split(' ')


"""
1. test-train, index-trainの近傍探索をする
2. testとindexのそれぞれの画像に対してrecognitionと同じようにsoft-votingなKNNを行う
3. 普通にtest-indexでtop100までretrievalする
4. test画像の分類結果と、retrievalしたtop100件のindex画像の分類結果を比較し、同一ラベルであれば検索結果の前の方に来るようにする（同一ラベル内での順序関係は維持したまま、全ての同一ラベルをそれ以外のindex画像よりも前に持ってくる）
5. 「top100件に出現してないけれども分類結果がtest画像と一致している」index画像を、4の「top100件のうち分類結果がtest画像と一致している」index画像の後ろに挿入する。
6. 5の操作で検索結果が100件以上にはみ出るので、top100以降をカットする
"""
score_thresh = 0.0
topk = 100

images = []
for test_id, pred, ids in tqdm.tqdm(zip(subm['id'], subm['test_pred_lid'], subm['index_id_list']), total=len(subm)):
    ids = ids[:topk]
    retrieved_list = index_train.loc[ids, 'landmark_id']

    if test_train.loc[test_id, 'score'] > score_thresh:
        whole_ids_same = index_train[
            (index_train['landmark_id'] == pred) & (index_train['score'] > score_thresh)].index
        similar_ids_same = index_train.loc[ids][(pred == index_train.loc[ids, 'landmark_id'])].index
        diff = set(whole_ids_same) - set(similar_ids_same)
        retrieved_list = pd.concat([
            retrieved_list,
            pd.Series(pred, index=diff)
        ])

    # use mergesort to keep relative order of original list.
    reranked_ids = (pred != retrieved_list).sort_values(kind='mergesort').index[:100]
    images.append(' '.join(reranked_ids))

subm['images'] = images

output_name = f'/fs2/groups2/gca50080/yokoo/landmark/output/stg2e45_rerank100_scorethresh0.0_ens0602w1_delfdbaqe_norerank.csv.gz'
subm[['id', 'images']].to_csv(output_name, compression='gzip', index=False)
