import os
from collections import Counter
import faiss
import tqdm
import numpy as np
import pandas as pd
import subprocess
from src import utils

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'


def predict_landmark_id(ids_query, feats_query, ids_train, feats_train, landmark_dict, topk=3):
    print('build index...')
    cpu_index = faiss.IndexFlatL2(feats_train.shape[1])
    cpu_index.add(feats_train)
    dists, topk_idx = cpu_index.search(x=feats_query, k=topk)
    print('query search done.')

    df = pd.DataFrame(ids_query, columns=['id'])
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

    pred = pd.DataFrame(rows)
    pred['landmark_id'], pred['score'] = list(
        zip(*pred['landmarks'].apply(lambda x: str(x).split(' '))))
    pred['score'] = pred['score'].astype(np.float32)
    pred = pred.sort_values('score', ascending=False).set_index('id')

    return pred


def reranking_submission(ids_index, feats_index,
                         ids_test, feats_test,
                         ids_train, feats_train,
                         subm, score_thresh=0.0, topk=100):

    train19_csv = pd.read_pickle(ROOT + 'input/train.pkl')[['id', 'landmark_id']]
    landmark_dict = train19_csv.set_index('id').sort_index().to_dict()['landmark_id']

    pred_index = predict_landmark_id(ids_index, feats_index, ids_train, feats_train, landmark_dict)
    pred_test = predict_landmark_id(ids_test, feats_test, ids_train, feats_train, landmark_dict)

    images = []
    for test_id, pred, ids in tqdm.tqdm(zip(subm['id'], subm['test_pred_lid'], subm['index_id_list']),
                                        total=len(subm)):
        ids = ids[:topk]
        retrieved_list = pred_index.loc[ids, 'landmark_id']

        if pred_test.loc[test_id, 'score'] > score_thresh:
            whole_ids_same = pred_index[
                (pred_index['landmark_id'] == pred) & (pred_index['score'] > score_thresh)].index
            similar_ids_same = pred_index.loc[ids][(pred == pred_index.loc[ids, 'landmark_id'])].index
            diff = set(whole_ids_same) - set(similar_ids_same)
            retrieved_list = pd.concat([
                retrieved_list,
                pd.Series(pred, index=diff)
            ])

        # use mergesort to keep relative order of original list.
        predefined_limit_topk = 100
        reranked_ids = (pred != retrieved_list).sort_values(kind='mergesort').index[:predefined_limit_topk]
        images.append(' '.join(reranked_ids))

    subm['images'] = images

    return subm


def main():
    index_dirs = [
        ROOT + 'experiments/v19c/feats_index19_ms_L2_ep4_scaleup_ep3_freqthresh-2_loss-cosface_pooling-G,G,G,G_verifythresh-30/',
        ROOT + 'experiments/v20c/feats_index19_ms_L2_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30/',
        ROOT + 'experiments/v21c/feats_index19_ms_L2_ep6_scaleup_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30/',
        ROOT + 'experiments/v22c/feats_index19_ms_L2_ep4_scaleup_ep3_base_margin-0.4_freqthresh-2_verifythresh-30/',
        ROOT + 'experiments/v23c/feats_index19_ms_L2_ep6_scaleup_ep5_augmentation-middle_epochs-7_freqthresh-3_verifythresh-30/',
        ROOT + 'experiments/v24c/feats_index19_ms_L2_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-cosface_verifythresh-30/',
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

    ids_index, feats_index = utils.prepare_ids_and_feats(index_dirs, weights, normalize=True)
    ids_test, feats_test = utils.prepare_ids_and_feats(test_dirs, weights, normalize=True)
    ids_train, feats_train = utils.prepare_ids_and_feats(train_dirs, weights, normalize=True)

    print('build index...')
    cpu_index = faiss.IndexFlatL2(feats_index.shape[1])
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    gpu_index.add(feats_index)
    dists, topk_idx = gpu_index.search(x=feats_test, k=100)
    print('query search done.')

    subm = pd.DataFrame(ids_test, columns=['id'])
    subm['images'] = np.apply_along_axis(' '.join, axis=1, arr=ids_index[topk_idx])

    subm = reranking_submission(ids_index, feats_index,
                                ids_test, feats_test,
                                ids_train, feats_train,
                                subm, score_thresh=0.0, topk=100)

    output_name = ROOT + f'output/submit_retrieval.csv.gz'
    subm.to_csv(output_name, compression='gzip', index=False)
    print('saved to ' + output_name)

    cmd = f'kaggle c submit -c landmark-retrieval-2019 -f {output_name} -m "" '
    print(cmd)
    subprocess.run(cmd, shell=True)


if __name__ == '__main__':
    main()
