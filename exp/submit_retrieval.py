import os
from collections import Counter
import faiss
import tqdm
import numpy as np
import pandas as pd
import subprocess
from src import utils

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'


def predict_landmark_id(ids_query, feats_query, ids_train, feats_train, landmark_dict, voting_k=3):
    print('build index...')
    cpu_index = faiss.IndexFlatIP(feats_train.shape[1])
    cpu_index.add(feats_train)
    sims, topk_idx = cpu_index.search(x=feats_query, k=voting_k)
    print('query search done.')

    df = pd.DataFrame(ids_query, columns=['id'])
    df['images'] = np.apply_along_axis(' '.join, axis=1, arr=ids_train[topk_idx])

    rows = []
    for imidx, (_, r) in tqdm.tqdm(enumerate(df.iterrows()), total=len(df)):
        image_ids = [name.split('/')[-1] for name in r.images.split(' ')]
        counter = Counter()
        for i, image_id in enumerate(image_ids[:voting_k]):
            landmark_id = landmark_dict[image_id]

            counter[landmark_id] += sims[imidx, i]

        landmark_id, score = counter.most_common(1)[0]
        rows.append({
            'id': r['id'],
            'landmarks': f'{landmark_id} {score:.9f}',
        })

    pred = pd.DataFrame(rows).set_index('id')
    pred['landmark_id'], pred['score'] = list(
        zip(*pred['landmarks'].apply(lambda x: str(x).split(' '))))
    pred['score'] = pred['score'].astype(np.float32) / voting_k

    return pred


def reranking_submission(ids_index, feats_index,
                         ids_test, feats_test,
                         ids_train, feats_train,
                         subm, 
                         topk=100, voting_k=3, thresh=0.6):

    train19_csv = pd.read_pickle(
        ROOT + 'input/train.pkl')[['id', 'landmark_id']]
    landmark_dict = train19_csv.set_index(
        'id').sort_index().to_dict()['landmark_id']

    pred_index = predict_landmark_id(
        ids_index, feats_index, ids_train, feats_train, landmark_dict, voting_k=voting_k)
    pred_test = predict_landmark_id(
        ids_test, feats_test, ids_train, feats_train, landmark_dict, voting_k=voting_k)

    assert np.all(subm['id'] == pred_test.index)
    subm['index_id_list'] = subm['images'].apply(lambda x: x.split(' ')[:topk])

    # to make higher score be inserted ealier position in insert-step
    pred_index = pred_index.sort_values('score', ascending=False)

    images = []
    for test_id, pred, ids in tqdm.tqdm(zip(subm['id'], pred_test['landmark_id'], subm['index_id_list']),
                                        total=len(subm)):
        retrieved_pred = pred_index.loc[ids, ['landmark_id', 'score']]

        # Sort-step
        is_positive: pd.Series = (pred == retrieved_pred['landmark_id'])
        # use mergesort to keep relative order of original list.
        sorted_retrieved_ids: list = (~is_positive).sort_values(
            kind='mergesort').index.to_list()

        # Insert-step
        whole_positives = pred_index[pred_index['landmark_id'] == pred]
        whole_positives = whole_positives[score + whole_positives['score'] > thresh]
        # keep the order by referring to the original index
        # in order to insert the samples in descending order of score
        diff = sorted(set(whole_positives.index) - set(ids),
                      key=whole_positives.index.get_loc)
        pos_cnt = is_positive.sum()
        reranked_ids = np.insert(sorted_retrieved_ids, pos_cnt, diff)[:topk]

        images.append(' '.join(reranked_ids))

    subm['images'] = images

    return subm


def main():
    index_dirs = [
        '../exp/v2clean/feats_index19_ms_L2_ep4_freqthresh-3_loss-arcface_verifythresh-30/']
    test_dirs = [
        '../exp/v2clean/feats_test19_ms_L2_ep4_freqthresh-3_loss-arcface_verifythresh-30/']
    train_dirs = [
        '../exp/v2clean/feats_train_ms_L2_ep4_freqthresh-3_loss-arcface_verifythresh-30/']

    ids_index, feats_index = utils.prepare_ids_and_feats(index_dirs, normalize=True)
    ids_test, feats_test = utils.prepare_ids_and_feats(test_dirs, normalize=True)
    ids_train, feats_train = utils.prepare_ids_and_feats(train_dirs, normalize=True)

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
                                subm, topk=100)

    output_name = ROOT + f'output/submit_retrieval.csv.gz'
    subm[['id', 'images']].to_csv(output_name, compression='gzip', index=False)
    print('saved to ' + output_name)

    cmd = f'kaggle c submit -c landmark-retrieval-2019 -f {output_name} -m "" '
    print(cmd)
    subprocess.run(cmd, shell=True)


if __name__ == '__main__':
    main()
