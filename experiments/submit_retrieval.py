import faiss
import numpy as np
import pandas as pd
import subprocess
from src import utils

setting = 'final'
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

ids_index, feats_index = utils.prepare_ids_and_feats(index_dirs, weights, normalize=True)
ids_test, feats_test = utils.prepare_ids_and_feats(test_dirs, weights, normalize=True)
ids_train, feats_train = utils.prepare_ids_and_feats(train_dirs, weights, normalize=True)

print('build index...')
cpu_index = faiss.IndexFlatL2(feats_index.shape[1])
gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
gpu_index.add(feats_index)
dists, topk_idx = gpu_index.search(x=feats_test, k=topk)
print('query search done.')

retrieval_result = pd.DataFrame(ids_test, columns=['id'])
retrieval_result['images'] = np.apply_along_axis(' '.join, axis=1, arr=ids_index[topk_idx])
output_name = f'../output/{setting}.csv.gz'
retrieval_result.to_csv(output_name, compression='gzip', index=False)
print('saved to ' + output_name)

cmd = f'kaggle c submit -c landmark-retrieval-2019 -f {output_name} -m "" '
print(cmd)
subprocess.run(cmd, shell=True)
