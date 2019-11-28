import argparse
import os
import subprocess

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

from cirtorch.datasets.genericdataset import ImagesFromList
from cirtorch.datasets.testdataset import configdataset
from cirtorch.utils.evaluate import compute_map_and_print
from cirtorch.utils.general import get_data_root
from src import utils


def extract_vectors(model,
                    images,
                    image_size=1024,
                    eval_transform=None,
                    bbxs=None,
                    scales=(1,),
                    tta_gem_p=1.0,
                    tqdm_desc=''
                    ):
    if eval_transform is None:
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    local_eval_loader = torch.utils.data.DataLoader(
        ImagesFromList(root='', images=images, imsize=image_size, bbxs=bbxs, transform=eval_transform),
        batch_size=1, shuffle=False, num_workers=8, pin_memory=True
    )

    ids, feats = [], []
    for i, x in tqdm(enumerate(local_eval_loader),
                     total=len(local_eval_loader),
                     desc=tqdm_desc,
                     miniters=None, ncols=55):

        batch_size, _, h, w = x.shape
        feat_blend = []

        with torch.no_grad():
            x = x.to('cuda')
            for s in scales:
                size = int(h * s // model.DIVIDABLE_BY * model.DIVIDABLE_BY), \
                       int(w * s // model.DIVIDABLE_BY * model.DIVIDABLE_BY)  # round off
                scaled_x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
                feat = model.extract_feat(scaled_x)
                if tta_gem_p != 1.0:
                    feat = feat ** tta_gem_p
                feat = feat.cpu().numpy()
                feat_blend.append(feat)

        feat_blend = np.mean(feat_blend, axis=0)
        feats.append(feat_blend)

    feats = np.concatenate(feats)
    if tta_gem_p != 1.0:
        feats = feats ** (1.0 / tta_gem_p)
    feats = utils.l2norm_numpy(feats)

    return feats


def eval_datasets(model,
                  datasets=('oxford5k', 'paris6k', 'roxford5k', 'rparis6k'),
                  ms=False,
                  tta_gem_p=1.0,
                  logger=None
                  ):
    model = model.eval()

    data_root = get_data_root()
    scales = [1 / 2 ** (1 / 2), 1.0, 2 ** (1 / 2)] if ms else [1.0]
    results = dict()

    for dataset in datasets:

        # prepare config structure for the test dataset
        cfg = configdataset(dataset, os.path.join(data_root, 'test'))
        images = [cfg['im_fname'](cfg, i) for i in range(cfg['n'])]
        qimages = [cfg['qim_fname'](cfg, i) for i in range(cfg['nq'])]
        bbxs = [tuple(cfg['gnd'][i]['bbx']) for i in range(cfg['nq'])]
        tqdm_desc = cfg['dataset']

        db_feats = extract_vectors(model,
                                   images=images,
                                   bbxs=None,
                                   scales=scales,
                                   tta_gem_p=tta_gem_p,
                                   tqdm_desc=tqdm_desc)
        query_feats = extract_vectors(model,
                                      images=qimages,
                                      bbxs=bbxs,
                                      scales=scales,
                                      tta_gem_p=tta_gem_p,
                                      tqdm_desc=tqdm_desc)

        scores = np.dot(db_feats, query_feats.T)
        ranks = np.argsort(-scores, axis=0)
        results[dataset] = compute_map_and_print(
            dataset, ranks, cfg['gnd'], kappas=[1, 5, 10], logger=logger)

    return results


if __name__ == '__main__':

    topk = 100

    parser = argparse.ArgumentParser()
    parser.add_argument('index_dirs', help='directories containing features of index')
    parser.add_argument('test_dirs', help='directories containing features of test')
    parser.add_argument('--setting', default='')
    parser.add_argument('-w', '--weights', default='1')
    parser.add_argument('-d', '--devices', default='0', help='gpu device indexes')
    args = parser.parse_args()

    index_dirs = args.index_dirs.split(',')
    test_dirs = args.test_dirs.split(',')
    setting = args.setting
    weights = list(map(int, args.weights.split(',')))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
    n_gpus = len(args.devices.split(','))

    ids_index, feats_index = utils.prepare_ids_and_feats(index_dirs, weights, normalize=True)
    ids_test, feats_test = utils.prepare_ids_and_feats(test_dirs, weights, normalize=True)
    # ids_train, feats_train = utils.prepare_ids_and_feats(train_dirs, weights, normalize=True)

    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    # co.float16 = False

    vres = []
    for _ in range(n_gpus):
        res = faiss.StandardGpuResources()
        vres.append(res)

    print('build index...')
    cpu_index = faiss.IndexFlatL2(feats_index.shape[1])
    gpu_index = faiss.index_cpu_to_gpu_multiple_py(vres, cpu_index, co)
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
