from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from easydict import EasyDict as edict
from util import timer


def gpux4_allpair_similarity(ds, prefix):
    # Use cache
    cache_data = load_cached_result(prefix)
    if cache_data is not None:
        return cache_data

    # Search with GpuMultiple
    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    vres = []
    for _ in range(4):
        res = faiss.StandardGpuResources()
        vres.append(res)

    cpu_index = faiss.IndexFlatIP(ds.feats_index.shape[1])
    gpu_index = faiss.index_cpu_to_gpu_multiple_py(vres, cpu_index, co)
    gpu_index.add(ds.feats_index)

    # 177sec
    with timer('Prepare all-pair similarity on index dataset'):
        ii_sims, ii_ids = gpu_index.search(x=ds.feats_index, k=100)

    with timer('Save results (index-index)'):
        fn_out = Path(prefix) / "index19_vs_index19_ids.npy"
        fn_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(fn_out), ii_ids)
        np.save(str(Path(prefix) / "index19_vs_index19_sims.npy"), ii_sims)

    with timer('Prepare all-pair similarity on test-index dataset'):
        ti_sims, ti_ids = gpu_index.search(x=ds.feats_test, k=100)

    with timer('Save results (test-index)'):
        np.save(str(Path(prefix) / "test19_vs_index19_ids.npy"), ti_ids)
        np.save(str(Path(prefix) / "test19_vs_index19_sims.npy"), ti_sims)

    return edict({
        'ti_sims': ti_sims, 'ti_ids': ti_ids,
        'ii_sims': ii_sims, 'ii_ids': ii_ids,
    })


def load_cached_result(prefix):
    cached_files = [
        Path(prefix) / "index19_vs_index19_ids.npy",
        Path(prefix) / "index19_vs_index19_sims.npy",
        Path(prefix) / "test19_vs_index19_ids.npy",
        Path(prefix) / "test19_vs_index19_sims.npy",
    ]
    for fp in cached_files:
        if not fp.exists():
            return

    return edict({
        'ti_sims': np.load(str(Path(prefix) / "test19_vs_index19_sims.npy")),
        'ti_ids': np.load(str(Path(prefix) / "test19_vs_index19_ids.npy")),
        'ii_sims': np.load(str(Path(prefix) / "index19_vs_index19_sims.npy")),
        'ii_ids': np.load(str(Path(prefix) / "index19_vs_index19_ids.npy")),
    })


def gpux4_euclidsearch_from_dataset(ds,
                                    fn_npy,
                                    lhs='test',
                                    rhs='index',
                                    topk=100):
    # Search with GpuMultiple
    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    vres = []
    for _ in range(4):
        res = faiss.StandardGpuResources()
        vres.append(res)

    cpu_index = faiss.IndexFlatL2(ds[f'feats_{rhs}'].shape[1])
    gpu_index = faiss.index_cpu_to_gpu_multiple_py(vres, cpu_index, co)
    gpu_index.add(ds[f'feats_{rhs}'])

    _, all_ranks = gpu_index.search(x=ds[f'feats_{lhs}'], k=topk)
    Path(fn_npy).parent.mkdir(parents=True, exist_ok=True)
    np.save(fn_npy, all_ranks)

    if lhs == 'test' and rhs == 'index':  # Retrieval task
        fn_sub = fn_npy.rstrip('.npy') + '.csv.gz'
        save_sub_from_top100ranks(ds, all_ranks, fn_sub, topk=topk)


def save_sub_from_top100ranks(ds, all_ranks, fn_sub, topk=100):
    qimlist = ds.ids_test.tolist()
    imlist = ds.ids_index.tolist()

    rows = []
    for i in range(all_ranks.shape[0]):
        images = " ".join([imlist[all_ranks[i, j]] for j in range(topk)])
        rows.append({'id': qimlist[i], 'images': images})

    df = pd.DataFrame(rows)
    df.loc[:, 'images'] = df.images.fillna('')
    df[['id', 'images']].to_csv(fn_sub, index=False, compression='gzip')
