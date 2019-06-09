from pathlib import Path
import h5py
import numpy as np


def l2norm_numpy(x):
    return x / np.linalg.norm(x, ord=2, axis=1, keepdims=True)


def load_scattered_h5data(filenames):
    """全てのファイルを読み込んだらidとfeatsをそれぞれconcat。idでソートして返す。
    :param filenames: h5形式のデータのパスのリスト
    :return: numpy.ndarray形式のidリストと対応するfeature
    """
    ids, feats = [], []

    for fn in filenames:
        with h5py.File(fn, 'r') as f:
            ids.append(f['ids'][()].astype(str))
            feats.append(f['feats'][()])

    ids = np.concatenate(ids, axis=0)
    feats = np.concatenate(feats, axis=0)

    order = np.argsort(ids)
    ids = ids[order]
    feats = feats[order]

    return ids, feats


def prepare_ids_and_feats(dirs, weights=None, normalize=True):
    """複数のdescriptorをload, normalize, weighting, concatして一本のdescriptorに統合して出力

    :param dirs: descriptorのファイルを持つディレクトリのリスト
    :param weights: 各descriptorに適用する重み係数
    :param normalize: 出力ベクトルを正規化するか
    :return: concatされた一本のdescriptor
    """
    if weights is None:
        weights = [1.0 for _ in range(len(dirs))]

    ids_list, feats_list = [], []
    for f_dir, w in zip(dirs, weights):
        print(f_dir)
        assert len(list(Path(f_dir).glob('*.h5'))) > 0

        ids, feats = load_scattered_h5data(Path(f_dir).glob('*.h5'))
        feats = l2norm_numpy(feats) * w
        ids_list.append(ids)
        feats_list.append(feats)

    if len(ids_list) > 1:
        # 全てのidが同じ順番かチェック
        assert all([np.all(ids_list[0] == ids) for ids in ids_list[1:]])
        # idがユニークかチェック
        assert len(ids_list[0]) == len(np.unique(ids_list[0]))

    ids_list = ids_list[0]
    feats_list = np.concatenate(feats_list, axis=1)
    if normalize:
        feats_list = l2norm_numpy(feats_list)

    return ids_list, feats_list
