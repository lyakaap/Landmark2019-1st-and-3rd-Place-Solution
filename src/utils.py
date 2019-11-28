import h5py
import json
import logging
import lockfile
import subprocess
import sys
import time
from collections import OrderedDict, deque
from pathlib import Path

import logzero
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import ParameterGrid, ParameterSampler
from tensorboardX import SummaryWriter


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def remove_redundant_keys(state_dict: OrderedDict):
    # remove DataParallel wrapping
    if 'module' in list(state_dict.keys())[0]:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                # str.replace() can't be used because of unintended key removal (e.g. se-module)
                new_state_dict[k[7:]] = v
    else:
        new_state_dict = state_dict

    return new_state_dict


def save_checkpoint(path, model, epoch, optimizer=None, save_arch=False, params=None):
    attributes = {
        'epoch': epoch,
        'state_dict': remove_redundant_keys(model.state_dict()),
    }

    if optimizer is not None:
        attributes['optimizer'] = optimizer.state_dict()

    if save_arch:
        attributes['arch'] = model

    if params is not None:
        attributes['params'] = params

    try:
        torch.save(attributes, path)
    except TypeError:
        if 'arch' in attributes:
            print('Model architecture will be ignored because the architecture includes non-pickable objects.')
            del attributes['arch']
            torch.save(attributes, path)


def load_checkpoint(path, model=None, optimizer=None, params=False, epoch=False):
    resume = torch.load(path)
    rets = dict()

    if model is not None:
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(remove_redundant_keys(resume['state_dict']))
        else:
            model.load_state_dict(remove_redundant_keys(resume['state_dict']))

        rets['model'] = model

    if optimizer is not None:
        optimizer.load_state_dict(resume['optimizer'])
        rets['optimizer'] = optimizer
    if params:
        rets['params'] = resume['params']
    if epoch:
        rets['epoch'] = resume['epoch']

    return rets


def load_model(path, is_inference=True):
    resume = torch.load(path)
    model = resume['arch']
    model.load_state_dict(resume['state_dict'])
    if is_inference:
        model.eval()
    return model


def l2norm_numpy(x):
    return x / np.linalg.norm(x, ord=2, axis=1, keepdims=True)


class WhiteningPCA:
    def __init__(self, eps=1e-16):
        self.eps = eps
        self.m = None
        self.P = None

    def fit(self, X: np.ndarray):
        """size should be N >> D"""
        N, D = X.shape

        self.m = X.mean(axis=0)  # (D,)
        X_centered = X - self.m
        C = X_centered.transpose() @ X_centered / N  # (D, D)

        S, U = np.linalg.eig(C)
        order = S.argsort()[::-1]
        S = S[order] + self.eps
        U = U[order]

        self.P = np.diag(S ** -0.5) @ U  # (D, D)

    def transform(self, X: np.ndarray, n_components=None):
        X_whitened = (X - self.m) @ self.P.transpose()

        if n_components is not None:
            X_whitened = X_whitened[:, :n_components]

        return X_whitened


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


def get_logger(log_dir, loglevel=logging.INFO, tensorboard_dir=None):
    from logzero import logger

    if not Path(log_dir).exists():
        Path(log_dir).mkdir(parents=True)
    logzero.loglevel(loglevel)
    logzero.logfile(log_dir + '/logfile')

    if tensorboard_dir is not None:
        if not Path(tensorboard_dir).exists():
            Path(tensorboard_dir).mkdir(parents=True)
        writer = SummaryWriter(tensorboard_dir)

        return logger, writer

    return logger


def get_optim(params, target):
    assert isinstance(target, nn.Module) or isinstance(target, dict)

    if isinstance(target, nn.Module):
        target = target.parameters()

    if params['optimizer'] == 'sgd':
        optimizer = optim.SGD(target, params['lr'], weight_decay=params['wd'])
    elif params['optimizer'] == 'momentum':
        optimizer = optim.SGD(target, params['lr'], momentum=0.9, weight_decay=params['wd'])
    elif params['optimizer'] == 'nesterov':
        optimizer = optim.SGD(target, params['lr'], momentum=0.9,
                              weight_decay=params['wd'], nesterov=True)
    elif params['optimizer'] == 'adam':
        optimizer = optim.Adam(target, params['lr'], weight_decay=params['wd'])
    elif params['optimizer'] == 'amsgrad':
        optimizer = optim.Adam(target, params['lr'], weight_decay=params['wd'], amsgrad=True)
    elif params['optimizer'] == 'rmsprop':
        optimizer = optim.RMSprop(target, params['lr'], weight_decay=params['wd'])
    else:
        raise ValueError

    return optimizer


def write_tuning_result(params: dict, result: dict, df_path: str):
    row = pd.DataFrame()
    for key in params['tuning_params']:
        row[key] = [params[key]]

    for key, val in result.items():
        row[key] = val

    with lockfile.FileLock(df_path):
        df_results = pd.read_csv(df_path)
        df_results = pd.concat([df_results, row], sort=False).reset_index(drop=True)
        df_results.to_csv(df_path, index=None)


def check_duplicate(df: pd.DataFrame, p: dict):
    """check if current params combination has already done"""

    new_key_is_included = not all(map(lambda x: x in df.columns, p.keys()))
    if new_key_is_included:
        return False

    for i in range(len(df)):  # for avoiding unexpected cast due to row-slicing
        is_dup = True
        for key, val in p.items():
            if df.loc[i, key] != val:
                is_dup = False
                break
        if is_dup:
            return True
    else:
        return False


def launch_tuning(mode: str='grid', n_iter: int=1, n_gpu: int=1, devices: str='0',
                  params: dict=None, space: dict=None, root: str='../',
                  save_interval: int=0, candidate_list=None):
    """
    Launch paramter search by specific way.
    Each trials are launched asynchronously by forking subprocess and all results of trials
    are automatically written in csv file.

    :param mode: the way of parameter search, one of 'grid or random'.
    :param n_iter: num of iteration for random search.
    :param n_gpu: num of gpu used at one trial.
    :param devices: gpu devices for tuning.
    :param params: training parameters.
                   the values designated as tuning parameters are overwritten
    :param space: paramter search space.
    :param root: path of the root directory.
    """

    gpu_list = deque(devices.split(','))

    if candidate_list is None:
        if mode == 'grid':
            candidate_list = list(ParameterGrid(space))
        elif mode == 'random':
            candidate_list = list(ParameterSampler(space, n_iter))
        else:
            raise ValueError

    params['tuning_params'] = list(candidate_list[0].keys())

    df_path = root + f'exp/{params["ex_name"]}/tuning/results.csv'
    if Path(df_path).exists() and Path(df_path).stat().st_size > 5:
        df_results = pd.read_csv(df_path)
    else:
        cols = list(candidate_list[0].keys())
        df_results = pd.DataFrame(columns=cols)
        df_results.to_csv(df_path, index=False)

    procs = []
    for p in candidate_list:

        if check_duplicate(df_results, p):
            print(f'skip: {p} because this setting is already experimented.')
            continue

        # overwrite hyper parameters for search
        for key, val in p.items():
            params[key] = val

        while True:
            if len(gpu_list) >= n_gpu:
                devices = ','.join([gpu_list.pop() for _ in range(n_gpu)])
                setting = '_'.join(f'{key}-{val}' for key, val in p.items())
                params_path = root + f'exp/{params["ex_name"]}/tuning/params_{setting}.json'
                with open(params_path, 'w') as f:
                    json.dump(params, f)
                break
            else:
                time.sleep(1)
                for i, (proc, dev) in enumerate(procs):
                    if proc.poll() is not None:
                        gpu_list += deque(dev.split(','))
                        del procs[i]

        cmd = f'{sys.executable} {params["ex_name"]}.py job ' \
              f'--tuning --params-path {params_path} --devices "{devices}" -s {save_interval}'
        procs.append((subprocess.Popen(cmd, shell=True), devices))

    while True:
        time.sleep(1)
        if all(proc.poll() is not None for i, (proc, dev) in enumerate(procs)):
            print('All parameter combinations have finished.')
            break

    show_tuning_result(params["ex_name"])


def show_tuning_result(ex_name, mode='markdown', sort_by=None, ascending=False):
    table = pd.read_csv(f'../exp/{ex_name}/tuning/results.csv')
    if sort_by is not None:
        table = table.sort_values(sort_by, ascending=ascending)

    if mode == 'markdown':
        from tabulate import tabulate
        print(tabulate(table, headers='keys', tablefmt='pipe', showindex=False))
    elif mode == 'latex':
        from tabulate import tabulate
        print(tabulate(table, headers='keys', tablefmt='latex', floatfmt='.2f', showindex=False))
    else:
        from IPython.core.display import display
        display(table)
