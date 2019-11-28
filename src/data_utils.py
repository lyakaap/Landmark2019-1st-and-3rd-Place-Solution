import bisect
import copy
import math

import cv2
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from albumentations import (
    Compose,
    Normalize,
    IAAAffine,
    RandomBrightnessContrast,
    MotionBlur,
    CLAHE,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from src import torch_custom

pd.set_option('mode.chained_assignment', None)


def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def _list_collate(batch):
    return [_ for _ in zip(*batch)]


def _cat_collate(batch):
    """concat if all tensors have same size."""
    img_ids, imgs = [_ for _ in zip(*batch)]
    imgs = [img[None] for img in imgs]
    if all(imgs[0].shape == img.shape for img in imgs):
        imgs = [torch.cat(imgs, dim=0)]
    return img_ids, imgs


class LandmarkDataset(Dataset):
    # (width, height)
    size_template = {
        'SS': ((384, 256), (256, 256), (256, 384)),
        'SS2': ((480, 320), (480, 352), (384, 384), (352, 480), (320, 480)),
        'S': ((512, 384), (448, 448), (384, 512)),
        'S2': ((512, 352), (512, 384), (448, 448), (384, 512), (352, 512)),
        'SM2': ((736, 512), (704, 544), (608, 608), (544, 704), (512, 736)),
        'M': ((800, 608), (704, 704), (608, 800)),
        'M2': ((800, 544), (800, 608), (704, 704), (608, 800), (544, 800)),
        'L': ((800, 608), (800, 800), (608, 800)),
        'L2': ((800, 544), (800, 608), (800, 800), (608, 800), (544, 800)),
    }

    def __init__(self,
                 paths,
                 class_ids=None,
                 transform=None,
                 aspect_gids=None,
                 scale='S'
                 ):
        self.paths = paths
        self.class_ids = class_ids
        self.aspect_gids = aspect_gids
        self.transform = transform
        self.scale = scale

    def __getitem__(self, index):
        img_path = str(self.paths[index])
        img = cv2.imread(img_path)
        assert img is not None, f'path: {img_path} is invalid.'
        img = img[..., ::-1]

        if self.aspect_gids is not None:
            gid = self.aspect_gids[index]
            img = cv2.resize(img, self.size_template[self.scale][gid])

        if self.transform is not None:
            img = self.transform(image=img)['image']

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        img_id = img_path.split('/')[-1].replace('.jpg', '')

        if self.class_ids is None:
            return img_id, img
        else:
            target = torch.tensor(self.class_ids[index]).long()
            return img_id, img, target

    def __len__(self):
        return len(self.paths)


def prepare_recognition_df(df_path=None,
                           df=None,
                           class_topk=1000,
                           least_freq_thresh=-1,
                           limit_samples_per_class=-1,
                           seed=7777777
                           ):
    if isinstance(df, pd.DataFrame):
        train = df
    else:
        if '.pkl' in df_path:
            train = pd.read_pickle(df_path)
        elif '.csv' in df_path:
            train = pd.read_csv(df_path)
        else:
            raise ValueError

    class_freq = train.landmark_id.value_counts()
    keep_index = []

    if class_topk > 0:  # 頻出上位k個まで
        use_landmark_ids = class_freq.index[:class_topk]
    elif least_freq_thresh > 0:  # 出現頻度が稀なlandmark_idを持つ画像は使わない
        use_landmark_ids = class_freq[(least_freq_thresh <= class_freq)].index
    else:
        use_landmark_ids = class_freq.index

    if limit_samples_per_class > 0:  # 出現頻度が多すぎるlandmark_idに関してはサンプリングを行う
        for id_ in class_freq[class_freq > limit_samples_per_class].index:
            idx = train.query('landmark_id == @id_').sample(
                n=limit_samples_per_class, random_state=seed).index
            keep_index.extend(idx)

        large_class_index = class_freq[class_freq > limit_samples_per_class].index
        use_landmark_ids = pd.Index(set(use_landmark_ids) - set(large_class_index))

        keep_index = pd.Index(keep_index)
        keep_index = keep_index.append(train[train['landmark_id'].isin(use_landmark_ids)].index)
    else:
        keep_index = train['landmark_id'].isin(use_landmark_ids)

    train_filtered = train.loc[keep_index]
    train_filtered['landmark_id'] = train_filtered['landmark_id'].astype('category')
    train_filtered['class_id'] = train_filtered['landmark_id'].cat.codes.astype('int64')

    return train_filtered


def prepare_grouped_loader_from_df(df,
                                   transform,
                                   batch_size,
                                   scale='S',
                                   is_train=True,
                                   num_workers=4
                                   ):
    class_ids = df['class_id'].values if 'class_id' in df.columns else None
    dataset = LandmarkDataset(paths=df['path'].values,
                              class_ids=class_ids,
                              aspect_gids=df['aspect_gid'].values,
                              transform=transform,
                              scale=scale)
    if is_train:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    gb_sampler = torch_custom.GroupedBatchSampler(sampler=sampler,
                                                  group_ids=df['aspect_gid'].values,
                                                  batch_size=batch_size,
                                                  drop_uneven=is_train)
    loader = DataLoader(dataset=dataset,
                        batch_sampler=gb_sampler,
                        pin_memory=True,
                        num_workers=num_workers)
    return loader


def build_transforms(mean=(0.485, 0.456, 0.406),
                     std=(0.229, 0.224, 0.225),
                     divide_by=255.0,
                     scale_limit=0.0,
                     shear_limit=0,
                     rotate_limit=0,
                     brightness_limit=0.0,
                     contrast_limit=0.0,
                     clahe_p=0.0,
                     blur_p=0.0,
                     autoaugment=False,
                     ):
    norm = Normalize(mean=mean, std=std, max_pixel_value=divide_by)

    if autoaugment:
        train_transform = Compose([
            torch_custom.AutoAugmentWrapper(p=1.0),
            torch_custom.RandomCropThenScaleToOriginalSize(limit=scale_limit, p=1.0),
            norm,
        ])
    else:
        train_transform = Compose([
            IAAAffine(rotate=(-rotate_limit, rotate_limit), shear=(-shear_limit, shear_limit), mode='constant'),
            RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=contrast_limit),
            MotionBlur(p=blur_p),
            CLAHE(p=clahe_p),
            torch_custom.RandomCropThenScaleToOriginalSize(limit=scale_limit, p=1.0),
            norm,
        ])
    eval_transform = Compose([norm])

    return train_transform, eval_transform


def make_train_loaders(params,
                       data_root,
                       use_clean_version=False,
                       train_transform=None,
                       eval_transform=None,
                       scale='S',
                       class_topk=1000,
                       limit_samples_per_class=-1,
                       test_size=0.1,
                       num_workers=4,
                       seed=77777
                       ):
    if use_clean_version:
        clean_train = pd.read_csv(params['clean_path'])
        clean_train_ids = np.concatenate(clean_train['images'].str.split(' '))
        train = pd.read_pickle(data_root + 'train.pkl')

        new_train = train[train['id'].isin(clean_train_ids)]
        assert len(clean_train) == new_train['landmark_id'].nunique()
        params['class_topk'] = len(clean_train)

        df = prepare_recognition_df(df=new_train,
                                    class_topk=params['class_topk'],
                                    least_freq_thresh=-1,
                                    limit_samples_per_class=limit_samples_per_class,
                                    seed=seed)
    else:
        df = prepare_recognition_df(df_path=data_root + 'train.pkl',
                                    class_topk=class_topk,
                                    least_freq_thresh=-1,
                                    limit_samples_per_class=limit_samples_per_class,
                                    seed=seed)

    if 'path' not in df.columns:
        if 'gld_v1' in data_root:
            df['path'] = df['id'].apply(lambda x: f'{data_root}/{split}/{x}.jpg')
        if 'gld_v2' in data_root:
            df['path'] = df['id'].apply(lambda x: f'{data_root}/{split}/{"/".join(x[:3])}/{x}.jpg')

    if '2' in scale:  # finer-grained aspect grouping
        bins = [0.67, 0.77, 1.33, 1.5]
    else:
        bins = [0.77, 1.33]

    df['aspect_ratio'] = df['height'] / df['width']
    df['aspect_gid'] = _quantize(df['aspect_ratio'], bins=bins)

    train_split, val_split = train_test_split(df, test_size=test_size, random_state=seed)

    data_loaders = dict()
    data_loaders['train'] = prepare_grouped_loader_from_df(
        train_split, train_transform, params['batch_size'],
        scale=scale, is_train=True, num_workers=num_workers)
    data_loaders['val'] = prepare_grouped_loader_from_df(
        val_split, eval_transform, params['test_batch_size'],
        scale=scale, is_train=False, num_workers=num_workers)

    return data_loaders


def make_predict_loaders(params,
                         data_root,
                         eval_transform=None,
                         scale='S',
                         splits=('index', 'test'),
                         num_workers=4,
                         n_blocks=1,
                         block_id=0,
                         ):
    """
    :param splits: 読み込むデータセットの種類
    :param block_id: ブロック分割したときのID
    """
    data_loaders = dict()

    if '2' in scale:  # finer-grained aspect grouping
        bins = [0.67, 0.77, 1.33, 1.5]
    else:
        bins = [0.77, 1.33]

    for split in splits:
        df = pd.read_pickle(f'{data_root}/{split}.pkl')
        n_per_block = math.ceil(len(df) / n_blocks)
        df = df.iloc[block_id * n_per_block: (block_id + 1) * n_per_block]

        if 'path' not in df.columns:
            if 'gld_v1' in data_root:
                df['path'] = df['id'].apply(lambda x: f'{data_root}/{split}/{x}.jpg')
            if 'gld_v2' in data_root:
                df['path'] = df['id'].apply(lambda x: f'{data_root}/{split}/{"/".join(x[:3])}/{x}.jpg')

        if scale == 'O':  # original resolution
            dataset = LandmarkDataset(paths=df['path'].values, transform=eval_transform)
            data_loaders[split] = DataLoader(dataset=dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             drop_last=False,
                                             pin_memory=True,
                                             num_workers=num_workers)
        else:
            df['aspect_ratio'] = df['height'] / df['width']
            df['aspect_gid'] = _quantize(df['aspect_ratio'], bins=bins)
            data_loaders[split] = prepare_grouped_loader_from_df(
                df, eval_transform, params['test_batch_size'],
                scale=scale, is_train=False, num_workers=num_workers)

    return data_loaders
