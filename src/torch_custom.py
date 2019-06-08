import itertools
import random
import math

import albumentations.augmentations.functional as F

import cv2
from PIL import Image
import numpy as np
import torch
from albumentations import ImageOnlyTransform
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import Sampler
from torch._six import int_classes as _int_classes

from src.autoaugment import ImageNetPolicy


class PiecewiseCyclicalLinearLR(_LRScheduler):
    r"""Set the learning rate of each parameter group using a piecewise
    cyclical linear schedule.

    When last_epoch=-1, sets initial lr as lr.

    _Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs
    https://arxiv.org/pdf/1802.10026
    Exploring loss function topology with cyclical learning rates
    https://arxiv.org/abs/1702.04283
    """

    def __init__(self, optimizer, c, alpha1=1e-2, alpha2=5e-4, last_epoch=-1):
        """
        :param c: cycle length
        :param alpha1: lr upper bound of cycle
        :param alpha2: lr lower bounf of cycle
        """
        self.c = c
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        super(PiecewiseCyclicalLinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):

        lrs = []
        for _ in range(len(self.base_lrs)):
            ti = ((self.last_epoch - 1) % self.c + 1) / self.c
            if 0 <= ti <= 0.5:
                lr = (1 - 2 * ti) * self.alpha1 + 2 * ti * self.alpha2
            elif 0.5 < ti <= 1.0:
                lr = (2 - 2 * ti) * self.alpha2 + (2 * ti - 1) * self.alpha1
            else:
                raise ValueError('t(i) is out of range [0,1].')
            lrs.append(lr)

        return lrs


class PolyLR(_LRScheduler):

    def __init__(self, optimizer, power=0.9, max_epoch=4e4, last_epoch=-1):
        """The argument name "epoch" also can be thought as "iter"."""
        self.power = power
        self.max_epoch = max_epoch
        self.last_epoch = last_epoch
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lrs = []
        for base_lr in self.base_lrs:
            lr = base_lr * (1.0 - (self.last_epoch / self.max_epoch)) ** self.power
            lrs.append(lr)

        return lrs


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    """cosine annealing scheduler with warmup.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(
        self,
        optimizer,
        T_max,
        eta_min,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )

        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            return self.get_lr_warmup()
        else:
            return self.get_lr_cos_annealing()

    def get_lr_warmup(self):
        if self.warmup_method == "constant":
            warmup_factor = self.warmup_factor
        elif self.warmup_method == "linear":
            alpha = self.last_epoch / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr * warmup_factor
            for base_lr in self.base_lrs
        ]

    def get_lr_cos_annealing(self):
        last_epoch = self.last_epoch - self.warmup_iters
        T_max = self.T_max - self.warmup_iters
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * last_epoch / T_max)) / 2
                for base_lr in self.base_lrs]


def _pad_const(x, target_height, target_width, value=255, center=True, pad_loc_seed=None):
    random.seed(pad_loc_seed)
    height, width = x.shape[:2]

    if height < target_height:
        if center:
            h_pad_top = int((target_height - height) / 2.0)
        else:
            h_pad_top = random.randint(a=0, b=target_height - height)
        h_pad_bottom = target_height - height - h_pad_top
    else:
        h_pad_top = 0
        h_pad_bottom = 0

    if width < target_width:
        if center:
            w_pad_left = int((target_width - width) / 2.0)
        else:
            w_pad_left = random.randint(a=0, b=target_width - width)
        w_pad_right = target_width - width - w_pad_left
    else:
        w_pad_left = 0
        w_pad_right = 0

    x = cv2.copyMakeBorder(x, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right,
                           cv2.BORDER_CONSTANT, value=value)
    return x


class RandomCropThenScaleToOriginalSize(ImageOnlyTransform):
    """Crop a random part of the input and rescale it to some size.

    Args:
        limit (float): maximum factor range for cropping region size.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        pad_value (int): pixel value for padding.
        p (float): probability of applying the transform. Default: 1.
    """

    def __init__(self, limit=0.1, interpolation=cv2.INTER_LINEAR, pad_value=0, p=1.0):
        super(RandomCropThenScaleToOriginalSize, self).__init__(p)
        self.limit = limit
        self.interpolation = interpolation
        self.pad_value = pad_value

    def apply(self, img, height_scale=1.0, width_scale=1.0, h_start=0, w_start=0, interpolation=cv2.INTER_LINEAR,
              pad_value=0, pad_loc_seed=None, **params):
        img_height, img_width = img.shape[:2]
        crop_height, crop_width = int(img_height * height_scale), int(img_width * width_scale)
        crop = self.random_crop(img, crop_height, crop_width, h_start, w_start, pad_value, pad_loc_seed)
        return F.resize(crop, img_height, img_width, interpolation)

    def get_params(self):
        height_scale = 1.0 + random.uniform(-self.limit, self.limit)
        width_scale = 1.0 + random.uniform(-self.limit, self.limit)
        return {'h_start': random.random(),
                'w_start': random.random(),
                'height_scale': height_scale,
                'width_scale': width_scale,
                'pad_loc_seed': random.random()}

    def update_params(self, params, **kwargs):
        if hasattr(self, 'interpolation'):
            params['interpolation'] = self.interpolation
        if hasattr(self, 'pad_value'):
            params['pad_value'] = self.pad_value
        params.update({'cols': kwargs['image'].shape[1], 'rows': kwargs['image'].shape[0]})
        return params

    @staticmethod
    def random_crop(img, crop_height, crop_width, h_start, w_start, pad_value=0, pad_loc_seed=None):
        height, width = img.shape[:2]

        if height < crop_height or width < crop_width:
            img = _pad_const(img, crop_height, crop_width, value=pad_value, center=False, pad_loc_seed=pad_loc_seed)

        y1 = max(int((height - crop_height) * h_start), 0)
        y2 = y1 + crop_height
        x1 = max(int((width - crop_width) * w_start), 0)
        x2 = x1 + crop_width
        img = img[y1:y2, x1:x2]
        return img


class AutoAugmentWrapper(ImageOnlyTransform):

    def __init__(self, p=1.0):
        super(AutoAugmentWrapper, self).__init__(p)
        self.autoaugment = ImageNetPolicy()

    def apply(self, img, **params):
        img = Image.fromarray(img)
        img = self.autoaugment(img)
        img = np.asarray(img)
        return img


# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
class GroupedBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that elements from the same group should appear in groups of batch_size.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.
    Arguments:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_uneven (bool): If ``True``, the sampler will drop the batches whose
            size is less than ``batch_size``
    """

    def __init__(self, sampler, group_ids, batch_size, drop_uneven=False):
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )
        self.sampler = sampler
        self.group_ids = torch.as_tensor(group_ids)
        assert self.group_ids.dim() == 1
        self.batch_size = batch_size
        self.drop_uneven = drop_uneven

        self.groups = torch.unique(self.group_ids).sort(0)[0]

        self._can_reuse_batches = False

    def _prepare_batches(self):
        dataset_size = len(self.group_ids)
        # get the sampled indices from the sampler
        sampled_ids = torch.as_tensor(list(self.sampler))
        # potentially not all elements of the dataset were sampled
        # by the sampler (e.g., DistributedSampler).
        # construct a tensor which contains -1 if the element was
        # not sampled, and a non-negative number indicating the
        # order where the element was sampled.
        # for example. if sampled_ids = [3, 1] and dataset_size = 5,
        # the order is [-1, 1, -1, 0, -1]
        order = torch.full((dataset_size,), -1, dtype=torch.int64)
        order[sampled_ids] = torch.arange(len(sampled_ids))

        # get a mask with the elements that were sampled
        mask = order >= 0

        # find the elements that belong to each individual cluster
        clusters = [(self.group_ids == i) & mask for i in self.groups]
        # get relative order of the elements inside each cluster
        # that follows the order from the sampler
        relative_order = [order[cluster] for cluster in clusters]
        # with the relative order, find the absolute order in the
        # sampled space
        permutation_ids = [s[s.sort()[1]] for s in relative_order]
        # permute each cluster so that they follow the order from
        # the sampler
        permuted_clusters = [sampled_ids[idx] for idx in permutation_ids]

        # splits each cluster in batch_size, and merge as a list of tensors
        splits = [c.split(self.batch_size) for c in permuted_clusters]
        merged = tuple(itertools.chain.from_iterable(splits))

        # now each batch internally has the right order, but
        # they are grouped by clusters. Find the permutation between
        # different batches that brings them as close as possible to
        # the order that we have in the sampler. For that, we will consider the
        # ordering as coming from the first element of each batch, and sort
        # correspondingly
        first_element_of_batch = [t[0].item() for t in merged]
        # get and inverse mapping from sampled indices and the position where
        # they occur (as returned by the sampler)
        inv_sampled_ids_map = {v: k for k, v in enumerate(sampled_ids.tolist())}
        # from the first element in each batch, get a relative ordering
        first_index_of_batch = torch.as_tensor(
            [inv_sampled_ids_map[s] for s in first_element_of_batch]
        )

        # permute the batches so that they approximately follow the order
        # from the sampler
        permutation_order = first_index_of_batch.sort(0)[1].tolist()
        # finally, permute the batches
        batches = [merged[i].tolist() for i in permutation_order]

        if self.drop_uneven:
            kept = []
            for batch in batches:
                if len(batch) == self.batch_size:
                    kept.append(batch)
            batches = kept
        return batches

    def __iter__(self):
        if self._can_reuse_batches:
            batches = self._batches
            self._can_reuse_batches = False
        else:
            batches = self._prepare_batches()
        self._batches = batches
        return iter(batches)

    def __len__(self):
        if not hasattr(self, "_batches"):
            self._batches = self._prepare_batches()
            self._can_reuse_batches = True
        return len(self._batches)


# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
class IterationBasedBatchSampler(BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations


class WeightedRandomSampler(Sampler):
    r"""Samples elements from [0,..,len(weights)-1] with given probabilities (weights).

    Note: Official WeightedRandomSampler implementation of PyTorch is very slow
          due to bad performance of torch.multinomial().
          Instead, this implementation uses numpy.random.choice().

    Arguments:
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
    """

    def __init__(self, weights, num_samples, replacement=True):
        if not isinstance(num_samples, _int_classes) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integeral "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.weights = weights / weights.sum()
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):
        return iter(np.random.choice(np.arange(len(self.weights)),
                                     size=self.num_samples,
                                     replace=self.replacement,
                                     p=self.weights))

    def __len__(self):
        return self.num_samples


class PseudoTripletSampler(Sampler):

    def __init__(self, class_ids, k=4):
        self.class_ids = class_ids
        self.k = k

    def __iter__(self):
        upper = len(self.class_ids) - len(self.class_ids) % k
        first_indexes = np.random.permutation(np.arange(0, upper, self.k))
        rand_indexes = np.concatenate(np.array([first_indexes + i for i in range(k)]).T)
        return iter(np.argsort(self.class_ids)[rand_indexes])

    def __len__(self):
        return len(self.class_ids)
