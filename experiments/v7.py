import json
import math
import os
import re
import time
from pathlib import Path

import click
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from albumentations import (
    HorizontalFlip,
    Compose,
    Normalize,
    RandomBrightness,
)
from sklearn.model_selection import ParameterGrid, ParameterSampler
from tqdm import tqdm

from src import qsub
from src import utils, data_utils, metrics
from src.modeling import models

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
MAX_CLASS_NUM = 14938 if ROOT == '/opt/landmark/' else 14951

params = {
    'ex_name': __file__.replace('.py', ''),
    'seed': 123456789,
    'lr': 1e-3,
    'batch_size': 32,
    'test_batch_size': 64,
    'optimizer': 'momentum',
    'epochs': 5,
    'wd': 1e-5,
    'model_name': 'fishnet150',
    'pooling': 'G,G,G,G',
    'class_topk': MAX_CLASS_NUM,
    'use_fc': True,
    'margin': 0.3,
    'fc_dim': 512,
    'scale_range': 0.1,
    'brightness_range': 0.1,
}


def build_transforms(mean=(0.485, 0.456, 0.406),
                     std=(0.229, 0.224, 0.225),
                     divide_by=255.0,
                     scale_range=0.1,
                     brightness_range=0.1):

    from src import torch_custom

    # CSAIL ResNet
    # norm = Normalize(mean=(102.9801, 115.9465, 122.7717), std=(1., 1., 1.), max_pixel_value=1, p=1.0)
    norm = Normalize(mean=mean, std=std, max_pixel_value=divide_by)

    train_transform = Compose([
        torch_custom.RandomCropThenScaleToOriginalSize(limit=scale_range, p=1.0),
        RandomBrightness(limit=brightness_range, p=0.5),
        HorizontalFlip(p=0.5),
        norm,
    ])
    eval_transform = Compose([
        norm,
    ])

    return train_transform, eval_transform


@click.group()
def cli():
    if not Path(ROOT + f'experiments/{params["ex_name"]}/train').exists():
        Path(ROOT + f'experiments/{params["ex_name"]}/train').mkdir(parents=True)
    if not Path(ROOT + f'experiments/{params["ex_name"]}/tuning').exists():
        Path(ROOT + f'experiments/{params["ex_name"]}/tuning').mkdir(parents=True)

    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    torch.cuda.manual_seed_all(params['seed'])
    torch.backends.cudnn.benchmark = False


@cli.command()
@click.option('--tuning', is_flag=True)
@click.option('--params-path', type=click.Path(), default=None, help='json file path for setting parameters')
@click.option('--devices', '-d', type=str, help='comma delimited gpu device list (e.g. "0,1")')
@click.option('--resume', type=str, default=None, help='checkpoint path')
@click.option('--save-interval', '-s', type=int, default=1, help='if 0 or negative value, not saving')
def job(tuning, params_path, devices, resume, save_interval):

    global params
    if tuning:
        with open(params_path, 'r') as f:
            params = json.load(f)
        mode_str = 'tuning'
        setting = '_'.join(f'{tp}-{params[tp]}' for tp in params['tuning_params'])
    else:
        mode_str = 'train'
        setting = ''

    exp_path = ROOT + f'experiments/{params["ex_name"]}/'
    os.environ['CUDA_VISIBLE_DEVICES'] = devices

    logger, writer = utils.get_logger(log_dir=exp_path + f'{mode_str}/log/{setting}',
                                      tensorboard_dir=exp_path + f'{mode_str}/tf_board/{setting}')
    train_transform, eval_transform = build_transforms(scale_range=params['scale_range'],
                                                       brightness_range=params['brightness_range'])
    data_loaders = data_utils.make_train_loaders(params=params,
                                                 data_root=ROOT + 'input/train2018',
                                                 train_transform=train_transform,
                                                 eval_transform=eval_transform,
                                                 class_topk=params['class_topk'],
                                                 num_workers=8)

    model = models.LandmarkFishNet(n_classes=params['class_topk'],
                                   model_name=params['model_name'],
                                   pooling_strings=params['pooling'].split(','),
                                   loss_module='arcface',
                                   s=30.0,
                                   margin=params['margin'],
                                   use_fc=params['use_fc'],
                                   fc_dim=params['fc_dim'],
                                   ).cuda()
    optimizer = utils.get_optim(params, model)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=params['epochs'] * len(data_loaders['train']), eta_min=1e-6)

    if len(devices.split(',')) > 1:
        model = nn.DataParallel(model)
    if resume is not None:
        model, optimizer = utils.load_checkpoint(path=resume, model=model, optimizer=optimizer)

    for epoch in range(params['epochs']):
        logger.info(f'Epoch {epoch}/{params["epochs"]} | lr: {optimizer.param_groups[0]["lr"]}')

        # ============================== train ============================== #
        model.train(True)

        losses = utils.AverageMeter()
        prec1 = utils.AverageMeter()

        for i, (_, x, y) in tqdm(enumerate(data_loaders['train']),
                                 total=len(data_loaders['train']),
                                 miniters=None, ncols=55):
            x = x.to('cuda')
            y = y.to('cuda')

            outputs = model(x, y)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            acc = metrics.accuracy(outputs, y)
            losses.update(loss.item(), x.size(0))
            prec1.update(acc, x.size(0))

            if i % 100 == 99:
                logger.info(f'{epoch+i/len(data_loaders["train"]):.2f}epoch | {setting} acc: {prec1.avg}')

        train_loss = losses.avg
        train_acc = prec1.avg

        # ============================== validation ============================== #
        model.train(False)
        losses.reset()
        prec1.reset()

        for i, (_, x, y) in tqdm(enumerate(data_loaders['val']),
                                 total=len(data_loaders['val']),
                                 miniters=None, ncols=55):
            x = x.to('cuda')
            y = y.to('cuda')

            with torch.no_grad():
                outputs = model(x, y)
                loss = criterion(outputs, y)

            acc = metrics.accuracy(outputs, y)
            losses.update(loss.item(), x.size(0))
            prec1.update(acc, x.size(0))

        val_loss = losses.avg
        val_acc = prec1.avg

        logger.info(f'[Val] Loss: \033[1m{val_loss:.4f}\033[0m | '
                    f'Acc: \033[1m{val_acc:.4f}\033[0m\n')

        writer.add_scalars('Loss', {'train': train_loss}, epoch)
        writer.add_scalars('Acc', {'train': train_acc}, epoch)
        writer.add_scalars('Loss', {'val': val_loss}, epoch)
        writer.add_scalars('Acc', {'val': val_acc}, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        if save_interval > 0:
            if (epoch + 1) == params['epochs'] or (epoch + 1) % save_interval == 0:
                output_file_name = exp_path + f'ep{epoch}_' + setting + '.pth'
                utils.save_checkpoint(path=output_file_name,
                                      model=model,
                                      epoch=epoch,
                                      optimizer=optimizer,
                                      params=params)

    if tuning:
        tuning_result = {}
        for key in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
            tuning_result[key] = [eval(key)]
        utils.write_tuning_result(params, tuning_result, exp_path + 'tuning/results.csv')


@cli.command()
@click.option('--mode', type=str, default='grid', help='Search method (tuning)')
@click.option('--n-iter', type=int, default=10, help='n of iteration for random parameter search (tuning)')
@click.option('--n-gpu', type=int, default=-1, help='n of used gpu at once')
@click.option('--devices', '-d', type=str, help='comma delimited gpu device list (e.g. "0,1")')
@click.option('--save-interval', '-s', type=int, default=-1, help='if 0 or negative value, not saving')
@click.option('--n-blocks', '-n', type=int, default=1)
@click.option('--block-id', '-i', type=int, default=0)
def tuning(mode, n_iter, n_gpu, devices, save_interval, n_blocks, block_id):

    if n_gpu == -1:
        n_gpu = len(devices.split(','))

    space = [{
        'pooling': ['G,G,G,G'],
        'epochs': [5],
        'batch_size': [32],
        # 'fc_dim': [128, 256, 512],
    }]

    if mode == 'grid':
        candidate_list = list(ParameterGrid(space))
    elif mode == 'random':
        candidate_list = list(ParameterSampler(space, n_iter, random_state=params['seed']))
    else:
        raise ValueError

    n_per_block = math.ceil(len(candidate_list) / n_blocks)
    candidate_chunk = candidate_list[block_id*n_per_block: (block_id+1)*n_per_block]

    utils.launch_tuning(mode=mode, n_iter=n_iter, n_gpu=n_gpu, devices=devices,
                        params=params, root=ROOT, save_interval=save_interval,
                        candidate_list=candidate_chunk)


@cli.command()
@click.option('--model-path', '-m', type=str)
@click.option('--devices', '-d', type=str, default="0", help='comma delimited gpu device list (e.g. "0,1")')
@click.option('--ms', is_flag=True)
@click.option('--scale', type=str, default='S')
@click.option('--batch-size', '-b', type=int, default=64)
@click.option('--splits', type=str, default='index,test')
@click.option('--n-blocks', '-n', type=int, default=1)
@click.option('--block-id', '-i', type=int, default=0)
def predict(model_path, devices, ms, scale, batch_size, splits, n_blocks, block_id):

    os.environ['CUDA_VISIBLE_DEVICES'] = devices

    ckpt = torch.load(model_path)
    params, state_dict = ckpt['params'], ckpt['state_dict']
    params['test_batch_size'] = batch_size

    splits = splits.split(',')

    model = models.LandmarkFishNet(n_classes=params['class_topk'],
                                   model_name=params['model_name'],
                                   pooling_strings=params['pooling'].split(','),
                                   loss_module='arcface',
                                   s=30.0,
                                   margin=params['margin'],
                                   use_fc=params['use_fc'],
                                   fc_dim=params['fc_dim'],
                                   )
    model.load_state_dict(state_dict)
    model = model.to('cuda').eval()

    train_transform, eval_transform = build_transforms()
    data_loaders = data_utils.make_predict_loaders(params,
                                                   eval_transform=eval_transform,
                                                   scale=scale,
                                                   splits=splits,
                                                   num_workers=8,
                                                   n_blocks=n_blocks,
                                                   block_id=block_id)

    exp_path = ROOT + f'experiments/{params["ex_name"]}/'

    file_suffix = model_path.split('/')[-1].replace('.pth', '')
    file_suffix = scale + '_' + file_suffix
    file_suffix = 'ms_' + file_suffix if ms else file_suffix

    scales = [0.75, 1.0, 1.25] if ms else [1.0]

    for split in splits:
        ids, feats = [], []
        for i, (img_id, x) in tqdm(enumerate(data_loaders[split]),
                                   total=len(data_loaders[split]),
                                   miniters=None, ncols=55):

            batch_size, _, h, w = x.shape
            feat_blend = np.zeros((batch_size, params['fc_dim']), dtype=np.float32)

            with torch.no_grad():
                x = x.to('cuda')

                for s in scales:
                    size = int(h * s // model.DIVIDABLE_BY * model.DIVIDABLE_BY),\
                           int(w * s // model.DIVIDABLE_BY * model.DIVIDABLE_BY)  # round off
                    scaled_x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
                    feat = model.extract_feat(scaled_x)
                    feat = feat.cpu().numpy()
                    feat_blend += feat

            feats.append(feat_blend)
            ids.extend(img_id)

        feats = np.concatenate(feats) / len(scales)

        output_path = Path(f'{exp_path}feats_{split}_{file_suffix}')
        output_path.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_path / f'block{block_id}.h5', 'a') as f:
            f.create_dataset('ids', data=np.array(ids, dtype=f'S{len(ids[0])}'))
            f.create_dataset('feats', data=feats)


@cli.command()
@click.argument('job-type')
@click.option('--n-gpu', type=int, default=-1, help='n of used gpu at once')
@click.option('--devices', '-d', type=str, help='comma delimited gpu device list (e.g. "0,1")')
@click.option('--save-interval', '-s', type=int, default=-1, help='if 0 or negative value, not saving')
@click.option('--model-path', '-m', default=None, type=str)
@click.option('--ms', is_flag=True)
@click.option('--scale', type=str, default='S')
@click.option('--batch-size', '-b', type=int, default=64)
@click.option('--splits', type=str, default='index,test')
@click.option('--n-blocks', '-n', type=int, default=1)
@click.option('--instance-type', type=str, default='rt_G.small')
def launch_qsub(job_type,
                n_gpu, devices, save_interval,  # tuning args
                model_path, ms, scale, batch_size, splits,  # predict args
                n_blocks, instance_type
                ):

    exp_path = ROOT + f'experiments/{params["ex_name"]}/'
    logger = utils.get_logger(log_dir=exp_path)
    job_ids = []
    for block_id in range(n_blocks):
        if job_type == 'tuning':
            cmd_with_args = [
                "python", "-W", "ignore", "v7.py", "tuning",
                "--n-gpu", str(n_gpu),
                "--devices", devices,
                "--save-interval", str(save_interval),
                "--n-blocks", str(n_blocks),
                "--block-id", str(block_id),
            ]
        elif job_type == 'predict':
            cmd_with_args = [
                "python", "-W", "ignore", "v7.py", "predict",
                "-m", model_path,
                "--splits", splits,
                "--ms" if ms else "",
                "--scale", scale,
                "--batch-size", str(batch_size),
                "--n-blocks", str(n_blocks),
                "--block-id", str(block_id),
            ]
        else:
            raise ValueError('job-type should be one of "tuning" or "predict"')
        proc = qsub.qsub(cmd_with_args,
                         n_hours=24,
                         instance_type=instance_type,
                         logger=logger)
        logger.info(f'Response from qsub: {proc.returncode}')

        m = re.match(r'Your job (\d+) \(', proc.stdout.decode('utf8'))
        job_id = int(m.group(1)) if m is not None else None
        logger.info(f'Job id: {job_id}')
        assert job_id is not None
        job_ids.append(job_id)
        time.sleep(1)

    qsub.monitor_jobs(job_ids, logger)


if __name__ == '__main__':
    cli()
