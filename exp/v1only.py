import json
import math
import os
import subprocess
import re
import time
from pathlib import Path

import click
import h5py
import numpy as np
from sklearn.model_selection import ParameterGrid, ParameterSampler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm

from src import qsub
from src import utils, data_utils, metrics
from src.modeling import models
from src.eval_retrieval import eval_datasets

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'

params = {
    'ex_name': __file__.replace('.py', ''),
    'seed': 123456789,
    'lr': 1e-3,
    'batch_size': 32,
    'test_batch_size': 64,
    'optimizer': 'momentum',
    'epochs': 5,
    'wd': 1e-5,
    'model_name': 'resnet101',
    'pooling': 'GeM',
    'class_topk': 14950,
    'use_fc': True,
    'loss': 'arcface',
    'margin': 0.3,
    's': 30,
    'theta_zero': 1.25,
    'fc_dim': 512,
    'scale_limit': 0.2,
    'shear_limit': 0,
    'brightness_limit': 0.0,
    'contrast_limit': 0.0,
    'augmentation': 'soft',
    'train_data': 'gld_v1',
}


@click.group()
def cli():
    if not Path(ROOT + f'exp/{params["ex_name"]}/train').exists():
        Path(ROOT + f'exp/{params["ex_name"]}/train').mkdir(parents=True)
    if not Path(ROOT + f'exp/{params["ex_name"]}/tuning').exists():
        Path(ROOT + f'exp/{params["ex_name"]}/tuning').mkdir(parents=True)

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

    exp_path = ROOT + f'exp/{params["ex_name"]}/'
    os.environ['CUDA_VISIBLE_DEVICES'] = devices

    logger, writer = utils.get_logger(log_dir=exp_path + f'{mode_str}/log/{setting}',
                                      tensorboard_dir=exp_path + f'{mode_str}/tf_board/{setting}')

    if params['augmentation'] == 'soft':
        params['scale_limit'] = 0.2
        params['brightness_limit'] = 0.1
    elif params['augmentation'] == 'middle':
        params['scale_limit'] = 0.3
        params['shear_limit'] = 4
        params['brightness_limit'] = 0.1
        params['contrast_limit'] = 0.1
    else:
        raise ValueError

    train_transform, eval_transform = data_utils.build_transforms(
        scale_limit=params['scale_limit'],
        shear_limit=params['shear_limit'],
        brightness_limit=params['brightness_limit'],
        contrast_limit=params['contrast_limit'],
    )

    data_loaders = data_utils.make_train_loaders(params=params,
                                                 data_root=ROOT + 'input/' + params['train_data'],
                                                 train_transform=train_transform,
                                                 eval_transform=eval_transform,
                                                 scale='S2',
                                                 test_size=0,
                                                 class_topk=params['class_topk'],
                                                 num_workers=8)

    model = models.LandmarkNet(n_classes=params['class_topk'],
                               model_name=params['model_name'],
                               pooling=params['pooling'],
                               loss_module=params['loss'],
                               s=params['s'],
                               margin=params['margin'],
                               theta_zero=params['theta_zero'],
                               use_fc=params['use_fc'],
                               fc_dim=params['fc_dim'],
                               ).cuda()
    optimizer = utils.get_optim(params, model)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=params['epochs'] * len(data_loaders['train']), eta_min=3e-6)
    start_epoch = 0

    if len(devices.split(',')) > 1:
        model = nn.DataParallel(model)

    for epoch in range(start_epoch, params['epochs']):

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

        writer.add_scalars('Loss', {'train': train_loss}, epoch)
        writer.add_scalars('Acc', {'train': train_acc}, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        if (epoch + 1) == params['epochs'] or (epoch + 1) % save_interval == 0:
            output_file_name = exp_path + f'ep{epoch}_' + setting + '.pth'
            utils.save_checkpoint(path=output_file_name,
                                  model=model,
                                  epoch=epoch,
                                  optimizer=optimizer,
                                  params=params)

    model = model.module
    datasets = ('oxford5k', 'paris6k', 'roxford5k', 'rparis6k')
    results = eval_datasets(model, datasets=datasets, ms=True, tta_gem_p=1.0, logger=logger)

    if tuning:
        tuning_result = {}
        for d in datasets:
            if d in ('oxford5k', 'paris6k'):
                tuning_result[d] = results[d]
            else:
                for key in ['mapE', 'mapM', 'mapH']:
                    mapE, mapM, mapH, mpE, mpM, mpH, kappas = results[d]
                    tuning_result[d + '-' + key] = [eval(key)]
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

    space = [
        {
            # 'loss': ['arcface', 'cosface'],
            'loss': ['arcface', 'cosface', 'softmax'],
            'epochs': [5],
            'augmentation': ['soft'],
        },
    ]

    if mode == 'grid':
        candidate_list = list(ParameterGrid(space))
    elif mode == 'random':
        candidate_list = list(ParameterSampler(space, n_iter, random_state=params['seed']))
    else:
        raise ValueError

    n_per_block = math.ceil(len(candidate_list) / n_blocks)
    candidate_chunk = candidate_list[block_id * n_per_block: (block_id + 1) * n_per_block]

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

    model = models.LandmarkNet(n_classes=params['class_topk'],
                               model_name=params['model_name'],
                               pooling=params['pooling'],
                               loss_module=params['loss'],
                               s=params['s'],
                               margin=params['margin'],
                               theta_zero=params['theta_zero'],
                               use_fc=params['use_fc'],
                               fc_dim=params['fc_dim'],
                               )
    model.load_state_dict(state_dict)
    model = model.to('cuda').eval()

    train_transform, eval_transform = data_utils.build_transforms()
    data_loaders = data_utils.make_predict_loaders(params,
                                                   data_root=ROOT + 'input/gld_v2',
                                                   eval_transform=eval_transform,
                                                   scale=scale,
                                                   splits=splits,
                                                   num_workers=8,
                                                   n_blocks=n_blocks,
                                                   block_id=block_id)

    exp_path = ROOT + f'exp/{params["ex_name"]}/'

    file_suffix = model_path.split('/')[-1].replace('.pth', '')
    file_suffix = scale + '_' + file_suffix
    file_suffix = 'ms_' + file_suffix if ms else file_suffix

    min_size = 128
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
                    th = max(min_size, int(h * s // model.DIVIDABLE_BY * model.DIVIDABLE_BY))
                    tw = max(min_size, int(w * s // model.DIVIDABLE_BY * model.DIVIDABLE_BY))  # round off

                    scaled_x = F.interpolate(x, size=(th, tw), mode='bilinear', align_corners=True)
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
@click.option('--mode', type=str, default='grid', help='Search method (tuning)')
@click.option('--n-iter', type=int, default=10, help='n of iteration for random parameter search (tuning)')
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
                mode, n_iter, n_gpu, devices, save_interval,  # tuning args
                model_path, ms, scale, batch_size, splits,  # predict args
                n_blocks, instance_type
                ):
    exp_path = ROOT + f'exp/{params["ex_name"]}/'
    logger = utils.get_logger(log_dir=exp_path)
    job_ids = []
    for block_id in range(n_blocks):
        if job_type == 'tuning':
            cmd_with_args = [
                "python", "-W", "ignore", __file__, "tuning",
                "--mode", mode,
                "--n-iter", str(n_iter),
                "--n-gpu", str(n_gpu),
                "--devices", devices,
                "--save-interval", str(save_interval),
                "--n-blocks", str(n_blocks),
                "--block-id", str(block_id),
            ]
            n_hours = 48
        elif job_type == 'predict':
            cmd_with_args = [
                "python", "-W", "ignore", __file__, "predict",
                "-m", model_path,
                "--splits", splits,
                "--scale", scale,
                "--batch-size", str(batch_size),
                "--n-blocks", str(n_blocks),
                "--block-id", str(block_id),
            ]
            n_hours = 4
            if ms:
                cmd_with_args.append("--ms")
        else:
            raise ValueError('job-type should be one of "tuning" or "predict"')
        proc = qsub.qsub(cmd_with_args,
                         n_hours=n_hours,
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


@cli.command()
@click.option('--devices', '-d', type=str, help='comma delimited gpu device list (e.g. "0,1")')
@click.option('--model-path', '-m', default=None, type=str)
@click.option('--ms', is_flag=True)
@click.option('--scale', type=str, default='S')
@click.option('--batch-size', '-b', type=int, default=64)
@click.option('--splits', type=str, default='index,test')
def multigpu_predict(devices, model_path, ms, scale, batch_size, splits):

    devices = devices.split(',')

    procs = []
    for block_id, d in enumerate(devices):
        cmd_with_args = [
            "python", __file__, "predict",
            "-d", d,
            "-m", model_path,
            "--splits", splits,
            "--scale", scale,
            "--batch-size", str(batch_size),
            "--n-blocks", str(len(devices)),
            "--block-id", str(block_id),
        ]
        if ms:
            cmd_with_args.append("--ms")
        procs.append(subprocess.Popen(cmd_with_args))

    while True:
        time.sleep(1)
        if all(proc.poll() is not None for proc in procs):
            print('All jobs have finished.')
            break


if __name__ == '__main__':
    cli()
