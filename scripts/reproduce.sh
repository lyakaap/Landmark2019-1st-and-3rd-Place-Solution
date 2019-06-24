#!/usr/bin/env bash

cd ./src/
python donwloader.py download

cd ../experiments/

# prepare model for later dataset cleaning & finetuning models of v7c
python v7.py tuning -d 0,1 --n-gpu 2 -s 4
python v7.py predict -d 0 --scale M --ms -b 32 --splits train,train2018_r800

# FishNet-150
python v19.py tuning -d 0,1,2,3,4,5,6,7 --n-gpu 2 -s 4
python v19c.py tuning -d 0,1,2,3,4,5,6,7 --n-gpu 2 -s 4
python v19c.py job --resume v19c/ep3_freqthresh-2_loss-cosface_pooling-G,G,G,G_verifythresh-30.pth -d 0,1,2,3
python v19c.py multigpu-predict -b 24 -d 0,1 --scale L2 --ms --splits train,test19,index19 -m v19c/ep4_scaleup_ep3_freqthresh-2_loss-cosface_pooling-G,G,G,G_verifythresh-30.pth

# SE-ResNeXt-101
python v20.py tuning -d 0,1,2,3 --n-gpu 2 -s 4
python v20c.py tuning -d 0,1,2,3 --n-gpu 2 -s 4
python v20c.py multigpu-predict -b 24 -d 0,1 --scale L2 --ms --splits train,test19,index19 -m v20c/ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30.pth

# ResNet-101
python v21.py tuning -d 0,1,2,3 --n-gpu 2 -s 4
python v21c.py tuning -d 0,1,2,3 --n-gpu 2 -s 4
python v21c.py job --resume v21c/ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30.pth -d 0,1,2,3
python v21c.py multigpu-predict -b 24 -d 0,1 -m v21c/ep6_scaleup_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30.pth --scale L2 --ms --splits test19,index19,train

# FishNet-150 preatrain with clean train18
python v22.py tuning -d 0,1,2,3 --n-gpu 2 -s 4
python v22c.py tuning -d 0,1,2,3 --n-gpu 2 -s 4
python v22c.py job --resume v22c/ep3_base_margin-0.4_freqthresh-2_verifythresh-30.pth -d 0,1,2,3
python v22c.py multigpu-predict -b 24 -d 0,1 -m v22c/ep4_scaleup_ep3_base_margin-0.3_freqthresh-2_verifythresh-20.pth --scale L2 --ms --splits test19,index19,train

# FishNet-150 pretrain with train18, freq=2,verify=30, augmentation=middle, 7epochs
python v23.py tuning -d 0,1,2,3 --n-gpu 2 -s 4
python v23c.py tuning -d 0,1,2,3 --n-gpu 2 -s 4
python v23c.py job --resume v23c/ep6_scaleup_ep5_augmentation-middle_epochs-7_freqthresh-3_verifythresh-30.pth -d 0,1,2,3
python v23c.py multigpu-predict -b 24 -d 0,1 -m v23c/ep6_scaleup_ep5_augmentation-middle_epochs-7_freqthresh-3_verifythresh-30.pth --scale L2 --ms --splits test19,index19,train

# SE-ResNeXt-101 CosFace
python v24.py tuning -d 0,1,2,3 --n-gpu 4 -s 1
python v24c.py tuning -d 0,1 --n-gpu 2 -s 1
python v24c.py multigpu-predict -b 24 -d 0,1 -m v24c/ep5_augmentation-middle_epochs-7_freqthresh-3_loss-cosface_verifythresh-30.pth --scale L2 --ms --splits test19,index19,train
