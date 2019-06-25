# Landmark2019-1st-and-3rd-Place-Solution

The 1st Place Solution of the Google Landmark 2019 Retrieval Challenge and the 3rd Place Solution of the Recognition Challenge.

**NOTE**: This solution code is not refactored and work in progress at this time. Stay tuned!

Our solution has been published! You can check from: [Large-scale Landmark Retrieval/Recognition under a Noisy and Diverse Dataset](https://arxiv.org/abs/1906.04087)

### Data
* Google Landmark Dataset v1: https://www.kaggle.com/google/google-landmarks-dataset
* Google Landmark Dataset v2: https://github.com/cvdfoundation/google-landmark
* Clean version of the v2: https://www.kaggle.com/confirm/cleaned-subsets-of-google-landmarks-v2/kernels

### Experiments
Model training and inference are done in `experiments/` directory.
```
# train models by various parameter settings with 4 gpus (each training is done with 2 gpus).
python vX.py tuning -d 0,1,2,3 --n-gpu 2

# predict
python vX.py predict -m vX/epX.pth -d 0
# predict with multiple gpus
python vX.py multigpu-predict -m vX/epX.pth --scale L2 --ms -b 32 -d 0,1

# ABCI command
python vX.py launch-qsub tuning -d 0,1,2,3 --n-gpu 2 --n-blocks 2 -s 1 --instance-type rt_F
python vX.py launch-qsub predict -m vX/ep --ms --scale L2 --batch-size 24 --splits train,test --n-blocks 64
```

### Prepare cleaned subset
You can skip this procedure to generate a cleaned subset.
Pre-computed files are available on [kaggle dataset](https://www.kaggle.com/confirm/cleaned-subsets-of-google-landmarks-v2).

To be able to use this code, please follow [these instructions](https://github.com/tensorflow/models/blob/master/research/delf/INSTALL_INSTRUCTIONS.md) to properly install the DELF library.
```
bash scripts/prepare_cleaned_subset.sh
```

### Reproduce
Following commands are for reproducing our results.
```
cd ./experiments/
bash donwload_train.sh # download data
bash setup.sh  # setup data to ready training
bash reproduce.sh  # train models and predict for reproducing

python submit_retrieval.py  # for retrieval challenge
python submit_recognition.py  # for retrieval challenge
```

### Reference
* https://github.com/filipradenovic/cnnimageretrieval-pytorch/tree/master/cirtorch
* https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py
* https://github.com/4uiiurz1/pytorch-adacos/blob/master/metrics.py
* https://github.com/kevin-ssy/FishNet
* https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/data/samplers/grouped_batch_sampler.py
