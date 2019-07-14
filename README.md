# Landmark2019-1st-and-3rd-Place-Solution

The 1st Place Solution of the Google Landmark 2019 Retrieval Challenge and the 3rd Place Solution of the Recognition Challenge.

Our solution has been published! You can check from: [Large-scale Landmark Retrieval/Recognition under a Noisy and Diverse Dataset](https://arxiv.org/abs/1906.04087)

### Environments
You can reproduce our environments using Dockerfile provided here https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/blob/master/docker/Dockerfile

### Data
* Google Landmark Dataset v1 (GLD-v1): https://www.kaggle.com/google/google-landmarks-dataset
* Google Landmark Dataset v2 (GLD-v2): https://github.com/cvdfoundation/google-landmark
* **Clean version of the v2 (Newly Released!)**: https://www.kaggle.com/confirm/cleaned-subsets-of-google-landmarks-v2/kernels

Dataset statistics:

| Dataset (train split) | # Samples  | # Landmarks  |
|-----------------------|------------|--------------|
| GLD-v1   | 1,225,029  | 14,951       |
| GLD-v2   | 4,132,914  | 203,094      |
| GLD-v2 (clean) | 1,580,470  | 81,313       |

### Prepare cleaned subset
(You can skip this procedure to generate a cleaned subset.
Pre-computed files are available on [kaggle dataset](https://www.kaggle.com/confirm/cleaned-subsets-of-google-landmarks-v2).)

Run `scripts/prepare_cleaned_subset.sh` for cleaning the GLD-v2 dataset.
The cleaning code requires DELF library ([install instructions](https://github.com/tensorflow/models/blob/master/research/delf/INSTALL_INSTRUCTIONS.md)).

### Reproduce
Prepare FishNet pretrained checkpoints first.
1. Download the checkpoint of FishNet-150 from https://www.dropbox.com/s/ajy9p6f97y45f1r/fishnet150_ckpt_welltrained.tar?dl=0
2. Place the checkpoint `src/FishNet/checkpoints/`
3. Execute src/FishNet/fix_checkpoint.py

Following commands are for reproducing our results.
```
cd ./experiments/
bash donwload_train.sh # download data
bash setup.sh  # setup data to ready training
bash reproduce.sh  # train models and predict for reproducing

python submit_retrieval.py  # for retrieval challenge
python submit_recognition.py  # for retrieval challenge
```

### Experiments
Model training and inference are done in `experiments/` directory.
```
# train models by various parameter settings with 4 gpus (each training is done with 2 gpus).
python vX.py tuning -d 0,1,2,3 --n-gpu 2

# predict
python vX.py predict -m vX/epX.pth -d 0
# predict with multiple gpus
python vX.py multigpu-predict -m vX/epX.pth --scale L2 --ms -b 32 -d 0,1
```

### Results

* Comparison among loss functions with a ResNet-101. Dataset used for training is GLD-v1.
For evaluation, GLD-v2, ROxford-5K, and RParis-6K are used. LB shows the Private LB scores.

| loss        | LB | ROxf (M) | ROxf (H) | RPar (M) | RPar (H) |
|-------------|-------|----------------|----------------|---------------|---------------|
| softmax     | 17.95 | 66.73          | 43.5           | 79.25         | 58.58         |
| contrastive | 20    | 67.3           | 44.3           | 80.6          | 61.5          |
| cosface     | **21.35** | **69.49**  | 44.78          | 80.06         | 62.95         |
| arcface     | 20.74 | 69.04          | **46.25**          | **82.35**         | **66.62**         |

* Comparison with dataset used for training. ArcFace and ResNet-101 are used for this experiment.
By using the clean version of the GLD-v2, we can obtain better results compared to GLD-v2 without cleaning.

| Dataset       | LB    | ROxf (M) | ROxf (H) | RPar (M) | RPar (H) |
|---------------|-------|----------|----------|----------|----------|
| v1            | 20.74 | 69.04    | 46.25    | 82.35    | 66.62    |
| v1 + v2       | 29.2  | 77       | 56.59    | 89.25    | 77.35    |
| v1 + v2-clean | **30.22** | **78.86**    | **59.93**    | 88.84    | **77.82**    |

* Our re-ranking method is much better than other recent re-ranking methods, and can be combined with the others for even higher scores.
A combination of ours and αQE achieves a competitive result to the score of 1st place with only a single model.


| Method               | Private LB | Public LB |
|----------------------|------------|-----------|
| baseline             | 30.22      | 27.81     |
| + Diffusion (k=1000) | 31.2       | 29.36     |
| + EGT (t=inf, k=100) | 30.33      | 28.44     |
| + αQE (α=3.0)        | 32.21      | 30.34     |
| + Ours               | 36.77      | 34.88     |
| + αQE+Ours           | **37.23**      | **35.5**      |

### Reference
* https://github.com/filipradenovic/cnnimageretrieval-pytorch/tree/master/cirtorch
* https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py
* https://github.com/4uiiurz1/pytorch-adacos/blob/master/metrics.py
* https://github.com/kevin-ssy/FishNet
* https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/data/samplers/grouped_batch_sampler.py

