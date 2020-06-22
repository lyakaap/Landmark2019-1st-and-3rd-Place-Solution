# Landmark2019-1st-and-3rd-Place-Solution

![pipeline](https://user-images.githubusercontent.com/27487010/69476665-0858c880-0e20-11ea-9fb4-5292f61e9c12.png)

The 1st Place Solution of the Google Landmark 2019 Retrieval Challenge and the 3rd Place Solution of the Recognition Challenge.

We have published two papers regarding our solution. You can check from:
- [Large-scale Landmark Retrieval/Recognition under a Noisy and Diverse Dataset](https://arxiv.org/abs/1906.04087)
  - Tech report describing our solution (4-pages)
- [Two-stage Discriminative Re-ranking for Large-scale Landmark Retrieval](https://arxiv.org/abs/2003.11211)
  - Extended version of above tech report, including more detailed explanation and additional experimental results (10-pages, CVPR Workshop 2020)

### Environments
You can reproduce our environments using Dockerfile provided here https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/blob/master/docker/Dockerfile

### Data
* Google Landmark Dataset v1 (GLD-v1): https://www.kaggle.com/google/google-landmarks-dataset
* Google Landmark Dataset v2 (GLD-v2): https://github.com/cvdfoundation/google-landmark
* **Clean version of the v2 (Newly Released!)**: https://www.kaggle.com/confirm/cleaned-subsets-of-google-landmarks-v2/kernels

Dataset statistics:

| Dataset (train split) | # Samples  | # Labels  |
|-----------------------|------------|--------------|
| GLD-v1   | 1,225,029  | 14,951       |
| GLD-v2   | 4,132,914  | 203,094      |
| GLD-v2 (clean) | 1,580,470  | 81,313       |

### Prepare cleaned subset
(You can skip this procedure to generate a cleaned subset.
Pre-computed files are available on [kaggle dataset](https://www.kaggle.com/confirm/cleaned-subsets-of-google-landmarks-v2).)

Run `scripts/prepare_cleaned_subset.sh` for cleaning the GLD-v2 dataset.
The cleaning code requires DELF library ([install instructions](https://github.com/tensorflow/models/blob/master/research/delf/INSTALL_INSTRUCTIONS.md)).

### exp
Model training and inference are done in `exp/` directory.
```
# train models by various parameter settings with 4 gpus (each training is done with 2 gpus).
python vX.py tuning -d 0,1,2,3 --n-gpu 2

# predict
python vX.py predict -m vX/epX.pth -d 0
# predict with multiple gpus
python vX.py multigpu-predict -m vX/epX.pth --scale L2 --ms -b 32 -d 0,1
```

### Results (retrieval challenge)
| Place | Team        |     Private    |     Public     |
|-------|-------------|:--------------:|:--------------:|
| 1st   | smlyaka (ours)    | **37.23** | **35.69** |
| 2nd   | imagesearch |      34.75     |      32.25     |
| 3rd   | Layer 6 AI  |      32.18     |      29.85     |


### Reference
* https://github.com/filipradenovic/cnnimageretrieval-pytorch/tree/master/cirtorch
* https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py
* https://github.com/4uiiurz1/pytorch-adacos/blob/master/metrics.py
* https://github.com/kevin-ssy/FishNet
* https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/data/samplers/grouped_batch_sampler.py

