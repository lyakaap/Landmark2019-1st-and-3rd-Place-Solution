# Landmark2019-1st-and-3rd-Place-Solution

The 1st Place Solution of the Google Landmark 2019 Retrieval Challenge and the 3rd Place Solution of the Recognition Challenge.

**NOTE**: This solution code is not refactored and work in progress at this time. Stay tuned!

Our solution was published! You can check from: [Large-scale Landmark Retrieval/Recognition under a Noisy and Diverse Dataset](https://arxiv.org/abs/1906.04087)

### Reproduce
Following commands are for reproducing our results.
```
bash donwload_train.sh # download data
bash setup.sh  # setup data to ready training
bash reproduce.sh  # train models and predict for reproducing
```

### Reference
* https://github.com/filipradenovic/cnnimageretrieval-pytorch/tree/master/cirtorch
* https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py
* https://github.com/4uiiurz1/pytorch-adacos/blob/master/metrics.py
* https://github.com/kevin-ssy/FishNet
* https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/data/samplers/grouped_batch_sampler.py
