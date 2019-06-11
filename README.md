# Landmark2019-1st-and-3rd-Place-Solution

Google Landmark 2019 1st Place Solution of Retrieval Challenge and 3rd Place Solution of Recognition Challenge.
Our solution was published! You can check from here:
https://arxiv.org/abs/1906.04087

Following commands are for reproduce our results.
```
bash donwload_train.sh # download data
bash setup.sh  # setup data to ready training
bash reproduce.sh  # train models and predict for reproducing
```

We borrow idea and code from these grate repositories,

* https://github.com/filipradenovic/cnnimageretrieval-pytorch/tree/master/cirtorch
* https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py
* https://github.com/4uiiurz1/pytorch-adacos/blob/master/metrics.py
* https://github.com/kevin-ssy/FishNet
* https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/data/samplers/grouped_batch_sampler.py

A List of papers related to Landmark Competition is here (Japanese),
https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/blob/master/retrieval_paper_memo.md
