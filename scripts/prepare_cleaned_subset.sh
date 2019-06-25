#!/bin/bash

#### Download DELF features
if [[ ! -f experiments/delf/pretrained/delf_gld_20190411/model/saved_model.pb ]]; then
    mkdir -p experiments/delf/{train,test}
    mkdir -p experiments/delf/pretrained/
    wget http://storage.googleapis.com/delf/delf_gld_20190411.tar.gz \
        -O experiments/delf/pretrained/delf_gld_20190411.tar.gz
    cd experiments/delf/pretrained && \
        tar zxvf ./delf_gld_20190411.tar.gz
fi

python -c 'import delf'

#### Extract train and test features
python experiments/extract_delf_features.py

#### Matching (test-train and train-train)
python experiments/matching_localfeatures.py

### Generate cleaned subset of training set
python experiments/cleaning_train2019.py
