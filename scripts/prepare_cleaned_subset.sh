#!/bin/bash

#### Download DELF features
if [[ ! -f exp/delf/pretrained/delf_gld_20190411/model/saved_model.pb ]]; then
    mkdir -p exp/delf/{train,test}
    mkdir -p exp/delf/pretrained/
    wget http://storage.googleapis.com/delf/delf_gld_20190411.tar.gz \
        -O exp/delf/pretrained/delf_gld_20190411.tar.gz
    cd exp/delf/pretrained && \
        tar zxvf ./delf_gld_20190411.tar.gz
fi

python -c 'import delf'

#### Extract train and test features
python exp/delf/extract_delf_features.py

#### Matching (test-train and train-train)
python exp/delf/matching_localfeatures.py

### Generate cleaned subset of training set
python exp/delf/cleaning_train2019.py
