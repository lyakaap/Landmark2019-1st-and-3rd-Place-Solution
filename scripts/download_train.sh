#!/usr/bin/env bash

WORKERS=16

wget https://s3.amazonaws.com/google-landmark/metadata/train.csv

for i in `seq 0 499`
do
    echo https://s3.amazonaws.com/google-landmark/train/images_$(printf "%03d" $i).tar >> url_list.txt
done

xargs -a url_list.txt -L 1 -P $WORKERS wget -c --tries=0

ls -1 | grep tar | xargs -L 1 -P $WORKERS tar -xvf
ls -1 | grep tar | xargs -L 1 -P $WORKERS rm


# download the train set of v1 dataset
cd ./src/
python donwloader.py download

# download 'oxford5k', 'paris6k', 'roxford5k', 'rparis6k' for evaluation
cd ..
python3 -m cirtorch.examples.download_test
