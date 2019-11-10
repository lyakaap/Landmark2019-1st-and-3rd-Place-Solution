#!/usr/bin/env bash

WORKERS=16

mkdir ../input/{train,index19,test19}
wget https://s3.amazonaws.com/google-landmark/metadata/train.csv -O ../input/train.csv
wget https://s3.amazonaws.com/google-landmark/metadata/index.csv -O ../input/index19.csv
wget https://s3.amazonaws.com/google-landmark/metadata/test.csv -O ../input/test19.csv

for i in `seq 0 499`
do
    echo https://s3.amazonaws.com/google-landmark/train/images_$(printf "%03d" $i).tar >> url_list_train.txt
done

for i in `seq 0 99`
do
    echo https://s3.amazonaws.com/google-landmark/index/images_$(printf "%03d" $i).tar >> url_list_index19.txt
done

for i in `seq 0 19`
do
    echo https://s3.amazonaws.com/google-landmark/test/images_$(printf "%03d" $i).tar >> url_list_test19.txt
done

for split in "train" "index19" "test19"
do
    xargs -a url_list_${split}.txt -L 1 -P $WORKERS wget -c --tries=0
    ls -1 | grep tar | xargs -IXXX -P $WORKERS tar -xvf XXX -C ../input/${split}
    # ls -1 | grep tar | xargs -L 1 -P $WORKERS rm
done


# download the train set of v1 dataset
kaggle datasets download -d google/google-landmarks-dataset
unzip google-landmarks-dataset.zip
mv train.csv ../input/train2018.csv
cd ../src/
python donwloader.py download
python prepare_dataset.py

# download 'oxford5k', 'paris6k', 'roxford5k', 'rparis6k' for evaluation
cd ..
python3 -m cirtorch.examples.download_test
