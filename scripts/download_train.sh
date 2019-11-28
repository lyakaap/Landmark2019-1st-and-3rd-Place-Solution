#!/usr/bin/env bash

WORKERS=16

mkdir ../input/gld_v2/{train,index,test}
wget https://s3.amazonaws.com/google-landmark/metadata/train.csv -O ../input/gld_v2/train.csv
wget https://s3.amazonaws.com/google-landmark/metadata/index.csv -O ../input/gld_v2/index.csv
wget https://s3.amazonaws.com/google-landmark/metadata/test.csv -O ../input/gld_v2/test.csv

for i in `seq 0 499`
do
    echo https://s3.amazonaws.com/google-landmark/train/images_$(printf "%03d" $i).tar >> url_list_train.txt
done

for i in `seq 0 99`
do
    echo https://s3.amazonaws.com/google-landmark/index/images_$(printf "%03d" $i).tar >> url_list_index.txt
done

for i in `seq 0 19`
do
    echo https://s3.amazonaws.com/google-landmark/test/images_$(printf "%03d" $i).tar >> url_list_test.txt
done

for split in "train" "index" "test"
do
    xargs -a url_list_${split}.txt -L 1 -P $WORKERS wget -c --tries=0
    ls -1 | grep tar | xargs -IXXX -P $WORKERS tar -xvf XXX -C ../input/gld_v2/${split}
    # ls -1 | grep tar | xargs -L 1 -P $WORKERS rm
done


# download the train set of v1 dataset
kaggle datasets download -d google/google-landmarks-dataset
unzip google-landmarks-dataset.zip
mv train.csv ../input/gld_v1/
cd ../src/
python donwloader.py download