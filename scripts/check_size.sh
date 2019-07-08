#!/usr/bin/env bash

SPLIT="train"

echo "id,width,height" > "${SPLIT}.csv"
echo "./${SPLIT}/*.jpg" | xargs identify | awk '
    {
        gsub("x",",",$3)
        match($1,"[0-9a-f]{16}")
        id = substr($1,RSTART,RLENGTH)
        printf "%s,%s\n",id,$3
    }
' >> "${SPLIT}.csv"



for i in 2 3
do
    for j in 30 40
    do
        python v2clean.py launch-qsub predict -m v2clean/ep4_freqthresh-${i}_loss-arcface_verifythresh-${j}.pth --ms --scale L2 --batch-size 24 --splits train --n-blocks 32 &
    done
done

for i in 2 3
do
    for j in 30 40
    do
        ls -1 feats_train_ms_L2_ep4_freqthresh-${i}_loss-arcface_verifythresh-${j} | wc -l
    done
done
