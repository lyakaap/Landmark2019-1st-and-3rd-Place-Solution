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
