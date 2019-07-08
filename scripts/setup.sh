#!/usr/bin/env bash

SPLIT="train2018_r800"

echo "id,height,width" > "${SPLIT}.csv"
echo "/fs2/groups2/gca50080/yokoo/landmark/input/${SPLIT}/*.jpg" | xargs identify | awk '
    {
        gsub("x",",",$3)
        match($1,"[0-9a-f]{16}")
        id = substr($1,RSTART,RLENGTH)
        printf "%s,%s\n",id,$3
    }
' >> "${SPLIT}.csv"

SPLIT="index"

echo "id,height,width" > "${SPLIT}.csv"
echo "/fs2/groups2/gca50080/yokoo/landmark/input/${SPLIT}/*.jpg" | xargs identify | awk '
    {
        gsub("x",",",$3)
        match($1,"[0-9a-f]{16}")
        id = substr($1,RSTART,RLENGTH)
        printf "%s,%s\n",id,$3
    }
' >> "${SPLIT}.csv"

SPLIT="test"

echo "id,height,width" > "${SPLIT}.csv"
echo "/fs2/groups2/gca50080/yokoo/landmark/input/${SPLIT}/*.jpg" | xargs identify | awk '
    {
        gsub("x",",",$3)
        match($1,"[0-9a-f]{16}")
        id = substr($1,RSTART,RLENGTH)
        printf "%s,%s\n",id,$3
    }
' >> "${SPLIT}.csv"


# recog-2019のtrainの絶対パスを事前にデータフレームに保存
echo """
import pandas as pd
from pathlib import Path
train = pd.read_pickle('./train.pkl')
paths = Path('./train/').glob('**/*.jpg')
df_path = pd.DataFrame(paths, columns=['path'])
df_path['path'] = df_path['path'].apply(lambda x: str(x.absolute()))
df_path['id'] = df_path['path'].apply(lambda x: x.split('/')[-1].replace('.jpg', ''))
train = train.merge(df_path, on='id')
train.to_pickle('./train.pkl')
""" | python

# recog-2018のtrainの絶対パスを事前にデータフレームに保存
echo """
import pandas as pd
from pathlib import Path
old_train = pd.read_csv('./old_train.csv')
hw_train = pd.read_csv('./train2018.csv')
old_train = old_train.merge(hw_train, on='id')
old_train.to_pickle('./train2018.pkl')
""" | python
unlink /fs2/groups2/gca50080/yokoo/landmark/input/old_train.csv

# identifyで失敗したやつの欠損補完
echo """
import pandas as pd
from pathlib import Path
import cv2
old_train = pd.read_csv('./old_train.csv')
hw_train = pd.read_csv('./train2018.csv')

paths = Path('./train2018/').glob('**/*.jpg')
paths = set(list(map(lambda x: str(x).split('/')[-1].replace('.jpg', ''), paths)))
old_train = old_train[old_train['id'].isin(paths)]
old_train = old_train.merge(hw_train, on='id', how='left')
old_train = old_train.set_index('id')
failed_ids = paths - set(hw_train['id'])

for id_ in failed_ids:
    img = cv2.imread(f'./train2018/{id_}.jpg')
    h, w, c = img.shape
    old_train.loc[id_, ['height', 'width']] = h, w

old_train = old_train.reset_index().sort_values(by='id')
old_train.to_pickle('./train2018.pkl')
""" | python

# stage2のindex/testの画像サイズ取得
"""
import pandas as pd
from pathlib import Path
import cv2
from tqdm import tqdm

index19 = pd.read_csv('./index19/index.csv', index_col=[0]).sort_index()
index_paths = list(Path('./index19/').glob('**/*.jpg'))

test19 = pd.read_csv('./test19/test.csv', index_col=[0]).sort_index()
test_paths = list(Path('./test19/').glob('**/*.jpg'))
for path in tqdm(test_paths):
    id_ = str(path).split('/')[-1].replace('.jpg', '')
    img = cv2.imread(str(path))
    h, w, c = img.shape
    test19.loc[id_, 'height'] = h
    test19.loc[id_, 'width'] = w
test19.reset_index().sort_values(by='id').to_pickle('./test19.pkl')

for path in tqdm(index_paths):
    id_ = str(path).split('/')[-1].replace('.jpg', '')
    img = cv2.imread(str(path))
    h, w, c = img.shape
    index19.loc[id_, 'height'] = h
    index19.loc[id_, 'width'] = w
index19.reset_index().sort_values(by='id').to_pickle('./index19.pkl')
""" | python