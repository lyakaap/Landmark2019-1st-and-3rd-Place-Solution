# Landmark2019-1st-and-3rd-Place-Solution

Google Landmark 2019 1st Place Solution of Retrieval Challenge and 3rd Place Solution of Recognition Challenge.

```
bash donwload_train.sh # download data
bash setup.sh  # setup data to ready training
bash reproduce.sh  # train models and predict for reproducing
```


We borrow idea and code from these grate repositories,

* https://github.com/filipradenovic/cnnimageretrieval-pytorch/tree/master/cirtorch
* https://github.com/kevin-ssy/FishNet
* https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/data/samplers/grouped_batch_sampler.py


## Ref

https://www.kaggle.com/c/landmark-retrieval-challenge
https://www.kaggle.com/c/landmark-recognition-challenge
https://www.kaggle.com/c/landmark-retrieval-2019/
https://www.kaggle.com/c/landmark-recognition-2019/

* retrieval 1st place
  * https://www.kaggle.com/c/landmark-retrieval-challenge/discussion/57855 
  * https://twitter.com/ducha_aiki/status/1008833406036107270
  
* recognition 3rd place https://drive.google.com/file/d/1rEAphWzeg94jcP6dPc2KDSGlmPxnuZ0e/view

* recognition 8th place https://drive.google.com/file/d/1wlyLUiiUiEt4v9Yv8ZyAQvm9l81im4MV/view

* recognition 10th place https://www.slideshare.net/Eduardyantov/kaggle-google-landmark-recognition

* retrieval 6th place <https://www.kaggle.com/c/landmark-retrieval-challenge/discussion/58482#latest-340957>

* 上位解法まとめ https://drive.google.com/file/d/12Zb0NZL3Ys6SPLiAvCroW5w3gZIYCB2X/view

* https://github.com/ducha-aiki/affnet

* https://github.com/ducha-aiki?tab=repositories

* Recog19th-place <https://github.com/jandaldrop/landmark-recognition-challenge>

* pytorch-delf matching code <https://github.com/nashory/DeLF-pytorch/blob/master/helper/matcher.py>

* 最近某探索の神資料： <http://yusukematsui.me/project/survey_pq/doc/ann_billion_2018.pdf>

* Whale 4th place <https://www.kaggle.com/c/humpback-whale-identification/discussion/82356>



## 指標について

### recognition
指標：GAP
計算方法：
1. 予測確率順に並べる
2. 予測確率の順位が上の方からprecisionを求めていく
3. 実際にランドマークが写っている写真の数で平均を取る．つまりランドマークが写っていない写真は何も予測しない．カウントされないわけではないので，何らかのクラスにいくらか確率を割り当ててしまうと，precisionの関係上ペナルティが生じる

具体例：

予測がこうなったとする
|画像|予測クラス|確率|
|:-:|:-:|:-:|
|A|3|0.7|
|B|2|0.9|
|C|1|0.2|
|D|3|0.6|
|E|0|0.3|

予測確率順にソートされ，正解かどうかのマッチングを行う（正解クラスが無し＝landmarkは写っていない）
|画像|予測クラス|確率|正解クラス|結果|Prec|
|:-:|:-:|:-:|:-:|:-:|:-:|
|B|2|0.9|2|o|1.0|
|A|3|0.7|2|x|-|
|D|3|0.6|3|o|0.67|
|E|0|0.3|無し|x|-|
|C|1|0.2|1|o|0.6|

この場合最終的に`(1.0 + 0.67 + 0.6) / 4 = 0.57`となる．分母に「無し」画像は含まれない

* probability calibrationをする必要があるのではないか．つまり，クラス不均衡データであれば調整する必要がある
* Average Precisionの性質上，high confidenceなmiss classificationにはとても重いペナルティがかかる（確率値自体は関係無く，ランク上位に来てるくせに間違っているとやばい）．

### retrieval
指標： MAP@100

具体例：あるクエリ画像Aがあり，同一のクラスの画像がB, C, D, E, Fの5個あったとする．予測結果は以下のように6個の画像をサジェストした．
```
A BCZEWD
```
このとき計算式は，
$$
\frac{1}{min(5, 100)} \Sigma_{k=1}^{min(6, 100)} P * \delta
$$
となる．具体的に計算すると，
`B:正解(1.0), C:正解(1.0), Z:不正解, E:正解(0.75), W:不正解, D:正解(0.67)`
よって $\frac{1}{5}(1.0+1.0+0.75+0.67)$

式を見て分かる通り，$n_q$にペナルティがかからない．つまり大前提として類似度上位100個MAXまで予測した方が得となる．
