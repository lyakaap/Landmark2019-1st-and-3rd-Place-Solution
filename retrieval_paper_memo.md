# Image Retreival / Metric Learning / (Face Recognition)｜論文メモ



## Face系

## [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)

CNNにより抽出した特徴ベクトルとFC層の重み(次元数：特徴ベクトルの次元数 x クラス数)を両方正規化して内積を取ると、出力のクラス数と同じ次元を持つベクトルが「cos類似度」とみなすことが出来る。
これに対して、$\cos(\theta_{y_i}+m)$のように正解クラスの要素にのみマージン$m$を足すことで、可能な限りクラス内分散をコンパクトにしようという手法。そのままsoftmaxを掛けてしまうと勾配が大きくなりすぎてしまうため、temperature scalingを行うことで補正する（s=30など）。

![arcface](https://user-images.githubusercontent.com/27487010/57187857-6b404c00-6f30-11e9-9f14-bf5feaed88a8.png)

他の類似手法との比較は↓ 

![arcface1](https://user-images.githubusercontent.com/27487010/57187323-9c684e80-6f27-11e9-948e-a83379f0c5ef.png)

性能的にはArc>Cos>Sphereらしい。
また、ArcFaceにIntra/Inter Lossを付加しても性能は上がらない。つまり、ArcFace単体で十分discriminativeな特徴抽出ができていることが伺える。



## [AdaCos: Adaptively Scaling Cosine Logits for Effectively Learning Deep Face Representations](https://arxiv.org/abs/1905.00292)

### 概要

顔認識のタスクでお馴染みのcosine softmax lossは、性能は高いものの、scaling parameter (s) やmargin (m) といった敏感なハイパーパラメーターをチューニングする必要があった。
提案手法では特にsの影響が大きいことを加味して、sをクラス数と学習時の角度の統計量により自動的に設定できるようにした。それにより設定が必要なハイパーパラメーターをmだけとしている。実験では複数のbenchmarkでArcFaceすら上回る精度でsotaを達成。P2SGradよりも実験の数値は良さそう。

* 関係ないクラスの重みと埋め込みベクトルとの角度のミニバッチ内平均は学習初期から最後の方までずっと2/π付近をキープしていることが分かった。
* 異なるsとmで、角度・予測確率の関係がどのように変わるかを示した。
  * sが小さすぎると、角度がいくら小さくなっても予測確率が1にあまり近付かなくなるし、sが大きすぎると、角度がそこそこ大きくても予測確率が1に近い値を取るようになってしまう。これはmに対しても同じようなことが言える。
  * なので、これらのハイパーパラメータはちゃんとチューニングしないと学習が上手く行かなくなる。
  * （とは言ってもs=30, m=0.4あたりであれば大体のケースで無難な感じになるのでは？）



## [P2SGrad: Refined Gradients for Optimizing Deep Face Models](https://arxiv.org/abs/1905.02479)

（上のAdaCosと同じ第一著者、しかも両方CVPR2019に採択されてて強い）

顔認識のタスクでお馴染みのcosine softmax lossの問題点を指摘し、それらを解消することで顔認識データセットでsotaを達成した。
cosine softmax lossの勾配計算は勾配の大きさと向きに分解することができる。また、更新の方向はどの種類のcosine softmax lossでも変わらないため、勾配の大きさが主な手法間の違いを生む要素となっている。

cosine softmax lossの問題点は以下の通りである：

* 精度がハイパーパラメータ (scaling, margin)とクラス数に敏感で、チューニング大変
* 予測確率とクラス間の角度がハイパーパラメータとクラス数に依存しているため、勾配の大きさにも影響を与える。
  * person Re-IDなどのopensetなタスクが目的なのに、クラス数に依存してしまうのはなんとなく気持ち悪い
  * Angular Margin-based Lossだと角度と勾配の大きさが比例しないため、直感的でなくなんとなく気持ち悪い

これらを解決できるような勾配計算を↓のように定義
$$
\begin{equation}
\begin{array}{l}{\tilde{G}_{\mathrm{P2SGrad}}\left(\vec{x}_{i}\right)=\sum_{j=1}^{C}\left(\cos \theta_{i, j}-\mathbb{1}\left(j=y_{i}\right)\right) \cdot \frac{\partial \cos \theta_{i, j}}{\partial \vec{x}_{i}}} \\ {\tilde{G}_{\mathrm{P2SGrad}}\left(\vec{W}_{j}\right)=\left(\cos \theta_{i, j}-\mathbb{1}\left(j=y_{i}\right)\right) \cdot \frac{\partial \cos \theta_{i, j}}{\partial \vec{W}_{j}}}\end{array}
\end{equation}
$$
xは入力、Wは重み、$\cos\theta_{i,j}$は入力のembeddingと重みjとのcos類似度
$\mathbb{1}\left(j=y_{i}\right)$ はjが正解クラスのとき1になって他は0になるような関数を表す



## [AdaptiveFace: Adaptive Margin and Sampling for Face Recognition](http://openaccess.thecvf.com/content_CVPR_2019/html/Liu_AdaptiveFace_Adaptive_Margin_and_Sampling_for_Face_Recognition_CVPR_2019_paper.html)

* 顔認識データセットはクラスごとにサンプル数が大きく異なるというクラス不均衡問題がある。

* クラス不均衡問題に対して、クラスごとに適切なマージンの値を学習させるようにして対処した。
* 実験から、サンプル数が少ないクラスほど、大きなマージンを好むことが分かった。
* 他にもHard Protype MiningとAdaptive Samplingを提案。
  * Hard Prototype Mining: Prototype（モデルの最終層の重み）単位で難しいサンプルをマイニング
  * Adaptive Data Sampling: サンプル単位のマイニング

## [UniformFace: Learning Deep Equidistributed Representation for Face Recognition](http://ivg.au.tsinghua.edu.cn/people/Yueqi_Duan/CVPR19_UniformFace%20Learning%20Deep%20Equidistributed%20Representation%20for%20Face%20Recognition.pdf)

* クラス間距離が均等になるような正則化項を提案

## [RegularFace: Deep Face Recognition via Exclusive Regularization](http://openaccess.thecvf.com/content_CVPR_2019/html/Zhao_RegularFace_Deep_Face_Recognition_via_Exclusive_Regularization_CVPR_2019_paper.html)

* クラス間距離が出来るだけ大きくなるような正則化項を提案



## Pooling手法あれこれ

- **Sum-pooling of Convolutions (SPoC)**
  - [Aggregating Deep Convolutional Features for Image Retrieval](https://arxiv.org/abs/1510.07493)
  - global average-pooling
- **weighted sum-pooling (CroW)**
  - [Cross-dimensional Weighting for Aggregated Deep Convolutional Features](https://arxiv.org/abs/1512.04065)
- **Maximum Activations of Convolutions (MAC)**
  - global max-pooling
- **Regional Maximum Activations of Convolutions (RMAC)**

  - [Particular object retrieval with integral max-pooling of CNN activations](https://arxiv.org/abs/1511.05879)
  - マルチスケールかつsliding windowでregional cropを作成→cropごとにglobal max-pooling→全てのpooling適用後のベクトルをsum-aggregation
- **Generalized Mean Pooling (GeM)**

  - [Fine-tuning CNN Image Retrieval with No Human Annotation](https://arxiv.org/abs/1711.02512)
  - $\rm{GeM}(X) = (\frac{1}{N}\sum x^p)^{1/p}$ 
  - p=1でmean, p=∞でmaxと等しい。論文ではp=3を推奨。SPoC/MACよりも高精度。
- **Compact Bilinear Pooling**

  - Fine-grained visual classificationタスクで高精度なBilinear Poolingを精度をそのままにコンパクトに。
    - Bilinear Poolingは、特徴マップのチャンネル同士のグラム行列のようなものをflattenしたもの。
    - 素のBilinear Poolingによる出力ベクトルの次元数はチャンネル数の二乗と、とても大きい次元数になってしまう。→Tensor sketch projectionを使ってコンパクトに。
- **NetVLAD pooling**
  - [NetVLAD: CNN architecture for weakly supervised place recognition](https://arxiv.org/abs/1511.07247)
    - VLADのcodebookのassignmentをsoftにすることで、end-to-endに学習することを可能にしたpooling手法。提案論文では、GPS情報を使ってpositive/negative ペアを作り出し、triplet lossによって学習することで従来のVLADを大きく上回る精度を達成した。



## [Learning with Average Precision: Training Image Retrieval with a Listwise Loss](https://arxiv.org/abs/1906.07589)

Average Precisionを直接最適化する系の話。Landmarkに関するデータセットでTriplet Lossを上回る精度を達成。ちょっと検証に使っているデータセットが少ない？他のAP Lossなどと比較して差分があまり無くない？結局CosFaceとかの方が強くない？みたいな疑問はある。

### Method

mean Average Precision, 通称mAPがImage Retrievalの評価指標としてはメジャーになってきている。しかし、APの計算にはソート及びindicator functionが含まれていて、微分不可能である。
そこで、このindicator functionを無くして、soft assignmentを使うことで微分可能にしようというのが本論文のモチベーション。これをList-wise Lossとして提案。
具体的には、クエリとそれに対応するデータベースのサンプルとのコサイン類似度と、[-1, 1]の区間のM個のビンそれぞれとをsoft assignする。この「類似度受け取ってソート」→「類似度受け取ってそれに応じたビンによるsoft assign」の変換がキーアイデア。ちなみにMは十分大きくないとgradientの近似精度は悪くなる気がするが、M=20程度で十分らしい。以下は通常のAPとquantized APの比較。1 - quantized APが提案手法のlossとなる。

* 通常AP: ランキング上位（類似度によるソート）からprecisionとincremental recallを計算
* quantized AP: m=1,2,...,Mで、m個目に対応するビンの代表点b_mとコサイン類似度とのtriangluar kernelを計算し、そいつがsoft assignの重みとなる。そのあと各ビンでのprecisionとincremental recallを計算。（このときb_m = 1 - (m-1) ⊿）

また、バッチサイズは大きくした方が、ミニバッチ内で計算するAPがglobalなAPにより近くなる。ただ、解像度も大きくしたいが、そうするとバッチサイズを大きくできない。



## [Combination of Multiple Global Descriptors for Image Retrieval](https://arxiv.org/abs/1903.10663)

SPoC/MAC/GeM によるdescriptorを組み合わせてimage rerieval の精度を上げようという手法。

* 抽出した複数のdescriptorにそれぞれ次元削減用のfc層をかませた後、最後に全てのdescriptorをconcatする。
* 特徴抽出器にはresnet50（stage-3とstage-4の間のmax-poolingを除去）などを使用。

* Ranking Loss (batch-hard triplet loss)だけではなく、Auxiliary Classification Lossも同時に学習することで、Ranking Loss単体よりも大きく精度を上げている。

* SPoC/MAC/GeMの組み合わせについて
  * 精度的には、GeM + SPoC/MACどちらかで十分っぽい。



## [Ensemble Feature for Person Re-Identification](https://arxiv.org/abs/1901.05798)

* 特徴マップを分割して、それぞれに対してpooling->次元削減->softmaxとすることで、特徴マップをいっぺんに扱うよりも良い精度を得られることを示した。
* AAP (adaptive average pooling)を提案。
  * 縦方向の特徴マップ分割＆poolingの操作を行っているだけ。
* 縦方向の特徴マップ分割により、分類器ごとに「顔担当」「足担当」みたいになってそう。
  * おそらくタスクの対象が人物など、ある程度決められた領域に同じパーツが来る場合でないと効果は発揮しない。さらにalignされている必要がある。

https://www.kaggle.com/c/humpback-whale-identification の7位が使っていた。



## [Attention-based Ensemble for Deep Metric Learning](https://arxiv.org/abs/1804.00382)

![](https://user-images.githubusercontent.com/27487010/57783199-af212580-7768-11e9-8b48-5e59f97a0cdf.png)

metric learningのattentionを使ったアンサンブルの話。アンサンブルは計算コストがかかる問題があるため、特徴抽出に用いるbackboneのネットワークとclassifierは共通のものを使い、途中のattentionの部分で別々の重みを学習させる。これにより、attentionごとに違う特徴に重みをつけるようになり、少ないコストで高い精度が得られる。



## [Defense Against Adversarial Images usingWeb-Scale Nearest-Neighbor Search](https://arxiv.org/abs/1903.01612)

![defense1](https://user-images.githubusercontent.com/27487010/57197171-e80a0f00-6f9e-11e9-9c79-7b9c9a4b3e65.png)

CVPR2019。タイトルの通り、web-scaleなデータベースを使ったKNNによってAdvasarial ExampleのDefenseをしてしまおうという論文。アイデアはシンプルだけど面白いし、最近傍探索の可能性を感じる。
提案のdefense手法に対してのシンプルなattack手法も提案している。

##### アルゴリズム

1. あらかじめ画像ごとにCNN特徴とsoftmax probabilityを計算し、データベースを構築する。
2. advararial attackをくらった入力画像のCNN特徴抽出を行い、データベースから近傍点をK個持ってくる
3. K個の近傍点のsoftmax probabilityをもとに重み付けをしてクラス分類を行う。重みの計算には次の3つを提案：
   1. Uniform Weighting: 全て均一の重み
   2. CBW-E (entropy based weighting): $w = |\log C + \sum_{c=1}^C s_c\log s_c|$     ($C$はクラス数, $s$はsoftmaxの値)
   3. CBW-D (diversity based weighting): $w = \sum_{m=2}^{M+1}(s'_1 - s'_m )^P$         ($M$は上位何個まで見るか, $s'$は降順にソートされたsoftmaxの値, $P$はハイパラ)

* CNNの各層の特徴についてもどのような特色があるのかを実験している。浅い層の特徴ほど摂動の影響が小さく、深い層の特徴ほど摂動の影響が大きくなっている。ただ、そもそも深い層の方が識別能力が高いため、摂動を結構大きくしても逆転するほどではない。

* *Deep k-Nearest Neighbors: Towards Confident, Interpretable and Robust Deep Learning* もKNNによるdefenseをしようという論文。
  * Deep k-Nearest Neighborsでは、CNNの各層の特徴マップによるKNNの予測結果を見て、各層の結果が一致していればいるほど高くなるような指標の"credibility"を提案。"credibility"は一般的に使われることの多いconfidence（softmaxの最大値）よりも信用でき、advesarial exampleの検知も可能。



## [End-to-end Learning of Deep Visual Representations for Image Retrieval](https://arxiv.org/abs/1610.07940)

Deep Image Retrieval: Learning global representations for image searchのジャーナルバージョン？

![e2e-image-ret](https://user-images.githubusercontent.com/27487010/57422229-3858c980-724a-11e9-9cc1-42af4c283345.png)

* triplet lossなどのmetric learningにはクリーンなデータセットが必要不可欠ながらも、大規模かつクリーンなデータセットが無いということを指摘。LandmarksデータセットからLocal descriptor + ransac matchingによりクリーンなデータセットを再構築した。
  * 同一クラス内の全組み合わせのransac matchingを行って、inliers countを重みとした重み付きグラフを構築。そのグラフにより、結びつきの弱いサンプルをノイズと判断して除外した。
  * 併せて、検出したlocal descriptorからbounding boxを取得。bounding boxをrefineするために、先程の「同一クラス内の全組み合わせのransac matching」により得られたアフィン変換の行列を再利用する。具体的には、元画像内でのbounding boxの座標と、アフィン変換先でのbounding boxの座標のgeometric medianを取る。これをiterativeに他のサンプルでも複数回refineしていく。
* RMAC poolingはスライディング窓方式でcropを作ってそれぞれglobal descriptorを抽出するため重い。そこで、Region proposal network (RPN: faster-RCNNの1stage目で使われてるやつ) を使ってRoI (landmark領域候補)を検出することで精度を落とさず高速な推論を可能にした。
  * 精度的にはVGGを使うとR-MACに勝つけどResNetだとあまり変わらない。これは精度の高いネットワークは自動的に背景領域の影響を小さく出来ているからと推測できる。
* PCAをshift (bias) + Fully connected layerで近似して高速化した（PCAはGPUで計算すると遅い）。
* PQとPCAによる次元削減で高速な検索が可能なデータベースを構築した。
* DBA (database-side augmentation)を提案。
  * データベースのサンプルを、そのサンプルに対する近傍のdescriptorによる重み付き平均を取ることでrefineする。
  * QEと似ている。QEはquery側をrefineするけどDBAはDB側をrefineするイメージ。そのためDBAはオフラインで一回やるだけで良くて、query searchのときには速度に影響しないのが強み。
  * 何故効くのか？→descriptorがよりクラス中心に近づくから。DBAは同一クラス同士（近傍のサンプルは同一クラスに大体属しているという仮定）でよりクラス中心に引きつけ合うようなことをしている。
* QEとDBAのkに関する実験を行い、QEのkは小さめにしてDBAは大きめにするのが良いんじゃね？と言っている。ただこれはデータセットのサンプル数や、単一クラスあたりのサンプル数によりそう。



## [Fine-tuning CNN Image Retrieval with No Human Annotation](https://arxiv.org/abs/1711.02512)

* 人手を要することなくlandmark retrieval taskのためのfine-tuningを行うことを可能とした。
  * local descriptorのinliers countの数によるクラスタリングをしている。
* 複数のlandmark画像からSfMによる三次元復元を行い、自動的にpositive/negative pairの作成を行った。
* Generalized Mean Pooling (GeM)を提案。MAC/SPoCより高性能。
* 特徴抽出器としてcontrastive lossを使用。triplet lossより性能は良かった。
* FC層 + shift (bias)で近似できる。PCAの基底ベクトルによりGeM poolingの後のFC
* ラベルを活用するため、Linear discriminant analysisを応用したsupervised whitening PCAを提案。通常のPCAに比べ高性能。
  * お気持ち：クラス内分散を小さくし、クラス間分散を大きくする。
  * end-to-endにPCAを学習するバージョン、offlineでPCAを学習するバージョン両方実験したけど前者は遅い上に精度もofflineの方が良いらしい。
* $\alpha$QE (query expansion)を提案。QEを行う際に、cos類似度によって近傍のdescriptorに重み付けを行う。



## [Revisiting Oxford and Paris: Large-Scale Image Retrieval Benchmarking](https://arxiv.org/abs/1803.11285)

### 概要

既存のImage Retrievalのベンチマークとしてしばしば用いられるデータセットは、次の問題点がある：

1. データセットの規模が小さい
2. 近年の高性能な手法だと、ほとんど完璧な精度を出せてしまうので適切な評価が出来ない
3. アノテーションに誤りがある

本論文では、Oxford/Paris Datasetに着目し、それらの改良版である **ROxford/RParis** というデータセットを作成した。また、ROxford/RParisにて、既存のImage Retrievalの様々な手法を評価することで横断的な手法間の比較を実現した。

### 備考

* Oxford 100k みたいに多数のdistractor (landmarkに関係ない画像) を用いることで水増しをしているデータセットもある。ただ、既存の高性能な手法に対しては、精度に影響がほとんど無いため意味が薄い模様。しかもdistractorと言っておきながら普通にlandmark画像も混じってたりする (False Negative) 。
  * 本論文ではdistractorによる水増しを行った **R1M** も提案している。
    * YFCC100Mからのサンプリング
    * 画像数は1001001枚
    * 高解像(1024x768) 
    * False Negativeが混ざらないよう、複数のモデルによるフィルタリングをしている。
    * おそらく公開はしていない
* ラベルごとに Easy/Hard/Unclear/Negativeなどのattributeを付けている。Easy/Hardは照明条件や、画像中に占めるlandmark領域の割合などで決定、Unclearはpositiveだけど画像単体では判断できないもの。評価の情報がよりリッチになることが期待できる。
* Evaluation Protocol:
  * Easy: Easyのみで評価
  * Medium: EasyとHardで評価
  * Hard: Hardのみで評価
* 以前のParisやOxfordなどでの評価はEasyのProtocolに近い。なので、あまり参考にならない。よってMediumとHardを見るべき。
  * いくつかのEasy Protocolでいいスコアを出す手法がMedium/Hard Protocolだと微妙だったりする。
* 大規模実験による結果から、精度ではまだまだlocal descriptorに軍配が上がる。ただ、CNNベースのglobal descriptor手法の方が計算コストがかからないのと、local/global 両者を組み合わせることで更に精度が向上することが分かった。



## [Large-Scale Image Retrieval with Attentive Deep Local Features](https://arxiv.org/abs/1612.06321)

DELFの論文。Deepなlocal descriptorを提案。

![](https://user-images.githubusercontent.com/27487010/57424728-9a69fc80-7253-11e9-8a94-c83e45e6a909.png)

* cnnのfeature mapのサブピクセル1つ1つはそれぞれlocal descriptorとみなせる。
  * (H, W, C)のfeature mapであれば、C次元のdescriptorがHxW個得られる。
  * global descriptorの問題点：空間方向にpoolingしてしまうため、局所的な特徴を捉えづらい。
* 1x1のconvを使用して、「どのサブピクセルが重要か」に関するattention scoreを計算。
  * landmark分類を学習させることでattentionを学習。
  * end-to-endに特徴抽出器もattentionも学習させることもできるが、別々に学習させる方が良い。
* 従来のhand-crafted (non deep)なlocal descriptorに比べて大きく精度向上。他のglobal descriptorと組み合わせることで当時のsota。



## [Detect-to-Retrieve: Efficient Regional Aggregation for Image Search](https://arxiv.org/abs/1812.01584)

[Large-Scale Image Retrieval with Attentive Deep Local Features](https://arxiv.org/abs/1612.06321)の後続的な論文。

![](https://user-images.githubusercontent.com/27487010/57821122-52565700-77ca-11e9-9d8e-9c29122f7f01.png)

* Landmark画像に対してdetectorを用いることで、landmark領域を検知。領域ごとにdelf特徴を抽出し、VLADによるdescriptorを構築。最後に全領域で統合 (mean pooling) し、ASMKによる類似度計算を行い画像のマッチングを行うことを提案。この領域をまたいだASMKをR-ASMKと名付け、ASMKに比べて計算コストを増やすことなく高精度を達成した。
  * delf特徴: attentionを用いてcnn feature mapから抽出されたlocal descriptor
  * VLAD: 複数のlocal descriptorから、それぞれの特徴とクラスタ中心とのresidualにより一本のdescriptorを生成する手法
  * ASMK: VLADをベースに、thresholded polynomial selectivity functionを用いて画像間の類似度を定義
* Google Landmark Datasetのうち、約94K枚の画像に対してバウンディングボックスのアノテーションをして、landmark detector (SSD/faster R-CNN) を学習させた。
  * アノテーションは画像一枚に対して一つのbounding boxをつけさせるようにした。
  * 推論時は一枚の画像に対して複数の領域を使う。



## [ATTENTION-AWARE GENERALIZED MEAN POOLING FOR IMAGE RETRIEVAL](https://arxiv.org/abs/1811.00202)

アイデアはいたってシンプルで、ResNet-101-GeMにResidual Attentionを加えている。
GeMに比べて精度が結構上がっているので意外。

roxford5k/rparis6kにて、QE/DBA/Diffusionを使うことで既存手法に比べて高い精度を達成した。

roxford5k/rparis6kにおいての、QE/DBAの「使用する近傍の数」「α」についてそれぞれの精度を検証している。意外にも、DBAのαは1, QEのβは0のとき (つまり普通のnQE) が一番精度が良かったらしい。

![](https://user-images.githubusercontent.com/27487010/58009315-210ebb80-7b29-11e9-8f23-aae12905ea1a.png)

 



## Few Shot Learning系の手法

Matching Networks, MAML, Prototypical Networksなど

https://towardsdatascience.com/advances-in-few-shot-learning-a-guided-tour-36bc10a68b77

* 従来のfew-shot learningのアプローチ: 最初にembeddingの学習→KNNなどによる分類
  - end-to-endではない
* Matching Networksのコア
  * 学習用のデータセットから、特定のクラスのデータをサンプリングすることで、few-shot learningのタスクを擬似的に作る。このタスクをepisodeと言ったりする。
  * 疑似タスクをたくさん作り出すことで、end-to-endにfew-shot learningを学習。
  * KNNをcos類似度 + softmaxにより微分可能にした。
  * 順番の影響を無くしたLSTMを使って、サポートセット全体を考慮しつつembeddingを学習させた。
* Prototypical Networksのコア
  * Matching Networksと目的は一緒だけど、もっとシンプル
  * LSTMの代わりにクラスごとの中心を"prototype"として使って学習させる。

## [One-shot Face Recognition by Promoting Underrepresented Classes](https://arxiv.org/abs/1707.05574)

既存の顔認識ベンチマークは学習用のセットと評価用のセットで異なる人物の画像を使って評価することが多い。
当論文では、学習用と評価用で同じ人物が登場し、人物ごと（クラスごと）にサンプル数に大きな偏りが存在するようなタスクを人工的に作り出し評価した。
また、few-shot learningでよく使われるKNNよりもMLR (multi logistic regression) の方が性能が良いことを指摘。ただ、MLRはfew-shotなクラスに対して精度が悪くなってしまう傾向があるため、そういったクラスの重みを適切に調節することを提案。

手法のキーアイデアは以下の二つ。

* あるクラスのembeddingと、対応するクラスを代表するような重み（全結合層の重み）同士ができるだけ近いdirectionを持つような正則化項の提案
  * これはcenter lossに近い。向こうはクラス中心を逐次的に更新していく方式だけど、こちらはクラス中心が重みとなっている。そのため、他のクラスとの関係性も考慮できる。
* few-shot (underrepresented) なクラスに関する重みは、サンプル数がそこそこ多いクラスに関する重みに比べてノルムが小さくなる傾向がある。それが精度低下につながっていると考え、few-shot (underrepresented) なクラスに関する重みが大きくなるような正則化項を提案。



## [Bag of Tricks and A Strong Baseline for Deep Person Re-identification](https://arxiv.org/abs/1903.07071)

既存のPerson Re-ID手法を提案した論文は、色んな精度向上のテクニックを使っており、割と不公平な比較が行われてしまっている。
そこで、そういったテクニックをまとめて強いPerson Re-IDのベースラインを作った。
コード：https://github.com/michuanhaohao/reid-strong-baseline

### Bag og Tricks

1. LR-warmup

2. Random Erasing

3. Label Smoothing

   * one-shot learningだと、実際の推論時には関係ないような学習用のラベルに過学習してしまう可能性がある。なのでone-shot learningとLabel Smoothingは相性が良い。

4. ResNetの最後のstride=2のconvのstrideを1にする

   * 解像度が保存されていい感じ。計算コストもほとんど増えない。

5. Batch Normalization Neck

   * 現在のPerson Re-IDのトレンドは、triplet lossとsoftmaxの組み合わせが多い。ただ、それぞれを同じlayerの出力からlossを計算すると精度に悪影響が出るのでは無いかということを指摘。それぞれを分けることで精度を大きく向上させている。

     ![](https://user-images.githubusercontent.com/27487010/57823829-e24cce80-77d3-11e9-90f0-2b2c186f4fc9.png)

6. Center Loss

これらのトリックを一個づつ加えていった結果がこちら。global descriptorのみで複雑なlocal descriptorベースの手法に近い精度を出すことが出来ている。

![](https://user-images.githubusercontent.com/27487010/57821957-74050d80-77cd-11e9-9ccc-8b449fd97a6b.png)



## [Brute-Force Facial Landmark Analysis With A 140,000-Way Classifier](https://arxiv.org/abs/1802.01777)

顔の特徴点抽出を行うタスクを、特徴点の座標の回帰ではなく、150Kクラスの分類問題に置き換えることでocclusionなどに対処し、ロバストなモデルを開発した。一つのクラスが特定の誰か一人に対応している。
学習時にはsoftmaxではなくmulti label lossという、クラスの数分の二値分類ロスを最小化するようにする。何でそうするかというと、150Kクラスもあるととても類似したクラスが出て来るので、オーバーラップを許したいからである。ただ推論時はsoftmaxに付け替える。



## [A Discriminative Feature Learning Approach for Deep Face Recognition](https://ydwen.github.io/papers/WenECCV16.pdf)

Center Lossの論文。
顔にまつわるタスク，例えば顔認証，認識，検知などのタスクは，辞書にないデータが予測時に入ってくる可能性があるので従来の他クラス判別の手法が使えない．
よって，顔画像をベクトル化して近傍法などを適用するなどの手法がよく使われる．そういったケースでは，次元の呪いがしばしば問題として起こってしまうので顔の特徴をよく表した分散表現を獲得出来ればかなり嬉しい．

Center Lossは，ミニバッチ内の特定クラスの平均との距離を最小化させるような損失．
softmax lossと組み合わせることで良い分離の分散表現を獲得できる．
似た論文でcontrastive-lossという，interクラスとの距離も損失に加えて，よりクラス間の分離を良くしようという試みもある．
目的関数はsoftmax + intra-class distances / inter-class distances となる



## [Learning Metrics from Teachers: Compact Networks for Image Embedding](https://arxiv.org/abs/1904.03624)

Image Embeddingにおいてdistillationをした論文。学習の方法は下図がわかりやすい。
よくあるmetric learningのように、二つの画像によって学習及びdistillationを行う。
Lossに用いる項は以下の通り (※ teacherは学習時はfreezeする)

1. $d^S$｜ 画像iと画像jとの距離 (S: student network)
2. $L^{abs}_{KD}$｜ teacher/student間のembeddingの距離
3. $L^{rel}_{KD}$｜teacher/student間の $d^T, d^S$ 同士の距離

![](https://user-images.githubusercontent.com/27487010/57994195-edb33900-7af6-11e9-9acc-5ef2be86de29.png)

さらに、distillationで有効であることが分かっている「hint-loss」と「attention-loss」はembeddingのdistillationでも有効であることを確かめた。これらのlossは全てmultitask learningの要領で同時に最適化される。

* hint-loss: teacher/studentの各層のfeature mapもヒントとしてdistillationに使う手法
* attention-loss: teacher/studentの各層のspatial attention mapをdistillationに使う手法
  * spatial attention map: ただのチャンネル方向のpooling。出力shapeはHxW。
  * 画像中のどの領域に注目したら良いかを伝達する狙い。



## [Hardness-Aware Deep Metric Learning](https://arxiv.org/abs/1903.05503)

CVPR2019 oral.

Triplet lossの学習にはpositive/negative pairを作成する必要があるが、その際にnegative-pairの考えられうる組み合わせはとても多い。そのため、hard negative miningなどのテクニックは必要不可欠である。しかし、hard negative miningはトリッキーな上に一部のnegativeなpairしかサンプリングされないためもったいない。また、実際のデータと学習に用いるデータとの分布にずれが発生してしまう。
そこで、easyなnegative pairも有効的に使って精度を上げることができないかについて取り組んだのがこの論文。

具体的にどうするかと言うと、negative sampleとanchorを混ぜたもの (linear interpolation) を学習に使う。イメージとしては、下の図のように「y- (negative sample) に y+ (positive sample) をミックスすることで擬似的にhardなnegative sampleを作ろう」という狙いがある。しかし、そのまま愚直にミックスするとmanifoldの外に出てしまう（ありえないようなサンプルになってしまう）恐れがあるため、negative sample の manifold の一番近いとこに戻すような操作をする。

![](https://user-images.githubusercontent.com/27487010/58157906-44b23d00-7cb4-11e9-8f7c-2137d152f44c.png)

他の主要な工夫点としては、negative sampleの難しさによって、どの程度positve sampleの混ぜ具合を大きくするかを決めている。
これにより、学習が進んでnegative pairとの距離が小さくなるにつれてnegative sampleに混ぜ込むanchorの割合が増えるため、学習の後半でも生成するnegative sampleの難しさを維持することができる。

### どうやってmanifoldに戻すのか

* feature: y を入力とするautoencoderを使う。これをgeneratorと呼ぶ。generatorのencoderはfeature spaceからembedding: zへとmappingし、augmentor (negative sampleを生成するモジュール) がzに対してanchorを混ぜる処理をしてz^を生成して、decoderがz^をfeature spaceにmappingする（y^）。
* y^とy が同じカテゴリか、つまりanchorを混ぜたあとでもラベルが変わらないようにするような目的関数を設定している。
* 元のnegative sampleもgeneratorに通して再構成誤差を最小化するようにする。
* generatorはメインのネットワークと同時に学習する。

![](https://user-images.githubusercontent.com/27487010/58162398-12f1a400-7cbd-11e9-8ab7-26f00fafe363.png)


**疑問**：Object detectionではnegative samplingの問題点をfocal lossによって解決した。似たようなことをmetric learningでも出来ないか？



## Towards Good Practices for Image Retrieval Based on CNN Features

R-MACを改善したという論文。

* Multi-scale input: 複数の解像度の入力画像を用意して、複数回forwardして最後に"early fusion"する。いわゆるmulti-scale inferenceのような"late fusion"と違って安定すると主張している。
  * 思ったこと：学習時にもforwardの回数が増えてしまうことから計算コストの面では良くないと思われる。
* Multi-layer concatenation: 最後の二つのFC層の出力をconcatしている。その分次元数が多くなってしまうので、主成分ベクトルとクラスタリングを使った次元削減をしている。
*  RPNの代わりに、CAMによってざっくりとしたRoIのマスキングをして、activationが一定以上の領域のみでR-MACを適用する。



## [Improved Deep Metric Learning with Multi-class N-pair Loss Objective](https://papers.nips.cc/paper/6200-improved-deep-metric-learning-with-multi-class-n-pair-loss-objective)

* N-pair mc lossを提案。
* triplet lossよりも複数種類のnegativeを学習時に考慮できるので精度良くなる。
* ミニバッチの作り方
  * 複数のクラスから、二つずつサンプリングする。これにより、anchor-positive-negativesのタプルを無駄なく構築できる。
  * triplet lossではhard negative miningをサンプルに対してやっていた。この論文では、もっと効率的にやろうということで、hard **class** negative miningを提案。
  * nagative classの抽出法：
    * まずランダムに一つのクラスを選択し、そのクラスの中から二つのサンプルを選択。
    * その二つのサンプルとのtriplet constraintを違反するような難しいクラスをgreedyに選択していく



## [Explore-Exploit Graph Traversal for Image Retrieval]([http://www.cs.toronto.edu/~mvolkovs/cvpr2019EGT.pdf](http://www.cs.toronto.edu/~mvolkovs/cvpr2019EGT.pdf))

CVPR2019. 画像検索におけるグラフベースのre-ranking手法の論文。

* 検索タスクにおけるグラフベースの手法には、Query ExpansionやDiffusionなどがある。これらの手法に比べ高速かつ高精度な手法を提案。
* 具体的には、K近傍グラフに対して、ExploreステップとExploitステップを交互に繰り返すことで、「クエリとの類似度は低いけど、クエリの近傍に対しての類似度が高いもの」を検索結果に追加していく。
* Exploreステップ
  * クエリを出発点として、プリム法のようにgreedyにエッジの重みが大きいところから近傍をたどっていく。そのため、Diffusionと違って再帰的に「近傍の近傍」を引っ張ってこれるのが強み。
  * K近傍グラフのエッジの重みについては、cos類似度でも良いし、RANSAC matchingによるinliers-countでも良い。後者の方が重いけど高性能、けどcos類似度でも普通に既存手法より良い。
* Exploitステップ
  * 何でもかんでも検索結果に追加していたら、クエリと無関係なものまで引っ張ってきてしまう恐れがあるので、しきい値によって追加するかどうかを決める。
* ROxford5kやRParis6k, Google Landmark DatasetでQE/Diffusionを大きく上回る精度を達成。
* たまに精度が良くないときがあるが、これはtopic-drift (無関係のものも拾ってきてしまう現象) が原因。