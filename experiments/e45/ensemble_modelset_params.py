"""
アンサンブルの組み合わせを記述する
"""
BASEDIR = 'yokoo/landmark/experiments/'

# 5/29 アンサンブルセット（スコア確認済みの９モデル）
# LB スコア https://docs.google.com/spreadsheets/d/1ybMlAkzTFNT2kAvuRUrXOTNfrtVkk9T6rgyTltiyzMc/edit#gid=1585242002
# 精度の相関：https://files.slack.com/files-pri/T0M91A51B-FK2M6NB7A/image.png
MODELSET_V0529_WEIGHT_V1 = [
    0.5,
    0.5,
    0.5,
    1.0,
    1.5,
    0.5,
    1.0,
    0.5,
    0.5,
]
MODELSET_V0529 = {
    'test': [
        BASEDIR + 'v19c/feats_test19_ms_L2_ep4_scaleup_ep3_freqthresh-2_loss-arcface_pooling-G,G,G,G_verifythresh-30/',
        BASEDIR + 'v19c/feats_test19_ms_L2_ep4_scaleup_ep3_freqthresh-2_loss-arcface_pooling-SMG,MG,G,G_verifythresh-30/',
        BASEDIR + 'v19c/feats_test19_ms_L2_ep4_scaleup_ep3_freqthresh-2_loss-adacos_pooling-G,G,G,G_verifythresh-30/',
        BASEDIR + 'v19c/feats_test19_ms_L2_ep4_scaleup_ep3_freqthresh-2_loss-cosface_pooling-G,G,G,G_verifythresh-30/',
        BASEDIR + 'v20c/feats_test19_ms_L2_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30/',
        BASEDIR + 'v21c/feats_test19_ms_L2_ep4_scaleup_ep3_augmentation-soft_epochs-5_freqthresh-3_loss-arcface_verifythresh-30/',
        BASEDIR + 'v21c/feats_test19_ms_L2_ep6_scaleup_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30/',
        BASEDIR + 'v22c/feats_test19_ms_L2_ep4_scaleup_ep3_base_margin-0.3_freqthresh-2_verifythresh-20/',
        BASEDIR + 'v22c/feats_test19_ms_L2_ep4_scaleup_ep3_base_margin-0.4_freqthresh-2_verifythresh-30/',
    ],
    'index': [
        BASEDIR + 'v19c/feats_index19_ms_L2_ep4_scaleup_ep3_freqthresh-2_loss-arcface_pooling-G,G,G,G_verifythresh-30/',  # 0.28477
        BASEDIR + 'v19c/feats_index19_ms_L2_ep4_scaleup_ep3_freqthresh-2_loss-arcface_pooling-SMG,MG,G,G_verifythresh-30/',  # 0.27500
        BASEDIR + 'v19c/feats_index19_ms_L2_ep4_scaleup_ep3_freqthresh-2_loss-adacos_pooling-G,G,G,G_verifythresh-30/',  # 0.27569
        BASEDIR + 'v19c/feats_index19_ms_L2_ep4_scaleup_ep3_freqthresh-2_loss-cosface_pooling-G,G,G,G_verifythresh-30/',  # 0.29039
        BASEDIR + 'v20c/feats_index19_ms_L2_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30/',  #
        BASEDIR + 'v21c/feats_index19_ms_L2_ep4_scaleup_ep3_augmentation-soft_epochs-5_freqthresh-3_loss-arcface_verifythresh-30/',  #
        BASEDIR + 'v21c/feats_index19_ms_L2_ep6_scaleup_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30/',  #
        BASEDIR + 'v22c/feats_index19_ms_L2_ep4_scaleup_ep3_base_margin-0.3_freqthresh-2_verifythresh-20/',  #
        BASEDIR + 'v22c/feats_index19_ms_L2_ep4_scaleup_ep3_base_margin-0.4_freqthresh-2_verifythresh-30/',  #
    ],
    'train': [
        BASEDIR + 'v19c/feats_train_ms_L2_ep4_scaleup_ep3_freqthresh-2_loss-arcface_pooling-G,G,G,G_verifythresh-30/',
        BASEDIR + 'v19c/feats_train_ms_L2_ep4_scaleup_ep3_freqthresh-2_loss-arcface_pooling-SMG,MG,G,G_verifythresh-30/',
        BASEDIR + 'v19c/feats_train_ms_L2_ep4_scaleup_ep3_freqthresh-2_loss-adacos_pooling-G,G,G,G_verifythresh-30/',
        BASEDIR + 'v19c/feats_train_ms_L2_ep4_scaleup_ep3_freqthresh-2_loss-cosface_pooling-G,G,G,G_verifythresh-30/',
        BASEDIR + 'v20c/feats_train_ms_L2_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30/',
        BASEDIR + 'v21c/feats_train_ms_L2_ep4_scaleup_ep3_augmentation-soft_epochs-5_freqthresh-3_loss-arcface_verifythresh-30/',
        BASEDIR + 'v21c/feats_train_ms_L2_ep6_scaleup_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30/',
        BASEDIR + 'v22c/feats_train_ms_L2_ep4_scaleup_ep3_base_margin-0.3_freqthresh-2_verifythresh-20/',
        BASEDIR + 'v22c/feats_train_ms_L2_ep4_scaleup_ep3_base_margin-0.4_freqthresh-2_verifythresh-30/',
    ],
}


MODELSET_V0531_WEIGHT_V1 = [
    1.0,
    1.0,
    1.0,
]
MODELSET_V0531 = {
    'test': [
        BASEDIR + 'v19c/feats_test19_ms_L2_ep4_scaleup_ep3_freqthresh-2_loss-cosface_pooling-G,G,G,G_verifythresh-30/',
        BASEDIR + 'v20c/feats_test19_ms_L2_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30/',
        BASEDIR + 'v21c/feats_test19_ms_L2_ep6_scaleup_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30/',
    ],
    'index': [
        BASEDIR + 'v19c/feats_index19_ms_L2_ep4_scaleup_ep3_freqthresh-2_loss-cosface_pooling-G,G,G,G_verifythresh-30/',  # 0.29039
        BASEDIR + 'v20c/feats_index19_ms_L2_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30/',  #
        BASEDIR + 'v21c/feats_index19_ms_L2_ep6_scaleup_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30/',  #
    ],
    'train': [
        BASEDIR + 'v19c/feats_train_ms_L2_ep4_scaleup_ep3_freqthresh-2_loss-cosface_pooling-G,G,G,G_verifythresh-30/',
        BASEDIR + 'v20c/feats_train_ms_L2_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30/',
        BASEDIR + 'v21c/feats_train_ms_L2_ep6_scaleup_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30/',
    ],
}


MODELSET_V0601_WEIGHT_V1 = [
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
]
MODELSET_V0601 = {
    'test': [
        BASEDIR + 'v19c/feats_test19_ms_L2_ep4_scaleup_ep3_freqthresh-2_loss-arcface_pooling-G,G,G,G_verifythresh-30/',
        BASEDIR + 'v19c/feats_test19_ms_L2_ep4_scaleup_ep3_freqthresh-2_loss-arcface_pooling-SMG,MG,G,G_verifythresh-30/',
        BASEDIR + 'v19c/feats_test19_ms_L2_ep4_scaleup_ep3_freqthresh-2_loss-adacos_pooling-G,G,G,G_verifythresh-30/',
        BASEDIR + 'v19c/feats_test19_ms_L2_ep4_scaleup_ep3_freqthresh-2_loss-cosface_pooling-G,G,G,G_verifythresh-30/',
        BASEDIR + 'v20c/feats_test19_ms_L2_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30/',
        BASEDIR + 'v21c/feats_test19_ms_L2_ep4_scaleup_ep3_augmentation-soft_epochs-5_freqthresh-3_loss-arcface_verifythresh-30/',
        BASEDIR + 'v21c/feats_test19_ms_L2_ep6_scaleup_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30/',
        BASEDIR + 'v22c/feats_test19_ms_L2_ep4_scaleup_ep3_base_margin-0.3_freqthresh-2_verifythresh-20/',
        BASEDIR + 'v22c/feats_test19_ms_L2_ep4_scaleup_ep3_base_margin-0.4_freqthresh-2_verifythresh-30/',
    ],
    'index': [
        BASEDIR + 'v19c/feats_index19_ms_L2_ep4_scaleup_ep3_freqthresh-2_loss-arcface_pooling-G,G,G,G_verifythresh-30/',  # 0.28477
        BASEDIR + 'v19c/feats_index19_ms_L2_ep4_scaleup_ep3_freqthresh-2_loss-arcface_pooling-SMG,MG,G,G_verifythresh-30/',  # 0.27500
        BASEDIR + 'v19c/feats_index19_ms_L2_ep4_scaleup_ep3_freqthresh-2_loss-adacos_pooling-G,G,G,G_verifythresh-30/',  # 0.27569
        BASEDIR + 'v19c/feats_index19_ms_L2_ep4_scaleup_ep3_freqthresh-2_loss-cosface_pooling-G,G,G,G_verifythresh-30/',  # 0.29039
        BASEDIR + 'v20c/feats_index19_ms_L2_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30/',  #
        BASEDIR + 'v21c/feats_index19_ms_L2_ep4_scaleup_ep3_augmentation-soft_epochs-5_freqthresh-3_loss-arcface_verifythresh-30/',  #
        BASEDIR + 'v21c/feats_index19_ms_L2_ep6_scaleup_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30/',  #
        BASEDIR + 'v22c/feats_index19_ms_L2_ep4_scaleup_ep3_base_margin-0.3_freqthresh-2_verifythresh-20/',  #
        BASEDIR + 'v22c/feats_index19_ms_L2_ep4_scaleup_ep3_base_margin-0.4_freqthresh-2_verifythresh-30/',  #
    ],
    'train': [
        BASEDIR + 'v19c/feats_train_ms_L2_ep4_scaleup_ep3_freqthresh-2_loss-arcface_pooling-G,G,G,G_verifythresh-30/',
        BASEDIR + 'v19c/feats_train_ms_L2_ep4_scaleup_ep3_freqthresh-2_loss-arcface_pooling-SMG,MG,G,G_verifythresh-30/',
        BASEDIR + 'v19c/feats_train_ms_L2_ep4_scaleup_ep3_freqthresh-2_loss-adacos_pooling-G,G,G,G_verifythresh-30/',
        BASEDIR + 'v19c/feats_train_ms_L2_ep4_scaleup_ep3_freqthresh-2_loss-cosface_pooling-G,G,G,G_verifythresh-30/',
        BASEDIR + 'v20c/feats_train_ms_L2_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30/',
        BASEDIR + 'v21c/feats_train_ms_L2_ep4_scaleup_ep3_augmentation-soft_epochs-5_freqthresh-3_loss-arcface_verifythresh-30/',
        BASEDIR + 'v21c/feats_train_ms_L2_ep6_scaleup_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30/',
        BASEDIR + 'v22c/feats_train_ms_L2_ep4_scaleup_ep3_base_margin-0.3_freqthresh-2_verifythresh-20/',
        BASEDIR + 'v22c/feats_train_ms_L2_ep4_scaleup_ep3_base_margin-0.4_freqthresh-2_verifythresh-30/',
    ],
}

MODELSET_V0601_WEIGHT_V2 = [
    0.5,
    0.5,
    0.5,
    1.0,
    1.9,
    0.5,
    1.2,
    0.5,
    0.5,
]

MODELSET_V0602_WEIGHT_V1 = [
    0.5,
    1.0,
    1.0,
    0.5,
    1.0,
    1.0,
]
MODELSET_V0602 = {
    'test': [
        BASEDIR + 'v19c/feats_test19_ms_L2_ep4_scaleup_ep3_freqthresh-2_loss-cosface_pooling-G,G,G,G_verifythresh-30/',
        BASEDIR + 'v20c/feats_test19_ms_L2_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30/',
        BASEDIR + 'v21c/feats_test19_ms_L2_ep6_scaleup_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30/',
        BASEDIR + 'v22c/feats_test19_ms_L2_ep4_scaleup_ep3_base_margin-0.4_freqthresh-2_verifythresh-30/',
        BASEDIR + 'v23c/feats_test19_ms_L2_ep6_scaleup_ep5_augmentation-middle_epochs-7_freqthresh-3_verifythresh-30/',
        BASEDIR + 'v24c/feats_test19_ms_L2_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-cosface_verifythresh-30/',
    ],
    'index': [
        BASEDIR + 'v19c/feats_index19_ms_L2_ep4_scaleup_ep3_freqthresh-2_loss-cosface_pooling-G,G,G,G_verifythresh-30/',  # 0.29039
        BASEDIR + 'v20c/feats_index19_ms_L2_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30/',  #
        BASEDIR + 'v21c/feats_index19_ms_L2_ep6_scaleup_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30/',  #
        BASEDIR + 'v22c/feats_index19_ms_L2_ep4_scaleup_ep3_base_margin-0.4_freqthresh-2_verifythresh-30/',  #
        BASEDIR + 'v23c/feats_index19_ms_L2_ep6_scaleup_ep5_augmentation-middle_epochs-7_freqthresh-3_verifythresh-30/',  #
        BASEDIR + 'v24c/feats_index19_ms_L2_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-cosface_verifythresh-30/',  #
    ],
    'train': [
        BASEDIR + 'v19c/feats_train_ms_L2_ep4_scaleup_ep3_freqthresh-2_loss-cosface_pooling-G,G,G,G_verifythresh-30/',
        BASEDIR + 'v20c/feats_train_ms_L2_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30/',
        BASEDIR + 'v21c/feats_train_ms_L2_ep6_scaleup_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-arcface_verifythresh-30/',
        BASEDIR + 'v22c/feats_train_ms_L2_ep4_scaleup_ep3_base_margin-0.4_freqthresh-2_verifythresh-30/',
        BASEDIR + 'v23c/feats_train_ms_L2_ep6_scaleup_ep5_augmentation-middle_epochs-7_freqthresh-3_verifythresh-30/',
        BASEDIR + 'v24c/feats_train_ms_L2_ep5_augmentation-middle_epochs-7_freqthresh-3_loss-cosface_verifythresh-30/',
    ],
}

MODELSET_V0602_WEIGHT_V2 = [
    0.6,
    1.1,
    1.0,
    0.6,
    1.0,
    1.1,
]

MODELSET_V0602_WEIGHT_V3 = [
    0.5,
    1.2,
    1.2,
    0.5,
    1.2,
    1.2,
]
