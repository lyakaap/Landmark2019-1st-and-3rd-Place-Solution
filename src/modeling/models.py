import torch
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels

from src import utils
from src.FishNet import models
from .metric_learning import ArcMarginProduct, AddMarginProduct, AdaCos
import cirtorch

ROOT = '../'


class LandmarkNet(nn.Module):

    DIVIDABLE_BY = 32

    def __init__(self,
                 n_classes,
                 model_name='resnet50',
                 pooling='GeM',
                 args_pooling: dict={},
                 use_fc=False,
                 fc_dim=512,
                 dropout=0.0,
                 loss_module='softmax',
                 s=30.0,
                 margin=0.50,
                 ls_eps=0.0,
                 theta_zero=0.785):
        """
        :param n_classes:
        :param model_name: name of model from pretrainedmodels
            e.g. resnet50, resnext101_32x4d, pnasnet5large
        :param pooling: One of ('SPoC', 'MAC', 'RMAC', 'GeM', 'Rpool', 'Flatten', 'CompactBilinearPooling')
        :param loss_module: One of ('arcface', 'cosface', 'softmax')
        """
        super(LandmarkNet, self).__init__()

        self.backbone = getattr(pretrainedmodels, model_name)(num_classes=1000)
        final_in_features = self.backbone.last_linear.in_features
        # HACK: work around for this issue https://github.com/Cadene/pretrained-models.pytorch/issues/120
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        self.pooling = getattr(cirtorch.pooling, pooling)(**args_pooling)

        self.use_fc = use_fc
        if use_fc:
            self.dropout = nn.Dropout(p=dropout)
            self.fc = nn.Linear(final_in_features, fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)
            self._init_params()
            final_in_features = fc_dim

        if loss_module == 'arcface':
            self.final = ArcMarginProduct(final_in_features, n_classes,
                                          s=s, m=margin, easy_margin=False, ls_eps=ls_eps)
        elif loss_module == 'cosface':
            self.final = AddMarginProduct(final_in_features, n_classes, s=s, m=margin)
        elif loss_module == 'adacos':
            self.final = AdaCos(final_in_features, n_classes, m=margin, theta_zero=theta_zero)
        else:
            self.final = nn.Linear(final_in_features, n_classes)

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x, label):
        feature = self.extract_feat(x)
        logits = self.final(feature, label)
        return logits

    def extract_feat(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.pooling(x).view(batch_size, -1)

        if self.use_fc:
            x = self.dropout(x)
            x = self.fc(x)
            x = self.bn(x)

        return x


class LandmarkFishNet(nn.Module):

    DIVIDABLE_BY = 32
    dims = [320, 832, 1600, 2112]

    pooling_factory = {
        'S': cirtorch.pooling.SPoC(),
        'M': cirtorch.pooling.MAC(),
        'G': cirtorch.pooling.GeM(p=3.0, freeze_p=True),
    }

    def __init__(self,
                 n_classes,
                 model_name='fishnet150',
                 use_fc=False,
                 fc_dim=512,
                 dropout=0.0,
                 loss_module='softmax',
                 s=30.0,
                 margin=0.50,
                 ls_eps=0.0,
                 theta_zero=0.785,
                 pooling_strings=('G', 'G', 'G', 'G'),
                 pretrained=True):
        """
        :param loss_module: One of ('arcface', 'cosface', 'softmax')

        feature dimension from fish-head layers (h,w: 224x224):
            7:[320, 56, 56], 8:[832, 28, 28], 9:[1600, 14, 14], 10:[2112, 7, 7]
        """
        super(LandmarkFishNet, self).__init__()

        self.backbone = models.__dict__[model_name]()
        if pretrained:
            ckpt_path = ROOT + f'src/FishNet/checkpoints/{model_name}_ckpt_welltrained.tar'
            ckpt = torch.load(ckpt_path)
            ckpt['state_dict'] = utils.remove_redundant_keys(ckpt['state_dict'])
            self.backbone.load_state_dict(ckpt['state_dict'], strict=True)
        self.use_fc = use_fc
        self.pooling_strings = pooling_strings
        final_in_features = self._calc_output_dim()

        if use_fc:
            self.dropout = nn.Dropout(p=dropout)
            self.fc = nn.Linear(final_in_features, fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)
            self._init_params()
            final_in_features = fc_dim

        if loss_module == 'arcface':
            self.final = ArcMarginProduct(final_in_features, n_classes,
                                          s=s, m=margin, easy_margin=False, ls_eps=ls_eps)
        elif loss_module == 'cosface':
            self.final = AddMarginProduct(final_in_features, n_classes, s=s, m=margin)
        elif loss_module == 'adacos':
            self.final = AdaCos(final_in_features, n_classes, m=margin, theta_zero=theta_zero)
        else:
            self.final = nn.Linear(final_in_features, n_classes)

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def _calc_output_dim(self):
        return sum(d * len(p) for d, p in zip(self.dims, self.pooling_strings))

    def _pool_concat(self, feats):
        batch_size = feats[0].shape[0]
        outputs = []
        for feat, pooling_str in zip(feats, self.pooling_strings):
            for p in pooling_str:
                output = self.pooling_factory[p](feat).view(batch_size, -1)
                outputs.append(output)
        return torch.cat(outputs, dim=1)

    def forward(self, x, label):
        feature = self.extract_feat(x)
        logits = self.final(feature, label)
        return logits

    def extract_feat(self, x):
        x = self.backbone.features(x)
        x = self._pool_concat(x)

        if self.use_fc:
            x = self.dropout(x)
            x = self.fc(x)
            x = self.bn(x)

        return x
