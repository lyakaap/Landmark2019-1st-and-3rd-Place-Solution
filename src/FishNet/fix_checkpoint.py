import torch
from src.FishNet import models
from src import utils

ckpt_path = '../src/FishNet/checkpoints/fishnet150_ckpt_welltrained.tar'
ckpt = torch.load(ckpt_path)
ckpt['state_dict'] = utils.remove_redundant_keys(ckpt['state_dict'])
model = models.__dict__[ckpt['arch']]()

# missing keysはmodelの定義にあるのにcheckpointにないもの
# unexpected keysはcheckpointにはあるのにmodelの定義にないもの

# mapping
# model:302 <- 310 conv
# model:303 <- 311 bn
# model:31 <- 313 conv
# model:932 <- 940:ckpt
# model:933 <- 941:ckpt
# model:941 <- 944

ckpt['state_dict']['fish.fish.3.0.2.weight'] = ckpt['state_dict']['fish.fish.3.1.0.weight']
del ckpt['state_dict']['fish.fish.3.1.0.weight']

for attr in ['weight', 'bias', 'running_mean', 'running_var']:
    ckpt['state_dict']['fish.fish.3.0.3.' + attr] = ckpt['state_dict']['fish.fish.3.1.1.' + attr]
    del ckpt['state_dict']['fish.fish.3.1.1.' + attr]

for attr in ['weight', 'bias']:
    ckpt['state_dict']['fish.fish.3.1.' + attr] = ckpt['state_dict']['fish.fish.3.1.3.' + attr]
    del ckpt['state_dict']['fish.fish.3.1.3.' + attr]

ckpt['state_dict']['fish.fish.9.3.2.weight'] = ckpt['state_dict']['fish.fish.9.4.0.weight']
del ckpt['state_dict']['fish.fish.9.4.0.weight']

for attr in ['weight', 'bias', 'running_mean', 'running_var']:
    ckpt['state_dict']['fish.fish.9.3.3.' + attr] = ckpt['state_dict']['fish.fish.9.4.1.' + attr]
    del ckpt['state_dict']['fish.fish.9.4.1.' + attr]

for attr in ['weight', 'bias']:
    ckpt['state_dict']['fish.fish.9.4.1.' + attr] = ckpt['state_dict']['fish.fish.9.4.4.' + attr]
    del ckpt['state_dict']['fish.fish.9.4.4.' + attr]

# check
model.load_state_dict(ckpt['state_dict'], strict=True)

torch.save(ckpt, ckpt_path)


# ======================= fishnet150 ↑ ↓ fishnet201 ==================== #


import torch
from src.FishNet import models
from src import utils

ckpt_path = '../src/FishNet/checkpoints/fishnet201_ckpt_welltrain.tar'
ckpt = torch.load(ckpt_path)
ckpt['state_dict'] = utils.remove_redundant_keys(ckpt['state_dict'])
model = models.__dict__[ckpt['arch']]()

# missing keysはmodelの定義にあるのにcheckpointにないもの
# unexpected keysはcheckpointにはあるのにmodelの定義にないもの

# mapping
# model: 302 <- 310 conv
# model: 303 <- 311 bn
# model: 31 <- 313 conv
# model: 930 <- 920 bn
# model: 933 <- 931 bn
# model: 941 <- 934 conv
# model: 932 <- 930 fc

for attr in ['weight']:
    ckpt['state_dict']['fish.fish.0.0.0.shortcut.2.' + attr] = ckpt['state_dict']['fish.fish.0.0.0.shortcut.' + attr]
    del ckpt['state_dict']['fish.fish.0.0.0.shortcut.' + attr]

for attr in ['weight']:
    ckpt['state_dict']['fish.fish.3.0.2.' + attr] = ckpt['state_dict']['fish.fish.3.1.0.' + attr]
    del ckpt['state_dict']['fish.fish.3.1.0.' + attr]

for attr in ['weight', 'bias', 'running_mean', 'running_var']:
    ckpt['state_dict']['fish.fish.3.0.3.' + attr] = ckpt['state_dict']['fish.fish.3.1.1.' + attr]
    del ckpt['state_dict']['fish.fish.3.1.1.' + attr]

for attr in ['weight', 'bias']:
    ckpt['state_dict']['fish.fish.3.1.' + attr] = ckpt['state_dict']['fish.fish.3.1.3.' + attr]
    del ckpt['state_dict']['fish.fish.3.1.3.' + attr]

for attr in ['weight', 'bias', 'running_mean', 'running_var']:
    ckpt['state_dict']['fish.fish.9.3.0.' + attr] = ckpt['state_dict']['fish.fish.9.2.0.' + attr]
    del ckpt['state_dict']['fish.fish.9.2.0.' + attr]

for attr in ['weight', 'bias', 'running_mean', 'running_var']:
    ckpt['state_dict']['fish.fish.9.3.3.' + attr] = ckpt['state_dict']['fish.fish.9.3.1.' + attr]
    del ckpt['state_dict']['fish.fish.9.3.1.' + attr]

for attr in ['weight']:
    ckpt['state_dict']['fish.fish.9.3.2.' + attr] = ckpt['state_dict']['fish.fish.9.3.0.' + attr]
    del ckpt['state_dict']['fish.fish.9.3.0.' + attr]

for attr in ['weight', 'bias']:
    ckpt['state_dict']['fish.fish.9.4.1.' + attr] = ckpt['state_dict']['fish.fish.9.3.4.' + attr]
    del ckpt['state_dict']['fish.fish.9.3.4.' + attr]

# check
model.load_state_dict(ckpt['state_dict'], strict=False)

torch.save(ckpt, ckpt_path)
