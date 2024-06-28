import timm
import torch
import torch.nn as nn
import copy
import timm
import torch
import torch.nn as nn
import requests
import torch
import numpy as np

from PIL import Image
from einops import rearrange, reduce, repeat
import math

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from timm.models import create_model
from tqdm import tqdm

from replknet import replknet_base
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class PatchEmbed(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = copy.deepcopy(model)

    def forward(self, x, **kwargs):
        x = self.model.patch_embed(x)
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.model.pos_drop(x + self.model.pos_embed)
        return x


class Residual(nn.Module):
    def __init__(self, *fn):
        super().__init__()
        self.fn = nn.Sequential(*fn)

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class Lambda(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)

def flatten(xs_list):
    return [x for xs in xs_list for x in xs]



def fourier(x):  # 2D Fourier transform
    f = torch.fft.fft2(x)
    f = f.abs() + 1e-6
    f = f.log()
    return f


def shift(x):  # shift Fourier transformed feature map
    b, c, h, w = x.shape
    return torch.roll(x, shifts=(int(h / 2), int(w / 2)), dims=(2, 3))


def make_segments(x, y):  # make segment for `plot_segment`
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def plot_segment(ax, xs, ys, cmap_name="plasma"):  # plot with cmap segments
    z = np.linspace(0.0, 1.0, len(ys))
    z = np.asarray(z)

    cmap = cm.get_cmap(cmap_name)
    norm = plt.Normalize(0.0, 1.0)
    segments = make_segments(xs, ys)
    lc = LineCollection(segments, array=z, cmap=cmap_name, norm=norm,
                        linewidth=2.5, alpha=1.0)
    ax.add_collection(lc)

    colors = [cmap(x) for x in xs]
    ax.scatter(xs, ys, color=colors, marker=marker, zorder=100)


import os
os.environ['CUDA_VISIBLE_DEVICES'] = "6"
device = torch.device("cuda:0")
### 1.model prepare

#name = "resnet50"
#name = 'replknet'
name = "vit_base_patch16_224"
print("Drawing: "+name)

if name == "resnet50":
    model = create_model(name, pretrained=True)
    model.to(device)
    named_parameters_list = []
    for n,p in model.named_parameters():
        named_parameters_list.append(n)
    # model → blocks. `blocks` is a sequence of blocks
    blocks = [
        nn.Sequential(model.conv1, model.bn1, model.act1, model.maxpool),
        *model.layer1,
        *model.layer2,
        *model.layer3,
        *model.layer4,
        nn.Sequential(model.global_pool, model.fc)
    ]
    drop_last_feature = True
elif name == 'replknet':
    model = replknet_base()
    pretrained_weights = "/checkpoints/RepLKNet-31B_ImageNet-1K_224.pth"
    if pretrained_weights.startswith("https://"):
        ckpt = torch.hub.load_state_dict_from_url(url=pretrained_weights, map_location="cpu")
    else:
        ckpt = torch.load(pretrained_weights, map_location="cpu")
    if "model" in ckpt:
        msg = model.load_state_dict(ckpt["model"])
    else:
        msg = model.load_state_dict(ckpt)   
    print(msg)

    model.to(device)
    model.eval()
    blocks = [*model.stem,
              *model.stages[0].blocks,
              *model.transitions[0],
              *model.stages[1].blocks,
              *model.transitions[1],
              *model.stages[2].blocks,
              *model.transitions[2],
              *model.stages[3].blocks,
              ]
    drop_last_feature = True
elif name == "vit_base_patch16_224":
    model = create_model("vit_base_patch16_224", pretrained=True)
    model.to(device)
    model.eval()
    blocks = [
        PatchEmbed(model),
        *flatten([[Residual(b.norm1, b.attn), Residual(b.norm2, b.mlp)]
                  for b in model.blocks]),
        nn.Sequential(model.norm, Lambda(lambda x: x[:, 0]), model.head),
    ]
    drop_last_feature = True


#### 2. plot fourier

#imagenet_mean = np.array([0.485, 0.456, 0.406])
#imagenet_std = np.array([0.229, 0.224, 0.225])

# load a sample ImageNet-1K image -- use the full val dataset for precise results
DATA_ROOT = '/dataset/imagenet/val'

batch_size = 128
dataset_size = 1280
num_sweep = 10

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
#normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])

torch.random.manual_seed(0)
perms = [torch.randperm(dataset_size) for _ in range(num_sweep)]
dataset = datasets.ImageFolder(DATA_ROOT, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

# accumulate `latents` by collecting hidden states of a model
latents = []
with torch.no_grad():
    for sweep in range(num_sweep):
        dataset_sweep = torch.utils.data.Subset(dataset, perms[sweep])
        data_loader = torch.utils.data.DataLoader(
            dataset_sweep,
            batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True)
        for images, targets in tqdm(data_loader):
            xs = images.to(device)
            # features = forward_features(model, images)
            # out = model(xs)
            latents_item = []
            with torch.no_grad():
                for block in blocks:
                    xs = block(xs)
                    latents_item.append(torch.mean(xs,dim=0))
            # torch.cuda.empty_cache()
        latents.append(latents_item)

latents_final = []
ds_len = len(latents)
layer_len = len(latents[0])
for layer_index in range(layer_len):
    layer_all_data = []
    for ds_index in range(ds_len):
        layer_all_data.append(latents[ds_index][layer_index])
    latents_final.append(torch.stack(layer_all_data,dim=0))

latents  = latents_final
if name in ["vit_base_patch16_224"]:  # for ViT: Drop CLS token
    latents = [latent[:, 1:] for latent in latents]
if drop_last_feature:
    latents = latents[:-1]  # drop logit (output)

######### 3.plot results

# aggregate feature map variances
variances = []
for latent in latents:  # `latents` is a list of hidden feature maps in latent spaces
    latent = latent.cpu()

    if len(latent.shape) == 3:  # for ViT
        b, n, c = latent.shape
        h, w = int(math.sqrt(n)), int(math.sqrt(n))
        latent = rearrange(latent, "b (h w) c -> b c h w", h=h, w=w)
    elif len(latent.shape) == 4:  # for CNN
        b, c, h, w = latent.shape
    else:
        raise Exception("shape: %s" % str(latent.shape))

    variances.append(latent.var(dim=[-1, -2]).mean(dim=[0, 1]))

# Plot Fig 9: "Feature map variance"
import numpy as np
import matplotlib.pyplot as plt

if name == "resnet50":  # for ResNet-50
    pools = [4, 8, 14]
    msas = []
    marker = "D"
    color = "tab:blue"
else:
    import warnings
    warnings.warn("The configuration for %s are not implemented." %name, Warning)
    pools, msas = [], []
    marker = "s"
    color = "tab:green"

depths = range(len(variances))

save_np = np.array(variances)
np.save("./fig_featuremap_var/vitb.npy", save_np)

# normalize
depth = len(depths) - 1
depths = (np.array(depths)) / depth
pools = (np.array(pools)) / depth
msas = (np.array(msas)) / depth

fig, ax = plt.subplots(1, 1, figsize=(6.5, 4), dpi=200)
ax.plot(depths, variances, marker=marker, color=color, markersize=7)

for pool in pools:
    ax.axvspan(pool - 1.0 / depth, pool + 0.0 / depth, color="tab:blue", alpha=0.15, lw=0)
for msa in msas:
    ax.axvspan(msa - 1.0 / depth, msa + 0.0 / depth, color="tab:gray", alpha=0.15, lw=0)

ax.set_xlim(left=0, right=1.0)
ax.set_ylim(bottom=0.0, )

ax.set_xlabel("Normalized depth")
ax.set_ylabel("Feature map variance")

#plt.show()
plt.savefig("./fig_featuremap_var/test.png")