import sys

import torch
import torch.nn as nn

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

sys.path.insert(0, '..')
import torchattacks
#from engine import evaluate, evaluate_PGD, evaluate_FGSM

from torchvision import models
from utils import get_imagenet_data, clean_accuracy
#from robustbench.utils import clean_accuracy
#from tqdm import tqdm
#from timm.utils import accuracy
#from timm.models import create_model
from replknet import replknet_base
import numpy as np
import matplotlib.pyplot as plt

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

print('[Data loaded]')

device = "cuda"

model = replknet_base()
pretrained_weights = "/checkpoints/RepLKNet-31B_ImageNet-1K_224.pth"

if pretrained_weights is not None:
    if pretrained_weights.startswith("https://"):
        ckpt = torch.hub.load_state_dict_from_url(url=pretrained_weights, map_location="cpu")
    else:
        ckpt = torch.load(pretrained_weights, map_location="cpu")
    if "model" in ckpt:
        msg = model.load_state_dict(ckpt["model"])
    else:
        msg = model.load_state_dict(ckpt)   

    print(msg)

stages = 4
blocks = [2,2,18,2]
rs = [15,14,13,6]
#get weight
for m in range(stages):
    R = rs[m]
    for n in range(blocks[m]):
        weight = model.stages[m].blocks[n*2].large_kernel.lkb_origin.conv.weight.data
        weight = weight.squeeze(1)
        a = np.array(weight)
        
        #draw
        distances = np.zeros_like(a)
        print(a.shape)
        max_dis = R * np.sqrt(2)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                for k in range(a.shape[2]):
                    distances[i, j, k] = np.sqrt((j - R)**2 + (k - R)**2)
                    distances[i, j, k] = distances[i, j, k] / max_dis
        result = np.multiply(distances, np.abs(a))
        mean_result = np.sum(result, axis=(1, 2))
        sorted_indices = np.argsort(mean_result)
        sorted_mean_result = mean_result[sorted_indices]
        
        save_npy = "./kernel_dis_npy/1K_s" + str(m) + "_b" + str(n) + ".npy"
        np.save(save_npy, sorted_mean_result)
        plt.plot(sorted_mean_result)
        save_name = "./kernel_dis_fig/1K_s" + str(m) + "_b" + str(n) + ".png"
        plt.savefig(save_name)