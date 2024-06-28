import sys

import torch
import torch.nn as nn

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

sys.path.insert(0, '..')
import torchattacks
from engine import evaluate, evaluate_PGD, evaluate_FGSM

from torchvision import models
from utils import get_imagenet_data, clean_accuracy
#from tqdm import tqdm
#from timm.utils import accuracy
#from timm.models import create_model
from replknet import replknet_base

budget = 0.5
step = 1
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

test_loader = get_imagenet_data(mean=mean, std=std)
print('[Data loaded]')

device = "cuda"

model = replknet_base()
pretrained_weights = "/checkpoints/RepLKNet-31B_ImageNet-22K-to-1K_224.pth"

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


model.to(device).eval()

# imagenet-val
test_stats = evaluate(test_loader, model, device)

# PGD attack
#test_stats = evaluate_PGD(test_loader, model, device, mean=mean, std=std, budget=budget, alpha=budget/4, step=step)

# FGSM attack
#test_stats = evaluate_FGSM(test_loader, model, device, mean=mean, std=std, budget=budget)

print(f"Accuracy of the network on 50000 test images: {test_stats['acc1']:.5f}%")
