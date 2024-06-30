import argparse
import logging
import os
import sys
import timm
from timm.models import create_model

import numpy as np
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as tvF
from torch.backends import cudnn
from convnext import convnext_base
from replknet import replknet_base
from timm.models.resnetv2 import _cfg

os.environ['CUDA_VISIBLE_DEVICES'] = "6"
# import models as MODEL

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--dir', type=str, default="./results/replknet_base_ep16")
parser.add_argument('--model_dir', type=str, default=None)
args = parser.parse_args()

# if not os.path.isdir("test_logs"):
#     os.mkdir("test_logs")
#
# logging.basicConfig(filename='test_logs/{}.log'.format(args.dir.split('/')[-1]), level=logging.INFO)
#
# logging.info(args)


cudnn.benchmark = False
cudnn.deterministic = True
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)



if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def get_model(model_name):
    if model_name == 'resnet_TAIG':
        # model = torchvision.models.resnet50(pretrained=True)
        # state_dict = torch.load("./pretrained_model/resnet50_a1_0-14fe96d1.pth")
        model = timm.create_model("resnet50", pretrained=False)
        state_dict = torch.load("official_ckpts/resnet50_a1_0-14fe96d1.pth")
        model.load_state_dict(state_dict, strict=True)
    elif model_name == 'convnext_base':
        model = convnext_base()
        state_dict = torch.load("official_ckpts/convnext_base_22k_1k_224.pth")["model"]
        model.load_state_dict(state_dict, strict=False)
    elif model_name == "vit_base":
        model = timm.create_model("vit_base_patch16_224", pretrained=False)
        state_dict = torch.load("/official_ckpts/vit_b.bin")
        model.load_state_dict(state_dict, strict=True)
    elif model_name == 'replknet_base':
        model = replknet_base()
        pretrained_weights = "/official_ckpts/RepLKNet-31B_ImageNet-22K-to-1K_224.pth"
        ckpt = torch.load(pretrained_weights, map_location="cpu")
        if "model" in ckpt:
            msg = model.load_state_dict(ckpt["model"])
        else:
            msg = model.load_state_dict(ckpt) 
        print(msg)  
    else:
        print('No implemation')
    return model

def normalize(x, ms=None):
    if ms == None:
        ms = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]
    for i in range(x.shape[1]):
        x[:,i] = (x[:,i] - ms[0][i]) / ms[1][i]
    return x

def load_img(img, trans):
    img_pil = tvF.to_pil_image(img)
    return trans(img_pil)

class NumpyImages(torch.utils.data.Dataset):
    def __init__(self, npy_dir, transforms=None):
        super(NumpyImages, self).__init__()
        npy_ls = []
        for npy_name in os.listdir(npy_dir):
            if npy_name[:5] == 'batch':
                npy_ls.append(npy_name)
        self.data = []
        for npy_ind in range(len(npy_ls)):
            self.data.append(np.load(npy_dir + '/batch_{}.npy'.format(npy_ind)))
        self.data = np.concatenate(self.data, axis=0)
        self.target = np.load(npy_dir+"/labels.npy")
    
    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]).float() / 255, self.target[index]
    
    def __len__(self,):
        return len(self.target)


dataset = NumpyImages(args.dir, transforms=None)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=200, shuffle=False, num_workers=4)

def test(model, trans, dataloader=dataloader):
    img_num = 0
    count = 0
    dir_ls = os.listdir(args.dir)
    for img, label in dataloader:
        label = label.to(device)
        img = img.to(device)
        with torch.no_grad():
            pred = torch.argmax(model(trans(img)), dim=1).view(1,-1)
        count += (label != pred.squeeze(0)).sum().item()
        img_num += len(img)
    return round(100. * count / img_num, 2)


trans_pnas = T.Compose([
    T.Resize((256,256)),
    T.CenterCrop((224,224)),
    T.Resize((331, 331)),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
trans_se = T.Compose([
    T.Resize((256,256)),
    T.CenterCrop((224,224)),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
trans_hh = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),  
        T.CenterCrop(224),
        #T.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



#model_name_list = ["resnet_TAIG","convnext_tiny", "convnext_base" ,  "convnext_large",
#                    "vit_base" ,  "vit_large", "bit_152x4", "replknet_base"]

model_name_list = ["replknet_base"]
print("results_path: ",args.dir)
for model_name in model_name_list:
    model = get_model(model_name)

    model.to(device)
    model.eval()
    acc = test(model = model, trans = trans_hh)
    print(model_name,"_acc:",acc)

