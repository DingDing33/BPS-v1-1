import json
import cv2
import numpy as np
import os
from PIL import Image
import pandas as pd
import pickle as pkl
import torch
import torch.nn as nn

from random import shuffle
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from annotator.util import resize_image
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomGrayscale, Compose


def resize_image_v2(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    kh = float(resolution) / H
    kw = float(resolution) / W
    H *= kh
    W *= kw
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if max(kh,kw) > 1 else cv2.INTER_AREA)
    return img



class COCODataset(Dataset):
    def __init__(self, 
                 image_resolution= 512,
                 source='pidinet',
                 root = './data/coco2017',
                 val_json = 'prompt_val2017.json',
                 train_json = 'prompt_train2017.json',
                 test_json = 'prompt_test2017_blip.json',
                 val_blip_json = 'prompt_val2017_blip.json',
                 train_blip_json = 'prompt_train2017_blip.json',
                 train = True):
        
        self.image_resolution = image_resolution
        self.root = root
        self.val_json = val_json
        self.train_json = train_json
        self.test_json = test_json
        self.source = source

        val_data, train_data, test_blip = self.get_data("val"), self.get_data("train"), self.get_data("test")
        self.val_train_blip = json.load(open(os.path.join(self.root, val_blip_json), "r")) + json.load(open(os.path.join(self.root, train_blip_json), "r"))
        self.train = train
        self.data =  val_data+train_data+test_blip
        self.data = self.data
    def get_data(self, train_val_test):
        if train_val_test == "val":
            json_list = json.load(open(os.path.join(self.root, self.val_json), "r"))
        elif train_val_test == "train":
            json_list = json.load(open(os.path.join(self.root, self.train_json), "r"))
        else:
            json_list = json.load(open(os.path.join(self.root, self.test_json), "r"))
            
        for i in json_list:
            i['sfile_name'] = os.path.join(self.root, train_val_test+'2017_'+self.source, i['file_name'].replace('jpg','npy'))
            i['file_name'] = os.path.join(self.root, train_val_test+'2017', i['file_name'])
            i['dataset'] = train_val_test
            
        return json_list
    def __getitem__(self, idx):
        item = self.data[idx]

        target_path = item['file_name']
        target = cv2.imread(target_path)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target = resize_image_v2(target, self.image_resolution)

        source_path = item['sfile_name']
        source = np.load(source_path)
        source = resize_image_v2(source[:,:,None], self.image_resolution)
        c_img = np.array(Image.fromarray(source).convert('RGB')).transpose(-1,0,1) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0
        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        if self.train:
            if item['dataset'] != "test":
                p = np.random.multinomial(1, [0.33, 0.33, 0.34], size=1).squeeze()
                if p[0] != 0:
                    prompt = "a high-quality, detailed, and professional image"
                elif p[1] != 0:
                    prompt = self.val_train_blip[idx]['prompt']
                else:
                    prompt = item['prompt']
            else:
                p = np.random.multinomial(1, [1/2.]*2, size=1).squeeze()
                if p[0] != 0:
                    prompt = "a high-quality, detailed, and professional image"
                else:
                    prompt = item['prompt']
        else:
            prompt = item['prompt']

        return dict(jpg=target, txt=prompt, hint=source[:,:,None], c_img = c_img)
