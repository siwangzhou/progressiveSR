import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from typing import Sequence
import numpy as np
from numpy import random
from einops import rearrange, reduce, repeat
import cv2
import math
import numpy as np

class MyRotateTransform:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


class MyDataset(Dataset):
    def __init__(self, alist=["D:/DYQ/DATASETS/DIV2K_Crop/DIV2K_Crop_train_HR", "D:/DYQ/DATASETS/Flickr2K_Crop/Flickr2K_Crop"]):
        super(MyDataset, self).__init__()
        self.alist=alist
        self.hrimgs=[]
        self.lrimgs = []
        i=0
        for path in self.alist:
            train_list=os.listdir(path)
            for name in train_list:
                i+=1
                img_path=path+"/"+name
                self.hrimgs.append(img_path)

    def __len__(self):
        return len(self.hrimgs)
    def __getitem__(self, index):
        hrsize = 64
        lrsize = 30
        self.transforms = transforms.Compose([transforms.RandomCrop(hrsize), ])
        temp_img = Image.open(self.hrimgs[index]).convert('RGB')
        sourceImg = self.transforms(temp_img)
        cropimg = sourceImg.resize((lrsize, lrsize), Image.BICUBIC)

        hr = transforms.ToTensor()(sourceImg)
        lr = transforms.ToTensor()(cropimg)
        flip_ran = random.randint(0, 2)

        if flip_ran == 0:
            # horizontal
            hr = torch.flip(hr, [1])
            lr = torch.flip(lr, [1])
        elif flip_ran == 1:
            # vertical
            hr = torch.flip(hr, [2])
            lr = torch.flip(lr, [2])

        rot_ran = random.randint(0, 3)

        if rot_ran != 0:
            # horizontal
            hr = torch.rot90(hr, rot_ran, [1, 2])
            lr = torch.rot90(lr, rot_ran, [1, 2])

        hr = (hr - 0.5) * 2.0
        lr = (lr - 0.5) * 2.0

        return lr, hr

class Test(Dataset):
    def __init__(self, alist=["D:/DYQ/DATASETS/Set5/HR"]):
        super(Test, self).__init__()
        self.alist = alist
        self.imgshr = []
        self.imgslr = []
        i=0
        for path in self.alist:
            test_list = os.listdir(path)
            for name in test_list:
                i+=1
                img_path = path+"/"+name
                self.imgshr.append(img_path)

    def __len__(self):
        return len(self.imgshr)

    # 对应32卷积核的下采样方法,为了适应32卷积核，需要将大图进行裁剪为32的倍数
    def __getitem__(self, index):
        scales = 32
        temp_img = Image.open(self.imgshr[index]).convert('RGB')
        size = (np.array(temp_img.size) / scales).astype(int)
        w = size[0]
        h = size[1]
        sourceImg = transforms.RandomCrop((h*scales, w*scales))(temp_img)
        cropimg = sourceImg.resize((w*15, h*15), Image.BICUBIC)

        hr = transforms.ToTensor()(sourceImg)
        lr = transforms.ToTensor()(cropimg)

        hr = (hr - 0.5) * 2.0
        lr = (lr - 0.5) * 2.0

        return lr, hr


if __name__ == '__main__':
    a = MyDataset()
    n = a.__len__()
    a.__getitem__(1)
