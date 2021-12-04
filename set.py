import torch
import cv2
import os
import argparse
import numpy as np
from PIL import Image
from torch.nn import *
from torch.optim import Adam
from torch.utils.data import Dataset,DataLoader

#自定义数据集的类
class AirplanesDataset(Dataset):
    def __init__(self,txt_path):
        super(AirplanesDataset, self).__init__()
        paths=open("%s" % txt_path,"r")
        data=[]
        for lines in paths:
            path=lines.rstrip('\n')
            data.append(path)
        self.data=data
        self.len=len(data)
    def __getitem__(self, index):
        image=cv2.imread("train_dataset/img/%s" %self.data[index]+".png",-1)
        label = cv2.imread("train_dataset/mask/%s"%self.data[index] +".png",0) #灰度图
        image = cv2.resize(image, dsize=(224, 224))
        label = cv2.resize(label, dsize=(224, 224))
        label = np.array(label)
        label = label / 255.0

        label[label > 0] = 1
        image = np.array(image)
        image = np.transpose(np.array(image), [2, 0, 1]) / 255
        return image,label
    def __len__(self):
        return self.len

