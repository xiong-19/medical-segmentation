from model import *
import numpy as np
import torch
import argparse
import copy
import cv2


NCLASSES = 2
BATCH_SIZE = 4

#文件的加载路径
parser = argparse.ArgumentParser()
parser.add_argument('--val_txt', type=str, default='val.txt', help='about validation')
parser.add_argument('--weights', type=str, default='model.pth', help='weights')
opt = parser.parse_args()
print(opt)

txt_path = opt.val_txt
weight=opt.weights




__all__ = ['SegmentationMetric']


#读取val.txt中的图片的名称
paths = open("%s" % txt_path, "r")
data = []

for lines in paths:
    path = lines.rstrip('\n')
    data.append(path)

device=torch.device("cuda:0"if torch.cuda.is_available() else "cpu")   #检测是否有GPU加速

model = Ma_Net(3, NCLASSES)              #初始化model

model.load_state_dict(torch.load(opt.weights))     #加载权重

model.to(device)

model.eval()

for i in range(len(data)):
    image1 = cv2.imread("train_dataset/img/%s" % data[i] + ".png", -1)
    label = cv2.imread("train_dataset/mask/%s" % data[i] + ".png",0)


    orininal_h = image1.shape[0]               # 读取的图像的高
    orininal_w = image1.shape[1]               # 读取的图像的宽

    image1 = cv2.resize(image1, dsize=(224, 224))

    label = cv2.resize(label, dsize=(224, 224))

    label = label / 255.0
    label[label > 0] = 1
    label = torch.tensor(label)

    #image1
    image1 = image1 / 255.0          # 图像归一化
    image1 = torch.from_numpy(image1)
    image1 = image1.permute(2, 0, 1)             # 显式的调转维度

    image1 = torch.unsqueeze(image1, dim=0)             # 改变维度,使得符合model input size
    image1 = image1.type(torch.FloatTensor)             # 数据转换,否则报错
    image1 = image1.to(device)

    # 放入GPU中计算
    model.eval()
    predict = model(image1).cpu()

    predict = torch.squeeze(predict)               # [1,1,416,416]---->[1,416,416]
    predict = predict.permute(1, 2, 0)

    predict = predict.detach().numpy()

    prc = predict.argmax(axis=-1)

    prc = torch.tensor(prc)
    print(prc.shape)
    cv2.imshow(prc)

