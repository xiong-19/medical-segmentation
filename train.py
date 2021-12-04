import torch
import cv2
import os
import argparse
import numpy as np
from PIL import Image
from torch.nn import *
from torch.optim import Adam
from set import *
from torch.utils.data import DataLoader
from model import *

BATCH_SIZE1 = 4         #训练的batch_size
NUM_CLASSES = 2                 #分割的种类数
LR= 1e-4                            #学习率
EPOCH = 40             #迭代次数


parser = argparse.ArgumentParser()
parser.add_argument('--gpu',action='store_true',default=True,help='whether use gpu')
parser.add_argument('--train_txt', type=str, default='train.txt', help='about trian')
parser.add_argument('--val_txt', type=str, default='val.txt', help='about val')

opt = parser.parse_args()
txt = opt.train_txt
val = opt.val_txt


train_dataset = AirplanesDataset(txt)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE1,
                          shuffle=True,drop_last=True)


criterion = CrossEntropyLoss()  # Loss
model = Ma_Net(3, NUM_CLASSES)



optimizer = Adam(model.parameters(),  # 优化器
                lr=LR)
                
model.load_state_dict(torch.load("model.pth"))


device=torch.device("cuda:0"if torch.cuda.is_available() else "cpu")   #检测是否有GPU加速
model.to(device)       #网络放入GPU里加速

def train(epoch):
    running_loss = 0.0
    for batch_idx,data in enumerate(train_loader,0):         #0是表示从0开始
        image =data[0]
        label = data[1]

        image,label = image.to(device),label.to(device)         #数据放进GPU里
        optimizer.zero_grad()                  #优化器参数清零
        #forword+backward+update
        image =image.type(torch.FloatTensor)        #转化数据类型,不转则会报错
        image = image.to(device)

        outputs=model(image)

        loss=criterion(outputs,label.long())        #进行loss计算

        lll=label.long().cpu().numpy()             #把label从GPU放进CPU

        loss.backward(retain_graph=True)                  #反向传播(求导)
        optimizer.step()            #优化器更新model权重
        running_loss += loss.item()       #收集loss的值

        if batch_idx % 5 == 0:
            print('[epoch: %d,idex: %2d] loss:%.3f' % (epoch+1,batch_idx+1,running_loss/36))  #训练集的数量,可根据数据集调整
    if (epoch+1)%10 == 0:
        torch.save(model.state_dict(),f='model.pth') #保存权重


for epoch in range(EPOCH):    #迭代次数
    train(epoch)


