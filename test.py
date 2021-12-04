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

class SegmentationMetric(object):                 #计算mIoU、accuracy的类
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        acc = round(acc,5)
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        mIoU =round(mIoU,4)
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))



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


sum_1 = 0  # 累加每张图片val的accuracy
sum_2 = 0  # 累积每张图片Val的mIoU

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
    #进行mIoU和accuracy的评测
    imgPredict =prc
    imgLabel = label
    #print(imgPredict.shape)
    #print(imgLabel.shape)
    metric = SegmentationMetric(3)
    metric.addBatch(imgPredict, imgLabel)
    acc = metric.pixelAccuracy()
    sum_1+=acc
    mIoU = metric.meanIntersectionOverUnion()
    sum_2+=mIoU
    print("%s.jpg :" % data[i])
    print("accuracy:  "+str(acc*100)+" %")
    print("mIoU:  " +str(mIoU))
    print("-------------------")


# 全部图片平均的accuracy和mIoU
sum_1=sum_1/len(data)
sum_2=sum_2/len(data)

sum_1 = round(sum_1,5)
sum_2 = round(sum_2,4)

print("M accuracy:  "+str(sum_1*100)+" %")
print("M mIoU:  " +str(sum_2))