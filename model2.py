import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
import numpy as np
from PIL import Image
import torch.utils.data as Data
import os
import time
import argparse
from shuangM import bilinear_interpolate

bn_momentum = 0.1  # BN层的momentum

torch.cuda.manual_seed(1)  # 设置随机种子

class Encoder_SSH(nn.Module):
    def __init__(self, input_channels):
        super(Encoder_SSH, self).__init__()

        self.enco1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU()
        )
        self.enco2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.ReLU()
        )
        self.enco3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU()
        )
        self.enco4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU()
        )
        self.enco5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU()
        )

    def forward(self, x):
        id = []

        rx1 = self.enco1(x)
        x1, id1 = F.max_pool2d(rx1, kernel_size=2, stride=2, return_indices=True)  # 保留最大值的位置
        id.append(id1)
        rx2 = self.enco2(x1)
        x2, id2 = F.max_pool2d(rx2, kernel_size=2, stride=2, return_indices=True)
        id.append(id2)
        rx3 = self.enco3(x2)
        x3, id3 = F.max_pool2d(rx3, kernel_size=2, stride=2, return_indices=True)
        id.append(id3)
        rx4 = self.enco4(x3)
        x4, id4 = F.max_pool2d(rx4, kernel_size=2, stride=2, return_indices=True)
        id.append(id4)
        rx5 = self.enco5(x4)
        x5, id5 = F.max_pool2d(rx5, kernel_size=2, stride=2, return_indices=True)
        id.append(id5)
        return x1,x3,x5,id



class aspp(nn.Module):
    def __init__(self,channel):
        super(aspp, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, 256, kernel_size=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel, 256, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channel, 256, kernel_size=3, dilation=4, padding=4),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(channel, 256, kernel_size=3, dilation=8, padding=8),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256 * 4, channel, kernel_size=1),
            nn.BatchNorm2d(channel, momentum=bn_momentum),
            nn.ReLU()
        )
    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x = self.conv5(torch.cat([x1, x2, x3, x4], dim=1))
        return x



class fusionchannel(nn.Module):
    def __init__(self,channel):
        super(fusionchannel,self).__init__()
        self.aspp = aspp(channel)

        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel,channel//8,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//8,channel,1,bias=False)
        )
        self.conv2 = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(channel,channel//8,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//8,channel,1,bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        fx = self.aspp(x)
        avg_x = self.conv1(fx)
        max_x = self.conv2(fx)
        out = avg_x + max_x
        return self.sigmoid(out) * x

class fusionspatial(nn.Module):
    def __init__(self,channel):
        super(fusionspatial, self).__init__()
        self.aspp = aspp(channel)
        self.conv = nn.Conv2d(2, 1, kernel_size = 7, padding=3, bias=False)  # 输入两个通道，一个是maxpool 一个是avgpool的
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        spatial_x = self.aspp(x)
        avg_out = torch.mean(spatial_x, dim = 1, keepdim = True)
        max_out,_ = torch.max(spatial_x, dim = 1, keepdim = True)
        xs = torch.cat([avg_out,max_out],dim = 1)
        xs = self.conv(xs)
        return self.sigmoid(xs)*x

class cascade_fusion(nn.Module):
    def __init__(self,channel):
        super(cascade_fusion, self).__init__()
        self.fuionc = fusionchannel(channel)
        self.fuions = fusionspatial(channel)
    def forward(self,x):
        x = self.fuionc(x)
        x = self.fuions(x)
        return x
class parallel_fusion(nn.Module):
    def __init__(self,channel):
        super(parallel_fusion, self).__init__()
        self.fusionc = fusionchannel(channel)
        self.fusions = fusionspatial(channel)
        self.conv = nn.Sequential(
            nn.Conv2d(2*channel,channel,kernel_size = 3, padding= 1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
    def forward(self,x):
        x = self.conv(torch.cat([self.fusionc(x),self.fusions(x)],dim = 1))
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        # assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        # padding = 3 if kernel_size == 7 else 1
        kernel_size = 7
        padding = 3
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 输入两个通道，一个是maxpool 一个是avgpool的
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)  # 对池化完的数据cat 然后进行卷积
        return self.sigmoid(x)
class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''
    def __init__(self, channels):
        super(AFF, self).__init__()

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, channels // 16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels // 16,momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels,momentum=bn_momentum),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels // 16,momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels,momentum=bn_momentum),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo



# 编码器+解码器
class Ma_Net(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Ma_Net, self).__init__()

        self.weights_new = self.state_dict()
        self.Encoder_SSH = Encoder_SSH(input_channels)
        self.aspp = aspp(512)


        self.cascade_fusion1 = parallel_fusion(512)
        self.cascade_fusion2 = parallel_fusion(256)
        self.cascade_fusion3 = parallel_fusion(64)



        self.deco1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU()
        )
        self.deco2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU()
        )
        self.deco3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.ReLU()
        )
        self.deco4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU()
        )
        self.deco5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(64, output_channels, kernel_size=3, stride=1, padding=1),
        )

        self.af1 = nn.Sequential(
            nn.Conv2d(512, output_channels, kernel_size=1, stride=1, padding=0),
        )
        self.af2 = nn.Sequential(
            nn.Conv2d(256, output_channels, kernel_size=1, stride=1, padding=0),
        )
        self.af3 = nn.Sequential(
            nn.Conv2d(128, output_channels, kernel_size=1, stride=1, padding=0),
        )
        self.af4 = nn.Sequential(
            nn.Conv2d(64, output_channels, kernel_size=1, stride=1, padding=0),
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(3, 3 // 3, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(3 // 3, 3, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv1 = nn.Conv2d(256 * 2,256,kernel_size=3,padding=1,stride=1)
        self.conv2 = nn.Conv2d(64 * 2, 64, kernel_size=3, padding=1, stride=1)



    def forward(self,x1):

        x11, x13, x15, id = self.Encoder_SSH(x1)

        xf5 = self.parallel_fusion3(x11)
        xf3 = self.parallel_fusion2(x13)
        xf1 = self.parallel_fusion1(x15)


        x = F.max_unpool2d(xf1, id[4], kernel_size=2, stride=2)
        x1 = self.deco1(x)
        x = F.max_unpool2d(x1, id[3], kernel_size=2, stride=2)
        x2 = self.deco2(x)

        x = self.conv1(torch.cat([x2, xf3], dim=1))

        x = F.max_unpool2d(x, id[2], kernel_size=2, stride=2)
        x3 = self.deco3(x)
        x = F.max_unpool2d(x3, id[1], kernel_size=2, stride=2)
        x4 = self.deco4(x)

        x4 = self.conv2(torch.cat([x4,xf5],dim=1))

        x = F.max_unpool2d(x4, id[0], kernel_size=2, stride=2)
        x5 = self.deco5(x)

        # fx1 = self.af1(x1)
        # fx2 = self.af2(x2)
        # fx3 = self.af3(x3)
        # fx4 = self.af4(x4)
        #
        # fx1 = bilinear_interpolate(fx1, (224, 224))
        #
        # fx2 = bilinear_interpolate(fx2, (224, 224))
        #
        # fx3 = bilinear_interpolate(fx3, (224, 224))
        #
        # fx4 = bilinear_interpolate(fx4, (224, 224))
        #
        # afx = fx1.add(fx2)
        # afx = afx.add(fx3)
        # afx = afx.add(fx4)
        # afx = afx.add(x5)

        return x5

    # 删掉VGG-16后面三个全连接层的权重
    def load_weights(self, weights_path):
        weights = torch.load(weights_path)
        del weights["classifier.0.weight"]
        del weights["classifier.0.bias"]
        del weights["classifier.3.weight"]
        del weights["classifier.3.bias"]
        del weights["classifier.6.weight"]
        del weights["classifier.6.bias"]

        names = []
        for key, value in self.encoder.state_dict().items():
            if "num_batches_tracked" in key:
                continue
            names.append(key)

        for name, dict in zip(names, weights.items()):
            self.weights_new[name] = dict[1]

        self.encoder.load_state_dict(self.weights_new)

