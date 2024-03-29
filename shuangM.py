import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def bilinear_interpolate(src, dst_size, align_corners=False):
    """
    双线性差值
    :param src: 原图像张量 NCHW
    :param dst_size: 目标图像spatial大小(H,W)
    :param align_corners: 换算坐标的不同方式
    :return: 目标图像张量NCHW
    """
    src_n, src_c, src_h, src_w = src.shape
    dst_n, dst_c, (dst_h, dst_w) = src_n, src_c, dst_size

    if src_h == dst_h and src_w == dst_w:
        return src.copy()
    """将dst的H和W坐标映射到src的H和W坐标"""
    hd = torch.arange(0, dst_h)
    wd = torch.arange(0, dst_w)
    if align_corners:
        h = float(src_h - 1) / (dst_h - 1) * hd
        w = float(src_w - 1) / (dst_w - 1) * wd
    else:
        h = float(src_h) / dst_h * (hd + 0.5) - 0.5
        w = float(src_w) / dst_w * (wd + 0.5) - 0.5

    h = torch.clamp(h, 0, src_h - 1)  # 防止越界，0相当于上边界padding
    w = torch.clamp(w, 0, src_w - 1)  # 防止越界，0相当于左边界padding

    h = h.view(dst_h, 1) # 1维dst_h个，变2维dst_h*1个
    w = w.view(1, dst_w)  # 1维dst_w个，变2维1*dst_w个
    h = h.repeat(1, dst_w).cuda()  # H方向重复1次，W方向重复dst_w次
    w = w.repeat(dst_h, 1).cuda()  # H方向重复dsth次，W方向重复1次

    """求出四点坐标"""
    h0 = torch.clamp(torch.floor(h), 0, src_h - 2) # -2相当于下边界padding
    w0 = torch.clamp(torch.floor(w), 0, src_w - 2) # -2相当于右边界padding
    h0 = h0.long().cuda()  # torch坐标必须是long
    w0 = w0.long().cuda()  # torch坐标必须是long

    h1 = (h0 + 1).cuda()
    w1 = (w0 + 1).cuda()

    """求出四点值"""
    q00 = src[..., h0, w0]
    q01 = src[..., h0, w1]
    q10 = src[..., h1, w0]
    q11 = src[..., h1, w1]

    """公式计算"""
    r0 = (w1 - w) * q00 + (w - w0) * q01 # 双线性插值的r0
    r1 = (w1 - w) * q10 + (w - w0) * q11 # 双线性差值的r1
    dst = (h1 - h) * r0 + (h - h0) * r1 # 双线性差值的q

    #del h, w, h0, w0, h1, w1,r0,r1
    #torch.cuda.empty_cache()

    return dst

