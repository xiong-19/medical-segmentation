import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import models


class Ma_Net(nn.Module):
    def __init__(self, classes):
        super(Ma_Net, self).__init__()
        self.deeplab = models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=classes, aux_loss=True)
        self.weights_new = self.state_dict()

    def forward(self, x):
        return self.deeplab(x)

    def load_weights(self, weights_path):
        weights = torch.load(weights_path)
        del weights["classifier.4.weight"]
        del weights["classifier.4.bias"]
        del weights["aux_classifier.4.weight"]
        del weights["aux_classifier.4.bias"]
        self.deeplab.load_state_dict(weights, strict=False)


if __name__ == '__main__':
    data = np.random.rand(4, 3, 224, 224)
    a = torch.tensor(data)
    a = a.to(torch.float32)
    m = Ma_Net(2)
    m.load_weights(r'C:\Users\20172\.cache\torch\hub\checkpoints\deeplabv3_resnet101_coco-586e9e4e.pth')
    b = m(a)
    print(b['out'].shape)