# coding:utf-8
import os
from pathlib import Path
from typing import List, Union
from torch import nn
from torch.nn import functional as F

from utils.augmentation_utils import ToTensor
from utils.box_utils import draw, rescale_bbox

import numpy as np
import torch
from PIL import Image

from .detector import Detector


class Mish(nn.Module):

    def forward(self, x):
        y = x * torch.tanh(F.softplus(x))

        return y


class CBMBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.setConvStruc(in_channels, kernel_size, out_channels, stride)
        self.setbn(out_channels)
        self.setMish()

    def setMish(self):
        self.mish = Mish()

    def setbn(self, out_channels):
        self.bn = nn.BatchNorm2d(out_channels)

    def setConvStruc(self, in_channels, kernel_size, out_channels, stride):
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2,
                              bias=False)

    def forward(self, x):
        return self.mish(self.bn(self.conv(x)))


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, hidden_channels=None):
        super().__init__()
        hidden_channels = self.setHChannel(hidden_channels, in_channels)
        self.setBlock(hidden_channels, in_channels)

    def setBlock(self, hidden_channels, in_channels):
        self.block = nn.Sequential(CBMBlock(in_channels, hidden_channels, 1),
                                   CBMBlock(hidden_channels, in_channels, 3), )

    def setHChannel(self, hidden_channels, in_channels):
        hidden_channels = hidden_channels or in_channels
        return hidden_channels

    def forward(self, x):
        return x + self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks):
        super().__init__()
        self.setDConv(in_channels, out_channels)

        self.setConvLayer(n_blocks, out_channels)

    def setConvLayer(self, n_blocks, out_channels):
        if n_blocks == 1:
            self.convblock1(out_channels)
        else:
            self.convblock2(n_blocks, out_channels)

    def convblock2(self, n_blocks, out_channels):
        self.setConv02(out_channels)
        self.setConv12(out_channels)
        self.setBlockConv2(n_blocks, out_channels)
        self.setConcatConv2(out_channels)

    def setConcatConv2(self, out_channels):
        self.concat_conv = CBMBlock(out_channels, out_channels, 1)

    def setBlockConv2(self, n_blocks, out_channels):
        self.blocks_conv = nn.Sequential(
            *[ResidualUnit(out_channels // 2) for _ in range(n_blocks)],
            CBMBlock(out_channels // 2, out_channels // 2, 1)
        )

    def setConv12(self, out_channels):
        self.split_conv1 = CBMBlock(out_channels, out_channels // 2, 1)

    def setConv02(self, out_channels):
        self.split_conv0 = CBMBlock(out_channels, out_channels // 2, 1)

    def convblock1(self, out_channels):
        self.setConv01(out_channels)
        self.setConv11(out_channels)
        self.setblockConv1(out_channels)
        self.setConcatConv1(out_channels)

    def setConcatConv1(self, out_channels):
        self.concat_conv = CBMBlock(out_channels * 2, out_channels, 1)

    def setblockConv1(self, out_channels):
        self.blocks_conv = nn.Sequential(
            ResidualUnit(out_channels, out_channels // 2),
            CBMBlock(out_channels, out_channels, 1)
        )

    def setConv11(self, out_channels):
        self.split_conv1 = CBMBlock(out_channels, out_channels, 1)

    def setConv01(self, out_channels):
        self.split_conv0 = CBMBlock(out_channels, out_channels, 1)

    def setDConv(self, in_channels, out_channels):
        self.downsample_conv = CBMBlock(
            in_channels, out_channels, 3, stride=2)

    def forward(self, x):
        x = self.downsample_conv(x)
        x0, x1 = self.setSplit(x)
        x1 = self.blocks_conv(x1)
        x = torch.cat([x1, x0], dim=1)
        return self.concat_conv(x)

    def setSplit(self, x):
        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)
        return x0, x1


class CSPDarkNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layers = self.setLayers()
        channels = self.setChannels()
        self.setConv1()
        self.setStages(channels, layers)

    def setStages(self, channels, layers):
        # Mlist = []
        # for i in range(5):
        #     Mlist.append((ResidualBlock(channels[i], channels[i + 1], layers[i])))
        self.stages = nn.ModuleList([ResidualBlock(channels[i], channels[i+1], layers[i]) for i in range(5)])

    def setConv1(self):
        self.conv1 = CBMBlock(3, 32, 3)

    def setChannels(self):
        channels = [32, 64, 128, 256, 512, 1024]
        return channels

    def setLayers(self):
        layers = [1, 2, 8, 8, 4]
        return layers

    def forward(self, x):
        x = self.setX(x)
        x3 = self.stages[2](x)
        x4, x5 = self.getXs(x3)

        return x3, x4, x5

    def getXs(self, x3):
        x4 = self.stages[3](x3)
        x5 = self.stages[4](x4)
        return x4, x5

    def load(self, model_path: Union[Path, str]):
        self.load_state_dict(torch.load(model_path))

    def setX(self, x):
        x = self.conv1(x)
        x = self.stages[0](x)
        x = self.stages[1](x)
        return x

    def set_freezed(self, freeze: bool):
        for param in self.parameters():
            param.requires_grad = not freeze


class CBLBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.setConv(in_channels, kernel_size, out_channels, stride)
        self.setBN(out_channels)
        self.setRelu()

    def setRelu(self):
        self.relu = nn.LeakyReLU(0.1)

    def setBN(self, out_channels):
        self.bn = nn.BatchNorm2d(out_channels)

    def setConv(self, in_channels, kernel_size, out_channels, stride):
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                              bias=False)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SPPBlock(nn.Module):
    def __init__(self, sizes=(5, 9, 13)):
        super().__init__()
        self.setMaxpools(sizes)

    def setMaxpools(self, sizes):
        # Mlist = {}
        # for size in sizes:
        #     Mlist = nn.MaxPool2d(size, 1, size // 2)
        self.maxpools = nn.ModuleList([nn.MaxPool2d(size, 1, size//2) for size in sizes])

    def forward(self, x):
        if self.getX1(x):
            x1 = self.getX1(x)
        x1.append(x)
        return torch.cat(x1, dim=1)

    def getX1(self, x):
        x1 = [pool(x) for pool in self.maxpools[::-1]]
        return x1


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.setUpsample(in_channels, out_channels)

    def setUpsample(self, in_channels, out_channels):
        self.upsample = nn.Sequential(CBLBlock(in_channels, out_channels, 1), nn.Upsample(scale_factor=2, mode="nearest"))

    def forward(self, x):
        if True:
            x_back = self.upsample(x)
        return x_back


def make_three_cbl(channels: list):
    cblss = nn.Sequential(CBLBlock(channels[0], channels[1], 1), CBLBlock(channels[1], channels[2], 3),
                          CBLBlock(channels[2], channels[1], 1), )
    return cblss


def make_five_cbl(channels: list):
    cblsss = nn.Sequential(CBLBlock(channels[0], channels[1], 1), CBLBlock(channels[1], channels[2], 3),
                           CBLBlock(channels[2], channels[1], 1), CBLBlock(channels[1], channels[2], 3),
                           CBLBlock(channels[2], channels[1], 1), )
    return cblsss


def make_yolo_head(channels: list):
    yolo_head = nn.Sequential(CBLBlock(channels[0], channels[1], 3), nn.Conv2d(channels[1], channels[2], 1), )
    return yolo_head


class Yolo(nn.Module):
    def __init__(self, n_classes, anchors: list = None, conf_thresh=0.1, nms_thresh=0.45):
        super().__init__()
        self.img_size_judge(416)

        anchors = self.setAnchors(anchors)

        anchors = self.calAnchors(anchors, 416)

        self.setSelfAnchors(anchors)

        self.setClassNum(n_classes)
        self.setImageSize(416)

        self.setBackbone()

        self.setConv1()
        self.setSPP()
        self.setConv2()

        self.setUpsample1()
        self.setP4Conv()
        self.setFiveConv1()

        self.setUpsample2()
        self.setP3Conv()
        self.setFiveConv2()

        channel = self.calChannel(n_classes)
        self.setYoloHead3(channel)

        self.setDownSample1()
        self.setFiveConv3()

        self.setYoloHead2(channel)

        self.setDownSample2()
        self.setFiveConv4()

        self.setYoloHead1(channel)

        self.setDetector(conf_thresh, 416, n_classes, nms_thresh)

    def setDetector(self, conf_thresh, image_size, n_classes, nms_thresh):
        self.detector = Detector(self.anchors, image_size, n_classes, conf_thresh, nms_thresh)

    def setYoloHead1(self, channel):
        self.yolo_head1 = make_yolo_head([512, 1024, channel])

    def setFiveConv4(self):
        self.make_five_conv4 = make_five_cbl([1024, 512, 1024])

    def setDownSample2(self):
        self.down_sample2 = CBLBlock(256, 512, 3, stride=2)

    def setYoloHead2(self, channel):
        self.yolo_head2 = make_yolo_head([256, 512, channel])

    def setFiveConv3(self):
        self.make_five_conv3 = make_five_cbl([512, 256, 512])

    def setDownSample1(self):
        self.down_sample1 = CBLBlock(128, 256, 3, stride=2)

    def setYoloHead3(self, channel):
        self.yolo_head3 = make_yolo_head([128, 256, channel])

    def calChannel(self, n_classes):
        channel = len(self.anchors[1]) * (5 + n_classes)
        return channel

    def setFiveConv2(self):
        self.make_five_conv2 = make_five_cbl([256, 128, 256])

    def setP3Conv(self):
        self.conv_for_P3 = CBLBlock(256, 128, 1)

    def setUpsample2(self):
        self.upsample2 = Upsample(256, 128)

    def setFiveConv1(self):
        self.make_five_conv1 = make_five_cbl([512, 256, 512])

    def setP4Conv(self):
        self.conv_for_P4 = CBLBlock(512, 256, 1)

    def setUpsample1(self):
        self.upsample1 = Upsample(512, 256)

    def setConv2(self):
        self.conv2 = make_three_cbl([2048, 512, 1024])

    def setSPP(self):
        self.SPP = SPPBlock()

    def setConv1(self):
        self.conv1 = make_three_cbl([1024, 512, 1024])

    def setBackbone(self):
        self.backbone = CSPDarkNet()

    def setImageSize(self, image_size):
        self.image_size = image_size

    def setClassNum(self, n_classes):
        self.n_classes = n_classes

    def setSelfAnchors(self, anchors):
        self.anchors = anchors.tolist()

    def calAnchors(self, anchors, image_size):
        anchors = np.array(anchors, dtype=np.float32)
        anchors = anchors * image_size / 416
        return anchors

    def setAnchors(self, anchors):
        anchors = anchors or [[[142, 110], [192, 243], [459, 401]], [[36, 75], [76, 55], [72, 146]],
                              [[12, 16], [19, 36], [40, 28]], ]
        return anchors

    def img_size_judge(self, image_size):
        if image_size <= 0 or image_size % 32 != 0:
            raise ValueError("image_size must be 32x")

    def forward(self, x: torch.Tensor):
        P4, P5, x2 = self.prepareParams(x)

        P3 = self.combineParams(P4, x2)

        P4 = self.concatP4(P3, P4)

        P5 = self.concatP5(P4, P5)

        y0, y1, y2 = self.getHeads(P3, P4, P5)

        return y0, y1, y2

    def getHeads(self, P3, P4, P5):
        y2 = self.yolo_head3(P3)
        y1 = self.yolo_head2(P4)
        y0 = self.yolo_head1(P5)
        return y0, y1, y2

    def concatP5(self, P4, P5):
        P4_downsample = self.down_sample2(P4)
        P5 = torch.cat([P4_downsample, P5], dim=1)
        P5 = self.make_five_conv4(P5)
        return P5

    def concatP4(self, P3, P4):
        P3_downsample = self.down_sample1(P3)
        P4 = torch.cat([P3_downsample, P4], dim=1)
        P4 = self.make_five_conv3(P4)
        return P4

    def combineParams(self, P4, x2):
        P4_upsample = self.upsample2(P4)
        P3 = self.conv_for_P3(x2)
        P3 = torch.cat([P3, P4_upsample], dim=1)
        P3 = self.make_five_conv2(P3)
        return P3

    def prepareParams(self, x):
        x2, x1, x0 = self.backbone(x)
        P4, P5, P5_upsample = self.setP(x0, x1)
        P4 = torch.cat([P4, P5_upsample], dim=1)
        P4 = self.make_five_conv1(P4)
        return P4, P5, x2

    def setP(self, x0, x1):
        P4 = self.conv_for_P4(x1)
        P5 = self.conv2(self.SPP(self.conv1(x0)))
        P5_upsample = self.upsample1(P5)
        return P4, P5, P5_upsample

    def detect(self, image: Union[str, np.ndarray], classes: List[str], show_conf=True) -> Image.Image:
        image = self.pathDetect(image)

        h, w, channels = image.shape
        self.channelDetect(channels)

        x = self.setX(image, True)

        y = self.predict(x)
        if not y:
            return Image.fromarray(image)

        bbox, conf, label = self.boxPred(classes, h, w, y)

        if not show_conf:
            conf = None

        image = draw(image, np.vstack(bbox), label, conf)
        return image

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        if self.detector:
            x_detect = self.detector(self(x))
        return x_detect

    def boxPred(self, classes, h, w, y):
        bbox, conf, label = [],[],[]
        for c, pred in y[0].items():
            boxes = rescale_bbox(pred[:, 1:], 416, h, w)
            bbox.append(boxes)
            conf.extend(pred[:, 0].tolist())
            label.extend([classes[c]] * pred.shape[0])
        return bbox, conf, label

    def calcPred(self, bbox, h, pred, w):
        pred = pred.numpy()
        boxes = self.calBoxes(h, pred.numpy(), w)
        bbox.append(boxes)
        return pred

    def calBoxes(self, h, pred, w):
        boxes = rescale_bbox(pred[:, 1:], 416, h, w)
        return boxes

    def setX(self, image, use_gpu):
        x = ToTensor(416).transform(image)
        if use_gpu:
            x = x.cuda()
        return x

    def channelDetect(self, channels):
        if channels != 3:
            raise ValueError('The input must be an RGB image with three channels!')

    def pathDetect(self, image):
        if isinstance(image, str):
            convert = os.path.exists(image)
            if convert:
                image = np.array(Image.open(image).convert('RGB'))
            else:
                raise FileNotFoundError("Image Path Error")
        return image

    def load(self, model_path: Union[Path, str]):
        self.load_state_dict(torch.load(model_path))
