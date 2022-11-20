import torch
from torch.utils.data import Dataset, DataLoader
from os import path
import cv2 as cv
import numpy as np
from utils.annotation_utils import AnnotationReader
from utils.box_utils import corner_to_center_numpy
from typing import Dict, List, Tuple, Union
from xml.etree import ElementTree as ET
from utils.augmentation_utils import Transformer
import random


class VOCDataset(Dataset):

    VOC2007_classes = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair','cow', 'diningtable', 'dog', 'horse','motorbike', 'person', 'pottedplant','sheep', 'sofa', 'train', 'tvmonitor'
    ]

    def __init__(self, root: Union[str, List[str]], imageSet: Union[str, List[str]],
                 transformer: Transformer = None, colorTransformer: Transformer = None, keepDifficult=False,
                 ifMosaic=False, ifMixup=False, imageSize=416):
        
        super().__init__()

        root = self.ifin(root)

        imageSet = self.ifin(imageSet)

        self.errorJudge(imageSet, root)

        self.setRoot(root)

        self.setImageSet(imageSet)

        self.setImageSize(imageSize)

        self.setIfMosaic(ifMosaic)

        self.setIfMixup(ifMixup)

        self.setClassNum()

        self.setKeepDifficult(keepDifficult)

        self.setClassIndex()

        self.setTransformer(transformer)

        self.setColorTransformer(colorTransformer)

        self.setAnnotationReader(keepDifficult)

        # Get all images and tag file paths in the specified dataset
        self.getFilePaths()

    def getFilePaths(self):
        self.imageNames, self.imagePaths, self.annotationPaths = [], [], []
        for root, imageSet in zip(self.root, self.imageSet):
            filepath = path.join(root, f'ImageSets/Main/{imageSet}.txt')
            with open(filepath) as f:
                for line in f.readlines():
                    line = line.strip()
                    if not line:
                        continue
                    self.selfAppend(line, root)


    def setColorTransformer(self, colorTransformer):
        self.colorTransformer = colorTransformer

    def setAnnotationReader(self, keepDifficult):
        self.annotationReader = AnnotationReader(self.class_to_index, keepDifficult)

    def setTransformer(self, transformer):
        self.transformer = transformer

    def selfAppend(self, line, root):
        self.imageNames.append(line)
        self.imagePaths.append(path.join(root, f'JPEGImages/{line}.jpg'))
        self.annotationPaths.append(path.join(root, f'Annotations/{line}.xml'))

    def setClassIndex(self):
        self.class_to_index = {c: i for i, c in enumerate(self.VOC2007_classes)}

    def setKeepDifficult(self, keepDifficult):
        self.keepDifficult = keepDifficult

    def setClassNum(self):
        self.n_classes = len(self.VOC2007_classes)

    def setIfMixup(self, ifMixup):
        self.ifMixup = ifMixup

    def setIfMosaic(self, ifMosaic):
        self.ifMosaic = ifMosaic

    def setImageSize(self, imageSize):
        self.imageSize = imageSize

    def setImageSet(self, imageSet):
        self.imageSet = imageSet

    def setRoot(self, root):
        self.root = root

    def errorJudge(self, imageSet, root):
        if len(root) != len(imageSet):
            raise ValueError("number `root` = number `imageSet`")

    def ifin(self, root):
        if isinstance(root, str):
            root = [root]
        return root

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, index: int):
        #generate samples

        # 50% mosaic data augmentation
        if self.ifMosaic and np.random.randint(2):
            image, bbox, label = self.mosaicMaker(index)

            # mixup
            bbox, image, label = self.mixup(bbox, image, label)

            # data augmentation
            bbox, image, label = self.dataAug(bbox, image, label)

        else:
            image, bbox, label = self.read_image_label(index)
            bbox, image, label = self.trans(bbox, image, label)

        image = self.imageCal(image)

        bbox = self.bboxCal(bbox)

        target = self.targetGen(bbox, label)

        return torch.from_numpy(image).permute(2, 0, 1), target

    def targetGen(self, bbox, label):
        target = np.hstack((bbox, label[:, np.newaxis]))
        return target

    def bboxCal(self, bbox):
        bbox = corner_to_center_numpy(bbox)
        return bbox

    def imageCal(self, image):
        image = image.astype(np.float32)
        image /= 255.0
        return image

    def trans(self, bbox, image, label):
        if self.transformer:
            image, bbox, label = self.transformer.transform(image, bbox, label)
        return bbox, image, label

    def dataAug(self, bbox, image, label):
        if self.colorTransformer:
            image, bbox, label = self.colorTransformer.transform(image, bbox, label)
        return bbox, image, label

    def mixup(self, bbox, image, label):
        if self.ifMixup and np.random.randint(2):
            indexx = np.random.randint(0, len(self))
            bboxx, imagee, labell = self.setMosaic(indexx)
            image = self.calculateImage(image, imagee, np.random.beta(8, 8))
            bbox = self.calculateBbox(bbox, bboxx)
            label = self.calculateLabel(label, labell)
        return bbox, image, label

    def setMosaic(self, indexx):
        imagee, bboxx, labell = self.mosaicMaker(indexx)
        return bboxx, imagee, labell

    def calculateLabel(self, label, labell):
        label = np.hstack((label, labell))
        return label

    def calculateBbox(self, bbox, bboxx):
        bbox = np.vstack((bbox, bboxx))
        return bbox

    def calculateImage(self, image, imagee, r):
        image = (image * r + imagee * (1 - r)).astype(np.uint8)
        return image

    def mosaicMaker(self, index: int):
        indexes = self.geneIndex()

        choices = self.getChoices(index, indexes)

        bboxes, images, labels = self.splitChoices(choices)

        imageSize = self.imageSize
        
        mean = np.array([123, 117, 104])

        mosaicImage = self.mosaicCal(imageSize, mean)

        xc = self.initC(imageSize)

        yc = self.initC(imageSize)

        self.loop1(bboxes, imageSize, images, labels, mosaicImage, xc, yc)

        bbox, label = self.setLB(bboxes, imageSize, labels)

        bboxHeight, bboxWidth = self.calBboxHW(bbox)

        mask = np.logical_and(bboxWidth > 1, bboxHeight > 1)
        
        bbox, label = bbox[mask], label[mask]

        bbox, label = self.setBL(bbox, label)


        bbox /= mosaicImage.shape[0]

        return mosaicImage, bbox, label

    def setLB(self, bboxes, imageSize, labels):
        label = np.hstack(labels)
        bbox = np.clip(np.vstack(bboxes), 0, 2 * imageSize)
        return bbox, label

    def setBL(self, bbox, label):
        if len(bbox) == 0:
            bbox = np.zeros((1, 4))

            label = np.array([0])
        return bbox, label

    def calBboxHW(self, bbox):
        bboxWidth = bbox[:, 2] - bbox[:, 0]
        bboxHeight = bbox[:, 3] - bbox[:, 1]
        return bboxHeight, bboxWidth

    def loop1(self, bboxes, imageSize, images, labels, mosaicImage, xc, yc):
        for i, (image, bbox, label) in enumerate(zip(images, bboxes, labels)):

            hi, s, wi = self.setHWS(image)
            image = self.resizeImg(hi, image, imageSize, s, wi)
            h, w, _ = image.shape

            x1a, x1b, x2a, x2b, y1a, y1b, y2a, y2b = 0, 0, 0, 0, 0, 0, 0, 0
            if i == 0:

                x1a, x1b, x2a, x2b, y1a, y1b, y2a, y2b = self.calI0(h, w, xc, yc)

            elif i == 1:

                x1a, x1b, x2a, x2b, y1a, y1b, y2a, y2b = self.calI1(h, imageSize, w, x1a, x1b, x2a, x2b, xc, y1a, y1b,
                                                                    y2a, y2b, yc)

            elif i == 2:

                x1a, x1b, x2a, x2b, y1a, y1b, y2a, y2b = self.calI2(h, imageSize, w, x1a, x1b, x2a, x2b, xc, y1a, y1b,
                                                                    y2a, y2b, yc)

            elif i == 3:

                x1a, x1b, x2a, x2b, y1a, y1b, y2a, y2b = self.calI3(h, imageSize, w, x1a, x1b, x2a, x2b, xc, y1a, y1b,
                                                                    y2a, y2b, yc)

            mosaicImage[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]

            dx, dy = self.iniD(x1a, x1b, y1a, y1b)

            bbox[:, [0, 2]] = dx + bbox[:, [0, 2]] * w
            bbox[:, [1, 3]] = dy + bbox[:, [1, 3]] * h

    def setHWS(self, image):
        hi, wi, _ = image.shape
        s = np.random.choice(np.arange(50, 210, 10)) / 100
        return hi, s, wi

    def iniD(self, x1a, x1b, y1a, y1b):
        dx = x1a - x1b
        dy = y1a - y1b
        return dx, dy

    def resizeImg(self, hi, image, imageSize, s, wi):
        if np.random.randint(2):
            r = imageSize / max(hi, wi)
            image = self.notOne(hi, image, r, s, wi)
        else:
            image = cv.resize(image, (int(imageSize * s), int(imageSize * s)))
        return image

    def notOne(self, hi, image, r, s, wi):
        if r != 1:
            image = cv.resize(image, (int(wi * r * s), int(hi * r * s)))
        return image

    def calI3(self, h, imageSize, w, x1a, x1b, x2a, x2b, xc, y1a, y1b, y2a, y2b, yc):
        x1a, x2a, y1a, y2a = self.calI31(h, imageSize, w, x1a, x2a, xc, y1a, y2a, yc)
        x1b, x2b, y1b, y2b = self.calI32(h, w, x1a, x1b, x2a, x2b, y1a, y1b, y2a, y2b)
        return x1a, x1b, x2a, x2b, y1a, y1b, y2a, y2b

    def calI32(self, h, w, x1a, x1b, x2a, x2b, y1a, y1b, y2a, y2b):
        x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
        return x1b, x2b, y1b, y2b

    def calI31(self, h, imageSize, w, x1a, x2a, xc, y1a, y2a, yc):
        x1a, y1a, x2a, y2a = xc, yc, min(xc + w, imageSize * 2), min(imageSize * 2, yc + h)
        return x1a, x2a, y1a, y2a

    def calI2(self, h, imageSize, w, x1a, x1b, x2a, x2b, xc, y1a, y1b, y2a, y2b, yc):
        x1a, x2a, y1a, y2a = self.calI21(h, imageSize, w, x1a, x2a, xc, y1a, y2a, yc)
        x1b, x2b, y1b, y2b = self.calI22(h, w, x1a, x1b, x2a, x2b, y1a, y1b, y2a, y2b)
        return x1a, x1b, x2a, x2b, y1a, y1b, y2a, y2b

    def calI22(self, h, w, x1a, x1b, x2a, x2b, y1a, y1b, y2a, y2b):
        x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        return x1b, x2b, y1b, y2b

    def calI21(self, h, imageSize, w, x1a, x2a, xc, y1a, y2a, yc):
        x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(imageSize * 2, yc + h)
        return x1a, x2a, y1a, y2a

    def calI1(self, h, imageSize, w, x1a, x1b, x2a, x2b, xc, y1a, y1b, y2a, y2b, yc):
        x1a, x2a, y1a, y2a = self.calI11(h, imageSize, w, x1a, x2a, xc, y1a, y2a, yc)
        x1b, x2b, y1b, y2b = self.calI12(h, w, x1a, x1b, x2a, x2b, y1a, y1b, y2a, y2b)
        return x1a, x1b, x2a, x2b, y1a, y1b, y2a, y2b

    def calI12(self, h, w, x1a, x1b, x2a, x2b, y1a, y1b, y2a, y2b):
        x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        return x1b, x2b, y1b, y2b

    def calI11(self, h, imageSize, w, x1a, x2a, xc, y1a, y2a, yc):
        x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, imageSize * 2), yc
        return x1a, x2a, y1a, y2a

    def calI0(self, h, w, xc, yc):
        x1a, x2a, y1a, y2a = self.calI01(h, w, xc, yc)
        x1b, x2b, y1b, y2b = self.calI02(h, w, x1a, x2a, y1a, y2a)
        return x1a, x1b, x2a, x2b, y1a, y1b, y2a, y2b

    def calI02(self, h, w, x1a, x2a, y1a, y2a):
        x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        return x1b, x2b, y1b, y2b

    def calI01(self, h, w, xc, yc):
        x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
        return x1a, x2a, y1a, y2a

    def initC(self, imageSize):
        xc = int(random.uniform(imageSize // 2, 3 * imageSize // 2))
        return xc

    def mosaicCal(self, imageSize, mean):
        mosaicImage = np.ones((imageSize * 2, imageSize * 2, 3)) * mean
        return mosaicImage

    def splitChoices(self, choices):
        images, bboxes, labels = [], [], []
        for i in choices:
            image, bbox, label = self.read_image_label(i)
            images.append(image)
            bboxes.append(bbox)
            labels.append(label)
        return bboxes, images, labels

    def geneIndex(self):
        indexes = list(range(len(self.imagePaths)))
        return indexes

    def getChoices(self, index, indexes):
        choices = random.sample(indexes[:index] + indexes[index + 1:], 3)
        choices.append(index)
        return choices

    def setTarget(self, index):
        target = np.array(self.annotationReader.read(self.annotationPaths[index]))
        return target

    def read_image_label(self, index: int):
        image = cv.cvtColor(cv.imread(self.imagePaths[index]), cv.COLOR_BGR2RGB)
        target = self.setTarget(index)
        bbox, label = target[:, :4], target[:, 4]
        return image, bbox, label

def dataLoader(dataset: VOCDataset, batch_size, num_workers=4, shuffle=True, drop_last=True, pin_memory=True):
    return DataLoader(
        dataset,
        batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=pin_memory,
        collate_fn=collateFn
    )

def collateFn(batch: List[Tuple[torch.Tensor, np.ndarray]]):

    images = []
    targets = []

    for img, target in batch:
        images.append(img.to(torch.float32))
        targets.append(torch.Tensor(target))

    return torch.stack(images, 0), targets

