from os import path
from typing import Dict, List, Tuple, Union
from xml.etree import ElementTree as ET
import random
from utils.annotation_utils import AnnotationReader
from utils.box_utils import corner_to_center_numpy
from utils.augmentation_utils import Transformer

import cv2 as cv
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


class VOCDataset(Dataset):
    # VOC Dataset

    VOC2007_classes = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair','cow', 'diningtable', 'dog', 'horse','motorbike', 'person', 'pottedplant','sheep', 'sofa', 'train', 'tvmonitor'
    ]

    def __init__(self, root: Union[str, List[str]], imageSet: Union[str, List[str]],
                 transformer: Transformer = None, colorTransformer: Transformer = None, keepDifficult=False,
                 ifMosaic=False, ifMixup=False, imageSize=416):
        
        super().__init__()
        
        if isinstance(root, str):
            root = [root]
            
        if isinstance(imageSet, str):
            imageSet = [imageSet]
            
        if len(root) != len(imageSet):
            raise ValueError("number `root` = number `image_set`")

        self.root = root
        
        self.image_set = imageSet
        
        self.image_size = imageSize
        
        self.ifMosaic = ifMosaic
        
        self.ifMixup = ifMixup

        self.n_classes = len(self.VOC2007_classes)
        
        self.keepDifficult = keepDifficult
        
        self.class_to_index = {c: i for i, c in enumerate(self.VOC2007_classes)}

        self.transformer = transformer    # Data augmentation
        
        self.colorTransformer = colorTransformer
        
        self.annotationReader = AnnotationReader(self.class_to_index, keepDifficult)

        # Get all images and tag file paths in the specified dataset
        self.imageNames = []
        
        self.imagePaths = []
        
        self.annotationPaths = []

        for root, imageSet in zip(self.root, self.image_set):
            with open(path.join(root, f'ImageSets/Main/{imageSet}.txt')) as f:
                for line in f.readlines():
                    line = line.strip()
                    if not line:
                        continue
                    self.imageNames.append(line)
                    self.imagePaths.append(
                        path.join(root, f'JPEGImages/{line}.jpg'))
                    self.annotationPaths.append(
                        path.join(root, f'Annotations/{line}.xml'))

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, index: int):
        #generate samples

        # 50% mosaic data augmentation
        if self.ifMosaic and np.random.randint(2):
            image, bbox, label = self.make_mosaic(index)

            # mixup
            if self.ifMixup and np.random.randint(2):
                indexx = np.random.randint(0, len(self))
                
                imagee, bboxx, labell = self.make_mosaic(indexx)
                r = np.random.beta(8, 8)
                
                image = (image*r + imagee * (1 - r)).astype(np.uint8)
                bbox = np.vstack((bbox, bboxx))
                label = np.hstack((label, labell))

            # data augmentation
            if self.colorTransformer:
                image, bbox, label = self.colorTransformer.transform(
                    image, bbox, label)

        else:
            image, bbox, label = self.read_image_label(index)
            if self.transformer:
                image, bbox, label = self.transformer.transform(
                    image, bbox, label)

        image = image.astype(np.float32)
        image /= 255.0
        
        bbox = corner_to_center_numpy(bbox)
        
        target = np.hstack((bbox, label[:, np.newaxis]))

        return torch.from_numpy(image).permute(2, 0, 1), target

    def make_mosaic(self, index: int):
        indexes = list(range(len(self.imagePaths)))
        choices = random.sample(indexes[:index]+indexes[index+1:], 3)
        choices.append(index)
        images, bboxes, labels = [], [], []
        for i in choices:
            image, bbox, label = self.read_image_label(i)
            images.append(image)
            bboxes.append(bbox)
            labels.append(label)


        img_size = self.image_size
        mean = np.array([123, 117, 104])
        mosaic_img = np.ones((img_size*2, img_size*2, 3))*mean
        xc = int(random.uniform(img_size//2, 3*img_size//2))
        yc = int(random.uniform(img_size//2, 3*img_size//2))


        for i, (image, bbox, label) in enumerate(zip(images, bboxes, labels)):

            ih, iw, _ = image.shape
            s = np.random.choice(np.arange(50, 210, 10))/100
            if np.random.randint(2):
                r = img_size / max(ih, iw)
                if r != 1:
                    image = cv.resize(image, (int(iw*r*s), int(ih*r*s)))
            else:
                image = cv.resize(image, (int(img_size*s), int(img_size*s)))


            h, w, _ = image.shape
            if i == 0:

                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc

                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:
                x1a, y1a, x2a, y2a = xc, max(
                    yc - h, 0), min(xc + w, img_size * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:
                x1a, y1a, x2a, y2a = max(
                    xc - w, 0), yc, xc, min(img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:
                x1a, y1a, x2a, y2a = xc, yc, min(
                    xc + w, img_size * 2), min(img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            mosaic_img[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]


            dx = x1a - x1b
            dy = y1a - y1b
            bbox[:, [0, 2]] = bbox[:, [0, 2]]*w+dx
            bbox[:, [1, 3]] = bbox[:, [1, 3]]*h+dy


        bbox = np.clip(np.vstack(bboxes), 0, 2*img_size)
        label = np.hstack(labels)


        bbox_w = bbox[:, 2] - bbox[:, 0]
        bbox_h = bbox[:, 3] - bbox[:, 1]
        mask = np.logical_and(bbox_w > 1, bbox_h > 1)
        bbox, label = bbox[mask], label[mask]
        if len(bbox) == 0:
            bbox = np.zeros((1, 4))
            label = np.array([0])


        bbox /= mosaic_img.shape[0]

        return mosaic_img, bbox, label

    def read_image_label(self, index: int):

        image = cv.cvtColor(
            cv.imread(self.imagePaths[index]), cv.COLOR_BGR2RGB)
        target = np.array(self.annotationReader.read(
            self.annotationPaths[index]))
        bbox, label = target[:, :4], target[:, 4]
        return image, bbox, label


def collate_fn(batch: List[Tuple[torch.Tensor, np.ndarray]]):

    images = []
    targets = []

    for img, target in batch:
        images.append(img.to(torch.float32))
        targets.append(torch.Tensor(target))

    return torch.stack(images, 0), targets


def make_data_loader(dataset: VOCDataset, batch_size, num_workers=4, shuffle=True, drop_last=True, pin_memory=True):
    return DataLoader(
        dataset,
        batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
