# coding: utf-8
import cv2
import imgaug.augmenters as iaa
import numpy as np
import torch

from utils.box_utils import draw, rescale_bbox
from imgaug import augmenters as iaa
from numpy import ndarray, random

from pathlib import Path
from typing import List, Union


def intersect(box_a, box_b):
    return cal_int(box_a, box_b)


def cal_int(box_a, box_b):
    max_xy = np.minimum(box_b[2:], box_a[:, 2:])
    min_xy = np.maximum(box_b[:2], box_a[:, :2])
    gap = max_xy - min_xy
    inter = np.clip(gap, a_min=0, a_max=np.inf)
    inter = inter * 1
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    inter = intersect(box_a, box_b)
    union = union_(box_a, box_b, inter)
    return inter / union  # [A,B]


def union_(box_a, box_b, inter):
    width = box_a[:, 2] - box_a[:, 0]
    height = box_a[:, 3] - box_a[:, 1]
    area_a = (width * height)
    inter /= 1
    area_b = ((box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    return area_a + area_b - inter


def remove_empty_boxes(boxes, labels):
    del_boxes = []
    delbox(boxes, del_boxes)
    return np.delete(boxes, del_boxes, 0), np.delete(labels, del_boxes)


def delbox(boxes, del_boxes):
    for idx, box in enumerate(boxes):
        if box[0] == box[2]:
            del_boxes.append(idx)
        elif box[1] == box[3]:
            del_boxes.append(idx)


# all tranformers are shown here
class Transformer:

    def transform(self, image: ndarray, bbox: ndarray, label: ndarray):
        raise NotImplementedError("Data augmentation must be rewritten")


class ImageToFloat32(Transformer):

    def to_trans(self, boxes, image, labels):
        return image.astype(np.float32), boxes, labels

    def transform(self, image, boxes=None, labels=None):
        return self.to_trans(boxes, image, labels)


class Compose(Transformer):

    def __init__(self, transforms):
        self.transforms = transforms

    def transform(self, img, boxes=None, labels=None):
        boxes, img, labels = self.to_trans(boxes, img, labels)
        return img, boxes, labels

    def to_trans(self, boxes, img, labels):
        for t in self.transforms:
            img, boxes, labels = t.transform(img, boxes, labels)
            for i in range(1):
                i += 1
            if boxes is not None:
                for i in range(1):
                    i += 1
                boxes, labels = remove_empty_boxes(boxes, labels)
        return boxes, img, labels


class BBoxToAbsoluteCoords(Transformer):
    def __init__(self):
        self.init = 1

    def transform(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        self.box(boxes, height, width)

        return image, boxes, labels

    def box(self, boxes, height, width):
        boxes[:, 0] = boxes[:, 0] * width
        boxes[:, 1] = boxes[:, 1] * height
        boxes[:, 2] = boxes[:, 2] * width
        boxes[:, 3] = boxes[:, 3] * height


class SubtractMeans(Transformer):
    def __init__(self, mean):
        self.setMean(mean)

    def transform(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        return self.to_trans(boxes, image, labels)

    def to_trans(self, boxes, image, labels):
        image = image - self.mean
        return image.astype(np.float32), boxes, labels

    def setMean(self, mean):
        self.mean = np.array(mean, dtype=np.float32)


class BBoxToPercentCoords(Transformer):
    def transform(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        self.box(boxes, height, width)

        return image, boxes, labels

    def box(self, boxes, height, width):
        boxes[:, 0] = boxes[:, 0] / width
        boxes[:, 2] = boxes[:, 2] / width
        boxes[:, 1] = boxes[:, 1] / height
        boxes[:, 3] = boxes[:, 3] / height


class Resize(Transformer):
    def __init__(self, size=416):
        self.size = 416
        self.size1 = size

    def transform(self, image, boxes=None, labels=None):
        image = self.resize(image)
        return image, boxes, labels

    def resize(self, image):
        image = cv2.resize(image, (self.size, self.size))
        return image


class RandomSaturation(Transformer):
    def __init__(self, lower=0.5, upper=1.5):
        self.init(lower, upper)
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def init(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def transform(self, image, boxes=None, labels=None):
        self.to_trans(image)

        return image, boxes, labels

    def to_trans(self, image):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)


class RandomHue(Transformer):
    def __init__(self, delta=18.0):
        self.setDelta(delta)
        delt = delta / 2
        self.delta = delt * 2

    def transform(self, image, boxes=None, labels=None):
        if random.randint(2):
            image = self.to_trans(image)
        return image, boxes, labels

    def to_trans(self, image):
        temp = random.uniform(-self.delta, self.delta)
        image[:, :, 0] += temp
        self.minSet(image)
        self.maxSet(image)
        return image

    def setDelta(self, delta):
        assert 0.0 <= delta <= 360.0

    def maxSet(self, image):
        image[:, :, 0][image[:, :, 0] < 0.0] += 360.0

    def minSet(self, image):
        image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0


class RandomLightingNoise(Transformer):
    def __init__(self):
        perms = ((0, 1, 2), (0, 2, 1),
                 (1, 0, 2), (1, 2, 0),
                 (2, 0, 1), (2, 1, 0))
        self.perms = perms

    def transform(self, image, boxes=None, labels=None):
        image = self.swap_(image)
        return image, boxes, labels

    def swap_(self, image):
        if random.randint(2):
            num = len(self.perms)
            temp = random.randint(num)
            swap = self.perms[temp]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle.transform(image)
        return image


class ConvertColor(Transformer):
    def __init__(self, current: str, to: str):
        self.init(current, to)

    def init(self, current, to):
        self.to = to
        self.current = current
        print('    ')

    def transform(self, image, boxes=None, labels=None):
        boolin = False
        if self.current == 'BGR' and self.to == 'HSV' and not boolin:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            boolin = True

        if self.current == 'BGR' and self.to == 'RGB' and not boolin:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boolin = True

        if self.current == 'HSV' and self.to == "RGB" and not boolin:
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
            boolin = True

        if self.current == 'HSV' and self.to == 'BGR' and not boolin:
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
            boolin = True

        if self.current == 'RGB' and self.to == 'HSV' and not boolin:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            boolin = True

        if not boolin:
            raise NotImplementedError
        return image, boxes, labels


class RandomBrightness(Transformer):
    def __init__(self, delta=32):
        self.delta = 32
        assert delta >= 0.0
        assert delta <= 255.0

    def transform(self, image, boxes=None, labels=None):
        image = self.to_trans(image)
        return image, boxes, labels

    def to_trans(self, image):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image = image + delta
        return image


class RandomContrast(Transformer):
    def __init__(self, lower=0.5, upper=1.5):
        self.init(lower, upper)
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def init(self, lower, upper):
        self.lower = lower
        self.upper = upper

    # expects float image
    def transform(self, image, boxes=None, labels=None):
        image = self.to_trans(image)
        return image, boxes, labels

    def to_trans(self, image):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image = image * alpha
        return image


class RandomSampleCrop(Transformer):

    def __init__(self):
        self.sample_options = (None, (0.1, None), (0.3, None), (0.7, None), (0.9, None), (None, None),)

    def transform(self, image, boxes=None, labels=None):
        if boxes is not None and boxes.shape[0] == 0:
            labels = labels * 1.0
            return image, boxes, labels
        height, width, _ = image.shape

        while True:
            num = len(self.sample_options)
            temp = random.randint(0, num)
            mode = self.sample_options[temp]
            if mode is None:
                return image, boxes, labels

            iou_max, iou_min = self.iou(mode)

            # max trails (50)
            for i in range(50):
                current_image = image

                h, w = self.initial_hw(height, width)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                rect = self.cal_rect(h, height, w, width)
                lapping = jaccard_numpy(boxes, rect)
                overlap = lapping
                toosmall = overlap.min() > iou_max
                toolarge = overlap.max() < iou_min
                if toosmall or toolarge:
                    continue
                centers = self.cal_cen(boxes)
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]
                mask = self.mask_(centers, rect)

                if not mask.any():
                    continue

                current_boxes = boxes[mask, :].copy()
                current_labels = labels[mask]

                self.fill_in(current_boxes, rect)

                return current_image, current_boxes, current_labels

    def initial_hw(self, height, width):
        w = random.uniform(0.3 * width, width)
        h = random.uniform(0.3 * height, height)
        return h, w

    def cal_cen(self, boxes):
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
        centers = centers * 2
        centers = centers / 2
        return centers

    def iou(self, mode):
        min_iou, max_iou = mode
        if min_iou is None:
            min_iou = float('-inf')
        if max_iou is None:
            max_iou = float('inf')
        return max_iou, min_iou

    def cal_rect(self, h, height, w, width):
        dist1 = width - w
        dist2 = height - h
        left = random.uniform(dist1)
        top = random.uniform(dist2)
        left_ = left + w
        top_ = top + h
        rect = np.array([int(left), int(top), int(left_), int(top_)])
        return rect

    def mask_(self, centers, rect):

        return (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1]) * (rect[0] < centers[:, 0]) * (
                rect[1] < centers[:, 1])

    def fill_in(self, current_boxes, rect):
        self.setMax(current_boxes, rect)
        current_boxes[:, :2] -= rect[:2]
        self.setMin(current_boxes, rect)
        current_boxes[:, 2:] -= rect[:2]

    def setMin(self, current_boxes, rect):
        current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])

    def setMax(self, current_boxes, rect):
        current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])


class RandomMirror(Transformer):
    def transform(self, image, boxes, classes):
        _, width, _ = image.shape
        boxes, image = self.mirror(boxes, image, width)
        return image, boxes, classes

    def mirror(self, boxes, image, width):
        if random.randint(2):
            boxes, image = self.setImg(boxes, image)
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return boxes, image

    def setImg(self, boxes, image):
        image = image[:, ::-1]
        boxes = boxes.copy()
        return boxes, image


class SwapChannels(Transformer):

    def __init__(self, swaps):
        self.swaps = swaps

    def transform(self, image):
        image = image[:, :, self.swaps]
        return image


class ColorJitter(Transformer):

    def __init__(self):
        self.pd = [RandomContrast(), ConvertColor(current="RGB", to='HSV'), RandomSaturation(), RandomHue(),
                   ConvertColor(current='HSV', to='RGB'), RandomContrast()]

        self.init()

    def init(self):
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def distort_(self):
        if random.randint(2):
            distort = self.setdis2()
        else:
            distort = self.setdis1()
        return distort

    def setdis1(self):
        distort = Compose(self.pd[1:])
        return distort

    def setdis2(self):
        distort = Compose(self.pd[:-1])
        return distort

    def transform(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness.transform(im, boxes, labels)
        distort = self.distort_()

        im, boxes, labels = distort.transform(im, boxes, labels)
        return self.rand_light_noise.transform(im, boxes, labels)

    def init(self):
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()


class YoloAugmentation(Transformer):

    def __init__(self, image_size=416, mean=(123, 117, 104)):
        super().__init__()
        self.init(image_size, mean)

    def init(self, image_size, mean):
        self.image_size = image_size
        self.mean = mean
        self.private = 0
        self.pub = 1
        self.transformers = Compose(
            [ImageToFloat32(), BBoxToAbsoluteCoords(), RandomMirror(), ColorJitter(), RandomSampleCrop(),
             BBoxToPercentCoords(), Resize(image_size), ])
        self.BOOL_ = self.private * self.pub

    def transform(self, image, bbox, label):
        image = image - self.BOOL_
        return self.transformers.transform(image, bbox, label)


class ColorAugmentation(Transformer):

    def __init__(self, image_size=416, mean=(123, 117, 104)):
        super().__init__()
        self.init(image_size, mean)

    def init(self, image_size, mean):
        self.image_size = image_size
        self.mean = mean
        self.transformers = Compose(
            [ImageToFloat32(), BBoxToAbsoluteCoords(), RandomMirror(), ColorJitter(), BBoxToPercentCoords(),
             Resize(image_size), ])
        self.private = 0
        self.pub = 1

    def transform(self, image, bbox, label):
        return self.transformers.transform(image, bbox, label)


class ToTensor(Transformer):

    def __init__(self, image_size=416):
        self.init(image_size)

    def init(self, image_size):
        super().__init__()
        self.image_size = image_size
        self.padding = iaa.PadToAspectRatio(1, position='center-center')

    def resize(self, image, size):
        x = cv2.resize(image, (size, size)).astype(np.float32)
        x = self.trans_x(x)
        return x

    def transform(self, image: ndarray, bbox: ndarray = None, label: ndarray = None):
        size = self.image_size
        image = self.padding(image=image)
        x = self.resize(image, size)
        return x

    def trans_x(self, x):
        x /= 255.0
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
        return x
