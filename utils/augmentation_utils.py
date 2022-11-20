# from https://github.com/amdegroot/ssd.pytorch

import cv2
import imgaug.augmenters as iaa
import numpy as np
import torch
from imgaug import augmenters as iaa
from numpy import ndarray, random



def intersect(box_a, box_b):
    return cal_int(box_a, box_b)


def cal_int(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):


    inter = intersect(box_a, box_b)
    union = union_(box_a, box_b, inter)
    return inter / union  # [A,B]


def union_(box_a, box_b, inter):
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return union


def remove_empty_boxes(boxes, labels):


    del_boxes = []
    for idx, box in enumerate(boxes):
        if box[0] == box[2] or box[1] == box[3]:
            del_boxes.append(idx)

    return np.delete(boxes, del_boxes, 0), np.delete(labels, del_boxes)

# all tranformers are shown here
class Transformer:

    def transform(self, image: ndarray, bbox: ndarray, label: ndarray):

        raise NotImplementedError("Data augmentation must be rewritten")


class Compose(Transformer):

    def __init__(self, transforms):
        self.transforms = transforms

    def transform(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t.transform(img, boxes, labels)
            if boxes is not None:
                boxes, labels = remove_empty_boxes(boxes, labels)
        return img, boxes, labels


class ImageToFloat32(Transformer):
    def transform(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


class SubtractMeans(Transformer):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def transform(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels


class BBoxToAbsoluteCoords(Transformer):
    def transform(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        self.box(boxes, height, width)

        return image, boxes, labels

    def box(self, boxes, height, width):
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height


class BBoxToPercentCoords(Transformer):
    def transform(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        self.box(boxes, height, width)

        return image, boxes, labels

    def box(self, boxes, height, width):
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height


class Resize(Transformer):
    def __init__(self, size=416):
        self.size = size

    def transform(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size, self.size))
        return image, boxes, labels


class RandomSaturation(Transformer):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def transform(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels


class RandomHue(Transformer):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def transform(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class RandomLightingNoise(Transformer):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def transform(self, image, boxes=None, labels=None):
        image = self.swap_(image)
        return image, boxes, labels

    def swap_(self, image):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle.transform(image)
        return image


class ConvertColor(Transformer):
    def __init__(self, current: str, to: str):
        self.to = to
        self.current = current

    def transform(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.to == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        elif self.current == 'BGR' and self.to == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        elif self.current == 'HSV' and self.to == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

        elif self.current == 'HSV' and self.to == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        elif self.current == 'RGB' and self.to == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        else:
            raise NotImplementedError
        return image, boxes, labels


class RandomContrast(Transformer):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def transform(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


class RandomBrightness(Transformer):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def transform(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


class RandomSampleCrop(Transformer):


    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),(0.3, None),(0.7, None),(0.9, None),(None, None),
        )

    def transform(self, image, boxes=None, labels=None):
        # guard against no boxes
        if boxes is not None and boxes.shape[0] == 0:
            return image, boxes, labels
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = self.sample_options[random.randint(
                0, len(self.sample_options))]
            if mode is None:
                return image, boxes, labels

            max_iou, min_iou = self.iou(mode)

            # max trails (50)
            for i in range(50):
                current_image = image

                h, w = self.initial_hw(height, width)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                rect = self.cal_rect(h, height, w, width)

                overlap = jaccard_numpy(boxes, rect)

                if overlap.max() < min_iou or overlap.min() > max_iou:
                    continue

                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

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

    def iou(self, mode):
        min_iou, max_iou = mode
        if min_iou is None:
            min_iou = float('-inf')
        if max_iou is None:
            max_iou = float('inf')
        return max_iou, min_iou

    def cal_rect(self, h, height, w, width):
        left = random.uniform(width - w)
        top = random.uniform(height - h)
        rect = np.array([int(left), int(top), int(left + w), int(top + h)])
        return rect

    def mask_(self, centers, rect):
        m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
        m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
        mask = m1 * m2
        return mask

    def fill_in(self, current_boxes, rect):
        current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
        current_boxes[:, :2] -= rect[:2]
        current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])
        current_boxes[:, 2:] -= rect[:2]


class RandomMirror(Transformer):
    def transform(self, image, boxes, classes):
        _, width, _ = image.shape
        boxes, image = self.mirror(boxes, image, width)
        return image, boxes, classes

    def mirror(self, boxes, image, width):
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return boxes, image


class SwapChannels(Transformer):


    def __init__(self, swaps):
        self.swaps = swaps

    def transform(self, image):

        image = image[:, :, self.swaps]
        return image


class ColorJitter(Transformer):

    def __init__(self):
        self.pd = [
            RandomContrast(),  # RGB
            ConvertColor(current="RGB", to='HSV'),  # HSV
            RandomSaturation(),  # HSV
            RandomHue(),  # HSV
            ConvertColor(current='HSV', to='RGB'),  # RGB
            RandomContrast()  # RGB
        ]

        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def transform(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness.transform(im, boxes, labels)
        boxes, im, labels = self.distort_(boxes, im, labels)
        return self.rand_light_noise.transform(im, boxes, labels)

    def distort_(self, boxes, im, labels):
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort.transform(im, boxes, labels)
        return boxes, im, labels


class YoloAugmentation(Transformer):

    def __init__(self, image_size=416, mean=(123, 117, 104)):
        super().__init__()
        self.image_size = image_size
        self.mean = mean
        self.transformers = Compose([
            ImageToFloat32(),
            BBoxToAbsoluteCoords(),
            RandomMirror(),
            ColorJitter(),
            RandomSampleCrop(),
            BBoxToPercentCoords(),
            Resize(image_size),
            # SubtractMeans(mean)
        ])

    def transform(self, image, bbox, label):
        return self.transformers.transform(image, bbox, label)


class ColorAugmentation(Transformer):

    def __init__(self, image_size=416, mean=(123, 117, 104)):
        super().__init__()
        self.image_size = image_size
        self.mean = mean
        self.transformers = Compose([
            ImageToFloat32(),
            BBoxToAbsoluteCoords(),
            RandomMirror(),
            ColorJitter(),
            BBoxToPercentCoords(),
            Resize(image_size),
            # SubtractMeans(mean)
        ])

    def transform(self, image, bbox, label):
        return self.transformers.transform(image, bbox, label)


class ToTensor(Transformer):


    def __init__(self, image_size=416):

        super().__init__()
        self.image_size = image_size
        self.padding = iaa.PadToAspectRatio(1, position='center-center')

    def transform(self, image: ndarray, bbox: ndarray = None, label: ndarray = None):

        size = self.image_size
        image = self.padding(image=image)
        x = self.resize(image, size)
        return x

    def resize(self, image, size):
        x = cv2.resize(image, (size, size)).astype(np.float32)
        x /= 255.0
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
        return x
