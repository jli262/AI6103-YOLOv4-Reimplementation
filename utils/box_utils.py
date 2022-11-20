# coding: utf-8
import cmapy
import numpy as np
import torch
from typing import List, Union
from numpy import ndarray
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor
from math import pi


def iou(bbox1: Tensor, bbox2: Tensor):
    A, B = sizeFact(bbox1, bbox2)
    bbox1, bbox2 = ctC(bbox1, bbox2)
    xy_max = maxXY(A, B, bbox1, bbox2)
    xy_min = minXY(A, B, bbox1, bbox2)

    inter = calcInt(xy_max, xy_min)
    area_prior = calcPrio(A, B, bbox1)
    area_bbox = calcBBox(A, B, bbox2)

    return inter / (area_prior + area_bbox - inter)


def calcBBox(A, B, bbox2):
    a = bbox2[:, 2] - bbox2[:, 0]
    b = bbox2[:, 3] - bbox2[:, 1]
    area_bbox = (a * b).unsqueeze(0).expand(A, B)
    return area_bbox


def calcPrio(A, B, bbox1):
    a = bbox1[:, 2] - bbox1[:, 0]
    b = bbox1[:, 3] - bbox1[:, 1]
    area_prior = (a * b).unsqueeze(1).expand(A, B)
    return area_prior


def calcInt(xy_max, xy_min):
    inter = (xy_max - xy_min).clamp(min=0)
    inter = inter[:, :, 0] * inter[:, :, 1]
    return inter


def maxXY(A, B, bbox1, bbox2):
    a = bbox1[:, 2:].unsqueeze(1).expand(A, B, 2)
    b = bbox2[:, 2:].unsqueeze(0).expand(A, B, 2)
    xy_max = torch.min(a, b)
    return xy_max


def minXY(A, B, bbox1, bbox2):
    a = bbox1[:, :2].unsqueeze(1).expand(A, B, 2)
    b = bbox2[:, :2].unsqueeze(0).expand(A, B, 2)
    xy_min = torch.max(a, b)
    return xy_min


def ctC(bbox1, bbox2):
    bbox1 = center_to_corner(bbox1)
    bbox2 = center_to_corner(bbox2)
    return bbox1, bbox2


def sizeFact(bbox1, bbox2):
    A = bbox1.size(0)
    B = bbox2.size(0)
    return A, B


def jaccard_overlap_numpy(box: np.ndarray, boxes: np.ndarray):
    xy_max, xy_min = calcMM(box, boxes)
    inter = calcInter(xy_max, xy_min)
    inter = inter[:, 0] * inter[:, 1]
    area_box, area_boxes = calcArea(box, boxes)
    iou = inter / (area_box + area_boxes - inter)
    return iou


def calcArea(box, boxes):
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    a = boxes[:, 2] - boxes[:, 0]
    b = boxes[:, 3] - boxes[:, 1]
    area_boxes = a * b
    return area_box, area_boxes


def calcInter(xy_max, xy_min):
    inter = np.clip(xy_max - xy_min, a_min=0, a_max=np.inf)
    return inter


def calcMM(box, boxes):
    xy_min = np.maximum(boxes[:, :2], box[:2])
    xy_max = np.minimum(boxes[:, 2:], box[2:])
    return xy_max, xy_min


def ciou(bbox1: Tensor, bbox2: Tensor):
    xy_max1, xy_min1 = calcXY(bbox1)
    xy_max2, xy_min2 = calcXY(bbox2)

    xy_max, xy_min = calcMMs(xy_max1, xy_max2, xy_min1, xy_min2)

    inter = calcInters(xy_max, xy_min)
    union = calcUn(bbox1, bbox2, inter)
    iou = inter / (union + 1e-7)
    xy_max, xy_min = calcMMRev(xy_max, xy_max1, xy_max2, xy_min, xy_min1, xy_min2)
    center_distance = (torch.pow(bbox1[..., :2] - bbox2[..., :2], 2)).sum(dim=-1)
    diag_distance = torch.pow(xy_max - xy_min, 2).sum(dim=-1)
    v = calcSimi(bbox1, bbox2)
    result = calcResult(center_distance, diag_distance, iou, v)
    return result


def calcResult(center_distance, diag_distance, iou, v):
    alpha = v / torch.clamp((1 - iou + v), min=1e-6)
    result = iou - center_distance / diag_distance.clamp(min=1e-6) - alpha * v
    return result


def calcSimi(bbox1, bbox2):
    a = bbox1[..., 3].clamp(min=1e-6)
    b = bbox2[..., 3].clamp(min=1e-6)
    v = 4 / (pi ** 2) * torch.pow(
        torch.atan(bbox1[..., 2] / a) - torch.atan(bbox2[..., 2] / b), 2
    )
    return v


def calcUn(bbox1, bbox2, inter):
    a = bbox1[..., 2]
    b = bbox1[..., 3]
    c = bbox2[..., 2]
    d = bbox2[..., 3]
    union = a * b + c * d - inter
    return union


def calcMMRev(xy_max, xy_max1, xy_max2, xy_min, xy_min1, xy_min2):
    xy_max = torch.max(xy_max1, xy_max2)
    xy_min = torch.min(xy_min1, xy_min2)
    return xy_max, xy_min


def calcInters(xy_max, xy_min):
    inter = (xy_max - xy_min).clamp(min=0)
    inter = inter[..., 0] * inter[..., 1]
    return inter


def calcMMs(xy_max1, xy_max2, xy_min1, xy_min2):
    xy_max = torch.min(xy_max1, xy_max2)
    xy_min = torch.max(xy_min1, xy_min2)
    return xy_max, xy_min


def calcXY(bbox2):
    a = bbox2[..., [0, 1]]
    b = bbox2[..., [2, 3]]
    c = bbox2[..., [0, 1]]
    d = bbox2[..., [2, 3]]
    xy_min2 = a - b / 2
    xy_max2 = c + d / 2
    return xy_max2, xy_min2


def center_to_corner(boxes: Tensor):
    a = boxes[:, :2] - boxes[:, 2:] / 2
    b = boxes[:, :2] + boxes[:, 2:] / 2
    return torch.cat((a, b), dim=1)


def center_to_corner_numpy(boxes: ndarray) -> ndarray:
    a = boxes[:, :2] - boxes[:, 2:] / 2
    b = boxes[:, :2] + boxes[:, 2:] / 2
    return np.hstack((a, b))


def corner_to_center(boxes: Tensor):
    a = (boxes[:, :2] + boxes[:, 2:]) / 2
    b = boxes[:, 2:] - boxes[:, :2]
    return torch.cat((a, b), dim=1)


def corner_to_center_numpy(boxes: ndarray) -> ndarray:
    a = (boxes[:, :2] + boxes[:, 2:]) / 2
    b = boxes[:, 2:] - boxes[:, :2]
    return np.hstack((a, b))


def decode(pred: Tensor, anchors: Union[List[List[int]], np.ndarray], n_classes: int, image_size: int):
    n_anchors = len(anchors)
    N, h, pred, w = predFact(n_anchors, n_classes, pred)
    step_h, step_w = zoomIn(h, image_size, w)
    anchors = calcAnchor(anchors, step_h, step_w)
    anchors = Tensor(anchors)

    cx, cy = calcCXY(N, h, n_anchors, w)
    ph, pw = calcPwh(N, anchors, h, n_anchors, w)

    out = torch.zeros_like(pred)
    out01(cx, cy, out, pred)
    out2(out, pred, pw)
    out3(out, ph, pred)
    out4(out, pred)
    zoomOut(out, step_h, step_w)
    return out


def calcPw(N, anchors, h, n_anchors, w):
    pw = anchors[:, 0].view(n_anchors, 1, 1).repeat(N, 1, h, w)
    return pw


def calcCy(N, h, n_anchors, w):
    cy = torch.linspace(0, h - 1, h).view(h, 1).repeat(N, n_anchors, 1, w)
    return cy


def zoomOut(out, step_h, step_w):
    out[..., [0, 2]] *= step_w
    out[..., [1, 3]] *= step_h


def calcPwh(N, anchors, h, n_anchors, w):
    pw = calcPw(N, anchors, h, n_anchors, w)
    ph = anchors[:, 1].view(n_anchors, 1, 1).repeat(N, 1, h, w)
    return ph, pw


def calcCXY(N, h, n_anchors, w):
    cx = torch.linspace(0, w - 1, w).repeat(N, n_anchors, h, 1)
    cy = calcCy(N, h, n_anchors, w)
    return cx, cy


def calcAnchor(anchors, step_h, step_w):
    anchors = [[i / step_w, j / step_h] for i, j in anchors]
    return anchors


def zoomIn(h, image_size, w):
    step_h = image_size / h
    step_w = image_size / w
    return step_h, step_w


def predFact(n_anchors, n_classes, pred):
    N, _, h, w = pred.shape
    pred = pred.view(N, n_anchors, n_classes + 5, h, w).permute(0, 1, 3, 4, 2).contiguous().cpu()
    return N, h, pred, w


def out4(out, pred):
    out[..., 4:] = pred[..., 4:].sigmoid()


def out3(out, ph, pred):
    out[..., 3] = ph * torch.exp(pred[..., 3])


def out2(out, pred, pw):
    out[..., 2] = pw * torch.exp(pred[..., 2])


def out01(cx, cy, out, pred):
    out0(cx, out, pred)
    out[..., 1] = cy + pred[..., 1].sigmoid()


def out0(cx, out, pred):
    out[..., 0] = cx + pred[..., 0].sigmoid()


def calc1(gi, gj, gt, i, j, k, target):
    gt[i, k, gi, gj, 4] = 1
    gt[i, k, gi, gj, 5 + int(target[j, 4])] = 1


def calcG(gh, gi, gj, gt, gw, i, k):
    gt[i, k, gi, gj, 2] = gw
    gt[i, k, gi, gj, 3] = gh


def calcC(cx, cy, gi, gj, gt, i, k):
    gt[i, k, gi, gj, 0] = cx
    gt[i, k, gi, gj, 1] = cy


def match(anchors: list, anchor_mask: list, targets: List[Tensor], h: int, w: int, n_classes: int, overlap_thresh=0.5):
    N, n_anchors = nLen(anchor_mask, targets)
    n_mask, p_mask = maskPN(N, h, n_anchors, w)
    anchors, gt = matchAnd(N, anchors, h, n_anchors, n_classes, w)

    for i in range(N):
        if len(targets[i]) == 0:
            continue

        target = tgForm(i, targets, w)
        tgFormer(h, i, target, targets)
        a = torch.zeros(target.size(0), 2)
        bbox = torch.cat((a, target[:, 2:4]), 1)
        iousss = iou(bbox, anchors)
        best_indexes = torch.argmax(iousss, dim=1)

        for j, best_i in enumerate(best_indexes):
            if best_i not in anchor_mask:
                continue
            k = anchor_mask.index(best_i)

            cx, cy, gh, gw = calcVal(j, target)
            gj, gi = int(cx), int(cy)
            calcTF(gi, gj, i, k, n_mask, p_mask)
            calcGt(cx, cy, gh, gi, gj, gt, gw, i, j, k, target)

    return p_mask, n_mask, gt


def calcGt(cx, cy, gh, gi, gj, gt, gw, i, j, k, target):
    calcC(cx, cy, gi, gj, gt, i, k)
    calcG(gh, gi, gj, gt, gw, i, k)
    calc1(gi, gj, gt, i, j, k, target)


def tgForm(i, targets, w):
    target = torch.zeros_like(targets[i])
    target[:, [0, 2]] = targets[i][:, [0, 2]] * w
    return target


def matchAnd(N, anchors, h, n_anchors, n_classes, w):
    gt = torch.zeros(N, n_anchors, h, w, n_classes + 5)
    anchors = torch.hstack((torch.zeros((len(anchors), 2)), Tensor(anchors)))
    return anchors, gt


def maskPN(N, h, n_anchors, w):
    p_mask = torch.zeros(N, n_anchors, h, w)
    n_mask = torch.ones(N, n_anchors, h, w)
    return n_mask, p_mask


def calcTF(gi, gj, i, k, n_mask, p_mask):
    p_mask[i, k, gi, gj] = 1
    n_mask[i, k, gi, gj] = 0


def calcVal(j, target):
    cx, gw = target[j, [0, 2]]
    cy, gh = target[j, [1, 3]]
    return cx, cy, gh, gw


def tgF4(i, target, targets):
    target[:, 4] = targets[i][:, 4]


def tgFormer(h, i, target, targets):
    target[:, [1, 3]] = targets[i][:, [1, 3]] * h
    tgF4(i, target, targets)


def nLen(anchor_mask, targets):
    N = len(targets)
    n_anchors = len(anchor_mask)
    return N, n_anchors


def calcx2(boxes):
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    return x2, y2


def calcx1(boxes):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    return x1, y1


def nms(boxes: Tensor, scores: Tensor, overlap_thresh=0.45, top_k=100):
    keep = []
    x1, x2, y1, y2 = calcBox(boxes)
    area = (x2 - x1) * (y2 - y1)
    indexes = indexFact(scores, top_k)
    if boxes.numel() == 0:
        return torch.LongTensor(keep)

    while indexes.numel():
        if indexes.numel() == 1:
            break
        i = keeApp(indexes, keep)
        left, right = calcLR(i, indexes, x1, x2)
        top, bottom = calcLR(i, indexes, y1, y2)
        inter = ((right - left) * (bottom - top)).clamp(min=0)
        indexes = calcLas(area, i, indexes, inter, overlap_thresh)
    return torch.LongTensor(keep)


def calcLas(area, i, indexes, inter, overlap_thresh):
    ious = inter / (area[i] + area[indexes] - inter)
    indexes = indexes[ious < overlap_thresh]
    return indexes


def keeApp(indexes, keep):
    i = indexes[0]
    keep.append(i)
    return i


def indexFact(scores, top_k):
    _, indexes = scores.sort(dim=0, descending=True)
    indexes = indexes[:top_k]
    return indexes


def calcLR(i, indexes, x1, x2):
    right = x2[indexes].clamp(max=x2[i].item())
    left = x1[indexes].clamp(min=x1[i].item())
    return left, right


def calcBox(boxes):
    boxes = center_to_corner(boxes)
    x1, y1 = calcx1(boxes)
    x2, y2 = calcx2(boxes)
    return x1, x2, y1, y2


def draw(image: Union[ndarray, Image.Image], bbox: ndarray, label: ndarray, conf: ndarray = None) -> Image.Image:
    bbox = center_to_corner_numpy(bbox).astype(np.int)
    font = ImageFont.truetype('resource/font/msyh.ttc', size=13)
    if isinstance(image, ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    color_indexes, label_unique = labelSet(label)
    image_draw = ImageDraw.Draw(image, 'RGBA')

    for i in range(bbox.shape[0]):
        x1, y1 = calcmax1(bbox, i)
        x2, y2 = calcmin2(bbox, i, image)

        color = colorChose(color_indexes, i, label, label_unique)
        image_draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        l, text, y1_, y2_ = locate(conf, font, i, label, y1)
        left, right = calcLRF(image, l, x1)
        drawRT(color, image_draw, left, right, y1_, y2_)
        image_draw.text([left + 2, y1_ + 2], text=text, font=font, embedded_color=color)
    return image


def calcLRF(image, l, x1):
    if x1 + l <= image.width - 1:
        right = x1 + l
    else:
        right = image.width - 1
    left = int(right - l)
    return left, right


def drawRT(color, image_draw, left, right, y1_, y2_):
    image_draw.rectangle([left, y1_, right, y2_], fill=color + 'AA', outline=color + 'DD')


def locate(conf, font, i, label, y1):
    y1_ = y1 if y1 - 23 < 0 else y1 - 23
    y2_ = y1 if y1_ < y1 else y1 + 23
    text = refText(conf, i, label)
    l = font.getlength(text) + 3
    return l, text, y1_, y2_


def boxSet1(bbox, h, h_, pad_y):
    bbox[:, [1, 3]] = (bbox[:, [1, 3]] - pad_y / 2) * h / h_


def boxSet0(bbox, pad_x, w, w_):
    bbox[:, [0, 2]] = (bbox[:, [0, 2]] - pad_x / 2) * w / w_


def refText(conf, i, label):
    text = label[i] if conf is None else f'{label[i]} | {conf[i]:.2f}'
    return text


def colorChose(color_indexes, i, label, label_unique):
    class_index = label_unique.index(label[i])
    color = to_hex_color(cmapy.color(
        'rainbow', color_indexes[class_index], True))
    return color


def calcmin2(bbox, i, image):
    x2 = min(image.width - 1, bbox[i, 2])
    y2 = min(image.height - 1, bbox[i, 3])
    return x2, y2


def calcmax1(bbox, i):
    x1 = max(0, bbox[i, 0])
    y1 = max(0, bbox[i, 1])
    return x1, y1


def labelSet(label):
    label_unique = np.unique(label).tolist()
    color_indexes = np.linspace(0, 255, len(label_unique), dtype=int)
    return color_indexes, label_unique


def to_hex_color(color):
    transcolor = [hex(c)[2:].zfill(2) for c in color]
    hexcolor = '#' + ''.join(transcolor)
    return hexcolor


def rescale_bbox(bbox: ndarray, image_size, h, w):
    pad_x, pad_y = calcPad(h, image_size, w)
    w_ = image_size - pad_x
    h_ = image_size - pad_y
    bbox = bboxSet(bbox, h, h_, pad_x, pad_y, w, w_)
    bbox = corner_to_center_numpy(bbox)
    return bbox


def bboxSet(bbox, h, h_, pad_x, pad_y, w, w_):
    bbox = center_to_corner_numpy(bbox)
    boxSet0(bbox, pad_x, w, w_)
    boxSet1(bbox, h, h_, pad_y)
    return bbox


def calcPad(h, image_size, w):
    pad_x = max(h - w, 0) * image_size / max(h, w)
    pad_y = max(w - h, 0) * image_size / max(h, w)
    return pad_x, pad_y
