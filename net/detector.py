# coding:utf-8
from typing import Tuple, Dict, List
import numpy as np

import torch
from utils.box_utils import decode, center_to_corner
from torchvision.ops import nms


class Detector:
    """ 探测器 """

    def __init__(self, anchors: list, image_size: int, n_classes: int, conf_thresh=0.25, nms_thresh=0.45):

        self.setAnchors(anchors)
        self.setClassNum(n_classes)
        self.setImageSize(image_size)
        self.setnmsThresh(nms_thresh)
        self.setConfThresh(conf_thresh)

    def setConfThresh(self, conf_thresh):
        self.conf_thresh = conf_thresh

    def setnmsThresh(self, nms_thresh):
        self.nms_thresh = nms_thresh

    def setImageSize(self, image_size):
        self.image_size = image_size

    def setClassNum(self, n_classes):
        self.n_classes = n_classes

    def setAnchors(self, anchors):
        self.anchors = anchors

    def __call__(self, preds: Tuple[torch.Tensor]) -> List[Dict[int, torch.Tensor]]:

        N = preds[0].size(0)

        batch_pred = self.getPredBatch(N, preds)

        out = self.getOutArray(batch_pred)

        return out

    def getOutArray(self, batch_pred):
        out = []
        for pred in batch_pred:
            pred[:, 5:] = self.calculatePred(pred)

            c, conf = self.getPredConf(pred)
            pred = self.precessPred(c, conf, pred)

            pred = self.precessPred2(pred)
            if not pred.size(0):
                continue

            classes_pred = self.getPredClass(pred)

            detections = self.setDetections(classes_pred, pred)

            out.append(detections)
        return out

    def calculatePred(self, pred):
        return pred[:, 5:] * pred[:, 4:5]

    def setDetections(self, classes_pred, pred):
        detections = {}
        for c in classes_pred:
            boxes, scores = self.getParas(c, pred)
            keep = self.calculateKeep(boxes, scores)
            detections[int(c)] = torch.cat(
                (scores[keep].unsqueeze(1), boxes[keep]), dim=1)
        return detections

    def calculateKeep(self, boxes, scores):
        keep = nms(center_to_corner(boxes), scores, self.nms_thresh)
        return keep

    def getParas(self, c, pred):
        mask = pred[:, -1] == c
        boxes = self.calculateBox(mask, pred)
        scores = self.calculateScore(mask, pred)
        return boxes, scores

    def calculateScore(self, mask, pred):
        scores = pred[:, 4][mask]
        return scores

    def calculateBox(self, mask, pred):
        boxes = pred[:, :4][mask]
        return boxes

    def getPredClass(self, pred):
        classes_pred = pred[:, -1].unique()
        return classes_pred

    def precessPred2(self, pred):
        pred = pred[pred[:, 4] >= self.conf_thresh]
        return pred

    def precessPred(self, c, conf, pred):
        pred = torch.cat((pred[:, :4], conf, c), dim=1)
        return pred

    def getPredConf(self, pred):
        conf, c = torch.max(pred[:, 5:], dim=1, keepdim=True)
        return c, conf

    def getPredBatch(self, N, preds):
        batch_pred = []
        for pred, anchors in zip(preds, self.anchors):
            pred_ = decode(pred, anchors, self.n_classes, self.image_size)

            batch_pred.append(pred_.view(N, -1, self.n_classes + 5))
        batch_pred = torch.cat(batch_pred, dim=1)
        return batch_pred

