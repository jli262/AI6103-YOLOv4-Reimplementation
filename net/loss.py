import torch
from torch import Tensor, nn

from utils.box_utils import match, decode, ciou, iou

from typing import List
import numpy as np

class YoloLoss(nn.Module):


    def __init__(self, anchors: list, n_classes: int, image_size: int, overlap_thresh=0.5):

        super().__init__()
        self.setAnchorsNum(anchors)
        self.setAnchors(anchors)
        self.setClassNum(n_classes)
        self.setImageSize(image_size)
        self.setOverlapThresh(overlap_thresh)

        self.setBalances()
        self.setLambdaBox()
        self.setLambdaObj(image_size)
        self.setLambdaCls(n_classes)

        self.setBceLoss()

    def setBceLoss(self):
        self.bce_loss = nn.BCELoss(reduction='mean')

    def setLambdaCls(self, n_classes):
        self.lambda_cls = n_classes / 80

    def setLambdaObj(self, image_size):
        self.lambda_obj = 5 * (image_size / 416) ** 2

    def setLambdaBox(self):
        self.lambda_box = 0.05

    def setBalances(self):
        self.balances = [0.4, 1, 4]

    def setOverlapThresh(self, overlap_thresh):
        self.overlap_thresh = overlap_thresh

    def setImageSize(self, image_size):
        self.image_size = image_size

    def setClassNum(self, n_classes):
        self.n_classes = n_classes

    def setAnchors(self, anchors):
        self.anchors = np.array(anchors).reshape(-1, 2)

    def setAnchorsNum(self, anchors):
        self.n_anchors = len(anchors[0])

    def forward(self, index: int,  pred: Tensor, targets: List[Tensor]):

        loss = 0
        N, _, h, w = pred.shape


        anchor_mask = list(
            range(index*self.n_anchors, (index+1)*self.n_anchors))
        pred = decode(pred, self.anchors[anchor_mask],
                      self.n_classes, self.image_size)

        step_h = self.calculateHW(h)
        step_w = self.calculateHW(w)

        anchors = [[i/step_w, j/step_h] for i, j in self.anchors]
        p_mask, n_mask, gt = match(
            anchors, anchor_mask, targets, h, w, self.n_classes, self.overlap_thresh)
        self.mark_ignore(pred, targets, n_mask)

        p_mask = self.setDevice(p_mask, pred)
        n_mask = self.setDevice(n_mask, pred)
        gt = self.setDevice(gt, pred)

        m = p_mask == 1
        if m.sum() != 0:

            iou = ciou(pred[..., :4], gt[..., :4])
            m &= torch.logical_not(torch.isnan(iou))
            loss = self.calculateLoss1(iou, loss, m)

            loss = self.calculateLoss2(gt, loss, m, pred)

        mask = n_mask.bool() | m
        loss = self.calculateLoss3(index, loss, m, mask, pred)

        return loss

    def calculateLoss3(self, index, loss, m, mask, pred):
        loss += self.bce_loss(pred[..., 4] * mask, m.type_as(pred) * mask) * \
                self.lambda_obj * self.balances[index]
        return loss

    def calculateLoss2(self, gt, loss, m, pred):
        loss += self.bce_loss(pred[..., 5:][m], gt[..., 5:][m]) * self.lambda_cls
        return loss

    def calculateLoss1(self, iou, loss, m):
        loss += torch.mean((1 - iou)[m]) * self.lambda_box
        return loss

    def setDevice(self, d, pred):
        d = d.to(pred.device)
        return d

    def calculateHW(self, x):
        step_x = self.image_size / x
        return step_x

    def mark_ignore(self, pred: Tensor, targets: List[Tensor], n_mask: Tensor):

        N, _, h, w, _ = pred.shape
        bbox = self.setBbox(pred)

        for i in range(N):
            if targets[i].size(0) == 0:
                continue

            box = self.setBox(bbox, i)
            target = self.calculateTarget(h, i, targets, w)

            max_iou = self.calculateMaxiou(box, i, pred, target)
            n_mask[i][max_iou > self.overlap_thresh] = 0

    def calculateMaxiou(self, box, i, pred, target):
        max_iou, _ = torch.max(iou(target, box), dim=0)
        max_iou = max_iou.view(pred[i].shape[:3])
        return max_iou

    def calculateTarget(self, h, i, targets, w):
        target = torch.zeros_like(targets[i][..., :4])
        target[:, [0, 2]] = targets[i][:, [0, 2]] * w
        target[:, [1, 3]] = targets[i][:, [1, 3]] * h
        return target

    def setBox(self, bbox, i):
        box = bbox[i].view(-1, 4)
        return box

    def setBbox(self, pred):
        bbox = pred[..., :4]
        return bbox
