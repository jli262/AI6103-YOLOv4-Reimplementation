from xml.etree import ElementTree as ET
from utils.box_utils import jaccard_overlap_numpy, center_to_corner_numpy, rescale_bbox
from utils.augmentation_utils import ToTensor
import numpy as np


import json
from pathlib import Path

from .dataset import VOCDataset
from .yolo import Yolo
from PIL import Image
from prettytable import PrettyTable
import torch
from torch import cuda

class EvalPipeline:
    def __init__(self, model_path: str, dataset: VOCDataset, image_size=416, anchors: list = None, conf_thresh=0.05, overlap_thresh=0.5, save_dir='eval', use_07_metric=False):
        self.setDevice()

        self.setDataset(dataset)
        self.setImageSize(image_size)
        self.setConfThresh(conf_thresh)

        self.setOverlapThresh(overlap_thresh)
        self.set07Metric(use_07_metric)
        self.setSaveDir(save_dir)
        self.setModelPath(model_path)
        self.setModel(anchors, image_size)
        self.setDConfThresh(conf_thresh)
        self.setModelDevice()
        self.setModelLoad(model_path)
        self.model.eval()
    def setModelLoad(self, model_path):
        self.model.load(model_path)

    def setModelDevice(self):
        self.model = self.model.to(self.device)

    def setDConfThresh(self, conf_thresh):
        self.model.detector.conf_thresh = conf_thresh

    def setModel(self, anchors, image_size):
        self.model = Yolo(self.dataset.n_classes, image_size, anchors)

    def setDevice(self):
        self.device = 'cuda'

    def setModelPath(self, model_path):
        self.model_path = Path(model_path)

    def setSaveDir(self, save_dir):
        self.save_dir = Path(save_dir)

    def set07Metric(self, use_07_metric):
        self.use_07_metric = use_07_metric

    def setOverlapThresh(self, overlap_thresh):
        self.overlap_thresh = overlap_thresh

    def setConfThresh(self, conf_thresh):
        self.conf_thresh = conf_thresh

    def setImageSize(self, image_size):
        self.image_size = image_size

    def setDataset(self, dataset):
        self.dataset = dataset

    @torch.no_grad()
    def eval(self):
        self._predict()
        self._get_ground_truth()
        return self._get_mAP()

    def _predict(self):

        self.setPreds()
        transformer = self.setTrans()

        print('In predicting...')
        self.predLoop(transformer)

    def predLoop(self, transformer):
        for i, (image_path, image_name) in enumerate(zip(self.dataset.imagePaths, self.dataset.imageNames)):
            print(f'\rRate of progress ：{i / len(self.dataset):.0%}', end='')

            h, image, w = self.setImageHW(image_path)

            out = self.getOutput(image, transformer)
            if not out:
                continue

            self.getResult(h, image_name, out, w)

    def getResult(self, h, image_name, out, w):
        for c, pred in out[0].items():
            mask, pred = self.setFactor(pred)
            if not mask.any():
                continue

            bbox, conf = self.resultFilter(h, mask, pred, w)
            self.saveResult(bbox, c, conf, image_name)

    def setFactor(self, pred):
        pred = pred.numpy()
        mask = pred[:, 0] > self.conf_thresh
        return mask, pred

    def saveResult(self, bbox, c, conf, image_name):
        self.preds[self.dataset.VOC2007_classes[c]][image_name] = {"bbox": bbox.tolist(),"conf": conf.tolist()}

    def calConf(self, mask, pred):
        conf = pred[:, 0][mask]
        return conf

    def getOutput(self, image, transformer):
        x = transformer.transform(image).to(self.device)
        out = self.model.predict(x)
        return out

    def resultFilter(self, h, mask, pred, w):
        conf = self.calConf(mask, pred)
        bbox = rescale_bbox(pred[:, 1:][mask], self.image_size, h, w)
        bbox = center_to_corner_numpy(bbox)
        return bbox, conf

    def setImageHW(self, image_path):
        image = np.array(Image.open(image_path).convert('RGB'))
        h, w, _ = image.shape
        return h, image, w

    def setTrans(self):
        transformer = ToTensor(self.image_size)
        return transformer

    def setPreds(self):
        self.preds = {c: {} for c in self.dataset.VOC2007_classes}

    def _get_ground_truth(self):

        self.getGroundTruth()
        self.getPositiveNums()

        print('\n\nFetching labels...')

        self.gtLoop()

    def gtLoop(self):
        for i, (anno_path, img_name) in enumerate(zip(self.dataset.annotationPaths, self.dataset.imageNames)):
            root = self.getRoot(anno_path)
            self.fetchObj(img_name, root)
            print(f'\rRate of progress：{i / len(self.dataset):.0%}', end='')

    def fetchObj(self, img_name, root):
        for obj in root.iter('object'):

            bbox, c, difficult = self.getLabelBox(img_name, obj)

            self.addMark(bbox, c, difficult, img_name)

    def addMark(self, bbox, c, difficult, img_name):
        self.ground_truths[c][img_name]['detected'].append(False)
        self.ground_truths[c][img_name]['bbox'].append(bbox)
        self.ground_truths[c][img_name]['difficult'].append(difficult)
        self.calPosNum(c, difficult)

    def calPosNum(self, c, difficult):
        self.n_positives[c] += (1 - difficult)

    def getLabelBox(self, img_name, obj):
        c = self.calC(obj)
        difficult = self.setDifficult(obj)
        bbox = self.findBbox(obj)
        bbox = self.findBboxes(bbox)
        self.clearGt(c, img_name)
        return bbox, c, difficult

    def clearGt(self, c, img_name):
        if not self.ground_truths[c].get(img_name):
            self.ground_truths[c][img_name] = {"detected": [], "bbox": [], "difficult": []}

    def findBboxes(self, bbox):
        bbox = [int(bbox.find('xmin').text),int(bbox.find('ymin').text),int(bbox.find('xmax').text),int(bbox.find('ymax').text),]
        return bbox

    def findBbox(self, obj):
        bbox = obj.find('bndbox')
        return bbox

    def setDifficult(self, obj):
        difficult = int(obj.find('difficult').text)
        return difficult

    def calC(self, obj):
        c = obj.find('name').text.lower().strip()
        return c

    def getRoot(self, anno_path):
        root = ET.parse(anno_path).getroot()
        return root

    def getPositiveNums(self):
        self.n_positives = {c: 0 for c in self.dataset.VOC2007_classes}

    def getGroundTruth(self):
        self.ground_truths = {c: {} for c in self.dataset.VOC2007_classes}

    def _get_mAP(self):
        result = {}

        print('\n\nCalculating AP...')
        mAP = self.mapObtain(result)

        self.saveResult0(result)

        return mAP

    def saveResult0(self, result):
        self.save_dir.mkdir(exist_ok=True, parents=True)
        p = self.calDir()
        self.importJson(p, result)

    def importJson(self, p, result):
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(result, f)

    def calDir(self):
        p = self.save_dir / (self.model_path.stem + '_AP.json')
        return p

    def mapObtain(self, result):
        mAP = 0
        mAP, table = self.generateTable(mAP, result)
        mAP = self.calMap(mAP)
        table.add_column("mAP", [f"{mAP:.2%}"] + [""] * (len(self.dataset.VOC2007_classes) - 1))
        print(table)
        return mAP

    def calMap(self, mAP):
        mAP /= len(self.dataset.VOC2007_classes)
        return mAP

    def generateTable(self, mAP, result):
        table = self.setTable()
        for c in self.dataset.VOC2007_classes:
            ap, precision, recall = self._get_AP(c)
            result[c] = {'AP': ap,'precision': precision,'recall': recall}
            mAP = self.accuMAP(ap, mAP)
            table.add_row([c, f"{ap:.2%}"])
        return mAP, table

    def accuMAP(self, ap, mAP):
        mAP += ap
        return mAP

    def setTable(self):
        table = PrettyTable(["class", "AP"])
        return table

    def _get_AP(self, c: str):
        bbox, conf, ground_truth, image_names, pred = self.initParas(c)

        self.combineBoxes(bbox, conf, image_names, pred)

        if not bbox:
            return 0, 0, 0

        bbox, conf, image_names = self.setParas(bbox, conf, image_names)

        bbox, image_names = self.sortBoxes(bbox, conf, image_names)

        fp, tp = self.calTPFP(bbox, ground_truth, image_names)

        precision, recall = self.calPR(c, fp, tp)

        ap = self.calAPs(precision, recall)

        return ap, precision.tolist(), recall.tolist()

    def calAPs(self, precision, recall):
        if not self.use_07_metric:
            prec, rec = self.calPrec(precision, recall)

            for i in range(prec.size - 1, 0, -1):
                inde = np.maximum(prec[i - 1], prec[i])
                prec[i - 1] = inde

            i = self.calIndex(i, rec)
            ap = self.calAp1(i, prec, rec)
        else:
            ap = 0
            ap = self.calAp2(ap, precision, recall)
        return ap

    def calAp2(self, ap, precision, recall):
        for r in np.arange(0, 1.1, 0.1):
            if np.any(recall >= r):
                ap = self.calAp3(ap, precision, r, recall)
        return ap

    def calAp3(self, ap, precision, r, recall):
        ap += np.max(precision[recall >= r]) / 11
        return ap

    def calAp1(self, i, prec, rec):
        ap = np.sum((rec[i + 1] - rec[i]) * prec[i + 1])
        return ap

    def calIndex(self, i, rec):
        i = np.where(rec[1:] != rec[:-1])[0]
        return i

    def calPrec(self, precision, recall):
        rec = self.calRec(recall)
        prec = np.concatenate(([0.], precision, [0.]))
        return prec, rec

    def calRec(self, recall):
        rec = np.concatenate(([0.], recall, [1.]))
        return rec

    def calPR(self, c, fp, tp):
        tp = tp.cumsum()
        fp = fp.cumsum()
        n_positives = self.n_positives[c]
        recall = self.calR(n_positives, tp)
        precision = self.calP(fp, tp)
        return precision, recall

    def calP(self, fp, tp):
        precision = tp / (tp + fp)
        return precision

    def calR(self, n_positives, tp):
        recall = tp / n_positives
        return recall

    def calTPFP(self, bbox, ground_truth, image_names):
        tp = self.initP(image_names)
        fp = self.initP(image_names)
        for i, image_name in enumerate(image_names):
            record = ground_truth.get(image_name)
            if not record:
                fp[i] = 1
                continue

            bbox_gt, bbox_pred = self.setBboxpg(bbox, i, record)
            iou_max, iou_max_index = self.calIOU(bbox_gt, bbox_pred)

            if iou_max < self.overlap_thresh:
                fp[i] = 1
            elif not record['difficult'][iou_max_index]:
                if record['detected'][iou_max_index]:
                    fp[i] = 1
                else:
                    record['detected'][iou_max_index] = True
                    tp[i] = 1

        return fp, tp

    def calIOU(self, bbox_gt, bbox_pred):
        iou = jaccard_overlap_numpy(bbox_pred, bbox_gt)
        iou_max = iou.max()
        iou_max_index = iou.argmax()
        return iou_max, iou_max_index

    def setBboxpg(self, bbox, i, record):
        bbox_pred = bbox[i]
        bbox_gt = np.array(record['bbox'])
        return bbox_gt, bbox_pred

    def initP(self, image_names):
        tp = np.zeros(len(image_names))
        return tp

    def bcAppend(self, bbox, conf, v):
        bbox.append(v['bbox'])
        conf.append(v['conf'])

    def sortBoxes(self, bbox, conf, image_names):
        index = np.argsort(-conf)
        bbox = bbox[index]
        conf = conf[index]
        image_names = image_names[index]
        return bbox, image_names

    def setParas(self, bbox, conf, image_names):
        bbox = np.vstack(bbox)
        conf = np.hstack(conf)
        image_names = np.array(image_names)
        return bbox, conf, image_names

    def combineBoxes(self, bbox, conf, image_names, pred):
        for image_name, v in pred.items():
            image_names.extend([image_name] * len(v['conf']))
            self.bcAppend(bbox, conf, v)


    def initParas(self, c):
        bbox = []
        conf = []
        pred = self.preds[c]
        ground_truth = self.ground_truths[c]
        image_names = []
        return bbox, conf, ground_truth, image_names, pred
