# coding: utf-8
from typing import Dict
from xml.etree import ElementTree as ET

class AnnotationReader:

    def __init__(self, class_to_index: Dict[str, int], keep_difficult=False):

        self.setKD(keep_difficult)
        self.selfFactor(class_to_index)

    def setKD(self, keep_difficult):
        self.keep_difficult = keep_difficult

    def selfFactor(self, index):
        self.class_to_index = index

    def read(self, file_path: str):
        root = ET.parse(file_path).getroot()
        h, w = self.imageSize(root)

        target = []
        for obj in root.iter('object'):
            if int(obj.find('difficult').text) and not self.keep_difficult :
                continue
            points = ['xmin', 'ymin', 'xmax', 'ymax']
            data = []
            bbox, name = self.objFill(obj)
            for i, pt in enumerate(points):
                pt = self.calcPt(bbox, h, i, pt, w)
                data.append(pt)
            self.appendNew(data, name, target)

        return target

    def appendNew(self, data, name, target):
        data.append(self.class_to_index[name])
        target.append(data)

    def calcPt(self, bbox, h, i, pt, w):
        pt = int(bbox.find(pt).text) - 1
        pt = pt / w if i % 2 == 0 else pt / h
        return pt

    def objFill(self, obj):
        name = obj.find('name').text.lower().strip()
        bbox = obj.find('bndbox')
        return bbox, name

    def imageSize(self, root):
        img_size = root.find('size')
        w = int(img_size.find('width').text)
        h = int(img_size.find('height').text)
        return h, w


