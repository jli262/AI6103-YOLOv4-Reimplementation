# coding:utf-8

import json
from pathlib import Path
from typing import List
import matplotlib as mpl
import matplotlib.pyplot as plt
from net.eval_pipeline import EvalPipeline
from net.dataset import VOCDataset
from net.yolo import Yolo

mpl.rc_file('resource/theme/matlab.mplstyle')

root = 'VOC2007_test'
dataset = VOCDataset(root, 'test')


def fileOpen():
    with open('mAPs.json', 'w', encoding='utf-8') as f:
        json.dump(mAPs, f)


def image_detect(models_path: str, images_path: str, classes: List[str], conf_thresh=0.6):
    nms_thresh = 0.45
    image_size = 416
    model = Yolo(len(classes), image_size, None, conf_thresh, nms_thresh)
    model = model.cuda()
    model.load(models_path)
    model.eval()
    return model.detect(images_path, classes, True)


def pathsort():
    model_paths.sort(key=lambda i: int(i.stem.split("_")[1]))


def iterate():
    global model_path
    for model_path in model_paths:
        ep = EvalPipeline(model_path, dataset, conf_thresh=0.001)
        iterations.append(int(model_path.stem[5:]))
        mAPs.append(ep.eval() * 100)


model_dir = Path('model/2022-11-15_13-49-27')
model_paths = [i for i in model_dir.glob('Yolo_*')]
pathsort()

mAPs = []
iterations = []
iterate()
fileOpen()

fig, ax = plt.subplots(1, 1, num='mAPcurve')
ax.plot(iterations, mAPs)
ax.set(xlabel='iteration', ylabel='mAP', title='mAP curve')
plt.show()


model_path = 'model/2022-11-15_13-49-27/Yolo_200.pth'
image_path = 'VOC2007_test/JPEGImages/007447.jpg'

image = image_detect(model_path, image_path, VOCDataset.VOC2007_classes, conf_thresh=0.2)
image.show()
