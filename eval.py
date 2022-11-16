# coding:utf-8
from net import EvalPipeline, VOCDataset

# load dataset
root = 'VOC2007_test'
dataset = VOCDataset(root, 'test')

model_path = 'model/2022-11-15_13-49-27/Yolo_200.pth'
eval_pipeline = EvalPipeline(model_path, dataset, conf_thresh=0.001)
eval_pipeline.eval()