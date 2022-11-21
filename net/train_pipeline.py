# coding:utf-8
import time
import torch
from torch import cuda, optim, nn
from torch.backends import cudnn
from .dataset import VOCDataset, dataLoader
from utils.log_utils import LossLogger, Logger
from .loss import YoloLoss
from .yolo import Yolo
import numpy as np
from utils.lr_schedule_utils import WarmUpCosLRSchedule, determin_lr, get_lr
from tqdm import tqdm
from pathlib import Path
from datetime import datetime


def exception_handler(train_func):
    def wrapper(train_pipeline, *args, **kwargs):
        try:
            return train_func(train_pipeline, *args, **kwargs)
        except BaseException as e:
            ifError(e, train_pipeline)

            exit()

    def ifError(e, train_pipeline):
        errorDetect(e)
        train_pipeline.save()
        cuda.empty_cache()

    def errorDetect(e):
        if not isinstance(e, KeyboardInterrupt):
            Logger("error").error(f"{e.__class__.__name__}: {e}", True)

    return wrapper


def time_delta(t: datetime):
    times = datetime.now() - t
    seconds = times.seconds % 60
    hours = times.seconds // 3600
    minutes = (times.seconds - times.seconds // 3600 * 3600) // 60
    return f'{hours:02}:{minutes:02}:{seconds:02}'


def make_optimizer(model: nn.Module, lr, momentum=0.9, weight_decay=5e-4):
    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_modules():
        setBiasApp(pg2, v)
        setWeightApp(k, pg0, pg1, v)

    optimizer = optim.SGD(pg0, lr, momentum=momentum, nesterov=True)
    setPara(optimizer, pg1, pg2, weight_decay)
    return optimizer


def setPara(optimizer, pg1, pg2, weight_decay):
    optimizer.add_param_group({"params": pg2})
    optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})


def setBiasApp(pg2, v):
    if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
        pg2.append(v.bias)


def setWeightApp(k, pg0, pg1, v):
    if isinstance(v, nn.BatchNorm2d) or "bn" in k:
        pg0.append(v.weight)
    elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
        pg1.append(v.weight)


class TrainPipeline:
    def __init__(self, n_classes: int, image_size: int, anchors: list, dataset: VOCDataset, darknet_path: str = None,
                 yolo_path: str = None, lr=0.01, momentum=0.9, weight_decay=4e-5, warm_up_ratio=0.02, freeze=True,
                 batch_size=4, freeze_batch_size=8, num_workers=4, freeze_epoch=20, start_epoch=0, max_epoch=60):

        save_frequency = 5
        save_dir = 'model'
        log_file: str = None
        log_dir = 'log'
        self.setDataset(dataset)
        self.setSaveDir(save_dir)
        self.setGPUDevice()
        self.setSaveFreq(save_frequency)
        self.setFBS(freeze_batch_size)
        self.setBatchSize(batch_size)

        self.setLr(lr)
        self.setFreeze(freeze)
        self.setMaxEpoch(max_epoch)
        self.setSEpoch(start_epoch)
        self.setCEpoch(start_epoch)
        self.setFEpoch(freeze_epoch)

        self.setModel(anchors, image_size, n_classes)
        self.setPrint(darknet_path, yolo_path)
        self.model.backbone.set_freezed(freeze)

        bs = self.setOptiandLoss(anchors, batch_size, freeze, freeze_batch_size, image_size, lr, max_epoch, momentum,
                                 n_classes, warm_up_ratio, weight_decay)

        self.dataLoading(bs, num_workers)

        self.logger = LossLogger(log_file, log_dir)

    def dataLoading(self, bs, num_workers):
        self.setWorkerNum(num_workers)
        self.setBatchNum(bs)
        self.setDataLoader(bs, num_workers)

    def setWorkerNum(self, num_workers):
        self.num_worksers = num_workers

    def setDataLoader(self, bs, num_workers):
        self.data_loader = dataLoader(self.dataset, bs, num_workers)

    def setBatchNum(self, bs):
        self.n_batches = len(self.dataset) // bs

    def setOptiandLoss(self, anchors, batch_size, freeze, freeze_batch_size, image_size, lr, max_epoch, momentum,
                       n_classes, warm_up_ratio, weight_decay):
        bs = self.setbs(batch_size, freeze, freeze_batch_size)
        lr_fit, lr_min_fit = determin_lr(lr, bs)
        self.setCriterion(anchors, image_size, n_classes)
        self.setOptimizer(lr_fit, momentum, weight_decay)
        self.set_lr_schedule(lr_fit, lr_min_fit, max_epoch, warm_up_ratio)
        return bs

    def set_lr_schedule(self, lr_fit, lr_min_fit, max_epoch, warm_up_ratio):
        self.lr_schedule = WarmUpCosLRSchedule(self.optimizer, lr_fit, lr_min_fit, max_epoch, warm_up_ratio)

    def setOptimizer(self, lr_fit, momentum, weight_decay):
        self.optimizer = make_optimizer(self.model, lr_fit, momentum, weight_decay)

    def setCriterion(self, anchors, image_size, n_classes):
        self.criterion = YoloLoss(anchors, n_classes, image_size)

    def setbs(self, batch_size, freeze, freeze_batch_size):
        bs = freeze_batch_size if freeze else batch_size
        return bs

    def setPrint(self, darknet_path, yolo_path):
        if darknet_path:
            self.model.backbone.load(darknet_path)
            print('Successfully loading Darknet53：' + darknet_path)
        if yolo_path:
            self.model.load(yolo_path)
            print('Successfully loading YOLO：' + yolo_path)

    def setModel(self, anchors, image_size, n_classes):
        self.model = Yolo(n_classes, anchors).to(self.device)

    def setGPUDevice(self):

        self.device = torch.device('cuda')


    def setFEpoch(self, freeze_epoch):
        self.free_epoch = freeze_epoch

    def setCEpoch(self, start_epoch):
        self.current_epoch = start_epoch

    def setSEpoch(self, start_epoch):
        self.start_epoch = start_epoch

    def setMaxEpoch(self, max_epoch):
        self.max_epoch = max_epoch

    def setFreeze(self, freeze):
        self.freeze = freeze

    def setLr(self, lr):
        self.lr = lr

    def setBatchSize(self, batch_size):
        self.batch_size = batch_size

    def setFBS(self, freeze_batch_size):
        self.freeze_batch_size = freeze_batch_size

    def setSaveFreq(self, save_frequency):
        self.save_frequency = save_frequency

    def setSaveDir(self, save_dir):
        self.save_dir = Path(save_dir)

    def setDataset(self, dataset):
        self.dataset = dataset

    def save(self):
        self.saveModelandLoss()

        path = self.saveModel()

        self.saveLoss()

        print(f'\nSave model in {path.absolute()}\n')

    def saveLoss(self):
        self.logger.save(f'train_losses_{self.current_epoch + 1}')

    def saveModel(self):
        self.model.eval()
        path = self.generatePath()
        torch.save(self.model.state_dict(), path)
        return path

    def generatePath(self):
        path = self.save_dir / f'Yolo_{self.current_epoch + 1}.pth'
        return path

    def saveModelandLoss(self):
        self.pbar.close()
        self.save_dir.mkdir(exist_ok=True, parents=True)

    @exception_handler
    def train(self):
        t = self.getTime()
        self.createDir(t)

        bar_format = self.set_bar_format()
        print('Training Begins!')
        self.trainLoop(bar_format)

        self.save()

    def trainLoop(self, bar_format):
        is_unfreezed = False
        for e in range(self.start_epoch, self.max_epoch):
            self.current_epoch = e

            self.freezeMethod(e, is_unfreezed)

            self.model.train()

            start_time = self.createBar(bar_format, e)

            iter, loss_value = self.trainPred(start_time)

            self.others(e, iter, loss_value)

            self.judgeIfMosaic(e)

            self.saveModelAtTimes(e)

    def saveModelAtTimes(self, e):
        if e > self.start_epoch and (e + 1 - self.start_epoch) % self.save_frequency == 0:
            self.save()

    def judgeIfMosaic(self, e):
        if e == self.max_epoch - self.lr_schedule.no_aug_epoch:
            self.dataset.ifMosaic = False

    def others(self, e, iter, loss_value):
        self.logger.record(loss_value / iter)
        self.pbar.close()
        self.lr_schedule.step(e)

    def trainPred(self, start_time):
        loss_value = 0
        for iter, (images, targets) in enumerate(self.data_loader, 1):
            preds = self.getPreds(images)

            loss = self.doBackward(preds, targets)

            loss_value = self.calLoss(loss, loss_value)

            self.barUpdate(iter, loss_value, start_time)
        return iter, loss_value

    def barUpdate(self, iter, loss_value, start_time):
        cost_time = time_delta(start_time)
        self.pbar.set_postfix_str(
            f'loss: {loss_value / iter:.4f}, lr: {get_lr(self.optimizer):.5f}, time: {cost_time}\33[0m')
        self.pbar.update()

    def calLoss(self, loss, loss_value):
        loss_value += loss.item()
        return loss_value

    def doBackward(self, preds, targets):
        self.optimizer.zero_grad()
        loss = self.lossAccumulate(preds, targets)
        loss.backward()
        self.optimizer.step()
        return loss

    def lossAccumulate(self, preds, targets):
        loss = 0
        for i, pred in enumerate(preds):
            loss += self.criterion(i, pred, targets)
        return loss

    def getPreds(self, images):
        preds = self.model(images.to(self.device))
        return preds

    def createBar(self, bar_format, e):
        self.pbar = tqdm(total=self.n_batches, bar_format=bar_format)
        self.pbar.set_description(f"\33[36m💫 Epoch {(e + 1):5d}/{self.max_epoch}")
        start_time = datetime.now()
        return start_time

    def freezeMethod(self, e, is_unfreezed):
        if self.freeze and e >= self.free_epoch and not is_unfreezed:
            print('\nFreeze begins: \n')
            self.setLrDl()
            self.calBatchSize()
            self.model.backbone.set_freezed(False)

    def setLrDl(self):
        self.data_loader = dataLoader(self.dataset, self.batch_size, self.num_worksers)
        self.lr_schedule.set_lr(*determin_lr(self.lr, self.batch_size))

    def calBatchSize(self):
        self.n_batches = len(self.dataset) // self.batch_size

    def set_bar_format(self):
        bar_format = '{desc}{n_fmt:>4s}/{total_fmt:<4s}|{bar}|{postfix}'
        return bar_format

    def createDir(self, t):
        self.save_dir = self.save_dir / t
        self.logger.save_dir = self.logger.save_dir / t

    def getTime(self):
        t = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
        return t
