from torch.optim import Optimizer
from math import pi, cos


class WarmUpCosLRSchedule:
    def __init__(self, optimizer: Optimizer, lr: float, min_lr: float, total_epoch: int, warm_up_ratio=0.05, no_aug_ratio=0.05, warm_up_factor=1/3):
        self.selfSet(lr, min_lr, optimizer)
        self.total_epoch = total_epoch
        self.warm_up_factor = warm_up_factor
        self.selfEpoch(no_aug_ratio, total_epoch, warm_up_ratio)

    def selfEpoch(self, no_aug_ratio, total_epoch, warm_up_ratio):
        self.warm_up_epoch = int(warm_up_ratio * total_epoch)
        self.no_aug_epoch = int(no_aug_ratio * total_epoch)

    def calcCositer(self):
        cos_iters = self.total_epoch - self.warm_up_factor - self.no_aug_epoch
        return cos_iters

    def selfSet(self, lr, min_lr, optimizer):
        self.lr = lr
        self.min_lr = min_lr
        self.optimizer = optimizer

    def step(self, epoch: int):
        if epoch < self.warm_up_epoch:
            lr = self.setSmall(epoch)
        elif epoch >= self.total_epoch-self.no_aug_epoch:
            lr = self.setLarge()
        else:
            product = (1 + cos(pi * (epoch - self.warm_up_epoch) / self.calcCositer()))
            lr = self.setElse(product)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def setLarge(self):
        lr = self.min_lr
        return lr

    def setSmall(self, epoch):
        delta = (1 - self.warm_up_factor) * epoch / self.warm_up_epoch
        lr = (self.warm_up_factor + delta) * self.lr
        return lr

    def set_lr(self, lr, min_lr):
        self.lr = lr
        self.min_lr = min_lr

    def setElse(self, product):
        lr = self.min_lr + 0.5 * (self.lr - self.min_lr) * product
        return lr

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def determin_lr(lr, batch_size):
    lr_fit, lr_min_fit = lrFIt(batch_size, 64, lr, 5e-2, 5e-4)
    return lr_fit, lr_min_fit


def lrFIt(batch_size, bs, lr, lr_max, lr_min):
    minfit, fitt = maxNum(batch_size, bs, lr, lr_min)
    lr_fit = min(fitt, lr_max)
    lr_min_fit = min(minfit, lr_max / 100)
    return lr_fit, lr_min_fit


def maxNum(batch_size, bs, lr, lr_min):
    asap = max(batch_size / bs * (lr / 100), lr_min / 100)
    bsbp = max(batch_size / bs * lr, lr_min)
    return asap, bsbp
