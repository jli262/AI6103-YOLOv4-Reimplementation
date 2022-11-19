from torch.optim import Optimizer
from math import pi, cos


class WarmUpCosLRSchedule:
    self.lr = lr
    self.min_lr = min_lr
    self.optimizer = optimizer
    self.warm_up_factor = warm_up_factor
    self.total_epoch = total_epoch
    self.warm_up_epoch = int(warm_up_ratio*total_epoch)
    self.no_aug_epoch = int(no_aug_ratio*total_epoch)

    def step(self, epoch: int):
        if epoch < self.warm_up_epoch:
            delta = (1 - self.warm_up_factor) * epoch / self.warm_up_epoch
            lr = (self.warm_up_factor + delta) * self.lr
        elif epoch >= self.total_epoch-self.no_aug_epoch:
            lr = self.min_lr
        else:
            cos_iters = self.total_epoch - self.warm_up_factor - self.no_aug_epoch
            lr = self.min_lr + 0.5 * (self.lr - self.min_lr) * (
                1 + cos(pi * (epoch - self.warm_up_epoch) / cos_iters)
            )

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def set_lr(self, lr, min_lr):
        self.lr = lr
        self.min_lr = min_lr


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def determin_lr(lr, batch_size):
    bs = 64
    lr_max = 5e-2
    lr_min = 5e-4
    lr_fit = min(max(batch_size/bs*lr, lr_min), lr_max)
    lr_min_fit = min(max(batch_size/bs*(lr/100), lr_min/100), lr_max/100)
    return lr_fit, lr_min_fit
