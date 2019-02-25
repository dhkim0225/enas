import math


class AvgMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class LRScheduler:
    def __init__(self, optimizer, args):
        self.last_lr_reset = 0
        self.lr_T_0 = args.child_lr_T_0
        self.child_lr_T_mul = args.child_lr_T_mul
        self.child_lr_min = args.child_lr_min
        self.child_lr_max = args.child_lr_max
        self.optimizer = optimizer

    def update(self, epoch):
        T_curr = epoch - self.last_lr_reset
        if T_curr == self.lr_T_0:
            self.last_lr_reset = epoch
            self.lr_T_0 = self.lr_T_0 * self.child_lr_T_mul
        rate = T_curr / self.lr_T_0 * math.pi
        lr = self.child_lr_min + 0.5 * (self.child_lr_max - self.child_lr_min) * (1.0 + math.cos(rate))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res
