import os
import sys
import random
import numpy as np
import torch
import utils
import argparse
import torch.utils
import torch.nn.functional as F

from micro_child import CNN
from micro_controller import Controller

import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler


parser = argparse.ArgumentParser("cifar10")
parser.add_argument('--data_path', type=str, default='./data/cifar10', help='db path')
parser.add_argument('--batch_size', type=int, default=160, help='batch size')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=10, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=150, help='num of training epochs')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')

parser.add_argument('--child_lr_max', type=float, default=0.05)
parser.add_argument('--child_lr_min', type=float, default=0.0005)
parser.add_argument('--child_lr_T_0', type=int, default=10)
parser.add_argument('--child_lr_T_mul', type=int, default=2)
parser.add_argument('--child_num_layers', type=int, default=6)
parser.add_argument('--child_out_filters', type=int, default=20)
parser.add_argument('--child_num_branches', type=int, default=5)
parser.add_argument('--child_num_cells', type=int, default=5)
parser.add_argument('--child_use_aux_heads', type=bool, default=False)

parser.add_argument('--controller_lr', type=float, default=0.0035)
parser.add_argument('--controller_tanh_constant', type=float, default=1.10)
parser.add_argument('--controller_op_tanh_reduce', type=float, default=2.5)

parser.add_argument('--lstm_size', type=int, default=64)
parser.add_argument('--lstm_num_layers', type=int, default=1)
parser.add_argument('--lstm_keep_prob', type=float, default=0)
parser.add_argument('--temperature', type=float, default=5.0)

parser.add_argument('--entropy_weight', type=float, default=0.0001)
parser.add_argument('--bl_dec', type=float, default=0.99)

args = parser.parse_args()

CIFAR_CLASSES = 10

baseline = None
epoch = 0

print('==> Preparing data..')
transforms_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transforms_valid = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

dataset_train = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transforms_train)
dataset_valid = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transforms_valid)
dataset_test = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transforms_test)

# split train dataset into train and valid. After that, make sampler for each dataset
# code from https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
num_train = len(dataset_train)
indices = list(range(num_train))
split = int(np.floor(0.1 * num_train))
train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, num_workers=4, sampler=train_sampler, pin_memory=True)
valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, num_workers=4, sampler=valid_sampler, pin_memory=True)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)


def main():
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    print("args = %s", args)

    model = CNN(args)
    model.cuda()

    controller = Controller()
    controller.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.child_lr_max,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    controller_optimizer = torch.optim.Adam(
        controller.parameters(),
        args.controller_lr,
        betas=(0.1, 0.999),
        eps=1e-3,
    )

    scheduler = utils.LRScheduler(optimizer, args)

    for epoch in range(args.epochs):
        lr = scheduler.update(epoch)
        print('epoch', epoch, lr)

        # training
        train_acc = train(model, controller, optimizer)
        print('train_acc', train_acc)
        train_controller(model, controller, controller_optimizer)

        # validation
        valid_acc = infer(model, controller)
        print('valid_acc', valid_acc)
        torch.save(model.state_dict(), os.path.join(args.save, 'weights.pth'))


def train(model, controller, optimizer):
    total_loss = utils.AvgMeter()
    total_top1 = utils.AvgMeter()

    for step, (data, target) in enumerate(train_loader):
        model.train()
        n = data.size(0)

        data = data.cuda()
        target = target.cuda()

        optimizer.zero_grad()

        controller.eval()
        dag, _, _ = controller()

        logits, _ = model(data, dag)
        loss = F.cross_entropy(logits, target)

        loss.backward()
        optimizer.step()

        prec1 = utils.accuracy(logits, target)[0]
        total_loss.update(loss.item(), n)
        total_top1.update(prec1.item(), n)

        if step % args.report_freq == 0:
            print('train step=', step,
                  'total_loss=', round(total_loss.avg, 4),
                  'acc=', round(total_top1.avg, 4))

    return total_top1.avg


def train_controller(model, controller, controller_optimizer):
    global baseline
    total_loss = utils.AvgMeter()
    total_reward = utils.AvgMeter()
    total_entropy = utils.AvgMeter()

    valid_iterator = iter(valid_loader)
    for step in range(300):
        try:
            data, targets = next(valid_iterator)
        except StopIteration:
            valid_iterator = iter(valid_loader)
            data, targets = next(valid_iterator)

        model.eval()
        n = data.size(0)

        data = data.cuda()
        target = target.cuda()

        controller_optimizer.zero_grad()

        controller.train()
        dag, log_prob, entropy = controller()

        with torch.no_grad():
            logits, _ = model(data, dag)
            reward = utils.accuracy(logits, target)[0]

        if args.entropy_weight is not None:
            reward += args.entropy_weight*entropy

        log_prob = torch.sum(log_prob)
        if baseline is None:
            baseline = reward
        baseline -= (1 - args.bl_dec) * (baseline - reward)

        loss = log_prob * (reward - baseline)
        loss = loss.sum()

        loss.backward()

        controller_optimizer.step()

        total_loss.update(loss.item(), n)
        total_reward.update(reward.item(), n)
        total_entropy.update(entropy.item(), n)

        if step % args.report_freq == 0:
            print('controller step=', step,
                  'total_loss=', round(total_loss.avg, 4),
                  'total_reward=', round(total_reward.avg, 4),
                  'baseline=', baseline.item())


def infer(model, controller):
    total_loss = utils.AvgMeter()
    total_top1 = utils.AvgMeter()
    model.eval()
    controller.eval()

    with torch.no_grad():
        test_iterator = iter(test_loader)
        for step in range(300):
            try:
                data, targets = next(test_iterator)
            except StopIteration:
                test_iterator = iter(test_loader)
                data, targets = next(test_iterator)

            data = data.cuda()
            target = target.cuda()

            dag, _, _ = controller()

            logits, _ = model(data, dag)
            loss = F.cross_entropy(logits, target)

            prec1 = utils.accuracy(logits, target)[0]
            n = data.size(0)
            total_loss.update(loss.item(), n)
            total_top1.update(prec1.item(), n)

            print('valid step=', step,
                  'loss=', loss.item(),
                  'acc=', prec1.item())
            print('normal cell', str(dag[0]))
            print('reduce cell', str(dag[1]))

    return total_top1.avg


if __name__ == '__main__':
    main() 

