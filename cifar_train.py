import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision
import argparse
import utils
from model import Network
from torchsummary import summary
from tqdm import tqdm
import numpy as np
from utils import _data_transforms_cifar10


def get_args():
    parser = argparse.ArgumentParser("Single_Path_One_Shot")
    parser.add_argument('--exp_name', type=str, default='spos_cifar', required=True, help='experiment name')
    parser.add_argument('--data_dir', type=str, default='/home/work/dataset/cifar', help='path to the dataset')
    parser.add_argument('--batch_size', type=int, default=96, help='batch size')
    parser.add_argument('--epochs', type=int, default=600, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight-decay', type=float, default=4e-5, help='weight decay')
    parser.add_argument('--train_interval', type=int, default=1000, help='print train loss frequency')
    parser.add_argument('--val_interval', type=int, default=5, help='validate and save frequency')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    if torch.cuda.is_available():
        print('Train on GPU!')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    train_transform, valid_transform = _data_transforms_cifar10(args)
    trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True,
                                            download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                               shuffle=True, pin_memory=True, num_workers=4)
    valset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False,
                                          download=True, transform=valid_transform)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                             shuffle=False, pin_memory=True, num_workers=4)

    model = Network(classes=10, gap_size=1).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1 - (epoch / args.epochs))
    summary(model, (3, 32, 32))
    print('Start training!')
    for epoch in range(args.epochs):
        print('epoch:%d, lr:%f' % (epoch, scheduler.get_lr()[0]))
        train(args, epoch, train_loader, device, model, criterion, optimizer, scheduler)
        scheduler.step()
        if (epoch + 1) % args.val_interval == 0:
            validate(args, epoch, val_loader, device, model)
            utils.save_checkpoint({'state_dict': model.state_dict(), }, epoch + 1, tag=args.exp_name)


def train(args, epoch, train_data, device, model, criterion, optimizer, scheduler):
    _loss = 0.0
    for step, (inputs, targets) in enumerate(tqdm(train_data)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        random = np.random.randint(4, size=20)
        outputs = model(inputs, random)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        _loss += loss.item()
        if (epoch + 1) % args.train_interval == 0:
            print('[epoch:%d, lr:%f] loss: %f' % (epoch + 1, scheduler.get_lr()[0], _loss / (step + 1)))


def validate(args, epoch, val_data, device, model):
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(val_data):
            inputs = inputs.to(device)
            targets = targets.to(device)
            random = np.random.randint(4, size=20)
            outputs = model(inputs, random)
            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            n = inputs.size(0)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
        print('[Val_Accuracy] epoch:%d, top1:%.5f%%, top5:%.5f%% ' % (epoch + 1, top1.avg, top5.avg))


if __name__ == '__main__':
    main()
