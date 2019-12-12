import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.datasets as data_sets
import argparse
import utils
from torchsummary import summary
from model import Network
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser("Single_Path_One_Shot")
    parser.add_argument('--exp_name', type=str, default='spos_cifar', required=True, help='experiment name')
    parser.add_argument('--train_dir', type=str, default='', help='path to training dataset')
    parser.add_argument('--val_dir', type=str, default='', help='path to validation dataset')
    parser.add_argument('--train_batch', type=int, default=64, help='batch size')
    parser.add_argument('--val_batch', type=int, default=2048, help='batch size')
    parser.add_argument('--epochs', type=int, default=120, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight-decay', type=float, default=4e-5, help='weight decay')
    parser.add_argument('--train_interval', type=int, default=1000, help='save frequency')
    parser.add_argument('--val_interval', type=int, default=10, help='save frequency')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_data_set = data_sets.ImageFolder(
        args.train_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    val_data_set = data_sets.ImageFolder(args.val_dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    train_data = torch.utils.data.DataLoader(
        train_data_set, batch_size=args.train_batch, shuffle=True,
        num_workers=4, pin_memory=True, sampler=None)
    val_data = torch.utils.data.DataLoader(
        val_data_set, batch_size=args.val_batch, shuffle=False,
        num_workers=4, pin_memory=True)
    print('num train_data:', len(train_data))
    print('num val_data:', len(val_data))

    model = Network()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1-(epoch / args.epochs))

    if torch.cuda.is_available():
        print('Train on GPU!')
        criterion = criterion.cuda()
        device = torch.device("cuda")
    else:
        criterion = criterion
        device = torch.device("cpu")
    model = model.to(device)
    summary(model, (3, 224, 224))

    print('Start training!')
    for epoch in range(args.epochs):
        lr = scheduler.get_lr()
        print('epoch:%d, lr:%f' % (epoch, scheduler.get_lr()[0]))
        train(args, epoch, train_data, device, model, criterion, optimizer)
        scheduler.step()
        if (epoch + 1) % args.val_interval == 0:
            validate(args, epoch, val_data, device, model)
            utils.save_checkpoint({'state_dict': model.state_dict(), }, epoch + 1, tag=args.exp_name)


def train(args, epoch, train_data, device, model, criterion, optimizer):
    # _loss for the loss of args.train_interval
    _loss = 0.0
    # _epoch_loss for the loss of the whole epoch
    _epoch_loss = 0.0
    for step, (inputs, labels) in enumerate(tqdm(train_data)):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _loss += loss.item()
        _epoch_loss += loss.item()
        if (step+1) % args.train_interval == 0:
            print('[epoch:%d, step:%d] loss: %f' % (epoch, step + 1, _loss / args.train_interval))
            _loss = 0.0
    print('[epoch:%d, step:%d] loss: %f' % (epoch, step + 1, _epoch_loss / (step+1)))


def validate(args, epoch, val_data, device, model):
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(tqdm(val_data)):
            inputs = inputs.to(device)
            labels = targets.to(device)
            outputs = model(inputs)
            prec1, prec5 = utils.accuracy(outputs, labels, topk=(1, 5))
            n = inputs.size(0)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
        print('[Val_Accuracy] top1:%.5f%%, top5:%.5f%% ' % (top1.avg, top5.avg))


if __name__ == '__main__':
    main()
