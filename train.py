import torch
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torchvision.datasets as data_sets
import argparse

from model import Network


def get_args():
    parser = argparse.ArgumentParser("Single_Path_One_Shot")
    parser.add_argument('--train_dir', type=str, default='/Dataset/ImageNet/train',
                        help='path to training dataset')
    parser.add_argument('--val_dir', type=str, default='/Dataset/ImageNet/val',
                        help='path to validation dataset')

    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--epochs', type=int, default=240, help='batch size')
    parser.add_argument('--total-iters', type=int, default=1200000, help='total iters')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight-decay', type=float, default=4e-5, help='weight decay')
    parser.add_argument('--train_interval', type=int, default=1000, help='save frequency')
    parser.add_argument('--val_interval', type=int, default=5, help='save frequency')

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    if torch.cuda.is_available():
        gpu = True
    else:
        gpu = False

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
        train_data_set, batch_size=args.batch_size, shuffle=True,
        num_workers=1, pin_memory=True, sampler=None)
    val_data = torch.utils.data.DataLoader(
        val_data_set, batch_size=args.batch_size, shuffle=False,
        num_workers=1, pin_memory=True)
    # print(len(train_data))
    # print(len(val_data))

    model = Network()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                    lambda step : (1.0-step/args.total_iters) if step <= args.total_iters else 0, last_epoch=-1)

    if gpu:
        criterion = criterion.cuda()
        device = torch.device("cuda")
    else:
        criterion = criterion
        device = torch.device("cpu")
    model = model.to(device)

    print('Start training!')
    for epoch in range(args.epochs):
        train(args, epoch, train_data, device, model, criterion, optimizer, scheduler)
        if (epoch + 1) % args.val_interval == 0:
            validate(epoch, val_data, device, model )


def train(args, epoch, train_data, device, model, criterion, optimizer, scheduler):
    _loss = 0.0
    for step, (inputs, targets) in enumerate(train_data):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs, random=np.random.randint(4, size=20))
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        _loss += loss.item()
        if step % args.train_interval == (args.train_interval-1):
            print('[%d, %5d] loss: %.3f' % (epoch, step+1, _loss / args.train_interval))
            _loss = 0.0


def validate(epoch, val_data, device, model):
        # validate
        correct, total = 0, 0
        with torch.no_grad():
            # for inputs, labels in val_data:
            for step, (inputs, targets) in enumerate(val_data):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs, random=np.random.randint(4, size=20))
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (targets == predicted).sum().item()
                print(step)
        accuracy = 100 * correct / total
        print('Accuracy : %d %%' % accuracy)
        # save model
        save_dir = './snapshots/' + str(epoch) + '-' + str(accuracy) + '-weights.pt'
        torch.save(model.state_dict(), save_dir)
        print('Successfully save the model.')


if __name__ == '__main__':
    main()
