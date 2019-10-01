import torch
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torchvision
import argparse
from torch.autograd import Variable
from model import Network
from torchsummary import summary


def get_args():
    parser = argparse.ArgumentParser("Single_Path_One_Shot")
    parser.add_argument('--train_dir', type=str, default='/home/lushun/Documents/Dataset/ImageNet/train',
                        help='path to training dataset')
    parser.add_argument('--val_dir', type=str, default='/home/lushun/Documents/Dataset/ImageNet/val',
                        help='path to validation dataset')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--epochs', type=int, default=1000, help='batch size')
    parser.add_argument('--total-iters', type=int, default=1200000, help='total iters')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight-decay', type=float, default=4e-5, help='weight decay')
    parser.add_argument('--train_interval', type=int, default=1000, help='save frequency')
    parser.add_argument('--val_interval', type=int, default=10, help='save frequency')
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=0)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=0)

    model = Network()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay)

    if torch.cuda.is_available():
        print('Train on GPU!')
        criterion = criterion.cuda()
        device = torch.device("cuda")
    else:
        criterion = criterion
        device = torch.device("cpu")
    model = model.to(device)
    summary(model, (3, 32, 32))
    print('Start training!')
    for epoch in range(args.epochs):
        train(args, epoch, train_loader, device, model, criterion, optimizer)
        if (epoch + 1) % args.val_interval == 0:
            validate(epoch, val_loader, device, model )


def train(args, epoch, train_data, device, model, criterion, optimizer):
    _loss = 0.0
    for step, (inputs, labels) in enumerate(train_data):
        inputs, labels = Variable(inputs), Variable(labels)
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs, random = np.random.randint(4, size=20))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _loss += loss.item()
        # if step % args.train_interval == (args.train_interval-1):
    print('[%d, %5d] loss: %.3f' % (epoch, step, _loss / (step+1)))


def validate(epoch, val_data, device, model):
        # validate
        correct, total = 0, 0
        with torch.no_grad():
            # for inputs, labels in val_data:
            for step, (inputs, targets) in enumerate(val_data):
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs, random = np.random.randint(4, size=20))
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (targets == predicted).sum().item()
                # print(step)
        accuracy = 100 * correct / total
        print('Accuracy : %d %%' % accuracy)
        # save model
        save_dir = './snapshots/epoch_' + str(epoch) + '_accuracy_' + str(accuracy) + '-weights.pt'
        torch.save(model.state_dict(), save_dir)
        print('Successfully save the model.')


if __name__ == '__main__':
    main()
