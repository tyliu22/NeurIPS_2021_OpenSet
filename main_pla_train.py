'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import numpy as np

# from models import *
from models.dla_part import DLA6
from utils import progress_bar
from Utils.MyDataLoader import subDataset
import torch.utils.data.dataloader as DataLoader



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

train_dataset_data, train_dataset_label = train_dataset.data, train_dataset.targets
test_dataset_data, test_dataset_label = test_dataset.data, test_dataset.targets
label_class = np.array(list(train_dataset.class_to_idx.values()))
# select 6 classes as training dataset, 4 dataset as testing dataset
np.random.shuffle(label_class)
selected_class, unselected_class= label_class[0:6], label_class[6:10]
print(' CIFAR10 training class:', selected_class)
print('CIFAR10 testing  class:', unselected_class)

selected_train_dataset_label, selected_train_dataset_data = np.empty(shape=[0]), np.empty(shape=[0,32,32,3])
selected_test_dataset_label, selected_test_dataset_data = np.empty(shape=[0]), np.empty(shape=[0,32,32,3])
for i in selected_class:
    selected_train_dataset_data = np.append(selected_train_dataset_data,
                                             train_dataset_data[np.where(train_dataset_label==i)], axis=0)
    selected_train_dataset_label = np.append(selected_train_dataset_label,
                                             np.array(train_dataset_label)[np.where(train_dataset_label==i)])
    selected_test_dataset_data = np.append(selected_test_dataset_data,
                                             test_dataset_data[np.where(test_dataset_label==i)], axis=0)
    selected_test_dataset_label = np.append(selected_test_dataset_label,
                                             np.array(test_dataset_label)[np.where(test_dataset_label==i)])
# selected_train_dataset_label = selected_train_dataset_label.astype(int)
# selected_test_dataset_label = selected_test_dataset_label.astype(int)

# rename label as [0 ,..., 5]
num_class = int(10)
for i in selected_class:
    selected_train_dataset_label[np.where(selected_train_dataset_label==i)] = num_class
    selected_test_dataset_label[np.where(selected_test_dataset_label == i)] = num_class
    num_class = num_class+1
selected_train_dataset_label = (selected_train_dataset_label - 10).astype(int)
selected_test_dataset_label = (selected_test_dataset_label - 10).astype(int)

# numpy dimension [10000, 3, 32, 32]
CIFAR10_train_data, CIFAR10_train_label = selected_train_dataset_data.transpose(0,3,1,2),\
                                      selected_train_dataset_label
CIFAR10_test_data, CIFAR10_test_label = selected_test_dataset_data.transpose(0,3,1,2),\
                                      selected_test_dataset_label

train_dataset = subDataset(CIFAR10_train_data, CIFAR10_train_label)
test_dataset = subDataset(CIFAR10_test_data, CIFAR10_test_label)
train_dataloader = DataLoader.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_dataloader = DataLoader.DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=2)

# trainset = torchvision.datasets.CIFAR10(
#     root='./data', train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(
#     trainset, batch_size=128, shuffle=True, num_workers=2)
#
# testset = torchvision.datasets.CIFAR10(
#     root='./data', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(
#     testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = DLA6()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt6.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        inputs, targets = inputs.type(torch.FloatTensor).to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, _ = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        # print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        progress_bar(batch_idx, len(train_dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dataloader):
            inputs, targets = inputs.type(torch.FloatTensor).to(device), targets.to(device)
            outputs, _ = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            progress_bar(batch_idx, len(test_dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt6.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+150):
    train(epoch)
    test(epoch)
scheduler.step()