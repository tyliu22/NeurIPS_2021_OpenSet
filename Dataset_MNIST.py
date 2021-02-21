#!/usr/bin/env python
# coding: utf-8
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data.dataloader as DataLoader
import matplotlib.pyplot as plt

from Utils.AUROC_Score_single import AUROC_score
from Utils.MyDataLoader import subDataset


r_seed = 0
torch.manual_seed(r_seed)
np.random.seed(r_seed)

transform = transforms.Compose([
    transforms.ToTensor()
])

# # Preparing Data   MNIST
train_dataset = datasets.MNIST('../data', train=True,
                               download=True, transform=transform)
test_dataset = datasets.MNIST('../data', train=False,
                              transform=transform)

train_dataset_data = train_dataset.data.numpy()
train_dataset_label = train_dataset.targets.numpy()
label_class = np.array(list(train_dataset.class_to_idx.values()))
# select 6 classes as training dataset, 4 dataset as testing dataset
np.random.shuffle(label_class)
selected_class = label_class[0:6]
unselected_class = label_class[6:10]
print('MNIST training class:', selected_class)
print('MNIST testing  class:', unselected_class)

selected_train_dataset_label = np.empty(shape=[0])
selected_train_dataset_data = np.empty(shape=[0,28,28])
for i in selected_class:
    selected_train_dataset_data = np.append(selected_train_dataset_data,
                                             train_dataset_data[np.where(train_dataset_label==i)], axis=0)
    selected_train_dataset_label = np.append(selected_train_dataset_label,
                                             train_dataset_label[np.where(train_dataset_label==i)])
num_class = int(10)
for i in selected_class:
    selected_train_dataset_label[np.where(selected_train_dataset_label==i)] = num_class
    num_class=num_class+1
selected_train_dataset_label = (selected_train_dataset_label - 10).astype(int)

unselected_train_dataset_label = np.empty(shape=[0])
unselected_train_dataset_data = np.empty(shape=[0,28,28])
for i in unselected_class:
    unselected_train_dataset_data = np.append(unselected_train_dataset_data,
                                             train_dataset_data[np.where(train_dataset_label==i)], axis=0)
    # unselected_train_dataset_label = np.append(unselected_train_dataset_label,
    #                                          train_dataset_label[np.where(train_dataset_label==i)])

mnist_train_data, mnist_train_label = selected_train_dataset_data[:, np.newaxis,:,:],\
                                      selected_train_dataset_label
mnist_test_data, mnist_test_label = unselected_train_dataset_data[:, np.newaxis,:,:],\
                                    unselected_train_dataset_label

train_dataset = subDataset(mnist_train_data, mnist_train_label)
test_dataset = subDataset(mnist_test_data, mnist_test_label)

train_dataloader = DataLoader.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 100, 3, 1)
        self.conv2 = nn.Conv2d(100, 100, 3, 1)
        self.conv3 = nn.Conv2d(100, 100, 3, 1)
        self.conv4 = nn.Conv2d(100, 100, 3, 1)
        self.conv5 = nn.Conv2d(100, 100, 3, 1)
        self.fc1 = nn.Linear(3*3*100, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 6)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 3*3*100)
        x = self.fc1(x)
        x_hidden = self.fc2(x)
        output = F.softmax(self.fc3(x_hidden), dim=1)
        # output = F.softmax(x, dim=1)
        return output, x_hidden

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 100, 5, 1)
#         self.conv2 = nn.Conv2d(100, 100, 5, 1)
#         self.conv3 = nn.Conv2d(100, 100, 5, 1)
#         self.conv4 = nn.Conv2d(100, 100, 5, 1)
#         self.fc1 = nn.Linear(3*3*100, 300)
#         self.fc2 = nn.Linear(300, 100)
#         self.fc3 = nn.Linear(100, 6)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv3(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = x.view(-1, 3*3*100)
#         x = self.fc1(x)
#         x_hidden = self.fc2(x)
#         output = F.softmax(self.fc3(x_hidden), dim=1)
#         # output = F.softmax(x, dim=1)
#         return output, x_hidden

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 100, 7, 1)
#         self.conv2 = nn.Conv2d(100, 100, 7, 1)
#         self.conv3 = nn.Conv2d(100, 100, 7, 1)
#         self.fc1 = nn.Linear(2*2*100, 100)
#         self.fc2 = nn.Linear(100, 10)
#         self.fc3 = nn.Linear(10, 6)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv3(x))
#         # x = F.max_pool2d(x, 2, 2)
#         x = x.view(-1, 2*2*100)
#         x = self.fc1(x)
#         x_hidden = self.fc2(x)
#         output = F.softmax(self.fc3(x_hidden), dim=1)
#         # output = F.softmax(x, dim=1)
#         return output, x_hidden

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # [128, 1, 28, 28]
        correct = 0
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, _ = model(data.float())
        loss = F.nll_loss(torch.log(output), target.long())
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct = pred.eq(target.view_as(pred)).sum().item()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy:{:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), correct/len(data)))

# plt.imshow(mnist_train_data[1].numpy().reshape(28,28), cmap='gray')
# plt.show()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
model = Net().to(device)
# if device == 'cuda':
#     model = torch.nn.DataParallel(model)
#     cudnn.benchmark = True

# model traiing
train_epoch = 20
optimizer = optim.Adam(model.parameters(), lr=1e-4)
for epoch in range(1, train_epoch):
    train(model, device, train_dataloader, optimizer, epoch)
    # test(model, device, test_loader)


mnist_train_data = torch.tensor(mnist_train_data)
mnist_test_data = torch.tensor(mnist_test_data)
# print('mnist_test Dataset shape:', mnist_test_data.shape[0])

# plot figure
# plt.imshow(mnist_train_data[1].numpy().reshape(28,28), cmap='gray')
# plt.show()

with torch.no_grad():
    result_mnist_train_last_layer, result_mnist_train_hidden = model(mnist_train_data.float().to(device))
    result_mnist_test_last_layer, result_mnist_test_hidden = model(mnist_test_data.float().to(device))

# *********************** AUROC_score ************************* #
num_train_sample = mnist_train_data.shape[0]
num_test_sample = mnist_test_data.shape[0]

print('===> AUROC_score start')
# ******************* Outlier Detection ********************** #
# def AUROC_score(train_data_last_layer, train_data_hidden, num_train_sample,
#                 test_data_last_layer, test_data_hidden, num_test_sample,
#                 r_seed=0, n_estimators=1000, verbose=0,
#                 max_samples=10000, contamination=0.01):

for i in range(2,11,1):
    AUROC_score(result_mnist_train_last_layer, result_mnist_train_hidden, num_train_sample,
                result_mnist_test_last_layer, result_mnist_test_hidden, num_test_sample,
                r_seed=0, n_estimators=1000, verbose=0, max_samples=10000, contamination=0.01*i)

print('Algorithm End')
