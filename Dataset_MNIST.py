#!/usr/bin/env python
# coding: utf-8
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch.utils.data.dataloader as DataLoader

from sklearn.ensemble import IsolationForest

from Utils.OutlierDetection import OutlierDetection
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


unselected_train_dataset_label = np.empty(shape=[0,0])
unselected_train_dataset_data = np.empty(shape=[0,28,28])
for i in unselected_class:
    unselected_train_dataset_data = np.append(unselected_train_dataset_data,
                                             train_dataset_data[np.where(train_dataset_label==i)], axis=0)
    unselected_train_dataset_label = np.append(unselected_train_dataset_label,
                                             train_dataset_label[np.where(train_dataset_label==i)])


mnist_train_data, mnist_train_label = selected_train_dataset_data[:, np.newaxis,:,:],\
                                      np.array(selected_train_dataset_label, dtype=float)
mnist_test_data, mnist_test_label = unselected_train_dataset_data[:, np.newaxis,:,:],\
                                    unselected_train_dataset_label

train_dataset = subDataset(mnist_train_data, mnist_train_label)
test_dataset = subDataset(mnist_test_data, mnist_test_label)

train_dataloader = DataLoader.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 100, 5, 1)
#         self.conv2 = nn.Conv2d(100, 100, 5, 1)
#         self.conv3 = nn.Conv2d(100, 100, 5, 1)
#         self.conv4 = nn.Conv2d(100, 100, 5, 1)
#         self.conv5 = nn.Conv2d(100, 50, 5, 1)
#         self.fc1 = nn.Linear(3*3*50, 300)
#         self.fc2 = nn.Linear(300, 100)
#         self.fc3 = nn.Linear(100, 10)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv4(x))
#         x = F.relu(self.conv5(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = x.view(-1, 3*3*50)
#         x1 = self.fc1(x)
#         x_hidden = self.fc2(x1)
#         output = F.softmax(self.fc3(x_hidden), dim=1)
#         # output = self.fc2(x)
#         # log_softmax
#         # output = F.softmax(x, dim=1)
#         return output, x_hiddennp.random.shuffle(label_class)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 100, 5, 1)
        self.conv2 = nn.Conv2d(100, 100, 5, 1)
        self.conv3 = nn.Conv2d(100, 100, 5, 1)
        self.conv4 = nn.Conv2d(100, 100, 5, 1)
        self.fc1 = nn.Linear(3*3*100, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 6)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 3*3*100)
        x = self.fc1(x)
        x_hidden = self.fc2(x)
        output = F.softmax(self.fc3(x_hidden), dim=1)
        # output = F.softmax(x, dim=1)
        return output, x_hidden

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # [128, 1, 28, 28]
        correct = 0
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, _ = model(data)
        loss = F.nll_loss(torch.log(output), target)
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct = pred.eq(target.view_as(pred)).sum().item()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy:{:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), correct/len(data)))


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
model = Net().to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

# model traiing
train_epoch = 7
optimizer = optim.Adam(model.parameters(), lr=5e-4)
for epoch in range(1, train_epoch):
    train(model, device, train_dataloader, optimizer, epoch)
    # test(model, device, test_loader)



# result_mnist_train_last_layer = []
# result_mnist_train_hidden = []
# with torch.no_grad():
#     for batch_idx, (data, target) in enumerate(train_dataloader):
#         # [128, 1, 28, 28]
#         data, target = data.to(device), target.to(device)
#         output_train, hidden_train = model(data)
#         result_mnist_train_last_layer.append(output_train)
#         result_mnist_train_hidden.append(hidden_train)
#
# result_mnist_train_last_layer = torch.cat(result_mnist_train_last_layer, dim=0)
# result_mnist_train_hidden = torch.cat(result_mnist_train_hidden, dim=0)

mnist_train_data = torch.tensor(mnist_train_data)
mnist_test_data = torch.tensor(mnist_test_data)
print('mnist_test Dataset shape:', mnist_test_data.shape[0])

# plot figure
# plt.imshow(x_noise_test[1].numpy().reshape(28,28), cmap='gray')
# plt.show()

with torch.no_grad():
    result_mnist_train_last_layer, result_mnist_train_hidden = model(mnist_train_data.to(device))
    result_mnist_test_last_layer, result_mnist_test_hidden = model(mnist_test_data.to(device))



# ******************* Outlier Detection ********************** #
sample_size = mnist_test_data.shape
outlier_detector_l1 = IsolationForest(random_state=r_seed, n_estimators=1000, verbose=0, max_samples=10000,
                                      contamination=0.05)
outlier_detector_l2 = IsolationForest(random_state=r_seed, n_estimators=1000, verbose=0, max_samples=10000,
                                      contamination=0.05)

result_mnist_train = result_cifar10_train.cpu().numpy()
result_mnist_train_base = result_cifar10_train_base.cpu().numpy()
# data argument
outlier_detector_l1.fit(result_cifar10_train)
outlier_detector_l2.fit(result_cifar10_train_base)


# **************** Tensor2numpy **************** #
result_Imagenet_crop_test = result_Imagenet_crop_test.cpu().numpy()
result_Imagenet_crop_test_base = result_Imagenet_crop_test_base.cpu().numpy()


# **************** outlier predict **************** #
outlier_cifar10_train = outlier_detector_l1.predict(result_cifar10_train)
outlier_cifar10_train += outlier_detector_l2.predict(result_cifar10_train_base)

outlier_Imagenet_crop = outlier_detector_l1.predict(result_Imagenet_crop_test)
outlier_Imagenet_crop += outlier_detector_l2.predict(result_Imagenet_crop_test_base)


# **************** outlier predict final **************** #
outlier_cifar10_train[outlier_cifar10_train <= 1] = -1
outlier_cifar10_train[outlier_cifar10_train > 1] = 0
outlier_cifar10_train[outlier_cifar10_train == 0] = \
    result_cifar10_train_base.argmax(axis=1)[outlier_cifar10_train == 0]

outlier_Imagenet_crop[outlier_Imagenet_crop <= 1] = -1
outlier_Imagenet_crop[outlier_Imagenet_crop > 1] = 0
outlier_Imagenet_crop[outlier_Imagenet_crop == 0] = \
    result_Imagenet_crop_test_base.argmax(axis=1)[outlier_Imagenet_crop == 0]

# **************** Print predict result **************** #
print('outlier_cifar10_train detection rate:', (outlier_cifar10_train == -1).sum() / outlier_cifar10_train.shape[0])
print('outlier_Imagenet_crop detection rate:', (outlier_Imagenet_crop == -1).sum() / outlier_Imagenet_crop.shape[0])
print('End')



