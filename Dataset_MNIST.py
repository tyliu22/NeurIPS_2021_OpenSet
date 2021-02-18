#!/usr/bin/env python
# coding: utf-8
import zipfile
from PIL import Image
from io import BytesIO
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms

from sklearn.manifold import TSNE
from OutlierDetection import OutlierDetection
import matplotlib.pyplot as plt

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
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128)


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
#         return output, x_hidden
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 100, 5, 1)
        self.conv2 = nn.Conv2d(100, 100, 5, 1)
        self.conv3 = nn.Conv2d(100, 100, 5, 1)
        self.conv4 = nn.Conv2d(100, 100, 5, 1)
        self.fc1 = nn.Linear(3*3*100, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)

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


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            test_loss += F.nll_loss(torch.log(output), target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


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
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)



result_mnist_train_last_layer = []
result_mnist_train_hidden = []
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(train_loader):
        # [128, 1, 28, 28]
        data, target = data.to(device), target.to(device)
        output_train, hidden_train = model(data)
        result_mnist_train_last_layer.append(output_train)
        result_mnist_train_hidden.append(hidden_train)

result_mnist_train_last_layer = torch.cat(result_mnist_train_last_layer, dim=0)
result_mnist_train_hidden = torch.cat(result_mnist_train_hidden, dim=0)

train_loader_whole = torch.utils.data.DataLoader(train_dataset, batch_size=60000)
test_loader_whole = torch.utils.data.DataLoader(test_dataset, batch_size=10000)

# LOAD DATA: CIFAR10 (training or testing dataset)
x_mnist_train, x_mnist_train_label = iter(train_loader_whole).next()
x_mnist_test, x_mnist_test_label = iter(test_loader_whole).next()



print('Loading outlier datasets')
omniglot_data_path = '/home/tianyliu/Data/OpenSet/Dataset/omniglot/python/images_evaluation.zip'
sample_size = 10000
# introduce noise to mnist
# x_mnist_test, x_mnist_test_label = iter(test_loader_whole).next()
x_mnist_noise_test = x_mnist_test.numpy().copy()
random_noise = np.random.uniform(
    0, 1, x_mnist_noise_test[np.where(x_mnist_noise_test == 0)].shape[0])
x_mnist_noise_test[np.where(x_mnist_noise_test == 0)] = random_noise


x_noise_test = np.random.uniform(0, 1, (sample_size, 1, 28, 28))
x_noise_test = torch.FloatTensor(x_noise_test)
print('x_noise_test Dataset shape:', x_noise_test.shape)


# plot figure
# plt.imshow(x_noise_test[1].numpy().reshape(28,28), cmap='gray')
# plt.show()

with torch.no_grad():
    result_mnist_test_last_layer, result_mnist_test_hidden = model(x_mnist_test.to(device))
    result_noise_last_layer, result_noise_hidden = model(x_noise_test.to(device))


abnormal_datasets_name = ['noise']
abnormal_datasets = [result_noise_last_layer, result_noise_hidden]

for i in range(1,11,1):
    OutlierDetection(result_mnist_train_last_layer, result_mnist_train_hidden,
                     result_mnist_test_last_layer, result_mnist_test_hidden, x_mnist_test_label,
                     abnormal_datasets_name, abnormal_datasets,
                     sample_size=10000, r_seed=0, n_estimators=1000, verbose=0,
                     max_samples=10000, contamination=0.01*i)
print('End')


# def plot_tsne(features, labels, save_eps=False):
#         ''' Plot TSNE figure. Set save_eps=True if you want to save a .eps file.
#         '''
#         tsne = TSNE(n_components=2, init='pca', random_state=0)
#         features_tsne = tsne.fit_transform(features)
#         x_min, x_max = np.min(features_tsne, 0), np.max(features_tsne, 0)
#         features_norm = (features_tsne - x_min) / (x_max - x_min)
#         for i in range(features_norm.shape[0]):
#             plt.text(features_norm[i, 0], features_norm[i, 1], str(labels[i]),
#                      color=plt.cm.Set1(labels[i] / 10.),
#                      fontdict={'weight': 'bold', 'size': 9})
#         plt.xticks([])
#         plt.yticks([])
#         plt.title('T-SNE')
#         if save_eps:
#             plt.savefig('tsne.eps', dpi=600, format='eps')
#         plt.show()
#
#
# plot_tsne(result_mnist_train_last_layer, x_mnist_train_label.cpu().numpy())
# plot_tsne(result_mnist_test_last_layer, x_mnist_test_label.cpu().numpy())