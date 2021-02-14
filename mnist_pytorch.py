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
from torchvision import datasets, transforms

from OutlierDetection import OutlierDetection

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


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        output = self.fc2(x)
        # log_softmax
        # output = F.softmax(x, dim=1)
        return output, x


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, _ = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))




use_cuda = torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")

model = LeNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
# train_loader  test_loader
for epoch in range(1, 6):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)



train_loader_whole = torch.utils.data.DataLoader(train_dataset, batch_size=60000)
test_loader_whole = torch.utils.data.DataLoader(test_dataset, batch_size=10000)

# LOAD DATA: CIFAR10 (training or testing dataset)
x_mnist_train, x_mnist_train_label = iter(train_loader_whole).next()
x_mnist_test, x_mnist_test_label = iter(test_loader_whole).next()



print('Loading outlier datasets')

omniglot_data_path = '/home/tianyliu/Data/OpenSet/Dataset/omniglot/python/images_evaluation.zip'
sample_size = 10000
# introduce noise to mnist
x_mnist_test, x_mnist_test_label = iter(test_loader_whole).next()
x_mnist_noise_test = x_mnist_test.numpy().copy()
random_noise = np.random.uniform(
    0, 1, x_mnist_noise_test[np.where(x_mnist_noise_test == 0)].shape[0])
x_mnist_noise_test[np.where(x_mnist_noise_test == 0)] = random_noise


# ## Omniglot
def load_omniglot_eval(data_path):
    omniglot_data_list = []
    with zipfile.ZipFile(data_path) as zf:
        for filename in zf.namelist():
            if '.png' in filename:
                zip_data = zf.read(filename)
                bytes_io = BytesIO(zip_data)
                pil_img = Image.open(bytes_io)
                pil_img = pil_img.resize((28, 28))
                omniglot_data_list.append([1 - np.array(pil_img) * 1.0])

    omniglot_data = np.concatenate(omniglot_data_list)
    return omniglot_data

omniglot_data = load_omniglot_eval(omniglot_data_path)
sample_idx = np.random.permutation(omniglot_data.shape[0])[:sample_size]
x_omniglot_test = omniglot_data[sample_idx]
print('Omniglot Dataset shape:', omniglot_data.shape)

x_noise_test = np.random.uniform(0, 1, (sample_size, 1, 28, 28))

# x_mnist_test, x_mnist_noise_test, x_omniglot_test, x_noise_test
x_mnist_noise_test = torch.FloatTensor(x_mnist_noise_test)
x_omniglot_test = torch.FloatTensor(x_omniglot_test).unsqueeze(1)
x_noise_test = torch.FloatTensor(x_noise_test)


print('x_mnist_test Dataset shape:', x_mnist_test.shape)
print('x_mnist_noise_test Dataset shape:', x_mnist_noise_test.shape)
print('x_omniglot_test Dataset shape:', x_omniglot_test.shape)
print('x_noise_test Dataset shape:', x_noise_test.shape)




with torch.no_grad():
    result_mnist_train_last_layer, result_mnist_train_hidden\
        = model(x_mnist_train.to(device))
    result_mnist_test_last_layer, result_mnist_test_hidden\
        = model(x_mnist_test.to(device))

    # x_mnist_test, x_mnist_noise_test, x_omniglot_test, x_noise_test
    result_mnist_noise_test_last_layer, result_mnist_noise_test_hidden\
        = model(x_mnist_noise_test.to(device))
    result_omniglot_test_last_layer, result_omniglot_test_hidden\
        = model(x_omniglot_test.to(device))
    result_noise_last_layer, result_noise_hidden\
        = model(x_noise_test.to(device))


# OutlierDetection(train_data_last_layer, train_data_hidden,
#                  test_data_last_layer, test_data_hidden, test_data_label,
#                  outlier_datasets_name, outlier_datasets,
#                  sample_size=10000, r_seed=0, n_estimators=1000, verbose=0,
#                  max_samples=10000, contamination=0.02)

abnormal_datasets_name = ['mnist noise', 'omniglottest', 'noise']
abnormal_datasets = [result_mnist_noise_test_last_layer, result_mnist_noise_test_hidden,
                     result_omniglot_test_last_layer, result_omniglot_test_hidden,
                     result_noise_last_layer, result_noise_hidden
                     ]

OutlierDetection(result_mnist_train_last_layer, result_mnist_train_hidden,
                 result_mnist_test_last_layer, result_mnist_test_hidden, x_mnist_test_label,
                 abnormal_datasets_name, abnormal_datasets,
                 sample_size=10000, r_seed=0, n_estimators=1000, verbose=0,
                 max_samples=10000, contamination=0.05)


print('End')




