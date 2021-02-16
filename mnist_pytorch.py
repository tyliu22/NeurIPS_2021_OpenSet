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

from sklearn.metrics import f1_score
from sklearn.ensemble import IsolationForest
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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 100, 3, 1)
        self.conv2 = nn.Conv2d(100, 100, 3, 1)
        self.conv3 = nn.Conv2d(100, 100, 3, 1)
        self.conv4 = nn.Conv2d(100, 50, 3, 1)
        # self.bn1 = nn.BatchNorm2d(100)
        # self.bn2 = nn.BatchNorm2d(100)
        # self.bn3 = nn.BatchNorm2d(100)
        # self.bn4 = nn.BatchNorm2d(50)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = self.fc1(x)
        x_hidden = self.fc2(x)
        output = F.softmax(self.fc3(x_hidden), dim=1)
        # output = self.fc2(x)
        # log_softmax
        # output = F.softmax(x, dim=1)
        return output, x


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # [128, 1, 28, 28]
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
print(device)
model = Net().to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True


optimizer = optim.Adam(model.parameters(), lr=0.0001)
# train_loader  test_loader
for epoch in range(1, 8):
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

# x_mnist_train = x_mnist_train
# x_mnist_test  = x_mnist_test

print('Loading outlier datasets')

omniglot_data_path = '/home/tianyliu/Data/OpenSet/Dataset/omniglot/python/images_evaluation.zip'
sample_size = 10000
# introduce noise to mnist
# x_mnist_test, x_mnist_test_label = iter(test_loader_whole).next()
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


x_noise_test = np.random.uniform(0, 1, (sample_size, 1, 28, 28))

# x_mnist_test, x_mnist_noise_test, x_omniglot_test, x_noise_test
x_mnist_noise_test = torch.FloatTensor(x_mnist_noise_test)
x_omniglot_test = torch.FloatTensor(x_omniglot_test).unsqueeze(1)
x_noise_test = torch.FloatTensor(x_noise_test)


print('x_mnist_test Dataset shape:', x_mnist_test.shape)
print('x_mnist_noise_test Dataset shape:', x_mnist_noise_test.shape)
print('x_omniglot_test Dataset shape:', x_omniglot_test.shape)
print('x_noise_test Dataset shape:', x_noise_test.shape)


# plot figure
# plt.imshow(x_noise_test[1].numpy().reshape(28,28), cmap='gray')
# plt.show()

# torch.cuda.clear_memory_allocated()
model = Net().to(device)
model.train(False)
with torch.no_grad():
    # result_mnist_train_last_layer, result_mnist_train_hidden = model(x_mnist_train.to(device))
    result_mnist_test_last_layer, result_mnist_test_hidden = model(x_mnist_test.to(device))
    result_mnist_noise_test_last_layer, result_mnist_noise_test_hidden = model(x_mnist_noise_test.to(device))
    result_omniglot_test_last_layer, result_omniglot_test_hidden = model(x_omniglot_test.to(device))
    result_noise_last_layer, result_noise_hidden = model(x_noise_test.to(device))


# **************** outlier detector training **************** #
print('outlier detector training')
sample_size = 10000
outlier_detector_l1 = IsolationForest(random_state=r_seed, n_estimators=1000, verbose=0, max_samples=10000,
                                      contamination=0.05)
outlier_detector_l2 = IsolationForest(random_state=r_seed, n_estimators=1000, verbose=0, max_samples=10000,
                                      contamination=0.05)

result_mnist_train = result_mnist_train_hidden.cpu().numpy()
result_mnist_train_base = result_mnist_train_last_layer.cpu().numpy()
# data argument
outlier_detector_l1.fit(result_mnist_train)
outlier_detector_l2.fit(result_mnist_train_base)



# **************** Tensor2numpy **************** #
result_mnist_test = result_mnist_test_hidden.cpu().numpy()
result_mnist_test_base = result_mnist_test_last_layer.cpu().numpy()

result_mnist_noise_test = result_mnist_noise_test_hidden.cpu().numpy()
result_mnist_noise_test_base = result_mnist_noise_test_last_layer.cpu().numpy()

result_omniglot_test = result_omniglot_test_hidden.cpu().numpy()
result_omniglot_test_base = result_omniglot_test_last_layer.cpu().numpy()

result_noise_test = result_noise_hidden.cpu().numpy()
result_noise_test_base = result_noise_last_layer.cpu().numpy()


# **************** outlier predict **************** #
print('outlier predict')
outlier_mnist_train = outlier_detector_l1.predict(result_mnist_train)
outlier_mnist_train += outlier_detector_l2.predict(result_mnist_train_base)

outlier_mnist_test = outlier_detector_l1.predict(result_mnist_test)
outlier_mnist_test += outlier_detector_l2.predict(result_mnist_test_base)

outlier_mnist_noise = outlier_detector_l1.predict(result_mnist_noise_test)
outlier_mnist_noise += outlier_detector_l2.predict(result_mnist_noise_test_base)

outlier_omniglot_test = outlier_detector_l1.predict(result_omniglot_test)
outlier_omniglot_test += outlier_detector_l2.predict(result_omniglot_test_base)

outlier_noise_test = outlier_detector_l1.predict(result_noise_test)
outlier_noise_test += outlier_detector_l2.predict(result_noise_test_base)

# **************** outlier predict final two layer detection **************** #
outlier_mnist_train[outlier_mnist_train <= 1] = -1
outlier_mnist_train[outlier_mnist_train > 1] = 0
outlier_mnist_train[outlier_mnist_train == 0] = \
    result_mnist_train_base.argmax(axis=1)[outlier_mnist_train == 0]

outlier_mnist_test[outlier_mnist_test <= 1] = -1
outlier_mnist_test[outlier_mnist_test > 1] = 0
outlier_mnist_test[outlier_mnist_test == 0] = \
    result_mnist_test_base.argmax(axis=1)[outlier_mnist_test == 0]

outlier_mnist_noise[outlier_mnist_noise <= 1] = -1
outlier_mnist_noise[outlier_mnist_noise > 1] = 0
outlier_mnist_noise[outlier_mnist_noise == 0] = \
    result_mnist_noise_test_base.argmax(axis=1)[outlier_mnist_noise == 0]

outlier_omniglot_test[outlier_omniglot_test <= 1] = -1
outlier_omniglot_test[outlier_omniglot_test > 1] = 0
outlier_omniglot_test[outlier_omniglot_test == 0] = \
    result_omniglot_test_base.argmax(axis=1)[outlier_omniglot_test == 0]

outlier_noise_test[outlier_noise_test <= 1] = -1
outlier_noise_test[outlier_noise_test > 1] = 0
outlier_noise_test[outlier_noise_test == 0] = \
    result_noise_test_base.argmax(axis=1)[outlier_noise_test == 0]


# **************** Print predict result **************** #
print('outlier_mnist_train detection rate:', (outlier_mnist_train == -1).sum() / outlier_mnist_train.shape[0])
print('outlier_mnist_test detection rate:', (outlier_mnist_test == -1).sum() / outlier_mnist_test.shape[0])
print('outlier_mnist_noise detection rate:', (outlier_mnist_noise == -1).sum() / outlier_mnist_noise.shape[0])
print('outlier_omniglot_test detection rate:', (outlier_omniglot_test == -1).sum() / outlier_omniglot_test.shape[0])
print('outlier_noise_test detection rate:', (outlier_noise_test == -1).sum() / outlier_noise_test.shape[0])


# **************** F1 Score result **************** #

base_pred = result_mnist_test_base.argmax(axis=1)
base_pred[outlier_mnist_test == -1] = -1

# total 20000 samples: 10000 cifar10, 10000 other samples
true_label = np.zeros(sample_size * 2)
true_label = true_label - 1
true_label[:sample_size] = x_mnist_test_label

mnist_noise_f1 = f1_score(true_label, np.concatenate([base_pred, outlier_mnist_noise]), average='macro')
omniglot_test_f1 = f1_score(true_label, np.concatenate([base_pred, outlier_omniglot_test]), average='macro')
noise_test_f1 = f1_score(true_label, np.concatenate([base_pred, outlier_noise_test]), average='macro')

print('mnist_noise detection f1 score:', mnist_noise_f1)
print('omniglot_test detection  f1 score:', omniglot_test_f1)
print('noise_test detection f1 score:', noise_test_f1)


print('End')

















#
# abnormal_datasets_name = ['mnist noise', 'omniglot test', 'noise']
# abnormal_datasets = [result_mnist_noise_test_last_layer, result_mnist_noise_test_hidden,
#                      result_omniglot_test_last_layer, result_omniglot_test_hidden,
#                      result_noise_last_layer, result_noise_hidden
#                      ]
#
# OutlierDetection(result_mnist_train_last_layer, result_mnist_train_hidden,
#                  result_mnist_test_last_layer, result_mnist_test_hidden, x_mnist_test_label,
#                  abnormal_datasets_name, abnormal_datasets,
#                  sample_size=10000, r_seed=0, n_estimators=1000, verbose=0,
#                  max_samples=10000, contamination=0.05)
#
#
# print('End')




def plot_tsne(features, labels, save_eps=False):
        ''' Plot TSNE figure. Set save_eps=True if you want to save a .eps file.
        '''
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        features_tsne = tsne.fit_transform(features)
        x_min, x_max = np.min(features_tsne, 0), np.max(features_tsne, 0)
        features_norm = (features_tsne - x_min) / (x_max - x_min)
        for i in range(features_norm.shape[0]):
            plt.text(features_norm[i, 0], features_norm[i, 1], str(labels[i]),
                     color=plt.cm.Set1(labels[i] / 10.),
                     fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        plt.title('T-SNE')
        if save_eps:
            plt.savefig('tsne.eps', dpi=600, format='eps')
        plt.show()


plot_tsne(result_mnist_train_base, x_mnist_train_label.cpu().numpy())

plot_tsne(result_mnist_test_base, x_mnist_test_label.cpu().numpy())


