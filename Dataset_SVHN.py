#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.dataloader as DataLoader
from sklearn.manifold import TSNE
from torchvision import datasets, transforms

from Utils.AUROC_Score_single import AUROC_score
from Utils.MyDataLoader import subDataset

r_seed = 0
torch.manual_seed(r_seed)
np.random.seed(r_seed)

transform = transforms.Compose([
    transforms.ToTensor()
])

# # Preparing Data : SVHN
train_dataset = datasets.SVHN('../data', split='train', transform=None,
                              target_transform=None, download=True)
test_dataset = datasets.SVHN('../data', split='train', transform=None,
                             target_transform=None, download=True)

train_dataset_data = train_dataset.data
train_dataset_label = train_dataset.labels
label_class = np.array(list(set(train_dataset.labels)))
# select 6 classes as training dataset, 4 dataset as testing dataset
np.random.shuffle(label_class)
selected_class = label_class[0:6]
unselected_class = label_class[6:10]
print('MNIST training class:', selected_class)
print('MNIST testing  class:', unselected_class)


selected_train_dataset_label = np.empty(shape=[0])
selected_train_dataset_data = np.empty(shape=[0,3,32,32])
for i in selected_class:
    selected_train_dataset_data = np.append(selected_train_dataset_data,
                                             train_dataset_data[np.where(train_dataset_label==i)], axis=0)
    selected_train_dataset_label = np.append(selected_train_dataset_label,
                                             train_dataset_label[np.where(train_dataset_label==i)])
num_class = int(10)
for i in selected_class:
    selected_train_dataset_label[np.where(selected_train_dataset_label==i)] = num_class
    num_class = num_class+1
selected_train_dataset_label = (selected_train_dataset_label - 10).astype(int)

unselected_train_dataset_label = np.empty(shape=[0])
unselected_train_dataset_data = np.empty(shape=[0,3,32,32])
for i in unselected_class:
    unselected_train_dataset_data = np.append(unselected_train_dataset_data,
                                             train_dataset_data[np.where(train_dataset_label==i)], axis=0)
    # unselected_train_dataset_label = np.append(unselected_train_dataset_label,
    #                                          train_dataset_label[np.where(train_dataset_label==i)])


SVHN_train_data, SVHN_train_label = selected_train_dataset_data,\
                                      selected_train_dataset_label
SVHN_test_data, SVHN_test_label = unselected_train_dataset_data,\
                                    unselected_train_dataset_label

train_dataset = subDataset(SVHN_train_data, SVHN_train_label)
test_dataset = subDataset(SVHN_test_data, SVHN_test_label)

train_dataloader = DataLoader.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
test_dataloader = DataLoader.DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=4)




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 100, 3, 1)
        self.conv2 = nn.Conv2d(100, 100, 3, 1)
        self.conv3 = nn.Conv2d(100, 100, 3, 1)
        self.conv4 = nn.Conv2d(100, 100, 3, 1)
        self.conv5 = nn.Conv2d(100, 100, 3, 1)
        self.fc1 = nn.Linear(4 * 4 * 100, 500)
        self.fc2 = nn.Linear(500, 100)
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
        x = x.view(-1, 4 * 4 * 100)
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
        output, _ = model(data.float())
        loss = F.nll_loss(torch.log(output), target.long())
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct = pred.eq(target.view_as(pred)).sum().item()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy:{:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), correct / len(data)))



use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
model = Net().to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

# model training
train_epoch = 10
optimizer = optim.Adam(model.parameters(), lr=1e-4)
for epoch in range(1, train_epoch):
    train(model, device, train_dataloader, optimizer, epoch)
    # test(model, device, test_loader)



train_dataloader = DataLoader.DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=4)
test_dataloader = DataLoader.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

result_SVHN_train_last_layer = []
result_SVHN_train_hidden = []
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(train_dataloader):
        # [128, 1, 28, 28]
        data, target = data.type(torch.FloatTensor).to(device), target.to(device)
        output_train, hidden_train = model(data)
        result_SVHN_train_last_layer.append(output_train)
        result_SVHN_train_hidden.append(hidden_train)

result_SVHN_train_last_layer = torch.cat(result_SVHN_train_last_layer, dim=0)
result_SVHN_train_hidden = torch.cat(result_SVHN_train_hidden, dim=0)


SVHN_train_data = torch.tensor(SVHN_train_data)
SVHN_test_data = torch.tensor(SVHN_test_data)
with torch.no_grad():
    # result_mnist_train_last_layer, result_mnist_train_hidden = model(mnist_train_data.float().to(device))
    result_SVHN_test_last_layer, result_SVHN_test_hidden = model(SVHN_test_data.float().to(device))

# *********************** AUROC_score ************************* #
num_train_sample = SVHN_train_data.shape[0]
num_test_sample = SVHN_test_data.shape[0]

print('===> AUROC_score start')
# ******************* Outlier Detection ********************** #
outlier_train_hidden_detector_label, outlier_test_hidden_detector_label = AUROC_score(
    F.softmax(result_SVHN_train_last_layer), result_SVHN_train_hidden, num_train_sample,
    F.softmax(result_SVHN_test_last_layer), result_SVHN_test_hidden, num_test_sample,
    r_seed=0, n_estimators=1000, verbose=0, max_samples=1.0, contamination=0.1)

print('Algorithm End')


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
        plt.savefig('/figures/tsne.eps', dpi=600, format='eps')
    plt.show()


result_mnist_train_hidden = result_SVHN_train_hidden.cpu().numpy()
result_mnist_test_hidden = result_SVHN_test_hidden.cpu().numpy()

tsne_data = np.append(result_mnist_train_hidden, result_mnist_test_hidden, axis=0)
tsne_label_predictor = np.append(outlier_train_hidden_detector_label, outlier_test_hidden_detector_label)

plot_tsne(tsne_data, tsne_label_predictor)










