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

from Utils.AUROC_Score import AUROC_score
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
    num_class = num_class+1
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















