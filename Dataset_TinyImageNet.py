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
import torchvision.models as models
from torch.optim import lr_scheduler
from Utils.train_model import train_model

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
test_dataloader = DataLoader.DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=4)









data_transforms = { 'train': transforms.Compose([transforms.ToTensor()]),
                    'val'  : transforms.Compose([transforms.ToTensor(),]) }

data_dir = 'images/64'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=100, shuffle=True, num_workers=64)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

#Load Resnet18 with pretrained weights
model_ft = models.resnet18()
#Finetune Final few layers to adjust for tiny imagenet input
model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 200)
# model_ft.conv1 = nn.Conv2d(3,64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
# model_ft.maxpool = nn.Sequential()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)
#Multi GPU
model_ft = torch.nn.DataParallel(model_ft, device_ids=[0, 1])
model_ft.load_state_dict(torch.load('./models/resnet18_224_w.pt'))

#Loss Function
criterion = nn.CrossEntropyLoss()
# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)



#Train
model_ft = train_model(model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=7)

#Load Resnet18 with pretrained weights
model_ft = models.resnet18()
#Finetune Final few layers to adjust for tiny imagenet input
model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 200)
model_ft.conv1 = nn.Conv2d(3,64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
model_ft.maxpool = nn.Sequential()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)
#Multi GPU
model_ft = torch.nn.DataParallel(model_ft, device_ids=[0, 1])
pretrained_dict = torch.load('./models/resnet18_224_w.pt')
model_ft_dict = model_ft.state_dict()
first_layer_weight = model_ft_dict['module.conv1.weight']
first_layer_bias  = model_ft_dict['module.conv1.bias']
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_ft_dict}

model_ft_dict.update(pretrained_dict)
model_ft_dict['module.conv1.weight'] = first_layer_weight
model_ft_dict['module.conv1.bias']   = first_layer_bias
model_ft.load_state_dict(model_ft_dict)


#Loss Function
criterion = nn.CrossEntropyLoss()
# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

#Train
model_ft = train_model(model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=7)
#Train
model_ft = train_model(model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=7)



