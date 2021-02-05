'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import argparse
from PIL import Image
from io import BytesIO
import numpy as np
import zipfile

import numpy as np
# import pandas as pd
# from sklearn.metrics import f1_score
from sklearn.ensemble import IsolationForest
# from keras import backend as K

from models import *

r_seed = 0
np.random.seed(r_seed)
# torch.random.seed(r_seed)


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', default=True, action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ******************************************************************* #
#                        load Model & CIFAR10
# ******************************************************************* #
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

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = DLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']


# ******************************************************************* #
#                        Making Prediction
# ******************************************************************* #
# CIFAR10    Data
trainloader_whole = torch.utils.data.DataLoader(
    trainset, batch_size=50000, shuffle=False)
testloader_whole = torch.utils.data.DataLoader(
    testset, batch_size=10000, shuffle=False)
# [50000, 3, 32, 32]

    # result_cifar10_train_base = net(x_cifar10_train)



# ******************************************************************* #
#                        Outlier Detection
# ******************************************************************* #
def load_image_eval(data_path):
    image_data_list = []
    str = ['.png', '.jpg', '.jpeg']
    with zipfile.ZipFile(data_path) as zf:
        for filename in zf.namelist():
            if '.png' in filename or '.jpg' in filename or '.jpeg' in filename:
                zip_data = zf.read(filename)
                bytes_io = BytesIO(zip_data)
                pil_img = Image.open(bytes_io)
                pil_img = pil_img.resize((32, 32))
                image_data_list.append([1 - np.array(pil_img) * 1.0])

    image_data = np.concatenate(image_data_list)
    return image_data

# LOAD DATA: CIFAR10 (testing dataset)


# LOAD DATA: Imagenet_crop
Imagenet_crop_data_path = '/home/tianyliu/Data/OpenSet/Dataset/Image/Imagenet_crop.zip'
Imagenet_crop_data = load_image_eval(Imagenet_crop_data_path)
x_Imagenet_crop_test = Imagenet_crop_data


# LOAD DATA: Imagenet_resize
# Imagenet_resize_data_path = '/home/tianyliu/Data/OpenSet/Dataset/Image/Imagenet_resize.zip'
# Imagenet_resize_data = load_image_eval(Imagenet_resize_data_path)
# # sample_idx = np.random.permutation(Imagenet_resize_data.shape[0])[:sample_size]
# x_Imagenet_resize_test = Imagenet_resize_data


# LOAD DATA: LSUN_crop
# LSUN_crop_data_path = '/home/tianyliu/Data/OpenSet/Dataset/Image/LSUN_crop.zip'
# LSUN_crop_data = load_image_eval(LSUN_crop_data_path)
# x_LSUN_crop_test = LSUN_crop_data

# LOAD DATA: LSUN_resize
# LSUN_resize_data_path = '/home/tianyliu/Data/OpenSet/Dataset/Image/LSUN_resize.zip'
# LSUN_resize_data = load_image_eval(LSUN_resize_data_path)
# x_LSUN_resize_test = LSUN_resize_data

# LOAD DATA: iSUN
# iSUN_data_path = '/home/tianyliu/Data/OpenSet/Dataset/Image/iSUN.zip'
# iSUN_data = load_image_eval(iSUN_data_path)
# x_iSUN_test = iSUN_data


# numpy To tensor  [10000, 3, 32, 32]
x_Imagenet_crop_test = torch.from_numpy(x_Imagenet_crop_test).permute(0, 3, 1, 2)
x_Imagenet_crop_test = x_Imagenet_crop_test.type(torch.FloatTensor)
# x_Imagenet_resize_test = torch.from_numpy(x_Imagenet_resize_test).permute(0, 3, 1, 2)
# x_LSUN_crop_test = torch.from_numpy(x_LSUN_crop_test).permute(0, 3, 1, 2)
# x_LSUN_resize_test = torch.from_numpy(x_LSUN_resize_test).permute(0, 3, 1, 2)
# x_iSUN_test = torch.from_numpy(x_iSUN_test).permute(0, 3, 1, 2)


print('Imagenet_crop Dataset shape:', x_Imagenet_crop_test.shape)
# print('Imagenet_resize Dataset shape:', x_Imagenet_resize_test.shape)
# print('LSUN_crop Dataset shape:', x_LSUN_crop_test.shape)
# print('LSUN_resize Dataset shape:', x_LSUN_resize_test.shape)
# print('iSUN Dataset shape:', x_iSUN_test.shape)




with torch.no_grad():
    x_cifar10_train,  x_cifar10_train_label= iter(trainloader_whole).next()
    x_cifar10_test, x_cifar10_test_label = iter(testloader_whole).next()
    result_cifar10_train_base, result_cifar10_train = net(x_cifar10_train.to(device))
    result_cifar10_test_base, result_cifar10_test = net(x_cifar10_test.to(device))

with torch.no_grad():
    result_Imagenet_crop_test_base, result_Imagenet_crop_test = net(x_Imagenet_crop_test.to(device))




sample_size = 10000
outlier_detector_l1 = IsolationForest(random_state=r_seed, n_estimators=1000, verbose=0, max_samples=10000,
                                      contamination=0.05)
outlier_detector_l2 = IsolationForest(random_state=r_seed, n_estimators=1000, verbose=0, max_samples=10000,
                                      contamination=0.05)

result_cifar10_train = result_cifar10_train.cpu().numpy()
result_cifar10_train_base = result_cifar10_train_base.cpu().numpy()
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