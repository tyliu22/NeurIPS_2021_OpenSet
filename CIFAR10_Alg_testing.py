'''Train CIFAR10 with PyTorch.'''
import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import argparse
from PIL import Image
from io import BytesIO
import zipfile

import numpy as np

from models import *
from Utils.OutlierDetection import OutlierDetection

r_seed = 0
np.random.seed(r_seed)

# ******************************************************************* #
#                       Para Setting
# ******************************************************************* #

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


# ******************************************************************* #
#                        Loading Dataset
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
                image = 1 - np.array(pil_img) * 1.0
                if len(image.shape) == 2:
                    color_image = np.expand_dims(image, axis=2)
                    color_image = np.concatenate((color_image, color_image, color_image), axis=-1)
                    image = color_image
                image_data_list.append([image])

    image_data = np.concatenate(image_data_list)
    return image_data


# LOAD DATA: CIFAR10 (training or testing dataset)
# Tensor
x_cifar10_train, x_cifar10_train_label = iter(trainloader_whole).next()
x_cifar10_test, x_cifar10_test_label = iter(testloader_whole).next()

# LOAD DATA: Imagenet_crop
Imagenet_crop_data_path = '/home/tianyliu/Data/OpenSet/Dataset/Image/Imagenet_crop.zip'
Imagenet_crop_data = load_image_eval(Imagenet_crop_data_path)
x_Imagenet_crop_test = Imagenet_crop_data

# LOAD DATA: Imagenet_resize
Imagenet_resize_data_path = '/home/tianyliu/Data/OpenSet/Dataset/Image/Imagenet_resize.zip'
Imagenet_resize_data = load_image_eval(Imagenet_resize_data_path)
# sample_idx = np.random.permutation(Imagenet_resize_data.shape[0])[:sample_size]
x_Imagenet_resize_test = Imagenet_resize_data


# LOAD DATA: LSUN_crop
LSUN_crop_data_path = '/home/tianyliu/Data/OpenSet/Dataset/Image/LSUN_crop.zip'
LSUN_crop_data = load_image_eval(LSUN_crop_data_path)
x_LSUN_crop_test = LSUN_crop_data

# LOAD DATA: LSUN_resize
LSUN_resize_data_path = '/home/tianyliu/Data/OpenSet/Dataset/Image/LSUN_resize.zip'
LSUN_resize_data = load_image_eval(LSUN_resize_data_path)
x_LSUN_resize_test = LSUN_resize_data



# numpy To tensor  [10000, 3, 32, 32]
x_Imagenet_crop_test = torch.from_numpy(x_Imagenet_crop_test).permute(0, 3, 1, 2)
x_Imagenet_resize_test = torch.from_numpy(x_Imagenet_resize_test).permute(0, 3, 1, 2)
x_LSUN_crop_test = torch.from_numpy(x_LSUN_crop_test).permute(0, 3, 1, 2)
x_LSUN_resize_test = torch.from_numpy(x_LSUN_resize_test).permute(0, 3, 1, 2)

# Double Tensor 2 Float Tensor
x_Imagenet_crop_test = x_Imagenet_crop_test.type(torch.FloatTensor)
x_Imagenet_resize_test = x_Imagenet_resize_test.type(torch.FloatTensor)
x_LSUN_crop_test = x_LSUN_crop_test.type(torch.FloatTensor)
x_LSUN_resize_test = x_LSUN_resize_test.type(torch.FloatTensor)


# Double Tensor 2 Float Tensor
x_Imagenet_crop_test = x_Imagenet_crop_test.type(torch.FloatTensor)
x_Imagenet_resize_test = x_Imagenet_resize_test.type(torch.FloatTensor)
x_LSUN_crop_test = x_LSUN_crop_test.type(torch.FloatTensor)
x_LSUN_resize_test = x_LSUN_resize_test.type(torch.FloatTensor)



print('x_cifar10_train Dataset shape:', x_cifar10_train.shape)
print('x_cifar10_test Dataset shape:', x_cifar10_test.shape)
print('Imagenet_crop Dataset shape:', x_Imagenet_crop_test.shape)
print('Imagenet_resize Dataset shape:', x_Imagenet_resize_test.shape)
print('LSUN_crop Dataset shape:', x_LSUN_crop_test.shape)
print('LSUN_resize Dataset shape:', x_LSUN_resize_test.shape)


with torch.no_grad():
    result_cifar10_train_last_layer, result_cifar10_train_hidden\
        = net(x_cifar10_train.to(device))
    result_cifar10_test_last_layer, result_cifar10_test_hidden\
        = net(x_cifar10_test.to(device))
    result_Imagenet_crop_test_last_layer, result_Imagenet_crop_hidden\
        = net(x_Imagenet_crop_test.to(device))
    result_Imagenet_resize_test_last_layer, result_Imagenet_resize_hidden \
        = net(x_Imagenet_resize_test.to(device))
    result_LSUN_crop_test_last_layer, result_LSUN_crop_hidden\
        = net(x_LSUN_crop_test.to(device))
    result_LSUN_resize_test_last_layer, result_LSUN_resize_hidden\
        = net(x_LSUN_resize_test.to(device))


# OutlierDetection(train_data_last_layer, train_data_hidden,
#                  test_data_last_layer, test_data_hidden, test_data_label,
#                  outlier_datasets_name, outlier_datasets,
#                  sample_size=10000, r_seed=0, n_estimators=1000, verbose=0,
#                  max_samples=10000, contamination=0.02)

abnormal_datasets_name = ['Imagenet crop', 'Imagenet resize', 'LSUN crop', 'LSUN resize']
abnormal_datasets = [result_Imagenet_crop_test_last_layer, result_Imagenet_crop_hidden,
                     result_Imagenet_resize_test_last_layer, result_Imagenet_resize_hidden,
                     result_LSUN_crop_test_last_layer, result_LSUN_crop_hidden,
                     result_LSUN_resize_test_last_layer, result_LSUN_resize_hidden
                     ]

OutlierDetection(result_cifar10_train_last_layer, result_cifar10_train_hidden,
                 result_cifar10_test_last_layer, result_cifar10_test_hidden, x_cifar10_test_label,
                 abnormal_datasets_name, abnormal_datasets,
                 sample_size=10000, r_seed=0, n_estimators=1000, verbose=0,
                 max_samples=10000, contamination=0.01)


