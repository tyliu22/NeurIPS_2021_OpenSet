#!/usr/bin/env python
# coding: utf-8

# test

'''
Directly copy from file : mnist_playground.ipynb
'''

# # Import Package
import tensorflow as tf
import zipfile
from PIL import Image
from io import BytesIO
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.ensemble import IsolationForest
from keras import backend as K
from keras.datasets import cifar10


tf.random.set_seed(0)
r_seed = 0
np.random.seed(r_seed)
omniglot_data_path = '/home/tianyliu/Data/OpenSet/Dataset/omniglot/python/images_evaluation.zip'
sample_size = 10000


# # Preparing Data
# ## CIFAR10
(x_cifar10_train, y_cifar10_train), (x_cifar10_test, y_cifar10_test)\
    = cifar10.load_data()



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



# LOAD DATA: Imagenet_crop
Imagenet_crop_data_path = '/home/tianyliu/Data/OpenSet/Dataset/Image/Imagenet_crop.zip'
Imagenet_crop_data = load_image_eval(Imagenet_crop_data_path)
# sample_idx = np.random.permutation(Imagenet_crop_data.shape[0])[:sample_size]
# x_Imagenet_crop_test = Imagenet_crop_data[sample_idx]
x_Imagenet_crop_test = Imagenet_crop_data

print('Imagenet_crop Dataset shape:', Imagenet_crop_data.shape)




# LOAD DATA: Imagenet_resize
# Imagenet_resize_data_path = '/home/tianyliu/Data/OpenSet/Dataset/Image/Imagenet_resize.zip'
# Imagenet_resize_data = load_image_eval(Imagenet_resize_data_path)
# # sample_idx = np.random.permutation(Imagenet_resize_data.shape[0])[:sample_size]
# x_Imagenet_resize_test = Imagenet_resize_data
# print('Imagenet_resize Dataset shape:', Imagenet_resize_data.shape)


# LOAD DATA: LSUN_crop
LSUN_crop_data_path = '/home/tianyliu/Data/OpenSet/Dataset/Image/LSUN_crop.zip'
LSUN_crop_data = load_image_eval(LSUN_crop_data_path)
sample_idx = np.random.permutation(LSUN_crop_data.shape[0])[:sample_size]
x_LSUN_crop_test = LSUN_crop_data[sample_idx]
print('LSUN_crop Dataset shape:', LSUN_crop_data.shape)

# LOAD DATA: LSUN_resize
LSUN_resize_data_path = '/home/tianyliu/Data/OpenSet/Dataset/Image/LSUN_resize.zip'
LSUN_resize_data = load_image_eval(LSUN_resize_data_path)
sample_idx = np.random.permutation(LSUN_resize_data.shape[0])[:sample_size]
x_LSUN_resize_test = LSUN_resize_data[sample_idx]
print('LSUN_resize Dataset shape:', LSUN_resize_data.shape)


# LOAD DATA: iSUN
iSUN_data_path = '/home/tianyliu/Data/OpenSet/Dataset/Image/iSUN.zip'
iSUN_data = load_image_eval(iSUN_data_path)
sample_idx = np.random.permutation(iSUN_data.shape[0])[:sample_size]
x_iSUN_test = iSUN_data[sample_idx]
print('iSUN Dataset shape:', iSUN_data.shape)


'''
    Build Classifcation Model
'''

basic_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=100,
                           kernel_size=(3, 3),
                           activation="relu",
                           input_shape=(32, 32, 3)),
    tf.keras.layers.Conv2D(filters=100, kernel_size=(3, 3), activation="relu"),
    tf.keras.layers.Conv2D(filters=100, kernel_size=(3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=100, kernel_size=(3, 3), activation="relu"),
    tf.keras.layers.Conv2D(filters=100, kernel_size=(3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(500),
    tf.keras.layers.Dense(100),  # layer -3
    tf.keras.layers.Dense(10),  # layer -2
    tf.keras.layers.Activation(activation='softmax')  # layer -1
])

basic_model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

# set layer output as a second Model
model_l2 = tf.keras.models.Model(inputs=basic_model.layers[0].input,
                                 outputs=basic_model.layers[-2].output)

# set layer output as a third Model
model_l3 = tf.keras.models.Model(inputs=basic_model.layers[0].input,
                                 outputs=basic_model.layers[-3].output)


K.set_value(basic_model.optimizer.learning_rate, 0.001)

# ## CIFAR10
# (x_cifar10_train, y_cifar10_train), (x_cifar10_test, y_cifar10_test)\
#     = cifar10.load_data()

basic_model.fit(x_cifar10_train, y_cifar10_train, epochs=5)
basic_model.evaluate(x_cifar10_test, y_cifar10_test, verbose=2)

# # Making Prediction

result_Imagenet_crop_test = model_l2.predict(x_Imagenet_crop_test)
# result_Imagenet_resize_test = model_l2.predict(x_Imagenet_resize_test)
result_LSUN_crop_test = model_l2.predict(x_LSUN_crop_test)
result_LSUN_resize_test = model_l2.predict(x_LSUN_resize_test)
result_iSUN_test = model_l2.predict(x_iSUN_test)

result_Imagenet_crop_test_base = basic_model.predict(x_Imagenet_crop_test)
# result_Imagenet_resize_test_base = basic_model.predict(x_Imagenet_resize_test)
result_LSUN_crop_test_base = basic_model.predict(x_LSUN_crop_test)
result_LSUN_resize_test_base = basic_model.predict(x_LSUN_resize_test)
result_iSUN_test_base = basic_model.predict(x_iSUN_test)


# # Multi-layer Ensemble
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore", category=FutureWarning)

# x_cifar10_train, y_cifar10_train), (x_cifar10_test, y_cifar10_test
result_cifar10_train = model_l2.predict(x_cifar10_train)
result_cifar10_train_base = basic_model.predict(x_cifar10_train)


sample_size = 10000
outlier_detector_l1 = IsolationForest(random_state=r_seed, n_estimators=1000, verbose=0, max_samples=10000,
                                      contamination=0.05)
outlier_detector_l2 = IsolationForest(random_state=r_seed, n_estimators=1000, verbose=0, max_samples=10000,
                                      contamination=0.05)

outlier_detector_l1.fit(result_cifar10_train)
outlier_detector_l2.fit(result_cifar10_train_base)


outlier_Imagenet_crop = outlier_detector_l1.predict(result_Imagenet_crop_test)
# outlier_Imagenet_resize = outlier_detector_l1.predict(result_Imagenet_resize_test)
outlier_LSUN_crop = outlier_detector_l1.predict(result_LSUN_crop_test)
outlier_LSUN_resize = outlier_detector_l1.predict(result_LSUN_resize_test)
outlier_iSUN = outlier_detector_l1.predict(result_iSUN_test)


outlier_Imagenet_crop += outlier_detector_l2.predict(result_Imagenet_crop_test_base)
# outlier_Imagenet_resize += outlier_detector_l2.predict(result_Imagenet_resize_test_base)
outlier_LSUN_crop += outlier_detector_l2.predict(result_LSUN_crop_test_base)
outlier_LSUN_resize += outlier_detector_l2.predict(result_LSUN_resize_test_base)
outlier_iSUN = outlier_detector_l1.predict(result_iSUN_test_base)


outlier_Imagenet_crop[outlier_Imagenet_crop <= 1] = -1
outlier_Imagenet_crop[outlier_Imagenet_crop > 1] = 0
outlier_Imagenet_crop[outlier_Imagenet_crop == 0] = \
    result_Imagenet_crop_test_base.argmax(axis=1)[outlier_Imagenet_crop == 0]

# outlier_Imagenet_resize[outlier_Imagenet_resize <= 1] = -1
# outlier_Imagenet_resize[outlier_Imagenet_resize > 1] = 0
# outlier_Imagenet_resize[outlier_Imagenet_resize == 0] = \
#     result_Imagenet_resize_test_base.argmax(axis=1)[outlier_Imagenet_resize == 0]

outlier_LSUN_crop[outlier_LSUN_crop <= 1] = -1
outlier_LSUN_crop[outlier_LSUN_crop > 1] = 0
outlier_LSUN_crop[outlier_LSUN_crop == 0] = \
    result_LSUN_crop_test_base.argmax(axis=1)[outlier_LSUN_crop == 0]

outlier_LSUN_resize[outlier_LSUN_resize <= 1] = -1
outlier_LSUN_resize[outlier_LSUN_resize > 1] = 0
outlier_LSUN_resize[outlier_LSUN_resize == 0] = \
    result_LSUN_resize_test_base.argmax(axis=1)[outlier_LSUN_resize == 0]

outlier_iSUN[outlier_iSUN <= 1] = -1
outlier_iSUN[outlier_iSUN > 1] = 0
outlier_iSUN[outlier_iSUN == 0] = \
    result_iSUN_test_base.argmax(axis=1)[outlier_iSUN == 0]

print('outlier_Imagenet_crop detection rate:', (outlier_Imagenet_crop == -1).sum() / outlier_Imagenet_crop.shape[0])
# print('outlier_Imagenet_resize detection rate:', (outlier_Imagenet_resize == -1).sum() / outlier_Imagenet_resize.shape[0])
print('outlier_LSUN_crop detection rate:', (outlier_LSUN_crop == -1).sum() / outlier_LSUN_crop.shape[0])
print('outlier_LSUN_resize detection rate:', (outlier_LSUN_resize == -1).sum() / outlier_LSUN_resize.shape[0])
print('outlier_iSUN detection rate:', (outlier_iSUN == -1).sum() / outlier_iSUN.shape[0])


base_pred = result_mnist_test_base.argmax(axis=1)
base_pred[outlier_mnist == -1] = -1

true_label = np.zeros(sample_size * 2)
true_label = true_label - 1
true_label[:sample_size] = y_mnist_test

omniglot_f1 = f1_score(true_label, np.concatenate([base_pred, outlier_omniglot]), average='macro')
mnist_noise_f1 = f1_score(true_label, np.concatenate([base_pred, outlier_mnist_noise]), average='macro')
noise_f1 = f1_score(true_label, np.concatenate([base_pred, outlier_noise]), average='macro')

playground_result = pd.DataFrame()
playground_result['detector'] = ['IsolationForest']
playground_result['omniglot'] = [omniglot_f1]
playground_result['mnist_noise'] = [mnist_noise_f1]
playground_result['noise'] = [noise_f1]

playground_result








#
# x_mnist_test = x_mnist_test.reshape(-1, 28, 28, 1)
# x_omniglot_test = x_omniglot_test.reshape(-1, 28, 28, 1)
# x_mnist_noise_test = x_mnist_noise_test.reshape(-1, 28, 28, 1)
# x_noise_test = x_noise_test.reshape(-1, 28, 28, 1)
#
# mniTr_l3_r = model_l3.predict(x_mnist_train)
# mniTe_l3_r = model_l3.predict(x_mnist_test)
# omnig_l3_r = model_l3.predict(x_omniglot_test)
# mniNo_l3_r = model_l3.predict(x_mnist_noise_test)
# noise_l3_r = model_l3.predict(x_noise_test)
#
#
# idx1 = 1
# idx2 = 2
# plt.scatter(mniTr_l3_r[:5000, idx1], mniTr_l3_r[:5000, idx2])
# plt.scatter(omnig_l3_r[:5000, idx1], omnig_l3_r[:5000, idx2])
# plt.scatter(noise_l3_r[:5000, idx1], noise_l3_r[:5000, idx2])
# plt.ylim([-50, 60])
# plt.xlim([-50, 60])
# plt.show()
#
# plt.scatter(mniTe_l3_r[:5000, idx1], mniTe_l3_r[:5000, idx2])
# plt.ylim([-50, 60])
# plt.xlim([-50, 60])
# plt.show()
#
# plt.scatter(mniTr_l3_r[:5000, idx1], mniTr_l3_r[:5000, idx2])
# plt.ylim([-50, 60])
# plt.xlim([-50, 60])
# plt.show()
#
# plt.scatter(omnig_l3_r[:5000, idx1], omnig_l3_r[:5000, idx2])
# plt.ylim([-50, 60])
# plt.xlim([-50, 60])
# plt.show()
#
# plt.scatter(noise_l3_r[:5000, idx1], noise_l3_r[:5000, idx2])
# plt.ylim([-50, 60])
# plt.xlim([-50, 60])
# plt.show()
#
#
#
# plt.scatter(mniTr_l3_r[:5000, 0], mniTr_l3_r[:5000, 1])
# plt.show()










