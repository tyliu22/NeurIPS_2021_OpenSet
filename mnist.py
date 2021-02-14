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
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.ensemble import IsolationForest
from keras import backend as K

tf.random.set_seed(0)
r_seed = 0
np.random.seed(r_seed)
omniglot_data_path = '/home/tianyliu/Data/OpenSet/Dataset/omniglot/python/images_evaluation.zip'
sample_size = 10000


# # Preparing Data
# ## MNIST and MNIST-Noise

mnist = tf.keras.datasets.mnist
(x_mnist_train, y_mnist_train), (x_mnist_test,
                                 y_mnist_test) = mnist.load_data()
x_mnist_train, x_mnist_test = x_mnist_train / 255.0, x_mnist_test / 255.0

# introduce noise to mnist
x_mnist_noise_test = x_mnist_test.copy()
random_noise = np.random.uniform(
    0, 1, x_mnist_noise_test[np.where(x_mnist_noise_test == 0)].shape[0])
x_mnist_noise_test[np.where(x_mnist_noise_test == 0)] = random_noise


# for i in range(1):
#     im = Image.fromarray(np.uint8(x_mnist_test[i] * 255))
#     plt.imshow(im)
#     plt.show()
# for i in range(1):
#     im = Image.fromarray(np.uint8(x_mnist_noise_test[i] * 255))
#     plt.imshow(im)
#     plt.show()


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

x_noise_test = np.random.uniform(0, 1, (sample_size, 28, 28))



'''
    Build Classifcation Model
'''
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


basic_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=100,
                           kernel_size=(3, 3),
                           activation="relu",
                           input_shape=(28, 28, 1)),
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

x_mnist_train = x_mnist_train.reshape(-1, 28, 28, 1)
x_mnist_test = x_mnist_test.reshape(-1, 28, 28, 1)


K.set_value(basic_model.optimizer.learning_rate, 0.001)

basic_model.fit(x_mnist_train, y_mnist_train, epochs=5)
basic_model.evaluate(x_mnist_test, y_mnist_test, verbose=2)

# # Making Prediction

# x_mnist_test = x_mnist_test.reshape(-1, 28, 28, 1)
x_omniglot_test = x_omniglot_test.reshape(-1, 28, 28, 1)
x_mnist_noise_test = x_mnist_noise_test.reshape(-1, 28, 28, 1)
x_noise_test = x_noise_test.reshape(-1, 28, 28, 1)

result_mnist_test = model_l2.predict(x_mnist_test[:sample_size])
result_omniglot_test = model_l2.predict(x_omniglot_test)
result_mnist_noise_test = model_l2.predict(x_mnist_noise_test[:sample_size])
result_noise_test = model_l2.predict(x_noise_test)

result_mnist_test_base = basic_model.predict(x_mnist_test[:sample_size])
result_omniglot_test_base = basic_model.predict(x_omniglot_test)
result_mnist_noise_test_base = basic_model.predict(
    x_mnist_noise_test[:sample_size])
result_noise_test_base = basic_model.predict(x_noise_test)




# result_mnist_test_mlayer = np.hstack(
#     [result_mnist_test, result_mnist_test_base])
# result_omniglot_test_mlayer = np.hstack(
#     [result_omniglot_test, result_omniglot_test_base])
# result_mnist_noise_test_mlayer = np.hstack(
#     [result_mnist_noise_test, result_mnist_noise_test_base])
# result_noise_test_mlayer = np.hstack(
#     [result_noise_test, result_noise_test_base])

# fig, ax = plt.subplots(nrows=2, ncols=5)
# counter = 0
# for row in ax:
#     for col in row:
#         col.hist(result_mnist_test_base[:, counter], facecolor='g')
#         col.hist(result_omniglot_test_base[:, counter])
#         counter += 1
# plt.show()

#
# fig, ax = plt.subplots(nrows=2, ncols=5)
# counter = 0
# for row in ax:
#     for col in row:
#         col.hist(result_mnist_test[:, counter], facecolor='g')
#         col.hist(result_omniglot_test[:, counter])
#         counter += 1
# plt.show()


# fig, ax = plt.subplots(nrows=2, ncols=5)
# counter = 0
# for row in ax:
#     for col in row:
#         col.hist(result_mnist_test[:, counter], facecolor='g')
#         col.hist(result_mnist_noise_test[:, counter])
#         counter += 1
# plt.show()


# fig, ax = plt.subplots(nrows=2, ncols=5)
# counter = 0
# for row in ax:
#     for col in row:
#         col.hist(result_mnist_test[:, counter], facecolor='g')
#         col.hist(result_noise_test[:, counter])
#         counter += 1
# plt.show()


# # Multi-layer Ensemble
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore", category=FutureWarning)


result_mnist_train = model_l2.predict(x_mnist_train)
result_mnist_train_base = basic_model.predict(x_mnist_train)


sample_size = 10000
outlier_detector_l1 = IsolationForest(random_state=r_seed, n_estimators=1000, verbose=0, max_samples=10000,
                                      contamination=0.05)
outlier_detector_l2 = IsolationForest(random_state=r_seed, n_estimators=1000, verbose=0, max_samples=10000,
                                      contamination=0.05)

outlier_detector_l1.fit(result_mnist_train)
outlier_detector_l2.fit(result_mnist_train_base)

outlier_mnist = outlier_detector_l1.predict(result_mnist_test)
outlier_omniglot = outlier_detector_l1.predict(result_omniglot_test)
outlier_mnist_noise = outlier_detector_l1.predict(result_mnist_noise_test)
outlier_noise = outlier_detector_l1.predict(result_noise_test)

outlier_mnist += outlier_detector_l2.predict(result_mnist_test_base)
outlier_omniglot += outlier_detector_l2.predict(result_omniglot_test_base)
outlier_mnist_noise += outlier_detector_l2.predict(result_mnist_noise_test_base)
outlier_noise += outlier_detector_l2.predict(result_noise_test_base)

outlier_mnist[outlier_mnist <= 1] = -1
outlier_mnist[outlier_mnist > 1] = 0
outlier_mnist[outlier_mnist == 0] = result_mnist_test_base.argmax(axis=1)[outlier_mnist == 0]

outlier_omniglot[outlier_omniglot <= 1] = -1
outlier_omniglot[outlier_omniglot > 1] = 0
outlier_omniglot[outlier_omniglot == 0] = result_omniglot_test_base.argmax(axis=1)[outlier_omniglot == 0]

outlier_mnist_noise[outlier_mnist_noise <= 1] = -1
outlier_mnist_noise[outlier_mnist_noise > 1] = 0
outlier_mnist_noise[outlier_mnist_noise == 0] = result_mnist_noise_test_base.argmax(axis=1)[outlier_mnist_noise == 0]

outlier_noise[outlier_noise <= 1] = -1
outlier_noise[outlier_noise > 1] = 0
outlier_noise[outlier_noise == 0] = result_noise_test_base.argmax(axis=1)[outlier_noise == 0]

print('mnist detection rate:', (outlier_mnist == -1).sum() / outlier_mnist.shape[0])
print('omniglot detection rate:', (outlier_omniglot == -1).sum() / outlier_omniglot.shape[0])
print('mnist_noise detection rate:', (outlier_mnist_noise == -1).sum() / outlier_mnist_noise.shape[0])
print('noise detection rate:', (outlier_noise == -1).sum() / outlier_noise.shape[0])

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


print('detector:IsolationForest')
print('omniglot detection  f1 score:', omniglot_f1)
print('mnist_noise detection f1 score:', mnist_noise_f1)
print('noise detection f1 score:', noise_f1)

print('End')








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










