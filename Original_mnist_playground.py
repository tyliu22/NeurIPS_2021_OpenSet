#!/usr/bin/env python
# coding: utf-8

# test

'''
Directly copy from file : mnist_playground.ipynb
'''

# # Import Package
import tensorflow as tf
import zipfile
from zipfile import ZipFile
import warnings
from IPython.core.display import display
from PIL import Image
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
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


# for i in range(1):
#     im = Image.fromarray(np.uint8(x_omniglot_test[i] * 255))
#     plt.imshow(im)
#     plt.show()

# for i in range(1):
#     im = Image.fromarray(np.uint8(x_noise_test[i] * 255))
#     plt.imshow(im)
#     plt.show()



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
                                 outputs=basic_model.layers[-2 ].output)

# set layer output as a second Model
model_l3 = tf.keras.models.Model(inputs=basic_model.layers[0].input,
                                 outputs=basic_model.layers[-3].output)

x_mnist_train = x_mnist_train.reshape(-1, 28, 28, 1)
x_mnist_test = x_mnist_test.reshape(-1, 28, 28, 1)


K.set_value(basic_model.optimizer.learning_rate, 0.001)

basic_model.fit(x_mnist_train, y_mnist_train, epochs=5)
basic_model.evaluate(x_mnist_test, y_mnist_test, verbose=2)

# # Making Prediction

x_mnist_test = x_mnist_test.reshape(-1, 28, 28, 1)
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

result_mnist_test_mlayer = np.hstack(
    [result_mnist_test, result_mnist_test_base])
result_omniglot_test_mlayer = np.hstack(
    [result_omniglot_test, result_omniglot_test_base])
result_mnist_noise_test_mlayer = np.hstack(
    [result_mnist_noise_test, result_mnist_noise_test_base])
result_noise_test_mlayer = np.hstack(
    [result_noise_test, result_noise_test_base])

fig, ax = plt.subplots(nrows=2, ncols=5)
counter = 0
for row in ax:
    for col in row:
        col.hist(result_mnist_test_base[:, counter], facecolor='g')
        col.hist(result_omniglot_test_base[:, counter])
        counter += 1
plt.show()

#
fig, ax = plt.subplots(nrows=2, ncols=5)
counter = 0
for row in ax:
    for col in row:
        col.hist(result_mnist_test[:, counter], facecolor='g')
        col.hist(result_omniglot_test[:, counter])
        counter += 1
plt.show()


fig, ax = plt.subplots(nrows=2, ncols=5)
counter = 0
for row in ax:
    for col in row:
        col.hist(result_mnist_test[:, counter], facecolor='g')
        col.hist(result_mnist_noise_test[:, counter])
        counter += 1
plt.show()


fig, ax = plt.subplots(nrows=2, ncols=5)
counter = 0
for row in ax:
    for col in row:
        col.hist(result_mnist_test[:, counter], facecolor='g')
        col.hist(result_noise_test[:, counter])
        counter += 1
plt.show()


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

playground_result



outlier_detector_l1 = IsolationForest(random_state=r_seed, n_estimators=1000, verbose=0, max_samples=10000)
outlier_detector_l2 = IsolationForest(random_state=r_seed, n_estimators=1000, verbose=0, max_samples=10000)

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

playground_result

# # Openset Sample Enrichment
import numpy as np
import math
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import pairwise_distances


# an implementation of Kernel Mean Matchin
# referenres:
#  1. Gretton, Arthur, et al. "Covariate shift by kernel mean matching." Dataset shift in machine learning 3.4 (2009): 5.
#  2. Huang, Jiayuan, et al. "Correcting sample selection bias by unlabeled data." Advances in neural information processing systems. 2006.
def kernel_mean_matching(X, Z, kern='lin', B=1.0, eps=None):
    nx = X.shape[0]
    nz = Z.shape[0]
    if eps == None:
        eps = B / math.sqrt(nz)
    if kern == 'lin':
        K = np.dot(Z, Z.T)
        kappa = np.sum(np.dot(Z, X.T) * float(nz) / float(nx), axis=1)
    elif kern == 'rbf':
        K = compute_rbf(Z, Z)
        kappa = np.sum(compute_rbf(Z, X), axis=1) * float(nz) / float(nx)
    else:
        raise ValueError('unknown kernel')

    K = matrix(K)
    kappa = matrix(kappa)
    G = matrix(np.r_[np.ones((1, nz)), -np.ones((1, nz)), np.eye(nz), -np.eye(nz)])
    h = matrix(np.r_[nz * (1 + eps), nz * (eps - 1), B * np.ones((nz,)), np.zeros((nz,))])

    sol = solvers.qp(K, -kappa, G, h)
    coef = np.array(sol['x'])
    return coef


def compute_rbf(X, Z):
    sigma = pairwise_distances(X).std()
    sigma = sigma ** 2
    print(sigma)
    K = np.zeros((X.shape[0], Z.shape[0]), dtype=float)
    for i, vx in enumerate(X):
        K[i, :] = np.exp(-np.sum((vx - Z) ** 2, axis=1) / (2.0 * sigma))
    return K


def sample_enrichment(r_seed, target_data, sample_size):
    np.random.seed(r_seed)
    domain_max = target_data.max(axis=0)
    domain_min = target_data.min(axis=0)
    domain_dim = target_data.shape[1]

    sample_enri = np.random.random(size=(sample_size, domain_dim))

    domain_gap = (domain_max - domain_min) * 1.2
    domain_mean = (domain_max + domain_min) / 2

    for dim_idx in range(domain_dim):
        sample_enri[:, dim_idx] = sample_enri[:, dim_idx] * domain_gap[
            dim_idx] + domain_mean[dim_idx] - domain_gap[dim_idx] / 2

    sample_coef = kernel_mean_matching(target_data, sample_enri, kern='rbf', B=50)
    return sample_enri, np.squeeze(sample_coef)


x_mnist_test = x_mnist_test.reshape(-1, 28, 28, 1)
x_omniglot_test = x_omniglot_test.reshape(-1, 28, 28, 1)
x_mnist_noise_test = x_mnist_noise_test.reshape(-1, 28, 28, 1)
x_noise_test = x_noise_test.reshape(-1, 28, 28, 1)

mniTr_l3_r = model_l3.predict(x_mnist_train)
mniTe_l3_r = model_l3.predict(x_mnist_test)
omnig_l3_r = model_l3.predict(x_omniglot_test)
mniNo_l3_r = model_l3.predict(x_mnist_noise_test)
noise_l3_r = model_l3.predict(x_noise_test)


idx1 = 1
idx2 = 2
plt.scatter(mniTr_l3_r[:5000, idx1], mniTr_l3_r[:5000, idx2], c='b')
plt.scatter(omnig_l3_r[:5000, idx1], omnig_l3_r[:5000, idx2], c='y')
plt.scatter(noise_l3_r[:5000, idx1], noise_l3_r[:5000, idx2], c='g')
plt.ylim([-50, 60])
plt.xlim([-50, 60])
plt.show()

plt.scatter(mniTe_l3_r[:5000, idx1], mniTe_l3_r[:5000, idx2])
plt.ylim([-50, 60]
plt.xlim([-50, 60])
plt.show()

plt.scatter(mniTr_l3_r[:5000, idx1], mniTr_l3_r[:5000, idx2])
plt.ylim([-50, 60])
plt.xlim([-50, 60])
plt.show()

plt.scatter(omnig_l3_r[:5000, idx1], omnig_l3_r[:5000, idx2])
plt.ylim([-50, 60])
plt.xlim([-50, 60])
plt.show()

plt.scatter(noise_l3_r[:5000, idx1], noise_l3_r[:5000, idx2])
plt.ylim([-50, 60])
plt.xlim([-50, 60])
plt.show()

# In[434]:


plt.scatter(mniTr_l3_r[:5000, 0], mniTr_l3_r[:5000, 1])
plt.show()

q_sample_list = []
q_weight_list = []

for i in range(2):
    temp_enri, temp_coef = sample_enrichment(i, mniTr_l3_r[i * 6000:(i + 1) * 6000], 6000)
    q_sample_list.append(temp_enri)
    q_weight_list.append(temp_coef)

# In[509]:


q_sample = np.vstack(q_sample_list)
q_weight = np.hstack(q_weight_list)

# In[510]:


theta = 0.005
cond1 = (q_weight < theta)
cond2 = (q_weight >= theta)

plt.scatter(q_sample[cond1, 0], q_sample[cond1, 1])
# plt.scatter(omnig_l3_r[:5000, 0], omnig_l3_r[:5000, 1])
plt.show()
plt.scatter(q_sample[cond2, 0], q_sample[cond2, 1])
plt.show()

plt.scatter(q_sample[cond1, 0], q_sample[cond1, 1])
plt.scatter(q_sample[cond2, 0], q_sample[cond2, 1])
plt.show()

# In[550]:


q_sample = np.vstack(q_sample_list)
q_weight = np.hstack(q_weight_list)

beta = 0.5
tau = 0.2
cond1 = (q_weight <= tau)
cond2 = (q_weight >= 2 * tau)
cond3 = (q_weight > tau) & (q_weight < 2 * tau)
q_weight[cond1] = q_weight[cond1] + beta
q_weight[cond3] = (1 - beta / tau) * q_weight[cond3] + 2 * beta

# In[551]:


theta = 1
cond1 = (q_weight < beta)
plt.scatter(q_sample[cond1, 0], q_sample[cond1, 1])
plt.show()

# In[552]:


q_weight.max()


# # NN Customer Loss

# In[9]:


class pq_risk(tf.keras.losses.Loss):

    def __init__(self, model, x_t, x_w, k, is_init):
        super().__init__(name='pq_risk')
        self.model = model
        self.x_t = x_t
        self.x_w = x_w
        self.k = k
        self.is_init = is_init

    def call(self, y_true, y_pred):
        Rs_all_hat = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

        if self.is_init:
            return Rs_all_hat

        y_t_pred = self.model(self.x_t)
        y_true_q = np.zeros(self.x_w.shape) + self.k

        Rt_k_hat = tf.keras.losses.sparse_categorical_crossentropy(y_true_q, y_t_pred)
        Rt_k_hat = tf.math.multiply(tf.convert_to_tensor(self.x_w, dtype=tf.float32), Rt_k_hat)
        Rt_k_hat = tf.reduce_mean(tf.math.abs(Rt_k_hat))

        y_true_p = np.zeros(y_true.shape) + self.k
        Rs_k_hat = tf.keras.losses.sparse_categorical_crossentropy(y_true_p, y_pred)
        Rs_k_hat = tf.reduce_mean(tf.math.abs(Rs_k_hat))

        return 2 * Rs_all_hat + tf.reduce_max([Rt_k_hat - Rs_k_hat, 0])

    # In[10]:


pq_detetor = tf.keras.models.Sequential([
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
    tf.keras.layers.Dense(4),  # layer -3
    tf.keras.layers.Dense(11),  # layer -2
    tf.keras.layers.Activation(activation='softmax')  # layer -1
])

pq_z_layer = tf.keras.models.Model(inputs=pq_detetor.layers[0].input,
                                   outputs=pq_detetor.layers[-3].output)

# In[15]:


x_mnist_train = x_mnist_train.reshape(-1, 28, 28, 1)
x_mnist_test = x_mnist_test.reshape(-1, 28, 28, 1)

# In[16]:


pq_detetor.compile(optimizer='adam',
                   loss=pq_risk(None, None, None, None, True),
                   metrics=['accuracy'])

K.set_value(pq_detetor.optimizer.learning_rate, 0.001)
pq_detetor.fit(x_mnist_train, y_mnist_train * 1.0, epochs=5, batch_size=16)


z = pq_z_layer(x_mnist_train)
print(z.shape)


pq_detetor.compile(optimizer='adam',
                   loss=pq_risk(pq_detetor, x_t, x_w, 10, False),
                   metrics=['accuracy'])

K.set_value(pq_detetor.optimizer.learning_rate, 0.001)
pq_detetor.fit(x_mnist_train, y_mnist_train * 1.0, epochs=5, batch_size=16)


x_t = q_sample
x_w = q_weight

# In[577]:


tf.keras.losses.mean_squared_error


# In[668]:


class auxiliary_risk(tf.keras.losses.Loss):

    def __init__(self, model, x_t, y_w, k):
        super().__init__(name='detector')
        self.model = model
        self.x_t = x_t
        self.y_w = y_w
        self.k = k

    def call(self, y_true, y_pred):
        y_true_ohe = tf.dtypes.cast(y_true, tf.int64)
        y_true_ohe = tf.one_hot(y_true_ohe, self.k + 1, on_value=1.0, off_value=0.0, axis=-1)
        Rs_all_hat = tf.math.square(y_true_ohe - y_pred)
        Rs_all_hat = tf.math.reduce_sum(Rs_all_hat, axis=1)
        Rs_all_hat = tf.reduce_mean(Rs_all_hat)

        y_q_pred = self.model(self.x_t)
        y_true_q = np.zeros(self.y_w.shape) + self.k
        y_true_q_ohe = tf.one_hot(y_true_q, self.k + 1, on_value=1.0, off_value=0.0, axis=-1)

        Rt_k_hat = tf.math.square(y_true_q_ohe - y_q_pred)
        Rt_k_hat = tf.math.reduce_sum(Rt_k_hat, axis=1)
        Rt_k_hat = tf.math.multiply(tf.convert_to_tensor(self.y_w, dtype=tf.float32), Rt_k_hat)
        Rt_k_hat = tf.reduce_mean(Rt_k_hat)

        y_true_q = np.zeros(y_pred.shape[0]) + self.k
        y_true_q_ohe = tf.one_hot(y_true_q, self.k + 1, on_value=1.0, off_value=0.0, axis=-1)
        Rs_k_hat = tf.math.square(y_true_q_ohe - y_pred)
        Rs_k_hat = tf.math.reduce_sum(Rs_k_hat, axis=1)
        Rs_k_hat = tf.reduce_mean(Rs_k_hat)

        #         y_t_pred = self.model(self.x_t)
        #         y_t_pred = tf.math.argmax(y_t_pred, axis=1, output_type=tf.dtypes.int64)
        #         y_true_q = np.zeros(self.y_w.shape) + self.k
        #         y_true_q = y_true_q.astype(int)
        #         Rt_k_hat = tf.keras.losses.mean_squared_error(y_true_q, y_t_pred)

        # #         Rt_k_hat = tf.math.multiply(tf.convert_to_tensor(self.y_w, dtype=tf.float32), Rt_k_hat)
        #         Rt_k_hat = tf.math.multiply(tf.convert_to_tensor(self.y_w, dtype=tf.int64), Rt_k_hat)
        #         Rt_k_hat = tf.reduce_mean(tf.math.abs(Rt_k_hat))

        #         y_true_p = np.zeros(y_true.shape) + self.k
        #         Rs_k_hat = tf.keras.losses.mean_squared_error(y_true_p, y_pred)
        #         Rs_k_hat = tf.reduce_mean(tf.math.abs(Rs_k_hat))
        #         Rt_k_hat = tf.dtypes.cast(Rt_k_hat,tf.float32)

        #         return Rs_all_hat
        return 2 * Rs_all_hat


#         Rs_all_hat = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

#         y_t_pred = self.model(self.x_t)
#         y_true_q = np.zeros(self.y_w.shape) + self.k

#         Rt_k_hat = tf.keras.losses.sparse_categorical_crossentropy(y_true_q, y_t_pred)
#         Rt_k_hat = tf.math.multiply(tf.convert_to_tensor(self.y_w, dtype=tf.float32), Rt_k_hat)
#         Rt_k_hat = tf.reduce_mean(tf.math.abs(Rt_k_hat))

#         y_true_p = np.zeros(y_true.shape) + self.k
#         Rs_k_hat = tf.keras.losses.sparse_categorical_crossentropy(y_true_p, y_pred)
#         Rs_k_hat = tf.reduce_mean(tf.math.abs(Rs_k_hat))

#         return 2*Rs_all_hat + tf.reduce_max([Rt_k_hat - Rs_k_hat, 0])
#         return Rs_all_hat + Rt_k_hat - Rs_k_hat

#         tf.keras.losses.mean_squared_error()
#         return Rt_k_hat - Rs_k_hat



tf.random.set_seed(0)
# basic_model structure
detetor = tf.keras.models.Sequential([
    tf.keras.layers.Dense(4),
    tf.keras.layers.Dense(11),
    tf.keras.layers.Activation(activation='softmax')
])

detetor.compile(optimizer='adam',
                loss=auxiliary_risk(detetor, x_t, x_w, 10),
                metrics=['accuracy'], steps_per_execution=256 * 2)

K.set_value(detetor.optimizer.learning_rate, 0.0001)
detetor.fit(mniTr_l3_r, y_mnist_train * 1.0, epochs=100, batch_size=16)

# In[687]:


K.set_value(detetor.optimizer.learning_rate, 0.000001)
detetor.fit(mniTr_l3_r, y_mnist_train * 1.0, epochs=10, batch_size=16)

# In[671]:


K.set_value(detetor.optimizer.learning_rate, 0.0001)
detetor.fit(mniTr_l3_r, y_mnist_train * 1.0, epochs=5)

# In[672]:


K.set_value(detetor.optimizer.learning_rate, 0.00001)
detetor.fit(mniTr_l3_r, y_mnist_train * 1.0, epochs=5)

# In[569]:


from sklearn.metrics import accuracy_score

y_test_pred = detetor.predict(mniTe_l3_r)
y_test_pred = y_test_pred.argmax(axis=1)
print(accuracy_score(y_mnist_test, y_test_pred))

# In[570]:


y_test_pred = detetor.predict(mniTr_l3_r)
y_test_pred = y_test_pred.argmax(axis=1)
(y_test_pred == 10).sum() / 60000

# In[571]:


# mniTr_l3_r = model_l3.predict(x_mnist_train)
# mniTe_l3_r = model_l3.predict(x_mnist_test)
# omnig_l3_r = model_l3.predict(x_omniglot_test)
# mniNo_l3_r = model_l3.predict(x_mnist_noise_test)
# noise_l3_r = model_l3.predict(x_noise_test)


# In[572]:


y_test_pred = detetor.predict(mniTe_l3_r)
y_test_pred = y_test_pred.argmax(axis=1)
(y_test_pred == 10).sum() / 10000

# In[573]:


y_test_pred = detetor.predict(omnig_l3_r)
y_test_pred = y_test_pred.argmax(axis=1)
(y_test_pred == 10).sum() / 12000

# In[574]:


y_test_pred = detetor.predict(mniNo_l3_r)
y_test_pred = y_test_pred.argmax(axis=1)
(y_test_pred == 10).sum() / 12000

# In[575]:


y_test_pred = detetor.predict(noise_l3_r)
y_test_pred = y_test_pred.argmax(axis=1)
(y_test_pred == 10).sum() / 12000




result_mnist_test = model_l2.predict(x_mnist_test[:sample_size])
result_omniglot_test = model_l2.predict(x_omniglot_test)
result_mnist_noise_test = model_l2.predict(x_mnist_noise_test[:sample_size])
result_noise_test = model_l2.predict(x_noise_test)

result_mnist_test_base = basic_model.predict(x_mnist_test[:sample_size])
result_omniglot_test_base = basic_model.predict(x_omniglot_test)
result_mnist_noise_test_base = basic_model.predict(
    x_mnist_noise_test[:sample_size])
result_noise_test_base = basic_model.predict(x_noise_test)



# # Play Ground


# def auxiliary_risk(x_t, y_t, k, model):
def auxiliary_risk():
    #     def base_loss(y_true, y_pred):
    #         Rs_all_hat = tf.keras.losses.sparse_categorical_crossentropy(
    #             y_true, y_pred)
    # #         loss_2 = tf.reduce_mean(tf.square(y_true-y_pred))
    # #         print(loss_2)
    #         #         print(x_t.shape)
    # #         print(x_t)
    # #         print(model)
    # #         y_t_pred = model.predict(x_t)
    #         #         y_t_pred[y_t_pred==k] = 1001 # inlier
    #         #         y_t_pred[y_t_pred!=k] = 1000 # outlier
    #         #         Rt_k_hat = tf.keras.losses.sparse_categorical_crossentropy(y_t, y_t_pred)

    #         # 1001 # inlier; 1000 - outlier
    # #         y_true_np = tf. make_tensor_proto(y_true)

    # #         y_pred_copy = ops.convert_to_tensor(y_pred).numpy()
    # #         y_true_copy = math_ops.cast(y_true, y_pred.dtype)
    # #         print(y_pred_copy)
    # #         print(y_true_copy)
    # #         y_true_np = y_true_np.numpy()
    # #         y_true_copy = tf.where(tf.equal(y_true, k), 1001.0, 1000.0)
    # #         y_pred_copy = tf.where(tf.equal(y_pred, k), 1001.0, 1000.0)

    # #         Rs_k_hat = tf.keras.losses.sparse_categorical_crossentropy(
    # #             y_true_copy, y_pred_copy)

    # #         del y_true_copy
    # #         del y_pred_copy

    #         return Rs_all_hat

    #         return Rs_all_hat + np.max([Rs_all_hat - Rs_k_hat, 0])
    def wrapper(y_true, y_pred):
        #         y_pred_copy = ops.convert_to_tensor_v2(y_pred)
        # #         y_true_copy = math_ops.cast(y_true, y_pred.dtype)
        #         print(y_pred_copy)
        #         print(y_pred_copy.eval())
        #         if y_true.shape[0] is not None:
        #         y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)

        #         with tf.compat.v1.Session() as sess:
        #             print(y_pred.eval())
        #         print(y_true.op, y_true.value_index)
        mse = tf.math.reduce_mean(tf.square(y_true - y_pred))
        reg = tf.math.reduce_mean(tf.square(0.5 - y_pred))
        print(mse)
        #         scce = tf.keras.losses.SparseCategoricalCrossentropy()
        #         print(scce(y_true, y_pred))
        return mse + reg

    #         return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

    return wrapper


# In[719]:


import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers as KL
from tensorflow.python.keras import models as KM
import numpy as np


class WbceLoss(KL.Layer):
    def __init__(self, **kwargs):
        super(WbceLoss, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """
        # inputs：Input tensor, or list/tuple of input tensors.
        如上，父类KL.Layer的call方法明确要求inputs为一个tensor，或者包含多个tensor的列表/元组
        所以这里不能直接接受多个入参，需要把多个入参封装成列表/元组的形式然后在函数中自行解包，否则会报错。
        """
        # 解包入参
        y_true, y_weight, y_pred = inputs
        # 复杂的损失函数
        bce_loss = K.binary_crossentropy(y_true, y_pred)
        wbce_loss = K.mean(bce_loss * y_weight)
        # 重点：把自定义的loss添加进层使其生效，同时加入metric方便在KERAS的进度条上实时追踪
        self.add_loss(wbce_loss, inputs=True)
        self.add_metric(wbce_loss, aggregation="mean", name="wbce_loss")
        return wbce_loss


# In[720]:


def my_model():
    # input layers
    input_img = KL.Input([64, 64, 3], name="img")
    input_lbl = KL.Input([64, 64, 1], name="lbl")
    input_weight = KL.Input([64, 64, 1], name="weight")

    predict = KL.Conv2D(2, [1, 1], padding="same")(input_img)
    my_loss = WbceLoss()([input_lbl, input_weight, predict])
    model = KM.Model(inputs=[input_img, input_lbl, input_weight], outputs=[predict, my_loss])
    model.compile(optimizer="adam", epochs=5)
    return model


# In[721]:


def get_fake_dataset():
    def map_fn(img, lbl, weight):
        inputs = {"img": img, "lbl": lbl, "weight": weight}
        targets = {}
        return inputs, targets

    fake_imgs = np.ones([500, 64, 64, 3])
    fake_lbls = np.ones([500, 64, 64, 1])
    fake_weights = np.ones([500, 64, 64, 1])
    fake_dataset = tf.data.Dataset.from_tensor_slices(
        (fake_imgs, fake_lbls, fake_weights)
    ).map(map_fn).batch(10)
    return fake_dataset


model = my_model()
my_dataset = get_fake_dataset()
model.fit(my_dataset)


result_mnist_train_subset = result_mnist_train[:5000]
pred_label = result_mnist_train.argmax(axis=1)[:5000]
X_embedded = TSNE(n_components=2).fit_transform(result_mnist_train_subset)


idx1 = 1
idx2 = 3
theta = 0.9
plt.scatter(result_mnist_train_subset[:, idx1], result_mnist_train_subset[:, idx2])
plt.show()

plt.scatter(temp_enri[np.squeeze(temp_coef) < theta, idx1], temp_enri[np.squeeze(temp_coef) < theta, idx2])
plt.scatter(temp_enri[np.squeeze(temp_coef) >= theta, idx1], temp_enri[np.squeeze(temp_coef) >= theta, idx2])
plt.show()

# In[275]:


result_mnist_train_subset = result_mnist_train[:5000]
pred_label = result_mnist_train.argmax(axis=1)[:5000]
X_embedded = TSNE(n_components=2).fit_transform(result_mnist_train_subset)

plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
plt.show()

sample_enri_list = []
sample_coef_list = []

for label in range(10):
    temp_enri, temp_coef = sample_enrichment(label, X_embedded[pred_label == label, :], 500)
    sample_enri_list.append(temp_enri)
    sample_coef_list.append(np.squeeze(temp_coef))

sample_enri = np.vstack(sample_enri_list)
sample_coef = np.hstack(sample_coef_list)

# In[289]:


for i in range(10):
    label = i
    plt.scatter(sample_enri_list[label][:, 0], sample_enri_list[label][:, 1])
    plt.scatter(X_embedded[pred_label == label, 0], X_embedded[pred_label == label, 1])

    plt.xlim([-80, 70])
    plt.ylim([-70, 80])
    plt.show()

# In[318]:


label = 0
theta = 0.9

plt.scatter(sample_enri_list[label][sample_coef_list[label] < theta, 0],
            sample_enri_list[label][sample_coef_list[label] < theta, 1])
plt.scatter(sample_enri_list[label][sample_coef_list[label] >= theta, 0],
            sample_enri_list[label][sample_coef_list[label] >= theta, 1])

plt.xlim([-80, 70])
plt.ylim([-70, 80])
plt.show()



plt.scatter(temp_enri[:, 0], temp_enri[:, 1])
plt.show()

# In[320]:


inlier_idx = sample_coef >= 0.9
oulier_idx = sample_coef < 0.9

plt.scatter(sample_enri[oulier_idx, 0], sample_enri[oulier_idx, 1])
plt.scatter(sample_enri[inlier_idx, 0], sample_enri[inlier_idx, 1])
plt.show()

# In[225]:


sample_enri_list = []
sample_coef_list = []

pred_label = result_mnist_train.argmax(axis=1)

for c_idx in range(1):
    temp_enri, temp_coef = sample_enrichment(c_idx, result_mnist_train[pred_label == c_idx, :], 5000)
    sample_enri_list.append(temp_enri)
    sample_coef_list.append(np.squeeze(temp_coef))

# In[177]:


idx1 = 1
idx2 = 6
plt.scatter(sample_enri_list[0][:, idx1], sample_enri_list[0][:, idx2])
plt.show()
plt.scatter(result_mnist_train[pred_label == 0, idx1], result_mnist_train[pred_label == 0, idx2])
plt.show()
plt.scatter(sample_enri_list[0][:, idx1], sample_enri_list[0][:, idx2])
plt.scatter(result_mnist_train[pred_label == 0, idx1], result_mnist_train[pred_label == 0, idx2])
plt.show()

# In[237]:


(sample_coef_list[0] > top_value).sum()

# In[238]:


(sample_coef_list[0] <= top_value).sum()

# In[255]:


sorted_coef = np.array(sample_coef_list[0])
top_rate = 0.9949999
n = sorted_coef.shape[0]
(-sorted_coef).sort()
top_value = sorted_coef[:int(n * top_rate)][-1]

plt.scatter(sample_enri_list[0][sample_coef_list[0] <= top_value, idx1],
            sample_enri_list[0][sample_coef_list[0] <= top_value, idx2])
plt.scatter(sample_enri_list[0][sample_coef_list[0] > top_value, idx1],
            sample_enri_list[0][sample_coef_list[0] > top_value, idx2])
plt.show()

# In[167]:


from sklearn.manifold import TSNE



X_embedded2 = TSNE(n_components=2).fit_transform(
    np.vstack([result_mnist_train[pred_label == 0, :], sample_enri_list[0]]))

# In[172]:


(pred_label == 0).sum()

# In[175]:


plt.scatter(X_embedded2[:5879, 0], X_embedded2[:5879, 1])
plt.show()
plt.scatter(X_embedded2[5879:12000, 0], X_embedded2[5879:12000, 1])
plt.show()

# In[173]:


# In[128]:


X_embedded = TSNE(n_components=2).fit_transform(np.vstack([result_mnist_train, sample_enri]))



plt.scatter(X_embedded[:2000, 0], X_embedded[:2000, 1])
plt.scatter(X_embedded[10000:12000, 0], X_embedded[10000:12000, 1])
plt.show()




sample_enri




sample_enri.shape



np.unique(sample_coef)




plt.hist(sample_coef)
plt.show()

# In[85]:


from sklearn.ensemble import GradientBoostingClassifier

# In[86]:


outlier_detector_sampled = GradientBoostingClassifier(n_estimators=500,
                                                      learning_rate=0.01,
                                                      max_depth=20,
                                                      random_state=0)
outlier_detector_sampled.fit(sample_enri, sample_label)


outlier_pred_result = outlier_detector_sampled.predict(result_merge_TSNE_embedded[:5000])
(outlier_pred_result == -1).sum()


outlier_pred_result = outlier_detector_sampled.predict(result_merge_TSNE_embedded[5000:])
(outlier_pred_result == -1).sum()


mnist_max = result_mnist_test.max(axis=0)
mnist_min = result_mnist_test.min(axis=0)

print(mnist_max)
print(mnist_min)

tsne_encoder = TSNE(n_components=3)
result_mnist_TSNE_embedded = tsne_encoder.fit_transform(result_mnist_test)

mnist_max = result_mnist_TSNE_embedded.max(axis=0)
mnist_min = result_mnist_TSNE_embedded.min(axis=0)

# In[286]:


sample_size = 20000
sample_dim = mnist_max.shape[0]
uniform_sample_set = np.random.random(size=(sample_size, sample_dim))
uniform_sample_set


mnist_gap = mnist_max * 2 - mnist_min * 2
for dim_idx in range(sample_dim):
    uniform_sample_set[:, dim_idx] = uniform_sample_set[:, dim_idx] * mnist_gap[dim_idx] - mnist_gap[dim_idx] / 2
uniform_sample_set


coef = kernel_mean_matching(result_mnist_TSNE_embedded, uniform_sample_set, kern='rbf', B=10)


uniform_label = np.ones(sample_size)
uniform_label[np.squeeze(coef < 0.8)] = -1
(uniform_label == -1).sum() / sample_size



from sklearn.ensemble import GradientBoostingClassifier

outlier_detector_sampled = GradientBoostingClassifier(n_estimators=500,
                                                      learning_rate=0.01,
                                                      max_depth=5,
                                                      random_state=0)
outlier_detector_sampled.fit(uniform_sample_set, uniform_label)

# In[332]:


outlier_pred_result = outlier_detector_sampled.predict(uniform_sample_set)
(outlier_pred_result == -1).sum()

# In[333]:


outlier_pred_result = outlier_detector_sampled.predict(result_mnist_TSNE_embedded)
(outlier_pred_result == -1).sum()

# In[336]:


result_omniglot_test_tsne = tsne_encoder.transform(result_omniglot_test)


# In[ ]:


def outlier_detection(outlier_detector, mnist, omniglot, mnist_noise, noise):
    outlier_detector.fit(mnist)
    print(mnist.shape)

    outlier_mnist = outlier_detector.predict(mnist)
    outlier_omniglot = outlier_detector.predict(omniglot)
    outlier_mnist_noise = outlier_detector.predict(mnist_noise)
    outlier_noise = outlier_detector.predict(noise)

    print('mnist detection rate:', (outlier_mnist == -1).sum() / outlier_mnist.shape[0])
    print('omniglot detection rate:', (outlier_omniglot == -1).sum() / outlier_omniglot.shape[0])
    print('mnist_noise detection rate:', (outlier_mnist_noise == -1).sum() / outlier_mnist_noise.shape[0])
    print('noise detection rate:', (outlier_noise == -1).sum() / outlier_noise.shape[0])

    true_label = np.ones(10000)
    true_label[5000:] = -1
    omniglot_f1 = f1_score(true_label, np.concatenate([outlier_mnist, outlier_omniglot]), average='macro')
    mnist_noise_f1 = f1_score(true_label, np.concatenate([outlier_mnist, outlier_mnist_noise]), average='macro')
    noise_f1 = f1_score(true_label, np.concatenate([outlier_mnist, outlier_noise]), average='macro')

    playground_result = pd.DataFrame()
    playground_result['detector'] = ['LocalOutlierFactor']
    playground_result['omniglot'] = [omniglot_f1]
    playground_result['mnist_noise'] = [mnist_noise_f1]
    playground_result['noise'] = [noise_f1]

    return playground_result

