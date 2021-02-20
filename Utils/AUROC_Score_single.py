import numpy as np
import torch

import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest


def AUROC_score(train_data_last_layer, train_data_hidden, num_train_sample,
                test_data_last_layer, test_data_hidden, num_test_sample,
                r_seed=0, n_estimators=1000, verbose=0, max_samples=10000, contamination=0.01):
    """
    The AUROC score Function
    =======================
    Input: (train_data, test_data, outlier_data)
        train_data      [tensor]: embeded training dataset
        test_data       [tensor]: embeded testing dataset
        num_data_sample         : label of testing dataset
    data type:
    """
    # outlier_detector_l1 = IsolationForest(random_state=r_seed, n_estimators=n_estimators, verbose=verbose, max_samples=max_samples,
    #                                       contamination=contamination)
    outlier_detector_l2 = IsolationForest(random_state=r_seed, n_estimators=n_estimators, verbose=verbose, max_samples=max_samples,
                                          contamination=contamination)

    # train_data_hidden = train_data_hidden.cpu().numpy()
    train_data_last_layer = train_data_last_layer.cpu().numpy()
    # data argument
    print('===> AUROC Detector Fit: Start')
    # outlier_detector_l1.fit(train_data_hidden)
    outlier_detector_l2.fit(train_data_last_layer)

    # **************** Tensor2numpy **************** #
    # test_data_hidden = test_data_hidden.cpu().numpy()
    test_data_last_layer = test_data_last_layer.cpu().numpy()


    print('===> outlier dataset prediction')
    # outlier predict
    # outlier_train_hidden = outlier_detector_l1.predict(train_data_hidden)
    outlier_train_last_layer = outlier_detector_l2.predict(train_data_last_layer)
    outlier_train = outlier_train_last_layer

    # outlier_test_hidden = outlier_detector_l1.predict(test_data_hidden)
    outlier_test_last_layer = outlier_detector_l2.predict(test_data_last_layer)
    outlier_test = outlier_test_last_layer

    # print('outlier testing hidden inlier number', np.sum(outlier_test_hidden == 1))

    # **************** outlier predict final **************** #
    # outlier_train[outlier_train <= 1] = -1
    # outlier_train[outlier_train > 1] = 0
    #
    # outlier_test[outlier_test <= 1] = -1
    # outlier_test[outlier_test > 1] = 0

    # **************** Print predict result **************** #
    print('Training dataset outlier detection rate:', (outlier_train == -1).sum() / outlier_train.shape[0])
    print('Testing dataset outlier detection rate:', (outlier_test == -1).sum() / outlier_test.shape[0])


    # **************** outlier predict **************** #
    train_label = np.ones(num_train_sample, dtype=int)
    test_label = np.zeros(num_test_sample, dtype=int)
    total_label = np.append(train_label, test_label)

    print('===> AUROC Detector Decision Score: Start')
    # train_score_hidden = outlier_detector_l1.decision_function(train_data_hidden)
    # test_score_hidden = outlier_detector_l1.decision_function(test_data_hidden)
    train_score_last_layer = outlier_detector_l2.decision_function(train_data_last_layer)
    test_score_last_layer = outlier_detector_l2.decision_function(test_data_last_layer)

    train_score = train_score_last_layer
    test_score =  test_score_last_layer
    total_data_score = np.append(train_score, test_score)

    total_data_score = total_data_score

    AUROC_score = roc_auc_score(total_label, total_data_score)
    print('', contamination)
    print('Outlier AUROC Score:', AUROC_score)

    print('===> AUROC_score End')
