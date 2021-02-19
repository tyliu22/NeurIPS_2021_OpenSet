import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest


def AUROC_score(train_data_last_layer, train_data_hidden, num_train_sample,
                test_data_last_layer, test_data_hidden, num_test_sample,
                r_seed=0, n_estimators=1000, verbose=0,
                max_samples=10000, contamination=0.01):
    """
    The AUROC score Function
    =======================
    Input: (train_data, test_data, outlier_data)
        train_data      [tensor]: embeded training dataset
        test_data       [tensor]: embeded testing dataset
        num_data_sample         : label of testing dataset
    data type:
    """
    outlier_detector_l1 = IsolationForest(random_state=r_seed, n_estimators=1000, verbose=0, max_samples=10000,
                                          contamination=0.1)
    outlier_detector_l2 = IsolationForest(random_state=r_seed, n_estimators=1000, verbose=0, max_samples=10000,
                                          contamination=0.1)

    train_data_hidden = train_data_hidden.cpu().numpy()
    train_data_last_layer = train_data_last_layer.cpu().numpy()
    # data argument
    print('===> AUROC Detector Fit: Start')
    outlier_detector_l1.fit(train_data_hidden)
    outlier_detector_l2.fit(train_data_last_layer)

    # **************** Tensor2numpy **************** #
    test_data_hidden = test_data_hidden.cpu().numpy()
    test_data_last_layer = test_data_last_layer.cpu().numpy()

    # **************** outlier predict **************** #
    train_label = np.ones(num_train_sample, dtype=int)
    test_label = np.zeros(num_test_sample, dtype=int)
    total_label = np.append(train_label, test_label)

    print('===> AUROC Detector Decision Score: Start')
    train_score_hidden = outlier_detector_l1.decision_function(train_data_hidden)
    test_score_hidden = outlier_detector_l1.decision_function(test_data_hidden)
    train_score_last_layer = outlier_detector_l2.decision_function(train_data_last_layer)
    test_score_last_layer = outlier_detector_l2.decision_function(test_data_last_layer)

    train_score = np.minimum(train_score_hidden, train_score_last_layer)
    test_score = np.minimum(test_score_hidden, test_score_last_layer)
    total_data_score = np.append(train_score, test_score)

    AUROC_score = roc_auc_score(total_label, total_data_score)
    print('Outlier AUROC Score:', AUROC_score)

    print('===> AUROC_score End')
