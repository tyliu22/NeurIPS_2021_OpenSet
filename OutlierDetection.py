import numpy as np

from sklearn.metrics import f1_score
from sklearn.ensemble import IsolationForest


def OutlierDetection(train_data_last_layer, train_data_hidden,
                     test_data_last_layer, test_data_hidden, test_data_label,
                     abnormal_datasets_name, abnormal_datasets,
                     sample_size=10000, r_seed=0, n_estimators=1000, verbose=0,
                     max_samples=10000, contamination=0.01):
    """
    The Outlier Detection Function
    =======================
    Input: (train_data, test_data, outlier_data)
        train_data      [tensor]: embeded training dataset
        test_data       [tensor]: embeded testing dataset
        test_data_label [tensor]: label of testing dataset

        outlier_datasets_name   : name of outlier datasets
            [outlier_dataset_1, outlier_dataset_2, ... ]
        outlier_dataset [tensor]: embeded outlier datasets
            [outlier_dataset_1_last_layer, outlier_dataset_1_data_hidden, ... ]

    data type:

    """

    # if num_of_para < 2:
    #     raise NameError('Input parameters less than 2, cannot calculate the detection rate')
    # build outlier detection model
    print('===> Outlier detector: starting')
    print('===> Parameter setting threshold:', contamination)
    outlier_detector_last_layer = IsolationForest(random_state=r_seed, n_estimators=n_estimators, verbose=verbose,
                                                  max_samples=max_samples, contamination=contamination)
    outlier_detector_hidden = IsolationForest(random_state=r_seed, n_estimators=n_estimators, verbose=verbose, max_samples=max_samples,
                                              contamination=contamination)

    train_data_last_layer = train_data_last_layer.cpu().numpy()
    train_data_hidden = train_data_hidden.cpu().numpy()

    print('===> outlier detector:training')
    # data argument
    outlier_detector_last_layer.fit(train_data_last_layer)
    outlier_detector_hidden.fit(train_data_hidden)

    print('===> outlier dataset prediction')
    # outlier predict
    outlier_train_hidden = outlier_detector_hidden.predict(train_data_hidden)
    outlier_train_last_layer = outlier_detector_last_layer.predict(train_data_last_layer)
    outlier_train = outlier_train_last_layer + outlier_train_hidden

    # **************** Tensor2numpy **************** #
    test_data_last_layer = test_data_last_layer.cpu().numpy()
    test_data_hidden = test_data_hidden.cpu().numpy()

    outlier_test_hidden = outlier_detector_hidden.predict(test_data_hidden)
    outlier_test_last_layer = outlier_detector_last_layer.predict(test_data_last_layer)
    outlier_test = outlier_test_last_layer + outlier_test_hidden


    # **************** outlier predict final **************** #
    outlier_train[outlier_train <= 1] = -1
    outlier_train[outlier_train > 1] = 0
    outlier_train[outlier_train == 0] = \
        train_data_last_layer.argmax(axis=1)[outlier_train == 0]

    outlier_test[outlier_test <= 1] = -1
    outlier_test[outlier_test > 1] = 0
    outlier_test[outlier_test == 0] = \
        test_data_last_layer.argmax(axis=1)[outlier_test == 0]


    # **************** Print predict result **************** #
    print('Training dataset outlier detection rate:', (outlier_train == -1).sum() / outlier_train.shape[0])
    print('Testing dataset outlier detection rate:', (outlier_test == -1).sum() / outlier_test.shape[0])


    num_of_datasets = len(abnormal_datasets_name)
    for i in range(num_of_datasets):
        # **************** Tensor2numpy **************** #
        abnormal_datasets_last_layer = abnormal_datasets[2*i].cpu().numpy()
        abnormal_datasets_hidden = abnormal_datasets[2*i+1].cpu().numpy()


        outlier_datasets_last_layer = outlier_detector_last_layer.predict(abnormal_datasets_last_layer)
        outlier_datasets_hidden = outlier_detector_hidden.predict(abnormal_datasets_hidden)
        outlier_datasets_sum = outlier_datasets_last_layer + outlier_datasets_hidden


        # **************** outlier predict final **************** #
        outlier_datasets_sum[outlier_datasets_sum <= 1] = -1
        outlier_datasets_sum[outlier_datasets_sum > 1] = 0
        outlier_datasets_sum[outlier_datasets_sum == 0] = \
            abnormal_datasets_last_layer.argmax(axis=1)[outlier_datasets_sum == 0]

        # **************** Print predict result **************** #
        print(abnormal_datasets_name[i], ' outlier detection rate:',
              (outlier_datasets_sum == -1).sum() / outlier_datasets_sum.shape[0])

        # **************** F1 Score result **************** #
        base_pred = test_data_last_layer.argmax(axis=1)
        base_pred[outlier_test == -1] = -1

        # total 20000 samples: 10000 cifar10, 10000 other samples
        true_label = np.zeros(sample_size * 2)
        true_label = true_label - 1
        true_label[:sample_size] = test_data_label

        outlier_datasets_f1 = f1_score(true_label,
                              np.concatenate([base_pred, outlier_datasets_sum]), average='macro')

        print(abnormal_datasets_name[i], 'detection f1 score:', outlier_datasets_f1)

    print('End')

