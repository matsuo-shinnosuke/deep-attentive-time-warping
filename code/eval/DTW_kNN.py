from tqdm import tqdm
import numpy as np
import torch
import collections
from fastdtw import fastdtw


def DTW_kNN(dataset, cfg):
    # train_data, train_label = dataset.train_data, dataset.train_label
    train_data, train_label = \
        np.concatenate([dataset.train_data, dataset.val_data]),\
        np.concatenate([dataset.train_label, dataset.val_label])
    test_data, test_label = dataset.test_data, dataset.test_label

    neighbor_data_list, neighbor_label_list, pred_list = [], [], []
    for i in tqdm(range(test_data.shape[0])):
        dist_list = []
        for j in range(train_data.shape[0]):
            dist_list.append(fastdtw(test_data[i], train_data[j], dist=2)[0])
        dist_list = np.array(dist_list)

        neighbor_index = np.argsort(dist_list)[:cfg.kNN_k]
        neighbor_data = train_data[neighbor_index]
        neighbor_data_list.extend(neighbor_data)
        neighbor_label = train_label[neighbor_index]
        neighbor_label_list.extend(neighbor_label)

        c = collections.Counter(neighbor_label)
        pred = c.most_common()[0][0]
        pred_list.append(pred)

    acc = sum(pred_list == test_label)/test_label.shape[0]

    return 1-acc, np.array(pred_list), np.array(neighbor_data_list), np.array(neighbor_label_list)
