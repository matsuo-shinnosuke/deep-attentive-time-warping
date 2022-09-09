
from fastdtw import fastdtw
import torch
import numpy as np
import logging
from utilities import create_path_img

log = logging.getLogger('__main__')


class DatasetMetricLearning(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.index = np.concatenate(
            [np.arange(dataset.N_train_data).repeat(dataset.N_train_data).reshape(-1, 1),
                np.tile(np.arange(dataset.N_train_data), dataset.N_train_data).reshape(-1, 1)], 1)
        self.data1, self.data2 = dataset.train_data, dataset.train_data
        self.label1, self.label2 = dataset.train_label, dataset.train_label

        log.debug('index:\n%s' % self.index)
        log.debug('data1 shape:\n%s' % str(self.data1.shape))
        log.debug('data2 shape:\n%s' % str(self.data2.shape))
        self.len = self.index.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data1 = self.data1[self.index[idx, 0]]
        data2 = self.data2[self.index[idx, 1]]
        label1 = self.label1[self.index[idx, 0]]
        label2 = self.label2[self.index[idx, 1]]
        sim = (label1 == label2)

        data1 = torch.tensor(data1).float()
        data2 = torch.tensor(data2).float()
        sim = torch.tensor(sim).float()

        return data1, data2, sim
