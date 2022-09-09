
import torch
import numpy as np
import logging
log = logging.getLogger('__main__')


class BalancedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, dataset, data_type, positive_ratio, negative_ratio, cfg):
        self.data_type = data_type

        if self.data_type == 'train':
            self.index = np.concatenate(
                [np.arange(dataset.N_train_data).repeat(dataset.N_train_data).reshape(-1, 1),
                 np.tile(np.arange(dataset.N_train_data), dataset.N_train_data).reshape(-1, 1)], 1)
            data1_label = dataset.train_label
            data2_label = dataset.train_label
            self.iteration = cfg.train_iteration
        if self.data_type == 'val':
            self.index = np.concatenate(
                [np.arange(dataset.N_val_data).repeat(dataset.N_train_data).reshape(-1, 1),
                 np.tile(np.arange(dataset.N_train_data), dataset.N_val_data).reshape(-1, 1)], 1)
            data1_label = dataset.val_label
            data2_label = dataset.train_label
            self.iteration = cfg.val_iteration

        self.labels = (data1_label[self.index[:, 0]]
                       == data2_label[self.index[:, 1]]).astype(int)
        self.positive_ratio = positive_ratio
        self.negative_ratio = negative_ratio

        log.debug('data type: %s' % self.data_type)
        log.debug('index:\n%s' % self.index)
        log.debug('labels:%s' % self.labels)
        log.debug('label1:%s' % data1_label[self.index[:, 0]])
        log.debug('label2:%s' % data2_label[self.index[:, 1]])
        log.debug('positive_ration: %d' % self.positive_ratio)
        log.debug('negative_ration: %d' % self.negative_ratio)

        self.labels_set = [0, 1]
        self.label_to_indices = {label: np.where(self.labels == label)[
            0] for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.cfg = cfg

    def __iter__(self):
        count = 0
        while count < self.iteration:
            indices = []
            indices.extend(self.label_to_indices[0][self.used_label_indices_count[0]:self.used_label_indices_count[0] + int(
                self.cfg.batch_size*self.negative_ratio/(self.positive_ratio + self.negative_ratio))])
            self.used_label_indices_count[0] += int(
                self.cfg.batch_size*self.negative_ratio/(self.positive_ratio + self.negative_ratio))
            if self.used_label_indices_count[0] + int(self.cfg.batch_size*self.negative_ratio/(self.positive_ratio + self.negative_ratio)) > len(self.label_to_indices[0]):
                np.random.shuffle(self.label_to_indices[0])
                self.used_label_indices_count[0] = 0

            indices.extend(self.label_to_indices[1][self.used_label_indices_count[1]:self.used_label_indices_count[1] + int(
                self.cfg.batch_size*self.positive_ratio/(self.positive_ratio + self.negative_ratio))])
            self.used_label_indices_count[1] += int(
                self.cfg.batch_size*self.positive_ratio/(self.positive_ratio + self.negative_ratio))
            if self.used_label_indices_count[1] + int(self.cfg.batch_size*self.positive_ratio/(self.positive_ratio + self.negative_ratio)) > len(self.label_to_indices[1]):
                np.random.shuffle(self.label_to_indices[1])
                self.used_label_indices_count[1] = 0

            np.random.shuffle(indices)
            yield indices

            count += 1

    def __len__(self):
        return self.iteration
