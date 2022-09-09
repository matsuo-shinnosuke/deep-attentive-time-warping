import numpy as np
from sklearn.preprocessing import StandardScaler
import json
import pandas as pd
import logging
log = logging.getLogger('__main__')


class get_UCRdataset():
    def __init__(self, cwd, cfg):
        self.cfg = cfg
        self.load_dataset(cwd, self.cfg.dataset_path)
        log.debug('train_data shape: %s' % str(self.train_data.shape))
        log.debug('train_label shape: %s' %
                  str(self.train_label.shape))
        log.debug('train_data.max: %.4f, train_data.min: %.4f' %
                  (self.train_data.max(), self.train_data.min()))
        log.debug('train_data.mean: %.4f, train_data.std: %.4f' %
                  (self.train_data.mean(), self.train_data.std()))
        log.debug('val_data shape: %s' % str(self.val_data.shape))
        log.debug('val_label shape: %s' % str(self.val_label.shape))
        log.debug('test_data shape: %s' % str(self.test_data.shape))
        log.debug('test_label shape: %s' % str(self.test_label.shape))
        log.debug('N_train_data: %d, N_val_data: %d, N_test_data: %d' %
                  (self.N_train_data, self.N_val_data, self.N_test_data))
        log.debug('length: %d, channel: %d' %
                  (self.length, self.channel))

    def load_dataset(self, cwd, dataset_path):
        with open(cwd+'/UCR_dataset_name.json', 'r') as fp:
            UCR_dataset_name_list = json.load(fp)
        self.dataset_name = UCR_dataset_name_list[str(self.cfg.dataset.ID)]

        read_train_data = pd.read_table('%sUCRArchive_2018/%s/%s_TRAIN.tsv' %
                                        (cwd+dataset_path, self.dataset_name, self.dataset_name), header=None).values
        self.train_data = read_train_data[:, 1:, None]
        self.train_label = read_train_data[:, 0]

        read_test_data = pd.read_table('%sUCRArchive_2018/%s/%s_TEST.tsv' %
                                       (cwd+dataset_path, self.dataset_name, self.dataset_name), header=None).values
        self.test_data = read_test_data[:, 1:, None]
        self.test_label = read_test_data[:, 0]

        # self.train_data, self.train_label, self.test_data, self.test_label = UCR_dataset.load_dataset(
        #     self.dataset_name)
        self.N_train_data, self.N_test_data = self.train_data.shape[0], self.test_data.shape[0]
        self.length = self.train_data.shape[1]
        self.channel = self.train_data.shape[2]

        self.split_train_val()

        if self.sample_train_data:
            if self.cfg.dataset.max_train_data < self.N_train_data:
                self.sample_train_data()

        if self.cfg.dataset.standardization:
            self.standardization()

    def split_train_val(self):
        np.random.seed(self.cfg.seed)
        suffle_index = np.random.permutation(np.arange(self.N_train_data))
        self.N_val_data = int(
            self.N_train_data*(1-self.cfg.dataset.train_val_split_ration_train))
        self.N_train_data -= self.N_val_data
        self.val_data = self.train_data[suffle_index][self.N_train_data:]
        self.val_label = self.train_label[suffle_index][self.N_train_data:]
        self.train_data = self.train_data[suffle_index][: self.N_train_data]
        self.train_label = self.train_label[suffle_index][: self.N_train_data]

    def sample_train_data(self):
        np.random.seed(self.cfg.seed)
        suffle_index = np.random.permutation(np.arange(self.N_train_data))
        self.sampled_train_data = self.train_data[suffle_index][:self.cfg.dataset.max_train_data]
        self.sampled_train_label = self.train_label[suffle_index][:self.cfg.dataset.max_train_data]
        self.N_sampled_train_data = self.sampled_train_data.shape[0]
        log.debug('sampled_train_data shape: %s' %
                  str(self.sampled_train_data.shape))
        log.debug('sampled_train_label shape: %s' %
                  str(self.sampled_train_label.shape))

    def standardization(self):
        scaler = StandardScaler()
        scaler.fit(self.train_data[:, :, 0])
        scaler.transform(self.train_data[:, :, 0])[:, :, None]
        scaler.transform(self.val_data[:, :, 0])[:, :, None]
        scaler.transform(self.test_data[:, :, 0])[:, :, None]
