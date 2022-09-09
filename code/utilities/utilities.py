import os
import numpy as np
import torch
import subprocess
import torchinfo
import re
import datetime
import pytz
import matplotlib.pyplot as plt
from tslearn.preprocessing import TimeSeriesResampler
import logging
log = logging.getLogger('__main__')


def get_date():
    return datetime.datetime.now(
        pytz.timezone('Asia/Tokyo')).strftime('%Y.%m.%d.%H.%M.%S')


def make_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)


def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)  # fix the initial value of the network weight
    torch.cuda.manual_seed(seed)  # for cuda
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True  # choose the determintic algorithm

    log.debug(np.random.rand(3))


def resampling_time_series_data(data, len):
    tsr = TimeSeriesResampler(len)
    data_re = tsr.fit_transform(data)
    return data_re


def create_path_img(path):
    path_img_max = 8
    img = np.zeros((path[-1][0]+1, path[-1][1]+1))
    for i in range(len(path)):
        img[path[i][0]][path[i][1]] = path_img_max
    return img


def determine_batch_size(model, input_shape, device):
    cmd = 'nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader'
    free_memory = str(subprocess.check_output(cmd))
    free_memory = int(re.findall(
        r'([+-]?[0-9]+\.?[0-9]*)', free_memory)[int(device[-1])])
    log.debug('free memory: %d' % free_memory)
    for i in range(100):
        if np.shape(np.shape(input_shape))[0] == 1:
            input = (2**i, )+input_shape
        else:
            input = ()
            for x in input_shape:
                input += ((2**i, )+x, )
        model_summary = str(torchinfo.summary(
            model, input, device=device, verbose=0))
        consumed_memory = re.findall(
            r'([+-]?[0-9]+\.?[0-9]*)', model_summary)[-1]
        consumed_memory = float(consumed_memory)
        log.debug('batch size: %d' % (2**i))
        log.debug('Estimated Total Size (MB): %.2f' % consumed_memory)
        if free_memory < consumed_memory:
            return 2**(i-1)


class TrainingCurve():
    def __init__(self, name, save_path, cfg):
        self.train_value_list, self.val_value_list = [], []
        self.save_path = save_path
        self.name = name
        self.cfg = cfg

    def save(self, train_value=np.nan, val_value=np.nan, show_fig=True, show_per_N=1):
        self.train_value_list.append(train_value)
        self.val_value_list.append(val_value)

        np.save('%s_%s.npy' % (self.save_path, self.name),
                np.array([self.train_value_list, self.val_value_list]))

        if show_fig == True:
            if (len(self.train_value_list) % show_per_N) == 0:
                font_size, fig_size = 15, (10, 5)
                _, ax = plt.subplots(figsize=fig_size, tight_layout=True)
                x = (np.arange(len(self.train_value_list))+1) * \
                    self.cfg.train_iteration
                ax.plot(x, self.train_value_list, color='C0', label='training')
                ax.plot(x, self.val_value_list, color='C1', label='validation')
                ax.set_xlabel('iteration', fontsize=font_size)
                ax.set_ylabel(self.name, fontsize=font_size)
                ax.legend(loc=1, fontsize=font_size)
                ax.grid()
                plt.savefig('%s_%s.pdf' % (self.save_path, self.name))
                plt.savefig('%s_%s.png' % (self.save_path, self.name))
                plt.clf()
                plt.close()


class SaveModel():
    def __init__(self, name, more_or_less, save_path, cfg):
        self.name = name
        self.more_or_less = more_or_less
        self.save_path = save_path
        self.cfg = cfg

        self.epoch = 0
        self.best_epoch = 0
        self.best_value = None

    def save(self, model, value):
        self.epoch += 1

        if self.epoch == 1:
            self.save_new_model(model, value)

        if self.more_or_less == 'more':
            if value >= self.best_value:
                self.remove_old_model(model)
                self.save_new_model(model, value)

        if self.more_or_less == 'less':
            if value <= self.best_value:
                self.remove_old_model(model)
                self.save_new_model(model, value)

    def remove_old_model(self, model):
        os.remove('%s_%s_epoch_%d_%s_%.4f.pkl' %
                  (self.save_path, model.__class__.__name__, self.best_epoch, self.name, self.best_value))

    def save_new_model(self, model, value):
        self.best_epoch, self.best_value = self.epoch, value
        torch.save(model.state_dict(), '%s_%s_epoch_%d_%s_%.4f.pkl' %
                   (self.save_path, model.__class__.__name__, self.best_epoch, self.name, self.best_value))
