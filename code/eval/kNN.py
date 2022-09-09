from tqdm import tqdm
import numpy as np
import torch
import collections
from loss import ContrastiveLoss


def kNN(model, dataset, val_or_test, cfg):
    model.eval()
    if val_or_test == 'val':
        test_data, test_label = dataset.val_data, dataset.val_label
    if val_or_test == 'test':
        test_data, test_label = dataset.test_data, dataset.test_label

    if cfg.sample_train_data and (cfg.dataset.max_train_data < dataset.N_train_data):
        train_data, train_label = dataset.sampled_train_data, dataset.sampled_train_label
    else:
        train_data, train_label = dataset.train_data, dataset.train_label

    neighbor_list, loss_list = [], []
    pred_list = []

    for i in tqdm(range(test_data.shape[0])):
        neighbor, loss = cal_dist(
            model, test_data[i], test_label[i], train_data, train_label, cfg)
        neighbor_list.append(neighbor)
        loss_list.append(loss)

    for i in range(test_data.shape[0]):
        result = neighbor_list[i][:cfg.kNN_k]
        c = collections.Counter(result)
        pred = c.most_common()[0][0]
        pred_list.append(pred)

    acc = sum(pred_list == test_label)/test_label.shape[0]

    return 1-acc, np.mean(np.array(loss_list)), np.array(pred_list), np.array(neighbor_list)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, test_data, test_label, train_data, train_label):
        self.test_data = test_data
        self.test_label = test_label
        self.train_data = train_data
        self.train_label = train_label
        self.len = self.train_label.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data1 = torch.tensor(self.test_data).float()
        data2 = torch.tensor(self.train_data[idx]).float()
        sim = torch.tensor(
            (self.test_label == self.train_label[idx]).astype(int)).float()

        return data1, data2, sim


def cal_dist(model, test_data, test_label, train_data, train_label, cfg):
    dist_list = []
    loss_list = []

    test_dataset = TestDataset(test_data, test_label, train_data, train_label)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    loss_function = ContrastiveLoss(cfg.tau)

    with torch.no_grad():
        for i, (data1, data2, sim) in enumerate(test_loader):
            data1, data2 = data1.to(cfg.device), data2.to(cfg.device)
            sim = sim.to(cfg.device)
            pred_path = model(data1, data2)
            loss, d = loss_function(pred_path, data1, data2, sim)
            dist_list.extend(d.cpu().data.numpy())
            loss_list.append(loss.item())

    dist_list = np.array(dist_list)

    # ASC
    index = np.argsort(dist_list)
    # DESC
    # index = np.argsort(dist_list)[::-1]

    neighbor = train_label[index]

    return neighbor, np.mean(np.array(loss_list))
