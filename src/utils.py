import torch
from torchvision import transforms
import numpy as np
from tslearn.datasets import UCR_UEA_datasets
import collections
from tqdm import tqdm

from model import ContrastiveLoss

def reproductibility(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)  # fix the initial value of the network weight
    torch.cuda.manual_seed(seed)  # for cuda
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True  # choose the determintic algorithm

def load_ucr_dataset(dataset_name):
    ucr = UCR_UEA_datasets()
    dataset_list = ucr.list_datasets()
    assert dataset_name in dataset_list, 'Not avilable dataset'

    X_train, y_train, X_test, y_test = ucr.load_dataset(dataset_name)
    return X_train, y_train, X_test, y_test

class kNearestNeighbor():
    def __init__(self, X_ref, y_ref, X_test, y_test, 
                 k=3, tau=1, batch_size=16, devive='cuda:0') -> None:
        self.X_ref, self.y_ref = X_ref, y_ref
        self.X_test, self.y_test = X_test, y_test
        self.k, self.tau = k, tau
        self.batch_size, self.device = batch_size, devive

        N_ref = self.X_ref.shape[0]
        sampling_size = 1000
        if N_ref > sampling_size:
            sampling_idx = np.arange(N_ref)
            np.random.shuffle(sampling_idx)
            sampling_idx = sampling_idx[:sampling_size]
            self.X_ref, self.y_ref = self.X_ref[sampling_idx], self.y_ref[sampling_idx]

    def prediction(self, model):
        model.eval()

        neighbor_list, loss_list = [], []
        pred_list = []

        for i in tqdm(range(self.X_test.shape[0]), leave=False):
            neighbor, loss = self.cal_dist(
                model, self.X_test[i], self.y_test[i], self.X_ref, self.y_ref)
            neighbor_list.append(neighbor)
            loss_list.append(loss)

        for i in range(self.X_test.shape[0]):
            result = neighbor_list[i][:self.k]
            c = collections.Counter(result)
            pred = c.most_common()[0][0]
            pred_list.append(pred)

        pred = np.array(pred_list)

        return pred
    
    def cal_dist(self, model, X_test, y_test, X_ref, y_ref):
        dist_list = []
        loss_list = []

        test_dataset = Dataset(X_test, y_test, X_ref, y_ref)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, num_workers=4)
        loss_function = ContrastiveLoss(self.tau)

        with torch.no_grad():
            for data1, data2, sim in tqdm(test_loader, leave=False):
                data1, data2 = data1.to(self.device), data2.to(self.device)
                sim = sim.to(self.device)
                pred_path = model(data1, data2)
                loss, d = loss_function(pred_path, data1, data2, sim)
                dist_list.extend(d.cpu().data.numpy())
                loss_list.append(loss.item())

        dist_list = np.array(dist_list)

        # ASC
        index = np.argsort(dist_list)
        # DESC
        # index = np.argsort(dist_list)[::-1]

        neighbor = y_ref[index]

        return neighbor, np.mean(np.array(loss_list))


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X_test, y_test, X_ref, y_ref):
        self.X_test, self.y_test = X_test, y_test
        self.X_ref, self.y_ref = X_ref, y_ref

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0, 1),
        ])
        
    def __len__(self):
        return len(self.y_ref)

    def __getitem__(self, idx):
        X_test = self.transform(self.X_test)[0].float()
        X_ref = self.transform(self.X_ref[idx])[0].float()
        sim = torch.tensor(int(self.y_test == self.y_ref[idx])).float()

        return X_test, X_ref, sim


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_ucr_dataset(dataset_name='Adiac')
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
