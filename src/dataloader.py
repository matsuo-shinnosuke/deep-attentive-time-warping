import torch
from torchvision import transforms
import numpy as np
from fastdtw import fastdtw

from utils import load_ucr_dataset

def create_path_img(path):
    length = path[-1][0]+1
    img = np.zeros((length, length))
    for i in range(len(path)):
        img[path[i][0]][path[i][1]] = 8
    return img

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y, stage='pre_training'):
        self.X, self.y, self.stage = X, y, stage
        
        N = self.X.shape[0]
        self.pair_index = np.concatenate(
            [np.arange(N).repeat(N).reshape(-1, 1),np.tile(np.arange(N), N).reshape(-1, 1)], 1)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0, 1),
        ])
        
    def __len__(self):
        return self.pair_index.shape[0]

    def __getitem__(self, idx):
        X1, X2 = self.X[self.pair_index[idx, 0]], self.X[self.pair_index[idx, 1]]
        y1, y2 = self.y[self.pair_index[idx, 0]], self.y[self.pair_index[idx, 1]]
        sim = int(y1 == y2)

        X1, X2 = self.transform(X1)[0].float(), self.transform(X2)[0].float()
        y1, y2 = torch.tensor(y1).long(), torch.tensor(y2).long()
        sim = torch.tensor(sim).float()

        if self.stage == 'pre-training':
            path = create_path_img(fastdtw(X1, X2, dist=2)[1])
            path = torch.tensor(path).float()
            return X1, X2, y1, y2, sim, path
        else:
            return X1, X2, y1, y2, sim
            
        
    
class BalancedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, y, positive_ratio=1, negative_ratio=2, iteration=100, batch_size=64):
        self.y = y

        self.positive_ratio = positive_ratio
        self.negative_ratio = negative_ratio
        self.iteration = iteration
        self.batch_size = batch_size

        N = self.y.shape[0]
        self.pair_index = np.concatenate(
            [np.arange(N).repeat(N).reshape(-1, 1),np.tile(np.arange(N), N).reshape(-1, 1)], 1)
        self.labels = (y[self.pair_index[:, 0]] == y[self.pair_index[:, 1]]).astype(int)
        self.labels_set = [0, 1]
        self.label_to_indices = {label: np.where(self.labels == label)[0] for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}

    def __iter__(self):
        count = 0
        while count < self.iteration:
            indices = []

            ## negative pair
            n_negative_pair = int(self.batch_size*self.negative_ratio/(self.positive_ratio + self.negative_ratio))
            indices.extend(self.label_to_indices[0][self.used_label_indices_count[0]:self.used_label_indices_count[0] + n_negative_pair])
            self.used_label_indices_count[0] += n_negative_pair

            if self.used_label_indices_count[0] + n_negative_pair > len(self.label_to_indices[0]):
                np.random.shuffle(self.label_to_indices[0])
                self.used_label_indices_count[0] = 0

            ## positive pair
            n_positive_pair = self.batch_size-n_negative_pair
            indices.extend(self.label_to_indices[1][self.used_label_indices_count[1]:self.used_label_indices_count[1] + n_positive_pair])
            self.used_label_indices_count[1] += n_positive_pair

            if self.used_label_indices_count[1] + n_positive_pair > len(self.label_to_indices[1]):
                np.random.shuffle(self.label_to_indices[1])
                self.used_label_indices_count[1] = 0

            np.random.shuffle(indices)
            yield indices

            count += 1

    def __len__(self):
        return self.iteration
    
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_ucr_dataset(dataset_name='Adiac')
    
    dataset = Dataset(X_train, y_train, stage='pre-training')
    batch_sampler = BalancedBatchSampler(
        y_train, positive_ratio=1, negative_ratio=2, iteration=100, batch_size=64)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_sampler=batch_sampler, num_workers=2)
    
    X1, X2, y1, y2, sim, path = next(iter(dataloader))
    print(X1.size(), X2.size())
    print(y1.size(), y2.size())
    print(sim.size(), path.size())
    
