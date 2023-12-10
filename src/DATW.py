import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
import copy

from dataloader import BalancedBatchSampler, Dataset
from model import BipartiteAttention, ContrastiveLoss
from utils import kNearestNeighbor, reproductibility, load_ucr_dataset

class DATW():
    def __init__(self,
                 batch_size=64,
                 lr=1e-4,
                 pre_training_num_epochs=10,
                 pre_training_iteration=100,
                 contrastive_learning_num_epochs=20,
                 contrastive_learning_iteration=500,
                 tau=1,
                 k=3,
                 seed=42,
                 device='cuda:0',
                 best_model=None) -> None:
        
        self.batch_size = batch_size
        self.lr = lr
        self.pre_training_num_epochs = pre_training_num_epochs
        self.pre_training_iteration = pre_training_iteration
        self.contrastive_learning_num_epochs = contrastive_learning_num_epochs
        self.contrastive_learning_iteration = contrastive_learning_iteration
        self.tau = tau
        self.k = k
        self.seed = seed
        self.device = device

        self.best_model = best_model

    def pre_training(self, X, y):
        reproductibility(self.seed)

        ### data loader ###
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=self.seed)
    
        train_dataset = Dataset(X_train, y_train, stage='pre-training')
        train_batch_sampler = BalancedBatchSampler(
            y_train, positive_ratio=1, negative_ratio=2, 
            iteration=self.pre_training_iteration, batch_size=self.batch_size)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_sampler=train_batch_sampler, num_workers=4)
    
        val_dataset = Dataset(X_val, y_val, stage='pre-training')
        val_batch_sampler = BalancedBatchSampler(
            y_val, positive_ratio=1, negative_ratio=2, 
            iteration=self.pre_training_iteration, batch_size=self.batch_size)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_sampler=val_batch_sampler, num_workers=4)
        
        ### define model & optimizer & loss function ###
        channel = X_train.shape[-1]
        model = BipartiteAttention(input_ch=channel).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, betas=(0.5, 0.999))
        loss_function = nn.MSELoss()    
        
        ### Training ###
        reproductibility(self.seed)
        best_loss = np.inf
        for epoch in range(self.pre_training_num_epochs):
            start_time = time.time()
            ####################################
            model.train()
            train_losses = []
            for X1, X2, _, _, _, path in tqdm(train_loader, leave=False):
                X1, X2 = X1.to(self.device), X2.to(self.device)
                path = path.to(self.device)

                y = model(X1, X2)
                loss = loss_function(F.softmax(y, dim=-1), F.softmax(path, dim=-1))
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_losses.append(loss.item())
                
            ####################################
            model.eval()
            val_losses = []
            with torch.no_grad():
                for X1, X2, _, _, _, path in tqdm(val_loader, leave=False):
                    X1, X2 = X1.to(self.device), X2.to(self.device)
                    path = path.to(self.device)

                    y = model(X1, X2)
                    loss = loss_function(F.softmax(y, dim=-1), F.softmax(path, dim=-1))
                    
                    val_losses.append(loss.item())
            
            ####################################
            end_time = time.time()
            epoch_time = end_time - start_time
            train_loss = torch.mean(torch.FloatTensor(train_losses)).item()
            val_loss = torch.mean(torch.FloatTensor(val_losses)).item()  

            if best_loss > val_loss:
                self.best_model = copy.deepcopy(model)
                best_loss = val_loss

            print('[%d/%d]-ptime: %.2f, train loss: %.6f, val loss: %.6f'
                    % ((epoch + 1), self.pre_training_num_epochs, epoch_time, train_loss, val_loss))  

    def contrastive_learning(self, X, y, is_pre_training=True):
        reproductibility(self.seed)
        
        ### data loader ###
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=self.seed)
    
        train_dataset = Dataset(X_train, y_train, stage='contrastive-learning')
        train_batch_sampler = BalancedBatchSampler(
            y_train, positive_ratio=1, negative_ratio=2, 
            iteration=self.contrastive_learning_iteration, batch_size=self.batch_size)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_sampler=train_batch_sampler, num_workers=4)
        
        ### define model & optimizer & loss function ###
        if is_pre_training:
            model = self.best_model
        else:
            model = BipartiteAttention(input_ch= X_train.shape[-1]).to(self.device)
            
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, betas=(0.5, 0.999))
        loss_function = ContrastiveLoss(tau=self.tau)
        knn = kNearestNeighbor(X_ref=X_train, y_ref=y_train, X_test=X_val, y_test=y_val,
                               k=self.k, tau=self.tau, batch_size=self.batch_size, devive=self.device)
        
        ### Training ###
        reproductibility(self.seed)
        best_acc = 0
        for epoch in range(self.contrastive_learning_num_epochs):
            start_time = time.time()
            ####################################
            model.train()
            train_losses = []
            for X1, X2, _, _, sim in tqdm(train_loader, leave=False):
                X1, X2 = X1.to(self.device), X2.to(self.device)
                sim = sim.to(self.device)

                pred_path = model(X1, X2)
                loss, _ = loss_function(pred_path, X1, X2, sim)
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_losses.append(loss.item())
                
            ####################################
            model.eval()
            pred = knn.prediction(model)
            val_acc = np.array(y_val==pred).mean()
            
            ####################################
            end_time = time.time()
            epoch_time = end_time - start_time
            train_loss = torch.mean(torch.FloatTensor(train_losses)).item()

            if best_acc < val_acc:
                self.best_model = copy.deepcopy(model)
                best_acc = val_acc

            print('[%d/%d]-ptime: %.2f, train loss: %.6f, val acc: %.4f'
                    % ((epoch + 1), self.contrastive_learning_num_epochs, epoch_time, train_loss, val_acc))  
            
    def predict(self, X_ref, y_ref, X_test, y_test):
        model = self.best_model
        knn = kNearestNeighbor(X_ref=X_ref, y_ref=y_ref, X_test=X_test, y_test=y_test,
                               k=self.k, tau=self.tau, batch_size=self.batch_size, devive=self.device)
        return knn.prediction(model)
    
    def cal_dist(self, X1, X2):
        self.best_model.eval()

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0, 1),
        ])
        X1, X2 = transform(X1).float(), transform(X2).float()

        self.best_model = self.best_model.to(self.device)
        X1, X2 = X1.to(self.device), X2.to(self.device)
        pred_path = model(X1, X2)
        pred_path_t = pred_path.transpose(1, 2)
        pred_path = F.softmax(pred_path, dim=2)
        pred_path_t = F.softmax(pred_path_t, dim=2)
        g_X2 = torch.matmul(pred_path_t, X1)
        g_X1 = torch.matmul(pred_path, X2)

        dist1 = (X1 - g_X1).pow(2).view(X1.size(0), -1).mean(dim=1)
        dist2 = (X2 - g_X2).pow(2).view(X2.size(0), -1).mean(dim=1)

        return (dist1+dist2)/2


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_ucr_dataset(dataset_name='ECG200')
    """
    X_train: (num_train_data, length, channel)
    y_train: (num_train_data, )
    X_test: (num_test_data, length, channel)
    y_test: (num_test_data, )
    """

    ### Training & Test ###
    datw = DATW(batch_size=64,
                lr=1e-4,
                pre_training_num_epochs=10,
                pre_training_iteration=100,
                contrastive_learning_num_epochs=20,
                contrastive_learning_iteration=500,
                tau=1,
                k=3,
                seed=42,
                device='cuda:0',
                best_model=None)
    
    datw.pre_training(X_train, y_train)
    datw.contrastive_learning(X_train, y_train)
    prediction = datw.predict(X_train, y_train, X_test, y_test)
    print('Accuracy: %.4f' % np.array(prediction==y_test).mean())  
    torch.save(datw.best_model, 'DATW_model.pkl')

    ### Distance calculation for a pair ###
    X1 = X_train[0] # (legth, channel)
    X2 = X_test[0] # (legth, channel)

    model = torch.load('DATW_model.pkl')
    datw = DATW(device='cuda:0',
                best_model=model)
    dist = datw.cal_dist(X1, X2)
