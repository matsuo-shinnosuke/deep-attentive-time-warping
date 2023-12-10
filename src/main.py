import torch
import numpy as np
from DATW import DATW
from utils import load_ucr_dataset

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

datw.pre_training(X=X_train, y=y_train)
datw.contrastive_learning(X=X_train, y=y_train)
prediction = datw.predict(X_ref=X_train, y_ref=y_train, 
                          X_test=X_test, y_test=y_test)
torch.save(datw.best_model, 'DATW_model.pkl')
print('Accuracy: %.4f' % np.array(prediction==y_test).mean())  

### Distance calculation for a pair ###
X1 = X_train[0] # (legth, channel)
X2 = X_test[0] # (legth, channel)

model = torch.load('DATW_model.pkl')
datw = DATW(device='cuda:0',
            best_model=model)
dist = datw.cal_dist(X1, X2)