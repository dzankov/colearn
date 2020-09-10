import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
from numpy.linalg import inv
from colearn.utils import GetMB


class TensorRidge:
    
    def __init__(self, lmb=0):
        
        self.lmb = lmb
        
    
    def _add_bias(self, X):
        
        X_new = np.ones((X.shape[0], X.shape[1] + 1))
        X_new[:, 1:] = X
        
        return X_new
    
    
    def prepare_input(self, X, y):
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        X = torch.from_numpy(self._add_bias(X).astype('float64')).cuda()
        y = torch.from_numpy(y.astype('float64')).cuda()
        I = torch.from_numpy(np.eye(X.shape[1]).astype('float64')).cuda()
        
        return X, y, I
    
    
    def fit(self, X, y):
        
        
        X, y, I = self.prepare_input(X, y)
        
        self.w_ = torch.inverse(X.transpose(0, 1).mm(X) + self.lmb * I).mm(X.transpose(0, 1)).mm(y)
        
        return self
    
    def predict(self, X):

        X = torch.from_numpy(self._add_bias(X).astype('float64')).cuda()
        
        return X.mm(self.w_).cpu()
		

class CudaMLPRegressor(nn.Module):
    
    def __init__(self, inp_dim, hidden_dim, out_dim, lmb=0, batch_size=100, n_epochs=100):
        
        super(CudaMLPRegressor, self).__init__()
        
        self.inp_dim  = inp_dim
        self.hidden_dim = hidden_dim 
        self.out_dim = out_dim
        
        self.net = nn.Sequential(
            nn.Linear(self.inp_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, self.out_dim))
        
        self.lmb = lmb
        self.batch_size = batch_size
        self.n_epochs = n_epochs
    
    
    def prepare_input(self, X, y):
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        X = torch.from_numpy(X.astype('float32')).cuda()
        y = torch.from_numpy(y.astype('float32')).cuda()
        
        return X, y
        
    
    def forward(self, x):
        
        out = self.net(x)
        
        return out
    
    def loss(self, y_pred, y_true):
        
        mse = nn.MSELoss()
        loss = mse(y_pred, y_true)
        
        return loss
    
    def get_mimi_bathes(self, X, y):
        
        X_data = GetMB(X, y)
        mb = DataLoader(X_data, batch_size=self.batch_size, shuffle=True)
        
        return mb
    
    def fit(self, X, y):
        
        X, y = self.prepare_input(X, y)
        
        optimizer = optim.Adam(self.parameters(), weight_decay=self.lmb)
        
        for epoch in range(self.n_epochs): 
  
            mb = self.get_mimi_bathes(X, y)
        
            for X_mb, y_mb in mb:
            
                y_out = self.forward(X_mb)
                loss = self.loss(y_out, y_mb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
    
    def predict(self, X):
        
        X = torch.from_numpy(X.astype('float32')).cuda()
        
        return self.forward(X).cpu().detach().numpy()