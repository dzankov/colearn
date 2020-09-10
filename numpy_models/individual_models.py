import torch
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from numpy.linalg import inv
from colearn.utils import GetMB


class Ridge:
    
    def __init__(self, lmb=0):
        
        self.lmb = lmb
        
    
    def _add_bias(self, X):
        
        X_new = np.ones((X.shape[0], X.shape[1] + 1))
        X_new[:, 1:] = X
        
        return X_new
    
    
    def fit(self, X, y):
        
        X = self._add_bias(X)
        I = np.eye(X.shape[1])
        
        self.w_ = inv(X.T.dot(X) + self.lmb * I).dot(X.T).dot(y)
        
        return self
    
    def predict(self, X):
	
        X = self._add_bias(X)
        
        return X.dot(self.w_)
    

	
	
class MLPRegressor(nn.Module):
    
    def __init__(self, inp_dim, hidden_dim, out_dim, lmb=0, batch_size=100, n_epochs=100):
        
        super(MLPRegressor, self).__init__()
        
        self.inp_dim  = inp_dim
        self.hidden_dim = hidden_dim 
        self.out_dim = out_dim
        
        self.net = nn.Sequential(
            nn.Linear(self.inp_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, self.out_dim)
            )
        
        self.lmb = lmb
        self.batch_size = batch_size
        self.n_epochs = n_epochs
    
    
    def check_input(self, X, y):
        
        X = X.astype('float32')
        y = y.astype('float32')
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
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
        
        X, y = self.check_input(X, y)
        
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
        
        X = torch.from_numpy(X.astype('float32'))
        
        return self.forward(X).detach().numpy()
    