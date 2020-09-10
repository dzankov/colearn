import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
from numpy.linalg import inv
from colearn.utils import GetMB, GetTauMB
from math import ceil


class TauCoopRidge:

    def __init__(self, alpha=0, lmb=1):
    
        self.alpha = alpha
        self.lmb = lmb

    def _add_bias(self, X):
    
        X_new = np.ones((X.shape[0], X.shape[1] + 1))
        X_new[:, 1:] = X
        
        return X_new

    def fit (self, X1, X2, X, yt, yk):
    
        X1 = self._add_bias(X1)
        X2 = self._add_bias(X2)
        X = self._add_bias(X)
        I = np.eye(X.shape[1])
                        
        z1 = inv( 
                  self.alpha * (X2 - X1).T.dot(X2 - X1) + 
                  (1 - self.alpha) * (X.T.dot(X) + self.lmb * I)
                )
                             
        z2 = (
               self.alpha * (X2 - X1).T.dot(yt) + 
               (1 - self.alpha) * X.T.dot(yk)
             )
        
        
        self.w_ = z1.dot(z2)
        
        return self
    

    def predict(self, X):
    
        X = self._add_bias(X)
        
        return X.dot(self.w_)
    

    def predict_const(self, X1, X2):
    
        const = self.predict(X2) - self.predict(X1)
    
        return const
    

    
class TauCoopMLP(nn.Module):

    def __init__(self, inp_dim, hidden_dim, out_dim, alpha=1, lmb=0, batch_size=100, n_epochs=100):
        
        super(TauCoopMLP, self).__init__()
        
        self.inp_dim  = inp_dim
        self.hidden_dim = hidden_dim 
        self.out_dim = out_dim
        
        self.net = nn.Sequential(
            nn.Linear(self.inp_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, self.out_dim)
            )
        
        self.alpha = alpha
        self.lmb = lmb
        self.batch_size = batch_size
        self.n_epochs = n_epochs

    
    def check_input(self, X1, X2, X, yt, yk):
        
        X1 = X1.astype('float32')
        X2 = X2.astype('float32')
        X = X.astype('float32')
        
        yt = yt.astype('float32')
        yk = yk.astype('float32')
        
        if yt.ndim == 1:
            yt = yt.reshape(-1, 1)
            
        if yk.ndim == 1:
            yk = yk.reshape(-1, 1)
        
        return X1, X2, X, yt, yk   
    
    def forward(self, x):
        
        out = self.net(x)
        
        return out
    
    
    def loss(self, yt_pred, yt_true, yk_pred, yk_true):
        
        mse = nn.MSELoss()
        
        tau_loss = mse(yt_pred, yt_true)
        acid_loss = mse(yk_pred, yk_true)
        
        loss = self.alpha * tau_loss + (1 - self.alpha) * acid_loss
        
        return loss
    
    def get_mimi_bathes(self, X1, X2, X, yt, yk):
        
        n_mb = ceil(len(X1) / self.batch_size)
        batch_size = ceil(len(X) / n_mb)
        
        X_tau = GetTauMB(X1, X2, yt)
        X_acid = GetMB(X, yk)
        
        tau_mb = DataLoader(X_tau, batch_size=self.batch_size, shuffle=True)
        acid_mb = DataLoader(X_acid, batch_size=batch_size, shuffle=True)
        
        return tau_mb, acid_mb
    
    def fit(self, X1, X2, X, yt, yk):
        
        X1, X2, X, yt, yk = self.check_input(X1, X2, X, yt, yk)
        
        optimizer = optim.Adam(self.parameters())
        
        for n_epoch in range(self.n_epochs): 
  
            tau_mb, acid_mb = self.get_mimi_bathes(X1, X2, X, yt, yk)
        
            for (X1_mb, X2_mb, yt_mb), (X_mb, yk_mb) in zip(tau_mb, acid_mb):
            
                yt_out = self.forward(X2_mb) - self.forward(X1_mb)
                yk_out = self.forward(X_mb)
                loss = self.loss(yt_out, yt_mb, yk_out, yk_mb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
    
    def predict(self, X):
        
        X = torch.from_numpy(X.astype('float32'))
        
        return self.forward(X).detach().numpy()
        
    
    def predict_const(self, X1, X2):
        
        const = self.predict(X2) - self.predict(X1) 
        
        return const
