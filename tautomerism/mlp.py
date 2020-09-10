import torch
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from colearn.utils import GetMB, GetTauMB
from math import ceil


class ConjugatedRidge:

    def __init__(self, alpha=0, lmb=1, cuda=False):

        self.alpha = alpha
        self.lmb = lmb
        self.cuda = cuda

    def _add_bias(self, X):

        X_new = np.ones((X.shape[0], X.shape[1] + 1))
        X_new[:, 1:] = X

        return X_new

    def array_to_tensor(self, X1, X2, X, YT, YA):

        if YT.ndim == 1:
            YT = YT.reshape(-1, 1)
        if YA.ndim == 1:
            YA = YA.reshape(-1, 1)

        X1 = torch.from_numpy(self._add_bias(X1.astype('float64')))
        X2 = torch.from_numpy(self._add_bias(X2.astype('float64')))
        X = torch.from_numpy(self._add_bias(X.astype('float64')))

        YT = torch.from_numpy(YT.astype('float64'))
        YA = torch.from_numpy(YA.astype('float64'))

        I = torch.from_numpy(np.eye(X.shape[1]))

        if self.cuda:
            X1, X2, X, YT, YA, I = X1.cuda(), X2.cuda(), X.cuda(), YT.cuda(), YA.cuda(), I.cuda()

        return X1, X2, X, YT, YA, I

    def fit(self, X1, X2, X, YT, YA):

        X1, X2, X, YT, YA, I = self.array_to_tensor(X1, X2, X, YT, YA)

        z1 = torch.inverse(self.alpha * (X2 - X1).transpose(0, 1).mm(X2 - X1) +
                           (1 - self.alpha) * X.transpose(0, 1).mm(X) + self.lmb * I)

        z2 = (self.alpha * (X2 - X1).transpose(0, 1).mm(YT) +
              (1 - self.alpha) * X.transpose(0, 1).mm(YA))

        self.w_ = z1.mm(z2)

        return self

    def predict_acidity(self, X):
        X = torch.from_numpy(self._add_bias(X).astype('float64'))
        if self.cuda:
            X = X.cuda()

        y_pred = X.mm(self.w_).cpu()
        return y_pred

    def predict_constant(self, X1, X2):
        const_pred = self.predict_acidity(X2) - self.predict_acidity(X1)
        return const_pred


class TauCoopMLP(nn.Module):

    def __init__(self, inp_dim, hidden_dim, out_dim, alpha=1, lmb=0, batch_size=100, n_epochs=100):
        
        super(TauCoopMLP, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(inp_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, out_dim)
            )
        
        self.lmb = lmb
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.optimizer = optim.Adam(self.parameters())

    
    def check_input(self, X1, X2, X, yT, yK):
        
        if yT.ndim == 1:
            yT = yT.reshape(-1, 1)
            
        if yK.ndim == 1:
            yK = yK.reshape(-1, 1)
        
        X1 = torch.from_numpy(X1.astype('float32')).cuda()
        X2 = torch.from_numpy(X2.astype('float32')).cuda()
        X = torch.from_numpy(X.astype('float32')).cuda()
        
        yT = torch.from_numpy(yT.astype('float32')).cuda()
        yK = torch.from_numpy(yK.astype('float32')).cuda()
        
        return X1, X2, X, yT, yK 
    
    def forward(self, x):
        
        out = self.net(x)
        
        return out
    
    
    def loss(self, yT_pred, yT_true, yK_pred, yK_true):
        
        mse = nn.MSELoss()
        
        tau_loss = mse(yT_pred, yT_true)
        acid_loss = mse(yK_pred, yK_true)
        
        loss = self.alpha * tau_loss + (1 - self.alpha) * acid_loss
        
        return loss
    
    
    def reset_params(self, m):
        if isinstance(m, nn.Linear):
            m.reset_parameters()

            
    def reset_weights(self):
        self.net.apply(self.reset_params)
    
    
    def get_mimi_bathes(self, X1, X2, X, yT, yK):
        
        n_mb = ceil(len(X1) / self.batch_size)
        batch_size = ceil(len(X) / n_mb)
        
        X_tau = GetTauMB(X1, X2, yT)
        X_acid = GetMB(X, yK)
        
        tau_mb = DataLoader(X_tau, batch_size=self.batch_size, shuffle=True)
        acid_mb = DataLoader(X_acid, batch_size=batch_size, shuffle=True)
        
        return tau_mb, acid_mb
    
    def fit(self, X1, X2, X, yT, yK):
        
        X1, X2, X, yT, yK = self.check_input(X1, X2, X, yT, yK)
        
        self.reset_weights()
        
        for n_epoch in range(self.n_epochs): 
  
            tau_mb, acid_mb = self.get_mimi_bathes(X1, X2, X, yT, yK)
        
            for (X1_mb, X2_mb, yT_mb), (X_mb, yK_mb) in zip(tau_mb, acid_mb):
            
                yT_out = self.forward(X2_mb) - self.forward(X1_mb)
                yK_out = self.forward(X_mb)
                loss = self.loss(yT_out, yT_mb, yK_out, yK_mb)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
    
    def predict(self, X):
        
        X = torch.from_numpy(X.astype('float32')).cuda()
        
        return self.forward(X).cpu().detach().numpy()
        
    
    def predict_const(self, X1, X2):
        
        const = self.predict(X2) - self.predict(X1) 
        
        return const