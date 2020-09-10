import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
from numpy.linalg import inv
from colearn.utils import GetArrheniusBasedMB


class ArrheniusWeightCoopRidgeTorch:
    
    def __init__(self, lmbA=1, lmbE=1, a=1):
        
        self.lmbA = lmbA
        self.lmbE = lmbE
        self.a = a
        
    def _add_bias(self, X):
        
        X_new = np.ones((X.shape[0], X.shape[1] + 1))
        X_new[:, 1:] = X
        
        return X_new
    
    def _get_diag(self, T):
        
        return np.diag(T)
    
    def prepare_input(self, XE, XK, yE, yK, T):
        
        if yE.ndim == 1:
            yE = yE.reshape(-1, 1)
        if yK.ndim == 1:
            yK = yK.reshape(-1, 1)
            
        XE = self._add_bias(XE)
        XK = self._add_bias(XK)
        T = self._get_diag(T)

        XE = torch.from_numpy(XE.astype('float32')).to('cuda')
        XK = torch.from_numpy(XK.astype('float32')).to('cuda')
        yE = torch.from_numpy(yE.astype('float32')).to('cuda')
        yK = torch.from_numpy(yK.astype('float32')).to('cuda')
        T = torch.from_numpy(T.astype('float32')).to('cuda')
        I = torch.eye(XK.shape[1]).to('cuda')
            
        return XE, XK, yE, yK, T, I

    
    def fit(self, XE, XK, yE, yK, T):
        
        XE, XK, yE, yK, T, I = self.prepare_input(XE, XK, yE, yK, T)
        
        a1 = torch.inverse(self.a * XK.transpose(0, 1).mm(XK) + self.lmbA * I)
        a2 = self.a * XK.transpose(0, 1).mm(yK)
        a3 = self.a * XK.transpose(0, 1).mm(T).mm(XK)

        e1 = torch.inverse(self.a * XK.transpose(0, 1).mm(T.transpose(0, 1)).mm(T).mm(XK) + 
                          (1 - self.a) * XE.transpose(0, 1).mm(XE) + self.lmbE * I)

        e2 = (1 - self.a) * XE.transpose(0, 1).mm(yE) - self.a * XK.transpose(0, 1).mm(T.transpose(0, 1)).mm(yK)
        e3 = self.a * XK.transpose(0, 1).mm(T.transpose(0, 1)).mm(XK)

        A, B, C, D = e1.mm(e2), e1.mm(e3), a1.mm(a2), a1.mm(a3)

        self.wA_ = torch.inverse(I - D.mm(B)).mm(C + D.mm(A))
        self.wE_ = torch.inverse(I - B.mm(D)).mm(A + B.mm(C))
        
        return self
    
    def predict_lgA(self, X):
        
        X = torch.from_numpy(self._add_bias(X).astype('float32')).to('cuda')
        
        return X.mm(self.wA_)
    
    def predict_E(self, X):
        
        X = torch.from_numpy(self._add_bias(X).astype('float32')).to('cuda')
        
        return X.mm(self.wE_)
    
    def predict_lgK(self, X, T):
        
        T = torch.from_numpy(self._get_diag(T).astype('float32')).to('cuda')
        lgK = self.predict_lgA(X) - T.mm(self.predict_E(X))
        
        return lgK

class ArrheniusWeightCoopRidge:
    
    def __init__(self, lmbA=1, lmbE=1, a=1):
        
        self.lmbA = lmbA
        self.lmbE = lmbE
        self.a = a
        
    def _add_bias(self, X):
        
        X_new = np.ones((X.shape[0], X.shape[1] + 1))
        X_new[:, 1:] = X
        
        return X_new
    
    def _get_diag(self, T):
        
        return np.diag(T)
    
    def fit(self, XE, XK, yE, yK, T):
        
        XE = self._add_bias(XE)
        XK = self._add_bias(XK)
        T = self._get_diag(T)
        I = np.eye(XK.shape[1])
        
        a1 = inv(self.a * XK.T.dot(XK) + self.lmbA * I)
        a2 = self.a * XK.T.dot(yK)
        a3 = self.a * XK.T.dot(T).dot(XK)
        
        e1 = inv(self.a * XK.T.dot(T.T).dot(T).dot(XK) + (1 - self.a) * XE.T.dot(XE) + self.lmbE * I)
        e2 = (1 - self.a) * XE.T.dot(yE) - self.a * XK.T.dot(T.T).dot(yK)
        e3 = self.a * XK.T.dot(T.T).dot(XK)
        
        A, B, C, D = e1.dot(e2), e1.dot(e3), a1.dot(a2), a1.dot(a3)
        
        self.wA_ = inv(I - D.dot(B)).dot(C + D.dot(A))
        self.wE_ = inv(I - B.dot(D)).dot(A + B.dot(C))
        
        return self
    
    def predict_lgA(self, X):
        
        X = self._add_bias(X)
        
        return X.dot(self.wA_)
    
    def predict_E(self, X):
        
        X = self._add_bias(X)
        
        return X.dot(self.wE_)
    
    def predict_lgK(self, X, T):
        
        T = self._get_diag(T)
        lgK = self.predict_lgA(X) - T.mm(self.predict_E(X))
        
        return lgK



class ArrheniusCoopRidge:
    
    def __init__(self, lmbA=1, lmbE=1):
        
        self.lmbA = lmbA
        self.lmbE = lmbE
        
    def _add_bias(self, X):
        
        X_new = np.ones((X.shape[0], X.shape[1] + 1))
        X_new[:, 1:] = X
        
        return X_new
    
    def _get_diag(self, T):
        
        return np.diag(T)
    
    def fit(self, XA, XE, XK, yA, yE, yK, T):
        
        XA = self._add_bias(XA)
        XE = self._add_bias(XE)
        XK = self._add_bias(XK)
        T = self._get_diag(T)
        I = np.eye(XA.shape[1])
        
        a1 = inv(XA.T.dot(XA) + XK.T.dot(XK) + self.lmbA * I)
        a2 = XA.T.dot(yA) + XK.T.dot(yK)
        a3 = XK.T.dot(T).dot(XK)
        
        e1 = inv((XK.T.dot(T.T).dot(T).dot(XK) + XE.T.dot(XE) + self.lmbE * I))
        e2 = XE.T.dot(yE) - XK.T.dot(T.T).dot(yK)
        e3 = XK.T.dot(T.T).dot(XK)
        
        A, B, C, D = e1.dot(e2), e1.dot(e3), a1.dot(a2), a1.dot(a3)
        
        self.wA_ = inv(I - D.dot(B)).dot(C + D.dot(A))
        self.wE_ = inv(I - B.dot(D)).dot(A + B.dot(C))
        
        return self
    
    def predict_lgA(self, X):
        
        X = self._add_bias(X)
        
        return X.dot(self.wA_)
    
    def predict_E(self, X):
        
        X = self._add_bias(X)
        
        return X.dot(self.wE_)
    
    def predict_lgK(self, X, T):
        
        T = self._get_diag(T)
        lgK = self.predict_lgA(X) - T.dot(self.predict_E(X))
        
        return lgK

	
class ArrheniusBasedRidge:
    
    def __init__(self, lmbA=1, lmbE=1):
        
        self.lmbA = lmbA
        self.lmbE = lmbE
        
        
    def _add_bias(self, X):
        
        X_new = np.ones((X.shape[0], X.shape[1] + 1))
        X_new[:, 1:] = X
        
        return X_new
    
    
    def _get_diag(self, T):
        
        return np.diag(T)
    
    
    def fit(self, XK, yK, T):
        
        XK = self._add_bias(XK)
        T = self._get_diag(T)
        I = np.eye(XK.shape[1])
        
        a1 = inv(XK.T.dot(XK) + self.lmbA * I)
        a2 = XK.T.dot(yK)
        a3 = XK.T.dot(T).dot(XK)

        e1 = inv((XK.T.dot(T.T).dot(T).dot(XK) + self.lmbE * I))
        e2 =  - XK.T.dot(T.T).dot(yK)
        e3 = XK.T.dot(T.T).dot(XK)
        
        A, B, C, D = e1.dot(e2), e1.dot(e3), a1.dot(a2), a1.dot(a3)
        
        self.wA_ = inv(I - D.dot(B)).dot(C + D.dot(A))
        self.wE_ = inv(I - B.dot(D)).dot(A + B.dot(C))
        
        return self
    
    
    def predict_lgA(self, X):
        
        X = self._add_bias(X)
        
        return X.dot(self.wA_)
    
    
    def predict_E(self, X):
        
        X = self._add_bias(X)
        
        return X.dot(self.wE_)
    
    
    def predict_lgK(self, X, T):
        
        T = self._get_diag(T)
        lgK = self.predict_lgA(X) - T.dot(self.predict_E(X))
        
        return lgK
		

	
class SlotNet(nn.Module):
    
    def __init__(self, inp_dim, hidden_dim, out_dim, lmb=0):
        
        super(SlotNet, self).__init__()
        
        self.inp_dim  = inp_dim
        self.hidden_dim = hidden_dim 
        self.out_dim = out_dim
        
        self.net = nn.Sequential(
            nn.Linear(self.inp_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, self.out_dim)
            )
        
        self.lmb = lmb

        self.optimizer = optim.Adam(self.parameters(), weight_decay=self.lmb)
    
    
    def forward(self, x):
        
        out = self.net(x)
        
        return out
    
    def update_weights(self, loss):
        
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
        
        return self
		
    def reset_params(self, m):
        if isinstance(m, nn.Linear):
            m.reset_parameters()

    def reset_weights(self):
        self.net.apply(self.reset_params)
		
		
class ArrheniusBasedMLP:
    
    def __init__(self, *args, lmbA=0, lmbE=0, batch_size=100, n_epochs=20):

        self.lmbA = lmbA
        self.lmbE = lmbE
        
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        
        self.net_A = SlotNet(*args, lmb=self.lmbA)
        self.net_E = SlotNet(*args, lmb=self.lmbE)
        
    
    def check_input(self, X, y, T):
        
        X = X.astype('float32')
        y = y.astype('float32')
        T = T.astype('float32')
            
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            
        if T.ndim == 1:
            T = T.reshape(-1, 1)
            
        return X, y, T
    
    
    def loss(self, y_pred, y_true):
        
        mse = nn.MSELoss()
        loss = mse(y_pred, y_true)
        
        return loss
    
    
    def get_mini_batches(self, X, y, T):
        
        X_data = GetArrheniusBasedMB(X, y, T)
        mb = DataLoader(X_data, batch_size=self.batch_size, shuffle=True)
        
        return mb
    

    def fit(self, X, y, T):
        
        X, y, T = self.check_input(X, y, T)
		
        self.net_A.reset_weights()
        self.net_E.reset_weights()

        for epoch in range(self.n_epochs): 
            
            mb = self.get_mini_batches(X, y, T)
        
            for X_mb, y_mb, T_mb in mb:
                
                y_out = self.net_A(X_mb) - T_mb * self.net_E(X_mb)
                
                loss = self.loss(y_out, y_mb)

                self.net_A.update_weights(loss)
                self.net_E.update_weights(loss)      
                
        return self
       
    
    def predict_lgA(self, X):
    
        X = torch.from_numpy(X.astype('float32'))

        return self.net_A(X).detach().numpy()
    
    def predict_E(self, X):
        
        X = torch.from_numpy(X.astype('float32'))
        
        return self.net_E(X).detach().numpy()
        
    def predict_lgK(self, X, T):
        
        T = T.reshape(-1, 1)

        return self.predict_lgA(X) - T * self.predict_E(X)