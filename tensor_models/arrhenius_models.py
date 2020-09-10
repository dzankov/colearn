import torch
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from colearn.utils import GetMB, GetArrheniusMB


class TensorArrheniusWeightCoopRidge:
    
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

        XE = torch.from_numpy(XE.astype('float64')).cuda()
        XK = torch.from_numpy(XK.astype('float64')).cuda()
        yE = torch.from_numpy(yE.astype('float64')).cuda()
        yK = torch.from_numpy(yK.astype('float64')).cuda()
        T = torch.from_numpy(T.astype('float64')).cuda()
        I = torch.from_numpy(np.eye(XK.shape[1]).astype('float64')).cuda()
            
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
        
        X = torch.from_numpy(self._add_bias(X).astype('float64')).cuda()
        
        return X.mm(self.wA_).cpu()
    
    def predict_E(self, X):
        
        X = torch.from_numpy(self._add_bias(X).astype('float64')).cuda()
        
        return X.mm(self.wE_).cpu()
    
    def predict_lgK(self, X, T):
        
        T = torch.from_numpy(self._get_diag(T).astype('float64'))
        lgK = self.predict_lgA(X) - T.mm(self.predict_E(X))
        
        return lgK
    
   

class SlotNet(nn.Module):
    
    def __init__(self, sequential, lmb=0):
        
        super(SlotNet, self).__init__()
    
        self.net = sequential
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

        
class ArrheniusNet:
    
    def __init__(self, seq_A, seq_B, a=0, lmbA=0, lmbE=0, batch_size=100, n_epochs=20):
        
        self.net_A = SlotNet(seq_A, lmb=lmbA)
        self.net_E = SlotNet(seq_A, lmb=lmbE)
        
        self.net_A.cuda()
        self.net_E.cuda()
        
        self.a = a
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
    
    @property
    def lmbA(self):
        return self.net_A.lmb
    
    
    @property
    def lmbE(self):
        return self.net_E.lmb
    
    
    def numpy_to_tensor(self, XE, XK, yE, yK, T):
        
        if yE.ndim == 1:
            yE = yE.reshape(-1, 1)
        if yK.ndim == 1:
            yK = yK.reshape(-1, 1)
            
        XE = torch.from_numpy(XE.astype('float32')).cuda()
        XK = torch.from_numpy(XK.astype('float32')).cuda()
        yE = torch.from_numpy(yE.astype('float32')).cuda()
        yK = torch.from_numpy(yK.astype('float32')).cuda()
        T = torch.from_numpy(T.astype('float32')).cuda()
        
        return XE, XK, yE, yK, T
    
    
    def loss(self, yE_true, yK_true, yE_pred, yK_pred):
        mse = nn.MSELoss()
        loss = self.a * mse(yK_pred, yK_true) + (1 - self.a) * mse(yE_pred, yE_true)
        return loss
    
    
    def get_mini_batches(self, XE, XK, yE, yK, T):
        
        XE_data = GetMB(XE, yE)
        XK_data = GetArrheniusMB(XK, yK, T)
        
        mb_E = DataLoader(XE_data, batch_size=self.batch_size, shuffle=True)
        mb_K = DataLoader(XK_data, batch_size=self.batch_size, shuffle=True)
        
        return mb_E, mb_K
    

    def fit(self, XE, XK, yE, yK, T):

        self.net_A.reset_weights()
        self.net_E.reset_weights()
        
        XE, XK, yE, yK, T = self.numpy_to_tensor(XE, XK, yE, yK, T)

        for epoch in range(self.n_epochs): 
            
            mb_E, mb_K = self.get_mini_batches(XE, XK, yE, yK, T)
        
            for (XE_mb, yE_mb), (XK_mb, yK_mb, T_mb) in zip(mb_E, mb_K):
                
                yK_out = self.net_A(XK_mb) - T_mb * self.net_E(XK_mb)
                yE_out = self.net_E(XE_mb)
                
                loss = self.loss(yE_mb, yK_mb, yE_out, yK_out)

                self.net_A.update_weights(loss)
                self.net_E.update_weights(loss)      
                
        return self
       
    
    def predict_lgA(self, X):
        X = torch.from_numpy(X.astype('float32')).cuda()
        return self.net_A(X).cpu().detach().numpy()
    
    
    def predict_E(self, X):
        X = torch.from_numpy(X.astype('float32')).cuda()
        return self.net_E(X).cpu().detach().numpy()
        
        
    def predict_lgK(self, X, T):
        T = T.reshape(-1, 1)
        return self.predict_lgA(X) - T * self.predict_E(X)