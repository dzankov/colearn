import torch
import numpy as np


class TensorSelectivityRidge:
    
    def __init__(self, a=1, b=1, c=1, lmbE=1, lmbS=1):
        
        self.a = a
        self.b = b
        self.c = c
        self.lmbE = lmbE
        self.lmbS = lmbS
        
        
    def _add_bias(self, X):
        
        X_new = np.ones((X.shape[0], X.shape[1] + 1))
        X_new[:, 1:] = X
        
        return X_new
    
    
    def prepare_input(self, XE, XS, XEP, XSP, yE, yS, yP):
        
        if yE.ndim == 1:
            yE = yE.reshape(-1, 1)
        if yS.ndim == 1:
            yS = yS.reshape(-1, 1)
        if yP.ndim == 1:
            yP = yP.reshape(-1, 1)
            
        XE = self._add_bias(XE)
        XS = self._add_bias(XS)
        XEP = self._add_bias(XEP)
        XSP = self._add_bias(XSP)

        XE = torch.from_numpy(XE.astype('float64')).cuda()
        XS = torch.from_numpy(XS.astype('float64')).cuda()
        XEP = torch.from_numpy(XEP.astype('float64')).cuda()
        XSP = torch.from_numpy(XSP.astype('float64')).cuda()
        
        yE = torch.from_numpy(yE.astype('float64')).cuda()
        yS = torch.from_numpy(yS.astype('float64')).cuda()
        yP = torch.from_numpy(yP.astype('float64')).cuda()
        I = torch.from_numpy(np.eye(XE.shape[1]).astype('float64')).cuda()
            
        return XE, XS, XEP, XSP, yE, yS, yP, I

    
    def fit(self, XE, XS, XEP, XSP, yE, yS, yP):
        
        XE, XS, XEP, XSP, yE, yS, yP, I = self.prepare_input(XE, XS, XEP, XSP, yE, yS, yP)
        
        a1 = torch.inverse(self.c * XEP.transpose(0, 1).mm(XEP) + 
                           self.a * XE.transpose(0, 1).mm(XE) + self.lmbE * I)
        a2 = (self.a * XE.transpose(0, 1).mm(yE) + self.c * XEP.transpose(0, 1).mm(yP))
        a3 = self.c * XEP.transpose(0, 1).mm(XSP)

        b1 = torch.inverse(self.c * XSP.transpose(0, 1).mm(XSP) + 
                           self.b * XS.transpose(0, 1).mm(XS) + self.lmbS * I)
        b2 = (self.b * XS.transpose(0, 1).mm(yS) - self.c * XSP.transpose(0, 1).mm(yP))
        b3 = self.c * XSP.transpose(0, 1).mm(XEP)

        A, B, C, D = a1.mm(a2), a1.mm(a3), b1.mm(b2), b1.mm(b3)

        self.wE_ = torch.inverse(I - B.mm(D)).mm(A + B.mm(C))
        self.wS_ = torch.inverse(I - D.mm(B)).mm(C + D.mm(A))
        
        return self
    
    
    def predict_lgkE(self, X):
        
        X = torch.from_numpy(self._add_bias(X).astype('float64')).cuda()
        
        return X.mm(self.wE_).cpu()
    
    
    def predict_lgkS(self, X):
        
        X = torch.from_numpy(self._add_bias(X).astype('float64')).cuda()
        
        return X.mm(self.wS_).cpu()
    

    def predict_Eper(self, XE, XS):
        
        per = self.predict_lgkE(XE) - self.predict_lgkS(XS)
        
        return per