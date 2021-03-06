import torch
import numpy as np


class TautomerismConjugatedRidge:

    def __init__(self, alpha=0, lmb=0, init_cuda=False):

        self.alpha = alpha
        self.lmb = lmb
        self.init_cuda = init_cuda

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

        if self.init_cuda:
            X1, X2, X, YT, YA, I = X1.cuda(), X2.cuda(), X.cuda(), YT.cuda(), YA.cuda(), I.cuda()
        return X1, X2, X, YT, YA, I

    def fit(self, X1, X2, X, YT, YA):

        X1, X2, X, YT, YA, I = self.array_to_tensor(X1, X2, X, YT, YA)

        z1 = torch.inverse(self.alpha * (X2 - X1).transpose(0, 1).mm(X2 - X1) + (1 - self.alpha) * X.transpose(0, 1).mm(X) + self.lmb * I)

        z2 = (self.alpha * (X2 - X1).transpose(0, 1).mm(YT) + (1 - self.alpha) * X.transpose(0, 1).mm(YA))

        self.W_ = z1.mm(z2)

        return self

    def predict_acidity(self, X):
        X = torch.from_numpy(self._add_bias(X).astype('float64'))
        if self.init_cuda:
            X = X.cuda()
        pred = X.mm(self.W_).cpu()
        return pred

    def predict_constant(self, X1, X2):
        pred = self.predict_acidity(X2) - self.predict_acidity(X1)
        return pred


class ArrheniusConjugatedRidge:

    def __init__(self, a=0, lmbA=0, lmbE=0, init_cuda=False):

        self.a = a
        self.lmbA = lmbA
        self.lmbE = lmbE
        self.init_cuda = init_cuda

    def _add_bias(self, X):

        X_new = np.ones((X.shape[0], X.shape[1] + 1))
        X_new[:, 1:] = X

        return X_new

    def array_to_tensor(self, XK, XE, YK, YE, T):

        if YK.ndim == 1:
            YK = YK.reshape(-1, 1)
        if YE.ndim == 1:
            YE = YE.reshape(-1, 1)

        XK = self._add_bias(XK)
        XE = self._add_bias(XE)
        T = np.diag(T)

        XK = torch.from_numpy(XK.astype('float64'))
        XE = torch.from_numpy(XE.astype('float64'))

        YK = torch.from_numpy(YK.astype('float64'))
        YE = torch.from_numpy(YE.astype('float64'))

        T = torch.from_numpy(T.astype('float64'))
        I = torch.from_numpy(np.eye(XK.shape[1]).astype('float64'))

        if self.init_cuda:
            XK, XE, YK, YE, T, I = XK.cuda(), XE.cuda(), YK.cuda(), YE.cuda(), T.cuda(), I.cuda()
        return XK, XE, YK, YE, T, I

    def fit(self, XK, XE, YK, YE, T):

        XK, XE, YK, YE, T, I = self.array_to_tensor(XK, XE, YK, YE, T)

        a1 = torch.inverse(self.a * XK.transpose(0, 1).mm(XK) + self.lmbA * I)
        a2 = self.a * XK.transpose(0, 1).mm(YK)
        a3 = self.a * XK.transpose(0, 1).mm(T).mm(XK)

        e1 = torch.inverse(self.a * XK.transpose(0, 1).mm(T.transpose(0, 1)).mm(T).mm(XK) + (1 - self.a) * XE.transpose(0, 1).mm(XE) + self.lmbE * I)
        e2 = (1 - self.a) * XE.transpose(0, 1).mm(YE) - self.a * XK.transpose(0, 1).mm(T.transpose(0, 1)).mm(YK)
        e3 = self.a * XK.transpose(0, 1).mm(T.transpose(0, 1)).mm(XK)

        A, B, C, D = e1.mm(e2), e1.mm(e3), a1.mm(a2), a1.mm(a3)

        self.WA_ = torch.inverse(I - D.mm(B)).mm(C + D.mm(A))
        self.WE_ = torch.inverse(I - B.mm(D)).mm(A + B.mm(C))

        return self

    def predict_lgA(self, X):

        X = torch.from_numpy(self._add_bias(X).astype('float64'))
        if self.init_cuda:
            X = X.cuda()
        pred = X.mm(self.WA_).cpu()
        return pred

    def predict_E(self, X):

        X = torch.from_numpy(self._add_bias(X).astype('float64'))
        if self.init_cuda:
            X = X.cuda()
        pred = X.mm(self.WE_).cpu()
        return pred

    def predict_lgK(self, X, T):

        T = torch.from_numpy(np.diag(T).astype('float64'))
        pred = self.predict_lgA(X) - T.mm(self.predict_E(X))
        return pred


class SelectivityConjugatedRidge:

    def __init__(self, a=1, b=1, c=1, lmbE=1, lmbS=1, init_cuda=False):

        self.a = a
        self.b = b
        self.c = c
        self.lmbE = lmbE
        self.lmbS = lmbS
        self.init_cuda = init_cuda

    def _add_bias(self, X):
        X_new = np.ones((X.shape[0], X.shape[1] + 1))
        X_new[:, 1:] = X
        return X_new

    def array_to_tensor(self, XE, XS, XEP, XSP, YE, YS, YP):

        if YP.ndim == 1:
            YP = YP.reshape(-1, 1)
        if YE.ndim == 1:
            YE = YE.reshape(-1, 1)
        if YS.ndim == 1:
            YS = YS.reshape(-1, 1)

        XE = self._add_bias(XE)
        XS = self._add_bias(XS)
        XEP = self._add_bias(XEP)
        XSP = self._add_bias(XSP)

        XE = torch.from_numpy(XE.astype('float64'))
        XS = torch.from_numpy(XS.astype('float64'))
        XEP = torch.from_numpy(XEP.astype('float64'))
        XSP = torch.from_numpy(XSP.astype('float64'))

        YE = torch.from_numpy(YE.astype('float64'))
        YS = torch.from_numpy(YS.astype('float64'))
        YP = torch.from_numpy(YP.astype('float64'))

        I = torch.from_numpy(np.eye(XE.shape[1]).astype('float64'))

        if self.init_cuda:
            XE, XS, XEP, XSP, YE, YS, YP, I = XE.cuda(), XS.cuda(), XEP.cuda(), XSP.cuda(), YE.cuda(), YS.cuda(), YP.cuda(), I.cuda()
        return XE, XS, XEP, XSP, YE, YS, YP, I

    def fit(self, XE, XS, XEP, XSP, YE, YS, YP):

        XE, XS, XEP, XSP, YE, YS, YP, I = self.array_to_tensor(XE, XS, XEP, XSP, YE, YS, YP)

        a1 = torch.inverse(self.c * XEP.transpose(0, 1).mm(XEP) + self.a * XE.transpose(0, 1).mm(XE) + self.lmbE * I)
        a2 = (self.a * XE.transpose(0, 1).mm(YE) + self.c * XEP.transpose(0, 1).mm(YP))
        a3 = self.c * XEP.transpose(0, 1).mm(XSP)

        b1 = torch.inverse(self.c * XSP.transpose(0, 1).mm(XSP) + self.b * XS.transpose(0, 1).mm(XS) + self.lmbS * I)
        b2 = (self.b * XS.transpose(0, 1).mm(YS) - self.c * XSP.transpose(0, 1).mm(YP))
        b3 = self.c * XSP.transpose(0, 1).mm(XEP)

        A, B, C, D = a1.mm(a2), a1.mm(a3), b1.mm(b2), b1.mm(b3)

        self.WE_ = torch.inverse(I - B.mm(D)).mm(A + B.mm(C))
        self.WS_ = torch.inverse(I - D.mm(B)).mm(C + D.mm(A))

        return self

    def predict_lgkE(self, X):
        X = torch.from_numpy(self._add_bias(X).astype('float64'))
        if self.init_cuda:
            X = X.cuda()
        pred = X.mm(self.WE_).cpu()
        return pred

    def predict_lgkS(self, X):
        X = torch.from_numpy(self._add_bias(X).astype('float64'))
        if self.init_cuda:
            X = X.cuda()
        pred = X.mm(self.WS_).cpu()
        return pred

    def predict_P(self, XE, XS):
        pred = self.predict_lgkE(XE) - self.predict_lgkS(XS)
        return pred