import torch
import numpy as np


class ConjugatedRidge:

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

        self.w_ = z1.mm(z2)

        return self

    def predict_acidity(self, X):
        X = torch.from_numpy(self._add_bias(X).astype('float64'))
        if self.init_cuda:
            X = X.cuda()
        pred = X.mm(self.w_).cpu()
        return pred

    def predict_constant(self, X1, X2):
        pred = self.predict_acidity(X2) - self.predict_acidity(X1)
        return pred