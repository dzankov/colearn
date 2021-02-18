class ArrheniusConjugatedRidge:

    def __init__(self, a=1, lmbA=0, lmbE=0, init_cuda=False):

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

    def predict_YA(self, X):

        X = torch.from_numpy(self._add_bias(X).astype('float64'))
        if self.init_cuda:
            X = X.cuda()
        pred = X.mm(self.WA_).cpu()
        return pred

    def predict_YE(self, X):

        X = torch.from_numpy(self._add_bias(X).astype('float64'))
        if self.init_cuda:
            X = X.cuda()
        pred = X.mm(self.WE_).cpu()
        return pred

    def predict_YK(self, X, T):
        T = torch.from_numpy(np.diag(T).astype('float64'))
        pred = self.predict_YA(X) - T.mm(self.predict_YE(X))
        return pred